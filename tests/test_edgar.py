"""Offline tests for the EDGAR client: no real network calls are made (httpx.MockTransport)."""
import json
from pathlib import Path

import httpx
import pytest

from filings_credit_memo.edgar import (
    SUBMISSIONS_URL,
    TICKERS_URL,
    EdgarClient,
    EdgarError,
    Filing,
    download_latest,
    find_latest_filing,
    ticker_to_cik,
    validate_user_agent,
)

UA = "Test User test.user@realdomain.org"
TICKERS = {
    "0": {"cik_str": 111, "ticker": "AAA", "title": "Alpha Corp"},
    "1": {"cik_str": 222, "ticker": "BBB", "title": "Beta Corp"},
}


def submissions(rows: list[tuple[str, str, str, str, str]]) -> dict:
    """rows: (accession, filing_date, report_date, form, primary_document)"""
    keys = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]
    return {"filings": {"recent": {k: [r[i] for r in rows] for i, k in enumerate(keys)}}}


# ---- User-Agent validation -------------------------------------------------------------------------------------


def test_user_agent_requires_email():
    with pytest.raises(EdgarError):
        validate_user_agent("Just A Name")
    with pytest.raises(EdgarError):
        validate_user_agent(None)
    with pytest.raises(EdgarError):
        validate_user_agent("   ")


def test_user_agent_rejects_placeholder_email():
    with pytest.raises(EdgarError, match="placeholder"):
        validate_user_agent("Your Name your-email@example.com")


def test_user_agent_accepts_real_looking_value():
    assert validate_user_agent("  Jane Doe jane.doe@university.edu ") == "Jane Doe jane.doe@university.edu"


# ---- ticker lookup and filing selection ------------------------------------------------------------------------


def test_ticker_to_cik_is_case_insensitive_and_raises_when_missing():
    assert ticker_to_cik(TICKERS, "aaa") == 111
    with pytest.raises(EdgarError, match="not found"):
        ticker_to_cik(TICKERS, "ZZZ")


def test_find_latest_filing_picks_newest_exact_form_only():
    data = submissions(
        [
            ("0001-26-000003", "2026-05-01", "2026-03-31", "10-Q", "q.htm"),
            ("0001-26-000002", "2026-02-20", "2025-12-31", "10-K", "new10k.htm"),
            ("0001-26-000004", "2026-03-15", "2025-12-31", "10-K/A", "amend.htm"),
            ("0001-25-000001", "2025-02-18", "2024-12-31", "10-K", "old10k.htm"),
        ]
    )
    f = find_latest_filing(data, "aaa", 111)
    assert (f.ticker, f.accession, f.primary_document, f.filing_date) == ("AAA", "0001-26-000002", "new10k.htm", "2026-02-20")


def test_find_latest_filing_raises_when_no_such_form_or_bad_format():
    data = submissions([("0001-26-000003", "2026-05-01", "2026-03-31", "10-Q", "q.htm")])
    with pytest.raises(EdgarError, match="no 10-K"):
        find_latest_filing(data, "AAA", 111)
    with pytest.raises(EdgarError, match="unexpected submissions format"):
        find_latest_filing({"filings": {"recent": {"form": []}}}, "AAA", 111)


def test_filing_url_uses_integer_cik_and_dashless_accession():
    f = Filing("AAA", 123456, "10-K", "0000123456-26-000012", "2026-02-20", "2025-12-31", "doc.htm")
    assert f.url == "https://www.sec.gov/Archives/edgar/data/123456/000012345626000012/doc.htm"


# ---- the client: headers, cache, retries, throttle, rate cap -----------------------------------------------------


def make_client(tmp_path: Path, handler, rps: float = 5.0, sleeps: list[float] | None = None, clock=None):
    sleeps = sleeps if sleeps is not None else []
    kwargs = {"clock": clock} if clock else {}
    return EdgarClient(UA, tmp_path / "cache", rps, httpx.MockTransport(handler), sleep=sleeps.append, **kwargs)


def test_client_sends_declared_user_agent(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, content=b"ok")

    with make_client(tmp_path, handler) as client:
        client.get("https://data.sec.gov/x")
    assert seen["ua"] == UA


def test_client_rejects_rate_above_sec_limit_or_nonpositive(tmp_path):
    for bad in (0, -1, 10.5):
        with pytest.raises(ValueError):
            EdgarClient(UA, tmp_path / "c", requests_per_second=bad)


def test_second_get_of_same_url_comes_from_disk_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"payload")

    with make_client(tmp_path, handler) as client:
        assert client.get("https://data.sec.gov/x") == b"payload"
        assert client.get("https://data.sec.gov/x") == b"payload"
        assert client.network_requests == 1
    assert len(calls) == 1


def test_retries_on_503_then_succeeds_and_backs_off(tmp_path):
    responses = [httpx.Response(503), httpx.Response(503), httpx.Response(200, content=b"fine")]

    def handler(request):
        return responses.pop(0)

    sleeps: list[float] = []
    with make_client(tmp_path, handler, sleeps=sleeps) as client:
        assert client.get("https://data.sec.gov/x") == b"fine"
    assert 1.0 in sleeps and 2.0 in sleeps  # exponential backoff between attempts


def test_gives_up_after_max_attempts_and_does_not_cache_failures(tmp_path):
    def handler(request):
        return httpx.Response(503)

    with make_client(tmp_path, handler) as client:
        with pytest.raises(EdgarError, match="503"):
            client.get("https://data.sec.gov/x")
        assert client.network_requests == 3
    assert list((tmp_path / "cache").iterdir()) == []


def test_403_fails_immediately_with_user_agent_hint(tmp_path):
    def handler(request):
        return httpx.Response(403)

    with make_client(tmp_path, handler) as client:
        with pytest.raises(EdgarError, match="User-Agent"):
            client.get("https://data.sec.gov/x")
        assert client.network_requests == 1


def test_throttle_waits_between_requests(tmp_path):
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    def handler(request):
        return httpx.Response(200, content=b"x")

    client = EdgarClient(UA, tmp_path / "c", 5.0, httpx.MockTransport(handler), sleep=sleep, clock=lambda: now[0])
    client.get("https://data.sec.gov/a")
    client.get("https://data.sec.gov/b")
    client.close()
    assert sleeps == [pytest.approx(0.2)]  # 1 / 5 requests per second; no wait before the first request


# ---- end to end -------------------------------------------------------------------------------------------------


def test_download_latest_saves_filing_and_metadata_and_is_idempotent(tmp_path):
    sub = submissions([("0000111-26-000009", "2026-02-20", "2025-12-31", "10-K", "alpha10k.htm")])
    log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        log.append(url)
        if url == TICKERS_URL:
            return httpx.Response(200, json=TICKERS)
        if url == SUBMISSIONS_URL.format(cik=111):
            return httpx.Response(200, json=sub)
        if url == "https://www.sec.gov/Archives/edgar/data/111/000011126000009/alpha10k.htm":
            return httpx.Response(200, content=b"<html>annual report</html>")
        return httpx.Response(404)

    out = tmp_path / "filings"
    with make_client(tmp_path, handler) as client:
        path = download_latest(client, "AAA", out)
        assert path == out / "AAA" / "0000111-26-000009.htm"
        assert path.read_bytes() == b"<html>annual report</html>"
        meta = json.loads((out / "AAA" / "0000111-26-000009.json").read_text())
        assert meta["form"] == "10-K" and meta["filing_date"] == "2026-02-20" and meta["url"].endswith("alpha10k.htm")
        first_run_requests = client.network_requests
        download_latest(client, "AAA", out)
        assert client.network_requests == first_run_requests  # second run touches no network
    assert len(log) == 3
