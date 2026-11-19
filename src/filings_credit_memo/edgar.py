"""A polite SEC EDGAR client: declared User-Agent, rate limit, disk cache, bounded retries.

Rules this code follows (from sec.gov "Accessing EDGAR Data", checked 2026-11-19; re-check before relying on them):
  * Declare a User-Agent that identifies you and includes a contact email.
  * Stay at or below 10 requests per second (we default to 5).
  * Download only what you need; no crawling. Everything fetched is cached on disk so it is never fetched twice.

The JSON field names used below (company_tickers.json and the submissions "filings.recent" arrays) come from the
author's knowledge of the EDGAR formats, not from the page above. The code checks for them and fails loudly if
they are missing, so a format change shows up as a clear error instead of silent wrong data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
SEC_MAX_REQUESTS_PER_SECOND = 10.0
DEFAULT_REQUESTS_PER_SECOND = 5.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
PLACEHOLDER_DOMAINS = ("example.com", "example.org")


class EdgarError(RuntimeError):
    """Any problem talking to or interpreting EDGAR."""


def validate_user_agent(value: str | None) -> str:
    """Return the cleaned User-Agent or raise. It must contain a real-looking contact email."""
    ua = (value or "").strip()
    match = EMAIL_RE.search(ua)
    if not ua or not match:
        raise EdgarError("SEC_USER_AGENT must be like 'Your Name your-email@domain.com' (contact email required).")
    if any(match.group(0).lower().endswith(d) for d in PLACEHOLDER_DOMAINS):
        raise EdgarError("SEC_USER_AGENT still has the placeholder email; put your real contact email in .env.")
    return ua


def load_user_agent(env_var: str = "SEC_USER_AGENT") -> str:
    load_dotenv()
    return validate_user_agent(os.environ.get(env_var))


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    form: str
    accession: str  # with dashes, e.g. 0000123456-26-000012
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession_nodash=self.accession.replace("-", ""), document=self.primary_document
        )


class EdgarClient:
    """HTTP client with a declared User-Agent, a request-rate cap, an on-disk cache and bounded retries."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < requests_per_second <= SEC_MAX_REQUESTS_PER_SECOND:
            raise ValueError(f"requests_per_second must be in (0, {SEC_MAX_REQUESTS_PER_SECOND}]")
        self._interval = 1.0 / requests_per_second
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.network_requests = 0
        self._http = httpx.Client(
            headers={"User-Agent": validate_user_agent(user_agent), "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _cache_path(self, url: str) -> Path:
        return self._cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".bin")

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get(self, url: str) -> bytes:
        """Fetch a URL, using the disk cache when possible. Only successful (200) responses are cached."""
        path = self._cache_path(url)
        if path.exists():
            return path.read_bytes()
        for attempt in range(MAX_ATTEMPTS):
            self._throttle()
            self.network_requests += 1
            try:
                response = self._http.get(url)
            except httpx.TransportError as err:
                if attempt == MAX_ATTEMPTS - 1:
                    raise EdgarError(f"network error fetching {url}: {err}") from err
                self._sleep(2.0**attempt)
                continue
            if response.status_code == 200:
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(response.content)
                os.replace(tmp, path)
                return response.content
            if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS - 1:
                self._sleep(2.0**attempt)
                continue
            hint = " (403: check your User-Agent and request rate)" if response.status_code == 403 else ""
            raise EdgarError(f"HTTP {response.status_code} for {url}{hint}")
        raise EdgarError(f"gave up on {url}")  # pragma: no cover

    def get_json(self, url: str) -> dict:
        try:
            return json.loads(self.get(url))
        except json.JSONDecodeError as err:
            raise EdgarError(f"response from {url} was not valid JSON") from err


def ticker_to_cik(tickers_json: dict, ticker: str) -> int:
    """Look up a CIK in company_tickers.json (a dict of index -> {cik_str, ticker, title})."""
    wanted = ticker.upper()
    for row in tickers_json.values():
        if str(row.get("ticker", "")).upper() == wanted:
            return int(row["cik_str"])
    raise EdgarError(f"ticker {ticker!r} not found in company_tickers.json")


_REQUIRED_KEYS = ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument")


def find_latest_filing(submissions: dict, ticker: str, cik: int, form: str = "10-K") -> Filing:
    """Newest filing of exactly this form type (so 10-K/A amendments and 10-Q are not picked up)."""
    try:
        recent = submissions["filings"]["recent"]
        columns = {k: recent[k] for k in _REQUIRED_KEYS}
    except KeyError as err:
        raise EdgarError(f"unexpected submissions format, missing {err}") from err
    rows = [dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)]
    matches = [r for r in rows if r["form"] == form]
    if not matches:
        raise EdgarError(f"no {form} in the recent filings of {ticker} (CIK {cik})")
    best = max(matches, key=lambda r: r["filingDate"])
    return Filing(
        ticker=ticker.upper(),
        cik=cik,
        form=best["form"],
        accession=best["accessionNumber"],
        filing_date=best["filingDate"],
        report_date=best["reportDate"],
        primary_document=best["primaryDocument"],
    )


def download_latest(client: EdgarClient, ticker: str, out_dir: Path, form: str = "10-K") -> Path:
    """Download the newest filing of `form` for `ticker` into out_dir/TICKER/. Returns the saved HTML path."""
    cik = ticker_to_cik(client.get_json(TICKERS_URL), ticker)
    filing = find_latest_filing(client.get_json(SUBMISSIONS_URL.format(cik=cik)), ticker, cik, form)
    target_dir = Path(out_dir) / filing.ticker
    target_dir.mkdir(parents=True, exist_ok=True)
    html_path = target_dir / f"{filing.accession}.htm"
    if not html_path.exists():
        html_path.write_bytes(client.get(filing.url))
    (target_dir / f"{filing.accession}.json").write_text(json.dumps(asdict(filing) | {"url": filing.url}, indent=2))
    return html_path
