import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from filings_credit_memo.extract import FieldResult  # noqa: E402
from filings_credit_memo.memo import Memo, section_from_result  # noqa: E402


def result(field, status, items=None):
    return FieldResult(field, "CCL", status, items or [], [])


APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_shows_memo_and_source_text(tmp_path, monkeypatch):
    item = {"title": "Weather", "summary": "Other words", "chunk_id": "CCL-1A-0001", "quote": "Weather can hurt us."}
    memo = Memo("CCL", {"Company ticker": "CCL"}, [section_from_result(result("risk_factors", "ok", [item])),
                section_from_result(result("covenants", "insufficient_evidence"))],
                {"CCL-1A-0001": {"text": "FULL CHUNK TEXT", "section": "1A", "lines": [1, 2]}}, model="m")
    (tmp_path / "data/outputs/memos").mkdir(parents=True)
    (tmp_path / "data/outputs/memos/CCL.json").write_text(json.dumps(memo.to_dict()))
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert any("FULL CHUNK TEXT" in t.value for t in at.text)
    assert any("Not determined" in i.value for i in at.info)


def test_app_without_memos_explains_what_to_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP).run(timeout=30)
    assert any("memo CCL" in e.value for e in at.error)
