from types import SimpleNamespace

from filings_credit_memo.extract import FieldResult
from filings_credit_memo.llm import LLMError
from filings_credit_memo.memo import (
    DEFAULT_QUESTIONS,
    Memo,
    build_memo,
    clean,
    header_for,
    render_markdown,
    section_from_answer,
    section_from_result,
)
from filings_credit_memo.parse import ParsedFiling
from filings_credit_memo.qa import AnswerResult


def filing(report_date: str = "2025-11-30") -> ParsedFiling:
    return ParsedFiling("CCL", "0000815097-26-000010", "10-K", "2026-01-27", report_date, "https://sec.example/x", 10, [])


class FakeStore:
    def __init__(self, chunks: dict[str, str]):
        self.chunks = chunks

    def get_all(self, ticker=None):
        hits = [
            SimpleNamespace(chunk_id=cid, text=text, section_key="1A", lines=(1, 2)) for cid, text in self.chunks.items()
        ]
        return hits, []


def result(field: str, status: str, items=None, rejected=None) -> FieldResult:
    return FieldResult(field, "CCL", status, items or [], rejected or [])  # type: ignore[arg-type]


def test_clean_strips_markup_and_collapses_whitespace():
    dirty = "See [click](http://evil.example)\n![img](http://evil.example/x.png) <b>hi</b> `code` **bold**"
    out = clean(dirty)
    for bad in ("[", "]", "<", ">", "`", "*"):
        assert bad not in out
    assert "\n" not in out


def test_clean_truncates_long_text():
    assert len(clean("word " * 500, limit=100)) <= 100


def test_header_uses_metadata_fiscal_year_end_and_never_guesses():
    assert header_for(filing())["Fiscal year end"] == "November 30, 2025"
    assert header_for(filing("not a date"))["Fiscal year end"] == "not available"


def test_ok_risk_factor_section_keeps_citation_and_quote():
    item = {"title": "Fuel [prices]", "summary": "Costs may rise.", "chunk_id": "CCL-1A-0001", "quote": "Fuel costs may rise."}
    sec = section_from_result(result("risk_factors", "ok", [item]))
    assert sec.status == "ok"
    assert sec.entries[0].chunk_id == "CCL-1A-0001"
    assert "[" not in sec.entries[0].text


def test_debt_entry_omits_blank_fields():
    item = {"instrument": "Term loan", "maturity": "2029", "amount": "", "chunk_id": "c", "quote": "q"}
    sec = section_from_result(result("debt_facilities", "ok", [item]))
    assert sec.entries[0].text == "Term loan (matures 2029)"


def test_missing_evidence_is_not_reported_as_absence():
    sec = section_from_result(result("covenants", "insufficient_evidence"))
    assert sec.status == "missing"
    assert "does not mean the filing is silent" in sec.note


def test_all_rejected_items_are_withheld_not_shown():
    rejected = [{"item": {"description": "made up"}, "reason": "quote not in chunk"}]
    sec = section_from_result(result("covenants", "unsupported", rejected=rejected))
    assert sec.status == "withheld"
    assert sec.entries == []
    assert "made up" not in sec.note


def test_going_concern_not_found_is_missing():
    item = {"status": "not_found", "chunk_id": "", "quote": ""}
    assert section_from_result(result("going_concern", "ok", [item])).status == "missing"


def test_going_concern_no_doubt_is_worded_cautiously():
    item = {"status": "no_doubt_stated", "chunk_id": "c", "quote": "q"}
    sec = section_from_result(result("going_concern", "ok", [item]))
    assert "does not express" in sec.entries[0].text


def test_invalid_output_is_an_error_section():
    assert section_from_result(result("risk_factors", "invalid_output")).status == "error"


def test_answer_statuses_map_to_sections():
    ok = AnswerResult("q?", "CCL", "answered", answer="$1", chunk_id="c1", quote="paid $1")
    assert section_from_answer(ok).status == "ok"
    for status, expected in (("not_found", "missing"), ("insufficient_evidence", "missing"),
                             ("unsupported", "withheld"), ("invalid_output", "error")):
        assert section_from_answer(AnswerResult("q?", "CCL", status)).status == expected


def fake_extractor(spec, ticker, store, embedder, llm, min_score):
    if spec.name == "risk_factors":
        item = {"title": "T", "summary": "S", "chunk_id": "CCL-1A-0001", "quote": "quote text here"}
        return result("risk_factors", "ok", [item])
    return result(spec.name, "insufficient_evidence")


def fake_answerer(store, embedder, llm, ticker, question, min_score):
    return AnswerResult(question, ticker, "answered", answer="42", chunk_id="CCL-5-0002", quote="42 holders")


def test_build_memo_collects_evidence_only_for_cited_chunks():
    store = FakeStore({"CCL-1A-0001": "full chunk one", "CCL-5-0002": "full chunk two", "CCL-9-0009": "unused"})
    memo = build_memo(filing(), store, None, SimpleNamespace(name="m"), extractor=fake_extractor, answerer=fake_answerer)  # type: ignore[arg-type]
    assert set(memo.evidence) == {"CCL-1A-0001", "CCL-5-0002"}
    assert memo.evidence["CCL-1A-0001"]["text"] == "full chunk one"
    assert len(memo.sections) == 4 + len(DEFAULT_QUESTIONS)


def test_build_memo_survives_a_model_failure_in_one_section():
    def flaky(spec, *args):
        if spec.name == "debt_facilities":
            raise LLMError("ollama down")
        return result(spec.name, "insufficient_evidence")

    memo = build_memo(filing(), FakeStore({}), None, SimpleNamespace(name="m"), extractor=flaky, answerer=fake_answerer)  # type: ignore[arg-type]
    by_key = {s.key: s for s in memo.sections}
    assert by_key["debt_facilities"].status == "error"
    assert by_key["risk_factors"].status == "missing"


def test_markdown_has_disclaimer_citations_and_no_advice_section():
    store = FakeStore({"CCL-1A-0001": "x", "CCL-5-0002": "y"})
    memo = build_memo(filing(), store, None, SimpleNamespace(name="m"), extractor=fake_extractor, answerer=fake_answerer)  # type: ignore[arg-type]
    md = render_markdown(memo)
    assert "not a credit opinion, rating or investment advice" in md
    assert "`CCL-1A-0001`" in md
    assert "November 30, 2025" in md
    for word in ("recommend", "buy", "sell", "rating:"):
        assert f"## {word}" not in md.lower()


def test_markdown_never_contains_raw_links_from_model_text():
    item = {"title": "T", "summary": "[x](http://evil.example)", "chunk_id": "c", "quote": "![i](http://evil.example/p.png)"}
    memo = Memo("CCL", {"Company ticker": "CCL"}, [section_from_result(result("risk_factors", "ok", [item]))], {})
    md = render_markdown(memo)
    assert "](" not in md
    assert "![" not in md


def test_risk_factor_summary_that_repeats_the_quote_is_dropped():
    quote = "Adverse weather conditions could have a material impact on our business."
    item = {"title": "Weather risk", "summary": quote, "chunk_id": "c", "quote": quote}
    sec = section_from_result(result("risk_factors", "ok", [item]))
    assert sec.entries[0].text == "Weather risk"


def test_lists_are_labeled_partial_but_going_concern_is_not():
    item = {"instrument": "Notes", "maturity": "2030", "amount": "", "chunk_id": "c", "quote": "q"}
    assert "partial" in section_from_result(result("debt_facilities", "ok", [item])).note
    gc = {"status": "doubt_expressed", "chunk_id": "c", "quote": "q"}
    assert "partial" not in section_from_result(result("going_concern", "ok", [gc])).note


def test_memo_round_trips_through_json(tmp_path):
    import json

    from filings_credit_memo.memo import load_memo, saved_memos

    item = {"title": "T", "summary": "Different words", "chunk_id": "CCL-1A-0001", "quote": "quote text here"}
    memo = Memo("CCL", {"Company ticker": "CCL"}, [section_from_result(result("risk_factors", "ok", [item]))],
                {"CCL-1A-0001": {"text": "full", "section": "1A", "lines": [1, 2]}}, model="m")
    path = tmp_path / "CCL.json"
    path.write_text(json.dumps(memo.to_dict()))
    loaded = load_memo(path)
    assert loaded.sections[0].entries[0].chunk_id == "CCL-1A-0001"
    assert loaded.evidence == memo.evidence
    assert render_markdown(loaded) == render_markdown(memo)
    assert list(saved_memos(tmp_path)) == ["CCL"]
