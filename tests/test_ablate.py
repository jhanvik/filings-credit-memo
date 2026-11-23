import json

import pytest

from filings_credit_memo import ablate
from filings_credit_memo.evaluate import RowResult
from filings_credit_memo.qa import PROMPTS, QA_SYSTEM, AnswerWithReasoning, answer_from_hits
from filings_credit_memo.retrieval import Hit


class FakeLLM:
    name = "fake"

    def __init__(self, reply: dict):
        self.reply = reply
        self.systems: list[str] = []
        self.schemas: list[dict] = []

    def complete(self, system, user, schema=None):
        self.systems.append(system)
        self.schemas.append(schema)
        return json.dumps(self.reply)


def _hit() -> Hit:
    text = "The facility matures in June 2030 and bears interest at SOFR."
    meta = {"ticker": "CCL", "section_key": "8", "start_line": 1, "end_line": 2}
    return Hit("CCL-8-0001", 0.8, text, meta)


def _reply(**extra) -> dict:
    base = {"status": "answered", "answer": "June 2030", "chunk_id": "CCL-8-0001", "quote": "The facility matures in June 2030"}
    return {**extra, **base}


def test_default_call_still_uses_the_baseline_prompt():
    llm = FakeLLM(_reply())
    result = answer_from_hits("When?", "CCL", [_hit()], llm)
    assert result.status == "answered"
    assert llm.systems == [QA_SYSTEM]


def test_relaxed_prompt_and_reasoning_schema_are_passed_through():
    llm = FakeLLM(_reply(reasoning="Excerpt 1 states the maturity."))
    result = answer_from_hits(
        "When?", "CCL", [_hit()], llm, system=PROMPTS["relaxed"], model_cls=AnswerWithReasoning
    )
    assert result.status == "answered"
    assert llm.systems == [PROMPTS["relaxed"]]
    assert list(llm.schemas[0]["properties"])[0] == "reasoning"  # reasoning is written before the answer


def test_reasoning_field_missing_is_an_invalid_output_not_a_crash():
    llm = FakeLLM(_reply())  # no reasoning field although the schema asks for it
    result = answer_from_hits("When?", "CCL", [_hit()], llm, model_cls=AnswerWithReasoning)
    assert result.status == "invalid_output"


def _row(id_, answerable, status, score=0.8, answer_correct=None) -> RowResult:
    return RowResult(
        id_, "CCL", "f", "q?", answerable, ("x",), status, "a", "q", "", score, 1 if answerable else None,
        answer_correct, None, 1.0, False,
    )


def test_changes_vs_baseline_counts_flips_in_both_directions():
    base = [_row("a", True, "not_found"), _row("b", True, "answered", answer_correct=True), _row("c", False, "not_found")]
    new = [_row("a", True, "answered", answer_correct=True), _row("b", True, "not_found"), _row("c", False, "answered")]
    gained, lost = ablate.changes_vs_baseline(base, new)
    assert gained == ["a"]
    assert lost == ["b", "c"]  # c is a refusal that turned into a false answer


def test_select_best_prefers_utility_and_penalises_wrong_answers():
    safe = [_row("a", True, "answered", answer_correct=True), _row("b", True, "not_found")]
    risky = [_row("a", True, "answered", answer_correct=True), _row("b", True, "answered", answer_correct=False)]
    entries = [(ablate.BY_NAME["relaxed_k5"], risky), (ablate.BY_NAME["base_k5"], safe)]
    assert ablate.select_best(entries).name == "base_k5"


def test_select_best_breaks_ties_toward_smaller_k_and_no_reasoning():
    res = [_row("a", True, "answered", answer_correct=True)]
    entries = [(ablate.BY_NAME["relaxed_reason_k5"], res), (ablate.BY_NAME["relaxed_k3"], res), (ablate.BY_NAME["relaxed_k5"], res)]
    assert ablate.select_best(entries).name == "relaxed_k3"


def test_report_lists_every_config_and_the_flips():
    base = [_row("a", True, "not_found"), _row("c", False, "not_found")]
    new = [_row("a", True, "answered", answer_correct=True), _row("c", False, "not_found")]
    text = ablate.render_ablation(
        [(ablate.BY_NAME["base_k5"], base), (ablate.BY_NAME["relaxed_k5"], new)], "Ablation report"
    )
    assert "base_k5" in text and "relaxed_k5" in text
    assert "gained ['a']" in text
    assert "not a finding" in text


def test_holdout_refuses_a_second_run(tmp_path):
    used = tmp_path / "HOLDOUT_USED.json"
    ablate.ensure_holdout_unused(used)  # first time: fine
    used.write_text(json.dumps({"used_at_utc": "2026-11-24T00:00:00+00:00"}))
    with pytest.raises(SystemExit):
        ablate.ensure_holdout_unused(used)


def test_holdout_needs_the_confirm_flag(capsys):
    assert ablate.main(["holdout", "--config", "base_k5"]) == 2
    assert "--confirm-final-run" in capsys.readouterr().err


def test_unknown_config_name_is_rejected(capsys):
    assert ablate.main(["run", "--only", "nope"]) == 2
    assert "Unknown config" in capsys.readouterr().err
