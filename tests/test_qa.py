"""Tests for free-form question answering with verification. Fully offline."""

import json

from filings_credit_memo.qa import QA_SYSTEM, answer_from_hits
from filings_credit_memo.retrieval import Hit

SENTENCE = "The company employed approximately 76,000 people as of December 31, 2025."
HIT = Hit("AAA-1-0000", 0.80, f"Human capital.\n{SENTENCE}\nWe value our people.", {
    "ticker": "AAA", "section_key": "1", "start_line": 10, "end_line": 12,
})


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user, schema=None):
        self.prompts.append((system, user, schema))
        return self.replies.pop(0)


def reply(answer="76,000", chunk_id="AAA-1-0000", quote=SENTENCE, status="answered"):
    return json.dumps({"status": status, "answer": answer, "chunk_id": chunk_id, "quote": quote})


def ask(llm, hits=(HIT,), min_score=0.65):
    return answer_from_hits("How many employees?", "aaa", list(hits), llm, min_score)


def test_verified_answer():
    res = ask(ScriptedLLM(reply()))
    assert res.status == "answered" and res.answer == "76,000" and res.attempts == 1
    assert res.ticker == "AAA" and res.top_score == 0.80
    json.dumps(res.to_dict())  # serialisable, and hits are left out
    assert "hits" not in res.to_dict()


def test_prompt_is_untrusted_data_with_context_and_schema():
    llm = ScriptedLLM(reply())
    ask(llm)
    system, user, schema = llm.prompts[0]
    assert system == QA_SYSTEM and "untrusted" in system
    assert "Question: How many employees?" in user and "[chunk_id=AAA-1-0000]" in user and "<context>" in user
    assert "$ref" not in json.dumps(schema)


def test_below_threshold_refuses_without_calling_the_model():
    llm = ScriptedLLM()
    res = ask(llm, min_score=0.9)
    assert res.status == "insufficient_evidence" and llm.prompts == []


def test_no_hits_is_insufficient_evidence():
    assert ask(ScriptedLLM(), hits=()).status == "insufficient_evidence"


def test_model_says_not_found():
    res = ask(ScriptedLLM(reply(answer="", chunk_id="", quote="", status="not_found")))
    assert res.status == "not_found" and res.answer == ""


def test_quote_not_verbatim_is_unsupported():
    res = ask(ScriptedLLM(reply(quote="The company employed about 90,000 people worldwide at year end.")))
    assert res.status == "unsupported" and "verbatim" in res.reason


def test_chunk_id_not_retrieved_is_unsupported():
    assert ask(ScriptedLLM(reply(chunk_id="ZZZ-9-0001"))).status == "unsupported"


def test_number_missing_from_quote_is_unsupported():
    res = ask(ScriptedLLM(reply(answer="90,000")))
    assert res.status == "unsupported" and "90,000" in res.reason


def test_empty_answer_is_unsupported():
    assert ask(ScriptedLLM(reply(answer="  "))).status == "unsupported"


def test_invalid_json_is_retried_then_accepted():
    llm = ScriptedLLM("not json", reply())
    res = ask(llm)
    assert res.status == "answered" and res.attempts == 2
    assert "invalid" in llm.prompts[1][1]


def test_always_invalid_gives_invalid_output():
    res = ask(ScriptedLLM("x", "y", "z"))
    assert res.status == "invalid_output" and res.attempts == 3
