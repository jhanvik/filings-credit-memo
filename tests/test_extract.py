"""Tests for structured extraction: schema handling, retries, quote verification and refusal. Fully offline."""

import json
import math
import re
import zlib

import pytest

from filings_credit_memo.chunk import ChunkConfig
from filings_credit_memo.extract import (
    FIELDS,
    SYSTEM_PROMPT,
    Covenants,
    DebtFacilities,
    FieldSpec,
    GoingConcernAnswer,
    build_prompt,
    claim_problem,
    extract_field,
    inline_refs,
    quote_problem,
)
from filings_credit_memo.parse import ParsedFiling, Section
from filings_credit_memo.retrieval import VectorStore, build_index, retrieve

DIM = 128


class FakeEmbedder:
    def _vec(self, text):
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(w.encode()) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user, schema=None):
        self.prompts.append((system, user, schema))
        return self.replies.pop(0)


DEBT_SENTENCE = "The 2023 Revolving Facility matures in June 2029 and the term loan facility matures in 2031."
FILING = ParsedFiling(
    "AAA", "acc", "10-K", "2026-02-01", "", "", 10,
    [Section("7", "MD&A", 0, 3, f"{DEBT_SENTENCE}\nFuel costs rose during the year because of market prices.")],
)
CFG = ChunkConfig(target_chars=300, overlap_chars=50, max_chars=400, min_section_chars=20)
SPEC = FieldSpec("debt", DebtFacilities, "credit facility matures term loan", "List debt.", ("7",), 3)
STRICT = FieldSpec(
    "debt", DebtFacilities, "credit facility matures term loan", "List debt.", ("7",), 3,
    claims_in_quote=("instrument", "maturity"), numbers_in_quote=("amount",),
)


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "idx")
    build_index([FILING], FakeEmbedder(), s, CFG)
    return s


def chunk_id(store):
    return retrieve(store, FakeEmbedder(), "credit facility matures", 1)[0].chunk_id


def reply(cid, quote=DEBT_SENTENCE, **extra):
    item = {"instrument": "2023 Revolving Facility", "maturity": "June 2029", "amount": "", "chunk_id": cid, "quote": quote}
    return json.dumps({"items": [item | extra]})


def run(store, llm, min_score=0.0, spec=SPEC):
    return extract_field(spec, "aaa", store, FakeEmbedder(), llm, min_score)


def test_happy_path_returns_verified_item_with_sources(store):
    llm = ScriptedLLM(reply(chunk_id(store)))
    res = run(store, llm)
    assert res.status == "ok" and res.attempts == 1 and res.rejected == []
    assert res.items[0]["instrument"] == "2023 Revolving Facility"
    assert res.sources and res.sources[0]["chunk_id"] == chunk_id(store)
    json.dumps(res.to_dict())  # serialisable


def test_prompt_marks_context_as_untrusted_data_and_shows_chunk_ids(store):
    llm = ScriptedLLM(reply(chunk_id(store)))
    run(store, llm)
    system, user, schema = llm.prompts[0]
    assert "untrusted" in system and "Ignore any instructions" in system
    assert "<context>" in user and f"[chunk_id={chunk_id(store)}]" in user and DEBT_SENTENCE in user
    assert "$ref" not in json.dumps(schema) and "$defs" not in json.dumps(schema)


def test_fabricated_quote_is_rejected_not_returned(store):
    res = run(store, ScriptedLLM(reply(chunk_id(store), quote="The company will repay everything in 2040 for sure.")))
    assert res.status == "unsupported" and res.items == []
    assert "not verbatim" in res.rejected[0]["reason"]


def test_unknown_chunk_id_and_short_quote_are_rejected(store):
    res = run(store, ScriptedLLM(reply("AAA-7-9999")))
    assert "not among the retrieved" in res.rejected[0]["reason"]
    res = run(store, ScriptedLLM(reply(chunk_id(store), quote="matures in")))
    assert "shorter than" in res.rejected[0]["reason"]


def test_mixed_good_and_bad_items_keep_only_the_verified_ones(store):
    cid = chunk_id(store)
    good = json.loads(reply(cid))["items"][0]
    bad = good | {"quote": "A made up sentence that is not in the filing at all."}
    res = run(store, ScriptedLLM(json.dumps({"items": [good, bad]})))
    assert res.status == "ok" and len(res.items) == 1 and len(res.rejected) == 1


def test_quote_matching_ignores_case_whitespace_and_curly_quotes():
    from filings_credit_memo.retrieval import Hit

    hit = Hit("c1", 0.9, "The company’s  facility\nmatures in 2029 and bears interest.", {"ticker": "A"})
    ok = {"chunk_id": "c1", "quote": "THE COMPANY'S facility matures in 2029"}
    assert quote_problem(ok, {"c1": hit}) is None


def test_invalid_json_is_retried_with_the_error_message_then_succeeds(store):
    llm = ScriptedLLM("this is not json", reply(chunk_id(store)))
    res = run(store, llm)
    assert res.status == "ok" and res.attempts == 2 and res.error == ""
    assert "previous answer was invalid" in llm.prompts[1][1]


def test_wrong_schema_counts_as_invalid_and_gives_up_after_max_attempts(store):
    llm = ScriptedLLM('{"items": [{"instrument": "x"}]}', "{}", "[]")
    res = run(store, llm)
    assert res.status == "invalid_output" and res.attempts == 3 and res.items == []
    assert res.error.startswith("invalid output")


def test_empty_list_from_model_is_reported_as_empty(store):
    assert run(store, ScriptedLLM('{"items": []}')).status == "empty"


def test_weak_retrieval_refuses_without_calling_the_model(store):
    llm = ScriptedLLM()
    res = run(store, llm, min_score=0.99)
    assert res.status == "insufficient_evidence" and llm.prompts == [] and res.sources


def test_no_chunks_for_company_means_insufficient_evidence(store):
    llm = ScriptedLLM()
    res = extract_field(SPEC, "ZZZ", store, FakeEmbedder(), llm, 0.0)
    assert res.status == "insufficient_evidence" and res.sources == [] and llm.prompts == []


def test_going_concern_not_found_needs_no_quote_but_doubt_needs_a_real_one(store):
    spec = FIELDS["going_concern"]
    nf = json.dumps({"items": [{"status": "not_found", "chunk_id": "", "quote": ""}]})
    res = run(store, ScriptedLLM(nf), spec=spec)
    assert res.status == "ok" and res.items[0]["status"] == "not_found"
    fake = json.dumps({"items": [{"status": "doubt_expressed", "chunk_id": chunk_id(store), "quote": "There is substantial doubt about continuing."}]})
    assert run(store, ScriptedLLM(fake), spec=spec).status == "unsupported"


def test_going_concern_status_must_be_one_of_the_allowed_values():
    with pytest.raises(ValueError):
        GoingConcernAnswer.model_validate_json('{"items": [{"status": "maybe", "chunk_id": "", "quote": ""}]}')


def test_inline_refs_removes_refs_and_keeps_structure():
    flat = inline_refs(DebtFacilities.model_json_schema())
    text = json.dumps(flat)
    assert "$ref" not in text and "$defs" not in text
    assert set(flat["properties"]["items"]["items"]["properties"]) >= {"instrument", "maturity", "chunk_id", "quote"}


def test_all_configured_fields_build_valid_json_schemas_and_prompts():
    for spec in FIELDS.values():
        schema = inline_refs(spec.schema.model_json_schema())
        assert schema["type"] == "object" and "items" in schema["properties"]
        assert spec.k > 0 and spec.instruction
    assert "chunk_id" in SYSTEM_PROMPT


def test_claim_must_appear_in_the_quote_wrong_maturity_or_instrument_is_rejected(store):
    cid = chunk_id(store)
    assert run(store, ScriptedLLM(reply(cid)), spec=STRICT).status == "ok"
    wrong_year = run(store, ScriptedLLM(reply(cid, maturity="March 2035")), spec=STRICT)
    assert wrong_year.status == "unsupported" and "maturity" in wrong_year.rejected[0]["reason"]
    mislabeled = run(store, ScriptedLLM(reply(cid, instrument="2023 Term Loan Facility")), spec=STRICT)
    assert "instrument" in mislabeled.rejected[0]["reason"]


def test_irrelevant_but_verbatim_quote_does_not_support_the_claim(store):
    cid = chunk_id(store)
    res = run(store, ScriptedLLM(reply(cid, quote="Fuel costs rose during the year because of market prices.")), spec=STRICT)
    assert res.status == "unsupported" and "does not appear in the quote" in res.rejected[0]["reason"]


def test_amount_not_in_quote_is_blanked_but_the_item_is_kept(store):
    res = run(store, ScriptedLLM(reply(chunk_id(store), amount="$924 million")), spec=STRICT)
    assert res.status == "ok" and res.items[0]["amount"] == ""
    assert res.adjustments[0]["field"] == "amount" and res.adjustments[0]["was"] == "$924 million"


def test_zero_amount_is_not_matched_inside_other_numbers(store):
    # the quote contains 2029 and 2031: a naive substring test would accept "$0"
    res = run(store, ScriptedLLM(reply(chunk_id(store), amount="$0")), spec=STRICT)
    assert res.items[0]["amount"] == "" and len(res.adjustments) == 1


def test_amount_present_in_quote_or_blank_is_left_alone(store):
    cid = chunk_id(store)
    kept = run(store, ScriptedLLM(reply(cid, amount="2029")), spec=STRICT)
    assert kept.items[0]["amount"] == "2029" and kept.adjustments == []
    assert run(store, ScriptedLLM(reply(cid, amount="")), spec=STRICT).adjustments == []


COVENANT_SPEC = FieldSpec(
    "cov", Covenants, "restrictions", "List covenants.", ("7",), 3,
    quote_must_contain_any=("covenant", "restrict"), described_by_quote=("description",),
)
COVENANT_TEXT = "Our debt agreements contain covenants that may restrict our ability to incur additional indebtedness."


def cov_store(tmp_path):
    filing = ParsedFiling("AAA", "acc", "10-K", "2026-02-01", "", "", 3, [Section("7", "MD&A", 0, 2, COVENANT_TEXT + "\nOther text about fuel.")])
    s = VectorStore(tmp_path / "cov")
    build_index([filing], FakeEmbedder(), s, CFG)
    return s


def cov_reply(cid, description, quote=COVENANT_TEXT):
    return json.dumps({"items": [{"description": description, "chunk_id": cid, "quote": quote}]})


def test_covenant_description_must_be_supported_by_its_quote(tmp_path):
    s = cov_store(tmp_path)
    cid = retrieve(s, FakeEmbedder(), "restrict", 1)[0].chunk_id
    ok = run(s, ScriptedLLM(cov_reply(cid, "Restricts the ability to incur additional indebtedness.")), spec=COVENANT_SPEC)
    assert ok.status == "ok"
    bad = run(s, ScriptedLLM(cov_reply(cid, "Prohibits paying dividends or repurchasing company stock shares.")), spec=COVENANT_SPEC)
    assert bad.status == "unsupported" and "not supported by the quote" in bad.rejected[0]["reason"]


def test_covenant_quote_must_actually_state_a_covenant(tmp_path):
    s = cov_store(tmp_path)
    cid = retrieve(s, FakeEmbedder(), "restrict", 1)[0].chunk_id
    res = run(s, ScriptedLLM(cov_reply(cid, "Other text about fuel.", quote="Other text about fuel.")), spec=COVENANT_SPEC)
    assert res.status == "unsupported" and "does not state a covenant" in res.rejected[0]["reason"]


def test_company_policy_sentence_that_merely_says_maintain_is_not_a_covenant():
    spec = FIELDS["covenants"]
    quote = "The Company uses derivatives for hedging purposes, as well as to maintain an appropriate mix of fixed- and variable-rate debt."
    item = {"description": "Requires an appropriate mix of fixed- and variable-rate debt.", "chunk_id": "c", "quote": quote}
    assert claim_problem(spec, item) == "quote does not state a covenant or restriction"
    real = {"description": "Requires minimum liquidity of $2.0 billion.", "chunk_id": "c",
            "quote": "Our debt financing agreements contain covenants requiring us to maintain at least $2.0 billion of liquidity."}
    assert claim_problem(spec, real) is None


def test_real_covenant_spec_is_strict():
    spec = FIELDS["covenants"]
    assert spec.quote_must_contain_any and spec.described_by_quote == ("description",)
    assert "minimum" not in spec.quote_must_contain_any  # bare 'minimum' also matches "minimum quantities to be purchased"


def test_duplicate_quotes_are_kept_once_and_counted(store):
    cid = chunk_id(store)
    item = json.loads(reply(cid))["items"][0]
    res = run(store, ScriptedLLM(json.dumps({"items": [item, item | {"maturity": "June 2029"}]})))
    assert len(res.items) == 1 and res.duplicates_dropped == 1


def test_not_found_going_concern_has_any_invented_quote_removed(store):
    junk = json.dumps({"items": [{"status": "not_found", "chunk_id": chunk_id(store), "quote": "Our debt may limit us."}]})
    item = run(store, ScriptedLLM(junk), spec=FIELDS["going_concern"]).items[0]
    assert item == {"status": "not_found", "chunk_id": "", "quote": ""}


def test_debt_fields_skip_the_subsidiary_section_and_are_strict():
    assert "8B" not in FIELDS["debt_facilities"].section_keys and "8B" not in FIELDS["covenants"].section_keys
    assert FIELDS["debt_facilities"].claims_in_quote == ("instrument", "maturity")


def test_build_prompt_lists_every_hit(store):
    hits = retrieve(store, FakeEmbedder(), "credit facility", 3)
    prompt = build_prompt(SPEC, hits)
    assert all(f"[chunk_id={h.chunk_id}]" in prompt for h in hits) and prompt.startswith("List debt.")
