"""Tests for scoring, threshold calibration and the report. Fully offline (fake embedder, scripted model)."""

import json
import math
import re
import zlib

import pytest

from filings_credit_memo.chunk import ChunkConfig
from filings_credit_memo.evalset import EvalRow
from filings_credit_memo.evaluate import (
    RowResult,
    answer_matches,
    best_threshold,
    effective_status,
    evidence_in_text,
    main,
    outcome,
    parse_numbers,
    render_report,
    run_eval,
    score_row,
    summarize,
    two_fold_calibration,
    utility,
    wilson,
    write_results_csv,
)
from filings_credit_memo.parse import ParsedFiling, Section, save_parsed
from filings_credit_memo.qa import AnswerResult
from filings_credit_memo.retrieval import Hit, VectorStore, build_index

DIM = 128
EVIDENCE = "The company employed approximately 76,000 people as of December 31, 2025."


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

    def complete(self, system, user, schema=None):
        return self.replies.pop(0)


# ------------------------------------------------------------------------------------------------ matching
def test_evidence_exact_ignores_case_whitespace_and_curly_quotes():
    assert evidence_in_text("the company’s   plan", "x\nThe company's plan y")


def test_evidence_split_across_chunks_counts_when_most_of_it_is_present():
    evidence = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    assert evidence_in_text(evidence, "alpha beta gamma delta epsilon zeta eta theta")  # ~80%
    assert not evidence_in_text(evidence, "alpha beta gamma delta")
    assert not evidence_in_text("", "anything")


def test_parse_numbers_applies_scale():
    assert parse_numbers("$63.4 billion and 1,200 million, 5") == [63.4e9, 1.2e9, 5.0]


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        ("$63.4 billion", ("63.4 billion",), True),
        ("63,400 million", ("63.4 billion",), True),
        ("about 76,000 employees", ("76,000",), True),
        ("$63.4 billion", ("54 billion",), False),
        ("76,000", ("760,000",), False),
        ("No, there is no substantial doubt.", ("no",), True),
        ("I know of none", ("no",), False),
        ("Yes. The agreement requires minimum liquidity.", ("yes",), True),
        ("The facility matures in June 2029.", ("June 2029",), True),
        ("The facility matures in 2029.", ("June 2029",), False),
        ("Audit Committee.", ("audit committee", "Safety Committee"), True),
        ("Safety Committee", ("audit committee", "Safety Committee"), True),
        ("", ("no",), False),
    ],
)
def test_answer_matches(predicted, gold, expected):
    assert answer_matches(predicted, gold) is expected


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.237, abs=0.002) and hi == pytest.approx(0.763, abs=0.002)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert math.isnan(wilson(0, 0)[0])


# ------------------------------------------------------------------------------------------------ scoring
def make_hit(chunk_id="AAA-1-0000", score=0.8, text=EVIDENCE):
    return Hit(chunk_id, score, text, {"ticker": "AAA", "section_key": "1", "start_line": 1, "end_line": 2})


def row(answerable=True, gold=("76,000",), evidence=EVIDENCE, id="AAA-employees", ticker="AAA", field="employees"):
    return EvalRow(id, ticker, field, "How many employees?", answerable, gold if answerable else (),
                   evidence if answerable else "")


def result(status="answered", answer="76,000", chunk_id="AAA-1-0000", top=0.8):
    return AnswerResult("q", "AAA", status, answer, chunk_id, EVIDENCE if status == "answered" else "", "", 1, top)


def test_score_row_correct_answer_with_evidence_in_top_hit():
    hits = [make_hit(text="unrelated"), make_hit("AAA-1-0001")]
    r = score_row(row(), hits, result(chunk_id="AAA-1-0001"), 1.5, False)
    assert r.hit_rank == 2 and r.answer_correct is True and r.cited_has_evidence is True
    assert outcome(r, 0.65) == "correct"


def test_score_row_wrong_answer_and_citation_without_evidence():
    r = score_row(row(), [make_hit(text="unrelated")], result(answer="90,000"), 1.0, False)
    assert r.hit_rank is None and r.answer_correct is False and r.cited_has_evidence is False
    assert outcome(r, 0.65) == "wrong"


def test_unanswerable_rows_have_no_retrieval_metrics():
    r = score_row(row(answerable=False), [make_hit()], result(), 1.0, False)
    assert r.hit_rank is None and r.answer_correct is None
    assert outcome(r, 0.65) == "false_answer"


def test_outcomes_for_every_status():
    a, u = row(), row(answerable=False)
    for status, exp_a, exp_u in [
        ("insufficient_evidence", "wrongful_refusal", "correct_refusal"),
        ("not_found", "wrongful_refusal", "correct_refusal"),
        ("unsupported", "blocked", "blocked_unanswerable"),
        ("invalid_output", "blocked", "blocked_unanswerable"),
    ]:
        assert outcome(score_row(a, [make_hit()], result(status=status), 1, False), 0.65) == exp_a
        assert outcome(score_row(u, [make_hit()], result(status=status), 1, False), 0.65) == exp_u


def test_threshold_turns_an_answer_into_a_refusal():
    r = score_row(row(), [make_hit(score=0.6)], result(top=0.6), 1, False)
    assert effective_status(r, 0.5) == "answered" and outcome(r, 0.5) == "correct"
    assert effective_status(r, 0.65) == "refused" and outcome(r, 0.65) == "wrongful_refusal"


def rr(ticker, answerable, correct, top, status="answered", id=None):
    return RowResult(id or f"{ticker}-{top}", ticker, "f", "q", answerable, ("a",), status, "a", "quote", "", top,
                     1 if answerable else None, correct if answerable and status == "answered" else None,
                     True if answerable and status == "answered" else None, 2.0, False)


DATA = [
    rr("A", True, True, 0.78), rr("A", True, True, 0.72), rr("A", False, None, 0.62), rr("A", False, None, 0.60),
    rr("B", True, True, 0.75), rr("B", True, False, 0.58), rr("B", False, None, 0.64), rr("B", False, None, 0.55),
]


def test_utility_and_best_threshold_prefer_refusing_low_scoring_wrong_answers():
    assert utility(DATA, 0.50) == 3 - 2 * (1 + 4)  # 3 correct; 1 wrong + 4 false answers shown
    assert utility(DATA, 0.70) == 3
    assert 0.65 <= best_threshold(DATA) <= 0.72  # middle of the plateau, between the 0.64 and 0.72 scores


def test_two_fold_calibration_reports_out_of_fold_utility():
    cal = two_fold_calibration(DATA, current=0.65)
    assert cal is not None and cal["folds"] == [["A"], ["B"]]
    assert cal["oof_utility_current"] == utility(DATA, 0.65)  # current threshold is fixed, so folds add up
    assert two_fold_calibration([r for r in DATA if r.ticker == "A"], 0.65) is None


def test_summarize_counts_and_cached_rows_excluded_from_latency():
    data = [rr("A", True, True, 0.8), rr("A", False, None, 0.5)]
    data[1].cached = True
    s = summarize(data, 0.65)
    assert s["correct"] == 1 and s["correct_refusal"] == 1 and s["hit@1"] == 1 and s["hit@5"] == 1
    assert s["shown"] == 1 and s["unsupported_shown"] == 0 and s["latency_n"] == 1


# --------------------------------------------------------------------------------- end to end, offline
CFG = ChunkConfig(target_chars=300, overlap_chars=50, max_chars=400, min_section_chars=20)
FILING = ParsedFiling("AAA", "acc", "10-K", "2026-02-01", "", "", 10,
                      [Section("1", "Business", 0, 3, f"{EVIDENCE}\nFuel costs rose because of market prices.")])


def scripted_answer(store):
    from filings_credit_memo.retrieval import retrieve

    cid = retrieve(store, FakeEmbedder(), "How many employees?", 1, "AAA")[0].chunk_id
    return json.dumps({"status": "answered", "answer": "76,000", "chunk_id": cid, "quote": EVIDENCE})


def test_run_eval_end_to_end_report_and_csv(tmp_path):
    store = VectorStore(tmp_path / "idx")
    build_index([FILING], FakeEmbedder(), store, CFG)
    rows = [row(), row(answerable=False, id="AAA-oos", field="out_of_scope")]
    refusal = json.dumps({"status": "not_found", "answer": "", "chunk_id": "", "quote": ""})
    results = run_eval(rows, store, FakeEmbedder(), ScriptedLLM(scripted_answer(store), refusal), k=3)
    assert [outcome(r, 0.0) for r in results] == ["correct", "correct_refusal"]

    report = render_report(results, "scripted", 3, 0.65)
    assert "Retrieval hit@1" in report and "Answer accuracy" in report and "Limitations" in report
    assert "95% CI" in report and "Refusal threshold" in report and "Rows that need attention" in report
    path = tmp_path / "out" / "results.csv"
    write_results_csv(results, path, 0.65)
    assert path.read_text(encoding="utf-8-sig").splitlines()[0].startswith("id,ticker,field")


def test_report_with_two_companies_includes_the_two_fold_check():
    report = render_report(DATA, "scripted", 5, 0.65)
    assert "2-fold" in report and "Best threshold on all rows" in report and "| 0.65 (current) |" in report


def test_cli_check_reports_label_problems(tmp_path, capsys):
    labels = tmp_path / "l.csv"
    labels.write_text("id,ticker,field,question,answerable,gold_answer,gold_evidence,where_found,notes\n"
                      "A,AAL,f,Q?,maybe,,,,\n")
    assert main(["--labels", str(labels), "--sections", str(tmp_path), "check"]) == 2
    assert "yes or no" in capsys.readouterr().err


def test_cli_check_passes_on_valid_labels(tmp_path, capsys):
    save_parsed(ParsedFiling("AAL", "acc", "10-K", "2026-02-01", "", "", 10, FILING.sections), tmp_path / "sections")
    labels = tmp_path / "l.csv"
    labels.write_text("id,ticker,field,question,answerable,gold_answer,gold_evidence,where_found,notes\n"
                      f'AAL-e,AAL,employees,How many?,yes,"76,000","{EVIDENCE}",Item 1,\n')
    assert main(["--labels", str(labels), "--sections", str(tmp_path / "sections"), "check"]) == 0
    assert "Labels OK: 1 labeled rows" in capsys.readouterr().out


def test_cli_template_refuses_to_overwrite(tmp_path, capsys):
    labels = tmp_path / "l.csv"
    assert main(["--labels", str(labels), "template"]) == 0
    assert main(["--labels", str(labels), "template"]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
