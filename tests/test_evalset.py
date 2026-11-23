"""Tests for the labeling sheet: template, validation, evidence check and text export."""

import csv

import pytest

from filings_credit_memo.evalset import (
    COLUMNS,
    TEMPLATE_QUESTIONS,
    TICKERS,
    EvalSetError,
    check_evidence,
    export_text,
    load_rows,
    write_template,
)
from filings_credit_memo.parse import ParsedFiling, Section, save_parsed

EVIDENCE = "The Company has approximately 76,000 employees worldwide."


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def good(**kw):
    base = {"id": "AAL-employees", "ticker": "AAL", "field": "employees", "question": "How many employees?",
            "answerable": "yes", "gold_answer": "76,000 | 76000", "gold_evidence": EVIDENCE}
    return base | kw


def test_template_has_every_company_and_question_and_is_blank(tmp_path):
    path = tmp_path / "eval" / "labels.csv"
    n = write_template(path)
    assert n == len(TICKERS) * len(TEMPLATE_QUESTIONS)
    rows, skipped = load_rows(path)
    assert rows == [] and skipped == n


def test_template_never_overwrites_labels(tmp_path):
    path = tmp_path / "labels.csv"
    write_template(path)
    with pytest.raises(FileExistsError):
        write_template(path)
    write_template(path, overwrite=True)


def test_valid_rows_load_with_alternatives_split(tmp_path):
    path = tmp_path / "l.csv"
    write_csv(path, [good(), good(id="AAL-x", answerable="NO", gold_answer="", gold_evidence="", question="Q?"),
                     {"id": "AAL-blank", "ticker": "AAL", "question": "Q?"}])
    rows, skipped = load_rows(path)
    assert skipped == 1 and [r.answerable for r in rows] == [True, False]
    assert rows[0].gold_answers == ("76,000", "76000") and rows[0].ticker == "AAL"


def test_all_problems_are_reported_together(tmp_path):
    path = tmp_path / "l.csv"
    write_csv(path, [
        good(answerable="maybe"),
        good(id="AAL-a", gold_answer=""),
        good(id="AAL-b", gold_evidence="too short"),
        good(id="AAL-c", answerable="no"),
        good(id="AAL-d", ticker="XYZ"),
        good(id="AAL-d"),
        {"id": "AAL-e", "ticker": "AAL", "question": "Q?", "gold_answer": "5"},
    ])
    with pytest.raises(EvalSetError) as err:
        load_rows(path)
    msg = str(err.value)
    for needle in ("yes or no", "needs gold_answer", "must be 20-600", "answerable=no but", "unknown ticker",
                   "duplicate id", "'answerable' is empty"):
        assert needle in msg


def test_missing_column_is_reported(tmp_path):
    path = tmp_path / "l.csv"
    path.write_text("id,ticker\nA,AAL\n")
    with pytest.raises(EvalSetError, match="missing column"):
        load_rows(path)


def make_sections(tmp_path):
    filing = ParsedFiling("AAL", "acc", "10-K", "2026-02-01", "", "http://x", 5,
                          [Section("1", "Business", 0, 2, f"Intro line.\n{EVIDENCE}\nMore."),
                           Section("7", "MD&A", 3, 4, "Revenue was 54 billion.")])
    save_parsed(filing, tmp_path / "sections")
    return tmp_path / "sections"


def test_evidence_check_accepts_verbatim_and_ignores_whitespace_and_quotes(tmp_path):
    sections = make_sections(tmp_path)
    path = tmp_path / "l.csv"
    write_csv(path, [good(gold_evidence="The Company has   approximately 76,000\nemployees worldwide.")])
    rows, _ = load_rows(path)
    assert check_evidence(rows, sections) == []


def test_evidence_check_flags_text_that_is_not_in_the_filing(tmp_path):
    sections = make_sections(tmp_path)
    path = tmp_path / "l.csv"
    write_csv(path, [good(gold_evidence="The Company has about 90,000 employees worldwide."),
                     good(id="DAL-e", ticker="DAL")])
    rows, _ = load_rows(path)
    problems = check_evidence(rows, sections)
    assert any("AAL-employees" in p and "verbatim" in p for p in problems)
    assert any("DAL" in p and "no parsed filing" in p for p in problems)


def test_unanswerable_rows_are_not_evidence_checked(tmp_path):
    path = tmp_path / "l.csv"
    write_csv(path, [good(answerable="no", gold_answer="", gold_evidence="")])
    rows, _ = load_rows(path)
    assert check_evidence(rows, tmp_path / "nowhere") == []


def test_export_text_writes_one_file_per_company_with_item_dividers(tmp_path):
    sections = make_sections(tmp_path)
    paths = export_text(sections, tmp_path / "txt")
    assert [p.name for p in paths] == ["AAL.txt"]
    text = paths[0].read_text()
    assert "##### ITEM 1: Business #####" in text and EVIDENCE in text and "##### ITEM 7: MD&A #####" in text
