import pytest

from filings_credit_memo.facts import answer_fact, format_date, match_field
from filings_credit_memo.parse import ParsedFiling

QUESTION = "What fiscal year end date does this annual report cover?"


def filing(report_date: str = "2025-11-30") -> ParsedFiling:
    return ParsedFiling(
        "CCL", "0000815097-26-000010", "10-K", "2026-01-27", report_date, "https://sec.example/x.htm", 10, []
    )


@pytest.mark.parametrize(
    "question",
    [
        QUESTION,
        "What is the company's fiscal year end?",
        "Which fiscal year-end date is covered by this filing?",
        "When does the fiscal year end?",
        "what is the year end date",
    ],
)
def test_router_matches_clear_fiscal_year_end_questions(question):
    assert match_field(question) == "fiscal_year_end"


@pytest.mark.parametrize(
    "question",
    [
        "What were total revenues for the fiscal year ended November 30, 2024?",  # a specific year
        "What was net income for the most recent fiscal year?",
        "What were the company's total revenues at fiscal year end?",  # asks for revenue, not a date
        "Which board committee oversees cybersecurity risk?",
        "How many employees did the company have at fiscal year end?",
        "",
    ],
)
def test_router_leaves_everything_else_to_retrieval(question):
    assert match_field(question) is None


def test_format_date_has_no_leading_zero_and_rejects_bad_values():
    assert format_date("2026-01-05") == "January 5, 2026"
    assert format_date("2025-11-30") == "November 30, 2025"
    for bad in ("", "2025-13-01", "11/30/2025", None):
        assert format_date(bad) is None  # type: ignore[arg-type]


def test_answer_fact_returns_date_with_provenance():
    result = answer_fact(QUESTION, filing())
    assert result is not None
    assert result.answer == "November 30, 2025"
    assert "period of report 2025-11-30" in result.source and "0000815097-26-000010" in result.source
    assert result.url == "https://sec.example/x.htm"


def test_answer_fact_falls_back_when_metadata_is_bad_or_question_is_other():
    assert answer_fact(QUESTION, filing("unknown")) is None
    assert answer_fact("Which board committee oversees cybersecurity risk?", filing()) is None
