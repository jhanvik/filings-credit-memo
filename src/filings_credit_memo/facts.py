"""Structured facts answered from filing metadata, not from retrieval or a language model.

Some questions have an exact, machine-readable answer: the fiscal year end is the filing's "period of report". Asking an
embedding search to find it on the cover page was unreliable (the cover chunk did not rank), and a model can grab a
prior-year date from a financial note. So the router sends only these questions here and everything else to retrieval.

The router is deliberately narrow: it answers only a clear "what is the fiscal year end date" question, and returns
None (so the normal pipeline runs) whenever the question mentions a specific year or the metadata is missing or odd.
The report_date field is assumed to be the period of report from EDGAR: verify that on sec.gov for your filings.

    uv run python -m filings_credit_memo.facts check
    uv run python -m filings_credit_memo.facts ask AAL "What fiscal year end date does this annual report cover?"
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from filings_credit_memo.evalset import load_rows
from filings_credit_memo.evaluate import answer_matches
from filings_credit_memo.parse import ParsedFiling, load_parsed

_FISCAL_YEAR_END = re.compile(
    r"\b(?:what|which|when)\b.*\b(?:fiscal[\s-]+year[\s-]+end(?:s|ed)?(?:\s+date)?|year[\s-]+end\s+date)\b", re.I
)
_SPECIFIC_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NOT_A_DATE_QUESTION = re.compile(
    r"\b(?:revenue|revenues|income|earnings|debt|cash|assets|liabilit\w*|employees|compensation|shares|dividends?)\b",
    re.I,
)


@dataclass(frozen=True)
class FactAnswer:
    field: str
    answer: str
    source: str  # human-readable provenance, shown wherever a citation would be
    url: str


def match_field(question: str) -> str | None:
    """The structured field this question asks for, or None to leave the question to retrieval."""
    if _SPECIFIC_YEAR.search(question) or _NOT_A_DATE_QUESTION.search(question):
        return None
    return "fiscal_year_end" if _FISCAL_YEAR_END.search(question) else None


def format_date(iso: str) -> str | None:
    """'2025-11-30' -> 'November 30, 2025'. None if the value is not a real ISO date."""
    try:
        d = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    return f"{d:%B} {d.day}, {d.year}"


def answer_fact(question: str, filing: ParsedFiling) -> FactAnswer | None:
    if match_field(question) != "fiscal_year_end":
        return None
    answer = format_date(filing.report_date)
    if answer is None:
        return None  # never guess: fall back to the normal pipeline
    source = (
        f"Filing metadata: period of report {filing.report_date} "
        f"({filing.ticker} {filing.form}, accession {filing.accession}, filed {filing.filing_date})"
    )
    return FactAnswer("fiscal_year_end", answer, source, filing.url)


def load_filings(sections_dir: Path) -> dict[str, ParsedFiling]:
    return {f.ticker.upper(): f for f in (load_parsed(p) for p in sorted(Path(sections_dir).glob("*.json")))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer structured questions from filing metadata.")
    parser.add_argument("--sections", type=Path, default=Path("data/processed/sections"))
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="score the labeled fiscal_year_end rows against the metadata answer")
    check.add_argument(
        "--labels", type=Path, nargs="+", default=[Path("eval/labels.csv"), Path("eval/labels_holdout.csv")]
    )
    ask = sub.add_parser("ask")
    ask.add_argument("ticker")
    ask.add_argument("question")
    args = parser.parse_args(argv)

    filings = load_filings(args.sections)
    if args.command == "ask":
        filing = filings.get(args.ticker.upper())
        result = answer_fact(args.question, filing) if filing else None
        print(
            f"{result.answer}\n{result.source}\n{result.url}"
            if result
            else "Not a structured question (or no metadata)."
        )
        return 0

    rows = [r for path in args.labels for r in load_rows(path)[0] if r.field == "fiscal_year_end" and r.answerable]
    correct = 0
    for row in rows:
        filing = filings.get(row.ticker.upper())
        result = answer_fact(row.question, filing) if filing else None
        ok = bool(result and answer_matches(result.answer, row.gold_answers))
        correct += ok
        shown = result.answer if result else None
        print(f"{row.id}: {'correct' if ok else 'WRONG'}  answer={shown!r}  gold={row.gold_answers}")
    print(f"\nMetadata route: {correct}/{len(rows)} correct on labeled fiscal_year_end rows (not a RAG result).")
    return 0 if correct == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
