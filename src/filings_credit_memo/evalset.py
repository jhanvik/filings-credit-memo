"""The hand-labeled evaluation set: file format, template, validation and a text dump to label from.

The labels are the most valuable part of the evaluation, and they only mean something if a person made them
independently of the system being tested. So:
  * the human reads the cleaned filing text (`export_text`), NOT the retrieval output;
  * gold evidence is a short passage copied word for word from that text, and `check_evidence` proves it is there;
  * rows whose `answerable` cell is empty are treated as not labeled yet and are skipped.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from filings_credit_memo.extract import _norm
from filings_credit_memo.parse import load_parsed

COLUMNS = ("id", "ticker", "field", "question", "answerable", "gold_answer", "gold_evidence", "where_found", "notes")
TICKERS = ("CCL", "DAL", "F", "DUK", "WMT", "AAL")
MIN_EVIDENCE_CHARS = 20
MAX_EVIDENCE_CHARS = 600
_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}

# (field, question). Same wording for every company so results are comparable; rewrite a question in the CSV if it
# does not fit a company (and say so in notes).
TEMPLATE_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("revenue", "What were the company's total revenues for the most recent fiscal year?"),
    ("employees", "How many people does the company employ?"),
    ("revolver_maturity", "When does the company's main revolving credit facility mature?"),
    (
        "covenants",
        "Do the company's debt agreements contain financial covenants, such as a minimum liquidity requirement?",
    ),
    ("cyber_oversight", "Which board committee oversees cybersecurity risk?"),
    (
        "going_concern",
        "Does the filing disclose substantial doubt about the company's ability to continue as a going concern?",
    ),
    ("out_of_scope", "What was the company's net income in fiscal 2019?"),
)


class EvalSetError(ValueError):
    """The labels file has problems. The message lists all of them so they can be fixed in one pass."""


@dataclass(frozen=True)
class EvalRow:
    id: str
    ticker: str
    field: str
    question: str
    answerable: bool
    gold_answers: tuple[str, ...]  # accepted alternatives, written in the CSV separated by |
    gold_evidence: str
    where_found: str = ""
    notes: str = ""


def write_template(path: Path, tickers: tuple[str, ...] = TICKERS, overwrite: bool = False) -> int:
    """Write the blank labeling sheet. Refuses to overwrite an existing file: that would destroy hours of labels."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; refusing to overwrite your labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for ticker in tickers:
            for field_name, question in TEMPLATE_QUESTIONS:
                n += 1
                writer.writerow([f"{ticker}-{field_name}", ticker, field_name, question, "", "", "", "", ""])
    return n


def load_rows(path: Path) -> tuple[list[EvalRow], int]:
    """Read and validate the labels. Returns (labeled rows, number of rows skipped as not labeled yet)."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise EvalSetError(f"missing column(s): {', '.join(missing_cols)}")
        raw = list(reader)

    rows: list[EvalRow] = []
    problems: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for number, rec in enumerate(raw, start=2):  # line 1 is the header
        get = {c: (rec.get(c) or "").strip() for c in COLUMNS}
        where = f"line {number} ({get['id'] or 'no id'})"
        flag = get["answerable"].lower()
        if not flag:
            if get["gold_answer"] or get["gold_evidence"]:
                problems.append(f"{where}: has an answer or evidence but 'answerable' is empty")
            skipped += 1
            continue
        if flag not in _YES | _NO:
            problems.append(f"{where}: answerable must be yes or no, not {get['answerable']!r}")
            continue
        answerable = flag in _YES
        if not get["id"]:
            problems.append(f"{where}: id is empty")
        elif get["id"] in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(get["id"])
        if get["ticker"].upper() not in TICKERS:
            problems.append(f"{where}: unknown ticker {get['ticker']!r}")
        if not get["question"]:
            problems.append(f"{where}: question is empty")
        answers = tuple(a.strip() for a in get["gold_answer"].split("|") if a.strip())
        evidence = get["gold_evidence"]
        if answerable:
            if not answers:
                problems.append(f"{where}: answerable=yes needs gold_answer")
            if not MIN_EVIDENCE_CHARS <= len(evidence) <= MAX_EVIDENCE_CHARS:
                problems.append(
                    f"{where}: gold_evidence must be {MIN_EVIDENCE_CHARS}-{MAX_EVIDENCE_CHARS} characters "
                    f"(a short passage copied from the text), it is {len(evidence)}"
                )
        elif answers or evidence:
            problems.append(
                f"{where}: answerable=no but gold_answer/gold_evidence are filled in; clear them or fix 'answerable'"
            )
        rows.append(
            EvalRow(
                get["id"], get["ticker"].upper(), get["field"], get["question"], answerable, answers,
                evidence, get["where_found"], get["notes"],
            )
        )
    if problems:
        raise EvalSetError(f"{len(problems)} problem(s) in {path}:\n  " + "\n  ".join(problems))
    return rows, skipped


def filing_text(sections_dir: Path, ticker: str) -> str:
    return "\n".join(s.text for s in load_parsed(Path(sections_dir) / f"{ticker}.json").sections)


def check_evidence(rows: list[EvalRow], sections_dir: Path) -> list[str]:
    """For answerable rows, confirm the gold evidence really is in that company's cleaned filing text (verbatim)."""
    problems: list[str] = []
    texts: dict[str, str] = {}
    for row in rows:
        if not row.answerable:
            continue
        if row.ticker not in texts:
            try:
                texts[row.ticker] = _norm(filing_text(sections_dir, row.ticker))
            except FileNotFoundError:
                texts[row.ticker] = ""
                problems.append(f"{row.ticker}: no parsed filing in {sections_dir}")
        if texts[row.ticker] and _norm(row.gold_evidence) not in texts[row.ticker]:
            problems.append(
                f"{row.id}: gold_evidence was not found verbatim in the {row.ticker} text "
                "(copy it again from the text file)"
            )
    return problems


def export_text(sections_dir: Path, out_dir: Path) -> list[Path]:
    """One plain-text file per company, with a divider before each Item, for reading and searching by hand."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file in sorted(Path(sections_dir).glob("*.json")):
        filing = load_parsed(file)
        parts = [f"{filing.ticker} {filing.form} filed {filing.filing_date}  ({filing.url})"]
        for section in filing.sections:
            parts.append(f"\n##### ITEM {section.key}: {section.title} #####\n{section.text}")
        path = out_dir / f"{filing.ticker}.txt"
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
