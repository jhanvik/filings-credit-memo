"""Credit memo generator: a fixed template filled only with verified, cited facts.

The memo has three kinds of content, and each says where it came from:
  1. Filing metadata (company, form, filing date, fiscal year end): exact values from the filing record, no model.
  2. Structured sections (risk factors, debt, covenants, going concern): items that passed the quote check in
     extract.py. Every item carries a chunk id and a verbatim quote.
  3. A few cited questions (dividends, holders of record): answered by the frozen QA pipeline, or "not determined".

What the memo never does: give a rating, a recommendation or investment advice (the template has no place for one),
or fill a gap with a guess. Missing evidence is stated as missing. Filing text and model output are untrusted, so
everything that came from them is cleaned before it is placed in the markdown (no links, images or HTML).

    uv run python -m filings_credit_memo.memo CCL
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from filings_credit_memo.extract import DEFAULT_MIN_SCORE, FIELDS, FieldResult, extract_field
from filings_credit_memo.facts import answer_fact, load_filings
from filings_credit_memo.llm import DEFAULT_MODEL, LLM, CachedLLM, LLMError, OllamaLLM
from filings_credit_memo.parse import ParsedFiling
from filings_credit_memo.qa import AnswerResult, answer_question
from filings_credit_memo.retrieval import Embedder, VectorStore

FISCAL_YEAR_QUESTION = "What fiscal year end date does this annual report cover?"
DEFAULT_QUESTIONS = (
    "What was the total amount of cash dividends the company paid on its common stock in its most recent fiscal year?",
    "How many holders of record does the company's common stock have?",
)
SECTION_TITLES = {
    "risk_factors": "Risk factors described by the company",
    "debt_facilities": "Debt instruments and credit facilities",
    "covenants": "Covenants and restrictions",
    "going_concern": "Going-concern language",
}
PARTIAL_NOTE = "This list is partial: the model reads only the few best-matching passages, so items may be missing."
MISSING_NOTE = (
    "Nothing verified was found in the retrieved text. This does not mean the filing is silent on the topic: "
    "the search or the model may have missed it. Check the filing."
)
DISCLAIMER = (
    "This memo was generated automatically from the company's SEC filing by a small local language model. It is a "
    "reading aid, not a credit opinion, rating or investment advice, and it can be wrong or incomplete. Each item "
    "is checked only for a verbatim quote in the cited text; the check does not prove that the quote answers the "
    "question. Verify every item against the filing before relying on it."
)

_MARKUP = re.compile(r"[<>\[\]`*_|#]")


def clean(text: str, limit: int = 600) -> str:
    """Make untrusted text safe to place in markdown: one line, no links, images, HTML or emphasis characters."""
    flat = " ".join(str(text).split())
    flat = _MARKUP.sub("", flat)
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


@dataclass
class Entry:
    text: str
    chunk_id: str = ""
    quote: str = ""


@dataclass
class Section:
    key: str
    title: str
    status: str  # ok | missing | withheld | error
    entries: list[Entry] = field(default_factory=list)
    note: str = ""


@dataclass
class Memo:
    ticker: str
    header: dict[str, str]
    sections: list[Section]
    evidence: dict[str, dict[str, Any]]  # chunk_id -> {"text", "section", "lines"} for every cited chunk
    model: str = ""
    min_score: float = DEFAULT_MIN_SCORE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memo:
        sections = [
            Section(s["key"], s["title"], s["status"], [Entry(**e) for e in s.get("entries", [])], s.get("note", ""))
            for s in data["sections"]
        ]
        return cls(
            data["ticker"], data["header"], sections, data.get("evidence", {}),
            data.get("model", ""), data.get("min_score", DEFAULT_MIN_SCORE),
        )


def load_memo(path: Path) -> Memo:
    return Memo.from_dict(json.loads(Path(path).read_text()))


def saved_memos(directory: Path) -> dict[str, Path]:
    """Ticker -> JSON path for every memo already generated (generating one needs the local model)."""
    return {p.stem.upper(): p for p in sorted(Path(directory).glob("*.json"))}


Extractor = Callable[..., FieldResult]
Answerer = Callable[..., AnswerResult]


def header_for(filing: ParsedFiling) -> dict[str, str]:
    fye = answer_fact(FISCAL_YEAR_QUESTION, filing)
    return {
        "Company ticker": filing.ticker.upper(),
        "Form": filing.form,
        "Filed": filing.filing_date,
        "Fiscal year end": fye.answer if fye else "not available",
        "Source": filing.url,
        "Accession": filing.accession,
    }


def _repeats(a: str, b: str) -> bool:
    """True when two texts start the same way (case and spacing ignored): the second adds nothing."""
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    return bool(na) and na[:60] == nb[:60]


_MATURITY_CUE = re.compile(r"\b(?:due|matur\w*|expir\w*)\b|\||\bthrough\b", re.IGNORECASE)
_RANGE = re.compile(r"\b(?:19|20)\d{2}\b\W+to\W+(?:19|20)\d{2}\b", re.IGNORECASE)
_ENDED = re.compile(r"\b(?:repaid|terminated|redeemed|retired)\b", re.IGNORECASE)


def _debt_text(item: dict[str, Any]) -> str:
    """Describe a debt item without claiming more than the quote shows.

    The quote check proves the date text appears in the quote, not that it is a maturity date: a model can copy the
    month a loan was signed or terminated. So "matures" is used only when the quote has a maturity cue (due, matures,
    expires, a table row); otherwise the date is labeled as just a date in the quote. Quotes that say the facility
    was repaid or terminated, or that show a range of maturities, are flagged. These are cheap text checks, not proof.
    """
    quote = item.get("quote", "")
    details = []
    if item.get("maturity"):
        label = "matures" if _MATURITY_CUE.search(quote) else "date in quote, not confirmed as maturity:"
        details.append(f"{label} {clean(item['maturity'], 80)}")
    if item.get("amount"):
        details.append(f"amount {clean(item['amount'], 80)}")
    if _ENDED.search(quote):
        details.append("quote says repaid or terminated")
    if _RANGE.search(quote):
        details.append("quote shows a range of maturities")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"{clean(item['instrument'], 200)}{suffix}"


def _entry_for(field_name: str, item: dict[str, Any]) -> Entry:
    quote = item.get("quote", "")
    cid = item.get("chunk_id", "")
    if field_name == "risk_factors":
        title, summary = clean(item["title"], 150), clean(item["summary"])
        if _repeats(summary, quote) or _repeats(summary, title):  # a small model often copies the quote as its summary
            return Entry(title, cid, quote)
        return Entry(f"{title}: {summary}", cid, quote)
    if field_name == "debt_facilities":
        return Entry(_debt_text(item), cid, quote)
    if field_name == "covenants":
        return Entry(clean(item["description"]), cid, quote)
    return Entry(clean(str(item)), cid, quote)


_GOING_CONCERN_TEXT = {
    "doubt_expressed": "The company expresses substantial doubt about its ability to continue as a going concern.",
    "no_doubt_stated": "The retrieved text does not express substantial doubt about continuing as a going concern.",
}


def section_from_result(res: FieldResult) -> Section:
    title = SECTION_TITLES.get(res.field, res.field)
    if res.status == "invalid_output":
        return Section(res.field, title, "error", note="The model did not return valid output.")
    if res.status in ("insufficient_evidence", "empty"):
        return Section(res.field, title, "missing", note=MISSING_NOTE)
    entries: list[Entry] = []
    if res.field == "going_concern":
        for item in res.items:
            text = _GOING_CONCERN_TEXT.get(item.get("status", ""))
            if text:
                entries.append(Entry(text, item.get("chunk_id", ""), item.get("quote", "")))
    else:
        entries = [_entry_for(res.field, item) for item in res.items]
    if entries:
        notes = [] if res.field == "going_concern" else [PARTIAL_NOTE]
        if res.rejected:
            notes.append(f"{len(res.rejected)} candidate item(s) failed verification and were withheld.")
        return Section(res.field, title, "ok", entries, " ".join(notes))
    if res.status == "unsupported":
        return Section(res.field, title, "withheld", note="Every candidate item failed verification and was withheld.")
    return Section(res.field, title, "missing", note=MISSING_NOTE)


def section_from_answer(res: AnswerResult) -> Section:
    title = clean(res.question, 200)
    if res.status == "answered":
        return Section("qa", title, "ok", [Entry(clean(res.answer, 300), res.chunk_id, res.quote)])
    if res.status == "unsupported":
        return Section("qa", title, "withheld", note="The model's answer failed verification and was withheld.")
    if res.status == "invalid_output":
        return Section("qa", title, "error", note="The model did not return valid output.")
    return Section("qa", title, "missing", note="Not determined from the retrieved text. Check the filing.")


def _evidence(store: VectorStore, ticker: str, sections: list[Section]) -> dict[str, dict[str, Any]]:
    wanted = {e.chunk_id for s in sections for e in s.entries if e.chunk_id}
    if not wanted:
        return {}
    hits, _ = store.get_all(ticker)
    return {
        h.chunk_id: {"text": h.text, "section": h.section_key, "lines": h.lines}
        for h in hits
        if h.chunk_id in wanted
    }


def build_memo(
    filing: ParsedFiling,
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    questions: tuple[str, ...] = DEFAULT_QUESTIONS,
    min_score: float = DEFAULT_MIN_SCORE,
    extractor: Extractor = extract_field,
    answerer: Answerer = answer_question,
) -> Memo:
    ticker = filing.ticker.upper()
    sections: list[Section] = []
    for name, spec in FIELDS.items():
        try:
            sections.append(section_from_result(extractor(spec, ticker, store, embedder, llm, min_score)))
        except LLMError as err:
            sections.append(Section(name, SECTION_TITLES.get(name, name), "error", note=f"Model problem: {clean(str(err), 200)}"))
    for question in questions:
        try:
            sections.append(section_from_answer(answerer(store, embedder, llm, ticker, question, min_score=min_score)))
        except LLMError as err:
            sections.append(Section("qa", clean(question, 200), "error", note=f"Model problem: {clean(str(err), 200)}"))
    return Memo(
        ticker, header_for(filing), sections, _evidence(store, ticker, sections),
        model=getattr(llm, "name", ""), min_score=min_score,
    )


def render_markdown(memo: Memo) -> str:
    out = [f"# Credit memo: {memo.ticker}", "", f"> {DISCLAIMER}", "", "## Filing"]
    out += [f"- **{label}:** {clean(value, 300)}" for label, value in memo.header.items()]
    out += ["", "Fiscal year end comes from the filing record, not from the language model.", ""]
    qa_started = False
    for sec in memo.sections:
        if sec.key == "qa" and not qa_started:
            out += ["## Specific questions", ""]
            qa_started = True
        elif sec.key != "qa":
            out += [f"## {sec.title}", ""]
        if sec.key == "qa":
            out += [f"**{sec.title}**", ""]
        if sec.status == "ok":
            for e in sec.entries:
                cite = f" (source `{e.chunk_id}`)" if e.chunk_id else ""
                out.append(f"- {e.text}{cite}")
                if e.quote:
                    out.append(f"  > {clean(e.quote, 400)}")
        else:
            out.append(f"_{sec.status.upper()}: {sec.note}_")
        if sec.status == "ok" and sec.note:
            out.append(f"_{sec.note}_")
        out.append("")
    out += ["---", f"Model: {memo.model or 'unknown'}. Retrieval similarity threshold: {memo.min_score}."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a cited credit memo for one company from the local index.")
    parser.add_argument("ticker")
    parser.add_argument("--sections", type=Path, default=Path("data/processed/sections"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/llm"))
    parser.add_argument("--out", type=Path, default=Path("data/outputs/memos"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args(argv)

    filings = load_filings(args.sections)
    filing = filings.get(args.ticker.upper())
    if filing is None:
        print(f"No parsed filing for {args.ticker}. Known: {', '.join(sorted(filings))}", file=sys.stderr)
        return 2
    store = VectorStore(args.index)
    if store.count() == 0:
        print("The index is empty. Run: uv run python -m filings_credit_memo.index build", file=sys.stderr)
        return 2

    from filings_credit_memo.retrieval import SentenceTransformerEmbedder  # heavy import, only for the real run

    llm = CachedLLM(OllamaLLM(args.model), args.cache)
    try:
        memo = build_memo(filing, store, SentenceTransformerEmbedder(), llm, min_score=args.min_score)
    except LLMError as err:
        print(f"Model problem: {err}", file=sys.stderr)
        return 3
    args.out.mkdir(parents=True, exist_ok=True)
    md_path, json_path = args.out / f"{memo.ticker}.md", args.out / f"{memo.ticker}.json"
    md_path.write_text(render_markdown(memo))
    json_path.write_text(json.dumps(memo.to_dict(), indent=1))
    print(render_markdown(memo))
    print(f"Saved {md_path} and {json_path}  (real model calls this run: {llm.calls})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
