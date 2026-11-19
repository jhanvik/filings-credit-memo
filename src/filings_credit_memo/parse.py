"""Clean an EDGAR 10-K (inline-XBRL HTML) into text lines and split it into its "Items".

Why this is harder than it looks (all observed in real filings, not assumed):
  * The file starts with a hidden inline-XBRL header full of machine tags that is not part of the report.
  * Every Item heading shows up in the table of contents AND in the body.
  * Page headers repeat ("Item 1A. Risk Factors (Continued)") and sentences cross-reference other Items.
  * Formatting varies per company: "Item 1A." vs "ITEM 1A. RISK FACTORS", heading and title on one line or two.

Strategy (each step is covered by a test):
  1. Strip hidden XBRL / scripts, keep inline text together, keep each table row on one line ("a | b | c").
  2. Find heading candidates: a line that starts with "Item <n>[A-C]" and whose remainder is empty or looks like the
     official title of that Item. Lines saying "(Continued)" are dropped.
  3. Skip the table of contents by locating the first "Item 1" candidate followed by a long stretch of text.
  4. Among the remaining candidates choose the best chain that follows the official Item order, preferring
     upper-case "ITEM" headings. This rejects page headers and cross-references.
The official titles below are written from knowledge of Form 10-K, not copied from sec.gov: verify against the form.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
# Keys that are not in the standard list or whose title varies by company: any remainder text is accepted.
FREE_TITLE_KEYS = {"4A", "6", "8A", "8B"}
PREAMBLE_KEY = "0"
PREAMBLE_TITLE = "Cover page and table of contents"
REQUIRED_KEYS = ("1", "1A", "7", "7A")
# Heuristic minimum sizes (characters) for sections that are long in a real 10-K. Tune after looking at data.
MIN_CHARS = {"1": 5_000, "1A": 5_000, "7": 5_000}
MIN_STATEMENTS_CHARS = 5_000
STATEMENTS_KEY = "FS"

MIN_BODY_CHARS = 3_000
MAX_HEADING_CHARS = 250
HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
BLOCK_TAGS = ["p", "div", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "section", "center", "hr"]
_WS = re.compile(r"[\s​﻿]+")
_PAGE_NUMBER = re.compile(r"^(?:[A-Z]-)?\d{1,3}$")  # a line holding only a page number such as "22" or "F-7"
_AUDIT_REPORT = re.compile(r"^report of independent registered public accounting firm\.?$", re.I)
_ITEM = re.compile(r"^item\s*(\d{1,2})([a-c])?\b\s*[.:\-–—]?\s*(.*)$", re.I)


class ParseError(RuntimeError):
    """The filing does not look the way this parser expects."""


# ---- step 1: HTML to clean lines ---------------------------------------------------------------------------------


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _tidy_row(row: str) -> str:
    row = re.sub(r"\$ \| ", "$", row)
    row = re.sub(r" \| \)", ")", row)
    return re.sub(r" \| %", "%", row)


def html_to_lines(html: bytes | str) -> list[str]:
    """Visible text of a filing as non-empty, whitespace-normalised lines. Table rows become 'cell | cell | cell'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    def drop(tags: list) -> None:
        for tag in tags:
            if not getattr(tag, "decomposed", False):
                tag.decompose()

    drop(soup.find_all(["ix:header", "script", "style", "head"]))
    drop(soup.find_all(style=HIDDEN_STYLE))

    for table in soup.find_all("table"):
        if getattr(table, "decomposed", False):
            continue
        rows = []
        for tr in table.find_all("tr"):
            cells = [c for c in (_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])) if c]
            if cells:
                rows.append(_tidy_row(" | ".join(cells)))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")

    lines = (_clean(raw) for raw in soup.get_text().split("\n"))
    return [line for line in lines if line and not _PAGE_NUMBER.match(line)]


# ---- step 2: heading candidates ----------------------------------------------------------------------------------


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _order(key: str) -> tuple[int, str]:
    digits = re.match(r"\d+", key)
    assert digits is not None
    return int(digits.group()), key[digits.end() :]


@dataclass(frozen=True)
class Candidate:
    line: int
    key: str
    remainder: str
    weight: float  # 2.0 for an upper-case "ITEM" heading, 1.0 otherwise


def _remainder_ok(key: str, remainder: str) -> bool:
    if not remainder or key in FREE_TITLE_KEYS:
        return True
    title = ITEM_TITLES.get(key)
    return title is not None and _alnum(remainder)[:6] == _alnum(title)[:6]


def find_item_candidates(lines: list[str]) -> list[Candidate]:
    out = []
    for i, line in enumerate(lines):
        if len(line) > MAX_HEADING_CHARS or "(continued)" in line.lower():
            continue
        match = _ITEM.match(line)
        if not match:
            continue
        key = match.group(1) + (match.group(2) or "").upper()
        if key not in ITEM_TITLES and key not in FREE_TITLE_KEYS:
            continue
        remainder = match.group(3).strip()
        if _remainder_ok(key, remainder):
            out.append(Candidate(i, key, remainder, 2.0 if line[:4] == "ITEM" else 1.0))
    return out


# ---- steps 3 and 4: pick the real headings -----------------------------------------------------------------------


def _find_body_start(cands: list[Candidate], cum: list[int], min_body_chars: int) -> int:
    """Line of the first 'Item 1' candidate followed by a long stretch of text (a table-of-contents entry is not)."""
    for idx, cand in enumerate(cands):
        if cand.key != "1":
            continue
        nxt = next((c for c in cands[idx + 1 :] if c.key != "1"), None)
        end = nxt.line if nxt else len(cum) - 1
        if cum[end] - cum[cand.line] >= min_body_chars:
            return cand.line
    raise ParseError("could not find the body of Item 1 (no 'Item 1' heading followed by a long section)")


def _select_chain(cands: list[Candidate]) -> list[Candidate]:
    """Best-scoring chain of candidates whose Items are in strictly increasing official order (earlier wins ties)."""
    n = len(cands)
    best = [0.0] * n
    prev = [-1] * n
    for i, c in enumerate(cands):
        score = c.weight - 1e-6 * i
        best[i] = score
        for j in range(i):
            if _order(cands[j].key) < _order(c.key) and best[j] + score > best[i]:
                best[i] = best[j] + score
                prev[i] = j
    if not n:
        return []
    i = max(range(n), key=lambda k: best[k])
    chain = []
    while i != -1:
        chain.append(cands[i])
        i = prev[i]
    return chain[::-1]


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    start_line: int
    end_line: int  # exclusive
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _relocate_statements(sections: list[Section], lines: list[str]) -> list[Section]:
    """Some companies (Ford) put the financial statements after the signature pages and leave Item 8 as a pointer.

    If Item 8 is tiny, split off everything from the first auditor's report found in a later section into its own
    section (key 'FS') so the statements and notes are not buried inside "Item 16".
    """
    item8 = next((s for s in sections if s.key == "8"), None)
    if item8 is None or item8.n_chars >= MIN_STATEMENTS_CHARS:
        return sections
    for idx, sec in enumerate(sections):
        if sec.start_line <= item8.start_line:
            continue
        hit = next((n for n in range(sec.start_line, sec.end_line) if _AUDIT_REPORT.match(lines[n])), None)
        if hit is None:
            continue
        head = Section(sec.key, sec.title, sec.start_line, hit, "\n".join(lines[sec.start_line : hit]))
        title = f"Financial statements and notes (found after Item {sec.key})"
        tail = Section(STATEMENTS_KEY, title, hit, sec.end_line, "\n".join(lines[hit : sec.end_line]))
        return [*sections[:idx], head, tail, *sections[idx + 1 :]]
    return sections


def split_sections(lines: list[str], min_body_chars: int = MIN_BODY_CHARS) -> list[Section]:
    cands = find_item_candidates(lines)
    cum = [0]
    for line in lines:
        cum.append(cum[-1] + len(line) + 1)
    start = _find_body_start(cands, cum, min_body_chars)
    chain = _select_chain([c for c in cands if c.line >= start])

    sections = []
    if chain[0].line > 0:
        sections.append(Section(PREAMBLE_KEY, PREAMBLE_TITLE, 0, chain[0].line, "\n".join(lines[: chain[0].line])))
    for pos, cand in enumerate(chain):
        end = chain[pos + 1].line if pos + 1 < len(chain) else len(lines)
        title = ITEM_TITLES.get(cand.key) or cand.remainder or (lines[cand.line + 1] if cand.line + 1 < end else "")
        sections.append(Section(cand.key, title, cand.line, end, "\n".join(lines[cand.line : end])))
    return _relocate_statements(sections, lines)


# ---- whole filings -----------------------------------------------------------------------------------------------


@dataclass
class ParsedFiling:
    ticker: str
    accession: str
    form: str
    filing_date: str
    report_date: str
    url: str
    n_lines: int
    sections: list[Section]

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)


def parse_filing(html_path: Path) -> ParsedFiling:
    """Parse one downloaded filing. Metadata comes from the .json file the downloader saved next to it."""
    html_path = Path(html_path)
    meta_path = html_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    lines = html_to_lines(html_path.read_bytes())
    return ParsedFiling(
        ticker=meta.get("ticker", html_path.parent.name),
        accession=meta.get("accession", html_path.stem),
        form=meta.get("form", ""),
        filing_date=meta.get("filing_date", ""),
        report_date=meta.get("report_date", ""),
        url=meta.get("url", ""),
        n_lines=len(lines),
        sections=split_sections(lines),
    )


def save_parsed(parsed: ParsedFiling, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{parsed.ticker}.json"
    path.write_text(json.dumps(asdict(parsed), indent=1, ensure_ascii=False))
    return path


def load_parsed(path: Path) -> ParsedFiling:
    data = json.loads(Path(path).read_text())
    data["sections"] = [Section(**s) for s in data["sections"]]
    return ParsedFiling(**data)


def sanity_problems(parsed: ParsedFiling) -> list[str]:
    """Plain-English problems worth a human look. An empty list means nothing looked wrong (not that it is perfect)."""
    problems = []
    for key in REQUIRED_KEYS:
        section = parsed.section(key)
        if section is None:
            problems.append(f"Item {key} not found")
        elif key in MIN_CHARS and section.n_chars < MIN_CHARS[key]:
            problems.append(f"Item {key} is only {section.n_chars:,} chars (expected at least {MIN_CHARS[key]:,})")
    statements = [s for s in parsed.sections if s.key in {"8", "8A", "8B", STATEMENTS_KEY}]
    if not any(s.n_chars >= MIN_STATEMENTS_CHARS for s in statements):
        problems.append("no financial statements section of reasonable size found (Item 8 / 8A / 8B / FS)")
    return problems


# ---- command line ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean downloaded 10-Ks and split them into Items.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/filings"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sections"))
    args = parser.parse_args(argv)

    html_files = sorted(args.raw.glob("*/*.htm"))
    if not html_files:
        print(f"No .htm filings under {args.raw}. Run the downloader first.", file=sys.stderr)
        return 2
    failures = 0
    for html_path in html_files:
        try:
            parsed = parse_filing(html_path)
        except ParseError as err:
            failures += 1
            print(f"{html_path.parent.name:5s} FAILED: {err}", file=sys.stderr)
            continue
        save_parsed(parsed, args.out)
        print(f"\n{parsed.ticker}  {parsed.form} filed {parsed.filing_date}  ({parsed.n_lines:,} lines)")
        for s in parsed.sections:
            print(f"  Item {s.key:>3s}  line {s.start_line:>6,}  {s.n_chars:>9,} chars  {s.title[:50]}")
        for problem in sanity_problems(parsed):
            failures += 1
            print(f"  !! {problem}")
    print(f"\nProblems flagged: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
