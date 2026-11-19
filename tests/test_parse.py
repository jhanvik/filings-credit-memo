"""Tests for the filing cleaner and Item splitter.

Most tests use tiny synthetic documents so they run offline in milliseconds. The last block runs against the real
downloaded filings and is skipped when data/raw/filings is not present (the data folder is git-ignored).
"""

import json
from pathlib import Path

import pytest

from filings_credit_memo.parse import (
    ParseError,
    find_item_candidates,
    html_to_lines,
    load_parsed,
    parse_filing,
    sanity_problems,
    save_parsed,
    split_sections,
)

BODY = "x" * 200  # one filler "paragraph"


def filler(n: int = 5) -> list[str]:
    return [BODY] * n


def keys(sections) -> list[str]:
    return [s.key for s in sections]


def starts(sections) -> dict[str, int]:
    return {s.key: s.start_line for s in sections}


# ---- html_to_lines -----------------------------------------------------------------------------------------------


def test_strips_hidden_xbrl_header_scripts_and_hidden_blocks():
    html = (
        '<?xml version="1.0"?><html><head><title>t</title></head><body>'
        '<div style="display:none"><ix:header><xbrli:context>SECRET-XBRL</xbrli:context></ix:header></div>'
        "<script>var hidden = 1;</script>"
        '<div style="display: none">HIDDEN-BLOCK</div>'
        "<div>Visible text</div></body></html>"
    )
    text = "\n".join(html_to_lines(html))
    assert "Visible text" in text
    for bad in ("SECRET-XBRL", "var hidden", "HIDDEN-BLOCK"):
        assert bad not in text


def test_inline_tags_do_not_split_a_sentence_but_blocks_do():
    html = "<div>Our <b>business</b> may <span>suffer</span> losses.</div><div>Second paragraph.</div>"
    assert html_to_lines(html) == ["Our business may suffer losses.", "Second paragraph."]


def test_non_breaking_spaces_and_zero_width_characters_are_normalised():
    html = "<div>ITEM\xa01A.​  RISK\xa0FACTORS</div>"
    assert html_to_lines(html) == ["ITEM 1A. RISK FACTORS"]


def test_table_rows_become_pipe_joined_lines_and_symbols_are_glued_to_numbers():
    html = (
        "<table><tr><td>Revenue</td><td>$</td><td>1,200</td><td>%</td></tr>"
        "<tr><td>Loss</td><td>(5</td><td>)</td></tr><tr><td></td></tr></table>"
    )
    assert html_to_lines(html) == ["Revenue | $1,200%", "Loss | (5)"]


def test_br_makes_a_new_line_and_page_numbers_are_dropped():
    html = "<div>line one<br/>line two</div><div>22</div><div>F-7</div><div>Revenue grew 22 percent</div>"
    assert html_to_lines(html) == ["line one", "line two", "Revenue grew 22 percent"]


def test_empty_document_gives_no_lines():
    assert html_to_lines("<html><body></body></html>") == []


# ---- heading candidates ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "key", "weight"),
    [
        ("ITEM 1A. RISK FACTORS", "1A", 2.0),
        ("Item 1A. Risk Factors", "1A", 1.0),
        ("ITEM 1A.", "1A", 2.0),
        ("Item 7A", "7A", 1.0),
        ("ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION", "7", 2.0),
        ("ITEM 6. (RESERVED)", "6", 2.0),
        ("ITEM 8A. CONSOLIDATED FINANCIAL STATEMENTS OF THE PARENT", "8A", 2.0),
        ("ITEM 14. PRINCIPAL ACCOUNTING FEES AND SERVICES", "14", 2.0),
        ("Item 1 Business", "1", 1.0),
        ("ITEM 1. | BUSINESS", "1", 2.0),
    ],
)
def test_real_style_headings_are_candidates(line, key, weight):
    (cand,) = find_item_candidates([line])
    assert (cand.key, cand.weight) == (key, weight)


@pytest.mark.parametrize(
    "line",
    [
        "Item 1A. Risk Factors (Continued)",
        "ITEM 1C. Cybersecurity (continued)",
        "Item 1A of this report describes the risks that could affect us",
        "See Item 7 for more information",
        "Item 99. Something else",
        "Item 7. " + "long " * 80,
        "Items 10 through 14 are incorporated by reference",
        "Item",
    ],
)
def test_page_headers_prose_and_junk_are_not_candidates(line):
    assert find_item_candidates([line]) == []


# ---- split_sections ----------------------------------------------------------------------------------------------


def toc() -> list[str]:
    return ["Item 1.", "Item 1A.", "Item 1B.", "Item 2.", "Item 7.", "Item 8."]


def test_table_of_contents_is_skipped_and_text_belongs_to_the_body_headings():
    lines = ["COVER PAGE", *toc(), "ITEM 1. BUSINESS", *filler(), "ITEM 1A. RISK FACTORS", *filler(), "ITEM 2. PROPERTIES"]
    lines += [*filler(), "ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS", *filler(), "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", *filler()]
    sections = split_sections(lines, min_body_chars=500)
    assert keys(sections) == ["0", "1", "1A", "2", "7", "8"]
    assert starts(sections)["1"] == lines.index("ITEM 1. BUSINESS")
    assert sections[0].text.splitlines()[0] == "COVER PAGE"
    assert sections[1].text.splitlines()[0] == "ITEM 1. BUSINESS"
    assert sum(s.n_chars for s in sections[1:2]) > 1000  # the body, not the one-word TOC entry


def test_sections_are_contiguous_and_cover_the_whole_document():
    lines = ["cover", "ITEM 1. BUSINESS", *filler(), "ITEM 1A. RISK FACTORS", *filler(), "ITEM 2. PROPERTIES", "none"]
    sections = split_sections(lines, min_body_chars=500)
    assert sections[0].start_line == 0 and sections[-1].end_line == len(lines)
    for a, b in zip(sections, sections[1:], strict=False):
        assert a.end_line == b.start_line


def test_uppercase_heading_beats_an_earlier_title_case_cross_reference():
    lines = ["ITEM 1. BUSINESS", *filler(), "Item 1A. Risk Factors", *filler(), "ITEM 1A. RISK FACTORS", *filler()]
    lines += ["ITEM 2. PROPERTIES", *filler()]
    sections = split_sections(lines, min_body_chars=500)
    assert starts(sections)["1A"] == lines.index("ITEM 1A. RISK FACTORS")
    assert next(s for s in sections if s.key == "1").n_chars > 1000  # the cross-reference did not cut Item 1 short


def test_forward_cross_reference_does_not_derail_the_chain():
    lines = ["ITEM 1. BUSINESS", *filler(), "ITEM 3. LEGAL PROCEEDINGS", "Item 8. Financial Statements and Supplementary Data"]
    lines += [*filler(), "ITEM 4. MINE SAFETY", *filler(), "ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS", *filler(), "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"]
    lines += filler()
    sections = split_sections(lines, min_body_chars=500)
    assert keys(sections) == ["1", "3", "4", "7", "8"]
    assert starts(sections)["8"] == lines.index("ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA")


def test_repeated_page_headers_of_the_same_item_collapse_into_one_section():
    lines = ["ITEM 1. BUSINESS", *filler(2), "Item 1. Business", *filler(2), "Item 1. Business", *filler(2)]
    lines += ["ITEM 1A. RISK FACTORS", *filler()]
    sections = split_sections(lines, min_body_chars=500)
    assert keys(sections) == ["1", "1A"]
    assert sections[0].text.count("Item 1. Business") == 2  # headers stay inside the section text


def test_continued_page_headers_do_not_split_a_section():
    lines = ["ITEM 1. BUSINESS", *filler(), "ITEM 1A.", *filler(2), "Item 1A. Risk Factors (Continued)", *filler(2)]
    lines += ["ITEM 2. PROPERTIES", *filler()]
    sections = split_sections(lines, min_body_chars=500)
    assert keys(sections) == ["1", "1A", "2"]


def test_heading_on_one_line_and_title_on_the_next_both_work():
    lines = ["ITEM 1.", "BUSINESS", *filler(), "ITEM 1A. RISK FACTORS", *filler()]
    sections = split_sections(lines, min_body_chars=500)
    assert keys(sections) == ["1", "1A"]
    assert sections[0].title == "Business"


def test_company_specific_items_are_kept_with_the_observed_title():
    lines = ["ITEM 1. BUSINESS", *filler(), "ITEM 4A. Information About Our Executive Officers.", *filler()]
    lines += ["ITEM 8A. PARENT COMPANY FINANCIAL STATEMENTS", *filler(), "ITEM 8B. SUBSIDIARY FINANCIAL STATEMENTS"]
    sections = split_sections(lines + filler(), min_body_chars=500)
    assert keys(sections) == ["1", "4A", "8A", "8B"]
    assert sections[1].title == "Information About Our Executive Officers."


def test_no_item_1_body_raises_a_clear_error():
    with pytest.raises(ParseError, match="Item 1"):
        split_sections(["Item 1.", "Item 1A.", "Item 2."], min_body_chars=500)
    with pytest.raises(ParseError):
        split_sections(["nothing here"], min_body_chars=500)


def test_financial_statements_after_the_signatures_are_split_into_their_own_section():
    lines = ["ITEM 1. BUSINESS", *filler(), "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", "See Item 15 for the statements."]
    lines += ["ITEM 15. EXHIBITS", *filler(2), "SIGNATURES", "By: | /s/ Someone"]
    lines += ["Report of Independent Registered Public Accounting Firm", *filler(40)]
    sections = split_sections(lines, min_body_chars=500)
    assert keys(sections) == ["1", "8", "15", "FS"]
    fs = sections[-1]
    assert fs.text.splitlines()[0] == "Report of Independent Registered Public Accounting Firm"
    assert "SIGNATURES" in sections[2].text and "SIGNATURES" not in fs.text


def test_statements_are_not_split_when_item_8_is_already_large():
    lines = ["ITEM 1. BUSINESS", *filler(), "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", "Report of Independent Registered Public Accounting Firm"]
    sections = split_sections([*lines, *filler(40), "ITEM 15. EXHIBITS", *filler()], min_body_chars=500)
    assert "FS" not in keys(sections)


# ---- parse_filing, persistence, sanity checks --------------------------------------------------------------------


H1 = "ITEM 1. BUSINESS"
H1A = "ITEM 1A. RISK FACTORS"
H7 = "ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS"
H7A = "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"
H8 = "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"
BIG = 12  # paragraphs: 12 x ~600 chars is above the 5,000-character "long section" heuristics
FULL = {H1: BIG, H1A: BIG, H7: BIG, H7A: 1, H8: BIG}


def html_filing(sections: dict[str, int]) -> str:
    parts = ["<html><body><div>Cover</div><div>Item 1.</div><div>Item 1A.</div>"]
    for heading, n in sections.items():
        parts.append(f"<div>{heading}</div>")
        parts.extend(f"<div>{BODY * 3} paragraph {i}</div>" for i in range(n))
    return "".join(parts) + "</body></html>"


def test_parse_filing_reads_metadata_saves_and_reloads(tmp_path):
    path = tmp_path / "ABC" / "0001-26-000001.htm"
    path.parent.mkdir()
    path.write_text(html_filing(FULL))
    meta = {"ticker": "ABC", "accession": "0001-26-000001", "form": "10-K", "filing_date": "2026-02-01"}
    path.with_suffix(".json").write_text(json.dumps(meta | {"report_date": "2025-12-31", "url": "http://x"}))

    parsed = parse_filing(path)
    assert (parsed.ticker, parsed.form, parsed.filing_date) == ("ABC", "10-K", "2026-02-01")
    assert (parsed.report_date, parsed.url) == ("2025-12-31", "http://x")
    assert [s.key for s in parsed.sections] == ["0", "1", "1A", "7", "7A", "8"]
    assert parsed.section("1A").text.startswith(H1A)
    assert parsed.section("99") is None

    saved = save_parsed(parsed, tmp_path / "out")
    assert saved == tmp_path / "out" / "ABC.json"
    assert load_parsed(saved) == parsed


def test_parse_filing_falls_back_to_folder_name_without_metadata(tmp_path):
    path = tmp_path / "XYZ" / "acc-1.htm"
    path.parent.mkdir()
    path.write_text(html_filing({H1: BIG, H1A: BIG}))
    parsed = parse_filing(path)
    assert (parsed.ticker, parsed.accession, parsed.form) == ("XYZ", "acc-1", "")


def make_parsed(tmp_path, sections: dict[str, int]):
    path = tmp_path / "T" / "a.htm"
    path.parent.mkdir(exist_ok=True)
    path.write_text(html_filing(sections))
    return parse_filing(path)


def test_sanity_passes_for_a_complete_filing(tmp_path):
    assert sanity_problems(make_parsed(tmp_path, FULL)) == []


def test_sanity_flags_missing_and_tiny_sections_and_missing_statements(tmp_path):
    parsed = make_parsed(tmp_path, {H1: BIG, H1A: 1, H7: BIG})
    problems = " | ".join(sanity_problems(parsed))
    assert "Item 7A not found" in problems
    assert "Item 1A is only" in problems
    assert "financial statements" in problems


# ---- real filings (skipped when the data is not on this machine) -------------------------------------------------

RAW = Path("data/raw/filings")
TICKERS = ["AAL", "CCL", "DAL", "DUK", "F", "WMT"]


@pytest.fixture(scope="module")
def real_filings():
    files = {t: next(iter((RAW / t).glob("*.htm")), None) for t in TICKERS}
    if not all(files.values()):
        pytest.skip("real filings not downloaded (run the downloader first)")
    return {t: parse_filing(p) for t, p in files.items()}


def test_real_filings_have_the_core_items_and_no_sanity_problems(real_filings):
    for ticker, parsed in real_filings.items():
        assert sanity_problems(parsed) == [], ticker


def test_real_filings_sections_are_ordered_and_contiguous(real_filings):
    for ticker, parsed in real_filings.items():
        for a, b in zip(parsed.sections, parsed.sections[1:], strict=False):
            assert a.end_line == b.start_line, (ticker, a.key, b.key)
        assert parsed.sections[-1].end_line == parsed.n_lines


def test_real_filings_start_each_section_at_its_own_heading(real_filings):
    for ticker, parsed in real_filings.items():
        for key in ("1", "1A", "7"):
            first_line = parsed.section(key).text.splitlines()[0].lower()
            assert first_line.startswith(f"item {key.lower()}"), (ticker, key, first_line)


def test_real_filings_contain_no_xbrl_markup_or_page_number_lines(real_filings):
    for ticker, parsed in real_filings.items():
        for section in parsed.sections:
            assert "xbrli:" not in section.text and "us-gaap:" not in section.text, ticker
            assert not any(line.isdigit() and len(line) <= 3 for line in section.text.splitlines()), ticker


def test_ford_statements_are_found_after_the_signatures(real_filings):
    assert real_filings["F"].section("8").n_chars < 5_000
    assert real_filings["F"].section("FS").n_chars > 100_000
