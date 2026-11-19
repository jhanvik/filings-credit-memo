"""Tests for chunking. The last block uses the real downloaded filings and is skipped if they are absent."""

from pathlib import Path

import pytest

from filings_credit_memo.chunk import (
    INDEX_KEYS,
    Chunk,
    ChunkConfig,
    chunk_filing,
    chunk_section,
    split_long_line,
)
from filings_credit_memo.parse import ParsedFiling, Section, parse_filing

CFG = ChunkConfig(target_chars=100, overlap_chars=30, max_chars=150, min_section_chars=10)


def make_filing(section_lines: list[str], key: str = "1A", start: int = 10) -> tuple[ParsedFiling, Section]:
    section = Section(key, "Risk Factors", start, start + len(section_lines), "\n".join(section_lines))
    filing = ParsedFiling("ABC", "0001-26-000001", "10-K", "2026-02-01", "2025-12-31", "", 999, [section])
    return filing, section


def lines_of(n: int, width: int = 40) -> list[str]:
    return [f"line{i:03d} " + "w" * (width - 8) for i in range(n)]


def test_config_validates_its_numbers():
    for bad in ({"overlap_chars": 1_200}, {"target_chars": 2_000}, {"overlap_chars": -1}):
        with pytest.raises(ValueError):
            ChunkConfig(**bad)


def test_chunks_respect_target_size_and_keep_whole_lines():
    filing, section = make_filing(lines_of(30))
    chunks = chunk_section(section, filing, CFG)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.text) <= CFG.target_chars + CFG.max_chars  # never wildly over
        assert all(line in section.text for line in c.text.split("\n"))


def test_every_line_is_covered_and_nothing_is_lost():
    filing, section = make_filing(lines_of(30))
    covered = set()
    for c in chunk_section(section, filing, CFG):
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered == set(range(section.start_line, section.start_line + 30))


def test_consecutive_chunks_overlap_but_always_make_progress():
    filing, section = make_filing(lines_of(30))  # each line is 41 chars with its newline, so 45 keeps one line
    chunks = chunk_section(section, filing, ChunkConfig(target_chars=100, overlap_chars=45, max_chars=150))
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start_line <= a.end_line  # overlap
        assert b.end_line > a.end_line  # progress
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_smaller_than_one_line_means_no_overlap_but_still_full_coverage():
    filing, section = make_filing(lines_of(30))
    chunks = chunk_section(section, filing, CFG)  # overlap 30 < 41-char lines
    assert all(b.start_line == a.end_line + 1 for a, b in zip(chunks, chunks[1:], strict=False))


def test_no_redundant_trailing_chunk_made_only_of_overlap():
    filing, section = make_filing(lines_of(9))
    chunks = chunk_section(section, filing, CFG)
    assert chunks[-1].end_line == section.start_line + 8
    assert chunks[-1].start_line > chunks[0].start_line or len(chunks) == 1


def test_line_numbers_are_absolute_and_ids_carry_ticker_and_item():
    filing, section = make_filing(lines_of(12), start=500)
    chunks = chunk_section(section, filing, CFG)
    assert chunks[0].start_line == 500
    assert chunks[0].chunk_id == "ABC-1A-0000"
    assert chunks[0].filing_date == "2026-02-01"


def test_single_short_section_gives_one_chunk():
    filing, section = make_filing(["only line"])
    (chunk,) = chunk_section(section, filing, CFG)
    assert chunk.text == "only line"


def test_long_line_is_split_on_sentences_without_losing_words():
    sentence = "This is a fairly ordinary sentence about risk. "
    line = (sentence * 10).strip()
    parts = split_long_line(line, 120)
    assert len(parts) > 1 and all(len(p) <= 120 for p in parts)
    assert " ".join(parts).split() == line.split()


def test_unbreakable_long_token_is_hard_split():
    parts = split_long_line("x" * 500, 100)
    assert all(len(p) <= 100 for p in parts) and "".join(parts) == "x" * 500


def test_short_line_is_returned_unchanged():
    assert split_long_line("short", 100) == ["short"]


def test_embed_text_has_a_context_header_and_label_for_non_item_sections():
    filing, section = make_filing(["body text"])
    chunk = chunk_section(section, filing, CFG)[0]
    assert chunk.embed_text == "ABC 10-K filed 2026-02-01 | Item 1A Risk Factors\nbody text"
    fs = Chunk("i", "ABC", "a", "2026-02-01", "10-K", "FS", "Financial statements and notes", 1, 2, "t")
    assert "Item" not in fs.label and "Financial statements and notes" in fs.label


def test_chunk_filing_skips_unlisted_and_tiny_sections():
    big = Section("1A", "Risk Factors", 0, 5, "\n".join(lines_of(5)))
    tiny = Section("1B", "Unresolved", 5, 6, "None.")
    cover = Section("0", "Cover", 6, 20, "\n".join(lines_of(14)))
    other = Section("11", "Executive Compensation", 20, 30, "\n".join(lines_of(10)))
    filing = ParsedFiling("ABC", "a", "10-K", "2026-02-01", "", "", 30, [cover, big, tiny, other])
    keys = {c.section_key for c in chunk_filing(filing, ChunkConfig(min_section_chars=50))}
    assert keys == {"1A"}
    assert "1A" in INDEX_KEYS and "0" not in INDEX_KEYS


# ---- real filings ------------------------------------------------------------------------------------------------

RAW = Path("data/raw/filings")
TICKERS = ["AAL", "CCL", "DAL", "DUK", "F", "WMT"]


@pytest.fixture(scope="module")
def real_chunks():
    files = {t: next(iter((RAW / t).glob("*.htm")), None) for t in TICKERS}
    if not all(files.values()):
        pytest.skip("real filings not downloaded")
    out = {}
    for t, path in files.items():
        filing = parse_filing(path)
        out[t] = (filing, chunk_filing(filing))
    return out


def test_real_chunks_are_bounded_unique_and_cover_every_indexed_line(real_chunks):
    cfg = ChunkConfig()
    for ticker, (filing, chunks) in real_chunks.items():
        assert chunks, ticker
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert max(len(c.text) for c in chunks) <= cfg.target_chars + cfg.max_chars, ticker
        covered = set()
        for c in chunks:
            covered.update(range(c.start_line, c.end_line + 1))
        for section in filing.sections:
            if section.key in INDEX_KEYS and section.n_chars >= cfg.min_section_chars:
                assert set(range(section.start_line, section.end_line)) <= covered, (ticker, section.key)


def test_real_chunk_text_comes_from_its_own_section(real_chunks):
    for ticker, (filing, chunks) in real_chunks.items():
        by_key = {s.key: s.text for s in filing.sections}
        for c in chunks[:: max(1, len(chunks) // 200)]:
            assert c.text.split("\n")[0] in by_key[c.section_key], (ticker, c.chunk_id)
