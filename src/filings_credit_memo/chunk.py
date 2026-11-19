"""Split parsed 10-K sections into overlapping, citable chunks.

Each chunk remembers where it came from (ticker, accession, Item, absolute line range in the cleaned text) so that an
answer can later cite its source and a human can check it. Chunks are packed from whole lines (a line is a paragraph or
a table row) so sentences and rows are not cut in half; only a line longer than `max_chars` is split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from filings_credit_memo.parse import ParsedFiling, Section

# Sections worth searching. Cover page, exhibits-only and "incorporated by reference" Items are left out on purpose.
INDEX_KEYS = ("1", "1A", "1C", "2", "3", "5", "7", "7A", "8", "8A", "8B", "FS", "9A", "15")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ITEM_KEY = re.compile(r"^\d+[A-C]?$")


@dataclass(frozen=True)
class ChunkConfig:
    target_chars: int = 1_200  # start a new chunk once adding a line would pass this
    overlap_chars: int = 200  # trailing lines repeated at the start of the next chunk
    max_chars: int = 1_800  # a single line longer than this is split on sentence ends
    min_section_chars: int = 200  # sections shorter than this ("None.", pointers) are not indexed

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_chars < self.target_chars <= self.max_chars:
            raise ValueError("need 0 <= overlap_chars < target_chars <= max_chars")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ticker: str
    accession: str
    filing_date: str
    form: str
    section_key: str
    section_title: str
    start_line: int  # absolute line numbers in the cleaned filing text, inclusive
    end_line: int
    text: str

    @property
    def label(self) -> str:
        item = f"Item {self.section_key} {self.section_title}" if _ITEM_KEY.match(self.section_key) else self.section_title
        return f"{self.ticker} {self.form} filed {self.filing_date} | {item}"

    @property
    def embed_text(self) -> str:
        """What gets embedded: the text with a one-line context header so 'Item 1A' chunks of different companies differ."""
        return f"{self.label}\n{self.text}"


def split_long_line(line: str, max_chars: int) -> list[str]:
    """Split on sentence ends, then hard-split anything still too long. Concatenating the pieces keeps every word."""
    if len(line) <= max_chars:
        return [line]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(line):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_section(section: Section, filing: ParsedFiling, cfg: ChunkConfig = ChunkConfig()) -> list[Chunk]:
    pieces: list[tuple[int, str]] = []
    for offset, line in enumerate(section.text.split("\n")):
        pieces.extend((section.start_line + offset, part) for part in split_long_line(line, cfg.max_chars))

    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    size = 0
    new_pieces = 0  # pieces added since the last emitted chunk (overlap does not count)

    def emit() -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{filing.ticker}-{section.key}-{len(chunks):04d}",
                ticker=filing.ticker,
                accession=filing.accession,
                filing_date=filing.filing_date,
                form=filing.form,
                section_key=section.key,
                section_title=section.title,
                start_line=current[0][0],
                end_line=current[-1][0],
                text="\n".join(text for _, text in current),
            )
        )

    for piece in pieces:
        if current and size + len(piece[1]) + 1 > cfg.target_chars:
            emit()
            keep: list[tuple[int, str]] = []
            kept = 0
            for earlier in reversed(current):
                if kept + len(earlier[1]) + 1 > cfg.overlap_chars:
                    break
                keep.insert(0, earlier)
                kept += len(earlier[1]) + 1
            current, size, new_pieces = keep, kept, 0
        current.append(piece)
        size += len(piece[1]) + 1
        new_pieces += 1
    if current and new_pieces:
        emit()
    return chunks


def chunk_filing(
    filing: ParsedFiling, cfg: ChunkConfig = ChunkConfig(), keys: tuple[str, ...] = INDEX_KEYS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in filing.sections:
        if section.key in keys and section.n_chars >= cfg.min_section_chars:
            chunks.extend(chunk_section(section, filing, cfg))
    return chunks
