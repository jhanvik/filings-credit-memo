"""Retrieval tests with a fake embedder (no model download) and a real ChromaDB in a temp folder."""

import math
import re
import zlib

import pytest

from filings_credit_memo.chunk import Chunk, ChunkConfig
from filings_credit_memo.parse import ParsedFiling, Section
from filings_credit_memo.retrieval import VectorStore, build_index, retrieve

DIM = 128


class FakeEmbedder:
    """Bag-of-words hashing: texts that share words get similar vectors. Deterministic, no network."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def filing(ticker: str, paragraphs: dict[str, list[str]]) -> ParsedFiling:
    sections, line = [], 0
    for key, paras in paragraphs.items():
        text = "\n".join(paras)
        sections.append(Section(key, "T", line, line + len(paras), text))
        line += len(paras)
    return ParsedFiling(ticker, f"acc-{ticker}", "10-K", "2026-02-01", "", "", line, sections)


CFG = ChunkConfig(target_chars=300, overlap_chars=50, max_chars=400, min_section_chars=20)
A = filing(
    "AAA",
    {
        "1A": ["Fuel price volatility could materially hurt our airline margins and cash flow."] * 3
        + ["Cybersecurity incidents may disrupt reservations systems and customer data privacy."] * 3,
        "7": ["Our revolving credit facility matures in 2027 and term loans mature in 2029 and 2031."] * 3,
    },
)
B = filing("BBB", {"1A": ["Tariffs on imported goods could increase our merchandise costs and pressure retail margins."] * 6})


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "idx")
    build_index([A, B], FakeEmbedder(), s, CFG)
    return s


def test_build_index_counts_chunks_and_search_finds_the_right_one(store):
    assert store.count() > 0
    hits = retrieve(store, FakeEmbedder(), "when does the revolving credit facility mature", k=3)
    assert hits[0].ticker == "AAA" and hits[0].section_key == "7"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_ticker_filter_only_returns_that_company(store):
    hits = retrieve(store, FakeEmbedder(), "margins and costs", k=10, ticker="bbb")
    assert hits and {h.ticker for h in hits} == {"BBB"}


def test_section_filter_and_combined_filters(store):
    hits = retrieve(store, FakeEmbedder(), "credit facility", k=10, section_keys=["7"])
    assert hits and {h.section_key for h in hits} == {"7"}
    both = retrieve(store, FakeEmbedder(), "margins", k=10, ticker="AAA", section_keys=["1A"])
    assert both and all(h.ticker == "AAA" and h.section_key == "1A" for h in both)


def test_k_larger_than_index_and_no_match_filter_are_safe(store):
    assert len(retrieve(store, FakeEmbedder(), "anything", k=10_000)) == store.count()
    assert retrieve(store, FakeEmbedder(), "anything", k=3, ticker="ZZZ") == []
    assert retrieve(store, FakeEmbedder(), "anything", k=0) == []


def test_empty_question_is_rejected(store):
    with pytest.raises(ValueError):
        retrieve(store, FakeEmbedder(), "   ?! ", k=3)


def test_rebuilding_does_not_duplicate_chunks(tmp_path):
    s = VectorStore(tmp_path / "idx")
    first = build_index([A, B], FakeEmbedder(), s, CFG)
    second = build_index([A, B], FakeEmbedder(), s, CFG)
    assert first == second == s.count()


def test_index_persists_on_disk(tmp_path):
    build_index([A], FakeEmbedder(), VectorStore(tmp_path / "idx"), CFG)
    assert VectorStore(tmp_path / "idx").count() > 0


def test_hit_carries_source_metadata_for_citations(store):
    hit = retrieve(store, FakeEmbedder(), "fuel price volatility", k=1)[0]
    start, end = hit.lines
    assert hit.metadata["accession"] == "acc-AAA" and hit.metadata["filing_date"] == "2026-02-01"
    assert 0 <= start <= end
    assert "Fuel price" in hit.text


def test_mismatched_lengths_are_rejected(tmp_path):
    s = VectorStore(tmp_path / "idx")
    chunk = Chunk("i", "A", "a", "d", "10-K", "1", "T", 0, 0, "t")
    with pytest.raises(ValueError):
        s.add([chunk], [])


def test_empty_filing_list_builds_an_empty_index(tmp_path):
    s = VectorStore(tmp_path / "idx")
    assert build_index([], FakeEmbedder(), s, CFG) == 0 and s.count() == 0
