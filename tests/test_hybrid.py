import pytest

from filings_credit_memo.chunk import Chunk
from filings_credit_memo.evalset import EvalRow
from filings_credit_memo.hybrid import Bm25Index, HybridRetriever, rrf, tokenize
from filings_credit_memo.retrieval import Hit, VectorStore
from filings_credit_memo.retrieval_eval import measure, render_comparison

AUDIT = "Report of Independent Registered Public Accounting Firm. We have audited the statements. KPMG LLP Dallas"
OTHERS = [
    "Revenue increased because of higher passenger volumes and fares during the year",
    "Fuel costs rose and the company hedged part of its expected fuel consumption",
    "The revolving credit facility matures in 2028 and bears interest at SOFR plus a margin",
    "Employees are represented by several labor unions under collective agreements",
]
QUESTION = "Which firm is the independent registered public accounting firm?"


def vec(*xs: float) -> list[float]:
    return list(xs)


class FakeStore:
    """Holds chunks with hand-made embeddings; get_all mirrors VectorStore.get_all."""

    def __init__(self, items: list[tuple[str, str, str, list[float]]]) -> None:
        self.items = items  # (chunk_id, ticker, text, embedding)
        self.calls = 0

    def get_all(self, ticker: str | None = None) -> tuple[list[Hit], list[list[float]]]:
        self.calls += 1
        rows = [r for r in self.items if ticker is None or r[1] == ticker.upper()]
        return [Hit(cid, 0.0, text, {"ticker": t}) for cid, t, text, _ in rows], [e for *_, e in rows]


class FakeEmbedder:
    def __init__(self, query: list[float]) -> None:
        self.query = query

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.query


def store_with_audit_last() -> FakeStore:
    # the query vector points at x; the audit chunk is orthogonal, so dense search ranks it last
    items = [(f"A-{i}", "A", text, vec(1.0 - 0.1 * i, 0.1 * i, 0.0)) for i, text in enumerate(OTHERS)]
    items.append(("A-audit", "A", AUDIT, vec(0.0, 0.0, 1.0)))
    items.append(("B-1", "B", "Another company's text about revenue", vec(1.0, 0.0, 0.0)))
    return FakeStore(items)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("Which firm is the Independent Registered firm?") == ["firm", "independent", "registered", "firm"]


def test_bm25_prefers_the_document_with_the_rare_term_and_skips_non_matches():
    index = Bm25Index([*OTHERS, AUDIT])
    top = index.top("independent registered public accounting firm", 10)
    assert top[0][0] == len(OTHERS)  # the audit chunk
    assert all(i == len(OTHERS) for i, _ in top)  # nothing else shares a term with the query


def test_bm25_empty_inputs():
    assert Bm25Index([]).top("anything", 5) == []
    assert Bm25Index(["alpha beta"]).top("", 5) == []


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([["a", "b", "c"], ["c", "a"]])
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"] > fused["b"]


def test_hybrid_lifts_a_chunk_that_dense_search_ranks_last():
    store = store_with_audit_last()
    hybrid = HybridRetriever(store, FakeEmbedder(vec(1.0, 0.0, 0.0)))
    dense_order = [h.chunk_id for h in hybrid.retrieve("revenue", 5, "A")]
    assert dense_order[-1] == "A-audit" or "A-audit" not in dense_order[:2]
    hits = hybrid.retrieve(QUESTION, 3, "A")
    assert "A-audit" in [h.chunk_id for h in hits]


def test_hit_score_is_the_dense_cosine_not_the_fused_score():
    store = store_with_audit_last()
    hits = HybridRetriever(store, FakeEmbedder(vec(1.0, 0.0, 0.0))).retrieve(QUESTION, 5, "A")
    by_id = {h.chunk_id: h.score for h in hits}
    assert by_id["A-audit"] == pytest.approx(0.0)
    assert by_id["A-0"] == pytest.approx(1.0)


def test_ticker_filter_and_cache():
    store = store_with_audit_last()
    hybrid = HybridRetriever(store, FakeEmbedder(vec(1.0, 0.0, 0.0)))
    assert {h.chunk_id for h in hybrid.retrieve("revenue", 10, "B")} == {"B-1"}
    hybrid.retrieve("revenue", 10, "b")
    hybrid.retrieve("fuel", 10, "B")
    assert store.calls == 1  # the per-company data is built once


def test_empty_question_and_empty_store():
    hybrid = HybridRetriever(FakeStore([]), FakeEmbedder(vec(1.0, 0.0, 0.0)))
    assert hybrid.retrieve("revenue", 5, "A") == []
    with pytest.raises(ValueError):
        hybrid.retrieve("   ", 5, "A")


def make_row(row_id: str, field: str, evidence: str, answerable: bool = True) -> EvalRow:
    return EvalRow(row_id, "A", field, QUESTION, answerable, ("KPMG",), evidence, "", "")


def test_measure_finds_the_rank_of_the_evidence_chunk():
    store = store_with_audit_last()
    hybrid = HybridRetriever(store, FakeEmbedder(vec(1.0, 0.0, 0.0)))
    rows = [
        make_row("A-auditor", "auditor", "Report of Independent Registered Public Accounting Firm. We have audited"),
        make_row("A-missing", "auditor", "This sentence appears in none of the retrieved chunks at all"),
        make_row("A-none", "out_of_scope", "", answerable=False),
    ]
    ranks = measure(rows, hybrid.retrieve)
    assert set(ranks) == {"A-auditor", "A-missing"}  # unanswerable rows are not measured
    assert ranks["A-auditor"] is not None and ranks["A-auditor"] <= 5
    assert ranks["A-missing"] is None


def test_render_comparison_reports_gains_losses_and_misses():
    rows = [make_row("r1", "auditor", "x" * 30), make_row("r2", "cover", "y" * 30)]
    report = render_comparison(rows, {"dense": {"r1": None, "r2": 2}, "hybrid": {"r1": 1, "r2": 7}})
    assert "| dense | 0/2 | 1/2 | 1/2 | 1/2 |" in report
    assert "| hybrid | 1/2 | 1/2 | 1/2 | 2/2 |" in report
    assert "gained ['r1']; lost ['r2']" in report
    assert "Still missing at k=5 with hybrid: ['r2']" in report


def chunk(cid: str, ticker: str, text: str) -> Chunk:
    return Chunk(cid, ticker, "acc", "2026-01-01", "10-K", "8", "Statements", 1, 2, text)


def test_vectorstore_get_all_returns_chunks_and_embeddings(tmp_path):
    store = VectorStore(tmp_path / "idx", collection="t_get_all")
    chunks = [chunk("A-1", "A", "alpha"), chunk("A-2", "A", "beta"), chunk("B-1", "B", "gamma")]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    hits, vectors = store.get_all("a")
    assert sorted(h.chunk_id for h in hits) == ["A-1", "A-2"]
    assert len(vectors) == 2 and all(len(v) == 2 for v in vectors)
    all_hits, all_vectors = store.get_all()
    assert len(all_hits) == 3 and len(all_vectors) == 3
    by_id = {h.chunk_id: v for h, v in zip(all_hits, all_vectors, strict=True)}
    assert by_id["B-1"] == pytest.approx([0.5, 0.5])
    assert all_hits[0].metadata["ticker"] in {"A", "B"}


def test_hybrid_retriever_works_on_a_real_chroma_store(tmp_path):
    store = VectorStore(tmp_path / "idx", collection="t_real")
    texts = [*OTHERS, AUDIT]
    chunks = [chunk(f"A-{i}", "A", t) for i, t in enumerate(texts)]
    store.add(chunks, [[1.0, 0.1 * i, 0.0] if i < len(OTHERS) else [0.0, 0.0, 1.0] for i in range(len(texts))])
    hits = HybridRetriever(store, FakeEmbedder(vec(1.0, 0.0, 0.0))).retrieve(QUESTION, 2, "A")
    assert hits[0].chunk_id == "A-0" or "A-4" in [h.chunk_id for h in hits]
    assert "A-4" in [h.chunk_id for h in hits]
    assert all(isinstance(h.score, float) for h in hits)


def test_describe_row_marks_the_gold_chunk_and_its_rank():
    from filings_credit_memo.retrieval_eval import describe_row

    gold = "Report of Independent Registered Public Accounting Firm. We have audited the statements"
    hits = [
        Hit("A-0", 0.5, "Unrelated revenue text", {"ticker": "A", "section_key": "7", "start_line": 1, "end_line": 2}),
        Hit("A-1", 0.4, gold + " KPMG", {"ticker": "A", "section_key": "8", "start_line": 5, "end_line": 6}),
    ]
    text = describe_row(make_row("A-auditor", "auditor", gold), "hybrid", hits, show=2)
    assert "[hybrid] gold chunk at rank 2" in text
    assert text.count("<- gold") == 1 and "Item 8 lines 5-6  <- gold" in text
    missing = describe_row(make_row("A-x", "auditor", "z" * 40), "dense", hits, show=1)
    assert "NOT in the top 2" in missing


def test_answer_matching_is_token_based_and_handles_numbers():
    from filings_credit_memo.retrieval_eval import answer_in_text, answer_needles

    row = EvalRow("r", "A", "revenue", QUESTION, True, ("32,237 million", "32.2 billion"), "x" * 30, "", "")
    assert answer_needles(row) == ["32,237", "32.2"]
    assert answer_in_text(row, "Total operating revenues | 32,237 | 30,357")
    assert not answer_in_text(row, "Total | 132,237 | 30,357")  # not a whole token
    assert not answer_in_text(row, "Total | 32,2371")
    firm = EvalRow("f", "A", "auditor", QUESTION, True, ("Ernst & Young", "Ernst"), "x" * 30, "", "")
    assert answer_in_text(firm, "/s/ Ernst & Young LLP Rogers, Arkansas")
    assert not answer_in_text(firm, "Ernstein is unrelated")


def test_render_comparison_adds_answer_table_when_given():
    rows = [make_row("r1", "auditor", "x" * 30)]
    report = render_comparison(rows, {"dense": {"r1": None}}, answer_ranks_by_name={"dense": {"r1": 2}})
    assert "answer@5" in report and "| dense | 0/1 | 1/1 | 1/1 | 1/1 |" in report
