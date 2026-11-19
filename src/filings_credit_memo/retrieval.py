"""Embeddings, a local vector store (ChromaDB) and retrieval.

The embedder is behind a tiny interface so tests use a fake one and the real model (sentence-transformers) is only
imported when actually used. The model name below is from the author's knowledge: verify it on Hugging Face.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from filings_credit_memo.chunk import Chunk, ChunkConfig, chunk_filing
from filings_credit_memo.parse import ParsedFiling

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION = "filings"
ADD_BATCH = 500


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings. The first use downloads the model from Hugging Face (needs internet once)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        query_prefix: str = BGE_QUERY_PREFIX,
        show_progress: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # imported late: heavy

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._show_progress = show_progress

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=self._show_progress
        )
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float  # cosine similarity: higher is closer
    text: str
    metadata: dict[str, Any]

    @property
    def ticker(self) -> str:
        return str(self.metadata["ticker"])

    @property
    def section_key(self) -> str:
        return str(self.metadata["section_key"])

    @property
    def lines(self) -> tuple[int, int]:
        return int(self.metadata["start_line"]), int(self.metadata["end_line"])


class VectorStore:
    """ChromaDB collection on disk, cosine distance."""

    def __init__(self, path: Path, collection: str = COLLECTION) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        self._name = collection
        self._collection = self._client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), ADD_BATCH):
            part = chunks[i : i + ADD_BATCH]
            self._collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i : i + ADD_BATCH],
                documents=[c.text for c in part],
                metadatas=[
                    {
                        "ticker": c.ticker,
                        "accession": c.accession,
                        "filing_date": c.filing_date,
                        "section_key": c.section_key,
                        "section_title": c.section_title,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                    }
                    for c in part
                ],
            )

    def query(
        self, embedding: list[float], k: int = 5, ticker: str | None = None, section_keys: list[str] | None = None
    ) -> list[Hit]:
        clauses: list[dict[str, Any]] = []
        if ticker:
            clauses.append({"ticker": ticker.upper()})
        if section_keys:
            clauses.append({"section_key": {"$in": section_keys}})
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
        total = self.count()
        if total == 0 or k <= 0:
            return []
        result = self._collection.query(query_embeddings=[embedding], n_results=min(k, total), where=where)
        return [
            Hit(cid, 1.0 - float(dist), doc, dict(meta))
            for cid, dist, doc, meta in zip(
                result["ids"][0], result["distances"][0], result["documents"][0], result["metadatas"][0], strict=True
            )
        ]


def build_index(
    filings: list[ParsedFiling], embedder: Embedder, store: VectorStore, cfg: ChunkConfig = ChunkConfig()
) -> int:
    """Rebuild the whole index from scratch (so re-running never leaves duplicates). Returns the chunk count."""
    chunks = [c for filing in filings for c in chunk_filing(filing, cfg)]
    store.reset()
    if chunks:
        store.add(chunks, embedder.embed_documents([c.embed_text for c in chunks]))
    return len(chunks)


def retrieve(
    store: VectorStore,
    embedder: Embedder,
    question: str,
    k: int = 5,
    ticker: str | None = None,
    section_keys: list[str] | None = None,
) -> list[Hit]:
    if not re.search(r"\w", question):
        raise ValueError("question is empty")
    return store.query(embedder.embed_query(question), k, ticker, section_keys)
