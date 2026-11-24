"""Hybrid retrieval: dense embeddings plus BM25 keyword ranking, merged with reciprocal rank fusion (RRF).

Why: a dense model matches meaning, but a question like "Which firm is the independent registered public accounting
firm?" is answered by a chunk that literally contains those words. A keyword ranker finds such chunks easily, while a
dense model can lose them among hundreds of financial-statement chunks. RRF merges the two rankings using only ranks,
so the two score scales never have to be compared.

Hit.score stays the dense cosine similarity so the refusal threshold keeps its meaning.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Protocol

import numpy as np

from filings_credit_memo.retrieval import Embedder, Hit

STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or s that the this to was were "
    "what when where which who whom why will with".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
POOL = 50  # candidates taken from each ranking before fusion
RRF_K = 60  # the usual constant from the original RRF paper; verify before citing it as a source


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Bm25Index:
    """Okapi BM25 over a fixed list of documents. Scores are only meaningful within one index."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        counts = [Counter(tokenize(d)) for d in docs]
        self._lengths = [sum(c.values()) for c in counts]
        n = len(docs)
        self._avg_len = sum(self._lengths) / n if n else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(counts):
            for term, tf in c.items():
                self._postings[term].append((i, tf))
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        """The n best documents as (index, score), best first; documents sharing no term with the query are left out."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                norm = 1 - self._b + self._b * self._lengths[i] / self._avg_len if self._avg_len else 1.0
                scores[i] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:n]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion: each ranking gives an item 1/(k + rank); the scores are added up."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] += 1.0 / (k + rank)
    return dict(fused)


class ChunkSource(Protocol):
    def get_all(self, ticker: str | None = None) -> tuple[list[Hit], list[list[float]]]: ...


class HybridRetriever:
    """Dense + BM25 retrieval over one company's chunks (or all chunks when ticker is None). Built lazily and cached."""

    def __init__(self, store: ChunkSource, embedder: Embedder, pool: int = POOL, rrf_k: int = RRF_K) -> None:
        self._store, self._embedder, self._pool, self._rrf_k = store, embedder, pool, rrf_k
        self._cache: dict[str | None, tuple[list[Hit], Any, Bm25Index]] = {}

    def _data(self, ticker: str | None) -> tuple[list[Hit], Any, Bm25Index]:
        key = ticker.upper() if ticker else None
        if key not in self._cache:
            hits, vectors = self._store.get_all(key)
            if len(hits) != len(vectors):
                raise ValueError("the store returned a different number of chunks and embeddings")
            if hits:
                matrix = np.asarray(vectors, dtype=float)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.where(norms == 0, 1.0, norms)
            else:
                matrix = np.zeros((0, 0))
            self._cache[key] = (hits, matrix, Bm25Index([h.text for h in hits]))
        return self._cache[key]

    def retrieve(self, question: str, k: int = 5, ticker: str | None = None) -> list[Hit]:
        if not re.search(r"\w", question):
            raise ValueError("question is empty")
        hits, matrix, bm25 = self._data(ticker)
        if not hits or k <= 0:
            return []
        query = np.asarray(self._embedder.embed_query(question), dtype=float)
        norm = float(np.linalg.norm(query))
        cosine = matrix @ (query / norm if norm else query)
        dense_order = sorted(range(len(hits)), key=lambda i: (-float(cosine[i]), i))[: self._pool]
        lexical_order = [i for i, _ in bm25.top(question, self._pool)]
        fused = rrf([[hits[i].chunk_id for i in dense_order], [hits[i].chunk_id for i in lexical_order]], self._rrf_k)
        index_of = {h.chunk_id: i for i, h in enumerate(hits)}
        ranked = sorted(fused, key=lambda cid: (-fused[cid], -float(cosine[index_of[cid]]), cid))[:k]
        out = []
        for cid in ranked:
            i = index_of[cid]
            out.append(Hit(cid, float(cosine[i]), hits[i].text, hits[i].metadata))
        return out
