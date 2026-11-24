"""Retrieval-only comparison: no language model is called, so a run takes seconds and costs nothing.

For every answerable labeled row it finds the rank of the first retrieved chunk that contains the gold evidence
(the same test evaluate.py uses) and reports hit@k for each retriever.

    uv run python -m filings_credit_memo.retrieval_eval compare
    uv run python -m filings_credit_memo.retrieval_eval build-cover --out data/index_cover
    uv run python -m filings_credit_memo.retrieval_eval compare --index-cover data/index_cover

The default labels are the dev set plus the first held-out set. That held-out set was used once and is spent: it is
fair for diagnosing retrieval, but numbers measured on it after tuning must not be reported as held-out.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from filings_credit_memo.chunk import INDEX_KEYS, ChunkConfig, chunk_filing
from filings_credit_memo.evalset import EvalRow, load_rows
from filings_credit_memo.evaluate import evidence_in_text
from filings_credit_memo.hybrid import HybridRetriever
from filings_credit_memo.parse import load_parsed
from filings_credit_memo.retrieval import DEFAULT_MODEL, Hit, VectorStore, retrieve

KS = (1, 3, 5, 10)
Retriever = Callable[[str, int, str | None], list[Hit]]


def evidence_rank(row: EvalRow, hits: list[Hit]) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if evidence_in_text(row.gold_evidence, h.text)), None)


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def answer_needles(row: EvalRow) -> list[str]:
    """Strings to look for in a chunk: the first number of a numeric answer, the whole phrase otherwise."""
    needles = []
    for alt in row.gold_answers:
        number = _NUMBER.search(alt)
        needle = number.group(0).rstrip(",") if number else alt.strip()
        if needle:
            needles.append(needle.lower())
    return needles


def answer_in_text(row: EvalRow, text: str) -> bool:
    """Loose test: a gold answer appears as a whole token. Numbers can match by coincidence, so it is a rough guide."""
    lowered = " ".join(text.lower().split())
    return any(re.search(rf"(?<![\w,.]){re.escape(n)}(?![\w])", lowered) for n in answer_needles(row))


def answer_rank(row: EvalRow, hits: list[Hit]) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if answer_in_text(row, h.text)), None)


def measure_answers(rows: list[EvalRow], retriever: Retriever, max_k: int = max(KS)) -> dict[str, int | None]:
    """Rank of the first chunk containing a gold answer string, per answerable row."""
    return {r.id: answer_rank(r, retriever(r.question, max_k, r.ticker)) for r in rows if r.answerable}


def measure(rows: list[EvalRow], retriever: Retriever, max_k: int = max(KS)) -> dict[str, int | None]:
    """Rank of the first chunk holding the gold evidence, per answerable row (None = not in the top max_k)."""
    return {r.id: evidence_rank(r, retriever(r.question, max_k, r.ticker)) for r in rows if r.answerable}


def describe_row(row: EvalRow, name: str, hits: list[Hit], show: int = 5, chars: int = 260) -> str:
    """Readable view of one retriever's results for one row: where the gold chunk is, and what ranked above it."""
    rank = evidence_rank(row, hits)
    where = f"gold chunk at rank {rank}" if rank else f"gold chunk NOT in the top {len(hits)}"
    answer_ranks = [i for i, h in enumerate(hits, 1) if answer_in_text(row, h.text)][:8]
    lines = [f"  [{name}] {where}; a gold answer string appears in chunks at ranks {answer_ranks or 'none'}"]
    for i, h in enumerate(hits[:show], 1):
        mark = "  <- gold" if evidence_in_text(row.gold_evidence, h.text) else ""
        start, end = h.lines
        snippet = h.text[:chars].replace("\n", " ")
        lines.append(f"    #{i} cos {h.score:.3f} Item {h.section_key} lines {start}-{end}{mark}")
        lines.append(f"       {snippet}")
    return "\n".join(lines)


def _hit_at(ranks: dict[str, int | None], k: int, ids: list[str] | None = None) -> tuple[int, int]:
    chosen = [i for i in ranks if ids is None or i in ids]
    return sum(1 for i in chosen if (ranks[i] or 10**9) <= k), len(chosen)


def _in_top5(ranks: dict[str, int | None], row_id: str) -> bool:
    rank = ranks[row_id]
    return rank is not None and rank <= 5


def render_comparison(
    rows: list[EvalRow],
    ranks_by_name: dict[str, dict[str, int | None]],
    ks: tuple[int, ...] = KS,
    answer_ranks_by_name: dict[str, dict[str, int | None]] | None = None,
) -> str:
    names = list(ranks_by_name)
    answerable = [r for r in rows if r.answerable]
    lines = [
        "# Retrieval comparison (no model calls)",
        "",
        f"{len(answerable)} answerable rows. hit@k = the gold evidence is inside the top k chunks.",
        "",
        "| Retriever | " + " | ".join(f"hit@{k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for name in names:
        cells = [f"{a}/{n}" for a, n in (_hit_at(ranks_by_name[name], k) for k in ks)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    if answer_ranks_by_name:
        lines += [
            "",
            "## answer@k: some top-k chunk contains a gold answer string (looser and noisier than hit@k)",
            "",
            "| Retriever | " + " | ".join(f"answer@{k}" for k in ks) + " |",
            "|---|" + "---|" * len(ks),
        ]
        for name in names:
            cells = [f"{a}/{n}" for a, n in (_hit_at(answer_ranks_by_name[name], k) for k in ks)]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    fields = sorted({r.field for r in answerable})
    lines += ["", "## hit@5 by field", "", "| Field | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for field in fields:
        ids = [r.id for r in answerable if r.field == field]
        cells = [f"{a}/{n}" for a, n in (_hit_at(ranks_by_name[name], 5, ids) for name in names)]
        lines.append(f"| {field} | " + " | ".join(cells) + " |")

    base = names[0]
    for name in names[1:]:
        new, old = ranks_by_name[name], ranks_by_name[base]
        gained = [i for i in new if _in_top5(new, i) and not _in_top5(old, i)]
        lost = [i for i in new if _in_top5(old, i) and not _in_top5(new, i)]
        lines += ["", f"- {name} vs {base} at k=5: gained {gained or 'none'}; lost {lost or 'none'}."]
    still = [i for i in ranks_by_name[names[-1]] if not _in_top5(ranks_by_name[names[-1]], i)]
    lines += ["", f"Still missing at k=5 with {names[-1]}: {still or 'none'}."]
    lines += [
        "",
        "How to read this: rows are few, so differences of one or two rows are not findings. These rows were also used",
        "to choose the retriever, so the numbers are optimistic; a fresh held-out set is the honest test.",
    ]
    return "\n".join(lines) + "\n"


def build_variant_index(sections: Path, out: Path, embedder, keys: tuple[str, ...]) -> int:
    """Build a separate index (never touches data/index) that also covers the given section keys."""
    filings = [load_parsed(f) for f in sorted(sections.glob("*.json"))]
    chunks = [c for filing in filings for c in chunk_filing(filing, ChunkConfig(), keys=keys)]
    store = VectorStore(out)
    store.reset()
    if chunks:
        store.add(chunks, embedder.embed_documents([c.embed_text for c in chunks]))
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare retrievers on the labeled rows without calling a model.")
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    default_labels = [Path("eval/labels.csv"), Path("eval/labels_holdout.csv")]
    compare.add_argument("--labels", type=Path, nargs="+", default=default_labels)
    compare.add_argument("--index", type=Path, default=Path("data/index"))
    compare.add_argument("--index-cover", type=Path, default=None, help="an index built with build-cover")
    compare.add_argument("--report", type=Path, default=Path("reports/retrieval_comparison.md"))
    compare.add_argument("--model", default=DEFAULT_MODEL)
    show = sub.add_parser("show", help="print what each retriever returns for chosen rows")
    show.add_argument("--rows", nargs="+", required=True, help="row ids such as WMT-auditor")
    show.add_argument("--labels", type=Path, nargs="+", default=default_labels)
    show.add_argument("--index", type=Path, default=Path("data/index"))
    show.add_argument("-k", type=int, default=5, help="chunks to print per retriever")
    show.add_argument("--model", default=DEFAULT_MODEL)
    build = sub.add_parser("build-cover")
    build.add_argument("--sections", type=Path, default=Path("data/processed/sections"))
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    from filings_credit_memo.retrieval import SentenceTransformerEmbedder  # heavy import, only for real runs

    if args.command == "build-cover":
        if args.out.resolve() == Path("data/index").resolve():
            print("Refusing to overwrite data/index; choose another --out.", file=sys.stderr)
            return 2
        embedder = SentenceTransformerEmbedder(args.model)
        total = build_variant_index(args.sections, args.out, embedder, INDEX_KEYS + ("0",))
        print(f"Indexed {total:,} chunks (standard sections plus the cover page) into {args.out}")
        return 0

    rows: list[EvalRow] = []
    for path in args.labels:
        loaded, _ = load_rows(path)
        rows += loaded
    embedder = SentenceTransformerEmbedder(args.model)
    if args.command == "show":
        by_id = {r.id: r for r in rows}
        unknown = [i for i in args.rows if i not in by_id]
        if unknown:
            print(f"Unknown row id(s): {unknown}", file=sys.stderr)
            return 2
        store = VectorStore(args.index)
        hybrid = HybridRetriever(store, embedder)
        for row_id in args.rows:
            row = by_id[row_id]
            print(f"\n=== {row.id}: {row.question}\n  gold answer: {' | '.join(row.gold_answers)}")
            print(f"  gold evidence: {row.gold_evidence[:300]}")
            for name, retriever in (
                ("dense", lambda q, k, t: retrieve(store, embedder, q, k, t)),
                ("hybrid", hybrid.retrieve),
            ):
                print(describe_row(row, name, retriever(row.question, 50, row.ticker), args.k))
        return 0
    ranks: dict[str, dict[str, int | None]] = {}
    answer_ranks: dict[str, dict[str, int | None]] = {}
    indexes = [("", args.index)] + ([("cover ", args.index_cover)] if args.index_cover else [])
    for label, path in indexes:
        store = VectorStore(path)
        if store.count() == 0:
            print(f"The index at {path} is empty.", file=sys.stderr)
            return 2
        dense = lambda q, k, t, s=store: retrieve(s, embedder, q, k, t)  # noqa: E731
        hybrid = HybridRetriever(store, embedder).retrieve
        for name, retriever in ((f"{label}dense", dense), (f"{label}hybrid", hybrid)):
            ranks[name] = measure(rows, retriever)
            answer_ranks[name] = measure_answers(rows, retriever)
    report = render_comparison(rows, ranks, answer_ranks_by_name=answer_ranks)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
