"""Command line for Phase 3.

    uv run python -m filings_credit_memo.index build
    uv run python -m filings_credit_memo.index search "What debt does the company have maturing in 2027?" --ticker AAL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from filings_credit_memo.chunk import chunk_filing
from filings_credit_memo.parse import load_parsed
from filings_credit_memo.retrieval import DEFAULT_MODEL, SentenceTransformerEmbedder, VectorStore, build_index, retrieve

SECTIONS = Path("data/processed/sections")
INDEX = Path("data/index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and query the local filings index.")
    parser.add_argument("--sections", type=Path, default=SECTIONS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="chunk, embed and index every parsed filing")
    search = sub.add_parser("search", help="show the top chunks for a question")
    search.add_argument("question")
    search.add_argument("--ticker")
    search.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.command == "build":
        files = sorted(args.sections.glob("*.json"))
        if not files:
            print(f"No parsed filings in {args.sections}. Run filings_credit_memo.parse first.", file=sys.stderr)
            return 2
        filings = [load_parsed(f) for f in files]
        for filing in filings:
            chunks = chunk_filing(filing)
            print(f"{filing.ticker:5s} {len(chunks):5,} chunks, {sum(len(c.text) for c in chunks):>10,} chars")
        embedder = SentenceTransformerEmbedder(args.model, show_progress=True)
        total = build_index(filings, embedder, VectorStore(args.index))
        print(f"Indexed {total:,} chunks with {args.model} into {args.index}")
        return 0

    store = VectorStore(args.index)
    if store.count() == 0:
        print("The index is empty. Run the build command first.", file=sys.stderr)
        return 2
    for rank, hit in enumerate(retrieve(store, SentenceTransformerEmbedder(args.model), args.question, args.k, args.ticker), 1):
        start, end = hit.lines
        print(f"\n#{rank}  score {hit.score:.3f}  {hit.ticker} Item {hit.section_key}  lines {start}-{end}")
        print(hit.text[:500].replace("\n", " ") + ("..." if len(hit.text) > 500 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
