"""Command line: download the latest 10-K for each ticker from SEC EDGAR.

Usage (from the project root):
    uv run python -m filings_credit_memo.download CCL DAL F DUK WMT AAL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from filings_credit_memo.edgar import EdgarClient, EdgarError, download_latest, load_user_agent

DEFAULT_OUT = Path("data/raw/filings")
DEFAULT_CACHE = Path("data/cache")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the latest filing of a form type for each ticker.")
    parser.add_argument("tickers", nargs="+", help="e.g. CCL DAL F")
    parser.add_argument("--form", default="10-K")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)

    try:
        user_agent = load_user_agent()
    except EdgarError as err:
        print(f"Setup problem: {err}", file=sys.stderr)
        return 2

    failures = 0
    with EdgarClient(user_agent, args.cache) as client:
        for ticker in args.tickers:
            try:
                path = download_latest(client, ticker, args.out, args.form)
                print(f"{ticker.upper():6s} saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
            except EdgarError as err:
                failures += 1
                print(f"{ticker.upper():6s} FAILED: {err}", file=sys.stderr)
        print(f"Network requests made this run: {client.network_requests}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
