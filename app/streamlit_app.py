"""Demo: read a generated credit memo and open the source text behind every citation.

    uv run streamlit run app/streamlit_app.py

The app only reads memos saved by `python -m filings_credit_memo.memo TICKER`. It does not call the model, so it runs
anywhere. All filing and model text is shown with st.text or cleaned markdown, never as raw HTML or links.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from filings_credit_memo.memo import DISCLAIMER, MISSING_NOTE, Memo, clean, load_memo, saved_memos

MEMO_DIR = Path("data/outputs/memos")
STATUS_LABEL = {"ok": "", "missing": "Not determined", "withheld": "Withheld (failed verification)", "error": "Error"}


def show_entry(memo: Memo, text: str, chunk_id: str, quote: str, key: str) -> None:
    st.markdown(f"- {text}")
    if not chunk_id:
        return
    with st.expander(f"Source: {chunk_id}"):
        st.caption("Quote the model gave (verified to appear verbatim in the chunk below)")
        st.text(quote)
        ev = memo.evidence.get(chunk_id)
        if ev:
            st.caption(f"Full source chunk, section {ev['section']}, lines {ev['lines']}")
            st.text(ev["text"])
        else:
            st.warning("The source chunk was not saved with this memo.")


def show_memo(memo: Memo) -> None:
    st.title(f"Credit memo: {memo.ticker}")
    st.warning(DISCLAIMER)
    for label, value in memo.header.items():
        if label == "Source" and value.startswith("https://www.sec.gov/"):
            st.markdown(f"**{label}:** [{clean(value, 200)}]({value})")
        else:
            st.markdown(f"**{label}:** {clean(value, 200)}")
    st.caption("Fiscal year end comes from the filing record, not from the language model.")
    for i, sec in enumerate(memo.sections):
        if sec.key == "qa":
            st.markdown(f"**{sec.title}**")
        else:
            st.subheader(sec.title)
        if sec.status == "ok":
            for j, e in enumerate(sec.entries):
                show_entry(memo, e.text, e.chunk_id, e.quote, f"{i}-{j}")
            if sec.note:
                st.caption(sec.note)
        else:
            st.info(f"{STATUS_LABEL.get(sec.status, sec.status)}: {sec.note or MISSING_NOTE}")
    st.divider()
    st.caption(f"Model: {memo.model or 'unknown'}. Retrieval similarity threshold: {memo.min_score}.")


def main() -> None:
    st.set_page_config(page_title="Filings credit memo", layout="wide")
    memos = saved_memos(MEMO_DIR)
    if not memos:
        st.error("No memos found. Generate one first: uv run python -m filings_credit_memo.memo CCL")
        return
    ticker = st.sidebar.selectbox("Company", list(memos))
    st.sidebar.caption("Memos are generated offline with a local model; this page only displays them.")
    show_memo(load_memo(memos[ticker]))


main()
