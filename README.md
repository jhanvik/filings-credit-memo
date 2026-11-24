# Filings credit memo assistant (with evaluation)

![CI](https://github.com/jhanvik/filings-credit-memo/actions/workflows/ci.yml/badge.svg)

A local, free-to-run pipeline that reads SEC 10-K filings, extracts credit-relevant facts with a small language model,
**checks every extracted item against the filing text**, and assembles a cited credit memo. Most of the project is about
measuring when the system is right, wrong or silent, not about the demo.

> **Status: a learning and portfolio project, not a credit tool.** It uses an 8B local model, six companies and small
> hand-labeled evaluation sets. The results below are honest about how weak that makes it. Nothing here is investment
> or credit advice.

## What it does

```
EDGAR 10-K
  -> parse into Items
  -> section-aware chunks
  -> embeddings (ChromaDB)
  -> retrieve
  -> local LLM -> JSON (Pydantic schema)
  -> quote check (verbatim in the cited chunk)
  -> credit memo (verified items only)

Fiscal year end: filing metadata (no model) -> credit memo
```

- **Structured extraction:** risk factors, debt instruments, covenants and going-concern language, each with a chunk id
  and a verbatim quote. Items whose quote is not found word for word in the cited chunk are rejected and counted.
- **Refusal:** if the best retrieved chunk is only weakly related (cosine similarity below 0.65, a provisional threshold),
  the system says "insufficient evidence" and does not call the model.
- **Metadata route:** the fiscal year end comes from the filing record (period of report), not from retrieval. This is a
  lookup, not a RAG result, and is reported separately below.
- **Memo and demo:** a fixed template fills only verified, cited items; a Streamlit page shows the memo and opens the
  source text behind each citation. The template has no rating, recommendation or advice section.
- **Untrusted text:** filings and model output are treated as untrusted. The prompts say to ignore instructions inside
  the excerpts, and text is stripped of links, images and HTML before it reaches the markdown memo.

## Data

SEC EDGAR annual reports (Form 10-K) for six companies: Carnival (CCL), Delta Air Lines (DAL), Ford (F), Duke Energy
(DUK), Walmart (WMT) and American Airlines (AAL). Filings are downloaded with `filings_credit_memo.download`, which
requires a contact email in a `SEC_USER_AGENT` environment variable (see `.env.example`). The `data/` folder is not
committed. SEC's website policy says information on sec.gov is public information that may be copied or further
distributed by users ([policy](https://www.sec.gov/files/privacy.htm), read on 2026-11-24); it does not address rights in
company-authored text, and I have not verified that point. The SEC's fair-access limits apply to downloads (the client caps
at 5 requests per second by default). This project is not affiliated with or endorsed by the SEC and uses no SEC or EDGAR logos.

## Models and tools

Python 3.12, Ollama with `llama3.1:8b`, `BAAI/bge-small-en-v1.5` embeddings, ChromaDB, Pydantic, httpx, Streamlit,
pytest, ruff, uv. Everything runs locally and costs nothing per call.

## Evaluation design

- Questions and gold answers were written by me **by reading the filings and searching their text**, not by looking at
  what the retriever returns. Each gold answer has a gold evidence sentence that must appear verbatim in the cleaned filing.
- Every question is run once with the refusal threshold switched off; the 0.65 threshold is applied afterwards, so the
  effect of refusing is measurable.
- Outcomes per question: correct, wrong, wrongful refusal (answer existed), false answer (unanswerable question was
  answered), correct refusal. Intervals are Wilson 95%. `utility = correct - 2 x (wrong + false answers)`; the weight of
  2 is an assumption.
- Configurations were chosen by a rule written before looking at results (utility, ties to fewer wrong answers, smaller
  k, no extra reasoning field).
- Two **held-out sets** were written separately and each run **once**, guarded in code (`--confirm-final-run` and a
  `HOLDOUT*_USED.json` file). For the first, I committed the labels after the run rather than before; the order is noted
  here because it is weaker than committing first.

## Results

Two configurations are compared: `base_k5` (answer directly from the top 5 chunks) and `base_reason_k5` (the model first
writes one sentence saying which excerpt answers the question). Full tables are in `reports/`.

| Question set | Answerable | `base_k5` correct | `base_reason_k5` correct | Wrong answers (base / reasoning) | False answers (base / reasoning) | Correct refusals (base / reasoning) |
|---|---|---|---|---|---|---|
| Dev (30 answerable, 12 not) | 30 | 8/30 (27%, CI 14-44%) | 13/30 (43%, CI 27-61%) | 0 / 1 | 0/12 / 0/12 | 12/12 / 12/12 |
| Held-out 1 (8 answerable, 4 not) | 8 | 0/8 | 0/8 | 0 / 1 | 0/4 / 0/4 | 4/4 / 4/4 |
| Held-out 2 (9 answerable, 3 not), frozen pipeline | 9 | 2/9 (22%, CI 6-55%) | 4/9 (44%, CI 19-73%) | 0 / 0 | 0/3 / 0/3 | 3/3 / 3/3 |

How to read this honestly:

- **Dev numbers are optimistic.** The prompt, k and selection rule were chosen on those rows. The dev gain (8 to 13) did
  not show up on held-out 1.
- **Held-out 1 failed (0/8).** Diagnosis from the per-row output (`reports/phase6b_findings.md`): for 6 of 8 rows the gold
  evidence was not in the top 5 chunks; one answer was verified but wrong (a prior-year date); one correct answer was
  blocked by the refusal threshold (top score 0.618 < 0.65).
- **Held-out 2 used the frozen pipeline** and is the cleanest number: 4/9 with reasoning, 2/9 without. Intervals are very
  wide and the sets are tiny, so this is evidence of direction, not a benchmark. Retrieval was not the main problem here
  (the gold evidence was in the top 5 for 8 of 9 rows); the 8B model often answered "not found", including when the true
  answer is an absence (for example "no dividends paid").
- **The system is safe but timid:** 0 false answers on 19 unanswerable questions in total. With so few unanswerable rows,
  I cannot claim a low false-answer rate, only that none was observed.
- **Fiscal year end (4/4 labeled rows) comes from filing metadata**, not from the model, after a first attempt through
  retrieval failed. It should not be counted as RAG accuracy.
- **Hybrid retrieval was tested and not adopted.** Dense + BM25 gained 2 rows at hit@5 (26/38 vs 24/38) against an
  adoption rule that required 3 on both measures. Details in `reports/retrieval_comparison.md`.
- The footers of `reports/holdout_report.md` and `reports/holdout2_report.md` had inherited dev-set boilerplate ("30
  questions", "in-sample", "decided on these same questions"). I corrected that text after the runs; the results were not touched.
- Latency: roughly 15 s median per question for a first, uncached call in the baseline runs on a MacBook with an
  8B model. I did not benchmark anything else.

## What the memo does and does not guarantee

Each memo item has a verbatim quote from the cited chunk, and numbers in an answer must appear in the quote. **That does
not prove the quote answers the question.** Examples from the generated memos:

- Duke Energy debt: a model listed a "364-day term loan facility (matures March 2024)" from a sentence saying the loan was
  *entered into* in March 2024, and listed two facilities from sentences saying they were *repaid and terminated*.
- Delta debt: a table row with maturities "2026 to 2028" was reduced to one year.

The memo layer now labels these (`date in quote, not confirmed as maturity`, `quote says repaid or terminated`,
`quote shows a range of maturities`) with simple text checks. These are heuristics, not proof, and the extractor itself
was not changed so earlier evaluation numbers stay valid. Lists are marked partial because the model sees only the best
few passages. "Going concern" is "not determined" for all six companies, and the memo cannot tell "no doubt stated" from
"not found".

## Limitations

- Six companies, one filing each, hand labels written by one person. No claim generalizes beyond this.
- An 8B local model refuses or misses a lot; absence statements ("none") are often returned as not found.
- Revenue and employee-count questions often miss in the top 5 (large tables, repeated figures); filings with several
  registrants (Duke Energy and subsidiaries) contain near-duplicate tables.
- The refusal threshold looks only at the top chunk's similarity and was set by eye, not calibrated.
- Tables parse poorly into text; debt tables especially.
- Quote verification checks existence, not meaning (see above). A second check that the quote supports the claim is future work.
- The Streamlit demo displays saved memos; it does not generate them, so it can run without the model.
- Not done: calibration of the threshold, table-aware chunking, allowing "answered: none" for absence statements,
  multi-registrant handling, multiple years per company.

## Where this sits relative to model risk guidance

US bank model risk guidance SR 11-7 was replaced by SR 26-2 (April 17, 2026). Its footnote 3 says generative AI and
agentic AI models "are novel and rapidly evolving. As such, they are not within the scope of this guidance," and it
continues with a "Nonetheless" sentence about governance practices that I have not reproduced here: read it in the
[primary source](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf). This project uses a generative
model, so SR 26-2 would not formally govern it, but the habits it practices (a labeled evaluation set, held-out testing,
documented limitations, refusal when evidence is missing, human verification) are the ones a bank would still want. I
am not a lawyer or a regulator; this is not a compliance statement.

## How to run

```
git clone https://github.com/jhanvik/filings-credit-memo.git
cd filings-credit-memo
uv sync
cp .env.example .env
```

Edit `.env` so `SEC_USER_AGENT` has your real name and email (the SEC requires it). Install and start Ollama, then:

```
ollama pull llama3.1:8b
uv run python -m filings_credit_memo.download CCL DAL F DUK WMT AAL
uv run python -m filings_credit_memo.parse
uv run python -m filings_credit_memo.index build
uv run python -m filings_credit_memo.memo CCL
uv run streamlit run app/streamlit_app.py
```

Tests and lint: `uv run pytest -q` and `uv run ruff check src tests app`. The tests use fakes and need neither Ollama
nor the data. In a clean clone, 224 pass and 7 skip themselves because they check the parser and chunker against the
real downloaded filings; run the downloader and parser first to include them.

### Docker (demo viewer only)

The image runs the Streamlit viewer. It does not include the model, the filings or any memos: mount your generated memos.

```
docker build -t filings-credit-memo .
docker run --rm -p 127.0.0.1:8501:8501 -v "$PWD/data/outputs/memos:/app/data/outputs/memos:ro" filings-credit-memo
```

Then open http://localhost:8501. Publishing the port on `127.0.0.1` keeps the viewer off your local network. The image contains the full locked dependency set (including PyTorch), so it is large.
`.env` is excluded by `.dockerignore`; never bake keys or filings into an image.

Evaluation commands (see each module's docstring): `evaluate check|run`, `ablate run|holdout`,
`retrieval_eval compare|show`, `facts check|ask`.

## Project structure

```
src/filings_credit_memo/
  edgar.py download.py     fetch filings from EDGAR (user agent, rate limiting, cache)
  parse.py chunk.py        clean HTML, split into Items, section-aware chunks
  index.py retrieval.py    embeddings, ChromaDB store, dense retrieval
  hybrid.py                dense + BM25 experiment (not adopted)
  llm.py                   Ollama client with response cache
  qa.py extract.py         cited answers and structured extraction with quote verification
  facts.py                 fiscal year end from filing metadata
  memo.py                  memo template, safe rendering, saved-memo loading
  evalset.py evaluate.py ablate.py retrieval_eval.py   evaluation tooling
app/streamlit_app.py       memo viewer with source expanders
eval/                      hand-labeled questions (dev, held-out 1, held-out 2)
reports/                   evaluation reports and findings
tests/
```
