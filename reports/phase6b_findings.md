# Phase 6b findings: what the first held-out run taught us

Written after the one-time held-out run. Numbers come from `reports/ablation_report.md`, `reports/holdout_report.md`
and `reports/retrieval_comparison.md`. Everything measured on the dev rows or the spent held-out rows is optimistic.

## 1. The dev gain did not carry over

- Dev (30 answerable, 12 not): the reasoning prompt (`base_reason_k5`) raised correct answers from 8/30 to 13/30 under a
  rule fixed in advance (utility, ties to the simpler setting). Intervals at n=30 are about +/-15 points.
- Held-out (8 answerable, 4 not, never used for choosing anything): **0/8 correct for both configurations**, 4/4 correct
  refusals (weak evidence, the system refuses most things). `base_reason_k5` also showed 1 wrong answer.
- The held-out rows were harder and of different types (fiscal year end, auditor). Eight rows is too few to rank
  configurations; the result says the dev gain should not be quoted as a general improvement.

## 2. Why the held-out rows failed (diagnosed from per-row output)

| Cause | Rows | Evidence |
|---|---|---|
| Gold evidence not in the top 5 chunks | 6 of 8 | blank `hit_rank` in the per-row CSV |
| Verified but wrong answer | CCL fiscal year end | model quoted a prior-year sentence (2023); the quote check only proves the quote exists, not that it answers the question |
| Correct answer blocked by the refusal threshold | AAL auditor | answer "KPMG LLP" was right and verified, but top score 0.618 < 0.65 |

## 3. Retrieval experiments (no model calls; 38 answerable tuning rows)

| Retriever | hit@5 | hit@10 | answer@5 |
|---|---|---|---|
| dense (current) | 24/38 | 27/38 | 28/38 |
| hybrid (dense + BM25, RRF) | 26/38 | 30/38 | 28/38 |
| dense + cover page indexed | 24/38 | 27/38 | 28/38 |
| hybrid + cover page indexed | 26/38 | 31/38 | 27/38 |

- Adoption rule set before the comparison: hybrid must beat dense by at least 3 rows on both hit@5 and answer@5, with no
  field losing more than 1 row. It gained 2 and 0, so **dense stays the default**. Hybrid is kept as an experiment.
- hit@k understates retrieval: the gold label names one sentence, but other retrieved chunks often also answer the
  question (for example every top chunk for the WMT auditor question was an audit report).
- answer@k is a loose string test and is meaningless for dates, which appear in almost every chunk.
- Hybrid hurt CCL-employees: the question says "employ", the filing says "employees", and the keyword side has no match.
  A standard stemmer (Snowball) does not fix this ("employees" -> "employe", "employ" -> "employ").
- Indexing the cover page did not help dense search: the "For the fiscal year ended" chunk ranked outside the top 50.

## 4. Decision for fiscal year end

Answer it from filing metadata (`report_date`, the period of report), not from retrieval: exact, no model call, with
provenance (accession, filing date, URL). Implemented in `facts.py` with a deliberately narrow router. It scores 4/4 on
the labeled fiscal year end rows. **This is a metadata lookup, not a RAG result, and is reported separately.**

## 5. Known limitations carried into the README

- Revenue and employee-count questions often miss in the top 5 (large statement tables, repeated figures).
- Filings with several registrants (Duke Energy and its subsidiaries) contain many near-duplicate tables; the
  consolidated total can rank far down.
- A valid citation does not prove a correct answer; the verifier checks the quote, not that it answers the question.
- The refusal threshold uses the top chunk's similarity and can block correct answers for some question types.
- The retriever and prompt were chosen on the same rows used for diagnosis. The honest test is a fresh held-out set.

## 6. Next

Write a fresh held-out set on narrative fields not tuned on (for example Item 5 dividends/repurchases, Item 2
headquarters, Item 3 legal proceedings, plus unanswerable rows), freeze the pipeline, run it once, report it as it comes.
