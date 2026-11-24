# Retrieval comparison (no model calls)

38 answerable rows. hit@k = the gold evidence is inside the top k chunks.

| Retriever | hit@1 | hit@3 | hit@5 | hit@10 |
|---|---|---|---|---|
| dense | 14/38 | 22/38 | 24/38 | 27/38 |
| hybrid | 15/38 | 22/38 | 26/38 | 30/38 |
| cover dense | 14/38 | 22/38 | 24/38 | 27/38 |
| cover hybrid | 16/38 | 23/38 | 26/38 | 31/38 |

## answer@k: some top-k chunk contains a gold answer string (looser and noisier than hit@k)

| Retriever | answer@1 | answer@3 | answer@5 | answer@10 |
|---|---|---|---|---|
| dense | 21/38 | 27/38 | 28/38 | 29/38 |
| hybrid | 18/38 | 26/38 | 28/38 | 32/38 |
| cover dense | 21/38 | 27/38 | 28/38 | 29/38 |
| cover hybrid | 19/38 | 25/38 | 27/38 | 32/38 |

## hit@5 by field

| Field | dense | hybrid | cover dense | cover hybrid |
|---|---|---|---|---|
| auditor | 1/4 | 2/4 | 1/4 | 1/4 |
| covenants | 4/6 | 6/6 | 4/6 | 6/6 |
| cyber_oversight | 6/6 | 6/6 | 6/6 | 6/6 |
| employees | 3/6 | 3/6 | 3/6 | 3/6 |
| fiscal_year_end | 1/4 | 1/4 | 1/4 | 2/4 |
| revenue | 3/6 | 2/6 | 3/6 | 2/6 |
| revolver_maturity | 6/6 | 6/6 | 6/6 | 6/6 |

- hybrid vs dense at k=5: gained ['DAL-employees', 'WMT-covenants', 'AAL-covenants', 'F-auditor']; lost ['CCL-employees', 'AAL-revenue'].

- cover dense vs dense at k=5: gained none; lost none.

- cover hybrid vs dense at k=5: gained ['DAL-employees', 'WMT-covenants', 'AAL-covenants', 'WMT-fiscal_year_end']; lost ['CCL-employees', 'AAL-revenue'].

Still missing at k=5 with cover hybrid: ['CCL-employees', 'DAL-revenue', 'F-revenue', 'F-employees', 'DUK-revenue', 'WMT-employees', 'AAL-revenue', 'DAL-fiscal_year_end', 'AAL-fiscal_year_end', 'F-auditor', 'DUK-auditor', 'WMT-auditor'].

How to read this: rows are few, so differences of one or two rows are not findings. These rows were also used
to choose the retriever, so the numbers are optimistic; a fresh held-out set is the honest test.
