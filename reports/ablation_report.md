# Ablation report (dev questions)

30 answerable and 12 not-answerable questions. Refusal threshold 0.65. Utility = correct - 2 x (wrong + false answers); the weight is an assumption.

| Config | Prompt | Reasoning | k | Correct | Wrong | Wrongful refusal | False answer | Correct refusal | Utility | hit@k | Gained | Lost | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_k5 | base | no | 5 | 8/30 | 0 | 22 | 0/12 | 12/12 | 8 | 22/30 | 0 | 0 | cached |
| base_reason_k5 | base | yes | 5 | 13/30 | 1 | 14 | 0/12 | 12/12 | 11 | 22/30 | 6 | 1 | cached |
| relaxed_k5 | relaxed | no | 5 | 8/30 | 0 | 19 | 0/12 | 12/12 | 8 | 22/30 | 1 | 1 | cached |
| relaxed_reason_k5 | relaxed | yes | 5 | 14/30 | 2 | 9 | 0/12 | 11/12 | 10 | 22/30 | 8 | 3 | cached |
| relaxed_k3 | relaxed | no | 3 | 12/30 | 1 | 15 | 0/12 | 12/12 | 10 | 20/30 | 5 | 1 | cached |
| relaxed_reason_k3 | relaxed | yes | 3 | 14/30 | 3 | 10 | 0/12 | 12/12 | 8 | 20/30 | 7 | 1 | cached |

Highest utility by the pre-set rule: **base_reason_k5** (in-sample, so optimistic).
- base_reason_k5 vs base_k5: gained ['CCL-revolver_maturity', 'CCL-covenants', 'DAL-revolver_maturity', 'DUK-revolver_maturity', 'WMT-revolver_maturity', 'AAL-revenue']; lost ['CCL-cyber_oversight'].
- relaxed_k5 vs base_k5: gained ['AAL-revenue']; lost ['CCL-cyber_oversight'].
- relaxed_reason_k5 vs base_k5: gained ['CCL-revolver_maturity', 'CCL-covenants', 'DAL-revolver_maturity', 'F-covenants', 'DUK-revolver_maturity', 'DUK-cyber_oversight', 'AAL-revenue', 'AAL-covenants']; lost ['CCL-cyber_oversight', 'WMT-employees', 'WMT-out_of_scope'].
- relaxed_k3 vs base_k5: gained ['CCL-revolver_maturity', 'CCL-covenants', 'F-covenants', 'DUK-cyber_oversight', 'AAL-revolver_maturity']; lost ['CCL-cyber_oversight'].
- relaxed_reason_k3 vs base_k5: gained ['CCL-revolver_maturity', 'CCL-covenants', 'DUK-revolver_maturity', 'DUK-cyber_oversight', 'WMT-revolver_maturity', 'AAL-revolver_maturity', 'AAL-covenants']; lost ['CCL-cyber_oversight'].

## How to read this

- Wilson 95% intervals on a rate from 30 questions are about +/-15 percentage points wide. One or two rows are not a finding.
- False answers are the cost of answering more: a looser prompt that raises Correct but also raises False answer has not necessarily improved.
- 'Gained' and 'Lost' count rows (answerable or not) whose result flipped between right and wrong compared with the baseline.
- The prompts, the k values and the choice rule were decided by one person on these same questions. Use the held-out run for the honest number.
