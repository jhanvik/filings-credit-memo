# Held-out evaluation (used once)

8 answerable and 4 not-answerable questions. Refusal threshold 0.65. Utility = correct - 2 x (wrong + false answers); the weight is an assumption.

| Config | Prompt | Reasoning | k | Correct | Wrong | Wrongful refusal | False answer | Correct refusal | Utility | hit@k | Gained | Lost | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_k5 | base | no | 5 | 0/8 | 0 | 8 | 0/4 | 4/4 | 0 | 2/8 | 0 | 0 | 15.1s (n=12) |
| base_reason_k5 | base | yes | 5 | 0/8 | 1 | 7 | 0/4 | 4/4 | -2 | 2/8 | 0 | 0 | 4.8s (n=12) |

Highest utility by the pre-set rule: **base_k5** (held-out run: configuration chosen on the dev set before this run).
- base_reason_k5 vs base_k5: gained none; lost none.

## How to read this

- Wilson 95% intervals on rates from this few questions (see the row counts above) are very wide. One or two rows are not a finding.
- False answers are the cost of answering more: a looser prompt that raises Correct but also raises False answer has not necessarily improved.
- 'Gained' and 'Lost' count rows (answerable or not) whose result flipped between right and wrong compared with the baseline.
- This is a held-out run made once: the configuration was chosen on the dev set, not on these rows. The rows were used afterwards to diagnose the failure (see phase6b_findings.md), so they are not a clean test for later changes. (Footer corrected after the run; the results were not changed.)
