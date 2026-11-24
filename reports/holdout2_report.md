# Held-out evaluation (used once)

9 answerable and 3 not-answerable questions. Refusal threshold 0.65. Utility = correct - 2 x (wrong + false answers); the weight is an assumption.

| Config | Prompt | Reasoning | k | Correct | Wrong | Wrongful refusal | False answer | Correct refusal | Utility | hit@k | Gained | Lost | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_k5 | base | no | 5 | 2/9 | 0 | 7 | 0/3 | 3/3 | 2 | 8/9 | 0 | 0 | 14.9s (n=12) |
| base_reason_k5 | base | yes | 5 | 4/9 | 0 | 5 | 0/3 | 3/3 | 4 | 8/9 | 2 | 0 | 5.5s (n=12) |

Highest utility by the pre-set rule: **base_reason_k5** (held-out run: pipeline frozen before this run).
- base_reason_k5 vs base_k5: gained ['DAL-dividends_paid', 'WMT-dividends_paid']; lost none.

## How to read this

- Wilson 95% intervals on rates from this few questions (see the row counts above) are very wide. One or two rows are not a finding.
- False answers are the cost of answering more: a looser prompt that raises Correct but also raises False answer has not necessarily improved.
- 'Gained' and 'Lost' count rows (answerable or not) whose result flipped between right and wrong compared with the baseline.
- This is a held-out run made once with the pipeline frozen beforehand; the configurations were chosen on the dev set, not on these rows. (Footer corrected after the run; the results were not changed.)
