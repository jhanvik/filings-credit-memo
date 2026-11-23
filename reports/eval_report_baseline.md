# Evaluation report

Model `ollama:llama3.1:8b`, top-k 5, refusal threshold 0.65. 42 labeled questions (30 answerable, 12 not answerable from the filing).

## Headline results

| Metric | Result |
|---|---|
| Retrieval hit@1 (answerable) | 13/30 = 43% (95% CI 27%-61%) |
| Retrieval hit@3 | 20/30 = 67% (95% CI 49%-81%) |
| Retrieval hit@5 | 22/30 = 73% (95% CI 56%-86%) |
| Answer accuracy (answerable) | 8/30 = 27% (95% CI 14%-44%) |
| Wrongful refusal (answerable) | 22/30 = 73% (95% CI 56%-86%) |
| Wrong answer shown (answerable) | 0/30 = 0% (95% CI 0%-11%) |
| Blocked by verification (answerable) | 0/30 = 0% (95% CI 0%-11%) |
| Correct refusal (not answerable) | 12/12 = 100% (95% CI 76%-100%) |
| False answer shown (not answerable) | 0/12 = 0% (95% CI 0%-24%) |
| Unsupported share of all answers shown | 0/8 = 0% (95% CI 0%-32%) |
| Citation contains the gold evidence (answered, answerable) | 5/8 = 62% (95% CI 31%-86%) |
| Latency, model calls only | n/a (every row was served from cache; see the earlier uncached run) |

"Unsupported" here means an answer that was shown but is wrong, or any answer shown to a question the filing does not answer. It is a measured proxy for hallucination, not a complete definition. Quotes are guaranteed verbatim by the verifier, so this measures whether the claim itself is right.

## By field

| Field | n | Correct | Wrong | Wrongful refusal | Blocked | Correct refusal | False answer |
|---|---|---|---|---|---|---|---|
| revenue | 6 | 0 | 0 | 6 | 0 | 0 | 0 |
| employees | 6 | 2 | 0 | 4 | 0 | 0 | 0 |
| revolver_maturity | 6 | 0 | 0 | 6 | 0 | 0 | 0 |
| covenants | 6 | 1 | 0 | 5 | 0 | 0 | 0 |
| cyber_oversight | 6 | 5 | 0 | 1 | 0 | 0 | 0 |
| going_concern | 6 | 0 | 0 | 0 | 0 | 6 | 0 |
| out_of_scope | 6 | 0 | 0 | 0 | 0 | 6 | 0 |

## Refusal threshold

The score threshold is applied before the model is called. Utility = correct − 2 × (wrong + false answers). The weight is an assumption, not a measurement: it says that a wrong answer shown is worse than a refusal.

| Threshold | Correct | Wrong | False answer | Wrongful refusal | Correct refusal | Utility |
|---|---|---|---|---|---|---|
| 0.50 | 9 | 0 | 0 | 21 | 12 | 9 |
| 0.55 | 9 | 0 | 0 | 21 | 12 | 9 |
| 0.60 | 9 | 0 | 0 | 21 | 12 | 9 |
| 0.65 (current) | 8 | 0 | 0 | 22 | 12 | 8 |
| 0.70 | 6 | 0 | 0 | 24 | 12 | 6 |
| 0.75 | 6 | 0 | 0 | 24 | 12 | 6 |
| 0.80 | 3 | 0 | 0 | 27 | 12 | 3 |
| 0.85 | 0 | 0 | 0 | 30 | 12 | 0 |

Best threshold on all rows: **0.55** (in-sample, so optimistic).

Company-wise 2-fold check (folds: AAL + DAL + F | CCL + DUK + WMT): thresholds chosen on the other half were 0.58 and 0.55. Out-of-fold utility 9, versus 8 at the current 0.65. If these two numbers are close, the data does not justify changing the threshold.

## Rows that need attention

- **CCL-revenue** — wrongful_refusal (top score 0.64, status not_found)
  - question: What were the company's total revenues for the most recent fiscal year?
  - gold: 26.6 billion | 26,622 million
  - model: (no answer)
- **CCL-employees** — wrongful_refusal (top score 0.64, status not_found)
  - question: How many people does the company employ?
  - gold: 160,000
  - model: (no answer)
- **CCL-revolver_maturity** — wrongful_refusal (top score 0.76, status not_found)
  - question: When does the company's main revolving credit facility mature?
  - gold: June 2030
  - model: (no answer)
- **CCL-covenants** — wrongful_refusal (top score 0.77, status not_found)
  - question: Do the company's debt agreements contain financial covenants, such as a minimum liquidity requirement?
  - gold: yes
  - model: (no answer)
- **DAL-revenue** — wrongful_refusal (top score 0.68, status not_found)
  - question: What were the company's total revenues for the most recent fiscal year?
  - gold: 63,364 million | 63.4 billion
  - model: (no answer)
- **DAL-employees** — wrongful_refusal (top score 0.60, status answered)
  - question: How many people does the company employ?
  - gold: 103,000
  - model: 103,000
- **DAL-revolver_maturity** — wrongful_refusal (top score 0.67, status not_found)
  - question: When does the company's main revolving credit facility mature?
  - gold: 2028 | 2026 to 2028
  - model: (no answer)
- **F-revenue** — wrongful_refusal (top score 0.71, status not_found)
  - question: What were the company's total revenues for the most recent fiscal year?
  - gold: 176.2 billion | 176,191 million
  - model: (no answer)
- **F-employees** — wrongful_refusal (top score 0.64, status not_found)
  - question: How many people does the company employ?
  - gold: 169,000
  - model: (no answer)
- **F-revolver_maturity** — wrongful_refusal (top score 0.77, status not_found)
  - question: When does the company's main revolving credit facility mature?
  - gold: April 2030 | April 17, 2030
  - model: (no answer)
- **F-covenants** — wrongful_refusal (top score 0.74, status not_found)
  - question: Do the company's debt agreements contain financial covenants, such as a minimum liquidity requirement?
  - gold: yes
  - model: (no answer)
- **DUK-revenue** — wrongful_refusal (top score 0.67, status not_found)
  - question: What were the company's total revenues for the most recent fiscal year?
  - gold: 32,237 million | 32.2 billion
  - model: (no answer)
- **DUK-revolver_maturity** — wrongful_refusal (top score 0.70, status not_found)
  - question: When does the company's main revolving credit facility mature?
  - gold: March 2030
  - model: (no answer)
- **DUK-covenants** — wrongful_refusal (top score 0.75, status not_found)
  - question: Do the company's debt agreements contain financial covenants, such as a minimum liquidity requirement?
  - gold: yes
  - model: (no answer)
- **DUK-cyber_oversight** — wrongful_refusal (top score 0.76, status not_found)
  - question: Which board committee oversees cybersecurity risk?
  - gold: Audit Committee
  - model: (no answer)
- **WMT-revenue** — wrongful_refusal (top score 0.70, status not_found)
  - question: What were the company's total revenues for the most recent fiscal year?
  - gold: 713.2 billion | 713,163 million
  - model: (no answer)
- **WMT-revolver_maturity** — wrongful_refusal (top score 0.74, status not_found)
  - question: When does the company's main revolving credit facility mature?
  - gold: April 2030 | April 2026
  - model: (no answer)
- **WMT-covenants** — wrongful_refusal (top score 0.69, status not_found)
  - question: Do the company's debt agreements contain financial covenants, such as a minimum liquidity requirement?
  - gold: yes
  - model: (no answer)
- **AAL-revenue** — wrongful_refusal (top score 0.67, status not_found)
  - question: What were the company's total revenues for the most recent fiscal year?
  - gold: 54.6 billion | 54,633 million | 54,626 million
  - model: (no answer)
- **AAL-employees** — wrongful_refusal (top score 0.66, status not_found)
  - question: How many people does the company employ?
  - gold: 139,100
  - model: (no answer)
- **AAL-revolver_maturity** — wrongful_refusal (top score 0.75, status not_found)
  - question: When does the company's main revolving credit facility mature?
  - gold: June 2029
  - model: (no answer)
- **AAL-covenants** — wrongful_refusal (top score 0.79, status not_found)
  - question: Do the company's debt agreements contain financial covenants, such as a minimum liquidity requirement?
  - gold: yes
  - model: (no answer)

## Limitations

- Small sample: confidence intervals above are wide, and a difference of one or two rows is not a finding.
- One labeler wrote the gold answers; there is no second annotator or agreement measure.
- Answer matching is lenient for very short golds (for example 'no'); inspect the per-row CSV.
- The threshold and any later prompt or chunking choices are tuned on the same rows, which makes the reported numbers optimistic. A fresh held-out set is the honest fix.
- Six companies, one filing each, one local 8B model: do not generalise beyond that.
