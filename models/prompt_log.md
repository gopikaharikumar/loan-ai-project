# Prompt log

Edit this file: add who ran each prompt, the date, and anything else your team did. The entries below summarise what happened so far.

| # | Component | Prompt (summary) | What went wrong | What we changed |
|---|---|---|---|---|
| 1 | Dataset | Asked the AI assistant to generate a synthetic personal-loan dataset with a realistic, noisy default label | The first idea (label by policy rules) would let the model rediscover the rule and score about 99% | Label drawn from a hidden default probability plus noise; documented that the data is synthetic |
| 2 | Dataset | Checked the first dataset with logistic regression and boosting | Default rate 25.6% (too high); weak gender proxy effect | Lowered loan sizes and intercept (default rate 14%); added an assumed pay gap and shorter credit history for women |
| 3 | Policy | Asked for a numbered, single-rule lending policy matched to dataset columns, with planted edge cases | Thresholds had to match the model features | 65 clauses; planted a thin-file vs high-income conflict and a gap (age at loan maturity) |
| 4 | Model | Asked for baseline, logistic regression and gradient boosting with a cost-based threshold | The naive 0.5 threshold caught only 30% of defaulters | Chose threshold 0.11 from error costs on validation; kept logistic regression (it beat boosting) |
| 5 | RAG v0 (no policy) | In-app prompt: explain the decision using general knowledge only | Invented loan amount (10x error), invented timelines and promises, no "not covered" statement, no clause cited | Kept as the "without RAG" comparison |
| 6 | RAG v1 | In-app prompt with retrieved policy clauses, cite clause numbers | Cited a non-existent "Clause 7"; irrelevant citation (11.1); recited internal rules; used jargon (PD); fairness wording leaked into customer text | Rewrote the rules: speak to the applicant, no jargon, no internal procedures; stopped passing clauses 8.5, 8.6 and 11.1; check now catches bare section citations |
| 7 | RAG v2 | Same, with the v1 fixes | Income-typo case copied a policy sentence word for word; the rejection mixed in a review-only reason | Rejections now get rejection reasons only; added "use your own words"; added a verbatim-copy style check |
| 8 | Checker | Automated check for protected terms | Flagged "low risk tier" as a protected term (false alarm) | Check now looks for "city" only |
| 9 | LLM call | Called gemini-2.5-flash-lite | 404: model no longer available to new users | Switched the default to gemini-3.5-flash-lite; the model name is one environment variable |
| 10 | App | Asked for a Streamlit app with a human checkpoint and audit log | A blocked override looked unblocked in a test, caused by test sequencing, not the app | Re-tested with the correct widget order: an override of credit score below 600 or DTI above 0.50 is blocked |
