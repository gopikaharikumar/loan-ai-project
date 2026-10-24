# AI-Powered Loan Decision and Explanation System

> **Educational prototype. Not financial, medical or legal advice. All applicant data is synthetic.**

Foundations of AI for Business, Final Group Project 1 (Finance).
A mid-sized retail bank prototype: a risk model recommends a decision, a RAG component explains it using the bank's lending policy, and a loan officer confirms every rejection and review case.

## How it works
1. `decision_engine.py` builds features, scores the applicant with a logistic regression risk model, applies the policy rules and risk bands, and records which policy clauses fired.
2. `rag.py` retrieves those clauses from `policy/lending_policy.md`, prompts an LLM (Gemini) to write a draft explanation that cites them, and runs automated checks (invented clauses, unsupported numbers, protected terms, changed decision).
3. `app.py` is the Streamlit app with the **human checkpoint**: the draft goes to a loan officer, overrides need a written reason (and Senior Credit Manager approval for "not eligible" rules), and every decision goes to an audit log.

The LLM only explains the decision. It never makes or changes it.

## Repo structure
```
app.py                     Streamlit app (entry point)
rag.py                     clause retrieval, prompt, LLM call, automated checks
decision_engine.py         features, risk model, policy rules
requirements.txt
policy/lending_policy.md   lending policy (the RAG source)
models/                    risk_model.joblib, prep_params.json, metrics.json, figures
data/                      synthetic applicants and train/val/test splits
src/                       generate_loan_dataset.py, prepare_data.py, train_model.py
prompt_log.md              prompt log (appendix deliverable)
```

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # optional: without it the app uses template explanations
streamlit run app.py
```
On Streamlit Community Cloud, put the key in the app's **Secrets** settings: `GEMINI_API_KEY = "your-key"`. Never commit keys.

## Reproduce the model
```bash
python src/generate_loan_dataset.py
python src/prepare_data.py --input data/loan_applicants.csv --outdir data/processed
python src/train_model.py --datadir data/processed --outdir models
```

## Key results (synthetic test set, 1,600 applicants, 14% default rate)
| Model | Precision | Recall | Total cost |
|---|---|---|---|
| Baseline: approve everyone | 0.00 | 0.00 | Rs 4.74 crore |
| Logistic regression, threshold 0.5 | 0.66 | 0.30 | Rs 2.81 crore |
| Logistic regression, cost-based threshold 0.11 | 0.31 | 0.79 | Rs 1.87 crore |

Cost assumptions: a missed default costs 50% of the loan, a wrongly rejected good customer costs 8% of the loan.

## Limits
Synthetic data, assumed costs, about 29% of good applicants rejected at the cost-based threshold, automated checks cannot judge clause relevance, only known policy gaps are detected automatically. See the app's "About and limits" tab.
