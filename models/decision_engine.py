"""
Decision engine (EDUCATIONAL PROTOTYPE): features -> model PD -> policy rules -> recommendation.

Clause numbers refer to policy/lending_policy.md. The engine decides WHICH clauses fired;
the LLM later only explains them. Used by both the notebook and the Streamlit app.

    python decision_engine.py          # runs the demo applicants
"""
import json
from pathlib import Path

import joblib
import pandas as pd

_here = Path(__file__).resolve().parent
ROOT = next((p for p in (_here, _here.parent, Path.cwd()) if (p / "models").exists()), _here)
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
PARAMS_PATH = ROOT / "models" / "prep_params.json"
ANNUAL_RATE = 0.14                  # same assumption as the data generator
APPROVE_BELOW, REJECT_AT = 0.10, 0.25   # policy 5.2-5.4

_art = None
_params = None


def _load():
    global _art, _params
    if _art is None:
        _art = joblib.load(MODEL_PATH)
        _params = json.load(open(PARAMS_PATH))
    return _art, _params


def emi(amount, months, rate=ANNUAL_RATE):
    r = rate / 12
    return amount * r * (1 + r) ** months / ((1 + r) ** months - 1)


def build_features(a: dict) -> pd.DataFrame:
    """Raw applicant dict -> the exact model feature row (same logic as prepare_data.py)."""
    _, params = _load()
    med = params["train_medians"]
    income = a.get("monthly_income")
    score = a.get("credit_score")
    income_outlier = income is not None and income > params["income_verify_limit"]
    new_emi = emi(a["loan_amount"], a["tenure_months"])
    inc_for_ratio = income if income else med["monthly_income"]
    row = {
        "age": a["age"],
        "years_in_current_job": a.get("years_in_current_job") if a.get("years_in_current_job") is not None
                                else med["years_in_current_job"],
        "monthly_income": med["monthly_income"] if (income is None or income_outlier) else income,
        "existing_monthly_emi": a.get("existing_monthly_emi", 0),
        "credit_score": med["credit_score"] if score is None else score,
        "credit_history_years": a["credit_history_years"],
        "num_past_delinquencies": a["num_past_delinquencies"],
        "loan_amount": a["loan_amount"],
        "tenure_months": a["tenure_months"],
        "proposed_emi": round(new_emi),
        "dti_ratio": round((a.get("existing_monthly_emi", 0) + new_emi) / inc_for_ratio, 3),
        "loan_to_income": round(a["loan_amount"] / (inc_for_ratio * 12), 3),
        "credit_score_missing": int(score is None),
        "thin_file": int(a["credit_history_years"] < 1 or score is None),
        "emp_self_employed": int(a["employment_type"] == "Self-employed"),
        "emp_business_owner": int(a["employment_type"] == "Business owner"),
    }
    return pd.DataFrame([row])[params["feature_cols"]], income_outlier


def evaluate(a: dict) -> dict:
    """Return PD, fired clauses, recommendation and whether a human must review."""
    art, _ = _load()
    X, income_outlier = build_features(a)
    pd_ = float(art["model"].predict_proba(X)[0, 1])
    f = X.iloc[0]
    dti, lti = f["dti_ratio"], f["loan_to_income"]
    score, delq, hist = a.get("credit_score"), a["num_past_delinquencies"], a["credit_history_years"]
    income, amt, emp = a.get("monthly_income"), a["loan_amount"], a["employment_type"]
    thin = bool(f["thin_file"])
    reject, review, not_covered = [], [], []

    def R(clause, msg): reject.append({"clause": clause, "reason": msg})
    def V(clause, msg): review.append({"clause": clause, "reason": msg})

    # ----- hard "not eligible" rules -----
    if not 21 <= a["age"] <= 65: R("2.1", "age is outside 21-65")
    if income is not None and income < 15000: R("2.2", "monthly income is below Rs 15,000")
    if not 50000 <= amt <= 2500000: R("1.2", "loan amount is outside Rs 50,000 - Rs 25,00,000")
    if not 12 <= a["tenure_months"] <= 60: R("1.3", "tenure is outside 12-60 months")
    if score is not None and score < 600: R("3.5", "credit score is below 600")
    if delq >= 3: R("3.6", "3 or more past delinquencies")
    if dti > 0.50: R("4.4", f"debt-to-income ratio {dti:.2f} is above 0.50")
    elif dti > 0.40 and (score is None or score < 700):
        R("4.3", f"debt-to-income ratio {dti:.2f} is above 0.40 and credit score is below 700")
    if lti > 2.0: R("4.6", f"loan-to-income ratio {lti:.2f} is above 2.0")
    if score is not None and 600 <= score < 650 and (amt > 500000 or dti > 0.40):
        R("3.4", "weak credit score (600-649) is eligible only up to Rs 5,00,000 and DTI 0.40")
    if thin and (amt > 200000 or dti > 0.35):
        R("3.10", "thin-file applicants are eligible only up to Rs 2,00,000 and DTI 0.35")

    # ----- manual-review triggers -----
    if emp == "Salaried" and f["years_in_current_job"] < 1: V("2.3", "salaried with under 1 year in current job")
    if emp != "Salaried" and f["years_in_current_job"] < 2: V("2.4", "self-employed/business owner with under 2 years in business")
    if score is not None and 600 <= score < 650: V("3.4", "weak credit score (600-649)")
    if delq == 2: V("3.7", "exactly 2 past delinquencies")
    if delq == 1 and score is not None and score < 700: V("3.8", "1 past delinquency and credit score below 700")
    if thin: V("3.9", "thin-file applicant cannot be approved automatically")
    if score is None and hist >= 1: V("3.11", "long credit history but no score on file (possible data error)")
    if 0.40 < dti <= 0.50 and score is not None and score >= 700: V("4.3", f"debt-to-income ratio {dti:.2f} is between 0.40 and 0.50")
    if 1.5 < lti <= 2.0: V("4.7", f"loan-to-income ratio {lti:.2f} is between 1.5 and 2.0")
    if income_outlier: V("4.8", "reported income above Rs 5,00,000 must be verified")
    if APPROVE_BELOW <= pd_ < REJECT_AT: V("5.3", f"model PD {pd_:.2f} is in the borderline band")

    # ----- known gaps in the policy (honest design: only KNOWN gaps are listed) -----
    if a["age"] + a["tenure_months"] / 12 > 65:
        not_covered.append({"clause": "10.1", "reason": "the policy has no rule on applicant age at loan maturity"})
    if emp not in ("Salaried", "Self-employed", "Business owner"):
        not_covered.append({"clause": "10.1", "reason": "employment type is not covered by the policy"})

    # ----- combine (policy 6.1-6.6) -----
    if reject:
        rec = "REJECT"
    elif pd_ >= REJECT_AT:
        rec = "REJECT"; R("5.4", f"model PD {pd_:.2f} is at or above {REJECT_AT}")
    elif review or not_covered:
        rec = "REVIEW"
    else:
        rec = "APPROVE"
    if not_covered and rec == "APPROVE":
        rec = "REVIEW"
    return {"pd": round(pd_, 3), "dti": dti, "loan_to_income": lti, "recommendation": rec,
            "human_review_required": rec != "APPROVE" or bool(not_covered),
            "reject_reasons": reject, "review_reasons": review, "not_covered": not_covered,
            "features": f.to_dict()}


DEMO = {
    "clear_approval": dict(age=38, employment_type="Salaried", years_in_current_job=8, monthly_income=95000,
                           existing_monthly_emi=8000, credit_score=790, credit_history_years=12,
                           num_past_delinquencies=0, loan_amount=400000, tenure_months=36),
    "clear_rejection": dict(age=29, employment_type="Self-employed", years_in_current_job=3, monthly_income=26000,
                            existing_monthly_emi=9000, credit_score=585, credit_history_years=4,
                            num_past_delinquencies=2, loan_amount=700000, tenure_months=48),
    "thin_file_high_income": dict(age=26, employment_type="Salaried", years_in_current_job=2, monthly_income=150000,
                                  existing_monthly_emi=0, credit_score=None, credit_history_years=0,
                                  num_past_delinquencies=0, loan_amount=180000, tenure_months=24),
    "policy_gap_age_tenure": dict(age=63, employment_type="Salaried", years_in_current_job=15, monthly_income=80000,
                                  existing_monthly_emi=5000, credit_score=760, credit_history_years=25,
                                  num_past_delinquencies=0, loan_amount=300000, tenure_months=60),
    "income_typo": dict(age=34, employment_type="Salaried", years_in_current_job=5, monthly_income=850000,
                        existing_monthly_emi=0, credit_score=720, credit_history_years=8,
                        num_past_delinquencies=0, loan_amount=500000, tenure_months=36),
}

if __name__ == "__main__":
    for name, a in DEMO.items():
        r = evaluate(a)
        print(f"\n== {name}: {r['recommendation']}  PD={r['pd']}  DTI={r['dti']}  human={r['human_review_required']}")
        for k in ("reject_reasons", "review_reasons", "not_covered"):
            for x in r[k]:
                print(f"   [{k.split('_')[0]}] clause {x['clause']}: {x['reason']}")
