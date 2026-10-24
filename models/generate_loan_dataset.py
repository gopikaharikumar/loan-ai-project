"""
Synthetic personal-loan dataset generator (EDUCATIONAL PROTOTYPE).

Design principle: the 'defaulted' label is drawn from a hidden default
probability (a noisy function of risk drivers), NOT from the bank's policy rules.
So a model can learn real signal but cannot reach ~100% accuracy.
Gender is NOT used in the default process. It only correlates with income
(an assumed pay gap), so gender acts as a proxy-bias test case.

Run:  python src/generate_loan_dataset.py
Output: data/loan_applicants.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N = 8000
rng = np.random.default_rng(SEED)

# ---------- Demographics ----------
gender = rng.choice(["Male", "Female"], N, p=[0.72, 0.28])
city_tier = rng.choice([1, 2, 3], N, p=[0.35, 0.40, 0.25])
age = np.clip(rng.normal(35, 9, N), 21, 65).round().astype(int)
employment = rng.choice(["Salaried", "Self-employed", "Business owner"], N, p=[0.60, 0.25, 0.15])

# ---------- Income (assumption: pay gap + tier effect, so gender is a proxy) ----------
median_by_tier = {1: 55000, 2: 38000, 3: 28000}
emp_mult = {"Salaried": 1.0, "Self-employed": 1.05, "Business owner": 1.25}
base = np.array([median_by_tier[t] for t in city_tier], dtype=float)
base *= np.array([emp_mult[e] for e in employment])
base *= np.where(gender == "Female", 0.82, 1.0)
income = np.clip(base * rng.lognormal(0, 0.5, N), 12000, 600000).round(-2)

# ---------- Work and credit history ----------
max_exp = age - 21
years_job = np.minimum(rng.exponential(4.5, N), max_exp).round(1)
hist_len = np.minimum(rng.gamma(2.5, 2.6, N), max_exp)
thin_file = rng.random(N) < 0.08           # first-time borrowers
hist_len = hist_len * np.where(gender == "Female", 0.75, 1.0)   # assumed career-break effect (proxy bias)
hist_len = np.where(thin_file, 0, hist_len).round(1)

delinq = rng.poisson(np.where(employment == "Salaried", 0.35, 0.55))
delinq = np.where(thin_file, 0, delinq)

# CIBIL-style score (300-900)
score = (660 + 4.0 * hist_len - 45 * delinq + 12 * np.log(income / 40000)
         + rng.normal(0, 55, N))
score = np.clip(score, 300, 900).round().astype(float)
# thin-file applicants often have NO score at all (a real-world edge case)
no_score = thin_file & (rng.random(N) < 0.6)
score[no_score] = np.nan

# ---------- Existing obligations and the requested loan ----------
has_emi = rng.random(N) < 0.65
existing_emi = np.where(has_emi, income * rng.beta(2, 6, N) * 0.8, 0).round(-1)

tenure = rng.choice([12, 24, 36, 48, 60], N, p=[0.10, 0.20, 0.35, 0.20, 0.15])
loan_amt = income * rng.lognormal(np.log(6), 0.55, N)
loan_amt = np.clip(loan_amt, 50000, 2500000).round(-3)

r = 0.14 / 12                               # assumed 14% p.a. reducing-balance
new_emi = loan_amt * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1)
dti = (existing_emi + new_emi) / income     # total EMI burden / income
lti = loan_amt / (income * 12)              # loan-to-annual-income

# ---------- Hidden default process (gender NOT included) ----------
score_filled = np.where(np.isnan(score), 640, score)
z = (-3.5
     + 5.0 * (dti - 0.35)
     - 0.011 * (score_filled - 700)
     + 0.45 * delinq
     + 0.35 * no_score
     - 0.04 * years_job
     + 0.30 * (employment == "Self-employed")
     + 0.15 * (employment == "Business owner")
     + 0.15 * (city_tier == 3)
     + rng.normal(0, 0.7, N))               # unobserved factors
p_default = 1 / (1 + np.exp(-z))
defaulted = (rng.random(N) < p_default).astype(int)

df = pd.DataFrame({
    "applicant_id": [f"APP{i:05d}" for i in range(1, N + 1)],
    "age": age,
    "gender": gender,
    "city_tier": city_tier,
    "employment_type": employment,
    "years_in_current_job": years_job,
    "monthly_income": income,
    "existing_monthly_emi": existing_emi,
    "credit_score": score,
    "credit_history_years": hist_len,
    "num_past_delinquencies": delinq,
    "loan_amount": loan_amt,
    "tenure_months": tenure,
    "proposed_emi": new_emi.round(0),
    "dti_ratio": dti.round(3),
    "loan_to_income": lti.round(3),
    "defaulted": defaulted,
})

# ---------- Real-world messiness ----------
for col, frac in [("years_in_current_job", 0.02), ("monthly_income", 0.01)]:
    df.loc[rng.random(N) < frac, col] = np.nan           # missing values
typo = rng.random(N) < 0.004
df.loc[typo, "monthly_income"] = df.loc[typo, "monthly_income"] * 10   # data-entry errors

import os; os.makedirs("data", exist_ok=True)
df.to_csv("data/loan_applicants.csv", index=False)
print(f"Saved {len(df)} rows -> data/loan_applicants.csv")
print(f"Default rate: {df.defaulted.mean():.1%}")
