"""
Phase 1: data preparation for the loan model (EDUCATIONAL PROTOTYPE).

Steps: inspect -> flag outliers/missing -> split (60/20/20, stratified) ->
fit imputation on TRAIN ONLY -> save processed train/val/test + parameters.

Run from the repo root:
    python src/prepare_data.py
or with paths:
    python prepare_data.py --input loan_applicants.csv --outdir processed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INCOME_VERIFY_LIMIT = 500_000      # policy clause 4.8
TARGET = "defaulted"

# Columns kept in the files for the fairness check but NEVER used as model features.
# city_tier is excluded by default because of policy clause 8.2; switch it on
# later to measure how much it matters (and to report it as a proxy risk).
FAIRNESS_ONLY = ["gender", "city_tier"]

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/loan_applicants.csv")
p.add_argument("--outdir", default="data/processed")
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.input)

# ---------- 1. Inspect ----------
print(f"Rows: {len(df)}   Default rate: {df[TARGET].mean():.1%}")
print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())
n_out = (df["monthly_income"] > INCOME_VERIFY_LIMIT).sum()
print(f"\nIncome above Rs {INCOME_VERIFY_LIMIT:,} (needs verification, clause 4.8): {n_out}")
print("\nDefault rate by employment type:")
print(df.groupby("employment_type")[TARGET].mean().round(3).to_string())
bands = pd.cut(df["credit_score"], [0, 600, 650, 700, 750, 901],
               labels=["<600", "600-649", "650-699", "700-749", "750+"], right=False)
print("\nDefault rate by credit score band (sanity check):")
print(df.groupby(bands, observed=True)[TARGET].mean().round(3).to_string())

# ---------- 2. Flags (created BEFORE imputing, so missingness stays visible) ----------
df["credit_score_missing"] = df["credit_score"].isna().astype(int)
df["income_missing"] = df["monthly_income"].isna().astype(int)
df["income_outlier"] = (df["monthly_income"] > INCOME_VERIFY_LIMIT).astype(int)
df["thin_file"] = ((df["credit_history_years"] < 1) | df["credit_score"].isna()).astype(int)
# outliers are treated as unverified: blank them so they get imputed, not trusted
df.loc[df["income_outlier"] == 1, "monthly_income"] = np.nan

# ---------- 3. Split BEFORE fitting any imputation ----------
train, temp = train_test_split(df, test_size=0.40, stratify=df[TARGET], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=SEED)

# ---------- 4. Impute using TRAIN statistics only ----------
impute_cols = ["credit_score", "monthly_income", "years_in_current_job"]
medians = {c: float(train[c].median()) for c in impute_cols}
for part in (train, val, test):
    for c in impute_cols:
        part[c] = part[c].fillna(medians[c])

# ---------- 5. One-hot encode employment type ----------
for part in (train, val, test):
    for e in ["Self-employed", "Business owner"]:       # Salaried = reference
        part[f"emp_{e.lower().replace('-', '_').replace(' ', '_')}"] = (part["employment_type"] == e).astype(int)

feature_cols = [
    "age", "years_in_current_job", "monthly_income", "existing_monthly_emi",
    "credit_score", "credit_history_years", "num_past_delinquencies",
    "loan_amount", "tenure_months", "proposed_emi", "dti_ratio", "loan_to_income",
    "credit_score_missing", "thin_file", "emp_self_employed", "emp_business_owner",
]

keep = ["applicant_id"] + feature_cols + FAIRNESS_ONLY + ["income_outlier", TARGET]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[keep].to_csv(out / f"{name}.csv", index=False)
    print(f"{name}: {len(part)} rows, default rate {part[TARGET].mean():.1%}")

json.dump({"seed": SEED, "train_medians": medians, "feature_cols": feature_cols,
           "fairness_only_cols": FAIRNESS_ONLY,
           "income_verify_limit": INCOME_VERIFY_LIMIT},
          open(out / "prep_params.json", "w"), indent=2)
print(f"\nSaved to {out}/  (train.csv, val.csv, test.csv, prep_params.json)")
