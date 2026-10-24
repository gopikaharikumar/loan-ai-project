"""
Phase 2: predictive model (EDUCATIONAL PROTOTYPE).

Baseline (approve everyone) -> logistic regression -> gradient boosting.
Threshold is chosen on VALIDATION from business error costs; TEST is used once.

Run from the repo root:
    python src/train_model.py
or:
    python train_model.py --datadir processed --outdir models
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Business assumptions (CHANGE THESE and defend them) ----------------
LGD = 0.50      # share of the loan lost when a borrower defaults
MARGIN = 0.08   # net lifetime profit earned on a good loan, as a share of the loan
# Cost of approving someone who defaults  = LGD    * loan_amount
# Cost of rejecting someone who would repay = MARGIN * loan_amount (profit forgone)
# -> with calibrated probabilities the cost-optimal cut is MARGIN / (MARGIN + LGD)
# ------------------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--datadir", default="data/processed")
p.add_argument("--outdir", default="models")
args = p.parse_args()
out = Path(args.outdir)
(out / "figures").mkdir(parents=True, exist_ok=True)

params = json.load(open(Path(args.datadir) / "prep_params.json"))
FEATS = params["feature_cols"]
tr, va, te = [pd.read_csv(Path(args.datadir) / f"{n}.csv") for n in ("train", "val", "test")]
Xtr, ytr = tr[FEATS], tr["defaulted"].values
Xva, yva = va[FEATS], va["defaulted"].values
Xte, yte = te[FEATS], te["defaulted"].values


def cost(y, reject, loan):
    """Return (total cost in Rs, missed defaults, good customers rejected)."""
    missed = (~reject) & (y == 1)
    lost = reject & (y == 0)
    return LGD * loan[missed].sum() + MARGIN * loan[lost].sum(), int(missed.sum()), int(lost.sum())


def report(y, reject, loan, label):
    """Metrics where 'positive' = predicted to default (i.e. rejected)."""
    tn, fp, fn, tp = confusion_matrix(y, reject.astype(int), labels=[0, 1]).ravel()
    c, _, _ = cost(y, reject, loan)
    return {"model": label,
            "approved_and_repaid_TN": int(tn), "rejected_but_would_repay_FP": int(fp),
            "approved_but_defaulted_FN": int(fn), "rejected_and_defaulted_TP": int(tp),
            "precision": round(precision_score(y, reject, zero_division=0), 3),
            "recall": round(recall_score(y, reject, zero_division=0), 3),
            "total_cost_rs": round(float(c))}


# ---------------- Models ----------------
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                        max_iter=250, random_state=42),
}
for m in models.values():
    m.fit(Xtr, ytr)

loan_va, loan_te = va["loan_amount"].values, te["loan_amount"].values
ts = np.arange(0.02, 0.60, 0.005)

# ---------------- Validation: AUC, calibration, cost-optimal threshold ----------------
val_rows, best = [], {}
plt.figure(figsize=(11, 4))
for name, m in models.items():
    pv = m.predict_proba(Xva)[:, 1]
    costs = np.array([cost(yva, pv >= t, loan_va)[0] for t in ts])
    t_star = float(ts[costs.argmin()])
    best[name] = {"t_star": t_star, "val_cost": float(costs.min())}
    val_rows.append({"model": name, "ROC-AUC": round(roc_auc_score(yva, pv), 3),
                     "PR-AUC": round(average_precision_score(yva, pv), 3),
                     "Brier": round(brier_score_loss(yva, pv), 4), "cost-optimal threshold": t_star})
    plt.subplot(1, 2, 1)
    pr, rc, _ = precision_recall_curve(yva, pv)
    plt.plot(rc, pr, label=f"{name} (PR-AUC {val_rows[-1]['PR-AUC']})")
    plt.subplot(1, 2, 2)
    plt.plot(ts, costs / 1e6, label=name)
    plt.axvline(t_star, ls=":", color="grey")
plt.subplot(1, 2, 1)
plt.axhline(yva.mean(), ls="--", color="k", label="Baseline (default rate)")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall (validation)"); plt.legend()
plt.subplot(1, 2, 2)
plt.xlabel("Rejection threshold (PD)"); plt.ylabel("Total cost (Rs million)")
plt.title("Cost vs threshold (validation)"); plt.legend()
plt.tight_layout(); plt.savefig(out / "figures" / "model_comparison.png", dpi=150); plt.close()

val_table = pd.DataFrame(val_rows)
print("\nVALIDATION\n", val_table.to_string(index=False))

# choose the model with the lowest validation cost at its own best threshold
champ = min(best, key=lambda k: best[k]["val_cost"])
T = best[champ]["t_star"]
print(f"\nChampion (lowest validation cost): {champ}, threshold {T:.3f}")

# ---------------- TEST: touched once ----------------
rows = []
base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
rows.append(report(yte, base.predict(Xte).astype(bool), loan_te, "Baseline: approve everyone"))
rows.append(report(yte, np.ones(len(yte), bool), loan_te, "Reject everyone"))
for name, m in models.items():
    pt = m.predict_proba(Xte)[:, 1]
    rows.append(report(yte, pt >= 0.5, loan_te, f"{name} @ 0.5 (naive)"))
    rows.append(report(yte, pt >= best[name]["t_star"], loan_te,
                       f"{name} @ {best[name]['t_star']:.3f} (cost-based)"))
test_table = pd.DataFrame(rows)
print("\nTEST (positive = rejected as likely defaulter)\n", test_table.to_string(index=False))
base_cost = test_table.iloc[0]["total_cost_rs"]
champ_cost = test_table[test_table.model.str.startswith(champ) & test_table.model.str.contains("cost-based")].iloc[0]["total_cost_rs"]
print(f"\nCost saving of champion vs approve-everyone on test: Rs {base_cost - champ_cost:,.0f} "
      f"({(base_cost - champ_cost) / base_cost:.0%})")

# ---------------- Candidate review bands (for policy clauses 5.2-5.4) ----------------
pt = models[champ].predict_proba(Xte)[:, 1]
band_rows = []
for a, r in [(0.10, 0.25), (round(T * 0.7, 2), round(T * 1.6, 2)), (round(T * 0.8, 2), round(T * 1.3, 2))]:
    lo, mid, hi = pt < a, (pt >= a) & (pt < r), pt >= r
    band_rows.append({"approve_below": a, "reject_at_or_above": r,
                      "auto_approve_%": round(lo.mean() * 100, 1), "review_%": round(mid.mean() * 100, 1),
                      "auto_reject_%": round(hi.mean() * 100, 1),
                      "default_rate_in_approve_%": round(yte[lo].mean() * 100, 1),
                      "default_rate_in_review_%": round(yte[mid].mean() * 100, 1),
                      "default_rate_in_reject_%": round(yte[hi].mean() * 100, 1)})
band_table = pd.DataFrame(band_rows)
print("\nREVIEW-BAND OPTIONS (test)\n", band_table.to_string(index=False))

# ---------------- Fairness check (gender and city tier are NOT model inputs) ----------------
approved = pt < T
fair = {}
for col in ("gender", "city_tier"):
    rates = pd.Series(approved).groupby(te[col].values).mean().round(3)
    fair[col] = {str(k): float(v) for k, v in rates.items()}
    print(f"\nApproval rate by {col}: {fair[col]}   (min/max ratio {rates.min() / rates.max():.2f}; 4/5 rule = 0.80)")

# ---------------- Save ----------------
joblib.dump({"model": models[champ], "features": FEATS, "threshold": T, "name": champ},
            out / "risk_model.joblib")
json.dump({"assumptions": {"LGD": LGD, "MARGIN": MARGIN}, "validation": val_rows,
           "champion": champ, "threshold": T, "test": rows, "bands": band_rows, "fairness": fair},
          open(out / "metrics.json", "w"), indent=2)
print(f"\nSaved model, metrics and figure to {out}/")
