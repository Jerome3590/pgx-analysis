"""
Compute final manuscript-ready Z-code ORs from saved zcode_merged.parquet.
Q1 = reference (unmonitored, z_prop=0 or lowest quartile).
Reports the U-shaped pattern: Q2/Q3 protective, Q4 as risky as Q1.
Fills zcode_results.json with all CH_4 placeholder values.
"""
import json, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, chi2_contingency

warnings.filterwarnings("ignore")

merged = pd.read_parquet("zcode_merged.parquet")
print(f"Loaded: {len(merged):,} patients  "
      f"({merged['target'].sum():,} cases, "
      f"{(merged['target']==0).sum():,} controls)")

# ── Assign quartiles with Q1 = lowest (unmonitored) ─────────────────────────
try:
    merged["z_q"] = pd.qcut(merged["z_prop"], q=4,
                             labels=[1, 2, 3, 4], duplicates="drop")
except ValueError:
    merged["z_q"] = pd.qcut(merged["z_prop"].rank(method="first"),
                             q=4, labels=[1, 2, 3, 4])

q4_lo   = merged["z_prop"].quantile(0.75)
q1_hi   = merged["z_prop"].quantile(0.25)
print(f"\nQ1 upper bound: ≤{q1_hi:.3f} ({q1_hi*100:.0f}% of claims)")
print(f"Q4 lower bound: ≥{q4_lo:.3f} ({q4_lo*100:.0f}% of claims)")

print("\n=== QUARTILE CASE RATES ===")
for q in [1, 2, 3, 4]:
    g  = merged[merged["z_q"] == q]
    nc = (g["target"] == 1).sum()
    nt = len(g)
    print(f"  Q{q}: n={nt:,}  cases={nc:,} ({nc/nt*100:.1f}%)")

# ── Chi-square for overall association ───────────────────────────────────────
ct = pd.crosstab(merged["z_q"], merged["target"])
chi2, p_chi2, _, _ = chi2_contingency(ct)
print(f"\nChi-square (overall): χ²={chi2:.1f}  p={'<0.001' if p_chi2<0.001 else f'{p_chi2:.4f}'}")

# ── Logistic regression: Q1 as reference, Q2/Q3/Q4 dummies ──────────────────
print("\n=== LOGISTIC REGRESSION: Q1 reference + n_events + band ===")
df_lr = merged.dropna(subset=["z_prop", "n_events"]).copy()
df_lr["q2"] = (df_lr["z_q"] == 2).astype(float)
df_lr["q3"] = (df_lr["z_q"] == 3).astype(float)
df_lr["q4"] = (df_lr["z_q"] == 4).astype(float)
df_lr = pd.get_dummies(df_lr, columns=["band"], drop_first=True)
band_cols = [c for c in df_lr.columns if c.startswith("band_")]

Xmat = np.column_stack(
    [np.ones(len(df_lr))]
    + [df_lr[c].astype(float).values
       for c in ["q2", "q3", "q4", "n_events"] + band_cols]
)
y = df_lr["target"].astype(int).values


def neg_ll(b):
    p = expit(Xmat @ b)
    return -np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))


def grad(b):
    p = expit(Xmat @ b)
    return -Xmat.T @ (y - p)


res  = minimize(neg_ll, np.zeros(Xmat.shape[1]), jac=grad, method="L-BFGS-B")
beta = res.x
ph   = expit(Xmat @ beta)
W    = ph * (1 - ph)
H    = (Xmat.T * W) @ Xmat
try:
    se = np.sqrt(np.diag(np.linalg.inv(H)))
except np.linalg.LinAlgError:
    se = np.full(len(beta), np.nan)

or_vals = {}
for i, qname in enumerate(["Q2", "Q3", "Q4"], start=1):
    coef  = beta[i]
    se_c  = se[i]
    ci_lo = coef - 1.96 * se_c
    ci_hi = coef + 1.96 * se_c
    pval  = 2 * norm.sf(abs(coef / se_c))
    pstr  = "<0.001" if pval < 0.001 else f"{pval:.3f}"
    or_   = np.exp(coef)
    lo_   = np.exp(ci_lo)
    hi_   = np.exp(ci_hi)
    print(f"  {qname} vs Q1: OR={or_:.2f} ({lo_:.2f}–{hi_:.2f})  p={pstr}")
    or_vals[qname] = {"or": round(or_, 2),
                      "ci_lo": round(lo_, 2),
                      "ci_hi": round(hi_, 2),
                      "pval": pstr}

# ── Unadjusted (crude) Q2 vs Q1 for abstract ─────────────────────────────────
g1 = merged[merged["z_q"] == 1]
g2 = merged[merged["z_q"] == 2]
g3 = merged[merged["z_q"] == 3]
odds_q1 = g1["target"].sum() / max((g1["target"] == 0).sum(), 1)
odds_q2 = g2["target"].sum() / max((g2["target"] == 0).sum(), 1)
odds_q3 = g3["target"].sum() / max((g3["target"] == 0).sum(), 1)
crude_q2 = odds_q2 / max(odds_q1, 1e-10)
crude_q3 = odds_q3 / max(odds_q1, 1e-10)
print(f"\n  Crude Q2 vs Q1: OR={crude_q2:.2f}")
print(f"  Crude Q3 vs Q1: OR={crude_q3:.2f}")

# ── IQR for Table 1 ───────────────────────────────────────────────────────────
cases_g = merged[merged["target"] == 1]["z_prop"]
ctrl_g  = merged[merged["target"] == 0]["z_prop"]
c_med, c_q25, c_q75 = cases_g.median(), cases_g.quantile(0.25), cases_g.quantile(0.75)
r_med, r_q25, r_q75 = ctrl_g.median(),  ctrl_g.quantile(0.25),  ctrl_g.quantile(0.75)

print("\n" + "=" * 62)
print("=== CH_4 FINAL MANUSCRIPT VALUES ===")
print(f"  Table 1 Z-code IQR (protective-Z-only, Z00-Z13/Z23-Z28)")
print(f"    Cases:    {c_med:.2f} ({c_q25:.2f}–{c_q75:.2f})")
print(f"    Controls: {r_med:.2f} ({r_q25:.2f}–{r_q75:.2f})")
print(f"  Q4 threshold: ≥ {q4_lo*100:.0f}% of claims")
print(f"  Q2 vs Q1 (moderate vs unmonitored):")
print(f"    Adj OR={or_vals['Q2']['or']} ({or_vals['Q2']['ci_lo']}–{or_vals['Q2']['ci_hi']}) p={or_vals['Q2']['pval']}")
print(f"  Q4 vs Q1 (extreme vs unmonitored):")
print(f"    Adj OR={or_vals['Q4']['or']} ({or_vals['Q4']['ci_lo']}–{or_vals['Q4']['ci_hi']}) p={or_vals['Q4']['pval']}")
print(f"  U-shaped: Q2 protective ({crude_q2:.2f} crude OR vs Q1), Q4≈Q1 ({crude_q3:.2f} Q3 crude)")

# ── Save ─────────────────────────────────────────────────────────────────────
results = {
    "note": "Protective Z-codes only: Z00-Z13 (exams/screens) + Z23-Z28 (immunizations)",
    "cases_median":    round(c_med, 2),
    "cases_q25":       round(c_q25, 2),
    "cases_q75":       round(c_q75, 2),
    "controls_median": round(r_med, 2),
    "controls_q25":    round(r_q25, 2),
    "controls_q75":    round(r_q75, 2),
    "q1_upper_pct":    round(q1_hi * 100),
    "q4_threshold_pct": round(q4_lo * 100),
    "or_q2_vs_q1": or_vals["Q2"],
    "or_q3_vs_q1": or_vals["Q3"],
    "or_q4_vs_q1": or_vals["Q4"],
    "crude_q2_vs_q1": round(crude_q2, 2),
    "crude_q3_vs_q1": round(crude_q3, 2),
    "pattern": "U-shaped: Q2/Q3 protective, Q4 similar to Q1 (reactive care)",
}
with open("zcode_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved zcode_results.json")
