"""
Z-code proportion analysis for CH_4 non_opioid_ed cohort.

Uses full medical claims history (all ICD-coded events across enrollment period).
Z-code proportion = Z-code ICD events / total ICD events (per patient).

Produces:
  - Table 1 IQR (cases vs controls)
  - Logistic regression OR + 95% CI (Q4 vs Q1 reference)
  - Bimodal extreme-density OR subgroups
"""
import boto3, io, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

s3     = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["65-74", "75-84", "85-114"]

ICD_PRIMARY = "primary_icd_diagnosis_code"

# =============================================================================
# 1. Load model_events for all geriatric bands (medical events only)
# =============================================================================
print("Loading model_events (medical events only) …")
frames = []
for band in BANDS:
    key  = (f"gold/cohorts_model_data/cohort_name=non_opioid_ed/"
            f"age_band={band}/model_events.parquet")
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    df   = pd.read_parquet(
        io.BytesIO(data),
        columns=["mi_person_key", "target", ICD_PRIMARY]
    )
    # Medical events only (ICD present, not drug-dispensing rows)
    df   = df[df[ICD_PRIMARY].notna()].copy()
    df["band"] = band
    frames.append(df)
    print(f"  {band}: {len(df):,} medical events")

med = pd.concat(frames, ignore_index=True)
print(f"Total medical events: {len(med):,}")

# =============================================================================
# 2. Per-patient Z-code proportion (full enrollment history)
# =============================================================================
med["is_z"] = med[ICD_PRIMARY].str.startswith("Z", na=False).astype(np.int8)

pat = (med.groupby(["mi_person_key", "target", "band"])
          .agg(total_icd=("is_z", "count"),
               z_icd    =("is_z", "sum"))
          .reset_index())
pat["z_prop"] = pat["z_icd"] / pat["total_icd"].clip(lower=1)

print(f"\nPatients with medical events: {len(pat):,}")

# =============================================================================
# 3. Add patients with zero medical events (z_prop = 0)
# =============================================================================
all_pts_frames = []
for band in BANDS:
    for split in ("model_train", "model_test"):
        key = (f"gold/final_model/non_opioid_ed/{band}/inputs/"
               f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            ff   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target",
                                            "n_events", "n_event_bin"])
            ff["band"] = band
            all_pts_frames.append(ff)
        except Exception as e:
            print(f"  SKIP {split}/{band}: {e}")

all_pts = pd.concat(all_pts_frames).drop_duplicates("mi_person_key")
print(f"Total analytic cohort: {len(all_pts):,} patients")

merged = all_pts.merge(
    pat[["mi_person_key", "z_prop", "total_icd", "z_icd"]],
    on="mi_person_key", how="left"
)
merged["z_prop"]    = merged["z_prop"].fillna(0.0)
merged["total_icd"] = merged["total_icd"].fillna(0).astype(int)
merged["z_icd"]     = merged["z_icd"].fillna(0).astype(int)

# =============================================================================
# 4. Table 1 IQR
# =============================================================================
print("\n=== TABLE 1 Z-CODE PROPORTION (full enrollment) ===")
for label, tgt in [("Cases", 1), ("Controls", 0)]:
    g = merged[merged["target"] == tgt]["z_prop"]
    q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
    print(f"  {label:10s}: median={q50:.2f} ({q25:.2f}–{q75:.2f})  n={len(g):,}")

# =============================================================================
# 5. Logistic regression: target ~ z_prop_quartile + n_events + band
# =============================================================================
print("\n=== LOGISTIC REGRESSION ===")
df_lr = merged.dropna(subset=["z_prop", "n_events"]).copy()

try:
    df_lr["z_q"] = pd.qcut(df_lr["z_prop"], q=4, labels=[1, 2, 3, 4],
                            duplicates="drop").astype(float)
except ValueError:
    df_lr["z_q"] = pd.qcut(df_lr["z_prop"].rank(method="first"),
                            q=4, labels=[1, 2, 3, 4]).astype(float)

df_lr = pd.get_dummies(df_lr, columns=["band"], drop_first=True)
band_cols = [c for c in df_lr.columns if c.startswith("band_")]

Xmat = np.column_stack(
    [np.ones(len(df_lr))]
    + [df_lr[c].astype(float).values for c in ["z_q", "n_events"] + band_cols]
)
y = df_lr["target"].astype(int).values


def neg_ll(b):
    p = expit(Xmat @ b)
    return -np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))


def neg_ll_grad(b):
    p = expit(Xmat @ b)
    return -Xmat.T @ (y - p)


res  = minimize(neg_ll, np.zeros(Xmat.shape[1]), jac=neg_ll_grad, method="L-BFGS-B")
beta = res.x
p_hat = expit(Xmat @ beta)
W     = p_hat * (1 - p_hat)
H     = (Xmat.T * W) @ Xmat
try:
    se = np.sqrt(np.diag(np.linalg.inv(H)))
except np.linalg.LinAlgError:
    se = np.full(len(beta), np.nan)

coef  = beta[1]
se_c  = se[1]
ci_lo = coef - 1.96 * se_c
ci_hi = coef + 1.96 * se_c
pval  = 2 * norm.sf(abs(coef / se_c))
q4_lo = df_lr["z_prop"].quantile(0.75)

print(f"  z_q coef={coef:.3f}  OR={np.exp(coef):.2f}  "
      f"95%CI [{np.exp(ci_lo):.2f}–{np.exp(ci_hi):.2f}]  p={pval:.4f}")
print(f"  Q4 lower bound (75th pct): {q4_lo:.2f} ({q4_lo*100:.0f}% of claims)")

# =============================================================================
# 6. Bimodal extreme-density subgroup analysis
# =============================================================================
print("\n=== EXTREME-DENSITY BIMODAL SUBGROUPS ===")
ext = merged[merged["n_event_bin"] == "extreme"].copy()
print(f"  Extreme-density patients: {len(ext):,}")
if len(ext) > 10:
    ext_median_z = ext["z_prop"].median()
    ext["z_high"] = (ext["z_prop"] >= ext_median_z).astype(int)
    print(f"  Z-prop split at median = {ext_median_z:.3f}")

    for subgroup, label in [(0, "low-Z (high-risk)"), (1, "high-Z (low-risk)")]:
        g = ext[ext["z_high"] == subgroup]
        n_case = (g["target"] == 1).sum()
        n_ctrl = (g["target"] == 0).sum()
        # Simple OR vs complement
        or_sg = (n_case / max(n_ctrl, 1)) / \
                ((ext["target"] == 1).sum() / max((ext["target"] == 0).sum(), 1))
        print(f"  {label}: n_case={n_case}, n_ctrl={n_ctrl}, OR≈{or_sg:.2f}")

# Also get proper OR for extreme-density subgroups via logistic regression
if len(ext) > 10:
    ext_lr = ext.dropna(subset=["z_prop", "n_events"]).copy()
    ext_lr["z_high"] = (ext_lr["z_prop"] >= ext_median_z).astype(float)
    Xe = np.column_stack([np.ones(len(ext_lr)), ext_lr["z_high"].values,
                          ext_lr["n_events"].astype(float).values])
    ye = ext_lr["target"].astype(int).values
    res_e = minimize(neg_ll.__class__(lambda b: -np.sum(ye*np.log(expit(Xe@b)+1e-15) +
                     (1-ye)*np.log(1-expit(Xe@b)+1e-15))),
                     np.zeros(3), method="L-BFGS-B")

    def neg_ll_ext(b):
        p = expit(Xe @ b)
        return -np.sum(ye * np.log(p + 1e-15) + (1 - ye) * np.log(1 - p + 1e-15))

    def grad_ext(b):
        p = expit(Xe @ b)
        return -Xe.T @ (ye - p)

    res_e = minimize(neg_ll_ext, np.zeros(3), jac=grad_ext, method="L-BFGS-B")
    be = res_e.x
    pe = expit(Xe @ be)
    We = pe * (1 - pe)
    He = (Xe.T * We) @ Xe
    try:
        se_e = np.sqrt(np.diag(np.linalg.inv(He)))
    except np.linalg.LinAlgError:
        se_e = np.full(3, np.nan)
    or_high = np.exp(be[1])
    ci_lo_e = np.exp(be[1] - 1.96 * se_e[1])
    ci_hi_e = np.exp(be[1] + 1.96 * se_e[1])
    print(f"  High-Z vs Low-Z OR (logit) = {or_high:.2f} ({ci_lo_e:.2f}–{ci_hi_e:.2f})")

# =============================================================================
# 7. Final manuscript values summary
# =============================================================================
print("\n" + "=" * 60)
print("=== CH_4 MANUSCRIPT VALUES ===")
cases_g   = merged[merged["target"] == 1]["z_prop"]
ctrl_g    = merged[merged["target"] == 0]["z_prop"]
c_med, c_q25, c_q75 = cases_g.median(), cases_g.quantile(0.25), cases_g.quantile(0.75)
r_med, r_q25, r_q75 = ctrl_g.median(),  ctrl_g.quantile(0.25),  ctrl_g.quantile(0.75)
print(f"  Table 1 Z-code IQR — Cases:    {c_med:.2f} ({c_q25:.2f}–{c_q75:.2f})")
print(f"  Table 1 Z-code IQR — Controls: {r_med:.2f} ({r_q25:.2f}–{r_q75:.2f})")
print(f"  Q4 threshold (75th pct):  ≥ {q4_lo*100:.0f}% of claims")
or_val   = np.exp(coef)
ci_lo_v  = np.exp(ci_lo)
ci_hi_v  = np.exp(ci_hi)
p_str    = "<0.001" if pval < 0.001 else f"{pval:.3f}"
print(f"  OR = {or_val:.2f} (95% CI {ci_lo_v:.2f}–{ci_hi_v:.2f}; p={p_str})")

results = {
    "cases_median":    round(c_med, 2),
    "cases_q25":       round(c_q25, 2),
    "cases_q75":       round(c_q75, 2),
    "controls_median": round(r_med, 2),
    "controls_q25":    round(r_q25, 2),
    "controls_q75":    round(r_q75, 2),
    "q4_threshold_pct": round(q4_lo * 100),
    "or":    round(or_val, 2),
    "ci_lo": round(ci_lo_v, 2),
    "ci_hi": round(ci_hi_v, 2),
    "pval":  p_str,
}
with open("data/zcode_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved zcode_results.json")
