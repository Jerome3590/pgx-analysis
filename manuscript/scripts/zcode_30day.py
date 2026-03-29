"""
Z-code proportion (30-day pre-index window) for CH_4 non_opioid_ed cohort.
Uses model_events.parquet which has ICD codes + first_o11_p_date (index date).
"""
import boto3, io
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

s3     = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["65-74", "75-84", "85-114"]
ICD_COLS = [
    "primary_icd_diagnosis_code", "two_icd_diagnosis_code",
    "three_icd_diagnosis_code",   "four_icd_diagnosis_code",
    "five_icd_diagnosis_code",    "six_icd_diagnosis_code",
    "seven_icd_diagnosis_code",   "eight_icd_diagnosis_code",
    "nine_icd_diagnosis_code",    "ten_icd_diagnosis_code",
]

all_frames = []

for band in BANDS:
    key  = (f"gold/cohorts_model_data/cohort_name=non_opioid_ed/"
            f"age_band={band}/model_events.parquet")
    print(f"Loading {band} …", end=" ", flush=True)
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    usecols = ["mi_person_key", "target", "event_date",
               "first_o11_p_date"] + ICD_COLS
    df = pd.read_parquet(io.BytesIO(data), columns=usecols)
    df["band"] = band
    all_frames.append(df)
    print(f"{len(df):,} rows")

events = pd.concat(all_frames, ignore_index=True)
print(f"\nTotal rows: {len(events):,}")

# Parse dates
events["event_date"]       = pd.to_datetime(events["event_date"])
events["first_o11_p_date"]  = pd.to_datetime(events["first_o11_p_date"])

# Controls have no first_o11_p_date → assign band-median of case index dates
for band in BANDS:
    case_mask  = (events["band"] == band) & (events["target"] == 1)
    ctrl_mask  = (events["band"] == band) & (events["target"] == 0)
    median_idx = events.loc[case_mask, "first_o11_p_date"].dropna().median()
    events.loc[ctrl_mask & events["first_o11_p_date"].isna(),
               "first_o11_p_date"] = median_idx
    print(f"  {band}: control pseudo-index = {median_idx.date()}")

# 30-day pre-index window
events["days_before"] = (events["first_o11_p_date"] - events["event_date"]).dt.days
win = events[(events["days_before"] > 0) & (events["days_before"] <= 30)].copy()
print(f"30-day pre-index events: {len(win):,}")

# Flag Z-code events (any ICD position starting with 'Z' or 'z')
def has_z(row):
    for col in ICD_COLS:
        val = row[col]
        if isinstance(val, str) and val.upper().startswith("Z"):
            return 1
    return 0

win["is_z"] = win[ICD_COLS].apply(
    lambda r: int(any(
        isinstance(v, str) and v.upper().startswith("Z")
        for v in r
    )), axis=1)

# Per-patient summary
pat = (win.groupby(["mi_person_key", "target", "band"])
         .agg(total_claims=("is_z", "count"),
              z_claims=("is_z", "sum"))
         .reset_index())
pat["z_prop"] = pat["z_claims"] / pat["total_claims"].clip(lower=1)

n_cases = (pat["target"] == 1).sum()
n_ctrl  = (pat["target"] == 0).sum()
print(f"\nPatients with ≥1 30-day event: {len(pat):,}  "
      f"({n_cases:,} cases, {n_ctrl:,} controls)")

# IQR by case/control
for label, mask in [("Cases", pat["target"] == 1),
                    ("Controls", pat["target"] == 0)]:
    g = pat[mask]["z_prop"]
    q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
    print(f"  {label}: median={q50:.3f}  IQR={q25:.3f}–{q75:.3f}  n={mask.sum():,}")

# Patients with NO pre-index events get z_prop = 0
all_patients = (events[["mi_person_key", "target", "band"]]
                .drop_duplicates("mi_person_key"))
merged = all_patients.merge(
    pat[["mi_person_key", "z_prop", "total_claims"]],
    on="mi_person_key", how="left")
merged["z_prop"]       = merged["z_prop"].fillna(0)
merged["total_claims"] = merged["total_claims"].fillna(0)

print(f"\nFull cohort (incl. no-event patients): {len(merged):,}")
for label, mask in [("Cases", merged["target"] == 1),
                    ("Controls", merged["target"] == 0)]:
    g = merged[mask]["z_prop"]
    q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
    print(f"  {label}: median={q50:.3f}  IQR={q25:.3f}–{q75:.3f}  n={mask.sum():,}")

# ── Also get n_events from final_features ───────────────────────────────────
nev_frames = []
for band in BANDS:
    for split in ["model_test", "model_train"]:
        key = (f"gold/final_model/non_opioid_ed/{band}/inputs/"
               f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            ff   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target", "n_events"])
            ff["band"] = band
            nev_frames.append(ff)
        except Exception as e:
            print(f"  {split}/{band}: {e}")
nev = pd.concat(nev_frames).drop_duplicates("mi_person_key")
merged2 = merged.merge(nev[["mi_person_key", "n_events"]], on="mi_person_key", how="left")

# ── Logistic regression ──────────────────────────────────────────────────────
print("\nLogistic regression: target ~ z_prop_quartile + n_events + band")
df_lr = merged2.dropna(subset=["z_prop", "n_events"]).copy()
# qcut with many zeros can collapse bins — use quartile ranks robustly
try:
    df_lr["z_q"] = pd.qcut(df_lr["z_prop"], q=4, labels=[1, 2, 3, 4],
                           duplicates="drop").astype(float)
except ValueError:
    # Fall back: rank-based quartile assignment
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
    return -np.sum(y * np.log(p + 1e-15) + (1-y) * np.log(1-p + 1e-15))

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

print(f"  z_q β={coef:.3f}  OR={np.exp(coef):.2f}  "
      f"95%CI [{np.exp(ci_lo):.2f}–{np.exp(ci_hi):.2f}]  p={pval:.4f}")

# ── Q4 threshold ─────────────────────────────────────────────────────────────
q4_lo = df_lr["z_prop"].quantile(0.75)
print(f"\n  Q4 lower bound (75th pct): {q4_lo:.3f} "
      f"({q4_lo*100:.0f}% of claims)")

# ── Final manuscript values ───────────────────────────────────────────────────
cases_m = merged[merged["target"] == 1]
ctrl_m  = merged[merged["target"] == 0]
print(f"\n=== CH_4 MANUSCRIPT VALUES ===")
print(f"  Z-code IQR Cases:    {cases_m['z_prop'].median():.2f}  "
      f"({cases_m['z_prop'].quantile(0.25):.2f}–{cases_m['z_prop'].quantile(0.75):.2f})")
print(f"  Z-code IQR Controls: {ctrl_m['z_prop'].median():.2f}  "
      f"({ctrl_m['z_prop'].quantile(0.25):.2f}–{ctrl_m['z_prop'].quantile(0.75):.2f})")
print(f"  Q4 threshold: ≥ {q4_lo*100:.0f}%")
print(f"  OR = {np.exp(coef):.2f} "
      f"(95% CI {np.exp(ci_lo):.2f}–{np.exp(ci_hi):.2f}; "
      f"p={'<0.001' if pval < 0.001 else f'{pval:.3f}'})")
