"""
Compute Z-code OR for CH_4 (non_opioid_ed).
Uses gold/final_model/non_opioid_ed/{band}/inputs/model_test/final_features.parquet
Outcome: ADE ED visit (target)
Predictor: Z-code proportion quartile (z_prop_q: 1–4)
Covariates: age, sex (female dummy), drug_count
"""
import boto3, io, re, warnings
import pandas as pd
import numpy as np
from scipy.stats import chi2
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

# ── 1. Inspect columns ────────────────────────────────────────────────────────
key_65 = "gold/final_model/non_opioid_ed/65-74/inputs/model_test/final_features.parquet"
data = s3.get_object(Bucket=BUCKET, Key=key_65)["Body"].read()
df65 = pd.read_parquet(io.BytesIO(data))

z_cols = [c for c in df65.columns if re.search(r"z_?code|zcode|z00|z_prop|zc_", c, re.I)]
age_cols = [c for c in df65.columns if "age" in c.lower()]
sex_cols = [c for c in df65.columns if re.search(r"sex|gender|female|male", c, re.I)]
print(f"Z-code related cols: {z_cols[:20]}")
print(f"Age cols:            {age_cols[:10]}")
print(f"Sex cols:            {sex_cols[:10]}")
print(f"Total cols:          {len(df65.columns)}")
print(f"Sample cols:         {list(df65.columns[:20])}")

# ── 2. Load all non_opioid_ed bands ─────────────────────────────────────────
bands = ["65-74", "75-84", "85-114"]
frames = []
for band in bands:
    key = f"gold/final_model/non_opioid_ed/{band}/inputs/model_test/final_features.parquet"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        df = pd.read_parquet(io.BytesIO(data))
        df["_band"] = band
        frames.append(df)
        print(f"\n{band}: n={len(df):,}, cases={df['target'].sum():,}")
    except Exception as e:
        print(f"SKIP {band}: {e}")

if not frames:
    print("No data loaded — exiting.")
    raise SystemExit(1)

df = pd.concat(frames, ignore_index=True)

# ── 3. Identify Z-code proxy and covariates ──────────────────────────────────
tgt_col = "target" if "target" in df.columns else "is_target_case"

# Drug count = sum of item_drug_* binary columns
drug_cols = [c for c in df.columns if c.startswith("item_drug_")]
df["drug_count"] = df[drug_cols].sum(axis=1)

# Check for direct Z-code proportion column
z_prop_col = None
for candidate in ("z_code_prop", "zcode_prop", "z_prop", "prop_zcode"):
    if candidate in df.columns:
        z_prop_col = candidate
        break

# If no direct Z-code column, look for item_diag_ with Z prefix or n_zcode cols
diag_z_cols = [c for c in df.columns if c.startswith("item_diag_Z") or c.startswith("item_diag_z")]
print(f"\nDirect Z-prop col: {z_prop_col}")
print(f"item_diag_Z* cols:  {len(diag_z_cols)} — {diag_z_cols[:10]}")

if z_prop_col is None and diag_z_cols:
    # Build Z-code proportion as: sum(Z-code diagnosis binary flags) / n_events
    df["z_sum"] = df[diag_z_cols].sum(axis=1)
    n_events_col = "n_events" if "n_events" in df.columns else None
    if n_events_col:
        df["z_prop"] = (df["z_sum"] / df[n_events_col]).clip(0, 1)
        z_prop_col = "z_prop"
        print(f"Built z_prop from {len(diag_z_cols)} item_diag_Z cols / n_events")
    else:
        df["z_prop"] = df["z_sum"] / df["z_sum"].max().clip(1)
        z_prop_col = "z_prop"
        print("Built z_prop as normalized z_sum (no n_events)")
elif z_prop_col is None:
    print("\nNo Z-code proportion data found — listing all item_diag columns:")
    diag_all = [c for c in df.columns if c.startswith("item_diag_")]
    print(f"  {len(diag_all)} item_diag_ cols: {diag_all[:30]}")
    raise SystemExit("Cannot run regression without Z-code data")

# Age column
age_col = next((c for c in ("age", "age_imputed", "age_at_index") if c in df.columns), None)
print(f"Age col: {age_col}")

# Female binary
female_col = next((c for c in ("female", "is_female", "sex_female") if c in df.columns), None)
if female_col is None:
    # Check gender string col
    gender_col = next((c for c in df.columns if "gender" in c.lower()), None)
    if gender_col:
        df["female"] = (df[gender_col] == "F").astype(int)
        female_col = "female"
print(f"Female col: {female_col}")

# ── 4. Summary stats for Z-code proportion ───────────────────────────────────
print(f"\nZ-code proportion ({z_prop_col}) summary:")
print(df.groupby(tgt_col)[z_prop_col].describe().round(4))

# ── 5. Build regression dataset ───────────────────────────────────────────────
reg_cols = [tgt_col, z_prop_col, "drug_count"]
if age_col:
    reg_cols.append(age_col)
if female_col:
    reg_cols.append(female_col)

reg = df[reg_cols].copy().dropna()
reg.columns = ["outcome", "z_prop", "drug_count"] + (
    ["age"] if age_col else []) + (["female"] if female_col else [])

# Quartile of Z-code proportion among cases + controls
reg["z_prop_q"] = pd.qcut(reg["z_prop"], q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(float)
print(f"\nZ-prop quartile distribution:\n{reg['z_prop_q'].value_counts().sort_index()}")

# ── 6. Logistic regression ───────────────────────────────────────────────────
print("\n--- Logistic Regression ---")
formula_parts = ["z_prop_q", "drug_count"]
if age_col:
    formula_parts.append("age")
if female_col:
    formula_parts.append("female")

formula = "outcome ~ " + " + ".join(formula_parts)
print(f"Formula: {formula}")

model = smf.logit(formula, data=reg).fit(disp=False)
print(model.summary2())

# Extract Z-code quartile OR and CI
params = model.params
conf = model.conf_int()
z_or  = np.exp(params["z_prop_q"])
z_lo  = np.exp(conf.loc["z_prop_q", 0])
z_hi  = np.exp(conf.loc["z_prop_q", 1])
z_p   = model.pvalues["z_prop_q"]

print(f"\n=== Z-code Quartile OR ===")
print(f"  OR  = {z_or:.2f}")
print(f"  95% CI = ({z_lo:.2f} – {z_hi:.2f})")
print(f"  p-value = {z_p:.4f}")
print(f"\n  Formatted: OR = {z_or:.2f} (95% CI {z_lo:.2f}–{z_hi:.2f})")

# Also show per-quartile ORs relative to Q1 baseline using dummy coding
reg["z_q1"] = (reg["z_prop_q"] == 1).astype(int)
reg["z_q2"] = (reg["z_prop_q"] == 2).astype(int)
reg["z_q3"] = (reg["z_prop_q"] == 3).astype(int)
reg["z_q4"] = (reg["z_prop_q"] == 4).astype(int)

formula2 = "outcome ~ z_q2 + z_q3 + z_q4 + drug_count"
if age_col:
    formula2 += " + age"
if female_col:
    formula2 += " + female"

model2 = smf.logit(formula2, data=reg).fit(disp=False)
params2 = model2.params
conf2 = model2.conf_int()

print(f"\n=== Per-Quartile ORs vs Q1 ===")
for q in ("z_q2", "z_q3", "z_q4"):
    if q in params2:
        or_q = np.exp(params2[q])
        lo_q = np.exp(conf2.loc[q, 0])
        hi_q = np.exp(conf2.loc[q, 1])
        p_q  = model2.pvalues[q]
        print(f"  {q}: OR = {or_q:.2f} (95% CI {lo_q:.2f}–{hi_q:.2f}), p = {p_q:.4f}")
