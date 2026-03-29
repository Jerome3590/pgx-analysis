"""
Compute counterfactual Δp̂ for lorazepam removal using the 85-114 CatBoost model.
Reports mean |Δp̂| for patients who have lorazepam present vs removed.
"""
import boto3, io, tempfile, os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BAND   = "85-114"
COHORT = "non_opioid_ed"
BINN   = "low"


def load_model(key, tmp):
    local = os.path.join(tmp, "model.cbm")
    s3.download_file(BUCKET, key, local)
    m = CatBoostClassifier()
    m.load_model(local)
    return m


def read_parquet(key, cols=None):
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(data), columns=cols)


print("Loading CatBoost model and test features …")
with tempfile.TemporaryDirectory() as tmp:
    model = load_model(
        f"gold/final_model/{COHORT}/{BAND}/bin_models/{BINN}/catboost.joblib", tmp
    )
    # joblib model won't work with load_model — use .cbm
    try:
        model = load_model(
            f"gold/final_model/{COHORT}/{BAND}/bin_models/{BINN}/"
            f"{COHORT}_{BAND.replace('-','_')}_best_catboost_model.cbm", tmp
        )
        print("  Loaded .cbm model")
    except Exception as e:
        print(f"  .cbm failed: {e}")

# Load test features
key_test = (f"gold/final_model/{COHORT}/{BAND}/inputs/"
            f"model_test/final_features.parquet")
ff = read_parquet(key_test)
print(f"  Test features: {ff.shape}")

# Find patients in the low bin with lorazepam present
lora_col = "item_drug_LORAZEPAM"
if lora_col not in ff.columns:
    print(f"  {lora_col} not in test features — looking for alternatives …")
    benz_cols = [c for c in ff.columns if any(
        b in c.upper() for b in ["LORAZEPAM", "DIAZEPAM", "ALPRAZOLAM"])]
    print(f"  Found: {benz_cols}")
    lora_col = benz_cols[0] if benz_cols else None

if lora_col is None:
    print("No benzodiazepine column found — exiting")
    exit()

ff_low = ff[ff["n_event_bin"] == BINN].copy()
lora_pts = ff_low[ff_low[lora_col] == 1]
print(f"  Low-bin patients with {lora_col}=1: {len(lora_pts):,}")

if len(lora_pts) == 0:
    print("No patients with lorazepam in low bin — trying all bins")
    lora_pts = ff[ff[lora_col] == 1]
    print(f"  All-bin patients with {lora_col}=1: {len(lora_pts):,}")

if len(lora_pts) == 0:
    print("No lorazepam patients found")
    exit()

# Get feature names from model
feat_names = model.feature_names_
X_with = lora_pts[[c for c in feat_names if c in lora_pts.columns]].copy()
# Align to model features
for col in feat_names:
    if col not in X_with.columns:
        X_with[col] = 0
X_with = X_with[feat_names].fillna(0)

# Counterfactual: set lorazepam to 0
X_without = X_with.copy()
if lora_col in X_without.columns:
    X_without[lora_col] = 0

p_with    = model.predict_proba(X_with.values)[:, 1]
p_without = model.predict_proba(X_without.values)[:, 1]
delta     = p_with - p_without

print(f"\n=== COUNTERFACTUAL: Remove {lora_col} ===")
print(f"  n patients: {len(lora_pts):,}")
print(f"  mean p̂(with lorazepam):    {p_with.mean():.4f}")
print(f"  mean p̂(without lorazepam): {p_without.mean():.4f}")
print(f"  mean Δp̂:                    {delta.mean():.4f}")
print(f"  mean |Δp̂| (%):              {delta.mean()*100:.1f}%")
print(f"  median Δp̂:                  {np.median(delta):.4f}")
print(f"  75th pctile Δp̂:             {np.percentile(delta,75):.4f}")
print()
print(f"  → Dashboard would show: 'Removing {lora_col.replace('item_drug_','').title()} "
      f"reduces risk by ~{delta.mean()*100:.0f}%'")
