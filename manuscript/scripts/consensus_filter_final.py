"""
Consensus Filter counts for CH_2.
MCCV-selected features = top N by SHAP importance (N from CH_2 text: 498 opioid, 89 polypharm).
FFA-confirmed = causal_responsibility > 0 in ffa_causal_factors.csv.
Consensus = top-N ∩ FFA-confirmed.
SHAP-only = top-N not FFA-confirmed.
FFA-only  = FFA-confirmed not in top-N (FFA found outside MCCV set).
"""
import boto3, io
import pandas as pd
import numpy as np

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
REP_BIN = "low"

MCCV_N = {"opioid_ed": 498, "non_opioid_ed": 89}
COHORTS = {
    "opioid_ed":     ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-114"],
}


def load_shap(cohort, band, bin_name=REP_BIN):
    ab = band.replace("-", "_")
    key = f"gold/final_model/{cohort}/{band}/bin_models/{bin_name}/{cohort}_{ab}_catboost_feature_importance.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


def load_ffa(cohort, band, bin_name=REP_BIN):
    band_us = band.replace("-", "_")
    key = f"gold/ffa_analysis/{cohort}/{band_us}/bin_models/{bin_name}/ffa_causal_factors.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


summary = []

for cohort, bands in COHORTS.items():
    n_mccv = MCCV_N[cohort]
    print(f"\n=== {cohort} (MCCV n={n_mccv}) ===")

    for band in bands:
        shap_df = load_shap(cohort, band)
        ffa_df  = load_ffa(cohort, band)

        if shap_df is None:
            print(f"  {band}: no SHAP data")
            continue

        # MCCV universe = top n_mccv features by SHAP importance
        shap_sorted = shap_df.sort_values("importance", ascending=False)
        mccv_set = set(shap_sorted.head(n_mccv)["feature"])

        if ffa_df is not None:
            ffa_confirmed = set(ffa_df.loc[ffa_df["causal_responsibility"] > 0, "feature"])
        else:
            ffa_confirmed = set()

        consensus = mccv_set & ffa_confirmed
        shap_only = mccv_set - ffa_confirmed
        ffa_only_in_mccv = ffa_confirmed & mccv_set - consensus  # should be empty
        ffa_only_outside = ffa_confirmed - mccv_set               # FFA found outside MCCV top-N

        pct = len(consensus) / n_mccv * 100

        print(f"  {band}: mccv={len(mccv_set)}, ffa_total={len(ffa_confirmed)}, "
              f"consensus={len(consensus)} ({pct:.0f}%), "
              f"shap_only={len(shap_only)}, ffa_only_outside_mccv={len(ffa_only_outside)}")

        summary.append({
            "cohort": cohort, "band": band,
            "consensus": len(consensus), "pct": pct,
            "shap_only": len(shap_only),
            "ffa_only": len(ffa_only_outside),
        })

df = pd.DataFrame(summary)
print("\n" + "=" * 65)
print("MANUSCRIPT VALUES:")
for cohort in COHORTS:
    sub = df[df["cohort"] == cohort]
    print(f"\n  {cohort}:")
    print(f"    Consensus %: {sub['pct'].mean():.0f}% (range {sub['pct'].min():.0f}%–{sub['pct'].max():.0f}%)")
    print(f"    SHAP-only:   {sub['shap_only'].mean():.0f} (range {sub['shap_only'].min()}–{sub['shap_only'].max()})")
    print(f"    FFA-only:    {sub['ffa_only'].mean():.0f} (range {sub['ffa_only'].min()}–{sub['ffa_only'].max()})")

all_pct = df["pct"].mean()
all_so  = df["shap_only"].mean()
all_fo  = df["ffa_only"].mean()
print(f"\n  OVERALL (all 7 bands):")
print(f"    Consensus %: {all_pct:.0f}%")
print(f"    SHAP-only:   {all_so:.0f}")
print(f"    FFA-only:    {all_fo:.0f}")
print(f"\n  FOR CH_2 TEXT:")
print(f"    '[XX%]'       → {all_pct:.0f}%")
print(f"    '[XX] SHAP-only' → {all_so:.0f}")
print(f"    '[XX] FFA-only'  → {all_fo:.0f}")
