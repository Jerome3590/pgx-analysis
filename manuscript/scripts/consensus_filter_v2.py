"""
Compute SHAP-only, FFA-only, and Consensus-Causal feature counts for CH_2.
Uses ffa_causal_factors.csv which contains BOTH shap_importance AND causal_responsibility.
Approach: per-band analysis using the MCCV feature set as the universe.
  - SHAP-confirmed: shap_importance >= 75th pct of the band's shap distribution
  - FFA-confirmed:  causal_responsibility > 0
  - Consensus:      both
  - SHAP-only:      SHAP but not FFA
  - FFA-only:       FFA but not SHAP (features in FFA file not in SHAP top-25%)
Uses a representative bin ("low") per band for per-band counts.
"""
import boto3, io
import pandas as pd
import numpy as np

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
REP_BIN = "low"  # representative bin per band

COHORTS = {
    "opioid_ed":     ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-114"],
}

MCCV_N = {"opioid_ed": 498, "non_opioid_ed": 89}


def load_ffa(cohort, band, bin_name=REP_BIN):
    band_us = band.replace("-", "_")
    key = f"gold/ffa_analysis/{cohort}/{band_us}/bin_models/{bin_name}/ffa_causal_factors.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


def load_shap(cohort, band, bin_name=REP_BIN):
    ab = band.replace("-", "_")
    key = f"gold/final_model/{cohort}/{band}/bin_models/{bin_name}/{cohort}_{ab}_catboost_feature_importance.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


print("=" * 65)
all_band_results = {}

for cohort, bands in COHORTS.items():
    mccv_n = MCCV_N[cohort]
    print(f"\n=== {cohort} (MCCV n={mccv_n}) ===")
    band_pcts = []
    band_shap_only_n = []
    band_ffa_only_n  = []

    for band in bands:
        # Load SHAP FI (defines the MCCV feature universe)
        shap_df = load_shap(cohort, band)
        # Load FFA causal factors (has both shap_importance + causal_responsibility)
        ffa_df  = load_ffa(cohort, band)

        if shap_df is None and ffa_df is None:
            print(f"  {band}: no data")
            continue

        # --- MCCV universe from SHAP FI file ---
        if shap_df is not None:
            mccv_features = set(shap_df["feature"])
            shap_thresh   = shap_df["importance"].quantile(0.75)
            shap_top      = set(shap_df.loc[shap_df["importance"] >= shap_thresh, "feature"])
        else:
            mccv_features = set()
            shap_top = set()

        # --- FFA-confirmed: causal_responsibility > 0 ---
        if ffa_df is not None:
            ffa_top = set(ffa_df.loc[ffa_df["causal_responsibility"] > 0, "feature"])
            ffa_all = set(ffa_df["feature"])
        else:
            ffa_top = set()
            ffa_all = set()

        # Restrict to MCCV universe (features that went through MCCV)
        shap_top_mccv = shap_top & mccv_features
        ffa_top_mccv  = ffa_top  & mccv_features
        ffa_only_extra = ffa_top - mccv_features  # FFA confirmed but NOT in MCCV

        consensus  = shap_top_mccv & ffa_top_mccv
        shap_only  = shap_top_mccv - ffa_top_mccv
        ffa_only   = ffa_top_mccv  - shap_top_mccv  # FFA in MCCV but not top SHAP

        pct = len(consensus) / mccv_n * 100 if mccv_n else 0
        band_pcts.append(pct)
        band_shap_only_n.append(len(shap_only))
        band_ffa_only_n.append(len(ffa_only) + len(ffa_only_extra))

        print(f"  {band}: MCCV={len(mccv_features)}, "
              f"SHAP-top25%={len(shap_top_mccv)}, "
              f"FFA-confirmed={len(ffa_top_mccv)}, "
              f"Consensus={len(consensus)} ({pct:.0f}%), "
              f"SHAP-only={len(shap_only)}, "
              f"FFA-only-in-MCCV={len(ffa_only)}, "
              f"FFA-only-outside-MCCV={len(ffa_only_extra)}")

    if band_pcts:
        avg_pct = np.mean(band_pcts)
        avg_shap_only = int(np.mean(band_shap_only_n))
        avg_ffa_only  = int(np.mean(band_ffa_only_n))
        print(f"\n  Summary for {cohort}:")
        print(f"    Mean consensus %: {avg_pct:.0f}% (range {min(band_pcts):.0f}%–{max(band_pcts):.0f}%)")
        print(f"    Mean SHAP-only:   {avg_shap_only} (range {min(band_shap_only_n)}–{max(band_shap_only_n)})")
        print(f"    Mean FFA-only:    {avg_ffa_only}  (range {min(band_ffa_only_n)}–{max(band_ffa_only_n)})")
        all_band_results[cohort] = {
            "pct": avg_pct, "pct_range": (min(band_pcts), max(band_pcts)),
            "shap_only": avg_shap_only, "shap_only_range": (min(band_shap_only_n), max(band_shap_only_n)),
            "ffa_only":  avg_ffa_only,  "ffa_only_range":  (min(band_ffa_only_n),  max(band_ffa_only_n)),
        }

print("\n" + "=" * 65)
print("CH_2 MANUSCRIPT VALUES:")
for cohort, r in all_band_results.items():
    print(f"\n  {cohort}:")
    print(f"    [XX%]  → {r['pct']:.0f}% (range {r['pct_range'][0]:.0f}%–{r['pct_range'][1]:.0f}%)")
    print(f"    [XX] SHAP-only → {r['shap_only']} (range {r['shap_only_range'][0]}–{r['shap_only_range'][1]})")
    print(f"    [XX] FFA-only  → {r['ffa_only']}  (range {r['ffa_only_range'][0]}–{r['ffa_only_range'][1]})")
