"""
Compute SHAP-only, FFA-only, and Consensus-Causal feature counts for CH_2.
SHAP threshold: feature importance >= 75th percentile (i.e., top 25% by SHAP)
FFA threshold:  causal_responsibility > 0  (any FFA identification)
Sources:
  SHAP: gold/final_model/{cohort}/{band}/bin_models/{bin}/{cohort}_{ab}_catboost_feature_importance.csv
  FFA:  gold/ffa_analysis/{cohort}/{band_us}/bin_models/{bin}/ffa_causal_factors.csv
"""
import boto3, io
import pandas as pd
import numpy as np

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BINS   = ["low", "medium", "high", "extreme"]

COHORTS = {
    "opioid_ed":     {"bands": ["13-24", "25-44", "45-54", "55-64"],   "mccv_n": 498},
    "non_opioid_ed": {"bands": ["65-74", "75-84", "85-114"],            "mccv_n": 89},
}


def load_shap(cohort, band, bin_name):
    ab = band.replace("-", "_")
    key = f"gold/final_model/{cohort}/{band}/bin_models/{bin_name}/{cohort}_{ab}_catboost_feature_importance.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


def load_ffa(cohort, band, bin_name):
    band_us = band.replace("-", "_")
    key = f"gold/ffa_analysis/{cohort}/{band_us}/bin_models/{bin_name}/ffa_causal_factors.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


print("Inspecting SHAP and FFA column names …")
# Sample one file to see columns
shap_sample = load_shap("opioid_ed", "25-44", "low")
ffa_sample  = load_ffa("opioid_ed",  "25_44".replace("_", "-"), "low")
if shap_sample is not None:
    print(f"SHAP cols: {list(shap_sample.columns)}")
    print(shap_sample.head(3).to_string())
if ffa_sample is not None:
    print(f"\nFFA cols: {list(ffa_sample.columns)}")
    print(ffa_sample.head(3).to_string())

print("\n" + "=" * 60)

all_results = {}
for cohort, cfg in COHORTS.items():
    print(f"\n=== {cohort} (MCCV n={cfg['mccv_n']}) ===")
    cohort_shap_only, cohort_ffa_only, cohort_consensus = [], [], []

    for band in cfg["bands"]:
        band_shap_only = set()
        band_ffa_only  = set()
        band_consensus = set()

        for bin_name in BINS:
            shap_df = load_shap(cohort, band, bin_name)
            ffa_df  = load_ffa(cohort, band, bin_name)

            if shap_df is None or ffa_df is None:
                continue

            # SHAP feature col
            shap_feat_col = next((c for c in shap_df.columns
                                  if c.lower() in ("feature", "feature_name", "column")),
                                 shap_df.columns[0])
            shap_val_col  = next((c for c in shap_df.columns
                                  if c.lower() in ("importance", "shap", "mean_shap", "value")),
                                 shap_df.columns[1])

            # Top 25% SHAP features
            thresh = shap_df[shap_val_col].quantile(0.75)
            shap_top = set(shap_df.loc[shap_df[shap_val_col] >= thresh, shap_feat_col])

            # FFA feature col
            ffa_feat_col = next((c for c in ffa_df.columns
                                 if c.lower() in ("feature", "feature_name", "drug", "drug_name")),
                                ffa_df.columns[0])
            ffa_val_col  = next((c for c in ffa_df.columns
                                 if "causal" in c.lower() or "responsibility" in c.lower()),
                                None)
            if ffa_val_col is None:
                ffa_val_col = ffa_df.columns[1]

            # FFA features with any causal responsibility > 0
            ffa_top = set(ffa_df.loc[ffa_df[ffa_val_col] > 0, ffa_feat_col])

            # Classify
            band_consensus.update(shap_top & ffa_top)
            band_shap_only.update(shap_top - ffa_top)
            band_ffa_only.update(ffa_top  - shap_top)

        n_consensus = len(band_consensus)
        n_shap_only = len(band_shap_only)
        n_ffa_only  = len(band_ffa_only)
        n_mccv      = cfg["mccv_n"]
        pct = n_consensus / n_mccv * 100 if n_mccv else 0
        print(f"  {band}: consensus={n_consensus} ({pct:.0f}%), shap_only={n_shap_only}, ffa_only={n_ffa_only}")

        cohort_consensus.extend(band_consensus)
        cohort_shap_only.extend(band_shap_only)
        cohort_ffa_only.extend(band_ffa_only)

    uniq_consensus = len(set(cohort_consensus))
    uniq_shap_only = len(set(cohort_shap_only))
    uniq_ffa_only  = len(set(cohort_ffa_only))
    pct_all = uniq_consensus / cfg["mccv_n"] * 100 if cfg["mccv_n"] else 0
    print(f"  AGGREGATE: consensus={uniq_consensus} ({pct_all:.0f}%), shap_only={uniq_shap_only}, ffa_only={uniq_ffa_only}")
    all_results[cohort] = {
        "consensus_pct": pct_all,
        "shap_only": uniq_shap_only,
        "ffa_only":  uniq_ffa_only,
    }

print("\n=== CH_2 Summary (both cohorts combined) ===")
total_consensus_pct = np.mean([v["consensus_pct"] for v in all_results.values()])
total_shap_only = sum(v["shap_only"] for v in all_results.values())
total_ffa_only  = sum(v["ffa_only"]  for v in all_results.values())
print(f"  Mean Consensus-Causal %: {total_consensus_pct:.0f}%")
print(f"  SHAP-only (across both cohorts): {total_shap_only}")
print(f"  FFA-only  (across both cohorts): {total_ffa_only}")
