"""
Extract FFA manuscript-ready metrics from S3 ffa_causal_factors.csv
and axp_explanations.parquet for CH_3 (opioid_ed) and CH_4 (non_opioid_ed).
"""
import boto3, io, json
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
PREFIX = "gold/ffa_analysis"


def read_csv(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        return None


def read_parquet(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        return None


def top_features(df, n=10, label_col="feature", score_col="causal_responsibility"):
    if df is None or df.empty:
        return []
    top = df.nlargest(n, score_col)
    return top[[label_col, score_col, "shap_importance", "rule_frequency"]].to_dict("records")


configs = {
    "non_opioid_ed": ["65_74", "75_84", "85_114"],
    "opioid_ed":     ["13_24", "25_44", "45_54", "55_64"],
}

results = {}

print("=" * 70)
print("FFA causal_factors — top features by causal_responsibility (bin=low)")
print("=" * 70)

for cohort, bands in configs.items():
    results[cohort] = {}
    for ab in bands:
        band = ab.replace("_", "-")
        key = f"{PREFIX}/{cohort}/{ab}/bin_models/low/ffa_causal_factors.csv"
        df = read_csv(key)
        if df is None:
            print(f"  SKIP {cohort}/{band}: not found")
            continue

        total_rules = int(df["total_rules"].iloc[0]) if "total_rules" in df.columns else 0
        n_features  = len(df)

        # Top drug features only (exclude meta-features)
        meta = {"n_events", "pgx_num_drugs", "pgx_num_cpic_drugs"}
        drugs = df[~df["feature"].isin(meta)].copy()
        top_drugs = drugs.nlargest(5, "causal_responsibility")

        print(f"\n  {cohort} / {band}  (n_features={n_features}, total_rules={total_rules:,})")
        for _, r in top_drugs.iterrows():
            name = r["feature"].replace("item_drug_", "").replace("item_icd_", "ICD:").replace("item_cpt_", "CPT:")
            print(f"    {name:<35s}  CR={r['causal_responsibility']:.4f}  SHAP={r['shap_importance']:.4f}  rules={int(r['rule_frequency']):,}")

        results[cohort][band] = {
            "n_features": n_features,
            "total_rules": total_rules,
            "top_drugs": top_drugs["feature"].str.replace("item_drug_", "").tolist()[:5],
            "top_cr": top_drugs["causal_responsibility"].tolist()[:5],
            "top_rule_freq": top_drugs["rule_frequency"].tolist()[:5],
        }

# ── Check axp_explanations for pair-level interaction data ────────────────
print("\n" + "=" * 70)
print("axp_explanations — checking for interaction pair data (non_opioid_ed, bin=low)")
print("=" * 70)

for ab in ["65_74", "75_84", "85_114"]:
    band = ab.replace("_", "-")
    for model in ("xgboost", "catboost"):
        key = f"{PREFIX}/non_opioid_ed/{ab}/bin_models/low/{model}/axp_explanations.parquet"
        df = read_parquet(key)
        if df is None:
            continue
        print(f"\n  non_opioid_ed / {band} / {model}  shape={df.shape}")
        print(f"  columns: {list(df.columns)[:20]}")
        # Show first row to understand structure
        if len(df) > 0:
            row = df.iloc[0].to_dict()
            for k, v in list(row.items())[:15]:
                print(f"    {k}: {str(v)[:80]}")
        break  # one model is enough to understand structure

# ── Save summary JSON ────────────────────────────────────────────────────
with open("ffa_manuscript_data.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved ffa_manuscript_data.json")
