"""
Extract FFA manuscript-ready metrics from S3 ffa_causal_factors.csv
and axp_explanations.parquet for CH_3 (opioid_ed) and CH_4 (non_opioid_ed).
Reads ALL density bins (low / medium / high / extreme) for all cohorts / age bands.
"""
import boto3, io, json
from pathlib import Path
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
PREFIX = "gold/ffa_analysis"
SCRIPT_DIR = Path(__file__).parent

COHORTS   = ["opioid_ed", "non_opioid_ed"]
AGE_BANDS = ["0_12", "13_24", "25_44", "45_54", "55_64", "65_74", "75_84", "85_114"]
DENSITY_BINS = ["low", "medium", "high", "extreme"]
META_FEATURES = {"n_events", "n_event_bin_ordinal", "pgx_num_drugs", "pgx_num_cpic_drugs"}


def read_csv(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def read_parquet(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def read_json_s3(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


results = {}

print("=" * 70)
print("FFA causal_factors — ALL bins — top features by causal_responsibility")
print("=" * 70)

for cohort in COHORTS:
    results[cohort] = {}
    for ab in AGE_BANDS:
        band = ab.replace("_", "-")
        results[cohort][band] = {}

        for bin_name in DENSITY_BINS:
            # ── Try manuscript summary JSON first (written by notebook 4 checkpoint cell) ──
            summary_key = f"{PREFIX}/{cohort}/{band}/{bin_name}/ffa_manuscript_summary.json"
            summary = read_json_s3(summary_key)

            # ── Fall back to raw causal_factors.csv ──
            csv_key = f"{PREFIX}/{cohort}/{ab}/bin_models/{bin_name}/ffa_causal_factors.csv"
            df = read_csv(csv_key)

            if df is None and summary is None:
                continue

            if df is not None:
                total_rules = int(df["total_rules"].iloc[0]) if "total_rules" in df.columns else 0
                n_features  = len(df)
                drugs = df[~df["feature"].isin(META_FEATURES)].copy()
                cr_col = "causal_responsibility" if "causal_responsibility" in drugs.columns else \
                         next((c for c in ("ir_score","ir","importance") if c in drugs.columns), None)
                top_drugs = drugs.nlargest(5, cr_col) if cr_col else drugs.head(5)

                print(f"  {cohort}/{band}/{bin_name}  n_features={n_features}  rules={total_rules:,}")
                for _, r in top_drugs.iterrows():
                    name = (r["feature"]
                            .replace("item_drug_", "")
                            .replace("item_icd_", "ICD:")
                            .replace("item_cpt_", "CPT:"))
                    cr_val = r[cr_col] if cr_col else 0
                    shap_val = r.get("shap_importance", 0)
                    rf_val   = r.get("rule_frequency",  0)
                    print(f"    {name:<35s}  CR={cr_val:.4f}  SHAP={shap_val:.4f}  rules={int(rf_val):,}")

                results[cohort][band][bin_name] = {
                    "n_features":    n_features,
                    "total_rules":   total_rules,
                    "top_drugs":     top_drugs["feature"].str.replace("item_drug_", "").tolist()[:5],
                    "top_cr":        top_drugs[cr_col].tolist()[:5] if cr_col else [],
                    "top_rule_freq": top_drugs["rule_frequency"].tolist()[:5] if "rule_frequency" in top_drugs.columns else [],
                }
            elif summary:
                # Use pre-built manuscript summary
                results[cohort][band][bin_name] = {
                    "n_features":  summary.get("n_causal_features", 0),
                    "total_rules": summary.get("n_ffa_rules", 0),
                    "top_drugs":   [f["feature"] for f in summary.get("top_features", [])[:5]],
                    "top_cr":      [f.get("ir_score", 0) for f in summary.get("top_features", [])[:5]],
                    "top_ffa_rules": summary.get("top_ffa_rules", [])[:5],
                    "top_rule_freq": [],
                }
                print(f"  {cohort}/{band}/{bin_name}  [from summary JSON]  "
                      f"n_features={summary.get('n_causal_features',0)}")

# ── Check axp_explanations for pair/triplet counts (CH_4 IE scores) ─────────
print("\n" + "=" * 70)
print("axp_explanations — pair IE scores (non_opioid_ed, geriatric bands, all bins)")
print("=" * 70)

ie_results = {}
for ab in ["65_74", "75_84", "85_114"]:
    band = ab.replace("_", "-")
    ie_results[band] = {}
    for bin_name in DENSITY_BINS:
        for model in ("xgboost", "catboost"):
            key = f"{PREFIX}/non_opioid_ed/{ab}/bin_models/{bin_name}/{model}/axp_explanations.parquet"
            df = read_parquet(key)
            if df is None:
                continue
            ie_col = next((c for c in ("interaction_effect","ie_score","ie","IE",
                                       "synergy_score","probability_shift") if c in df.columns), None)
            rule_col = next((c for c in ("rule","rule_str","antecedent","items") if c in df.columns), None)
            n_rules = len(df)
            top_rules = []
            if ie_col and rule_col:
                top = df.nlargest(10, ie_col)[[rule_col, ie_col]]
                top_rules = top.rename(columns={rule_col: "rule", ie_col: "ie_score"}).to_dict("records")
            ie_results[band][f"{bin_name}_{model}"] = {
                "n_rules": n_rules, "top_rules": top_rules
            }
            print(f"  non_opioid_ed/{band}/{bin_name}/{model}  shape={df.shape}  "
                  f"ie_col={ie_col}")
            break  # one model per bin is enough

# ── Save summary JSON files ───────────────────────────────────────────
out_ffa  = SCRIPT_DIR.parent / "data/ffa_manuscript_data.json"
out_ie   = SCRIPT_DIR.parent / "data/ffa_ie_ci.json"

out_ffa.write_text(json.dumps(results,    indent=2, default=str), encoding="utf-8")
out_ie.write_text( json.dumps(ie_results, indent=2, default=str), encoding="utf-8")

print(f"\nSaved {out_ffa}")
print(f"Saved {out_ie}")

# ── Console summary for manuscript ─────────────────────────────────────────
print()
print("=" * 70)
print("Top causal drug per cohort/band/bin (CH_3, CH_4 placeholder update)")
print("=" * 70)
for cohort in COHORTS:
    for band, bins in results[cohort].items():
        for bin_name, entry in bins.items():
            if entry.get("top_drugs"):
                cr = entry["top_cr"][0] if entry.get("top_cr") else "?"
                print(f"  {cohort}/{band}/{bin_name}: {entry['top_drugs'][0]}  CR={cr}")
