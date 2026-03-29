import boto3, json

s3 = boto3.client("s3", region_name="us-east-1")
DASHBOARD_BUCKET = "jerome-dixon.io"
PREFIX = "vcu/pgx-risk-calculator/visualizations/causal"

targets = [
    ("non_opioid_ed", "65-74",  "low"),
    ("non_opioid_ed", "75-84",  "low"),
    ("non_opioid_ed", "85-114", "low"),
    ("opioid_ed",     "25-44",  "low"),
    ("opioid_ed",     "13-24",  "low"),
]

for cohort, band, bin_name in targets:
    key = f"{PREFIX}/{cohort}/{band}/{bin_name}/causal_data.json"
    try:
        obj = s3.get_object(Bucket=DASHBOARD_BUCKET, Key=key)
        d = json.loads(obj["Body"].read())
    except Exception as e:
        print(f"SKIP {cohort}/{band}/{bin_name}: {e}")
        continue

    print(f"\n{'='*70}")
    print(f"  {cohort} / {band} / bin={bin_name}")
    print(f"{'='*70}")

    summary = d.get("summary", {})
    print(f"  Summary: {json.dumps(summary, indent=4)}")

    factors = d.get("top_causal_factors", [])
    print(f"\n  top_causal_factors ({len(factors)} total) — top 15:")
    for i, f in enumerate(factors[:15]):
        feat = f.get("feature", f.get("feature_name", "?"))
        imp  = f.get("importance", f.get("combined_importance", 0))
        shap = f.get("shap_norm", f.get("shap_importance", ""))
        ffa  = f.get("ffa_norm",  f.get("ffa_importance", ""))
        ir   = f.get("ir", f.get("intervention_rate", ""))
        print(f"    {i+1:2d}. {feat:<45s} imp={imp:.4f}"
              f"  shap={shap:.3f}" if isinstance(shap, float) else
              f"  ir={ir}" if ir != "" else "")

    # Check for interaction/pair keys
    print(f"\n  All keys in causal_data: {list(d.keys())}")
    notes = d.get("notes", {})
    if notes:
        print(f"  Notes: {json.dumps(notes, indent=4)}")
