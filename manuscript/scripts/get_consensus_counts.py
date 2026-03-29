import boto3, json
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

configs = [
    ("opioid_ed",     ["13_24","25_44","45_54","55_64"]),
    ("non_opioid_ed", ["65_74","75_84","85_114"]),
]

print("=== Consensus-Causal Feature Counts (SHAP ∩ FFA allowed codes) ===")
for cohort, bands in configs:
    for b in bands:
        key = f"gold/bupar/allowed_codes/allowed_codes_shap_ffa_{cohort}_{b}.json"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            codes = json.loads(obj["Body"].read())
            band_disp = b.replace("_", "-")
            print(f"  {cohort:15s} / {band_disp:7s}: {len(codes):4d} features")
        except Exception as e:
            print(f"  SKIP {cohort}/{b}: {e}")

# Also check what FFA output files exist on S3
print("\n=== Scanning for FFA causal output files ===")
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix="gold/"):
    for obj in page.get("Contents", []):
        k = obj["Key"]
        if any(x in k for x in ["causal_factor", "ffa_result", "feature_importance_axp", "ffa_causal"]):
            print(f"  {obj['LastModified'].strftime('%Y-%m-%d %H:%M')}  {obj['Size']:>10,}  {k}")
