import boto3, json

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

cohorts = {
    "opioid_ed":     ["13-24","25-44","45-54","55-64"],
    "non_opioid_ed": ["65-74","75-84","85-114"],
}

def s3sel(key, expr):
    r = s3.select_object_content(
        Bucket=BUCKET, Key=key, ExpressionType="SQL", Expression=expr,
        InputSerialization={"CSV": {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"},
        OutputSerialization={"JSON": {}},
    )
    out = ""
    for ev in r["Payload"]:
        if "Records" in ev:
            out += ev["Records"]["Payload"].decode()
    return int(json.loads(out.strip())["_1"])

totals = {}
for c, bands in cohorts.items():
    totals[c] = {}
    for b in bands:
        ab = b.replace("-", "_")
        key = f"gold/final_model/{c}/{b}/{c}_{ab}_train_final_features_no_leakage.csv"
        try:
            total = s3sel(key, "SELECT COUNT(*) FROM S3Object")
            cases = s3sel(key, "SELECT COUNT(*) FROM S3Object WHERE target = '1'")
            ctrl  = total - cases
            totals[c][b] = {"total": total, "cases": cases, "controls": ctrl}
            ratio = ctrl/cases if cases else 0
            print(f"{c:20s} | {b:8s} | total={total:8,} | cases={cases:7,} | ctrl={ctrl:7,} | {ratio:.1f}x")
        except Exception as e:
            print(f"  ERROR {c}/{b}: {e}")

print("\n=== GRAND TOTALS ===")
for c, bands in totals.items():
    gc = sum(v.get("cases",0)    for v in bands.values())
    gr = sum(v.get("controls",0) for v in bands.values())
    print(f"{c}: cases={gc:,}  controls={gr:,}")

with open("data/cohort_counts_train.json", "w") as f:
    json.dump(totals, f, indent=2)
print("Saved cohort_counts_train.json")
