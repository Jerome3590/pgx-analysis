"""Count rows in train_final_features CSVs via S3 Select (no full download)."""
import boto3, json

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

cohorts = {
    "opioid_ed":     ["0-12","13-24","25-44","45-54","55-64","65-74","75-84","85-114"],
    "non_opioid_ed": ["0-12","13-24","25-44","45-54","55-64","65-74","75-84","85-114"],
}

def s3_select_count(key):
    resp = s3.select_object_content(
        Bucket=BUCKET, Key=key,
        ExpressionType="SQL",
        Expression="SELECT COUNT(*) FROM S3Object",
        InputSerialization={"CSV": {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"},
        OutputSerialization={"JSON": {}},
    )
    result = ""
    for event in resp["Payload"]:
        if "Records" in event:
            result += event["Records"]["Payload"].decode()
    return int(json.loads(result.strip())["_1"])

def s3_select_case_ctrl(key):
    """Count target==1 (cases) and target==0 (controls)."""
    results = {}
    for tgt in ("1", "0"):
        resp = s3.select_object_content(
            Bucket=BUCKET, Key=key,
            ExpressionType="SQL",
            Expression=f"SELECT COUNT(*) FROM S3Object s WHERE s.target = '{tgt}'",
            InputSerialization={"CSV": {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"},
            OutputSerialization={"JSON": {}},
        )
        out = ""
        for event in resp["Payload"]:
            if "Records" in event:
                out += event["Records"]["Payload"].decode()
        results[tgt] = int(json.loads(out.strip())["_1"])
    return results["1"], results["0"]

totals = {}
for c, bands in cohorts.items():
    totals[c] = {}
    for b in bands:
        ab = b.replace("-", "_")
        key = f"gold/final_model/{c}/{b}/{c}_{ab}_train_final_features_no_leakage.csv"
        try:
            cases, ctrl = s3_select_case_ctrl(key)
            totals[c][b] = {"cases": cases, "controls": ctrl}
            print(f"{c:20s} | {b:8s} | cases={cases:8,} | controls={ctrl:8,} | ratio={ctrl/cases if cases else 0:.1f}x")
        except Exception as e:
            print(f"  ERROR {c}/{b}: {e}")

print("\n=== TOTALS (training set, 2016-2018) ===")
for c, bands in totals.items():
    gc = sum(v["cases"]    for v in bands.values())
    gr = sum(v["controls"] for v in bands.values())
    print(f"{c}: cases={gc:,}  controls={gr:,}")

with open("data/cohort_counts_train.json","w") as f:
    json.dump(totals, f, indent=2)
print("Saved cohort_counts_train.json")
