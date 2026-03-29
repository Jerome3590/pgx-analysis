import boto3, io, json
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

COHORTS = ["opioid_ed", "non_opioid_ed"]
BANDS   = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]


def read_parquet_cols(key, cols):
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(data), columns=cols)
    except Exception:
        return None


totals = {}
for c in COHORTS:
    totals[c] = {}
    for b in BANDS:
        cases_total = 0
        ctrl_total  = 0
        pkeys_seen  = set()
        found_any   = False
        for split in ("model_train", "model_test"):
            key = f"gold/final_model/{c}/{b}/inputs/{split}/final_features.parquet"
            df  = read_parquet_cols(key, ["mi_person_key", "target"])
            if df is None:
                continue
            found_any = True
            df = df[~df["mi_person_key"].isin(pkeys_seen)]
            pkeys_seen.update(df["mi_person_key"].tolist())
            cases_total += int((df["target"] == 1).sum())
            ctrl_total  += int((df["target"] == 0).sum())
        if not found_any:
            print(f"  SKIP {c}/{b}: no parquet found")
            continue
        totals[c][b] = {"cases": cases_total, "controls": ctrl_total}
        print(f"{c:20s} | {b:8s} | cases={cases_total:8,} | controls={ctrl_total:8,}")

print("\n=== TOTALS ===")
for c, bands in totals.items():
    tc = sum(v["cases"] for v in bands.values())
    tr = sum(v["controls"] for v in bands.values())
    print(f"{c}: total cases={tc:,}  total controls={tr:,}")

with open("cohort_counts.json", "w") as f:
    json.dump(totals, f, indent=2)
print("Saved cohort_counts.json")
