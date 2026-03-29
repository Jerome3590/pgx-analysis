"""
Compute drug count (item_drug_* columns) and n_events from final_features.parquet test sets.
"""
import boto3, io, pandas as pd
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

configs = [
    ("opioid_ed",    ["13-24", "25-44", "45-54", "55-64"]),
    ("non_opioid_ed", ["65-74", "75-84", "85-114"]),
]


def iqr(s, fmt=".0f"):
    s = s.dropna()
    return f"{s.median():{fmt}} ({s.quantile(0.25):{fmt}}\u2013{s.quantile(0.75):{fmt}})"


for cohort, bands in configs:
    print(f"\n=== {cohort} ===")
    for band in bands:
        key = f"gold/final_model/{cohort}/{band}/inputs/model_test/final_features.parquet"
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            df = pd.read_parquet(io.BytesIO(data))

            drug_cols = [c for c in df.columns if c.startswith("item_drug_")]
            df["drug_count"] = df[drug_cols].sum(axis=1)

            tgt = next((c for c in ("target", "is_target_case") if c in df.columns), None)
            if tgt is None:
                print(f"  {band}: no target col, cols={list(df.columns[:8])}")
                continue

            cases = df[df[tgt] == 1]
            ctrls = df[df[tgt] == 0]
            print(f"  {band}:")
            print(f"    N: cases={len(cases):,}, controls={len(ctrls):,}")
            print(f"    drug_count cases: {iqr(cases['drug_count'])}")
            print(f"    drug_count ctrls: {iqr(ctrls['drug_count'])}")
            if "n_events" in df.columns:
                print(f"    n_events   cases: {iqr(cases['n_events'])}")
                print(f"    n_events   ctrls: {iqr(ctrls['n_events'])}")
        except Exception as e:
            print(f"  SKIP {band}: {e}")
