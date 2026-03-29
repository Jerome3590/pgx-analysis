"""
Inspect model_events.parquet for non_opioid_ed to find Z-code and demographic columns.
Then compute Z-code OR via logistic regression.
"""
import boto3, io, re
import pandas as pd
import numpy as np

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

# Try both slash and underscore age band formats
for band_key in ["65_74", "65-74"]:
    key = f"gold/cohorts_model_data/cohort_name=non_opioid_ed/age_band={band_key}/model_events.parquet"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        df = pd.read_parquet(io.BytesIO(data))
        print(f"Found at: {key}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nis_target_case dist:\n{df['is_target_case'].value_counts()}")
        icd_cols = [c for c in df.columns if "icd" in c.lower() or "diag" in c.lower()]
        print(f"\nICD/Diag cols: {icd_cols}")
        if icd_cols:
            # Sample a few values from first ICD col
            print(f"\nSample {icd_cols[0]} values:\n{df[icd_cols[0]].dropna().head(10).tolist()}")
        break
    except Exception as e:
        print(f"  MISS {band_key}: {e}")
