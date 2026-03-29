"""
Test multiple span thresholds to find [XX,XXX]-sized clusters for CH_3.
Loads all four opioid_ed bands once, then sweeps thresholds.
"""
import boto3, io
import pandas as pd
import numpy as np

s3     = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["13-24", "25-44", "45-54", "55-64"]
DATE_COLS = {"13-24": "current_event_date"}  # 13-24 uses different col name

all_spans = []

for band in BANDS:
    ab = band.replace("-", "_")
    date_col = DATE_COLS.get(band, "event_date")
    print(f"Loading {band} …", end=" ", flush=True)

    # Case IDs
    cases = set()
    for split in ["model_test", "model_train"]:
        key  = f"gold/final_model/opioid_ed/{band}/inputs/{split}/final_features.parquet"
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        fp   = pd.read_parquet(io.BytesIO(data), columns=["mi_person_key", "target"])
        cases |= set(fp.loc[fp["target"] == 1, "mi_person_key"].unique())

    # Event intervals
    key2 = f"gold/dtw_filter/opioid_ed/{band}/event_intervals_opioid_ed_{ab}.parquet"
    data2 = s3.get_object(Bucket=BUCKET, Key=key2)["Body"].read()
    ei    = pd.read_parquet(io.BytesIO(data2),
                            columns=["mi_person_key", date_col])
    ei = ei[ei["mi_person_key"].isin(cases)].copy()
    ei[date_col] = pd.to_datetime(ei[date_col])

    span = (
        ei.groupby("mi_person_key")[date_col]
        .agg(first="min", last="max")
        .assign(span_days=lambda d: (d["last"] - d["first"]).dt.days)
        .reset_index()
    )
    all_spans.append(span[["mi_person_key", "span_days"]])
    print(f"{len(span):,} cases")

combined = pd.concat(all_spans, ignore_index=True)
TOTAL_CASES = 26710
n_dtw = len(combined)
print(f"\nDTW-matched cases: {n_dtw:,} / {TOTAL_CASES:,}  "
      f"({n_dtw/TOTAL_CASES*100:.1f}%)")
print(f"span_days: p10={combined['span_days'].quantile(.10):.0f}  "
      f"p25={combined['span_days'].quantile(.25):.0f}  "
      f"p50={combined['span_days'].quantile(.50):.0f}  "
      f"p75={combined['span_days'].quantile(.75):.0f}  "
      f"p90={combined['span_days'].quantile(.90):.0f}")

# Sweep thresholds
print("\nThreshold sweep (applied to full 26,710 by extrapolation):")
for months in [3, 6, 9, 12, 15, 18]:
    thr = months * 30.4
    r = combined[combined["span_days"] <  thr]
    c = combined[combined["span_days"] >= thr]
    pct_r = len(r) / n_dtw
    pct_c = len(c) / n_dtw
    n_r_extrap = round(pct_r * TOTAL_CASES)
    n_c_extrap = TOTAL_CASES - n_r_extrap
    print(f"  {months:2d} mo: Rapid={len(r):5,} ({pct_r*100:.0f}%) "
          f"→ extrap {n_r_extrap:,}  |  "
          f"Chronic={len(c):5,} ({pct_c*100:.0f}%) "
          f"→ extrap {n_c_extrap:,}")
