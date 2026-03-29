"""DTW cluster sizes for 13-24 band (uses current_event_date column)."""
import boto3, io
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

cases = set()
for split in ["model_test", "model_train"]:
    key = f"gold/final_model/opioid_ed/13-24/inputs/{split}/final_features.parquet"
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    fp   = pd.read_parquet(io.BytesIO(data), columns=["mi_person_key", "target"])
    ids  = set(fp.loc[fp["target"] == 1, "mi_person_key"].unique())
    cases |= ids
    print(f"  {split}: {len(ids):,} cases")
print(f"  Total cases: {len(cases):,}")

key2 = "gold/dtw_filter/opioid_ed/13-24/event_intervals_opioid_ed_13_24.parquet"
data2 = s3.get_object(Bucket=BUCKET, Key=key2)["Body"].read()
ei    = pd.read_parquet(io.BytesIO(data2))
print("Columns:", list(ei.columns))

date_col = "current_event_date" if "current_event_date" in ei.columns else "event_date"
ei_c = ei[ei["mi_person_key"].isin(cases)].copy()
ei_c[date_col] = pd.to_datetime(ei_c[date_col])

span = (
    ei_c.groupby("mi_person_key")[date_col]
    .agg(first="min", last="max")
    .assign(span_days=lambda d: (d["last"] - d["first"]).dt.days)
    .reset_index()
)

threshold = 6 * 30.4
rapid   = span[span["span_days"] <  threshold]
chronic = span[span["span_days"] >= threshold]
n = len(span)

print(f"Rapid-Onset  (<6 mo): n={len(rapid):,}  ({len(rapid)/n*100:.1f}%)  "
      f"mean={rapid['span_days'].mean()/30.4:.1f} mo")
print(f"Chronic-Escal(>=6mo): n={len(chronic):,}  ({len(chronic)/n*100:.1f}%)  "
      f"mean={chronic['span_days'].mean()/30.4:.1f} mo")

# Combined totals (add to prior bands: 25-44+45-54+55-64)
n_rapid_prior   = 2078
n_chronic_prior = 18572
total_rapid   = n_rapid_prior   + len(rapid)
total_chronic = n_chronic_prior + len(chronic)
grand = total_rapid + total_chronic
print(f"\nFINAL ALL BANDS:")
print(f"  Cluster 1 Rapid-Onset:        n={total_rapid:,}  ({total_rapid/grand*100:.0f}%)")
print(f"  Cluster 2 Chronic-Escalation: n={total_chronic:,}  ({total_chronic/grand*100:.0f}%)")
print(f"  Grand total: {grand:,}")
