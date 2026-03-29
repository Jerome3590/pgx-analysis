"""Diagnose Z-code proportion calculation — medical events only vs all events."""
import boto3, io
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
ICD_COLS = [
    "primary_icd_diagnosis_code", "two_icd_diagnosis_code",
    "three_icd_diagnosis_code",   "four_icd_diagnosis_code",
]

key = ("gold/cohorts_model_data/cohort_name=non_opioid_ed/"
       "age_band=65-74/model_events.parquet")
data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
df = pd.read_parquet(
    io.BytesIO(data),
    columns=["mi_person_key", "target", "event_date",
             "first_o11_p_date", "drug_name"] + ICD_COLS
)

has_icd  = df["primary_icd_diagnosis_code"].notna()
has_drug = df["drug_name"].notna()
print(f"Total rows      : {len(df):,}")
print(f"Medical (ICD)   : {has_icd.sum():,}  ({has_icd.mean()*100:.1f}%)")
print(f"Drug dispensing : {has_drug.sum():,}  ({has_drug.mean()*100:.1f}%)")
print(f"Both null       : {(~has_icd & ~has_drug).sum():,}")
print()

# 30-day case window
df["event_date"]      = pd.to_datetime(df["event_date"])
df["first_o11_p_date"] = pd.to_datetime(df["first_o11_p_date"])
cases = df[df["target"] == 1].copy()
cases["days_before"] = (cases["first_o11_p_date"] - cases["event_date"]).dt.days
win = cases[(cases["days_before"] > 0) & (cases["days_before"] <= 30)]

print(f"Case 30-day events: {len(win):,}")
n_med  = win["primary_icd_diagnosis_code"].notna().sum()
n_drug = win["drug_name"].notna().sum()
print(f"  Medical (ICD) : {n_med:,}")
print(f"  Drug          : {n_drug:,}")

any_z = win["primary_icd_diagnosis_code"].str.startswith("Z", na=False)
print(f"  Z-codes in medical ICD: {any_z.sum():,}")

# Per-patient z_prop using medical-only denominator
med = win[win["primary_icd_diagnosis_code"].notna()].copy()
med["is_z"] = med["primary_icd_diagnosis_code"].str.startswith("Z", na=False).astype(int)
pp = (med.groupby("mi_person_key")
        .agg(total=("is_z", "count"), z=("is_z", "sum"))
        .reset_index())
pp["z_prop"] = pp["z"] / pp["total"]

n_pts = len(pp)
med50 = pp["z_prop"].median()
q25   = pp["z_prop"].quantile(0.25)
q75   = pp["z_prop"].quantile(0.75)
print(f"\nCases with medical events in 30-day window: {n_pts:,}")
print(f"  z_prop median={med50:.3f}  IQR={q25:.3f}-{q75:.3f}")
print(f"  z_prop > 0: {(pp['z_prop'] > 0).sum():,} ({(pp['z_prop'] > 0).mean()*100:.1f}%)")
