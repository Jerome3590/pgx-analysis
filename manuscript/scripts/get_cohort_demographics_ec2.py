"""
Run on EC2 (where cohort parquets are fully local).
Extracts age, sex, enrollment months, drug count, Z-code proportion
for CH_3 (opioid_ed) and CH_4 (non_opioid_ed) manuscript demographics tables.

Usage on EC2:
    python get_cohort_demographics_ec2.py
"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust to local EC2 path
COHORT_BASE = Path("/home/pgx3874/pgx-analysis/gold/cohorts")
# OR directly from S3 via awswrangler if available:
# import awswrangler as wr
# df = wr.s3.read_parquet("s3://pgxdatalake/gold/cohorts/cohort_name=.../")

def iqr(s, fmt=".0f"):
    return f"{s.median():{fmt}} ({s.quantile(.25):{fmt}}–{s.quantile(.75):{fmt}})"

def load_cohort(cohort, band, years=(2016, 2017, 2018, 2019)):
    frames = []
    for yr in years:
        p = COHORT_BASE / f"cohort_name={cohort}" / f"event_year={yr}" / f"age_band={band}" / "cohort.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            # Try S3
            try:
                import boto3, io
                s3 = boto3.client("s3")
                key = f"gold/cohorts/cohort_name={cohort}/event_year={yr}/age_band={band}/cohort.parquet"
                data = s3.get_object(Bucket="pgxdatalake", Key=key)["Body"].read()
                frames.append(pd.read_parquet(io.BytesIO(data)))
            except Exception:
                pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def patient_demographics(df, cohort, band):
    """
    Aggregate claim-level parquet to patient-level demographics.
    Columns expected: mi_person_key, event_date, age_imputed, member_gender, target (if present)
    """
    # Group by person — take first occurrence per patient
    grp = df.groupby("mi_person_key").agg(
        age=("age_imputed", "first"),
        gender=("member_gender", "first"),
        n_claims=("event_date", "count"),
        target=("target", "max") if "target" in df.columns else ("mi_person_key", "count"),
        first_date=("event_date", "min"),
        last_date=("event_date", "max"),
    ).reset_index()

    grp["enrollment_months"] = ((grp["last_date"] - grp["first_date"]).dt.days / 30.44).round()

    # Drug count from drug_name column (distinct drugs in 30-day pre-index window)
    # This requires index date alignment — approximated by counting distinct drug_name per patient
    if "drug_name" in df.columns:
        drug_ct = df.groupby("mi_person_key")["drug_name"].nunique().reset_index()
        drug_ct.columns = ["mi_person_key", "drug_count"]
        grp = grp.merge(drug_ct, on="mi_person_key", how="left")

    return grp


print("=" * 60)
print("CH_3: opioid_ed demographics")
print("=" * 60)
for band in ["13-24", "25-44", "45-54", "55-64"]:
    raw = load_cohort("opioid_ed", band)
    if raw is None:
        print(f"  SKIP {band}: no data found")
        continue
    pat = patient_demographics(raw, "opioid_ed", band)
    cases = pat[pat["target"] == 1] if "target" in pat.columns else pat
    ctrls = pat[pat["target"] == 0] if "target" in pat.columns else pd.DataFrame()
    print(f"\n  opioid_ed / {band}:")
    print(f"    Cases n={len(cases)}, Controls n={len(ctrls)}")
    if len(cases):
        print(f"    Age cases:   {iqr(cases['age'])}")
        print(f"    Age ctrls:   {iqr(ctrls['age']) if len(ctrls) else 'N/A'}")
        print(f"    Female cases: {(cases['gender']=='F').mean()*100:.1f}%")
        print(f"    Female ctrls: {(ctrls['gender']=='F').mean()*100:.1f}%" if len(ctrls) else "")
        print(f"    Enroll mo cases: {iqr(cases['enrollment_months'])}")
        print(f"    Enroll mo ctrls: {iqr(ctrls['enrollment_months']) if len(ctrls) else 'N/A'}")

print("\n" + "=" * 60)
print("CH_4: non_opioid_ed demographics")
print("=" * 60)
raw_geri = []
for band in ["65-74", "75-84", "85-114"]:
    raw = load_cohort("non_opioid_ed", band)
    if raw is not None:
        raw["_band"] = band
        raw_geri.append(raw)

if raw_geri:
    all_geri = pd.concat(raw_geri, ignore_index=True)
    pat = patient_demographics(all_geri, "non_opioid_ed", "65-114")
    cases = pat[pat["target"] == 1] if "target" in pat.columns else pat
    ctrls = pat[pat["target"] == 0] if "target" in pat.columns else pd.DataFrame()
    print(f"\n  non_opioid_ed / 65-114 (combined):")
    print(f"    Cases n={len(cases)}, Controls n={len(ctrls)}")
    if len(cases):
        print(f"    Age cases:   {iqr(cases['age'])}")
        print(f"    Age ctrls:   {iqr(ctrls['age']) if len(ctrls) else 'N/A'}")
        print(f"    Female cases: {(cases['gender']=='F').mean()*100:.1f}%")
        print(f"    Female ctrls: {(ctrls['gender']=='F').mean()*100:.1f}%" if len(ctrls) else "")
        if "drug_count" in pat.columns:
            print(f"    Drug count cases: {iqr(cases['drug_count'])}")
            print(f"    Drug count ctrls: {iqr(ctrls['drug_count']) if len(ctrls) else 'N/A'}")
