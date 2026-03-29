"""
Compute demographics from gold/cohorts/{cohort}/{year}/{band}/cohort.parquet.
Columns: mi_person_key, age_imputed, member_gender, event_date, drug_name,
         is_target_case (1=case, 0=control), days_to_target_event
"""
import boto3, io, pandas as pd, numpy as np
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
YEARS  = [2016, 2017, 2018, 2019]


def load_cohort_parquet(cohort, band, years=YEARS):
    frames = []
    for yr in years:
        key = f"gold/cohorts/cohort_name={cohort}/event_year={yr}/age_band={band}/cohort.parquet"
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            frames.append(pd.read_parquet(io.BytesIO(data)))
        except Exception as e:
            print(f"    SKIP {cohort}/{band}/{yr}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else None


def iqr(s, fmt=".0f"):
    s = s.dropna()
    return f"{s.median():{fmt}} ({s.quantile(0.25):{fmt}}\u2013{s.quantile(0.75):{fmt}})"


def patient_level(df):
    """Aggregate claim-level cohort.parquet to one row per patient."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Drug count in 30-day pre-index window (days_to_target_event in [-30, 0])
    drugs30 = df[
        df["drug_name"].notna() &
        (df["days_to_target_event"] >= -30) &
        (df["days_to_target_event"] <= 0)
    ]
    dc = drugs30.groupby("mi_person_key")["drug_name"].nunique().rename("drug_count_30d")

    pat = df.groupby("mi_person_key").agg(
        age      =("age_imputed", "first"),
        gender   =("member_gender", "first"),
        is_case  =("is_target_case", "max"),
        min_date =("event_date", "min"),
        max_date =("event_date", "max"),
    ).join(dc, how="left")
    pat["drug_count_30d"]  = pat["drug_count_30d"].fillna(0)
    pat["enroll_months"]   = ((pat["max_date"] - pat["min_date"]).dt.days / 30.44).round()
    return pat.reset_index()


def print_stats(pat, label):
    cases = pat[pat["is_case"] == 1]
    ctrls = pat[pat["is_case"] == 0]
    print(f"\n{label}")
    print(f"  N: cases={len(cases):,}, controls={len(ctrls):,}")
    pct_f_c = (cases["gender"] == "F").mean() * 100
    pct_f_t = (ctrls["gender"] == "F").mean() * 100
    print(f"  Female: cases={pct_f_c:.1f}%, controls={pct_f_t:.1f}%")
    print(f"  Age    (cases): {iqr(cases['age'])}")
    print(f"  Age    (ctrls): {iqr(ctrls['age'])}")
    print(f"  Enroll months (cases): {iqr(cases['enroll_months'])}")
    print(f"  Enroll months (ctrls): {iqr(ctrls['enroll_months'])}")
    print(f"  Drug ct 30d (cases): {iqr(cases['drug_count_30d'])}")
    print(f"  Drug ct 30d (ctrls): {iqr(ctrls['drug_count_30d'])}")


# ── CH_4: non_opioid_ed ───────────────────────────────────────────────────────
print("=" * 60)
print("CH_4: non_opioid_ed demographics")
print("=" * 60)
geri_frames = []
for band in ["65-74", "75-84", "85-114"]:
    print(f"  Loading non_opioid_ed/{band}...")
    df = load_cohort_parquet("non_opioid_ed", band)
    if df is not None:
        df["_band"] = band
        geri_frames.append(df)

if geri_frames:
    geri = pd.concat(geri_frames, ignore_index=True)
    pat_geri = patient_level(geri)
    print_stats(pat_geri, "non_opioid_ed COMBINED (65-114)")
    for band in ["65-74", "75-84", "85-114"]:
        ids = geri[geri["_band"] == band]["mi_person_key"].unique()
        print_stats(pat_geri[pat_geri["mi_person_key"].isin(ids)], f"  {band}")

# ── CH_3: opioid_ed ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CH_3: opioid_ed demographics")
print("=" * 60)
opi_frames = []
for band in ["13-24", "25-44", "45-54", "55-64"]:
    print(f"  Loading opioid_ed/{band}...")
    df = load_cohort_parquet("opioid_ed", band)
    if df is not None:
        df["_band"] = band
        opi_frames.append(df)

if opi_frames:
    opi = pd.concat(opi_frames, ignore_index=True)
    pat_opi = patient_level(opi)
    print_stats(pat_opi, "opioid_ed COMBINED (13-64)")
    for band in ["13-24", "25-44", "45-54", "55-64"]:
        ids = opi[opi["_band"] == band]["mi_person_key"].unique()
        print_stats(pat_opi[pat_opi["mi_person_key"].isin(ids)], f"  {band}")
