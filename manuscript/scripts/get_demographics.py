"""
Compute demographic IQR statistics from cohort parquets for CH_3 and CH_4.
Uses gold/cohorts/cohort_name={}/event_year=2019/age_band={}/cohort.parquet (holdout year).
"""
import boto3, io, pandas as pd, numpy as np
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"


def load_cohort(cohort, band, year=2019):
    key = f"gold/cohorts/cohort_name={cohort}/event_year={year}/age_band={band}/cohort.parquet"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(data))
    except Exception as e:
        print(f"  MISS {cohort}/{band}/{year}: {e}")
        return None


def iqr_str(series, fmt=".0f"):
    m  = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{m:{fmt}} ({q1:{fmt}}–{q3:{fmt}})"


# Peek at columns first
print("=== Cohort parquet columns ===")
for cohort, band in [("opioid_ed","25-44"), ("non_opioid_ed","65-74")]:
    df = load_cohort(cohort, band)
    if df is not None:
        print(f"\n{cohort}/{band}: shape={df.shape}")
        print("  columns:", list(df.columns)[:30])
        print("  dtypes:", df.dtypes.head(15).to_string())
        break

print("\n\n=== CH_3: opioid_ed demographics (2019 holdout) ===")
OPIOID_BANDS = ["13-24","25-44","45-54","55-64"]
opi_all = []
for band in OPIOID_BANDS:
    df = load_cohort("opioid_ed", band)
    if df is not None:
        opi_all.append(df)
        print(f"\n  opioid_ed/{band}: n={len(df)}, cols={list(df.columns[:10])}")

if opi_all:
    opi = pd.concat(opi_all, ignore_index=True)
    print(f"\n  Combined opioid_ed 2019: n={len(opi)}")
    # Check for target/case column
    case_col = next((c for c in ("target","case","is_case","ade_flag","oud_flag") if c in opi.columns), None)
    print(f"  Case column: {case_col}")
    age_col  = next((c for c in ("age","age_at_index","member_age","age_years") if c in opi.columns), None)
    sex_col  = next((c for c in ("sex","gender","member_gender","female") if c in opi.columns), None)
    enroll_col = next((c for c in ("enrollment_months","months_enrolled","enroll_months") if c in opi.columns), None)
    drug_col = next((c for c in ("drug_count","n_drugs","drug_count_30d") if c in opi.columns), None)
    print(f"  age={age_col}, sex={sex_col}, enroll={enroll_col}, drug={drug_col}")
    if age_col and case_col:
        cases = opi[opi[case_col]==1]
        ctrls = opi[opi[case_col]==0]
        print(f"  Cases: {len(cases)}, Controls: {len(ctrls)}")
        print(f"  Age (cases): {iqr_str(cases[age_col])}")
        print(f"  Age (ctrls): {iqr_str(ctrls[age_col])}")

print("\n\n=== CH_4: non_opioid_ed demographics (2019 holdout) ===")
GERI_BANDS = ["65-74","75-84","85-114"]
geri_all = []
for band in GERI_BANDS:
    df = load_cohort("non_opioid_ed", band)
    if df is not None:
        geri_all.append(df)

if geri_all:
    geri = pd.concat(geri_all, ignore_index=True)
    print(f"  Combined non_opioid_ed 2019: n={len(geri)}")
    case_col = next((c for c in ("target","case","is_case","ade_flag") if c in geri.columns), None)
    age_col  = next((c for c in ("age","age_at_index","member_age","age_years") if c in geri.columns), None)
    sex_col  = next((c for c in ("sex","gender","member_gender","female") if c in geri.columns), None)
    drug_col = next((c for c in ("drug_count","n_drugs","drug_count_30d","n_drugs_30d") if c in geri.columns), None)
    zcode_col= next((c for c in ("z_code_prop","zcode_prop","z_code_pct","routine_pct","z_prop") if c in geri.columns), None)
    print(f"  case={case_col}, age={age_col}, sex={sex_col}, drug={drug_col}, zcode={zcode_col}")
    if case_col:
        cases = geri[geri[case_col]==1]
        ctrls = geri[geri[case_col]==0]
        print(f"  Cases: {len(cases)}, Controls: {len(ctrls)}")
        if age_col:
            print(f"  Age (cases): {iqr_str(cases[age_col])}")
            print(f"  Age (ctrls): {iqr_str(ctrls[age_col])}")
        if drug_col:
            print(f"  Drug count 30d (cases): {iqr_str(cases[drug_col])}")
            print(f"  Drug count 30d (ctrls): {iqr_str(ctrls[drug_col])}")
        if zcode_col:
            print(f"  Z-code prop (cases): {iqr_str(cases[zcode_col], '.2f')}")
            print(f"  Z-code prop (ctrls): {iqr_str(ctrls[zcode_col], '.2f')}")
        if sex_col:
            print(f"  Female % (cases): {cases[sex_col].mean()*100:.1f}%")
            print(f"  Female % (ctrls): {ctrls[sex_col].mean()*100:.1f}%")
    print(f"\n  All columns: {list(geri.columns)}")
