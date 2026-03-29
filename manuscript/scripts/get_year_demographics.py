"""
Get per-year case/control counts for CH_3 (opioid_ed) index-year rows
and 2019-only demographics for CH_4 (non_opioid_ed).
Also computes Z-code proportion for CH_4.
"""
import boto3, io, pandas as pd
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"


def load_year(cohort, band, year):
    key = f"gold/cohorts/cohort_name={cohort}/event_year={year}/age_band={band}/cohort.parquet"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(data))
    except Exception:
        return None


def iqr(s, fmt=".0f"):
    s = s.dropna()
    return f"{s.median():{fmt}} ({s.quantile(0.25):{fmt}}\u2013{s.quantile(0.75):{fmt}})"


# ── CH_3: Per-year case counts for opioid_ed ─────────────────────────────────
print("=" * 60)
print("CH_3: opioid_ed — per index year (2016–2018 training set)")
print("=" * 60)
opi_bands = ["13-24", "25-44", "45-54", "55-64"]
year_totals = {}
for yr in [2016, 2017, 2018]:
    yr_frames = []
    for band in opi_bands:
        df = load_year("opioid_ed", band, yr)
        if df is not None:
            yr_frames.append(df)
    if yr_frames:
        combined = pd.concat(yr_frames, ignore_index=True)
        pat = combined.groupby("mi_person_key").agg(is_case=("is_target_case", "max")).reset_index()
        n_cases = (pat["is_case"] == 1).sum()
        n_ctrl  = (pat["is_case"] == 0).sum()
        year_totals[yr] = {"cases": n_cases, "ctrls": n_ctrl}
        print(f"  {yr}: cases={n_cases:,}, controls={n_ctrl:,}")

total_cases = sum(v["cases"] for v in year_totals.values())
total_ctrls = sum(v["ctrls"] for v in year_totals.values())
print(f"\n  Totals (2016–2018): cases={total_cases:,}, controls={total_ctrls:,}")
print("  Percent breakdown:")
for yr, v in year_totals.items():
    pct_c = v["cases"] / total_cases * 100 if total_cases else 0
    pct_t = v["ctrls"] / total_ctrls * 100 if total_ctrls else 0
    print(f"    {yr}: cases={v['cases']:,} ({pct_c:.1f}%), controls={v['ctrls']:,} ({pct_t:.1f}%)")


# ── CH_4: 2019-only demographics for non_opioid_ed ───────────────────────────
print("\n" + "=" * 60)
print("CH_4: non_opioid_ed — 2019 holdout demographics")
print("=" * 60)
geri_frames_2019 = []
for band in ["65-74", "75-84", "85-114"]:
    df = load_year("non_opioid_ed", band, 2019)
    if df is not None:
        df["_band"] = band
        geri_frames_2019.append(df)

if geri_frames_2019:
    geri = pd.concat(geri_frames_2019, ignore_index=True)
    geri["event_date"] = pd.to_datetime(geri["event_date"])

    # Z-code proportion: claims with primary_icd starting with Z / total claims per patient
    geri["is_zcode"] = geri["primary_icd_diagnosis_code"].fillna("").str.startswith("Z").astype(int)

    # Drug count in 30-day pre-index window
    drugs30 = geri[
        geri["drug_name"].notna() &
        (geri["days_to_target_event"] >= -30) &
        (geri["days_to_target_event"] <= 0)
    ]
    dc = drugs30.groupby("mi_person_key")["drug_name"].nunique().rename("drug_count_30d")

    # Z-code proportion per patient (all claims in window)
    window = geri[(geri["days_to_target_event"] >= -30) & (geri["days_to_target_event"] <= 0)]
    zc = window.groupby("mi_person_key").agg(
        z_sum=("is_zcode", "sum"),
        total=("is_zcode", "count")
    )
    zc["z_prop"] = zc["z_sum"] / zc["total"]

    pat = geri.groupby("mi_person_key").agg(
        age     =("age_imputed", "first"),
        gender  =("member_gender", "first"),
        is_case =("is_target_case", "max"),
    ).join(dc, how="left").join(zc[["z_prop"]], how="left")
    pat["drug_count_30d"] = pat["drug_count_30d"].fillna(0)

    for label, mask in [("CASES", pat["is_case"] == 1), ("CONTROLS", pat["is_case"] == 0)]:
        sub = pat[mask]
        print(f"\n  {label} (n={len(sub):,}):")
        pct_f = (sub["gender"] == "F").mean() * 100
        print(f"    Female: {pct_f:.1f}%  (n={int(len(sub)*pct_f/100):,})")
        print(f"    Age: {iqr(sub['age'])}")
        print(f"    Drug count 30d: {iqr(sub['drug_count_30d'])}")
        print(f"    Z-code proportion: {iqr(sub['z_prop'], fmt='.3f')}")

    print("\n  Per-band (2019 only):")
    for band in ["65-74", "75-84", "85-114"]:
        ids = geri[geri["_band"] == band]["mi_person_key"].unique()
        sub = pat[pat.index.isin(ids)]
        cases = sub[sub["is_case"] == 1]
        ctrls = sub[sub["is_case"] == 0]
        print(f"    {band}: cases={len(cases)}, ctrls={len(ctrls)}, "
              f"age_cases={iqr(cases['age'])}, female_cases={((cases['gender']=='F').mean()*100):.1f}%")
