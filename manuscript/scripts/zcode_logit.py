"""
Z-code OR for CH_4 using model_events.parquet (non_opioid_ed).
Columns available: mi_person_key, target, event_date, first_o11_p_date,
  member_gender, age_band, primary_icd_diagnosis_code … ten_icd_diagnosis_code
"""
import boto3, io, warnings
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

ICD_COLS = [
    "primary_icd_diagnosis_code", "two_icd_diagnosis_code",
    "three_icd_diagnosis_code",   "four_icd_diagnosis_code",
    "five_icd_diagnosis_code",
]
NEED_COLS = ["mi_person_key", "target", "event_date", "first_o11_p_date",
             "member_gender", "age_band"] + ICD_COLS
BAND_MID  = {"65-74": 70, "75-84": 80, "85-114": 95}


def load_band(cohort, band):
    key = f"gold/cohorts_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet"
    print(f"  Loading {band}…", end=" ", flush=True)
    buf = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    tbl = pq.read_table(io.BytesIO(buf), columns=NEED_COLS)
    df  = tbl.to_pandas()
    df["_band"] = band
    print(f"{len(df):,} rows, cases={df['target'].sum():,}")
    return df


# ── Load all bands ────────────────────────────────────────────────────────────
print("Loading model_events.parquet for non_opioid_ed …")
frames = []
for band in ["65-74", "75-84", "85-114"]:
    try:
        frames.append(load_band("non_opioid_ed", band))
    except Exception as e:
        print(f"  SKIP {band}: {e}")

df = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows: {len(df):,}  |  unique patients: {df['mi_person_key'].nunique():,}")

# ── Z-code flag on every claim (full history) ────────────────────────────────
# first_o11_p_date is the first outpatient contact date — NOT a valid 30-day
# pre-index reference for controls (controls have no ED event).  Use full
# claim history instead to compute each patient's overall Z-code proportion.
df["is_zcode"] = df[ICD_COLS].apply(
    lambda r: int(any(isinstance(v, str) and v.startswith("Z") for v in r)),
    axis=1,
)

# ── Patient-level aggregation (all claims) ────────────────────────────────────
pat = df.groupby("mi_person_key").agg(
    z_sum   =("is_zcode", "sum"),
    n_claims=("is_zcode", "count"),
    target  =("target",   "max"),
    gender  =("member_gender", "first"),
    band    =("_band",    "first"),
).reset_index()

pat["z_prop"]  = pat["z_sum"] / pat["n_claims"].clip(1)
pat["female"]  = (pat["gender"] == "F").astype(int)
pat["age_mid"] = pat["band"].map(BAND_MID).fillna(80)

print(f"\nPatient summary:")
print(f"  Total patients: {len(pat):,}")
print(f"  Cases: {pat['target'].sum():,}, Controls: {(pat['target']==0).sum():,}")
print(f"\nZ-code proportion by case/control:")
print(pat.groupby("target")["z_prop"].describe().round(4))

# ── Z-code proportion IQR for table ──────────────────────────────────────────
def iqr(s):
    s = s.dropna()
    return f"{s.median():.3f} ({s.quantile(.25):.3f}\u2013{s.quantile(.75):.3f})"

cases = pat[pat["target"] == 1]
ctrls = pat[pat["target"] == 0]
print(f"\nZ-code proportion IQR:")
print(f"  Cases:    {iqr(cases['z_prop'])}")
print(f"  Controls: {iqr(ctrls['z_prop'])}")

# ── Quartile regression ───────────────────────────────────────────────────────
reg = pat[["target", "z_prop", "female", "age_mid"]].dropna().copy()
reg["z_prop_q"] = pd.qcut(reg["z_prop"], q=4, labels=[1, 2, 3, 4],
                           duplicates="drop").astype(float)

print(f"\nZ-prop quartile distribution (target 0/1):")
print(pd.crosstab(reg["z_prop_q"], reg["target"]))

model = smf.logit("target ~ z_prop_q + female + age_mid", data=reg).fit(disp=False)
print(model.summary2())

params = model.params
conf   = model.conf_int()
z_or   = np.exp(params["z_prop_q"])
z_lo   = np.exp(conf.loc["z_prop_q", 0])
z_hi   = np.exp(conf.loc["z_prop_q", 1])
z_p    = model.pvalues["z_prop_q"]

print(f"\n=== Z-code Quartile OR (per 1-quartile increase) ===")
print(f"  OR  = {z_or:.2f}")
print(f"  95% CI = {z_lo:.2f}–{z_hi:.2f}")
print(f"  p = {z_p:.4f}")
print(f"  Formatted: OR = {z_or:.2f} (95% CI {z_lo:.2f}–{z_hi:.2f})")

# Per-quartile (vs Q1 reference) dummy approach
reg["z_q2"] = (reg["z_prop_q"] == 2).astype(int)
reg["z_q3"] = (reg["z_prop_q"] == 3).astype(int)
reg["z_q4"] = (reg["z_prop_q"] == 4).astype(int)

model2 = smf.logit("target ~ z_q2 + z_q3 + z_q4 + female + age_mid", data=reg).fit(disp=False)
params2 = model2.params
conf2   = model2.conf_int()
print(f"\n=== Per-Quartile ORs (Q1 = reference) ===")
for q in ("z_q2", "z_q3", "z_q4"):
    or_q = np.exp(params2[q])
    lo_q = np.exp(conf2.loc[q, 0])
    hi_q = np.exp(conf2.loc[q, 1])
    p_q  = model2.pvalues[q]
    print(f"  Q{q[-1]} vs Q1: OR = {or_q:.2f} (95% CI {lo_q:.2f}–{hi_q:.2f}), p = {p_q:.4f}")
