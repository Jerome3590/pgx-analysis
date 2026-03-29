"""
Inventory Z-codes in non_opioid_ed model_events to separate:
  - Protective monitoring (Z00-Z13, Z23-Z28): routine exams, screenings, immunizations
  - Drug-related (Z79.x): long-term medication use
  - History/status (Z80-Z99): personal history, device presence — potentially post-hoc
  - Social factors (Z55-Z65): SDH codes
"""
import boto3, io
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["65-74", "75-84", "85-114"]
ICD_PRIMARY = "primary_icd_diagnosis_code"

all_frames = []
for band in BANDS:
    key  = (f"gold/cohorts_model_data/cohort_name=non_opioid_ed/"
            f"age_band={band}/model_events.parquet")
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    df   = pd.read_parquet(io.BytesIO(data),
                           columns=["mi_person_key", "target", ICD_PRIMARY])
    df   = df[df[ICD_PRIMARY].notna()].copy()
    df["band"] = band
    all_frames.append(df)

med = pd.concat(all_frames, ignore_index=True)
med["icd"] = med[ICD_PRIMARY].str.upper().str.strip()

# Extract Z-codes only
z_events = med[med["icd"].str.startswith("Z", na=False)].copy()
print(f"Total medical events:  {len(med):,}")
print(f"Z-code events:         {len(z_events):,}  ({len(z_events)/len(med)*100:.1f}%)")
print(f"  of which in cases:   {(z_events['target']==1).sum():,}")
print(f"  of which in controls:{(z_events['target']==0).sum():,}")
print()

# Top 30 Z-codes overall
print("=== TOP 30 Z-CODES (all patients) ===")
top = z_events["icd"].value_counts().head(30)
for code, cnt in top.items():
    # Classify
    num = int(code[1:4]) if code[1:4].isdigit() else 99
    if   0  <= num <= 13:  cat = "MONITORING (exam/screen)"
    elif 23 <= num <= 28:  cat = "MONITORING (immunization)"
    elif 29 <= num <= 29:  cat = "MONITORING (other preventive)"
    elif 41 <= num <= 53:  cat = "PROCEDURE encounter"
    elif 55 <= num <= 65:  cat = "SOCIAL FACTORS"
    elif 66 <= num <= 76:  cat = "ENCOUNTER (non-exam)"
    elif 79 <= num <= 79:  cat = "DRUG STATUS (Z79)"
    elif 80 <= num <= 99:  cat = "HISTORY/STATUS (post-hoc?)"
    else:                  cat = "OTHER"
    print(f"  {code:8s}  {cnt:8,}  [{cat}]")

print()

# Z-code category summary
print("=== Z-CODE CATEGORY BREAKDOWN ===")
def classify_z(code):
    if not isinstance(code, str):
        return "OTHER"
    c = code.upper().strip()
    if not c.startswith("Z") or not c[1:4].replace("0","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6","").replace("7","").replace("8","").replace("9","") == "":
        return "OTHER"
    try:
        num = int(c[1:4])
    except ValueError:
        return "OTHER"
    if   0  <= num <= 13: return "Z00-Z13 MONITORING"
    elif 23 <= num <= 29: return "Z23-Z29 IMMUNIZATION/PREVENTIVE"
    elif 41 <= num <= 53: return "Z41-Z53 PROCEDURE"
    elif 55 <= num <= 65: return "Z55-Z65 SOCIAL"
    elif 66 <= num <= 76: return "Z66-Z76 ENCOUNTER OTHER"
    elif 79 <= num <= 79: return "Z79 LONG-TERM DRUG USE"
    elif 80 <= num <= 99: return "Z80-Z99 HISTORY/STATUS"
    else:                 return "OTHER Z"

z_events["category"] = z_events["icd"].apply(classify_z)
cat_summary = z_events.groupby("category").agg(
    n_events=("icd", "count"),
    n_cases=("target", "sum"),
    n_controls=("target", lambda x: (x == 0).sum())
).reset_index().sort_values("n_events", ascending=False)
for _, row in cat_summary.iterrows():
    print(f"  {row['category']:35s} events={row['n_events']:8,} "
          f"cases={row['n_cases']:5,}  ctrl={row['n_controls']:8,}")

print()
# Specifically flag Z79 (long-term drug use) — this may co-occur WITH the ADE
z79 = z_events[z_events["icd"].str.startswith("Z79", na=False)]
print(f"=== Z79 LONG-TERM DRUG USE CODES ===  ({len(z79):,} events)")
print(z79["icd"].value_counts().head(20))

print()
# Protective monitoring only (Z00-Z13, Z23-Z29)
monitoring = z_events[z_events["category"].isin(
    ["Z00-Z13 MONITORING", "Z23-Z29 IMMUNIZATION/PREVENTIVE"])]
print(f"=== PROTECTIVE MONITORING Z-CODES (Z00-Z13, Z23-Z29) ===")
print(f"  Total: {len(monitoring):,}  "
      f"({len(monitoring)/len(med)*100:.1f}% of all medical events)")
print(f"  In cases: {(monitoring['target']==1).sum():,}")
print(f"  In controls: {(monitoring['target']==0).sum():,}")

# Per-patient monitoring proportion (excluding Z79, Z80-Z99)
print()
print("=== PER-PATIENT MONITORING PROPORTION (protective only) ===")
med["is_monitoring"] = med["icd"].str.startswith("Z", na=False)
# Exclude Z79 and Z80-Z99
def is_protective(code):
    if not isinstance(code, str):
        return False
    c = code.upper().strip()
    if not c.startswith("Z"):
        return False
    try:
        num = int(c[1:4])
    except (ValueError, IndexError):
        return False
    return 0 <= num <= 29  # Z00-Z29 only (excl. Z79, Z80-Z99)

med["is_protective_z"] = med["icd"].apply(is_protective).astype(int)
med["is_drug_status"]  = med["icd"].str.startswith("Z79", na=False).astype(int)
med["is_history"]      = med["icd"].apply(
    lambda c: isinstance(c, str) and c.upper().startswith("Z") and
    c[1:4].isdigit() and int(c[1:4]) >= 80).astype(int)

pp = (med.groupby(["mi_person_key", "target"])
         .agg(total=("icd", "count"),
              protective_z=("is_protective_z", "sum"),
              drug_status_z=("is_drug_status", "sum"),
              history_z=("is_history", "sum"))
         .reset_index())
pp["z_protective_prop"] = pp["protective_z"] / pp["total"].clip(lower=1)
pp["z_drugrx_prop"]     = pp["drug_status_z"] / pp["total"].clip(lower=1)
pp["z_history_prop"]    = pp["history_z"] / pp["total"].clip(lower=1)

for metric in ["z_protective_prop", "z_drugrx_prop", "z_history_prop"]:
    print(f"\n  {metric}:")
    for label, tgt in [("Cases", 1), ("Controls", 0)]:
        g = pp[pp["target"] == tgt][metric]
        q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
        print(f"    {label}: median={q50:.3f} ({q25:.3f}–{q75:.3f})  n={len(g):,}")
