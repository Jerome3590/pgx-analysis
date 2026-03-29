"""
Z-code proportion broken down by n_event_bin × case/control.
Tests whether Z-code accumulation is density-driven (post-hoc artifact)
or genuine protective monitoring signal.

Z-code categories:
  MONITORING  : Z00-Z29  (exams, screenings, immunizations — truly preventive)
  DRUG_STATUS : Z79      (long-term drug use — correlates with polypharmacy)
  HISTORY     : Z80-Z99  (personal/family history, device presence — often post-hoc)
"""
import boto3, io
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["65-74", "75-84", "85-114"]
ICD_PRIMARY = "primary_icd_diagnosis_code"


def classify_z(code):
    if not isinstance(code, str):
        return None
    c = code.upper().strip()
    if not c.startswith("Z"):
        return None
    try:
        num = int(c[1:4])
    except (ValueError, IndexError):
        return None
    if   0  <= num <= 29: return "MONITORING"    # Z00-Z29
    elif 79 <= num <= 79: return "DRUG_STATUS"   # Z79
    elif 80 <= num <= 99: return "HISTORY"       # Z80-Z99
    else:                 return "OTHER_Z"       # Z30-Z78 excl Z79


# ── Load medical events ───────────────────────────────────────────────────────
print("Loading model_events (medical events only) …")
frames = []
for band in BANDS:
    key  = (f"gold/cohorts_model_data/cohort_name=non_opioid_ed/"
            f"age_band={band}/model_events.parquet")
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    df   = pd.read_parquet(io.BytesIO(data),
                           columns=["mi_person_key", "target", ICD_PRIMARY])
    df   = df[df[ICD_PRIMARY].notna()].copy()
    df["band"] = band
    frames.append(df)

med = pd.concat(frames, ignore_index=True)
med["icd"] = med[ICD_PRIMARY].str.upper().str.strip()
med["z_cat"] = med["icd"].apply(classify_z)
print(f"Medical events: {len(med):,}")

# ── Load n_event_bin from final_features ─────────────────────────────────────
print("Loading n_event_bin from final_features …")
ff_frames = []
for band in BANDS:
    for split in ("model_train", "model_test"):
        key = (f"gold/final_model/non_opioid_ed/{band}/inputs/"
               f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            ff   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target",
                                            "n_events", "n_event_bin"])
            ff["band"] = band
            ff_frames.append(ff)
        except Exception:
            pass

ff_all = (pd.concat(ff_frames, ignore_index=True)
            .sort_values("target", ascending=False)
            .drop_duplicates("mi_person_key")
            [["mi_person_key", "n_event_bin", "n_events"]])

med2 = med.merge(ff_all, on="mi_person_key", how="left")
print(f"After bin join: {med2['n_event_bin'].notna().sum():,} events with bin info")

# ── Per-patient summary ───────────────────────────────────────────────────────
def agg_patient(g):
    total = len(g)
    return pd.Series({
        "total":       total,
        "n_z":         (g["z_cat"].notna()).sum(),
        "n_monitoring":(g["z_cat"] == "MONITORING").sum(),
        "n_drug":      (g["z_cat"] == "DRUG_STATUS").sum(),
        "n_history":   (g["z_cat"] == "HISTORY").sum(),
    })

pp = (med2.groupby(["mi_person_key", "target", "n_event_bin"])
          .apply(agg_patient)
          .reset_index())
pp["z_prop"]          = pp["n_z"]          / pp["total"].clip(lower=1)
pp["monitor_prop"]    = pp["n_monitoring"] / pp["total"].clip(lower=1)
pp["drug_prop"]       = pp["n_drug"]       / pp["total"].clip(lower=1)
pp["history_prop"]    = pp["n_history"]    / pp["total"].clip(lower=1)

BINS_ORDER = ["low", "medium", "high", "extreme"]

# ── Table 1: Z-code proportion by bin × case/control ─────────────────────────
print("\n" + "=" * 72)
print("Z-CODE PROPORTION BY BIN (median, IQR)  — cases vs controls")
print("=" * 72)
print(f"{'Bin':10s}  {'Group':10s}  {'All-Z':16s}  "
      f"{'Monitor(Z00-29)':18s}  {'DrugRx(Z79)':14s}  {'Hist(Z80+)':14s}  n")
print("-" * 110)
for b in BINS_ORDER:
    for label, tgt in [("Cases", 1), ("Controls", 0)]:
        sub = pp[(pp["n_event_bin"] == b) & (pp["target"] == tgt)]
        if len(sub) == 0:
            continue
        def iqr(s):
            q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
            return f"{q50:.2f} ({q25:.2f}-{q75:.2f})"
        print(f"{b:10s}  {label:10s}  {iqr(sub['z_prop']):16s}  "
              f"{iqr(sub['monitor_prop']):18s}  "
              f"{iqr(sub['drug_prop']):14s}  "
              f"{iqr(sub['history_prop']):14s}  {len(sub):,}")

# ── Cross-bin trend for controls (is Z-prop density-driven?) ──────────────────
print("\n" + "=" * 72)
print("MEDIAN Z-PROP ACROSS BINS — controls only (density-driven test)")
print("=" * 72)
ctrl = pp[pp["target"] == 0]
for b in BINS_ORDER:
    sub = ctrl[ctrl["n_event_bin"] == b]
    if len(sub) == 0:
        continue
    print(f"  {b:10s}: all-Z={sub['z_prop'].median():.3f}  "
          f"monitor={sub['monitor_prop'].median():.3f}  "
          f"drug={sub['drug_prop'].median():.3f}  "
          f"history={sub['history_prop'].median():.3f}  "
          f"n={len(sub):,}")

# ── Specific Z-code frequency breakdown within extreme bin ────────────────────
print("\n" + "=" * 72)
print("TOP 20 Z-CODES IN EXTREME BIN (cases vs controls)")
print("=" * 72)
ext_med = med2[med2["n_event_bin"] == "extreme"]
ext_z   = ext_med[ext_med["z_cat"].notna()]
for label, tgt in [("Cases", 1), ("Controls", 0)]:
    sub = ext_z[ext_z["target"] == tgt]
    top = sub["icd"].value_counts().head(20)
    total_events = len(ext_med[ext_med["target"] == tgt])
    print(f"\n  {label} (total events in extreme bin: {total_events:,}):")
    for code, cnt in top.items():
        cat = classify_z(code)
        pct = cnt / total_events * 100
        print(f"    {code:8s}  {cnt:6,} ({pct:4.1f}%)  [{cat}]")

# ── Key finding summary ───────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("KEY FINDINGS")
print("=" * 72)
for b in BINS_ORDER:
    ctrl_sub = pp[(pp["n_event_bin"] == b) & (pp["target"] == 0)]
    case_sub = pp[(pp["n_event_bin"] == b) & (pp["target"] == 1)]
    if len(ctrl_sub) == 0:
        continue
    ctrl_mon = ctrl_sub["monitor_prop"].median()
    ctrl_hist = ctrl_sub["history_prop"].median()
    case_mon = case_sub["monitor_prop"].median() if len(case_sub) > 0 else float("nan")
    case_hist = case_sub["history_prop"].median() if len(case_sub) > 0 else float("nan")
    print(f"  {b:10s}  ctrl-monitor={ctrl_mon:.3f}  ctrl-history={ctrl_hist:.3f}  "
          f"case-monitor={case_mon:.3f}  case-history={case_hist:.3f}")

print("\nInterpretation:")
print("  If history_prop RISES with bin → Z80+ codes are density artifacts")
print("  If monitor_prop FALLS with bin → high-density patients get less routine care")
print("  Cross-bin case vs control difference in monitor_prop = true protection signal")
