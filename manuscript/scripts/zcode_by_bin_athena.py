"""
Z-code (protective monitoring) proportion by n_event_bin x case/control.

Hypothesis:
  - Low-bin z_prop=0: never visit hospital, truly unmonitored -> high ADE risk
  - Low-medium bin z_prop>0: routine monitoring present -> lowest ADE risk
  - High-extreme bin: protective Z-codes diluted by procedure/treatment codes
    -> monitoring proportion drops even as total Z-codes rise

Analysis:
  1. Athena: protective Z-codes (Z00-Z13, Z23-Z28) proportion per patient (full history)
  2. Join n_event_bin from final_features
  3. Within each bin: compare cases vs controls z_prop; within-bin OR

Produces zcode_results.json with all CH_4 manuscript values.
"""
import boto3, io, json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, mannwhitneyu

s3     = boto3.client("s3",     region_name="us-east-1")
athena = boto3.client("athena", region_name="us-east-1")
BUCKET = "pgxdatalake"
OUTPUT = "s3://pgxdatalake/athena-query-results/"
BANDS  = ["65-74", "75-84", "85-114"]
BINS   = ["low", "medium", "high", "extreme"]


def athena_query(sql, workgroup="APCD", wait=240):
    r   = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": "medical_raw"},
        ResultConfiguration={"OutputLocation": OUTPUT},
        WorkGroup=workgroup,
    )
    qid = r["QueryExecutionId"]
    for i in range(wait // 3):
        time.sleep(3)
        st    = athena.get_query_execution(QueryExecutionId=qid)
        state = st["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if i % 10 == 0:
            print(f"  ... {i*3}s state={state}")
    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena {state}: "
                           f"{st['QueryExecution']['Status'].get('StateChangeReason','')}")
    rows, tok = [], None
    while True:
        kw = {"QueryExecutionId": qid, "MaxResults": 1000}
        if tok:
            kw["NextToken"] = tok
        res = athena.get_query_results(**kw)
        rows.extend(res["ResultSet"]["Rows"])
        tok = res.get("NextToken")
        if not tok:
            break
    return rows


# ── Load cohort with n_event_bin ──────────────────────────────────────────────
print("Loading non_opioid_ed cohort + n_event_bin …")
ff_rows = []
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
            ff_rows.append(ff)
        except Exception as e:
            print(f"  SKIP {split}/{band}: {e}")

cohort = (pd.concat(ff_rows, ignore_index=True)
            .sort_values("target", ascending=False)
            .drop_duplicates("mi_person_key"))
cohort["mi_person_key"] = pd.to_numeric(cohort["mi_person_key"],
                                         errors="coerce").astype("Int64")
n_cases = cohort["target"].sum()
n_ctrl  = (cohort["target"] == 0).sum()
print(f"  {n_cases:,} cases, {n_ctrl:,} controls  ({len(cohort):,} total)")
print(f"  n_event_bin distribution:")
print(cohort["n_event_bin"].value_counts().to_string())

# ── Athena: protective Z-code proportion ─────────────────────────────────────
print("\nQuerying Athena (protective Z00-Z13, Z23-Z28 only) …")
sql = """
SELECT
    mi_person_key,
    COUNT(*) AS total_claims,
    SUM(CASE
            WHEN REGEXP_LIKE(CAST(primary_icd_diagnosis_code AS VARCHAR),
                             '^Z(0[0-9]|1[0-3]|2[3-8])')
              OR REGEXP_LIKE(CAST(two_icd_diagnosis_code AS VARCHAR),
                             '^Z(0[0-9]|1[0-3]|2[3-8])')
              OR REGEXP_LIKE(CAST(three_icd_diagnosis_code AS VARCHAR),
                             '^Z(0[0-9]|1[0-3]|2[3-8])')
            THEN 1 ELSE 0
        END) AS protective_z_claims
FROM medical_raw.medical_partitioned
WHERE age_band IN ('65-74','75-84','85-94','95-114')
  AND CAST(event_year AS INTEGER) BETWEEN 2015 AND 2019
GROUP BY mi_person_key
"""
rows   = athena_query(sql)
header = [c["VarCharValue"] for c in rows[0]["Data"]]
zdf    = pd.DataFrame(
    [dict(zip(header, [c.get("VarCharValue","") for c in r["Data"]]))
     for r in rows[1:]]
)
zdf["total_claims"]        = pd.to_numeric(zdf["total_claims"],        errors="coerce").fillna(0)
zdf["protective_z_claims"] = pd.to_numeric(zdf["protective_z_claims"], errors="coerce").fillna(0)
zdf["z_prop"]              = zdf["protective_z_claims"] / zdf["total_claims"].clip(lower=1)
zdf["mi_person_key"]       = pd.to_numeric(zdf["mi_person_key"],       errors="coerce").astype("Int64")
print(f"  {len(zdf):,} patient rows  "
      f"median z_prop={zdf['z_prop'].median():.3f}  "
      f"pct_zero={((zdf['z_prop']==0).mean()*100):.1f}%")

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = cohort.merge(zdf[["mi_person_key","z_prop","total_claims"]],
                      on="mi_person_key", how="left")
# Patients not found in Athena: assign z_prop=0 (no claims in geriatric bands)
merged["z_prop"]      = merged["z_prop"].fillna(0.0)
merged["total_claims"] = merged["total_claims"].fillna(0).astype(int)
print(f"\nMerged cohort: {len(merged):,} patients")
pct_zero = (merged["z_prop"] == 0).mean() * 100
print(f"  z_prop=0 (no protective Z-codes): {pct_zero:.1f}% of cohort")

# ── Table 1 IQR ──────────────────────────────────────────────────────────────
print("\n=== TABLE 1: Z-CODE PROPORTION IQR ===")
iqr_out = {}
for label, tgt in [("Cases", 1), ("Controls", 0)]:
    g = merged[merged["target"] == tgt]["z_prop"]
    q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
    iqr_out[label] = (q25, q50, q75, len(g))
    print(f"  {label}: median={q50:.2f} ({q25:.2f}–{q75:.2f})  n={len(g):,}")
    print(f"    z_prop=0: {(g==0).sum():,} ({(g==0).mean()*100:.1f}%)")

# ── Within-bin analysis: the key non-linear story ────────────────────────────
print("\n=== WITHIN-BIN Z-PROP BY CASE/CONTROL ===")
print(f"{'Bin':10s} {'Group':10s} {'n':>6s} {'z=0%':>6s} "
      f"{'median':>8s} {'IQR':>20s} {'Mann-W p':>10s}")
print("-" * 80)

bin_results = {}
for bname in BINS:
    bin_results[bname] = {}
    cases_b = merged[(merged["n_event_bin"] == bname) & (merged["target"] == 1)]
    ctrl_b  = merged[(merged["n_event_bin"] == bname) & (merged["target"] == 0)]
    if len(cases_b) == 0 and len(ctrl_b) == 0:
        continue

    for label, g in [("Cases", cases_b["z_prop"]),
                     ("Controls", ctrl_b["z_prop"])]:
        if len(g) == 0:
            print(f"  {bname:10s} {label:10s}: no data")
            continue
        q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
        pz = (g == 0).mean() * 100
        print(f"  {bname:10s} {label:10s} {len(g):6,} {pz:5.1f}% "
              f"{q50:8.3f} ({q25:.3f}–{q75:.3f})")
        bin_results[bname][label] = {
            "n": len(g), "median": q50, "q25": q25, "q75": q75,
            "pct_zero": pz
        }

    # Mann-Whitney U test within bin (cases vs controls)
    if len(cases_b) > 0 and len(ctrl_b) > 0:
        stat, pval = mannwhitneyu(cases_b["z_prop"], ctrl_b["z_prop"],
                                  alternative="two-sided")
        direction = ("cases < controls"
                     if cases_b["z_prop"].median() < ctrl_b["z_prop"].median()
                     else "cases > controls")
        print(f"  {'':10s} {'MW-U p':10s} {pval:.4f}  [{direction}]")
        bin_results[bname]["mw_pval"] = pval
        bin_results[bname]["direction"] = direction

# ── Within-bin OR for each bin ────────────────────────────────────────────────
print("\n=== WITHIN-BIN LOGISTIC REGRESSION (z_prop_Q4 vs Q1, per bin) ===")

def logit_or(y, z):
    """Simple logit OR for z quartile (Q4 vs Q1 reference) within a bin."""
    try:
        df = pd.DataFrame({"y": y, "z": z})
        df["z_q"] = pd.qcut(df["z"].rank(method="first"), q=4,
                             labels=[1, 2, 3, 4]).astype(float)
        df["q4"] = (df["z_q"] == 4).astype(float)
        Xm = np.column_stack([np.ones(len(df)), df["q4"].values])
        yv = df["y"].astype(int).values

        def nll(b):
            p = expit(Xm @ b)
            return -np.sum(yv*np.log(p+1e-15) + (1-yv)*np.log(1-p+1e-15))

        def grd(b):
            p = expit(Xm @ b)
            return -Xm.T @ (yv - p)

        res  = minimize(nll, np.zeros(2), jac=grd, method="L-BFGS-B")
        b    = res.x
        ph   = expit(Xm @ b)
        W    = ph * (1 - ph)
        H    = (Xm.T * W) @ Xm
        se   = np.sqrt(np.diag(np.linalg.inv(H)))
        or_  = np.exp(b[1])
        lo   = np.exp(b[1] - 1.96*se[1])
        hi   = np.exp(b[1] + 1.96*se[1])
        pv   = 2 * norm.sf(abs(b[1] / se[1]))
        return or_, lo, hi, pv
    except Exception as e:
        return float("nan"), float("nan"), float("nan"), float("nan")

bin_or = {}
for bname in BINS:
    sub = merged[merged["n_event_bin"] == bname].dropna(subset=["z_prop"])
    if len(sub) < 20:
        print(f"  {bname}: insufficient data (n={len(sub)})")
        continue
    or_, lo, hi, pv = logit_or(sub["target"].values, sub["z_prop"].values)
    pstr = "<0.001" if pv < 0.001 else f"{pv:.3f}"
    print(f"  {bname:10s}: Q4 vs Q1 OR={or_:.2f} ({lo:.2f}–{hi:.2f})  p={pstr}")
    bin_or[bname] = {"or": round(float(or_),2), "ci_lo": round(float(lo),2),
                     "ci_hi": round(float(hi),2), "pval": pstr}

# ── Overall quartile with Q4 as reference (per manuscript narrative) ──────────
print("\n=== OVERALL Q4 as REFERENCE (non-linear narrative) ===")
df_q = merged.copy()
try:
    df_q["z_q"] = pd.qcut(df_q["z_prop"], q=4, labels=[1,2,3,4], duplicates="drop")
except ValueError:
    df_q["z_q"] = pd.qcut(df_q["z_prop"].rank(method="first"), q=4,
                           labels=[1,2,3,4])

q4_lo = df_q["z_prop"].quantile(0.75)
print(f"  Q4 threshold: ≥ {q4_lo:.2f} ({q4_lo*100:.0f}% of claims)")

# Crude ORs vs Q4 reference
for q in [1,2,3]:
    g4  = df_q[df_q["z_q"] == 4]
    gq  = df_q[df_q["z_q"] == q]
    o4  = g4["target"].sum() / max((g4["target"]==0).sum(), 1)
    oq  = gq["target"].sum() / max((gq["target"]==0).sum(), 1)
    n4  = len(g4); nq = len(gq)
    print(f"  Q{q} vs Q4 crude OR={oq/max(o4,1e-9):.2f}  "
          f"(Q{q} case%={(gq['target']==1).mean()*100:.1f}%  "
          f"Q4 case%={(g4['target']==1).mean()*100:.1f}%)")

# ── Summary of non-linear pattern ────────────────────────────────────────────
print("\n=== NON-LINEAR SUMMARY ===")
print("Bin   | ctrl z_prop=0% | ctrl median z | case median z | direction")
for bname in BINS:
    cm = bin_results.get(bname, {}).get("Cases",  {})
    rm = bin_results.get(bname, {}).get("Controls", {})
    cpz = cm.get("pct_zero", float("nan"))
    rpz = rm.get("pct_zero", float("nan"))
    cmed = cm.get("median", float("nan"))
    rmed = rm.get("median", float("nan"))
    dir_ = bin_results.get(bname, {}).get("direction", "n/a")
    print(f"{bname:8s}| ctrl_0%={rpz:4.1f}  ctrl_med={rmed:.3f}  "
          f"case_0%={cpz:4.1f}  case_med={cmed:.3f}  [{dir_}]")

# ── Save results ──────────────────────────────────────────────────────────────
cases_g = merged[merged["target"] == 1]["z_prop"]
ctrl_g  = merged[merged["target"] == 0]["z_prop"]

results = {
    "note": "Protective Z-codes only: Z00-Z13 (exams/screens) + Z23-Z28 (immunizations)",
    "cases_median":    round(float(cases_g.median()), 2),
    "cases_q25":       round(float(cases_g.quantile(0.25)), 2),
    "cases_q75":       round(float(cases_g.quantile(0.75)), 2),
    "controls_median": round(float(ctrl_g.median()), 2),
    "controls_q25":    round(float(ctrl_g.quantile(0.25)), 2),
    "controls_q75":    round(float(ctrl_g.quantile(0.75)), 2),
    "q4_threshold_pct": round(q4_lo * 100),
    "within_bin": bin_results,
    "within_bin_or": bin_or,
}

merged.to_parquet("zcode_merged.parquet", index=False)
with open("zcode_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved zcode_merged.parquet and zcode_results.json")
