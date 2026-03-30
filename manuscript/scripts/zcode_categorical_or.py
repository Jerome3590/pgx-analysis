"""
Extend Z-code analysis with categorical Q4 vs Q1 OR and extreme-density subgroups.
Saves zcode_results.json with all CH_4 manuscript values.
"""
import boto3, io, json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

s3     = boto3.client("s3",     region_name="us-east-1")
athena = boto3.client("athena", region_name="us-east-1")
BUCKET = "pgxdatalake"
OUTPUT = "s3://pgxdatalake/athena-query-results/"
BANDS  = ["65-74", "75-84", "85-114"]


def athena_query(sql, workgroup="APCD", wait=180):
    r = athena.start_query_execution(
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
        reason = st["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Athena {state}: {reason}")
    rows, next_tok = [], None
    while True:
        kwargs = {"QueryExecutionId": qid, "MaxResults": 1000}
        if next_tok:
            kwargs["NextToken"] = next_tok
        res = athena.get_query_results(**kwargs)
        rows.extend(res["ResultSet"]["Rows"])
        next_tok = res.get("NextToken")
        if not next_tok:
            break
    return rows


# ── Load cohort ──────────────────────────────────────────────────────────────
print("Loading non_opioid_ed cohort …")
cohort_rows = []
for band in BANDS:
    for split in ["model_test", "model_train"]:
        key  = (f"gold/final_model/non_opioid_ed/{band}/inputs/"
                f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            fp   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target",
                                            "n_events", "n_event_bin"])
            fp["band"] = band
            cohort_rows.append(fp)
        except Exception as e:
            print(f"  SKIP {split}/{band}: {e}")

cohort = pd.concat(cohort_rows, ignore_index=True)
cohort = (cohort.sort_values("target", ascending=False)
                .drop_duplicates("mi_person_key"))
cohort["mi_person_key"] = pd.to_numeric(cohort["mi_person_key"],
                                         errors="coerce").astype("Int64")
print(f"  {cohort['target'].sum():,} cases, "
      f"{(cohort['target']==0).sum():,} controls  ({len(cohort):,} total)")

# ── Athena Z-code proportions ─────────────────────────────────────────────────
print("\nQuerying Athena …")
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
        END) AS z_claims
FROM medical_raw.medical_partitioned
WHERE age_band IN ('65-74', '75-84', '85-94', '95-114')
  AND CAST(event_year AS INTEGER) BETWEEN 2015 AND 2019
GROUP BY mi_person_key
"""
print("  Z-code filter: protective monitoring only (Z00-Z13 exams/screens + Z23-Z28 immunizations)")
rows = athena_query(sql)
header = [c["VarCharValue"] for c in rows[0]["Data"]]
records = [dict(zip(header, [c.get("VarCharValue", "") for c in r["Data"]]))
           for r in rows[1:]]
zdf = pd.DataFrame(records)
zdf["total_claims"] = pd.to_numeric(zdf["total_claims"], errors="coerce").fillna(0)
zdf["z_claims"]     = pd.to_numeric(zdf["z_claims"],     errors="coerce").fillna(0)
zdf["z_prop"]       = zdf["z_claims"] / zdf["total_claims"].clip(lower=1)
zdf["mi_person_key"] = pd.to_numeric(zdf["mi_person_key"],
                                      errors="coerce").astype("Int64")
print(f"  Athena rows: {len(zdf):,}  "
      f"median z_prop={zdf['z_prop'].median():.3f}")

merged = cohort.merge(zdf[["mi_person_key", "z_prop", "total_claims"]],
                      on="mi_person_key", how="inner")
print(f"  Merged: {len(merged):,} "
      f"({merged['target'].sum():,} cases, "
      f"{(merged['target']==0).sum():,} controls)")

# Save for reuse (avoids repeat Athena query)
merged.to_parquet("zcode_merged.parquet", index=False)
print("  Saved zcode_merged.parquet")

# ── IQR by case/control ──────────────────────────────────────────────────────
print("\n=== Z-CODE IQR (full enrollment 2015-2019) ===")
for label, tgt in [("Cases", 1), ("Controls", 0)]:
    g = merged[merged["target"] == tgt]["z_prop"]
    q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
    print(f"  {label}: median={q50:.2f} ({q25:.2f}–{q75:.2f})  n={len(g):,}")

# ── Within-bin case rate (removes density confounding) ─────────────────────
print("\n=== WITHIN-BIN Z-PROP BY CASE/CONTROL ===")
for bname in ["low", "medium", "high", "extreme"]:
    sub = merged[merged["n_event_bin"] == bname]
    if len(sub) == 0:
        continue
    for label, tgt in [("Cases", 1), ("Controls", 0)]:
        g = sub[sub["target"] == tgt]["z_prop"]
        if len(g) == 0:
            continue
        q25, q50, q75 = g.quantile([0.25, 0.50, 0.75])
        print(f"  {bname:8s} {label:10s}: median={q50:.3f} ({q25:.3f}–{q75:.3f})  "
              f"n={len(g):,}")

# ── Quartile distribution ─────────────────────────────────────────────────────
print("\n=== QUARTILE CASE RATES ===")
df_q = merged.copy()
try:
    df_q["z_q"] = pd.qcut(df_q["z_prop"], q=4, labels=[1, 2, 3, 4],
                           duplicates="drop")
except ValueError:
    df_q["z_q"] = pd.qcut(df_q["z_prop"].rank(method="first"),
                           q=4, labels=[1, 2, 3, 4])

q4_lo = df_q["z_prop"].quantile(0.75)
print(f"  Q4 lower bound (75th pct): {q4_lo:.2f} ({q4_lo*100:.0f}% of claims)")

for q in [1, 2, 3, 4]:
    sub = df_q[df_q["z_q"] == q]
    nc  = (sub["target"] == 1).sum()
    nct = (sub["target"] == 0).sum()
    pct = nc / len(sub) * 100 if len(sub) > 0 else 0
    print(f"  Q{q}: n={len(sub):,}  cases={nc:,} ({pct:.1f}%)  controls={nct:,}")

# ── Crude OR Q1 vs Q4 (Q4=reference, no n_events adjustment) ────────────────
print("\n=== CRUDE OR: Q1 vs Q4 (protective comparison, unadjusted) ===")
for q in [1, 2, 3]:
    g_ref = df_q[df_q["z_q"] == 4]
    g_q   = df_q[df_q["z_q"] == q]
    odds_ref = g_ref["target"].sum() / max((g_ref["target"]==0).sum(), 1)
    odds_q   = g_q["target"].sum()   / max((g_q["target"]==0).sum(), 1)
    crude_or = odds_q / max(odds_ref, 1e-10)
    print(f"  Q{q} vs Q4 crude OR = {crude_or:.2f}")

# ── Categorical logistic regression Q4 vs Q1 reference ──────────────────────
print("\n=== CATEGORICAL LOGISTIC REGRESSION (Q1 as reference) ===")
df_lr = df_q.dropna(subset=["z_prop", "n_events"]).copy()
df_lr["q2"] = (df_lr["z_q"] == 2).astype(float)
df_lr["q3"] = (df_lr["z_q"] == 3).astype(float)
df_lr["q4"] = (df_lr["z_q"] == 4).astype(float)
df_lr = pd.get_dummies(df_lr, columns=["band"], drop_first=True)
band_cols = [c for c in df_lr.columns if c.startswith("band_")]

Xmat = np.column_stack(
    [np.ones(len(df_lr))]
    + [df_lr[c].astype(float).values
       for c in ["q2", "q3", "q4", "n_events"] + band_cols]
)
y = df_lr["target"].astype(int).values


def neg_ll(b):
    p = expit(Xmat @ b)
    return -np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))


def neg_ll_grad(b):
    p = expit(Xmat @ b)
    return -Xmat.T @ (y - p)


res  = minimize(neg_ll, np.zeros(Xmat.shape[1]), jac=neg_ll_grad, method="L-BFGS-B")
beta = res.x
p_hat = expit(Xmat @ beta)
W     = p_hat * (1 - p_hat)
H     = (Xmat.T * W) @ Xmat
try:
    se = np.sqrt(np.diag(np.linalg.inv(H)))
except np.linalg.LinAlgError:
    se = np.full(len(beta), np.nan)

for i, qname in enumerate(["Q2", "Q3", "Q4"], start=1):
    coef  = beta[i]
    se_c  = se[i]
    ci_lo = coef - 1.96 * se_c
    ci_hi = coef + 1.96 * se_c
    pval  = 2 * norm.sf(abs(coef / se_c))
    pstr  = "<0.001" if pval < 0.001 else f"{pval:.3f}"
    print(f"  {qname} vs Q1: OR={np.exp(coef):.2f} "
          f"({np.exp(ci_lo):.2f}–{np.exp(ci_hi):.2f})  p={pstr}")
    if qname == "Q4":
        or_q4, ci_lo_q4, ci_hi_q4, pval_q4 = (
            np.exp(coef), np.exp(ci_lo), np.exp(ci_hi), pstr)

# ── Extreme-density subgroup ──────────────────────────────────────────────────
print("\n=== EXTREME-DENSITY BIMODAL ANALYSIS ===")
ext = merged[merged["n_event_bin"] == "extreme"].copy()
print(f"  Extreme-density patients: {len(ext):,} "
      f"({ext['target'].sum():,} cases, {(ext['target']==0).sum():,} controls)")

if len(ext) > 20:
    ext_median = ext["z_prop"].median()
    ext["z_high"] = (ext["z_prop"] >= ext_median).astype(int)
    print(f"  Z-prop split at median={ext_median:.3f}")

    for hi, lab in [(0, "Low Z (high-risk)"), (1, "High Z (low-risk)")]:
        g = ext[ext["z_high"] == hi]
        nc = (g["target"] == 1).sum()
        nt = (g["target"] == 0).sum()
        print(f"  {lab}: cases={nc}, controls={nt}")

    # Logistic regression within extreme-density
    ext_lr = ext.dropna(subset=["z_prop", "n_events"]).copy()
    Xe = np.column_stack([np.ones(len(ext_lr)),
                          ext_lr["z_high"].astype(float).values,
                          ext_lr["n_events"].astype(float).values])
    ye = ext_lr["target"].astype(int).values

    def neg_ll_e(b):
        p = expit(Xe @ b)
        return -np.sum(ye * np.log(p + 1e-15) + (1 - ye) * np.log(1 - p + 1e-15))

    def grad_e(b):
        p = expit(Xe @ b)
        return -Xe.T @ (ye - p)

    res_e = minimize(neg_ll_e, np.zeros(3), jac=grad_e, method="L-BFGS-B")
    be    = res_e.x
    pe    = expit(Xe @ be)
    We    = pe * (1 - pe)
    He    = (Xe.T * We) @ Xe
    try:
        se_e = np.sqrt(np.diag(np.linalg.inv(He)))
    except np.linalg.LinAlgError:
        se_e = np.full(3, np.nan)
    or_ext   = np.exp(be[1])
    ci_lo_e  = np.exp(be[1] - 1.96 * se_e[1])
    ci_hi_e  = np.exp(be[1] + 1.96 * se_e[1])
    pval_e   = 2 * norm.sf(abs(be[1] / se_e[1]))
    pstr_e   = "<0.001" if pval_e < 0.001 else f"{pval_e:.3f}"
    print(f"  High-Z vs Low-Z OR = {or_ext:.2f} "
          f"({ci_lo_e:.2f}–{ci_hi_e:.2f})  p={pstr_e}")
    # Low-Z OR = 1/or_ext (reference flipped)
    or_low   = 1.0 / or_ext if or_ext > 0 else float("nan")
    or_high_ci_lo = 1.0 / ci_hi_e
    or_high_ci_hi = 1.0 / ci_lo_e
    print(f"  Low-Z vs High-Z OR  = {or_low:.2f} "
          f"({or_high_ci_lo:.2f}–{or_high_ci_hi:.2f})")
else:
    print("  Not enough extreme-density patients for subgroup analysis")
    or_ext = float("nan"); ci_lo_e = float("nan"); ci_hi_e = float("nan")
    or_low = float("nan"); ext_median = float("nan")

# ── IQR for Table 1 ──────────────────────────────────────────────────────────
cases_g = merged[merged["target"] == 1]["z_prop"]
ctrl_g  = merged[merged["target"] == 0]["z_prop"]

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "cases_median": round(cases_g.median(), 2),
    "cases_q25":    round(cases_g.quantile(0.25), 2),
    "cases_q75":    round(cases_g.quantile(0.75), 2),
    "controls_median": round(ctrl_g.median(), 2),
    "controls_q25":    round(ctrl_g.quantile(0.25), 2),
    "controls_q75":    round(ctrl_g.quantile(0.75), 2),
    "q4_threshold_pct": round(q4_lo * 100),
    "or_q4_vs_q1":  round(or_q4, 2),
    "ci_lo_q4":     round(ci_lo_q4, 2),
    "ci_hi_q4":     round(ci_hi_q4, 2),
    "pval_q4":      pval_q4,
    "extreme_z_median": round(float(ext_median), 3) if not np.isnan(float(ext_median)) else None,
    "or_ext_high_z": round(or_ext, 2),
    "ci_lo_ext":    round(ci_lo_e, 2),
    "ci_hi_ext":    round(ci_hi_e, 2),
    "or_ext_low_z": round(or_low, 2),
    "ci_lo_low_z":  round(or_high_ci_lo, 2),
    "ci_hi_low_z":  round(or_high_ci_hi, 2),
}

print("\n" + "=" * 60)
print("=== FINAL CH_4 MANUSCRIPT VALUES ===")
print(f"  Table 1 Z-code IQR")
print(f"    Cases:    {results['cases_median']:.2f} ({results['cases_q25']:.2f}–{results['cases_q75']:.2f})")
print(f"    Controls: {results['controls_median']:.2f} ({results['controls_q25']:.2f}–{results['controls_q75']:.2f})")
print(f"  Q4 threshold: ≥ {results['q4_threshold_pct']}% of claims")
print(f"  Q4 vs Q1 OR = {results['or_q4_vs_q1']} ({results['ci_lo_q4']}–{results['ci_hi_q4']})")
print(f"  Extreme-density High-Z OR = {results['or_ext_high_z']} ({results['ci_lo_ext']}–{results['ci_hi_ext']})")
print(f"  Extreme-density Low-Z  OR = {results['or_ext_low_z']} ({results['ci_lo_low_z']}–{results['ci_hi_low_z']})")

with open("data/zcode_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved zcode_results.json")
