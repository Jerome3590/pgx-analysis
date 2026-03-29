"""
Z-code proportion IQR and logistic regression OR for CH_4 (non_opioid_ed).
Uses Athena medical_raw.medical_partitioned for full claims history.
Steps:
1. Load non_opioid_ed case/control patient IDs from final_features.parquet
2. Query Athena for per-patient Z-code claim proportion in geriatric age bands
3. Join cohort labels; compute IQR by case/control
4. Run logistic regression with covariates age, sex, drug_count → OR + 95%CI
"""
import boto3, io, time
import pandas as pd
import numpy as np

s3     = boto3.client("s3",     region_name="us-east-1")
athena = boto3.client("athena", region_name="us-east-1")
BUCKET = "pgxdatalake"
OUTPUT = "s3://pgxdatalake/athena-query-results/"
BANDS  = ["65-74", "75-84", "85-114"]
AB_MAP = {"65-74": "65-74", "75-84": "75-84", "85-114": "85-94"}  # Athena band → APCD band


def athena_query(sql, workgroup="APCD", wait=120):
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
    # Paginate results
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


# ── Step 1: Load cohort patient IDs ─────────────────────────────────────────
print("Loading non_opioid_ed cohort …")
cohort_rows = []
for band in BANDS:
    for split in ["model_test", "model_train"]:
        key  = (f"gold/final_model/non_opioid_ed/{band}/inputs/"
                f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            fp   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target", "n_events"])
            fp["band"] = band
            cohort_rows.append(fp)
        except Exception as e:
            print(f"  {split}/{band}: {e}")

cohort = pd.concat(cohort_rows, ignore_index=True)
# Keep one row per patient (prefer target=1 if duplicate)
cohort = (cohort.sort_values("target", ascending=False)
                .drop_duplicates("mi_person_key"))
n_cases    = cohort["target"].sum()
n_controls = (cohort["target"] == 0).sum()
print(f"  {n_cases:,} cases, {n_controls:,} controls  (total {len(cohort):,})")

# ── Step 2: Athena Z-code proportion query ───────────────────────────────────
# Query all geriatric bands 2015-2019 (pre-index period)
print("\nQuerying Athena for Z-code proportions (geriatric bands 2015–2019) …")
sql = """
SELECT
    mi_person_key,
    COUNT(*) AS total_claims,
    SUM(CASE
            WHEN CAST(primary_icd_diagnosis_code AS VARCHAR) LIKE 'Z%'
              OR CAST(two_icd_diagnosis_code     AS VARCHAR) LIKE 'Z%'
              OR CAST(three_icd_diagnosis_code   AS VARCHAR) LIKE 'Z%'
            THEN 1 ELSE 0
        END) AS z_claims
FROM medical_raw.medical_partitioned
WHERE age_band IN ('65-74', '75-84', '85-94', '95-114')
  AND CAST(event_year AS INTEGER) BETWEEN 2015 AND 2019
GROUP BY mi_person_key
"""
print(f"  SQL: {sql[:100].strip()} …")
rows = athena_query(sql)
# Parse results
header = [c["VarCharValue"] for c in rows[0]["Data"]]
print(f"  Returned {len(rows)-1:,} patient rows (header: {header})")

records = []
for row in rows[1:]:
    vals = [c.get("VarCharValue", "") for c in row["Data"]]
    records.append(dict(zip(header, vals)))
zdf = pd.DataFrame(records)
zdf["total_claims"] = pd.to_numeric(zdf["total_claims"], errors="coerce").fillna(0)
zdf["z_claims"]     = pd.to_numeric(zdf["z_claims"],     errors="coerce").fillna(0)
zdf["z_prop"]       = zdf["z_claims"] / zdf["total_claims"].clip(lower=1)
zdf["mi_person_key"] = pd.to_numeric(zdf["mi_person_key"], errors="coerce").astype("Int64")
cohort["mi_person_key"] = pd.to_numeric(cohort["mi_person_key"], errors="coerce").astype("Int64")
print(f"  Z-code prop stats: mean={zdf['z_prop'].mean():.4f}  "
      f"median={zdf['z_prop'].median():.4f}")

# ── Step 3: Merge cohort labels ───────────────────────────────────────────────
merged = cohort.merge(zdf[["mi_person_key", "z_prop", "total_claims"]],
                      on="mi_person_key", how="inner")
print(f"\nMerged: {len(merged):,} patients  "
      f"({merged['target'].sum():,} cases, "
      f"{(merged['target']==0).sum():,} controls)")

for label, grp in [("Cases", merged[merged["target"]==1]),
                   ("Controls", merged[merged["target"]==0])]:
    q25, q50, q75 = (grp["z_prop"].quantile(0.25),
                     grp["z_prop"].quantile(0.50),
                     grp["z_prop"].quantile(0.75))
    print(f"  {label}: median={q50:.3f}  IQR={q25:.3f}–{q75:.3f}  "
          f"n={len(grp):,}")

# ── Step 4: Logistic regression ───────────────────────────────────────────────
print("\nLogistic regression: target ~ z_prop_quartile + n_events + band")
from scipy.optimize import minimize
from scipy.special import expit

df_lr = merged.dropna(subset=["z_prop", "n_events", "target"]).copy()
df_lr["z_q"] = pd.qcut(df_lr["z_prop"], q=4, labels=[1, 2, 3, 4],
                        duplicates="drop").astype(float)
df_lr = pd.get_dummies(df_lr, columns=["band"], drop_first=True)
band_cols = [c for c in df_lr.columns if c.startswith("band_")]

X_cols = ["z_q", "n_events"] + band_cols
Xmat = np.column_stack([np.ones(len(df_lr))] +
                       [df_lr[c].astype(float).values for c in X_cols])
y    = df_lr["target"].astype(int).values

# MLE via scipy
def neg_ll(b):
    p = expit(Xmat @ b)
    return -np.sum(y * np.log(p + 1e-15) + (1-y) * np.log(1-p + 1e-15))

def neg_ll_grad(b):
    p = expit(Xmat @ b)
    return -Xmat.T @ (y - p)

b0  = np.zeros(Xmat.shape[1])
res = minimize(neg_ll, b0, jac=neg_ll_grad, method="L-BFGS-B")
beta = res.x

# Hessian-based SE via Fisher information
p_hat = expit(Xmat @ beta)
W     = p_hat * (1 - p_hat)
H     = (Xmat.T * W) @ Xmat
try:
    cov = np.linalg.inv(H)
    se  = np.sqrt(np.diag(cov))
except np.linalg.LinAlgError:
    se = np.full(len(beta), np.nan)

from scipy.stats import norm
coef  = beta[1]         # z_q coefficient (index 0 = intercept)
se_c  = se[1]
ci_lo = coef - 1.96 * se_c
ci_hi = coef + 1.96 * se_c
pval  = 2 * norm.sf(abs(coef / se_c))
print(f"\n  z_prop_quartile: β={coef:.3f}  OR={np.exp(coef):.3f}  "
      f"95%CI [{np.exp(ci_lo):.3f}–{np.exp(ci_hi):.3f}]  p={pval:.4f}")

print(f"\n=== CH_4 MANUSCRIPT VALUES ===")
cases_grp = merged[merged["target"] == 1]
ctrl_grp  = merged[merged["target"] == 0]
q25c, q50c, q75c = (cases_grp["z_prop"].quantile(0.25),
                    cases_grp["z_prop"].quantile(0.50),
                    cases_grp["z_prop"].quantile(0.75))
q25k, q50k, q75k = (ctrl_grp["z_prop"].quantile(0.25),
                    ctrl_grp["z_prop"].quantile(0.50),
                    ctrl_grp["z_prop"].quantile(0.75))
print(f"  Z-code IQR Cases:    {q50c:.3f} ({q25c:.3f}–{q75c:.3f})")
print(f"  Z-code IQR Controls: {q50k:.3f} ({q25k:.3f}–{q75k:.3f})")
print(f"  OR = {np.exp(coef):.2f} (95% CI {np.exp(ci_lo):.2f}–{np.exp(ci_hi):.2f}; "
      f"p={'<0.001' if pval < 0.001 else f'{pval:.3f}'})")
