"""
Query Athena for total unique APCD patient count (2016-2019) for CH_3 attrition caption.
"""
import boto3, time

glue   = boto3.client("glue",   region_name="us-east-1")
athena = boto3.client("athena", region_name="us-east-1")
OUTPUT = "s3://pgxdatalake/athena-query-results/"

# ── List tables in relevant databases ────────────────────────────────────────
for db in ["medical_raw", "bronze_medical", "cohorts", "silver_medical", "medical"]:
    try:
        tables = glue.get_tables(DatabaseName=db)["TableList"]
        names  = [t["Name"] for t in tables[:15]]
        print(f"{db}: {names}")
    except Exception as e:
        print(f"{db}: {e}")

print()

# ── Find which table has mi_person_key ────────────────────────────────────────
for db, tbl in [
    ("medical_raw",   "medical_claims"),
    ("bronze_medical","medical_claims"),
    ("cohorts",       "opioid_ed"),
    ("silver_medical","medical_claims"),
    ("medical",       "claims"),
]:
    try:
        t = glue.get_table(DatabaseName=db, Name=tbl)
        cols = [c["Name"] for c in t["Table"]["StorageDescriptor"]["Columns"]]
        print(f"{db}.{tbl}: {cols[:10]}")
    except Exception as e:
        pass  # table doesn't exist


def athena_query(sql, db="default", workgroup="APCD", wait_sec=60):
    r = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": db},
        ResultConfiguration={"OutputLocation": OUTPUT},
        WorkGroup=workgroup,
    )
    qid = r["QueryExecutionId"]
    for _ in range(wait_sec):
        time.sleep(2)
        st = athena.get_query_execution(QueryExecutionId=qid)
        state = st["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
    if state != "SUCCEEDED":
        reason = st["QueryExecution"]["Status"].get("StateChangeReason", "")
        print(f"  FAILED: {reason}")
        return None
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    return rows


# ── Check cohorts database ────────────────────────────────────────────────────
print("\n=== cohorts database tables ===")
try:
    tables = glue.get_tables(DatabaseName="cohorts")["TableList"]
    for t in tables:
        name = t["Name"]
        cols = [c["Name"] for c in t["StorageDescriptor"]["Columns"][:5]]
        print(f"  {name}: {cols}")
except Exception as e:
    print(f"  {e}")

# ── Try counting unique patients from silver_medical ─────────────────────────
print("\n=== Unique patient count query ===")
try:
    tables = glue.get_tables(DatabaseName="silver_medical")["TableList"]
    tbl_name = tables[0]["Name"] if tables else None
    if tbl_name:
        cols = [c["Name"] for c in tables[0]["StorageDescriptor"]["Columns"]]
        print(f"  silver_medical.{tbl_name} columns: {cols[:10]}")
        # Try count
        if "mi_person_key" in cols:
            rows = athena_query(
                f"SELECT COUNT(DISTINCT mi_person_key) AS n FROM silver_medical.{tbl_name} "
                "WHERE year >= 2016 AND year <= 2019",
                db="silver_medical"
            )
            if rows:
                print(f"  Unique patients 2016-2019: {rows[1]['Data'][0]['VarCharValue']}")
except Exception as e:
    print(f"  {e}")
