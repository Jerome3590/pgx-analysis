"""Query medical_raw.medical for total unique APCD patients 2016-2019."""
import boto3, time

glue   = boto3.client("glue",   region_name="us-east-1")
athena = boto3.client("athena", region_name="us-east-1")
OUTPUT = "s3://pgxdatalake/athena-query-results/"


def athena_query(sql, db="default", workgroup="APCD", wait_sec=120):
    r = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": db},
        ResultConfiguration={"OutputLocation": OUTPUT},
        WorkGroup=workgroup,
    )
    qid = r["QueryExecutionId"]
    print(f"  query id: {qid}")
    for i in range(wait_sec // 2):
        time.sleep(2)
        st = athena.get_query_execution(QueryExecutionId=qid)
        state = st["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if i % 10 == 0:
            print(f"  ... waiting ({i*2}s, state={state})")
    if state != "SUCCEEDED":
        reason = st["QueryExecution"]["Status"].get("StateChangeReason", "")
        print(f"  FAILED: {state} — {reason}")
        return None
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    return rows


# ── Inspect medical_raw.medical columns ─────────────────────────────────────
print("=== medical_raw.medical columns ===")
try:
    t = glue.get_table(DatabaseName="medical_raw", Name="medical")
    cols = [c["Name"] for c in t["Table"]["StorageDescriptor"]["Columns"]]
    print(f"  {cols[:20]}")
    part_keys = [k["Name"] for k in t["Table"].get("PartitionKeys", [])]
    print(f"  partition keys: {part_keys}")
except Exception as e:
    print(f"  {e}")

# ── Also check medical_partitioned ────────────────────────────────────────────
print("\n=== medical_raw.medical_partitioned columns ===")
try:
    t = glue.get_table(DatabaseName="medical_raw", Name="medical_partitioned")
    cols = [c["Name"] for c in t["Table"]["StorageDescriptor"]["Columns"]]
    print(f"  {cols[:20]}")
    part_keys = [k["Name"] for k in t["Table"].get("PartitionKeys", [])]
    print(f"  partition keys: {part_keys}")
except Exception as e:
    print(f"  {e}")

# ── Count unique patients in the APCD 2016-2019 ──────────────────────────────
print("\n=== Count unique patients 2016-2019 ===")
# Try the partitioned table first (faster due to partition pruning)
for tbl, db in [("medical_partitioned", "medical_raw"), ("medical", "medical_raw")]:
    try:
        t = glue.get_table(DatabaseName=db, Name=tbl)
        all_cols = [c["Name"] for c in t["Table"]["StorageDescriptor"]["Columns"]]
        part_keys = [k["Name"] for k in t["Table"].get("PartitionKeys", [])]
        all_fields = all_cols + part_keys

        # Find the person key column name
        person_col = None
        for c in ["mi_person_key", "person_key", "memberid", "member_id",
                  "mi_person", "enrollee_id"]:
            if c.lower() in [f.lower() for f in all_fields]:
                person_col = c
                break

        # Find date/year column
        date_col = None
        year_col = None
        for c in ["service_year", "claim_year", "year", "service_date",
                  "admission_date", "from_date"]:
            if c.lower() in [f.lower() for f in all_fields]:
                if "year" in c.lower():
                    year_col = c
                else:
                    date_col = c
                break

        print(f"\n  {db}.{tbl}: person_col={person_col}, year_col={year_col}, date_col={date_col}")

        if person_col:
            if year_col:
                sql = (f"SELECT COUNT(DISTINCT {person_col}) AS n_patients "
                       f"FROM {db}.{tbl} "
                       f"WHERE {year_col} BETWEEN 2016 AND 2019")
            elif date_col:
                sql = (f"SELECT COUNT(DISTINCT {person_col}) AS n_patients "
                       f"FROM {db}.{tbl} "
                       f"WHERE YEAR({date_col}) BETWEEN 2016 AND 2019")
            else:
                sql = (f"SELECT COUNT(DISTINCT {person_col}) AS n_patients "
                       f"FROM {db}.{tbl}")

            print(f"  SQL: {sql[:120]}")
            rows = athena_query(sql, db=db)
            if rows and len(rows) > 1:
                n = rows[1]["Data"][0].get("VarCharValue", "?")
                print(f"  >>> TOTAL UNIQUE PATIENTS: {n}")
                break
    except Exception as e:
        print(f"  {tbl}: {e}")
