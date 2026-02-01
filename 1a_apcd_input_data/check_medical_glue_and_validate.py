#!/usr/bin/env python3
"""
Check Glue tables exist for medical bronze/silver/gold; create crawlers/tables if missing.

Uses explicit databases per layer: bronze_medical, silver_medical, gold_medical.
S3 paths:
  s3://pgxdatalake/bronze/medical/         -> database bronze_medical
  s3://pgxdatalake/silver/imputed/medical_partitioned/ -> database silver_medical
  s3://pgxdatalake/gold/medical/          -> database gold_medical

Athena row-count QA (validate no information loss bronze→gold) lives in aws-pgx-setup/glue:
  python aws-pgx-setup/glue/validate_medical_row_coverage.py --athena-output s3://pgxdatalake/athena-query-results/ --credentials-dir /mnt/c/Projects

Runs locally (no EC2 required). Uses AWS profile "mushin" by default (--profile to override).
Omit --database to use per-layer databases; pass --database pgxdatalake for single-database (legacy).

Usage (from project root):
  python 1a_apcd_input_data/check_medical_glue_and_validate.py
  python 1a_apcd_input_data/check_medical_glue_and_validate.py --create-missing
  python 1a_apcd_input_data/check_medical_glue_and_validate.py --update-prefix
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import boto3  # noqa: E402

BUCKET = "pgxdatalake"
REGION = "us-east-1"
# (logical_name, s3_prefix, suggested_table_name, database_name)
# Databases are explicit: bronze_medical, silver_medical, gold_medical
MEDICAL_LAYERS = [
    ("bronze_medical", "bronze/medical/", "bronze_medical", "bronze_medical"),
    ("silver_medical_partitioned", "silver/imputed/medical_partitioned/", "silver_medical_partitioned", "silver_medical"),
    ("gold_medical", "gold/medical/", "gold_medical", "gold_medical"),
]
GLUE_DATABASE_DEFAULT = "pgxdatalake"
GLUE_ROLE_ARN_DEFAULT = "arn:aws:iam::535362115856:role/service-role/AWSGlueServiceRole-pgx-data-model"


def get_tables_for_database(glue, database: str):
    out = []
    try:
        paginator = glue.get_paginator("get_tables")
        for page in paginator.paginate(DatabaseName=database):
            for t in page.get("TableList", []):
                name = t.get("Name", "")
                loc = (t.get("StorageDescriptor") or {}).get("Location", "")
                out.append((name, loc))
    except glue.exceptions.EntityNotFoundException:
        pass
    return out


def table_covers_location(location: str, s3_prefix: str) -> bool:
    if not location:
        return False
    m = re.match(r"s3://[^/]+/(.+)", location)
    path = m.group(1) if m else location
    path = path.rstrip("/") + "/"
    return path == s3_prefix or path.startswith(s3_prefix.rstrip("/") + "/")


def find_table_for_prefix(glue, database: str, s3_prefix: str):
    for name, loc in get_tables_for_database(glue, database):
        if table_covers_location(loc, s3_prefix):
            return name
    return None


def list_crawlers_targeting(glue, s3_path: str):
    s3_path = s3_path.rstrip("/") + "/"
    out = []
    paginator = glue.get_paginator("get_crawlers")
    for page in paginator.paginate():
        for c in page.get("Crawlers", []):
            for t in (c.get("Targets") or {}).get("S3Targets") or []:
                p = (t.get("Path") or "").rstrip("/") + "/"
                if p == s3_path or s3_path.startswith(p) or p.startswith(s3_path):
                    out.append(c.get("Name"))
                    break
    return out


def create_crawler_for_medical_layer(
    glue,
    database: str,
    layer_name: str,
    table_name: str,
    s3_prefix: str,
    role_arn: str,
) -> str:
    """Create a Glue crawler for the given S3 prefix. TablePrefix = layer prefix only (bronze_, silver_, gold_)."""
    crawler_name = f"pgx_medical_{layer_name}"
    s3_path = f"s3://{BUCKET}/{s3_prefix}"
    layer_prefix = layer_name.split("_")[0] + "_"
    config = {"Version": 1.0, "Grouping": {"TableGroupingPolicy": "CombineCompatibleSchemas"}}
    try:
        glue.create_crawler(
            Name=crawler_name,
            Role=role_arn,
            DatabaseName=database,
            Description=f"Medical {layer_name}",
            Targets={"S3Targets": [{"Path": s3_path}]},
            TablePrefix=layer_prefix,
            SchemaChangePolicy={
                "UpdateBehavior": "UPDATE_IN_DATABASE",
                "DeleteBehavior": "DEPRECATE_IN_DATABASE",
            },
            RecrawlPolicy={"RecrawlBehavior": "CRAWL_EVERYTHING"},
            Configuration=json.dumps(config),
        )
        return crawler_name
    except glue.exceptions.AlreadyExistsException:
        return crawler_name


def update_crawler_table_prefix(glue, crawler_name: str, new_prefix: str) -> bool:
    try:
        r = glue.get_crawler(Name=crawler_name)
        c = r["Crawler"]
        state = c.get("State", "")
        if state == "RUNNING":
            glue.stop_crawler(Name=crawler_name)
            for _ in range(60):
                r2 = glue.get_crawler(Name=crawler_name)
                if r2["Crawler"].get("State") == "READY":
                    break
                time.sleep(5)
            c = glue.get_crawler(Name=crawler_name)["Crawler"]
        glue.update_crawler(
            Name=crawler_name,
            Role=c["Role"],
            DatabaseName=c["DatabaseName"],
            Description=c.get("Description", ""),
            Targets=c["Targets"],
            TablePrefix=new_prefix,
            SchemaChangePolicy=c.get("SchemaChangePolicy", {}),
            RecrawlPolicy=c.get("RecrawlPolicy", {}),
            Configuration=c.get("Configuration", "{}"),
        )
        return True
    except Exception:
        return False


def run_crawler_and_wait(glue, crawler_name: str, timeout_sec: int = 600):
    glue.start_crawler(Name=crawler_name)
    start = time.time()
    while time.time() - start < timeout_sec:
        r = glue.get_crawler(Name=crawler_name)
        state = r["Crawler"]["State"]
        if state == "READY":
            return True
        if state in ("FAILED", "STOPPING"):
            return False
        time.sleep(10)
    return False


def run_athena_count(athena, database: str, table: str, output_location: str) -> Optional[int]:
    q = f'SELECT COUNT(*) AS cnt FROM "{table}"'  # noqa: S608 table from Glue
    try:
        resp = athena.start_query_execution(
            QueryString=q,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": output_location},
        )
        qid = resp["QueryExecutionId"]
        for _ in range(120):
            r = athena.get_query_execution(QueryExecutionId=qid)
            state = r["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                rr = athena.get_query_results(QueryExecutionId=qid)
                rows = rr.get("ResultSet", {}).get("Rows", [])
                if len(rows) >= 2:
                    val = rows[1].get("Data", [{}])[0].get("VarCharValue", "0")
                    return int(val)
                return None
            if state in ("FAILED", "CANCELLED"):
                return None
            time.sleep(2)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Check Glue tables for medical bronze/silver/gold; create crawlers if missing; validate row counts."
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Glue database name (optional; if unset, uses per-layer: bronze_medical, silver_medical, gold_medical)",
    )
    parser.add_argument("--create-missing", action="store_true", help="Create crawlers and run them if table missing")
    parser.add_argument("--update-prefix", action="store_true", help="Update crawlers' TablePrefix to layer-only (bronze_, silver_, gold_)")
    parser.add_argument("--role-arn", default=GLUE_ROLE_ARN_DEFAULT, help="Glue crawler IAM role ARN")
    parser.add_argument("--profile", default="mushin", help="AWS profile name")
    parser.add_argument("--credentials-dir", default=None, metavar="DIR", help="Directory with .aws/credentials (e.g. /mnt/c/Projects)")
    args = parser.parse_args()

    if args.credentials_dir:
        base = Path(args.credentials_dir).resolve()
        creds = base / ".aws" / "credentials"
        if not creds.exists():
            creds = base / "credentials"
        config = base / ".aws" / "config"
        if not config.exists():
            config = base / "config"
        if creds.exists():
            os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(creds)
        if config.exists():
            os.environ["AWS_CONFIG_FILE"] = str(config)

    session = boto3.Session(profile_name=args.profile, region_name=REGION)
    glue = session.client("glue")
    use_per_layer_db = args.database is None
    if use_per_layer_db:
        databases = list({layer[3] for layer in MEDICAL_LAYERS})
        print(f"Using per-layer databases: {databases}")
    else:
        databases = [args.database]

    for database in databases:
        try:
            glue.get_database(Name=database)
        except glue.exceptions.EntityNotFoundException:
            print(f"Database '{database}' does not exist. Create it in Glue console or run:")
            print(f"  aws glue create-database --database-input '{{\"Name\": \"{database}\"}}'")
            sys.exit(1)

    if args.update_prefix:
        print("Updating crawler TablePrefix to layer-only (bronze_, silver_, gold_)...")
        for logical_name, _s3_prefix, _suggested_table, _db in MEDICAL_LAYERS:
            crawler_name = f"pgx_medical_{logical_name}"
            layer_prefix = logical_name.split("_")[0] + "_"
            try:
                glue.get_crawler(Name=crawler_name)
                if update_crawler_table_prefix(glue, crawler_name, layer_prefix):
                    print(f"  [OK] {crawler_name}: TablePrefix -> '{layer_prefix}'")
                else:
                    print(f"  [FAIL] {crawler_name}: update failed")
            except glue.exceptions.EntityNotFoundException:
                print(f"  [SKIP] {crawler_name}: crawler does not exist")
        print("Re-run crawlers to create tables with correct names. You may drop old doubled tables in Glue.\n")

    layer_to_table = {}
    layer_to_database = {}
    for logical_name, s3_prefix, suggested_table, database_name in MEDICAL_LAYERS:
        db = database_name if use_per_layer_db else args.database
        layer_to_database[logical_name] = db
        s3_path = f"s3://{BUCKET}/{s3_prefix}"
        table_name = find_table_for_prefix(glue, db, s3_prefix)
        if table_name:
            layer_to_table[logical_name] = table_name
            print(f"  [OK] {logical_name}: database '{db}' table '{table_name}' -> {s3_path}")
        else:
            layer_to_table[logical_name] = None
            print(f"  [MISSING] {logical_name}: no table in database '{db}' for {s3_path}")

    if not args.validate_only and any(v is None for v in layer_to_table.values()):
        for logical_name, s3_prefix, suggested_table, database_name in MEDICAL_LAYERS:
            if layer_to_table.get(logical_name):
                continue
            db = database_name if use_per_layer_db else args.database
            s3_path = f"s3://{BUCKET}/{s3_prefix}"
            crawlers = list_crawlers_targeting(glue, s3_path)
            if crawlers:
                print(f"  Found crawler(s) for {logical_name}: {crawlers}. Run them to create table.")
                if args.create_missing:
                    for cn in crawlers:
                        print(f"  Starting crawler {cn}...")
                        if run_crawler_and_wait(glue, cn):
                            table_name = find_table_for_prefix(glue, db, s3_prefix)
                            if table_name:
                                layer_to_table[logical_name] = table_name
                        else:
                            print(f"  Crawler {cn} failed or timed out.")
            elif args.create_missing:
                crawler_name = create_crawler_for_medical_layer(glue, db, logical_name, suggested_table, s3_prefix, args.role_arn)
                print(f"  Created crawler {crawler_name}; starting...")
                if run_crawler_and_wait(glue, crawler_name):
                    table_name = find_table_for_prefix(glue, db, s3_prefix)
                    if table_name:
                        layer_to_table[logical_name] = table_name
                        print(f"  Table '{table_name}' created.")
                else:
                    print(f"  Crawler {crawler_name} failed or timed out.")
            else:
                print(f"  To create table: run with --create-missing (will create crawler for {s3_path})")

    # Athena row-count QA lives in aws-pgx-setup/glue (keeps pipeline focused on workflow)
    print("\nAthena row-count QA: python aws-pgx-setup/glue/validate_medical_row_coverage.py --athena-output s3://pgxdatalake/athena-query-results/ [--credentials-dir /mnt/c/Projects]")


if __name__ == "__main__":
    main()
