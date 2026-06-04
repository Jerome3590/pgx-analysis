#!/usr/bin/env python3
"""
FAERS time-to-onset QA on S3 parquet (therapy start → adverse-event onset).

Matches `1a_apcd_input_data/faers/faers.qmd` cleaned view: HO/DE outcomes,
0 < duration < 30 days, deduplicated per primaryid + caseid (shortest duration).

Usage (from project root):
  python py_helpers/faers_time_to_onset.py
  python py_helpers/faers_time_to_onset.py --base s3://pgxdatalake-backups/gold_backup_20251109T193811Z/faers
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import duckdb

DEFAULT_FAERS_S3_BASE = "s3://pgxdatalake-backups/gold_backup_20251109T193811Z/faers"


def connect_faers_s3(
    profile: str = "pgx",
    s3_region: str = "us-east-1",
) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with httpfs + AWS credentials for FAERS S3 reads."""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    if profile:
        try:
            conn.execute(f"CALL load_aws_credentials('{profile}');")
        except Exception as e:
            print(f"[warn] load_aws_credentials: {e}", file=sys.stderr)
    conn.execute(f"SET s3_region='{s3_region}';")
    return conn


def _resolve_faers_columns(col_names: dict[str, str]) -> dict[str, Optional[str]]:
    def col(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in col_names:
                return col_names[n.lower()]
        return None

    return {
        "duration": col("duration"),
        "start": col("start_dt", "drug_date"),
        "event": col("event_dt", "event_date"),
        "outc": col("outc_cod"),
        "primaryid": col("primaryid"),
        "caseid": col("caseid"),
    }


def _duration_expression(cols: dict[str, Optional[str]]) -> str:
    duration_col = cols["duration"]
    start_col = cols["start"]
    event_col = cols["event"]
    if duration_col:
        return f"CAST({duration_col} AS BIGINT)"
    if not start_col or not event_col:
        raise ValueError("Need duration column or start/drug_date + event/event_date")
    start_parse = f"""COALESCE(
        TRY_CAST({start_col} AS DATE),
        TRY_CAST(strptime(CAST({start_col} AS VARCHAR), '%Y%m%d') AS DATE),
        TRY_CAST(strptime(CAST({start_col} AS VARCHAR), '%Y-%m-%d') AS DATE)
    )"""
    event_parse = f"""COALESCE(
        TRY_CAST({event_col} AS DATE),
        TRY_CAST(strptime(CAST({event_col} AS VARCHAR), '%Y%m%d') AS DATE),
        TRY_CAST(strptime(CAST({event_col} AS VARCHAR), '%Y-%m-%d') AS DATE)
    )"""
    return f"CAST(datediff('day', {start_parse}, {event_parse}) AS BIGINT)"


def _partition_clause(cols: dict[str, Optional[str]]) -> tuple[str, str]:
    partition_parts = [c for c in [cols["primaryid"], cols["caseid"]] if c]
    if partition_parts:
        return (
            f"PARTITION BY {', '.join(partition_parts)}",
            ", ".join(partition_parts),
        )
    return "", "1"


def run_faers_time_to_onset_qa(
    base: str = DEFAULT_FAERS_S3_BASE,
    profile: str = "pgx",
    *,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> dict[str, Any]:
    """
    Query FAERS parquet on S3 and print summary tables.

    Returns dict with keys ``summary``, ``cleaned``, ``buckets`` (pandas DataFrames).
    """
    own_conn = conn is None
    if own_conn:
        conn = connect_faers_s3(profile=profile)

    glob_path = f"{base.rstrip('/')}/**/*.parquet"
    print(f"[path] {glob_path}")

    cols_df = conn.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{glob_path}') LIMIT 0"
    ).fetchdf()
    col_names = {str(c).lower(): str(c) for c in cols_df["column_name"]}
    print("[schema]", ", ".join(sorted(col_names.keys())))

    cols = _resolve_faers_columns(col_names)
    duration_expr = _duration_expression(cols)
    partition_sql, dedup_keys = _partition_clause(cols)

    outc_sql = ""
    if cols["outc"]:
        outc_sql = f"AND ({cols['outc']} IN ('HO', 'DE'))"

    sql_all_positive = f"""
    WITH base AS (
        SELECT {dedup_keys}, {duration_expr} AS duration
        FROM read_parquet('{glob_path}')
        WHERE {duration_expr} IS NOT NULL
          {outc_sql}
    ),
    deduped AS (
        SELECT duration
        FROM (
            SELECT
                {dedup_keys},
                duration,
                ROW_NUMBER() OVER ({partition_sql} ORDER BY duration ASC) AS rn
            FROM base
            WHERE duration > 0
        )
        WHERE rn = 1
    )
    SELECT
        CAST(COUNT(*) AS BIGINT) AS n_positive,
        CAST(COUNT(*) FILTER (WHERE duration <= 7) AS BIGINT) AS n_1_7,
        CAST(COUNT(*) FILTER (WHERE duration <= 14) AS BIGINT) AS n_1_14,
        CAST(COUNT(*) FILTER (WHERE duration <= 21) AS BIGINT) AS n_1_21,
        CAST(COUNT(*) FILTER (WHERE duration < 30) AS BIGINT) AS n_1_29,
        ROUND(100.0 * COUNT(*) FILTER (WHERE duration <= 7) / COUNT(*), 2) AS pct_le_7,
        ROUND(100.0 * COUNT(*) FILTER (WHERE duration <= 14) / COUNT(*), 2) AS pct_le_14,
        ROUND(100.0 * COUNT(*) FILTER (WHERE duration <= 21) / COUNT(*), 2) AS pct_le_21,
        ROUND(100.0 * COUNT(*) FILTER (WHERE duration < 30) / COUNT(*), 2) AS pct_lt_30,
        CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration) AS DOUBLE) AS median_days
    FROM deduped
    """

    sql_lt30 = f"""
    WITH base AS (
        SELECT {dedup_keys}, {duration_expr} AS duration
        FROM read_parquet('{glob_path}')
        WHERE {duration_expr} IS NOT NULL
          AND {duration_expr} > 0
          AND {duration_expr} < 30
          {outc_sql}
    ),
    deduped AS (
        SELECT duration
        FROM (
            SELECT
                {dedup_keys},
                duration,
                ROW_NUMBER() OVER ({partition_sql} ORDER BY duration ASC) AS rn
            FROM base
        )
        WHERE rn = 1
    )
    SELECT
        CAST(COUNT(*) AS BIGINT) AS n_faers_qmd_cleaned,
        CAST(COUNT(*) FILTER (WHERE duration <= 21) AS BIGINT) AS n_le_21,
        ROUND(100.0 * COUNT(*) FILTER (WHERE duration <= 21) / COUNT(*), 2) AS pct_le_21
    FROM deduped
    """

    sql_buckets = f"""
    WITH base AS (
        SELECT {dedup_keys}, {duration_expr} AS duration
        FROM read_parquet('{glob_path}')
        WHERE {duration_expr} IS NOT NULL
          AND {duration_expr} > 0
          AND {duration_expr} < 30
          {outc_sql}
    ),
    deduped AS (
        SELECT duration
        FROM (
            SELECT {dedup_keys}, duration,
                ROW_NUMBER() OVER ({partition_sql} ORDER BY duration ASC) AS rn
            FROM base
        ) WHERE rn = 1
    ),
    bucketed AS (
        SELECT
            CASE
                WHEN duration <= 7 THEN '1-7'
                WHEN duration <= 14 THEN '8-14'
                WHEN duration <= 21 THEN '15-21'
                ELSE '22-29'
            END AS bucket,
            duration
        FROM deduped
    )
    SELECT bucket,
           CAST(COUNT(*) AS BIGINT) AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM bucketed
    GROUP BY bucket
    ORDER BY MIN(duration)
    """

    summary_df = conn.sql(sql_all_positive).fetchdf()
    cleaned_df = conn.sql(sql_lt30).fetchdf()
    buckets_df = conn.sql(sql_buckets).fetchdf()

    print("\n[all positive durations, deduped per primaryid/caseid]")
    print(summary_df.to_string(index=False))

    print("\n[faers.qmd cleaned: 0 < duration < 30, deduped]")
    print(cleaned_df.to_string(index=False))

    print("\n[buckets, faers.qmd clean]")
    print(buckets_df.to_string(index=False))

    if own_conn:
        conn.close()

    return {
        "summary": summary_df,
        "cleaned": cleaned_df,
        "buckets": buckets_df,
        "glob_path": glob_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAERS time-to-onset QA (S3 parquet, faers.qmd-style filters)."
    )
    parser.add_argument("--base", default=DEFAULT_FAERS_S3_BASE)
    parser.add_argument("--profile", default="pgx")
    args = parser.parse_args()

    try:
        run_faers_time_to_onset_qa(base=args.base, profile=args.profile)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
