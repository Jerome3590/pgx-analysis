#!/usr/bin/env python3
"""
QA: verify one index date per target patient in cohort parquet files.

Checks (per cohort partition):
  - Distinct target patients vs distinct first_*_date among targets
  - Duplicate index dates (should be 0)
  - Optional: targets with >1 distinct index date in file (should be 0)

Usage:
  python 2_create_cohort/qa_index_date_uniqueness.py \\
    --path s3://pgxdatalake/gold/cohorts/cohort_name=non_opioid_ed/event_year=2019/age_band=65-74/cohort.parquet

  python 2_create_cohort/qa_index_date_uniqueness.py --cohort non_opioid_ed --year 2019 --age-band 65-74
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from py_helpers.constants import get_opioid_icd_sql_condition


def _cohort_s3_path(cohort_name: str, event_year: int, age_band: str) -> str:
    return (
        f"s3://pgxdatalake/gold/cohorts/cohort_name={cohort_name}/"
        f"event_year={event_year}/age_band={age_band}/cohort.parquet"
    )


def run_qa(parquet_path: str, cohort_name: str) -> dict:
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    is_opioid = cohort_name.strip().lower() == "opioid_ed"
    index_col = "first_opioid_ed_date" if is_opioid else "first_ed_non_opioid_date"

    summary = conn.sql(
        f"""
        SELECT
            CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS distinct_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) AS distinct_targets,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS BIGINT) AS distinct_controls,
            CAST(COUNT(*) AS BIGINT) AS total_event_rows
        FROM read_parquet('{parquet_path}')
        """
    ).fetchdf()
    row = summary.iloc[0]

    index_dup = conn.sql(
        f"""
        WITH targets AS (
            SELECT
                mi_person_key,
                MIN(CAST({index_col} AS DATE)) AS index_date_min,
                MAX(CAST({index_col} AS DATE)) AS index_date_max,
                COUNT(DISTINCT CAST({index_col} AS DATE)) AS n_distinct_index_dates
            FROM read_parquet('{parquet_path}')
            WHERE is_target_case = 1
              AND {index_col} IS NOT NULL
            GROUP BY mi_person_key
        )
        SELECT
            CAST(COUNT(*) AS BIGINT) AS target_patients_with_index,
            CAST(COUNT(CASE WHEN n_distinct_index_dates > 1 THEN 1 END) AS BIGINT) AS patients_multiple_index_dates,
            CAST(COUNT(CASE WHEN index_date_min <> index_date_max THEN 1 END) AS BIGINT) AS patients_min_max_mismatch
        FROM targets
        """
    ).fetchdf()
    idx = index_dup.iloc[0]

    distinct_index_dates = conn.sql(
        f"""
        SELECT CAST(COUNT(DISTINCT {index_col}) AS BIGINT) AS distinct_index_dates
        FROM read_parquet('{parquet_path}')
        WHERE is_target_case = 1 AND {index_col} IS NOT NULL
        """
    ).fetchone()[0]

    opioid_icd = get_opioid_icd_sql_condition()
    f1120_primary = conn.sql(
        f"""
        SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS n
        FROM read_parquet('{parquet_path}')
        WHERE primary_icd_diagnosis_code = 'F1120'
        """
    ).fetchone()[0]
    f1120_any_col = conn.sql(
        f"""
        SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS n
        FROM read_parquet('{parquet_path}')
        WHERE {opioid_icd}
        """
    ).fetchone()[0]

    return {
        "path": parquet_path,
        "cohort_name": cohort_name,
        "index_col": index_col,
        "distinct_patients": int(row["distinct_patients"]),
        "distinct_targets": int(row["distinct_targets"]),
        "distinct_controls": int(row["distinct_controls"]),
        "total_event_rows": int(row["total_event_rows"]),
        "target_patients_with_index": int(idx["target_patients_with_index"]),
        "patients_multiple_index_dates": int(idx["patients_multiple_index_dates"]),
        "distinct_index_dates_among_targets": int(distinct_index_dates),
        "f1120_distinct_primary_only": int(f1120_primary),
        "f1120_distinct_any_diagnosis_col": int(f1120_any_col),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QA cohort index-date uniqueness")
    parser.add_argument("--path", help="Local or s3:// path to cohort.parquet")
    parser.add_argument("--cohort", default="non_opioid_ed", choices=["non_opioid_ed", "opioid_ed"])
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--age-band", default="65-74")
    args = parser.parse_args()

    path = args.path or _cohort_s3_path(args.cohort, args.year, args.age_band)
    print(f"[QA] path: {path}")
    result = run_qa(path, args.cohort)

    print(f"[QA] cohort: {result['cohort_name']} | index column: {result['index_col']}")
    print(f"  Distinct patients: {result['distinct_patients']:,}")
    print(f"  Targets: {result['distinct_targets']:,} | Controls: {result['distinct_controls']:,}")
    print(f"  Event rows: {result['total_event_rows']:,}")
    print(f"  Targets with index date: {result['target_patients_with_index']:,}")
    print(f"  Distinct index dates (targets): {result['distinct_index_dates_among_targets']:,}")
    print(
        f"  Patients with >1 distinct index date: {result['patients_multiple_index_dates']:,} "
        f"(expect 0)"
    )
    # Multiple patients may share the same calendar index date; failure mode is >1 date per patient.
    ok = (
        result["patients_multiple_index_dates"] == 0
        and result["distinct_targets"] == result["target_patients_with_index"]
    )
    print(f"[QA] index uniqueness check: {'PASS' if ok else 'FAIL'}")
    if result["cohort_name"] != "opioid_ed":
        print(
            f"  F1120 distinct patients - primary only: {result['f1120_distinct_primary_only']:,} | "
            f"any of 10 ICD cols: {result['f1120_distinct_any_diagnosis_col']:,}"
        )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
