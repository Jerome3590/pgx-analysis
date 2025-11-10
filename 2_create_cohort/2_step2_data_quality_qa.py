#!/usr/bin/env python3
"""
step2_data_quality_qa.py

Final QA for cohort outputs produced by 2_create_cohort.
Validates that all steps were successfully applied and that final cohort
data structures (opioid_ed, ed_non_opioid) are consistent.

Usage:
  python step2_data_quality_qa.py --age-band 65-74 --event-year 2019 --cohorts both --save-results
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

# Project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3
from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
from helpers_1997_13.s3_utils import get_cohort_parquet_path, save_to_s3_json


def build_qa_output_path(cohort_name: str, age_band: str, event_year: int, bucket: str = "pgxdatalake") -> str:
    # Save under GOLD qa_results with cohort partitions
    return (
        f"s3://{bucket}/gold/qa_results/"
        f"cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}/qa.json"
    )


def describe_columns(conn, s3_path: str) -> List[str]:
    try:
        cols = conn.sql(
            f"""
            SELECT column_name
            FROM (
                DESCRIBE SELECT * FROM read_parquet('{s3_path}') LIMIT 0
            )
            """
        ).fetchall()
        return [c[0] for c in cols]
    except Exception:
        return []


def run_cohort_qa(conn, cohort_name: str, age_band: str, event_year: int, logger: logging.Logger) -> Dict[str, Any]:
    s3_path = get_cohort_parquet_path(cohort_name, age_band, event_year)

    results: Dict[str, Any] = {
        "cohort": cohort_name,
        "age_band": age_band,
        "event_year": event_year,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "UNKNOWN",
        "metrics": {},
        "schema": {
            "columns": [],
            "required_columns_missing": []
        }
    }

    logger.info(f"QA reading cohort parquet: {s3_path}")
    try:
        # Create view for cohort data
        conn.sql(f"CREATE OR REPLACE VIEW cohort_qa AS SELECT * FROM read_parquet('{s3_path}')")

        # Schema
        cols = describe_columns(conn, s3_path)
        results["schema"]["columns"] = cols

        required = [
            "mi_person_key", "event_date", "event_year", "drug_name",
        ]
        missing = [c for c in required if c not in cols]
        results["schema"]["required_columns_missing"] = missing

        # Build enriched view with normalized drug name for frequency analysis
        try:
            conn.sql(
                """
                CREATE OR REPLACE VIEW cohort_enriched AS
                SELECT
                  *,
                  COALESCE(NULLIF(standardized_drug_name, ''), LOWER(drug_name)) AS normalized_drug_name
                FROM cohort_qa
                """
            )
        except Exception as e:
            logger.debug(f"Could not create cohort_enriched (normalized drug name): {e}")

        # Core metrics
        row = conn.sql(
            """
            SELECT
              COUNT(*) as total_records,
              COUNT(DISTINCT mi_person_key) as distinct_patients,
              MIN(event_date) as earliest_event,
              MAX(event_date) as latest_event
            FROM cohort_qa
            """
        ).fetchone()

        results["metrics"].update({
            "total_records": row[0],
            "distinct_patients": row[1],
            "earliest_event": str(row[2]),
            "latest_event": str(row[3]),
        })

        # Control/target ratio if available
        ctrl_ratio = None
        try:
            row2 = conn.sql(
                """
                SELECT
                  COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
                  COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
                FROM cohort_qa
                """
            ).fetchone()
            if row2 and row2[0] and row2[0] > 0:
                ctrl_ratio = row2[1] / row2[0]
        except Exception:
            ctrl_ratio = None

        results["metrics"]["control_to_target_ratio"] = ctrl_ratio

        # Event type distribution (if present)
        try:
            dist = conn.sql(
                """
                SELECT event_type, COUNT(*) as count
                FROM cohort_qa
                GROUP BY event_type
                ORDER BY count DESC
                """
            ).fetchall()
            results["metrics"]["event_type_distribution"] = {k: v for k, v in dist}
        except Exception:
            pass

        # Frequency counts (top 50) for normalized drug names, ICD and CPT where present
        results["frequencies"] = {}

        # Normalized drug_name counts
        try:
            drug_freq = conn.sql(
                """
                SELECT normalized_drug_name, COUNT(*) as frequency
                FROM cohort_enriched
                WHERE normalized_drug_name IS NOT NULL AND normalized_drug_name <> ''
                GROUP BY normalized_drug_name
                ORDER BY frequency DESC
                LIMIT 50
                """
            ).fetchall()
            results["frequencies"]["normalized_drug_name"] = [
                {"value": d, "count": c} for d, c in drug_freq
            ]
        except Exception as e:
            logger.debug(f"Drug frequency unavailable: {e}")

        # ICD code counts for all ICD code columns present
        try:
            def is_icd_code_col(col: str) -> bool:
                cl = col.lower()
                if "icd" not in cl or "code" not in cl:
                    return False
                # prefer columns that explicitly end with _code
                if not cl.endswith("_code"):
                    return False
                # exclude descriptions/labels/ccs/rollups
                excluded_markers = ["desc", "description", "label", "ccs", "rollup"]
                return not any(m in cl for m in excluded_markers)

            icd_code_cols = [c for c in cols if is_icd_code_col(c)]
            for icd_col in icd_code_cols:
                try:
                    freq = conn.sql(
                        f"""
                        SELECT {icd_col} as code, COUNT(*) as frequency
                        FROM cohort_qa
                        WHERE {icd_col} IS NOT NULL AND {icd_col} <> ''
                        GROUP BY {icd_col}
                        ORDER BY frequency DESC
                        LIMIT 50
                        """
                    ).fetchall()
                    results["frequencies"][icd_col] = [{"value": d, "count": c} for d, c in freq]
                except Exception as e:
                    logger.debug(f"ICD frequency unavailable for {icd_col}: {e}")
        except Exception:
            pass

        # CPT/procedure code counts for all relevant CPT/procedure columns present (preferred feature signals)
        try:
            def is_cpt_code_col(col: str) -> bool:
                cl = col.lower()
                if not cl.endswith("_code"):
                    return False
                if not ("procedure" in cl or "cpt" in cl):
                    return False
                excluded_markers = ["desc", "description", "label", "name"]
                return not any(m in cl for m in excluded_markers)

            cpt_code_cols = [c for c in cols if is_cpt_code_col(c)]
            for cpt_col in cpt_code_cols:
                try:
                    freq = conn.sql(
                        f"""
                        SELECT {cpt_col} as code, COUNT(*) as frequency
                        FROM cohort_qa
                        WHERE {cpt_col} IS NOT NULL AND {cpt_col} <> ''
                        GROUP BY {cpt_col}
                        ORDER BY frequency DESC
                        LIMIT 50
                        """
                    ).fetchall()
                    results["frequencies"][cpt_col] = [{"value": d, "count": c} for d, c in freq]
                except Exception as e:
                    logger.debug(f"CPT/procedure frequency unavailable for {cpt_col}: {e}")
        except Exception:
            pass

        results["status"] = "PASS" if not missing else "WARN"
        return results

    except Exception as e:
        logger.error(f"Cohort QA failed for {cohort_name}: {e}")
        results["status"] = "ERROR"
        results["error"] = str(e)
        return results


def main():
    ap = argparse.ArgumentParser(description="Final QA for 2_create_cohort outputs")
    ap.add_argument("--age-band", help="Single age band to validate (e.g., 65-74)")
    ap.add_argument("--age-bands", help="Comma-separated list of age bands to validate")
    ap.add_argument("--all-age-bands", action="store_true", help="Validate all standard age bands")
    ap.add_argument("--event-year", type=int, help="Single event year to validate")
    ap.add_argument("--event-years", help="Comma-separated list of event years to validate (e.g., '2018,2019')")
    ap.add_argument("--all-event-years", action="store_true", help="Validate all standard event years")
    ap.add_argument("--cohorts", choices=["both", "opioid_ed", "ed_non_opioid"], default="both")
    ap.add_argument("--save-results", action="store_true", help="Save QA JSON to S3")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    # For logging, use a generic run_id if many bands
    run_band = args.age_band or ("all" if args.all_age_bands else (args.age_bands or "multi"))
    run_year = args.event_year if args.event_year is not None else ("all" if (args.all_event_years or args.event_years) else "unknown")
    logger, log_buffer = setup_logging("final_cohort_qa", str(run_band), str(run_year))
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    conn = create_simple_duckdb_connection(logger)

    cohorts = ["opioid_ed", "ed_non_opioid"] if args.cohorts == "both" else [args.cohorts]

    # Resolve age bands to process
    standard_bands = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-94", "95-114"]
    if args.all_age_bands:
        age_bands = standard_bands
    elif args.age_bands:
        age_bands = [b.strip() for b in args.age_bands.split(',') if b.strip()]
    elif args.age_band:
        age_bands = [args.age_band]
    else:
        raise SystemExit("Provide --age-band, --age-bands, or --all-age-bands")

    # Resolve event years to process
    standard_years = [2016, 2017, 2018, 2019, 2020]
    if args.all_event_years:
        event_years = standard_years
    elif args.event_years:
        try:
            event_years = [int(y.strip()) for y in args.event_years.split(',') if y.strip()]
        except ValueError:
            raise SystemExit("--event-years must be a comma-separated list of integers")
    elif args.event_year is not None:
        event_years = [args.event_year]
    else:
        raise SystemExit("Provide --event-year, --event-years, or --all-event-years")

    all_results: Dict[str, Any] = {
        "age_bands": age_bands,
        "event_years": event_years,
        "timestamp": datetime.utcnow().isoformat(),
        "results": {}
    }

    for band in age_bands:
        for year in event_years:
            for name in cohorts:
                res = run_cohort_qa(conn, name, band, year, logger)
                all_results["results"].setdefault(band, {}).setdefault(year, {})[name] = res

                if args.save_results:
                    out_path = build_qa_output_path(name, band, year)
                    try:
                        save_to_s3_json(res, out_path, logger)
                    except Exception as e:
                        logger.warning(f"Could not save QA JSON for {name} {band} {year}: {e}")

    # Save master log to S3
    try:
        save_logs_to_s3(log_buffer, "final_cohort_qa", args.age_band, args.event_year)
    except Exception:
        pass

    # Print a compact summary
    for band in age_bands:
        for year in event_years:
            for name in cohorts:
                r = all_results["results"][band][year][name]
                logger.info(
                    f"{band} {year} {name} → status={r.get('status')} records={r.get('metrics',{}).get('total_records')} "
                    f"patients={r.get('metrics',{}).get('distinct_patients')} ratio={r.get('metrics',{}).get('control_to_target_ratio')}"
                )


if __name__ == "__main__":
    main()


