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
from helpers_1997_13.constants import OPIOID_ICD_CODES


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
            "cohort_name", "age_band", "is_target_case", "target"
        ]
        missing = [c for c in required if c not in cols]
        results["schema"]["required_columns_missing"] = missing
        
        # Cohort group validation: check cohort_name column exists and has correct values
        cohort_validation = {
            "cohort_name_exists": "cohort_name" in cols,
            "cohort_name_values": {},
            "cohort_name_mismatch": False,
            "is_control_only": False
        }
        
        if "cohort_name" in cols:
            try:
                # Get distinct cohort_name values
                cohort_names = conn.sql(
                    """
                    SELECT DISTINCT cohort_name, COUNT(*) as count
                    FROM cohort_qa
                    GROUP BY cohort_name
                    ORDER BY count DESC
                    """
                ).fetchall()
                cohort_validation["cohort_name_values"] = {name: count for name, count in cohort_names}
                
                # Expected cohort_name based on file being checked
                expected_cohort_name = "OPIOID_ED" if cohort_name == "opioid_ed" else "ED_NON_OPIOID"
                
                # Check if all records have the correct cohort_name
                mismatch_count = conn.sql(
                    f"""
                    SELECT COUNT(*) 
                    FROM cohort_qa 
                    WHERE cohort_name != '{expected_cohort_name}'
                    """
                ).fetchone()[0]
                cohort_validation["cohort_name_mismatch"] = mismatch_count > 0
                cohort_validation["expected_cohort_name"] = expected_cohort_name
                cohort_validation["mismatch_count"] = mismatch_count
                
                if mismatch_count > 0:
                    logger.warning(
                        f"⚠️  Cohort name mismatch: Found {mismatch_count} records with cohort_name != '{expected_cohort_name}'"
                    )
            except Exception as e:
                logger.debug(f"Could not validate cohort_name: {e}")
                cohort_validation["error"] = str(e)
        
        results["cohort_validation"] = cohort_validation
        
        # Partition validation: check age_band and event_year match partition
        partition_validation = {
            "age_band_match": True,
            "event_year_match": True,
            "age_band_mismatch_count": 0,
            "event_year_mismatch_count": 0
        }
        
        if "age_band" in cols:
            try:
                age_band_mismatch = conn.sql(
                    f"""
                    SELECT COUNT(*) 
                    FROM cohort_qa 
                    WHERE age_band != '{age_band}'
                    """
                ).fetchone()[0]
                partition_validation["age_band_mismatch_count"] = age_band_mismatch
                partition_validation["age_band_match"] = age_band_mismatch == 0
                if age_band_mismatch > 0:
                    logger.warning(f"⚠️  Age band mismatch: Found {age_band_mismatch} records with age_band != '{age_band}'")
            except Exception as e:
                logger.debug(f"Could not validate age_band: {e}")
        
        if "event_year" in cols:
            try:
                event_year_mismatch = conn.sql(
                    f"""
                    SELECT COUNT(*) 
                    FROM cohort_qa 
                    WHERE event_year != {event_year}
                    """
                ).fetchone()[0]
                partition_validation["event_year_mismatch_count"] = event_year_mismatch
                partition_validation["event_year_match"] = event_year_mismatch == 0
                if event_year_mismatch > 0:
                    logger.warning(f"⚠️  Event year mismatch: Found {event_year_mismatch} records with event_year != {event_year}")
            except Exception as e:
                logger.debug(f"Could not validate event_year: {e}")
        
        results["partition_validation"] = partition_validation

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

        # Control/target ratio and validation
        ctrl_ratio = None
        target_cases = 0
        control_cases = 0
        is_target_case_validation = {
            "column_exists": "is_target_case" in cols,
            "valid_values": True,
            "invalid_value_count": 0,
            "is_control_only": False
        }
        
        try:
            if "is_target_case" in cols:
                # Check for invalid values (should only be 0 or 1)
                invalid_values = conn.sql(
                    """
                    SELECT COUNT(*) 
                    FROM cohort_qa 
                    WHERE is_target_case NOT IN (0, 1) OR is_target_case IS NULL
                    """
                ).fetchone()[0]
                is_target_case_validation["invalid_value_count"] = invalid_values
                is_target_case_validation["valid_values"] = invalid_values == 0
                
                if invalid_values > 0:
                    logger.warning(f"⚠️  Invalid is_target_case values: Found {invalid_values} records with values not in (0, 1)")
                
                # Get target/control counts
                row2 = conn.sql(
                    """
                    SELECT
                      COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
                      COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
                    FROM cohort_qa
                    """
                ).fetchone()
                
                if row2:
                    target_cases = row2[0] or 0
                    control_cases = row2[1] or 0
                    
                    # Check if control-only cohort
                    is_target_case_validation["is_control_only"] = target_cases == 0 and control_cases > 0
                    
                    if target_cases > 0:
                        ctrl_ratio = control_cases / target_cases
                    elif control_cases > 0:
                        # Control-only cohort
                        ctrl_ratio = None
                        logger.info(f"ℹ️  Control-only cohort detected: {control_cases} controls, 0 targets")
                    else:
                        logger.warning("⚠️  Empty cohort: no target cases and no control cases")
                
                # Validate 5:1 ratio (with tolerance for control-only cohorts)
                if target_cases > 0:
                    ratio_validation = {
                        "expected_ratio": 5.0,
                        "actual_ratio": ctrl_ratio,
                        "ratio_within_tolerance": False,
                        "tolerance": 0.5  # Allow 4.5:1 to 5.5:1
                    }
                    if ctrl_ratio:
                        ratio_validation["ratio_within_tolerance"] = (
                            4.5 <= ctrl_ratio <= 5.5
                        )
                        if not ratio_validation["ratio_within_tolerance"]:
                            logger.warning(
                                f"⚠️  Control-to-target ratio out of tolerance: {ctrl_ratio:.2f}:1 "
                                f"(expected ~5:1, tolerance ±0.5)"
                            )
                    results["metrics"]["ratio_validation"] = ratio_validation
                    
        except Exception as e:
            logger.debug(f"Could not validate is_target_case or ratio: {e}")
            is_target_case_validation["error"] = str(e)

        results["metrics"]["control_to_target_ratio"] = ctrl_ratio
        results["metrics"]["target_cases"] = target_cases
        results["metrics"]["control_cases"] = control_cases
        results["is_target_case_validation"] = is_target_case_validation
        
        # Cohort separation validation: check that opioid patients don't appear in ed_non_opioid cohort
        cohort_separation_validation = {
            "opioid_patients_in_ed_non_opioid": 0,
            "separation_valid": True
        }
        
        if cohort_name == "ed_non_opioid" and "primary_icd_diagnosis_code" in cols:
            try:
                # Check for opioid ICD codes in ed_non_opioid cohort
                # Create a tuple string for SQL IN clause
                opioid_codes_tuple = "(" + ", ".join([f"'{code}'" for code in OPIOID_ICD_CODES]) + ")"
                opioid_patients = conn.sql(
                    f"""
                    SELECT COUNT(DISTINCT mi_person_key) 
                    FROM cohort_qa 
                    WHERE primary_icd_diagnosis_code IN {opioid_codes_tuple}
                    """
                ).fetchone()[0]
                cohort_separation_validation["opioid_patients_in_ed_non_opioid"] = opioid_patients
                cohort_separation_validation["separation_valid"] = opioid_patients == 0
                
                if opioid_patients > 0:
                    logger.error(
                        f"❌ Cohort separation violation: Found {opioid_patients} opioid patients "
                        f"in ed_non_opioid cohort (should be 0)"
                    )
            except Exception as e:
                logger.debug(f"Could not validate cohort separation: {e}")
                cohort_separation_validation["error"] = str(e)
        
        results["cohort_separation_validation"] = cohort_separation_validation

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

        # Determine overall status
        status_issues = []
        
        if missing:
            status_issues.append(f"Missing required columns: {missing}")
        
        if cohort_validation.get("cohort_name_mismatch", False):
            status_issues.append("Cohort name mismatch detected")
        
        if not partition_validation.get("age_band_match", True):
            status_issues.append("Age band mismatch detected")
        
        if not partition_validation.get("event_year_match", True):
            status_issues.append("Event year mismatch detected")
        
        if not is_target_case_validation.get("valid_values", True):
            status_issues.append("Invalid is_target_case values detected")
        
        if not cohort_separation_validation.get("separation_valid", True):
            status_issues.append("Cohort separation violation detected")
        
        if status_issues:
            results["status"] = "FAIL"
            results["status_issues"] = status_issues
            logger.error(f"❌ QA FAILED for {cohort_name} ({age_band}, {event_year}): {len(status_issues)} issue(s)")
        elif missing:
            results["status"] = "WARN"
        else:
            results["status"] = "PASS"
            logger.info(f"✅ QA PASSED for {cohort_name} ({age_band}, {event_year})")
        
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
                status = r.get('status', 'UNKNOWN')
                metrics = r.get('metrics', {})
                ratio = metrics.get('control_to_target_ratio')
                ratio_str = f"{ratio:.2f}:1" if ratio else "N/A (control-only)" if metrics.get('control_cases', 0) > 0 else "N/A"
                
                # Add validation status indicators
                cohort_val = r.get('cohort_validation', {})
                partition_val = r.get('partition_validation', {})
                separation_val = r.get('cohort_separation_validation', {})
                
                validation_flags = []
                if cohort_val.get('cohort_name_mismatch', False):
                    validation_flags.append("COHORT_NAME_MISMATCH")
                if not partition_val.get('age_band_match', True):
                    validation_flags.append("AGE_BAND_MISMATCH")
                if not partition_val.get('event_year_match', True):
                    validation_flags.append("EVENT_YEAR_MISMATCH")
                if not separation_val.get('separation_valid', True):
                    validation_flags.append("SEPARATION_VIOLATION")
                
                flags_str = f" [{', '.join(validation_flags)}]" if validation_flags else ""
                
                logger.info(
                    f"{band} {year} {name} → status={status}{flags_str} "
                    f"records={metrics.get('total_records')} "
                    f"patients={metrics.get('distinct_patients')} "
                    f"targets={metrics.get('target_cases', 0)} "
                    f"controls={metrics.get('control_cases', 0)} "
                    f"ratio={ratio_str}"
                )


if __name__ == "__main__":
    main()


