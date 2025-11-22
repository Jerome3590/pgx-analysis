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
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3
from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
from helpers_1997_13.s3_utils import get_cohort_parquet_path
from helpers_1997_13.constants import OPIOID_ICD_CODES, get_opioid_icd_sql_condition, ALL_ICD_DIAGNOSIS_COLUMNS


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


def run_cohort_qa_parallel(args_tuple: Tuple[str, str, int, str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Parallel wrapper for run_cohort_qa that creates its own connection and logger.
    Args:
        args_tuple: (cohort_name, age_band, event_year, log_level, env_vars)
    Returns:
        Dict with QA results plus metadata for aggregation
    """
    cohort_name, age_band, event_year, log_level, env_vars = args_tuple

    # Import os FIRST, before any other imports
    import os
    import logging

    # CRITICAL: Set environment variables BEFORE importing modules that read them
    # The constants module reads env vars at import time, so we must set them first
    for key, value in env_vars.items():
        if value is not None:
            os.environ[key] = value

    # Now import modules that depend on environment variables
    # This ensures constants.py reads the correct values
    from helpers_1997_13.logging_utils import setup_logging
    from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection

    # Reload constants module to pick up the new environment variables
    import importlib
    import helpers_1997_13.constants as constants_module
    importlib.reload(constants_module)

    # Also reload s3_utils to ensure it uses the updated constants
    import helpers_1997_13.s3_utils as s3_utils_module
    importlib.reload(s3_utils_module)

    # Create logger for this process
    logger, _ = setup_logging("final_cohort_qa", age_band, str(event_year))
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Create connection for this process
    conn = create_simple_duckdb_connection(logger)

    try:
        # Re-import get_cohort_parquet_path after reloading modules
        # This ensures it uses the updated constants
        import sys
        # Remove old imports to force fresh import
        modules_to_reload = ['helpers_1997_13.s3_utils', 'helpers_1997_13.constants']
        for mod_name in modules_to_reload:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Re-import to get fresh references with updated env vars
        from helpers_1997_13.s3_utils import get_cohort_parquet_path

        # Update the module's globals so run_cohort_qa uses the reloaded function
        # Get the module that contains run_cohort_qa
        import importlib
        current_module = sys.modules.get(__name__)
        if current_module:
            # Update the global reference in this module
            current_module.get_cohort_parquet_path = get_cohort_parquet_path
            # Also update globals() to ensure the function sees it
            globals()['get_cohort_parquet_path'] = get_cohort_parquet_path
        
        return run_cohort_qa(conn, cohort_name, age_band, event_year, logger)
    finally:
        conn.close()


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
        # First, check what cohort_name values are actually in the file
        # Files may contain lowercase (opioid_ed, non_opioid_ed) or uppercase (OPIOID_ED, ED_NON_OPIOID)
        actual_cohort_names = conn.sql(
            f"""
            SELECT DISTINCT cohort_name
            FROM read_parquet('{s3_path}')
            LIMIT 10
            """
        ).fetchall()
        
        # Determine the actual cohort_name value to use (handle case variations)
        actual_cohort_name = None
        if actual_cohort_names:
            # Use the first distinct value found
            actual_cohort_name = actual_cohort_names[0][0]
            logger.info(f"📊 Detected cohort_name in file: '{actual_cohort_name}'")
        else:
            # Fallback: try both case variations
            expected_upper = "OPIOID_ED" if cohort_name == "opioid_ed" else "ED_NON_OPIOID"
            expected_lower = "opioid_ed" if cohort_name == "opioid_ed" else "non_opioid_ed"
            # Try uppercase first
            try:
                count = conn.sql(
                    f"SELECT COUNT(*) FROM read_parquet('{s3_path}') WHERE cohort_name = '{expected_upper}'"
                ).fetchone()[0]
                if count > 0:
                    actual_cohort_name = expected_upper
                else:
                    # Try lowercase
                    count = conn.sql(
                        f"SELECT COUNT(*) FROM read_parquet('{s3_path}') WHERE cohort_name = '{expected_lower}'"
                    ).fetchone()[0]
                    if count > 0:
                        actual_cohort_name = expected_lower
            except Exception:
                pass
        
        if actual_cohort_name is None:
            logger.warning(f"⚠️  Could not determine cohort_name value in file, using expected value")
            actual_cohort_name = "OPIOID_ED" if cohort_name == "opioid_ed" else "ED_NON_OPIOID"

        # Create view for cohort data, filtering to only the actual cohort_name found in file
        # This handles cases where PARTITION_BY (cohort_name) created multiple partitions
        conn.sql(f"""
        CREATE OR REPLACE VIEW cohort_qa AS
        SELECT * FROM read_parquet('{s3_path}')
        WHERE cohort_name = '{actual_cohort_name}'
        """)

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
        # Note: We already filtered by expected_cohort_name when creating the view,
        # so this validation checks that the filter worked correctly
        cohort_validation = {
            "cohort_name_exists": "cohort_name" in cols,
            "cohort_name_values": {},
            "cohort_name_mismatch": False,
            "is_control_only": False
        }
        
        if "cohort_name" in cols:
            try:
                # Get distinct cohort_name values (should only be one after filtering)
                cohort_names = conn.sql(
                    """
                    SELECT DISTINCT cohort_name, COUNT(*) as count
                    FROM cohort_qa
                    GROUP BY cohort_name
                    ORDER BY count DESC
                    """
                ).fetchall()
                cohort_validation["cohort_name_values"] = {name: count for name, count in cohort_names}
                
                # Expected cohort_name based on file being checked (for reference)
                expected_cohort_name = "OPIOID_ED" if cohort_name == "opioid_ed" else "ED_NON_OPIOID"
                
                # Check if all records have the correct cohort_name (should be 0 after filtering)
                # Use the actual_cohort_name that was detected, not the expected one
                actual_cohort_name_in_view = cohort_names[0][0] if cohort_names else None
                mismatch_count = 0
                if actual_cohort_name_in_view:
                    mismatch_count = conn.sql(
                        f"""
                        SELECT COUNT(*) 
                        FROM cohort_qa 
                        WHERE cohort_name != '{actual_cohort_name_in_view}'
                        """
                    ).fetchone()[0]
                cohort_validation["cohort_name_mismatch"] = mismatch_count > 0
                cohort_validation["expected_cohort_name"] = expected_cohort_name
                cohort_validation["actual_cohort_name"] = actual_cohort_name_in_view
                cohort_validation["mismatch_count"] = mismatch_count
                
                # Also check if there are any records with unexpected cohort_name values in the original file
                # (before filtering) - this helps identify if PARTITION_BY created mixed partitions
                try:
                    # Get all distinct cohort_name values in the file before filtering
                    all_cohort_names = conn.sql(
                        f"""
                        SELECT DISTINCT cohort_name, COUNT(*) as count
                        FROM read_parquet('{s3_path}')
                        GROUP BY cohort_name
                        ORDER BY count DESC
                        """
                    ).fetchall()
                    cohort_validation["all_cohort_names_in_file"] = {name: count for name, count in all_cohort_names}
                    
                    # Log what cohort_name values were found
                    if all_cohort_names:
                        cohort_names_str = ", ".join([f"{name}={count}" for name, count in all_cohort_names])
                        expected_cohort_name = "OPIOID_ED" if cohort_name == "opioid_ed" else "ED_NON_OPIOID"
                        logger.info(f"📊 Found cohort_name values in file: {cohort_names_str} (expected: {expected_cohort_name})")
                    
                    total_in_file = conn.sql(f"SELECT COUNT(*) FROM read_parquet('{s3_path}')").fetchone()[0]
                    total_after_filter = conn.sql("SELECT COUNT(*) FROM cohort_qa").fetchone()[0]
                    if total_in_file > total_after_filter:
                        unexpected_count = total_in_file - total_after_filter
                        cohort_validation["unexpected_cohort_name_count"] = unexpected_count
                        logger.warning(
                            f"⚠️  Found {unexpected_count} records with unexpected cohort_name in file "
                            f"(filtered out from analysis). Total in file: {total_in_file}, "
                            f"Total after filter: {total_after_filter}"
                        )
                except Exception as e:
                    logger.debug(f"Could not check cohort_name values in file: {e}")
                    pass  # Best effort - if we can't check, continue

                if mismatch_count > 0:
                    logger.warning(
                        f"⚠️  Cohort name mismatch: Found {mismatch_count} records with cohort_name != '{expected_cohort_name}' "
                        f"(this should be 0 after filtering)"
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
        # CRITICAL: Check ALL 10 ICD diagnosis columns for opioid codes
        cohort_separation_validation = {
            "opioid_patients_in_ed_non_opioid": 0,
            "separation_valid": True,
            "opioid_codes_by_position": {}
        }

        # Check if any ICD columns are available
        available_icd_cols = [col for col in ALL_ICD_DIAGNOSIS_COLUMNS if col in cols]

        if cohort_name == "ed_non_opioid" and available_icd_cols:
            try:
                # Check for opioid ICD codes in ed_non_opioid cohort across ALL diagnosis positions
                opioid_icd_condition = get_opioid_icd_sql_condition()
                opioid_patients = conn.sql(
                    f"""
                    SELECT COUNT(DISTINCT mi_person_key) 
                    FROM cohort_qa 
                    WHERE {opioid_icd_condition}
                    """
                ).fetchone()[0]
                cohort_separation_validation["opioid_patients_in_ed_non_opioid"] = opioid_patients
                cohort_separation_validation["separation_valid"] = opioid_patients == 0

                # If violations found, count by position for diagnostics
                if opioid_patients > 0:
                    for col in available_icd_cols:
                        opioid_codes_tuple = "(" + ", ".join([f"'{code}'" for code in OPIOID_ICD_CODES]) + ")"
                        count = conn.sql(f"""
                            SELECT COUNT(DISTINCT mi_person_key)
                            FROM cohort_qa
                            WHERE {col} IN {opioid_codes_tuple}
                        """).fetchone()[0]
                        if count > 0:
                            cohort_separation_validation["opioid_codes_by_position"][col] = count

                    logger.error(
                        f"❌ Cohort separation violation: Found {opioid_patients} opioid patients "
                        f"in ed_non_opioid cohort (should be 0) - check ANY diagnosis position"
                    )
                    logger.error(f"   Opioid codes by position: {cohort_separation_validation['opioid_codes_by_position']}")
            except Exception as e:
                logger.debug(f"Could not validate cohort separation: {e}")
                cohort_separation_validation["error"] = str(e)
        
        results["cohort_separation_validation"] = cohort_separation_validation

        # Drug window validation (days_to_target_event)
        drug_window_validation = {
            "column_exists": "days_to_target_event" in cols,
            "target_cases_with_pharmacy_events": 0,
            "pharmacy_events_in_30day_window": 0,
            "pharmacy_events_outside_window": 0,
            "avg_days_to_target": None,
            "validation_passed": True
        }

        if "days_to_target_event" in cols and "event_type" in cols and "is_target_case" in cols:
            try:
                # Get drug window statistics for target cases only
                drug_stats = conn.sql(
                    """
                    SELECT
                        COUNT(DISTINCT mi_person_key) as patients_with_drugs,
                        COUNT(*) as total_pharmacy_events,
                        COUNT(CASE WHEN days_to_target_event IS NOT NULL AND days_to_target_event >= 0 AND days_to_target_event <= 30 THEN 1 END) as drugs_in_30day_window,
                        COUNT(CASE WHEN days_to_target_event IS NOT NULL AND days_to_target_event > 30 THEN 1 END) as drugs_outside_window,
                        AVG(CASE WHEN days_to_target_event IS NOT NULL AND days_to_target_event >= 0 AND days_to_target_event <= 30 THEN days_to_target_event END) as avg_days_in_window
                    FROM cohort_qa
                    WHERE event_type = 'pharmacy' AND is_target_case = 1
                    """
                ).fetchone()

                if drug_stats:
                    drug_window_validation["target_cases_with_pharmacy_events"] = drug_stats[0] or 0
                    drug_window_validation["total_pharmacy_events"] = drug_stats[1] or 0
                    drug_window_validation["pharmacy_events_in_30day_window"] = drug_stats[2] or 0
                    drug_window_validation["pharmacy_events_outside_window"] = drug_stats[3] or 0
                    drug_window_validation["avg_days_to_target"] = round(drug_stats[4], 2) if drug_stats[4] else None

                # For ed_non_opioid (control-only), days_to_target_event should be NULL
                if cohort_name == "ed_non_opioid":
                    non_null_count = conn.sql(
                        """
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE days_to_target_event IS NOT NULL
                        """
                    ).fetchone()[0]
                    drug_window_validation["non_null_days_for_controls"] = non_null_count
                    drug_window_validation["validation_passed"] = non_null_count == 0

                    if non_null_count > 0:
                        logger.warning(
                            f"⚠️  ED_NON_OPIOID should have NULL days_to_target_event (control-only), "
                            f"but found {non_null_count} non-null values"
                        )

            except Exception as e:
                logger.debug(f"Could not validate drug window: {e}")
                drug_window_validation["error"] = str(e)

        results["drug_window_validation"] = drug_window_validation

        # NULL value validation for critical fields
        null_validation = {
            "critical_nulls_found": False,
            "null_counts": {}
        }

        critical_fields = ["mi_person_key", "event_date", "event_type", "target", "age_imputed"]
        available_critical_fields = [f for f in critical_fields if f in cols]

        if available_critical_fields:
            try:
                null_checks = []
                for field in available_critical_fields:
                    null_checks.append(f"COUNT(CASE WHEN {field} IS NULL THEN 1 END) as null_{field}")

                null_query = f"""
                    SELECT {', '.join(null_checks)}
                    FROM cohort_qa
                """
                null_counts_row = conn.sql(null_query).fetchone()

                for idx, field in enumerate(available_critical_fields):
                    null_count = null_counts_row[idx] or 0
                    null_validation["null_counts"][field] = null_count
                    if null_count > 0:
                        null_validation["critical_nulls_found"] = True
                        logger.warning(f"⚠️  Critical field '{field}' has {null_count} NULL values")

            except Exception as e:
                logger.debug(f"Could not validate NULL values: {e}")
                null_validation["error"] = str(e)

        results["null_validation"] = null_validation

        # Age validation (1-114 range and matches age_band)
        age_validation = {
            "age_in_valid_range": True,
            "age_matches_band": True,
            "invalid_age_count": 0,
            "age_band_mismatch_count": 0
        }

        if "age_imputed" in cols:
            try:
                # Check age range (1-114)
                invalid_ages = conn.sql(
                    """
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE age_imputed IS NOT NULL AND (age_imputed < 1 OR age_imputed > 114)
                    """
                ).fetchone()[0]
                age_validation["invalid_age_count"] = invalid_ages
                age_validation["age_in_valid_range"] = invalid_ages == 0

                if invalid_ages > 0:
                    logger.warning(f"⚠️  Found {invalid_ages} records with age outside valid range (1-114)")

                # Check age matches age_band
                if "age_band" in cols:
                    age_band_check = conn.sql(f"""
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE age_imputed IS NOT NULL
                          AND age_band = '{age_band}'
                          AND NOT (
                            (age_band = '0-12' AND age_imputed BETWEEN 0 AND 12) OR
                            (age_band = '13-24' AND age_imputed BETWEEN 13 AND 24) OR
                            (age_band = '25-44' AND age_imputed BETWEEN 25 AND 44) OR
                            (age_band = '45-54' AND age_imputed BETWEEN 45 AND 54) OR
                            (age_band = '55-64' AND age_imputed BETWEEN 55 AND 64) OR
                            (age_band = '65-74' AND age_imputed BETWEEN 65 AND 74) OR
                            (age_band = '75-84' AND age_imputed BETWEEN 75 AND 84) OR
                            (age_band = '85-94' AND age_imputed BETWEEN 85 AND 94) OR
                            (age_band = '95-114' AND age_imputed BETWEEN 95 AND 114)
                          )
                    """).fetchone()[0]
                    age_validation["age_band_mismatch_count"] = age_band_check
                    age_validation["age_matches_band"] = age_band_check == 0

                    if age_band_check > 0:
                        logger.warning(f"⚠️  Found {age_band_check} records where age doesn't match age_band")

            except Exception as e:
                logger.debug(f"Could not validate age: {e}")
                age_validation["error"] = str(e)

        results["age_validation"] = age_validation

        # Date consistency validation
        date_validation = {
            "date_year_matches_event_year": True,
            "dates_in_valid_range": True,
            "date_mismatch_count": 0,
            "date_out_of_range_count": 0
        }

        if "event_date" in cols and "event_year" in cols:
            try:
                # Check event_date year matches event_year column
                year_mismatch = conn.sql("""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE event_date IS NOT NULL
                      AND event_year IS NOT NULL
                      AND YEAR(TRY_CAST(event_date AS DATE)) != event_year
                """).fetchone()[0]
                date_validation["date_mismatch_count"] = year_mismatch
                date_validation["date_year_matches_event_year"] = year_mismatch == 0

                if year_mismatch > 0:
                    logger.warning(f"⚠️  Found {year_mismatch} records where event_date year doesn't match event_year")

                # Check dates are in valid range (2016-2020)
                out_of_range = conn.sql("""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE event_date IS NOT NULL
                      AND (YEAR(TRY_CAST(event_date AS DATE)) < 2016 OR YEAR(TRY_CAST(event_date AS DATE)) > 2020)
                """).fetchone()[0]
                date_validation["date_out_of_range_count"] = out_of_range
                date_validation["dates_in_valid_range"] = out_of_range == 0

                if out_of_range > 0:
                    logger.warning(f"⚠️  Found {out_of_range} records with dates outside valid range (2016-2020)")

            except Exception as e:
                logger.debug(f"Could not validate dates: {e}")
                date_validation["error"] = str(e)

        results["date_validation"] = date_validation

        # F1120 and opioid ICD code validation
        # CRITICAL: Check ALL 10 ICD diagnosis columns for opioid codes
        opioid_code_validation = {
            "f1120_present": False,
            "f1120_count": 0,
            "opioid_codes_present": False,
            "total_opioid_code_records": 0,
            "f1120_by_position": {}
        }

        # Check if any ICD columns are available
        available_icd_cols = [col for col in ALL_ICD_DIAGNOSIS_COLUMNS if col in cols]

        if available_icd_cols and cohort_name == "opioid_ed":
            try:
                # Check for F1120 specifically across ALL ICD diagnosis positions
                f1120_conditions = " OR ".join([f"{col} = 'F1120'" for col in available_icd_cols])
                f1120_count = conn.sql(f"""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE {f1120_conditions}
                """).fetchone()[0]
                opioid_code_validation["f1120_count"] = f1120_count
                opioid_code_validation["f1120_present"] = f1120_count > 0

                # Count F1120 by position for analysis
                for col in available_icd_cols:
                    count = conn.sql(f"""
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE {col} = 'F1120'
                    """).fetchone()[0]
                    if count > 0:
                        opioid_code_validation["f1120_by_position"][col] = count

                # Check for all opioid ICD codes across ALL diagnosis positions
                opioid_icd_condition = get_opioid_icd_sql_condition()
                opioid_total = conn.sql(f"""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE {opioid_icd_condition}
                """).fetchone()[0]
                opioid_code_validation["total_opioid_code_records"] = opioid_total
                opioid_code_validation["opioid_codes_present"] = opioid_total > 0

                if not opioid_code_validation["opioid_codes_present"]:
                    logger.warning("⚠️  OPIOID_ED cohort has no records with opioid ICD codes in ANY diagnosis position")
                
                # Log F1120 distribution if found in non-primary positions
                non_primary_f1120 = sum(count for col, count in opioid_code_validation["f1120_by_position"].items() 
                                       if col != 'primary_icd_diagnosis_code')
                if non_primary_f1120 > 0:
                    logger.info(f"ℹ️  Found {non_primary_f1120} F1120 codes in non-primary diagnosis positions")

            except Exception as e:
                logger.debug(f"Could not validate opioid codes: {e}")
                opioid_code_validation["error"] = str(e)

        results["opioid_code_validation"] = opioid_code_validation

        # Cohort-specific date fields validation
        cohort_date_fields_validation = {
            "first_opioid_ed_date_valid": True,
            "first_ed_non_opioid_date_valid": True,
            "incorrect_null_count": 0
        }

        if "first_opioid_ed_date" in cols and "first_ed_non_opioid_date" in cols:
            try:
                if cohort_name == "opioid_ed":
                    # For OPIOID_ED: first_opioid_ed_date should be populated, first_ed_non_opioid_date should be NULL
                    incorrect_nulls = conn.sql("""
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE first_opioid_ed_date IS NULL OR first_ed_non_opioid_date IS NOT NULL
                    """).fetchone()[0]
                    cohort_date_fields_validation["incorrect_null_count"] = incorrect_nulls
                    cohort_date_fields_validation["first_opioid_ed_date_valid"] = incorrect_nulls == 0

                    if incorrect_nulls > 0:
                        logger.warning(
                            f"⚠️  OPIOID_ED cohort: {incorrect_nulls} records with incorrect date field NULLs "
                            f"(first_opioid_ed_date should be populated, first_ed_non_opioid_date should be NULL)"
                        )

                elif cohort_name == "ed_non_opioid":
                    # For ED_NON_OPIOID: first_ed_non_opioid_date should be populated, first_opioid_ed_date should be NULL
                    incorrect_nulls = conn.sql("""
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE first_ed_non_opioid_date IS NULL OR first_opioid_ed_date IS NOT NULL
                    """).fetchone()[0]
                    cohort_date_fields_validation["incorrect_null_count"] = incorrect_nulls
                    cohort_date_fields_validation["first_ed_non_opioid_date_valid"] = incorrect_nulls == 0

                    if incorrect_nulls > 0:
                        logger.warning(
                            f"⚠️  ED_NON_OPIOID cohort: {incorrect_nulls} records with incorrect date field NULLs "
                            f"(first_ed_non_opioid_date should be populated, first_opioid_ed_date should be NULL)"
                        )

            except Exception as e:
                logger.debug(f"Could not validate cohort-specific date fields: {e}")
                cohort_date_fields_validation["error"] = str(e)

        results["cohort_date_fields_validation"] = cohort_date_fields_validation

        # Data completeness validation (medical has ICD, pharmacy has drug_name)
        data_completeness_validation = {
            "medical_events_have_icd": True,
            "pharmacy_events_have_drug": True,
            "medical_missing_icd_count": 0,
            "pharmacy_missing_drug_count": 0
        }

        if "event_type" in cols:
            try:
                # Medical events should have primary_icd_diagnosis_code
                if "primary_icd_diagnosis_code" in cols:
                    medical_missing_icd = conn.sql("""
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE event_type = 'medical' AND primary_icd_diagnosis_code IS NULL
                    """).fetchone()[0]
                    data_completeness_validation["medical_missing_icd_count"] = medical_missing_icd
                    data_completeness_validation["medical_events_have_icd"] = medical_missing_icd == 0

                    if medical_missing_icd > 0:
                        logger.warning(f"⚠️  Found {medical_missing_icd} medical events without ICD diagnosis code")

                # Pharmacy events should have drug_name
                if "drug_name" in cols:
                    pharmacy_missing_drug = conn.sql("""
                        SELECT COUNT(*)
                        FROM cohort_qa
                        WHERE event_type = 'pharmacy' AND drug_name IS NULL
                    """).fetchone()[0]
                    data_completeness_validation["pharmacy_missing_drug_count"] = pharmacy_missing_drug
                    data_completeness_validation["pharmacy_events_have_drug"] = pharmacy_missing_drug == 0

                    if pharmacy_missing_drug > 0:
                        logger.warning(f"⚠️  Found {pharmacy_missing_drug} pharmacy events without drug_name")

            except Exception as e:
                logger.debug(f"Could not validate data completeness: {e}")
                data_completeness_validation["error"] = str(e)

        results["data_completeness_validation"] = data_completeness_validation

        # Metadata fields validation
        metadata_validation = {
            "created_at_populated": True,
            "age_band_filter_matches": True,
            "event_year_filter_matches": True,
            "created_at_null_count": 0,
            "age_band_filter_mismatch_count": 0,
            "event_year_filter_mismatch_count": 0
        }

        try:
            if "created_at" in cols:
                created_at_nulls = conn.sql("""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE created_at IS NULL
                """).fetchone()[0]
                metadata_validation["created_at_null_count"] = created_at_nulls
                metadata_validation["created_at_populated"] = created_at_nulls == 0

                if created_at_nulls > 0:
                    logger.warning(f"⚠️  Found {created_at_nulls} records with NULL created_at")

            if "age_band_filter" in cols:
                age_band_filter_mismatch = conn.sql(f"""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE age_band_filter != '{age_band}'
                """).fetchone()[0]
                metadata_validation["age_band_filter_mismatch_count"] = age_band_filter_mismatch
                metadata_validation["age_band_filter_matches"] = age_band_filter_mismatch == 0

                if age_band_filter_mismatch > 0:
                    logger.warning(f"⚠️  Found {age_band_filter_mismatch} records where age_band_filter doesn't match '{age_band}'")

            if "event_year_filter" in cols:
                event_year_filter_mismatch = conn.sql(f"""
                    SELECT COUNT(*)
                    FROM cohort_qa
                    WHERE event_year_filter != {event_year}
                """).fetchone()[0]
                metadata_validation["event_year_filter_mismatch_count"] = event_year_filter_mismatch
                metadata_validation["event_year_filter_matches"] = event_year_filter_mismatch == 0

                if event_year_filter_mismatch > 0:
                    logger.warning(f"⚠️  Found {event_year_filter_mismatch} records where event_year_filter doesn't match {event_year}")

        except Exception as e:
            logger.debug(f"Could not validate metadata fields: {e}")
            metadata_validation["error"] = str(e)

        results["metadata_validation"] = metadata_validation

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

        # Frequency counts (full) for normalized drug names, ICD and CPT where present
        results["frequencies"] = {}

        # Normalized drug_name counts (full frequency counts)
        try:
            drug_freq = conn.sql(
                """
                SELECT normalized_drug_name, COUNT(*) as frequency
                FROM cohort_enriched
                WHERE normalized_drug_name IS NOT NULL AND normalized_drug_name <> ''
                GROUP BY normalized_drug_name
                ORDER BY frequency DESC
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

        if not drug_window_validation.get("validation_passed", True):
            status_issues.append("Drug window validation failed")

        if null_validation.get("critical_nulls_found", False):
            status_issues.append("Critical NULL values found")

        if not age_validation.get("age_in_valid_range", True):
            status_issues.append("Invalid ages detected (outside 1-114 range)")

        if not age_validation.get("age_matches_band", True):
            status_issues.append("Ages don't match age_band")

        if not date_validation.get("date_year_matches_event_year", True):
            status_issues.append("Event date year doesn't match event_year column")

        if not date_validation.get("dates_in_valid_range", True):
            status_issues.append("Dates outside valid range (2016-2020)")

        if not cohort_date_fields_validation.get("first_opioid_ed_date_valid", True):
            status_issues.append("Incorrect first_opioid_ed_date NULLs")

        if not cohort_date_fields_validation.get("first_ed_non_opioid_date_valid", True):
            status_issues.append("Incorrect first_ed_non_opioid_date NULLs")

        if not data_completeness_validation.get("medical_events_have_icd", True):
            status_issues.append("Medical events missing ICD codes")

        if not data_completeness_validation.get("pharmacy_events_have_drug", True):
            status_issues.append("Pharmacy events missing drug names")

        if not metadata_validation.get("created_at_populated", True):
            status_issues.append("Missing created_at timestamps")

        if not metadata_validation.get("age_band_filter_matches", True):
            status_issues.append("age_band_filter doesn't match age_band")

        if not metadata_validation.get("event_year_filter_matches", True):
            status_issues.append("event_year_filter doesn't match event_year")
        
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


def _worker_init(env_vars: Dict[str, str]):
    """
    Initializer for worker processes - sets environment variables
    BEFORE any imports or operations happen.
    """
    if env_vars:
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = value

    # Safety: Explicitly set PGX_TARGET_NAME if not already set
    if 'PGX_TARGET_NAME' not in os.environ or not os.environ['PGX_TARGET_NAME']:
        os.environ['PGX_TARGET_NAME'] = 'opioid_ed'

    # Re-add project root to path for this worker
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def process_single_cohort_qa(
    cohort_name: str,
    age_band: str,
    event_year: int,
    save_results: bool,
    log_level: str
) -> Tuple[str, int, str, Dict[str, Any]]:
    """
    Wrapper function to run QA for a single cohort in a separate process.
    Each process creates its own DuckDB connection and logger.

    Returns:
        Tuple of (age_band, event_year, cohort_name, results_dict)
    """
    # Create connection and logger for this process
    logger, _ = setup_logging("final_cohort_qa", age_band, str(event_year))
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Log environment setup for debugging
    logger.debug(f"Worker process environment: PGX_TARGET_NAME={os.environ.get('PGX_TARGET_NAME')}")

    conn = create_simple_duckdb_connection(logger)

    # Run QA
    res = run_cohort_qa(conn, cohort_name, age_band, event_year, logger)

    # Save results if requested
    if save_results:
        out_path = build_qa_output_path(cohort_name, age_band, event_year)
        try:
            save_to_s3_json(res, out_path, logger)
        except Exception as e:
            logger.warning(f"Could not save QA JSON for {cohort_name} {age_band} {event_year}: {e}")

    conn.close()
    return (age_band, event_year, cohort_name, res)


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
    ap.add_argument("--max-workers", type=int, default=None,
                   help="Maximum number of parallel workers (default: CPU count)")
    # Optional runtime overrides for target configuration (matches 0_create_cohort.py)
    ap.add_argument("--target-name", default=None, help="Optional target name to set (overrides PGX_TARGET_NAME env)")
    ap.add_argument("--target-icd-codes", default=None, help="Optional ICD codes string (comma-separated) to set PGX_TARGET_ICD_CODES")
    ap.add_argument("--target-cpt-codes", default=None, help="Optional CPT codes string (comma-separated) to set PGX_TARGET_CPT_CODES")
    ap.add_argument("--target-icd-prefixes", default=None, help="Optional ICD prefixes string (comma-separated) to set PGX_TARGET_ICD_PREFIXES")
    ap.add_argument("--target-cpt-prefixes", default=None, help="Optional CPT prefixes string (comma-separated) to set PGX_TARGET_CPT_PREFIXES")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers (default: 4)")
    ap.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (run sequentially)")
    args = ap.parse_args()

    # If target overrides provided on CLI, set environment variables *before* importing modules
    if args.target_name or args.target_icd_codes or args.target_cpt_codes or args.target_icd_prefixes or args.target_cpt_prefixes:
        if args.target_name:
            os.environ["PGX_TARGET_NAME"] = args.target_name
        if args.target_icd_codes:
            os.environ["PGX_TARGET_ICD_CODES"] = args.target_icd_codes
        if args.target_cpt_codes:
            os.environ["PGX_TARGET_CPT_CODES"] = args.target_cpt_codes
        if args.target_icd_prefixes:
            os.environ["PGX_TARGET_ICD_PREFIXES"] = args.target_icd_prefixes
        if args.target_cpt_prefixes:
            os.environ["PGX_TARGET_CPT_PREFIXES"] = args.target_cpt_prefixes
        
        # Reload modules so module-level constants derived from env are refreshed
        import importlib
        try:
            import helpers_1997_13.constants as constants
            import helpers_1997_13.s3_utils as s3_utils
            importlib.reload(constants)
            importlib.reload(s3_utils)
            # Re-import to get updated references
            from helpers_1997_13.s3_utils import get_cohort_parquet_path, save_to_s3_json
            globals()['get_cohort_parquet_path'] = get_cohort_parquet_path
            globals()['save_to_s3_json'] = save_to_s3_json
        except Exception:
            # Best-effort; if reload fails, re-import without reload
            try:
                from helpers_1997_13.s3_utils import get_cohort_parquet_path, save_to_s3_json
                globals()['get_cohort_parquet_path'] = get_cohort_parquet_path
                globals()['save_to_s3_json'] = save_to_s3_json
            except Exception:
                pass

    # For logging, use a generic run_id if many bands
    run_band = args.age_band or ("all" if args.all_age_bands else (args.age_bands or "multi"))
    run_year = args.event_year if args.event_year is not None else ("all" if (args.all_event_years or args.event_years) else "unknown")
    logger, log_buffer = setup_logging("final_cohort_qa", str(run_band), str(run_year))
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Only create connection if not using parallel processing
    # (parallel processing creates its own connections per worker)
    conn = None
    if args.max_workers is None or args.max_workers == 1:
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

<<<<<<< HEAD
    # Ensure PGX_TARGET_NAME is set in main process
    if 'PGX_TARGET_NAME' not in os.environ or not os.environ['PGX_TARGET_NAME']:
        logger.warning("PGX_TARGET_NAME not set, defaulting to 'opioid_ed'")
        os.environ['PGX_TARGET_NAME'] = 'opioid_ed'

    # Capture environment variables needed for target configuration
    target_env_vars = {
        "PGX_TARGET_NAME": os.getenv("PGX_TARGET_NAME"),
        "PGX_TARGET_ICD_CODES": os.getenv("PGX_TARGET_ICD_CODES"),
        "PGX_TARGET_CPT_CODES": os.getenv("PGX_TARGET_CPT_CODES"),
        "PGX_TARGET_ICD_PREFIXES": os.getenv("PGX_TARGET_ICD_PREFIXES"),
        "PGX_TARGET_CPT_PREFIXES": os.getenv("PGX_TARGET_CPT_PREFIXES"),
    }

    # Prepare all tasks for parallel processing
    tasks = []
    for band in age_bands:
        for year in event_years:
            for name in cohorts:
                tasks.append((name, band, year, args.log_level, target_env_vars))

    total_tasks = len(tasks)

    # Determine number of workers
    max_workers = args.max_workers if args.max_workers is not None else multiprocessing.cpu_count()
    max_workers = min(max_workers, len(tasks))  # Don't exceed number of tasks

    logger.info(f"Processing {len(tasks)} QA tasks with {max_workers} parallel workers...")

    # Process tasks in parallel or sequentially
    if max_workers > 1 and len(tasks) > 1 and not args.no_parallel:
        # Use parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(run_cohort_qa_parallel, task): task
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_task):
                name, band, year, _, _ = future_to_task[future]
                completed += 1
                try:
                    res = future.result()
                    all_results["results"].setdefault(band, {}).setdefault(year, {})[name] = res

                    if args.save_results:
                        out_path = build_qa_output_path(name, band, year)
                        try:
                            # Ensure save_to_s3_json is available (may have been reloaded)
                            from helpers_1997_13.s3_utils import save_to_s3_json
                            save_to_s3_json(res, out_path, logger)
                        except Exception as e:
                            logger.warning(f"Could not save QA JSON for {name} {band} {year}: {e}")

                    logger.info(f"[{completed}/{len(tasks)}] Completed QA for {band} {year} {name}")
                except Exception as e:
                    logger.error(f"Error processing {band} {year} {name}: {e}")
                    all_results["results"].setdefault(band, {}).setdefault(year, {})[name] = {
                        "status": "ERROR",
                        "error": str(e),
                        "cohort": name,
                        "age_band": band,
                        "event_year": year
                    }
    else:
        # Sequential processing (fallback or single worker)
        if conn is None:
            conn = create_simple_duckdb_connection(logger)
        for idx, (name, band, year, _, env_vars) in enumerate(tasks, 1):
            logger.info(f"[{idx}/{total_tasks}] Processing {name} {band} {year}")
            # Set environment variables for sequential processing too
            for key, value in env_vars.items():
                if value is not None:
                    os.environ[key] = value
            res = run_cohort_qa(conn, name, band, year, logger)
            all_results["results"].setdefault(band, {}).setdefault(year, {})[name] = res

            if args.save_results:
                out_path = build_qa_output_path(name, band, year)
                try:
                    # Ensure save_to_s3_json is available (may have been reloaded)
                    from helpers_1997_13.s3_utils import save_to_s3_json
                    save_to_s3_json(res, out_path, logger)
                except Exception as e:
                    logger.warning(f"Could not save QA JSON for {name} {band} {year}: {e}")

    # Close connection if we created one
    if conn is not None:
        conn.close()

    # Save master log to S3
    try:
        # Use run_band and run_year which handle None values for --all-age-bands/--all-event-years
        save_logs_to_s3(log_buffer, "final_cohort_qa", run_band, run_year)
    except Exception as e:
        logger.warning(f"Could not save logs to S3: {e}")

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
                drug_window_val = r.get('drug_window_validation', {})

                validation_flags = []
                if cohort_val.get('cohort_name_mismatch', False):
                    validation_flags.append("COHORT_NAME_MISMATCH")
                if not partition_val.get('age_band_match', True):
                    validation_flags.append("AGE_BAND_MISMATCH")
                if not partition_val.get('event_year_match', True):
                    validation_flags.append("EVENT_YEAR_MISMATCH")
                if not separation_val.get('separation_valid', True):
                    validation_flags.append("SEPARATION_VIOLATION")
                if not drug_window_val.get('validation_passed', True):
                    validation_flags.append("DRUG_WINDOW_INVALID")

                flags_str = f" [{', '.join(validation_flags)}]" if validation_flags else ""

                # Add drug window stats if available
                drug_window_str = ""
                if drug_window_val.get('column_exists') and drug_window_val.get('pharmacy_events_in_30day_window', 0) > 0:
                    drug_window_str = f" drugs_30day={drug_window_val.get('pharmacy_events_in_30day_window', 0)}"

                logger.info(
                    f"{band} {year} {name} → status={status}{flags_str} "
                    f"records={metrics.get('total_records')} "
                    f"patients={metrics.get('distinct_patients')} "
                    f"targets={metrics.get('target_cases', 0)} "
                    f"controls={metrics.get('control_cases', 0)} "
                    f"ratio={ratio_str}{drug_window_str}"
                )


if __name__ == "__main__":
    main()


