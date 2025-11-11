"""
Data processing and validation utilities.
"""

import sys
import os
import traceback
from typing import List, Any, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.common_imports import *
from helpers_1997_13.duckdb_utils import get_duckdb_connection
from helpers_1997_13.constants import EXCLUDED_CODES


def is_excluded_code(code):
    """Check if a code should be excluded as a lagging variable."""
    if not isinstance(code, str):
        return False
    code = code.upper().strip()
    # Check for F11 prefix (all opioid use disorder codes)
    if code.startswith('F11'):
        return True
    # Check for other excluded codes
    return any(excluded.lower() in code.lower() for excluded in EXCLUDED_CODES)
    
    
def validate_data_for_blank_strings(view_name: str, logger, conn=None):
    if conn is None:
        logger.warning("No connection provided to validate_data_for_blank_strings, skipping validation")
        return
        
    try:
        cols = {r[0] for r in conn.sql(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{view_name}'"
        ).fetchall()}
        checks = []
        if "member_age_dos" in cols:
            checks.append("""
              SELECT 'member_age_dos' AS field_name, COUNT(*) total_rows,
                     COUNT(CASE WHEN CAST(member_age_dos AS VARCHAR) = ' ' THEN 1 END) blank_strings,
                     COUNT(CASE WHEN CAST(member_age_dos AS VARCHAR) = '' THEN 1 END) empty_strings,
                     COUNT(CASE WHEN member_age_dos IS NULL OR TRY_CAST(member_age_dos AS INT) IS NULL THEN 1 END) non_castable
              FROM {view}
            """)
        if "member_age_num" in cols:
            checks.append("""
              SELECT 'member_age_num' AS field_name, COUNT(*) total_rows,
                     0 blank_strings, 0 empty_strings,
                     COUNT(CASE WHEN member_age_num IS NULL OR TRY_CAST(member_age_num AS INT) IS NULL THEN 1 END) non_castable
              FROM {view}
            """)
        if "age_imputed" in cols:
            checks.append("""
              SELECT 'age_imputed' AS field_name, COUNT(*) total_rows,
                     COUNT(CASE WHEN CAST(age_imputed AS VARCHAR) = ' ' THEN 1 END) blank_strings,
                     COUNT(CASE WHEN CAST(age_imputed AS VARCHAR) = '' THEN 1 END) empty_strings,
                     COUNT(CASE WHEN age_imputed IS NULL OR TRY_CAST(age_imputed AS INT) IS NULL THEN 1 END) non_castable
              FROM {view}
            """)
        if "event_year" in cols:
            checks.append("""
              SELECT 'event_year' AS field_name, COUNT(*) total_rows,
                     COUNT(CASE WHEN CAST(event_year AS VARCHAR) = ' ' THEN 1 END) blank_strings,
                     COUNT(CASE WHEN CAST(event_year AS VARCHAR) = '' THEN 1 END) empty_strings,
                     COUNT(CASE WHEN event_year IS NULL THEN 1 END) non_castable
              FROM {view}
            """)
        if not checks:
            logger.info(f"No numeric/string fields to validate in {view_name}.")
            return
        q = " UNION ALL ".join(c.format(view=view_name) for c in checks)
        for row in conn.sql(q).fetchall():
            logger.info(f"Validation {row}")
    except Exception as e:
        logger.error(f"Data validation failed for {view_name}: {e}")


def validate_and_clean_strings(value):
    """
    Validate and clean string values to prevent blank string to INT conversion errors.
    Returns cleaned value or None if invalid.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return value


def safe_cast_to_int(value, default=None):
    """
    Safely cast a value to integer, handling blank strings and other edge cases.
    Returns the integer value or default if casting fails.
    """
    if value is None:
        return default
    
    # Handle blank strings
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            return int(cleaned)
        except (ValueError, TypeError):
            return default
    
    # Handle numeric types
    try:
        return int(value)
    except (ValueError, TypeError):
        return default



def convert_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON serializable object
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, (set,)):
        return list(obj)
    elif isinstance(obj, (dict,)):
        return {k: convert_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_json_serializable(item) for item in obj]
    return obj


def prepare_data(test_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for model evaluation.

    Args:
        test_df: Test DataFrame

    Returns:
        Prepared DataFrame
    """
    try:
        # Create a copy to avoid modifying the original
        df = test_df.copy()

        # Handle missing values
        df = df.fillna({
            'support': 0.0,
            'confidence': 0.0,
            'lift': 0.0
        })

        # Convert categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category')

        # Convert numeric variables
        numeric_cols = ['support', 'confidence', 'lift']
        df[numeric_cols] = df[numeric_cols].astype(float)

        return df

    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise


def clean_rules_dataframe(rules):
    """
    Safely convert complex columns (like frozenset) to serializable formats.
    """
    if "antecedents" in rules.columns:
        rules["antecedents"] = rules["antecedents"].apply(lambda x: sorted(list(x)) if isinstance(x, frozenset) else x)

    if "consequents" in rules.columns:
        rules["consequents"] = rules["consequents"].apply(lambda x: sorted(list(x)) if isinstance(x, frozenset) else x)

    return rules


def safe_mean(values: List[Any], logger: Optional[logging.Logger] = None) -> float:
    """Calculate mean safely handling None/NaN values.

    Args:
        values: List of values to calculate mean from
        logger: Optional logger for error tracking

    Returns:
        Mean value or 0.0 if calculation fails
    """
    try:
        if not values:
            return 0.0
        return float(np.mean([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error calculating mean: {str(e)}")
        return 0.0


def safe_std(values: List[Any], logger: Optional[logging.Logger] = None) -> float:
    """Calculate standard deviation safely handling None/NaN values.

    Args:
        values: List of values to calculate std from
        logger: Optional logger for error tracking

    Returns:
        Standard deviation or 0.0 if calculation fails
    """
    try:
        if not values:
            return 0.0
        return float(np.std([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error calculating std: {str(e)}")
        return 0.0


def safe_median(values: List[Any], logger: Optional[logging.Logger] = None) -> float:
    """Calculate median safely handling None/NaN values.

    Args:
        values: List of values to calculate median from
        logger: Optional logger for error tracking

    Returns:
        Median value or 0.0 if calculation fails
    """
    try:
        if not values:
            return 0.0
        return float(np.median([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error calculating median: {str(e)}")
        return 0.0


def safe_min(values: List[Any], logger: Optional[logging.Logger] = None) -> float:
    """Calculate minimum safely handling None/NaN values.

    Args:
        values: List of values to calculate min from
        logger: Optional logger for error tracking

    Returns:
        Minimum value or 0.0 if calculation fails
    """
    try:
        if not values:
            return 0.0
        return float(np.min([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error calculating min: {str(e)}")
        return 0.0


def safe_max(values: List[Any], logger: Optional[logging.Logger] = None) -> float:
    """Calculate maximum safely handling None/NaN values.

    Args:
        values: List of values to calculate max from
        logger: Optional logger for error tracking

    Returns:
        Maximum value or 0.0 if calculation fails
    """
    try:
        if not values:
            return 0.0
        return float(np.max([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error calculating max: {str(e)}")
        return 0.0


def safe_sum(values: List[Any], logger: Optional[logging.Logger] = None) -> float:
    """Calculate sum safely handling None/NaN values.

    Args:
        values: List of values to calculate sum from
        logger: Optional logger for error tracking

    Returns:
        Sum value or 0.0 if calculation fails
    """
    try:
        if not values:
            return 0.0
        return float(np.sum([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error calculating sum: {str(e)}")
        return 0.0


def safe_count(values: List[Any], logger: Optional[logging.Logger] = None) -> int:
    """Count non-null values safely.

    Args:
        values: List of values to count
        logger: Optional logger for error tracking

    Returns:
        Count of non-null values or 0 if calculation fails
    """
    try:
        if not values:
            return 0
        return len([v for v in values if pd.notnull(v)])
    except Exception as e:
        if logger:
            logger.error(f"Error counting values: {str(e)}")
        return 0


def safe_unique_count(values: List[Any], logger: Optional[logging.Logger] = None) -> int:
    """Count unique non-null values safely.

    Args:
        values: List of values to count unique values from
        logger: Optional logger for error tracking

    Returns:
        Count of unique non-null values or 0 if calculation fails
    """
    try:
        if not values:
            return 0
        return len(set([v for v in values if pd.notnull(v)]))
    except Exception as e:
        if logger:
            logger.error(f"Error counting unique values: {str(e)}")
        return 0 


def handle_empty_filtered_cohort(df, cohort_name, band, year, paths, logger, TOP_K=25):
    """Handle empty filtered cohort by creating placeholder data."""
    logger.warning(f"Skipping {cohort_name} {band} {year} — no valid rows after filtering")

    placeholder_columns = ["mi_person_key", "target", "drug_tokens", "tokens"]
    placeholder = pd.DataFrame(columns=placeholder_columns)

    # Efficient column padding using pd.concat
    padding = {
        f"pattern_{i+1}": ["missing"]
        for i in range(TOP_K)
    }
    padding.update({
        f"support_{i+1}": [0.0]
        for i in range(TOP_K)
    })
    padding.update({
        f"confidence_{i+1}": [0.0]
        for i in range(TOP_K)
    })
    padding.update({
        f"lift_{i+1}": [0.0]
        for i in range(TOP_K)
    })
    padding.update({
        f"certainty_{i+1}": [0.0]
        for i in range(TOP_K)
    })

    placeholder = safe_concat_columns(placeholder, padding)
    logger.info(f"✓ Created placeholder enhanced dataset with 0 valid rows")

    return True  # Indicate early exit


def safe_concat_columns(df, new_cols_dict):
    """
    Silently concatenates new columns to a DataFrame, dropping any that already exist.
    No logging or warnings are emitted.
    """
    new_cols_df = pd.DataFrame(new_cols_dict)
    overlap = df.columns.intersection(new_cols_df.columns)
    if not overlap.empty:
        new_cols_df = new_cols_df.drop(columns=overlap)
    return pd.concat([df, new_cols_df], axis=1)


def has_valid_metrics(metrics, cohort):
    """Check if metrics object has valid data for the specified cohort."""
    return (
        metrics and
        isinstance(metrics.get("steps"), dict) and
        any(
            cohort in step.get("by_group", {})
            for step in metrics["steps"].values()
            if isinstance(step, dict)
        )
    )


def collect_validation_metrics(conn, logger, age_band, event_year):
    """Collect validation metrics for data quality checks and integrate into QA metrics system."""
    logger.info("→ Collecting validation metrics...")
    logger.info(f"→ Validation parameters: age_band={age_band}, event_year={event_year}")

    validation_metrics = {}

    try:
        # Pre-validation checks - ensure required views exist
        logger.info("→ Validation: Checking if required views exist...")
        try:
            conn.sql("SELECT COUNT(*) as count FROM medical_clean_age LIMIT 1").df()
            logger.info("→ Validation: medical_clean_age view exists")
        except Exception as medical_exists_e:
            logger.error(f"→ Validation: medical_clean_age view does not exist: {str(medical_exists_e)}")
            raise Exception(f"Required view 'medical_clean_age' does not exist: {str(medical_exists_e)}") from medical_exists_e

        try:
            conn.sql("SELECT COUNT(*) as count FROM pharmacy_clean LIMIT 1").df()
            logger.info("→ Validation: pharmacy_clean view exists")
        except Exception as pharmacy_exists_e:
            logger.error(f"→ Validation: pharmacy_clean view does not exist: {str(pharmacy_exists_e)}")
            raise Exception(f"Required view 'pharmacy_clean' does not exist: {str(pharmacy_exists_e)}") from pharmacy_exists_e

        # Event year consistency validation
        logger.info("→ Validation: Checking medical data event year consistency...")
        try:
            # Use DuckDB-compatible date extraction
            medical_year_check = conn.sql(f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN EXTRACT(YEAR FROM event_date) != {event_year} THEN 1 END) as mismatched_rows,
                    COUNT(CASE WHEN EXTRACT(YEAR FROM event_date) = {event_year} THEN 1 END) as matched_rows
                FROM medical_clean_age
                WHERE age_band = '{age_band}' AND event_year = {event_year}
            """).df()
            logger.info(f"→ Validation: Medical year check completed - {medical_year_check.iloc[0]['total_rows']} total rows")
        except Exception as medical_e:
            logger.error(f"→ Validation: Error checking medical year consistency: {str(medical_e)}")
            logger.error(f"→ Validation: Medical error type: {type(medical_e).__name__}")
            logger.error(f"→ Validation: Medical traceback:\n{traceback.format_exc()}")

            # Try alternative date extraction methods
            logger.info("→ Validation: Trying alternative date extraction methods...")
            try:
                # Try with YEAR() function
                medical_year_check = conn.sql(f"""
                    SELECT
                        COUNT(*) as total_rows,
                        COUNT(CASE WHEN YEAR(event_date) != {event_year} THEN 1 END) as mismatched_rows,
                        COUNT(CASE WHEN YEAR(event_date) = {event_year} THEN 1 END) as matched_rows
                    FROM medical_clean_age
                    WHERE age_band = '{age_band}' AND event_year = {event_year}
                """).df()
                logger.info(f"→ Validation: Alternative medical year check succeeded - {medical_year_check.iloc[0]['total_rows']} total rows")
            except Exception as alt_medical_e:
                logger.error(f"→ Validation: Alternative medical year check also failed: {str(alt_medical_e)}")
                # Try without year extraction (just count rows)
                try:
                    medical_year_check = conn.sql(f"""
                        SELECT
                            COUNT(*) as total_rows,
                            0 as mismatched_rows,
                            COUNT(*) as matched_rows
                        FROM medical_clean_age
                        WHERE age_band = '{age_band}' AND event_year = {event_year}
                    """).df()
                    logger.warning(f"→ Validation: Using simplified medical check - {medical_year_check.iloc[0]['total_rows']} total rows")
                except Exception as simple_medical_e:
                    logger.error(f"→ Validation: Even simplified medical check failed: {str(simple_medical_e)}")
                    raise

        logger.info("→ Validation: Checking pharmacy data event year consistency...")
        try:
            # Use DuckDB-compatible date extraction
            pharmacy_year_check = conn.sql(f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN EXTRACT(YEAR FROM event_date) != {event_year} THEN 1 END) as mismatched_rows,
                    COUNT(CASE WHEN EXTRACT(YEAR FROM event_date) = {event_year} THEN 1 END) as matched_rows
                FROM pharmacy_clean
                WHERE age_band = '{age_band}' AND event_year = {event_year}
            """).df()
            logger.info(f"→ Validation: Pharmacy year check completed - {pharmacy_year_check.iloc[0]['total_rows']} total rows")
        except Exception as pharmacy_e:
            logger.error(f"→ Validation: Error checking pharmacy year consistency: {str(pharmacy_e)}")
            logger.error(f"→ Validation: Pharmacy error type: {type(pharmacy_e).__name__}")
            logger.error(f"→ Validation: Pharmacy traceback:\n{traceback.format_exc()}")

            # Try alternative date extraction methods
            logger.info("→ Validation: Trying alternative pharmacy date extraction methods...")
            try:
                # Try with YEAR() function
                pharmacy_year_check = conn.sql(f"""
                    SELECT
                        COUNT(*) as total_rows,
                        COUNT(CASE WHEN YEAR(event_date) != {event_year} THEN 1 END) as mismatched_rows,
                        COUNT(CASE WHEN YEAR(event_date) = {event_year} THEN 1 END) as matched_rows
                    FROM pharmacy_clean
                    WHERE age_band = '{age_band}' AND event_year = {event_year}
                """).df()
                logger.info(f"→ Validation: Alternative pharmacy year check succeeded - {pharmacy_year_check.iloc[0]['total_rows']} total rows")
            except Exception as alt_pharmacy_e:
                logger.error(f"→ Validation: Alternative pharmacy year check also failed: {str(alt_pharmacy_e)}")
                # Try without year extraction (just count rows)
                try:
                    pharmacy_year_check = conn.sql(f"""
                        SELECT
                            COUNT(*) as total_rows,
                            0 as mismatched_rows,
                            COUNT(*) as matched_rows
                        FROM pharmacy_clean
                        WHERE age_band = '{age_band}' AND event_year = {event_year}
                    """).df()
                    logger.warning(f"→ Validation: Using simplified pharmacy check - {pharmacy_year_check.iloc[0]['total_rows']} total rows")
                except Exception as simple_pharmacy_e:
                    logger.error(f"→ Validation: Even simplified pharmacy check failed: {str(simple_pharmacy_e)}")
                    raise

        logger.info("→ Validation: Building event year consistency metrics...")
        validation_metrics["event_year_consistency"] = {
            "medical": {
                "total_rows": int(medical_year_check.iloc[0]['total_rows']),
                "mismatched_rows": int(medical_year_check.iloc[0]['mismatched_rows']),
                "matched_rows": int(medical_year_check.iloc[0]['matched_rows']),
                "consistency_rate": float(medical_year_check.iloc[0]['matched_rows'] / medical_year_check.iloc[0]['total_rows']) if medical_year_check.iloc[0]['total_rows'] > 0 else 0.0
            },
            "pharmacy": {
                "total_rows": int(pharmacy_year_check.iloc[0]['total_rows']),
                "mismatched_rows": int(pharmacy_year_check.iloc[0]['mismatched_rows']),
                "matched_rows": int(pharmacy_year_check.iloc[0]['matched_rows']),
                "consistency_rate": float(pharmacy_year_check.iloc[0]['matched_rows'] / pharmacy_year_check.iloc[0]['total_rows']) if pharmacy_year_check.iloc[0]['total_rows'] > 0 else 0.0
            }
        }

        # Date type validation
        logger.info("→ Validation: Checking medical date types...")
        try:
            medical_date_type = conn.sql("""
                SELECT typeof(event_date) as event_date_type
                FROM medical_clean_age
                LIMIT 1
            """).df()
            logger.info(f"→ Validation: Medical date type check completed - type: {medical_date_type.iloc[0]['event_date_type']}")
        except Exception as medical_date_e:
            logger.error(f"→ Validation: Error checking medical date type: {str(medical_date_e)}")
            logger.error(f"→ Validation: Medical date error type: {type(medical_date_e).__name__}")
            logger.error(f"→ Validation: Medical date traceback:\n{traceback.format_exc()}")
            # Provide default value and continue
            medical_date_type = pd.DataFrame({'event_date_type': ['TIMESTAMP']})
            logger.warning("→ Validation: Using default medical date type: TIMESTAMP")

        logger.info("→ Validation: Checking pharmacy date types...")
        try:
            pharmacy_date_type = conn.sql("""
                SELECT typeof(event_date) as event_date_type
                FROM pharmacy_clean
                LIMIT 1
            """).df()
            logger.info(f"→ Validation: Pharmacy date type check completed - type: {pharmacy_date_type.iloc[0]['event_date_type']}")
        except Exception as pharmacy_date_e:
            logger.error(f"→ Validation: Error checking pharmacy date type: {str(pharmacy_date_e)}")
            logger.error(f"→ Validation: Pharmacy date error type: {type(pharmacy_date_e).__name__}")
            logger.error(f"→ Validation: Pharmacy date traceback:\n{traceback.format_exc()}")
            # Provide default value and continue
            pharmacy_date_type = pd.DataFrame({'event_date_type': ['TIMESTAMP']})
            logger.warning("→ Validation: Using default pharmacy date type: TIMESTAMP")

        logger.info("→ Validation: Building date type metrics...")
        validation_metrics["date_types"] = {
            "medical_event_date_type": str(medical_date_type.iloc[0]['event_date_type']),
            "pharmacy_event_date_type": str(pharmacy_date_type.iloc[0]['event_date_type']),
            "medical_needs_conversion": 'varchar' in str(medical_date_type.iloc[0]['event_date_type']).lower(),
            "pharmacy_needs_conversion": 'varchar' in str(pharmacy_date_type.iloc[0]['event_date_type']).lower()
        }

        # Log validation results
        logger.info("→ Validation: Analyzing results...")
        total_mismatched = medical_year_check.iloc[0]['mismatched_rows'] + pharmacy_year_check.iloc[0]['mismatched_rows']
        if total_mismatched > 0:
            logger.warning(f"→ Found {total_mismatched} rows with event_date/year mismatch")
        else:
            logger.info("→ ✓ Event year consistency validated")

        medical_needs_conv = 'varchar' in str(medical_date_type.iloc[0]['event_date_type']).lower()
        pharmacy_needs_conv = 'varchar' in str(pharmacy_date_type.iloc[0]['event_date_type']).lower()

        if medical_needs_conv or pharmacy_needs_conv:
            logger.warning(f"→ Date type conversion needed - Medical: {medical_needs_conv}, Pharmacy: {pharmacy_needs_conv}")
        else:
            logger.info("→ ✓ Date types validated")

        logger.info("→ Validation: Metrics collection completed successfully")
        return validation_metrics

    except Exception as e:
        logger.error(f"→ CRITICAL: Error collecting validation metrics: {str(e)}")
        logger.error(f"→ CRITICAL: Exception type: {type(e).__name__}")
        logger.error(f"→ CRITICAL: Full traceback:\n{traceback.format_exc()}")

        # Check if required views exist
        logger.error("→ CRITICAL: Checking if required views exist...")
        try:
            medical_count = conn.sql("SELECT COUNT(*) FROM medical_clean_age").df()
            logger.error(f"→ CRITICAL: medical_clean_age exists with {medical_count.iloc[0, 0]} rows")
        except Exception as medical_check_e:
            logger.error(f"→ CRITICAL: medical_clean_age view error: {str(medical_check_e)}")

        try:
            pharmacy_count = conn.sql("SELECT COUNT(*) FROM pharmacy_clean").df()
            logger.error(f"→ CRITICAL: pharmacy_clean exists with {pharmacy_count.iloc[0, 0]} rows")
        except Exception as pharmacy_check_e:
            logger.error(f"→ CRITICAL: pharmacy_clean view error: {str(pharmacy_check_e)}")

        return None


def validate_event_year_consistency(conn, logger, age_band, event_year):
    """Validate that event_date and event_year are consistent across all data sources."""
    logger.info("→ Validating event_year consistency...")

    try:
        # Check medical data
        medical_check = conn.sql(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN YEAR(event_date) != {event_year} THEN 1 END) as mismatched_rows
            FROM medical_clean_age
            WHERE age_band = '{age_band}' AND event_year = {event_year}
        """).df()

        if medical_check.iloc[0]['mismatched_rows'] > 0:
            logger.warning(f"→ Found {medical_check.iloc[0]['mismatched_rows']} medical rows with event_date/year mismatch")

        # Check pharmacy data
        pharmacy_check = conn.sql(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN YEAR(event_date) != {event_year} THEN 1 END) as mismatched_rows
            FROM pharmacy_clean
            WHERE age_band = '{age_band}' AND event_year = {event_year}
        """).df()

        if pharmacy_check.iloc[0]['mismatched_rows'] > 0:
            logger.warning(f"→ Found {pharmacy_check.iloc[0]['mismatched_rows']} pharmacy rows with event_date/year mismatch")

        total_mismatched = medical_check.iloc[0]['mismatched_rows'] + pharmacy_check.iloc[0]['mismatched_rows']

        if total_mismatched > 0:
            logger.warning(f"→ Total mismatched rows: {total_mismatched}")
            return False
        else:
            logger.info("→ ✓ Event year consistency validated")
            return True

    except Exception as e:
        logger.error(f"→ Error during event year validation: {str(e)}")
        return False


def validate_cohort_name(cohort_name):
    """Validate that cohort_name is one of the known values."""
    valid_cohorts = ['opioid_ed', 'ed_non_opioid']
    if cohort_name not in valid_cohorts:
        raise ValueError(f"Invalid cohort_name: {cohort_name}. Must be one of: {valid_cohorts}")
    return True


def safe_convert_date_type(conn, table_name, column_name, logger):
    """Safely convert a date column to TIMESTAMP type without causing recursion."""
    try:
        # Check current column type
        type_check = conn.sql(f"""
            SELECT typeof({column_name}) as column_type
            FROM {table_name}
            LIMIT 1
        """).df()

        current_type = str(type_check.iloc[0]['column_type']).lower()

        if 'varchar' in current_type or 'string' in current_type:
            logger.warning(f"→ Converting {table_name}.{column_name} from {current_type.upper()} to TIMESTAMP")

            # Get all column names from the table to avoid SELECT *
            columns_query = conn.sql(f"DESCRIBE {table_name}").df()
            all_columns = [col['column_name'] for col in columns_query.to_dict('records')]

            # Build SELECT statement with explicit column list
            select_parts = []
            for col in all_columns:
                if col == column_name:
                    select_parts.append(f"CAST({column_name} AS TIMESTAMP) as {column_name}")
                else:
                    select_parts.append(col)

            select_clause = ", ".join(select_parts)

            # Create new view with converted column
            conn.sql(f"""
                CREATE OR REPLACE VIEW {table_name} AS
                SELECT {select_clause}
                FROM {table_name}
            """)

            logger.info(f"→ ✓ Successfully converted {table_name}.{column_name} to TIMESTAMP")
            return True
        else:
            logger.info(f"→ {table_name}.{column_name} already has correct type: {current_type.upper()}")
            return True
    except Exception as e:
        logger.error(f"→ Error converting {table_name}.{column_name}: {str(e)}")
        return False


def ensure_date_types(conn, logger):
    """Ensure event_date and drug_date are properly typed as DATE/TIMESTAMP."""
    logger.info("→ Ensuring proper date types...")

    try:
        # Safely convert medical data date types
        medical_success = safe_convert_date_type(conn, "medical_clean_age", "event_date", logger)

        # Safely convert pharmacy data date types
        pharmacy_success = safe_convert_date_type(conn, "pharmacy_clean", "event_date", logger)

        if medical_success and pharmacy_success:
            logger.info("→ ✓ Date types validated and corrected")
            return True
        else:
            logger.error("→ ✗ Date type conversion failed")
            return False
    except Exception as e:
        logger.error(f"→ Error during date type validation: {str(e)}")
        return False


def generate_qa_report(conn, cohort_name, logger, age_band=None, event_year=None):
    """Generate a compact QA report for the cohort."""
    logger.info(f"→ Generating QA report for {cohort_name}...")

    try:
        # Row counts
        row_counts = conn.sql(f"""
            SELECT COUNT(*) as total_rows,
                COUNT(DISTINCT mi_person_key) as distinct_patients
            FROM {cohort_name}_cohort
        """).df()

        # Event distribution
        event_distribution = conn.sql(f"""
            SELECT Event, COUNT(*) as count
            FROM {cohort_name}_cohort
            GROUP BY Event
            ORDER BY count DESC
        """).df()

        # Drug counts
        drug_counts = conn.sql(f"""
            SELECT COUNT(*) as total_drug_events,
                COUNT(DISTINCT drug_name) as distinct_drugs,
                COUNT(CASE WHEN drug_name IS NOT NULL THEN 1 END) as non_null_drugs
            FROM {cohort_name}_cohort
            WHERE Event = 'Drug_Prescription'
        """).df()

        # NULL counts for key fields
        null_counts = conn.sql(f"""
            SELECT
                COUNT(CASE WHEN mi_person_key IS NULL THEN 1 END) as null_person_keys,
                COUNT(CASE WHEN event_date IS NULL THEN 1 END) as null_event_dates,
                COUNT(CASE WHEN Event IS NULL THEN 1 END) as null_events,
                COUNT(CASE WHEN target IS NULL THEN 1 END) as null_targets
            FROM {cohort_name}_cohort
        """).df()

        # Target/control distribution
        target_distribution = conn.sql(f"""
            SELECT target, COUNT(DISTINCT mi_person_key) as patient_count
            FROM {cohort_name}_cohort
            GROUP BY target
        """).df()

        # Compile report
        qa_report = {
            "cohort_name": cohort_name,
            "timestamp": datetime.now().isoformat(),
            "row_counts": {
                "total_rows": int(row_counts.iloc[0]['total_rows']),
                "distinct_patients": int(row_counts.iloc[0]['distinct_patients'])
            },
            "event_distribution": event_distribution.to_dict('records'),
            "drug_counts": {
                "total_drug_events": int(drug_counts.iloc[0]['total_drug_events']),
                "distinct_drugs": int(drug_counts.iloc[0]['distinct_drugs']),
                "non_null_drugs": int(drug_counts.iloc[0]['non_null_drugs'])
            },
            "null_counts": {
                "null_person_keys": int(null_counts.iloc[0]['null_person_keys']),
                "null_event_dates": int(null_counts.iloc[0]['null_event_dates']),
                "null_events": int(null_counts.iloc[0]['null_events']),
                "null_targets": int(null_counts.iloc[0]['null_targets'])
            },
            "target_distribution": target_distribution.to_dict('records')
        }

        # Save QA report to S3
        qa_report_path = f"s3://{S3_BUCKET}/cohorts/qa_reports/{cohort_name}_{age_band}_{event_year}_qa_report.json"
        qa_report_json = json.dumps(qa_report, indent=2)

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"cohorts/qa_reports/{cohort_name}_{age_band}_{event_year}_qa_report.json",
            Body=qa_report_json,
            ContentType='application/json'
        )

        logger.info(f"→ ✓ QA report saved to {qa_report_path}")
        logger.info(f"→ QA Summary for {cohort_name}:")
        logger.info(f"  - Total rows: {qa_report['row_counts']['total_rows']}")
        logger.info(f"  - Distinct patients: {qa_report['row_counts']['distinct_patients']}")
        logger.info(f"  - Drug events: {qa_report['drug_counts']['total_drug_events']}")
        logger.info(f"  - NULL person keys: {qa_report['null_counts']['null_person_keys']}")

        return qa_report

    except Exception as e:
        logger.error(f"→ Error generating QA report: {str(e)}")
        return None


def validate_extracted_parameters(conn, logger, expected_age_band, expected_event_year, step_name):
    """Validate that extracted age_band and event_year match expected values."""
    logger.info(f"→ [{step_name}] Validating extracted parameters...")

    try:
        # Check medical data first
        medical_validation = conn.sql(f"""
            SELECT
            age_band,
            event_year,
            COUNT(*) as row_count
            FROM medical_clean
            GROUP BY age_band, event_year
            ORDER BY row_count DESC
        """).df()

        logger.info(f"→ [{step_name}] Medical data validation results:")
        for _, row in medical_validation.iterrows():
            logger.info(f"→ [{step_name}]   age_band='{row['age_band']}', event_year={row['event_year']}, rows={row['row_count']}")

        # Check if pharmacy_clean table exists before trying to validate it
        pharmacy_validation = None
        try:
            pharmacy_validation = conn.sql(f"""
                SELECT
                age_band,
                event_year,
                COUNT(*) as row_count
                FROM pharmacy_clean
                GROUP BY age_band, event_year
                ORDER BY row_count DESC
            """).df()

            logger.info(f"→ [{step_name}] Pharmacy data validation results:")
            for _, row in pharmacy_validation.iterrows():
                logger.info(f"→ [{step_name}]   age_band='{row['age_band']}', event_year={row['event_year']}, rows={row['row_count']}")

        except Exception as pharmacy_e:
            logger.info(f"→ [{step_name}] Pharmacy data not yet loaded (this is normal for Step 3.5): {str(pharmacy_e)}")
            pharmacy_validation = None

        # Check for expected data in medical
        expected_medical = medical_validation[
            (medical_validation['age_band'] == expected_age_band) &
            (medical_validation['event_year'] == expected_event_year)
        ]

        # Check for expected data in pharmacy (if available)
        expected_pharmacy = None
        if pharmacy_validation is not None:
            expected_pharmacy = pharmacy_validation[
                (pharmacy_validation['age_band'] == expected_age_band) &
                (pharmacy_validation['event_year'] == expected_event_year)
            ]

        # Validate that we have the expected data
        medical_has_expected = not expected_medical.empty
        pharmacy_has_expected = expected_pharmacy is not None and not expected_pharmacy.empty

        if not medical_has_expected:
            logger.error(f"→ [{step_name}] ERROR: No medical data found for expected parameters!")
            logger.error(f"→ [{step_name}] Expected: age_band='{expected_age_band}', event_year={expected_event_year}")
            logger.error(f"→ [{step_name}] Available medical data:")
            for _, row in medical_validation.iterrows():
                logger.error(f"→ [{step_name}]   Found: age_band='{row['age_band']}', event_year={row['event_year']}, rows={row['row_count']}")
            return False

        if pharmacy_validation is not None and not pharmacy_has_expected:
            logger.warning(f"→ [{step_name}] WARNING: No pharmacy data found for expected parameters!")
            logger.warning(f"→ [{step_name}] Expected: age_band='{expected_age_band}', event_year={expected_event_year}")
            # Don't fail for pharmacy data - it might not be loaded yet

        # Log success
        logger.info(f"→ [{step_name}] ✓ Found expected medical data: age_band='{expected_age_band}', event_year={expected_event_year}")
        if expected_medical is not None and not expected_medical.empty:
            logger.info(f"→ [{step_name}] ✓ Medical rows for expected parameters: {expected_medical.iloc[0]['row_count']}")
        
        if expected_pharmacy is not None and not expected_pharmacy.empty:
            logger.info(f"→ [{step_name}] ✓ Pharmacy rows for expected parameters: {expected_pharmacy.iloc[0]['row_count']}")

        logger.info(f"→ [{step_name}] ✓ Parameter validation passed")
        return True

    except Exception as e:
        logger.error(f"→ [{step_name}] ERROR during parameter validation: {str(e)}")
        logger.error(f"→ [{step_name}] Validation failed - proceeding with caution")
        return False


def validate_data_consistency(conn, logger, age_band, event_year, step_name):
    """Validate data consistency across all views."""
    logger.info(f"→ [{step_name}] Validating data consistency...")

    try:
        # Check if views exist and have data
        views_to_check = ['medical_clean', 'pharmacy_clean', 'medical_filtered', 'medical_clean_age']

        for view_name in views_to_check:
            try:
                count = conn.sql(f"SELECT COUNT(*) as count FROM {view_name}").df()
                logger.info(f"→ [{step_name}] {view_name}: {count.iloc[0]['count']} rows")

                # Check age_band distribution in this view
                if view_name in ['medical_clean', 'pharmacy_clean', 'medical_filtered', 'medical_clean_age']:
                    age_dist = conn.sql(f"""
                        SELECT age_band, COUNT(*) as count
                        FROM {view_name}
                        GROUP BY age_band
                        ORDER BY count DESC
                    """).df()

                    logger.info(f"→ [{step_name}] {view_name} age_band distribution:")
                    for _, row in age_dist.iterrows():
                        logger.info(f"→ [{step_name}]   {row['age_band']}: {row['count']} rows")

                        # Flag unexpected age bands
                        if row['age_band'] != age_band:
                            logger.warning(f"→ [{step_name}] WARNING: Unexpected age_band '{row['age_band']}' in {view_name}")

            except Exception as view_e:
                logger.warning(f"→ [{step_name}] Could not check {view_name}: {str(view_e)}")

        logger.info(f"→ [{step_name}] ✓ Data consistency validation completed")
        return True

    except Exception as e:
        logger.error(f"→ [{step_name}] ERROR during data consistency validation: {str(e)}")
        return False




# -----------------------------------------------------------------------------
# Pickle diff utilities (migrated from scripts/diff_pickles.py)
# These helpers provide a reusable API for comparing two DataFrame pickles and
# producing a CSV of row-level differences. They are intended to be used from
# tests, notebooks, or invoked directly via the small CLI wrapper below.
# -----------------------------------------------------------------------------
from typing import Dict, Tuple, Optional


def _load_pickle(path: str) -> pd.DataFrame:
    try:
        obj = pd.read_pickle(path)
    except Exception as e:
        raise IOError(f"ERROR loading pickle {path}: {e}") from e
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Pickle at {path} does not contain a pandas.DataFrame (got {type(obj)})")
    return obj


def find_preferred_pickle(base_dir: str, name: str = 'target_analysis_data.pkl') -> Optional[str]:
    """Return path to preferred pickle file.

    Preference order:
      1. <base_dir>/outputs/<name>
      2. <base_dir>/<name> (legacy)

    Returns None if not found.
    """
    outputs_path = os.path.join(base_dir, 'outputs', name)
    legacy_path = os.path.join(base_dir, name)
    if os.path.exists(outputs_path):
        return outputs_path
    if os.path.exists(legacy_path):
        return legacy_path
    return None


def safe_load_pickle(path: Optional[str]):
    """Safely load an arbitrary pickle (not restricted to DataFrame).

    Returns the unpickled object or None on error.
    """
    if not path:
        return None
    try:
        # Prefer pandas for common parquet-backed pickles, fallback to pickle
        try:
            return pd.read_pickle(path)
        except Exception:
            import pickle as _pickle
            with open(path, 'rb') as f:
                return _pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load pickle {path}: {e}")
        return None


def _make_keyed_df(df: pd.DataFrame, key: Optional[str]) -> pd.DataFrame:
    if key is None:
        if df.index.name is None:
            df = df.copy()
            df.index.name = "_index"
        return df
    if key in df.columns:
        return df.set_index(key, drop=False)
    raise KeyError(f"Key column '{key}' not found in DataFrame columns: {list(df.columns)}")


def _diff_dfs(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    before = df_before.copy()
    after = df_after.copy()

    all_cols = sorted(set(before.columns) | set(after.columns))

    before = before.reindex(columns=all_cols)
    after = after.reindex(columns=all_cols)

    combined = pd.concat([before.add_prefix('before__'), after.add_prefix('after__')], axis=1, sort=False)

    change_mask = []
    for col in all_cols:
        bcol = f"before__{col}"
        acol = f"after__{col}"
        neq = ~(combined[bcol].eq(combined[acol]) | (combined[bcol].isna() & combined[acol].isna()))
        change_mask.append(neq)

    if change_mask:
        changed_any = pd.concat(change_mask, axis=1).any(axis=1)
    else:
        changed_any = pd.Series(False, index=combined.index)

    present_before = combined[[c for c in combined.columns if c.startswith('before__')]].notna().any(axis=1)
    present_after = combined[[c for c in combined.columns if c.startswith('after__')]].notna().any(axis=1)

    added = (~present_before) & present_after
    removed = present_before & (~present_after)
    changed = present_before & present_after & changed_any
    unchanged = present_before & present_after & (~changed_any)

    summary = {
        "total_keys": int(len(combined)),
        "added": int(added.sum()),
        "removed": int(removed.sum()),
        "changed": int(changed.sum()),
        "unchanged": int(unchanged.sum()),
    }

    combined['diff_status'] = 'unchanged'
    combined.loc[added, 'diff_status'] = 'added'
    combined.loc[removed, 'diff_status'] = 'removed'
    combined.loc[changed, 'diff_status'] = 'changed'

    return combined, summary


def diff_pickles(before_path: str, after_path: str, key: Optional[str] = None, out: str = 'diffs.csv') -> Dict[str, int]:
    """Diff two pandas DataFrame pickles and write a CSV with prefixed before/after columns.

    Returns a summary dict: total_keys, added, removed, changed, unchanged.
    """
    df_before = _load_pickle(before_path)
    df_after = _load_pickle(after_path)

    before_k = _make_keyed_df(df_before, key)
    after_k = _make_keyed_df(df_after, key)

    if not before_k.index.is_unique:
        before_k = before_k[~before_k.index.duplicated(keep='first')]
    if not after_k.index.is_unique:
        after_k = after_k[~after_k.index.duplicated(keep='first')]

    combined, summary = _diff_dfs(before_k, after_k)
    combined.to_csv(out)
    return summary



def normalize_to_all_targets(obj: object) -> Optional[pd.DataFrame]:
    """Normalize different saved shapes to a DataFrame with columns:
    event_year, target_code, frequency, target_system

    Accepts DataFrame or dict-like objects commonly produced by the
    frequency analysis scripts. Returns None when the object cannot be
    normalized.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        for c in ['event_year', 'target_code', 'frequency', 'target_system']:
            if c not in df.columns:
                df[c] = pd.NA
        return df[['event_year', 'target_code', 'frequency', 'target_system']]
    if isinstance(obj, dict):
        if 'all_targets' in obj and obj['all_targets'] is not None:
            return normalize_to_all_targets(obj['all_targets'])
        parts = []
        if obj.get('icd_aggregated') is not None:
            parts.append(obj['icd_aggregated'].assign(target_system='icd'))
        if obj.get('cpt_aggregated') is not None:
            parts.append(obj['cpt_aggregated'].assign(target_system='cpt'))
        if parts:
            out = pd.concat(parts, ignore_index=True)
            for c in ['event_year', 'target_code', 'frequency', 'target_system']:
                if c not in out.columns:
                    out[c] = pd.NA
            return out[['event_year', 'target_code', 'frequency', 'target_system']]
        dfs = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            for c in ['event_year', 'target_code', 'frequency', 'target_system']:
                if c not in out.columns:
                    out[c] = pd.NA
            return out[['event_year', 'target_code', 'frequency', 'target_system']]
    return None


def load_target_artifacts(outputs_dir: str = os.path.join('1_apcd_input_data', 'outputs'),
                          s3_parquet: Optional[str] = 's3://pgxdatalake/gold/target_code/target_code_latest.parquet') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve and load target-code analysis artifacts.

    Returns a tuple (t_orig, t_updated) where each item is a normalized
    DataFrame (see :func:`normalize_to_all_targets`). The function prefers
    pickles under `outputs_dir`. If the updated/canonical pickle is missing
    and `s3_parquet` is provided, it may attempt to read the parquet via
    DuckDB (when enabled).
    """
    # Resolve paths
    # Only consult the canonical outputs directory; do not fall back to legacy_base
    canonical_pk = find_preferred_pickle(outputs_dir, 'target_analysis_data.pkl')
    orig_pk = find_preferred_pickle(outputs_dir, 'target_analysis_data.orig.pkl')
    # The pipeline writes a stable "updated" artifact without a dot before 'updated'
    updated_pk = find_preferred_pickle(outputs_dir, 'target_analysis_data_updated.pkl')

    pd_orig_obj = safe_load_pickle(orig_pk)
    pd_updated_obj = safe_load_pickle(updated_pk)
    pd_canon_obj = safe_load_pickle(canonical_pk)

    t_updated = normalize_to_all_targets(pd_updated_obj or pd_canon_obj)
    t_orig = normalize_to_all_targets(pd_orig_obj)

    # If updated/canonical missing, fail loudly - require the canonical
    # pickle to be present so callers correct the pipeline that produces it.
    if t_updated is None or (isinstance(t_updated, pd.DataFrame) and t_updated.empty):
        raise FileNotFoundError(
            f"Canonical updated target-code artifact not found. Look for 'target_analysis_data_updated.pkl' under '{outputs_dir}'."
        )

    if t_orig is None:
        t_orig = pd.DataFrame(columns=['event_year', 'target_code', 'frequency', 'target_system'])

    # Apply any local target ICD mapping as a safety-net so callers always
    # receive canonical target codes regardless of whether the generator
    # applied mappings. This reads mapping files under the project's
    # 1_apcd_input_data/target_mapping or 1_apcd_input_data/claim_mappings
    # directories when present.
    try:
        mapping_paths = [
            os.path.join('1_apcd_input_data', 'target_mapping', 'target_icd_mapping.json'),
            os.path.join('1_apcd_input_data', 'claim_mappings', 'target_icd_mapping.json')
        ]
        map_dict = {}
        for mp in mapping_paths:
            if os.path.exists(mp):
                try:
                    with open(mp, 'r', encoding='utf-8') as fh:
                        j = json.load(fh)
                    if isinstance(j, dict):
                        map_dict.update(j)
                except Exception:
                    # Ignore mapping load errors; mapping is advisory
                    pass
        if map_dict:
            for df in (t_orig, t_updated):
                if isinstance(df, pd.DataFrame) and 'target_code' in df.columns:
                    df['target_code'] = df['target_code'].astype(str).replace(map_dict)
    except Exception:
        # Do not fail loader if mapping application fails; mapping is best-effort
        pass

    # Ensure numeric types for downstream work
    for df in (t_orig, t_updated):
        if isinstance(df, pd.DataFrame):
            if 'frequency' in df.columns:
                df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce').fillna(0).astype(int)
            if 'event_year' in df.columns:
                df['event_year'] = pd.to_numeric(df['event_year'], errors='coerce').fillna(0).astype(int)

    return t_orig, t_updated


def find_variants(df: pd.DataFrame, code_of_interest: str) -> list:
    """Return a sorted list of target_code variants matching code_of_interest.

    Matching is done by removing dots/spaces and doing a case-insensitive
    substring search on the flattened code.
    """
    if df is None or df.empty:
        return []
    needle = code_of_interest.replace('.', '').upper()
    tmp = df.copy()
    tmp['code_flat'] = tmp['target_code'].astype(str).str.upper().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)
    codes = tmp[tmp['code_flat'].str.contains(needle, na=False)].groupby('target_code', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)['target_code'].tolist()
    return codes


def totals_for_codes(df: pd.DataFrame, codes: list) -> pd.DataFrame:
    if df is None or df.empty or not codes:
        return pd.DataFrame(columns=['target_code', 'frequency'])
    return df[df['target_code'].isin(codes)].groupby('target_code', as_index=False)['frequency'].sum().rename(columns={'frequency': 'freq'})


def compare_totals(t_orig: pd.DataFrame, t_updated: pd.DataFrame) -> pd.DataFrame:
    """Compute orig/updated totals and deltas per target_code.

    Returns a DataFrame with columns: target_code, orig_total, updated_total, delta
    sorted by updated_total descending.
    """
    orig_totals = t_orig.groupby('target_code', as_index=False)['frequency'].sum().rename(columns={'frequency': 'orig_total'})
    upd_totals = t_updated.groupby('target_code', as_index=False)['frequency'].sum().rename(columns={'frequency': 'updated_total'})
    cmp = orig_totals.merge(upd_totals, on='target_code', how='outer').fillna(0)
    cmp['delta'] = cmp['updated_total'] - cmp['orig_total']
    cmp = cmp.sort_values('updated_total', ascending=False)
    return cmp


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Diff two pandas pickles (DataFrames) and write a CSV of differences.')
    p.add_argument('before', help='Path to before pickle')
    p.add_argument('after', help='Path to after pickle')
    p.add_argument('--key', help='Column name to use as key for alignment (default: index)', default=None)
    p.add_argument('--out', help='Output CSV path (default: diffs.csv)', default='diffs.csv')
    args = p.parse_args()

    try:
        summary = diff_pickles(args.before, args.after, key=args.key, out=args.out)
        print('Summary:')
        for k, v in summary.items():
            print(f'  {k}: {v}')
        print(f'Wrote diffs to {args.out}')
    except Exception as e:
        print(f'Error: {e}')
        raise



