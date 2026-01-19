#!/usr/bin/env python3
"""
Analyze Z codes (routine examinations/administrative encounters) across cohorts.

This script:
1. Identifies Z codes in model_events data
2. Calculates time windows (days from target event date)
3. Compares Z code patterns between standard and extreme cohorts
4. Generates summary statistics and visualizations
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import COHORT_NAMES, AGE_BANDS, ALL_ICD_DIAGNOSIS_COLUMNS

MODEL_DATA_ROOT = PROJECT_ROOT / "4a_model_data"
OUTPUT_DIR = PROJECT_ROOT / "4b_event_filter" / "outputs" / "z_code_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_z_code(code: str) -> bool:
    """Check if a code is a Z code (starts with Z)."""
    if pd.isna(code):
        return False
    code_str = str(code).strip().upper()
    return code_str.startswith('Z') and len(code_str) >= 2


def extract_z_codes_from_event(event_row: pd.Series) -> List[str]:
    """Extract all Z codes from an event row (checking all ICD diagnosis columns)."""
    z_codes = []
    for col in ALL_ICD_DIAGNOSIS_COLUMNS:
        if col in event_row.index:
            code = event_row.get(col)
            if is_z_code(code):
                z_codes.append(str(code).strip().upper())
    return z_codes


def calculate_days_from_target(event_date: pd.Timestamp, target_date: pd.Timestamp) -> Optional[float]:
    """Calculate days from target event date (negative = before, positive = after)."""
    if pd.isna(event_date) or pd.isna(target_date):
        return None
    try:
        delta = (pd.to_datetime(event_date) - pd.to_datetime(target_date)).days
        return float(delta)
    except:
        return None


def get_target_date_field(cohort_name: str) -> str:
    """Get the target date field name for a cohort."""
    if "opioid" in cohort_name.lower():
        return "first_opioid_ed_date"
    else:
        return "first_ed_non_opioid_date"


def get_patient_target_dates(cohort_name: str, age_band: str) -> pd.DataFrame:
    """Get patient-level target dates from cohort data using DuckDB."""
    import os
    
    # Try to find cohort parquet files
    cohort_root = PROJECT_ROOT / "data" / "gold_cohorts"
    if not cohort_root.exists():
        # Try alternative location
        env_path = os.getenv("LOCAL_DATA_PATH")
        if env_path:
            cohort_root = Path(env_path)
        else:
            cohort_root = PROJECT_ROOT / "data" / "gold_cohorts"
    
    target_date_field = get_target_date_field(cohort_name)
    
    # Query cohort data for target dates using DuckDB
    conn = duckdb.connect()
    
    # Build UNION query for all years
    union_queries = []
    for year in [2016, 2017, 2018, 2019]:
        cohort_file = cohort_root / f"cohort_name={cohort_name}" / f"event_year={year}" / f"age_band={age_band}" / "cohort.parquet"
        if cohort_file.exists():
            union_queries.append(f"""
                SELECT DISTINCT 
                    mi_person_key as patient_id,
                    {target_date_field} as target_date
                FROM read_parquet('{cohort_file}')
                WHERE {target_date_field} IS NOT NULL
            """)
    
    if not union_queries:
        conn.close()
        return pd.DataFrame(columns=['patient_id', 'target_date'])
    
    # Combine all years and get first target date per patient
    query = f"""
    WITH all_target_dates AS (
        {' UNION ALL '.join(union_queries)}
    )
    SELECT 
        patient_id,
        MIN(target_date) as target_date
    FROM all_target_dates
    GROUP BY patient_id
    """
    
    target_dates_df = conn.execute(query).df()
    conn.close()
    
    return target_dates_df


def analyze_z_codes_in_cohort(
    cohort_name: str,
    age_band: str,
    is_extreme: bool = False
) -> pd.DataFrame:
    """
    Analyze Z codes in a cohort.
    
    Returns:
        DataFrame with columns: z_code, days_from_target, event_date, target_date, patient_id
    """
    import os
    
    age_band_fname = age_band.replace("-", "_")
    
    if is_extreme:
        cohort_path = MODEL_DATA_ROOT / f"cohort_name={cohort_name}_extreme_density" / f"age_band={age_band}"
    else:
        cohort_path = MODEL_DATA_ROOT / f"cohort_name={cohort_name}" / f"age_band={age_band}"
    
    # Try to load filtered version first, then fall back to regular
    parquet_file = cohort_path / "model_events_no_protocols.parquet"
    if not parquet_file.exists():
        parquet_file = cohort_path / "model_events.parquet"
    
    if not parquet_file.exists():
        print(f"  [SKIP] File not found: {parquet_file}")
        return pd.DataFrame()
    
    print(f"  [LOAD] {parquet_file.name}")
    
    # Use DuckDB to query parquet file
    conn = duckdb.connect()
    
    # Get patient ID column name (check what's available)
    sample_query = f"SELECT * FROM read_parquet('{parquet_file}') LIMIT 1"
    sample_df = conn.execute(sample_query).df()
    
    if len(sample_df) == 0:
        conn.close()
        print(f"  [SKIP] Empty file")
        return pd.DataFrame()
    
    patient_id_col = 'mi_person_key' if 'mi_person_key' in sample_df.columns else 'patient_id'
    if patient_id_col not in sample_df.columns:
        conn.close()
        print(f"  [SKIP] Patient ID column not found")
        return pd.DataFrame()
    
    # Get target dates from cohort data
    target_dates_df = get_patient_target_dates(cohort_name, age_band)
    
    if len(target_dates_df) == 0:
        print(f"  [WARN] No target dates found - will analyze without time windows")
    
    # Build ICD column list for SQL
    icd_columns = ', '.join([f"'{col}'" for col in ALL_ICD_DIAGNOSIS_COLUMNS if col in sample_df.columns])
    
    # Query events with Z codes using DuckDB
    # Check each ICD column for Z codes
    z_code_conditions = []
    available_icd_cols = [col for col in ALL_ICD_DIAGNOSIS_COLUMNS if col in sample_df.columns]
    
    if not available_icd_cols:
        conn.close()
        print(f"  [SKIP] No ICD diagnosis columns found")
        return pd.DataFrame()
    
    for col in available_icd_cols:
        z_code_conditions.append(f"UPPER(TRIM(CAST({col} AS VARCHAR))) LIKE 'Z%'")
    
    z_condition = " OR ".join(z_code_conditions)
    
    # Build UNION query to extract each Z code from each column
    union_parts = []
    for col in available_icd_cols:
        union_parts.append(f"""
            SELECT 
                {patient_id_col} as patient_id,
                event_date,
                UPPER(TRIM(CAST({col} AS VARCHAR))) as z_code
            FROM read_parquet('{parquet_file}')
            WHERE UPPER(TRIM(CAST({col} AS VARCHAR))) LIKE 'Z%'
        """)
    
    # Get target dates table if available
    if len(target_dates_df) > 0:
        # Register target dates as a temporary table
        conn.register('target_dates', target_dates_df)
        target_join = "LEFT JOIN target_dates td ON events.patient_id = td.patient_id"
        
        # Query to extract Z codes and calculate time windows with target dates
        query = f"""
        WITH z_code_events AS (
            {' UNION ALL '.join(union_parts)}
        )
        SELECT DISTINCT
            events.patient_id,
            events.event_date,
            events.z_code,
            td.target_date,
            CASE 
                WHEN events.event_date IS NOT NULL AND td.target_date IS NOT NULL
                THEN CAST(datediff('day', events.event_date::DATE, td.target_date::DATE) AS INTEGER)
                ELSE NULL
            END as days_from_target
        FROM z_code_events events
        {target_join}
        WHERE events.z_code IS NOT NULL AND LENGTH(events.z_code) >= 2
        """
    else:
        # Query without target dates
        query = f"""
        WITH z_code_events AS (
            {' UNION ALL '.join(union_parts)}
        )
        SELECT DISTINCT
            events.patient_id,
            events.event_date,
            events.z_code,
            NULL as target_date,
            NULL as days_from_target
        FROM z_code_events events
        WHERE events.z_code IS NOT NULL AND LENGTH(events.z_code) >= 2
        """
    
    result_df = conn.execute(query).df()
    conn.close()
    
    if len(result_df) == 0:
        print(f"  [INFO] No Z codes found")
        return pd.DataFrame()
    
    # Add metadata columns
    result_df['cohort'] = cohort_name
    result_df['age_band'] = age_band
    result_df['is_extreme'] = is_extreme
    
    # Reorder columns
    column_order = ['z_code', 'days_from_target', 'event_date', 'target_date', 
                    'patient_id', 'cohort', 'age_band', 'is_extreme']
    result_df = result_df[[col for col in column_order if col in result_df.columns]]
    
    return result_df


def analyze_extreme_vs_standard(cohort_name: str, age_band: str) -> pd.DataFrame:
    """Analyze Z codes comparing extreme vs standard cohorts for a specific cohort/age_band."""
    all_results = []
    
    print(f"Cohort: {cohort_name}, Age band: {age_band}")
    print("-" * 80)
    
    # Standard (non-extreme) cohort
    standard_df = analyze_z_codes_in_cohort(cohort_name, age_band, is_extreme=False)
    if len(standard_df) > 0:
        all_results.append(standard_df)
        print(f"  Standard: {len(standard_df)} Z code events")
    
    # Extreme cohort
    extreme_df = analyze_z_codes_in_cohort(cohort_name, age_band, is_extreme=True)
    if len(extreme_df) > 0:
        all_results.append(extreme_df)
        print(f"  Extreme: {len(extreme_df)} Z code events")
    
    if not all_results:
        print("  No Z codes found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for Z code analysis."""
    stats = {}
    
    if len(df) == 0:
        return stats
    
    # Overall statistics
    stats['total_z_code_events'] = len(df)
    stats['unique_z_codes'] = df['z_code'].nunique()
    stats['unique_patients'] = df['patient_id'].nunique()
    
    # By cohort type
    for is_extreme in [False, True]:
        cohort_type = 'extreme' if is_extreme else 'standard'
        subset = df[df['is_extreme'] == is_extreme]
        
        stats[f'{cohort_type}_total_events'] = len(subset)
        stats[f'{cohort_type}_unique_z_codes'] = subset['z_code'].nunique()
        stats[f'{cohort_type}_unique_patients'] = subset['patient_id'].nunique()
        
        if len(subset) > 0:
            days_data = subset['days_from_target'].dropna()
            if len(days_data) > 0:
                stats[f'{cohort_type}_days_mean'] = float(days_data.mean())
                stats[f'{cohort_type}_days_median'] = float(days_data.median())
                stats[f'{cohort_type}_days_std'] = float(days_data.std())
                stats[f'{cohort_type}_days_min'] = float(days_data.min())
                stats[f'{cohort_type}_days_max'] = float(days_data.max())
                stats[f'{cohort_type}_days_q25'] = float(days_data.quantile(0.25))
                stats[f'{cohort_type}_days_q75'] = float(days_data.quantile(0.75))
                
                # Count events before vs after target
                stats[f'{cohort_type}_events_before_target'] = int((days_data < 0).sum())
                stats[f'{cohort_type}_events_after_target'] = int((days_data >= 0).sum())
    
    # Top Z codes
    top_z_codes = df['z_code'].value_counts().head(20)
    stats['top_z_codes'] = top_z_codes.to_dict()
    
    # Z code categories
    z_categories = {}
    for z_code in df['z_code'].unique():
        category = z_code[:3] if len(z_code) >= 3 else z_code
        if category not in z_categories:
            z_categories[category] = 0
        z_categories[category] += len(df[df['z_code'] == z_code])
    
    stats['z_code_categories'] = dict(sorted(z_categories.items(), key=lambda x: x[1], reverse=True))
    
    return stats


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Z codes: Extreme vs Standard cohorts')
    parser.add_argument('--cohort', type=str, default='opioid_ed',
                       help='Cohort name (default: opioid_ed)')
    parser.add_argument('--age-band', type=str, default=None,
                       help='Age band (default: all age bands)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Z Code Analysis: Standard vs Extreme Cohorts")
    print("=" * 80)
    print()
    
    # Analyze extreme vs standard for specified cohort(s)
    cohort_name = args.cohort
    age_bands_to_analyze = [args.age_band] if args.age_band else AGE_BANDS
    
    all_results = []
    for age_band in age_bands_to_analyze:
        df = analyze_extreme_vs_standard(cohort_name, age_band)
        if len(df) > 0:
            all_results.append(df)
        print()
    
    if not all_results:
        print("No Z codes found. Exiting.")
        return
    
    df = pd.concat(all_results, ignore_index=True)
    
    if len(df) == 0:
        print("No Z codes found. Exiting.")
        return
    
    print()
    print("=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Save full results
    cohort_fname = cohort_name.replace("_", "-")
    if args.age_band:
        age_fname = args.age_band.replace("-", "_")
        output_csv = OUTPUT_DIR / f"z_code_analysis_{cohort_fname}_{age_fname}.csv"
    else:
        output_csv = OUTPUT_DIR / f"z_code_analysis_{cohort_fname}_all_ages.csv"
    df.to_csv(output_csv, index=False)
    print(f"[OK] Saved full results: {output_csv}")
    print(f"      Total Z code events: {len(df):,}")
    
    # Generate and save summary statistics
    stats = generate_summary_statistics(df)
    
    import json
    if args.age_band:
        age_fname = args.age_band.replace("-", "_")
        stats_json = OUTPUT_DIR / f"z_code_summary_{cohort_fname}_{age_fname}.json"
    else:
        stats_json = OUTPUT_DIR / f"z_code_summary_{cohort_fname}_all_ages.json"
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved summary statistics: {stats_json}")
    
    # Print summary
    print()
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()
    print(f"Total Z code events: {stats['total_z_code_events']:,}")
    print(f"Unique Z codes: {stats['unique_z_codes']}")
    print(f"Unique patients: {stats['unique_patients']:,}")
    print()
    
    print("Standard Cohort:")
    print(f"  Events: {stats.get('standard_total_events', 0):,}")
    print(f"  Unique Z codes: {stats.get('standard_unique_z_codes', 0)}")
    print(f"  Unique patients: {stats.get('standard_unique_patients', 0):,}")
    if 'standard_days_mean' in stats:
        print(f"  Days from target - Mean: {stats['standard_days_mean']:.1f}, Median: {stats['standard_days_median']:.1f}")
        print(f"  Events before target: {stats.get('standard_events_before_target', 0):,}")
        print(f"  Events after target: {stats.get('standard_events_after_target', 0):,}")
    
    print()
    print("Extreme Cohort:")
    print(f"  Events: {stats.get('extreme_total_events', 0):,}")
    print(f"  Unique Z codes: {stats.get('extreme_unique_z_codes', 0)}")
    print(f"  Unique patients: {stats.get('extreme_unique_patients', 0):,}")
    if 'extreme_days_mean' in stats:
        print(f"  Days from target - Mean: {stats['extreme_days_mean']:.1f}, Median: {stats['extreme_days_median']:.1f}")
        print(f"  Events before target: {stats.get('extreme_events_before_target', 0):,}")
        print(f"  Events after target: {stats.get('extreme_events_after_target', 0):,}")
    
    print()
    print("Top Z Code Categories:")
    for category, count in list(stats.get('z_code_categories', {}).items())[:10]:
        print(f"  {category}: {count:,} events")
    
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
