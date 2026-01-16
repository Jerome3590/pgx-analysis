#!/usr/bin/env python3
"""
Create predictive time-window features between consecutive drug events.

This script creates features that are predictive (not leakage):
- Time intervals between consecutive drug events
- Time intervals between consecutive ICD/CPT events
- Temporal patterns in drug sequences (without referencing target event)

These features can be calculated for both target and control patients.

Usage:
    python create_predictive_time_features.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import duckdb
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_drug_time_intervals(
    model_data_path: Path,
    cohort_name: str,
    target_value: int = None,
    is_target_case: bool = None
) -> pd.DataFrame:
    """
    Extract time intervals between consecutive drug events for each patient.
    
    Returns DataFrame with features:
    - drug_interval_mean: Mean time (days) between consecutive drug events
    - drug_interval_median: Median time between consecutive drug events
    - drug_interval_std: Std dev of time intervals
    - drug_interval_min: Minimum interval
    - drug_interval_max: Maximum interval
    - drug_interval_count: Number of intervals (n_drug_events - 1)
    """
    con = duckdb.connect()
    
    # Extract drug events with dates, ordered by patient and date
    query = f"""
    WITH drug_events AS (
        SELECT
            mi_person_key,
            event_date,
            drug_name
        FROM read_parquet('{model_data_path}')
        WHERE drug_name IS NOT NULL
          AND drug_name != ''
          AND event_date IS NOT NULL
          {('AND target = ' + str(target_value)) if target_value is not None else ''}
          {('AND is_target_case = ' + str(int(is_target_case))) if is_target_case is not None else ''}
    ),
    ranked_drugs AS (
        SELECT
            mi_person_key,
            event_date,
            drug_name,
            LAG(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date) as prev_event_date
        FROM drug_events
    ),
    intervals AS (
        SELECT
            mi_person_key,
            event_date,
            prev_event_date,
            CASE
                WHEN prev_event_date IS NOT NULL
                THEN DATEDIFF('day', prev_event_date, event_date)
                ELSE NULL
            END as interval_days
        FROM ranked_drugs
    )
    SELECT
        mi_person_key,
        COUNT(*) as drug_interval_count,
        AVG(interval_days) as drug_interval_mean,
        MEDIAN(interval_days) as drug_interval_median,
        STDDEV(interval_days) as drug_interval_std,
        MIN(interval_days) as drug_interval_min,
        MAX(interval_days) as drug_interval_max
    FROM intervals
    WHERE interval_days IS NOT NULL
    GROUP BY mi_person_key
    """
    
    result = con.execute(query).df()
    con.close()
    
    return result


def extract_icd_time_intervals(
    model_data_path: Path,
    cohort_name: str,
    target_value: int = None,
    is_target_case: bool = None
) -> pd.DataFrame:
    """Extract time intervals between consecutive ICD events."""
    con = duckdb.connect()
    
    query = f"""
    WITH all_icds AS (
        SELECT mi_person_key, event_date, primary_icd_diagnosis_code as icd FROM read_parquet('{model_data_path}')
        WHERE primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code != ''
          AND event_date IS NOT NULL
          {'AND target = ' + str(target_value) if target_value is not None else ''}
        UNION ALL
        SELECT mi_person_key, event_date, two_icd_diagnosis_code as icd FROM read_parquet('{model_data_path}')
        WHERE two_icd_diagnosis_code IS NOT NULL AND two_icd_diagnosis_code != ''
          AND event_date IS NOT NULL
          {'AND target = ' + str(target_value) if target_value is not None else ''}
        UNION ALL
        SELECT mi_person_key, event_date, three_icd_diagnosis_code as icd FROM read_parquet('{model_data_path}')
        WHERE three_icd_diagnosis_code IS NOT NULL AND three_icd_diagnosis_code != ''
          AND event_date IS NOT NULL
          {'AND target = ' + str(target_value) if target_value is not None else ''}
    ),
    ranked_icds AS (
        SELECT
            mi_person_key,
            event_date,
            LAG(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date) as prev_event_date
        FROM all_icds
    ),
    intervals AS (
        SELECT
            mi_person_key,
            CASE
                WHEN prev_event_date IS NOT NULL
                THEN DATEDIFF('day', prev_event_date, event_date)
                ELSE NULL
            END as interval_days
        FROM ranked_icds
    )
    SELECT
        mi_person_key,
        COUNT(*) as icd_interval_count,
        AVG(interval_days) as icd_interval_mean,
        MEDIAN(interval_days) as icd_interval_median,
        STDDEV(interval_days) as icd_interval_std,
        MIN(interval_days) as icd_interval_min,
        MAX(interval_days) as icd_interval_max
    FROM intervals
    WHERE interval_days IS NOT NULL
    GROUP BY mi_person_key
    """
    
    result = con.execute(query).df()
    con.close()
    
    return result


def extract_cpt_time_intervals(
    model_data_path: Path,
    cohort_name: str,
    target_value: int = None,
    is_target_case: bool = None
) -> pd.DataFrame:
    """Extract time intervals between consecutive CPT events."""
    con = duckdb.connect()
    
    query = f"""
    WITH cpt_events AS (
        SELECT
            mi_person_key,
            event_date,
            procedure_code as cpt
        FROM read_parquet('{model_data_path}')
        WHERE procedure_code IS NOT NULL
          AND procedure_code != ''
          AND event_date IS NOT NULL
          {('AND target = ' + str(target_value)) if target_value is not None else ''}
          {('AND is_target_case = ' + str(int(is_target_case))) if is_target_case is not None else ''}
    ),
    ranked_cpts AS (
        SELECT
            mi_person_key,
            event_date,
            LAG(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date) as prev_event_date
        FROM cpt_events
    ),
    intervals AS (
        SELECT
            mi_person_key,
            CASE
                WHEN prev_event_date IS NOT NULL
                THEN DATEDIFF('day', prev_event_date, event_date)
                ELSE NULL
            END as interval_days
        FROM ranked_cpts
    )
    SELECT
        mi_person_key,
        COUNT(*) as cpt_interval_count,
        AVG(interval_days) as cpt_interval_mean,
        MEDIAN(interval_days) as cpt_interval_median,
        STDDEV(interval_days) as cpt_interval_std,
        MIN(interval_days) as cpt_interval_min,
        MAX(interval_days) as cpt_interval_max
    FROM intervals
    WHERE interval_days IS NOT NULL
    GROUP BY mi_person_key
    """
    
    result = con.execute(query).df()
    con.close()
    
    return result


def create_predictive_time_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """Create predictive time-window features for both target and control patients."""
    
    age_band_fname = age_band.replace("-", "_")
    
    model_data_path = (
        project_root
        / "model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )
    
    if not model_data_path.exists():
        raise FileNotFoundError(f"Model data not found: {model_data_path}")
    
    print(f"[INFO] Creating predictive time features from {model_data_path}")
    
    # Check if this is non_opioid_ed (has is_target_case column)
    con_check = duckdb.connect()
    has_is_target_case = False
    try:
        con_check.execute(
            f"SELECT is_target_case FROM read_parquet('{model_data_path}') LIMIT 1"
        ).df()
        has_is_target_case = True
    except Exception:
        pass
    con_check.close()
    
    # Extract features for target patients
    print("\n[INFO] Extracting time intervals for target patients...")
    if has_is_target_case:
        drug_target = extract_drug_time_intervals(model_data_path, cohort_name, is_target_case=True)
        icd_target = extract_icd_time_intervals(model_data_path, cohort_name, is_target_case=True)
        cpt_target = extract_cpt_time_intervals(model_data_path, cohort_name, is_target_case=True)
    else:
        drug_target = extract_drug_time_intervals(model_data_path, cohort_name, target_value=1)
        icd_target = extract_icd_time_intervals(model_data_path, cohort_name, target_value=1)
        cpt_target = extract_cpt_time_intervals(model_data_path, cohort_name, target_value=1)
    
    # Merge target features
    target_features = drug_target.merge(icd_target, on='mi_person_key', how='outer')
    target_features = target_features.merge(cpt_target, on='mi_person_key', how='outer')
    target_features['target'] = 1
    print(f"[INFO] Created features for {len(target_features)} target patients")
    
    # Extract features for control patients
    print("\n[INFO] Extracting time intervals for control patients...")
    if has_is_target_case:
        drug_control = extract_drug_time_intervals(model_data_path, cohort_name, is_target_case=False)
        icd_control = extract_icd_time_intervals(model_data_path, cohort_name, is_target_case=False)
        cpt_control = extract_cpt_time_intervals(model_data_path, cohort_name, is_target_case=False)
    else:
        drug_control = extract_drug_time_intervals(model_data_path, cohort_name, target_value=0)
        icd_control = extract_icd_time_intervals(model_data_path, cohort_name, target_value=0)
        cpt_control = extract_cpt_time_intervals(model_data_path, cohort_name, target_value=0)
    
    # Merge control features
    control_features = drug_control.merge(icd_control, on='mi_person_key', how='outer')
    control_features = control_features.merge(cpt_control, on='mi_person_key', how='outer')
    control_features['target'] = 0
    print(f"[INFO] Created features for {len(control_features)} control patients")
    
    # Combine target and control
    all_features = pd.concat([target_features, control_features], ignore_index=True)
    
    # Fill NaN with 0 (patients with no intervals)
    feature_cols = [c for c in all_features.columns if c != 'mi_person_key' and c != 'target']
    all_features[feature_cols] = all_features[feature_cols].fillna(0)
    
    # Ensure mi_person_key is string
    all_features['mi_person_key'] = all_features['mi_person_key'].astype(str)
    
    print(f"\n[INFO] Total features created: {len(all_features)} patients, {len(feature_cols)} features")
    print("[INFO] Feature breakdown:")
    print(f"  Drug interval features: {len([c for c in feature_cols if 'drug_interval' in c])}")
    print(f"  ICD interval features: {len([c for c in feature_cols if 'icd_interval' in c])}")
    print(f"  CPT interval features: {len([c for c in feature_cols if 'cpt_interval' in c])}")
    
    # Save results
    output_dir = (
        project_root
        / "6_dtw_analysis"
        / "outputs"
        / "feature_engineering"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"predictive_time_features_{cohort_name}_{age_band_fname}.csv"
    
    print(f"\n[INFO] Saving predictive time features to {output_path}")
    all_features.to_csv(output_path, index=False)
    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Create predictive time-window features"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    create_predictive_time_features(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

