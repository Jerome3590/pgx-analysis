#!/usr/bin/env python3
"""
Create BupaR Post-Target Analysis CSV

Analyzes BupaR post-target outputs to identify which features (ICD/CPT codes, drugs)
appear primarily after the target event, indicating post-target leakage.

For opioid_ed cohort: Target is F1120 (opioid dependence ICD code)
For non_opioid_ed cohort: Target is ED visit (HCG line with drug window)

This script:
1. Loads post-target traces/features from BupaR outputs
2. Compares with pre-target features to identify post-target leakage
3. Creates a summary CSV with feature names and is_post_target_leakage flag
"""

import argparse
import sys
import os
import platform
import pandas as pd
from pathlib import Path
from typing import Set, Dict
import json

# Detect operating system and set project root
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

if IS_WINDOWS:
    # Windows: Use current workspace directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
elif IS_LINUX:
    # Linux/EC2: Use EC2 path
    PROJECT_ROOT = Path('/home/pgx3874/pgx-analysis')
else:
    # Fallback: Use current file's parent directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def extract_features_from_traces(traces_df: pd.DataFrame) -> Set[str]:
    """
    Extract unique feature names from BupaR traces.
    
    Traces contain activity sequences like "ICD:F1120", "CPT:80307", "DRUG:SUBOXONE"
    """
    features = set()
    
    if traces_df.empty or 'trace' not in traces_df.columns:
        return features
    
    for trace in traces_df['trace']:
        if pd.isna(trace):
            continue
        
        # Split trace by separator (typically ">>" or ",")
        activities = str(trace).replace('>>', ',').split(',')
        for activity in activities:
            activity = activity.strip()
            if ':' in activity:
                # Extract code/drug name after prefix (ICD:, CPT:, DRUG:)
                parts = activity.split(':', 1)
                if len(parts) == 2:
                    prefix, code = parts
                    # Store as feature name (e.g., "item_icd_80307", "item_drug_SUBOXONE")
                    if prefix.upper() == 'ICD':
                        features.add(f"item_icd_{code.strip()}")
                    elif prefix.upper() == 'CPT':
                        features.add(f"item_cpt_{code.strip()}")
                    elif prefix.upper() == 'DRUG':
                        features.add(f"item_drug_{code.strip()}")
    
    return features


def extract_features_from_patient_features(features_df: pd.DataFrame) -> Set[str]:
    """
    Extract feature names from BupaR patient features DataFrame.
    
    Looks for columns that represent feature counts or indicators.
    """
    features = set()
    
    if features_df.empty:
        return features
    
    # Look for columns that might contain feature information
    # This is a simplified approach - may need adjustment based on actual BupaR output format
    for col in features_df.columns:
        if 'feature' in col.lower() or 'item' in col.lower():
            # If column contains feature names
            if features_df[col].dtype == 'object':
                features.update(features_df[col].dropna().unique())
    
    return features


def analyze_post_target_leakage_from_events(
    cohort: str,
    age_band: str,
    project_root: Path,
    post_target_threshold: float = 0.8,
    min_events: int = 5
) -> pd.DataFrame:
    """
    Analyze event-level data to identify post-target leakage features.
    
    For each feature (ICD/CPT code, drug), calculates:
    - Total occurrences in target cases
    - Occurrences BEFORE target event (F1120 for opioid_ed, ED visit for non_opioid_ed)
    - Occurrences AFTER target event
    - Post-target ratio (post / total)
    
    A feature is flagged as leakage if:
    - post_target_ratio > threshold (default 0.8 = 80% of occurrences are post-target)
    - AND has minimum number of events (default 5) for statistical significance
    
    Args:
        cohort: Cohort name (opioid_ed or non_opioid_ed)
        age_band: Age band
        project_root: Project root directory
        post_target_threshold: Threshold for post-target ratio (0.0-1.0)
        min_events: Minimum number of events required for analysis
    
    Returns:
        DataFrame with columns: feature, is_post_target_leakage, is_pre_target_predictive,
        pre_target_ratio, post_target_ratio, pre_count, post_count, total_count, unique_patients
    """
    import duckdb
    
    age_band_fname = age_band_to_fname(age_band)
    
    # Find model_events.parquet file
    # Check EC2 path first (/mnt/nvme), then project root
    data_root = os.getenv("PGX_DATA_ROOT", "")
    if not data_root and IS_LINUX:
        data_root = "/mnt/nvme"
    elif not data_root:
        data_root = str(project_root)
    
    model_data_paths = [
        Path(data_root) / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        project_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        project_root / "model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
    ]
    
    model_data_path = None
    for path in model_data_paths:
        if path.exists():
            model_data_path = path
            break
    
    if not model_data_path:
        print(f"[ERROR] Model events file not found. Checked:")
        for path in model_data_paths:
            print(f"  - {path}")
        return pd.DataFrame()
    
    print(f"Loading event data from: {model_data_path}")
    
    con = duckdb.connect()
    
    # Query to analyze each feature's pre/post target distribution
    # For opioid_ed cohort, find first F1120 event date for each patient
    # For non_opioid_ed cohort, find first ED visit (HCG) event date for each patient
    # Escape the path for SQL
    model_data_path_str = str(model_data_path).replace("'", "''")
    
    if cohort == "opioid_ed":
        # Find first F1120 event date for each target patient
        query = f"""
    WITH target_patients AS (
        SELECT DISTINCT
            mi_person_key,
            MIN(CAST(event_date AS DATE)) as target_date
        FROM read_parquet('{model_data_path_str}')
        WHERE target = 1
          AND (
            primary_icd_diagnosis_code LIKE '%F1120%'
            OR two_icd_diagnosis_code LIKE '%F1120%'
            OR three_icd_diagnosis_code LIKE '%F1120%'
            OR four_icd_diagnosis_code LIKE '%F1120%'
            OR five_icd_diagnosis_code LIKE '%F1120%'
            OR six_icd_diagnosis_code LIKE '%F1120%'
            OR seven_icd_diagnosis_code LIKE '%F1120%'
            OR eight_icd_diagnosis_code LIKE '%F1120%'
            OR nine_icd_diagnosis_code LIKE '%F1120%'
            OR ten_icd_diagnosis_code LIKE '%F1120%'
          )
        GROUP BY mi_person_key
        HAVING target_date IS NOT NULL
    ),
    """
    else:
        # For non_opioid_ed, find first ED event (where hcg_line IS NOT NULL) for each patient
        query = f"""
    WITH target_patients AS (
        SELECT DISTINCT
            mi_person_key,
            MIN(CASE WHEN hcg_line IS NOT NULL THEN CAST(event_date AS DATE) END) as target_date
        FROM read_parquet('{model_data_path_str}')
        WHERE target = 1
        GROUP BY mi_person_key
        HAVING target_date IS NOT NULL
    ),
    """
    
    # Continue with the rest of the query (same for both cohorts)
    query += f"""
    events_with_target_dates AS (
        SELECT 
            e.mi_person_key,
            e.event_date,
            e.target,
            t.target_date,
            -- Extract codes/drugs from all relevant columns
            COALESCE(
                e.primary_icd_diagnosis_code,
                e.two_icd_diagnosis_code,
                e.three_icd_diagnosis_code,
                e.four_icd_diagnosis_code,
                e.five_icd_diagnosis_code,
                e.six_icd_diagnosis_code,
                e.seven_icd_diagnosis_code,
                e.eight_icd_diagnosis_code,
                e.nine_icd_diagnosis_code,
                e.ten_icd_diagnosis_code
            ) as icd_code,
            e.procedure_code as cpt_code,
            e.drug_name
        FROM read_parquet('{model_data_path_str}') e
        LEFT JOIN target_patients t ON e.mi_person_key = t.mi_person_key
        WHERE e.target = 1  -- Only analyze target cases
    ),
    -- Flatten to individual code/drug events
    feature_events AS (
        SELECT 
            mi_person_key,
            event_date,
            target_date,
            CASE 
                WHEN event_date < target_date THEN 'pre'
                WHEN event_date >= target_date THEN 'post'
                ELSE 'unknown'
            END as timing,
            'item_icd_' || REPLACE(UPPER(icd_code), '.', '') as feature
        FROM events_with_target_dates
        WHERE icd_code IS NOT NULL AND target_date IS NOT NULL
        
        UNION ALL
        
        SELECT 
            mi_person_key,
            event_date,
            target_date,
            CASE 
                WHEN event_date < target_date THEN 'pre'
                WHEN event_date >= target_date THEN 'post'
                ELSE 'unknown'
            END as timing,
            'item_cpt_' || UPPER(REPLACE(REPLACE(cpt_code, ' ', ''), '.', '')) as feature
        FROM events_with_target_dates
        WHERE cpt_code IS NOT NULL AND target_date IS NOT NULL
        
        UNION ALL
        
        SELECT 
            mi_person_key,
            event_date,
            target_date,
            CASE 
                WHEN event_date < target_date THEN 'pre'
                WHEN event_date >= target_date THEN 'post'
                ELSE 'unknown'
            END as timing,
            'item_drug_' || UPPER(REPLACE(REPLACE(drug_name, ' ', '_'), '-', '_')) as feature
        FROM events_with_target_dates
        WHERE drug_name IS NOT NULL AND target_date IS NOT NULL
    ),
    -- Calculate statistics per feature
    feature_stats AS (
        SELECT 
            feature,
            COUNT(*) as total_count,
            SUM(CASE WHEN timing = 'pre' THEN 1 ELSE 0 END) as pre_count,
            SUM(CASE WHEN timing = 'post' THEN 1 ELSE 0 END) as post_count,
            COUNT(DISTINCT mi_person_key) as unique_patients,
            CAST(SUM(CASE WHEN timing = 'pre' THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) as pre_target_ratio,
            CAST(SUM(CASE WHEN timing = 'post' THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) as post_target_ratio
        FROM feature_events
        WHERE timing IN ('pre', 'post')
        GROUP BY feature
        HAVING COUNT(*) >= {min_events}  -- Minimum events for statistical significance
    )
    SELECT 
        feature,
        total_count,
        pre_count,
        post_count,
        unique_patients,
        pre_target_ratio,
        post_target_ratio,
        CASE 
            WHEN post_target_ratio >= {post_target_threshold} THEN 1
            ELSE 0
        END as is_post_target_leakage,
        CASE 
            WHEN pre_target_ratio >= 0.8 THEN 1  -- Primarily pre-target (predictive)
            ELSE 0
        END as is_pre_target_predictive
    FROM feature_stats
    ORDER BY post_target_ratio DESC, total_count DESC
    """.format(min_events=min_events, post_target_threshold=post_target_threshold)
    
    try:
        results_df = con.execute(query).df()
        con.close()
        
        print(f"[OK] Analyzed {len(results_df)} features from event data")
        
        if len(results_df) > 0:
            leakage_count = results_df['is_post_target_leakage'].sum()
            predictive_count = results_df.get('is_pre_target_predictive', pd.Series([0]*len(results_df))).sum()
            
            # Check if there are any pre-F1120 events at all
            total_pre_events = results_df.get('pre_count', pd.Series([0]*len(results_df))).sum()
            total_post_events = results_df.get('post_count', pd.Series([0]*len(results_df))).sum()
            
            if total_pre_events == 0 and total_post_events > 0:
                print(f"\n   [WARN] CRITICAL FINDING: No pre-F1120 events found!")
                print(f"   All {total_post_events:,} events occur AFTER the target event (F1120).")
                print(f"   This means ALL features are post-target leakage and should be filtered.")
                print(f"   Consider checking the data filtering or cohort definition.")
            else:
                print(f"   Features with post-F1120 ratio >= {post_f1120_threshold:.0%} (leakage): {leakage_count}")
                print(f"   Features with pre-F1120 ratio >= 80% (predictive): {predictive_count}")
                print(f"   Features with mixed timing: {len(results_df) - leakage_count - predictive_count}")
                print(f"   Total pre-F1120 events: {total_pre_events:,}")
                print(f"   Total post-F1120 events: {total_post_events:,}")
        
        return results_df
        
    except Exception as e:
        print(f"[ERROR] Failed to analyze event data: {e}")
        import traceback
        traceback.print_exc()
        con.close()
        return pd.DataFrame()


def analyze_post_target_leakage(
    cohort: str,
    age_band: str,
    project_root: Path,
    use_event_data: bool = True,
    post_target_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Analyze post-target leakage features using event-level data (preferred) or BupaR outputs.
    
    Args:
        cohort: Cohort name (opioid_ed or non_opioid_ed)
        age_band: Age band
        project_root: Project root directory
        use_event_data: If True, use event-level data for accurate analysis (default: True)
        post_target_threshold: Threshold for post-target ratio (0.0-1.0, default: 0.8)
    
    Returns:
        DataFrame with leakage analysis results
    """
    # Cohort-specific target terminology
    target_name = "F1120" if cohort == "opioid_ed" else "ED visit (HCG)"
    
    if use_event_data:
        print(f"\n[INFO] Using event-level data for post-target leakage analysis")
        print(f"       This provides accurate pre/post {target_name} ratios for each feature")
        return analyze_post_target_leakage_from_events(
            cohort=cohort,
            age_band=age_band,
            project_root=project_root,
            post_target_threshold=post_target_threshold
        )
    
    # Fallback to BupaR outputs (less accurate, but available if event data missing)
    print(f"\n[INFO] Using BupaR outputs for post-target leakage analysis")
    age_band_fname = age_band_to_fname(age_band)
    output_dir = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname
    
    # Paths to BupaR output files
    post_traces_path = output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_post_f1120_traces_bupar.csv"
    pre_traces_path = output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_pre_f1120_traces_bupar.csv"
    post_features_path = output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_post_f1120_patient_features_bupar.csv"
    pre_features_path = output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_pre_f1120_patient_features_bupar.csv"
    
    # Extract features from post-F1120 outputs
    post_features = set()
    
    if post_traces_path.exists():
        print(f"Loading post-F1120 traces from: {post_traces_path}")
        post_traces = pd.read_csv(post_traces_path)
        post_features.update(extract_features_from_traces(post_traces))
        print(f"  Found {len(post_features)} features in post-F1120 traces")
    
    if post_features_path.exists():
        print(f"Loading post-F1120 patient features from: {post_features_path}")
        post_patient_features = pd.read_csv(post_features_path)
        post_features.update(extract_features_from_patient_features(post_patient_features))
        print(f"  Total unique post-F1120 features: {len(post_features)}")
    
    # Extract features from pre-F1120 outputs (for comparison)
    pre_features = set()
    
    if pre_traces_path.exists():
        print(f"Loading pre-F1120 traces from: {pre_traces_path}")
        pre_traces = pd.read_csv(pre_traces_path)
        pre_features.update(extract_features_from_traces(pre_traces))
        print(f"  Found {len(pre_features)} features in pre-F1120 traces")
    
    if pre_features_path.exists():
        print(f"Loading pre-F1120 patient features from: {pre_features_path}")
        pre_patient_features = pd.read_csv(pre_features_path)
        pre_features.update(extract_features_from_patient_features(pre_patient_features))
        print(f"  Total unique pre-F1120 features: {len(pre_features)}")
    
    # Identify post-target leakage features
    # A feature is post-target leakage if it appears in post-F1120 but not (or rarely) in pre-F1120
    # For now, we'll mark features that appear in post-F1120 as potential leakage
    # A more sophisticated approach would compare frequencies
    
    all_features = post_features | pre_features
    post_target_leakage_features = post_features - pre_features  # Features only in post-F1120
    
    print(f"\nPost-target leakage analysis:")
    print(f"  Total unique features: {len(all_features)}")
    print(f"  Features only in post-F1120 (likely leakage): {len(post_target_leakage_features)}")
    print(f"  Features in both pre and post: {len(post_features & pre_features)}")
    print(f"  Features only in pre-F1120: {len(pre_features - post_features)}")
    
    # Simple binary classification
    results = []
    for feature in sorted(all_features):
        is_leakage = 1 if feature in post_target_leakage_features else 0
        results.append({
            'feature': feature,
            'is_post_target_leakage': is_leakage,
            'post_target_ratio': 1.0 if is_leakage else 0.0,
            'pre_count': 0,
            'post_count': 0,
            'total_count': 0
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Create BupaR post-target analysis CSV from BupaR outputs"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)"
    )
    parser.add_argument(
        "--post-target-threshold",
        type=float,
        default=0.8,
        help="Threshold for post-target ratio to flag as leakage (0.0-1.0, default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=5,
        help="Minimum number of events required for analysis (default: 5)"
    )
    parser.add_argument(
        "--use-bupar-outputs",
        action="store_true",
        help="Use BupaR outputs instead of event-level data (less accurate)"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = PROJECT_ROOT
    
    print(f"\n{'='*80}")
    print(f"Creating BupaR Post-Target Analysis: {args.cohort} / {args.age_band}")
    print(f"{'='*80}")
    
    # Analyze post-target leakage
    results_df = analyze_post_target_leakage(
        cohort=args.cohort,
        age_band=args.age_band,
        project_root=project_root,
        use_event_data=not args.use_bupar_outputs,
        post_target_threshold=args.post_target_threshold
    )
    
    if results_df.empty:
        print(f"\n[ERROR] No results generated. Check that event data or BupaR outputs are available.")
        sys.exit(1)
    
    # Save results
    age_band_fname = age_band_to_fname(args.age_band)
    output_dir = project_root / "3b_feature_importance_eda" / "outputs" / args.cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    # Check if file exists in wrong location (features/ subdirectory) and move it
    wrong_location = output_dir / "features" / f"{args.cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    if wrong_location.exists() and not output_path.exists():
        print(f"\n[INFO] Found file in wrong location: {wrong_location}")
        print(f"       Moving to correct location: {output_path}")
        wrong_location.rename(output_path)
    elif wrong_location.exists() and output_path.exists():
        # Both exist - remove the one in wrong location
        print(f"\n[INFO] File exists in both locations. Removing wrong location: {wrong_location}")
        wrong_location.unlink()
    
    results_df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Saved post-target analysis to: {output_path}")
    print(f"   Total features analyzed: {len(results_df)}")
    print(f"   Post-target leakage features: {results_df['is_post_target_leakage'].sum()}")
    
    # Show statistics
    # Cohort-specific target terminology for output
    target_name = "F1120" if args.cohort == "opioid_ed" else "ED visit (HCG)"
    ratio_col = 'post_target_ratio' if 'post_target_ratio' in results_df.columns else 'post_f1120_ratio'
    pre_ratio_col = 'pre_target_ratio' if 'pre_target_ratio' in results_df.columns else 'pre_f1120_ratio'
    
    if ratio_col in results_df.columns:
        leakage_features = results_df[results_df['is_post_target_leakage'] == 1]
        predictive_features = results_df[results_df.get('is_pre_target_predictive', pd.Series([0]*len(results_df))) == 1]
        
        if len(leakage_features) > 0:
            print(f"\n   [WARN] Post-target leakage features (post-{target_name} ratio >= {args.post_target_threshold:.0%}):")
            print(f"   {'='*80}")
            for idx, row in leakage_features.head(20).iterrows():
                post_ratio_pct = row[ratio_col] * 100
                pre_ratio_pct = row.get(pre_ratio_col, 0) * 100
                # Handle NaN/None values safely
                pre_val = row.get('pre_count', 0)
                post_val = row.get('post_count', 0)
                total_val = row.get('total_count', 0)
                pre_count = int(pre_val) if pd.notna(pre_val) else 0
                post_count = int(post_val) if pd.notna(post_val) else 0
                total_count = int(total_val) if pd.notna(total_val) else 0
                print(f"     {row['feature']:40s} | Post: {post_ratio_pct:5.1f}% | Pre: {pre_ratio_pct:5.1f}% | Pre: {pre_count:4d} | Post: {post_count:4d} | Total: {total_count:4d}")
            if len(leakage_features) > 20:
                print(f"     ... and {len(leakage_features) - 20} more")
            print(f"   {'='*80}")
        
        if len(predictive_features) > 0:
            print(f"\n   [OK] Pre-target predictive features (pre-{target_name} ratio >= 80%):")
            print(f"   {'='*80}")
            for idx, row in predictive_features.head(20).iterrows():
                post_ratio_pct = row.get(ratio_col, 0) * 100
                pre_ratio_pct = row.get(pre_ratio_col, 0) * 100
                pre_val = row.get('pre_count', 0)
                post_val = row.get('post_count', 0)
                total_val = row.get('total_count', 0)
                pre_count = int(pre_val) if pd.notna(pre_val) else 0
                post_count = int(post_val) if pd.notna(post_val) else 0
                total_count = int(total_val) if pd.notna(total_val) else 0
                print(f"     {row['feature']:40s} | Pre: {pre_ratio_pct:5.1f}% | Post: {post_ratio_pct:5.1f}% | Pre: {pre_count:4d} | Post: {post_count:4d} | Total: {total_count:4d}")
            if len(predictive_features) > 20:
                print(f"     ... and {len(predictive_features) - 20} more")
            print(f"   {'='*80}")
        
        # Show summary statistics
        print(f"\n   Summary Statistics:")
        if pre_ratio_col in results_df.columns:
            print(f"     Mean pre-{target_name} ratio: {results_df[pre_ratio_col].mean():.2%}")
            print(f"     Median pre-{target_name} ratio: {results_df[pre_ratio_col].median():.2%}")
        print(f"     Mean post-{target_name} ratio: {results_df[ratio_col].mean():.2%}")
        print(f"     Median post-{target_name} ratio: {results_df[ratio_col].median():.2%}")
        print(f"     Features with post-ratio > 50%: {(results_df[ratio_col] > 0.5).sum()}")
        print(f"     Features with post-ratio > 80%: {(results_df[ratio_col] > 0.8).sum()}")
        print(f"     Features with post-ratio > 90%: {(results_df[ratio_col] > 0.9).sum()}")
        if pre_ratio_col in results_df.columns:
            print(f"     Features with pre-ratio > 80%: {(results_df[pre_ratio_col] > 0.8).sum()}")
    else:
        # Fallback display for BupaR-based analysis
        leakage_features = results_df[results_df['is_post_target_leakage'] == 1]
        if len(leakage_features) > 0:
            print(f"\n   Sample post-target leakage features (first 10):")
            for feature in leakage_features['feature'].head(10):
                print(f"     - {feature}")
            if len(leakage_features) > 10:
                print(f"     ... and {len(leakage_features) - 10} more")


if __name__ == "__main__":
    main()
