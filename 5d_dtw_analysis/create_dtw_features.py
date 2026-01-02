#!/usr/bin/env python3
"""
Create patient-level DTW trajectory features.

This script extracts DTW-based trajectory features from patient sequences:
- Builds patient trajectories from model_data (already filtered by aggregated feature importances in Step 4a)
- Computes DTW distances to prototype trajectories
- Creates patient-level features for model training

Output:
- Saves to: outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv
- This intermediate file is then merged with other features by add_dtw_features_to_model_data.py

NOTE: This script no longer requires FP-Growth itemsets. The model_data is already filtered
by aggregated feature importances from Step 3, so all events in model_data are used for
trajectory construction.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import subprocess
import shutil
from typing import Optional, Dict, List, Set, Tuple
import duckdb

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NOTE: load_fpgrowth_itemsets function removed - no longer needed
# model_data is already filtered by aggregated feature importances (Step 4a),
# so we don't need to filter again using FP-Growth itemsets.


def extract_patient_trajectories(
    model_data_path: Path,
    allowed_codes: Optional[Set[str]],
    cohort_name: str,
    item_type: str = "combined",
    target_filter: Optional[int] = None,
    cutoff_dates: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    """
    Extract patient trajectories from model_data.
    
    Parameters:
    -----------
    model_data_path : Path
        Path to model_data parquet file (already filtered by feature importances in Step 4a)
    allowed_codes : Optional[Set[str]]
        Set of allowed activity codes (None = use all codes in model_data)
        NOTE: model_data is already filtered, so this is typically None
    cohort_name : str
        Cohort name (e.g., opioid_ed)
    item_type : str
        Type of items to extract (drug, icd, cpt, combined)
    target_filter : Optional[int]
        If provided, filter by target value (0=control, 1=target, None=both)
    cutoff_dates : Optional[Dict[str, str]]
        Dictionary mapping mi_person_key to cutoff date (events before this date)
        Format: {'patient_id': 'YYYY-MM-DD'}
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping mi_person_key to list of activity codes
    """
    if not model_data_path.exists():
        logger.warning(f"Model data file not found: {model_data_path}")
        return {}
    
    con = duckdb.connect()
    
    # Build query to extract activities - need to include target column
    path_str = str(model_data_path)
    
    # Build target filter clause
    target_clause = ""
    if target_filter is not None:
        target_clause = f"AND target = {target_filter}"
    
    # Register cutoff dates table if provided
    # For target patients: use cutoff date (events before target event = no leakage)
    # For control patients: cutoff_date is NULL, so no filtering (use all events, matching BupaR)
    cutoff_join = ""
    cutoff_where = ""
    if cutoff_dates:
        # Create DataFrame from cutoff dates dict (includes NULL for controls)
        cutoff_df = pd.DataFrame([
            {'mi_person_key': str(k), 'cutoff_date': pd.to_datetime(v) if v is not None else None} 
            for k, v in cutoff_dates.items()
        ])
        # Only apply cutoff for patients with a cutoff date (target patients)
        # Controls have NULL cutoff_date, so they get all events (matching BupaR logic)
        if not cutoff_df.empty:
            con.register('cutoff_dates', cutoff_df)
            cutoff_join = """
            LEFT JOIN cutoff_dates cd ON CAST(e.mi_person_key AS VARCHAR) = CAST(cd.mi_person_key AS VARCHAR)
            """
            cutoff_where = """
            AND (cd.cutoff_date IS NULL OR e.event_date < CAST(cd.cutoff_date AS TIMESTAMP))
            """
    
    if item_type == "drug" or item_type == "combined":
        if cutoff_dates and cutoff_join:
            drug_query = f"""
            SELECT e.mi_person_key, e.event_date, 
                   'DRUG:' || e.drug_name as activity
            FROM read_parquet('{path_str}') e
            {cutoff_join}
            WHERE e.drug_name IS NOT NULL AND e.drug_name != '' {cutoff_where} {target_clause}
            """
        else:
            drug_query = f"""
            SELECT mi_person_key, event_date, 
                   'DRUG:' || drug_name as activity
            FROM read_parquet('{path_str}')
            WHERE drug_name IS NOT NULL AND drug_name != '' {target_clause}
            """
    else:
        drug_query = f"SELECT mi_person_key, event_date, NULL as activity FROM read_parquet('{path_str}') WHERE 1=0"
    
    if item_type == "icd" or item_type == "combined":
        if cutoff_dates and cutoff_join:
            icd_query = f"""
            WITH all_icds AS (
                SELECT e.mi_person_key, e.event_date, e.primary_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE 1=1 {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.two_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE 1=1 {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.three_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE 1=1 {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.four_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE 1=1 {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.five_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE 1=1 {cutoff_where} {target_clause}
            )
            SELECT mi_person_key, event_date, 
                   'ICD:' || icd as activity
            FROM all_icds
            WHERE icd IS NOT NULL AND icd != ''
            """
        else:
            icd_query = f"""
            WITH all_icds AS (
                SELECT mi_person_key, event_date, primary_icd_diagnosis_code as icd FROM read_parquet('{path_str}')
                UNION ALL
                SELECT mi_person_key, event_date, two_icd_diagnosis_code as icd FROM read_parquet('{path_str}')
                UNION ALL
                SELECT mi_person_key, event_date, three_icd_diagnosis_code as icd FROM read_parquet('{path_str}')
                UNION ALL
                SELECT mi_person_key, event_date, four_icd_diagnosis_code as icd FROM read_parquet('{path_str}')
                UNION ALL
                SELECT mi_person_key, event_date, five_icd_diagnosis_code as icd FROM read_parquet('{path_str}')
            )
            SELECT mi_person_key, event_date, 
                   'ICD:' || icd as activity
            FROM all_icds
            WHERE icd IS NOT NULL AND icd != '' {target_clause}
            """
    else:
        icd_query = f"SELECT mi_person_key, event_date, NULL as activity FROM read_parquet('{path_str}') WHERE 1=0"
    
    if item_type == "cpt" or item_type == "combined":
        if cutoff_dates and cutoff_join:
            cpt_query = f"""
            SELECT e.mi_person_key, e.event_date,
                   'CPT:' || e.procedure_code as activity
            FROM read_parquet('{path_str}') e
            {cutoff_join}
            WHERE e.procedure_code IS NOT NULL AND e.procedure_code != '' {cutoff_where} {target_clause}
            """
        else:
            cpt_query = f"""
            SELECT mi_person_key, event_date,
                   'CPT:' || procedure_code as activity
            FROM read_parquet('{path_str}')
            WHERE procedure_code IS NOT NULL AND procedure_code != '' {target_clause}
            """
    else:
        cpt_query = f"SELECT mi_person_key, event_date, NULL as activity FROM read_parquet('{path_str}') WHERE 1=0"
    
    query = f"""
    WITH drug_events AS ({drug_query}),
         icd_events AS ({icd_query}),
         cpt_events AS ({cpt_query}),
         all_events AS (
             SELECT * FROM drug_events WHERE activity IS NOT NULL
             UNION ALL
             SELECT * FROM icd_events WHERE activity IS NOT NULL
             UNION ALL
             SELECT * FROM cpt_events WHERE activity IS NOT NULL
         )
    SELECT mi_person_key, event_date, activity
    FROM all_events
    ORDER BY mi_person_key, event_date
    """
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        logger.warning("No trajectory data extracted")
        return {}
    
    # No filtering needed - model_data is already filtered by feature importances
    # (allowed_codes is None, meaning use all events in model_data)
    
    # Exclude F1120 from trajectories (for final model)
    df = df[~df['activity'].str.contains('F1120', case=False, na=False)]
    
    # Group by patient to create trajectories
    trajectories = {}
    for patient_id in df['mi_person_key'].unique():
        patient_data = df[df['mi_person_key'] == patient_id].sort_values('event_date')
        trajectory = patient_data['activity'].tolist()
        if trajectory:
            trajectories[patient_id] = trajectory
    
    logger.info(f"Extracted trajectories for {len(trajectories)} patients ({item_type})")
    
    return trajectories


def compute_dtw_distances_to_prototypes(
    patient_trajectories: Dict[str, List[str]],
    n_prototypes: int = 5
) -> pd.DataFrame:
    """
    Compute DTW distances from each patient to prototype trajectories.
    
    Prototypes are selected as median-length trajectories from clusters.
    """
    if not DTW_AVAILABLE:
        raise ImportError("dtaidistance package not available. Install with: pip install dtaidistance")
    
    if not patient_trajectories:
        logger.warning("No patient trajectories provided")
        return pd.DataFrame(columns=['mi_person_key'])
    
    logger.info(f"Computing DTW distances for {len(patient_trajectories)} patients")
    
    # Encode all trajectories
    encoded_trajectories = {}
    all_items = set()
    for traj in patient_trajectories.values():
        all_items.update(traj)
    
    # Create global encoding map
    unique_items = sorted(all_items)
    global_encoding = {item: idx for idx, item in enumerate(unique_items)}
    
    for pid, traj in patient_trajectories.items():
        encoded = [global_encoding[item] for item in traj]
        encoded_trajectories[pid] = encoded
    
    # Select prototype trajectories (median-length trajectories)
    trajectory_lengths = [(pid, len(traj)) for pid, traj in patient_trajectories.items()]
    trajectory_lengths.sort(key=lambda x: x[1])
    
    # Select prototypes evenly spaced by length
    n_patients = len(trajectory_lengths)
    if n_prototypes > 1:
        prototype_indices = [
            trajectory_lengths[int(i * (n_patients - 1) / (n_prototypes - 1))][0]
            for i in range(n_prototypes)
        ]
    else:
        prototype_indices = [trajectory_lengths[n_patients // 2][0]]
    
    prototype_trajectories = {
        pid: encoded_trajectories[pid]
        for pid in prototype_indices
    }
    
    logger.info(f"Selected {len(prototype_trajectories)} prototype trajectories")
    
    # Compute DTW distances
    features_list = []
    
    for pid, encoded_traj in encoded_trajectories.items():
        if not encoded_traj:
            continue
        
        feature_row = {'mi_person_key': pid}
        
        # Compute distance to each prototype
        for proto_idx, proto_pid in enumerate(prototype_indices):
            proto_traj = prototype_trajectories[proto_pid]
            
            if proto_traj:
                try:
                    distance = dtw.distance(encoded_traj, proto_traj)
                    feature_row[f'dtw_distance_to_prototype_{proto_idx}'] = distance
                except Exception as e:
                    logger.warning(f"Error computing DTW distance for patient {pid} to prototype {proto_idx}: {e}")
                    feature_row[f'dtw_distance_to_prototype_{proto_idx}'] = np.inf
            else:
                feature_row[f'dtw_distance_to_prototype_{proto_idx}'] = np.inf
        
        # Compute statistics
        distances = [v for k, v in feature_row.items() if k.startswith('dtw_distance_to_prototype_')]
        if distances:
            feature_row['dtw_min_distance'] = min(distances)
            feature_row['dtw_max_distance'] = max(distances)
            feature_row['dtw_mean_distance'] = np.mean(distances)
            feature_row['dtw_std_distance'] = np.std(distances)
        
        # Trajectory characteristics
        original_traj = patient_trajectories[pid]
        feature_row['trajectory_length'] = len(original_traj)
        feature_row['trajectory_diversity'] = len(set(original_traj))
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"Created {len(features_df.columns) - 1} DTW features for {len(features_df)} patients")
    
    return features_df


def create_all_dtw_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    split_type: str = "target",
    event_year: str = "train",
    n_prototypes: int = 5,
    item_types: List[str] = None
) -> pd.DataFrame:
    """
    Create all DTW features for specified item types.
    
    Parameters:
    -----------
    project_root : Path
        Project root directory
    cohort_name : str
        Cohort name (e.g., opioid_ed)
    age_band : str
        Age band (e.g., 0-12)
    split_type : str
        Split type (target or combined)
    event_year : str
        Event year label (train, 2019, etc.)
    n_prototypes : int
        Number of prototype trajectories to use
    item_types : List[str]
        List of item types to process (drug, icd, cpt, combined)
        
    Returns:
    --------
    pd.DataFrame
        Combined patient-level DTW features
    """
    if item_types is None:
        item_types = ["combined"]  # Default to combined
    
    age_band_fname = age_band.replace("-", "_")

    # Model data path (prefer protocol-filtered version if available).
    # Use canonical 4a_model_data for all cohorts.
    # NOTE: model_data is already filtered by aggregated feature importances (Step 4a),
    # so we don't need to filter again using FP-Growth itemsets.
    model_data_dir = (
        project_root
        / "4a_model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    model_data_filtered = model_data_dir / "model_events_no_protocols.parquet"
    model_data_path = (
        model_data_filtered
        if model_data_filtered.exists()
        else model_data_dir / "model_events.parquet"
    )
    
    if not model_data_path.exists():
        logger.error(f"Model data not found: {model_data_path}")
        return pd.DataFrame()
    
    # No need to filter by itemsets - model_data is already filtered by feature importances
    # Use all events in model_data for trajectory construction
    allowed_codes = None  # None means use all codes in model_data
    logger.info("Using all events from model_data (already filtered by feature importances in Step 4a)")
    
    # Get cutoff dates using the same logic as BupaR analysis
    # For target patients: use first_opioid_ed_date or first_ed_non_opioid_date (cohort-specific)
    #   - Events before target date = no leakage (excludes target event itself, matching BupaR)
    # For control patients: NULL means no cutoff (use all events, matching BupaR logic)
    # NOTE: Anything after target code date is leakage - use target event date field directly
    con = duckdb.connect()
    
    # Determine which target date field to use based on cohort
    # opioid_ed uses first_opioid_ed_date, non_opioid_ed uses first_ed_non_opioid_date
    if "opioid" in cohort_name.lower():
        target_date_field = "first_opioid_ed_date"
    else:
        target_date_field = "first_ed_non_opioid_date"
    
    cutoff_dates_df = con.execute(f"""
        WITH patient_target_dates AS (
            SELECT DISTINCT
                mi_person_key,
                target,
                CAST({target_date_field} AS DATE) as target_event_date
            FROM read_parquet('{model_data_path}')
            GROUP BY mi_person_key, target, {target_date_field}
        )
        SELECT 
            mi_person_key,
            -- For target patients: use target event date (events before target = no leakage, matching BupaR)
            -- For control patients: NULL means no cutoff (use all events, matching BupaR logic)
            CASE 
                WHEN target = 1 AND target_event_date IS NOT NULL 
                THEN target_event_date
                ELSE NULL  -- Controls: no cutoff, use all events (same as BupaR)
            END as cutoff_date
        FROM patient_target_dates
    """).df()
    
    logger.info(f"Using {target_date_field} for target patients (events before target), NULL for controls (all events, matching BupaR)")
    
    # Get base patient list (both target and control)
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key, target
        FROM read_parquet('{model_data_path}')
        WHERE target IN (0, 1)
        """
    ).df()
    con.close()
    
    if base_df.empty:
        logger.error("No patients found in model_data")
        return pd.DataFrame()
    
    if cutoff_dates_df.empty:
        logger.error("No cutoff dates found")
        return pd.DataFrame()
    
    logger.info(f"Creating DTW features for {len(base_df)} patients ({len(base_df[base_df['target']==1])} target, {len(base_df[base_df['target']==0])} control)")
    
    # Collect all trajectories for prototype creation
    all_trajectories_combined = {}
    
    # Process each item type
    for item_type in item_types:
        logger.info(f"\nProcessing {item_type} trajectories...")
        
        # Extract trajectories using cutoff dates (matching BupaR logic)
        # Convert cutoff_dates_df to dict format (includes NULL for controls, matching BupaR)
        cutoff_dates_dict = cutoff_dates_df.set_index('mi_person_key')['cutoff_date'].to_dict()
        # Convert dates to strings if needed, preserve None for controls
        cutoff_dates_dict = {str(k): str(v) if pd.notna(v) else None for k, v in cutoff_dates_dict.items()}
        
        patient_trajectories = extract_patient_trajectories(
            model_data_path=model_data_path,
            allowed_codes=allowed_codes,
            cohort_name=cohort_name,
            item_type=item_type,
            target_filter=None,  # Include both target and control
            cutoff_dates=cutoff_dates_dict
        )
        
        if not patient_trajectories:
            logger.warning(f"No patient trajectories for {item_type}, skipping")
            continue
        
        # Add item type prefix to avoid collisions when combining
        prefixed_trajectories = {
            f"{item_type}_{pid}": traj 
            for pid, traj in patient_trajectories.items()
        }
        all_trajectories_combined.update(prefixed_trajectories)
    
    if not all_trajectories_combined:
        logger.warning("No trajectories extracted for any item type")
        return pd.DataFrame()
    
    logger.info(f"Total trajectories for prototype creation: {len(all_trajectories_combined)}")
    
    # Create prototypes from combined trajectories (target + control)
    logger.info(f"Creating {n_prototypes} prototypes from combined trajectories...")
    dtw_features_combined = compute_dtw_distances_to_prototypes(
        patient_trajectories=all_trajectories_combined,
        n_prototypes=n_prototypes
    )
    
    # Remove item type prefix from patient IDs and merge back to base_df
    if not dtw_features_combined.empty:
        dtw_features_combined['mi_person_key'] = dtw_features_combined['mi_person_key'].str.replace(
            r'^(drug|icd|cpt|combined)_', '', regex=True
        )
        
        # Merge with base_df to ensure all patients are included
        combined_features = base_df.merge(
            dtw_features_combined,
            on='mi_person_key',
            how='left'
        )
    else:
        combined_features = base_df.copy()
    
    # Fill NaN values with appropriate defaults
    for col in combined_features.columns:
        if col not in ['mi_person_key', 'target']:
            if 'distance' in col.lower():
                combined_features[col] = combined_features[col].fillna(np.inf)
            elif 'length' in col.lower() or 'diversity' in col.lower():
                combined_features[col] = combined_features[col].fillna(0)
    
    logger.info(f"\nCreated {len(combined_features.columns) - 2} DTW features for {len(combined_features)} patients")
    
    return combined_features


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create DTW features from patient trajectories")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--split_type", default="target", help="Split type (target or combined)")
    parser.add_argument("--event_year", default="train", help="Event year label (train, 2019, etc.)")
    parser.add_argument("--n_prototypes", type=int, default=5, help="Number of prototype trajectories")
    parser.add_argument("--item_types", nargs="+", default=["combined"], 
                       help="Item types to process (drug, icd, cpt, combined)")
    parser.add_argument("--output", help="Output CSV path (optional)")
    
    args = parser.parse_args()
    
    if not DTW_AVAILABLE:
        logger.error("dtaidistance package not available. Install with: pip install dtaidistance")
        return
    
    project_root = PROJECT_ROOT
    
    # Create DTW features
    dtw_features = create_all_dtw_features(
        project_root=project_root,
        cohort_name=args.cohort,
        age_band=args.age_band,
        split_type=args.split_type,
        event_year=args.event_year,
        n_prototypes=args.n_prototypes,
        item_types=args.item_types
    )
    
    if dtw_features.empty:
        logger.error("No features created. Check inputs and logs.")
        return
    
    # Set output path - intermediate file for DTW features only
    if not args.output:
        age_band_fname = args.age_band.replace("-", "_")
        feature_eng_dir = (
            project_root
            / "5d_dtw_analysis"
            / "outputs"
            / "feature_engineering"
        )
        feature_eng_dir.mkdir(parents=True, exist_ok=True)
        args.output = feature_eng_dir / f"dtw_features_{args.cohort}_{age_band_fname}.csv"
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtw_features.to_csv(output_path, index=False)
    
    print(f"\nCreated {len(dtw_features.columns) - 1} DTW features for {len(dtw_features)} patients")
    print(f"Output format: Ready for merging with other features (uses mi_person_key)")
    print(f"Saved to: {output_path}")
    
    # Upload to S3 gold location (intermediate file)
    age_band_fname = args.age_band.replace("-", "_")
    s3_path = f"s3://pgxdatalake/gold/feature_engineering/6_dtw/{args.cohort}/{args.age_band}/dtw_features_{args.cohort}_{age_band_fname}.csv"
    
    # Check for AWS CLI
    aws_cli = shutil.which("aws")
    if aws_cli:
        try:
            print(f"\nUploading to S3: {s3_path}")
            result = subprocess.run(
                [aws_cli, "s3", "cp", str(output_path), s3_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"S3 upload successful: {s3_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"S3 upload failed: {e.stderr}")
            print(f"Warning: Could not upload to S3: {e.stderr}")
    else:
        logger.info("AWS CLI not found, skipping S3 upload")
        print("Note: AWS CLI not found, skipping S3 upload")
    
    print(f"\nFeature columns ({len(dtw_features.columns)} total):")
    for col in dtw_features.columns[:20]:  # Show first 20
        print(f"  - {col}")
    if len(dtw_features.columns) > 20:
        print(f"  ... and {len(dtw_features.columns) - 20} more")


if __name__ == "__main__":
    main()

