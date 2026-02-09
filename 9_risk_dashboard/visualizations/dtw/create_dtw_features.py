#!/usr/bin/env python3
"""
Create patient-level DTW trajectory features.

This script extracts DTW-based trajectory features from patient sequences:
- Builds patient trajectories from model_data, restricted to **SHAP/FFA important codes**
  when available (same as BupaR/FP-Growth). Uses get_shap_ffa_allowed_codes_combined();
  if that fails or returns empty, falls back to all events in model_data.
- Computes DTW distances to prototype trajectories
- Creates patient-level features for model training

Output:
- Saves to: outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv
- This intermediate file is then merged with other features by add_dtw_features_to_model_data.py
- Adds admin_icd_event_count (from 1b_apcd_event_filter/administrative_codes_lookup.json) for
  Routine vs No Routine (appointments) comparison in the dashboard.
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
# Repo root (pgx-analysis) for SHAP/FFA paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NOTE: load_fpgrowth_itemsets function removed - no longer needed
# model_data is already filtered by aggregated feature importances (Step 4a),
# so we don't need to filter again using FP-Growth itemsets.

# ICD diagnosis columns in model_events (must match 4_model_data)
ICD_DIAGNOSIS_COLUMNS = [
    "primary_icd_diagnosis_code",
    "two_icd_diagnosis_code",
    "three_icd_diagnosis_code",
    "four_icd_diagnosis_code",
    "five_icd_diagnosis_code",
    "six_icd_diagnosis_code",
    "seven_icd_diagnosis_code",
    "eight_icd_diagnosis_code",
    "nine_icd_diagnosis_code",
    "ten_icd_diagnosis_code",
]


def _load_administrative_icd_codes(project_root: Path) -> Set[str]:
    """Load administrative ICD codes from 1b_apcd_event_filter/administrative_codes_lookup.json."""
    path = project_root / "1b_apcd_event_filter" / "administrative_codes_lookup.json"
    if not path.exists():
        logger.warning("Administrative codes lookup not found at %s; routine vs no routine will use trajectory intensity", path)
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("administrative_codes", {}).get("icd", [])
        return set(str(c) for c in codes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load administrative ICD codes: %s", exc)
        return set()


def _compute_admin_icd_event_count(
    model_data_path: Path,
    project_root: Path,
) -> pd.DataFrame:
    """
    Compute per-patient count of events that have at least one administrative ICD code
    (routine appointment / admin visit). Used for Routine vs No Routine comparison.
    """
    admin_icd = _load_administrative_icd_codes(project_root)
    if not admin_icd:
        return pd.DataFrame(columns=["mi_person_key", "admin_icd_event_count"])

    con = duckdb.connect()
    # Build condition: row has admin ICD if any of the ICD columns is in the admin set
    admin_list = ", ".join(f"'{c}'" for c in sorted(admin_icd))
    icd_conditions = " OR ".join(f"{col} IN ({admin_list})" for col in ICD_DIAGNOSIS_COLUMNS)
    query = f"""
        WITH events_with_admin_icd AS (
            SELECT mi_person_key
            FROM read_parquet('{str(model_data_path).replace(chr(92), "/")}')
            WHERE {icd_conditions}
        )
        SELECT mi_person_key, COUNT(*)::INTEGER as admin_icd_event_count
        FROM events_with_admin_icd
        GROUP BY mi_person_key
    """
    try:
        df = con.execute(query).df()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute admin ICD event count: %s", exc)
        con.close()
        return pd.DataFrame(columns=["mi_person_key", "admin_icd_event_count"])
    con.close()
    return df


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

    # Filter to allowed codes (e.g. SHAP/FFA important) when provided
    if allowed_codes is not None:
        df["_code"] = df["activity"].str.split(":", n=1).str.get(1)
        df = df[df["_code"].isin(allowed_codes)].drop(columns=["_code"])

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


def extract_trajectories_with_time_windows(
    model_data_path: Path,
    allowed_codes: Optional[Set[str]],
    cohort_name: str,
    item_type: str = "combined",
    target_filter: Optional[int] = None,
    cutoff_dates: Optional[Dict[str, str]] = None,
    research_mode: bool = False
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Extract patient trajectories WITH time windows for research/analysis.
    
    This function captures ALL trajectories with timestamps and time intervals
    to help identify clinical vs protocol sequences.
    
    Parameters:
    -----------
    model_data_path : Path
        Path to model_data parquet file
    allowed_codes : Optional[Set[str]]
        Set of allowed activity codes (None = use all codes)
    cohort_name : str
        Cohort name (e.g., opioid_ed)
    item_type : str
        Type of items to extract (drug, icd, cpt, combined)
    target_filter : Optional[int]
        If provided, filter by target value (0=control, 1=target, None=both)
    cutoff_dates : Optional[Dict[str, str]]
        Dictionary mapping mi_person_key to cutoff date (events before this date)
        If research_mode=True, cutoff_dates are ignored (capture ALL events)
    research_mode : bool
        If True, ignore cutoff dates and capture ALL trajectories for research
    
    Returns:
    --------
    Tuple[Dict[str, List[str]], pd.DataFrame]
        - Dictionary mapping mi_person_key to list of activity codes (trajectories)
        - DataFrame with detailed trajectory data: mi_person_key, event_date, activity, 
          days_since_previous, sequence_position, trajectory_length
    """
    if not model_data_path.exists():
        logger.warning(f"Model data file not found: {model_data_path}")
        return {}, pd.DataFrame()
    
    con = duckdb.connect()
    path_str = str(model_data_path)
    
    # Build target filter clause
    target_clause = ""
    if target_filter is not None:
        target_clause = f"AND target = {target_filter}"
    
    # In research mode, ignore cutoff dates to capture ALL trajectories
    cutoff_join = ""
    cutoff_where = ""
    if not research_mode and cutoff_dates:
        cutoff_df = pd.DataFrame([
            {'mi_person_key': str(k), 'cutoff_date': pd.to_datetime(v) if v is not None else None} 
            for k, v in cutoff_dates.items()
        ])
        if not cutoff_df.empty:
            con.register('cutoff_dates', cutoff_df)
            cutoff_join = """
            LEFT JOIN cutoff_dates cd ON CAST(e.mi_person_key AS VARCHAR) = CAST(cd.mi_person_key AS VARCHAR)
            """
            cutoff_where = """
            AND (cd.cutoff_date IS NULL OR e.event_date < CAST(cd.cutoff_date AS TIMESTAMP))
            """
    
    # Build queries for each item type (same as extract_patient_trajectories)
    if item_type == "drug" or item_type == "combined":
        if cutoff_join:
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
        if cutoff_join:
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
        if cutoff_join:
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
    
    # Query to get all events with time windows
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
         ),
         events_with_seq AS (
             SELECT 
                 mi_person_key,
                 event_date,
                 activity,
                 ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as sequence_position
             FROM all_events
         ),
         events_with_intervals AS (
             SELECT 
                 e1.mi_person_key,
                 e1.event_date,
                 e1.activity,
                 e1.sequence_position,
                 e2.event_date as previous_event_date,
                 CASE 
                     WHEN e2.event_date IS NULL THEN NULL
                     ELSE DATEDIFF('day', e2.event_date, e1.event_date)
                 END as days_since_previous
             FROM events_with_seq e1
             LEFT JOIN events_with_seq e2
                 ON e1.mi_person_key = e2.mi_person_key
                 AND e1.sequence_position = e2.sequence_position + 1
         ),
         trajectory_lengths AS (
             SELECT 
                 mi_person_key,
                 COUNT(*) as trajectory_length
             FROM events_with_intervals
             GROUP BY mi_person_key
         )
    SELECT 
        e.mi_person_key,
        e.event_date,
        e.activity,
        e.sequence_position,
        e.days_since_previous,
        t.trajectory_length
    FROM events_with_intervals e
    INNER JOIN trajectory_lengths t ON e.mi_person_key = t.mi_person_key
    ORDER BY e.mi_person_key, e.sequence_position
    """
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        logger.warning("No trajectory data extracted")
        return {}, pd.DataFrame()
    
    # Exclude F1120 from trajectories (for final model)
    df = df[~df['activity'].str.contains('F1120', case=False, na=False)]
    
    # Create trajectory dictionary (activity sequences)
    trajectories = {}
    for patient_id in df['mi_person_key'].unique():
        patient_data = df[df['mi_person_key'] == patient_id].sort_values('sequence_position')
        trajectory = patient_data['activity'].tolist()
        if trajectory:
            trajectories[patient_id] = trajectory
    
    logger.info(f"Extracted trajectories with time windows for {len(trajectories)} patients ({item_type})")
    
    return trajectories, df


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

    # Model data path - use model_events.parquet directly (skip DTW filtering for now)
    # Use canonical 4_model_data for all cohorts.
    # NOTE: model_data is already filtered by aggregated feature importances (Step 4a),
    # so we don't need to filter again using FP-Growth itemsets.
    model_data_dir = (
        project_root
        / "4_model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    # Skip protocol filtering - use model_events.parquet directly
    model_data_path = model_data_dir / "model_events.parquet"
    
    if not model_data_path.exists():
        logger.error(f"Model data not found: {model_data_path}")
        return pd.DataFrame()
    
    # Prefer SHAP/FFA important codes for trajectory construction (same as BupaR/FP-Growth)
    allowed_codes = None
    try:
        from py_helpers.shap_ffa_fpgrowth_utils import get_shap_ffa_allowed_codes_combined

        allowed_codes = get_shap_ffa_allowed_codes_combined(
            cohort_name, age_band, top_n=500, project_root=REPO_ROOT
        )
        if allowed_codes:
            logger.info(
                "Using SHAP/FFA important codes for DTW trajectories (%s codes)",
                len(allowed_codes),
            )
        else:
            allowed_codes = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load SHAP/FFA codes for DTW: %s", exc)
        allowed_codes = None
    if allowed_codes is None:
        logger.info("Using all events from model_data for DTW (no SHAP/FFA filter)")
    
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
    
    # Add sequence + time window features
    logger.info("\nAdding sequence + time window features...")
    # Convert cutoff_dates_df to dict format for sequence features
    cutoff_dates_dict_for_seq = cutoff_dates_df.set_index('mi_person_key')['cutoff_date'].to_dict()
    # Convert dates to strings if needed, preserve None for controls
    cutoff_dates_dict_for_seq = {str(k): str(v) if pd.notna(v) else None for k, v in cutoff_dates_dict_for_seq.items()}
    
    sequence_features = create_sequence_time_window_features(
        model_data_path=model_data_path,
        cohort_name=cohort_name,
        cutoff_dates_dict=cutoff_dates_dict_for_seq
    )
    
    if not sequence_features.empty:
        # Merge sequence features
        combined_features = combined_features.merge(
            sequence_features,
            on='mi_person_key',
            how='left'
        )
        logger.info(f"Added {len(sequence_features.columns) - 1} sequence + time window features")
    else:
        logger.warning("No sequence + time window features created")

    # Add admin ICD event count for Routine vs No Routine (appointments) comparison
    admin_icd_df = _compute_admin_icd_event_count(model_data_path, REPO_ROOT)
    if not admin_icd_df.empty:
        combined_features = combined_features.merge(
            admin_icd_df,
            on='mi_person_key',
            how='left'
        )
        combined_features["admin_icd_event_count"] = combined_features["admin_icd_event_count"].fillna(0).astype(int)
        logger.info("Added admin_icd_event_count for routine vs no routine comparison (from administrative_codes_lookup.json)")
    
    logger.info(f"\nTotal features created: {len(combined_features.columns) - 2} (including sequence + time window features)")
    
    return combined_features


def create_sequence_time_window_features(
    model_data_path: Path,
    cohort_name: str,
    cutoff_dates_dict: Dict[str, Optional[str]]
) -> pd.DataFrame:
    """
    Create sequence + time window features from patient trajectories.
    
    Features include:
    - Time window statistics (mean, median, std, min, max of intervals)
    - Sequence length features
    - Early/late time window features (first few and last few intervals)
    - Sequence pattern features (common subsequences)
    
    Parameters:
    -----------
    model_data_path : Path
        Path to model_data parquet file
    cohort_name : str
        Cohort name
    cutoff_dates_dict : Dict[str, Optional[str]]
        Dictionary mapping mi_person_key to cutoff date
        
    Returns:
    --------
    pd.DataFrame
        Patient-level features with mi_person_key and sequence/time window features
    """
    if not model_data_path.exists():
        logger.warning(f"Model data file not found: {model_data_path}")
        return pd.DataFrame()
    
    # Extract trajectories with time windows
    trajectories_dict, trajectory_df = extract_trajectories_with_time_windows(
        model_data_path=model_data_path,
        allowed_codes=None,  # Use all codes
        cohort_name=cohort_name,
        item_type="combined",
        target_filter=None,
        cutoff_dates=cutoff_dates_dict,
        research_mode=False
    )
    
    if trajectory_df.empty:
        logger.warning("No trajectory data extracted")
        return pd.DataFrame()
    
    # Calculate time window statistics per patient
    con = duckdb.connect()
    
    # Register trajectory_df as a table
    con.register('trajectory_data', trajectory_df)
    
    # Calculate features
    features_query = """
    WITH patient_stats AS (
        SELECT 
            mi_person_key,
            COUNT(*) as seq_length,
            COUNT(DISTINCT activity) as seq_diversity,
            AVG(days_since_previous) as time_window_mean,
            MEDIAN(days_since_previous) as time_window_median,
            STDDEV(days_since_previous) as time_window_std,
            MIN(days_since_previous) as time_window_min,
            MAX(days_since_previous) as time_window_max,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY days_since_previous) as time_window_q25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY days_since_previous) as time_window_q75
        FROM trajectory_data
        WHERE days_since_previous IS NOT NULL
        GROUP BY mi_person_key
    ),
    early_intervals AS (
        SELECT 
            mi_person_key,
            AVG(days_since_previous) as time_window_early_mean,
            MEDIAN(days_since_previous) as time_window_early_median
        FROM (
            SELECT 
                mi_person_key,
                days_since_previous,
                ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY sequence_position) as rn
            FROM trajectory_data
            WHERE days_since_previous IS NOT NULL
        )
        WHERE rn <= 3
        GROUP BY mi_person_key
    ),
    late_intervals AS (
        SELECT 
            mi_person_key,
            AVG(days_since_previous) as time_window_late_mean,
            MEDIAN(days_since_previous) as time_window_late_median
        FROM (
            SELECT 
                mi_person_key,
                days_since_previous,
                ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY sequence_position DESC) as rn
            FROM trajectory_data
            WHERE days_since_previous IS NOT NULL
        )
        WHERE rn <= 3
        GROUP BY mi_person_key
    ),
    sequence_patterns AS (
        SELECT 
            mi_person_key,
            -- First 3 activities as a pattern
            LIST(activity ORDER BY sequence_position LIMIT 3) as seq_pattern_start,
            -- Last 3 activities as a pattern
            LIST(activity ORDER BY sequence_position DESC LIMIT 3) as seq_pattern_end
        FROM trajectory_data
        GROUP BY mi_person_key
    )
    SELECT 
        ps.mi_person_key,
        ps.seq_length,
        ps.seq_diversity,
        ps.time_window_mean,
        ps.time_window_median,
        ps.time_window_std,
        ps.time_window_min,
        ps.time_window_max,
        ps.time_window_q25,
        ps.time_window_q75,
        COALESCE(ei.time_window_early_mean, 0) as time_window_early_mean,
        COALESCE(ei.time_window_early_median, 0) as time_window_early_median,
        COALESCE(li.time_window_late_mean, 0) as time_window_late_mean,
        COALESCE(li.time_window_late_median, 0) as time_window_late_median,
        sp.seq_pattern_start,
        sp.seq_pattern_end
    FROM patient_stats ps
    LEFT JOIN early_intervals ei ON ps.mi_person_key = ei.mi_person_key
    LEFT JOIN late_intervals li ON ps.mi_person_key = li.mi_person_key
    LEFT JOIN sequence_patterns sp ON ps.mi_person_key = sp.mi_person_key
    """
    
    try:
        features_df = con.execute(features_query).df()
    except Exception as e:
        logger.warning(f"Error calculating sequence features: {e}")
        features_df = pd.DataFrame()
    finally:
        con.close()
    
    if features_df.empty:
        return pd.DataFrame()
    
    # Convert sequence patterns to string features (for easier handling)
    if 'seq_pattern_start' in features_df.columns:
        features_df['seq_pattern_start_str'] = features_df['seq_pattern_start'].apply(
            lambda x: '_'.join(x) if isinstance(x, list) and x else ''
        )
        features_df['seq_pattern_end_str'] = features_df['seq_pattern_end'].apply(
            lambda x: '_'.join(x) if isinstance(x, list) and x else ''
        )
        # Drop list columns (keep string versions)
        features_df = features_df.drop(columns=['seq_pattern_start', 'seq_pattern_end'], errors='ignore')
    
    # Fill NaN values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'mi_person_key':
            features_df[col] = features_df[col].fillna(0)
    
    return features_df


def save_trajectory_research_outputs(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    trajectory_data: pd.DataFrame,
    trajectories: Dict[str, List[str]],
    item_type: str = "combined"
) -> None:
    """
    Save comprehensive trajectory research outputs to help identify clinical vs protocol sequences.
    
    Outputs:
    - All trajectories with timestamps and time windows
    - Common sequence patterns
    - Time window statistics
    - Sequence frequency analysis
    """
    age_band_fname = age_band.replace("-", "_")
    output_dir = (
        project_root
        / "10d_dtw_dashboard_visual"
        / "outputs"
        / "research"
        / cohort_name
        / age_band
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save detailed trajectory data with time windows
    trajectory_file = output_dir / f"trajectories_with_time_windows_{item_type}_{cohort_name}_{age_band_fname}.parquet"
    trajectory_data.to_parquet(trajectory_file, index=False)
    logger.info(f"Saved trajectory data with time windows: {trajectory_file}")
    
    # 2. Calculate and save time window statistics
    time_window_stats = trajectory_data.groupby('mi_person_key').agg({
        'days_since_previous': ['mean', 'median', 'std', 'min', 'max'],
        'trajectory_length': 'first'
    }).reset_index()
    time_window_stats.columns = ['mi_person_key', 'mean_interval_days', 'median_interval_days', 
                                 'std_interval_days', 'min_interval_days', 'max_interval_days', 'trajectory_length']
    
    stats_file = output_dir / f"time_window_stats_{item_type}_{cohort_name}_{age_band_fname}.csv"
    time_window_stats.to_csv(stats_file, index=False)
    logger.info(f"Saved time window statistics: {stats_file}")
    
    # 3. Identify common sequence patterns (2-3 event sequences)
    sequence_patterns = []
    for pid, traj in trajectories.items():
        # Extract 2-event sequences
        for i in range(len(traj) - 1):
            seq_2 = f"{traj[i]} -> {traj[i+1]}"
            sequence_patterns.append({
                'mi_person_key': pid,
                'sequence': seq_2,
                'sequence_length': 2,
                'position_in_trajectory': i
            })
        # Extract 3-event sequences
        for i in range(len(traj) - 2):
            seq_3 = f"{traj[i]} -> {traj[i+1]} -> {traj[i+2]}"
            sequence_patterns.append({
                'mi_person_key': pid,
                'sequence': seq_3,
                'sequence_length': 3,
                'position_in_trajectory': i
            })
    
    if sequence_patterns:
        patterns_df = pd.DataFrame(sequence_patterns)
        # Count frequency of each sequence pattern
        sequence_freq = patterns_df.groupby(['sequence', 'sequence_length']).size().reset_index(name='frequency')
        sequence_freq = sequence_freq.sort_values('frequency', ascending=False)
        
        patterns_file = output_dir / f"common_sequences_{item_type}_{cohort_name}_{age_band_fname}.csv"
        sequence_freq.to_csv(patterns_file, index=False)
        logger.info(f"Saved common sequence patterns: {patterns_file}")
        
        # Save top 100 most frequent sequences
        top_sequences_file = output_dir / f"top_100_sequences_{item_type}_{cohort_name}_{age_band_fname}.csv"
        sequence_freq.head(100).to_csv(top_sequences_file, index=False)
        logger.info(f"Saved top 100 sequences: {top_sequences_file}")
    
    # 4. Identify protocol-like sequences (events < 7 days apart)
    protocol_sequences = trajectory_data[
        (trajectory_data['days_since_previous'].notna()) &
        (trajectory_data['days_since_previous'] < 7)
    ].copy()
    
    if not protocol_sequences.empty:
        protocol_file = output_dir / f"protocol_like_sequences_{item_type}_{cohort_name}_{age_band_fname}.parquet"
        protocol_sequences.to_parquet(protocol_file, index=False)
        logger.info(f"Saved protocol-like sequences (< 7 days): {protocol_file}")
        
        # Protocol sequence patterns
        protocol_patterns = []
        for pid in protocol_sequences['mi_person_key'].unique():
            patient_protocol = protocol_sequences[protocol_sequences['mi_person_key'] == pid]
            for idx, row in patient_protocol.iterrows():
                if row['sequence_position'] > 1:  # Has previous event
                    prev_row = trajectory_data[
                        (trajectory_data['mi_person_key'] == pid) &
                        (trajectory_data['sequence_position'] == row['sequence_position'] - 1)
                    ]
                    if not prev_row.empty:
                        protocol_patterns.append({
                            'mi_person_key': pid,
                            'sequence': f"{prev_row.iloc[0]['activity']} -> {row['activity']}",
                            'days_apart': row['days_since_previous']
                        })
        
        if protocol_patterns:
            protocol_patterns_df = pd.DataFrame(protocol_patterns)
            protocol_patterns_freq = protocol_patterns_df.groupby('sequence').agg({
                'days_apart': ['mean', 'count']
            }).reset_index()
            protocol_patterns_freq.columns = ['sequence', 'mean_days_apart', 'frequency']
            protocol_patterns_freq = protocol_patterns_freq.sort_values('frequency', ascending=False)
            
            protocol_patterns_file = output_dir / f"protocol_sequence_patterns_{item_type}_{cohort_name}_{age_band_fname}.csv"
            protocol_patterns_freq.to_csv(protocol_patterns_file, index=False)
            logger.info(f"Saved protocol sequence patterns: {protocol_patterns_file}")
    
    logger.info(f"\nResearch outputs saved to: {output_dir}")


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
    parser.add_argument("--research_mode", action="store_true", 
                       help="Research mode: capture ALL trajectories with time windows (no cutoff dates)")
    
    args = parser.parse_args()
    
    if not DTW_AVAILABLE:
        logger.error("dtaidistance package not available. Install with: pip install dtaidistance")
        return
    
    project_root = PROJECT_ROOT
    
    # Research mode: capture ALL trajectories with time windows for analysis
    if args.research_mode:
        logger.info("=" * 80)
        logger.info("RESEARCH MODE: Capturing ALL trajectories with time windows")
        logger.info("=" * 80)
        
        model_data_dir = (
            project_root
            / "4_model_data"
            / f"cohort_name={args.cohort}"
            / f"age_band={args.age_band}"
        )
        model_data_path = (
            model_data_dir / "model_events_no_protocols.parquet"
            if (model_data_dir / "model_events_no_protocols.parquet").exists()
            else model_data_dir / "model_events.parquet"
        )
        
        if not model_data_path.exists():
            logger.error(f"Model data not found: {model_data_path}")
            return
        
        # Extract trajectories with time windows (no cutoff dates in research mode)
        for item_type in args.item_types:
            logger.info(f"\nExtracting {item_type} trajectories with time windows (research mode)...")
            trajectories, trajectory_data = extract_trajectories_with_time_windows(
                model_data_path=model_data_path,
                allowed_codes=None,
                cohort_name=args.cohort,
                item_type=item_type,
                target_filter=None,  # Include both target and control
                cutoff_dates=None,  # Ignored in research mode
                research_mode=True
            )
            
            if trajectories and not trajectory_data.empty:
                # Save comprehensive research outputs
                save_trajectory_research_outputs(
                    project_root=project_root,
                    cohort_name=args.cohort,
                    age_band=args.age_band,
                    trajectory_data=trajectory_data,
                    trajectories=trajectories,
                    item_type=item_type
                )
            else:
                logger.warning(f"No trajectories extracted for {item_type}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Research mode complete. Review outputs in 10d_dtw_dashboard_visual/outputs/research/")
        logger.info("=" * 80)
        return
    
    # Normal mode: Create DTW features
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
            / "10d_dtw_dashboard_visual"
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

