#!/usr/bin/env python3
"""
Create patient-level DTW trajectory features.

This script extracts DTW-based trajectory features from patient sequences:
- Builds patient trajectories from model_data
- Computes DTW distances to prototype trajectories
- Creates patient-level features for model training

Output:
- Saves to: outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv
- This intermediate file is then merged with other features by add_dtw_features_to_model_data.py
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
from datetime import datetime

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available. Visualizations will be skipped.")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from 1_dtw/ -> 3b_feature_importance_eda/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))

# Import OS-aware path utilities
try:
    from py_helpers.env_utils import get_data_root, is_linux
except ImportError:
    # Fallback if env_utils not available
    def get_data_root() -> Path:
        """Fallback: return project root / data"""
        return PROJECT_ROOT / "data"
    
    def is_linux() -> bool:
        """Fallback: check if Linux"""
        import os
        return os.name == "posix"

# Define get_model_data_root locally (same logic as 4a_model_data/create_model_data.py)
def get_model_data_root() -> Path:
    """Get the root directory for model data (OS-aware)."""
    data_root = get_data_root()
    if is_linux():
        # On Linux/EC2: use /mnt/nvme/4a_model_data
        return data_root / "4a_model_data"
    else:
        # On Windows/local dev: use PROJECT_ROOT/4a_model_data
        return PROJECT_ROOT / "4a_model_data"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logger.error("dtaidistance package not available. Install with: pip install dtaidistance")


def _load_administrative_codes_for_dtw(cohort_name: str) -> Optional[Dict[str, Set[str]]]:
    """
    Load administrative codes from 4b_dtw_filter for exclusion from DTW trajectories.
    
    Sources:
    1. Hardcoded administrative codes (Z codes, CPT codes from filter_protocol_events.py)
    2. Research outputs (code_analysis_protocol_vs_clinical CSV)
    3. administrative_codes_lookup.json
    
    Parameters:
    -----------
    cohort_name : str
        Cohort name (for finding research outputs)
    
    Returns:
    --------
    Optional[Dict[str, Set[str]]]
        Dictionary with keys 'icd', 'cpt', 'drug' containing sets of administrative codes
        Returns None if loading fails (will skip exclusion)
    """
    admin_codes = {
        'icd': set(),
        'cpt': set(),
        'drug': set(),
    }
    
    try:
        # 1. Load hardcoded administrative codes (from filter_protocol_events.py logic)
        # Z00: General health examinations (preventive/routine)
        z00_codes = ['Z00.00', 'Z00.01', 'Z00.110', 'Z00.111', 'Z00.12', 'Z00.121', 'Z00.129', 'Z00.3', 'Z00.70', 'Z00.8']
        # Z01: Special examinations (preventive/routine)
        z01_codes = ['Z01.00', 'Z01.01', 'Z01.10', 'Z01.30', 'Z01.31', 'Z01.41', 'Z01.411', 'Z01.419', 'Z01.42', 'Z01.70',
                     'Z01.810', 'Z01.811', 'Z01.812', 'Z01.818', 'Z01.82', 'Z01.83', 'Z01.84', 'Z01.89']
        # Z02: Administrative examinations
        z02_codes = ['Z02.0', 'Z02.1', 'Z02.2', 'Z02.5', 'Z02.83', 'Z02.89', 'Z02.9']
        # Z03: Medical observation for suspected conditions (ruled out - administrative)
        z03_codes = ['Z03.6', 'Z03.71', 'Z03.72', 'Z03.73', 'Z03.74', 'Z03.75', 'Z03.79', 'Z03.89']
        # Z04: Examination for legal/administrative purposes
        z04_codes = ['Z04.1', 'Z04.3', 'Z04.41', 'Z04.42', 'Z04.8', 'Z04.89', 'Z04.9']
        # Z08: Follow-up examination after treatment for malignant neoplasm
        z08_codes = ['Z08']
        # Z09: Follow-up examination after treatment for other conditions
        z09_codes = ['Z09']
        # Z34: Supervision of normal pregnancy (preventive/routine)
        z34_codes = ['Z34.00', 'Z34.01', 'Z34.02', 'Z34.03', 'Z34.80', 'Z34.81', 'Z34.82', 'Z34.83', 'Z34.90', 'Z34.91', 'Z34.92', 'Z34.93']
        # Z39: Encounter for maternal postpartum care and examination
        z39_codes = ['Z39.0', 'Z39.1', 'Z39.2']
        # Z51: Encounters for other aftercare and medical care
        z51_codes = ['Z51.0', 'Z51.11', 'Z51.12', 'Z51.5', 'Z51.6', 'Z51.81', 'Z51.89']
        # V72: Other medical examination (preventive/administrative)
        v72_codes = ['V72.31', 'V72.40', 'V72.41', 'V72.42', 'V72.5', 'V72.61', 'V72.7', 'V72.81', 'V72.83', 'V72.85']
        
        all_icd_codes = z00_codes + z01_codes + z02_codes + z03_codes + z04_codes + z08_codes + z09_codes + z34_codes + z39_codes + z51_codes + v72_codes
        for code in all_icd_codes:
            admin_codes['icd'].add(code)  # With dots
            admin_codes['icd'].add(code.replace('.', ''))  # Without dots
        
        # CPT codes
        # CPT 99000-99099: Administrative services
        admin_cpt_99000 = ['99000', '99001', '99024', '99050', '99051', '99053', '99058', '99070', '99078']
        # CPT 99400-99499: Preventive medicine
        admin_cpt_99400 = ['99401', '99402', '99403', '99404', '99406', '99407', '99408', '99409', '99420', '99429', '99441', '99442', '99443', '99444', '99460', '99462', '99464', '99471', '99472', '99480', '99484', '99487', '99490', '99495', '99496', '99497', '99499']
        # CPT 99381-99397: Preventive visits
        admin_cpt_99381 = ['99381', '99382', '99383', '99384', '99385', '99386', '99387', '99391', '99392', '99393', '99394', '99395', '99396', '99397']
        # CPT 99211: Level 1 office visit (minimal complexity)
        admin_cpt_level1_office = ['99211']
        # S codes: Administrative billing codes
        admin_s_codes = ['S0201', 'S0109', 'S9083', 'S0990XA', 'S0028']
        
        for code in admin_cpt_99000 + admin_cpt_99400 + admin_cpt_99381 + admin_cpt_level1_office + admin_s_codes:
            admin_codes['cpt'].add(str(code))
        
        # 2. Try to load from administrative_codes_lookup.json
        lookup_path = PROJECT_ROOT / "4b_dtw_filter" / "administrative_codes_lookup.json"
        if lookup_path.exists():
            try:
                with open(lookup_path, 'r') as f:
                    lookup_data = json.load(f)
                    admin_data = lookup_data.get('administrative_codes', {})
                    
                    # Add ICD codes
                    for icd_code in admin_data.get('icd', []):
                        icd_str = str(icd_code).strip()
                        if icd_str:
                            admin_codes['icd'].add(icd_str)
                            admin_codes['icd'].add(icd_str.replace('.', ''))
                    
                    # Add CPT codes
                    for cpt_code in admin_data.get('cpt', []):
                        cpt_str = str(cpt_code).strip()
                        if cpt_str:
                            admin_codes['cpt'].add(cpt_str)
                    
                    # Add drug codes
                    for drug_code in admin_data.get('drug', []):
                        drug_str = str(drug_code).strip()
                        if drug_str:
                            admin_codes['drug'].add(drug_str)
                    
                    logger.info(f"Loaded administrative codes from lookup file: {len(admin_codes['icd'])} ICD, {len(admin_codes['cpt'])} CPT, {len(admin_codes['drug'])} drug")
            except Exception as e:
                logger.warning(f"Could not load administrative codes from lookup file: {e}")
        
        # 3. Try to load from research outputs (code_analysis_protocol_vs_clinical CSV)
        # This would require age_band, so we skip it here (can be added if needed)
        
        if admin_codes['icd'] or admin_codes['cpt'] or admin_codes['drug']:
            logger.info(f"Loaded {len(admin_codes['icd'])} ICD, {len(admin_codes['cpt'])} CPT, {len(admin_codes['drug'])} drug administrative codes for exclusion")
            return admin_codes
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Error loading administrative codes: {e}. Will not exclude administrative codes from trajectories.")
        return None


def load_fpgrowth_itemsets(itemsets_path: Path) -> Set[str]:
    """Load FP-Growth itemsets and extract all unique codes."""
    if not itemsets_path.exists():
        logger.warning(f"Itemsets file not found: {itemsets_path}")
        return set()
    
    try:
        with open(itemsets_path, 'r') as f:
            data = json.load(f)
        
        allowed_codes = set()
        for row in data:
            for code in row.get("itemsets", []):
                allowed_codes.add(code)
        
        logger.info(f"Loaded {len(allowed_codes)} unique codes from {itemsets_path.name}")
        return allowed_codes
    except Exception as e:
        logger.error(f"Error loading itemsets from {itemsets_path}: {e}")
        return set()


def extract_patient_trajectories(
    model_data_path: Path,
    allowed_codes: Optional[Set[str]] = None,
    cohort_name: str = "",
    item_type: str = "combined",
    target_filter: Optional[int] = None,
    cutoff_dates: Optional[Dict[str, str]] = None
) -> Dict[str, List[str]]:
    """
    Extract patient trajectories from model_data.
    
    Parameters:
    -----------
    model_data_path : Path
        Path to model_data parquet file
    allowed_codes : Optional[Set[str]]
        Optional set of allowed activity codes to filter by (if None, allows all codes except F1120)
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
    Tuple[Dict[str, List[str]], Dict[str, Dict]]
        (trajectories, trajectory_metadata)
        - trajectories: Dictionary mapping mi_person_key to list of activity codes
        - trajectory_metadata: Dictionary mapping mi_person_key to temporal metadata dict
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
    # For target patients: cutoff_date is set (events before target)
    # For control patients: cutoff_date is NULL (use all events)
    cutoff_join = ""
    cutoff_where = ""
    if cutoff_dates:
        # Create DataFrame from cutoff dates dict (includes NULL for controls)
        cutoff_df = pd.DataFrame([
            {'mi_person_key': str(k), 'cutoff_date': pd.to_datetime(v) if v is not None else None} 
            for k, v in cutoff_dates.items()
        ])
        if not cutoff_df.empty:
            con.register('cutoff_dates', cutoff_df)
            cutoff_join = """
            LEFT JOIN cutoff_dates cd ON CAST(e.mi_person_key AS VARCHAR) = CAST(cd.mi_person_key AS VARCHAR)
            """
            # Only apply cutoff for patients with a cutoff date (target patients)
            # Controls have NULL cutoff_date, so they get all events
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
                WHERE e.primary_icd_diagnosis_code IS NOT NULL AND e.primary_icd_diagnosis_code != '' {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.two_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE e.two_icd_diagnosis_code IS NOT NULL AND e.two_icd_diagnosis_code != '' {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.three_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE e.three_icd_diagnosis_code IS NOT NULL AND e.three_icd_diagnosis_code != '' {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.four_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE e.four_icd_diagnosis_code IS NOT NULL AND e.four_icd_diagnosis_code != '' {cutoff_where} {target_clause}
                UNION ALL
                SELECT e.mi_person_key, e.event_date, e.five_icd_diagnosis_code as icd 
                FROM read_parquet('{path_str}') e
                {cutoff_join}
                WHERE e.five_icd_diagnosis_code IS NOT NULL AND e.five_icd_diagnosis_code != '' {cutoff_where} {target_clause}
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
        return {}, {}
    
    # Filter by allowed codes if provided (optional filtering)
    # If allowed_codes is None or empty, allow all codes except F1120
    if allowed_codes and len(allowed_codes) > 2:  # More than just F1120 variants
        df = df[df['activity'].isin(allowed_codes)]
    
    # Exclude F1120 from trajectories (for final model)
    df = df[~df['activity'].str.contains('F1120', case=False, na=False)]
    
    # Exclude administrative codes (from 4b_dtw_filter)
    administrative_codes = _load_administrative_codes_for_dtw(cohort_name)
    if administrative_codes:
        # Build exclusion patterns for administrative codes
        exclude_patterns = []
        
        # ICD codes: format as 'ICD:Z00.00', 'ICD:Z0000', etc.
        for icd_code in administrative_codes.get('icd', set()):
            icd_str = str(icd_code).strip()
            if icd_str:
                # Add both formats: with and without dots
                exclude_patterns.append(f'ICD:{icd_str}')
                exclude_patterns.append(f'ICD:{icd_str.replace(".", "")}')
                # If no dots, try adding dot version
                if '.' not in icd_str and len(icd_str) >= 5:
                    if icd_str.startswith('Z') or icd_str.startswith('V'):
                        exclude_patterns.append(f'ICD:{icd_str[:3]}.{icd_str[3:]}')
        
        # CPT codes: format as 'CPT:99024', etc.
        for cpt_code in administrative_codes.get('cpt', set()):
            cpt_str = str(cpt_code).strip()
            if cpt_str:
                exclude_patterns.append(f'CPT:{cpt_str}')
        
        # Drug codes: format as 'DRUG:DRUGNAME', etc.
        for drug_code in administrative_codes.get('drug', set()):
            drug_str = str(drug_code).strip()
            if drug_str:
                exclude_patterns.append(f'DRUG:{drug_str}')
        
        # Filter out administrative codes
        if exclude_patterns:
            exclude_mask = df['activity'].isin(exclude_patterns)
            n_excluded = exclude_mask.sum()
            if n_excluded > 0:
                logger.info(f"Excluding {n_excluded} administrative events from trajectories")
                df = df[~exclude_mask]
    
    # Group by patient to create trajectories with temporal information
    trajectories = {}
    trajectory_metadata = {}  # Store temporal metadata for each trajectory
    
    for patient_id in df['mi_person_key'].unique():
        patient_data = df[df['mi_person_key'] == patient_id].sort_values('event_date')
        trajectory = patient_data['activity'].tolist()
        
        if trajectory:
            trajectories[patient_id] = trajectory
            
            # Calculate temporal metrics
            event_dates = pd.to_datetime(patient_data['event_date'])
            first_event_date = event_dates.min()
            last_event_date = event_dates.max()
            temporal_span_days = (last_event_date - first_event_date).days if len(event_dates) > 1 else 0
            
            # Calculate intervals between consecutive events
            if len(event_dates) > 1:
                intervals = (event_dates.diff().dropna()).dt.days
                interval_mean = intervals.mean() if len(intervals) > 0 else 0
                interval_median = intervals.median() if len(intervals) > 0 else 0
                interval_std = intervals.std() if len(intervals) > 0 else 0
                interval_min = intervals.min() if len(intervals) > 0 else 0
                interval_max = intervals.max() if len(intervals) > 0 else 0
            else:
                interval_mean = interval_median = interval_std = interval_min = interval_max = 0
            
            # Temporal density: events per month
            temporal_density = len(trajectory) / (temporal_span_days / 30.0) if temporal_span_days > 0 else len(trajectory)
            
            trajectory_metadata[patient_id] = {
                'first_event_date': first_event_date,
                'last_event_date': last_event_date,
                'temporal_span_days': temporal_span_days,
                'temporal_density': temporal_density,
                'interval_mean': interval_mean,
                'interval_median': interval_median,
                'interval_std': interval_std,
                'interval_min': interval_min,
                'interval_max': interval_max,
                'n_intervals': len(intervals) if len(event_dates) > 1 else 0
            }
    
    logger.info(f"Extracted trajectories for {len(trajectories)} patients ({item_type})")
    
    return trajectories, trajectory_metadata


def encode_trajectory(trajectory: List[str]) -> Tuple[List[int], Dict[str, int]]:
    """Encode trajectory to integer sequence and return encoding map."""
    unique_items = sorted(set(trajectory))
    encoding_map = {item: idx for idx, item in enumerate(unique_items)}
    encoded = [encoding_map[item] for item in trajectory]
    return encoded, encoding_map


def compute_dtw_distances_to_prototypes(
    patient_trajectories: Dict[str, List[str]],
    trajectory_metadata: Optional[Dict[str, Dict]] = None,
    n_prototypes: int = 5
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    """
    Compute DTW distances from each patient to prototype trajectories.
    
    Prototypes are selected as median-length trajectories from clusters.
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]
        (features_df, patient_trajectories, prototype_indices)
    """
    if not DTW_AVAILABLE:
        raise ImportError("dtaidistance package not available. Install with: pip install dtaidistance")
    
    if not patient_trajectories:
        logger.warning("No patient trajectories provided")
        return pd.DataFrame(columns=['mi_person_key']), {}, []
    
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
    prototype_indices = [
        trajectory_lengths[int(i * (n_patients - 1) / (n_prototypes - 1))][0]
        for i in range(n_prototypes)
    ] if n_prototypes > 1 else [trajectory_lengths[n_patients // 2][0]]
    
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
        
        # Add temporal metrics if available
        if trajectory_metadata and pid in trajectory_metadata:
            meta = trajectory_metadata[pid]
            feature_row['trajectory_temporal_span_days'] = meta['temporal_span_days']
            feature_row['trajectory_temporal_density'] = meta['temporal_density']
            feature_row['trajectory_interval_mean_days'] = meta['interval_mean']
            feature_row['trajectory_interval_median_days'] = meta['interval_median']
            feature_row['trajectory_interval_std_days'] = meta['interval_std']
            feature_row['trajectory_interval_min_days'] = meta['interval_min']
            feature_row['trajectory_interval_max_days'] = meta['interval_max']
            feature_row['trajectory_n_intervals'] = meta['n_intervals']
        else:
            # Default values if metadata not available
            feature_row['trajectory_temporal_span_days'] = 0
            feature_row['trajectory_temporal_density'] = 0
            feature_row['trajectory_interval_mean_days'] = 0
            feature_row['trajectory_interval_median_days'] = 0
            feature_row['trajectory_interval_std_days'] = 0
            feature_row['trajectory_interval_min_days'] = 0
            feature_row['trajectory_interval_max_days'] = 0
            feature_row['trajectory_n_intervals'] = 0
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"Created {len(features_df.columns) - 1} DTW features for {len(features_df)} patients")
    
    return features_df, patient_trajectories, prototype_indices


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
    
    # Model data path - OS-aware resolution (supports /mnt/nvme on Linux, PROJECT_ROOT on Windows)
    # Try data root first (for EC2/Linux), then fall back to project root
    model_data_base = get_model_data_root()
    
    # Check both locations (data root and project root)
    model_data_paths = [
        model_data_base / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events.parquet",
        project_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events.parquet"
    ]
    
    model_data_path = None
    for path in model_data_paths:
        if path.exists():
            model_data_path = path
            logger.info(f"Found model_data at: {model_data_path}")
            break
    
    if model_data_path is None:
        # Use first path as default (will error if not found)
        model_data_path = model_data_paths[0]
        logger.warning(f"Model data not found in any location. Will try: {model_data_path}")
    
    if not model_data_path.exists():
        logger.error(f"Model data not found: {model_data_path}")
        return pd.DataFrame()
    
    # No FP-Growth filtering - use all codes except F1120
    # allowed_codes is None, which means extract_patient_trajectories will allow all codes
    allowed_codes = None
    
    logger.info("Using all codes from model_data (no FP-Growth filtering)")
    
    # Get cutoff dates for all patients (target and control)
    con = duckdb.connect()
    
    # Determine target event identification based on cohort
    # opioid_ed: target event is F1120 (opioid use disorder ICD code)
    # non_opioid_ed: target event is polypharmacy ED visit (identified by HCG fields)
    if "opioid" in cohort_name.lower():
        # For opioid_ed: find first F1120 event date
        logger.info("Using F1120 event date for target patients (opioid_ed cohort)")
        cutoff_dates_df = con.execute(f"""
            WITH target_cutoffs AS (
                SELECT DISTINCT
                    mi_person_key,
                    MIN(CASE 
                        WHEN primary_icd_diagnosis_code LIKE '%F1120%' 
                             OR two_icd_diagnosis_code LIKE '%F1120%'
                             OR three_icd_diagnosis_code LIKE '%F1120%'
                             OR four_icd_diagnosis_code LIKE '%F1120%'
                             OR five_icd_diagnosis_code LIKE '%F1120%'
                        THEN event_date 
                        END) as cutoff_date
                FROM read_parquet('{model_data_path}')
                WHERE target = 1
                GROUP BY mi_person_key
                HAVING cutoff_date IS NOT NULL
            ),
            control_cutoffs AS (
                SELECT 
                    mi_person_key,
                    NULL as cutoff_date  -- Controls: no cutoff, use all events
                FROM read_parquet('{model_data_path}')
                WHERE target = 0
                GROUP BY mi_person_key
            )
            SELECT mi_person_key, cutoff_date FROM target_cutoffs
            UNION ALL
            SELECT mi_person_key, cutoff_date FROM control_cutoffs
        """).df()
    else:
        # For non_opioid_ed: find first polypharmacy ED event (HCG-based ED visit)
        # Polypharmacy ED = medical event with hcg_setting = 'ED' or hcg_line IS NOT NULL
        logger.info("Using first polypharmacy ED event date for target patients (non_opioid_ed cohort)")
        cutoff_dates_df = con.execute(f"""
            WITH target_cutoffs AS (
                SELECT DISTINCT
                    mi_person_key,
                    MIN(CASE 
                        WHEN (hcg_setting = 'ED' OR hcg_line IS NOT NULL)
                             AND primary_icd_diagnosis_code NOT LIKE '%F1120%'
                             AND (two_icd_diagnosis_code IS NULL OR two_icd_diagnosis_code NOT LIKE '%F1120%')
                             AND (three_icd_diagnosis_code IS NULL OR three_icd_diagnosis_code NOT LIKE '%F1120%')
                             AND (four_icd_diagnosis_code IS NULL OR four_icd_diagnosis_code NOT LIKE '%F1120%')
                             AND (five_icd_diagnosis_code IS NULL OR five_icd_diagnosis_code NOT LIKE '%F1120%')
                        THEN event_date 
                        END) as cutoff_date
                FROM read_parquet('{model_data_path}')
                WHERE target = 1
                GROUP BY mi_person_key
                HAVING cutoff_date IS NOT NULL
            ),
            control_cutoffs AS (
                SELECT 
                    mi_person_key,
                    NULL as cutoff_date  -- Controls: no cutoff, use all events
                FROM read_parquet('{model_data_path}')
                WHERE target = 0
                GROUP BY mi_person_key
            )
            SELECT mi_person_key, cutoff_date FROM target_cutoffs
            UNION ALL
            SELECT mi_person_key, cutoff_date FROM control_cutoffs
        """).df()
    
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
        
        # Extract trajectories using cutoff dates
        # Convert cutoff_dates_df to dict format
        cutoff_dates_dict = cutoff_dates_df.set_index('mi_person_key')['cutoff_date'].to_dict()
        # Convert dates to strings if needed
        cutoff_dates_dict = {str(k): str(v) for k, v in cutoff_dates_dict.items()}
        
        result = extract_patient_trajectories(
            model_data_path=model_data_path,
            allowed_codes=allowed_codes,
            cohort_name=cohort_name,
            item_type=item_type,
            target_filter=None,  # Include both target and control
            cutoff_dates=cutoff_dates_dict
        )
        
        # Handle tuple return (trajectories, metadata)
        if isinstance(result, tuple):
            patient_trajectories, trajectory_metadata = result
        else:
            # Backward compatibility: if old format (dict only)
            patient_trajectories = result
            trajectory_metadata = {}
        
        if not patient_trajectories:
            logger.warning(f"No patient trajectories for {item_type}, skipping")
            continue
        
        # Add item type prefix to avoid collisions when combining
        prefixed_trajectories = {
            f"{item_type}_{pid}": traj 
            for pid, traj in patient_trajectories.items()
        }
        all_trajectories_combined.update(prefixed_trajectories)
        
        # Store metadata with prefixed keys
        if trajectory_metadata:
            if not hasattr(create_all_dtw_features, '_all_metadata'):
                create_all_dtw_features._all_metadata = {}
            for pid, meta in trajectory_metadata.items():
                prefixed_pid = f"{item_type}_{pid}"
                create_all_dtw_features._all_metadata[prefixed_pid] = meta
    
    if not all_trajectories_combined:
        logger.warning("No trajectories extracted for any item type")
        return pd.DataFrame()
    
    logger.info(f"Total trajectories for prototype creation: {len(all_trajectories_combined)}")
    
    # Create prototypes from combined trajectories (target + control)
    logger.info(f"Creating {n_prototypes} prototypes from combined trajectories...")
    
    # Get combined metadata
    combined_metadata = getattr(create_all_dtw_features, '_all_metadata', {})
    
    dtw_features_combined, trajectories_for_viz, prototype_indices = compute_dtw_distances_to_prototypes(
        patient_trajectories=all_trajectories_combined,
        trajectory_metadata=combined_metadata if combined_metadata else None,
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
            elif 'temporal' in col.lower() or 'interval' in col.lower():
                # Fill temporal metrics with 0 for patients without trajectories
                combined_features[col] = combined_features[col].fillna(0)
    
    logger.info(f"\nCreated {len(combined_features.columns) - 2} DTW features for {len(combined_features)} patients")
    
    # Store trajectories for visualization (remove item type prefix)
    trajectories_for_viz_clean = {}
    for pid, traj in trajectories_for_viz.items():
        clean_pid = pid.replace('drug_', '').replace('icd_', '').replace('cpt_', '').replace('combined_', '')
        if clean_pid not in trajectories_for_viz_clean:
            trajectories_for_viz_clean[clean_pid] = traj
    
    return combined_features, trajectories_for_viz_clean


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
    
    # Extract trajectories if returned
    trajectories_for_viz = {}
    if isinstance(dtw_features, tuple):
        dtw_features, trajectories_for_viz = dtw_features
    
    if dtw_features.empty:
        logger.error("No features created. Check inputs and logs.")
        return
    
    # Set output path - intermediate file for DTW features only
    if not args.output:
        age_band_fname = args.age_band.replace("-", "_")
        feature_eng_dir = project_root / "3b_feature_importance_eda" / "outputs" / "feature_engineering"
        feature_eng_dir.mkdir(parents=True, exist_ok=True)
        args.output = feature_eng_dir / f"dtw_features_{args.cohort}_{age_band_fname}.csv"
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtw_features.to_csv(output_path, index=False)
    
    print(f"\nCreated {len(dtw_features.columns) - 1} DTW features for {len(dtw_features)} patients")
    print(f"Output format: Ready for merging with other features (uses mi_person_key)")
    print(f"Saved to: {output_path}")
    
    # Create visualizations if trajectories are available
    if trajectories_for_viz:
        try:
            import sys
            viz_script = Path(__file__).parent / "create_dtw_visualizations.py"
            if viz_script.exists():
                sys.path.insert(0, str(Path(__file__).parent))
                from create_dtw_visualizations import create_dtw_visualizations
                
                age_band_fname = args.age_band.replace("-", "_")
                # Use PROJECT_ROOT for outputs (not data root) - outputs go in project directory
                plots_dir = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / args.cohort / age_band_fname / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                viz_files = create_dtw_visualizations(
                    patient_trajectories=trajectories_for_viz,
                    dtw_features_df=dtw_features,
                    cohort_name=args.cohort,
                    age_band=args.age_band,
                    output_dir=plots_dir,
                    n_sample_trajectories=10
                )
                
                if viz_files:
                    print(f"\nCreated {len(viz_files)} DTW visualization files:")
                    for f in viz_files:
                        print(f"  - {f}")
                    
                    # Upload visualizations to S3
                    try:
                        from py_helpers.checkpoint_utils import upload_file_to_s3
                        from py_helpers.common_imports import S3_BUCKET
                        
                        s3_base_path = f"s3://{S3_BUCKET}/gold/feature_importance/{args.cohort}/{args.age_band}/plots/"
                        for viz_file in viz_files:
                            s3_viz_path = f"{s3_base_path}{viz_file.name}"
                            if upload_file_to_s3(viz_file, s3_viz_path, check_exists=True):
                                print(f"  Uploaded to S3: {s3_viz_path}")
                            else:
                                logger.warning(f"Failed to upload {viz_file.name} to S3 (non-critical)")
                    except Exception as e:
                        logger.warning(f"Could not upload visualizations to S3: {e}")
            else:
                logger.warning(f"Visualization script not found: {viz_script}")
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
            import traceback
            traceback.print_exc()
    
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

