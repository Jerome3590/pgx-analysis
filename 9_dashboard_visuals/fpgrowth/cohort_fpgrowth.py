#!/usr/bin/env python3
"""
Cohort-Specific FPGrowth Feature Importance Analysis

Processes each cohort separately to find cohort-specific patterns across:
- drug_name (pharmacy events)
- icd_code (medical diagnosis codes)
- cpt_code (medical procedure codes)

FP-Growth uses the same SHAP/FFA combined allowed codes file as BupaR and DTW (required
prerequisite); we never use all items.

Outputs to: s3://pgxdatalake/gold/fpgrowth/cohort/{item_type}/cohort_name={cohort}/age_band={age}/event_year={year}/
"""

import os
import shutil
import sys
import time
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import boto3
import psutil
import duckdb
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

try:
    from py_helpers.constants import COHORT_NAMES, AGE_BANDS, EVENT_YEARS
except ImportError:
    COHORT_NAMES = ["opioid_ed", "non_opioid_ed"]
    AGE_BANDS = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
    EVENT_YEARS = ["2016", "2017", "2018", "2019", "2020"]


# Script lives in 9_dashboard_visuals/fpgrowth; outputs go to 10_risk_dashboard/visualizations/fpgrowth
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 9_dashboard_visuals
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

# FP-Growth parameters (very permissive since data is pre-filtered by SHAP/FFA to important features only)
MIN_SUPPORT = 0.01       # 1% support (find rare but meaningful patterns in pre-filtered important features)
MIN_CONFIDENCE = 0.2     # 20% confidence (permissive - we capture weak associations between important features)

# CPT-specific parameters (also permissive since working with curated important features)
MIN_SUPPORT_CPT = 0.05   # 5% support for CPT codes (permissive)
MIN_CONFIDENCE_CPT = 0.3 # 30% confidence for CPT (permissive)

# Rule limits (focus on most important rules)
MAX_RULES_PER_COHORT = 1000  # Keep top 1000 rules by lift (practical limit)

# Target-only pass: single pool of target transactions (no density bins), so use lower support
# to get multi-item itemsets and thus rules (association_rules needs ≥2-item itemsets).
MIN_SUPPORT_TARGET_ONLY = 0.005  # 0.5% (lower than main run so smaller cohorts still get rules)

# Target-focused rule mining
TARGET_FOCUSED = True  # Only generate rules that predict target outcomes
TARGET_ICD_CODES = ['F11.20', 'F11.21', 'F11.22', 'F11.23', 'F11.24', 'F11.25', 'F11.29']  # Opioid dependence codes
TARGET_HCG_LINES = [
    "P51 - ER Visits and Observation Care",
    "O11 - Emergency Room",
    "P33 - Urgent Care Visits"
]  # ED visits (HCG Line codes)
TARGET_PREFIXES = ['TARGET_ICD:', 'TARGET_ED:']  # Prefixes for target items in transactions

# Processing parameters: one core per (cohort, age_band) combo, capped by CPU count
_ncpu = getattr(os, "cpu_count", lambda: 4)() or 4
MAX_WORKERS = min(_ncpu, len(COHORT_NAMES) * len(AGE_BANDS))
# DuckDB threads per connection (each item type has its own connection; 3 threads per item type)
DUCKDB_THREADS = 3

# Training window for FP-Growth: consolidate all years in model data to maximize transactions and produce rules
TRAIN_YEARS = [2016, 2017, 2018, 2019]  # All years in model_events (consolidating produces more itemsets/rules)
# Business rule: rules must reflect patterns that persist across years (not rare/single-year flukes)
MIN_YEARS_FOR_RULE = 2  # Pattern must appear in at least 2 of the 4 TRAIN years (see README)

# Transaction density bins (based on histogram/percentiles)
DENSITY_BINS = ['low', 'medium', 'high', 'extreme']  # Process in this order
# No minimum transaction count per bin: data is already filtered by SHAP/FFA feature importance; process every non-empty bin to maximize rules

# Itemset filtering (minimal threshold - data already filtered to important features via SHAP/FFA)
MIN_ITEMSET_LIFT = 1.0  # No lift filtering (lift=1.0 means independence; we accept all patterns since features are pre-curated)

# DRY RUN MODE (only applies when running cohort_fpgrowth.py directly as batch; dashboard uses run_single_cohort_fpgrowth per combo)
DRY_RUN = False  # Set to True to limit to DRY_RUN_LIMIT when testing batch runs
DRY_RUN_LIMIT = 5  # Number of cohort combinations to process when DRY_RUN is True

COHORTS_TO_PROCESS = ['opioid_ed', 'non_opioid_ed']  # Specify cohorts to process

# FP-Growth item types by cohort. Each type gets a separate graph network; user selects which to view.
# - non_opioid_ed (polypharmacy): drug_name only
# - opioid_ed: drug_name, icd_code, cpt_code (each has its own network)
COHORT_ITEM_TYPES = {
    "non_opioid_ed": ["drug_name"],
    "opioid_ed": ["drug_name", "icd_code", "cpt_code"],
}
# All item types (for validation and batch summary; medical_code kept for backward compat if added later)
ALL_ITEM_TYPES = ["drug_name", "icd_code", "cpt_code", "medical_code"]
# Default for unknown cohort: drug + icd + cpt
ITEM_TYPES = ALL_ITEM_TYPES  # used where a single list is expected (e.g. batch summary iteration)


def get_item_types_for_cohort(cohort_name: str) -> list:
    """Return item types to run for this cohort. Each type produces a distinct graph network (user selects which to view)."""
    return COHORT_ITEM_TYPES.get(cohort_name, ["drug_name", "icd_code", "cpt_code"])
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/cohort"
LOCAL_DATA_PATH = Path("/mnt/nvme/cohorts")  # Instance storage (NVMe SSD for fast I/O)

# Optional model_data root (filtered to important features + 5:1 control ratio).
# If a model_data file exists for a given (cohort, age_band), FP-Growth will
# prefer it over the raw GOLD cohorts parquet.
# NOTE: The canonical location for model_data in this project is now 4_model_data.
MODEL_DATA_ROOT = REPO_ROOT / "4_model_data"
USE_MODEL_DATA_IF_AVAILABLE = True

# Restrict FP-Growth to SHAP/FFA combined allowed codes (same prerequisite as BupaR/DTW; we never use all items).

# Local FP-Growth outputs (step 10: risk dashboard visualization outputs only)
LOCAL_OUTPUT_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth" / "outputs"


def _normalize_code(s: str) -> str:
    """Normalize code for set membership (e.g. F11.20 and F1120 match)."""
    if not s or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().replace(".", "").replace("-", "")


def _load_allowed_codes_by_type(
    cohort_name: str, age_band: str, item_type: str, project_root: Path
) -> set:
    """
    Load SHAP/FFA combined allowed codes from the same JSON as BupaR/DTW (required prerequisite).
    Returns the set of normalized codes for the given item_type (drug_name, icd_code, cpt_code, medical_code).
    Raises FileNotFoundError if the file is missing; ValueError if empty.
    """
    age_band_fname = age_band.replace("-", "_")
    path = project_root / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
    path = path / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP/FFA allowed codes file is required (prerequisite). Not found: {path}. "
            "Generate the combined file before running FP-Growth (same as BupaR/DTW)."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    allowed_codes = {str(c).strip() for c in raw if c is not None and str(c).strip()}
    if not allowed_codes:
        raise ValueError(
            f"SHAP/FFA allowed codes file is empty: {path}. Cannot run FP-Growth without allowed codes."
        )
    try:
        from py_helpers.shap_ffa_fpgrowth_utils import _parse_feature_name
    except ImportError:
        _parse_feature_name = None

    drug_set = set()
    icd_set = set()
    cpt_set = set()
    for c in allowed_codes:
        s = str(c).strip()
        norm = _normalize_code(s)
        if not norm:
            continue
        if s.startswith("cpt_"):
            cpt_set.add(_normalize_code(s[4:]))
        elif s.startswith("icd_"):
            icd_set.add(_normalize_code(s[4:]))
        elif s.startswith("drug_"):
            drug_set.add(_normalize_code(s[5:]))
        elif _parse_feature_name:
            typ, code = _parse_feature_name(s)
            raw_norm = _normalize_code(code) if code else norm
            if typ == "cpt":
                cpt_set.add(raw_norm)
            elif typ == "icd":
                icd_set.add(raw_norm)
            elif typ == "drug":
                drug_set.add(raw_norm)
            else:
                drug_set.add(norm)
                icd_set.add(norm)
                cpt_set.add(norm)
        else:
            drug_set.add(norm)
            icd_set.add(norm)
            cpt_set.add(norm)

    if item_type == "drug_name":
        return drug_set
    if item_type == "icd_code":
        return icd_set
    if item_type == "cpt_code":
        return cpt_set
    if item_type == "medical_code":
        return drug_set | icd_set | cpt_set
    return drug_set | icd_set | cpt_set


# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logger(name: str = 'cohort_fpgrowth') -> logging.Logger:
    """Setup logger with console output."""
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def _model_data_path(cohort_name: str, age_band: str, project_root: Path | None = None) -> Path | None:
    """Return path to model_events parquet if it exists; else None. Uses shared resolver (same as BupaR/DTW)."""
    root = Path(project_root).resolve() if project_root is not None else REPO_ROOT
    try:
        from py_helpers.model_data_paths import resolve_model_events_path
        return resolve_model_events_path(root, cohort_name, age_band)
    except Exception:  # noqa: BLE001
        return None


def _model_data_paths(cohort_name: str, age_band: str, project_root: Path | None = None) -> list:
    """Return 1 or 2 paths for model_events (2 only for 85-114 when 85-94 + 95-114 exist). Uses shared resolver (same as BupaR/DTW)."""
    root = Path(project_root).resolve() if project_root is not None else REPO_ROOT
    try:
        from py_helpers.model_data_paths import resolve_model_events_paths
        return resolve_model_events_paths(root, cohort_name, age_band)
    except Exception:  # noqa: BLE001
        return []


def _model_data_from_sql(paths: list) -> str:
    """Build DuckDB FROM clause: single read_parquet or UNION of two (for 85-114 = 85-94 + 95-114)."""
    if not paths:
        raise ValueError("model_data paths list is empty")
    normalized = [str(p).replace("\\", "/") for p in paths]
    if len(normalized) == 1:
        return f"read_parquet('{normalized[0]}')"
    if len(normalized) == 2:
        return f"(SELECT * FROM read_parquet('{normalized[0]}') UNION ALL SELECT * FROM read_parquet('{normalized[1]}'))"
    raise ValueError(f"Expected 1 or 2 model_data paths, got {len(paths)}")

# =============================================================================
# COHORT PROCESSING
# =============================================================================

def log_memory_cpu(logger, stage: str = ""):
    """Log current memory and CPU usage to help detect hangs and resource issues."""
    try:
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_percent = mem.percent
        mem_avail_gb = mem.available / (1024**3)
        # Process CPU (this process); interval=None for non-blocking snapshot
        proc = psutil.Process()
        cpu_proc = proc.cpu_percent(interval=None)
        # System-wide CPU (short interval to avoid long blocks; 0.1s)
        cpu_sys = psutil.cpu_percent(interval=0.1)
        logger.info(
            f"[RESOURCE {stage}] mem_used={mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB ({mem_percent:.1f}%) avail={mem_avail_gb:.1f}GB | "
            f"cpu_process={cpu_proc:.1f}% cpu_system={cpu_sys:.1f}%"
        )
        if mem_percent > 85:
            logger.warning("WARNING: HIGH MEMORY: %.1f%% - may cause OOM", mem_percent)
        return mem_percent
    except Exception as e:
        logger.error(f"Error getting resource info: {e}")
        return 0.0


def log_memory(logger, stage: str = ""):
    """Log current memory and CPU (calls log_memory_cpu for consistency)."""
    return log_memory_cpu(logger, stage)


def assign_transaction_density(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Calculate transaction sizes per patient and assign Transaction_Density bins.
    
    Args:
        df: DataFrame with mi_person_key and item columns
        logger: Logger instance
    
    Returns:
        DataFrame with Transaction_Density column added
    """
    if len(df) == 0:
        df = df.copy()
        df["Transaction_Density"] = pd.Series(dtype=object)
        return df

    # Calculate transaction size per patient
    logger.info(f"Calculating transaction sizes per patient...")
    transaction_sizes = df.groupby('mi_person_key')['item'].size().reset_index(name='transaction_size')
    if len(transaction_sizes) == 0:
        df = df.copy()
        df["Transaction_Density"] = pd.Series(dtype=object)
        return df

    # Calculate percentiles for binning
    sizes = transaction_sizes['transaction_size'].values
    p25 = np.percentile(sizes, 25)
    p50 = np.percentile(sizes, 50)
    p75 = np.percentile(sizes, 75)
    p95 = np.percentile(sizes, 95)
    
    logger.info(f"Transaction size percentiles:")
    logger.info(f"  P25: {p25:.1f} items")
    logger.info(f"  P50 (median): {p50:.1f} items")
    logger.info(f"  P75: {p75:.1f} items")
    logger.info(f"  P95: {p95:.1f} items")
    logger.info(f"  Max: {max(sizes):,} items")
    
    # Assign density bins based on percentiles
    def assign_density(size):
        if size <= p25:
            return 'low'
        elif size <= p50:
            return 'medium'
        elif size <= p95:
            return 'high'
        else:
            return 'extreme'
    
    transaction_sizes['Transaction_Density'] = transaction_sizes['transaction_size'].apply(assign_density)
    
    # Log distribution
    density_counts = transaction_sizes['Transaction_Density'].value_counts()
    logger.info(f"Transaction density distribution:")
    for density in DENSITY_BINS:
        count = density_counts.get(density, 0)
        pct = (count / len(transaction_sizes)) * 100 if len(transaction_sizes) > 0 else 0
        logger.info(f"  {density}: {count:,} ({pct:.1f}%)")
    
    # Merge density back to original dataframe
    df_with_density = df.merge(
        transaction_sizes[['mi_person_key', 'Transaction_Density', 'transaction_size']],
        on='mi_person_key',
        how='left'
    )
    
    return df_with_density


def get_transactions_by_density(df: pd.DataFrame, density: str, logger: logging.Logger) -> List[List[str]]:
    """
    Get transactions for a specific density level.
    
    Args:
        df: DataFrame with mi_person_key, item, and Transaction_Density columns
        density: Density level ('low', 'medium', 'high', 'extreme')
        logger: Logger instance
    
    Returns:
        List of transactions (each transaction is a list of items)
    """
    df_density = df[df['Transaction_Density'] == density]
    
    if len(df_density) == 0:
        return []
    
    # Ensure all items are strings and valid before creating transactions
    transactions = (
        df_density.groupby('mi_person_key')['item']
        .apply(lambda x: sorted(set(str(item).strip() for item in x.tolist() if pd.notna(item) and str(item).strip())))
        .tolist()
    )
    
    # Filter out empty transactions
    transactions = [t for t in transactions if len(t) > 0]
    
    logger.info(f"  {density}: {len(transactions):,} transactions")
    
    return transactions


def filter_itemsets_by_lift(
    itemsets: pd.DataFrame,
    df_encoded: pd.DataFrame,
    min_lift: float,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Filter itemsets by lift to remove common/trivial itemsets.
    
    Lift measures how much more likely items are to appear together than by chance.
    Lift = 1.0 means items are independent (not interesting)
    Lift > 1.0 means positive association (interesting)
    Lift < 1.0 means negative association (also interesting, but we filter these out)
    
    Args:
        itemsets: DataFrame with 'itemsets' and 'support' columns
        df_encoded: Encoded transaction DataFrame (needed to calculate individual item supports)
        min_lift: Minimum lift threshold (e.g., 1.1 = 10% more likely than chance)
        logger: Logger instance
    
    Returns:
        Filtered DataFrame with only itemsets above min_lift threshold
    """
    if len(itemsets) == 0:
        return itemsets
    
    logger.info(f"Filtering {len(itemsets):,} itemsets by lift (min_lift={min_lift})...")
    
    # Calculate individual item supports (needed for lift calculation)
    item_supports = {}
    total_transactions = len(df_encoded)
    
    for col in df_encoded.columns:
        item_supports[col] = df_encoded[col].sum() / total_transactions
    
    # Calculate lift for each itemset
    def calculate_lift(row):
        itemset = row['itemsets']
        itemset_support = row['support']
        
        # For single-item itemsets, lift is undefined (or 1.0 by convention)
        if len(itemset) == 1:
            return 1.0  # Single items don't have lift
        
        # For multi-item itemsets: lift = itemset_support / (item1_support * item2_support * ...)
        expected_support = 1.0
        for item in itemset:
            if item in item_supports:
                expected_support *= item_supports[item]
            else:
                # Item not found in transactions (shouldn't happen, but handle gracefully)
                return 0.0
        
        if expected_support == 0:
            return 0.0
        
        lift = itemset_support / expected_support
        return lift
    
    itemsets['lift'] = itemsets.apply(calculate_lift, axis=1)
    
    # Filter by lift threshold
    original_count = len(itemsets)
    itemsets_filtered = itemsets[itemsets['lift'] >= min_lift].copy()
    filtered_count = len(itemsets_filtered)
    removed_count = original_count - filtered_count
    
    logger.info(f"  Original itemsets: {original_count:,}")
    logger.info(f"  Filtered itemsets: {filtered_count:,} (lift >= {min_lift})")
    logger.info(f"  Removed common/trivial: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
    
    if filtered_count > 0:
        logger.info(f"  Lift range: {itemsets_filtered['lift'].min():.3f} - {itemsets_filtered['lift'].max():.3f}")
    
    return itemsets_filtered.drop(columns=['lift'])  # Remove lift column (not needed in output)


def ensure_subsets_for_association_rules(
    itemsets_filtered: pd.DataFrame,
    itemsets_original: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Ensure every subset of every kept itemset is present so mlxtend association_rules
    can look up antecedent/consequent support (it raises KeyError otherwise).
    Adds missing subsets from itemsets_original; does not change lift filtering.
    """
    if len(itemsets_filtered) == 0:
        return itemsets_filtered
    original_by_itemset = {}
    for _, row in itemsets_original.iterrows():
        k = row["itemsets"]
        original_by_itemset[frozenset(k) if not isinstance(k, frozenset) else k] = row["support"]
    kept = set()
    for v in itemsets_filtered["itemsets"].values:
        kept.add(frozenset(v) if not isinstance(v, frozenset) else v)
    to_add = []
    for itemset in list(kept):
        n = len(itemset)
        if n <= 1:
            continue
        for r in range(1, n):
            for sub in combinations(itemset, r):
                sub_fs = frozenset(sub)
                if sub_fs not in kept:
                    support = original_by_itemset.get(sub_fs)
                    if support is not None:
                        to_add.append({"support": support, "itemsets": sub_fs})
                        kept.add(sub_fs)
    if not to_add:
        return itemsets_filtered
    logger.debug("Adding %d subset itemsets so association_rules can look up antecedent/consequent support", len(to_add))
    extra = pd.DataFrame(to_add)
    return pd.concat([itemsets_filtered, extra], ignore_index=True)


def filter_rules_by_year_support(
    rules: pd.DataFrame,
    df: pd.DataFrame,
    min_years: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Keep only rules whose pattern appears in at least min_years distinct years.
    Ensures rules are not driven by rare/single-year flukes (business rule: 2 of 4 years).
    df must have columns: mi_person_key, item, event_year (item = prefixed, e.g. DRUG:xxx).
    """
    if len(rules) == 0 or "antecedents" not in rules.columns or "consequents" not in rules.columns:
        return rules
    if "event_year" not in df.columns or min_years < 1:
        return rules

    # (mi_person_key, event_year) -> set of items in that person-year
    df_yr = df[["mi_person_key", "event_year", "item"]].dropna(subset=["event_year"])
    person_year_items = (
        df_yr.groupby(["mi_person_key", "event_year"])["item"]
        .apply(lambda x: set(x.dropna().astype(str)))
        .to_dict()
    )

    keep = []
    for _, row in rules.iterrows():
        ant = row["antecedents"]
        con = row["consequents"]
        if not isinstance(ant, (set, frozenset)):
            ant = frozenset(ant) if hasattr(ant, "__iter__") and not isinstance(ant, str) else frozenset()
        if not isinstance(con, (set, frozenset)):
            con = frozenset(con) if hasattr(con, "__iter__") and not isinstance(con, str) else frozenset()
        needed = ant | con
        if not needed:
            keep.append(False)
            continue
        years_with_pattern = set()
        for (_, yr), items in person_year_items.items():
            if needed <= items:
                years_with_pattern.add(yr)
        keep.append(len(years_with_pattern) >= min_years)

    kept = sum(keep)
    dropped = len(rules) - kept
    logger.info(
        "Rule year filter (pattern in >= %d of 4 train years): kept %d, dropped %d",
        min_years, kept, dropped,
    )
    return rules.loc[np.array(keep)].reset_index(drop=True)


def process_single_cohort(
    item_type: str,
    cohort_name: str,
    age_band: str,
    event_year: int,
    local_data_path: Path,
    s3_output_base: str,
    min_support: float,
    min_confidence: float,
    project_root: Path | None = None,
) -> Dict:
    """
    Process a single cohort for a single item type.
    
    Returns:
        Dictionary with processing metrics
    """
    # Create process-specific logger
    logger = setup_logger(f'cohort_{cohort_name}_{age_band}_{event_year}_{item_type}')
    
    cohort_id = f"{cohort_name}/{age_band}/{event_year}"
    logger.info(f"Processing {item_type} for {cohort_id}")
    log_memory(logger, "START")
    
    start_time = time.time()
    
    # Convert age_band to filename format (hyphens to underscores for EC2 compatibility)
    age_band_fname = age_band.replace("-", "_")
    
    try:
        # Simple in-memory connection (no AWS needed for local parquet reads)
        con = duckdb.connect(':memory:')
        con.sql(f"SET threads = {DUCKDB_THREADS}")

        # Prefer DTW-filtered model_data (protocol events removed) if available,
        # then fall back to regular model_data, then raw GOLD cohorts parquet.
        # This ensures FP-Growth only uses useful signals (non-protocol events) for itemsets and rules.
        # Use smart path resolution (checks 3b, $PGX_DATA_ROOT, /mnt/nvme/, 4_model_data, 4a_model_data).
        # For 85-114, may union 85-94 + 95-114 when single 85-114 partition is not present.
        # Pass project_root so subprocess uses same base as parent (BupaR/DTW use same paths).
        model_data_paths = _model_data_paths(cohort_name, age_band, project_root=project_root)
        if model_data_paths:
            resolved_str = " + ".join(str(p) for p in model_data_paths)
            logger.info(f"Resolved model_events for {cohort_name}/{age_band}: {resolved_str}")
        else:
            logger.info(f"Resolved model_events for {cohort_name}/{age_band}: none (tried 3b, /mnt/nvme/4_model_data, $PGX_DATA_ROOT, 4_model_data, 4a_model_data; both age_band formats)")

        # Special handling for aggregated training window ("train" = all years, 2016–2019)
        event_label = str(event_year)
        if event_label == "train":
            # Training FP-Growth requires model_data (filtered important items + 5:1 control).
            # This keeps memory usage manageable and ensures only useful signals are used.
            if USE_MODEL_DATA_IF_AVAILABLE:
                if model_data_paths:
                    model_data_from = _model_data_from_sql(model_data_paths)
                    parquet_file = model_data_paths[0]  # for logging / schema check
                    file_type = "no_protocols" if "no_protocols" in str(parquet_file) else "regular"
                    if len(model_data_paths) == 2:
                        logger.info(f"Using {file_type} model_data for TRAIN FP-Growth (85-114 = 85-94 + 95-114): {model_data_paths[0]} + {model_data_paths[1]}")
                    else:
                        logger.info(f"Using {file_type} model_data for TRAIN FP-Growth (2016–2019): {parquet_file}")
                else:
                    try:
                        from py_helpers.model_data_paths import get_model_events_paths_checked, get_path_check_listings
                        _root = (Path(project_root).resolve() if project_root is not None else REPO_ROOT)
                        paths_checked = get_model_events_paths_checked(_root, cohort_name, age_band)
                        path_listings = get_path_check_listings(paths_checked) if paths_checked else []
                    except Exception:  # noqa: BLE001
                        paths_checked = []
                        path_listings = []
                    logger.warning(
                        "TRAIN FP-Growth requested for %s/%s but model_data file not found. Checked: 3b, $PGX_DATA_ROOT, /mnt/nvme/, 4_model_data, 4a_model_data (for 85-114 also 85-94+95-114). Run create_model_data.py first for this cohort/age_band.",
                        cohort_name, age_band,
                    )
                    logger.error(
                        "[ERROR_PARAMS] step=4_fpgrowth cohort_name=%s age_band=%s item_type=%s error=TRAIN model_data not found paths_checked=%s",
                        cohort_name, age_band, item_type,
                        " | ".join(paths_checked) if paths_checked else "(none)",
                    )
                    if path_listings:
                        logger.error(
                            "[ERROR_PARAMS] step=4_fpgrowth path_listings: %s",
                            " ; ".join(path_listings),
                        )
                    return {
                        'item_type': item_type,
                        'cohort_name': cohort_name,
                        'age_band': age_band,
                        'event_year': event_year,
                        'error': 'TRAIN model_data not found',
                        'paths_checked': paths_checked,
                        'path_listings': path_listings,
                    }
            else:
                logger.warning("USE_MODEL_DATA_IF_AVAILABLE is False, but TRAIN FP-Growth requires model_data")
                logger.error(
                    "[ERROR_PARAMS] step=4_fpgrowth cohort_name=%s age_band=%s item_type=%s error=TRAIN requires model_data but USE_MODEL_DATA_IF_AVAILABLE is False",
                    cohort_name, age_band, item_type,
                )
                return {
                    'item_type': item_type,
                    'cohort_name': cohort_name,
                    'age_band': age_band,
                    'event_year': event_year,
                    'error': 'TRAIN requires model_data but USE_MODEL_DATA_IF_AVAILABLE is False'
                }
        else:
            # For non-TRAIN years, prefer model_data if available, else fall back to raw GOLD cohorts
            if USE_MODEL_DATA_IF_AVAILABLE and model_data_paths:
                model_data_from = _model_data_from_sql(model_data_paths)
                parquet_file = model_data_paths[0]
                file_type = "no_protocols" if "no_protocols" in str(parquet_file) else "regular"
                if len(model_data_paths) == 2:
                    logger.info(f"Using {file_type} model_data for FP-Growth (85-114 = 85-94 + 95-114): {model_data_paths[0]} + {model_data_paths[1]}")
                else:
                    logger.info(f"Using {file_type} model_data for FP-Growth: {parquet_file}")
            else:
                model_data_from = None  # will use parquet_file for GOLD
                parquet_file = (
                    local_data_path
                    / f"cohort_name={cohort_name}"
                    / f"event_year={event_year}"
                    / f"age_band={age_band_fname}"
                    / "cohort.parquet"
                )

        if model_data_from is None and not parquet_file.exists():
            logger.warning("Cohort file not found: %s", parquet_file)
            logger.error(
                "[ERROR_PARAMS] step=4_fpgrowth cohort_name=%s age_band=%s item_type=%s error=File not found path=%s",
                cohort_name, age_band, item_type, str(parquet_file),
            )
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'File not found',
                'path': str(parquet_file),
            }
        if model_data_from is None:
            model_data_from = f"read_parquet('{str(parquet_file).replace(chr(92), '/')}')"

        # Determine event_year filter (single year vs aggregated TRAIN window).
        # When "train", accept numeric years 2016-2019 or literal 'train' (some model_data uses event_year='train').
        event_label = str(event_year)
        if event_label == "train":
            year_list = ", ".join(str(y) for y in TRAIN_YEARS)
            event_filter = f"(event_year IN ({year_list}) OR event_year = 'train')"
        else:
            event_filter = f"event_year = {event_year}"

        # Verify parquet schema before querying (use first path for union case)
        try:
            schema_check = con.execute(f"DESCRIBE SELECT * FROM {model_data_from} LIMIT 0").fetchall()
            available_cols = {row[0] for row in schema_check}
            keys_received = sorted(available_cols)
            keys_expected_item = ["mi_person_key", "target", "event_year", item_type if item_type != "medical_code" else "drug_name/icd/cpt columns"]
            logger.info("keys_expected (parquet, for item_type=%s): %s", item_type, keys_expected_item)
            logger.info("keys_received (parquet): %s", keys_received[:30] if len(keys_received) > 30 else keys_received)
            logger.info("Parquet schema has %s columns: %s...", len(available_cols), keys_received[:20])
        except Exception as e:
            logger.error("Failed to read parquet schema from %s: %s", parquet_file, e)
            logger.error(
                "[ERROR_PARAMS] step=4_fpgrowth cohort_name=%s age_band=%s item_type=%s error=Schema read failure "
                "keys_expected=parquet_schema keys_received=N/A (read failed) path=%s exception=%s",
                cohort_name, age_band, item_type, str(parquet_file), str(e),
            )
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': f'Schema read failure: {e}',
                'path': str(parquet_file),
            }

        # Build query based on item type. Target cohort only (target=1); controls would add noise to the network.
        # We include `target` in the SELECT for the later target-only FP-Growth pass (same cohort, no density stratification).
        if item_type == 'drug_name':
            # 4_model_data already encodes event context; filter directly on drug_name.
            if 'drug_name' not in available_cols:
                keys_expected = ["drug_name", "mi_person_key", "target", "event_year"]
                keys_received = sorted(available_cols)
                logger.error("Column 'drug_name' not found. keys_expected=%s keys_received=%s", keys_expected, keys_received[:40])
                logger.error(
                    "[ERROR_PARAMS] step=4_fpgrowth cohort_name=%s age_band=%s item_type=%s error=Column drug_name not in parquet keys_expected=%s keys_received=%s path=%s",
                    cohort_name, age_band, item_type, keys_expected, keys_received[:40], str(parquet_file),
                )
                return {
                    'item_type': item_type,
                    'cohort_name': cohort_name,
                    'age_band': age_band,
                    'event_year': event_year,
                    'error': f"Column 'drug_name' not in parquet (has: {len(available_cols)} columns)"
                }
            query = f"""
            SELECT mi_person_key, drug_name as item, target, event_year
            FROM {model_data_from}
            WHERE
                drug_name IS NOT NULL
                AND drug_name != ''
                AND target = 1
                AND {event_filter}
            """
        elif item_type == 'icd_code':
            # Collect from ALL ICD diagnosis columns (primary through ten)
            query = f"""
            WITH all_icds AS (
                SELECT mi_person_key, primary_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE primary_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, two_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE two_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, three_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE three_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, four_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE four_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, five_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE five_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, six_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE six_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, seven_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE seven_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, eight_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE eight_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, nine_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE nine_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, ten_icd_diagnosis_code as icd, target, event_year
                FROM {model_data_from}
                WHERE ten_icd_diagnosis_code IS NOT NULL AND target = 1 AND {event_filter}
            )
            SELECT mi_person_key, icd as item, target, event_year FROM all_icds WHERE icd != ''
            """
        elif item_type == 'cpt_code':
            query = f"""
            SELECT mi_person_key, procedure_code as item, target, event_year
            FROM {model_data_from}
            WHERE
                procedure_code IS NOT NULL
                AND procedure_code != ''
                AND target = 1
                AND {event_filter}
            """
        elif item_type == 'medical_code':
            # Combined ICD (all 10 diagnosis positions) + CPT codes in a single transaction space
            query = f"""
            WITH all_med_codes AS (
                SELECT mi_person_key, primary_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, two_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE two_icd_diagnosis_code IS NOT NULL AND two_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, three_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE three_icd_diagnosis_code IS NOT NULL AND three_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, four_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE four_icd_diagnosis_code IS NOT NULL AND four_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, five_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE five_icd_diagnosis_code IS NOT NULL AND five_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, six_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE six_icd_diagnosis_code IS NOT NULL AND six_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, seven_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE seven_icd_diagnosis_code IS NOT NULL AND seven_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, eight_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE eight_icd_diagnosis_code IS NOT NULL AND eight_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, nine_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE nine_icd_diagnosis_code IS NOT NULL AND nine_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, ten_icd_diagnosis_code as code, target, event_year
                FROM {model_data_from}
                WHERE ten_icd_diagnosis_code IS NOT NULL AND ten_icd_diagnosis_code != '' AND target = 1 AND {event_filter}
                UNION ALL
                SELECT mi_person_key, procedure_code as code, target, event_year
                FROM {model_data_from}
                WHERE procedure_code IS NOT NULL AND procedure_code != '' AND target = 1 AND {event_filter}
            )
            SELECT mi_person_key, code as item, target, event_year FROM all_med_codes WHERE code != ''
            """
        else:
            raise ValueError(f"Unknown item_type: {item_type}")
        
        # Load data
        df = con.execute(query).df()
        log_memory(logger, "After data extraction")
        con.close()

        # Normalize target column so filtering works (Parquet/DuckDB may give int/float/object)
        if "target" in df.columns:
            target_orig = df["target"].dtype
            df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
            if target_orig != df["target"].dtype:
                logger.info("Coerced target column from %s to int; value_counts: %s", target_orig, df["target"].value_counts().to_dict())

        # Coerce event_year so TRAIN_YEARS comparison works (parquet may have int, string "2016", or "train").
        if "event_year" in df.columns and len(df) > 0:
            event_year_orig = df["event_year"].dtype
            # Map literal "train" to first training year so rows are kept and comparable to TRAIN_YEARS
            train_mask = df["event_year"].astype(str).str.strip().str.lower() == "train"
            if train_mask.any():
                df.loc[train_mask, "event_year"] = TRAIN_YEARS[0]
            df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
            if event_year_orig != df["event_year"].dtype:
                logger.info("Coerced event_year from %s to numeric for TRAIN_YEARS comparison", event_year_orig)
            df = df[df["event_year"].notna()].copy()
            df["event_year"] = df["event_year"].astype(int)
        
        if len(df) == 0:
            logger.warning("No %s data for %s", item_type, cohort_id)
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No data'
            }
        
        # Restrict to SHAP/FFA combined allowed codes (required; same file as BupaR/DTW)
        try:
            allowed = _load_allowed_codes_by_type(cohort_name, age_band, item_type, REPO_ROOT)
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': str(e),
            }
        item_upper = df["item"].astype(str).str.strip().str.upper().str.replace(".", "").str.replace("-", "")
        allowed_upper = {c.strip().upper().replace(".", "").replace("-", "") for c in allowed}
        before = len(df)
        df = df[item_upper.isin(allowed_upper)].copy()
        logger.info(f"Filtered to SHAP/FFA allowed items: {before:,} -> {len(df):,} rows ({len(allowed_upper)} codes)")
        if len(df) == 0:
            logger.warning("No rows left for %s after allowed-codes filter", cohort_id)
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No data after allowed-codes filter'
            }
        
        # Add item type prefix (like BupaR does: DRUG:, ICD:, CPT:) to ensure proper encoding
        item_prefix = {
            'drug_name': 'DRUG:',
            'icd_code': 'ICD:',
            'cpt_code': 'CPT:',
            'medical_code': 'MED:'
        }.get(item_type, '')
        
        if item_prefix:
            logger.info(f"Adding {item_prefix} prefix to items for proper encoding")
            df['item'] = item_prefix + df['item'].astype(str).str.strip()
        else:
            df['item'] = df['item'].astype(str).str.strip()
        
        # Filter out empty/invalid items
        before_filter = len(df)
        df = df[df['item'] != item_prefix].copy()  # Remove items that are just the prefix
        df = df[df['item'].notna()].copy()
        after_filter = len(df)
        if before_filter != after_filter:
            logger.info(f"Filtered out {before_filter - after_filter:,} empty/invalid items")
        
        if len(df) == 0:
            logger.warning("No valid items remaining after cleanup for %s", cohort_id)
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No valid items after cleanup'
            }

        n_persons = df["mi_person_key"].nunique()
        n_rows = len(df)
        logger.info(f"Target cohort size for {cohort_id} ({item_type}): {n_persons:,} persons, {n_rows:,} item rows")

        # For opioid_ed, only use events from the current event_year for rule mining
        if cohort_name == "opioid_ed":
            logger.info(f"Running FP-Growth for opioid_ed cohort for each year in TRAIN_YEARS: {TRAIN_YEARS}, aggregating at patient level (mi_person_key)")
            # Find patients present in any year, but keep linkage by mi_person_key
            patient_years = df[df['event_year'].isin(TRAIN_YEARS)][['mi_person_key', 'event_year']].drop_duplicates()
            patients_multi_year = patient_years['mi_person_key'].unique()
            logger.info(f"Total unique patients in TRAIN_YEARS: {len(patients_multi_year):,}")
            all_year_itemsets = []
            all_year_rules = []
            for year in TRAIN_YEARS:
                logger.info(f"Processing opioid_ed for year {year}")
                df_year = df[(df['event_year'] == year) & (df['mi_person_key'].isin(patients_multi_year))].copy()
                logger.info(f"Year {year}: {len(df_year):,} rows, {df_year['mi_person_key'].nunique():,} patients")
                if len(df_year) == 0:
                    logger.warning(f"No data for opioid_ed in year {year}")
                    continue
                # Assign Transaction_Density for this year
                df_year = assign_transaction_density(df_year, logger)
                for density in DENSITY_BINS:
                    transactions = get_transactions_by_density(df_year, density, logger)
                    if len(transactions) == 0:
                        continue
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                    density_support = min_support
                    if density == 'extreme':
                        density_support = max(min_support * 0.5, 0.01)
                    itemsets_density = fpgrowth(df_encoded, min_support=density_support, use_colnames=True)
                    itemsets_density = itemsets_density.sort_values('support', ascending=False).reset_index(drop=True)
                    itemsets_original = itemsets_density.copy()
                    if len(itemsets_density) > 0:
                        itemsets_density = filter_itemsets_by_lift(
                            itemsets_density,
                            df_encoded,
                            MIN_ITEMSET_LIFT,
                            logger
                        )
                    if len(itemsets_density) > 0:
                        n_single = sum(1 for _, r in itemsets_density.iterrows() if len(r["itemsets"]) == 1)
                        n_multi = len(itemsets_density) - n_single
                        # Attach year and patient info for aggregation
                        itemsets_density['event_year'] = year
                        all_year_itemsets.append(itemsets_density)
                        itemsets_for_rules = ensure_subsets_for_association_rules(itemsets_density, itemsets_original, logger)
                        rules_density = association_rules(itemsets_for_rules, metric="confidence", min_threshold=min_confidence)
                        rules_density = rules_density.sort_values("lift", ascending=False).reset_index(drop=True)
                        rules_density['event_year'] = year
                        all_year_rules.append(rules_density)
            # Combine all years' results, keeping patient linkage
            if len(all_year_itemsets) == 0:
                logger.warning("No frequent itemsets for opioid_ed in any TRAIN_YEARS")
                return {
                    'item_type': item_type,
                    'cohort_name': cohort_name,
                    'age_band': age_band,
                    'event_year': event_year,
                    'error': 'No frequent itemsets in any year'
                }
            itemsets = pd.concat(all_year_itemsets, ignore_index=True).drop_duplicates(subset=['itemsets', 'event_year']).sort_values(['event_year', 'support'], ascending=[True, False]).reset_index(drop=True)
            if len(all_year_rules) > 0:
                rules = pd.concat(all_year_rules, ignore_index=True).drop_duplicates(subset=['antecedents', 'consequents', 'event_year']).sort_values(['event_year', 'lift'], ascending=[True, False]).reset_index(drop=True)
            else:
                rules = pd.DataFrame()
            # Assign Transaction_Density to the combined df for downstream compatibility
            df = assign_transaction_density(df, logger)
            log_memory(logger, "After density assignment (all years, patient-linked)")
            # Replace all_itemsets/all_rules for downstream
            all_itemsets = [itemsets]
            all_rules = [rules]
            density_counts = {d: 0 for d in DENSITY_BINS}  # Not meaningful in this context
            logger.info(f"Combined opioid_ed itemsets: {len(itemsets):,}, rules: {len(rules):,} (patient-linked)")
        else:
            # non_opioid_ed (and any other cohort): assign density and process by density bins
            logger.info(f"Assigning Transaction_Density to {len(df):,} rows...")
            df = assign_transaction_density(df, logger)
            log_memory(logger, "After density assignment")

            all_itemsets = []
            all_rules = []
            density_counts = {}

            logger.info(f"Processing transactions by density level...")
            for density in DENSITY_BINS:
                log_memory_cpu(logger, f"Start density={density}")
                transactions = get_transactions_by_density(df, density, logger)
                density_counts[density] = len(transactions)

                if len(transactions) == 0:
                    logger.info("No %s density transactions - skipping", density)
                    continue

                try:
                    logger.info(f"Processing {density} density transactions (n={len(transactions):,})...")

                    # Encode transactions
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                    log_memory(logger, f"After encoding ({density})")

                    # Adjust support threshold based on density (lower support for extreme)
                    density_support = min_support
                    if density == 'extreme':
                        density_support = max(min_support * 0.5, 0.01)  # At least 1% support
                        logger.info(f"Using adjusted support threshold {density_support:.4f} for {density} density")

                    # Run FP-Growth
                    itemsets_density = fpgrowth(df_encoded, min_support=density_support, use_colnames=True)
                    itemsets_density = itemsets_density.sort_values('support', ascending=False).reset_index(drop=True)
                    itemsets_original = itemsets_density.copy()  # needed so association_rules can look up subset supports after lift filter

                    # Filter out common/trivial itemsets by lift BEFORE generating rules
                    if len(itemsets_density) > 0:
                        itemsets_density = filter_itemsets_by_lift(
                            itemsets_density,
                            df_encoded,
                            MIN_ITEMSET_LIFT,
                            logger
                        )
                        log_memory(logger, f"After filtering itemsets by lift ({density})")
                    
                    if len(itemsets_density) > 0:
                        # Diagnostic: association_rules requires itemsets of size >= 2
                        n_single = sum(1 for _, r in itemsets_density.iterrows() if len(r["itemsets"]) == 1)
                        n_multi = len(itemsets_density) - n_single
                        all_itemsets.append(itemsets_density)
                        log_memory(logger, f"After FP-Growth ({density})")
                        # association_rules needs every antecedent/consequent in the itemsets DataFrame; lift filter may have removed subsets
                        itemsets_for_rules = ensure_subsets_for_association_rules(itemsets_density, itemsets_original, logger)
                        rules_density = association_rules(itemsets_for_rules, metric="confidence", min_threshold=min_confidence)
                        rules_density = rules_density.sort_values("lift", ascending=False).reset_index(drop=True)
                        all_rules.append(rules_density)
                        logger.info(
                            f"  {density}: {len(itemsets_density):,} itemsets ({n_single} single-item, {n_multi} multi-item) → {len(rules_density):,} rules"
                        )
                        log_memory(logger, f"After rule generation ({density})")
                    else:
                        logger.warning(f"No itemsets remaining after lift filtering for {density} density")
                        continue
                        
                except MemoryError as e:
                    logger.error("Memory error processing %s density transactions: %s", density, e)
                    logger.warning(f"   Skipping {density} density transactions due to memory constraints")
                except Exception as e:
                    logger.error("Error processing %s density transactions: %s", density, e)
                    logger.warning(f"   Skipping {density} density transactions")

        # Combine results across density bins (low/medium/high/extreme). Data is target cohort only (no controls).
        if len(all_itemsets) == 0:
            logger.warning("No frequent itemsets for %s", cohort_id)
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No frequent itemsets'
            }
        
        # Combine itemsets (deduplicate if needed); use consistent columns to avoid concat FutureWarning
        non_empty_itemsets = [x for x in all_itemsets if x is not None and len(x) > 0]
        if non_empty_itemsets:
            cols = ['support', 'itemsets']
            to_concat = [df[cols] for df in non_empty_itemsets if set(cols).issubset(df.columns)]
            itemsets = pd.concat(to_concat, ignore_index=True) if to_concat else pd.DataFrame()
            if len(itemsets) > 0:
                itemsets = itemsets.drop_duplicates(subset=['itemsets'])
                itemsets = itemsets.sort_values('support', ascending=False).reset_index(drop=True)
        else:
            itemsets = pd.DataFrame()

        # Combine rules; use consistent columns to avoid concat FutureWarning
        non_empty_rules = [r for r in all_rules if r is not None and len(r) > 0]
        if non_empty_rules:
            rule_cols = list(non_empty_rules[0].columns)
            to_concat = [r.reindex(columns=rule_cols) for r in non_empty_rules]
            rules = pd.concat(to_concat, ignore_index=True)
            rules = rules.drop_duplicates(subset=['antecedents', 'consequents'])
            rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
        else:
            rules = pd.DataFrame()

        # Diagnostic when 0 rules: association_rules needs itemsets of size >= 2
        n_single_total = sum(1 for _, r in itemsets.iterrows() if len(r["itemsets"]) == 1)
        n_multi_total = len(itemsets) - n_single_total
        if len(rules) == 0:
            logger.warning(
                "0 rules for %s: %d single-item and %d multi-item itemsets at min_support=%.4f. "
                "association_rules requires 2+ item itemsets. Trying fallback with lower support (%.4f).",
                cohort_id, n_single_total, n_multi_total, min_support, MIN_SUPPORT_TARGET_ONLY,
            )
            # Fallback: run on all target transactions (no density split) with lower support
            tx_all = (
                df.groupby("mi_person_key")["item"]
                .apply(lambda x: sorted(set(x.tolist())))
                .tolist()
            )
            tx_all = [t for t in tx_all if len(t) > 0]
            if len(tx_all) >= 10:
                te_f = TransactionEncoder()
                te_ary_f = te_f.fit(tx_all).transform(tx_all)
                df_enc_f = pd.DataFrame(te_ary_f, columns=te_f.columns_)
                support_f = min(min_support, MIN_SUPPORT_TARGET_ONLY)
                itemsets_f = fpgrowth(df_enc_f, min_support=support_f, use_colnames=True)
                itemsets_f_original = itemsets_f.copy() if len(itemsets_f) > 0 else itemsets_f
                if len(itemsets_f) > 0:
                    itemsets_f = filter_itemsets_by_lift(itemsets_f, df_enc_f, MIN_ITEMSET_LIFT, logger)
                if len(itemsets_f) > 0:
                    itemsets_for_rules_f = ensure_subsets_for_association_rules(itemsets_f, itemsets_f_original, logger)
                    rules_f = association_rules(itemsets_for_rules_f, metric="confidence", min_threshold=min_confidence)
                    if len(rules_f) > 0:
                        rules_f = rules_f.sort_values("lift", ascending=False).reset_index(drop=True)
                        rules = rules_f
                        itemsets = itemsets_f.drop_duplicates(subset=["itemsets"]).sort_values("support", ascending=False).reset_index(drop=True)
                        logger.info(
                            "Fallback (all target transactions, support=%.4f): %d itemsets → %d rules",
                            support_f, len(itemsets_f), len(rules),
                        )
            else:
                logger.warning("Fallback skipped: insufficient transactions (%d) for %s", len(tx_all), cohort_id)

        # Business rule (all cohorts): exclude rules that appear in only one year (pattern must exist in 2 of 4 years)
        event_label = str(event_year)
        if event_label == "train" and "event_year" in df.columns and len(rules) > 0:
            rules = filter_rules_by_year_support(rules, df, MIN_YEARS_FOR_RULE, logger)

        # Create encoding map
        encoding_map = {}
        for idx, row in itemsets.iterrows():
            itemset = row['itemsets']
            if len(itemset) == 1:
                item = list(itemset)[0]
                encoding_map[item] = {
                    'support': float(row['support']),
                    'rank': int(idx)
                }
        
        # Save to S3
        s3_path = f"{s3_output_base}/{item_type}/cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}"
        
        # Convert frozensets to lists
        itemsets_json = itemsets.copy()
        itemsets_json['itemsets'] = itemsets_json['itemsets'].apply(lambda x: list(x))
        
        rules_json = rules.copy()
        if len(rules) > 0:
            rules_json['antecedents'] = rules_json['antecedents'].apply(lambda x: list(x))
            rules_json['consequents'] = rules_json['consequents'].apply(lambda x: list(x))
        
        # Upload to S3
        s3_client = boto3.client('s3')
        bucket = 'pgxdatalake'
        prefix = s3_path.replace('s3://pgxdatalake/', '')
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/encoding_map.json",
            Body=json.dumps(encoding_map, indent=2)
        )
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/itemsets.json",
            Body=itemsets_json.to_json(orient='records', indent=2)
        )
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/rules.json",
            Body=rules_json.to_json(orient='records', indent=2)
        )
        
        # Self-describing metadata: target-only, SHAP/FFA-gated; per FFA/SHAP these rules predict risk
        metrics = {
            'item_type': item_type,
            'cohort_name': cohort_name,
            'age_band': age_band,
            'event_year': event_year,
            'population': 'target_only',
            'feature_source': 'shap_ffa_allowed_codes',
            'purpose': 'shap_ffa_risk_cooccurrence',
            'density_binning': True,
            'train_years': list(TRAIN_YEARS) if str(event_year) == 'train' else None,
            'min_years_for_rule': MIN_YEARS_FOR_RULE if str(event_year) == 'train' else None,
            'density_bin_definitions': {
                'low': '<=P25 transaction size',
                'medium': 'P25-P50',
                'high': 'P50-P95',
                'extreme': '>P95',
            },
            'unique_items': len(df['item'].unique()),
            'total_transactions': sum(density_counts.values()),
            'density_distribution': density_counts,
            'frequent_itemsets': len(itemsets),
            'association_rules': len(rules),
            'encoding_map_size': len(encoding_map),
            'processing_time_seconds': time.time() - start_time
        }
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/metrics.json",
            Body=json.dumps(metrics, indent=2)
        )

        # ------------------------------------------------------------------
        # Save local copies: visualization artifacts under cohort/age_band only
        # Directory layout: outputs/{cohort_name}/{age_band_fname}/
        # ------------------------------------------------------------------
        try:
            age_band_fname = age_band.replace("-", "_")
            artifact_dir = LOCAL_OUTPUT_ROOT / cohort_name / age_band_fname
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # encoding_map
            (artifact_dir / f"{item_type}_encoding_map.json").write_text(
                json.dumps(encoding_map, indent=2)
            )

            # itemsets
            itemsets_json.to_json(
                artifact_dir / f"{item_type}_itemsets.json",
                orient="records",
                indent=2,
            )

            # rules
            rules_json.to_json(
                artifact_dir / f"{item_type}_rules.json",
                orient="records",
                indent=2,
            )

            # metrics
            (artifact_dir / f"{item_type}_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )

            logger.info(f"Saved local FP-Growth outputs (target cohort, combined across density bins) under {artifact_dir}")
        except Exception as e:
            logger.warning(f"Failed to write local FP-Growth outputs: {e}")

        # ------------------------------------------------------------------
        # Target-only outputs: same count as combined when extraction is target-only;
        # different count when extraction is target+control (we run separate FP-Growth on target=1).
        # ------------------------------------------------------------------
        try:
            ab_fname = age_band.replace("-", "_")
            artifact_dir = LOCAL_OUTPUT_ROOT / cohort_name / ab_fname
            itemsets_path = artifact_dir / f"{item_type}_itemsets.json"
            rules_path = artifact_dir / f"{item_type}_rules.json"
            itemsets_to_path = artifact_dir / f"{item_type}_itemsets_target_only.json"
            rules_to_path = artifact_dir / f"{item_type}_rules_target_only.json"

            # If we have both target and control in df, run separate target-only FP-Growth (different counts).
            has_control = "target" in df.columns and (df["target"] == 0).any()
            if has_control:
                logger.info("Target+control data: running separate target-only FP-Growth (target_only count will differ from combined)")
                target_mask = df["target"] == 1
                df_target = df.loc[target_mask].copy()
                tx_target = (
                    df_target.groupby("mi_person_key")["item"]
                    .apply(lambda x: sorted(set(x.tolist())))
                    .tolist()
                )
                if len(tx_target) < 10:
                    logger.warning("Insufficient target-only transactions (%d); skipping target-only outputs", len(tx_target))
                else:
                    te_t = TransactionEncoder()
                    te_ary_t = te_t.fit(tx_target).transform(tx_target)
                    df_enc_t = pd.DataFrame(te_ary_t, columns=te_t.columns_)
                    support_t = min(min_support, MIN_SUPPORT_TARGET_ONLY)
                    itemsets_t = fpgrowth(df_enc_t, min_support=support_t, use_colnames=True)
                    itemsets_t_original = itemsets_t.copy() if len(itemsets_t) > 0 else itemsets_t
                    if len(itemsets_t) > 0:
                        itemsets_t = filter_itemsets_by_lift(itemsets_t, df_enc_t, MIN_ITEMSET_LIFT, logger)
                    try:
                        if len(itemsets_t) > 0:
                            itemsets_for_rules_t = ensure_subsets_for_association_rules(itemsets_t, itemsets_t_original, logger)
                            rules_t = association_rules(itemsets_for_rules_t, metric="confidence", min_threshold=min_confidence)
                        else:
                            rules_t = pd.DataFrame(columns=["antecedents", "consequents"])
                    except Exception as e_rules:
                        logger.warning("Target-only association_rules failed: %s", e_rules)
                        rules_t = pd.DataFrame(columns=["antecedents", "consequents"])
                    itemsets_t_json = itemsets_t.copy()
                    if "itemsets" in itemsets_t_json.columns:
                        itemsets_t_json["itemsets"] = itemsets_t_json["itemsets"].apply(lambda x: list(x))
                    rules_t_json = rules_t.copy()
                    if "antecedents" in rules_t_json.columns:
                        rules_t_json["antecedents"] = rules_t_json["antecedents"].apply(lambda x: list(x))
                    if "consequents" in rules_t_json.columns:
                        rules_t_json["consequents"] = rules_t_json["consequents"].apply(lambda x: list(x))
                    itemsets_t_json.to_json(itemsets_to_path, orient="records", indent=2)
                    rules_t_json.to_json(rules_to_path, orient="records", indent=2)
                    s3_client.put_object(Bucket=bucket, Key=f"{prefix}/itemsets_target_only.json", Body=itemsets_t_json.to_json(orient="records", indent=2).encode("utf-8"))
                    s3_client.put_object(Bucket=bucket, Key=f"{prefix}/rules_target_only.json", Body=rules_t_json.to_json(orient="records", indent=2).encode("utf-8"))
                    logger.info("Saved target-only FP-Growth (target=1 only): %d itemsets, %d rules", len(itemsets_t), len(rules_t))
            elif itemsets_path.exists() and rules_path.exists():
                # Target-only extraction: combined already is target cohort → same count
                shutil.copy2(itemsets_path, itemsets_to_path)
                shutil.copy2(rules_path, rules_to_path)
                itemsets_body = itemsets_path.read_bytes()
                rules_body = rules_path.read_bytes()
                s3_client.put_object(Bucket=bucket, Key=f"{prefix}/itemsets_target_only.json", Body=itemsets_body)
                s3_client.put_object(Bucket=bucket, Key=f"{prefix}/rules_target_only.json", Body=rules_body)
                logger.info("Target-only = combined (same count): copied itemsets and rules to *_target_only.json")
            else:
                logger.warning("Target-only: combined files not found; skipping copy to *_target_only.json")
        except Exception as e:
            logger.warning("Target-only copy encountered an error: %s", e)

        elapsed = time.time() - start_time
        log_memory(logger, "END")
        logger.info("[OK] %s %s: %d itemsets, %d rules in %.1fs", cohort_id, item_type, len(itemsets), len(rules), elapsed)

        return metrics
        
    except Exception as e:
        logger.error("Failed %s %s: %s", cohort_id, item_type, e)
        return {
            'item_type': item_type,
            'cohort_name': cohort_name,
            'age_band': age_band,
            'event_year': event_year,
            'error': str(e)
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger = setup_logger()
    
    logger.info("="*80)
    logger.info("COHORT-SPECIFIC FPGROWTH FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Min Support: {MIN_SUPPORT}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE}")
    logger.info(f"Min Itemset Lift: {MIN_ITEMSET_LIFT} (filtering common/trivial itemsets)")
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info("Item types by cohort: %s", {c: get_item_types_for_cohort(c) for c in COHORT_NAMES})
    logger.info(f"S3 Output: {S3_OUTPUT_BASE}")
    logger.info(f"Local Data: {LOCAL_DATA_PATH}")
    logger.info(f"Local Data Exists: {LOCAL_DATA_PATH.exists()}")
    logger.info(f"Model Data Fallback: 3b → $PGX_DATA_ROOT → /mnt/nvme/4_model_data → 4_model_data → 4a_model_data")
    logger.info("="*80)
    
    # Check if at least one data source exists (either raw cohorts or model_data)
    model_data_exists = _model_data_path("opioid_ed", "13_24") is not None  # Test with one cohort
    if not LOCAL_DATA_PATH.exists() and not model_data_exists:
        logger.error("Local data path does not exist: %s", LOCAL_DATA_PATH)
        logger.error("Model data not found in any fallback location (3b, $PGX_DATA_ROOT, /mnt/nvme/, 4_model_data, 4a_model_data)")
        logger.error(
            "  On EC2, sync from S3 with:\n"
            "    aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ /mnt/nvme/cohorts/"
        )
        logger.error(
            "  For local development, sync to ./data/cohorts_F1120/ and "
            "either set LOCAL_DATA_PATH accordingly or export LOCAL_DATA_PATH, "
            "or generate filtered model_data first."
        )
        sys.exit(1)
    
    # Generate all cohort combinations (item types depend on cohort)
    cohort_jobs = []
    for cohort_name in COHORT_NAMES:
        item_types = get_item_types_for_cohort(cohort_name)
        for item_type in item_types:
            for age_band in AGE_BANDS:
                # Per-year jobs (2016–2020, etc.)
                for event_year in EVENT_YEARS:
                    cohort_jobs.append((item_type, cohort_name, age_band, event_year))
                # Aggregated TRAIN window (2016–2019), label as 'train' for dashboard visuals
                cohort_jobs.append((item_type, cohort_name, age_band, "train"))
    
    # Apply DRY_RUN limit if enabled
    if DRY_RUN and len(cohort_jobs) > DRY_RUN_LIMIT:
        logger.info("DRY RUN: Limiting from %d to %d cohort combinations", len(cohort_jobs), DRY_RUN_LIMIT)
        cohort_jobs = cohort_jobs[:DRY_RUN_LIMIT]
    
    total_jobs = len(cohort_jobs)
    logger.info(f"Total cohort jobs: {total_jobs}")
    if DRY_RUN:
        logger.info(f"DRY RUN MODE: Processing only {DRY_RUN_LIMIT} combinations (set DRY_RUN = False for full run)")
    else:
        logger.info(f"FULL RUN MODE: Processing all cohorts")
    logger.info("="*80)
    
    # Process cohorts in parallel
    all_metrics = []
    completed = 0
    failed = 0
    
    overall_start = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(
                process_single_cohort,
                item_type, cohort_name, age_band, event_year,
                LOCAL_DATA_PATH, S3_OUTPUT_BASE,
                MIN_SUPPORT, MIN_CONFIDENCE
            ): (item_type, cohort_name, age_band, event_year)
            for item_type, cohort_name, age_band, event_year in cohort_jobs
        }
        
        # Process results as they complete
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                metrics = future.result()
                all_metrics.append(metrics)
                completed += 1
                
                if 'error' in metrics:
                    failed += 1
                
                # Progress update every 10 completions
                if completed % 10 == 0:
                    elapsed = time.time() - overall_start
                    rate = completed / elapsed
                    remaining = (total_jobs - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{total_jobs} ({completed/total_jobs*100:.1f}%) - "
                              f"ETA: {remaining/60:.1f} min")
            except Exception as e:
                logger.error("Job %s raised exception: %s", job, e)
                failed += 1
    
    # Final summary
    total_time = time.time() - overall_start
    successful = completed - failed
    
    logger.info("\n" + "="*80)
    logger.info("COHORT ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Runtime: {total_time/60:.1f} minutes")
    logger.info(f"Total Jobs: {total_jobs}")
    logger.info(f"Successful: {successful} ({successful/total_jobs*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total_jobs*100:.1f}%)")
    logger.info("="*80)
    
    # Summary by item type (only types that were run)
    item_types_seen = sorted(set(m['item_type'] for m in all_metrics))
    for item_type in item_types_seen:
        item_metrics = [m for m in all_metrics if m['item_type'] == item_type and 'error' not in m]
        if item_metrics:
            total_itemsets = sum(m.get('frequent_itemsets', 0) for m in item_metrics)
            total_rules = sum(m.get('association_rules', 0) for m in item_metrics)
            logger.info("  %s: %d cohorts, %s itemsets, %s rules", item_type, len(item_metrics), f"{total_itemsets:,}", f"{total_rules:,}")
    
    # EC2 Auto-Shutdown (optional)
    shutdown_ec2(logger)


def shutdown_ec2(logger: logging.Logger, enable: bool = False):
    """
    Automatically shutdown EC2 instance after analysis completes.
    
    Args:
        logger: Logger instance
        enable: Set to True to enable auto-shutdown, False to skip
    """
    if not enable:
        logger.info("\n" + "="*80)
        logger.info("EC2 Auto-Shutdown: DISABLED")
        logger.info("="*80)
        logger.info("To enable auto-shutdown, set enable=True in shutdown_ec2() call")
        logger.info("Instance will continue running.")
        logger.info("\nTo manually stop this instance later:")
        logger.info("  aws ec2 stop-instances --instance-ids $(ec2-metadata --instance-id | cut -d ' ' -f 2)")
        logger.info("Or use AWS Console: EC2 > Instances > Select instance > Instance State > Stop")
        return
    
    logger.info("\n" + "="*80)
    logger.info("Shutting down EC2 instance...")
    logger.info("="*80)
    
    import subprocess
    import requests
    import shutil
    
    # Get instance ID from EC2 metadata service
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            timeout=2
        )
        if response.status_code == 200:
            instance_id = response.text.strip()
            logger.info(f"Instance ID: {instance_id}")
            
            # Find AWS CLI
            aws_cmd = shutil.which("aws")
            if not aws_cmd:
                # Try common paths
                for path in ["/usr/local/bin/aws", "/usr/bin/aws", 
                           "/home/ec2-user/.local/bin/aws", 
                           "/home/ubuntu/.local/bin/aws"]:
                    if Path(path).exists():
                        aws_cmd = path
                        break
            
            if aws_cmd:
                # Stop the instance (use terminate-instances for permanent deletion)
                shutdown_cmd = [aws_cmd, "ec2", "stop-instances", "--instance-ids", instance_id]
                
                logger.info(f"Running: {' '.join(shutdown_cmd)}")
                result = subprocess.run(shutdown_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("[OK] EC2 instance stop command sent successfully")
                    logger.info("Instance will stop in a few moments.")
                    logger.info("Note: This is a STOP (not terminate), so you can restart it later.")
                    if result.stdout:
                        logger.info(f"\nAWS Response:\n{result.stdout}")
                else:
                    logger.error("EC2 stop command failed with exit code %s", result.returncode)
                    if result.stderr:
                        logger.error(f"Error: {result.stderr}")
                    logger.error("Check AWS credentials and IAM permissions.")
            else:
                logger.error("AWS CLI not found. Cannot shutdown instance.")
                logger.error("Install AWS CLI or ensure it's in your PATH.")
                logger.error("Manual shutdown: aws ec2 stop-instances --instance-ids " + instance_id)
        else:
            logger.error("Metadata service returned status code %s", response.status_code)
            logger.error("Could not retrieve instance ID.")
    
    except requests.exceptions.RequestException as e:
        logger.error("Could not retrieve instance ID from metadata service.")
        logger.error(f"Error: {e}")
        logger.error("If running on EC2, check that metadata service is accessible.")
        logger.error("\nManual shutdown command:")
        logger.error("  aws ec2 stop-instances --instance-ids <your-instance-id>")
    
    except Exception as e:
        logger.error("Unexpected error during shutdown: %s", e)


if __name__ == "__main__":
    main()


