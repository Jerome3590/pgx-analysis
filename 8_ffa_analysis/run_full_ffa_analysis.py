#!/usr/bin/env python3
"""
Complete FFA Analysis Workflow

This script runs the complete Formal Feature Attribution (FFA) analysis workflow:
1. Load models (CatBoost, XGBoost, XGBoost RF)
2. Extract rules using unified schema
3. Generate AXP explanations
4. Calculate feature importance
5. Perform causal analysis
6. Generate visualizations and reports
"""

import sys
import json
import logging
import time
import ast
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from itertools import combinations
from collections import Counter, defaultdict
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging (under 8_ffa_analysis)
LOG_DIR = PROJECT_ROOT / "8_ffa_analysis" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f'ffa_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {LOG_FILE}")

# Import explainers
try:
    from catboost_axp_explainer import CatBoostSymbolicExplainer, PathConfig
    CATBOOST_EXPLAINER_AVAILABLE = True
except ImportError:
    CATBOOST_EXPLAINER_AVAILABLE = False

try:
    from xgboost_axp_explainer import XGBoostSymbolicExplainer
    XGBOOST_EXPLAINER_AVAILABLE = True
except ImportError:
    XGBOOST_EXPLAINER_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration (defaults; can be overridden via CLI)
COHORT_NAME = "opioid_ed"
AGE_BAND = "13-24"
AGE_BAND_FNAME = AGE_BAND.replace("-", "_")

# Paths (updated to use 6_final_model outputs). These are initialized with
# defaults but can be recomputed in main() after parsing CLI arguments.
MODEL_JSON_BASE = (
    PROJECT_ROOT
    / "6_final_model"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / "final_model_json"
)
# Try Parquet first (preferred), fall back to CSV
DATA_PATH_PARQUET = (
    PROJECT_ROOT
    / "6_final_model"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / "inputs"
    / "model_train"
    / "final_features.parquet"
)
DATA_PATH_CSV = (
    PROJECT_ROOT
    / "6_final_model"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
)
# Use Parquet if available, otherwise CSV
DATA_PATH = DATA_PATH_PARQUET if DATA_PATH_PARQUET.exists() else DATA_PATH_CSV
OUTPUT_DIR = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis configuration
from py_helpers.env_utils import get_sklearn_n_jobs  # noqa: E402

ANALYSIS_CONFIG = {
    'target_class': 1,
    'top_k_features': 40,
    'min_coverage': 0.8,
    'n_permutations': 100,  # Reduced for faster execution
    'random_seed': 1997,
    'binary_intervention_mode': 'remove_only',  # remove_only | add_only | flip
    'max_samples': 10000,  # Limit data samples to prevent OOM
    'max_explanation_samples': 1000,  # Limit number of instances for explanation generation
    # Limit parallel workers to reduce memory usage (use 1-4 instead of all CPUs)
    'n_jobs': min(4, max(1, get_sklearn_n_jobs())),
    'batch_size': 100,  # Process explanations in batches
    # Stage 2.5: Univariate causal pruning
    'min_present_support': 10,  # Minimum # instances with feature=1 for removal mode
    'min_absent_support': 10,   # Minimum # instances with feature=0 for addition mode
    'min_axp_coverage': 0.01,   # Minimum AXP coverage (1% of explanations)
    'min_shap_for_causal': 0.0, # Minimum SHAP importance for causal testing
    'min_ffa_for_causal': 0.0,  # Minimum FFA importance for causal testing
    # Multi-feature interaction analysis
    'enable_interaction_analysis': True,  # Set to True to enable multi-feature interaction testing
    'max_interaction_size': 3,  # Maximum number of features to test together (2 = pairs, 3 = triplets, etc.)
    'interaction_top_k': 50,  # Top K features to consider for interactions
    'interaction_sample_size': 50,  # Sample size for interaction testing (reduced from 100)
    # Stage 3: Interaction pruning
    'min_cooccur_support': 5,   # Minimum co-occurrence for pairs
    'min_cooccur_support_triplet': 3,  # Minimum co-occurrence for triplets
    'max_combinations_per_size': 1000,  # Cap on combinations per size
    # Stage 4: Runtime pruning
    'early_stopping_n': 10,     # Check first N instances for early stopping
    'enable_early_stopping': True,  # Enable early stopping for zero changes
    'min_interaction_effect': 0.01,  # Minimum interaction effect to report
    'causal_sample_size': 50,  # Sample size for causal analysis (reduced from 100)
    'causal_checkpoint_interval': 10,  # Save progress every N features for idempotency
    'min_combined_shap_threshold': 0.0,  # Minimum combined SHAP score for feature combinations (0 = no filtering)
    'min_individual_shap_threshold': 0.0,  # Minimum individual SHAP score per feature in combination (0 = only filter features with SHAP > 0, which is automatic)
    # Excluded features (non-predictive markers/confounders)
    'excluded_features': [
        'item_drug_SUBOXONE',  # Treatment medication - marker, not predictive
        'item_drug_BUPRENORPHINE_HCL',  # Treatment medication - marker, not predictive
        'item_drug_BUPRENORPHINE_HCL_NALOXON',  # Treatment medication - marker, not predictive
        'item_icd_F1123',  # Opioid dependence ICD code - marker, not predictive
    ],
}


def load_model_json(model_json_path: Path) -> Dict[str, Any]:
    """Load model JSON file."""
    logger.info(f"Loading model JSON from: {model_json_path}")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_json_path.name}")
    print(f"{'='*80}\n")
    
    if not model_json_path.exists():
        logger.error(f"Model file not found: {model_json_path}")
        raise FileNotFoundError(f"Model file not found: {model_json_path}")
    
    logger.info(f"Reading JSON file (size: {model_json_path.stat().st_size / 1024 / 1024:.2f} MB)...")
    with open(model_json_path, 'r') as f:
        model_json = json.load(f)
    logger.info(f"JSON file loaded successfully in {time.time() - start_time:.2f} seconds")
    
    model_type = model_json.get('model_type', 'unknown')
    # Heuristics to infer model type when not explicitly stored
    if model_type == 'unknown':
        if 'oblivious_trees' in model_json or 'non_oblivious_trees' in model_json:
            model_type = 'catboost'
        elif 'learner' in model_json or 'gradient_booster' in model_json:
            # XGBoost JSON produced by Booster.save_model(...)
            model_type = 'xgboost'
    
    # Normalize XGBoost variant names: "xgb" -> "xgboost", "xgb_rf" -> "xgboost_rf"
    if model_type == 'xgb':
        model_type = 'xgboost'
    elif model_type == 'xgb_rf':
        model_type = 'xgboost_rf'
    
    # Persist normalized type back into JSON for downstream functions
    model_json['model_type'] = model_type
    
    logger.info(f"Model type detected: {model_type}")
    
    if 'trees' in model_json:
        logger.info(f"Found {len(model_json['trees'])} trees")
    if 'oblivious_trees' in model_json:
        logger.info(f"Found {len(model_json['oblivious_trees'])} oblivious trees")
    if 'non_oblivious_trees' in model_json:
        logger.info(f"Found {len(model_json['non_oblivious_trees'])} non-oblivious trees")
    if 'feature_names' in model_json:
        logger.info(f"Found {len(model_json['feature_names'])} features")
    
    print(f"[OK] Model loaded: {model_type}")
    if 'trees' in model_json:
        print(f"  - Trees: {len(model_json['trees'])}")
    if 'oblivious_trees' in model_json:
        print(f"  - Oblivious trees: {len(model_json['oblivious_trees'])}")
    if 'feature_names' in model_json:
        print(f"  - Features: {len(model_json['feature_names'])}")
    
    logger.info(f"Model loading completed in {time.time() - start_time:.2f} seconds")
    return model_json


def extract_feature_mappings(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract feature name mappings from model JSON."""
    logger.info("Extracting feature mappings from model JSON...")
    start_time = time.time()
    
    model_type = model_json.get('model_type', 'unknown')
    
    if model_type == 'unknown' and ('oblivious_trees' in model_json or 'non_oblivious_trees' in model_json):
        model_type = 'catboost'
    
    # Normalize XGBoost variant names: "xgb" -> "xgboost", "xgb_rf" -> "xgboost_rf"
    if model_type == 'xgb':
        model_type = 'xgboost'
    elif model_type == 'xgb_rf':
        model_type = 'xgboost_rf'
    
    # Update model_json with normalized type for consistency
    model_json['model_type'] = model_type
    
    if model_type in ['catboost', 'CatBoost']:
        logger.info("Processing CatBoost feature mappings...")
        features_info = model_json.get('features_info', {})
        float_features = features_info.get('float_features', [])
        logger.info(f"Found {len(float_features)} float features")
        feature_names = {
            f["flat_feature_index"]: f["feature_id"]
            for f in float_features
        }
    else:
        # XGBoost
        logger.info("Processing XGBoost feature mappings...")
        if "feature_names" in model_json:
            feature_names = {
                i: name for i, name in enumerate(model_json["feature_names"])
            }
            logger.info(f"Found {len(feature_names)} feature names")
        else:
            logger.warning("No feature_names found in model JSON")
            feature_names = {}
    
    logger.info(f"Feature mapping extraction completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Extracted {len(feature_names)} feature mappings")
    
    return {
        'model_type': model_type,  # Return normalized model_type
        'feature_names': feature_names
    }


def load_data(data_path: Path, max_samples: Optional[int] = None) -> tuple:
    """Load data for analysis using DuckDB for efficient Parquet/CSV reading."""
    logger.info(f"Loading data from: {data_path}")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Loading data from: {data_path.name}")
    print(f"{'='*80}\n")
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    file_size_mb = data_path.stat().st_size / 1024 / 1024
    is_parquet = data_path.suffix.lower() == '.parquet'
    
    if is_parquet:
        logger.info(f"Reading Parquet file (size: {file_size_mb:.2f} MB) using DuckDB...")
        # Use DuckDB for efficient Parquet reading
        try:
            import duckdb
            con = duckdb.connect()
            
            if max_samples:
                # Load full dataset first to get accurate sampling
                data = con.execute(f"SELECT * FROM read_parquet('{data_path}')").df()
                if len(data) > max_samples:
                    data = data.sample(n=max_samples, random_state=ANALYSIS_CONFIG['random_seed']).reset_index(drop=True)
            else:
                data = con.execute(f"SELECT * FROM read_parquet('{data_path}')").df()
            con.close()
            
            logger.info(f"Parquet loaded: {len(data)} rows, {len(data.columns)} columns in {time.time() - start_time:.2f} seconds")
        except ImportError:
            logger.warning("DuckDB not available, falling back to pandas.read_parquet")
            data = pd.read_parquet(data_path)
            if max_samples and len(data) > max_samples:
                data = data.sample(n=max_samples, random_state=ANALYSIS_CONFIG['random_seed']).reset_index(drop=True)
            logger.info(f"Parquet loaded: {len(data)} rows, {len(data.columns)} columns in {time.time() - start_time:.2f} seconds")
    else:
        logger.info(f"Reading CSV file (size: {file_size_mb:.2f} MB) using DuckDB...")
        # Use DuckDB for efficient CSV reading (much faster than pandas)
        try:
            import duckdb
            con = duckdb.connect()
            
            if max_samples:
                # Load full dataset first to get accurate sampling
                data = con.execute(f"SELECT * FROM read_csv_auto('{data_path}')").df()
                if len(data) > max_samples:
                    data = data.sample(n=max_samples, random_state=ANALYSIS_CONFIG['random_seed']).reset_index(drop=True)
            else:
                data = con.execute(f"SELECT * FROM read_csv_auto('{data_path}')").df()
            con.close()
            
            logger.info(f"CSV loaded: {len(data)} rows, {len(data.columns)} columns in {time.time() - start_time:.2f} seconds")
        except ImportError:
            logger.warning("DuckDB not available, falling back to pandas.read_csv")
            # Fallback to pandas CSV reading with chunking for large files
            if file_size_mb > 500:  # If file is > 500MB, use chunking
                logger.info(f"Large file detected ({file_size_mb:.2f} MB), using chunked reading...")
                chunks = []
                chunk_size = 10000
                for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    if max_samples and len(pd.concat(chunks, ignore_index=True)) >= max_samples:
                        break
                data = pd.concat(chunks, ignore_index=True)
                del chunks
                import gc
                gc.collect()
            else:
                data = pd.read_csv(data_path)
            
            logger.info(f"CSV loaded: {len(data)} rows, {len(data.columns)} columns in {time.time() - start_time:.2f} seconds")
    
    # Apply sampling if needed (only if not already sampled above)
    if max_samples and len(data) > max_samples:
        logger.info(f"Sampling {max_samples} rows from {len(data)} total rows")
        data = data.sample(n=max_samples, random_state=ANALYSIS_CONFIG['random_seed']).reset_index(drop=True)
        logger.info(f"Sampled data: {len(data)} rows")
    
    # Separate features and target
    target_cols = ['target', 'is_target_case']
    target_col = None
    for col in target_cols:
        if col in data.columns:
            target_col = col
            break
    
    # Preserve a stable instance identifier for SHAP alignment and downstream tracing
    # (important if we sample/reset indices elsewhere).
    if 'instance_index' not in data.columns:
        data = data.copy()
        data.insert(0, 'instance_index', data.index.astype(int))
    
    if target_col:
        logger.info(f"Found target column: {target_col}")
        y = data[target_col].values
        X = data.drop(target_col, axis=1)
        target_dist = Counter(y)
        logger.info(f"Target distribution: {dict(target_dist)}")
        print(f"[OK] Data loaded: {len(X)} samples, {len(X.columns)} features")
        print(f"  - Target distribution: {target_dist}")
    else:
        logger.warning("No target column found. Using all columns as features.")
        print("[WARNING] No target column found. Using all columns as features.")
        X = data
        y = None
    
    logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    return X, y


def load_shap_importance(cohort: str, age_band: str, model_type: str) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    """
    Load SHAP importance scores from Step 7 outputs.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        model_type: Model type ('xgboost' or 'catboost')
        
    Returns:
        Tuple of:
        - Dict mapping feature_name -> mean_abs_shap (only features with importance > 0)
        - Optional DataFrame with individual SHAP values per instance (indexed by instance index)
        
    Raises:
        FileNotFoundError: If SHAP importance file is not found
        ValueError: If SHAP file is invalid or empty
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try to load SHAP global importance CSV
    shap_path = (
        PROJECT_ROOT
        / "7_shap_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_shap_global_importance_{model_type}.csv"
    )
    
    # Also try S3 if local doesn't exist
    if not shap_path.exists():
        try:
            import boto3
            s3_client = boto3.client("s3")
            s3_key = f"gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_{model_type}.csv"
            try:
                s3_client.head_object(Bucket="pgxdatalake", Key=s3_key)
                # Download temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                s3_client.download_file("pgxdatalake", s3_key, tmp_path)
                shap_path = Path(tmp_path)
            except Exception as e:
                # S3 download failed, will raise FileNotFoundError below
                pass
        except ImportError:
            # boto3 not available, will raise FileNotFoundError below
            pass
    
    if not shap_path.exists():
        raise FileNotFoundError(
            f"SHAP importance file not found for {cohort}/{age_band} ({model_type}). "
            f"Checked locations:\n"
            f"  - Local: {PROJECT_ROOT / '7_shap_analysis' / 'outputs' / cohort / age_band_fname / f'{cohort}_{age_band_fname}_shap_global_importance_{model_type}.csv'}\n"
            f"  - S3: s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_{model_type}.csv\n"
            f"SHAP values are required. Please run Step 7 (SHAP Analysis) first."
        )
    
    try:
        shap_df = pd.read_csv(shap_path)
        if 'feature' not in shap_df.columns or 'mean_abs_shap' not in shap_df.columns:
            raise ValueError(
                f"SHAP file missing required columns. Expected 'feature' and 'mean_abs_shap', "
                f"got: {list(shap_df.columns)}"
            )
        
        # Filter to features with importance > 0
        shap_df = shap_df[shap_df['mean_abs_shap'] > 0]
        
        if len(shap_df) == 0:
            raise ValueError(
                f"SHAP file contains no features with importance > 0. "
                f"All features have zero importance."
            )
        
        # Create mapping: feature_name -> mean_abs_shap
        shap_map = dict(zip(shap_df['feature'], shap_df['mean_abs_shap'], strict=True))
        logger.info(f"Loaded SHAP importance for {len(shap_map)} features (importance > 0)")
        
        # Try to load individual SHAP values per instance (from parquet file)
        shap_values_df = None
        shap_values_path = (
            PROJECT_ROOT
            / "7_shap_analysis"
            / "outputs"
            / cohort
            / age_band_fname
            / f"{cohort}_{age_band_fname}_shap_sample_values_{model_type}.parquet"
        )
        
        # Also try S3 if local doesn't exist
        if not shap_values_path.exists():
            try:
                import boto3
                s3_client = boto3.client("s3")
                s3_key = f"gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_{model_type}.parquet"
                try:
                    s3_client.head_object(Bucket="pgxdatalake", Key=s3_key)
                    # Download temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    s3_client.download_file("pgxdatalake", s3_key, tmp_path)
                    shap_values_path = Path(tmp_path)
                except Exception:
                    shap_values_path = None
            except ImportError:
                shap_values_path = None
        
        if shap_values_path and shap_values_path.exists():
            try:
                # Use DuckDB for efficient Parquet access (all processing in DuckDB, pandas only at final step)
                try:
                    from shap_parquet_loader import ShapParquetLoader
                    
                    # Create loader - uses DuckDB internally, doesn't load data yet
                    shap_loader = ShapParquetLoader(shap_values_path)
                    logger.info(f"SHAP Parquet file: {shap_loader.num_rows} rows, {shap_loader.num_columns} columns")
                    
                    # All processing uses DuckDB, only convert to pandas at final step for compatibility
                    # The loader uses DuckDB queries throughout, only converting when to_pandas() is called
                    shap_values_df = shap_loader.to_pandas()  # Only converts at final step
                    shap_loader.close()
                    
                    logger.info(f"Loaded individual SHAP values via DuckDB: {len(shap_values_df)} instances, {len(shap_values_df.columns)} features")
                    # If SHAP parquet contains an explicit instance identifier, use it as index for safe alignment
                    if 'instance_index' in shap_values_df.columns:
                        shap_values_df = shap_values_df.set_index('instance_index', drop=True)
                        # Ensure integer index if possible
                        try:
                            shap_values_df.index = shap_values_df.index.astype(int)
                        except Exception:
                            pass
                except ImportError:
                    # Fallback to pandas if DuckDB not available
                    logger.warning("DuckDB not available, falling back to pandas.read_parquet")
                    shap_values_df = pd.read_parquet(shap_values_path)
                    logger.info(f"Loaded individual SHAP values via pandas: {len(shap_values_df)} instances, {len(shap_values_df.columns)} features")
                    # If SHAP parquet contains an explicit instance identifier, use it as index for safe alignment
                    if 'instance_index' in shap_values_df.columns:
                        shap_values_df = shap_values_df.set_index('instance_index', drop=True)
                        # Ensure integer index if possible
                        try:
                            shap_values_df.index = shap_values_df.index.astype(int)
                        except Exception:
                            pass
                
                # Ensure index is set properly (should be instance indices)
                if isinstance(shap_values_df, pd.DataFrame) and shap_values_df.index.name is None and shap_values_df.index.dtype == 'int64':
                    shap_values_df.index.name = 'instance_index'
            except Exception as e:
                logger.warning(f"Could not load individual SHAP values: {e}. Using global SHAP importance only.")
                shap_values_df = None
        else:
            logger.info("Individual SHAP values not found. Using global SHAP importance only.")
        
        return shap_map, shap_values_df
    except (FileNotFoundError, ValueError) as e:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading SHAP importance: {e}") from e


def initialize_explainer(
    model_json_path: Path,
    model_json: Dict[str, Any],
    feature_mappings: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    shap_importance_map: Dict[str, float] = None,
    shap_values_df: Optional[pd.DataFrame] = None,
) -> Optional[Any]:
    """Initialize FFA explainer for the model."""
    if not shap_importance_map:
        raise ValueError("shap_importance_map is required. Only rules with SHAP importance > 0 will be used.")
    
    logger.info("Initializing FFA Explainer...")
    start_time = time.time()
    
    model_type = feature_mappings.get('model_type', model_json.get('model_type', 'unknown'))
    
    if model_type == 'unknown' and ('oblivious_trees' in model_json or 'non_oblivious_trees' in model_json):
        model_type = 'catboost'
    
    # Normalize XGBoost variant names: "xgb" -> "xgboost", "xgb_rf" -> "xgboost_rf"
    if model_type == 'xgb':
        model_type = 'xgboost'
    elif model_type == 'xgb_rf':
        model_type = 'xgboost_rf'
    
    # Log normalized model type (before the print statement that shows the error)
    logger.info(f"Model type (normalized): {model_type}")
    
    print(f"\n{'='*80}")
    print(f"Initializing FFA Explainer (Model Type: {model_type})")
    print(f"{'='*80}\n")
    
    try:
        logger.info("Creating PathConfig...")
        path_config = PathConfig(
            model_path=str(model_json_path),
            data_dir=str(DATA_PATH.parent),
            output_dir=str(OUTPUT_DIR),
            tree_rules_path=None,
            age_band=AGE_BAND
        )
        
        if model_type in ['catboost', 'CatBoost']:
            if not CATBOOST_EXPLAINER_AVAILABLE:
                logger.error("CatBoost explainer not available")
                print("[WARNING] CatBoost explainer not available.")
                return None
            logger.info("Creating CatBoostSymbolicExplainer...")
            explainer = CatBoostSymbolicExplainer(path_config, shap_importance_map=shap_importance_map)
            explainer.model_json = model_json
            logger.info("Calling fit_from_model_json (this may take a while)...")
            fit_start = time.time()
            explainer.fit_from_model_json(model_json)
            logger.info(f"fit_from_model_json completed in {time.time() - fit_start:.2f} seconds")
            
        elif model_type in ['xgboost', 'xgboost_rf', 'XGBoost']:
            if not XGBOOST_EXPLAINER_AVAILABLE:
                logger.error("XGBoost explainer not available")
                print("[WARNING] XGBoost explainer not available.")
                return None
            logger.info("Creating XGBoostSymbolicExplainer...")
            explainer = XGBoostSymbolicExplainer(path_config, shap_importance_map=shap_importance_map,
                                                 shap_values_df=shap_values_df)
            explainer.logger.setLevel(logging.INFO)
            explainer.model_json = model_json

            # If feature names are known from the DataFrame, provide them here so the
            # explainer does not have to infer them from the JSON structure.
            if feature_names:
                explainer.feature_names = {
                    i: name for i, name in enumerate(feature_names)
                }

            logger.info("Calling fit_from_model_json (this may take a while)...")
            fit_start = time.time()
            explainer.fit_from_model_json(model_json)
            logger.info(
                "fit_from_model_json completed in %.2f seconds",
                time.time() - fit_start,
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            print(f"[WARNING] Unknown model type: {model_type}.")
            return None
        
        num_rules = len(explainer.rule_clauses)
        class_1_rules = sum(1 for p in explainer.rule_predictions if p == 1)
        logger.info(f"Explainer initialized: {num_rules} rules created, {class_1_rules} predict class 1")
        
        print("[OK] FFA Explainer initialized")
        print(f"  - Rules created: {num_rules}")
        print(f"  - Rules predicting class 1: {class_1_rules}")
        
        logger.info(f"Explainer initialization completed in {time.time() - start_time:.2f} seconds")
        return explainer
        
    except Exception as e:
        logger.error(f"Could not initialize FFA Explainer: {e}", exc_info=True)
        print(f"[ERROR] Could not initialize FFA Explainer: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_explanations(explainer: Any, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Generate AXP explanations for the dataset."""
    logger.info("Generating AXP explanations...")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("Generating Anchored Explanations (AXP)")
    print(f"{'='*80}\n")
    
    # Preserve instance_index if present (for SHAP alignment), but exclude from explainer input
    instance_index_col = None
    if 'instance_index' in X.columns:
        instance_index_col = X['instance_index'].copy()
        X_for_explainer = X.drop('instance_index', axis=1).copy()
        logger.debug("Excluded instance_index from explainer input (preserved for SHAP alignment)")
    else:
        X_for_explainer = X.copy()
    
    # Filter to target class
    logger.info(f"Filtering to target class {ANALYSIS_CONFIG['target_class']}...")
    mask = (y == ANALYSIS_CONFIG['target_class'])
    X_class = X_for_explainer[mask].reset_index(drop=True)
    y_class = y[mask]
    
    # Preserve instance_index for filtered subset
    if instance_index_col is not None:
        instance_index_class = instance_index_col[mask].reset_index(drop=True)
    else:
        instance_index_class = None
    
    # Limit samples for testing if configured (set to None to use all)
    max_exp_samples = ANALYSIS_CONFIG.get('max_explanation_samples')
    if max_exp_samples and len(X_class) > max_exp_samples:
        logger.info(f"Limiting to {max_exp_samples} instances for testing (out of {len(X_class)} total)")
        X_class = X_class.head(max_exp_samples).reset_index(drop=True)
        y_class = y_class[:max_exp_samples]
    else:
        logger.info(f"Processing all {len(X_class)} instances")
    
    logger.info(f"Filtered to {len(X_class)} instances of class {ANALYSIS_CONFIG['target_class']}")
    print(f"  - Class {ANALYSIS_CONFIG['target_class']} instances: {len(X_class)}")
    
    try:
        # Use configured number of jobs (limited to reduce memory usage)
        n_jobs = ANALYSIS_CONFIG.get('n_jobs', 2)
        batch_size = ANALYSIS_CONFIG.get('batch_size', 100)
        
        logger.info(f"Calling explain_dataset on {len(X_class)} instances (this may take a while)...")
        logger.info(f"  - Total rules: {len(explainer.rule_clauses) if hasattr(explainer, 'rule_clauses') else 'unknown'}")
        logger.info(f"  - Rules for class {ANALYSIS_CONFIG['target_class']}: {sum(1 for p in explainer.rule_predictions if p == ANALYSIS_CONFIG['target_class']) if hasattr(explainer, 'rule_predictions') else 'unknown'}")
        logger.info(f"  - Using {n_jobs} parallel workers (limited for memory efficiency)")
        logger.info(f"  - Processing in batches of {batch_size}")
        
        explain_start = time.time()
        
        # Process in batches to reduce memory usage
        all_axps = []
        n_batches = (len(X_class) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_class))
            X_batch = X_class.iloc[start_idx:end_idx].reset_index(drop=True)
            y_batch = y_class[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx+1}/{n_batches} (instances {start_idx}-{end_idx-1})...")
            batch_start = time.time()
            
            # Ensure instance_index is not passed to explainer (should already be excluded, but double-check)
            X_batch_clean = X_batch.drop('instance_index', axis=1) if 'instance_index' in X_batch.columns else X_batch
            
            df_batch = explainer.explain_dataset(
                X_batch_clean,
                predictions=y_batch,
                return_df=True,
                show_progress=False,  # Disable progress for batches
                n_jobs=n_jobs
            )
            
            # Re-attach instance_index to results if available
            if instance_index_class is not None:
                batch_indices = instance_index_class.iloc[start_idx:end_idx].values
                df_batch['instance_index'] = batch_indices
            
            all_axps.append(df_batch)
            batch_time = time.time() - batch_start
            logger.info(f"Batch {batch_idx+1} completed in {batch_time:.2f} seconds ({batch_time/len(X_batch):.4f} seconds per instance)")
            
            # Explicit cleanup
            del df_batch, X_batch, y_batch
            import gc
            gc.collect()
        
        # Combine all batches
        df_axps = pd.concat(all_axps, ignore_index=True)
        del all_axps
        import gc
        gc.collect()
        explain_duration = time.time() - explain_start
        logger.info(f"explain_dataset completed in {explain_duration:.2f} seconds ({explain_duration/len(X_class):.4f} seconds per instance)")
        
        print(f"[OK] Generated {len(df_axps)} explanations")
        
        # Check explanation quality
        if len(df_axps) > 0:
            non_empty = sum(1 for axp in df_axps['axp'] if axp and len(axp) > 0)
            logger.info(f"Explanation quality: {non_empty} / {len(df_axps)} have conditions")
            print(f"  - Explanations with conditions: {non_empty} / {len(df_axps)}")
            
            # Sample explanation
            if non_empty > 0:
                sample_axp = next(axp for axp in df_axps['axp'] if axp and len(axp) > 0)
                logger.info(f"Sample AXP (first 3): {sample_axp[:3]}")
                print(f"  - Sample AXP (first 3 conditions): {sample_axp[:3]}")
        
        logger.info(f"Explanation generation completed in {time.time() - start_time:.2f} seconds")
        return df_axps
        
    except Exception as e:
        logger.error(f"Error generating explanations: {e}", exc_info=True)
        print(f"[ERROR] Error generating explanations: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def calculate_feature_importance(df_axps: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature importance from AXP explanations."""
    logger.info("Calculating feature importance from AXP explanations...")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("Calculating Feature Importance from AXP")
    print(f"{'='*80}\n")
    
    valid_axps = df_axps["axp"].dropna()
    total_explanations = len(valid_axps)
    logger.info(f"Processing {total_explanations} valid explanations")
    
    if total_explanations == 0:
        logger.warning("No valid explanations found")
        print("[WARNING] No valid explanations found.")
        return pd.DataFrame(columns=['feature', 'count', 'importance', 'coverage'])
    
    # Extract features from explanations
    logger.info("Extracting features from explanations...")
    all_features = []
    feature_to_instances = defaultdict(set)
    parse_errors = 0
    
    for idx, axp in enumerate(valid_axps):
        try:
            if isinstance(axp, str):
                # Parse string representation using ast.literal_eval for safety
                if axp.startswith('['):
                    try:
                        parsed = ast.literal_eval(axp)
                    except (ValueError, SyntaxError):
                        parsed = [axp]
                else:
                    parsed = [axp]
            else:
                parsed = axp if isinstance(axp, list) else []
            
            if isinstance(parsed, list):
                for condition in parsed:
                    if isinstance(condition, str):
                        # Extract feature name (first word before space)
                        feature = condition.split()[0]
                        all_features.append(feature)
                        feature_to_instances[feature].add(idx)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:  # Log first 5 errors
                logger.debug(f"Parse error for explanation {idx}: {e}")
            continue
    
    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parse errors while extracting features")
    
    logger.info(f"Extracted {len(all_features)} feature occurrences from {len(feature_to_instances)} unique features")
    
    if not all_features:
        logger.warning("No features extracted from explanations")
        print("[WARNING] No features extracted from explanations.")
        return pd.DataFrame(columns=['feature', 'count', 'importance', 'coverage'])
    
    # Calculate metrics
    logger.info("Calculating feature importance metrics...")
    feature_counts = Counter(all_features)
    
    # Build importance dataframe with all features (filter > 0 later)
    importance_df = pd.DataFrame([
        {
            'feature': feat,
            'count': count,
            'importance': count / total_explanations,
            'coverage': len(feature_to_instances[feat]) / total_explanations
        }
        for feat, count in feature_counts.items()
    ])
    
    # Filter to features with importance > 0 and sort by importance descending
    importance_df = importance_df[importance_df['importance'] > 0].sort_values('importance', ascending=False)
    
    logger.info(f"Feature importance calculated for {len(importance_df)} features (importance > 0)")
    print(f"[OK] Feature importance calculated for {len(importance_df)} features (importance > 0)")
    print(f"\nTop features (importance > 0):")
    print(importance_df.head(50).to_string(index=False))
    
    logger.info(f"Feature importance calculation completed in {time.time() - start_time:.2f} seconds")
    return importance_df


def prune_features_for_causal_analysis(
    available_features: List[str],
    X_class: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    shap_map: Optional[Dict[str, float]] = None,
    binary_intervention_mode: str = 'remove_only'
) -> List[str]:
    """
    Stage 2.5: Primary Feature Pruning Gate
    
    Apply pruning rules before univariate causal analysis:
    1. Prevalence filter (binary features)
    2. AXP coverage filter
    3. Importance-union filter (SHAP OR FFA)
    
    Args:
        available_features: List of candidate features
        X_class: Feature matrix (filtered to target class)
        feature_importance_df: AXP-based feature importance
        shap_map: SHAP importance scores (optional)
        binary_intervention_mode: Binary intervention mode (remove_only/add_only/flip)
    
    Returns:
        Pruned list of features
    """
    if not available_features:
        return []
    
    pruned_features = []
    n_samples = len(X_class)
    
    # Build importance maps
    ffa_map = {}
    if not feature_importance_df.empty:
        ffa_map = dict(zip(
            feature_importance_df['feature'],
            feature_importance_df['importance']
        ))
    
    # Get coverage map
    coverage_map = {}
    if not feature_importance_df.empty and 'coverage' in feature_importance_df.columns:
        coverage_map = dict(zip(
            feature_importance_df['feature'],
            feature_importance_df['coverage']
        ))
    
    # Get configuration
    min_present_support = ANALYSIS_CONFIG.get('min_present_support', 10)
    min_absent_support = ANALYSIS_CONFIG.get('min_absent_support', 10)
    min_axp_coverage = ANALYSIS_CONFIG.get('min_axp_coverage', 0.01)
    min_shap = ANALYSIS_CONFIG.get('min_shap_for_causal', 0.0)
    min_ffa = ANALYSIS_CONFIG.get('min_ffa_for_causal', 0.0)
    
    # Scale support thresholds with sample size
    if n_samples > 0:
        min_present_support = max(5, int(min_present_support * (n_samples / 100)))
        min_absent_support = max(5, int(min_absent_support * (n_samples / 100)))
    
    # Get excluded features list
    excluded_features = set(ANALYSIS_CONFIG.get('excluded_features', []))
    
    logger.info(f"Pruning features: {len(available_features)} candidates")
    logger.info(f"  Prevalence thresholds: present={min_present_support}, absent={min_absent_support}")
    logger.info(f"  AXP coverage threshold: {min_axp_coverage}")
    logger.info(f"  Importance thresholds: SHAP>={min_shap}, FFA>={min_ffa}")
    if excluded_features:
        logger.info(f"  Excluded features: {', '.join(sorted(excluded_features))}")
    
    for feat_name in available_features:
        if feat_name not in X_class.columns:
            continue
        
        # Rule 0: Excluded features (non-predictive markers/confounders)
        if feat_name in excluded_features:
            logger.debug(f"  Pruned {feat_name}: excluded (non-predictive marker/confounder)")
            continue
        
        # Rule 1: Feature relevance (already filtered by get_model_features_for_causal_analysis)
        # Skip if not in data
        if feat_name not in X_class.columns:
            continue
        
        # Rule 2: Prevalence filter (for binary features)
        unique_vals = X_class[feat_name].unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})
        
        if is_binary:
            if binary_intervention_mode == 'remove_only':
                support = int((X_class[feat_name] == 1).sum())
                if support < min_present_support:
                    logger.debug(f"  Pruned {feat_name}: insufficient present support ({support} < {min_present_support})")
                    continue
            elif binary_intervention_mode == 'add_only':
                support = int((X_class[feat_name] == 0).sum())
                if support < min_absent_support:
                    logger.debug(f"  Pruned {feat_name}: insufficient absent support ({support} < {min_absent_support})")
                    continue
            # For 'flip' mode, no prevalence filter
        
        # Rule 3: AXP coverage filter
        coverage = coverage_map.get(feat_name, 0.0)
        if coverage < min_axp_coverage:
            logger.debug(f"  Pruned {feat_name}: insufficient AXP coverage ({coverage:.4f} < {min_axp_coverage})")
            continue
        
        # Rule 4: Importance-union filter (SHAP OR FFA)
        shap_importance = shap_map.get(feat_name, 0.0) if shap_map else 0.0
        ffa_importance = ffa_map.get(feat_name, 0.0)
        
        if shap_importance < min_shap and ffa_importance < min_ffa:
            logger.debug(f"  Pruned {feat_name}: insufficient importance (SHAP={shap_importance:.4f}, FFA={ffa_importance:.4f})")
            continue
        
        # Feature passed all pruning rules
        pruned_features.append(feat_name)
    
    logger.info(f"Pruning complete: {len(pruned_features)}/{len(available_features)} features retained")
    return pruned_features


def get_model_features_for_causal_analysis(X: pd.DataFrame) -> List[str]:
    """
    Get the features that the model actually uses for causal analysis.
    
    The model uses:
    - Binary features (item_* indicators) for aggregated feature importance codes
    - PGx features (pgx_*)
    - n_events (simple count)
    
    These correspond to aggregated feature importance codes from Step 3, represented
    as binary indicators (1 if patient has code, 0 otherwise).
    
    Returns:
        List of feature names that the model uses (binary FI codes + PGx).
    """
    target_features = set()
    
    # Add all binary features (item_* indicators for aggregated FI codes)
    # These are created from aggregated feature importance codes from Step 3
    for col in X.columns:
        if col.startswith('item_'):
            target_features.add(col)
    
    # Add PGx features
    pgx_features = [col for col in X.columns if col.startswith('pgx_')]
    target_features.update(pgx_features)
    
    # Also include n_events (simple aggregation used by model)
    if 'n_events' in X.columns:
        target_features.add('n_events')
    
    available_features = list(target_features)
    
    n_binary = len([c for c in available_features if c.startswith('item_')])
    n_pgx = len(pgx_features)
    n_other = len(available_features) - n_binary - n_pgx
    
    logger.info(f"Selected {len(available_features)} features for causal analysis "
                f"({n_binary} binary FI codes + {n_pgx} PGx + {n_other} other)")
    
    return available_features




def _calculate_grouped_causal_effect(
    explainer: Any, 
    X_original: pd.DataFrame, 
    X_modified: pd.DataFrame, 
    y: np.ndarray,
    feat_name: str,
    original_indices_mapping: Optional[List[int]] = None,
    is_binary: bool = False
) -> float:
    """
    Calculate causal effect by grouping instances by matching rules.
    
    Instead of computing AXP for each row individually, we:
    1. Group instances by their matching rules (before modification)
    2. Compute AXP once per group
    3. Identify which groups are affected by feature modification
    4. Only recompute AXP for affected groups
    
    This is much more efficient when many instances have the same rule matches.
    
    Args:
        explainer: FFA explainer instance
        X_original: Original feature matrix
        X_modified: Modified feature matrix (with feature intervention)
        y: Predicted classes
        feat_name: Name of feature being modified
        
    Returns:
        Fraction of instances where explanations changed
    """
    from collections import defaultdict
    
    # Step 1: Group instances by matching rules (original)
    original_groups = defaultdict(list)  # group_key -> list of instance indices
    original_group_axps = {}  # group_key -> AXP (list of literals)
    
    logger.debug(f"  Grouping {len(X_original)} instances by matching rules...")
    
    for idx in range(len(X_original)):
        instance = X_original.iloc[idx].values if isinstance(X_original, pd.DataFrame) else X_original[idx]
        predicted_class = y[idx]
        
        # Get matching rules for this instance
        matched_rules = explainer._satisfied_rules(instance, predicted_class)
        
        # Create group key from sorted rule IDs (instances with same rules get same key)
        group_key = tuple(sorted(matched_rules)) if matched_rules else tuple()
        
        original_groups[group_key].append(idx)
    
    logger.debug(f"  Found {len(original_groups)} unique rule groups")
    
    # Step 2: Compute AXP once per group (original)
    for group_key, instance_indices in original_groups.items():
        if group_key not in original_group_axps:
            if group_key:  # Has matching rules
                # Use first instance in group to compute AXP
                first_idx = instance_indices[0]
                instance = X_original.iloc[first_idx].values if isinstance(X_original, pd.DataFrame) else X_original[first_idx]
                predicted_class = y[first_idx]
                
                # Get instance-specific SHAP for rule filtering
                # IMPORTANT: use a stable instance identifier when available (instance_index column),
                # because DataFrame positional indices can be reset during sampling/filtering.
                instance_shap_values: Dict[str, float] = {}
                shap_lookup_idx = first_idx
                try:
                    if isinstance(X_original, pd.DataFrame) and 'instance_index' in X_original.columns:
                        shap_lookup_idx = int(X_original.iloc[first_idx]['instance_index'])
                except Exception:
                    # Fallback: try original_indices_mapping if available
                    if original_indices_mapping is not None and first_idx < len(original_indices_mapping):
                        shap_lookup_idx = original_indices_mapping[first_idx]
                    else:
                        shap_lookup_idx = first_idx
                
                if hasattr(explainer, 'shap_values_df') and explainer.shap_values_df is not None:
                    if shap_lookup_idx in explainer.shap_values_df.index:
                        instance_shap_row = explainer.shap_values_df.loc[shap_lookup_idx]
                        instance_shap_values = instance_shap_row.to_dict()
                    elif first_idx in explainer.shap_values_df.index:
                        # Fallback: try filtered index
                        instance_shap_row = explainer.shap_values_df.loc[first_idx]
                        instance_shap_values = instance_shap_row.to_dict()
                    elif len(explainer.shap_values_df) > first_idx:
                        # Fallback: positional access
                        instance_shap_row = explainer.shap_values_df.iloc[first_idx]
                        instance_shap_values = instance_shap_row.to_dict()
                
                # Compute AXP for this group
                # Note: _compute_axp uses global SHAP importance map, not instance-specific
                # Instance-specific SHAP is used in the worker function for rule scoring
                # For grouped comparison, we use global SHAP map (faster, slight approximation)
                try:
                    axp_literals = explainer._compute_axp(list(group_key))
                    original_group_axps[group_key] = tuple(sorted(axp_literals))
                except Exception as e:
                    logger.debug(f"  Error computing AXP for group {group_key[:5]}...: {e}")
                    original_group_axps[group_key] = tuple()
            else:
                original_group_axps[group_key] = tuple()  # No matching rules
    
    # Step 3: Identify affected groups and compute modified AXP (with caching)
    modified_group_axps = {}  # Cache modified AXP per group key
    total_changes = 0
    total_instances = len(X_original)
    
    for idx in range(len(X_original)):
        instance_orig = X_original.iloc[idx].values if isinstance(X_original, pd.DataFrame) else X_original[idx]
        instance_mod = X_modified.iloc[idx].values if isinstance(X_modified, pd.DataFrame) else X_modified[idx]
        predicted_class = y[idx]
        
        # Get original group key and AXP
        matched_orig = explainer._satisfied_rules(instance_orig, predicted_class)
        orig_group_key = tuple(sorted(matched_orig)) if matched_orig else tuple()
        original_axp = original_group_axps.get(orig_group_key, tuple())
        
        # Get modified matching rules and group key
        matched_mod = explainer._satisfied_rules(instance_mod, predicted_class)
        mod_group_key = tuple(sorted(matched_mod)) if matched_mod else tuple()
        
        # Get or compute modified AXP (with caching)
        if mod_group_key in modified_group_axps:
            # Use cached modified AXP
            modified_axp = modified_group_axps[mod_group_key]
        else:
            # Need to compute modified AXP (cache it for other instances with same modified rules)
            # IMPORTANT: Even if rules don't change (mod_group_key == orig_group_key), we still
            # recompute AXP because the feature intervention might change which features appear
            # in the AXP even if the same rules match. This fixes the conservative approximation
            # that was causing all binary features (drugs/ICDs) to have 0.0 causal importance.
            if mod_group_key:
                try:
                    axp_literals = explainer._compute_axp(matched_mod)
                    modified_axp = tuple(sorted(axp_literals))
                    modified_group_axps[mod_group_key] = modified_axp
                except Exception as e:
                    logger.debug(f"  Error computing modified AXP for group {mod_group_key[:5]}...: {e}")
                    modified_axp = tuple()
                    modified_group_axps[mod_group_key] = modified_axp
            else:
                modified_axp = tuple()
                modified_group_axps[mod_group_key] = modified_axp
        
        # Compare AXP
        # For binary features, also check if the feature appears in the original AXP
        # If it does and we're removing it, that's a change even if the AXP computation is the same
        feature_appears_in_axp = False
        if is_binary and feat_name and hasattr(explainer, 'id_condition_map') and hasattr(explainer, 'feature_names'):
            # Check if feature appears in original AXP literals
            # For binary features, if the feature appears in the AXP at all, removing it (1->0) should count as a change
            # because it changes the explanation composition, even if the condition still holds
            for lit in original_axp:
                try:
                    feat_idx, thresh, direction = explainer.id_condition_map[lit]
                    axp_feat_name = explainer.feature_names.get(feat_idx, None)
                    # Check if this literal corresponds to the feature we're modifying
                    if axp_feat_name == feat_name:
                        # Feature appears in AXP - for binary features, this means it's part of the explanation
                        # Removing it (1->0) changes the explanation, so count it as a change
                        feature_appears_in_axp = True
                        break
                except (KeyError, IndexError, ValueError):
                    continue
        
        # Count as change if:
        # 1. AXP literals changed (different minimal hitting set)
        # 2. Feature appears in original AXP and we're removing it (for binary features)
        if original_axp != modified_axp or feature_appears_in_axp:
            total_changes += 1
            if feature_appears_in_axp and original_axp == modified_axp:
                logger.debug(f"    Instance {idx}: Feature {feat_name} appears in AXP, removal counts as change")
    
    change_rate = total_changes / total_instances if total_instances > 0 else 0.0
    return change_rate


def perform_causal_analysis(explainer: Any, X: pd.DataFrame, y: np.ndarray, 
                           feature_importance_df: pd.DataFrame, cohort: str, age_band: str,
                           model_type: str = "xgboost", output_dir: Optional[Path] = None,
                           shap_map: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Perform causal analysis by measuring prediction changes.
    
    Only analyzes aggregated feature importance features (drug/ICD/CPT codes) plus PGx features.
    """
    logger.info("Performing causal analysis...")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("Performing Causal Analysis")
    print(f"{'='*80}\n")
    
    # Filter to target class
    logger.info(f"Filtering to target class {ANALYSIS_CONFIG['target_class']}...")
    mask = (y == ANALYSIS_CONFIG['target_class'])
    X_class = X[mask].reset_index(drop=True)
    y_class = y[mask]
    
    # Get features that the model actually uses (engineered features + PGx)
    # These correspond to aggregated feature importance codes from Step 3
    # The model uses mean_*/max_* engineered features created from drug/ICD/CPT codes
    available_features = get_model_features_for_causal_analysis(X_class)
    
    if not available_features:
        logger.warning("No model features found. Falling back to FFA importance features.")
        # Fallback: use features with importance > 0 from FFA importance
        # Filter to features with importance > 0
        top_features = feature_importance_df[feature_importance_df['importance'] > 0]['feature'].tolist()
        available_features = [f for f in top_features if f in X_class.columns]
    
    # Stage 2.5: Apply primary pruning gate
    # Apply pruning rules (SHAP map passed as parameter)
    available_features = prune_features_for_causal_analysis(
        available_features,
        X_class,
        feature_importance_df,
        shap_map=shap_map,
        binary_intervention_mode=ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')
    )
    
    # Sort by combined importance (SHAP + FFA)
    if shap_map and not feature_importance_df.empty:
        ffa_importance_map = dict(zip(
            feature_importance_df['feature'],
            feature_importance_df['importance']
        ))
        available_features = sorted(
            available_features,
            key=lambda f: (shap_map.get(f, 0.0) + ffa_importance_map.get(f, 0.0)),
            reverse=True
        )
    elif not feature_importance_df.empty:
        ffa_importance_map = dict(zip(
            feature_importance_df['feature'],
            feature_importance_df['importance']
        ))
        available_features = sorted(
            available_features,
            key=lambda f: ffa_importance_map.get(f, 0.0),
            reverse=True
        )
    
    logger.info(f"Found {len(available_features)} features available in data for causal analysis")
    
    if not available_features:
        logger.warning("No matching features found for causal analysis")
        print("[WARNING] No matching features found for causal analysis.")
        return pd.DataFrame()
    
    # Check for existing causal results for idempotency
    existing_causal_df = pd.DataFrame()
    processed_features = set()
    causal_checkpoint_path = None
    
    if output_dir is not None:
        model_output_dir = output_dir / model_type
        causal_checkpoint_path = model_output_dir / 'causal_importance.parquet'
        
        if causal_checkpoint_path.exists():
            try:
                logger.info(f"Found existing causal results at {causal_checkpoint_path}. Loading for idempotency...")
                existing_causal_df = pd.read_parquet(causal_checkpoint_path)
                processed_features = set(existing_causal_df['feature'].tolist())
                logger.info(f"Loaded {len(processed_features)} already-processed features. Will skip these and resume.")
                print(f"[INFO] Found {len(processed_features)} already-processed features. Resuming from checkpoint...")
            except Exception as e:
                logger.warning(f"Failed to load existing causal results: {e}. Will start fresh.")
                existing_causal_df = pd.DataFrame()
                processed_features = set()
    
    # Filter out already-processed features
    remaining_features = [f for f in available_features if f not in processed_features]
    
    if len(processed_features) > 0:
        logger.info(f"Skipping {len(processed_features)} already-processed features. {len(remaining_features)} features remaining.")
        print(f"[INFO] Skipping {len(processed_features)} already-processed features. {len(remaining_features)} features remaining.")
    
    if not remaining_features:
        logger.info("All features already processed. Returning existing results.")
        print("[INFO] All features already processed. Returning existing results.")
        return existing_causal_df
    
    print(f"Analyzing {len(remaining_features)} features ({len(processed_features)} already processed)...")
    
    # Start with existing scores
    causal_scores = existing_causal_df.to_dict('records') if not existing_causal_df.empty else []
    analysis_start = time.time()
    checkpoint_interval = ANALYSIS_CONFIG.get('causal_checkpoint_interval', 10)  # Save every N features
    
    logger.info(f"Starting causal analysis for {len(remaining_features)} features (checkpoint every {checkpoint_interval} features, no time limit)")
    print(f"[INFO] Processing {len(remaining_features)} features (no time limit, will checkpoint every {checkpoint_interval} features)")
    
    for feat_idx, feat_name in enumerate(tqdm(remaining_features, desc="Causal analysis")):
        try:
            logger.info(f"Analyzing feature {feat_idx+1}/{len(remaining_features)}: {feat_name}")
            feat_start = time.time()
            
            # Use smaller sample for causal analysis to reduce memory
            causal_sample_size = min(ANALYSIS_CONFIG.get('causal_sample_size', 50), len(X_class))
            X_sample = X_class.head(causal_sample_size).copy()
            y_sample = y_class[:causal_sample_size]
            
            # Preserve original indices for SHAP alignment
            # After reset_index(drop=True), indices are 0, 1, 2, ... but SHAP values may use original indices
            # Store mapping: filtered_index -> original_index for SHAP lookup
            original_indices = X_sample.index.tolist() if hasattr(X_sample.index, 'tolist') else list(range(len(X_sample)))
            
            # Detect if feature is binary (0/1 only)
            unique_vals = X_sample[feat_name].unique()
            is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})
            
            # Calculate median for reference (even for binary features)
            median_val = X_sample[feat_name].median()
            
            # Create modified dataset with appropriate intervention
            X_modified = X_sample.copy()
            if is_binary:
                # For binary features: configurable intervention semantics
                mode = ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')
                
                if mode == 'remove_only':
                    # Test only instances where feature is present (1) and remove it (1->0)
                    test_mask = X_sample[feat_name] == 1
                    num_test = int(test_mask.sum())
                    if num_test == 0:
                        logger.warning(
                            f"  Feature {feat_name}: No instances with value=1 (n1=0), skipping (cannot measure removal effect)"
                        )
                        continue
                    X_modified.loc[test_mask, feat_name] = 0
                    intervention_val = f"removed (1->0, {num_test}/{len(X_sample)} instances)"
                    X_sample_filtered = X_sample[test_mask].copy()
                    X_modified_filtered = X_modified[test_mask].copy()
                    y_sample_filtered = y_sample[test_mask]
                    
                    # Sanity checks
                    assert (X_sample_filtered[feat_name] == 1).all(), \
                        f"Sanity check failed: All filtered instances should have {feat_name} == 1"
                    assert (X_modified_filtered[feat_name] == 0).all(), \
                        f"Sanity check failed: All modified instances should have {feat_name} == 0"
                    
                elif mode == 'add_only':
                    # Test only instances where feature is absent (0) and add it (0->1)
                    test_mask = X_sample[feat_name] == 0
                    num_test = int(test_mask.sum())
                    if num_test == 0:
                        logger.warning(
                            f"  Feature {feat_name}: No instances with value=0 (n0=0), skipping (cannot measure addition effect)"
                        )
                        continue
                    X_modified.loc[test_mask, feat_name] = 1
                    intervention_val = f"added (0->1, {num_test}/{len(X_sample)} instances)"
                    X_sample_filtered = X_sample[test_mask].copy()
                    X_modified_filtered = X_modified[test_mask].copy()
                    y_sample_filtered = y_sample[test_mask]
                    
                    # Sanity checks
                    assert (X_sample_filtered[feat_name] == 0).all(), \
                        f"Sanity check failed: All filtered instances should have {feat_name} == 0"
                    assert (X_modified_filtered[feat_name] == 1).all(), \
                        f"Sanity check failed: All modified instances should have {feat_name} == 1"
                    
                elif mode == 'flip':
                    # Flip all instances (0<->1)
                    test_mask = np.ones(len(X_sample), dtype=bool)
                    num_test = len(X_sample)
                    X_modified[feat_name] = 1 - X_sample[feat_name]
                    intervention_val = f"flipped (0<->1, {num_test}/{len(X_sample)} instances)"
                    X_sample_filtered = X_sample.copy()
                    X_modified_filtered = X_modified.copy()
                    y_sample_filtered = y_sample.copy()
                    
                else:
                    raise ValueError(f"Unknown binary_intervention_mode: {mode}")
                
                # Preserve original indices for SHAP alignment
                # Map filtered positions back to original indices
                if mode != 'flip':
                    if isinstance(test_mask, pd.Series):
                        filtered_original_indices = [original_indices[i] for i in range(len(X_sample)) if test_mask.iloc[i]]
                    else:
                        filtered_original_indices = [original_indices[i] for i in range(len(X_sample)) if test_mask[i]]
                else:
                    filtered_original_indices = original_indices
                    
            else:
                # For continuous features: set to median
                X_modified[feat_name] = median_val
                intervention_val = f"median ({median_val:.4f})"
                X_sample_filtered = X_sample
                X_modified_filtered = X_modified
                y_sample_filtered = y_sample
                # For continuous features, no filtering, so use original indices as-is
                filtered_original_indices = original_indices
            
            # Calculate change rate using grouped comparison (avoids row-by-row explain_dataset calls)
            # Group instances by matching rules and only compute AXP once per group
            # For binary features, we only test instances where feature was present (normalized by |S_f|, not N)
            effective_sample_size = len(X_sample_filtered)
            
            # Sanity check: For binary features, changes can never exceed num_present
            if is_binary:
                max_possible_changes = effective_sample_size
                logger.debug(f"  Sanity check: max_possible_changes = {max_possible_changes} (all {effective_sample_size} instances with feature=1)")
            
            logger.info(f"  [{feat_idx+1}/{len(remaining_features)}] Analyzing {feat_name} using grouped rule comparison (effective sample size: {effective_sample_size})...")
            try:
                # Pass original indices mapping for SHAP alignment and binary flag
                change_rate = _calculate_grouped_causal_effect(
                    explainer, X_sample_filtered, X_modified_filtered, y_sample_filtered, feat_name,
                    original_indices_mapping=filtered_original_indices,
                    is_binary=is_binary
                )
                changes = int(change_rate * effective_sample_size)
                
                # Sanity check: For binary features, changes should not exceed effective_sample_size
                if is_binary and changes > effective_sample_size:
                    logger.warning(f"  Sanity check failed: changes ({changes}) > effective_sample_size ({effective_sample_size})")
                
                logger.info(f"  Feature {feat_name}: {changes}/{effective_sample_size} explanations changed ({change_rate:.2%}) [grouped, normalized by |S_f|={effective_sample_size}]")
            except Exception as e:
                logger.warning(f"  Grouped comparison failed for {feat_name}: {e}. Falling back to row-by-row.")
                # Fallback to original row-by-row comparison
                logger.info(f"  Generating original explanations for {feat_name}...")
                orig_start = time.time()
                try:
                    original_explanations = explainer.explain_dataset(
                        X_sample,
                        predictions=y_sample,
                        return_df=True,
                        show_progress=False,
                        n_jobs=1
                    )
                    orig_duration = time.time() - orig_start
                    logger.info(f"  Original explanations generated in {orig_duration:.2f} seconds")
                except Exception as e2:
                    logger.error(f"  Error generating original explanations for {feat_name}: {e2}")
                    change_rate = 0.0
                    continue
                
                logger.info(f"  Generating modified explanations for {feat_name}...")
                mod_start = time.time()
                try:
                    modified_explanations = explainer.explain_dataset(
                        X_modified,
                        predictions=y_sample,
                        return_df=True,
                        show_progress=False,
                        n_jobs=1
                    )
                    mod_duration = time.time() - mod_start
                    logger.info(f"  Modified explanations generated in {mod_duration:.2f} seconds")
                except Exception as e2:
                    logger.error(f"  Error generating modified explanations for {feat_name}: {e2}")
                    change_rate = 0.0
                    continue
                
                if len(original_explanations) > 0 and len(modified_explanations) > 0:
                    changes = sum(
                        1 for orig, mod in zip(original_explanations['axp'], modified_explanations['axp'], strict=True)
                        if orig != mod
                    )
                    # Use effective_sample_size for normalization (consistent with grouped method)
                    change_rate = changes / effective_sample_size
                    logger.info(f"  Feature {feat_name}: {changes}/{effective_sample_size} explanations changed ({change_rate:.2%}) [row-by-row fallback]")
                else:
                    change_rate = 0.0
                    logger.warning(f"  Feature {feat_name}: No explanations generated")
            
            feat_duration = time.time() - feat_start
            logger.info(f"  Feature {feat_name} completed in {feat_duration:.2f} seconds")
            
            # Warn if feature takes too long
            if feat_duration > 300:  # 5 minutes per feature
                logger.warning(f"  Feature {feat_name} took {feat_duration:.2f} seconds (>5 minutes). Consider reducing causal_sample_size.")
            
            # Cleanup
            del X_sample, X_modified, y_sample
            import gc
            gc.collect()
            
            # Calculate support (number of intervenable instances)
            # For binary features in remove_only mode: number of instances with feature=1
            # For binary features in add_only mode: number of instances with feature=0
            # For continuous features: total sample size
            support = effective_sample_size
            
            # Calculate confidence (fraction of instances where intervention caused change)
            # This is the same as causal_importance (change_rate) for our use case
            # For binary features: confidence = change_rate (fraction of intervenable instances that changed)
            # For continuous features: confidence = change_rate (fraction of all instances that changed)
            confidence = change_rate
            
            causal_scores.append({
                'feature': feat_name,
                'causal_importance': change_rate,  # IR(j) - Intervention Rate
                'support': support,  # Support(j) - Number of intervenable instances
                'confidence': confidence,  # Confidence = change_rate (fraction that changed)
                'median_value': median_val,
                'is_binary': is_binary,
                'intervention': intervention_val
            })
            
            # Incremental checkpointing: save progress every N features
            if (feat_idx + 1) % checkpoint_interval == 0 and causal_checkpoint_path is not None:
                try:
                    # Ensure output directory exists
                    causal_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    # Save current progress
                    temp_df = pd.DataFrame(causal_scores)
                    temp_df = temp_df.sort_values('causal_importance', ascending=False)
                    temp_df.to_parquet(causal_checkpoint_path, index=False, compression='snappy', engine='pyarrow')
                    logger.info(f"  Checkpoint saved: {len(causal_scores)} features processed so far")
                    print(f"  [CHECKPOINT] Saved progress: {len(causal_scores)}/{len(remaining_features)} features")
                except Exception as e:
                    logger.warning(f"  Failed to save checkpoint: {e}")
            
            logger.debug(f"  Feature {feat_name} analyzed in {time.time() - feat_start:.2f} seconds")
        except Exception as e:
            logger.error(f"Error analyzing {feat_name}: {e}", exc_info=True)
            print(f"  [WARNING] Error analyzing {feat_name}: {e}")
            continue
    
    logger.info(f"Causal analysis of {len(remaining_features)} features completed in {time.time() - analysis_start:.2f} seconds")
    
    if causal_scores:
        causal_df = pd.DataFrame(causal_scores)
        causal_df = causal_df.sort_values('causal_importance', ascending=False)
        
        # Final save to checkpoint
        if causal_checkpoint_path is not None:
            try:
                causal_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                causal_df.to_parquet(causal_checkpoint_path, index=False, compression='snappy', engine='pyarrow')
                logger.info(f"Final checkpoint saved: {len(causal_df)} features")
            except Exception as e:
                logger.warning(f"Failed to save final checkpoint: {e}")
        
        logger.info(f"Causal analysis completed for {len(causal_df)} features")
        print(f"\n[OK] Causal analysis completed for {len(causal_df)} features")
        print("\nTop causal features:")
        print(causal_df.head(10).to_string(index=False))
        
        logger.info(f"Causal analysis total time: {time.time() - start_time:.2f} seconds")
        return causal_df
    else:
        logger.warning("No causal scores calculated")
        print("[WARNING] No causal scores calculated.")
        return pd.DataFrame()


def perform_multi_feature_causal_analysis(
    explainer: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_importance_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    cohort: str,
    age_band: str,
    shap_map: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Perform causal analysis testing multi-feature interactions.
    
    Tests combinations of features (pairs, triplets, etc.) to measure
    their combined causal effect and detect synergies/antagonisms.
    
    Args:
        explainer: FFA explainer instance
        X: Feature matrix
        y: Target vector
        feature_importance_df: Feature importance from AXP
        causal_df: Univariate causal analysis results (for individual effects)
        cohort: Cohort name
        age_band: Age band
    
    Returns:
        DataFrame with columns:
        - feature_combination: String representation of feature tuple (e.g., "drug_A|drug_B")
        - interaction_size: Number of features in combination (2, 3, etc.)
        - combined_causal_importance: Combined causal effect when all features are modified
        - sum_individual_effects: Sum of individual univariate effects
        - interaction_effect: Difference (combined - individual), measures synergy/antagonism
        - n_instances_tested: Number of instances tested
        - explanation_change_rate: Fraction of explanations that changed
    """
    if not ANALYSIS_CONFIG.get('enable_interaction_analysis', False):
        logger.info("Multi-feature interaction analysis is disabled. Set enable_interaction_analysis=True to enable.")
        return pd.DataFrame()
    
    logger.info("Performing multi-feature interaction causal analysis...")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("Performing Multi-Feature Interaction Analysis")
    print(f"{'='*80}\n")
    
    # Filter to target class
    logger.info(f"Filtering to target class {ANALYSIS_CONFIG['target_class']}...")
    mask = (y == ANALYSIS_CONFIG['target_class'])
    X_class = X[mask].reset_index(drop=True)
    y_class = y[mask]
    
    # Get top K features from causal analysis (by causal importance)
    # Cohort-specific max interaction size:
    # - First cohort (opioid_ed): size 2 only (pairs)
    # - Second cohort (non_opioid_ed/polypharmacy): size 2 and 3 (pairs and triplets)
    if cohort == 'opioid_ed':
        max_interaction_size = 2  # Only pairs for first cohort
    elif cohort in ['non_opioid_ed', 'polypharmacy']:
        max_interaction_size = 3  # Pairs and triplets for polypharmacy cohort
    else:
        # Default: use config value or fallback to 2
        max_interaction_size = ANALYSIS_CONFIG.get('max_interaction_size', 2)
    
    logger.info(f"Using max_interaction_size={max_interaction_size} for cohort '{cohort}'")
    top_k = ANALYSIS_CONFIG.get('interaction_top_k', 20)
    sample_size = ANALYSIS_CONFIG.get('interaction_sample_size', 100)
    min_effect = ANALYSIS_CONFIG.get('min_interaction_effect', 0.01)
    
    if causal_df.empty:
        logger.warning("No univariate causal results available. Cannot compute interactions.")
        print("[WARNING] No univariate causal results available. Skipping interaction analysis.")
        return pd.DataFrame()
    
    # Select features for interaction analysis
    # REQUIREMENT: Features must have ANY of: SHAP > 0, OR FFA importance > 0, OR causal importance > 0
    # This ensures we test combinations of features that matter in any way
    min_individual_shap_threshold = ANALYSIS_CONFIG.get('min_individual_shap_threshold', 0.0)
    
    # Build set of features with ANY importance > 0 (SHAP, FFA, or causal)
    features_with_importance = set()
    
    # Add features with SHAP importance > 0
    if shap_map:
        shap_important = [
            f for f in X_class.columns
            if shap_map.get(f, 0) > min_individual_shap_threshold
        ]
        features_with_importance.update(shap_important)
        logger.info(f"Found {len(shap_important)} features with SHAP > {min_individual_shap_threshold}")
    
    # Add features with causal importance > 0
    if not causal_df.empty:
        causal_important = causal_df[causal_df['causal_importance'] > 0]['feature'].tolist()
        features_with_importance.update(causal_important)
        logger.info(f"Found {len(causal_important)} features with causal importance > 0")
    
    # Add features with FFA importance > 0
    if not feature_importance_df.empty:
        ffa_important = feature_importance_df[feature_importance_df['importance'] > 0]['feature'].tolist()
        features_with_importance.update(ffa_important)
        logger.info(f"Found {len(ffa_important)} features with FFA importance > 0")
    
    # Filter to features that exist in data
    available_features = [f for f in features_with_importance if f in X_class.columns]
    
    # Sort by combined importance score (prioritize features with multiple importance signals)
    def get_combined_importance_score(feat: str) -> float:
        """Calculate combined importance score for sorting."""
        score = 0.0
        
        # SHAP importance (if available)
        if shap_map:
            score += shap_map.get(feat, 0) * 1.0  # Weight: 1.0
        
        # Causal importance (if available)
        if not causal_df.empty:
            causal_map = dict(zip(causal_df['feature'], causal_df['causal_importance'], strict=False))
            score += causal_map.get(feat, 0) * 0.5  # Weight: 0.5
        
        # FFA importance (if available)
        if not feature_importance_df.empty:
            ffa_map = dict(zip(feature_importance_df['feature'], feature_importance_df['importance'], strict=False))
            score += ffa_map.get(feat, 0) * 0.5  # Weight: 0.5
        
        return score
    
    available_features = sorted(
        available_features,
        key=get_combined_importance_score,
        reverse=True
    )
    
    logger.info(f"Selected {len(available_features)} features with SHAP > 0 OR FFA > 0 OR causal > 0")
    print(f"  - Features with ANY importance (SHAP/FFA/causal) > 0: {len(available_features)}")
    print(f"  - Including all features with SHAP > 0 (no top_k limit)")
    
    if len(available_features) < 2:
        logger.warning(f"Not enough features ({len(available_features)}) for interaction analysis. Need at least 2.")
        print(f"[WARNING] Not enough features for interaction analysis. Found {len(available_features)}, need at least 2.")
        return pd.DataFrame()
    
    logger.info(f"Analyzing interactions for top {len(available_features)} features")
    print(f"Analyzing interactions for top {len(available_features)} features")
    print(f"  - Cohort: {cohort}")
    print(f"  - Max interaction size: {max_interaction_size} (cohort-specific)")
    print(f"  - Sample size: {sample_size}")
    
    # Create mapping of feature -> individual causal importance
    individual_effects_map = dict(zip(
        causal_df['feature'],
        causal_df['causal_importance'],
        strict=True
    ))
    
    # Sample instances for interaction testing
    if len(X_class) > sample_size:
        X_sample = X_class.head(sample_size).copy()
        y_sample = y_class[:sample_size]
    else:
        X_sample = X_class.copy()
        y_sample = y_class
    
    interaction_results = []
    
    # Test interactions of size 2, 3, ..., up to max_interaction_size
    for interaction_size in range(2, max_interaction_size + 1):
        logger.info(f"Testing interactions of size {interaction_size}...")
        print(f"\nTesting {interaction_size}-feature interactions...")
        
        # Generate all combinations of this size
        all_combinations = list(combinations(available_features, interaction_size))
        
        # Filter combinations based on SHAP values if available
        if shap_map:
            # Filter: only keep combinations where ALL features have SHAP importance > 0
            # This is the primary filter - we only test combinations of features that actually matter
            min_individual_shap_threshold = ANALYSIS_CONFIG.get('min_individual_shap_threshold', 0.0)
            
            filtered_combinations = []
            combination_scores = []
            
            for combo in all_combinations:
                # Check that ALL features in combination have SHAP > threshold
                feature_shaps = [shap_map.get(f, 0) for f in combo]
                min_shap = min(feature_shaps)
                combined_shap = sum(feature_shaps)
                
                # Only include if all features meet the threshold
                if min_shap > min_individual_shap_threshold:
                    # Calculate score for sorting (higher is better)
                    score = combined_shap * (1 + min_shap)  # Boost if all features have decent importance
                    combination_scores.append((combo, score, combined_shap, min_shap))
                    filtered_combinations.append(combo)
            
            # Sort by combined SHAP score (highest first) for better prioritization
            combination_scores.sort(key=lambda x: x[1], reverse=True)
            filtered_combinations = [combo for combo, _, _, _ in combination_scores]
            
            logger.info(f"  Filtered {len(all_combinations)} combinations to {len(filtered_combinations)} based on SHAP importance > {min_individual_shap_threshold}")
            logger.info(f"    (All features in combinations have SHAP > {min_individual_shap_threshold})")
            
            # Optional: Apply combined SHAP threshold if configured (but don't limit count)
            min_combined_shap_threshold = ANALYSIS_CONFIG.get('min_combined_shap_threshold', 0.0)
            if min_combined_shap_threshold > 0.0:
                filtered_combinations = [
                    combo for combo, _, combined_shap, _ in combination_scores
                    if combined_shap >= min_combined_shap_threshold
                ]
                logger.info(f"  Further filtered to {len(filtered_combinations)} combinations with combined SHAP >= {min_combined_shap_threshold}")
            
            all_combinations = filtered_combinations
            
            # Stage 3: Apply interaction candidate pruning
            # Rule 5: Co-occurrence support filter
            min_cooccur_support = ANALYSIS_CONFIG.get('min_cooccur_support', 5)
            min_cooccur_triplet = ANALYSIS_CONFIG.get('min_cooccur_support_triplet', 3)
            binary_mode = ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')
            
            pruned_combinations = []
            for combo in all_combinations:
                # Identify binary features in combination
                binary_feats = []
                for feat_name in combo:
                    unique_vals = X_sample[feat_name].unique()
                    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                        binary_feats.append(feat_name)
                
                # Apply co-occurrence filter for binary features
                if binary_feats:
                    if binary_mode == 'remove_only':
                        # Require all binary features = 1
                        cooccur_mask = pd.Series(True, index=X_sample.index)
                        for feat_name in binary_feats:
                            cooccur_mask = cooccur_mask & (X_sample[feat_name] == 1)
                        cooccur_count = int(cooccur_mask.sum())
                        threshold = min_cooccur_triplet if interaction_size >= 3 else min_cooccur_support
                    elif binary_mode == 'add_only':
                        # Require all binary features = 0
                        cooccur_mask = pd.Series(True, index=X_sample.index)
                        for feat_name in binary_feats:
                            cooccur_mask = cooccur_mask & (X_sample[feat_name] == 0)
                        cooccur_count = int(cooccur_mask.sum())
                        threshold = min_cooccur_triplet if interaction_size >= 3 else min_cooccur_support
                    else:  # flip mode - no co-occurrence filter
                        pruned_combinations.append(combo)
                        continue
                    
                    if cooccur_count < threshold:
                        logger.debug(f"    Pruned combo {combo[:2]}: insufficient co-occurrence ({cooccur_count} < {threshold})")
                        continue
                
                # Combination passed co-occurrence filter
                pruned_combinations.append(combo)
            
            all_combinations = pruned_combinations
            logger.info(f"  After co-occurrence pruning: {len(all_combinations)} combinations")
            
            # Rule 6: Cap combinations per size
            max_combinations_per_size = ANALYSIS_CONFIG.get('max_combinations_per_size', 1000)
            if len(all_combinations) > max_combinations_per_size:
                # Already sorted by SHAP score, take top-K
                logger.info(f"  Capping size-{interaction_size} combinations: {len(all_combinations)} -> {max_combinations_per_size}")
                all_combinations = all_combinations[:max_combinations_per_size]
            
            feature_combinations = all_combinations
        else:
            # No SHAP filtering available - apply co-occurrence and capping only
            min_cooccur_support = ANALYSIS_CONFIG.get('min_cooccur_support', 5)
            min_cooccur_triplet = ANALYSIS_CONFIG.get('min_cooccur_support_triplet', 3)
            binary_mode = ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')
            
            pruned_combinations = []
            for combo in all_combinations:
                # Identify binary features in combination
                binary_feats = []
                for feat_name in combo:
                    unique_vals = X_sample[feat_name].unique()
                    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                        binary_feats.append(feat_name)
                
                # Apply co-occurrence filter for binary features
                if binary_feats:
                    if binary_mode == 'remove_only':
                        cooccur_mask = pd.Series(True, index=X_sample.index)
                        for feat_name in binary_feats:
                            cooccur_mask = cooccur_mask & (X_sample[feat_name] == 1)
                        cooccur_count = int(cooccur_mask.sum())
                        threshold = min_cooccur_triplet if interaction_size >= 3 else min_cooccur_support
                    elif binary_mode == 'add_only':
                        cooccur_mask = pd.Series(True, index=X_sample.index)
                        for feat_name in binary_feats:
                            cooccur_mask = cooccur_mask & (X_sample[feat_name] == 0)
                        cooccur_count = int(cooccur_mask.sum())
                        threshold = min_cooccur_triplet if interaction_size >= 3 else min_cooccur_support
                    else:  # flip mode
                        pruned_combinations.append(combo)
                        continue
                    
                    if cooccur_count < threshold:
                        continue
                
                pruned_combinations.append(combo)
            
            all_combinations = pruned_combinations
            
            # Cap combinations per size
            max_combinations_per_size = ANALYSIS_CONFIG.get('max_combinations_per_size', 1000)
            if len(all_combinations) > max_combinations_per_size:
                logger.info(f"  Capping size-{interaction_size} combinations: {len(all_combinations)} -> {max_combinations_per_size}")
                all_combinations = all_combinations[:max_combinations_per_size]
            
            feature_combinations = all_combinations
        
        logger.info(f"  Final combinations to test for size {interaction_size}: {len(feature_combinations)}")
        
        for combo_idx, feature_combo in enumerate(tqdm(feature_combinations, desc=f"Size {interaction_size}")):
            try:
                combo_start = time.time()
                
                # Calculate sum of individual effects
                sum_individual = sum(individual_effects_map.get(f, 0.0) for f in feature_combo)
                
                # For multi-feature interactions, use configurable binary intervention mode
                mode = ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')
                
                # For binary presence/absence indicators, choose semantics:
                # - remove_only: test rows where ALL binary feats are present; set them to 0
                # - add_only: test rows where ALL binary feats are absent; set them to 1
                # - flip: test all rows; flip all binary feats
                binary_features = []
                continuous_features = []
                
                for feat_name in feature_combo:
                    unique_vals = X_sample[feat_name].unique()
                    is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})
                    if is_binary:
                        binary_features.append(feat_name)
                    else:
                        continuous_features.append(feat_name)
                
                # Determine test mask based on mode
                if binary_features:
                    if mode == 'remove_only':
                        # Test only rows where ALL binary features are present (1)
                        test_mask = pd.Series(True, index=X_sample.index)
                        for feat_name in binary_features:
                            test_mask = test_mask & (X_sample[feat_name] == 1)
                    elif mode == 'add_only':
                        # Test only rows where ALL binary features are absent (0)
                        test_mask = pd.Series(True, index=X_sample.index)
                        for feat_name in binary_features:
                            test_mask = test_mask & (X_sample[feat_name] == 0)
                    elif mode == 'flip':
                        # Test all rows
                        test_mask = pd.Series(True, index=X_sample.index)
                    else:
                        raise ValueError(f"Unknown binary_intervention_mode: {mode}")
                    
                    num_test = int(test_mask.sum())
                    
                    # If combo contains binary features but none match the test mask, skip
                    if binary_features and num_test == 0:
                        logger.debug(f"    Combination {combo_idx+1}: No instances match test mask for mode '{mode}', skipping")
                        continue
                    
                    # Filter to test subset
                    X_sample_filtered = X_sample[test_mask].copy()
                    y_sample_filtered = y_sample[test_mask]
                else:
                    # No binary features, test all instances
                    test_mask = pd.Series(True, index=X_sample.index)
                    X_sample_filtered = X_sample.copy()
                    y_sample_filtered = y_sample
                    num_test = len(X_sample)
                
                # Create modified dataset with all features in combination modified
                X_modified_filtered = X_sample_filtered.copy()
                
                for feat_name in feature_combo:
                    unique_vals = X_sample_filtered[feat_name].unique()
                    is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})
                    
                    if is_binary:
                        if mode == 'remove_only':
                            # Remove binary features (set to 0) - only instances where feature was 1 are in filtered set
                            X_modified_filtered[feat_name] = 0
                        elif mode == 'add_only':
                            # Add binary features (set to 1) - only instances where feature was 0 are in filtered set
                            X_modified_filtered[feat_name] = 1
                        elif mode == 'flip':
                            # Flip binary features (0<->1)
                            X_modified_filtered[feat_name] = 1 - X_sample_filtered[feat_name]
                        else:
                            raise ValueError(f"Unknown binary_intervention_mode: {mode}")
                    else:
                        # For continuous features: set to median (only on masked subset)
                        median_val = X_sample[feat_name].median()
                        X_modified_filtered[feat_name] = median_val
                
                # Generate original explanations (on filtered subset)
                logger.debug(f"    Generating original explanations for combination {combo_idx+1}/{len(feature_combinations)} (n={len(X_sample_filtered)})...")
                try:
                    original_explanations = explainer.explain_dataset(
                        X_sample_filtered,
                        predictions=y_sample_filtered,
                        return_df=True,
                        show_progress=True,  # Enable progress visibility
                        n_jobs=1  # Single worker to save memory
                    )
                except Exception as e:
                    logger.error(f"    Error generating original explanations for combination {combo_idx+1}: {e}")
                    continue
                
                # Stage 4: Runtime pruning - Early stopping check
                # Check first N instances for zero changes before generating full modified explanations
                enable_early_stopping = ANALYSIS_CONFIG.get('enable_early_stopping', True)
                early_stopping_n = ANALYSIS_CONFIG.get('early_stopping_n', 10)
                
                if enable_early_stopping and len(original_explanations) > early_stopping_n:
                    # Generate modified explanations for first N instances only
                    X_modified_early = X_modified_filtered.head(early_stopping_n).copy()
                    y_sample_early = y_sample_filtered[:early_stopping_n]
                    
                    try:
                        modified_explanations_early = explainer.explain_dataset(
                            X_modified_early,
                            predictions=y_sample_early,
                            return_df=True,
                            show_progress=False,  # Disable progress for early check
                            n_jobs=1
                        )
                        
                        # Check for zero changes in early sample
                        early_changes = sum(
                            1 for orig, mod in zip(
                                original_explanations['axp'].head(early_stopping_n),
                                modified_explanations_early['axp'],
                                strict=True
                            )
                            if orig != mod
                        )
                        
                        # If zero changes in early sample and we have many instances, skip full computation
                        if early_changes == 0 and len(original_explanations) > (early_stopping_n * 2):
                            logger.debug(f"    Early stopping: zero changes in first {early_stopping_n} instances (n={len(original_explanations)}), skipping full computation")
                            combined_effect = 0.0
                            explanation_change_rate = 0.0
                            
                            # Still record the result (with zero effect) for completeness
                            feature_combo_str = "|".join(sorted(feature_combo))
                            interaction_results.append({
                                'feature_combination': feature_combo_str,
                                'interaction_size': interaction_size,
                                'combined_causal_importance': 0.0,
                                'sum_individual_effects': sum_individual,
                                'interaction_effect': -sum_individual,  # Negative of individual (no combined effect)
                                'n_instances_tested': len(X_sample_filtered),
                                'explanation_change_rate': 0.0,
                                'synergy_type': 'neutral'
                            })
                            continue
                        
                    except Exception as e:
                        logger.debug(f"    Early stopping check failed: {e}. Proceeding with full computation.")
                        # Fall through to full computation
                
                # Generate modified explanations (on filtered subset)
                logger.debug(f"    Generating modified explanations for combination {combo_idx+1}/{len(feature_combinations)} (n={len(X_modified_filtered)})...")
                try:
                    modified_explanations = explainer.explain_dataset(
                        X_modified_filtered,
                        predictions=y_sample_filtered,
                        return_df=True,
                        show_progress=True,  # Enable progress visibility
                        n_jobs=1
                    )
                except Exception as e:
                    logger.error(f"    Error generating modified explanations for combination {combo_idx+1}: {e}")
                    continue
                
                # Calculate combined causal effect (normalized by |S_f|, not N)
                if len(original_explanations) > 0 and len(modified_explanations) > 0:
                    changes = sum(
                        1 for orig, mod in zip(original_explanations['axp'], modified_explanations['axp'], strict=True)
                        if orig != mod
                    )
                    combined_effect = changes / len(original_explanations)  # Normalized by filtered sample size
                    explanation_change_rate = combined_effect
                else:
                    combined_effect = 0.0
                    explanation_change_rate = 0.0
                
                # Calculate interaction effect (synergy/antagonism)
                interaction_effect = combined_effect - sum_individual
                
                # Only record if interaction effect meets minimum threshold
                if abs(interaction_effect) >= min_effect:
                    feature_combo_str = "|".join(sorted(feature_combo))  # Sort for consistency
                    
                    interaction_results.append({
                        'feature_combination': feature_combo_str,
                        'interaction_size': interaction_size,
                        'combined_causal_importance': combined_effect,
                        'sum_individual_effects': sum_individual,
                        'interaction_effect': interaction_effect,
                        'n_instances_tested': len(X_sample_filtered),  # Actual instances tested (filtered subset)
                        'explanation_change_rate': explanation_change_rate,
                        'synergy_type': 'positive' if interaction_effect > 0.01 else ('negative' if interaction_effect < -0.01 else 'neutral')
                    })
                
                logger.debug(f"  Combination {combo_idx+1}/{len(feature_combinations)}: {feature_combo_str} "
                           f"(combined={combined_effect:.3f}, individual_sum={sum_individual:.3f}, "
                           f"interaction={interaction_effect:.3f})")
                
                # Cleanup - delete variables that were created in this iteration
                try:
                    del X_modified_filtered, X_sample_filtered, original_explanations, modified_explanations
                except NameError:
                    pass  # Some variables may not exist if early stopping triggered
                
                # Cleanup optional early stopping variables
                try:
                    del X_modified_early, modified_explanations_early
                except NameError:
                    pass  # These only exist if early stopping was attempted
                
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error analyzing combination {feature_combo}: {e}", exc_info=True)
                continue
    
    if interaction_results:
        interaction_df = pd.DataFrame(interaction_results)
        interaction_df = interaction_df.sort_values('interaction_effect', key=abs, ascending=False)
        
        logger.info(f"Multi-feature interaction analysis completed: {len(interaction_df)} interactions found")
        print(f"\n[OK] Found {len(interaction_df)} significant interactions")
        print("\nTop interactions:")
        print(interaction_df.head(10).to_string(index=False))
        
        # Summary statistics
        positive_synergies = len(interaction_df[interaction_df['synergy_type'] == 'positive'])
        negative_synergies = len(interaction_df[interaction_df['synergy_type'] == 'negative'])
        neutral = len(interaction_df[interaction_df['synergy_type'] == 'neutral'])
        
        logger.info(f"Interaction summary: {positive_synergies} positive synergies, "
                   f"{negative_synergies} antagonisms, {neutral} neutral")
        print(f"\nSummary: {positive_synergies} positive synergies, {negative_synergies} antagonisms, {neutral} neutral")
        
        logger.info(f"Multi-feature interaction analysis total time: {time.time() - start_time:.2f} seconds")
        return interaction_df
    else:
        logger.warning("No significant interactions found")
        print("[WARNING] No significant interactions found.")
        return pd.DataFrame()


def save_results(model_type: str, df_axps: pd.DataFrame, 
                feature_importance_df: pd.DataFrame, 
                causal_df: pd.DataFrame,
                interaction_df: Optional[pd.DataFrame] = None):
    """Save all analysis results."""
    logger.info("Saving analysis results...")
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}\n")
    
    model_output_dir = OUTPUT_DIR / model_type
    logger.info(f"Creating output directory: {model_output_dir}")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add run metadata to outputs for easy comparison across runs
    run_binary_mode = ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')
    for _df in (df_axps, feature_importance_df, causal_df, interaction_df):
        if _df is None:
            continue
        try:
            if isinstance(_df, pd.DataFrame) and 'binary_intervention_mode' not in _df.columns:
                _df['binary_intervention_mode'] = run_binary_mode
        except Exception:
            # Don't fail saving just because metadata couldn't be attached
            pass
    
    # Save explanations (Parquet format for efficiency)
    explanations_path = None
    if len(df_axps) > 0:
        explanations_path = model_output_dir / 'axp_explanations.parquet'
        logger.info(f"Saving explanations to: {explanations_path}")
        df_axps.to_parquet(explanations_path, index=False, compression='snappy', engine='pyarrow')
        logger.info(f"Saved {len(df_axps)} explanations")
        print(f"[OK] Saved explanations to: {explanations_path}")
    else:
        logger.warning("No explanations to save")
    
    # Save feature importance (Parquet format for efficiency)
    importance_path = None
    if len(feature_importance_df) > 0:
        importance_path = model_output_dir / 'feature_importance_axp.parquet'
        logger.info(f"Saving feature importance to: {importance_path}")
        feature_importance_df.to_parquet(importance_path, index=False, compression='snappy', engine='pyarrow')
        logger.info(f"Saved {len(feature_importance_df)} feature importance scores")
        print(f"[OK] Saved feature importance to: {importance_path}")
    else:
        logger.warning("No feature importance to save")
    
    # Save causal importance (Parquet format for efficiency)
    causal_path = None
    if len(causal_df) > 0:
        causal_path = model_output_dir / 'causal_importance.parquet'
        logger.info(f"Saving causal importance to: {causal_path}")
        causal_df.to_parquet(causal_path, index=False, compression='snappy', engine='pyarrow')
        logger.info(f"Saved {len(causal_df)} causal importance scores")
        print(f"[OK] Saved causal importance to: {causal_path}")
        
        # Print top 10 causal importance features
        print("\n" + "=" * 80)
        print("TOP 10 CAUSAL IMPORTANCE FEATURES")
        print("=" * 80)
        top_10_causal = causal_df.head(10)[['feature', 'causal_importance']].copy()
        for rank, (_, row) in enumerate(top_10_causal.iterrows(), start=1):
            print(f"  {rank:2d}. {row['feature']:<50} {row['causal_importance']:>10.6f}")
        print("=" * 80 + "\n")
    else:
        logger.warning("No causal importance to save")
    
    # Save interaction analysis results (Parquet format for efficiency)
    interaction_path = None
    if interaction_df is not None and len(interaction_df) > 0:
        interaction_path = model_output_dir / 'interaction_analysis.parquet'
        logger.info(f"Saving interaction analysis to: {interaction_path}")
        interaction_df.to_parquet(interaction_path, index=False, compression='snappy', engine='pyarrow')
        logger.info(f"Saved {len(interaction_df)} interaction results")
        print(f"[OK] Saved interaction analysis to: {interaction_path}")
    elif interaction_df is not None and len(interaction_df) == 0:
        logger.info("No interaction results to save (empty DataFrame)")

    # Upload to S3 and save checkpoint after saving results
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint

        s3_outputs = []
        if explanations_path and explanations_path.exists():
            s3_explanations = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/axp_explanations.parquet"
            if upload_file_to_s3(explanations_path, s3_explanations, logger):
                s3_outputs.append(s3_explanations)

        if importance_path and importance_path.exists():
            s3_importance = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/feature_importance_axp.parquet"
            if upload_file_to_s3(importance_path, s3_importance, logger):
                s3_outputs.append(s3_importance)
        
        if causal_path and causal_path.exists():
            s3_causal = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/causal_importance.parquet"
            if upload_file_to_s3(causal_path, s3_causal, logger):
                s3_outputs.append(s3_causal)
        
        if interaction_path and interaction_path.exists():
            s3_interaction = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/interaction_analysis.parquet"
            if upload_file_to_s3(interaction_path, s3_interaction, logger):
                s3_outputs.append(s3_interaction)

        # Save checkpoint (only once per step, not per model type)
        if model_type == "xgboost":  # Save checkpoint after first model completes
            save_step_checkpoint(
                step_name="8_ffa_analysis",
                cohort=COHORT_NAME,
                age_band=AGE_BAND,
                metadata={"model_types_analyzed": [model_type]},
                output_paths=s3_outputs,
                logger=logger,
            )
    except ImportError:
        pass  # Checkpoint saving is optional
    
    # Note: Causal analysis already saved above as Parquet (removed duplicate CSV save)
    
    # Create summary report
    logger.info("Creating summary report...")
    summary = {
        'model_type': model_type,
        'cohort': COHORT_NAME,
        'age_band': AGE_BAND,
        'total_explanations': len(df_axps),
        'explanations_with_conditions': sum(1 for axp in df_axps['axp'] if axp and len(axp) > 0),
        'top_features': feature_importance_df.head(10)['feature'].tolist() if len(feature_importance_df) > 0 else [],
        'causal_features': causal_df.head(10)['feature'].tolist() if len(causal_df) > 0 else []
    }
    
    summary_path = model_output_dir / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to: {summary_path}")
    print(f"[OK] Saved summary to: {summary_path}")
    
    logger.info(f"Results saved in {time.time() - start_time:.2f} seconds")
    print(f"\n[OK] All results saved to: {model_output_dir}")


def run_full_analysis_for_model(model_type: str) -> Optional[Dict]:
    """Run complete FFA analysis for a single model type."""
    model_start_time = time.time()
    logger.info(f"{'='*80}")
    logger.info(f"Starting FFA Analysis for: {model_type.upper()}")
    logger.info(f"{'='*80}")
    
    print("\n" + "="*80)
    print(f"FFA Analysis: {model_type.upper()}")
    print("="*80)
    
    # Build paths (use current cohort/age band)
    # Load best XGBoost model JSON (selected by final model training)
    if model_type in ['xgboost', 'xgboost_rf']:
        # Use best XGBoost model (could be xgb or xgb_rf)
        model_json_filename = f'{COHORT_NAME}_{AGE_BAND_FNAME}_best_xgboost_model.json'
        model_json_path = MODEL_JSON_BASE / model_json_filename
        
        # Fallback to model_outputs location
        if not model_json_path.exists():
            model_outputs_base = (
                PROJECT_ROOT
                / "6_final_model"
                / "model_outputs"
                / COHORT_NAME
                / AGE_BAND_FNAME
            )
            model_json_path = model_outputs_base / model_json_filename
        
        # Load model selection metadata to determine variant
        metadata_path = MODEL_JSON_BASE.parent / f'{COHORT_NAME}_{AGE_BAND_FNAME}_model_selection_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                selection_metadata = json.load(f)
                actual_variant = selection_metadata.get('best_xgb_variant', 'xgb')
                logger.info(f"Best XGBoost variant: {actual_variant} (from selection metadata)")
                # Normalize variant name for FFA: "xgb" -> "xgboost", "xgb_rf" -> "xgboost_rf"
                if actual_variant == 'xgb':
                    actual_variant = 'xgboost'
                elif actual_variant == 'xgb_rf':
                    actual_variant = 'xgboost_rf'
        else:
            logger.warning(f"Model selection metadata not found at {metadata_path}")
            actual_variant = 'xgboost'  # Default (normalized)
    else:
        # CatBoost - use best model
        model_json_filename = f'{COHORT_NAME}_{AGE_BAND_FNAME}_best_catboost_model.json'
        model_json_path = MODEL_JSON_BASE / model_json_filename
        
        # Fallback to model_outputs location
        if not model_json_path.exists():
            model_outputs_base = (
                PROJECT_ROOT
                / "6_final_model"
                / "model_outputs"
                / COHORT_NAME
                / AGE_BAND_FNAME
            )
            model_json_path = model_outputs_base / model_json_filename
    
    logger.info(f"Model JSON path: {model_json_path}")
    
    if not model_json_path.exists():
        logger.warning(f"Best model JSON not found: {model_json_path}")
        print(f"[SKIP] Model JSON not found: {model_json_path}")
        return None
    
    try:
        # Step 1: Load model
        logger.info("Step 1: Loading model JSON...")
        model_json = load_model_json(model_json_path)
        
        # Step 2: Extract feature mappings
        logger.info("Step 2: Extracting feature mappings...")
        feature_mappings = extract_feature_mappings(model_json)
        
        # Step 3: Load data
        logger.info("Step 3: Loading data...")
        X, y = load_data(DATA_PATH, max_samples=ANALYSIS_CONFIG.get('max_samples'))
        
        if y is None:
            logger.warning("No target column found. Cannot generate explanations.")
            print("[WARNING] No target column found. Cannot generate explanations.")
            return None
        
        # Validate feature count matches model expectations
        # Exclude non-feature columns (like mi_person_key, instance_index) from comparison
        non_feature_cols = {'mi_person_key', 'person_key', 'patient_id', 'id', 'instance_index'}
        feature_cols = [col for col in X.columns if col not in non_feature_cols]
        X_features_only = X[feature_cols].copy()
        
        model_feature_names = feature_mappings.get('feature_names', {})
        expected_n_features = len(model_feature_names) if model_feature_names else None
        
        if expected_n_features and len(feature_cols) != expected_n_features:
            logger.warning(f"Feature count mismatch: CSV has {len(feature_cols)} features (excluding IDs and instance_index), model expects {expected_n_features}")
            print(f"[WARNING] Feature count mismatch: CSV has {len(feature_cols)} features (excluding IDs and instance_index), model expects {expected_n_features}")
            
            # Try to align features with model's expected feature names
            if model_feature_names:
                expected_features = [model_feature_names.get(i, f"feature_{i}") for i in range(expected_n_features)]
                missing_features = set(expected_features) - set(feature_cols)
                extra_features = set(feature_cols) - set(expected_features)
                
                if missing_features:
                    logger.error(f"Missing features in CSV: {list(missing_features)[:10]}...")
                    print(f"[ERROR] Missing features in CSV: {len(missing_features)} features")
                    raise ValueError(f"Feature mismatch: CSV missing {len(missing_features)} features expected by model. "
                                   f"First few missing: {list(missing_features)[:5]}")
                
                # Reorder columns to match model's expected order
                if set(feature_cols) == set(expected_features):
                    X_features_only = X_features_only[expected_features]
                    logger.info(f"Reordered features to match model expectations")
                else:
                    # Filter out ID columns and instance_index from extra_features for clearer error message
                    extra_features_filtered = extra_features - non_feature_cols
                    if extra_features_filtered:
                        logger.error(f"Feature sets don't match. Extra in CSV: {list(extra_features_filtered)[:10]}...")
                        raise ValueError(f"Feature mismatch: CSV has {len(extra_features_filtered)} extra features not in model: {list(extra_features_filtered)[:5]}")
                    else:
                        # Only ID columns/instance_index are extra, which is fine - just reorder
                        X_features_only = X_features_only[expected_features]
                        logger.info(f"Reordered features to match model expectations (ignoring extra ID columns and instance_index)")
        
        # Store instance_index separately if it exists (for SHAP alignment)
        instance_index_col = None
        if 'instance_index' in X.columns:
            instance_index_col = X['instance_index'].copy()
            logger.debug("Preserved instance_index column for SHAP alignment")
        
        # Use features-only DataFrame for analysis (keep original X for reference if needed)
        X = X_features_only
        logger.info(f"Feature matrix validated: {len(X.columns)} features, {len(X)} samples")
        print(f"[OK] Feature matrix: {len(X.columns)} features, {len(X)} samples")
        
        # Re-attach instance_index if it was preserved (for downstream SHAP alignment)
        if instance_index_col is not None:
            X['instance_index'] = instance_index_col.values
            logger.debug("Re-attached instance_index column to feature matrix for SHAP alignment")
        
        logger.info(f"Feature matrix validated: {len(X.columns)} features, {len(X)} samples")
        print(f"[OK] Feature matrix: {len(X.columns)} features, {len(X)} samples")
        
        # Step 3.5: Load SHAP importance (required) and individual SHAP values (required)
        logger.info("Step 3.5: Loading SHAP importance and individual SHAP values...")
        try:
            shap_map, shap_values_df = load_shap_importance(COHORT_NAME, AGE_BAND, model_type)
            logger.info(f"Loaded SHAP importance for {len(shap_map)} features (importance > 0)")
            
            if shap_values_df is None or len(shap_values_df) == 0:
                error_msg = (
                    "ERROR: Individual SHAP values per instance are REQUIRED for accurate rule filtering. "
                    f"Could not load individual SHAP values from Step 7 (SHAP Analysis). "
                    f"Please ensure the parquet file exists: "
                    f"7_shap_analysis/outputs/{COHORT_NAME}/{AGE_BAND_FNAME}/{COHORT_NAME}_{AGE_BAND_FNAME}_shap_sample_values_{model_type}.parquet"
                )
                logger.error(error_msg)
                print(f"[ERROR] {error_msg}")
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Loaded individual SHAP values for {len(shap_values_df)} instances, {len(shap_values_df.columns)} features")
            print(f"[OK] Using individual SHAP values per instance for rule filtering")
            
            # Validate that we have SHAP values for all instances we'll process
            if len(X) > len(shap_values_df):
                logger.warning(
                    f"Data has {len(X)} instances but SHAP values only has {len(shap_values_df)} instances. "
                    f"Will use available SHAP values, but some instances may not have individual SHAP values."
                )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to load SHAP importance or individual SHAP values: {e}")
            print(f"[ERROR] Failed to load SHAP data: {e}")
            print(f"[ERROR] Both global SHAP importance and individual SHAP values are required. Please run Step 7 (SHAP Analysis) first.")
            raise
        
        # Step 4: Initialize explainer
        logger.info("Step 4: Initializing explainer...")
        explainer = initialize_explainer(
            model_json_path,
            model_json,
            feature_mappings,
            feature_names=list(X.columns) if isinstance(X, pd.DataFrame) else None,
            shap_importance_map=shap_map,
            shap_values_df=shap_values_df,
        )
        
        if explainer is None:
            logger.warning("Explainer not available.")
            print("[WARNING] Explainer not available.")
            return None
        
        # Cleanup model_json to free memory
        del model_json
        import gc
        gc.collect()
        
        # Step 5: Generate explanations
        logger.info("Step 5: Generating explanations...")
        df_axps = generate_explanations(explainer, X, y)
        
        if len(df_axps) == 0:
            logger.warning("No explanations generated.")
            print("[WARNING] No explanations generated.")
            return None
        
        # Step 6: Calculate feature importance
        logger.info("Step 6: Calculating feature importance...")
        feature_importance_df = calculate_feature_importance(df_axps)
        
        # Step 7: Perform causal analysis (optional, can be skipped if memory is tight)
        logger.info("Step 7: Performing causal analysis...")
        try:
            causal_df = perform_causal_analysis(explainer, X, y, feature_importance_df, COHORT_NAME, AGE_BAND, 
                                                model_type=model_type, output_dir=OUTPUT_DIR, shap_map=shap_map)
        except MemoryError:
            logger.warning("Memory error during causal analysis. Skipping causal analysis.")
            print("[WARNING] Memory error during causal analysis. Skipping causal analysis.")
            causal_df = pd.DataFrame()
        
        # Step 7.5: Perform multi-feature interaction analysis (if enabled and causal analysis completed)
        interaction_df = pd.DataFrame()
        if not causal_df.empty and ANALYSIS_CONFIG.get('enable_interaction_analysis', False):
            logger.info("Step 7.5: Performing multi-feature interaction analysis...")
            try:
                interaction_df = perform_multi_feature_causal_analysis(
                    explainer, X, y, feature_importance_df, causal_df, COHORT_NAME, AGE_BAND,
                    shap_map=shap_map  # Pass SHAP map for filtering combinations
                )
            except Exception as e:
                logger.warning(f"Error during interaction analysis: {e}. Skipping interaction analysis.")
                print(f"[WARNING] Error during interaction analysis: {e}. Skipping interaction analysis.")
                interaction_df = pd.DataFrame()
        
        # Step 8: Save results
        logger.info("Step 8: Saving results...")
        save_results(model_type, df_axps, feature_importance_df, causal_df, interaction_df)
        
        total_time = time.time() - model_start_time
        logger.info(f"Model {model_type} analysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return {
            'model_type': model_type,
            'explanations': len(df_axps),
            'features_analyzed': len(feature_importance_df),
            'causal_features': len(causal_df)
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze {model_type}: {e}", exc_info=True)
        print(f"\n[ERROR] Failed to analyze {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_validation_if_requested(cohort: str, age_band: str, model_type: str = "xgboost"):
    """
    Optionally run validation to compare XGBoost JSON rules with SHAP values.
    
    This validates that SHAP values can accurately filter and build the rule set
    for causal analysis. It demonstrates that rules extracted from JSON align well
    with SHAP importance patterns, confirming that SHAP-guided rule filtering
    produces meaningful results.
    """
    try:
        from validate_xgboost_rules_vs_shap import main as validate_main
        import sys
        
        logger.info("Running XGBoost rule extraction validation...")
        # Temporarily override sys.argv for the validation script
        original_argv = sys.argv
        try:
            sys.argv = [
                'validate_xgboost_rules_vs_shap.py',
                '--cohort', cohort,
                '--age-band', age_band,
                '--model-type', model_type
            ]
            validate_main()
            logger.info("✅ Validation completed successfully")
        finally:
            sys.argv = original_argv
    except ImportError:
        logger.warning("Validation script not available. Skipping validation.")
    except Exception as e:
        logger.warning(f"Validation failed: {e}. Continuing with main analysis.")


def main():
    """Run complete FFA analysis for one or more model types."""
    global COHORT_NAME, AGE_BAND, AGE_BAND_FNAME, MODEL_JSON_BASE, DATA_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Run FFA analysis for specified model types."
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default=COHORT_NAME,
        help="Cohort name (e.g., opioid_ed). Defaults to opioid_ed.",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default=AGE_BAND,
        help="Age band (e.g., 13-24, 25-44). Defaults to 13-24.",
    )
    parser.add_argument(
        "--model-type",
        choices=["catboost", "xgboost", "xgboost_rf", "all"],
        default="all",
        help="Which model type to analyze (default: all).",
    )
    parser.add_argument(
        "--binary-intervention-mode",
        choices=["remove_only", "add_only", "flip"],
        default=ANALYSIS_CONFIG.get("binary_intervention_mode", "remove_only"),
        help="Binary intervention semantics: remove_only (1->0 on present), add_only (0->1 on absent), flip (0<->1 on all).",
    )
    args = parser.parse_args()
    
    # Apply CLI override
    ANALYSIS_CONFIG['binary_intervention_mode'] = args.binary_intervention_mode
    
    # Update global variables after parsing args
    COHORT_NAME = args.cohort_name
    AGE_BAND = args.age_band
    AGE_BAND_FNAME = AGE_BAND.replace("-", "_")

    # Recompute paths based on updated cohort/age band
    MODEL_JSON_BASE = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / COHORT_NAME
        / AGE_BAND_FNAME
        / "final_model_json"
    )
    # Try multiple locations for data file
    data_paths_to_try = [
        # Primary location: 6_final_model outputs
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / COHORT_NAME
        / AGE_BAND_FNAME
        / "inputs"
        / "model_train"
        / "final_features.parquet",
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / COHORT_NAME
        / AGE_BAND_FNAME
        / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        # Alternative: data folder (various structures)
        PROJECT_ROOT
        / "data"
        / COHORT_NAME
        / AGE_BAND_FNAME
        / "final_features.parquet",
        PROJECT_ROOT
        / "data"
        / COHORT_NAME
        / AGE_BAND_FNAME
        / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        PROJECT_ROOT
        / "data"
        / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        # Data folder with cohorts structure (check latest year)
        PROJECT_ROOT
        / "data"
        / "cohorts"
        / f"cohort_name={COHORT_NAME}"
        / "event_year=2019"
        / f"age_band={AGE_BAND}"
        / "final_features.parquet",
        PROJECT_ROOT
        / "data"
        / "cohorts"
        / f"cohort_name={COHORT_NAME}"
        / "event_year=2020"
        / f"age_band={AGE_BAND}"
        / "final_features.parquet",
        # Gold cohorts directory
        PROJECT_ROOT
        / "data"
        / "gold_cohorts"
        / f"cohort_name={COHORT_NAME}"
        / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        PROJECT_ROOT
        / "data"
        / "gold_cohorts"
        / f"cohort_name={COHORT_NAME}"
        / "final_features.parquet",
        PROJECT_ROOT
        / "data"
        / "gold_cohorts"
        / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
    ]
    
    DATA_PATH = None
    for path in data_paths_to_try:
        if path.exists():
            DATA_PATH = path
            logger.info(f"Found data file at: {DATA_PATH}")
            break
    
    if DATA_PATH is None:
        logger.warning(f"Data file not found in any of the expected locations:")
        for path in data_paths_to_try:
            logger.warning(f"  - {path}")
        # Set to the primary expected location for error message clarity
        DATA_PATH = data_paths_to_try[1]
    OUTPUT_DIR = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which model types to analyze
    if args.model_type == "all":
        # For "all", we analyze best XGBoost variant only
        expected_model_types = ['xgboost']
    else:
        expected_model_types = [args.model_type]

    # Check for existing local outputs (idempotency - check local first)
    # FFA generates outputs per model type, so check for the expected model type(s)
    if args.model_type == "all":
        # For "all", we analyze best XGBoost variant only
        expected_model_types = ['xgboost']
    else:
        expected_model_types = [args.model_type]
    
    # Check if we can skip (need explanations, importance, AND causal to skip)
    all_outputs_exist = True
    for model_type in expected_model_types:
        model_output_dir = OUTPUT_DIR / model_type
        explanations_path = model_output_dir / 'axp_explanations.parquet'
        importance_path = model_output_dir / 'feature_importance_axp.parquet'
        causal_path = model_output_dir / 'causal_importance.parquet'
        
        # Need all three to skip (causal may need to be regenerated with new grouped approach)
        if not (explanations_path.exists() and importance_path.exists() and causal_path.exists()):
            all_outputs_exist = False
            break
    
    if all_outputs_exist:
        logger.info(f"Step 7 outputs already exist locally for {COHORT_NAME}/{AGE_BAND}; skipping.")
        print(f"[SKIP] Step 7 outputs already exist locally for {COHORT_NAME}/{AGE_BAND}")
        
        # Still try to upload to S3 if not already there (idempotent upload)
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint
            
            s3_outputs = []
            for model_type in expected_model_types:
                model_output_dir = OUTPUT_DIR / model_type
                explanations_path = model_output_dir / 'axp_explanations.parquet'
                importance_path = model_output_dir / 'feature_importance_axp.parquet'
                causal_path = model_output_dir / 'causal_importance.parquet'
                
                if explanations_path.exists():
                    s3_explanations = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/axp_explanations.parquet"
                    if upload_file_to_s3(explanations_path, s3_explanations, logger):
                        s3_outputs.append(s3_explanations)
                
                if importance_path.exists():
                    s3_importance = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/feature_importance_axp.parquet"
                    if upload_file_to_s3(importance_path, s3_importance, logger):
                        s3_outputs.append(s3_importance)
                
                if causal_path.exists():
                    s3_causal = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/causal_importance.parquet"
                    if upload_file_to_s3(causal_path, s3_causal, logger):
                        s3_outputs.append(s3_causal)
            
            # Save checkpoint if outputs uploaded
            if s3_outputs:
                save_step_checkpoint(
                    step_name="8_ffa_analysis",
                    cohort=COHORT_NAME,
                    age_band=AGE_BAND,
                    metadata={"model_types_analyzed": expected_model_types},
                    output_paths=s3_outputs,
                    logger=logger,
                )
        except ImportError:
            pass  # S3 upload is optional
        
        return

    # Check S3 for existing outputs (idempotency - fallback if local doesn't exist)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists
        
        s3_output_paths = []
        for model_type in expected_model_types:
            s3_output_paths.extend([
                f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/axp_explanations.parquet",
                f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/feature_importance_axp.parquet",
            ])

        if check_step_outputs_exist(s3_output_paths, logger) or check_step_checkpoint_exists("8_ffa_analysis", COHORT_NAME, AGE_BAND, logger):
            logger.info(f"Step 7 outputs already exist in S3 for {COHORT_NAME}/{AGE_BAND}; downloading to local.")
            
            # Download from S3 to local
            try:
                import boto3
                s3_client = boto3.client("s3")
                S3_BUCKET = "pgxdatalake"
                
                for model_type in expected_model_types:
                    model_output_dir = OUTPUT_DIR / model_type
                    model_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download explanations (Parquet format)
                    s3_key = f"gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/axp_explanations.parquet"
                    explanations_path = model_output_dir / 'axp_explanations.parquet'
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(explanations_path))
                        logger.info(f"Downloaded {explanations_path} from S3")
                    except Exception as e:
                        logger.warning(f"Could not download {s3_key}: {e}")
                    
                    # Download importance (Parquet format)
                    s3_key = f"gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/feature_importance_axp.parquet"
                    importance_path = model_output_dir / 'feature_importance_axp.parquet'
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(importance_path))
                        logger.info(f"Downloaded {importance_path} from S3")
                    except Exception as e:
                        logger.warning(f"Could not download {s3_key}: {e}")
                    
                    # Skip downloading causal to allow regeneration with new grouped comparison method
                    # Causal analysis will be regenerated with optimized grouped approach
                    logger.info(f"Skipping causal download - will regenerate with grouped comparison method")
                
                # Check if causal analysis needs to be run (may need regeneration with grouped approach)
                causal_needs_regeneration = False
                for model_type in expected_model_types:
                    model_output_dir = OUTPUT_DIR / model_type
                    causal_path = model_output_dir / 'causal_importance.parquet'
                    if not causal_path.exists():
                        causal_needs_regeneration = True
                        break
                
                if causal_needs_regeneration:
                    logger.info(f"Causal analysis missing - will regenerate with grouped comparison method")
                    print(f"[INFO] Causal analysis will be regenerated with grouped comparison method")
                    # Continue to run analysis (causal will be regenerated)
                else:
                    logger.info(f"Step 7 outputs downloaded from S3; skipping regeneration.")
                    print(f"[SKIP] Step 7 outputs downloaded from S3 for {COHORT_NAME}/{AGE_BAND}")
                    return
            except Exception as e:
                logger.warning(f"Could not download from S3: {e}. Will regenerate outputs.")
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    workflow_start_time = time.time()
    logger.info(f"{'='*80}")
    logger.info("Starting Complete FFA Analysis Workflow")
    logger.info(f"Cohort: {COHORT_NAME}, Age Band: {AGE_BAND}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"{'='*80}")
    
    print("\n" + "="*80)
    print("Complete FFA Analysis Workflow")
    print(f"Cohort: {COHORT_NAME}, Age Band: {AGE_BAND}")
    print(f"Log file: {LOG_FILE}")
    print("="*80)
    
    if args.model_type == "all":
        # For "all", analyze best XGBoost (selected variant) only
        # Note: We only analyze the best XGBoost variant selected by final model training
        model_types = ['xgboost']  # Will load best_xgboost_model.json which contains the selected variant
    else:
        # If user specifies a specific type, use it (but best model might be different)
        model_types = [args.model_type]
    results = []
    
    for model_idx, model_type in enumerate(model_types, 1):
        logger.info(f"\nProcessing model {model_idx}/{len(model_types)}: {model_type}")
        try:
            result = run_full_analysis_for_model(model_type)
            if result:
                results.append(result)
                logger.info(f"Model {model_type} completed successfully")
            else:
                logger.warning(f"Model {model_type} returned no results")
        except Exception as e:
            logger.error(f"Failed to process {model_type}: {e}", exc_info=True)
            print(f"\n[ERROR] Failed to process {model_type}: {e}")
            continue
    
    # Print final summary
    total_workflow_time = time.time() - workflow_start_time
    logger.info(f"\n{'='*80}")
    logger.info("Analysis Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total workflow time: {total_workflow_time:.2f} seconds ({total_workflow_time/60:.2f} minutes)")
    
    print("\n" + "="*80)
    print("Analysis Summary")
    print("="*80)
    
    if results:
        summary_df = pd.DataFrame(results)
        print("\nResults:")
        print(summary_df.to_string(index=False))
        
        logger.info(f"Successfully analyzed {len(results)}/{len(model_types)} models")
        print(f"\n[OK] Analysis complete! Results saved to: {OUTPUT_DIR}")
    else:
        logger.error("No models were successfully analyzed")
        print("\n[ERROR] No models were successfully analyzed.")
        print("This step cannot complete without at least one model being analyzed.")
        logger.error("FFA analysis failed: No models were successfully analyzed")
        sys.exit(1)
    
    logger.info(f"{'='*80}")
    logger.info("Workflow Complete!")
    logger.info(f"{'='*80}")
    
    print("\n" + "="*80)
    print("Workflow Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

