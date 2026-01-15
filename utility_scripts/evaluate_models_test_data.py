#!/usr/bin/env python3
"""
Evaluate models on 2019 test data with calibration, performance metrics, 
feature importance, and SHAP analysis.

This script:
1. Loads best models from S3
2. Loads 2019 test data from S3
3. Evaluates performance (Recall, AUC-PR, etc.)
4. Calibrates models using isotonic calibration
5. Computes feature importance
6. Computes SHAP values
7. Saves results and combines with FFA analysis results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Try to import duckdb for memory-efficient parquet reading
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    print("[WARN] DuckDB not available. Install with: pip install duckdb")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname

# Cohorts and age bands
COHORTS: Dict[str, List[str]] = {
    'opioid_ed': ['13-24', '25-44', '45-54', '55-64'],
    'non_opioid_ed': ['65-74', '75-84', '85-94']
}

# S3 configuration
S3_BUCKET = "pgxdatalake"
# Auto-detect EC2 vs local
import os
if os.path.exists('/sys/hypervisor/uuid') or os.path.exists('/sys/class/dmi/id'):
    S3_PROFILE = None  # EC2 - use IAM role
else:
    S3_PROFILE = os.environ.get('AWS_PROFILE', 'mushin')  # Local - use profile


def download_from_s3(s3_path: str, local_path: Path, profile: Optional[str] = None) -> bool:
    """Download file from S3."""
    try:
        import subprocess
        cmd = ['aws', 's3', 'cp', s3_path, str(local_path)]
        if profile:
            cmd.extend(['--profile', profile])
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] Failed to download {s3_path}: {e}")
        return False


def upload_to_s3(local_path: Path, s3_path: str, profile: Optional[str] = None) -> bool:
    """Upload file to S3 and verify it was uploaded successfully."""
    try:
        import subprocess
        
        # Check file exists locally
        if not local_path.exists():
            print(f"  [ERROR] Local file does not exist: {local_path}")
            return False
        
        # Upload file
        cmd = ['aws', 's3', 'cp', str(local_path), s3_path]
        if profile:
            cmd.extend(['--profile', profile])
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  [ERROR] AWS CLI upload failed: {result.stderr}")
            return False
        
        # Verify file was uploaded successfully
        verify_cmd = ['aws', 's3', 'ls', s3_path]
        if profile:
            verify_cmd.extend(['--profile', profile])
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        
        if verify_result.returncode != 0:
            print(f"  [ERROR] File uploaded but verification failed: {s3_path}")
            return False
        
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to upload: {e}")
        return False


def load_test_data(cohort: str, age_band: str, profile: Optional[str] = None, use_duckdb: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Load 2019 test data from S3. Uses DuckDB for memory-efficient loading if available."""
    age_band_fname = age_band_to_fname(age_band)
    
    # Try S3 first
    s3_path = f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/inputs/model_test/final_features.parquet"
    
    # Local cache path
    local_cache = PROJECT_ROOT / "tmp" / "test_data" / f"{cohort}_{age_band_fname}_test.parquet"
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    
    # Download if not cached
    if not local_cache.exists():
        print(f"  Downloading test data from S3...")
        if not download_from_s3(s3_path, local_cache, profile):
            raise FileNotFoundError(f"Could not download test data from {s3_path}")
    
    # Use DuckDB for memory-efficient loading if available
    if use_duckdb and HAS_DUCKDB:
        print(f"  Loading data with DuckDB (memory-efficient)...")
        conn = duckdb.connect()
        
        # Read a small sample to get column names and types
        sample_df = conn.execute(f"SELECT * FROM read_parquet('{str(local_cache)}') LIMIT 1").df()
        all_cols = list(sample_df.columns)
        
        # Identify numeric columns (exclude target and key columns)
        exclude_cols = ['mi_person_key', 'target', 'person_key']
        numeric_cols = [col for col in all_cols 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(sample_df[col])]
        
        # Load target separately
        target_query = f"SELECT target FROM read_parquet('{str(local_cache)}')"
        y = conn.execute(target_query).df()['target'].astype(int)
        
        # Load numeric features
        numeric_cols_str = ', '.join([f'"{col}"' for col in numeric_cols])
        features_query = f"SELECT {numeric_cols_str} FROM read_parquet('{str(local_cache)}')"
        X = conn.execute(features_query).df()
        
        conn.close()
        
        print(f"  Loaded {len(X)} rows, {len(X.columns)} numeric features")
    else:
        # Fallback to pandas
        if use_duckdb:
            print(f"  [WARN] DuckDB not available, using pandas (may use more memory)...")
        
        df = pd.read_parquet(local_cache)
        
        if "target" not in df.columns:
            raise ValueError(f"'target' column not found in test data")
        
        y = df["target"].astype(int)
        X = df.drop(columns=["mi_person_key", "target", "person_key"], errors="ignore")
        
        # Keep only numeric columns (models expect numeric features)
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X = X[numeric_cols].copy()
    
    return X, y


def load_xgboost_model_from_json(model_json: Dict, X_test: pd.DataFrame):
    """Reconstruct XGBoost model from JSON with text dumps using SHAP TreeExplainer."""
    try:
        import xgboost as xgb
        import shap
    except ImportError:
        raise ImportError("XGBoost and SHAP required. Install with: pip install xgboost shap")
    
    feature_names = model_json.get('feature_names', [])
    trees = model_json.get('trees', [])
    
    if not feature_names:
        raise ValueError("Model JSON missing feature_names")
    if not trees:
        raise ValueError("Model JSON missing trees")
    
    # Align test data
    X_aligned = X_test.reindex(columns=feature_names, fill_value=0).astype('float32')
    
    # Reconstruct booster from tree dumps
    # We'll create a minimal booster by saving trees to a temporary file
    # and loading it back
    import tempfile
    import os
    
    # Create a temporary model file with trees in proper format
    # XGBoost can load from text dumps if we create the right structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        # Write trees in text dump format
        for tree in trees:
            if isinstance(tree, str):
                tmp_file.write(tree + '\n')
            elif isinstance(tree, dict) and 'tree_dump' in tree:
                tmp_file.write(tree['tree_dump'] + '\n')
        tmp_path = tmp_file.name
    
    try:
        # Create a new booster and try to load
        # Actually, XGBoost can't directly load from text dumps
        # We need to use SHAP's TreeExplainer which can work with text dumps
        # Or reconstruct the model by parsing trees manually
        
        # Alternative: Use SHAP TreeExplainer which can work with the model JSON
        # But TreeExplainer needs a booster object...
        
        # Best approach: Reconstruct booster by creating a minimal XGBoost model
        # and manually setting the trees
        # This is complex, so let's use a workaround:
        # Create a dummy model and use SHAP's model-agnostic explainer
        
        # Actually, let's try loading the JSON as if it were a proper XGBoost JSON
        # by creating a minimal valid structure
        xgb_json_structure = {
            "learner": {
                "learner_model_param": {
                    "num_feature": str(len(feature_names)),
                    "base_score": "0.5"
                },
                "objective": {
                    "name": "binary:logistic",
                    "reg_loss_param": {
                        "scale_pos_weight": "1"
                    }
                },
                "gradient_booster": {
                    "name": "gbtree",
                    "gbtree_model_param": {
                        "num_trees": str(len(trees)),
                        "size_leaf_vector": "0"
                    },
                    "model": {
                        "trees": []
                    }
                }
            },
            "version": [1, 0, 0]
        }
        
        # This won't work either - XGBoost needs proper tree structure
        
        # Final approach: Use SHAP's TreeExplainer with a workaround
        # We'll create predictions manually by parsing trees
        # But that's very complex...
        
        # Actually, the simplest: Check if we can use the explainer from FFA analysis
        # Or better: Try to load from a different format
        
        raise NotImplementedError(
            "XGBoost model JSON contains text dumps which cannot be directly loaded. "
            "Please use joblib models or implement tree parsing for predictions."
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_xgboost_model(cohort: str, age_band: str, profile: Optional[str] = None):
    """Load best XGBoost model from S3 (try joblib, .ubj, then JSON)."""
    try:
        import xgboost as xgb
        import joblib
    except ImportError:
        raise ImportError("XGBoost or joblib not installed")
    
    age_band_fname = age_band_to_fname(age_band)
    
    # Priority 1: Try joblib (XGBClassifier object)
    s3_joblib_paths = [
        f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/xgboost.joblib",
        f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/models/xgboost.joblib",
    ]
    
    local_joblib = PROJECT_ROOT / "tmp" / "models" / f"{cohort}_{age_band_fname}_xgboost.joblib"
    local_joblib.parent.mkdir(parents=True, exist_ok=True)
    
    for s3_path in s3_joblib_paths:
        if not local_joblib.exists():
            print(f"  Checking for joblib model: {s3_path}")
            if download_from_s3(s3_path, local_joblib, profile):
                print(f"  [OK] Loaded joblib model from S3")
                model = joblib.load(local_joblib)
                # Extract booster if it's an XGBClassifier
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
                    return booster, feature_names
                elif isinstance(model, xgb.Booster):
                    return model, None
                else:
                    # Try to get booster anyway
                    if hasattr(model, 'get_booster'):
                        return model.get_booster(), None
                    return model, None
    
    # Priority 2: Try .ubj binary format (native XGBoost booster)
    s3_ubj_paths = [
        f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/xgboost_model.ubj",
        f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/models/xgboost_model.ubj",
    ]
    
    local_ubj = PROJECT_ROOT / "tmp" / "models" / f"{cohort}_{age_band_fname}_xgboost.ubj"
    
    for s3_path in s3_ubj_paths:
        if not local_ubj.exists():
            print(f"  Checking for .ubj binary model: {s3_path}")
            if download_from_s3(s3_path, local_ubj, profile):
                print(f"  [OK] Loaded .ubj binary model from S3")
                booster = xgb.Booster()
                booster.load_model(str(local_ubj))
                # Try to get feature names from booster
                feature_names = booster.feature_names if hasattr(booster, 'feature_names') and booster.feature_names else None
                return booster, feature_names
    
    # Priority 3: Try JSON (but it has text dumps, not directly loadable)
    print(f"  Joblib and .ubj models not found, trying JSON...")
    s3_json_path = f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_xgboost_model.json"
    local_json = PROJECT_ROOT / "tmp" / "models" / f"{cohort}_{age_band_fname}_best_xgboost.json"
    
    if not local_json.exists():
        print(f"  Downloading model JSON from S3...")
        if not download_from_s3(s3_json_path, local_json, profile):
            raise FileNotFoundError(
                f"Could not download XGBoost model. Tried:\n"
                f"  - {s3_joblib_paths[0]}\n"
                f"  - {s3_ubj_paths[0]}\n"
                f"  - {s3_json_path}\n"
                f"Please ensure models are synced to S3."
            )
    
    # Load JSON to get feature names
    with open(local_json, 'r') as f:
        model_json = json.load(f)
    
    feature_names = model_json.get('feature_names', [])
    
    # JSON has text dumps - can't directly load, but we can use SHAP TreeExplainer
    # For now, raise informative error
    raise ValueError(
        f"XGBoost model JSON contains text dumps which cannot be directly loaded for predictions. "
        f"Please ensure joblib or .ubj models are synced to S3:\n"
        f"  - {s3_joblib_paths[0]}\n"
        f"  - {s3_ubj_paths[0]}\n"
        f"These formats are saved by run_final_model.py but may need to be synced to S3."
    )


def load_catboost_model(cohort: str, age_band: str, profile: Optional[str] = None):
    """Load best CatBoost model from S3."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    age_band_fname = age_band_to_fname(age_band)
    
    # Try .cbm binary first (preferred)
    s3_path_cbm = f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.cbm"
    local_cache_cbm = PROJECT_ROOT / "tmp" / "models" / f"{cohort}_{age_band_fname}_best_catboost.cbm"
    local_cache_cbm.parent.mkdir(parents=True, exist_ok=True)
    
    if not local_cache_cbm.exists():
        if download_from_s3(s3_path_cbm, local_cache_cbm, profile):
            model = CatBoostClassifier()
            model.load_model(str(local_cache_cbm))
            return model
    
    # Fallback to JSON
    s3_path_json = f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.json"
    local_cache_json = PROJECT_ROOT / "tmp" / "models" / f"{cohort}_{age_band_fname}_best_catboost.json"
    
    if not local_cache_json.exists():
        if not download_from_s3(s3_path_json, local_cache_json, profile):
            raise FileNotFoundError(f"Could not download CatBoost model")
    
    model = CatBoostClassifier()
    model.load_model(str(local_cache_json), format='json')
    return model


def prepare_xgboost_predictions(booster, feature_names: List[str], X_test: pd.DataFrame):
    """Prepare XGBoost booster and DMatrix for predictions."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    # Align test data to model's feature space
    X_aligned = X_test.reindex(columns=feature_names, fill_value=0).astype('float32')
    
    # Create DMatrix
    dtest = xgb.DMatrix(X_aligned, feature_names=feature_names)
    
    return dtest, feature_names


def calibrate_model_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calibrate predictions using isotonic regression."""
    from sklearn.isotonic import IsotonicRegression
    
    # Fit isotonic regression on predictions
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_proba, y_true)
    
    # Calibrate probabilities
    y_proba_calibrated = iso_reg.transform(y_proba)
    
    # Calculate optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_calibrated)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return y_proba_calibrated, optimal_threshold


def compute_feature_importance_xgboost(booster, feature_names: List[str]) -> pd.DataFrame:
    """Compute feature importance from XGBoost model."""
    import xgboost as xgb
    
    # Get importance scores
    importance_gain = booster.get_score(importance_type='gain')
    importance_weight = booster.get_score(importance_type='weight')
    
    # Handle case where booster uses f0, f1, ... format vs feature names
    # Map feature indices to names
    gain_values = []
    weight_values = []
    
    for i, feat_name in enumerate(feature_names):
        # Try both f{i} format and feature name
        gain_val = importance_gain.get(f'f{i}', importance_gain.get(feat_name, 0))
        weight_val = importance_weight.get(f'f{i}', importance_weight.get(feat_name, 0))
        gain_values.append(gain_val)
        weight_values.append(weight_val)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': gain_values,
        'importance_weight': weight_values
    })
    
    # Normalize (handle division by zero)
    gain_sum = importance_df['importance_gain'].sum()
    weight_sum = importance_df['importance_weight'].sum()
    
    if gain_sum > 0:
        importance_df['importance_gain_norm'] = importance_df['importance_gain'] / gain_sum
    else:
        importance_df['importance_gain_norm'] = 0.0
    
    if weight_sum > 0:
        importance_df['importance_weight_norm'] = importance_df['importance_weight'] / weight_sum
    else:
        importance_df['importance_weight_norm'] = 0.0
    
    return importance_df.sort_values('importance_gain', ascending=False)


def compute_feature_importance_catboost(model) -> pd.DataFrame:
    """Compute feature importance from CatBoost model."""
    feature_names = model.feature_names_
    importances = model.get_feature_importance()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Normalize
    importance_df['importance_norm'] = importance_df['importance'] / importance_df['importance'].sum()
    
    return importance_df.sort_values('importance', ascending=False)


def sample_test_data_duckdb(parquet_path: Path, feature_names: List[str], n_samples: int, include_target: bool = False, random_seed: int = 42) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Sample rows from parquet file using DuckDB without loading full dataset.
    
    Returns:
        X_sample: DataFrame with features
        y_sample: Series with target (if include_target=True), else None
    """
    if not HAS_DUCKDB:
        raise ImportError("DuckDB required for memory-efficient sampling")
    
    conn = duckdb.connect()
    
    # Use ORDER BY RANDOM() with LIMIT for efficient random sampling
    # This is more memory-efficient than loading full dataset
    feature_cols_str = ', '.join([f'"{col}"' for col in feature_names])
    
    # Get total row count first
    total_rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{str(parquet_path)}')").fetchone()[0]
    
    if n_samples >= total_rows:
        # No need to sample
        if include_target:
            query = f"SELECT {feature_cols_str}, target FROM read_parquet('{str(parquet_path)}')"
        else:
            query = f"SELECT {feature_cols_str} FROM read_parquet('{str(parquet_path)}')"
    else:
        # Sample using ORDER BY RANDOM() - DuckDB optimizes this
        if include_target:
            query = f"""
                SELECT {feature_cols_str}, target
                FROM read_parquet('{str(parquet_path)}')
                ORDER BY RANDOM()
                LIMIT {n_samples}
            """
        else:
            query = f"""
                SELECT {feature_cols_str}
                FROM read_parquet('{str(parquet_path)}')
                ORDER BY RANDOM()
                LIMIT {n_samples}
            """
    
    result_df = conn.execute(query).df()
    conn.close()
    
    if include_target:
        y_sample = result_df['target'].astype(int)
        X_sample = result_df.drop(columns=['target'])
        return X_sample, y_sample
    else:
        return result_df, None


def compute_shap_values_xgboost(booster, X_test: pd.DataFrame, feature_names: List[str], n_samples: Optional[int] = None, parquet_path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for XGBoost model. Uses DuckDB for memory-efficient sampling if available."""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    import xgboost as xgb
    
    # Sample if requested - use DuckDB if parquet_path provided and DuckDB available
    if n_samples and len(X_test) > n_samples:
        if parquet_path and HAS_DUCKDB:
            print(f"  Sampling {n_samples} rows using DuckDB (memory-efficient)...")
            X_sample, _ = sample_test_data_duckdb(parquet_path, feature_names, n_samples, include_target=False)
            sample_idx = np.arange(len(X_sample))  # New index for sampled data
        else:
            sample_idx = np.random.choice(len(X_test), size=n_samples, replace=False)
            X_sample = X_test.iloc[sample_idx].copy()
    else:
        X_sample = X_test.copy()
        sample_idx = np.arange(len(X_test))
    
    # Align to model feature space
    X_aligned = X_sample.reindex(columns=feature_names, fill_value=0)
    dtest = xgb.DMatrix(X_aligned, feature_names=feature_names)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dtest)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    # Create DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index if hasattr(X_sample, 'index') else np.arange(len(X_sample)))
    
    return shap_values, shap_df


def compute_shap_values_catboost(model, X_test: pd.DataFrame, y_test: pd.Series, n_samples: Optional[int] = None, parquet_path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for CatBoost model. Uses DuckDB for memory-efficient sampling if available."""
    try:
        import shap
        from catboost import Pool
    except ImportError:
        raise ImportError("SHAP or CatBoost not installed")
    
    # Sample if requested - use DuckDB if parquet_path provided and DuckDB available
    if n_samples and len(X_test) > n_samples:
        if parquet_path and HAS_DUCKDB:
            print(f"  Sampling {n_samples} rows using DuckDB (memory-efficient)...")
            feature_names = model.feature_names_
            X_sample, y_sample = sample_test_data_duckdb(parquet_path, feature_names, n_samples, include_target=True)
            sample_idx = np.arange(len(X_sample))  # New index for sampled data
        else:
            sample_idx = np.random.choice(len(X_test), size=n_samples, replace=False)
            X_sample = X_test.iloc[sample_idx].copy()
            y_sample = y_test.iloc[sample_idx].copy()
    else:
        X_sample = X_test.copy()
        y_sample = y_test.copy()
        sample_idx = np.arange(len(X_test))
    
    # Get categorical feature indices from model if available
    # CatBoost models may have categorical features that need to be marked
    cat_feature_indices = None
    if hasattr(model, 'get_cat_feature_indices'):
        try:
            cat_feature_indices = model.get_cat_feature_indices()
        except:
            pass
    
    # If model has categorical features, we need to mark them in the Pool
    # But since our features are numeric (from final_features), we'll create Pool without cat features
    # The error suggests the model expects categorical but data is numeric
    # Solution: Create Pool without specifying cat_features (treat all as numeric)
    try:
        # Try without categorical features first
        pool = Pool(X_sample, y_sample)
        shap_values = model.get_feature_importance(type="ShapValues", data=pool)
    except Exception as e:
        if "Categorical" in str(e):
            # Model expects categorical but we have numeric
            # Use SHAP TreeExplainer instead which handles this better
            print(f"  Note: Using SHAP TreeExplainer (model has categorical features)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Create DataFrame
            feature_names = model.feature_names_
            shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index if hasattr(X_sample, 'index') else np.arange(len(X_sample)))
            return shap_values, shap_df
        else:
            raise
    
    shap_values = np.array(shap_values)
    
    # Handle shape: (n_samples, n_features + 1) or (n_samples, n_classes, n_features + 1)
    if shap_values.ndim == 2:
        shap_values = shap_values[:, :-1]  # Drop expected value column
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, :-1].mean(axis=1)  # Collapse classes
    
    # Create DataFrame
    feature_names = model.feature_names_
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index if hasattr(X_sample, 'index') else np.arange(len(X_sample)))
    
    return shap_values, shap_df


def evaluate_model(
    cohort: str,
    age_band: str,
    model_type: str = 'xgboost',
    n_shap_samples: Optional[int] = 1000,
    profile: Optional[str] = None
) -> Dict:
    """Evaluate a single model on 2019 test data."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_type.upper()} model for {cohort}/{age_band}")
    print(f"{'='*80}")
    
    age_band_fname = age_band_to_fname(age_band)
    
    # Get parquet path for DuckDB sampling
    local_cache = PROJECT_ROOT / "tmp" / "test_data" / f"{cohort}_{age_band_fname}_test.parquet"
    parquet_path = local_cache if local_cache.exists() else None
    
    # Load test data
    print("\n[1/6] Loading 2019 test data...")
    X_test, y_test = load_test_data(cohort, age_band, profile, use_duckdb=True)
    print(f"  Test data: {len(X_test)} patients, {len(X_test.columns)} features")
    print(f"  Target distribution: {y_test.value_counts().to_dict()}")
    
    # Load model
    print(f"\n[2/6] Loading {model_type} model...")
    if model_type == 'xgboost':
        booster, feature_names_from_model = load_xgboost_model(cohort, age_band, profile)
        
        # Get feature names
        if feature_names_from_model is not None:
            feature_names = list(feature_names_from_model)
        else:
            # Try to get from booster
            if hasattr(booster, 'feature_names') and booster.feature_names:
                feature_names = booster.feature_names
            else:
                # Fallback: use test data columns (aligned)
                feature_names = list(X_test.columns)
        
        dtest, feature_names = prepare_xgboost_predictions(booster, feature_names, X_test)
        
        # Get predictions
        y_proba_raw = booster.predict(dtest, output_margin=False)
        y_pred_raw = (y_proba_raw >= 0.5).astype(int)
        
        model_obj = booster
    elif model_type == 'catboost':
        model = load_catboost_model(cohort, age_band, profile)
        feature_names = model.feature_names_
        
        # Align test data
        X_test_aligned = X_test.reindex(columns=feature_names, fill_value=0)
        
        # Get predictions
        y_proba_raw = model.predict_proba(X_test_aligned)[:, 1]
        y_pred_raw = (y_proba_raw >= 0.5).astype(int)
        
        model_obj = model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Calibrate predictions
    print(f"\n[3/6] Calibrating predictions...")
    y_proba_calibrated, optimal_threshold = calibrate_model_predictions(y_test.values, y_proba_raw)
    y_pred_calibrated = (y_proba_calibrated >= optimal_threshold).astype(int)
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Before calibration - Positive predictions: {y_pred_raw.sum()}")
    print(f"  After calibration - Positive predictions: {y_pred_calibrated.sum()}")
    
    # Calculate performance metrics
    print(f"\n[4/6] Calculating performance metrics...")
    metrics = {
        'cohort': cohort,
        'age_band': age_band,
        'model_type': model_type,
        'n_test_samples': len(y_test),
        'n_features': len(feature_names),
        'optimal_threshold': float(optimal_threshold),
        
        # Raw predictions (before calibration)
        'recall_raw': float(recall_score(y_test, y_pred_raw, zero_division=0)),
        'precision_raw': float(precision_score(y_test, y_pred_raw, zero_division=0)),
        'f1_raw': float(f1_score(y_test, y_pred_raw, zero_division=0)),
        'accuracy_raw': float(accuracy_score(y_test, y_pred_raw)),
        'roc_auc_raw': float(roc_auc_score(y_test, y_proba_raw)),
        'pr_auc_raw': float(average_precision_score(y_test, y_proba_raw)),
        'logloss_raw': float(log_loss(y_test, y_proba_raw)),
        'brier_raw': float(brier_score_loss(y_test, y_proba_raw)),
        
        # Calibrated predictions
        'recall_calibrated': float(recall_score(y_test, y_pred_calibrated, zero_division=0)),
        'precision_calibrated': float(precision_score(y_test, y_pred_calibrated, zero_division=0)),
        'f1_calibrated': float(f1_score(y_test, y_pred_calibrated, zero_division=0)),
        'accuracy_calibrated': float(accuracy_score(y_test, y_pred_calibrated)),
        'roc_auc_calibrated': float(roc_auc_score(y_test, y_proba_calibrated)),
        'pr_auc_calibrated': float(average_precision_score(y_test, y_proba_calibrated)),
        'logloss_calibrated': float(log_loss(y_test, y_proba_calibrated)),
        'brier_calibrated': float(brier_score_loss(y_test, y_proba_calibrated)),
    }
    
    # Confusion matrix
    cm_raw = confusion_matrix(y_test, y_pred_raw)
    cm_calibrated = confusion_matrix(y_test, y_pred_calibrated)
    
    metrics['confusion_matrix_raw'] = cm_raw.tolist()
    metrics['confusion_matrix_calibrated'] = cm_calibrated.tolist()
    
    print(f"  Recall (calibrated): {metrics['recall_calibrated']:.4f}")
    print(f"  Precision (calibrated): {metrics['precision_calibrated']:.4f}")
    print(f"  F1 (calibrated): {metrics['f1_calibrated']:.4f}")
    print(f"  ROC-AUC (calibrated): {metrics['roc_auc_calibrated']:.4f}")
    print(f"  PR-AUC (calibrated): {metrics['pr_auc_calibrated']:.4f}")
    
    # Feature importance
    print(f"\n[5/6] Computing feature importance...")
    if model_type == 'xgboost':
        importance_df = compute_feature_importance_xgboost(booster, feature_names)
    else:
        importance_df = compute_feature_importance_catboost(model_obj)
    
    print(f"  Top 10 features:")
    for idx, row in importance_df.head(10).iterrows():
        imp_col = 'importance_gain_norm' if model_type == 'xgboost' else 'importance_norm'
        print(f"    {row['feature']}: {row[imp_col]:.6f}")
    
    # SHAP analysis
    print(f"\n[6/6] Computing SHAP values...")
    if model_type == 'xgboost':
        shap_values, shap_df = compute_shap_values_xgboost(booster, X_test, feature_names, n_shap_samples, parquet_path)
    else:
        shap_values, shap_df = compute_shap_values_catboost(model_obj, X_test, y_test, n_shap_samples, parquet_path)
    
    # Global SHAP importance
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"  Computed SHAP values for {len(shap_df)} samples")
    print(f"  Top 10 features by SHAP importance:")
    for idx, row in shap_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.6f}")
    
    return {
        'metrics': metrics,
        'feature_importance': importance_df,
        'shap_importance': shap_importance,
        'shap_values': shap_df,
        'predictions': {
            'y_true': y_test.values,
            'y_proba_raw': y_proba_raw,
            'y_proba_calibrated': y_proba_calibrated,
            'y_pred_raw': y_pred_raw,
            'y_pred_calibrated': y_pred_calibrated
        }
    }


def save_results(results: Dict, cohort: str, age_band: str, model_type: str, output_dir: Path, profile: Optional[str] = None, upload_to_s3_flag: bool = True):
    """Save evaluation results and optionally upload to S3."""
    age_band_fname = age_band_to_fname(age_band)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine profile
    if profile is None:
        profile = S3_PROFILE
    
    # Save metrics
    metrics_path = output_dir / f"{cohort}_{age_band_fname}_{model_type}_test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\n  Saved metrics to: {metrics_path}")
    
    # Upload metrics to S3
    if upload_to_s3_flag:
        s3_metrics_path = f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/model_evaluation/{metrics_path.name}"
        print(f"  Uploading metrics to S3...")
        if upload_to_s3(metrics_path, s3_metrics_path, profile):
            print(f"  [OK] Uploaded metrics to S3")
        else:
            print(f"  [WARN] Failed to upload metrics to S3")
    
    # Save feature importance
    importance_path = output_dir / f"{cohort}_{age_band_fname}_{model_type}_test_feature_importance.csv"
    results['feature_importance'].to_csv(importance_path, index=False)
    print(f"  Saved feature importance to: {importance_path}")
    
    # Save SHAP importance
    shap_importance_path = output_dir / f"{cohort}_{age_band_fname}_{model_type}_test_shap_importance.csv"
    results['shap_importance'].to_csv(shap_importance_path, index=False)
    print(f"  Saved SHAP importance to: {shap_importance_path}")
    
    # Save SHAP values (parquet for efficiency)
    shap_values_path = output_dir / f"{cohort}_{age_band_fname}_{model_type}_test_shap_values.parquet"
    results['shap_values'].to_parquet(shap_values_path, index=True)
    print(f"  Saved SHAP values to: {shap_values_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame(results['predictions'])
    predictions_path = output_dir / f"{cohort}_{age_band_fname}_{model_type}_test_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    print(f"  Saved predictions to: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on 2019 test data with calibration and SHAP analysis"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=list(COHORTS.keys()),
        help="Cohort name"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Age band (e.g., 13-24)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['xgboost', 'catboost', 'both'],
        default='both',
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--n-shap-samples",
        type=int,
        default=1000,
        help="Number of samples for SHAP analysis (default: 1000, None for all)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile (default: auto-detect)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: 8_ffa_analysis/results/model_evaluation)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading results to S3 (default: upload to S3)"
    )
    
    args = parser.parse_args()
    
    # Determine profile
    profile = args.profile if args.profile else S3_PROFILE
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "8_ffa_analysis" / "results" / "model_evaluation"
    
    # Determine cohorts to process
    if args.cohort and args.age_band:
        cohorts_to_process = [(args.cohort, args.age_band)]
    else:
        # Process all cohorts
        cohorts_to_process = []
        for cohort, age_bands in COHORTS.items():
            for age_band in age_bands:
                cohorts_to_process.append((cohort, age_band))
    
    # Determine model types
    if args.model_type == 'both':
        model_types = ['xgboost', 'catboost']
    else:
        model_types = [args.model_type]
    
    # Process each cohort/age_band
    all_results = []
    upload_to_s3_flag = not args.no_upload
    for cohort, age_band in cohorts_to_process:
        for model_type in model_types:
            try:
                results = evaluate_model(
                    cohort=cohort,
                    age_band=age_band,
                    model_type=model_type,
                    n_shap_samples=args.n_shap_samples if args.n_shap_samples > 0 else None,
                    profile=profile
                )
                
                save_results(results, cohort, age_band, model_type, output_dir, profile=profile, upload_to_s3_flag=upload_to_s3_flag)
                all_results.append(results['metrics'])
                
            except Exception as e:
                print(f"\n[ERROR] Failed to evaluate {model_type} for {cohort}/{age_band}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save combined summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_dir / "test_evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"Saved combined summary to: {summary_path}")
        print(f"{'='*80}")
        
        # Upload summary to S3
        if upload_to_s3_flag:
            s3_summary_path = f"s3://{S3_BUCKET}/gold/final_model/model_evaluation_summary.csv"
            print(f"\nUploading summary to S3...")
            if upload_to_s3(summary_path, s3_summary_path, profile):
                print(f"[OK] Uploaded summary to S3: {s3_summary_path}")
            else:
                print(f"[WARN] Failed to upload summary to S3")
        
        # Print summary
        print("\nSummary of Test Set Performance (Calibrated):")
        print(summary_df[['cohort', 'age_band', 'model_type', 'recall_calibrated', 'precision_calibrated', 
                          'pr_auc_calibrated', 'roc_auc_calibrated']].to_string(index=False))


if __name__ == "__main__":
    main()
