"""
Feature Importance Model Training Utilities
Functions for training, prediction, and feature importance extraction
Supports CatBoost, XGBoost, and XGBoost RF mode
"""

import sys
import site

# Ensure user site-packages (where xgboost/catboost may be installed) are visible
try:
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidate_paths = [user_site]
    else:
        candidate_paths = list(user_site)
    for p in candidate_paths:
        if p and p not in sys.path:
            sys.path.append(p)
except Exception:
    # If anything goes wrong here, fall back to default sys.path behavior
    pass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings
import logging

warnings.filterwarnings("ignore")

# Import from common helpers
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_helpers.common_imports import *
from py_helpers.env_utils import get_xgb_cpu_nthread

logger = logging.getLogger(__name__)


# ============================================================================
# Unified Permutation Importance (for fair comparison across all models)
# ============================================================================

def get_permutation_importance(model, X_test, y_test, feature_names, scoring='recall', n_repeats=5, random_state=42):
    """
    Calculate permutation-based feature importance for any model.
    This provides fair comparison across all model types.
    
    Args:
        model: Trained model (any sklearn-compatible model, including CatBoost)
        X_test: Test features (DataFrame or array)
        y_test: Test labels (array)
        feature_names: List of feature names
        scoring: Scoring function ('recall', 'log_loss', or callable)
        n_repeats: Number of permutation repeats (default: 5)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: ['feature', 'importance']
    """
    # Check if this is a CatBoost model
    is_catboost = isinstance(model, CatBoostClassifier)

    if is_catboost:
        # Delegate to the CatBoost-specific helper, which correctly aligns
        # feature names with the model's trained feature set and handles
        # any constant-feature pruning that occurred during training.
        return get_importance_catboost(
            model,
            feature_names,
            X_test=X_test,
            y_test=y_test,
            scoring=scoring,
        )
    else:
        # For other models (e.g., XGBoost), implement permutation importance
        # manually so we can log fine-grained progress and control memory use.
        # Convert to numpy array
        if isinstance(X_test, pd.DataFrame):
            X_test_for_perm = X_test.values
        else:
            X_test_for_perm = X_test

        # Optional: downsample the permutation-importance evaluation set to
        # control memory usage, especially for very large holdout cohorts.
        # This is controlled via the PGX_PERM_MAX_ROWS env var; if unset,
        # we use the full holdout (backwards compatible).
        try:
            max_rows_env = os.getenv("PGX_PERM_MAX_ROWS")
            if max_rows_env is not None and int(max_rows_env) > 0:
                max_rows = int(max_rows_env)
                n_rows = X_test_for_perm.shape[0]
                if n_rows > max_rows:
                    rng = np.random.RandomState(random_state)
                    idx = rng.choice(n_rows, size=max_rows, replace=False)
                    X_test_for_perm = X_test_for_perm[idx]
                    if isinstance(y_test, (pd.Series, pd.DataFrame)):
                        y_test = y_test.values
                    y_test = y_test[idx]
        except Exception:
            # Best-effort safeguard; if anything goes wrong, fall back to full data
            pass

        # Define scoring function
        if scoring == 'recall':
            from sklearn.metrics import recall_score, make_scorer
            scorer = make_scorer(recall_score, zero_division=0)
        elif scoring == 'log_loss':
            from sklearn.metrics import log_loss, make_scorer
            def neg_log_loss(y_true, y_pred_proba):
                return -log_loss(y_true, y_pred_proba)
            scorer = make_scorer(neg_log_loss, needs_proba=True)
        else:
            scorer = scoring

        # Manual permutation importance with progress logging.
        X_perm = X_test_for_perm.copy()
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_arr = y_test.values
        else:
            y_arr = np.asarray(y_test)

        n_rows, n_features = X_perm.shape
        rng = np.random.RandomState(random_state)

        # Baseline score on the (possibly downsampled) eval set.
        baseline_score = scorer(model, X_perm, y_arr)

        logger.info(
            "Permutation importance: baseline score=%.6f on %d rows × %d features",
            baseline_score,
            n_rows,
            n_features,
        )

        importances = np.zeros(n_features, dtype=float)

        # Choose a logging interval so users can see progress.
        log_every = max(1, n_features // 20)  # ~5% increments

        for j in range(n_features):
            # Save original column
            original_col = X_perm[:, j].copy()

            scores = []
            for _ in range(n_repeats):
                # Permute this column in place
                perm_idx = rng.permutation(n_rows)
                X_perm[:, j] = original_col[perm_idx]

                score_perm = scorer(model, X_perm, y_arr)
                scores.append(score_perm)

            # Restore original column
            X_perm[:, j] = original_col

            mean_score = np.mean(scores)
            importances[j] = baseline_score - mean_score

            if (j + 1) % log_every == 0 or j == n_features - 1:
                logger.info(
                    "Permutation importance progress: %d/%d features (%.1f%%)",
                    j + 1,
                    n_features,
                    100.0 * (j + 1) / n_features,
                )

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df


# ============================================================================
# CatBoost
# ============================================================================

def train_catboost(X_train, y_train, params):
    """
    Train CatBoost model (Python) - uses categorical features
    
    Args:
        X_train: Training features (DataFrame with categorical columns)
        y_train: Training labels (binary 0/1)
        params: Dictionary of CatBoost parameters
        
    Returns:
        Trained CatBoost model
    """
    # Filter out constant features (all same value) - CatBoost requires at least one non-constant feature
    if isinstance(X_train, pd.DataFrame):
        # Find constant features (zero variance)
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X_train = X_train.drop(columns=constant_features)
            if len(X_train.columns) == 0:
                raise ValueError("All features are constant. Cannot train CatBoost model.")
    
    # Identify categorical columns (string/object type)
    # CatBoost requires all categorical columns to be explicitly listed
    # For our feature importance, all item_* columns are categorical (item names or empty strings)
    categorical_features = [col for col in X_train.columns if col.startswith('item_')]
    
    # Convert to indices (CatBoost uses 0-based indices)
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_features] if categorical_features else None
    
    # Create CatBoost Pool
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_indices  # All item_* columns are categorical
    )
    
    # Decide CPU vs GPU:
    # 1) If params include task_type/devices, honor those first.
    # 2) Else, use PGX_CATBOOST_TASK_TYPE env var (CPU/GPU).
    # 3) Else, default to GPU for backwards compatibility.
    task_type_param = params.get("task_type")
    devices_param = params.get("devices")

    if task_type_param is not None:
        task_type = str(task_type_param).upper()
    else:
        task_type_env = os.getenv("PGX_CATBOOST_TASK_TYPE")
        # Default to CPU when nothing is specified, to avoid unexpected
        # CUDA/driver issues on machines without a compatible GPU stack.
        task_type = task_type_env.upper() if task_type_env else "CPU"

    use_gpu = task_type == "GPU"

    # Set up CatBoost parameters
    catboost_params = {
        'iterations': params.get('iterations', 100),
        'learning_rate': params.get('learning_rate', 0.1),
        'depth': params.get('depth', 6),
        'loss_function': 'Logloss',
        'eval_metric': 'Recall',
        'verbose': params.get('verbose', False),
        'random_seed': params.get('random_seed', 42),
        'allow_writing_files': False,  # Disable file writing for parallel processing
    }

    if use_gpu:
        catboost_params.update({
            'task_type': 'GPU',
            'devices': str(devices_param) if devices_param is not None else '0',
        })
    else:
        catboost_params.update({
            'task_type': 'CPU',
        })
    
    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, verbose=False)
    
    return model


def predict_catboost(model, X_test):
    """Predict with CatBoost - returns binary predictions"""
    # All item_* columns are categorical (item names or empty strings)
    categorical_features = [col for col in X_test.columns if col.startswith('item_')]
    
    cat_indices = [X_test.columns.get_loc(col) for col in categorical_features] if categorical_features else None
    
    test_pool = Pool(
        data=X_test,
        cat_features=cat_indices  # All item_* columns are categorical
    )
    
    pred_proba = model.predict_proba(test_pool)[:, 1]
    pred = (pred_proba > 0.5).astype(int)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in CatBoost predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_catboost(model, X_test):
    """Predict probabilities with CatBoost"""
    # Treat all item_* columns as categorical, matching train_catboost / predict_catboost
    categorical_features = [col for col in X_test.columns if col.startswith('item_')]
    cat_indices = [X_test.columns.get_loc(col) for col in categorical_features] if categorical_features else None

    test_pool = Pool(
        data=X_test,
        cat_features=cat_indices
    )
    
    pred_proba = model.predict_proba(test_pool)[:, 1]
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in CatBoost probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_catboost(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from CatBoost model using CatBoost's built-in method.
    Uses CatBoost's get_feature_importance with Pool for permutation-based calculation.
    """
    if X_test is not None and y_test is not None:
        # Ensure X_test is a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        # Ensure y_test is a numpy array and has correct length
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
        
        # Verify lengths match
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test and y_test length mismatch: X_test has {len(X_test)} rows, "
                f"y_test has {len(y_test)} values"
            )
        
        # Get the features that the model was actually trained on
        # CatBoost models store feature names in feature_names_ attribute
        if hasattr(model, 'feature_names_'):
            model_feature_names = model.feature_names_
        else:
            # Fallback: use feature_names passed in (should match training features)
            model_feature_names = feature_names
        
        # Align X_test to only include features the model was trained on
        # This is important because train_catboost removes constant features per split
        X_test_aligned = X_test[model_feature_names].copy()
        
        # Use CatBoost's built-in permutation importance
        # Create Pool for test data
        categorical_features = [col for col in X_test_aligned.columns if col.startswith('item_')]
        cat_indices = [X_test_aligned.columns.get_loc(col) for col in categorical_features] if categorical_features else None
        
        test_pool = Pool(
            data=X_test_aligned,
            label=y_test,
            cat_features=cat_indices
        )
        
        # Get permutation-based importance using PredictionValuesChange (permutation-based)
        # This is CatBoost's built-in permutation importance
        importance = model.get_feature_importance(
            data=test_pool,
            type='PredictionValuesChange'  # Permutation-based importance
        )
        
        # Verify lengths match
        if len(importance) != len(model_feature_names):
            raise ValueError(
                f"Feature importance length ({len(importance)}) doesn't match model features length ({len(model_feature_names)}). "
                f"X_test had {len(X_test.columns)} columns, but model was trained on {len(model_feature_names)} features."
            )
        
        importance_df = pd.DataFrame({
            'feature': model_feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        # Fallback to native importance (for backward compatibility)
        importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


# ============================================================================
# XGBoost
# ============================================================================


def _xgb_device_params(params: dict) -> dict:
    """
    Normalize and extract XGBoost device/tree_method settings.

    - For newer XGBoost (2.x), GPU is enabled via device='cuda' with
      tree_method='hist'. The legacy 'gpu_hist' value is no longer valid.
    - We therefore normalize any 'gpu_hist' value to 'hist' but preserve
      the user's explicit 'device' and 'predictor' choices.
    """
    overrides: dict = {}

    tree_method = params.get("tree_method")
    if tree_method:
        if tree_method == "gpu_hist":
            # Normalize legacy GPU setting to the modern API.
            overrides["tree_method"] = "hist"
        else:
            overrides["tree_method"] = tree_method

    # Pass through predictor / device unchanged (e.g., 'gpu_predictor', 'cuda')
    if "predictor" in params:
        overrides["predictor"] = params["predictor"]
    if "device" in params:
        overrides["device"] = params["device"]

    return overrides


def train_xgboost(X_train, y_train, params):
    """Train XGBoost model (gradient boosting mode, GPU if available)."""
    nthread = get_xgb_cpu_nthread()
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "n_estimators": params.get("n_estimators", 100),
        "subsample": params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "random_state": params.get("random_seed", 42),
        # Use environment-aware CPU threads; outer MC-CV may also parallelize.
        "n_jobs": nthread,
        "verbosity": 0,
    }

    # Merge in device / tree_method settings based on GPU detection and user prefs.
    xgb_params.update(_xgb_device_params(params))

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    return model


def train_xgboost_rf(X_train, y_train, params):
    """Train XGBoost in Random Forest mode (GPU if available)."""
    n_features = X_train.shape[1]
    max_features = params.get("max_features", None)
    if max_features is None:
        max_features = int(np.sqrt(n_features))

    nthread = get_xgb_cpu_nthread()
    xgb_rf_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "n_estimators": params.get("n_estimators", 100),
        "subsample": params.get("subsample", 0.8),  # RF typically uses subsampling
        "colsample_bytree": max_features / n_features,  # RF-style feature sampling
        "random_state": params.get("random_seed", 42),
        "n_jobs": nthread,
        "verbosity": 0,
        "booster": "gbtree",
        # Default CPU tree method for RF; _xgb_device_params() will override if GPU works.
        "tree_method": "hist",
    }

    xgb_rf_params.update(_xgb_device_params(params))

    model = xgb.XGBClassifier(**xgb_rf_params)
    model.fit(X_train, y_train)
    return model


def _xgb_prepare_dmatrix_input(X):
    """
    Helper to convert pandas objects into a form compatible with xgboost.DMatrix.
    """
    # Preserve DataFrame structure so feature names are kept for XGBoost's
    # internal feature validation. DMatrix can consume pandas directly.
    if isinstance(X, pd.Series):
        return X.to_frame()
    return X


def _xgb_predict_raw(model, X):
    """
    Robust prediction helper that works across XGBoost versions by always
    going through a Booster + DMatrix when possible.
    """
    X_prepared = _xgb_prepare_dmatrix_input(X)

    # Prefer going through the Booster API, which is stable across versions.
    booster = None
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
        except Exception:
            booster = None
    if isinstance(model, xgb.Booster):
        booster = model

    if booster is not None:
        dmat = xgb.DMatrix(X_prepared)
        # For binary:logistic this returns probabilities for the positive class.
        raw_pred = booster.predict(dmat)
        return raw_pred

    # Fallback: rely on the estimator's own predict_proba implementation.
    try:
        proba = model.predict_proba(X_prepared)
        # Handle (n_samples,) vs (n_samples, 2) shapes
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    except Exception:
        # Last-resort fallback to plain predict
        pred = model.predict(X_prepared)
        return pred


def predict_xgboost(model, X_test):
    """Predict with XGBoost - returns binary predictions"""
    pred_proba = _xgb_predict_raw(model, X_test)
    pred = (pred_proba > 0.5).astype(int)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in XGBoost predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_xgboost(model, X_test):
    """Predict probabilities with XGBoost"""
    pred_proba = _xgb_predict_raw(model, X_test)
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in XGBoost probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_xgboost(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Combined gain + permutation importance for XGBoost:

    1. **Gain screen (annotation):** compute the model's built-in
       tree importance (gain / Gini) for all features to see which
       features were ever used in splits.
    2. **Permutation importance (primary score):** run permutation
       importance on the full feature set (with optional row caps
       via PGX_PERM_MAX_ROWS) to obtain metric-aligned importance.

    If X_test / y_test are not provided, falls back to gain importance only.
    """
    # Always compute gain-based importance first.
    logger.info(
        "XGBoost importance: computing gain (Gini) for %d features",
        len(feature_names),
    )
    gain_importance = getattr(model, "feature_importances_", None)
    if gain_importance is None:
        # Some XGBoost variants may not expose feature_importances_;
        # in that case, we fall back directly to permutation or zeros.
        gain_importance = np.zeros(len(feature_names), dtype=float)

    gain_df = pd.DataFrame(
        {"feature": feature_names, "gain_importance": gain_importance}
    )

    if X_test is None or y_test is None:
        # No evaluation set → return gain screen only.
        gain_df = gain_df.sort_values("gain_importance", ascending=False)
        gain_df = gain_df.rename(columns={"gain_importance": "importance"})
        return gain_df

    # For annotation, mark all features with strictly positive gain.
    positive_mask = gain_df["gain_importance"] > 0
    n_positive = int(positive_mask.sum())
    logger.info(
        "XGBoost importance: %d/%d features have gain_importance > 0 "
        "(env PGX_XGB_PERM_TOP_K=%r is informational only)",
        n_positive,
        len(feature_names),
        os.getenv("PGX_XGB_PERM_TOP_K"),
    )

    # Run permutation importance on the FULL feature set. XGBoost's
    # Booster expects the same feature dimensionality at prediction
    # time as at training time; passing a reduced column set leads to
    # shape-mismatch errors. We rely on PGX_PERM_MAX_ROWS inside
    # get_permutation_importance to control memory and runtime.
    logger.info(
        "XGBoost importance: running permutation importance on full feature "
        "set (%d features); PGX_PERM_MAX_ROWS=%r",
        len(feature_names),
        os.getenv("PGX_PERM_MAX_ROWS"),
    )
    perm_df = get_permutation_importance(
        model, X_test, y_test, feature_names, scoring=scoring
    )

    # Merge gain + permutation across all features.
    all_features_df = perm_df.merge(gain_df, on="feature", how="left")

    # Annotate whether each feature has strictly positive gain (ever used in a split).
    all_features_df["gain_gt_zero"] = all_features_df["gain_importance"] > 0

    logger.info(
        "XGBoost importance: permutation importance complete for %d features; "
        "%d features flagged gain_gt_zero",
        len(all_features_df),
        int(all_features_df["gain_gt_zero"].sum()),
    )

    return all_features_df



