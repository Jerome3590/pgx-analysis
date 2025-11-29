"""
Feature Importance Model Training Utilities
Functions for training, prediction, and feature importance extraction
Supports CatBoost, Random Forest, XGBoost, XGBoost RF mode, LightGBM, ExtraTrees,
LogisticRegression, LinearSVC, ElasticNet, and LASSO
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Import from common helpers
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from helpers_1997_13.common_imports import *


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
        # For CatBoost, use CatBoost's built-in permutation importance
        # CatBoost has get_feature_importance with Pool that handles categoricals properly
        categorical_features = [col for col in X_test.columns if col.startswith('item_')]
        cat_indices = [X_test.columns.get_loc(col) for col in categorical_features] if categorical_features else None
        
        test_pool = Pool(
            data=X_test,
            label=y_test,
            cat_features=cat_indices
        )
        
        # Use CatBoost's built-in permutation-based importance
        importance = model.get_feature_importance(
            data=test_pool,
            type='PredictionValuesChange'  # Permutation-based importance
        )
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        # For other models, use sklearn's permutation_importance
        # Convert to numpy array
        if isinstance(X_test, pd.DataFrame):
            X_test_for_perm = X_test.values
        else:
            X_test_for_perm = X_test
        
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
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_test_for_perm, y_test,
            scoring=scorer,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean
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
        'thread_count': 1  # Use 1 thread per worker to avoid conflicts
    }
    
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
# Random Forest
# ============================================================================

def train_random_forest(X_train, y_train, params):
    """Train Random Forest model (scikit-learn)"""
    n_features = X_train.shape[1]
    mtry = params.get('mtry', None)
    if mtry is None:
        mtry = int(np.sqrt(n_features))
    
    rf_params = {
        'n_estimators': params.get('ntree', 100),
        'max_features': mtry,
        'min_samples_leaf': params.get('nodesize', 1),
        'max_depth': params.get('maxnodes', None),
        'n_jobs': 1,  # Use 1 thread per worker
        'random_state': params.get('random_seed', 42),
        'verbose': 0
    }
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    
    return model


def predict_random_forest(model, X_test):
    """Predict with Random Forest - returns binary predictions"""
    pred = model.predict(X_test)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in Random Forest predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_random_forest(model, X_test):
    """Predict probabilities with Random Forest"""
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in Random Forest probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_random_forest(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from Random Forest model.
    Falls back to Gini importance if X_test/y_test not provided.
    """
    if X_test is not None and y_test is not None:
        return get_permutation_importance(model, X_test, y_test, feature_names, scoring=scoring)
    else:
        # Fallback to Gini importance (for backward compatibility)
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


# ============================================================================
# XGBoost
# ============================================================================

def train_xgboost(X_train, y_train, params):
    """Train XGBoost model (gradient boosting mode)"""
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'n_estimators': params.get('n_estimators', 100),
        'subsample': params.get('subsample', 1.0),
        'colsample_bytree': params.get('colsample_bytree', 1.0),
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    return model


def train_xgboost_rf(X_train, y_train, params):
    """Train XGBoost in Random Forest mode"""
    n_features = X_train.shape[1]
    max_features = params.get('max_features', None)
    if max_features is None:
        max_features = int(np.sqrt(n_features))
    
    xgb_rf_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'n_estimators': params.get('n_estimators', 100),
        'subsample': params.get('subsample', 0.8),  # RF typically uses subsampling
        'colsample_bytree': max_features / n_features,  # RF-style feature sampling
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbosity': 0,
        'tree_method': 'hist',  # Efficient tree construction
        'booster': 'gbtree'  # Use tree booster (not linear)
    }
    
    model = xgb.XGBClassifier(**xgb_rf_params)
    model.fit(X_train, y_train)
    
    return model


def predict_xgboost(model, X_test):
    """Predict with XGBoost - returns binary predictions"""
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = (pred_proba > 0.5).astype(int)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in XGBoost predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_xgboost(model, X_test):
    """Predict probabilities with XGBoost"""
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in XGBoost probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_xgboost(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from XGBoost model.
    Falls back to gain importance if X_test/y_test not provided.
    """
    if X_test is not None and y_test is not None:
        return get_permutation_importance(model, X_test, y_test, feature_names, scoring=scoring)
    else:
        # Fallback to gain importance (for backward compatibility)
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


# ============================================================================
# LightGBM
# ============================================================================

def train_lightgbm(X_train, y_train, params):
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': params.get('num_leaves', 31),
        'learning_rate': params.get('learning_rate', 0.1),
        'feature_fraction': params.get('feature_fraction', 1.0),
        'bagging_fraction': params.get('bagging_fraction', 1.0),
        'bagging_freq': params.get('bagging_freq', 0),
        'verbose': -1,
        'n_jobs': 1,  # Use 1 thread per worker
        'random_state': params.get('random_seed', 42),
        'num_threads': 1
    }
    
    num_boost_round = params.get('n_estimators', 100)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=0)]  # Suppress output (replaces verbose_eval)
    )
    
    return model


def predict_lightgbm(model, X_test):
    """Predict with LightGBM - returns binary predictions"""
    pred_proba = model.predict(X_test, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
    pred = (pred_proba > 0.5).astype(int)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in LightGBM predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_lightgbm(model, X_test):
    """Predict probabilities with LightGBM"""
    pred_proba = model.predict(X_test, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in LightGBM probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_lightgbm(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from LightGBM model.
    Uses custom permutation importance since LightGBM Booster is not sklearn-compatible.
    Falls back to gain importance if X_test/y_test not provided.
    """
    if X_test is not None and y_test is not None:
        # LightGBM Booster doesn't work with sklearn's permutation_importance
        # Use custom permutation importance
        return get_lightgbm_permutation_importance(model, X_test, y_test, feature_names, scoring=scoring)
    else:
        # Fallback to gain importance (for backward compatibility)
        importance = model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


def get_lightgbm_permutation_importance(model, X_test, y_test, feature_names, scoring='recall', n_repeats=5, random_state=42):
    """
    Custom permutation importance for LightGBM Booster object.
    
    Args:
        model: Trained LightGBM Booster model
        X_test: Test features (DataFrame or array)
        y_test: Test labels (array)
        feature_names: List of feature names
        scoring: Scoring function ('recall' or 'log_loss')
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with columns: ['feature', 'importance']
    """
    import numpy as np
    from sklearn.metrics import recall_score, log_loss
    
    # Ensure X_test is a numpy array
    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    
    # Ensure y_test is numpy array
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    
    # Get baseline score
    if scoring == 'recall':
        y_pred_baseline = predict_lightgbm(model, X_test_array)
        baseline_score = recall_score(y_test, y_pred_baseline, zero_division=0)
    elif scoring == 'log_loss':
        y_pred_proba_baseline = predict_proba_lightgbm(model, X_test_array)
        baseline_score = -log_loss(y_test, y_pred_proba_baseline)  # Negate for consistency
    else:
        raise ValueError(f"Unsupported scoring: {scoring}")
    
    # Calculate importance for each feature
    importances = []
    np.random.seed(random_state)
    
    for feature_idx, feature_name in enumerate(feature_names):
        feature_importances = []
        
        for repeat in range(n_repeats):
            # Shuffle the feature
            X_test_shuffled = X_test_array.copy()
            shuffled_values = X_test_shuffled[:, feature_idx].copy()
            np.random.shuffle(shuffled_values)
            X_test_shuffled[:, feature_idx] = shuffled_values
            
            # Calculate score with shuffled feature
            if scoring == 'recall':
                y_pred_shuffled = predict_lightgbm(model, X_test_shuffled)
                shuffled_score = recall_score(y_test, y_pred_shuffled, zero_division=0)
            elif scoring == 'log_loss':
                y_pred_proba_shuffled = predict_proba_lightgbm(model, X_test_shuffled)
                shuffled_score = -log_loss(y_test, y_pred_proba_shuffled)
            
            # Importance is the drop in score (baseline - shuffled)
            importance = baseline_score - shuffled_score
            feature_importances.append(importance)
        
        # Average importance across repeats
        importances.append(np.mean(feature_importances))
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


# ============================================================================
# ExtraTrees
# ============================================================================

def train_extratrees(X_train, y_train, params):
    """Train ExtraTrees (Extremely Randomized Trees) model"""
    n_features = X_train.shape[1]
    max_features = params.get('max_features', None)
    if max_features is None:
        max_features = int(np.sqrt(n_features))
    
    et_params = {
        'n_estimators': params.get('n_estimators', 100),
        'max_features': max_features,
        'min_samples_leaf': params.get('min_samples_leaf', 1),
        'max_depth': params.get('max_depth', None),
        'n_jobs': 1,  # Use 1 thread per worker
        'random_state': params.get('random_seed', 42),
        'verbose': 0
    }
    
    model = ExtraTreesClassifier(**et_params)
    model.fit(X_train, y_train)
    
    return model


def predict_extratrees(model, X_test):
    """Predict with ExtraTrees - returns binary predictions"""
    pred = model.predict(X_test)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in ExtraTrees predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_extratrees(model, X_test):
    """Predict probabilities with ExtraTrees"""
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in ExtraTrees probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_extratrees(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from ExtraTrees model.
    Falls back to Gini importance if X_test/y_test not provided.
    """
    if X_test is not None and y_test is not None:
        return get_permutation_importance(model, X_test, y_test, feature_names, scoring=scoring)
    else:
        # Fallback to Gini importance (for backward compatibility)
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


# ============================================================================
# LogisticRegression
# ============================================================================

def train_logistic_regression(X_train, y_train, params):
    """Train LogisticRegression model"""
    lr_params = {
        'penalty': params.get('penalty', 'l2'),
        'C': params.get('C', 1.0),
        'solver': params.get('solver', 'lbfgs'),
        'max_iter': params.get('max_iter', 5000),  # Increased for small datasets
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbose': 0
    }
    
    # Adjust solver based on penalty
    if lr_params['penalty'] == 'l1':
        lr_params['solver'] = 'liblinear'  # Only solver that supports L1
    elif lr_params['penalty'] == 'elasticnet':
        lr_params['solver'] = 'saga'  # Only solver that supports elasticnet
    
    # Suppress convergence warnings for small datasets
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model = LogisticRegression(**lr_params)
        model.fit(X_train, y_train)
    
    return model


def predict_logistic_regression(model, X_test):
    """Predict with LogisticRegression - returns binary predictions"""
    pred = model.predict(X_test)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in LogisticRegression predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_logistic_regression(model, X_test):
    """Predict probabilities with LogisticRegression"""
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in LogisticRegression probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_logistic_regression(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from LogisticRegression model.
    Falls back to coefficient magnitude if X_test/y_test not provided.
    """
    if X_test is not None and y_test is not None:
        return get_permutation_importance(model, X_test, y_test, feature_names, scoring=scoring)
    else:
        # Fallback to coefficient magnitude (for backward compatibility)
        importance = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


# ============================================================================
# LinearSVC
# ============================================================================

def train_linearsvc(X_train, y_train, params):
    """Train LinearSVC model"""
    svc_params = {
        'penalty': params.get('penalty', 'l2'),
        'C': params.get('C', 1.0),
        'loss': params.get('loss', 'squared_hinge'),
        'max_iter': params.get('max_iter', 5000),  # Increased for small datasets
        'random_state': params.get('random_seed', 42),
        'dual': params.get('dual', True),
        'verbose': 0
    }
    
    # Suppress convergence warnings for small datasets
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model = LinearSVC(**svc_params)
        model.fit(X_train, y_train)
    
    return model


def predict_linearsvc(model, X_test):
    """Predict with LinearSVC - returns binary predictions"""
    pred = model.predict(X_test)
    
    # Handle NA values
    if np.any(np.isnan(pred)):
        print("Warning: NA values in LinearSVC predictions, replacing with 0")
        pred = np.nan_to_num(pred, nan=0)
    
    return pred


def predict_proba_linearsvc(model, X_test):
    """Predict probabilities with LinearSVC (uses decision_function and sigmoid)"""
    decision_scores = model.decision_function(X_test)
    
    # Convert to probabilities using sigmoid
    pred_proba = 1 / (1 + np.exp(-decision_scores))
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in LinearSVC probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_linearsvc(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """
    Get permutation-based feature importance from LinearSVC model.
    Falls back to coefficient magnitude if X_test/y_test not provided.
    """
    if X_test is not None and y_test is not None:
        return get_permutation_importance(model, X_test, y_test, feature_names, scoring=scoring)
    else:
        # Fallback to coefficient magnitude (for backward compatibility)
        importance = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df


# ============================================================================
# ElasticNet (LogisticRegression with elasticnet penalty)
# ============================================================================

def train_elasticnet(X_train, y_train, params):
    """Train ElasticNet LogisticRegression model"""
    en_params = {
        'penalty': 'elasticnet',
        'C': params.get('C', 1.0),
        'l1_ratio': params.get('l1_ratio', 0.5),
        'solver': 'saga',  # Only solver that supports elasticnet
        'max_iter': params.get('max_iter', 5000),  # Increased for small datasets
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbose': 0
    }
    
    # Suppress convergence warnings for small datasets
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model = LogisticRegression(**en_params)
        model.fit(X_train, y_train)
    
    return model


def predict_elasticnet(model, X_test):
    """Predict with ElasticNet - uses LogisticRegression predict"""
    return predict_logistic_regression(model, X_test)


def predict_proba_elasticnet(model, X_test):
    """Predict probabilities with ElasticNet - uses LogisticRegression predict_proba"""
    return predict_proba_logistic_regression(model, X_test)


def get_importance_elasticnet(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """Get permutation-based feature importance from ElasticNet model"""
    return get_importance_logistic_regression(model, feature_names, X_test, y_test, scoring)


# ============================================================================
# LASSO (LogisticRegression with L1 penalty)
# ============================================================================

def train_lasso(X_train, y_train, params):
    """Train LASSO LogisticRegression model"""
    lasso_params = {
        'penalty': 'l1',
        'C': params.get('C', 1.0),
        'solver': 'liblinear',  # Only solver that supports L1
        'max_iter': params.get('max_iter', 5000),  # Increased for small datasets
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbose': 0
    }
    
    # Suppress convergence warnings for small datasets
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model = LogisticRegression(**lasso_params)
        model.fit(X_train, y_train)
    
    return model


def predict_lasso(model, X_test):
    """Predict with LASSO - uses LogisticRegression predict"""
    return predict_logistic_regression(model, X_test)


def predict_proba_lasso(model, X_test):
    """Predict probabilities with LASSO - uses LogisticRegression predict_proba"""
    return predict_proba_logistic_regression(model, X_test)


def get_importance_lasso(model, feature_names, X_test=None, y_test=None, scoring='recall'):
    """Get permutation-based feature importance from LASSO model"""
    return get_importance_logistic_regression(model, feature_names, X_test, y_test, scoring)

