"""
Feature Importance Model Training Utilities
Functions for training, prediction, and feature importance extraction
Supports CatBoost, Random Forest, XGBoost, XGBoost RF mode, LightGBM, ExtraTrees,
LogisticRegression, LinearSVC, ElasticNet, and LASSO
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
    # Identify categorical columns (object/string type)
    categorical_features = list(X_train.select_dtypes(include=['object', 'category']).columns)
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]
    
    # Create CatBoost Pool
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_indices if cat_indices else None
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
    categorical_features = list(X_test.select_dtypes(include=['object', 'category']).columns)
    cat_indices = [X_test.columns.get_loc(col) for col in categorical_features]
    
    test_pool = Pool(
        data=X_test,
        cat_features=cat_indices if cat_indices else None
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
    categorical_features = list(X_test.select_dtypes(include=['object', 'category']).columns)
    cat_indices = [X_test.columns.get_loc(col) for col in categorical_features]
    
    test_pool = Pool(
        data=X_test,
        cat_features=cat_indices if cat_indices else None
    )
    
    pred_proba = model.predict_proba(test_pool)[:, 1]
    
    # Handle NA values
    if np.any(np.isnan(pred_proba)):
        print("Warning: NA values in CatBoost probability predictions, replacing with 0.5")
        pred_proba = np.nan_to_num(pred_proba, nan=0.5)
    
    return pred_proba


def get_importance_catboost(model, feature_names):
    """Get feature importance from CatBoost model"""
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


def get_importance_random_forest(model, feature_names):
    """Get feature importance from Random Forest model"""
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


def get_importance_xgboost(model, feature_names):
    """Get feature importance from XGBoost model"""
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
        verbose_eval=False
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


def get_importance_lightgbm(model, feature_names):
    """Get feature importance from LightGBM model"""
    importance = model.feature_importance(importance_type='gain')
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
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


def get_importance_extratrees(model, feature_names):
    """Get feature importance from ExtraTrees model"""
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
        'max_iter': params.get('max_iter', 1000),
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbose': 0
    }
    
    # Adjust solver based on penalty
    if lr_params['penalty'] == 'l1':
        lr_params['solver'] = 'liblinear'  # Only solver that supports L1
    elif lr_params['penalty'] == 'elasticnet':
        lr_params['solver'] = 'saga'  # Only solver that supports elasticnet
    
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


def get_importance_logistic_regression(model, feature_names):
    """Get feature importance from LogisticRegression model (uses absolute value of coefficients)"""
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
        'max_iter': params.get('max_iter', 1000),
        'random_state': params.get('random_seed', 42),
        'dual': params.get('dual', True),
        'verbose': 0
    }
    
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


def get_importance_linearsvc(model, feature_names):
    """Get feature importance from LinearSVC model (uses absolute value of coefficients)"""
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
        'max_iter': params.get('max_iter', 1000),
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbose': 0
    }
    
    model = LogisticRegression(**en_params)
    model.fit(X_train, y_train)
    
    return model


def predict_elasticnet(model, X_test):
    """Predict with ElasticNet - uses LogisticRegression predict"""
    return predict_logistic_regression(model, X_test)


def predict_proba_elasticnet(model, X_test):
    """Predict probabilities with ElasticNet - uses LogisticRegression predict_proba"""
    return predict_proba_logistic_regression(model, X_test)


def get_importance_elasticnet(model, feature_names):
    """Get feature importance from ElasticNet - uses LogisticRegression importance"""
    return get_importance_logistic_regression(model, feature_names)


# ============================================================================
# LASSO (LogisticRegression with L1 penalty)
# ============================================================================

def train_lasso(X_train, y_train, params):
    """Train LASSO LogisticRegression model"""
    lasso_params = {
        'penalty': 'l1',
        'C': params.get('C', 1.0),
        'solver': 'liblinear',  # Only solver that supports L1
        'max_iter': params.get('max_iter', 1000),
        'random_state': params.get('random_seed', 42),
        'n_jobs': 1,  # Use 1 thread per worker
        'verbose': 0
    }
    
    model = LogisticRegression(**lasso_params)
    model.fit(X_train, y_train)
    
    return model


def predict_lasso(model, X_test):
    """Predict with LASSO - uses LogisticRegression predict"""
    return predict_logistic_regression(model, X_test)


def predict_proba_lasso(model, X_test):
    """Predict probabilities with LASSO - uses LogisticRegression predict_proba"""
    return predict_proba_logistic_regression(model, X_test)


def get_importance_lasso(model, feature_names):
    """Get feature importance from LASSO - uses LogisticRegression importance"""
    return get_importance_logistic_regression(model, feature_names)

