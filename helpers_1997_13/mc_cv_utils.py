"""
Monte Carlo Cross-Validation Utilities
Functions for running MC-CV analysis with multiple models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
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
from helpers_1997_13.feature_importance_model_utils import (
    train_catboost, train_random_forest, train_xgboost, train_xgboost_rf,
    train_lightgbm, train_extratrees, train_logistic_regression,
    train_linearsvc, train_elasticnet, train_lasso,
    predict_catboost, predict_random_forest, predict_xgboost,
    predict_lightgbm, predict_extratrees, predict_logistic_regression,
    predict_linearsvc, predict_elasticnet, predict_lasso,
    predict_proba_catboost, predict_proba_random_forest, predict_proba_xgboost,
    predict_proba_lightgbm, predict_proba_extratrees, predict_proba_logistic_regression,
    predict_proba_linearsvc, predict_proba_elasticnet, predict_proba_lasso,
    get_importance_catboost, get_importance_random_forest, get_importance_xgboost,
    get_importance_lightgbm, get_importance_extratrees, get_importance_logistic_regression,
    get_importance_linearsvc, get_importance_elasticnet, get_importance_lasso
)
from helpers_1997_13.model_utils import calculate_recall, calculate_logloss


def run_single_split(
    split_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_train_all: pd.DataFrame,
    y_train_all: np.ndarray,
    method: str,
    model_params: Dict,
    scaling_metric: str,
    data_catboost: Optional[pd.DataFrame] = None,
    X_test_all: Optional[pd.DataFrame] = None,
    y_test_all: Optional[np.ndarray] = None,
    test_data_catboost: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Run a single MC-CV split for a given method
    
    Args:
        split_idx: Index of this split
        train_idx: Training indices (from training data)
        test_idx: Test indices (from test data, if separate test data provided)
        X_train_all: All training features (format depends on method)
        y_train_all: All training labels
        method: Model type ('catboost', 'random_forest', 'xgboost', 'xgboost_rf',
                           'lightgbm', 'extratrees', 'logistic_regression',
                           'linearsvc', 'elasticnet', 'lasso')
        model_params: Model parameters dictionary
        scaling_metric: Metric to use for scaling ('recall' or 'logloss')
        data_catboost: Optional CatBoost-formatted training data (for CatBoost method)
        X_test_all: Optional test features (if None, uses X_train_all)
        y_test_all: Optional test labels (if None, uses y_train_all)
        test_data_catboost: Optional CatBoost-formatted test data (for CatBoost method)
        
    Returns:
        Dictionary with split results
    """
    try:
        # Use separate test data if provided, otherwise use same data
        X_train = X_train_all.iloc[train_idx].copy()
        y_train = y_train_all[train_idx]

        if X_test_all is not None and y_test_all is not None:
            # For ALL models, evaluate on the full temporal holdout set (e.g., all 2019)
            # in every split. Splits only change which rows are used for training.
            X_test = X_test_all.copy()
            y_test = y_test_all.copy()

            # Ensure test features match train features (same columns, same order)
            # Add missing columns as zeros, remove extra columns
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        else:
            # Original behavior: use same data for train and test
            X_test = X_train_all.iloc[test_idx].copy()
            y_test = y_train_all[test_idx]
        
        # Drop mi_person_key if present (not a feature)
        if 'mi_person_key' in X_train.columns:
            X_train = X_train.drop(columns=['mi_person_key'])
        if 'mi_person_key' in X_test.columns:
            X_test = X_test.drop(columns=['mi_person_key'])
        
        # Train model
        if method == 'catboost':
            if data_catboost is not None:
                X_train_cb = data_catboost.iloc[train_idx].drop(
                    columns=['target', 'mi_person_key'] if 'mi_person_key' in data_catboost.columns else ['target']
                )
            else:
                X_train_cb = X_train

            # Use the full temporal holdout set for CatBoost as well
            if test_data_catboost is not None:
                X_test_cb = test_data_catboost.drop(
                    columns=['target', 'mi_person_key'] if 'mi_person_key' in test_data_catboost.columns else ['target']
                )
            else:
                X_test_cb = X_test
            
            model = train_catboost(X_train_cb, y_train, model_params.get('catboost', {}))
            y_pred = predict_catboost(model, X_test_cb)
            y_pred_proba = predict_proba_catboost(model, X_test_cb)
            # Use permutation importance for fair comparison (requires X_test and y_test)
            feature_importance = get_importance_catboost(model, X_train_cb.columns.tolist(), X_test=X_test_cb, y_test=y_test)
            
        elif method == 'random_forest':
            model = train_random_forest(X_train, y_train, model_params.get('random_forest', {}))
            y_pred = predict_random_forest(model, X_test)
            y_pred_proba = predict_proba_random_forest(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_random_forest(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'xgboost':
            model = train_xgboost(X_train, y_train, model_params.get('xgboost', {}))
            y_pred = predict_xgboost(model, X_test)
            y_pred_proba = predict_proba_xgboost(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_xgboost(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'xgboost_rf':
            model = train_xgboost_rf(X_train, y_train, model_params.get('xgboost_rf', {}))
            y_pred = predict_xgboost(model, X_test)
            y_pred_proba = predict_proba_xgboost(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_xgboost(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'lightgbm':
            model = train_lightgbm(X_train, y_train, model_params.get('lightgbm', {}))
            y_pred = predict_lightgbm(model, X_test)
            y_pred_proba = predict_proba_lightgbm(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_lightgbm(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'extratrees':
            model = train_extratrees(X_train, y_train, model_params.get('extratrees', {}))
            y_pred = predict_extratrees(model, X_test)
            y_pred_proba = predict_proba_extratrees(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_extratrees(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'logistic_regression':
            model = train_logistic_regression(X_train, y_train, model_params.get('logistic_regression', {}))
            y_pred = predict_logistic_regression(model, X_test)
            y_pred_proba = predict_proba_logistic_regression(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_logistic_regression(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'linearsvc':
            model = train_linearsvc(X_train, y_train, model_params.get('linearsvc', {}))
            y_pred = predict_linearsvc(model, X_test)
            y_pred_proba = predict_proba_linearsvc(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_linearsvc(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'elasticnet':
            model = train_elasticnet(X_train, y_train, model_params.get('elasticnet', {}))
            y_pred = predict_elasticnet(model, X_test)
            y_pred_proba = predict_proba_elasticnet(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_elasticnet(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        elif method == 'lasso':
            model = train_lasso(X_train, y_train, model_params.get('lasso', {}))
            y_pred = predict_lasso(model, X_test)
            y_pred_proba = predict_proba_lasso(model, X_test)
            # Use permutation importance for fair comparison
            feature_importance = get_importance_lasso(model, X_train.columns.tolist(), X_test=X_test, y_test=y_test)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate metrics
        recall = calculate_recall(y_test, y_pred)
        logloss = calculate_logloss(y_test, y_pred_proba)
        
        # Scale importance by metric
        if scaling_metric == 'recall':
            scale_factor = recall if recall > 0 else 0.001  # Avoid division by zero
        elif scaling_metric == 'logloss':
            scale_factor = 1.0 / logloss if logloss > 0 else 0.001
        else:
            scale_factor = 1.0
        
        feature_importance['scaled_importance'] = feature_importance['importance'] * scale_factor
        feature_importance['split'] = split_idx
        feature_importance['recall'] = recall
        feature_importance['logloss'] = logloss
        
        return {
            'split': split_idx,
            'feature_importance': feature_importance,
            'recall': recall,
            'logloss': logloss,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'split': split_idx,
            'status': 'error',
            'error': str(e)
        }


def run_mc_cv_method(
    data: pd.DataFrame,
    method: str,
    split_indices: List[Dict[str, np.ndarray]],
    model_params: Dict,
    scaling_metric: str = 'recall',
    n_jobs: int = 1,
    data_catboost: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    test_data_catboost: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Run MC-CV for a single method
    
    Args:
        data: Training data frame with target column (format depends on method)
        method: Model type ('catboost', 'random_forest', 'xgboost', 'xgboost_rf',
                           'lightgbm', 'extratrees', 'logistic_regression',
                           'linearsvc', 'elasticnet', 'lasso')
        split_indices: List of dictionaries with 'train_idx' (from training data) and 'test_idx' (from test data)
        model_params: Model parameters dictionary
        scaling_metric: Metric to use for scaling ('recall' or 'logloss')
        n_jobs: Number of parallel jobs
        data_catboost: Optional CatBoost-formatted training data (for CatBoost method)
        test_data: Optional test data frame (if None, uses split_indices from data)
        test_data_catboost: Optional CatBoost-formatted test data (for CatBoost method)
        
    Returns:
        DataFrame with aggregated feature importance results
    """
    # Prepare training data based on method
    if method == 'catboost':
        if data_catboost is not None:
            X_train_all = data_catboost.drop(columns=['target'])
            y_train_all = data_catboost['target'].values
        else:
            X_train_all = data.drop(columns=['target'])
            y_train_all = data['target'].values
    else:
        X_train_all = data.drop(columns=['target'])
        y_train_all = data['target'].values
    
    # Prepare test data if provided
    if test_data is not None:
        if method == 'catboost':
            if test_data_catboost is not None:
                X_test_all = test_data_catboost.drop(columns=['target'])
                y_test_all = test_data_catboost['target'].values
            else:
                X_test_all = test_data.drop(columns=['target'])
                y_test_all = test_data['target'].values
        else:
            X_test_all = test_data.drop(columns=['target'])
            y_test_all = test_data['target'].values
    else:
        # Use same data for train and test (original behavior)
        X_test_all = X_train_all
        y_test_all = y_train_all
    
    feature_names = X_train_all.columns.tolist()
    n_splits = len(split_indices)
    
    print(f"\n--- Running MC-CV for {method} ({n_splits} splits) ---")
    
    # Run splits in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_single_split)(
            i,
            split_indices[i]['train_idx'],
            split_indices[i]['test_idx'],
            X_train_all,
            y_train_all,
            method,
            model_params,
            scaling_metric,
            data_catboost,
            X_test_all,
            y_test_all,
            test_data_catboost
        )
        for i in range(n_splits)
    )
    
    # Aggregate results
    successful_splits = [r for r in results if r['status'] == 'success']
    failed_splits = [r for r in results if r['status'] == 'error']
    
    if len(successful_splits) == 0:
        # Log first few errors for debugging
        error_samples = failed_splits[:5] if len(failed_splits) > 0 else []
        error_messages = [f"Split {r['split']}: {r.get('error', 'Unknown error')}" for r in error_samples]
        error_summary = "\n".join(error_messages)
        raise ValueError(f"No successful splits for method {method}. Sample errors:\n{error_summary}")
    
    # Combine feature importance from all splits
    all_importance = pd.concat([r['feature_importance'] for r in successful_splits], ignore_index=True)
    
    # Aggregate by feature
    aggregated = all_importance.groupby('feature').agg({
        'scaled_importance': ['mean', 'std', 'count'],
        'importance': ['mean', 'std'],
        'recall': 'mean',
        'logloss': 'mean'
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        'feature',
        'scaled_importance_mean',
        'scaled_importance_std',
        'scaled_importance_count',
        'importance_mean',
        'importance_std',
        'recall_mean',
        'logloss_mean'
    ]
    
    # Sort by scaled importance
    aggregated = aggregated.sort_values('scaled_importance_mean', ascending=False)
    
    print(f"Completed {len(successful_splits)}/{n_splits} splits for {method}")
    print(f"  Mean Recall: {aggregated['recall_mean'].iloc[0]:.4f}")
    print(f"  Mean LogLoss: {aggregated['logloss_mean'].iloc[0]:.4f}")
    
    return aggregated

