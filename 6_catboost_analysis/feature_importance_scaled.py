#!/usr/bin/env python3
"""
Scaled Feature Importance Calculation
Based on approach from: https://github.com/Jerome3590/phts/tree/main/graft-loss/feature_importance

Final feature importance = normalized values scaled by best model metrics
- Original project used concordance index
- This implementation uses Recall as the scaling metric
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, precision_score, f1_score
import logging

logger = logging.getLogger(__name__)


def calculate_scaled_feature_importance(
    models: List[CatBoostClassifier],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance_type: str = 'PredictionValuesChange',
    metric: str = 'recall'
) -> pd.DataFrame:
    """
    Calculate normalized feature importance scaled by best model's performance metric.
    
    Args:
        models: List of trained CatBoost models (e.g., from cross-validation or ensemble)
        X_test: Test features
        y_test: Test labels
        importance_type: Type of CatBoost feature importance ('PredictionValuesChange', 'LossFunctionChange', 'ShapValues')
        metric: Metric to use for scaling ('recall', 'auc', 'f1', 'precision', 'accuracy')
    
    Returns:
        DataFrame with columns: ['feature', 'importance_raw', 'importance_normalized', 'importance_scaled', 'rank']
    """
    logger.info(f"Calculating scaled feature importance using {metric} metric")
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    n_features = len(feature_names)
    
    # Store importance and metrics for each model
    model_importances = []
    model_metrics = []
    
    # Calculate importance and metrics for each model
    for i, model in enumerate(models):
        # Get feature importance
        importance = model.get_feature_importance(type=importance_type)
        
        # Calculate model metric
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        if metric == 'recall':
            metric_value = recall_score(y_test, y_pred)
        elif metric == 'auc':
            if y_pred_proba is not None:
                metric_value = roc_auc_score(y_test, y_pred_proba)
            else:
                logger.warning(f"Model {i} doesn't support predict_proba, skipping AUC calculation")
                continue
        elif metric == 'f1':
            metric_value = f1_score(y_test, y_pred)
        elif metric == 'precision':
            metric_value = precision_score(y_test, y_pred)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_test, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        model_importances.append(importance)
        model_metrics.append(metric_value)
        
        logger.debug(f"Model {i}: {metric}={metric_value:.4f}, max_importance={max(importance):.4f}")
    
    if not model_importances:
        raise ValueError("No valid models provided")
    
    # Convert to numpy arrays
    importance_matrix = np.array(model_importances)  # Shape: (n_models, n_features)
    metrics_array = np.array(model_metrics)  # Shape: (n_models,)
    
    # Step 1: Average importance across models
    avg_importance = np.mean(importance_matrix, axis=0)  # Shape: (n_features,)
    
    # Step 2: Normalize importance (0-1 scale)
    min_imp = np.min(avg_importance)
    max_imp = np.max(avg_importance)
    if max_imp > min_imp:
        normalized_importance = (avg_importance - min_imp) / (max_imp - min_imp)
    else:
        # All features have same importance
        normalized_importance = np.ones(n_features) / n_features
    
    # Step 3: Scale by best model's metric
    best_metric = np.max(metrics_array)
    best_model_idx = np.argmax(metrics_array)
    
    logger.info(f"Best model index: {best_model_idx}, {metric}={best_metric:.4f}")
    
    # Final scaled importance = normalized importance * best metric
    scaled_importance = normalized_importance * best_metric
    
    # Create DataFrame
    result_df = pd.DataFrame({
        'feature': feature_names,
        'importance_raw': avg_importance,
        'importance_normalized': normalized_importance,
        'importance_scaled': scaled_importance,
        'rank': np.argsort(scaled_importance)[::-1] + 1
    }).sort_values('importance_scaled', ascending=False)
    
    logger.info(f"Feature importance calculated: {len(result_df)} features")
    logger.info(f"Top 5 features by scaled importance:")
    for idx, row in result_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance_scaled']:.6f} (normalized: {row['importance_normalized']:.4f}, raw: {row['importance_raw']:.4f})")
    
    return result_df


def calculate_scaled_feature_importance_single_model(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance_type: str = 'PredictionValuesChange',
    metric: str = 'recall'
) -> pd.DataFrame:
    """
    Simplified version for single model (wraps multi-model version).
    
    Args:
        model: Single trained CatBoost model
        X_test: Test features
        y_test: Test labels
        importance_type: Type of CatBoost feature importance
        metric: Metric to use for scaling
    
    Returns:
        DataFrame with scaled feature importance
    """
    return calculate_scaled_feature_importance(
        models=[model],
        X_test=X_test,
        y_test=y_test,
        importance_type=importance_type,
        metric=metric
    )


def get_top_features(
    scaled_importance_df: pd.DataFrame,
    top_n: int = 1000,
    min_scaled_importance: Optional[float] = None
) -> List[str]:
    """
    Get top N features based on scaled importance, optionally filtered by minimum threshold.
    
    Args:
        scaled_importance_df: DataFrame from calculate_scaled_feature_importance
        top_n: Number of top features to return
        min_scaled_importance: Optional minimum scaled importance threshold
    
    Returns:
        List of feature names
    """
    df = scaled_importance_df.copy()
    
    if min_scaled_importance is not None:
        df = df[df['importance_scaled'] >= min_scaled_importance]
    
    top_features = df.head(top_n)['feature'].tolist()
    
    logger.info(f"Selected {len(top_features)} features (top_n={top_n}, min_threshold={min_scaled_importance})")
    
    return top_features


def filter_transactions_by_features(
    transactions_df: pd.DataFrame,
    feature_column: str,
    top_features: List[str]
) -> pd.DataFrame:
    """
    Filter transactions to only include top features.
    
    Args:
        transactions_df: DataFrame with transactions (e.g., from FP-Growth preprocessing)
        feature_column: Column name containing feature/item names
        top_features: List of feature names to keep
    
    Returns:
        Filtered DataFrame
    """
    top_features_set = set(top_features)
    
    # Filter rows where feature is in top_features
    filtered_df = transactions_df[transactions_df[feature_column].isin(top_features_set)].copy()
    
    logger.info(f"Filtered transactions: {len(transactions_df)} -> {len(filtered_df)} rows")
    logger.info(f"Features kept: {len(top_features_set)}")
    
    return filtered_df

