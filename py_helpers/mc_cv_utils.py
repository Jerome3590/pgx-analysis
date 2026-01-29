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
import logging
from pathlib import Path
import json

warnings.filterwarnings("ignore")

# Import from common helpers
import sys
import os

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_helpers.common_imports import *
from py_helpers.feature_importance_model_utils import (
    train_catboost,
    train_xgboost,
    train_xgboost_rf,
    predict_catboost,
    predict_xgboost,
    predict_proba_catboost,
    predict_proba_xgboost,
    get_importance_catboost,
    get_importance_xgboost,
)
from py_helpers.model_utils import calculate_recall, calculate_logloss


logger = logging.getLogger(__name__)


def _log_mc(message: str) -> None:
    """
    Helper for MC-CV logging: sends messages to both stdout (for interactive runs)
    and the configured Python logger (for timestamped log files / S3 logs).
    """
    # Ensure users see progress in real time in long-running jobs
    print(message, flush=True)
    # And also capture it in the central log stream
    logger.info(message)


def run_single_split(
    split_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_train_all,
    y_train_all: np.ndarray,
    method: str,
    model_params: Dict,
    scaling_metric: str,
    data_catboost: Optional[pd.DataFrame] = None,
    X_test_all: Optional[object] = None,
    y_test_all: Optional[np.ndarray] = None,
    test_data_catboost: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    sparse_for_xgb: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Dict:
    """
    Run a single MC-CV split for a given method
    
    Args:
        split_idx: Index of this split
        train_idx: Training indices (from training data)
        test_idx: Test indices (from test data, if separate test data provided)
        X_train_all: All training features (format depends on method)
        y_train_all: All training labels
        method: Model type ('catboost', 'xgboost', 'xgboost_rf')
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
        # Lightweight per-split logging to help debug long-running MC-CV jobs.
        _log_mc(f"[MC-CV] Starting split {split_idx} for {method}")
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
        
        # Drop mi_person_key if present (not a feature) – only relevant for DataFrames
        if "mi_person_key" in X_train.columns:
            X_train = X_train.drop(columns=["mi_person_key"])
        if "mi_person_key" in X_test.columns:
            X_test = X_test.drop(columns=["mi_person_key"])
        
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
            
        elif method == "xgboost":
            _log_mc(f"[MC-CV] Split {split_idx} ({method}): training model")
            model = train_xgboost(X_train, y_train, model_params.get("xgboost", {}))

            _log_mc(f"[MC-CV] Split {split_idx} ({method}): predicting on full holdout")
            y_pred = predict_xgboost(model, X_test)
            y_pred_proba = predict_proba_xgboost(model, X_test)

            # Use permutation importance for fair comparison
            feat_names = feature_names or X_train.columns.tolist()
            _log_mc(
                f"[MC-CV] Split {split_idx} ({method}): starting permutation importance "
                f"on {X_test.shape[0]} rows × {X_test.shape[1]} features"
            )
            feature_importance = get_importance_xgboost(
                model, feat_names, X_test=X_test, y_test=y_test
            )
            _log_mc(f"[MC-CV] Split {split_idx} ({method}): permutation importance done")

        elif method == "xgboost_rf":
            _log_mc(f"[MC-CV] Split {split_idx} ({method}): training model")
            model = train_xgboost_rf(
                X_train, y_train, model_params.get("xgboost_rf", {})
            )

            _log_mc(f"[MC-CV] Split {split_idx} ({method}): predicting on full holdout")
            y_pred = predict_xgboost(model, X_test)
            y_pred_proba = predict_proba_xgboost(model, X_test)

            # Use permutation importance for fair comparison
            feat_names = feature_names or X_train.columns.tolist()
            _log_mc(
                f"[MC-CV] Split {split_idx} ({method}): starting permutation importance "
                f"on {X_test.shape[0]} rows × {X_test.shape[1]} features"
            )
            feature_importance = get_importance_xgboost(
                model, feat_names, X_test=X_test, y_test=y_test
            )
            _log_mc(f"[MC-CV] Split {split_idx} ({method}): permutation importance done")
            
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

        _log_mc(
            f"[MC-CV] Completed split {split_idx} for {method} "
            f"(recall={recall:.4f}, logloss={logloss:.4f})"
        )

        # ------------------------------------------------------------------
        # Optional per-split checkpointing: persist feature importance and
        # metrics immediately so finished work survives process restarts.
        # ------------------------------------------------------------------
        if checkpoint_dir:
            try:
                ckpt_dir_path = Path(checkpoint_dir)
                ckpt_dir_path.mkdir(parents=True, exist_ok=True)
                fi_path = ckpt_dir_path / f"split_{split_idx}_importance.parquet"
                meta_path = ckpt_dir_path / f"split_{split_idx}_meta.json"

                feature_importance.to_parquet(fi_path, index=False)
                with meta_path.open("w") as f:
                    json.dump(
                        {
                            "split": int(split_idx),
                            "method": method,
                            "recall": float(recall),
                            "logloss": float(logloss),
                        },
                        f,
                    )

                _log_mc(
                    f"[MC-CV] Saved checkpoint for split {split_idx} "
                    f"({method}) to {fi_path}"
                )
            except Exception as e:
                _log_mc(
                    f"[MC-CV] WARNING: failed to save checkpoint for split "
                    f"{split_idx} ({method}): {e}"
                )

        return {
            'split': split_idx,
            'feature_importance': feature_importance,
            'recall': recall,
            'logloss': logloss,
            'status': 'success'
        }
        
    except Exception as e:
        _log_mc(f"[MC-CV] ERROR in split {split_idx} for {method}: {e}")
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
    test_data_catboost: Optional[pd.DataFrame] = None,
    checkpoint_dir: Optional[str] = None,
    force_rerun_checkpoints: bool = False,
) -> pd.DataFrame:
    """
    Run MC-CV for a single method
    
    Args:
        data: Training data frame with target column (format depends on method)
        method: Model type ('catboost', 'xgboost', 'xgboost_rf')
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

    # Capture feature names once; we keep everything dense for stability.
    feature_names = X_train_all.columns.tolist()
    n_splits = len(split_indices)

    _log_mc(f"--- Running MC-CV for {method} ({n_splits} splits) ---")

    # ------------------------------------------------------------------
    # Optional checkpoint loading: reuse completed splits from disk so
    # we don't lose progress after long runs or unexpected restarts.
    # ------------------------------------------------------------------
    preloaded_results: List[Dict] = []
    splits_to_run = list(range(n_splits))
    ckpt_dir_path: Optional[Path] = None

    if checkpoint_dir is not None:
        ckpt_dir_path = Path(checkpoint_dir)
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)

        if not force_rerun_checkpoints:
            for i in range(n_splits):
                fi_path = ckpt_dir_path / f"split_{i}_importance.parquet"
                meta_path = ckpt_dir_path / f"split_{i}_meta.json"
                if fi_path.exists() and meta_path.exists():
                    try:
                        fi_df = pd.read_parquet(fi_path)
                        with meta_path.open("r") as f:
                            meta = json.load(f)

                        preloaded_results.append(
                            {
                                "split": i,
                                "feature_importance": fi_df,
                                "recall": float(meta.get("recall", np.nan)),
                                "logloss": float(meta.get("logloss", np.nan)),
                                "status": "success",
                            }
                        )
                        splits_to_run.remove(i)
                        _log_mc(
                            f"[MC-CV] {method}: loaded checkpoint for split {i} "
                            f"from {fi_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to load checkpoint for %s split %d from %s: %s; "
                            "will recompute this split.",
                            method,
                            i,
                            fi_path,
                            e,
                        )

    # Run remaining splits in parallel. Use verbose logging so we can see progress
    # (e.g., 'Done 10 out of 200' style messages) in long-running jobs.
    results_new: List[Dict] = []
    if splits_to_run:
        _log_mc(
            f"[MC-CV] {method}: running {len(splits_to_run)} splits; "
            f"{len(preloaded_results)} loaded from checkpoints"
        )
        results_new = Parallel(n_jobs=n_jobs, verbose=10)(
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
                test_data_catboost,
                feature_names,
                False,
                str(ckpt_dir_path) if ckpt_dir_path is not None else None,
            )
            for i in splits_to_run
        )
    else:
        _log_mc(
            f"[MC-CV] {method}: all {n_splits} splits already completed via "
            "checkpoints; skipping computation."
        )

    # Combine preloaded + newly computed results and sort by split index
    results = preloaded_results + results_new
    results.sort(key=lambda r: r["split"])
    
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

    _log_mc(f"Completed {len(successful_splits)}/{n_splits} splits for {method}")
    _log_mc(f"  Mean Recall: {aggregated['recall_mean'].iloc[0]:.4f}")
    _log_mc(f"  Mean LogLoss: {aggregated['logloss_mean'].iloc[0]:.4f}")
    
    return aggregated

