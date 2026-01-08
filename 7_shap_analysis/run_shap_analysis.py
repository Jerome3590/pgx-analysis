#!/usr/bin/env python3
"""
Run SHAP analysis for final models for a given (cohort, age_band).

Outputs:
  7_shap_analysis/outputs/{cohort}/{age_band_fname}/
    - {cohort}_{age_band_fname}_shap_global_importance_xgboost.csv
    - {cohort}_{age_band_fname}_shap_global_importance_catboost.csv
    - {cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet
    - {cohort}_{age_band_fname}_shap_sample_values_catboost.parquet
    - summary bar / beeswarm plots (PNG) per model
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # type: ignore


def _load_final_features(cohort: str, age_band: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load final features using DuckDB for efficient CSV reading.
    Only converts to pandas at the final step for compatibility with SHAP.
    """
    import duckdb

    age_band_fname = age_band_to_fname(age_band)
    features_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Final features file not found: {features_path}")

    # Use DuckDB to read CSV efficiently (more memory efficient than pandas)
    con = duckdb.connect()
    try:
        # Read CSV using DuckDB (more memory efficient than pandas)
        # DuckDB handles large files better by streaming/chunking internally
        df = con.execute(f"SELECT * FROM read_csv_auto('{str(features_path)}')").df()

        if "target" not in df.columns:
            raise ValueError(f"'target' column not found in {features_path}")

        y = df["target"].astype(int)
        X = df.drop(columns=["mi_person_key", "target"], errors="ignore")

        # Keep numeric columns only (model is trained on numeric features)
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X = X[numeric_cols].copy()
        return X, y
    finally:
        con.close()


from py_helpers.env_utils import get_xgb_cpu_nthread  # noqa: E402


# ============================================================================
# Two-Pass SHAP Analysis Functions
# ============================================================================

def compute_global_shap_signal(
    booster,  # xgb.Booster
    X: pd.DataFrame,
    chunk_rows: int = 500,
) -> pd.DataFrame:
    """
    Pass 1: Compute global SHAP signal per feature (streamed, memory-efficient).
    
    Uses XGBoost's fast pred_contribs=True path for exact TreeSHAP.
    Accumulates mean_abs_shap and mean_signed_shap per feature.
    
    Args:
        booster: XGBoost Booster object
        X: Feature DataFrame (will be aligned to model's feature space)
        chunk_rows: Number of rows to process per chunk
        
    Returns:
        DataFrame with columns: feature, mean_abs_shap, mean_signed_shap
        Sorted by mean_abs_shap descending
    """
    import xgboost as xgb  # type: ignore
    
    expected = booster.feature_names
    if expected is None:
        raise ValueError("Booster has no feature_names; cannot align SHAP to columns.")
    
    print(f"Computing global SHAP signal for {len(expected)} features using {len(X)} rows...")
    
    # Align input to model feature space (CRITICAL: prevents feature mismatch)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    X = X.reindex(columns=expected, fill_value=0).astype("float32")
    
    abs_sum = np.zeros(len(expected), dtype=np.float64)
    signed_sum = np.zeros(len(expected), dtype=np.float64)
    n_total = 0
    
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        d = xgb.DMatrix(X.iloc[start:stop], feature_names=expected)
        contrib = booster.predict(d, pred_contribs=True)  # (rows, n_features+1)
        shap = contrib[:, :-1]  # exclude bias column
        
        abs_sum += np.abs(shap).sum(axis=0)
        signed_sum += shap.sum(axis=0)
        n_total += shap.shape[0]
        
        if (start // chunk_rows + 1) % 10 == 0:
            print(f"  Processed {stop}/{len(X)} rows...")
    
    mean_abs = abs_sum / max(n_total, 1)
    mean_signed = signed_sum / max(n_total, 1)
    
    out = pd.DataFrame({
        "feature": expected,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,  # Using mean_shap for consistency with existing code
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    
    print(f"Completed global SHAP signal computation: {n_total} rows processed")
    return out


def select_signal_features_topk(global_df: pd.DataFrame, k: int = 500) -> list[str]:
    """
    Select features with signal using Top K approach.
    
    Args:
        global_df: DataFrame from compute_global_shap_signal
        k: Number of top features to select
        
    Returns:
        List of feature names
    """
    k = int(k)
    return global_df.head(k)["feature"].tolist()


def select_signal_features_threshold(global_df: pd.DataFrame, min_mean_abs: float = 0.0005) -> list[str]:
    """
    Select features with signal using threshold approach.
    
    Args:
        global_df: DataFrame from compute_global_shap_signal
        min_mean_abs: Minimum mean_abs_shap threshold
        
    Returns:
        List of feature names
    """
    return global_df.loc[global_df["mean_abs_shap"] >= float(min_mean_abs), "feature"].tolist()


def write_row_shap_for_selected_features(
    booster,  # xgb.Booster
    X: pd.DataFrame,
    selected_features: list[str],
    out_path: Path,
    chunk_rows: int = 200,
    row_id: pd.Series | None = None,
) -> None:
    """
    Pass 2: Write per-row SHAP values for selected features only (streamed to parquet).
    
    Args:
        booster: XGBoost Booster object
        X: Feature DataFrame (will be aligned to model's feature space)
        selected_features: List of feature names to include in output
        out_path: Path to output parquet file
        chunk_rows: Number of rows to process per chunk
        row_id: Optional Series with row identifiers (e.g., mi_person_key)
    """
    import xgboost as xgb  # type: ignore
    
    expected = booster.feature_names
    if expected is None:
        raise ValueError("Booster has no feature_names; cannot align SHAP to columns.")
    
    # Align input to model feature space
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    X = X.reindex(columns=expected, fill_value=0).astype("float32")
    
    # Column indices for slicing SHAP contributions
    feat_to_idx = {f: i for i, f in enumerate(expected)}
    sel = [f for f in selected_features if f in feat_to_idx]
    if not sel:
        raise ValueError("No selected features exist in model feature list.")
    
    sel_idx = np.array([feat_to_idx[f] for f in sel], dtype=np.int32)
    
    # Row ids
    if row_id is None:
        row_id = pd.Series(np.arange(len(X)), name="row_id")
    else:
        row_id = row_id.reset_index(drop=True)
        row_id.name = row_id.name or "row_id"
    
    print(f"Writing row-level SHAP for {len(sel)} selected features ({len(X)} rows)...")
    
    # Collect chunks in memory (for single parquet file output)
    chunks = []
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        d = xgb.DMatrix(X.iloc[start:stop], feature_names=expected)
        contrib = booster.predict(d, pred_contribs=True)  # (rows, n_features+1)
        
        shap_sel = contrib[:, sel_idx]  # only selected features
        bias = contrib[:, -1].reshape(-1, 1)  # bias
        
        df_chunk = pd.DataFrame(shap_sel, columns=sel)
        df_chunk["bias"] = bias
        df_chunk.insert(0, row_id.name, row_id.iloc[start:stop].values)
        chunks.append(df_chunk)
        
        if (start // chunk_rows + 1) % 50 == 0:
            print(f"  Processed {stop}/{len(X)} rows...")
    
    # Combine and write to single parquet file
    result_df = pd.concat(chunks, ignore_index=True)
    
    # Use DuckDB for efficient parquet writing
    import duckdb
    con_parquet = duckdb.connect()
    try:
        con_parquet.register('shap_df', result_df)
        con_parquet.execute(f"COPY shap_df TO '{str(out_path)}' (FORMAT PARQUET)")
    except Exception as e:
        print(f"Warning: DuckDB Parquet write failed ({e}), falling back to pandas")
        result_df.to_parquet(out_path, index=False, engine='pyarrow')
    finally:
        con_parquet.close()
    
    print(f"Saved row-level SHAP values to {out_path}")


def compute_global_shap_signal_catboost(
    model,  # CatBoostClassifier
    X: pd.DataFrame,
    y: pd.Series,
    cat_feature_indices: list[int] | None = None,
    chunk_rows: int = 500,
) -> pd.DataFrame:
    """
    Pass 1: Compute global SHAP signal per feature for CatBoost (streamed, memory-efficient).
    
    Uses CatBoost's get_feature_importance(type="ShapValues") with chunked Pool objects.
    
    Args:
        model: CatBoostClassifier object
        X: Feature DataFrame
        y: Target Series
        cat_feature_indices: List of categorical feature indices (optional)
        chunk_rows: Number of rows to process per chunk
        
    Returns:
        DataFrame with columns: feature, mean_abs_shap, mean_shap
        Sorted by mean_abs_shap descending
    """
    from catboost import Pool  # type: ignore
    
    feature_names = list(X.columns)
    print(f"Computing global SHAP signal for {len(feature_names)} features using {len(X)} rows...")
    
    abs_sum = np.zeros(len(feature_names), dtype=np.float64)
    signed_sum = np.zeros(len(feature_names), dtype=np.float64)
    n_total = 0
    
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        X_chunk = X.iloc[start:stop]
        y_chunk = y.iloc[start:stop]
        
        # Create Pool for this chunk
        if cat_feature_indices:
            pool_chunk = Pool(X_chunk, y_chunk, cat_features=cat_feature_indices)
        else:
            pool_chunk = Pool(X_chunk, y_chunk)
        
        # Get SHAP values for this chunk
        shap_chunk = model.get_feature_importance(type="ShapValues", data=pool_chunk)
        shap_chunk = np.array(shap_chunk)
        
        # CatBoost returns: (n_samples, n_features + 1) [last col = expected value]
        # or (n_samples, n_classes, n_features + 1) for multiclass
        if shap_chunk.ndim == 2:
            shap_feat = shap_chunk[:, :-1]  # drop expected value column
        elif shap_chunk.ndim == 3:
            shap_feat = shap_chunk[:, :, :-1].mean(axis=1)  # collapse classes
        else:
            raise ValueError(f"Unexpected CatBoost SHAP array shape: {shap_chunk.shape}")
        
        abs_sum += np.abs(shap_feat).sum(axis=0)
        signed_sum += shap_feat.sum(axis=0)
        n_total += shap_feat.shape[0]
        
        if (start // chunk_rows + 1) % 10 == 0:
            print(f"  Processed {stop}/{len(X)} rows...")
    
    mean_abs = abs_sum / max(n_total, 1)
    mean_signed = signed_sum / max(n_total, 1)
    
    out = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    
    print(f"Completed global SHAP signal computation: {n_total} rows processed")
    return out


def write_row_shap_for_selected_features_catboost(
    model,  # CatBoostClassifier
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
    out_path: Path,
    cat_feature_indices: list[int] | None = None,
    chunk_rows: int = 200,
    row_id: pd.Series | None = None,
) -> None:
    """
    Pass 2: Write per-row SHAP values for selected features only (streamed to parquet).
    
    Args:
        model: CatBoostClassifier object
        X: Feature DataFrame
        y: Target Series
        selected_features: List of feature names to include in output
        out_path: Path to output parquet file
        cat_feature_indices: List of categorical feature indices (optional)
        chunk_rows: Number of rows to process per chunk
        row_id: Optional Series with row identifiers (e.g., mi_person_key)
    """
    from catboost import Pool  # type: ignore
    
    feature_names = list(X.columns)
    
    # Column indices for selected features
    sel = [f for f in selected_features if f in feature_names]
    if not sel:
        raise ValueError("No selected features exist in feature list.")
    
    sel_idx = [feature_names.index(f) for f in sel]
    
    # Row ids
    if row_id is None:
        row_id = pd.Series(np.arange(len(X)), name="row_id")
    else:
        row_id = row_id.reset_index(drop=True)
        row_id.name = row_id.name or "row_id"
    
    print(f"Writing row-level SHAP for {len(sel)} selected features ({len(X)} rows)...")
    
    # Collect chunks in memory (for single parquet file output)
    chunks = []
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        X_chunk = X.iloc[start:stop]
        y_chunk = y.iloc[start:stop]
        
        # Create Pool for this chunk
        if cat_feature_indices:
            pool_chunk = Pool(X_chunk, y_chunk, cat_features=cat_feature_indices)
        else:
            pool_chunk = Pool(X_chunk, y_chunk)
        
        # Get SHAP values for this chunk
        shap_chunk = model.get_feature_importance(type="ShapValues", data=pool_chunk)
        shap_chunk = np.array(shap_chunk)
        
        # Extract feature SHAP values (exclude expected value)
        if shap_chunk.ndim == 2:
            shap_feat = shap_chunk[:, :-1]  # drop expected value column
            bias = shap_chunk[:, -1].reshape(-1, 1)  # expected value (bias)
        elif shap_chunk.ndim == 3:
            shap_feat = shap_chunk[:, :, :-1].mean(axis=1)  # collapse classes
            bias = shap_chunk[:, :, -1].mean(axis=1).reshape(-1, 1)  # expected value
        else:
            raise ValueError(f"Unexpected CatBoost SHAP array shape: {shap_chunk.shape}")
        
        # Select only the features we want
        shap_sel = shap_feat[:, sel_idx]
        
        df_chunk = pd.DataFrame(shap_sel, columns=sel)
        df_chunk["bias"] = bias
        df_chunk.insert(0, row_id.name, row_id.iloc[start:stop].values)
        chunks.append(df_chunk)
        
        if (start // chunk_rows + 1) % 50 == 0:
            print(f"  Processed {stop}/{len(X)} rows...")
    
    # Combine and write to single parquet file
    result_df = pd.concat(chunks, ignore_index=True)
    
    # Use DuckDB for efficient parquet writing
    import duckdb
    con_parquet = duckdb.connect()
    try:
        con_parquet.register('shap_df', result_df)
        con_parquet.execute(f"COPY shap_df TO '{str(out_path)}' (FORMAT PARQUET)")
    except Exception as e:
        print(f"Warning: DuckDB Parquet write failed ({e}), falling back to pandas")
        result_df.to_parquet(out_path, index=False, engine='pyarrow')
    finally:
        con_parquet.close()
    
    print(f"Saved row-level SHAP values to {out_path}")


def _load_best_models(cohort: str, age_band: str):
    """
    Load the best models selected by the final model training step.
    
    Returns:
        - best_catboost_model: CatBoost model loaded from .cbm binary
        - model_selection_metadata: Dict with selection information
    """
    age_band_fname = age_band_to_fname(age_band)
    
    # Load model selection metadata
    metadata_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
    )
    
    import json
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            model_selection_metadata = json.load(f)
    else:
        print(f"Warning: Model selection metadata not found at {metadata_path}")
        model_selection_metadata = {}
    
    # Try loading CatBoost binary model from models directory first (preferred, consistent with XGBoost)
    cb_binary_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "models"
        / "catboost_model.cbm"
    )
    
    # Fallback to model_outputs location
    if not cb_binary_path.exists():
        cb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / "models"
            / "catboost_model.cbm"
        )
    
    # Fallback to final_model_json location (legacy)
    if not cb_binary_path.exists():
        cb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "outputs"
            / cohort
            / age_band_fname
            / "final_model_json"
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
    
    # Final fallback to model_outputs root (legacy)
    if not cb_binary_path.exists():
        cb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
    
    # Try loading from JSON if binary not found (CatBoost can load from JSON)
    if not cb_binary_path.exists():
        cb_json_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "outputs"
            / cohort
            / age_band_fname
            / "final_model_json"
            / f"{cohort}_{age_band_fname}_best_catboost_model.json"
        )
        if cb_json_path.exists():
            print(f"CatBoost binary (.cbm) not found, loading from JSON: {cb_json_path}")
            from catboost import CatBoostClassifier  # type: ignore
            cb_model = CatBoostClassifier()
            cb_model.load_model(str(cb_json_path))
            print(f"Loaded best CatBoost model from JSON: {cb_json_path}")
            return cb_model, model_selection_metadata
    
    if not cb_binary_path.exists():
        raise FileNotFoundError(
            f"Best CatBoost model binary or JSON not found. Checked:\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / 'models' / 'catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'final_model_json' / f'{cohort}_{age_band_fname}_best_catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / f'{cohort}_{age_band_fname}_best_catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'final_model_json' / f'{cohort}_{age_band_fname}_best_catboost_model.json'}\n"
            f"Please run 6_final_model_selection/run_final_model.py first or download model from S3."
        )
    
    from catboost import CatBoostClassifier  # type: ignore
    cb_model = CatBoostClassifier()
    cb_model.load_model(str(cb_binary_path))
    print(f"Loaded best CatBoost model from {cb_binary_path}")
    
    return cb_model, model_selection_metadata


def _load_best_xgboost_model(cohort: str, age_band: str):
    """
    Load the best XGBoost model saved by the final model training step.

    Prefers native XGBoost booster binary model (UBJ format, most reliable for SHAP).
    Falls back to joblib if binary not available.

    Returns:
        - best_xgboost_model: XGBoost model (loaded from binary or joblib)
    """
    import xgboost as xgb  # type: ignore
    
    age_band_fname = age_band_to_fname(age_band)

    # Try loading native XGBoost booster binary model first (preferred for SHAP)
    xgb_binary_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "models"
        / "xgboost_model.ubj"
    )

    # Fallback to model_outputs location
    if not xgb_binary_path.exists():
        xgb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / "models"
            / "xgboost_model.ubj"
        )

    if xgb_binary_path.exists():
        # Load from native binary model (most reliable for SHAP, avoids base_score issues)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_binary_path))
        print(f"Loaded best XGBoost model from native binary: {xgb_binary_path}")
        return xgb_model

    # Fallback to joblib if JSON not available
    xgb_joblib_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "models"
        / "xgboost.joblib"
    )

    if not xgb_joblib_path.exists():
        xgb_joblib_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / "models"
            / "xgboost.joblib"
        )

    if not xgb_joblib_path.exists():
        raise FileNotFoundError(
            f"Best XGBoost model not found. Checked:\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'xgboost_model.ubj'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / 'models' / 'xgboost_model.ubj'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'xgboost.joblib'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / 'models' / 'xgboost.joblib'}\n"
            f"Please run 6_final_model_selection/run_final_model.py first."
        )

    # Load from joblib and convert to booster for SHAP
    xgb_model = joblib.load(str(xgb_joblib_path))
    print(f"Loaded best XGBoost model from joblib: {xgb_joblib_path}")
    
    # Convert to booster and fix base_score issue for SHAP compatibility
    if hasattr(xgb_model, 'get_booster'):
        import tempfile
        import os
        import json
        import ast
        
        booster = xgb_model.get_booster()
        
        # Fix base_score in booster config if it's in string array format
        config = json.loads(booster.save_config())
        learner_model_param = config.get('learner', {}).get('learner_train_param', {})
        base_score_str = learner_model_param.get('base_score', '0.5')
        
        # Check if base_score is in problematic format like '[1.6610055E-1]'
        if isinstance(base_score_str, str) and base_score_str.startswith('[') and base_score_str.endswith(']'):
            try:
                # Parse the array string and extract the float value
                base_score_value = ast.literal_eval(base_score_str)
                if isinstance(base_score_value, list) and len(base_score_value) > 0:
                    base_score_value = float(base_score_value[0])
                else:
                    base_score_value = float(base_score_value)
                
                # Update the config with the fixed base_score
                learner_model_param['base_score'] = str(base_score_value)
                config['learner']['learner_train_param'] = learner_model_param
                
                # Save to temp file and reload to apply the fix
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
                    json.dump(config, tmp_json, indent=2)
                    tmp_json_path = tmp_json.name
                
                # Load the fixed config into a new booster
                booster.load_config(tmp_json_path)
                try:
                    os.unlink(tmp_json_path)
                except:
                    pass
                
                print(f"Fixed base_score from '{base_score_str}' to '{base_score_value}'")
            except Exception as e:
                print(f"[WARNING] Could not fix base_score: {e}")
        
        # Save booster to temp binary (UBJ) and reload into new model
        with tempfile.NamedTemporaryFile(suffix='.ubj', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        booster.save_model(tmp_path)
        xgb_model_for_shap = xgb.XGBClassifier()
        xgb_model_for_shap.load_model(tmp_path)
        try:
            os.unlink(tmp_path)
        except:
            pass
        print("Converted joblib model to booster format for SHAP compatibility")
        return xgb_model_for_shap
    
    return xgb_model


def _fit_models_for_shap(X: pd.DataFrame, y: pd.Series, cohort: str, age_band: str, random_seed: int = 42):
    """
    Load best CatBoost and XGBoost models for SHAP analysis.

    Uses the best models selected by the final model training step.
    """
    # Load best CatBoost model
    cb_model, model_selection_metadata = _load_best_models(cohort, age_band)

    # Load best XGBoost model (instead of retraining)
    try:
        xgb_model = _load_best_xgboost_model(cohort, age_band)
    except FileNotFoundError:
        # Fallback: if model not found, retrain (shouldn't happen in normal workflow)
        print("Warning: Best XGBoost model not found. Retraining from scratch...")
        import xgboost as xgb  # type: ignore
        nthread = get_xgb_cpu_nthread()
        from py_helpers.env_utils import is_linux
        device = "cpu" if is_linux() else "cuda"
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=random_seed,
        )
        try:
            xgb_model.fit(X, y)
        except Exception:
            xgb_model.set_params(tree_method="hist")
            if "device" in xgb_model.get_params():
                xgb_model.set_params(device="cpu")
            xgb_model.fit(X, y)

    return xgb_model, cb_model


def run_shap_analysis(
    cohort: str,
    age_band: str,
    n_background: int = 1000,
    n_eval: int = 2000,
) -> bool:
    """
    Run SHAP analysis for XGBoost and CatBoost models.
    
    Returns:
        bool: True if at least one model was successfully analyzed, False otherwise
    """
    import matplotlib.pyplot as plt

    try:
        import shap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'shap' library is required for SHAP analysis. "
            "Install with: pip install shap"
        ) from e

    age_band_fname = age_band_to_fname(age_band)
    out_dir = (
        PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading final features for {cohort}, {age_band}...")
    # Load full data including mi_person_key for row IDs
    import duckdb
    age_band_fname = age_band_to_fname(age_band)
    features_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Final features file not found: {features_path}")
    
    con = duckdb.connect()
    try:
        df_full = con.execute(f"SELECT * FROM read_csv_auto('{str(features_path)}')").df()
        if "target" not in df_full.columns:
            raise ValueError(f"'target' column not found in {features_path}")
        
        y = df_full["target"].astype(int)
        row_id = df_full.get("mi_person_key", None)
        X_full = df_full.drop(columns=["mi_person_key", "target"], errors="ignore")
        
        # Keep numeric columns only (model is trained on numeric features)
        numeric_cols = [c for c in X_full.columns if pd.api.types.is_numeric_dtype(X_full[c])]
        X_full = X_full[numeric_cols].copy()
    finally:
        con.close()
    
    print(f"Final feature matrix: {X_full.shape[0]} rows, {X_full.shape[1]} features.")

    print("Loading best models for SHAP...")
    xgb_clf, cb_clf = _fit_models_for_shap(X_full, y, cohort, age_band)

    s3_outputs = []  # Track S3 uploads for checkpointing
    
    # Track whether at least one model was successfully analyzed
    models_analyzed = []

    # ------------------- XGBoost SHAP (Two-Pass Approach) -------------------
    print("=" * 80)
    print("XGBoost SHAP Analysis (Two-Pass: Global Signal → Row-Level for Selected Features)")
    print("=" * 80)
    
    try:
        import xgboost as xgb  # type: ignore
        
        if not hasattr(xgb_clf, 'get_booster'):
            raise ValueError("XGBoost model does not have get_booster() method")
        
        booster = xgb_clf.get_booster()
        
        # Pass 1: Compute global SHAP signal (streamed, memory-efficient)
        print("\n[Pass 1] Computing global SHAP signal per feature...")
        global_shap_df = compute_global_shap_signal(booster, X_full, chunk_rows=500)
        
        # Save global importance CSV (all features with signal)
        xgb_imp_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
        )
        # Filter to features with mean_abs_shap > 0 for consistency
        global_shap_df_filtered = global_shap_df[global_shap_df['mean_abs_shap'] > 0].copy()
        global_shap_df_filtered.to_csv(xgb_imp_path, index=False)
        print(f"✅ Saved global SHAP importance to {xgb_imp_path}")
        print(f"   Features with signal: {len(global_shap_df_filtered)} (from {len(global_shap_df)} total)")
        
        # Select features with signal (Top K approach, default 500)
        # Can be changed to threshold: select_signal_features_threshold(global_shap_df, min_mean_abs=0.0005)
        selected_features = select_signal_features_topk(global_shap_df_filtered, k=500)
        print(f"\n[Feature Selection] Selected {len(selected_features)} features with signal (Top K=500)")
        
        # Pass 2: Write per-row SHAP for selected features only
        print(f"\n[Pass 2] Computing row-level SHAP for {len(selected_features)} selected features...")
        xgb_shap_sample_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
        )
        write_row_shap_for_selected_features(
            booster=booster,
            X=X_full,
            selected_features=selected_features,
            out_path=xgb_shap_sample_path,
            chunk_rows=200,
            row_id=row_id,
        )
        
        # Create summary plots using selected features
        # Load a sample from parquet file for plotting (limit to n_eval rows for memory efficiency)
        print("\n[Plots] Creating summary plots...")
        shap_sample_df = pd.read_parquet(xgb_shap_sample_path)
        # Limit to n_eval rows for plotting to avoid memory issues
        plot_sample_size = min(n_eval, len(shap_sample_df))
        shap_sample_df_plot = shap_sample_df.head(plot_sample_size)
        
        # Extract SHAP values (exclude row_id and bias columns)
        shap_cols = [c for c in selected_features if c in shap_sample_df_plot.columns]
        shap_values_plot = shap_sample_df_plot[shap_cols].values
        
        # Get corresponding feature values using row_id if available, otherwise use index
        if row_id is not None and 'row_id' in shap_sample_df_plot.columns:
            row_ids_plot = shap_sample_df_plot['row_id'].values
            # Map row_ids to indices in X_full
            row_id_to_idx = {rid: idx for idx, rid in enumerate(row_id)}
            row_indices = [row_id_to_idx.get(rid, i) for i, rid in enumerate(row_ids_plot)]
            X_plot = X_full[shap_cols].iloc[row_indices].reset_index(drop=True)
        else:
            # Use first plot_sample_size rows
            X_plot = X_full[shap_cols].iloc[:plot_sample_size].reset_index(drop=True)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot,
            X_plot,
            feature_names=selected_features,
            show=False,
            plot_type="bar",
        )
        bar_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_summary_bar_xgboost.png"
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_plot,
            X_plot,
            feature_names=selected_features,
            show=False,
            plot_type="dot",
        )
        beeswarm_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_summary_beeswarm_xgboost.png"
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=300)
        plt.close()

        print(f"✅ Saved XGBoost SHAP summary plots to {out_dir}")
        
        models_analyzed.append("xgboost")
        
        # Upload XGBoost SHAP outputs
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
            if xgb_imp_path.exists():
                s3_xgb_imp = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
                if upload_file_to_s3(xgb_imp_path, s3_xgb_imp):
                    s3_outputs.append(s3_xgb_imp)
            if xgb_shap_sample_path.exists():
                s3_xgb_sample = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
                if upload_file_to_s3(xgb_shap_sample_path, s3_xgb_sample):
                    s3_outputs.append(s3_xgb_sample)
        except ImportError:
            pass
            
    except Exception as e:
        print(f"[ERROR] XGBoost SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # ------------------- CatBoost SHAP (Two-Pass Approach) -------------------
    if cb_clf is not None:
        try:
            print("=" * 80)
            print("CatBoost SHAP Analysis (Two-Pass: Global Signal → Row-Level for Selected Features)")
            print("=" * 80)
            
            feature_names_cb = list(X_full.columns)
            
            # Identify categorical features (item_* features that were marked as categorical during training)
            # CatBoost requires us to specify categorical features when creating Pool
            cat_feature_indices = [
                i for i, name in enumerate(feature_names_cb)
                if name.startswith('item_')
            ]
            
            if cat_feature_indices:
                print(f"Marking {len(cat_feature_indices)} item_* features as categorical for CatBoost SHAP")
            
            # Pass 1: Compute global SHAP signal (streamed, memory-efficient)
            print("\n[Pass 1] Computing global SHAP signal per feature...")
            global_shap_df_cb = compute_global_shap_signal_catboost(
                model=cb_clf,
                X=X_full,
                y=y,
                cat_feature_indices=cat_feature_indices if cat_feature_indices else None,
                chunk_rows=500,
            )
            
            # Save global importance CSV (all features with signal)
            cb_imp_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
            )
            # Filter to features with mean_abs_shap > 0 for consistency
            global_shap_df_cb_filtered = global_shap_df_cb[global_shap_df_cb['mean_abs_shap'] > 0].copy()
            global_shap_df_cb_filtered.to_csv(cb_imp_path, index=False)
            print(f"✅ Saved global SHAP importance to {cb_imp_path}")
            print(f"   Features with signal: {len(global_shap_df_cb_filtered)} (from {len(global_shap_df_cb)} total)")
            
            # Select features with signal (Top K approach, default 500)
            selected_features_cb = select_signal_features_topk(global_shap_df_cb_filtered, k=500)
            print(f"\n[Feature Selection] Selected {len(selected_features_cb)} features with signal (Top K=500)")
            
            # Pass 2: Write per-row SHAP for selected features only
            print(f"\n[Pass 2] Computing row-level SHAP for {len(selected_features_cb)} selected features...")
            cb_shap_sample_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet"
            )
            write_row_shap_for_selected_features_catboost(
                model=cb_clf,
                X=X_full,
                y=y,
                selected_features=selected_features_cb,
                out_path=cb_shap_sample_path,
                cat_feature_indices=cat_feature_indices if cat_feature_indices else None,
                chunk_rows=200,
                row_id=row_id,
            )
            
            # Create summary plots using selected features
            # Load a sample from parquet file for plotting (limit to n_eval rows for memory efficiency)
            print("\n[Plots] Creating summary plots...")
            shap_cb_sample_df = pd.read_parquet(cb_shap_sample_path)
            # Limit to n_eval rows for plotting to avoid memory issues
            plot_sample_size_cb = min(n_eval, len(shap_cb_sample_df))
            shap_cb_sample_df_plot = shap_cb_sample_df.head(plot_sample_size_cb)
            
            # Extract SHAP values (exclude row_id and bias columns)
            shap_cols_cb = [c for c in selected_features_cb if c in shap_cb_sample_df_plot.columns]
            shap_values_plot_cb = shap_cb_sample_df_plot[shap_cols_cb].values
            
            # Get corresponding feature values using row_id if available, otherwise use index
            if row_id is not None and 'row_id' in shap_cb_sample_df_plot.columns:
                row_ids_plot_cb = shap_cb_sample_df_plot['row_id'].values
                # Map row_ids to indices in X_full
                row_id_to_idx_cb = {rid: idx for idx, rid in enumerate(row_id)}
                row_indices_cb = [row_id_to_idx_cb.get(rid, i) for i, rid in enumerate(row_ids_plot_cb)]
                X_plot_cb = X_full[shap_cols_cb].iloc[row_indices_cb].reset_index(drop=True)
            else:
                # Use first plot_sample_size_cb rows
                X_plot_cb = X_full[shap_cols_cb].iloc[:plot_sample_size_cb].reset_index(drop=True)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_plot_cb,
                X_plot_cb,
                feature_names=shap_cols_cb,
                show=False,
                plot_type="bar",
            )
            cb_bar_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_summary_bar_catboost.png"
            )
            plt.tight_layout()
            plt.savefig(cb_bar_path, dpi=300)
            plt.close()

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_plot_cb,
                X_plot_cb,
                feature_names=shap_cols_cb,
                show=False,
                plot_type="dot",
            )
            cb_beeswarm_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_summary_beeswarm_catboost.png"
            )
            plt.tight_layout()
            plt.savefig(cb_beeswarm_path, dpi=300)
            plt.close()

            print(f"✅ Saved CatBoost SHAP summary plots to {out_dir}")
            
            # Mark CatBoost as successfully analyzed
            models_analyzed.append("catboost")

            # Upload CatBoost SHAP outputs if they exist
            try:
                from py_helpers.checkpoint_utils import upload_file_to_s3

                if cb_imp_path.exists():
                    s3_cb_imp = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
                    if upload_file_to_s3(cb_imp_path, s3_cb_imp):
                        s3_outputs.append(s3_cb_imp)
                if cb_shap_sample_path.exists():
                    s3_cb_sample = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet"
                    if upload_file_to_s3(cb_shap_sample_path, s3_cb_sample):
                        s3_outputs.append(s3_cb_sample)
            except ImportError:
                pass
        except Exception as e:
            print(f"[ERROR] CatBoost SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Save checkpoint after all SHAP analysis completes (only if at least one model was analyzed)
    if models_analyzed:
        try:
            from py_helpers.checkpoint_utils import save_step_checkpoint

            save_step_checkpoint(
                step_name="7_shap_analysis",
                cohort=cohort,
                age_band=age_band,
                metadata={"n_background": n_background, "n_eval": n_eval, "models_analyzed": models_analyzed},
                output_paths=s3_outputs,
            )
        except ImportError:
            pass  # Checkpoint saving is optional
    
    # Return True if at least one model was analyzed
    return len(models_analyzed) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for final models for a given cohort/age_band."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    parser.add_argument(
        "--n_background",
        type=int,
        default=1000,
        help="Number of background samples for SHAP (default: 1000).",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=2000,
        help="Number of evaluation samples for SHAP (default: 2000).",
    )
    args = parser.parse_args()

    age_band_fname = args.age_band.replace("-", "_")
    out_dir = (
        PROJECT_ROOT / "7_shap_analysis" / "outputs" / args.cohort / age_band_fname
    )

    # Check for existing local outputs (idempotency - check local first)
    # SHAP generates outputs for both XGBoost and CatBoost (if available)
    expected_outputs = [
        f"{args.cohort}_{age_band_fname}_shap_global_importance_xgboost.csv",
        f"{args.cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet",
        f"{args.cohort}_{age_band_fname}_shap_summary_bar_xgboost.png",
        f"{args.cohort}_{age_band_fname}_shap_summary_beeswarm_xgboost.png",
    ]
    
    # CatBoost outputs are optional (model might not be available)
    optional_outputs = [
        f"{args.cohort}_{age_band_fname}_shap_global_importance_catboost.csv",
        f"{args.cohort}_{age_band_fname}_shap_sample_values_catboost.parquet",
        f"{args.cohort}_{age_band_fname}_shap_summary_bar_catboost.png",
        f"{args.cohort}_{age_band_fname}_shap_summary_beeswarm_catboost.png",
    ]

    all_required_exist = all((out_dir / fname).exists() for fname in expected_outputs)
    
    if all_required_exist:
        print(f"[SKIP] Step 7 outputs already exist locally for {args.cohort}/{args.age_band}")
        
        # Still try to upload to S3 if not already there (idempotent upload)
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint
            
            s3_outputs = []
            for fname in expected_outputs + optional_outputs:
                local_path = out_dir / fname
                if local_path.exists():
                    if fname.endswith('.csv'):
                        s3_path = f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    elif fname.endswith('.parquet'):
                        s3_path = f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    else:
                        continue  # Skip PNG files for S3 upload (they're large and optional)
                    
                    if upload_file_to_s3(local_path, s3_path):
                        s3_outputs.append(s3_path)
            
            # Save checkpoint if outputs uploaded
            if s3_outputs:
                save_step_checkpoint(
                    step_name="7_shap_analysis",
                    cohort=args.cohort,
                    age_band=args.age_band,
                    metadata={"n_background": args.n_background, "n_eval": args.n_eval, "models_analyzed": ["xgboost"]},
                    output_paths=s3_outputs,
                )
        except ImportError:
            pass  # S3 upload is optional
        
        return

    # Check S3 for existing outputs (idempotency - fallback if local doesn't exist)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists

        s3_output_paths = [
            f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_shap_global_importance_xgboost.csv",
            f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet",
        ]

        # Only skip if outputs actually exist (not just checkpoint)
        # Checkpoint might exist but outputs might be missing
        s3_outputs_exist = check_step_outputs_exist(s3_output_paths)
        
        if s3_outputs_exist:
            print(f"[SKIP] Step 7 outputs already exist in S3 for {args.cohort}/{args.age_band}; downloading to local.")
            
            # Download from S3 to local
            try:
                import boto3
                s3_client = boto3.client("s3")
                S3_BUCKET = "pgxdatalake"
                
                out_dir.mkdir(parents=True, exist_ok=True)
                
                downloaded_files = []
                # Download XGBoost outputs (required)
                for fname in expected_outputs:
                    s3_key = f"gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    local_path = out_dir / fname
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                        print(f"Downloaded {local_path} from S3")
                        downloaded_files.append(local_path)
                    except Exception as e:
                        print(f"Warning: Could not download {s3_key}: {e}")
                
                # Try to download CatBoost outputs (optional)
                for fname in optional_outputs:
                    s3_key = f"gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    local_path = out_dir / fname
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                        print(f"Downloaded {local_path} from S3")
                        downloaded_files.append(local_path)
                    except Exception:
                        pass  # CatBoost outputs are optional
                
                # Verify that required files actually exist before skipping
                all_required_exist = all((out_dir / fname).exists() for fname in expected_outputs)
                if all_required_exist:
                    print(f"[SKIP] Step 7 outputs downloaded from S3 for {args.cohort}/{args.age_band}")
                    return
                else:
                    print(f"[WARNING] Required SHAP outputs missing after download attempt. Will regenerate.")
            except Exception as e:
                print(f"Warning: Could not download from S3: {e}. Will regenerate outputs.")
        elif check_step_checkpoint_exists("7_shap_analysis", args.cohort, args.age_band):
            # Checkpoint exists but outputs don't - this is inconsistent, regenerate
            print(f"[WARNING] Step 7 checkpoint exists in S3 but outputs are missing. Will regenerate outputs.")
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    success = run_shap_analysis(
        cohort=args.cohort,
        age_band=args.age_band,
        n_background=args.n_background,
        n_eval=args.n_eval,
    )
    
    if not success:
        print("\n[ERROR] No models were successfully analyzed.")
        print("This step cannot complete without at least one model being analyzed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

