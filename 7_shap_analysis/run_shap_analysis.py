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
import logging
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
from py_helpers.event_density_utils import (  # type: ignore
    DENSITY_BINS,
    cohort_aggregate_final_model_has_artifacts,
    final_model_bin_has_trained_artifacts,
    resolve_step6_cohort_age_dir,
    resolve_step6_train_features_csv,
    validate_per_bin_outputs,
)


def _load_final_features(cohort: str, age_band: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load final features using DuckDB for efficient CSV reading.
    Only converts to pandas at the final step for compatibility with SHAP.
    """
    import duckdb

    age_band_fname = age_band_to_fname(age_band)
    features_path = resolve_step6_train_features_csv(PROJECT_ROOT, cohort, age_band)
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
    
    # Stream chunks to DuckDB then write parquet (avoids holding full result in memory)
    import duckdb
    con_parquet = duckdb.connect()
    try:
        first = True
        for start in range(0, len(X), chunk_rows):
            stop = min(start + chunk_rows, len(X))
            d = xgb.DMatrix(X.iloc[start:stop], feature_names=expected)
            contrib = booster.predict(d, pred_contribs=True)  # (rows, n_features+1)
            shap_sel = contrib[:, sel_idx]
            bias = contrib[:, -1].reshape(-1, 1)
            df_chunk = pd.DataFrame(shap_sel, columns=sel)
            df_chunk["bias"] = bias
            df_chunk.insert(0, row_id.name, row_id.iloc[start:stop].values)
            if first:
                con_parquet.register("shap_chunk", df_chunk)
                con_parquet.execute("CREATE TABLE shap_accum AS SELECT * FROM shap_chunk")
                first = False
            else:
                con_parquet.register("shap_chunk", df_chunk)
                con_parquet.execute("INSERT INTO shap_accum SELECT * FROM shap_chunk")
            if (start // chunk_rows + 1) % 50 == 0:
                print(f"  Processed {stop}/{len(X)} rows...")
        con_parquet.execute(f"COPY shap_accum TO '{str(out_path)}' (FORMAT PARQUET)")
    except Exception as e:
        print(f"Warning: DuckDB Parquet write failed ({e}), falling back to pandas chunked write")
        # Fallback: write chunks to parquet one by one (pyarrow can append to same file with proper API)
        chunks = []
        for start in range(0, len(X), chunk_rows):
            stop = min(start + chunk_rows, len(X))
            d = xgb.DMatrix(X.iloc[start:stop], feature_names=expected)
            contrib = booster.predict(d, pred_contribs=True)
            shap_sel = contrib[:, sel_idx]
            bias = contrib[:, -1].reshape(-1, 1)
            df_chunk = pd.DataFrame(shap_sel, columns=sel)
            df_chunk["bias"] = bias
            df_chunk.insert(0, row_id.name, row_id.iloc[start:stop].values)
            chunks.append(df_chunk)
        pd.concat(chunks, ignore_index=True).to_parquet(out_path, index=False, engine="pyarrow")
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
    
    original_columns = list(X.columns)
    feature_names_expected = _catboost_feature_names(model, X)
    cat_feature_names = _cat_feature_names_for_model(
        model,
        original_columns,
        feature_names_expected,
        cat_feature_indices,
    )
    X = _align_for_catboost_model(model, X, cat_feature_names)
    feature_names = list(X.columns)
    aligned_cat_feature_indices = _cat_feature_indices_after_alignment(
        feature_names,
        cat_feature_names,
    )
    print(f"Computing global SHAP signal for {len(feature_names)} features using {len(X)} rows...")
    
    abs_sum = np.zeros(len(feature_names), dtype=np.float64)
    signed_sum = np.zeros(len(feature_names), dtype=np.float64)
    n_total = 0
    
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        X_chunk = X.iloc[start:stop]
        y_chunk = y.iloc[start:stop]
        
        # Create Pool for this chunk
        if aligned_cat_feature_indices:
            pool_chunk = Pool(X_chunk, y_chunk, cat_features=aligned_cat_feature_indices)
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


def _catboost_feature_names(model, X: pd.DataFrame) -> list[str]:
    names = None
    try:
        names = model.feature_names_
    except Exception:
        names = None
    if not names:
        try:
            names = model.get_feature_names()
        except Exception:
            names = None
    return list(names) if names else list(X.columns)


def _cat_feature_names_for_model(
    model,
    original_columns: list[str],
    expected_columns: list[str],
    cat_feature_indices: list[int] | None,
) -> set[str]:
    names: set[str] = set()
    try:
        for i in model.get_cat_feature_indices():
            if 0 <= i < len(expected_columns):
                names.add(expected_columns[i])
    except Exception:
        pass
    if cat_feature_indices:
        names.update({
            original_columns[i]
            for i in cat_feature_indices
            if 0 <= i < len(original_columns)
        })
    if not names:
        names.update(c for c in expected_columns if c.startswith("item_"))
    return names


def _align_for_catboost_model(
    model,
    X: pd.DataFrame,
    cat_feature_names: set[str] | None = None,
) -> pd.DataFrame:
    expected = _catboost_feature_names(model, X)
    cat_feature_names = cat_feature_names or set()
    X_aligned = X.copy()
    for col in X_aligned.columns:
        if col in cat_feature_names:
            X_aligned[col] = X_aligned[col].fillna("__MISSING__").astype(str)
        else:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce").fillna(0)
    missing = [c for c in expected if c not in X_aligned.columns]
    if missing:
        print(f"[INFO] Adding {len(missing)} missing CatBoost feature columns as 0 for SHAP alignment.")
    X_aligned = X_aligned.reindex(columns=expected, fill_value=0)
    for col in cat_feature_names.intersection(expected):
        X_aligned[col] = X_aligned[col].fillna("__MISSING__").astype(str)
    return X_aligned


def _shap_s3_prefix(cohort: str, age_band: str, bin_name: str | None = None) -> str:
    prefix = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}"
    if bin_name:
        prefix = f"{prefix}/bin_models/{bin_name}"
    return prefix


def _shap_s3_key_prefix(cohort: str, age_band: str, bin_name: str | None = None) -> str:
    prefix = f"gold/shap_analysis/{cohort}/{age_band}"
    if bin_name:
        prefix = f"{prefix}/bin_models/{bin_name}"
    return prefix


def _cat_feature_indices_after_alignment(
    aligned_columns: list[str],
    cat_feature_names: set[str],
) -> list[int] | None:
    aligned_indices = [
        i for i, name in enumerate(aligned_columns)
        if name in cat_feature_names and i < len(aligned_columns)
    ]
    dropped = len(cat_feature_names) - len(aligned_indices)
    if dropped > 0:
        print(
            f"[INFO] Dropped {dropped} CatBoost categorical feature markers not present "
            "after model feature alignment."
        )
    return aligned_indices or None


def _required_shap_models(model_selection_metadata: dict | None) -> set[str]:
    required = {"xgboost"}
    selected = str((model_selection_metadata or {}).get("selected_model", "")).lower()
    if selected == "catboost":
        required.add("catboost")
    return required


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
    
    original_columns = list(X.columns)
    feature_names_expected = _catboost_feature_names(model, X)
    cat_feature_names = _cat_feature_names_for_model(
        model,
        original_columns,
        feature_names_expected,
        cat_feature_indices,
    )
    X = _align_for_catboost_model(model, X, cat_feature_names)
    feature_names = list(X.columns)
    aligned_cat_feature_indices = _cat_feature_indices_after_alignment(
        feature_names,
        cat_feature_names,
    )
    
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
    
    # Stream chunks to DuckDB then write parquet (avoids holding full result in memory)
    import duckdb
    con_parquet = duckdb.connect()
    try:
        first = True
        for start in range(0, len(X), chunk_rows):
            stop = min(start + chunk_rows, len(X))
            X_chunk = X.iloc[start:stop]
            y_chunk = y.iloc[start:stop]
            if aligned_cat_feature_indices:
                pool_chunk = Pool(X_chunk, y_chunk, cat_features=aligned_cat_feature_indices)
            else:
                pool_chunk = Pool(X_chunk, y_chunk)
            shap_chunk = model.get_feature_importance(type="ShapValues", data=pool_chunk)
            shap_chunk = np.array(shap_chunk)
            if shap_chunk.ndim == 2:
                shap_feat = shap_chunk[:, :-1]
                bias = shap_chunk[:, -1].reshape(-1, 1)
            elif shap_chunk.ndim == 3:
                shap_feat = shap_chunk[:, :, :-1].mean(axis=1)
                bias = shap_chunk[:, :, -1].mean(axis=1).reshape(-1, 1)
            else:
                raise ValueError(f"Unexpected CatBoost SHAP array shape: {shap_chunk.shape}")
            shap_sel = shap_feat[:, sel_idx]
            df_chunk = pd.DataFrame(shap_sel, columns=sel)
            df_chunk["bias"] = bias
            df_chunk.insert(0, row_id.name, row_id.iloc[start:stop].values)
            if first:
                con_parquet.register("shap_chunk", df_chunk)
                con_parquet.execute("CREATE TABLE shap_accum AS SELECT * FROM shap_chunk")
                first = False
            else:
                con_parquet.register("shap_chunk", df_chunk)
                con_parquet.execute("INSERT INTO shap_accum SELECT * FROM shap_chunk")
            if (start // chunk_rows + 1) % 50 == 0:
                print(f"  Processed {stop}/{len(X)} rows...")
        con_parquet.execute(f"COPY shap_accum TO '{str(out_path)}' (FORMAT PARQUET)")
    except Exception as e:
        print(f"Warning: DuckDB Parquet write failed ({e}), falling back to pandas chunked write")
        chunks = []
        for start in range(0, len(X), chunk_rows):
            stop = min(start + chunk_rows, len(X))
            X_chunk = X.iloc[start:stop]
            y_chunk = y.iloc[start:stop]
            if aligned_cat_feature_indices:
                pool_chunk = Pool(X_chunk, y_chunk, cat_features=aligned_cat_feature_indices)
            else:
                pool_chunk = Pool(X_chunk, y_chunk)
            shap_chunk = model.get_feature_importance(type="ShapValues", data=pool_chunk)
            shap_chunk = np.array(shap_chunk)
            if shap_chunk.ndim == 2:
                shap_feat = shap_chunk[:, :-1]
                bias = shap_chunk[:, -1].reshape(-1, 1)
            elif shap_chunk.ndim == 3:
                shap_feat = shap_chunk[:, :, :-1].mean(axis=1)
                bias = shap_chunk[:, :, -1].mean(axis=1).reshape(-1, 1)
            else:
                raise ValueError(f"Unexpected CatBoost SHAP array shape: {shap_chunk.shape}")
            shap_sel = shap_feat[:, sel_idx]
            df_chunk = pd.DataFrame(shap_sel, columns=sel)
            df_chunk["bias"] = bias
            df_chunk.insert(0, row_id.name, row_id.iloc[start:stop].values)
            chunks.append(df_chunk)
        pd.concat(chunks, ignore_index=True).to_parquet(out_path, index=False, engine="pyarrow")
    finally:
        con_parquet.close()
    print(f"Saved row-level SHAP values to {out_path}")


def _load_best_models(cohort: str, age_band: str, bin_name: str | None = None):
    """
    Load the best CatBoost model from Step 6 outputs.

    When ``bin_name`` is set, only paths under
    ``6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/{bin_name}/`` are
    used (plus that bin's ``final_model_json``). Cohort-level fallbacks are used
    only when ``bin_name`` is None, so per-bin SHAP never loads the aggregate model.
    """
    import json

    age_band_fname = age_band_to_fname(age_band)
    _bin_infix = ("bin_models", bin_name) if bin_name else ()
    _model_base = resolve_step6_cohort_age_dir(PROJECT_ROOT, cohort, age_band)
    _bin_base = _model_base.joinpath(*_bin_infix) if _bin_infix else _model_base

    # Load model selection metadata
    metadata_path = _bin_base / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
    if not metadata_path.exists():
        metadata_path = _model_base / f"{cohort}_{age_band_fname}_model_selection_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            model_selection_metadata = json.load(f)
    else:
        print(f"Warning: Model selection metadata not found at {metadata_path}")
        model_selection_metadata = {}

    stem = f"{cohort}_{age_band_fname}_best_catboost_model"
    age_band_h = age_band_fname.replace("_", "-")

    # Native .cbm (Step 6 writes both models/catboost_model.cbm and final_model_json/*_best_catboost_model.cbm)
    cb_binary_candidates: list[Path] = [
        _bin_base / "models" / "catboost_model.cbm",
        _bin_base / "final_model_json" / f"{stem}.cbm",
    ]
    if not bin_name:
        cb_binary_candidates.append(
            PROJECT_ROOT
            / "6_final_model"
            / "outputs"
            / cohort
            / age_band_fname
            / "final_model_json"
            / f"{stem}.cbm",
        )

    for cb_path in cb_binary_candidates:
        if cb_path.exists():
            from catboost import CatBoostClassifier  # type: ignore

            cb_model = CatBoostClassifier()
            cb_model.load_model(str(cb_path))
            print(f"Loaded best CatBoost model from {cb_path}")
            return cb_model, model_selection_metadata

    # JSON
    cb_json_candidates: list[Path] = [_bin_base / "final_model_json" / f"{stem}.json"]
    if not bin_name:
        cb_json_candidates.append(
            PROJECT_ROOT
            / "6_final_model"
            / "outputs"
            / cohort
            / age_band_fname
            / "final_model_json"
            / f"{stem}.json",
        )
    for cb_json_path in cb_json_candidates:
        if cb_json_path.exists():
            print(f"CatBoost binary (.cbm) not found, loading from JSON: {cb_json_path}")
            from catboost import CatBoostClassifier  # type: ignore

            cb_model = CatBoostClassifier()
            cb_model.load_model(str(cb_json_path))
            print(f"Loaded best CatBoost model from JSON: {cb_json_path}")
            return cb_model, model_selection_metadata

    # Joblib (saved by run_final_model.py in models/)
    cb_joblib_candidates: list[Path] = [_bin_base / "models" / "catboost.joblib"]
    for cb_joblib_path in cb_joblib_candidates:
        if cb_joblib_path.exists():
            print(f"CatBoost binary (.cbm) not found, loading from joblib: {cb_joblib_path}")
            cb_model = joblib.load(str(cb_joblib_path))
            print(f"Loaded best CatBoost model from joblib: {cb_joblib_path}")
            return cb_model, model_selection_metadata

    lines = [
        "Best CatBoost model not found (.cbm, .json, or catboost.joblib). Checked:",
        *[f"  - {p}" for p in cb_binary_candidates],
        *[f"  - {p}" for p in cb_json_candidates],
        *[f"  - {p}" for p in cb_joblib_candidates],
        f"Run Step 6: python 6_final_model/run_final_model.py --cohort {cohort} --age_band {age_band_h}",
    ]
    if bin_name:
        lines.append(
            f"Or sync per-bin artifact from S3: aws s3 cp "
            f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band_h}/bin_models/{bin_name}/catboost_model.cbm "
            f"{_bin_base / 'models' / 'catboost_model.cbm'}"
        )
    else:
        lines.append(
            f"Or sync from S3: aws s3 cp s3://pgxdatalake/gold/final_model/{cohort}/{age_band_h}/catboost_model.cbm "
            f"{PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'catboost_model.cbm'}"
        )
    raise FileNotFoundError("\n".join(lines))


def _load_best_xgboost_model(cohort: str, age_band: str, bin_name: str | None = None):
    """
    Load the best XGBoost model saved by the final model training step.

    Prefers native XGBoost booster binary model (UBJ format, most reliable for SHAP).
    Falls back to joblib if binary not available.

    When ``bin_name`` is set, only that bin's ``bin_models/{bin_name}/models/`` are
    searched; cohort-level paths are not used (avoid wrong model for per-bin SHAP).

    Returns:
        - best_xgboost_model: XGBoost model (loaded from binary or joblib)
    """
    import xgboost as xgb  # type: ignore
    
    age_band_fname = age_band_to_fname(age_band)
    _bin_infix = ("bin_models", bin_name) if bin_name else ()
    _model_base = resolve_step6_cohort_age_dir(PROJECT_ROOT, cohort, age_band)
    _bin_base = _model_base.joinpath(*_bin_infix) if _bin_infix else _model_base

    xgb_binary_candidates: list[Path] = [_bin_base / "models" / "xgboost_model.ubj"]

    for xgb_binary_path in xgb_binary_candidates:
        if xgb_binary_path.exists():
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(str(xgb_binary_path))
            print(f"Loaded best XGBoost model from native binary: {xgb_binary_path}")
            return xgb_model

    xgb_joblib_candidates: list[Path] = [_bin_base / "models" / "xgboost.joblib"]

    xgb_joblib_path = next((p for p in xgb_joblib_candidates if p.exists()), None)

    if xgb_joblib_path is None:
        age_band_h = age_band_fname.replace("_", "-")
        lines = [
            "Best XGBoost model not found. Checked:",
            *[f"  - {p}" for p in xgb_binary_candidates],
            *[f"  - {p}" for p in xgb_joblib_candidates],
            f"Run Step 6: python 6_final_model/run_final_model.py --cohort {cohort} --age_band {age_band_h}",
        ]
        if bin_name:
            lines.append(
                f"Or sync: aws s3 cp s3://pgxdatalake/gold/final_model/{cohort}/{age_band_h}/bin_models/{bin_name}/xgboost_model.ubj "
                f"{_bin_base / 'models' / 'xgboost_model.ubj'}"
            )
        raise FileNotFoundError("\n".join(lines))

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


def _fit_models_for_shap(X: pd.DataFrame, y: pd.Series, cohort: str, age_band: str, random_seed: int = 42, bin_name: str | None = None):
    """
    Load best CatBoost and XGBoost models for SHAP analysis.

    Uses the best models selected by the final model training step.
    """
    # Load best CatBoost model
    cb_model, model_selection_metadata = _load_best_models(cohort, age_band, bin_name=bin_name)

    # Load best XGBoost model (instead of retraining)
    try:
        xgb_model = _load_best_xgboost_model(cohort, age_band, bin_name=bin_name)
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

    return xgb_model, cb_model, model_selection_metadata


def run_shap_analysis(
    cohort: str,
    age_band: str,
    n_background: int = 1000,
    n_eval: int = 2000,
    max_rows: int | None = None,
    bin_name: str | None = None,
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
    if bin_name:
        out_dir = (
            PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname / "bin_models" / bin_name
        )
    else:
        out_dir = (
            PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading final features for {cohort}, {age_band}{f' (bin={bin_name})' if bin_name else ''}...")
    # Load full data including mi_person_key for row IDs
    import duckdb
    age_band_fname = age_band_to_fname(age_band)
    features_path = resolve_step6_train_features_csv(PROJECT_ROOT, cohort, age_band)
    if not features_path.exists():
        raise FileNotFoundError(f"Final features file not found: {features_path}")
    
    con = duckdb.connect()
    try:
        # Use DuckDB to limit rows when max_rows set (EC2 memory); otherwise full read
        _bin_filter = f" WHERE n_event_bin = '{bin_name}'" if bin_name else ""
        if max_rows and max_rows > 0:
            df_full = con.execute(
                f"SELECT * FROM read_csv_auto('{str(features_path)}'){_bin_filter} LIMIT {int(max_rows)}"
            ).df()
            print(f"Loaded sample of {len(df_full)} rows (max_rows={max_rows}{f', bin={bin_name}' if bin_name else ''}) for SHAP.")
        else:
            df_full = con.execute(f"SELECT * FROM read_csv_auto('{str(features_path)}'){_bin_filter}").df()
        if bin_name and len(df_full) == 0:
            print(
                f"[SKIP] No training rows for n_event_bin={bin_name!r} in {features_path.name}; "
                "cannot run per-bin SHAP."
            )
            return False
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
    xgb_clf, cb_clf, model_selection_metadata = _fit_models_for_shap(X_full, y, cohort, age_band, bin_name=bin_name)

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
        # Load a sample from parquet via DuckDB (limit rows for EC2 memory)
        print("\n[Plots] Creating summary plots...")
        con_plot = duckdb.connect()
        try:
            shap_sample_df_plot = con_plot.execute(
                f"SELECT * FROM read_parquet('{str(xgb_shap_sample_path)}') LIMIT {n_eval}"
            ).df()
        finally:
            con_plot.close()
        plot_sample_size = len(shap_sample_df_plot)
        
        # Extract SHAP values (exclude row_id and bias columns)
        shap_cols = [c for c in selected_features if c in shap_sample_df_plot.columns]
        shap_values_plot = shap_sample_df_plot[shap_cols].values
        
        # Get corresponding feature values using row_id if available, otherwise use index
        if row_id is not None and 'row_id' in shap_sample_df_plot.columns:
            row_ids_plot = shap_sample_df_plot['row_id'].values
            # Map row_ids to indices in X_full
            row_id_to_idx = {rid: idx for idx, rid in enumerate(row_id)}
            row_indices = [row_id_to_idx.get(rid, i) for i, rid in enumerate(row_ids_plot)]
            X_plot = X_full.reindex(columns=shap_cols, fill_value=0).iloc[row_indices].reset_index(drop=True)
        else:
            # Use first plot_sample_size rows
            X_plot = X_full.reindex(columns=shap_cols, fill_value=0).iloc[:plot_sample_size].reset_index(drop=True)
        
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
                s3_xgb_imp = f"{_shap_s3_prefix(cohort, age_band, bin_name)}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
                if upload_file_to_s3(xgb_imp_path, s3_xgb_imp):
                    s3_outputs.append(s3_xgb_imp)
            if xgb_shap_sample_path.exists():
                s3_xgb_sample = f"{_shap_s3_prefix(cohort, age_band, bin_name)}/{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
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
            
            # Load parquet sample via DuckDB (limit rows for EC2 memory)
            print("\n[Plots] Creating summary plots...")
            con_cb = duckdb.connect()
            try:
                shap_cb_sample_df_plot = con_cb.execute(
                    f"SELECT * FROM read_parquet('{str(cb_shap_sample_path)}') LIMIT {n_eval}"
                ).df()
            finally:
                con_cb.close()
            plot_sample_size_cb = len(shap_cb_sample_df_plot)
            
            # Extract SHAP values (exclude row_id and bias columns)
            shap_cols_cb = [c for c in selected_features_cb if c in shap_cb_sample_df_plot.columns]
            shap_values_plot_cb = shap_cb_sample_df_plot[shap_cols_cb].values
            
            # Get corresponding feature values using row_id if available, otherwise use index
            if row_id is not None and 'row_id' in shap_cb_sample_df_plot.columns:
                row_ids_plot_cb = shap_cb_sample_df_plot['row_id'].values
                # Map row_ids to indices in X_full
                row_id_to_idx_cb = {rid: idx for idx, rid in enumerate(row_id)}
                row_indices_cb = [row_id_to_idx_cb.get(rid, i) for i, rid in enumerate(row_ids_plot_cb)]
                X_plot_cb = X_full.reindex(columns=shap_cols_cb, fill_value=0).iloc[row_indices_cb].reset_index(drop=True)
            else:
                # Use first plot_sample_size_cb rows
                X_plot_cb = X_full.reindex(columns=shap_cols_cb, fill_value=0).iloc[:plot_sample_size_cb].reset_index(drop=True)
            
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
                    s3_cb_imp = f"{_shap_s3_prefix(cohort, age_band, bin_name)}/{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
                    if upload_file_to_s3(cb_imp_path, s3_cb_imp):
                        s3_outputs.append(s3_cb_imp)
                if cb_shap_sample_path.exists():
                    s3_cb_sample = f"{_shap_s3_prefix(cohort, age_band, bin_name)}/{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet"
                    if upload_file_to_s3(cb_shap_sample_path, s3_cb_sample):
                        s3_outputs.append(s3_cb_sample)
            except ImportError:
                pass
        except Exception as e:
            print(f"[ERROR] CatBoost SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()

    required_models = _required_shap_models(model_selection_metadata)
    missing_models = required_models.difference(models_analyzed)
    if missing_models:
        raise RuntimeError(
            "Step 7 SHAP did not produce all required model outputs. "
            f"Completed: {sorted(models_analyzed)}; missing: {sorted(missing_models)}"
        )

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

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for final models for a given cohort/age_band."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    parser.add_argument(
        "--bin",
        default=None,
        metavar="BIN",
        help="Optional density bin: low|medium|high|extreme. Per-bin SHAP under bin_models/{bin}/. "
        "Omit for cohort-level (aggregate) models under outputs/{cohort}/{age_band}/ only.",
    )
    parser.add_argument(
        "--skip-missing-bin",
        action="store_true",
        help="If --bin is set but Step 6 did not train that bin (no artifacts), exit 0 with a message instead of failing.",
    )
    parser.add_argument(
        "--skip-empty-bin",
        action="store_true",
        help="If --bin has Step 6 artifacts but no matching training rows, exit 0 instead of failing batch runs.",
    )
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
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Max training rows to load for SHAP (default: all). Set e.g. 50000 on EC2 to reduce memory.",
    )
    args = parser.parse_args()

    if args.bin is not None and args.bin not in DENSITY_BINS:
        parser.error(f"--bin must be one of {list(DENSITY_BINS)}, got {args.bin!r}")

    age_band_fname = args.age_band.replace("-", "_")

    if args.bin and args.skip_missing_bin:
        if not final_model_bin_has_trained_artifacts(PROJECT_ROOT, args.cohort, args.age_band, args.bin):
            print(
                f"[SKIP] No Step 6 per-bin model for cohort={args.cohort} age_band={args.age_band} bin={args.bin!r} "
                f"(expected under 6_final_model/outputs/.../bin_models/{args.bin}/). "
                "Train Step 6 for this bin or omit --skip-missing-bin to surface errors."
            )
            sys.exit(0)

    if args.bin and not final_model_bin_has_trained_artifacts(PROJECT_ROOT, args.cohort, args.age_band, args.bin):
        print(
            f"[ERROR] Step 6 has no trained model for bin={args.bin!r} "
            f"({args.cohort} / {args.age_band}). "
            "Run 6_final_model/run_final_model.py (per-bin mode) or use --skip-missing-bin to skip in batch runs."
        )
        sys.exit(1)

    if not args.bin and not cohort_aggregate_final_model_has_artifacts(PROJECT_ROOT, args.cohort, args.age_band):
        print(
            f"[ERROR] No cohort-level Step 6 models under 6_final_model/outputs/{args.cohort}/{age_band_fname}/. "
            "Omit --bin only when aggregate training produced models/, or pass --bin <density_bin> for per-bin SHAP."
        )
        sys.exit(1)

    # Validate per-bin artifacts exist in outputs/ (never model_outputs/).
    # When --bin is given: raise if that bin is missing (fast fail before model load).
    # When no --bin:       print all-bin status as a warning (aggregate SHAP is legacy; per-bin preferred).
    validate_per_bin_outputs(
        PROJECT_ROOT,
        args.cohort,
        args.age_band,
        bins=(args.bin,) if args.bin else None,
        raise_on_missing=bool(args.bin),
    )

    # File logger — logs/7_shap_analysis/{cohort}_{age_band}[_{bin}].log
    _logs_dir = PROJECT_ROOT / "logs" / "7_shap_analysis"
    _logs_dir.mkdir(parents=True, exist_ok=True)
    _bin_suffix = f"_{args.bin}" if args.bin else ""
    _log_path = _logs_dir / f"shap_{args.cohort}_{age_band_fname}{_bin_suffix}.log"
    _logger = logging.getLogger(f"7_shap_analysis.{args.cohort}.{age_band_fname}")
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        _fh = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
        _fh.setFormatter(_fmt)
        _logger.addHandler(_fh)
    _logger.propagate = False
    _logger.info("SHAP analysis start: cohort=%s age_band=%s bin=%s", args.cohort, args.age_band, args.bin)

    if args.bin:
        out_dir = (
            PROJECT_ROOT / "7_shap_analysis" / "outputs" / args.cohort / age_band_fname / "bin_models" / args.bin
        )
    else:
        out_dir = (
            PROJECT_ROOT / "7_shap_analysis" / "outputs" / args.cohort / age_band_fname
        )

    model_base = resolve_step6_cohort_age_dir(PROJECT_ROOT, args.cohort, args.age_band)
    metadata_probe_path = model_base / f"{args.cohort}_{age_band_fname}_model_selection_metadata.json"
    if args.bin:
        bin_metadata_probe_path = model_base / "bin_models" / args.bin / f"{args.cohort}_{age_band_fname}_model_selection_metadata.json"
        if bin_metadata_probe_path.exists():
            metadata_probe_path = bin_metadata_probe_path
    try:
        with open(metadata_probe_path) as f:
            metadata_probe = json.load(f)
    except Exception:
        metadata_probe = {}
    required_model_outputs = _required_shap_models(metadata_probe)

    # Check for existing local outputs (idempotency - check local first)
    # SHAP always needs XGBoost for FFA, plus the best-overall model when it differs.
    _pfx = f"{args.cohort}_{age_band_fname}"
    expected_outputs = [
        f"{_pfx}_shap_global_importance_xgboost.csv",
        f"{_pfx}_shap_sample_values_xgboost.parquet",
    ]
    if "catboost" in required_model_outputs:
        expected_outputs.extend(
            [
                f"{_pfx}_shap_global_importance_catboost.csv",
                f"{_pfx}_shap_sample_values_catboost.parquet",
            ]
        )
    
    optional_outputs = [
        f"{_pfx}_shap_summary_bar_xgboost.png",
        f"{_pfx}_shap_summary_beeswarm_xgboost.png",
        f"{_pfx}_shap_summary_bar_catboost.png",
        f"{_pfx}_shap_summary_beeswarm_catboost.png",
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
                        s3_path = f"{_shap_s3_prefix(args.cohort, args.age_band, args.bin)}/{fname}"
                    elif fname.endswith('.parquet'):
                        s3_path = f"{_shap_s3_prefix(args.cohort, args.age_band, args.bin)}/{fname}"
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
                    metadata={"n_background": args.n_background, "n_eval": args.n_eval, "models_analyzed": ["xgboost", "catboost"]},
                    output_paths=s3_outputs,
                )
        except ImportError:
            pass  # S3 upload is optional
        
        return

    # Check S3 for existing outputs (idempotency - fallback if local doesn't exist)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists

        s3_prefix = _shap_s3_prefix(args.cohort, args.age_band, args.bin)
        s3_key_prefix = _shap_s3_key_prefix(args.cohort, args.age_band, args.bin)
        s3_output_paths = [
            *[
                f"{s3_prefix}/{fname}"
                for fname in expected_outputs
            ],
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
                # Download required XGBoost and CatBoost outputs
                for fname in expected_outputs:
                    s3_key = f"{s3_key_prefix}/{fname}"
                    local_path = out_dir / fname
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                        print(f"Downloaded {local_path} from S3")
                        downloaded_files.append(local_path)
                    except Exception as e:
                        print(f"Warning: Could not download {s3_key}: {e}")
                
                # Try to download plot outputs (optional)
                for fname in optional_outputs:
                    s3_key = f"{s3_key_prefix}/{fname}"
                    local_path = out_dir / fname
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                        print(f"Downloaded {local_path} from S3")
                        downloaded_files.append(local_path)
                    except Exception:
                        pass
                
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
        max_rows=args.max_rows,
        bin_name=args.bin,
    )  
    if not success:
        if args.bin and args.skip_empty_bin:
            print(
                f"[SKIP] Step 7 produced no SHAP outputs for {args.cohort}/{args.age_band} "
                f"bin={args.bin!r}; continuing because --skip-empty-bin was set."
            )
            sys.exit(0)
        _logger.error("No models were successfully analyzed.")
        print("\n[ERROR] No models were successfully analyzed.")
        print("This step cannot complete without at least one model being analyzed.")
        sys.exit(1)

    _logger.info("SHAP analysis complete: cohort=%s age_band=%s bin=%s", args.cohort, args.age_band, args.bin)
    try:
        from py_helpers.fe_monitor import mirror_log_to_s3
        mirror_log_to_s3("7_shap_analysis", args.cohort, args.age_band, _log_path, _logger)
    except Exception:
        pass


if __name__ == "__main__":
    main()

