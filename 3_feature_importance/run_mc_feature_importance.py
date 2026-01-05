#!/usr/bin/env python3
"""
Monte-Carlo feature-importance runner for the final, leakage-filtered feature set.

This script:
  - Mirrors `build_final_features` logic from `6_final_model_selection/run_final_model.py`
    to assemble the patient-level feature matrix (with target-leakage
    removal already applied).
  - Runs N Monte-Carlo train/test splits with XGBoost (CPU on Linux, GPU on Windows if available).
  - Aggregates feature importances across runs, producing a CSV in the same
    schema as the legacy `*_aggregated_feature_importance.csv` files:

      feature,
      scaled_importance_mean,
      scaled_importance_std,
      scaled_importance_count,
      importance_mean,
      importance_std,
      recall_mean,
      logloss_mean

Usage (example):

    python 3_feature_importance/run_mc_feature_importance.py \
        --cohort opioid_ed \
        --age_band 13-24 \
        --n_runs 25
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, recall_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import duckdb
import matplotlib.pyplot as plt

# Ensure project root on path so we can import final_model utilities
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # noqa: E402
from py_helpers.env_utils import get_xgb_cpu_nthread, get_data_root, is_linux  # noqa: E402

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET  # noqa: E402
except ImportError:
    import boto3  # noqa: E402
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"


def _load_feature_table(path: Path, required: bool = True) -> pd.DataFrame:
    """Simplified loader mirroring 6_final_model.run_final_model._load_feature_table."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required feature file not found: {path}")
        print(f"Feature file not found (skipping): {path}")
        return pd.DataFrame()
    print(f"Loading features from {path}")
    return pd.read_csv(path)


def _remove_target_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious target-leakage features based on naming conventions,
    mirroring the logic in the legacy remove_target_leakage.py script and
    6_final_model.run_final_model.remove_target_leakage_features.

    This drops:
      - Columns starting with 'post_'
      - Columns containing 'time_to' / 'time_to_'
      - Time-window features with suffixes like '_30d', '_90d', '_180d'
        (except those with 'interval' in the name)
      - Datetime helper columns: 'target_time', 'first_time'
      - DTW-derived features (any column with 'dtw' in its name)
      - Any feature whose name contains 'F1120'
    """
    cols = list(df.columns)
    leakage: set[str] = set()

    post_features = [c for c in cols if c.startswith("post_")]
    leakage.update(post_features)

    time_to_features = [
        c for c in cols if "time_to" in c.lower() or "time_to_" in c.lower()
    ]
    leakage.update(time_to_features)

    time_window_features = [
        c
        for c in cols
        if any(x in c for x in ["_30d", "_90d", "_180d"])
        and "interval" not in c.lower()
    ]
    leakage.update(time_window_features)

    datetime_features = [c for c in ("target_time", "first_time") if c in cols]
    leakage.update(datetime_features)

    dtw_features = [c for c in cols if "dtw" in c.lower()]
    leakage.update(dtw_features)

    f1120_features = [c for c in cols if "F1120" in c.upper()]
    leakage.update(f1120_features)

    if leakage:
        kept = [c for c in cols if c not in leakage]
        print(
            "Removing potential target-leakage features:\n  "
            + ", ".join(sorted(leakage))
        )
        return df[kept].copy()

    return df


def _validate_s3_file_has_controls(s3_path: str) -> dict:
    """
    Validate that an S3 parquet file contains both cases (target=1) and controls (target=0).
    Uses DuckDB's S3 support to query without downloading the entire file.
    
    Returns:
        dict with keys: has_controls (bool), n_cases (int), n_controls (int), error (str or None)
    """
    import duckdb
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                COUNT(*) FILTER (WHERE target = 0) AS n_controls
            FROM read_parquet('{s3_path}')
            """
        ).fetchone()
        
        n_cases = result[0] if result else 0
        n_controls = result[1] if result else 0
        has_controls = n_controls > 0
        
        return {
            "has_controls": has_controls,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "error": None,
        }
    except Exception as e:
        return {
            "has_controls": False,
            "n_cases": 0,
            "n_controls": 0,
            "error": str(e),
        }
    finally:
        con.close()


def _resolve_model_events_path(cohort: str, age_band: str) -> Path:
    """
    Resolve the path to model_events.parquet, checking multiple locations.
    
    Priority on Linux/EC2:
    1. get_data_root()/4a_model_data/... (/mnt/nvme/4a_model_data/...)
    2. PROJECT_ROOT/4a_model_data/... (fallback)
    3. Try downloading from S3 to get_data_root() if not found locally
    
    Priority on Windows:
    1. PROJECT_ROOT/4a_model_data/... (Windows/local dev)
    2. get_data_root()/4a_model_data/... (fallback)
    3. Try downloading from S3 to PROJECT_ROOT if not found locally
    
    Returns:
        Path to model_events.parquet file
    """
    age_band_fname = age_band_to_fname(age_band)
    data_root = get_data_root()
    is_linux_system = is_linux()
    
    # Build candidate paths - prioritize data root on Linux, project root on Windows
    if is_linux_system:
        # On Linux/EC2: prioritize /mnt/nvme
        candidates = [
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer data root on Linux
        download_dest = candidates[0]
    else:
        # On Windows: prioritize project root
        candidates = [
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer project root on Windows
        download_dest = candidates[0]
    
    # Check each candidate
    for path in candidates:
        if path.exists():
            print(f"Found model_events.parquet at: {path}")
            return path
    
    # Log which paths we checked
    print(f"Model data not found locally. Checked paths:")
    for path in candidates:
        print(f"  - {path} (exists: {path.exists()})")
    
    # If not found locally, try downloading from S3
    s3_key_candidates = [
        f"gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet",
        f"gold/model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet",
        f"gold/model_data/{cohort}/{age_band}/model_events.parquet",
    ]
    
    download_dest.parent.mkdir(parents=True, exist_ok=True)
    
    for s3_key in s3_key_candidates:
        try:
            # Check if file exists in S3
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            s3_path = f"s3://{S3_BUCKET}/{s3_key}"
            
            # Validate controls BEFORE downloading (using DuckDB S3 support)
            print(f"Checking S3 file for controls: {s3_path}")
            validation_result = _validate_s3_file_has_controls(s3_path)
            
            if validation_result.get("error"):
                print(f"Warning: Could not validate S3 file {s3_path}: {validation_result['error']}")
                print("Proceeding with download and will validate after...")
            elif not validation_result.get("has_controls", False):
                print(
                    f"ERROR: S3 file {s3_path} is missing controls! "
                    f"Cases: {validation_result.get('n_cases', 0)}, Controls: {validation_result.get('n_controls', 0)}"
                )
                print(
                    f"This file should be regenerated with controls. "
                    f"Please run: python 4a_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
                )
                print("Skipping this S3 file and trying next candidate...")
                continue  # Skip this S3 file, try next candidate
            
            # Download the file
            print(f"Downloading model_events.parquet from S3: {s3_path}")
            print(f"Downloading to: {download_dest}")
            import io
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            with open(download_dest, 'wb') as f:
                f.write(obj['Body'].read())
            print(f"Saved to: {download_dest}")
            
            # Validate again after download (double-check)
            import duckdb
            con = duckdb.connect()
            try:
                result = con.execute(
                    f"""
                    SELECT 
                        COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                        COUNT(*) FILTER (WHERE target = 0) AS n_controls
                    FROM read_parquet('{download_dest}')
                    """
                ).fetchone()
                
                n_cases = result[0] if result else 0
                n_controls = result[1] if result else 0
                if n_controls == 0:
                    print(
                        f"ERROR: Downloaded file is missing controls! "
                        f"Cases: {n_cases}, Controls: {n_controls}"
                    )
                    print(
                        f"This file should be regenerated with controls. "
                        f"Please run: python 4a_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
                    )
                    # Delete the invalid file
                    download_dest.unlink()
                    print("Deleted invalid file. Trying next S3 candidate...")
                    continue
                else:
                    print(f"Validation passed: {n_cases} cases, {n_controls} controls")
            finally:
                con.close()
            
            return download_dest
        except Exception as e:
            print(f"S3 key not found or error: {s3_key} - {e}")
            continue
    
    # If all checks failed, raise error with helpful message
    error_msg = (
        f"Model data not found for cohort={cohort}, age_band={age_band}.\n"
        f"Checked locations:\n"
    )
    for path in candidates:
        error_msg += f"  - {path} (exists: {path.exists()})\n"
    error_msg += f"\nS3 locations checked:\n"
    for s3_key in s3_key_candidates:
        error_msg += f"  - s3://{S3_BUCKET}/{s3_key}\n"
    raise FileNotFoundError(error_msg)


def build_final_features_for_mc(cohort: str, age_band: str) -> pd.DataFrame:
    """
    Build feature matrix for Step 3 (Feature Importance) from raw model_events.parquet.
    
    Step 3 runs BEFORE Step 5 (PGx Feature Engineering), so it uses only raw features
    from the model_events.parquet dataset. Step 5 will add PGx features later, which
    will be used in Step 6 (Final Model Selection).
    
    Note: FP-Growth, BupaR, and DTW features are no longer used in the pipeline;
    they are only used for dashboard visualizations in Step 10.
    
    Inputs:
      - 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
        (or model_events_no_protocols.parquet if Step 4b has run)
    """
    age_band_fname = age_band_to_fname(age_band)

    events_path = _resolve_model_events_path(cohort, age_band)

    print(f"Loading model data (cases + controls) from {events_path}")
    con = duckdb.connect()
    grouped = con.execute(
        f"""
        SELECT
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            CAST(target AS INTEGER)        AS target,
            COUNT(*)                       AS n_events
        FROM read_parquet('{events_path}')
        GROUP BY mi_person_key, target
        """
    ).df()
    con.close()

    grouped["target"] = grouped["target"].astype(int).clip(lower=0, upper=1)

    # Step 3 uses only raw features from model_events.parquet
    # No feature engineering outputs are loaded here (Step 5 adds PGx features later)
    final = grouped.copy()
    final = final.dropna(subset=["target"])
    final = _remove_target_leakage_features(final)
    return final


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare numeric feature matrix X and label y from the assembled DataFrame."""
    feature_cols = [c for c in df.columns if c not in ("mi_person_key", "target")]
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns available for MC feature importance.")

    X = df[numeric_feature_cols].replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(0)
    y = df["target"].astype(int)

    return X, y, numeric_feature_cols


def run_mc_feature_importance(
    cohort: str,
    age_band: str,
    n_runs: int = 25,
    test_size: float = 0.3,
    random_seed: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    """Run Monte-Carlo CV for multiple models and aggregate feature importances.

    Models:
      - XGBoost (gradient boosted trees, CPU on Linux, GPU on Windows if available)
      - XGBoost RF (XGBRFClassifier, CPU on Linux, GPU on Windows if available)
      - CatBoost (if installed; CPU only)

    This function is idempotent - it will skip if results already exist locally or in S3,
    unless force=True is specified.
    """
    age_band_fname = age_band_to_fname(age_band)
    out_dir = PROJECT_ROOT / "3_feature_importance" / "outputs" / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing aggregated results (idempotency)
    agg_path = out_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    
    if not force and agg_path.exists():
        print(f"✓ Aggregated feature importance already exists locally: {agg_path}")
        print("  Skipping Monte-Carlo feature importance computation.")
        print("  Use --force to rerun.")
        return pd.read_csv(agg_path)

    # Check S3 if not found locally
    if not force:
        s3_key_agg = (
            f"gold/feature_importance/{cohort}/{age_band}/"
            f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_agg)
            print(f"✓ Aggregated feature importance exists in S3: s3://{S3_BUCKET}/{s3_key_agg}")
            print("  Downloading instead of recomputing...")
            import io
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_agg)
            agg_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            agg_df.to_csv(agg_path, index=False)
            print(f"  Saved locally: {agg_path}")
            return agg_df
        except Exception:
            # File doesn't exist in S3, proceed with computation
            pass

    # Assemble final feature matrix (with leakage removal baked in)
    df = build_final_features_for_mc(cohort, age_band)
    if df.empty:
        raise ValueError(f"No data assembled for cohort={cohort}, age_band={age_band}")

    X, y, feature_names = _prepare_xy(df)

    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(
            "XGBoost is required for Monte-Carlo feature importance. "
            "Install with: pip install xgboost"
        ) from exc

    try:
        from catboost import CatBoostClassifier  # type: ignore

        have_catboost = True
    except Exception:
        have_catboost = False
        print(
            "CatBoost not available; skipping CatBoost feature importance. "
            "Install with: pip install catboost"
        )

    rng = np.random.default_rng(random_seed)

    model_keys = ["xgb", "xgb_rf"]
    if have_catboost:
        model_keys.append("catboost")

    # Storage for per-run metrics and importances, per model
    per_feature_importances: Dict[str, Dict[str, List[float]]] = {
        m: {f: [] for f in feature_names} for m in model_keys
    }
    per_feature_scaled: Dict[str, Dict[str, List[float]]] = {
        m: {f: [] for f in feature_names} for m in model_keys
    }
    aucs: Dict[str, List[float]] = {m: [] for m in model_keys}
    pr_aucs: Dict[str, List[float]] = {m: [] for m in model_keys}
    recalls: Dict[str, List[float]] = {m: [] for m in model_keys}
    loglosses: Dict[str, List[float]] = {m: [] for m in model_keys}

    nthread = get_xgb_cpu_nthread()
    
    # Determine device: CPU on Linux, CUDA on Windows (if available)
    device = "cpu" if is_linux() else "cuda"

    for run_idx in range(n_runs):
        rs = int(rng.integers(0, np.iinfo(np.int32).max))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )

        # --------------------
        # XGBoost (boosting)
        # --------------------
        xgb_clf = xgb.XGBClassifier(
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
            random_state=rs,
        )
        try:
            xgb_clf.fit(X_train, y_train)
        except Exception:
            # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
            xgb_clf.set_params(tree_method="hist")
            if "device" in xgb_clf.get_params():
                xgb_clf.set_params(device="cpu")
            xgb_clf.fit(X_train, y_train)

        for model_name, clf in [("xgb", xgb_clf)]:
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            recalls[model_name].append(recall_score(y_test, y_pred))
            loglosses[model_name].append(log_loss(y_test, y_proba))
            aucs[model_name].append(roc_auc_score(y_test, y_proba))
            pr_aucs[model_name].append(average_precision_score(y_test, y_proba))

            importances = np.asarray(clf.feature_importances_, dtype=float)
            if importances.shape[0] != len(feature_names):
                raise RuntimeError(
                    f"Feature importance length mismatch for model {model_name}."
                )
            mean_imp = float(importances.mean()) if importances.size > 0 else 0.0
            if mean_imp > 0:
                scaled = importances / mean_imp
            else:
                scaled = np.zeros_like(importances)

            for fname, imp, imp_scaled in zip(feature_names, importances, scaled):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        # --------------------
        # XGBoost RF (XGBRFClassifier)
        # --------------------
        xgbrf_clf = xgb.XGBRFClassifier(
            n_estimators=500,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=rs + 1,
        )
        try:
            xgbrf_clf.fit(X_train, y_train)
        except Exception:
            # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
            xgbrf_clf.set_params(tree_method="hist")
            if "device" in xgbrf_clf.get_params():
                xgbrf_clf.set_params(device="cpu")
            xgbrf_clf.fit(X_train, y_train)

        for model_name, clf in [("xgb_rf", xgbrf_clf)]:
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            recalls[model_name].append(recall_score(y_test, y_pred))
            loglosses[model_name].append(log_loss(y_test, y_proba))
            aucs[model_name].append(roc_auc_score(y_test, y_proba))
            pr_aucs[model_name].append(average_precision_score(y_test, y_proba))

            importances = np.asarray(clf.feature_importances_, dtype=float)
            if importances.shape[0] != len(feature_names):
                raise RuntimeError(
                    f"Feature importance length mismatch for model {model_name}."
                )
            mean_imp = float(importances.mean()) if importances.size > 0 else 0.0
            if mean_imp > 0:
                scaled = importances / mean_imp
            else:
                scaled = np.zeros_like(importances)

            for fname, imp, imp_scaled in zip(feature_names, importances, scaled):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        # --------------------
        # CatBoost (optional)
        # --------------------
        if have_catboost:
            cb_clf = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                grow_policy="SymmetricTree",
                random_seed=rs + 2,
                verbose=False,
            )
            try:
                cb_clf.fit(X_train, y_train)
            except Exception:
                # Fallback is still CPU since CatBoost manages devices internally
                cb_clf = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    grow_policy="SymmetricTree",
                    random_seed=rs + 2,
                    verbose=False,
                )
                cb_clf.fit(X_train, y_train)

            model_name = "catboost"
            y_proba = cb_clf.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            recalls[model_name].append(recall_score(y_test, y_pred))
            loglosses[model_name].append(log_loss(y_test, y_proba))
            aucs[model_name].append(roc_auc_score(y_test, y_proba))
            pr_aucs[model_name].append(average_precision_score(y_test, y_proba))

            importances = np.asarray(cb_clf.get_feature_importance(), dtype=float)
            if importances.shape[0] != len(feature_names):
                raise RuntimeError(
                    f"Feature importance length mismatch for model {model_name}."
                )
            mean_imp = float(importances.mean()) if importances.size > 0 else 0.0
            if mean_imp > 0:
                scaled = importances / mean_imp
            else:
                scaled = np.zeros_like(importances)

            for fname, imp, imp_scaled in zip(feature_names, importances, scaled):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        print(
            f"[MC] Run {run_idx + 1}/{n_runs} "
            f"XGB_recall={recalls['xgb'][-1]:.4f} "
            f"XGB_logloss={loglosses['xgb'][-1]:.4f}"
        )

    # Aggregate across runs per model
    # (out_dir already created above during idempotency check)

    model_label_map = {
        "xgb": "xgboost",
        "xgb_rf": "xgboost_rf",
        "catboost": "catboost",
    }

    results = {}

    for model_name in model_keys:
        records = []
        recall_mean = float(np.mean(recalls[model_name])) if recalls[model_name] else float("nan")
        logloss_mean = (
            float(np.mean(loglosses[model_name])) if loglosses[model_name] else float("nan")
        )
        auc_mean = float(np.mean(aucs[model_name])) if aucs[model_name] else float("nan")
        pr_auc_mean = float(np.mean(pr_aucs[model_name])) if pr_aucs[model_name] else float("nan")

        for fname in feature_names:
            imp_values = np.array(
                per_feature_importances[model_name][fname], dtype=float
            )
            scaled_values = np.array(
                per_feature_scaled[model_name][fname], dtype=float
            )

            records.append(
                {
                    "feature": fname,
                    "scaled_importance_mean": float(scaled_values.mean())
                    if scaled_values.size
                    else 0.0,
                    "scaled_importance_std": float(scaled_values.std(ddof=0))
                    if scaled_values.size
                    else 0.0,
                    "scaled_importance_count": int(
                        np.count_nonzero(scaled_values > 0.0)
                    ),
                    "importance_mean": float(imp_values.mean())
                    if imp_values.size
                    else 0.0,
                    "importance_std": float(imp_values.std(ddof=0))
                    if imp_values.size
                    else 0.0,
                    "recall_mean": recall_mean,
                    "logloss_mean": logloss_mean,
                    "auc_mean": auc_mean,
                    "pr_auc_mean": pr_auc_mean,
                }
            )

        fi_df = pd.DataFrame.from_records(records)
        fi_df = fi_df.sort_values("scaled_importance_mean", ascending=False)
        label = model_label_map[model_name]

        out_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_{label}_feature_importance_mc{n_runs}.csv"
        )
        fi_df.to_csv(out_path, index=False)
        print(
            f"\nSaved Monte-Carlo {label} feature importances to {out_path} "
            f"(top 10 features shown below)."
        )
        print(fi_df.head(10).to_string(index=False))
        results[model_name] = fi_df

        # Basic visuals: top 50 barplot and raw vs scaled scatter
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        top_n = 50
        top_df = fi_df.head(top_n)

        plt.figure(figsize=(10, max(6, top_n * 0.18)))
        plt.barh(top_df["feature"][::-1], top_df["scaled_importance_mean"][::-1])
        plt.xlabel("Scaled importance (mean)")
        plt.title(
            f"{label} top {top_n} features\n"
            f"recall_mean={recall_mean:.3f}, "
            f"logloss_mean={logloss_mean:.3f}, "
            f"AUC_mean={auc_mean:.3f}, PR-AUC_mean={pr_auc_mean:.3f}"
        )
        plt.tight_layout()
        bar_path = (
            plots_dir
            / f"{cohort}_{age_band_fname}_{label}_top{top_n}_features_mc{n_runs}.png"
        )
        plt.savefig(bar_path, dpi=150)
        plt.close()

        # Scatter: raw vs scaled importance
        plt.figure(figsize=(6, 5))
        plt.scatter(
            fi_df["importance_mean"],
            fi_df["scaled_importance_mean"],
            alpha=0.6,
            s=10,
        )
        plt.xlabel("Raw importance (mean)")
        plt.ylabel("Scaled importance (mean)")
        plt.title(f"{label} importance: raw vs scaled")
        plt.tight_layout()
        scatter_path = (
            plots_dir
            / f"{cohort}_{age_band_fname}_{label}_normalized_vs_scaled_mc{n_runs}.png"
        )
        plt.savefig(scatter_path, dpi=150)
        plt.close()

    # For backward compatibility, also write an "aggregated" file name based on XGBoost boosting
    if "xgb" in results:
        agg_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
        results["xgb"].to_csv(agg_path, index=False)
        print(f"Saved aggregated feature importance (XGBoost) to {agg_path}")

    # Return the XGBoost boosting table by default
    return results.get("xgb", pd.DataFrame())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Monte-Carlo XGBoost feature importance on final features. "
        "This script is idempotent - it will skip if results already exist unless --force is used."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=25,
        help="Number of Monte-Carlo CV runs (default: 25)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if results already exist",
    )
    args = parser.parse_args()

    run_mc_feature_importance(
        cohort=args.cohort,
        age_band=args.age_band,
        n_runs=args.n_runs,
        force=args.force,
    )


if __name__ == "__main__":
    main()

