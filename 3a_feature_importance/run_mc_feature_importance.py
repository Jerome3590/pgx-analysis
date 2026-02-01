#!/usr/bin/env python3
"""
Monte-Carlo feature-importance runner for the final, leakage-filtered feature set.

This script:
  - Uses **cohort data** (Step 2 cohort.parquet) to build the patient-level feature matrix.
    It does not depend on Step 4 (model_events.parquet).
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
import os
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
from py_helpers.s3_utils import normalize_cohort_name, get_cohort_parquet_path  # noqa: E402

# Event years to combine when loading cohort data (matches 4_model_data default)
DEFAULT_EVENT_YEARS = [2016, 2017, 2018, 2019]

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET  # noqa: E402
except ImportError:
    import boto3  # noqa: E402
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

# Historical bucket for aggregated FI (written here and read by 1b; never cleared)
PGX_REPO_BUCKET = "pgx-repository"
PGX_REPO_FI_PREFIX = "pgx-analysis/3_feature_importance/outputs"


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


def _validate_cohort_file_has_controls(path_or_s3: str) -> dict:
    """
    Validate that a cohort.parquet file contains both cases (is_target_case=1) and controls (is_target_case=0).
    path_or_s3: local path or s3:// URL.
    Returns:
        dict with keys: has_controls (bool), n_cases (int), n_controls (int), error (str or None)
    """
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE is_target_case = 1) AS n_cases,
                COUNT(*) FILTER (WHERE is_target_case = 0) AS n_controls
            FROM read_parquet('{path_or_s3}')
            """
        ).fetchone()
        n_cases = int(result[0]) if result else 0
        n_controls = int(result[1]) if result else 0
        return {
            "has_controls": n_controls > 0,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "error": None,
        }
    except Exception as e:
        return {"has_controls": False, "n_cases": 0, "n_controls": 0, "error": str(e)}
    finally:
        con.close()


def _cohort_local_root() -> Path:
    """Local root for syncing cohort.parquet from S3 (NVMe on EC2). DuckDB uses only local paths."""
    return get_data_root() / "gold" / "cohorts"


def _resolve_cohort_parquet_paths(cohort: str, age_band: str) -> List[str]:
    """
    Resolve paths to cohort.parquet for (cohort, age_band) across DEFAULT_EVENT_YEARS.
    Returns only **local** paths: if a file exists only on S3, it is synced to local (NVMe/data) first,
    so DuckDB never sees mixed local/S3 paths.
    """
    cohort_slug = normalize_cohort_name(cohort)
    local_root = _cohort_local_root()
    # Candidate local roots to check before syncing (same layout as 4_model_data)
    data_root = get_data_root()
    check_roots: List[Path] = [
        data_root / "gold" / "cohorts",
        data_root / "data" / "gold_cohorts",
        PROJECT_ROOT / "data" / "gold_cohorts",
    ]
    if os.environ.get("LOCAL_DATA_PATH"):
        check_roots.insert(0, Path(os.environ["LOCAL_DATA_PATH"]))

    found: List[str] = []
    for year in DEFAULT_EVENT_YEARS:
        rel = f"cohort_name={cohort_slug}/event_year={year}/age_band={age_band}/cohort.parquet"
        local_path = local_root / rel
        # Prefer existing local file from any check root
        for root in check_roots:
            p = root / rel
            if p.exists():
                found.append(str(p))
                break
        else:
            # Not found locally: try S3 and sync to local_root (NVMe) then use that path
            s3_path = get_cohort_parquet_path(cohort_slug, age_band, year)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(s3_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                s3_client.head_object(Bucket=bucket, Key=key)
            except Exception:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                with open(local_path, "wb") as f:
                    f.write(obj["Body"].read())
                print(f"Synced cohort.parquet from S3 to local: {local_path}")
                found.append(str(local_path))
            except Exception as e:
                print(f"Warning: could not sync {s3_path} to {local_path}: {e}")
    return found


def build_final_features_for_mc(cohort: str, age_band: str, prefer_filtered: bool = True) -> pd.DataFrame:
    """
    Build feature matrix for Step 3 (Feature Importance) from **cohort data** (Step 2 cohort.parquet).

    Does not use Step 4 (model_events.parquet). Loads cohort.parquet for (cohort, age_band)
    across DEFAULT_EVENT_YEARS (2016–2019); files missing locally are synced from S3 to
    local (NVMe) first so DuckDB sees only local paths. Aggregates to patient-level:
    mi_person_key, target = MAX(is_target_case), n_events = COUNT(*).

    prefer_filtered is kept for API compatibility but has no effect when using cohort data.
    """
    cohort_paths = _resolve_cohort_parquet_paths(cohort, age_band)
    if not cohort_paths:
        raise FileNotFoundError(
            f"Cohort data not found for cohort={cohort}, age_band={age_band}. "
            f"Checked local gold/cohorts and S3 gold/cohorts/ for event years {DEFAULT_EVENT_YEARS}. "
            "Run Step 2 (2_create_cohort) first to produce cohort.parquet files."
        )

    # Validate at least one file has controls
    for path in cohort_paths[:1]:
        v = _validate_cohort_file_has_controls(path)
        if v.get("error"):
            print(f"Warning: Could not validate cohort file: {path} - {v['error']}")
        elif not v.get("has_controls", False):
            print(
                f"Warning: Cohort file has no controls: {path} "
                f"(cases={v.get('n_cases', 0)}, controls={v.get('n_controls', 0)})"
            )

    print(f"Loading cohort data (cases + controls) from {len(cohort_paths)} file(s)")
    paths_sql = ", ".join(repr(p) for p in cohort_paths)
    con = duckdb.connect()
    grouped = con.execute(
        f"""
        SELECT
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            CAST(MAX(is_target_case) AS INTEGER) AS target,
            COUNT(*)::BIGINT AS n_events
        FROM read_parquet([{paths_sql}])
        GROUP BY mi_person_key
        """
    ).df()
    con.close()

    grouped["target"] = grouped["target"].astype(int).clip(lower=0, upper=1)

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
    baseline: bool = False,
) -> pd.DataFrame:
    """Run Monte-Carlo CV for multiple models and aggregate feature importances.

    Uses cohort data (Step 2 cohort.parquet) to build the feature matrix. Does not
    depend on Step 4 (model_events.parquet).

    Default (baseline=False): write to outputs/{cohort}/. Baseline aggregated FI is
    expected to already exist (e.g. on S3); the 1b event filter uses it for the FI-based filter.

    When baseline=True: write to outputs/{cohort}/_baseline/. Use this only when
    generating baseline FI for the first time; normal pipeline runs should omit --baseline.

    Models:
      - XGBoost (gradient boosted trees, CPU on Linux, GPU on Windows if available)
      - XGBoost RF (XGBRFClassifier, CPU on Linux, GPU on Windows if available)
      - CatBoost (if installed; CPU only)

    This function is idempotent - it will skip if results already exist locally or in S3,
    unless force=True is specified.
    """
    age_band_fname = age_band_to_fname(age_band)
    # Optional: write to NVMe on EC2 (set PGX_FEATURE_IMPORTANCE_OUTPUTS e.g. /mnt/nvme/feature_importance/outputs)
    _outputs_base = os.environ.get("PGX_FEATURE_IMPORTANCE_OUTPUTS")
    if _outputs_base:
        out_dir = Path(_outputs_base) / cohort
        print(f"[INFO] Writing Step 3 outputs to NVMe: {out_dir}")
    else:
        out_dir = PROJECT_ROOT / "3a_feature_importance" / "outputs" / cohort
    if baseline:
        out_dir = out_dir / "_baseline"
        print("[INFO] Baseline run: writing to _baseline subfolder (original aggregated FI for 1b event filter)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing aggregated results (idempotency)
    agg_path = out_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"

    if not force and agg_path.exists():
        print(f"✓ Aggregated feature importance already exists locally: {agg_path}")
        print("  Skipping Monte-Carlo feature importance computation.")
        print("  Use --force to rerun.")
        return pd.read_csv(agg_path)

    # Check S3 if not found locally (use _baseline in S3 key when baseline=True)
    if not force:
        s3_suffix = "_baseline/" if baseline else ""
        s3_key_agg = (
            f"gold/feature_importance/{cohort}/{age_band}/{s3_suffix}"
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

    # Assemble final feature matrix from cohort data (with leakage removal baked in)
    df = build_final_features_for_mc(cohort, age_band, prefer_filtered=not baseline)
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
        print(f"[INFO] CatBoost is available - will run for all {n_runs} MC CV splits")
    else:
        print(f"[INFO] CatBoost not available - only running XGBoost models for {n_runs} MC CV splits")

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

    print(f"\n[INFO] Starting Monte-Carlo CV with {n_runs} splits")
    print(f"[INFO] Models to run: {', '.join(model_keys)}")
    
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

            for fname, imp, imp_scaled in zip(feature_names, importances, scaled, strict=True):
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

            for fname, imp, imp_scaled in zip(feature_names, importances, scaled, strict=True):
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

            for fname, imp, imp_scaled in zip(feature_names, importances, scaled, strict=True):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        # Log progress for all models
        log_msg = f"[MC] Run {run_idx + 1}/{n_runs} "
        log_msg += f"XGB_recall={recalls['xgb'][-1]:.4f} XGB_logloss={loglosses['xgb'][-1]:.4f}"
        if "xgb_rf" in recalls and recalls["xgb_rf"]:
            log_msg += f" XGB_RF_recall={recalls['xgb_rf'][-1]:.4f}"
        if "catboost" in recalls and recalls["catboost"]:
            log_msg += f" CatBoost_recall={recalls['catboost'][-1]:.4f}"
        print(log_msg)

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
        
        # Filter zero-importance features and remove duplicates before saving
        agg_df = results["xgb"].copy()
        
        # Filter out features with zero or negative importance (no signal)
        initial_count = len(agg_df)
        if "scaled_importance_mean" in agg_df.columns:
            agg_df = agg_df[agg_df["scaled_importance_mean"] > 1e-10].copy()
            filtered_count = len(agg_df)
            if filtered_count < initial_count:
                print(f"[INFO] Filtered out {initial_count - filtered_count} features with zero/negative importance")
                print(f"[INFO] Keeping {filtered_count} features with importance > 0")
        elif "importance_mean" in agg_df.columns:
            agg_df = agg_df[agg_df["importance_mean"] > 1e-10].copy()
            filtered_count = len(agg_df)
            if filtered_count < initial_count:
                print(f"[INFO] Filtered out {initial_count - filtered_count} features with zero/negative importance")
                print(f"[INFO] Keeping {filtered_count} features with importance > 0")
        
        # Remove duplicate features (keep first occurrence, which should be highest importance after sorting)
        initial_count = len(agg_df)
        agg_df = agg_df.drop_duplicates(subset=["feature"], keep="first")
        if len(agg_df) < initial_count:
            print(f"[INFO] Removed {initial_count - len(agg_df)} duplicate features")
        
        # Ensure sorted by importance (descending)
        if "scaled_importance_mean" in agg_df.columns:
            agg_df = agg_df.sort_values("scaled_importance_mean", ascending=False)
        elif "importance_mean" in agg_df.columns:
            agg_df = agg_df.sort_values("importance_mean", ascending=False)
        
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved aggregated feature importance (XGBoost) to {agg_path}")
        print(f"[INFO] Final aggregated CSV contains {len(agg_df)} unique features with signal")
        
        # Upload to S3: pgxdatalake (pipeline) and pgx-repository (historical; 1b reads from here)
        import io
        obj_bytes = agg_df.to_csv(index=False).encode('utf-8')
        s3_suffix = "_baseline/" if baseline else ""
        filename_agg = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        s3_key_agg = f"gold/feature_importance/{cohort}/{age_band}/{s3_suffix}{filename_agg}"
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key_agg,
                Body=io.BytesIO(obj_bytes),
                ContentType='text/csv'
            )
            print(f"✓ Uploaded aggregated feature importance to S3: s3://{S3_BUCKET}/{s3_key_agg}")
        except Exception as e:
            print(f"[WARN] Failed to upload to pgxdatalake: {e}")
            print(f"  File saved locally at: {agg_path}")
        # Historical copy in pgx-repository: flat layout pgx-analysis/3_feature_importance/outputs/{cohort}_{age_band}_aggregated_feature_importance.csv
        repo_key = f"{PGX_REPO_FI_PREFIX}/{filename_agg}"
        try:
            s3_client.put_object(
                Bucket=PGX_REPO_BUCKET,
                Key=repo_key,
                Body=io.BytesIO(obj_bytes),
                ContentType='text/csv'
            )
            print(f"✓ Uploaded to historical bucket: s3://{PGX_REPO_BUCKET}/{repo_key}")
        except Exception as e:
            print(f"[WARN] Failed to upload to pgx-repository (historical): {e}")

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
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="First-pass run: write to outputs/{cohort}/_baseline/. "
        "Default is no baseline: write to outputs/{cohort}/. "
        "Use --baseline only when generating baseline FI for the first time; baseline is usually already on S3.",
    )
    args = parser.parse_args()

    run_mc_feature_importance(
        cohort=args.cohort,
        age_band=args.age_band,
        n_runs=args.n_runs,
        force=args.force,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()

