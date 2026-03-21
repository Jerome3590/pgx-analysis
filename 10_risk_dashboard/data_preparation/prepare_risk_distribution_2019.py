#!/usr/bin/env python3
"""
Build the 2019 holdout risk distribution for the dashboard (idempotent).

Uses the same ensemble and feature schema as the Lambda risk API. For each
cohort/age_band: loads 2019 test features, runs ensemble predictions, bins
probabilities, and writes risk_distribution_2019.json next to feature_schema
and models. Also computes baseline_risk (actual 2019 outcome rate) so that when
the user enters no Drug, ICD, or CPT codes, the API returns this baseline;
as the user adds codes, risk increases to the model's classification probability.

Inputs (must exist from pipeline):
- 6_final_model/outputs/{cohort}/{age_band_fname}/inputs/model_test/final_features.parquet
- 10_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/feature_schema.json
- 10_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/*.joblib (or .json for catboost)

Outputs (idempotent overwrite):
- 10_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/risk_distribution_2019.json
  (bins, counts, n_patients, baseline_risk, description, bin_edges_pct)

Usage:
    python prepare_risk_distribution_2019.py --cohort opioid_ed
    python prepare_risk_distribution_2019.py --all
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from catboost import CatBoostClassifier
    import xgboost as xgb
    import joblib
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

# Cohort/age_band sets (match py_helpers.constants.REQUIRED_COHORTS)
_AGE_BANDS = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
REQUIRED_COHORTS = {"opioid_ed": _AGE_BANDS, "non_opioid_ed": _AGE_BANDS}

FINAL_MODEL_DIR = PROJECT_ROOT / "6_final_model" / "outputs"
MODELS_OUTPUT_DIR = PROJECT_ROOT / "10_risk_dashboard" / "outputs" / "models"

# 10 bins: [0,10), [10,20), ..., [90,100] (%); bin_centers for display
BIN_EDGES_PCT = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BIN_CENTERS_PCT = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]


def load_feature_schema(cohort: str, age_band: str) -> Optional[Dict[str, Any]]:
    """Load feature_schema.json from dashboard models output (same as Lambda)."""
    age_band_fname = age_band.replace("-", "_")
    path = MODELS_OUTPUT_DIR / cohort / age_band_fname / "feature_schema.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_model_from_dashboard(cohort: str, age_band: str, model_type: str) -> Optional[Any]:
    """Load model from 10_risk_dashboard/outputs/models (same artifacts Lambda uses)."""
    if not LIBS_AVAILABLE:
        return None
    age_band_fname = age_band.replace("-", "_")
    base = MODELS_OUTPUT_DIR / cohort / age_band_fname
    if model_type == "catboost":
        for name in ["catboost.joblib", "catboost.json"]:
            p = base / name
            if p.exists():
                model = CatBoostClassifier()
                model.load_model(str(p))
                return model
        return None
    if model_type in ("xgboost", "xgboost_rf"):
        p = base / f"{model_type}.joblib"
        if p.exists():
            return joblib.load(p)
    return None


def load_calibration_model(cohort: str, age_band: str, model_type: str) -> Optional[Any]:
    """Load Platt calibration model (LogisticRegression) if present; returns None otherwise."""
    if not LIBS_AVAILABLE:
        return None
    age_band_fname = age_band.replace("-", "_")
    p = MODELS_OUTPUT_DIR / cohort / age_band_fname / f"calibration_{model_type}.joblib"
    if p.exists():
        return joblib.load(p)
    return None


def build_2019_feature_matrix(
    test_df: pd.DataFrame,
    feature_schema: Dict[str, Any],
) -> np.ndarray:
    """
    Align 2019 test dataframe to schema feature order; fill missing with defaults.
    Returns (n_samples, n_features) float array.
    """
    features = feature_schema.get("features", [])
    defaults = feature_schema.get("defaults", {})
    if not features:
        return np.zeros((0, 0))

    rows = []
    for _, row in test_df.iterrows():
        vec = []
        for f in features:
            if f in test_df.columns and pd.notna(row.get(f)):
                val = row[f]
                vec.append(float(val))
            else:
                vec.append(float(defaults.get(f, 0.0)))
        rows.append(vec)
    return np.array(rows, dtype=np.float64)


def predict_ensemble_batch(
    X: np.ndarray,
    cohort: str,
    age_band: str,
    model_weights: Dict[str, float],
) -> np.ndarray:
    """
    Run ensemble prediction on rows X; same logic as Lambda predict_risk.
    Applies Platt calibration when calibration_*.joblib is present so the
    histogram distribution matches the calibrated scores returned by the Lambda.
    Returns 1D array of ensemble probabilities in [0, 1].
    """
    if not LIBS_AVAILABLE or X.size == 0:
        return np.array([])

    model_types = ["catboost", "xgboost", "xgboost_rf"]
    predictions: Dict[str, np.ndarray] = {}
    n_calibrated = 0
    for model_type in model_types:
        model = load_model_from_dashboard(cohort, age_band, model_type)
        if model is None:
            continue
        try:
            if model_type == "catboost":
                proba = model.predict_proba(X)[:, 1]
            elif model_type in ("xgboost", "xgboost_rf"):
                if isinstance(model, xgb.Booster):
                    dmat = xgb.DMatrix(X)
                    proba = model.predict(dmat)
                    proba = np.clip(proba, 0.0, 1.0)
                else:
                    proba = model.predict_proba(X)[:, 1]
            else:
                continue
            proba = np.asarray(proba, dtype=np.float64)
            calibrator = load_calibration_model(cohort, age_band, model_type)
            if calibrator is not None:
                proba = np.clip(
                    calibrator.predict_proba(proba.reshape(-1, 1))[:, 1], 0.0, 1.0
                )
                n_calibrated += 1
            predictions[model_type] = proba
        except Exception as e:
            print(f"    Warning: {model_type} prediction failed: {e}")
            continue

    if not predictions:
        return np.zeros(len(X))

    if n_calibrated > 0:
        print(f"    Platt calibration applied to {n_calibrated}/{len(predictions)} models")
    else:
        print(f"    No calibration models found — using raw probabilities (re-run training to generate calibration files)")

    # Weighted average (same as Lambda)
    total_weight = sum(model_weights.get(m, 0.0) for m in predictions)
    if total_weight <= 0:
        total_weight = len(predictions)
        weights = {m: 1.0 / total_weight for m in predictions}
    else:
        weights = {m: model_weights.get(m, 0.0) / total_weight for m in predictions}
    ensemble = np.zeros(len(X), dtype=np.float64)
    for m, w in weights.items():
        ensemble += w * predictions[m]
    return ensemble


def compute_distribution(ensemble_scores: np.ndarray) -> Tuple[List[float], List[int], int]:
    """
    Bin ensemble probabilities (0-1) into 10 bins; return bin centers (%), counts, n_patients.
    """
    n = len(ensemble_scores)
    if n == 0:
        return BIN_CENTERS_PCT, [0] * 10, 0
    # Bin edges in probability: 0, 0.1, ..., 1.0
    bin_edges = np.array([e / 100.0 for e in BIN_EDGES_PCT])
    counts, _ = np.histogram(ensemble_scores, bins=bin_edges)
    return BIN_CENTERS_PCT, counts.tolist(), int(n)


def build_distribution_for_cohort_age(
    cohort: str,
    age_band: str,
) -> bool:
    """Load 2019 test, run ensemble, bin, write risk_distribution_2019.json. Idempotent."""
    age_band_fname = age_band.replace("-", "_")
    test_path = (
        FINAL_MODEL_DIR / cohort / age_band_fname / "inputs" / "model_test" / "final_features.parquet"
    )
    if not test_path.exists():
        print(f"  Skip (no 2019 test data): {test_path}")
        return False

    feature_schema = load_feature_schema(cohort, age_band)
    if not feature_schema or not feature_schema.get("features"):
        print(f"  Skip (no feature_schema): {MODELS_OUTPUT_DIR / cohort / age_band_fname / 'feature_schema.json'}")
        return False

    test_df = pd.read_parquet(test_path)
    if len(test_df) == 0:
        print(f"  Skip (empty test): {test_path}")
        return False

    # Actual 2019 outcome rate = baseline risk when user enters no Drug/ICD/CPT codes
    baseline_risk: Optional[float] = None
    if "target" in test_df.columns:
        baseline_risk = float(test_df["target"].mean())
        print(f"  Baseline (2019 outcome rate): {baseline_risk:.4f}")

    X = build_2019_feature_matrix(test_df, feature_schema)
    if X.shape[0] == 0:
        print(f"  Skip (no rows after align): {cohort}/{age_band}")
        return False

    model_weights = feature_schema.get("model_weights", {
        "catboost": 1.0, "xgboost": 1.0, "xgboost_rf": 1.0
    })
    ensemble_scores = predict_ensemble_batch(X, cohort, age_band, model_weights)
    bins_pct, counts, n_patients = compute_distribution(ensemble_scores)

    # Risk band thresholds from 2019 distribution percentiles (low < 33rd <= medium < 67th <= high)
    risk_band_thresholds: Optional[Dict[str, float]] = None
    if len(ensemble_scores) > 0:
        p33, p67 = float(np.percentile(ensemble_scores, 33)), float(np.percentile(ensemble_scores, 67))
        risk_band_thresholds = {"low_medium": p33, "medium_high": p67}
        print(f"  Risk band thresholds (33rd/67th %ile): low_medium={p33:.4f}, medium_high={p67:.4f}")

    out_path = MODELS_OUTPUT_DIR / cohort / age_band_fname / "risk_distribution_2019.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "bins": bins_pct,
        "counts": counts,
        "n_patients": n_patients,
        "description": "2019 holdout predicted-probability distribution (ensemble)",
        "bin_edges_pct": BIN_EDGES_PCT,
    }
    if baseline_risk is not None:
        payload["baseline_risk"] = baseline_risk
    if risk_band_thresholds is not None:
        payload["risk_band_thresholds"] = risk_band_thresholds
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {n_patients} patients -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build 2019 holdout risk distribution for dashboard (idempotent)."
    )
    parser.add_argument("--cohort", choices=["opioid_ed", "non_opioid_ed"], help="Cohort to process")
    parser.add_argument("--all", action="store_true", help="Process all cohorts")
    args = parser.parse_args()

    if not LIBS_AVAILABLE:
        print("Warning: catboost/xgboost/joblib not available. Install to generate distributions.")
        return 0

    if args.all:
        cohorts: List[Tuple[str, List[str]]] = [
            ("opioid_ed", list(REQUIRED_COHORTS["opioid_ed"])),
            ("non_opioid_ed", list(REQUIRED_COHORTS["non_opioid_ed"])),
        ]
    elif args.cohort:
        cohorts = [(args.cohort, list(REQUIRED_COHORTS[args.cohort]))]
    else:
        parser.print_help()
        return 0

    tasks = [(c, ab) for c, age_bands in cohorts for ab in age_bands]
    n_workers = max(1, os.cpu_count() or 1)
    print(f"Building 2019 risk distributions (idempotent, {n_workers} workers)...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(build_distribution_for_cohort_age, c, ab): (c, ab)
            for c, ab in tasks
        }
        for future in as_completed(futures):
            c, ab = futures[future]
            try:
                ok = future.result()
                print(f"  {'Done' if ok else 'Skip'}: {c}/{ab}", flush=True)
            except Exception as e:
                print(f"  Error {c}/{ab}: {e}", flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
