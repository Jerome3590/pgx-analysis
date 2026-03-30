#!/usr/bin/env python3
"""
Shared event-density (n_event) binning utilities.

Single source of truth for the low / medium / high / extreme bin scheme used
consistently across:

  - Model training  (6_final_model/run_final_model.py)  → n_event_bin feature + per-bin models
  - DTW trajectories (9_dashboard_visuals/dtw/create_dtw_trajectories.py) → event_density_bin
  - FP-Growth        (9_dashboard_visuals/fpgrowth/cohort_fpgrowth.py)    → Transaction_Density
  - BupaR / others   (any step that reads model_events.parquet)

Bin definition
--------------
Given a numeric series of per-patient event counts (or any utilization metric):

  low     : value ≤ P25
  medium  : P25 < value ≤ P50
  high    : P50 < value ≤ P95
  extreme : value > P95

Cut-points: P25, P50, P95 of the population being analysed.

Threshold persistence
---------------------
Thresholds are saved as JSON so that downstream steps (inference, visualisation)
can use exactly the same cuts as the model that was trained:

  {cohort}/{age_band_fname}/n_event_bin_thresholds.json
  (relative to 6_final_model/outputs/ or any caller-supplied cache_dir)

API
---
  DENSITY_BINS          – ordered tuple of bin labels
  compute_bin_thresholds(series)               → dict
  assign_n_event_bin(value, thresholds)        → str
  assign_n_event_bins(series, thresholds=None) → pd.Series[str]
  save_thresholds(thresholds, path)
  load_thresholds(path)                        → dict | None
  load_or_compute_thresholds(model_events_path, cache_path=None, cohort=None, age_band=None) → dict
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSITY_BINS: tuple = ("low", "medium", "high", "extreme")

_DEFAULT_THRESHOLDS: Dict[str, float] = {"p25": 5.0, "p50": 15.0, "p95": 50.0}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_bin_thresholds(series: pd.Series) -> Dict[str, Any]:
    """
    Compute P25 / P50 / P95 cut-points from a numeric series of event counts.

    Returns a dict:
      {"p25": float, "p50": float, "p95": float, "n": int, "min": float, "max": float}
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        warnings.warn("compute_bin_thresholds: empty series; using default thresholds.")
        return dict(_DEFAULT_THRESHOLDS)
    p25 = float(np.percentile(s, 25))
    p50 = float(np.percentile(s, 50))
    p95 = float(np.percentile(s, 95))
    return {
        "p25": p25,
        "p50": p50,
        "p95": p95,
        "n": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def assign_n_event_bin(value: float, thresholds: Dict[str, float]) -> str:
    """
    Assign a single numeric value to a density bin.

    Parameters
    ----------
    value      : numeric event count (or rate)
    thresholds : dict with keys 'p25', 'p50', 'p95'

    Returns
    -------
    One of: 'low', 'medium', 'high', 'extreme'
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "low"
    if np.isnan(v):
        return "low"
    p25 = thresholds.get("p25", _DEFAULT_THRESHOLDS["p25"])
    p50 = thresholds.get("p50", _DEFAULT_THRESHOLDS["p50"])
    p95 = thresholds.get("p95", _DEFAULT_THRESHOLDS["p95"])
    if v <= p25:
        return "low"
    if v <= p50:
        return "medium"
    if v <= p95:
        return "high"
    return "extreme"


def assign_n_event_bins(
    series: pd.Series,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Assign every element of a numeric Series to a density bin.

    If thresholds is None, they are computed from the series itself (dynamic,
    suitable for visualisation steps that run before model training).

    Returns a pd.Series of dtype str with values in DENSITY_BINS.
    """
    if thresholds is None:
        thresholds = compute_bin_thresholds(series)
    return series.apply(lambda v: assign_n_event_bin(v, thresholds))


# ---------------------------------------------------------------------------
# Threshold persistence
# ---------------------------------------------------------------------------

def save_thresholds(thresholds: Dict[str, Any], path: Path) -> None:
    """Serialize thresholds dict to JSON at *path* (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)


def load_thresholds(path: Path) -> Optional[Dict[str, Any]]:
    """Load thresholds JSON; returns None if file is missing or invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "p25" in data and "p50" in data and "p95" in data:
            return data
        warnings.warn(f"load_thresholds: file {path} missing p25/p50/p95 keys; ignoring.")
        return None
    except Exception as exc:
        warnings.warn(f"load_thresholds: could not read {path}: {exc}")
        return None


def load_or_compute_thresholds(
    model_events_path: Path,
    cache_path: Optional[Path] = None,
    cohort: Optional[str] = None,
    age_band: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return n_event bin thresholds for *model_events_path*.

    Priority:
      1. Load from *cache_path* if it exists and contains p25/p50/p95.
      2. Compute from model_events.parquet (COUNT(*) per mi_person_key).
      3. Fall back to default thresholds with a warning.

    Computed thresholds are saved to *cache_path* (if provided) for future
    steps to reuse, so the first caller (typically Step 6 model training)
    defines the canonical thresholds for that (cohort, age_band).

    Parameters
    ----------
    model_events_path : path to model_events.parquet
    cache_path        : where to read/write the JSON (e.g. 6_final_model/outputs/{cohort}/{ab}/n_event_bin_thresholds.json)
    cohort, age_band  : stored in the thresholds JSON for provenance
    """
    if cache_path is not None:
        cached = load_thresholds(cache_path)
        if cached is not None:
            return cached

    model_events_path = Path(model_events_path)
    if not model_events_path.exists():
        warnings.warn(
            f"load_or_compute_thresholds: model_events not found at {model_events_path}; "
            "using default thresholds."
        )
        return dict(_DEFAULT_THRESHOLDS)

    try:
        import duckdb  # type: ignore

        path_str = str(model_events_path).replace("\\", "/")
        con = duckdb.connect(":memory:")
        n_events_df = con.execute(
            f"""
            SELECT COUNT(*) AS n_events
            FROM read_parquet('{path_str}')
            GROUP BY mi_person_key
            """
        ).df()
        con.close()
        thresholds = compute_bin_thresholds(n_events_df["n_events"])
    except Exception as exc:
        warnings.warn(
            f"load_or_compute_thresholds: could not compute from parquet ({exc}); "
            "using default thresholds."
        )
        return dict(_DEFAULT_THRESHOLDS)

    if cohort:
        thresholds["cohort"] = cohort
    if age_band:
        thresholds["age_band"] = age_band

    if cache_path is not None:
        try:
            save_thresholds(thresholds, cache_path)
        except Exception as exc:
            warnings.warn(f"load_or_compute_thresholds: could not save to {cache_path}: {exc}")

    return thresholds


# ---------------------------------------------------------------------------
# Convenience: standard cache path relative to project root
# ---------------------------------------------------------------------------

def default_threshold_cache_path(project_root: Path, cohort: str, age_band: str) -> Path:
    """
    Canonical on-disk location for n_event_bin thresholds.

      <project_root>/6_final_model/outputs/<cohort>/<age_band_fname>/n_event_bin_thresholds.json

    This path is written by Step 6 model training and read by DTW / FP-Growth
    / BupaR / inference steps.
    """
    age_band_fname = age_band.replace("-", "_")
    return (
        Path(project_root)
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "n_event_bin_thresholds.json"
    )


# ---------------------------------------------------------------------------
# Step 6 per-bin vs cohort-level artifact detection (SHAP / FFA orchestration)
# ---------------------------------------------------------------------------


def _age_band_fname(age_band: str) -> str:
    return age_band.replace("-", "_")


def final_model_bin_base(
    project_root: Path, cohort: str, age_band: str, bin_name: str
) -> Path:
    """``.../6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/{bin_name}`` (project root only)."""
    abf = _age_band_fname(age_band)
    return (
        Path(project_root)
        / "6_final_model"
        / "outputs"
        / cohort
        / abf
        / "bin_models"
        / bin_name
    )


def resolve_step6_cohort_age_dir(project_root: Path, cohort: str, age_band: str) -> Path:
    """
    Single canonical ``.../6_final_model/outputs/{cohort}/{age_band_fname}/`` for loading models/features.

    Prefers the first candidate directory that exists and has ``models/`` or ``bin_models/``; otherwise
    returns the first candidate (typically project root) so callers can surface a clear FileNotFoundError.
    """
    for d in _step6_cohort_age_output_dirs(project_root, cohort, age_band):
        if not d.is_dir():
            continue
        if (d / "models").is_dir() or (d / "bin_models").is_dir():
            return d
    return _step6_cohort_age_output_dirs(project_root, cohort, age_band)[0]


def resolve_step6_train_features_csv(project_root: Path, cohort: str, age_band: str) -> Path:
    """Resolve ``..._train_final_features_no_leakage.csv`` under project or data root."""
    abf = _age_band_fname(age_band)
    fname = f"{cohort}_{abf}_train_final_features_no_leakage.csv"
    for d in _step6_cohort_age_output_dirs(project_root, cohort, age_band):
        p = d / fname
        if p.exists():
            return p
    return _step6_cohort_age_output_dirs(project_root, cohort, age_band)[0] / fname


def _step6_cohort_age_output_dirs(project_root: Path, cohort: str, age_band: str) -> list[Path]:
    """
    Step 6 output directory(ies) for (cohort, age_band): repo + data root if different.

    On EC2, training often writes under ``DATA_ROOT/6_final_model/outputs/`` while orchestration
    uses ``PROJECT_ROOT`` — both must be checked so SHAP/FFA see every trained bin.
    """
    abf = _age_band_fname(age_band)
    p1 = Path(project_root) / "6_final_model" / "outputs" / cohort / abf
    out: list[Path] = [p1]
    try:
        from py_helpers.env_utils import get_data_root

        p2 = Path(get_data_root()) / "6_final_model" / "outputs" / cohort / abf
        if p2.resolve() != p1.resolve():
            out.append(p2)
    except Exception:
        pass
    return out


def _bin_folder_has_trained_artifacts(bdir: Path, cohort: str, age_band: str) -> bool:
    """True if one ``bin_models/{bin}/`` tree has deployment or JSON markers."""
    if not bdir.is_dir():
        return False
    abf = _age_band_fname(age_band)
    stem = f"{cohort}_{abf}_best"
    models_dir = bdir / "models"
    if models_dir.is_dir():
        markers = (
            "xgboost_model.ubj",
            "xgboost.joblib",
            "catboost_model.cbm",
            "catboost.joblib",
        )
        if any((models_dir / m).exists() for m in markers):
            return True
    fj = bdir / "final_model_json"
    if fj.is_dir():
        if (fj / f"{stem}_xgboost_model.json").exists():
            return True
        if (fj / f"{stem}_catboost_model.cbm").exists() or (fj / f"{stem}_catboost_model.json").exists():
            return True
    return False


def final_model_bin_has_trained_artifacts(
    project_root: Path, cohort: str, age_band: str, bin_name: str
) -> bool:
    """
    True if Step 6 produced a usable model for this density bin (under ``bin_models/{bin}/``).

    Bins are omitted when ``train_and_evaluate`` never ran for that slice (e.g. too few patients).
    We treat presence of model files under ``models/`` or ``final_model_json/`` as success.

    Checks both ``PROJECT_ROOT/6_final_model/outputs/...`` and ``get_data_root()/6_final_model/outputs/...``
    when they differ (EC2 NVMe vs repo).
    """
    for cohort_dir in _step6_cohort_age_output_dirs(project_root, cohort, age_band):
        bdir = cohort_dir / "bin_models" / bin_name
        if _bin_folder_has_trained_artifacts(bdir, cohort, age_band):
            return True
    return False


def list_trained_density_bins(project_root: Path, cohort: str, age_band: str) -> list[str]:
    """Return bins that have Step 6 artifacts, in ``DENSITY_BINS`` order."""
    return [
        b
        for b in DENSITY_BINS
        if final_model_bin_has_trained_artifacts(project_root, cohort, age_band, b)
    ]


def _aggregate_cohort_age_dir_has_artifacts(root: Path, cohort: str, age_band: str) -> bool:
    """True if ``root`` is ``.../outputs/{cohort}/{abf}/`` (non-bin aggregate) with model files."""
    if not root.is_dir():
        return False
    abf = _age_band_fname(age_band)
    stem = f"{cohort}_{abf}_best"
    models_dir = root / "models"
    if models_dir.is_dir():
        markers = (
            "xgboost_model.ubj",
            "xgboost.joblib",
            "catboost_model.cbm",
            "catboost.joblib",
        )
        if any((models_dir / m).exists() for m in markers):
            return True
    fj = root / "final_model_json"
    if fj.is_dir():
        if (fj / f"{stem}_xgboost_model.json").exists():
            return True
        if (fj / f"{stem}_catboost_model.cbm").exists() or (fj / f"{stem}_catboost_model.json").exists():
            return True
    return False


def cohort_aggregate_final_model_has_artifacts(
    project_root: Path, cohort: str, age_band: str
) -> bool:
    """
    True if cohort-level (non-bin) Step 6 outputs exist under ``outputs/.../{cohort}/{ab}/models/``
    or ``final_model_json/`` (aggregate / legacy train mode).

    Checks project and ``get_data_root()`` when paths differ.
    """
    for cohort_dir in _step6_cohort_age_output_dirs(project_root, cohort, age_band):
        if _aggregate_cohort_age_dir_has_artifacts(cohort_dir, cohort, age_band):
            return True
    return False


def validate_per_bin_outputs(
    project_root: Path,
    cohort: str,
    age_band: str,
    bins: "tuple | None" = None,
    raise_on_missing: bool = True,
) -> "dict[str, bool]":
    """
    Validate that density-bin Step 6 artifacts exist exclusively under
    ``6_final_model/outputs/{cohort}/{age_band}/bin_models/{bin}/``
    (never ``model_outputs/``).

    Parameters
    ----------
    project_root      : repository root
    cohort, age_band  : target cohort/age band
    bins              : bins to check; None → all DENSITY_BINS
    raise_on_missing  : raise FileNotFoundError listing missing bins + fix
                        commands when True (default). Set False for a
                        non-fatal status report.

    Returns
    -------
    dict mapping each checked bin name → bool (True = artifacts found in outputs/)
    """
    if bins is None:
        bins = DENSITY_BINS
    abf = _age_band_fname(age_band)
    results: dict[str, bool] = {
        b: final_model_bin_has_trained_artifacts(project_root, cohort, age_band, b)
        for b in bins
    }
    missing = [b for b, ok in results.items() if not ok]
    found = [b for b, ok in results.items() if ok]

    print(f"\n=== Per-bin Step 6 output validation: {cohort} / {age_band} ===")
    print(f"  Canonical path: 6_final_model/outputs/{cohort}/{abf}/bin_models/<bin>/")
    for b in bins:
        mark = "OK     " if results[b] else "MISSING"
        bpath = (
            Path(project_root) / "6_final_model" / "outputs" / cohort / abf / "bin_models" / b
        )
        print(f"  [{mark}] {b:<8s}  {bpath}")
    print(f"  Summary — found: {found or '(none)'}  |  missing: {missing or '(none)'}")
    print()

    if missing and raise_on_missing:
        ab_h = age_band
        lines = [
            f"Missing Step 6 per-bin models for {cohort}/{age_band}: {missing}",
            f"Expected under: 6_final_model/outputs/{cohort}/{abf}/bin_models/<bin>/",
            "",
            "Re-train Step 6 per-bin models:",
            f"  python 6_final_model/run_final_model.py "
            f"--cohort {cohort} --age_band {ab_h} --train-mode per_bin",
            "",
            "Or sync all bin artifacts from S3:",
            f"  aws s3 sync s3://pgxdatalake/gold/final_model/{cohort}/{ab_h}/bin_models/ \\",
            f"         6_final_model/outputs/{cohort}/{abf}/bin_models/",
        ]
        raise FileNotFoundError("\n".join(lines))

    return results
