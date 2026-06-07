#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import REQUIRED_COHORTS, age_band_to_fname  # type: ignore
from py_helpers.event_density_utils import (  # type: ignore
    DENSITY_BINS,
    final_model_bin_has_trained_artifacts,
    resolve_step6_cohort_age_dir,
    resolve_step6_train_features_csv,
)

DEFAULT_FEATURES = [
    "pgx_num_drugs",
    "pgx_num_cpic_drugs",
    "n_events",
    "n_event_bin_ordinal",
    "event_span_days",
    "event_rate_per30",
    "early_event_rate_per30",
    "late_event_rate_per30",
    "event_rate_delta_per30",
    "event_rate_ratio_late_vs_early",
    "event_burstiness",
    "mean_inter_event_days",
    "median_inter_event_days",
    "recent30_event_count",
    "recent90_event_count",
    "recent30_event_fraction",
    "recent90_event_fraction",
]

UNIT_LABELS = {
    "event_span_days": "days",
    "event_rate_per30": "events per 30 days",
    "early_event_rate_per30": "events per 30 days",
    "late_event_rate_per30": "events per 30 days",
    "event_rate_delta_per30": "events per 30 days",
    "mean_inter_event_days": "days",
    "median_inter_event_days": "days",
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_. -]+", "_", str(name)).strip()


def _load_xgboost_model(cohort: str, age_band: str, bin_name: str | None):
    import xgboost as xgb  # type: ignore

    age_band_fname = age_band_to_fname(age_band)
    base = resolve_step6_cohort_age_dir(PROJECT_ROOT, cohort, age_band)
    if bin_name:
        base = base / "bin_models" / bin_name
    candidates = [
        base / "models" / "xgboost_model.ubj",
        base / "models" / "xgboost.joblib",
        base / "final_model_json" / f"{cohort}_{age_band_fname}_best_xgboost_model.ubj",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".joblib":
            return joblib.load(path), path
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model, path
    raise FileNotFoundError("No XGBoost model found. Checked:\n" + "\n".join(str(p) for p in candidates))


def _load_feature_rows(cohort: str, age_band: str, bin_name: str | None, max_rows: int, seed: int) -> tuple[pd.DataFrame, Path]:
    path = resolve_step6_train_features_csv(PROJECT_ROOT, cohort, age_band)
    if not path.exists():
        raise FileNotFoundError(f"Final feature CSV not found: {path}")
    df = pd.read_csv(path)
    if bin_name and "n_event_bin" in df.columns:
        df = df[df["n_event_bin"].astype(str) == str(bin_name)].copy()
    if df.empty:
        return pd.DataFrame(), path
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    X = df.drop(columns=["mi_person_key", "target"], errors="ignore")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0), path


def _model_feature_names(model, X: pd.DataFrame) -> list[str]:
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        if names:
            return list(names)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return list(X.columns)


def _compute_shap_contribs(model, X: pd.DataFrame) -> pd.DataFrame:
    import xgboost as xgb  # type: ignore

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    expected = booster.feature_names or list(X.columns)
    X = X.reindex(columns=expected, fill_value=0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype("float32")
    dmatrix = xgb.DMatrix(X, feature_names=expected)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=expected)


def _plot_numeric(feature: str, data: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    unit = UNIT_LABELS.get(feature)
    xlabel = f"{feature} Values ({unit})" if unit else f"{feature} Values"
    fig = plt.figure(figsize=(10, 8), dpi=150)
    gs = fig.add_gridspec(21, 1)
    ax = fig.add_subplot(gs[:20, 0])
    hist_ax = fig.add_subplot(gs[20:, 0], sharex=ax)
    ax.scatter(data["feature_value"], data["shap_value"], color="blue", alpha=0.6, s=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SHAP Values")
    ax.set_title(f"SHAP Partial Dependency Plot for {feature}")
    ax.grid(True, alpha=0.2)
    hist_ax.hist(data["feature_value"], bins=30, color="grey")
    hist_ax.set_yticks([])
    hist_ax.set_ylabel("")
    hist_ax.grid(False)
    hist_ax.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def create_dependence_plots_for_scope(
    cohort: str,
    age_band: str,
    bin_name: str | None,
    features: list[str],
    output_root: Path,
    max_rows: int,
    seed: int,
) -> dict:
    model, model_path = _load_xgboost_model(cohort, age_band, bin_name)
    X_raw, features_path = _load_feature_rows(cohort, age_band, bin_name, max_rows=max_rows, seed=seed)
    if X_raw.empty:
        return {"cohort": cohort, "age_band": age_band, "bin": bin_name, "status": "skipped_empty_bin"}

    expected = _model_feature_names(model, X_raw)
    X = X_raw.reindex(columns=expected, fill_value=0)
    shap_df = _compute_shap_contribs(model, X)

    age_band_fname = age_band_to_fname(age_band)
    out_dir = output_root / cohort / age_band_fname
    if bin_name:
        out_dir = out_dir / bin_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cohort": cohort,
        "age_band": age_band,
        "bin": bin_name,
        "model": "xgboost",
        "model_path": str(model_path),
        "features_path": str(features_path),
        "n_rows": int(len(X)),
        "plot_type": "numeric_shap_dependence",
        "style_reference": "survival_analysis/final_model/survival_analysis_final_catboost_model.qmd",
        "features": [],
    }

    for feature in features:
        if feature not in X.columns or feature not in shap_df.columns:
            continue
        data = pd.DataFrame({"feature_value": X[feature], "shap_value": shap_df[feature]})
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            continue
        safe_feature = _safe_name(feature)
        csv_path = out_dir / f"shap_dependence_numeric_{safe_feature}.csv"
        png_path = out_dir / f"shap_plot_numeric_{safe_feature}.png"
        data.to_csv(csv_path, index=False)
        _plot_numeric(feature, data, png_path)
        manifest["features"].append({"feature": feature, "csv": str(csv_path), "png": str(png_path)})

    manifest_path = out_dir / "shap_dependence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def iter_scope(cohorts: list[str] | None, age_bands: list[str] | None):
    for cohort, bands in REQUIRED_COHORTS.items():
        if cohorts and cohort not in cohorts:
            continue
        for age_band in bands:
            if age_bands and age_band not in age_bands:
                continue
            yield cohort, age_band


def main() -> int:
    parser = argparse.ArgumentParser(description="Create numeric manuscript SHAP dependence plots from existing Step 6 models without retraining.")
    parser.add_argument("--cohort", action="append", choices=sorted(REQUIRED_COHORTS), help="Cohort to process. Repeatable. Defaults to all.")
    parser.add_argument("--age-band", action="append", help="Age band to process. Repeatable. Defaults to all.")
    parser.add_argument("--bin", action="append", choices=DENSITY_BINS, help="Density bin to process. Repeatable. Defaults to trained bins when present.")
    parser.add_argument("--feature", action="append", help="Numeric feature to plot. Repeatable. Defaults to grouping/interpretation candidate features.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "manuscript" / "final_model" / "images" / "dependence_plots_test")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = args.feature or DEFAULT_FEATURES
    manifests = []
    for cohort, age_band in iter_scope(args.cohort, args.age_band):
        bins = args.bin or [b for b in DENSITY_BINS if final_model_bin_has_trained_artifacts(PROJECT_ROOT, cohort, age_band, b)]
        if not bins:
            bins = [None]
        for bin_name in bins:
            try:
                print(f"→ SHAP dependence: {cohort} / {age_band}{f' / {bin_name}' if bin_name else ''}")
                manifest = create_dependence_plots_for_scope(
                    cohort=cohort,
                    age_band=age_band,
                    bin_name=bin_name,
                    features=features,
                    output_root=args.output_root,
                    max_rows=args.max_rows,
                    seed=args.seed,
                )
                manifests.append(manifest)
                print(f"  features plotted: {len(manifest.get('features', []))}")
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "shap_dependence_run_manifest.json").write_text(json.dumps(manifests, indent=2))
    print(f"Wrote manuscript SHAP dependence outputs to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
