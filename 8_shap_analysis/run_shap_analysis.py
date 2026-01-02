#!/usr/bin/env python3
"""
Run SHAP analysis for final models for a given (cohort, age_band).

Outputs:
  8_shap_analysis/outputs/{cohort}/{age_band_fname}/
    - {cohort}_{age_band_fname}_shap_global_importance_xgboost.csv
    - {cohort}_{age_band_fname}_shap_global_importance_catboost.csv
    - {cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet
    - {cohort}_{age_band_fname}_shap_sample_values_catboost.parquet
    - summary bar / beeswarm plots (PNG) per model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # type: ignore


def _load_final_features(cohort: str, age_band: str) -> Tuple[pd.DataFrame, pd.Series]:
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

    df = pd.read_csv(features_path)
    if "target" not in df.columns:
        raise ValueError(f"'target' column not found in {features_path}")

    y = df["target"].astype(int)
    X = df.drop(columns=["mi_person_key", "target"], errors="ignore")

    # Keep numeric columns only (model is trained on numeric features)
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols].copy()
    return X, y


from py_helpers.env_utils import get_xgb_cpu_nthread  # noqa: E402


def _fit_models_for_shap(X: pd.DataFrame, y: pd.Series, random_seed: int = 42):
    """
    Fit XGBoost and CatBoost models with the same hyperparameters used in
    6b_final_model_selection/run_final_model.py for use in SHAP analysis.

    We refit here to avoid depending on serialized binaries; this is acceptable
    because we are computing SHAP values on the same data distribution used in
    final training.
    """
    import xgboost as xgb  # type: ignore

    nthread = get_xgb_cpu_nthread()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=nthread,
        random_state=random_seed,
    )
    try:
        xgb_clf.fit(X, y)
    except Exception:
        xgb_clf.set_params(tree_method="hist")
        if "device" in xgb_clf.get_params():
            xgb_clf.set_params(device="cpu")
        xgb_clf.fit(X, y)

    try:
        from catboost import CatBoostClassifier  # type: ignore

        cb_clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            grow_policy="SymmetricTree",
            random_seed=random_seed,
            verbose=False,
        )
        cb_clf.fit(X, y)
    except Exception:
        cb_clf = None

    return xgb_clf, cb_clf


def run_shap_analysis(
    cohort: str,
    age_band: str,
    n_background: int = 1000,
    n_eval: int = 2000,
) -> None:
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
        PROJECT_ROOT / "8_shap_analysis" / "outputs" / cohort / age_band_fname
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading final features for {cohort}, {age_band}...")
    X, y = _load_final_features(cohort, age_band)
    print(f"Final feature matrix: {X.shape[0]} rows, {X.shape[1]} features.")

    # Sample background and evaluation sets for SHAP efficiency
    rng = np.random.default_rng(42)
    idx_all = np.arange(X.shape[0])
    rng.shuffle(idx_all)

    bg_idx = idx_all[: min(n_background, len(idx_all))]
    eval_idx = idx_all[: min(n_eval, len(idx_all))]
    X_bg = X.iloc[bg_idx]
    X_eval = X.iloc[eval_idx]

    print("Fitting models for SHAP...")
    xgb_clf, cb_clf = _fit_models_for_shap(X, y)

    feature_names = list(X.columns)

    # ------------------- XGBoost SHAP -------------------
    print("Computing SHAP values for XGBoost...")
    expl_xgb = shap.TreeExplainer(
        xgb_clf, data=X_bg, feature_perturbation="interventional"
    )
    shap_xgb = expl_xgb.shap_values(X_eval)

    # Global importance
    mean_abs_xgb = np.abs(shap_xgb).mean(axis=0)
    xgb_imp_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_xgb}
    ).sort_values("mean_abs_shap", ascending=False)
    xgb_imp_path = (
        out_dir
        / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    )
    xgb_imp_df.to_csv(xgb_imp_path, index=False)
    print(f"Saved XGBoost SHAP global importance to {xgb_imp_path}")

    # Sample SHAP values
    xgb_shap_sample_path = (
        out_dir
        / f"{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
    )
    shap_sample_df = pd.DataFrame(
        shap_xgb, columns=feature_names, index=X_eval.index
    )
    shap_sample_df.to_parquet(xgb_shap_sample_path, index=True)
    print(f"Saved XGBoost SHAP sample values to {xgb_shap_sample_path}")

    # Summary plots
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_xgb,
        X_eval,
        feature_names=feature_names,
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
        shap_xgb,
        X_eval,
        feature_names=feature_names,
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

    print(f"Saved XGBoost SHAP summary plots to {out_dir}")

    # ------------------- CatBoost SHAP -------------------
    if cb_clf is not None:
        try:
            print("Computing SHAP values for CatBoost...")
            from catboost import Pool  # type: ignore

            pool_eval = Pool(X_eval, y.iloc[eval_idx])
            shap_cb = cb_clf.get_feature_importance(
                type="ShapValues", data=pool_eval
            )
            shap_cb = np.array(shap_cb)

            # CatBoost returns:
            # - binary/Regression: (n_samples, n_features + 1) [last col = expected value]
            # - multiclass: (n_samples, n_classes, n_features + 1)
            # We want per-feature SHAP values with the expected value stripped.
            if shap_cb.ndim == 2:
                # (n_samples, n_features + 1)
                shap_cb_feat = shap_cb[:, :-1]  # drop expected value column
            elif shap_cb.ndim == 3:
                # (n_samples, n_classes, n_features + 1) → collapse classes
                shap_cb_feat = shap_cb[:, :, :-1].mean(axis=1)
            else:
                raise ValueError(
                    f"Unexpected CatBoost SHAP array shape: {shap_cb.shape}"
                )

            shap_cb_mean_abs = np.abs(shap_cb_feat).mean(axis=0).ravel()

            cb_imp_df = pd.DataFrame(
                {"feature": feature_names, "mean_abs_shap": shap_cb_mean_abs}
            ).sort_values("mean_abs_shap", ascending=False)
            cb_imp_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
            )
            cb_imp_df.to_csv(cb_imp_path, index=False)
            print(f"Saved CatBoost SHAP global importance to {cb_imp_path}")

            cb_shap_sample_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet"
            )
            shap_cb_sample_df = pd.DataFrame(
                shap_cb_feat,
                index=X_eval.index,
                columns=feature_names,
            )
            shap_cb_sample_df.to_parquet(cb_shap_sample_path, index=True)
            print(f"Saved CatBoost SHAP sample values to {cb_shap_sample_path}")

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_cb_feat,
                X_eval,
                feature_names=feature_names,
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
                shap_cb_feat,
                X_eval,
                feature_names=feature_names,
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

            print(f"Saved CatBoost SHAP summary plots to {out_dir}")
        except Exception as e:
            print(f"CatBoost SHAP analysis failed; skipping. {e}")


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

    run_shap_analysis(
        cohort=args.cohort,
        age_band=args.age_band,
        n_background=args.n_background,
        n_eval=args.n_eval,
    )


if __name__ == "__main__":
    main()

