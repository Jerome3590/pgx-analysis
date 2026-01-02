#!/usr/bin/env python3
"""
Combine SHAP (distributional) and FFA (structural) importance into a single
consensus table for a given (cohort, age_band).

Outputs:
  9_combined_shap_ffa/outputs/{cohort}/{age_band_fname}/
    - {cohort}_{age_band_fname}_combined_shap_ffa_importance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _paths_for(cohort: str, age_band_fname: str) -> Tuple[Path, Path, Path]:
    shap_path = (
        PROJECT_ROOT
        / "8_shap_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    )
    # Prefer the aggregated combined-weighted importance table from the
    # visualization step when available, but fall back to the core
    # AXP-based feature importance for the model type (xgboost) when the
    # combined file has not yet been generated.
    ffa_vis_path = (
        PROJECT_ROOT
        / "7_ffa_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / "visualizations"
        / "combined_weighted_feature_importance.csv"
    )
    ffa_core_path = (
        PROJECT_ROOT
        / "7_ffa_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / "xgboost"
        / "feature_importance_axp.csv"
    )
    ffa_path = ffa_vis_path if ffa_vis_path.exists() else ffa_core_path
    lookup_path = (
        PROJECT_ROOT
        / "feature_encoding_outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_feature_lookup.csv"
    )
    return shap_path, ffa_path, lookup_path


def run_combined(cohort: str, age_band: str) -> pd.DataFrame:
    age_band_fname = age_band.replace("-", "_")
    shap_path, ffa_path, lookup_path = _paths_for(cohort, age_band_fname)

    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP importance not found: {shap_path}")
    if not ffa_path.exists():
        raise FileNotFoundError(f"FFA combined importance not found: {ffa_path}")

    shap_df = pd.read_csv(shap_path)
    ffa_df = pd.read_csv(ffa_path)

    # Normalize scores and compute ranks
    shap_df = shap_df.copy()
    ffa_df = ffa_df.copy()

    shap_df["shap_rank"] = shap_df["mean_abs_shap"].rank(
        ascending=False, method="dense"
    )
    shap_max = shap_df["mean_abs_shap"].max() or 1.0
    shap_df["shap_norm"] = shap_df["mean_abs_shap"] / shap_max

    # Support both the combined-weighted FFA output (with weighted_importance)
    # and the core AXP feature importance (with importance) by aliasing to a
    # common column name for downstream normalization.
    if "weighted_importance" in ffa_df.columns:
        ffa_df["ffa_score"] = ffa_df["weighted_importance"]
    elif "importance" in ffa_df.columns:
        ffa_df["ffa_score"] = ffa_df["importance"]
    else:
        raise KeyError(
            "FFA importance file must contain either 'weighted_importance' or 'importance' column"
        )

    ffa_df["ffa_rank"] = ffa_df["ffa_score"].rank(
        ascending=False, method="dense"
    )
    ffa_max = ffa_df["ffa_score"].max() or 1.0
    ffa_df["ffa_norm"] = ffa_df["ffa_score"] / ffa_max

    # Merge on feature name
    merged = pd.merge(
        shap_df,
        ffa_df,
        on="feature",
        how="outer",
        suffixes=("_shap", "_ffa"),
    )

    # Combined score: simple average of normalized SHAP and FFA where both present,
    # otherwise use whichever is available.
    shap_norm = merged["shap_norm"].fillna(0.0)
    ffa_norm = merged["ffa_norm"].fillna(0.0)
    has_shap = merged["shap_norm"].notna()
    has_ffa = merged["ffa_norm"].notna()

    combined_score = np.where(
        has_shap & has_ffa,
        0.5 * shap_norm + 0.5 * ffa_norm,
        np.where(has_shap, shap_norm, ffa_norm),
    )
    merged["combined_score"] = combined_score
    merged["combined_rank"] = merged["combined_score"].rank(
        ascending=False, method="dense"
    )

    # Enrich with feature lookup metadata when available
    if lookup_path.exists():
        lookup_df = pd.read_csv(lookup_path)
        keep_cols = [
            "feature_name",
            "group",
            "description",
            "itemset_type",
            "itemset_items",
        ]
        keep_cols = [c for c in keep_cols if c in lookup_df.columns]
        merged = merged.merge(
            lookup_df[keep_cols],
            left_on="feature",
            right_on="feature_name",
            how="left",
        )

    # Reorder columns for readability
    cols_order = []
    for col in [
        "feature",
        "feature_name",
        "group",
        "description",
        "itemset_type",
        "itemset_items",
        "mean_abs_shap",
        "shap_rank",
        "shap_norm",
        "weighted_importance",
        "ffa_rank",
        "ffa_norm",
        "weighted_coverage",
        "model_count",
        "combined_score",
        "combined_rank",
    ]:
        if col in merged.columns and col not in cols_order:
            cols_order.append(col)

    other_cols = [c for c in merged.columns if c not in cols_order]
    merged = merged[cols_order + other_cols]

    out_dir = PROJECT_ROOT / "9_combined_shap_ffa" / "outputs" / cohort / age_band_fname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"{cohort}_{age_band_fname}_combined_shap_ffa_importance.csv"
    )
    merged.to_csv(out_path, index=False)
    print(f"Saved combined SHAP+FFA importance to {out_path}")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine SHAP and FFA importance into a consensus table."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    args = parser.parse_args()

    run_combined(args.cohort, args.age_band)


if __name__ == "__main__":
    main()

