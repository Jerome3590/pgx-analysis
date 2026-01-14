#!/usr/bin/env python3
"""
Create simple DTW feature visualizations for a given cohort / age_band.

This script is intentionally lightweight and works for both standard cohorts
and extreme-density cohorts (e.g., opioid_ed_extreme_density):

- Loads `dtw_added_features_{cohort}_{age_band_fname}.csv`
  (prefers `feature_engineering_outputs/6_dtw/...`, falls back to
  `5d_dtw_analysis/outputs/feature_engineering/...`).
- Produces:
  - Histogram of `dtw_min_distance`
  - Histogram of `trajectory_length`
  - Scatter plot of `trajectory_length` vs `dtw_min_distance`
- Saves plots under:
  - `feature_engineering_outputs/6_dtw/{cohort}/{age_band}/plots/`
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_dtw_features(cohort_name: str, age_band: str) -> tuple[pd.DataFrame, Path]:
    """Load dtw_added_features for a cohort/age_band and return (df, base_dir)."""
    age_band_fname = age_band.replace("-", "_")

    # Preferred location: mirrored DTW features under 5_feature_engineering/feature_engineering_outputs
    fe_root = (
        PROJECT_ROOT
        / "5_feature_engineering"
        / "feature_engineering_outputs"
        / "6_dtw"
        / cohort_name
        / age_band
    )
    dtw_added_path = fe_root / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"

    if not dtw_added_path.exists():
        # Fallback: local DTW outputs
        local_root = PROJECT_ROOT / "4b_dtw_filter" / "outputs" / "feature_engineering"
        dtw_added_path = local_root / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
        fe_root = local_root  # plots will be placed alongside features

    if not dtw_added_path.exists():
        raise FileNotFoundError(
            f"DTW added-features not found for {cohort_name} / {age_band} at {dtw_added_path}"
        )

    df = pd.read_csv(dtw_added_path)
    return df, fe_root


def create_dtw_plots(cohort_name: str, age_band: str) -> None:
    df, base_dir = _load_dtw_features(cohort_name, age_band)
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Ensure expected columns exist
    has_min = "dtw_min_distance" in df.columns
    has_len = "trajectory_length" in df.columns

    # Histogram: dtw_min_distance
    if has_min:
        series = df["dtw_min_distance"].replace([float("inf")], pd.NA).dropna()
        if not series.empty:
            plt.figure(figsize=(8, 6))
            plt.hist(series, bins=40, edgecolor="black")
            plt.xlabel("dtw_min_distance")
            plt.ylabel("Number of patients")
            plt.title(f"DTW minimum distance distribution\n{cohort_name} / {age_band}")
            plt.tight_layout()
            out_path = plots_dir / f"{cohort_name}_{age_band.replace('-', '_')}_dtw_min_distance_hist.png"
            plt.savefig(out_path, dpi=200)
            plt.close()

    # Histogram: trajectory_length
    if has_len:
        series = df["trajectory_length"].dropna()
        if not series.empty:
            plt.figure(figsize=(8, 6))
            plt.hist(series, bins=40, edgecolor="black")
            plt.xlabel("trajectory_length")
            plt.ylabel("Number of patients")
            plt.title(f"Trajectory length distribution\n{cohort_name} / {age_band}")
            plt.tight_layout()
            out_path = plots_dir / f"{cohort_name}_{age_band.replace('-', '_')}_trajectory_length_hist.png"
            plt.savefig(out_path, dpi=200)
            plt.close()

    # Scatter: trajectory_length vs dtw_min_distance
    if has_min and has_len:
        sub = df[[ "trajectory_length", "dtw_min_distance"]].replace(
            [float("inf")], pd.NA
        ).dropna()
        if not sub.empty:
            plt.figure(figsize=(8, 6))
            plt.scatter(
                sub["trajectory_length"],
                sub["dtw_min_distance"],
                alpha=0.3,
                s=10,
            )
            plt.xlabel("trajectory_length")
            plt.ylabel("dtw_min_distance")
            plt.title(
                f"Trajectory length vs DTW min distance\n{cohort_name} / {age_band}"
            )
            plt.tight_layout()
            out_path = plots_dir / f"{cohort_name}_{age_band.replace('-', '_')}_trajectory_vs_dtw_min.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create basic DTW feature plots for a cohort / age_band."
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed or opioid_ed_extreme_density)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 25-44)",
    )

    args = parser.parse_args()
    create_dtw_plots(cohort_name=args.cohort_name, age_band=args.age_band)


if __name__ == "__main__":
    main()

