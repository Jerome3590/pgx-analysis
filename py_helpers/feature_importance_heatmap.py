"""
Cross-age-band aggregated feature importance heatmap for Step 3a outputs.

Builds a heatmap (feature × age_band) from aggregated feature importance CSVs
written by 3a_feature_importance/run_mc_feature_importance.py.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Non-interactive backend when no display (e.g. CI / headless)
if not matplotlib.get_backend().startswith("module://"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass


def create_aggregated_fi_heatmap(
    cohort: str,
    age_bands: List[str],
    outputs_base: Path,
    top_n: int = 50,
    importance_col: Optional[str] = None,
) -> Optional[Path]:
    """
    Create cross-age-band feature importance heatmap from Step 3a aggregated CSVs.

    Loads {cohort}_{age_band_fname}_aggregated_feature_importance.csv for each
    age_band from outputs_base / cohort, builds feature × age_band matrix (union
    of top_n features across age bands), and saves a heatmap to
    outputs_base/cohort/plots/{cohort}_aggregated_fi_heatmap.png.

    Args:
        cohort: Cohort name (e.g. opioid_ed, non_opioid_ed).
        age_bands: List of age bands (e.g. ["13-24", "25-44"]).
        outputs_base: Base directory for Step 3a outputs (e.g. 3a_feature_importance/outputs).
        top_n: Number of top features per age band to include in union (default 50).
        importance_col: Column name for importance (default: first of scaled_importance_mean,
            importance_mean, importance_scaled, importance_normalized).

    Returns:
        Path to saved heatmap PNG, or None if no CSVs found / < 2 age bands.
    """
    cohort_dir = outputs_base / cohort
    if not cohort_dir.exists():
        return None

    # Load aggregated CSVs per age band
    all_dfs: List[pd.DataFrame] = []
    loaded_bands: List[str] = []
    for age_band in age_bands:
        age_band_fname = age_band.replace("-", "_")
        csv_path = cohort_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "feature" not in df.columns:
            continue
        col = importance_col
        if not col:
            for c in ("scaled_importance_mean", "importance_mean", "importance_scaled", "importance_normalized"):
                if c in df.columns:
                    col = c
                    break
        if not col or col not in df.columns:
            continue
        df = df[["feature", col]].copy()
        df["age_band"] = age_band
        df.rename(columns={col: "importance"}, inplace=True)
        all_dfs.append(df)
        loaded_bands.append(age_band)

    if len(loaded_bands) < 2:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    # Union of top N features per age band (unique, order by mean importance later)
    top_features_set = set()
    for _ab in loaded_bands:
        sub = combined[combined["age_band"] == _ab].nlargest(top_n, "importance")
        top_features_set.update(sub["feature"].tolist())
    top_features = list(top_features_set)
    if not top_features:
        return None

    # Pivot: feature × age_band, values = importance (fill 0 if missing)
    pivot = combined.pivot_table(
        index="feature",
        columns="age_band",
        values="importance",
        aggfunc="first",
    ).reindex(index=top_features, columns=loaded_bands).fillna(0.0)

    # Order features by mean importance across age bands
    pivot["_mean"] = pivot[loaded_bands].mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    pivot = pivot.reindex(columns=loaded_bands)

    # Cap number of rows for readable heatmap
    max_rows = 80
    if len(pivot) > max_rows:
        pivot = pivot.iloc[:max_rows]

    plots_dir = cohort_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = plots_dir / f"{cohort}_aggregated_fi_heatmap.png"

    fig, ax = plt.subplots(figsize=(max(8, len(loaded_bands) * 1.8), max(10, len(pivot) * 0.22)))
    sns.heatmap(
        pivot,
        annot=False,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Importance"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Aggregated Feature Importance by Age Band — {cohort}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Age Band")
    ax.set_ylabel("Feature")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(heatmap_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()

    return heatmap_path


def create_combined_cohorts_fi_heatmap(
    outputs_base: Path,
    cohorts: Dict[str, List[str]],
    top_n: int = 80,
    importance_col: Optional[str] = None,
) -> Optional[Path]:
    """
    Create one final feature importance heatmap for both cohorts.

    Combines all age bands per cohort by **summing** normalized weighted feature
    importance scores (scaled_importance_mean) across age bands for each cohort.
    Result: one heatmap with rows = top features (union), columns = cohort names,
    cell = sum of importance across age bands for that (feature, cohort).

    Args:
        outputs_base: Base directory for Step 3a outputs (e.g. 3a_feature_importance/outputs).
        cohorts: Dict cohort -> list of age_bands (e.g. {"opioid_ed": ["13-24", ...], "non_opioid_ed": ["65-74", ...]}).
        top_n: Number of top features to show (by max summed importance across cohorts).
        importance_col: Column name for importance (default: scaled_importance_mean, then importance_mean).

    Returns:
        Path to saved heatmap PNG, or None if no data.
    """
    # Collect (cohort, feature, importance) from each cohort/age_band CSV; sum importance per (cohort, feature)
    summed: Dict[str, Dict[str, float]] = {c: {} for c in cohorts}

    for cohort, age_bands in cohorts.items():
        cohort_dir = outputs_base / cohort
        if not cohort_dir.exists():
            continue
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            csv_path = cohort_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if "feature" not in df.columns:
                continue
            col = importance_col
            if not col:
                for c in ("scaled_importance_mean", "importance_mean", "importance_scaled", "importance_normalized"):
                    if c in df.columns:
                        col = c
                        break
            if not col or col not in df.columns:
                continue
            for _, row in df.iterrows():
                f = row["feature"]
                v = float(row[col]) if pd.notna(row[col]) else 0.0
                summed[cohort][f] = summed[cohort].get(f, 0.0) + v

    cohort_names = [c for c in cohorts if summed.get(c)]
    if not cohort_names:
        return None

    # Union of features that appear in any cohort, ordered by max summed importance across cohorts
    all_features: Dict[str, float] = {}
    for c in cohort_names:
        for f, v in summed[c].items():
            all_features[f] = max(all_features.get(f, 0.0), v)
    top_features = sorted(all_features.keys(), key=lambda x: -all_features[x])[:top_n]
    if not top_features:
        return None

    # Matrix: rows = features, columns = cohorts, values = summed normalized weighted importance
    pivot = pd.DataFrame(
        {c: [summed[c].get(f, 0.0) for f in top_features] for c in cohort_names},
        index=top_features,
    )

    plots_dir = outputs_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = plots_dir / "combined_cohorts_feature_importance_heatmap.png"

    fig, ax = plt.subplots(figsize=(max(6, len(cohort_names) * 3), max(10, len(pivot) * 0.2)))
    sns.heatmap(
        pivot,
        annot=False,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Sum of normalized weighted importance (across age bands)"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "Combined feature importance — both cohorts (sum of normalized weighted importance across age bands)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Feature")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(heatmap_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()

    return heatmap_path
