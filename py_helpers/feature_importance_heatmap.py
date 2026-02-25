"""
Cross-age-band aggregated feature importance heatmap for Step 3a outputs.

Builds a heatmap (feature × age_band) from aggregated feature importance CSVs
written by 3a_feature_importance/run_mc_feature_importance.py.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Canonical age-band order for heatmap columns (0-12 through 85-114)
CANONICAL_AGE_BAND_ORDER = [
    "0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"
]

# Model identifiers for dashboard filter (aggregated = cross-model; others = per-model CSVs)
FI_MODEL_LABELS = ["aggregated", "catboost", "xgboost", "xgboost_rf"]

# Non-interactive backend when no display (e.g. CI / headless)
if not matplotlib.get_backend().startswith("module://"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass


def _resolve_project_root(outputs_base: Path) -> Optional[Path]:
    """Infer repo root from 3a_feature_importance/outputs when possible."""
    try:
        if outputs_base.name == "outputs" and outputs_base.parent.name == "3a_feature_importance":
            return outputs_base.parent.parent
    except Exception:
        pass
    return None


def get_aggregated_fi_heatmap_data(
    cohort: str,
    age_bands: List[str],
    outputs_base: Path,
    top_n: int = 50,
    importance_col: Optional[str] = None,
    max_rows: int = 80,
    project_root: Optional[Path] = None,
    filter_final: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build feature × age_band heatmap data from Step 3a aggregated CSVs.

    Returns a dict suitable for JSON and client-side Plotly: row_labels, column_labels, matrix, metric.
    Returns None if no data or < 2 age bands.
    When filter_final is True (default), drops admin/Z and target-leakage features (final FI only).
    """
    cohort_dir = outputs_base / cohort
    if not cohort_dir.exists():
        return None
    proj = project_root or _resolve_project_root(outputs_base)

    def _filter_fi(df: pd.DataFrame) -> pd.DataFrame:
        if proj and filter_final:
            try:
                from py_helpers.feature_importance_filters import filter_fi_df_final
                return filter_fi_df_final(df, proj, cohort=cohort)
            except ImportError:
                pass
        return df

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
        df = _filter_fi(df)
        if df.empty:
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

    column_order = [b for b in CANONICAL_AGE_BAND_ORDER if b in loaded_bands]
    combined = pd.concat(all_dfs, ignore_index=True)

    top_features_set = set()
    for _ab in loaded_bands:
        sub = combined[combined["age_band"] == _ab].nlargest(top_n, "importance")
        top_features_set.update(sub["feature"].tolist())
    top_features = list(top_features_set)
    if not top_features:
        return None

    pivot = combined.pivot_table(
        index="feature",
        columns="age_band",
        values="importance",
        aggfunc="first",
    ).reindex(index=top_features, columns=column_order).fillna(0.0)

    pivot["_mean"] = pivot[column_order].mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    pivot = pivot.reindex(columns=column_order)
    if len(pivot) > max_rows:
        pivot = pivot.iloc[:max_rows]

    row_labels = list(pivot.index.astype(str))
    column_labels = list(pivot.columns.astype(str))
    matrix = pivot.values.tolist()

    return {
        "cohort": cohort,
        "row_labels": row_labels,
        "column_labels": column_labels,
        "matrix": matrix,
        "metric": "importance",
    }


def _importance_col_for_df(df: pd.DataFrame, importance_col: Optional[str] = None) -> Optional[str]:
    """Return importance column name from aggregated or per-model FI DataFrame."""
    if importance_col and importance_col in df.columns:
        return importance_col
    for c in ("scaled_importance_mean", "importance_mean", "importance_scaled", "importance_normalized"):
        if c in df.columns:
            return c
    return None


def get_fi_heatmap_data_for_model(
    cohort: str,
    age_bands: List[str],
    outputs_base: Path,
    model: str,
    top_n: int = 50,
    importance_col: Optional[str] = None,
    max_rows: int = 80,
    project_root: Optional[Path] = None,
    filter_final: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build feature × age_band heatmap data for a specific model (or aggregated).

    For model=="aggregated" uses aggregated CSVs; otherwise uses
    {cohort}_{age_band_fname}_{model}_feature_importance_mc*.csv.
    Returns same shape as get_aggregated_fi_heatmap_data; None if < 2 age bands.
    When filter_final is True (default), drops admin/Z and target-leakage features.
    """
    cohort_dir = outputs_base / cohort
    if not cohort_dir.exists():
        return None
    proj = project_root or _resolve_project_root(outputs_base)

    def _filter_fi_model(df: pd.DataFrame) -> pd.DataFrame:
        if proj and filter_final:
            try:
                from py_helpers.feature_importance_filters import filter_fi_df_final
                return filter_fi_df_final(df, proj, cohort=cohort)
            except ImportError:
                pass
        return df

    all_dfs: List[pd.DataFrame] = []
    loaded_bands: List[str] = []
    for age_band in age_bands:
        age_band_fname = age_band.replace("-", "_")
        if model == "aggregated":
            csv_path = cohort_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        else:
            # Per-model: e.g. opioid_ed_65_74_xgboost_feature_importance_mc25.csv
            candidates = list(cohort_dir.glob(f"{cohort}_{age_band_fname}_{model}_feature_importance_mc*.csv"))
            csv_path = candidates[0] if candidates else None
        if not csv_path or not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "feature" not in df.columns:
            continue
        df = _filter_fi_model(df)
        if df.empty:
            continue
        col = _importance_col_for_df(df, importance_col)
        if not col:
            continue
        df = df[["feature", col]].copy()
        df["age_band"] = age_band
        df.rename(columns={col: "importance"}, inplace=True)
        all_dfs.append(df)
        loaded_bands.append(age_band)

    if len(loaded_bands) < 2:
        return None

    column_order = [b for b in CANONICAL_AGE_BAND_ORDER if b in loaded_bands]
    combined = pd.concat(all_dfs, ignore_index=True)
    top_features_set = set()
    for _ab in loaded_bands:
        sub = combined[combined["age_band"] == _ab].nlargest(top_n, "importance")
        top_features_set.update(sub["feature"].tolist())
    top_features = list(top_features_set)
    if not top_features:
        return None

    pivot = combined.pivot_table(
        index="feature",
        columns="age_band",
        values="importance",
        aggfunc="first",
    ).reindex(index=top_features, columns=column_order).fillna(0.0)
    pivot["_mean"] = pivot[column_order].mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    pivot = pivot.reindex(columns=column_order)
    if len(pivot) > max_rows:
        pivot = pivot.iloc[:max_rows]

    return {
        "cohort": cohort,
        "model": model,
        "row_labels": list(pivot.index.astype(str)),
        "column_labels": list(pivot.columns.astype(str)),
        "matrix": pivot.values.tolist(),
        "metric": "importance",
    }


def get_single_age_band_fi(
    cohort: str,
    age_band: str,
    outputs_base: Path,
    model: str,
    top_n: int = 100,
    importance_col: Optional[str] = None,
    project_root: Optional[Path] = None,
    filter_final: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load feature importance for one (cohort, age_band, model). Returns
    { cohort, model, age_band, features: [ { feature, importance } ] } or None.
    When filter_final is True (default), drops admin/Z and target-leakage features.
    """
    cohort_dir = outputs_base / cohort
    if not cohort_dir.exists():
        return None
    proj = project_root or _resolve_project_root(outputs_base)

    def _filter_fi_single(df: pd.DataFrame) -> pd.DataFrame:
        if proj and filter_final:
            try:
                from py_helpers.feature_importance_filters import filter_fi_df_final
                return filter_fi_df_final(df, proj, cohort=cohort)
            except ImportError:
                pass
        return df

    age_band_fname = age_band.replace("-", "_")
    if model == "aggregated":
        csv_path = cohort_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    else:
        candidates = list(cohort_dir.glob(f"{cohort}_{age_band_fname}_{model}_feature_importance_mc*.csv"))
        csv_path = candidates[0] if candidates else None
    if not csv_path or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns:
        return None
    df = _filter_fi_single(df)
    if df.empty:
        return None
    col = _importance_col_for_df(df, importance_col)
    if not col:
        return None
    df = df[["feature", col]].copy()
    df = df.nlargest(top_n, col)
    features = [{"feature": str(row["feature"]), "importance": float(row[col])} for _, row in df.iterrows()]
    return {
        "cohort": cohort,
        "model": model,
        "age_band": age_band,
        "features": features,
    }


def discover_fi_available(outputs_base: Path) -> Dict[str, Any]:
    """
    Scan 3a_feature_importance/outputs for available cohort/age_band/model combinations.
    Returns: { "cohorts": { cohort: { "age_bands": [...], "models": [...] } } }
    """
    result: Dict[str, Any] = {"cohorts": {}}
    for cohort_dir in outputs_base.iterdir():
        if not cohort_dir.is_dir() or cohort_dir.name.startswith("_"):
            continue
        cohort = cohort_dir.name
        if cohort in ("plots",):
            continue
        age_bands_set: set = set()
        models_set: set = set()
        for p in cohort_dir.iterdir():
            if p.is_dir():
                continue
            if not p.suffix == ".csv":
                continue
            stem = p.stem
            # aggregated: {cohort}_{age_band_fname}_aggregated_feature_importance
            if "_aggregated_feature_importance" in stem:
                models_set.add("aggregated")
                rest = stem.replace(f"{cohort}_", "").replace("_aggregated_feature_importance", "")
                if "_" in rest:
                    age_band = rest.replace("_", "-")
                    age_bands_set.add(age_band)
            # per-model: {cohort}_{age_band_fname}_{model}_feature_importance_mc*
            for m in ("catboost", "xgboost", "xgboost_rf"):
                if f"_{m}_feature_importance_mc" in stem:
                    models_set.add(m)
                    rest = stem.replace(f"{cohort}_", "").replace(f"_{m}_feature_importance_mc", "")
                    rest = re.sub(r"_\d+$", "", rest)  # drop mc run suffix e.g. _25
                    if "_" in rest:
                        age_band = rest.replace("_", "-")
                        age_bands_set.add(age_band)
                    break
        age_bands_sorted = sorted(age_bands_set, key=lambda b: (CANONICAL_AGE_BAND_ORDER.index(b) if b in CANONICAL_AGE_BAND_ORDER else 99))
        result["cohorts"][cohort] = {
            "age_bands": age_bands_sorted,
            "models": sorted(models_set, key=lambda m: (FI_MODEL_LABELS.index(m) if m in FI_MODEL_LABELS else 99)),
        }
    return result


def build_fi_dashboard_jsons(
    outputs_base: Path,
    top_n: int = 50,
    single_band_top_n: int = 100,
    importance_col: Optional[str] = None,
    max_rows: int = 80,
    project_root: Optional[Path] = None,
    filter_final: bool = True,
) -> Dict[str, Any]:
    """
    Convert all available 3a FI CSVs to dashboard JSONs: index + heatmaps (all age bands) + single-age feature lists.
    Uses final feature importances only (admin/Z and target-leakage removed) when filter_final is True.
    Writes:
      - outputs_base/feature_importance_index.json
      - outputs_base/{cohort}/plots/{cohort}_{model}_fi_heatmap.json (when 2+ age bands; aggregated uses existing name)
      - outputs_base/{cohort}/plots/{cohort}_{model}_{age_band_fname}_fi.json
    Returns the index dict and list of paths written.
    """
    index = discover_fi_available(outputs_base)
    written: List[Path] = []
    proj = project_root or _resolve_project_root(outputs_base)

    index_path = outputs_base / "feature_importance_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    written.append(index_path)

    for cohort, info in index.get("cohorts", {}).items():
        age_bands = info.get("age_bands") or []
        models = info.get("models") or []
        if not age_bands or not models:
            continue
        plots_dir = (outputs_base / cohort) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for model in models:
            if len(age_bands) >= 2:
                data = get_fi_heatmap_data_for_model(
                    cohort, age_bands, outputs_base, model,
                    top_n=top_n, importance_col=importance_col, max_rows=max_rows,
                    project_root=proj, filter_final=filter_final,
                )
                if data:
                    name = "aggregated_fi_heatmap" if model == "aggregated" else f"{model}_fi_heatmap"
                    json_name = f"{cohort}_{name}.json"
                    path = plots_dir / json_name
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    written.append(path)

            for age_band in age_bands:
                single = get_single_age_band_fi(
                    cohort, age_band, outputs_base, model,
                    top_n=single_band_top_n, importance_col=importance_col,
                    project_root=proj, filter_final=filter_final,
                )
                if single:
                    age_band_fname = age_band.replace("-", "_")
                    path = plots_dir / f"{cohort}_{model}_{age_band_fname}_fi.json"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(single, f, indent=2)
                    written.append(path)

    return {"index": index, "written": [str(p) for p in written]}


def write_aggregated_fi_heatmap_json(
    cohort: str,
    age_bands: List[str],
    outputs_base: Path,
    top_n: int = 50,
    importance_col: Optional[str] = None,
    max_rows: int = 80,
    project_root: Optional[Path] = None,
    filter_final: bool = True,
) -> Optional[Path]:
    """Build heatmap data (final FI: admin/leakage removed) and write to outputs_base/cohort/plots/{cohort}_aggregated_fi_heatmap.json."""
    data = get_aggregated_fi_heatmap_data(
        cohort, age_bands, outputs_base,
        top_n=top_n, importance_col=importance_col, max_rows=max_rows,
        project_root=project_root, filter_final=filter_final,
    )
    if not data:
        return None
    plots_dir = (outputs_base / cohort) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    json_path = plots_dir / f"{cohort}_aggregated_fi_heatmap.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return json_path


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
    Also writes outputs_base/cohort/plots/{cohort}_aggregated_fi_heatmap.json for dashboard.

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

    data = get_aggregated_fi_heatmap_data(
        cohort, age_bands, outputs_base,
        top_n=top_n, importance_col=importance_col,
        project_root=_resolve_project_root(outputs_base), filter_final=True,
    )
    if not data:
        return None

    row_labels = data["row_labels"]
    column_labels = data["column_labels"]
    pivot = pd.DataFrame(data["matrix"], index=row_labels, columns=column_labels)

    plots_dir = cohort_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    json_path = plots_dir / f"{cohort}_aggregated_fi_heatmap.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    heatmap_path = plots_dir / f"{cohort}_aggregated_fi_heatmap.png"

    fig, ax = plt.subplots(figsize=(max(8, len(column_labels) * 1.8), max(10, len(pivot) * 0.22)))
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
    project_root: Optional[Path] = None,
    filter_final: bool = True,
) -> Optional[Path]:
    """
    Create one final feature importance heatmap for both cohorts.
    When filter_final is True (default), uses only final FI (admin/Z and leakage removed).

    Combines all age bands per cohort by **summing** normalized weighted feature
    importance scores (scaled_importance_mean) across age bands for each cohort.
    Result: one heatmap with rows = top features (union), columns = cohort names,
    cell = sum of importance across age bands for that (feature, cohort).

    Args:
        outputs_base: Base directory for Step 3a outputs (e.g. 3a_feature_importance/outputs).
        cohorts: Dict cohort -> list of age_bands (e.g. REQUIRED_COHORTS; both opioid_ed and non_opioid_ed use full set).
        top_n: Number of top features to show (by max summed importance across cohorts).
        importance_col: Column name for importance (default: scaled_importance_mean, then importance_mean).

    Returns:
        Path to saved heatmap PNG, or None if no data.
    """
    proj = project_root or _resolve_project_root(outputs_base)

    def _filter_fi_combined(d: pd.DataFrame, c: str) -> pd.DataFrame:
        if proj and filter_final:
            try:
                from py_helpers.feature_importance_filters import filter_fi_df_final
                return filter_fi_df_final(d, proj, cohort=c)
            except ImportError:
                pass
        return d

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
            df = _filter_fi_combined(df, cohort)
            if df.empty:
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
