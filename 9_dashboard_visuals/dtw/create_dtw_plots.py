#!/usr/bin/env python3
"""
Create 3D (or 1D for polypharmacy) Plotly cluster plots of trajectories.

Same pattern as FP-Growth visuals: writes HTML (and optional PNG) to the DTW plots dir
so create_dtw_visuals can upload them to the dashboard bucket.

- For opioid_ed (and other multi-axis cohorts): 3D scatter — axes = counts of top 3 codes;
  points colored by KMeans cluster.
- For polypharmacy (non_opioid_ed): 1D — one axis = count of top code; points colored by cluster.

Code counts are derived from seq_pattern_str in the DTW features CSV (activity sequence per patient).
"""

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    PLOTLY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    SKLEARN_AVAILABLE = False


# Polypharmacy cohort: one axis only (1D plot)
POLYPHARMACY_COHORT = "non_opioid_ed"

# Tokens to exclude from code counts (missing/placeholder values in seq_pattern_str)
_SKIP_TOKENS = frozenset({"nan", "none", "null", ""})


def _ensure_plots_dir(plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)


def _code_counts_from_seq_pattern_str(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Build patient x code count matrix from DTW features with seq_pattern_str.

    Returns:
        count_df: index = mi_person_key, columns = code, values = count.
        target_series: target per mi_person_key if present, else None.
    """
    if "seq_pattern_str" not in df.columns:
        return pd.DataFrame(), None
    target_series = None
    if "target" in df.columns:
        target_series = df.set_index("mi_person_key")["target"].drop_duplicates()

    rows = []
    for _, row in df.iterrows():
        pid = row["mi_person_key"]
        seq = row.get("seq_pattern_str") or ""
        if not isinstance(seq, str):
            seq = str(seq)
        tokens = (s.strip() for s in seq.split("_") if s.strip())
        counts = Counter(s for s in tokens if s.lower() not in _SKIP_TOKENS)
        rows.append({"mi_person_key": pid, **counts})
    if not rows:
        return pd.DataFrame(), target_series
    count_df = pd.DataFrame(rows).set_index("mi_person_key").fillna(0).astype(int)
    return count_df, target_series


def _top_codes(count_df: pd.DataFrame, n: int) -> List[str]:
    """Top n codes by total count across patients."""
    if count_df.empty or n <= 0:
        return []
    totals = count_df.sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


def _cluster_points(
    count_df: pd.DataFrame,
    code_cols: List[str],
    n_clusters: int = 5,
) -> np.ndarray:
    """KMeans cluster labels (0 .. n_clusters-1)."""
    if not SKLEARN_AVAILABLE or not code_cols or count_df.empty:
        return np.zeros(len(count_df), dtype=int)
    X = count_df[code_cols].values
    n_clusters = min(n_clusters, max(1, len(count_df) - 1), max(1, X.shape[1]))
    if n_clusters < 2:
        return np.zeros(len(count_df), dtype=int)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(X)


def create_trajectory_cluster_plots(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    dtw_df: Optional[pd.DataFrame] = None,
    n_clusters: int = 5,
    force: bool = False,
) -> List[Path]:
    """
    Create 3D (or 1D for polypharmacy) Plotly trajectory cluster plots and write to plots dir.

    If dtw_df is None, loads DTW features from
    project_root/10_risk_dashboard/visualizations/dtw/outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv.

    Returns list of written paths (HTML, and PNG if kaleido available).
    """
    if not PLOTLY_AVAILABLE:
        print("[WARN] Plotly not available; skipping DTW trajectory cluster plots")
        return []
    if not SKLEARN_AVAILABLE:
        print("[WARN] sklearn not available; skipping DTW trajectory cluster plots")
        return []

    age_band_fname = age_band.replace("-", "_")
    plots_dir = (
        project_root
        / "10_risk_dashboard"
        / "visualizations"
        / "dtw"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "plots"
    )

    if dtw_df is None:
        fe_dir = project_root / "10_risk_dashboard" / "visualizations" / "dtw" / "outputs" / "feature_engineering"
        csv_path = fe_dir / f"dtw_features_{cohort_name}_{age_band_fname}.csv"
        if not csv_path.exists():
            print(f"[WARN] DTW features not found: {csv_path}; skipping cluster plots")
            return []
        dtw_df = pd.read_csv(csv_path)
        if "mi_person_key" in dtw_df.columns:
            dtw_df["mi_person_key"] = dtw_df["mi_person_key"].astype(str)

    if "seq_pattern_str" not in dtw_df.columns:
        print("[WARN] DTW features have no seq_pattern_str; skipping trajectory cluster plots")
        return []

    count_df, target_series = _code_counts_from_seq_pattern_str(dtw_df)
    if count_df.empty:
        print("[WARN] No code counts from seq_pattern_str; skipping cluster plots")
        return []

    is_polypharmacy = cohort_name == POLYPHARMACY_COHORT
    n_axes = 1 if is_polypharmacy else 3
    top_codes = _top_codes(count_df, n_axes)
    if len(top_codes) < n_axes:
        print(f"[WARN] Fewer than {n_axes} codes for axes; using {len(top_codes)}")
        if not top_codes:
            return []

    code_cols = top_codes[:n_axes]
    labels = _cluster_points(count_df, code_cols, n_clusters=n_clusters)
    count_df = count_df.copy()
    count_df["cluster"] = labels

    _ensure_plots_dir(plots_dir)
    written: List[Path] = []

    if is_polypharmacy:
        # 1D: x = count of top code, y = 0 (or small jitter for visibility)
        np.random.seed(42)
        x = count_df[code_cols[0]].values
        y = np.zeros(len(x)) + np.random.uniform(-0.1, 0.1, size=len(x))
        hover_list = []
        for i in count_df.index:
            t = "" if target_series is None or i not in target_series.index else f", target={target_series.loc[i]}"
            hover_list.append(f"mi_person_key={i}{t}, {code_cols[0]}={count_df.loc[i, code_cols[0]]}")
        fig = go.Figure()
        for c in sorted(count_df["cluster"].unique()):
            mask = count_df["cluster"] == c
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    name=f"Cluster {c}",
                    text=[hover_list[j] for j in np.where(mask)[0]],
                    hoverinfo="text",
                )
            )
        fig.update_layout(
            title=f"DTW trajectory clusters (polypharmacy) — {cohort_name} {age_band}<br>Axis: count of top code '{code_cols[0]}'",
            xaxis_title=f"Count of '{code_cols[0]}'",
            yaxis_title="",
            yaxis=dict(showticklabels=False, zeroline=True),
            height=500,
            showlegend=True,
        )
        fname = f"dtw_trajectory_cluster_1d_{cohort_name}_{age_band_fname}.html"
    else:
        # 3D: x, y, z = counts of top 3 codes
        x = count_df[code_cols[0]].values
        y = count_df[code_cols[1]].values
        z = count_df[code_cols[2]].values
        hover_list = []
        for i in count_df.index:
            t = "" if target_series is None or i not in target_series.index else f", target={target_series.loc[i]}"
            hover_list.append(
                f"mi_person_key={i}{t}<br>"
                f"{code_cols[0]}={count_df.loc[i, code_cols[0]]}, "
                f"{code_cols[1]}={count_df.loc[i, code_cols[1]]}, "
                f"{code_cols[2]}={count_df.loc[i, code_cols[2]]}"
            )
        fig = go.Figure()
        for c in sorted(count_df["cluster"].unique()):
            mask = count_df["cluster"] == c
            fig.add_trace(
                go.Scatter3d(
                    x=x[mask],
                    y=y[mask],
                    z=z[mask],
                    mode="markers",
                    name=f"Cluster {c}",
                    text=[hover_list[j] for j in np.where(mask)[0]],
                    hoverinfo="text",
                )
            )
        fig.update_layout(
            title=f"DTW trajectory clusters — {cohort_name} {age_band}<br>Axes: top 3 codes (counts)",
            scene=dict(
                xaxis_title=code_cols[0],
                yaxis_title=code_cols[1],
                zaxis_title=code_cols[2],
            ),
            height=700,
            showlegend=True,
        )
        fname = f"dtw_trajectory_cluster_3d_{cohort_name}_{age_band_fname}.html"

    out_html = plots_dir / fname
    if not force and out_html.exists():
        print(f"[INFO] Plot already exists: {out_html}; skipping (use --force to re-run)")
        return [out_html]

    fig.write_html(str(out_html), config={"responsive": True})
    written.append(out_html)
    print(f"[INFO] Wrote {out_html}")

    # Optional PNG (requires kaleido)
    try:
        out_png = plots_dir / fname.replace(".html", ".png")
        fig.write_image(str(out_png))
        written.append(out_png)
        print(f"[INFO] Wrote {out_png}")
    except Exception:
        pass

    return written


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Create DTW trajectory cluster plots (3D or 1D).")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent.parent.parent)
    parser.add_argument("--cohort-name", required=True)
    parser.add_argument("--age-band", required=True)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    create_trajectory_cluster_plots(
        project_root=args.project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        n_clusters=args.n_clusters,
        force=args.force,
    )


if __name__ == "__main__":
    main()
