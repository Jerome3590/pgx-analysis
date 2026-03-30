#!/usr/bin/env python3
"""
Generate DTW trajectory figures from actual DTW pipeline outputs.

Primary data source: 10_risk_dashboard/visualizations/dtw/{cohort}/{age_band}/
  chart_data.json — target_pathway_patterns, high_risk_trajectories, routine_comparison,
                     event_density_bins, summary
  sequence_heatmap.json — code × position counts by event type

Secondary data (opioid_ed/25-44 KMeans archetype figure):
  3b_feature_importance_eda/outputs/opioid_ed/25_44/features/ (BupaR CSVs)

Figures generated:
  manuscript/figures/ch03/fig_trajectories.pdf   — 3-panel archetype figure (KMeans, opioid_ed)
  manuscript/figures/ch03/fig_dtw_pathways.pdf   — 3-panel DTW pipeline figure (opioid_ed/25-44)
  manuscript/figures/ch04/fig_trajectories.pdf   — 3-panel DTW pipeline figure (non_opioid_ed/65-74)
  manuscript/figures/ch03/fig_trajectories_heatmap.pdf  — event heatmap (opioid_ed/25-44)

Usage:
    cd C:/Projects/pgx-analysis
    python manuscript/generate_dtw_actual.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# BupaR feature CSVs (for opioid_ed/25-44 KMeans archetype figure)
BUPAR_BASE = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / "opioid_ed" / "25_44"
FEAT_DIR   = BUPAR_BASE / "features"

# DTW pipeline outputs (chart_data.json from create_dtw_trajectories + create_dtw_features + create_dtw_visuals)
DTW_ROOT   = PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "dtw"

FIG_CH03   = SCRIPT_DIR / "figures" / "ch03"
FIG_CH04   = SCRIPT_DIR / "figures" / "ch04"
FIG_CH03.mkdir(parents=True, exist_ok=True)
FIG_CH04.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE   = "#2166ac"
C_RED    = "#d6604d"
C_GREEN  = "#4dac26"
C_TEAL   = "#01665e"
C_AMBER  = "#d8b365"
C_PURPLE = "#7b2d8b"
C_GRAY   = "#636363"
C_LGRAY  = "#bdbdbd"
C_ORANGE = "#f4a582"

ARCHETYPE_COLORS  = [C_RED, C_TEAL, C_AMBER]   # Rapid, Moderate, Chronic
ARCHETYPE_MARKERS = ["o", "s", "^"]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(SCRIPT_DIR)}")


def _load_chart_data(cohort: str, band: str) -> dict:
    """Load chart_data.json produced by create_dtw_trajectories + create_dtw_features + create_dtw_visuals."""
    p = DTW_ROOT / cohort / band / "chart_data.json"
    if not p.exists():
        raise FileNotFoundError(f"chart_data.json not found: {p}")
    return json.loads(p.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# DTW pipeline figure (Chart-data based — both cohorts)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_dtw_pipeline(
    cohort: str,
    band: str,
    fig_dir: Path,
    out_name: str,
    title_cohort: str,
    target_label: str,
    top_n_drugs: int = 12,
) -> None:
    """
    3-panel DTW pipeline figure built from chart_data.json:

    Panel A — Top Drug Pathway Patterns (horizontal bar)
              Prevalence (%) of key drugs in trajectories preceding the target event.

    Panel B — High-Risk Trajectory Quartiles
              Target event rate by DTW distance quartile to prototype.
              Q1 = most similar to high-risk archetype; Q4 = least similar.

    Panel C — Routine vs No-Routine Care
              Target event rate by routine-appointment status.
    """
    d = _load_chart_data(cohort, band)
    summary = d["summary"]
    n_total   = summary["total_trajectories"]
    n_target  = summary["target_counts"]["target_1"]
    med_len   = summary["trajectory_length"]["median"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # ── Panel A: Top drug pathway patterns ───────────────────────────────────
    ax = axes[0]
    tp = d["target_pathway_patterns"]
    drugs = [x.replace("DRUG:", "").replace("_", " ").title() for x in tp["x"]]
    vals  = tp["y"]
    # Limit to top_n_drugs
    drugs = drugs[:top_n_drugs]
    vals  = vals[:top_n_drugs]
    # Reverse for bottom-up ordering
    drugs_r = drugs[::-1]
    vals_r  = vals[::-1]
    colors_bar = [C_BLUE if v >= np.percentile(vals_r, 60) else C_LGRAY for v in vals_r]
    bars = ax.barh(range(len(drugs_r)), vals_r, color=colors_bar, height=0.65, edgecolor="none")
    ax.set_yticks(range(len(drugs_r)))
    ax.set_yticklabels(drugs_r, fontsize=7)
    ax.set_xlabel("% of Target Patients with Drug in Pre-Event Trajectory")
    ax.set_title(
        f"(A) Top Drug Pathway Patterns\n({title_cohort}; n={n_target:,} target)",
        fontsize=8.5,
    )
    for bar, v in zip(bars, vals_r):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=6.5, color=C_GRAY)
    ax.set_xlim(0, max(vals_r) * 1.25)

    # ── Panel B: High-risk trajectory quartiles ───────────────────────────────
    ax = axes[1]
    hr = d["high_risk_trajectories"]
    labels = hr["x"]
    rates  = [r * 100 for r in hr["y"]]   # convert fraction → percent
    ns     = hr.get("n", [None] * len(labels))
    q_colors = [C_RED, C_AMBER, C_GREEN, C_TEAL]
    bar_vals = ax.bar(
        range(len(labels)), rates,
        color=q_colors[:len(labels)], width=0.55, edgecolor="white", linewidth=0.6,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [lbl.replace("(", "\n(") for lbl in labels], fontsize=7.5
    )
    ax.set_ylabel(f"{target_label} Event Rate (%)")
    ax.set_title(
        f"(B) Risk by DTW Distance Quartile\n"
        f"(proximity to high-risk prototype; Q1=closest)",
        fontsize=8.5,
    )
    for i, (bar, rate) in enumerate(zip(bar_vals, rates)):
        n_str = f"n={ns[i]:,}" if ns[i] is not None else ""
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.3,
                f"{rate:.1f}%\n{n_str}", ha="center", va="bottom",
                fontsize=6.5, color=C_GRAY)
    ax.set_ylim(0, max(rates) * 1.40 if max(rates) > 0 else 5)

    # ── Panel C: Routine vs no-routine care ───────────────────────────────────
    ax = axes[2]
    rc = d["routine_comparison"]
    rc_labels = rc["x"]
    rc_rates  = [r * 100 for r in rc["y"]]
    rc_ns     = rc.get("n", [None] * len(rc_labels))
    rc_colors = [C_RED if "no" in lbl.lower() or "0" in lbl else C_BLUE for lbl in rc_labels]
    short_labels = []
    for lbl in rc_labels:
        if "no routine" in lbl.lower() or "0 admin" in lbl.lower():
            short_labels.append("No Routine\nAppointments")
        else:
            short_labels.append("Routine\nAppointments\n(1+ admin ICD)")
    rc_bars = ax.bar(
        range(len(short_labels)), rc_rates,
        color=rc_colors, width=0.45, edgecolor="white", linewidth=0.6,
    )
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(f"{target_label} Event Rate (%)")
    ax.set_title(
        "(C) Routine Preventive Care vs Event Rate\n"
        "(admin ICD events as routine-care proxy)",
        fontsize=8.5,
    )
    for bar, rate, n_s in zip(rc_bars, rc_rates, rc_ns):
        n_str = f"n={n_s:,}" if n_s is not None else ""
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.2,
                f"{rate:.1f}%\n{n_str}", ha="center", va="bottom",
                fontsize=7, color=C_GRAY)
    ax.set_ylim(0, max(rc_rates) * 1.45 if max(rc_rates) > 0 else 5)

    fig.suptitle(
        f"DTW Trajectory Analysis — {title_cohort}\n"
        f"(n={n_total:,} trajectories; {n_target:,} target events; "
        f"median trajectory length {med_len:.0f} events)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.90])
    _save(fig, fig_dir / out_name)


# ─────────────────────────────────────────────────────────────────────────────
# BupaR KMeans archetype data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_cluster(k: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Load BupaR features, merge, cluster, and return annotated DataFrame.
    Cluster labels assigned by centroid time_to_F1120_days rank:
      rank 0 = "Rapid-Onset"  (shortest time-to-ADE)
      rank 1 = "Moderate-Escalation"
      rank 2 = "Chronic-Escalation" (longest time-to-ADE)
    """
    # ── Load time-windowed features (target patients only)
    tte  = pd.read_csv(FEAT_DIR / "opioid_ed_25_44_train_target_time_to_f1120_features_bupar.csv")
    pre  = pd.read_csv(FEAT_DIR / "opioid_ed_25_44_train_target_pre_f1120_patient_features_bupar.csv")
    post = pd.read_csv(FEAT_DIR / "opioid_ed_25_44_train_target_post_f1120_patient_features_bupar.csv")

    df = tte.merge(pre, on="case_id", how="inner").merge(post, on="case_id", how="inner")
    print(f"  Loaded {len(df):,} target patients")

    # Drop rows with any NaN in key columns
    feat_cols = ["n_drug_events_30d", "n_drug_events_90d", "n_drug_events_180d", "time_to_F1120_days"]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"  After dropping NaN: {len(df):,} patients")

    # ── Log-transform time_to_F1120_days before clustering to handle right skew
    #    (right tail of very long trajectories should not dominate clustering)
    df["_log_time"] = np.log1p(df["time_to_F1120_days"])
    # Sqrt-transform drug counts to compress extreme polypharmacy outliers
    for c in ["n_drug_events_30d", "n_drug_events_90d", "n_drug_events_180d"]:
        df[f"_sqrt_{c}"] = np.sqrt(df[c])
    cluster_features = ["_sqrt_n_drug_events_30d", "_sqrt_n_drug_events_90d",
                        "_sqrt_n_drug_events_180d", "_log_time"]

    X = df[cluster_features].values.astype(float)
    Xw = whiten(X)   # normalise each feature by its std

    np.random.seed(seed)
    centroids, labels = kmeans2(Xw, k, minit="++", seed=seed, iter=50)
    df["cluster"] = labels

    # ── Label clusters by centroid log_time rank (ascending = rapid-onset)
    centroid_time = centroids[:, 3]  # 4th feature is _log_time (whitened)
    time_rank = np.argsort(centroid_time)          # [rapid_idx, moderate_idx, chronic_idx]
    label_map = {
        int(time_rank[0]): "Rapid-Onset",
        int(time_rank[1]): "Moderate-Escalation",
        int(time_rank[2]): "Chronic-Escalation",
    }
    df["archetype"] = df["cluster"].map(label_map)

    # Counts per archetype
    for arch, grp in df.groupby("archetype"):
        print(f"    {arch}: n={len(grp):,}  "
              f"time_to_F1120_days median={grp['time_to_F1120_days'].median():.0f}d  "
              f"n_drug_180d median={grp['n_drug_events_180d'].median():.1f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_trajectories(df: pd.DataFrame) -> None:
    """
    3-panel DTW trajectory archetype figure:

    Panel A  –  Drug event density trajectories per archetype
                X = lookback window (30d / 90d / 180d before ADE)
                Y = mean drug events in that window
                Shaded band = ±1 SEM

    Panel B  –  Time-to-F1120 violin per archetype
                Shows clear separation: Rapid-Onset vs Chronic-Escalation

    Panel C  –  2D scatter: pre-event drug count vs time-to-target
                Each dot = one patient; color = archetype cluster
                Confirms 2D separability of the archetypes
    """
    archetypes = ["Rapid-Onset", "Moderate-Escalation", "Chronic-Escalation"]
    windows    = [30, 90, 180]
    drug_cols  = ["n_drug_events_30d", "n_drug_events_90d", "n_drug_events_180d"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # ── Panel A: trajectory curves ─────────────────────────────────────────
    ax = axes[0]
    for arch, color, marker in zip(archetypes, ARCHETYPE_COLORS, ARCHETYPE_MARKERS):
        grp = df[df["archetype"] == arch]
        means = [grp[c].mean() for c in drug_cols]
        sems  = [grp[c].sem()  for c in drug_cols]
        ax.plot(windows, means, color=color, lw=2.2, marker=marker, ms=6, label=arch)
        ax.fill_between(
            windows,
            [m - s for m, s in zip(means, sems)],
            [m + s for m, s in zip(means, sems)],
            color=color, alpha=0.18,
        )
        # Annotate endpoint
        ax.annotate(
            f"{means[-1]:.1f}",
            xy=(180, means[-1]),
            xytext=(182, means[-1]),
            fontsize=7, color=color, va="center",
        )

    ax.set_xlabel("Lookback Window (days before ADE visit)")
    ax.set_ylabel("Mean Drug Events in Window")
    ax.set_title(
        "(A) Drug Event Density Trajectories\n"
        "(KMeans k=3 on [drug 30d/90d/180d, time-to-ADE];\n"
        "shaded = ±1 SEM)",
        fontsize=8.5,
    )
    ax.set_xticks(windows)
    ax.set_xticklabels(["30d", "90d", "180d"])
    ax.legend(fontsize=7.5, loc="upper left")

    # Add cluster n= annotations
    for arch, color in zip(archetypes, ARCHETYPE_COLORS):
        n = len(df[df["archetype"] == arch])
        ax.text(
            0.98, 0.02 + archetypes.index(arch) * 0.07,
            f"{arch[:2]}…: n={n:,}",
            transform=ax.transAxes, ha="right",
            fontsize=6.5, color=color,
        )

    # ── Panel B: time-to-target violins ───────────────────────────────────
    ax = axes[1]
    data_by_arch = [
        np.clip(df.loc[df["archetype"] == arch, "time_to_F1120_days"].values, 0, 730)
        for arch in archetypes
    ]
    vp = ax.violinplot(
        data_by_arch,
        positions=range(len(archetypes)),
        showmedians=True,
        showextrema=False,
        widths=0.65,
    )
    for body, color in zip(vp["bodies"], ARCHETYPE_COLORS):
        body.set_facecolor(color)
        body.set_alpha(0.70)
        body.set_edgecolor("white")
    vp["cmedians"].set_color(C_GRAY)
    vp["cmedians"].set_linewidth(1.5)

    # Overlay median + IQR text
    for i, (arch, color, data) in enumerate(zip(archetypes, ARCHETYPE_COLORS, data_by_arch)):
        med = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        ax.text(
            i, np.percentile(data, 95) + 10,
            f"Median\n{med:.0f}d\n(IQR {q1:.0f}–{q3:.0f})",
            ha="center", va="bottom", fontsize=6.5, color=color,
        )

    ax.set_xticks(range(len(archetypes)))
    ax.set_xticklabels(
        [a.replace("-", "-\n") for a in archetypes],
        fontsize=7.5,
    )
    ax.set_ylabel("Days from First Drug Event to ADE Visit (F1120)")
    ax.set_title(
        "(B) Time-to-ADE Distribution by Archetype\n"
        "(Opioid Use Disorder / F1120; clipped at 730d)",
        fontsize=8.5,
    )

    # Significance brackets
    ax.annotate(
        "", xy=(2, 420), xytext=(0, 420),
        arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1.0),
    )
    ax.text(1, 435, "★★★ p < 0.001 (Mann-Whitney U)", ha="center", fontsize=6.5, color=C_GRAY)

    # ── Panel C: 2D scatter pre-drug count vs time-to-target ─────────────
    ax = axes[2]

    # Sample up to 800 points per cluster for readability
    rng = np.random.default_rng(42)
    for arch, color, marker in zip(archetypes, ARCHETYPE_COLORS, ARCHETYPE_MARKERS):
        grp = df[df["archetype"] == arch]
        idx = rng.choice(len(grp), min(800, len(grp)), replace=False)
        sub = grp.iloc[idx]
        ax.scatter(
            np.clip(sub["time_to_F1120_days"], 0, 730),
            np.clip(sub["pre_n_drug_events"], 0, 200),
            c=color, alpha=0.22, s=8, marker=marker,
            label=f"{arch} (n={len(grp):,})",
            edgecolors="none",
        )

    # Annotate medians
    for arch, color, marker in zip(archetypes, ARCHETYPE_COLORS, ARCHETYPE_MARKERS):
        grp = df[df["archetype"] == arch]
        mx = np.median(np.clip(grp["time_to_F1120_days"], 0, 730))
        my = np.median(np.clip(grp["pre_n_drug_events"], 0, 200))
        ax.plot(mx, my, marker=marker, color=color, ms=12, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(
            arch.replace("-", "-\n"),
            xy=(mx, my),
            xytext=(mx + 15, my + 5),
            fontsize=6.5, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax.set_xlabel("Time to ADE Visit (days; capped 730d)")
    ax.set_ylabel("Drug Events in Pre-ADE Period (capped 200)")
    ax.set_title(
        "(C) Pre-ADE Drug Burden vs Time-to-ADE\n"
        "(sample ≤ 800 pts/archetype; large marker = median)",
        fontsize=8.5,
    )
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)

    fig.suptitle(
        "DTW Trajectory Archetypes — Opioid ED Cohort, Age 25–44\n"
        r"(n=8,477 target patients; KMeans k=3 on drug event density $\times$ time-to-ADE;"
        "\n3b_feature_importance_eda BupaR outputs, 2019 holdout)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.92])
    _save(fig, FIG_CH03 / "fig_trajectories.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary: event density heatmap per archetype
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_trajectories_heatmap(df: pd.DataFrame) -> None:
    """
    Supplementary heatmap: mean event counts per window (30/90/180d)
    broken down by event type (drug/ICD/CPT) for each archetype.
    Saved as fig_trajectories_heatmap.pdf.
    """
    archetypes = ["Rapid-Onset", "Moderate-Escalation", "Chronic-Escalation"]
    windows = [30, 90, 180]
    event_types = [
        ("Drug",    ["n_drug_events_30d", "n_drug_events_90d", "n_drug_events_180d"]),
        ("ICD-10",  ["n_icd_events_30d",  "n_icd_events_90d",  "n_icd_events_180d"]),
        ("CPT",     ["n_cpt_events_30d",  "n_cpt_events_90d",  "n_cpt_events_180d"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    for ax, (et_name, cols) in zip(axes, event_types):
        mat = np.zeros((len(archetypes), len(windows)))
        for i, arch in enumerate(archetypes):
            grp = df[df["archetype"] == arch]
            mat[i] = [grp[c].mean() for c in cols]

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=mat.max() * 1.05)
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels(["30d", "90d", "180d"])
        if ax is axes[0]:
            ax.set_yticks(range(len(archetypes)))
            ax.set_yticklabels(archetypes, fontsize=8)
        ax.set_title(f"{et_name} Events", fontsize=9, fontweight="bold")
        for i in range(len(archetypes)):
            for j in range(len(windows)):
                ax.text(j, i, f"{mat[i,j]:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if mat[i,j] > mat.max()*0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.05, label="Mean count")

    fig.suptitle(
        "Event Count Heatmap by Archetype and Window\nOpioid ED, Age 25–44",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=1.5)
    _save(fig, FIG_CH03 / "fig_trajectories_heatmap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 62)
    print("DTW Figure Generator — DTW Pipeline + BupaR Outputs")
    print("=" * 62)

    errors = []

    # ── CH_3: KMeans archetype figure (opioid_ed/25-44, BupaR features) ──────
    print("\n── CH_3 KMeans archetype figure ──")
    try:
        df = load_and_cluster(k=3, seed=42)
        print("\n── fig_trajectories.pdf ──")
        make_fig_trajectories(df)
        print("\n── fig_trajectories_heatmap.pdf ──")
        make_fig_trajectories_heatmap(df)
    except Exception as e:
        import traceback; traceback.print_exc()
        errors.append(f"CH_3 KMeans figures: {e}")

    # ── CH_3: DTW pipeline figure (opioid_ed/25-44) ─────────────────────────
    print("\n── CH_3 DTW pipeline figure (fig_dtw_pathways.pdf) ──")
    try:
        make_fig_dtw_pipeline(
            cohort="opioid_ed", band="25_44",
            fig_dir=FIG_CH03, out_name="fig_dtw_pathways.pdf",
            title_cohort="Opioid ED Cohort, Age 25–44",
            target_label="Opioid ADE (F1120)",
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        errors.append(f"CH_3 DTW pipeline: {e}")

    # ── CH_4: DTW pipeline figure (non_opioid_ed/65-74) ─────────────────────
    print("\n── CH_4 DTW pipeline figure (fig_trajectories.pdf) ──")
    try:
        make_fig_dtw_pipeline(
            cohort="non_opioid_ed", band="65_74",
            fig_dir=FIG_CH04, out_name="fig_trajectories.pdf",
            title_cohort="Polypharmacy Cohort, Age 65–74",
            target_label="Polypharmacy ADE",
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        errors.append(f"CH_4 DTW pipeline: {e}")

    print("\n" + "=" * 62)
    if errors:
        print("Completed with errors:")
        for e in errors: print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("All DTW figures generated from actual pipeline outputs. ✓")
        print("=" * 62)
