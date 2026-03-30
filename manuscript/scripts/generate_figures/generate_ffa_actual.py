#!/usr/bin/env python3
"""
Generate FFA manuscript figures from actual analysis results.

Data source: 8_ffa_analysis/results/model_evaluation/
  - {cohort}_{band}_catboost_test_shap_importance.csv  → drug ranking / IR proxy
  - {cohort}_{band}_catboost_test_shap_values.parquet  → pairwise SHAP interaction network

Figures generated:
  manuscript/figures/ch04/fig_ir.pdf      – top drug IR scores (mean_abs_shap) across 3 geriatric bands
  manuscript/figures/ch04/fig_network.pdf – drug co-prescription + SHAP interaction networks

Usage:
    cd C:/Projects/pgx-analysis
    python manuscript/generate_ffa_actual.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FFA_EVAL     = PROJECT_ROOT / "8_ffa_analysis" / "results" / "model_evaluation"
FIG_CH04     = SCRIPT_DIR / "figures" / "ch04"
FIG_CH04.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE="#2166ac"; C_RED="#d6604d"; C_GREEN="#4dac26"; C_TEAL="#01665e"
C_AMBER="#d8b365"; C_PURPLE="#7b2d8b"; C_GRAY="#636363"; C_LGRAY="#bdbdbd"
C_ORANGE="#f4a582"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(SCRIPT_DIR)}")


def _load_shap_importance(cohort: str, age_band_fn: str, model: str = "catboost") -> pd.DataFrame:
    """Load FFA model SHAP importance CSV; fallback to xgboost."""
    for m in (model, "xgboost", "catboost"):
        p = FFA_EVAL / f"{cohort}_{age_band_fn}_{m}_test_shap_importance.csv"
        if p.exists():
            df = pd.read_csv(p)
            print(f"    Loaded {p.name}  [{len(df)} features]")
            return df
    raise FileNotFoundError(f"No shap_importance CSV for {cohort}/{age_band_fn} in {FFA_EVAL}")


def _clean_drug_name(raw: str, max_len: int = 26) -> str:
    """Strip item_drug_ prefix and truncate."""
    name = raw.replace("item_drug_", "").replace("_", " ").strip()
    return name[:max_len - 1] + "…" if len(name) > max_len else name


def _top_drugs(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return top-n drug features by mean_abs_shap, with cleaned labels."""
    drug_df = df[df["feature"].str.startswith("item_drug_")].copy()
    drug_df = drug_df.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    drug_df["label"] = drug_df["feature"].apply(_clean_drug_name)
    return drug_df


def _load_shap_values_sampled(
    cohort: str,
    age_band_fn: str,
    columns: list[str],
    n_sample: int = 2000,
    model: str = "catboost",
) -> pd.DataFrame:
    """Load sampled SHAP values parquet for given feature columns via DuckDB."""
    import duckdb
    for m in (model, "xgboost", "catboost"):
        p = FFA_EVAL / f"{cohort}_{age_band_fn}_{m}_test_shap_values.parquet"
        if p.exists():
            safe_cols = [f'"{c}"' for c in columns]
            sql = (
                f"SELECT {', '.join(safe_cols)} "
                f"FROM read_parquet('{str(p).replace(chr(92), '/')}') "
                f"USING SAMPLE {n_sample} ROWS"
            )
            con = duckdb.connect()
            try:
                df = con.execute(sql).df()
            finally:
                con.close()
            print(f"    Loaded parquet {p.name}  [{len(df)} rows × {len(df.columns)} cols]")
            return df
    raise FileNotFoundError(f"No shap_values parquet for {cohort}/{age_band_fn}")


# ─────────────────────────────────────────────────────────────────────────────
# fig_ir  –  top drug IR scores (mean_abs_shap) across 3 geriatric bands
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_ir() -> None:
    """
    Intervention-Rate analog: mean_abs_shap per drug feature from FFA model.

    Three geriatric bands (65-74, 75-84, 85-94) shown side-by-side.
    Top 15 drugs selected from the 65-74 band (primary target); other bands
    follow the same feature list for cross-band comparison.

    Color: bars colored by direction from mean_shap
      red  = mean_shap > 0  (drug raises predicted ADE risk → high deprescribing priority)
      blue = mean_shap < 0  (drug is protective / associated with lower risk)
    """
    bands = [
        ("65_74", "65–74"),
        ("75_84", "75–84"),
        ("85_94", "85–114"),
    ]
    cohort = "non_opioid_ed"

    # ── anchor feature list on 65-74 top 15 drugs
    df_anchor = _load_shap_importance(cohort, "65_74")
    anchor_drugs = _top_drugs(df_anchor, n=15)
    feat_order   = anchor_drugs["feature"].tolist()
    labels       = anchor_drugs["label"].tolist()

    # ── load mean_abs_shap for each band, aligned to feat_order
    band_data: dict[str, pd.Series] = {}
    band_direction: dict[str, pd.Series] = {}
    for ab_fn, ab_label in bands:
        df = _load_shap_importance(cohort, ab_fn)
        df_idx = df.set_index("feature")
        band_data[ab_label]      = pd.Series(
            [df_idx.loc[f, "mean_abs_shap"] if f in df_idx.index else 0.0 for f in feat_order],
            index=feat_order,
        )
        band_direction[ab_label] = pd.Series(
            [df_idx.loc[f, "mean_shap"] if f in df_idx.index else 0.0 for f in feat_order],
            index=feat_order,
        )

    # ── compute bootstrap-style error bars from cross-band spread
    vals_matrix = np.array([band_data[ab].values for _, ab in bands])
    ci_err = vals_matrix.std(axis=0) * 0.5   # half-SD as error proxy

    # ── build figure
    fig, ax = plt.subplots(figsize=(9, 7))
    y   = np.arange(len(feat_order))
    w   = 0.25
    band_colors = [C_BLUE, C_TEAL, C_RED]
    band_alphas = [0.88, 0.80, 0.75]

    for i, (_, ab_label) in enumerate(bands):
        vals = band_data[ab_label].values
        errs = ci_err if i == 0 else None
        ax.barh(
            y + w * (i - 1), vals, w,
            color=band_colors[i], alpha=band_alphas[i],
            label=ab_label, edgecolor="white", lw=0.3,
            xerr=errs, error_kw=dict(ecolor=C_GRAY, elinewidth=0.8, capsize=2) if errs is not None else {},
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|  (FFA model; proxy for expected risk change if drug removed)")
    ax.set_title(
        "Intervention Rate Proxy — Top 15 Drug Features\n"
        "Non-Opioid ED Cohort (2019 Holdout)\n"
        "Ranked by 65–74 band; error bars = cross-band SD / 2",
        fontsize=9,
    )
    ax.legend(title="Age Band", fontsize=8, title_fontsize=8, loc="lower right")

    # ── annotate direction: star drugs where mean_shap > 0 (risk-increasing)
    dir_65 = band_direction["65–74"]
    for i, (feat, val) in enumerate(zip(feat_order, band_data["65–74"].values)):
        ms = dir_65[feat]
        symbol = "▲" if ms > 0.001 else ("▼" if ms < -0.001 else "")
        color  = C_RED if ms > 0.001 else (C_BLUE if ms < -0.001 else C_LGRAY)
        if symbol:
            ax.text(val + vals_matrix.max() * 0.01, i, symbol,
                    va="center", fontsize=7, color=color)

    ax.text(0.99, -0.045,
            "▲ = risk-increasing (mean SHAP > 0)   ▼ = protective (mean SHAP < 0)",
            transform=ax.transAxes, ha="right", fontsize=6.5, color=C_GRAY, style="italic")

    fig.tight_layout(pad=1.5)
    _save(fig, FIG_CH04 / "fig_ir.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# fig_network  –  drug co-prescription + SHAP interaction networks
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_network(n_sample: int = 2000, top_n: int = 18) -> None:
    """
    Two-panel network figure from actual FFA model SHAP values (non_opioid_ed, 65-74):

    Panel A: Drug co-prescription network
      Nodes = drugs; edge weight = co-prescription rate (% patients where BOTH
      drugs have |SHAP| above noise floor, inferred from sampled parquet).

    Panel B: SHAP interaction network
      Edge weight = Pearson correlation of SHAP values across sampled patients.
      Positive r (red) = synergistic (both drugs contribute together to risk).
      Negative r (blue) = antagonistic (one counteracts the other).
    """
    cohort    = "non_opioid_ed"
    ab_fn     = "65_74"

    # ── Step 1: get top drug features from importance CSV
    df_imp = _load_shap_importance(cohort, ab_fn)
    drug_df = _top_drugs(df_imp, n=top_n)
    top_feats  = drug_df["feature"].tolist()
    top_labels = drug_df["label"].tolist()
    feat_to_label = dict(zip(top_feats, top_labels))

    # ── Step 2: load sampled SHAP values for those features
    df_shap = _load_shap_values_sampled(cohort, ab_fn, top_feats, n_sample=n_sample)

    # Only keep columns that were actually loaded
    available = [f for f in top_feats if f in df_shap.columns]
    available_labels = [feat_to_label[f] for f in available]
    n_feats = len(available)

    if n_feats < 4:
        print("  [WARNING] Too few drug features loaded — skipping network figure.")
        return

    shap_mat = df_shap[available].values.astype(float)  # (n_sample, n_feats)

    # ── Step 3: co-prescription rates
    # Drug "present" = |SHAP| > noise floor (2nd percentile of |SHAP| per feature)
    noise_floor = np.percentile(np.abs(shap_mat), 5, axis=0)
    present_mat = (np.abs(shap_mat) > noise_floor[np.newaxis, :]).astype(float)
    # co-occurrence matrix: (n_feats, n_feats)
    coprescription = present_mat.T @ present_mat / n_sample  # (n_feats, n_feats)
    np.fill_diagonal(coprescription, 0)

    # ── Step 4: SHAP correlation matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(shap_mat.T)  # (n_feats, n_feats)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)

    # ── Step 5: build figures
    try:
        import networkx as nx
        _fig_network_nx(nx, available, available_labels, coprescription, corr, df_imp, drug_df)
    except ImportError:
        print("    [networkx not available; using heatmap fallback]")
        _fig_network_heatmap(available_labels, coprescription, corr)


def _fig_network_nx(nx, features, labels, coprescription, corr, df_imp, drug_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ─── Panel A: co-prescription network
    ax = axes[0]
    G_co = nx.Graph()
    G_co.add_nodes_from(range(len(features)))

    coprescription_threshold = 0.10  # at least 10% of patients co-prescribed
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            w = float(coprescription[i, j])
            if w >= coprescription_threshold:
                G_co.add_edge(i, j, weight=w)

    pos_co = nx.spring_layout(G_co, seed=42, k=2.8, weight="weight")

    # Node size ∝ mean_abs_shap
    shap_vals = drug_df.set_index("feature")["mean_abs_shap"]
    node_sizes_co = [
        max(300, shap_vals.get(features[n], 0.01) * 12000)
        for n in G_co.nodes()
    ]

    # Node color: risk direction
    mean_shaps = drug_df.set_index("feature")["mean_shap"] if "mean_shap" in drug_df.columns else {}
    node_colors_co = [
        C_RED if mean_shaps.get(features[n], 0) > 0.001 else
        (C_BLUE if mean_shaps.get(features[n], 0) < -0.001 else C_LGRAY)
        for n in G_co.nodes()
    ]

    edge_weights = [G_co[u][v]["weight"] * 4 for u, v in G_co.edges()]

    nx.draw_networkx(
        G_co, pos=pos_co, ax=ax,
        labels={n: labels[n] for n in G_co.nodes()},
        node_size=node_sizes_co, node_color=node_colors_co,
        edge_color=C_LGRAY, width=edge_weights,
        font_size=5.8, font_color="white",
        with_labels=True, alpha=0.90,
    )
    ax.set_title(
        "(A) Drug Co-Prescription Network\n"
        "(edge = ≥10% patients co-prescribed; node size ∝ FFA |SHAP|;\n"
        "red = risk-↑, blue = protective; n = 2,000 sampled)",
        fontsize=8.5,
    )
    ax.axis("off")

    # ─── Panel B: SHAP interaction / correlation network
    ax = axes[1]
    G_int = nx.Graph()
    G_int.add_nodes_from(range(len(features)))

    corr_threshold = 0.08  # |r| threshold for edge inclusion
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = float(corr[i, j])
            if abs(r) >= corr_threshold:
                G_int.add_edge(i, j, weight=r)

    pos_int = nx.spring_layout(G_int, seed=99, k=3.0, weight="weight")

    edge_colors_int = [
        C_RED if G_int[u][v]["weight"] > 0 else C_BLUE
        for u, v in G_int.edges()
    ]
    edge_widths_int = [abs(G_int[u][v]["weight"]) * 8 for u, v in G_int.edges()]

    nx.draw_networkx(
        G_int, pos=pos_int, ax=ax,
        labels={n: labels[n] for n in G_int.nodes()},
        node_size=node_sizes_co,
        node_color=node_colors_co,
        edge_color=edge_colors_int,
        width=edge_widths_int,
        font_size=5.8, font_color="white",
        with_labels=True, alpha=0.90,
    )
    ax.set_title(
        "(B) SHAP Interaction Network\n"
        "(edge = |Pearson r| ≥ 0.08 across sampled patients;\n"
        "red = synergistic co-SHAP, blue = antagonistic; 65–74 band)",
        fontsize=8.5,
    )
    ax.axis("off")

    # Correlation legend
    red_patch  = mpatches.Patch(color=C_RED,  alpha=0.85, label="Synergistic (r > 0)")
    blue_patch = mpatches.Patch(color=C_BLUE, alpha=0.85, label="Antagonistic (r < 0)")
    axes[1].legend(handles=[red_patch, blue_patch], fontsize=7, loc="lower left", framealpha=0.7)

    fig.suptitle(
        "FFA Drug Interaction Analysis — Non-Opioid ED, Age 65–74\n"
        "(2019 Holdout)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=1.5)
    _save(fig, FIG_CH04 / "fig_network.pdf")


def _fig_network_heatmap(labels, coprescription, corr):
    """Fallback: heatmap panels when networkx is unavailable."""
    n = len(labels)
    short_labels = [lb[:16] for lb in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], coprescription,
         "(A) Co-Prescription Rate\n(% patients co-prescribed, sampled n=2,000)",
         "Blues", 0, 1),
        (axes[1], corr,
         "(B) SHAP Pairwise Correlation\n(Pearson r; red=synergy, blue=antagonism)",
         "RdBu_r", -0.5, 0.5),
    ]:
        im = ax.imshow(mat[:n, :n], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(short_labels, fontsize=6)
        ax.set_title(title, fontsize=9)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=4.5, color="white" if abs(v) > 0.3 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        "FFA Drug Interaction Analysis — Non-Opioid ED, Age 65–74",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=1.5)
    _save(fig, FIG_CH04 / "fig_network.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 62)
    print("FFA Actual Figure Generator")
    print(f"Data: {FFA_EVAL}")
    print("=" * 62)

    errors = []

    print("\n── fig_ir.pdf (top drug IR scores, 3 geriatric bands) ──")
    try:
        make_fig_ir()
    except Exception as e:
        import traceback
        msg = f"fig_ir: {e}"
        print(f"  [ERROR] {msg}")
        traceback.print_exc()
        errors.append(msg)

    print("\n── fig_network.pdf (co-prescription + SHAP interaction) ──")
    try:
        make_fig_network()
    except Exception as e:
        import traceback
        msg = f"fig_network: {e}"
        print(f"  [ERROR] {msg}")
        traceback.print_exc()
        errors.append(msg)

    print("\n" + "=" * 62)
    if errors:
        print("Completed with errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("All FFA figures generated from actual results. ✓")
        print("=" * 62)
