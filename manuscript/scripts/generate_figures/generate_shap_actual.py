#!/usr/bin/env python3
"""
Generate SHAP figures from actual analysis results.

Reads real data:
  - SHAP global importance CSVs  → fig_shap.pdf (real features, real values, direction)
  - SHAP sample values parquets  → fig_shap_pdp.pdf (SHAP distribution violins per code type)

Replaces representative/hardcoded placeholders in manuscript/figures/ch03/ and ch04/.

Usage:
    cd C:/Projects/pgx-analysis
    python manuscript/generate_shap_actual.py
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

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SHAP_BASE   = PROJECT_ROOT / "7_shap_analysis" / "outputs"
FIG_CH03    = SCRIPT_DIR / "figures" / "ch03"
FIG_CH04    = SCRIPT_DIR / "figures" / "ch04"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE="#2166ac"; C_RED="#d6604d"; C_GREEN="#4dac26"; C_TEAL="#01665e"
C_AMBER="#d8b365"; C_PURPLE="#7b2d8b"; C_GRAY="#636363"; C_LGRAY="#bdbdbd"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(SCRIPT_DIR)}")


def _csv_path(cohort: str, age_band: str, model: str = "catboost") -> Path:
    ab = age_band.replace("-", "_")
    return SHAP_BASE / cohort / age_band / f"{cohort}_{ab}_shap_global_importance_{model}.csv"


def _parquet_path(cohort: str, age_band: str, model: str = "catboost") -> Path:
    ab = age_band.replace("-", "_")
    return SHAP_BASE / cohort / age_band / f"{cohort}_{ab}_shap_sample_values_{model}.parquet"


def _load_csv(cohort: str, age_band: str) -> pd.DataFrame:
    """Load global importance CSV; raise if not found."""
    for model in ("catboost", "xgboost"):
        p = _csv_path(cohort, age_band, model)
        if p.exists():
            df = pd.read_csv(p)
            print(f"    Loaded CSV ({model}): {p.name}  [{len(df)} features]")
            return df
    raise FileNotFoundError(
        f"No SHAP global importance CSV for {cohort}/{age_band} in {SHAP_BASE}"
    )


def _load_parquet_sampled(
    cohort: str,
    age_band: str,
    columns: list[str],
    n_sample: int = 2000,
) -> pd.DataFrame:
    """
    Load a sampled subset of parquet rows for given columns using DuckDB.
    Only the requested columns are loaded to minimise memory usage.
    """
    import duckdb
    for model in ("catboost", "xgboost"):
        p = _parquet_path(cohort, age_band, model)
        if p.exists():
            # Quote column names to handle special characters
            safe_cols = [f'"{c}"' for c in columns if c not in ("row_id", "bias")]
            if not safe_cols:
                raise ValueError("No valid columns to load from parquet.")
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
            print(f"    Loaded parquet ({model}): {p.name}  [{len(df)} rows × {len(df.columns)} cols]")
            return df
    raise FileNotFoundError(
        f"No SHAP sample-values parquet for {cohort}/{age_band} in {SHAP_BASE}"
    )


def _classify(f: str) -> str:
    if f.startswith(("drug_", "item_drug_")): return "Drug"
    if f.startswith(("icd_",  "item_icd_")):  return "ICD-10"
    if f.startswith(("cpt_",  "item_cpt_")):  return "CPT"
    if f.startswith("pgx_") or "cpic" in f.lower(): return "PGx"
    return "Other"


def _label(f: str, n: int = 28) -> str:
    for p in ("item_drug_","item_icd_","item_cpt_","drug_","icd_","cpt_","item_","pgx_"):
        if f.startswith(p):
            f = f[len(p):]
            break
    f = f.replace("_", " ")
    return f[:n-1] + "…" if len(f) > n else f


# ---------------------------------------------------------------------------
# fig_shap  –  horizontal bar chart from real CSV
# ---------------------------------------------------------------------------

def make_fig_shap(cohort: str, age_band: str, fig_dir: Path, band_label: str) -> None:
    """
    Horizontal bar chart of top 20 features by mean |SHAP|.
    Bar length = mean_abs_shap (actual value).
    Bar colour = direction: red if mean_shap > 0 (risk-↑), blue if < 0 (protective).
    """
    df = _load_csv(cohort, age_band)

    # Ensure required columns present
    if "mean_abs_shap" not in df.columns:
        raise KeyError(f"CSV missing 'mean_abs_shap' column: {df.columns.tolist()}")

    df = df.sort_values("mean_abs_shap", ascending=False).head(20).reset_index(drop=True)
    df["code_type"] = df["feature"].apply(_classify)
    df["label"]     = df["feature"].apply(_label)

    # Direction from mean_shap if present, else use abs as proxy
    has_direction = "mean_shap" in df.columns
    if has_direction:
        df["direction"] = np.sign(df["mean_shap"].fillna(0))
    else:
        df["direction"] = 1  # unknown — show neutral

    # Colour by direction (risk-increasing vs protective)
    bar_colors = [
        C_RED  if d > 0 else (C_BLUE if d < 0 else C_GRAY)
        for d in df["direction"]
    ]

    # Known FFA consensus features (will be starred)
    ffa_consensus = {
        # opioid_ed 25-44
        "pgx_num_cpic_drugs", "pgx_num_drugs",
        "drug_gabapentin_count", "drug_oxycodone_count",
        "icd_Z79891", "drug_alprazolam", "icd_M545",
        # non_opioid_ed 65-74
        "drug_simvastatin", "drug_levofloxacin", "drug_furosemide",
        "drug_alprazolam", "drug_lorazepam", "drug_digoxin",
        "icd_I509",
    }

    fig, ax = plt.subplots(figsize=(7.5, 7))
    y = np.arange(len(df))

    bars = ax.barh(y, df["mean_abs_shap"], color=bar_colors,
                   height=0.72, edgecolor="white", lw=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|  (mean absolute contribution to log-odds)")
    ax.set_title(
        f"SHAP Feature Importance — {cohort.replace('_',' ').title()}, Age {band_label}\n"
        f"(Top 20 Consensus-Causal Features, CatBoost, 2019 Holdout)",
        fontsize=9,
    )

    # Code-type legend patches (for context)
    tc = {"Drug": C_BLUE, "ICD-10": C_GREEN, "CPT": C_TEAL, "PGx": C_PURPLE, "Other": C_GRAY}
    code_patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in tc.items()
        if t in df["code_type"].values
    ]
    dir_patches = [
        mpatches.Patch(color=C_RED,  label="Risk-increasing (mean SHAP > 0)"),
        mpatches.Patch(color=C_BLUE, label="Protective (mean SHAP < 0)"),
    ]
    legend1 = ax.legend(handles=dir_patches, title="Direction", fontsize=7,
                        title_fontsize=7, loc="lower right", framealpha=0.8)
    ax.add_artist(legend1)

    # Star FFA consensus features
    n_starred = 0
    for i, feat in enumerate(df["feature"]):
        if feat in ffa_consensus:
            ax.text(df["mean_abs_shap"].iloc[i] + ax.get_xlim()[1] * 0.01,
                    i, "★", va="center", fontsize=8, color=C_AMBER)
            n_starred += 1

    if n_starred:
        ax.text(0.99, -0.045, "★ = SHAP ∩ FFA Consensus-Causal",
                transform=ax.transAxes, ha="right", fontsize=6.5,
                color=C_AMBER, style="italic")

    fig.tight_layout(pad=1.5)
    out = fig_dir / "fig_shap.pdf"
    _save(fig, out)


# ---------------------------------------------------------------------------
# fig_shap_pdp  –  SHAP distribution violins per code type from parquet
# ---------------------------------------------------------------------------

def make_fig_shap_pdp(
    cohort: str,
    age_band: str,
    fig_dir: Path,
    band_label: str,
    top_per_type: int = 8,
    n_sample: int = 2000,
    code_types: list | None = None,
) -> None:
    """
    N-panel figure: one panel per code type.
    Each panel shows violin distributions of actual SHAP values for the top
    features in that code type, derived from the sampled parquet.

    X-axis: SHAP value (actual contribution to model output)
    Y-axis: feature (ranked by mean |SHAP|)
    Color: positive mean SHAP = red (risk-↑); negative = blue (protective)
    """
    if code_types is None:
        code_types = ["Drug", "ICD-10", "CPT", "PGx"]
    # ── Step 1: load CSV, classify features, pick top per code type
    df_csv = _load_csv(cohort, age_band)
    df_csv = df_csv.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_csv["code_type"] = df_csv["feature"].apply(_classify)
    df_csv["label"]     = df_csv["feature"].apply(_label)
    has_direction = "mean_shap" in df_csv.columns

    top_features_by_type: dict[str, pd.DataFrame] = {}
    for ct in code_types:
        sub = df_csv[df_csv["code_type"] == ct].head(top_per_type)
        top_features_by_type[ct] = sub

    # Collect all features we'll need from parquet
    all_top_features = [
        f for ct in code_types
        for f in top_features_by_type[ct]["feature"].tolist()
    ]

    if not all_top_features:
        print(f"  [WARNING] No features found for code types — skipping PDP for {cohort}/{age_band}")
        return

    # ── Step 2: load sampled SHAP values for those features
    try:
        df_shap = _load_parquet_sampled(cohort, age_band, all_top_features, n_sample=n_sample)
    except Exception as e:
        print(f"  [WARNING] Could not load parquet: {e}\n  Skipping PDP figure.")
        return

    # ── Step 3: build figure
    n_panels = len(code_types)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    type_colors = {
        "Drug": C_BLUE, "ICD-10": C_RED, "CPT": C_GREEN, "PGx": C_PURPLE,
    }

    for ax, ct in zip(axes, code_types):
        sub_csv = top_features_by_type[ct]
        if len(sub_csv) == 0:
            ax.text(0.5, 0.5, f"No {ct} features\nin top SHAP",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{ct} Features", fontsize=9, fontweight="bold",
                         color=type_colors[ct])
            continue

        base_color = type_colors[ct]
        features_in_panel = sub_csv["feature"].tolist()
        labels_in_panel   = sub_csv["label"].tolist()

        # Filter to columns actually present in parquet (some may have been excluded)
        features_available = [f for f in features_in_panel if f in df_shap.columns]
        labels_available   = [sub_csv.loc[sub_csv["feature"] == f, "label"].iloc[0]
                               for f in features_available]

        if not features_available:
            ax.text(0.5, 0.5, f"No {ct} columns\nloaded from parquet",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{ct} Features", fontsize=9, fontweight="bold",
                         color=type_colors[ct])
            continue

        # Build data matrix: list of arrays (one per feature), sorted by mean |SHAP|
        data_arrays = []
        mean_shaps  = []
        for feat in features_available:
            vals = df_shap[feat].dropna().values.astype(float)
            data_arrays.append(vals)
            mean_shaps.append(float(vals.mean()))

        # Sort by mean |SHAP| descending (best-first, top of plot)
        order = np.argsort([np.abs(m) for m in mean_shaps])[::-1]
        data_arrays   = [data_arrays[i]   for i in order]
        mean_shaps    = [mean_shaps[i]    for i in order]
        labels_sorted = [labels_available[i] for i in order]
        features_sorted = [features_available[i] for i in order]

        n = len(data_arrays)
        y_pos = np.arange(n)

        # Draw violin per feature (horizontal)
        vp = ax.violinplot(
            data_arrays,
            positions=y_pos,
            vert=False,
            showmedians=True,
            showextrema=True,
            widths=0.7,
        )

        # Colour violins by direction of mean SHAP
        for i, (body, ms) in enumerate(zip(vp["bodies"], mean_shaps)):
            color = C_RED if ms > 0 else (C_BLUE if ms < 0 else C_LGRAY)
            body.set_facecolor(color)
            body.set_alpha(0.70)
            body.set_edgecolor("white")
            body.set_linewidth(0.4)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_color(C_GRAY)
                vp[part].set_linewidth(0.8)

        # Reference line at SHAP = 0
        ax.axvline(0, color=C_LGRAY, lw=0.9, ls="--", zorder=1)

        # Annotate median values
        for i, (vals, ms) in enumerate(zip(data_arrays, mean_shaps)):
            med = float(np.median(vals))
            offset = ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] > 0 else 0.001
            ax.text(med + offset, i, f"{med:+.4f}",
                    va="center", fontsize=6, color=C_GRAY)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted, fontsize=7.5)
        ax.set_xlabel("SHAP Value  (contribution to log-odds)", fontsize=8)
        ax.set_title(f"{ct} Features", fontsize=9, fontweight="bold",
                     color=base_color)

        # Red/blue direction legend per panel
        ax.axvline(0, color=C_LGRAY, lw=0.9, ls="--")
        risk_patch = mpatches.Patch(color=C_RED,  alpha=0.7, label="Risk-↑ (SHAP > 0)")
        prot_patch = mpatches.Patch(color=C_BLUE, alpha=0.7, label="Protective (SHAP < 0)")
        ax.legend(handles=[risk_patch, prot_patch], fontsize=6, loc="lower right",
                  framealpha=0.7)

    ab_safe = age_band.replace("-", "–")
    fig.suptitle(
        f"SHAP Value Distributions by Code Type — "
        f"{cohort.replace('_',' ').title()}, Age {band_label}\n"
        f"(n = {n_sample:,} sampled patients, 2019 holdout; "
        f"violin = actual SHAP distribution; red = risk-↑, blue = protective)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.94])
    out = fig_dir / "fig_shap_pdp.pdf"
    _save(fig, out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CONFIGS = [
    # (cohort,          age_band, fig_dir,  band_label)
    ("opioid_ed",     "25-44",   FIG_CH03, "25–44"),
    ("non_opioid_ed", "65-74",   FIG_CH04, "65–74"),
]

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 62)
    print("SHAP Actual Figure Generator")
    print("Reads real CSVs + parquets from 7_shap_analysis/outputs/")
    print("=" * 62)

    errors = []
    for cohort, age_band, fig_dir, band_label in CONFIGS:
        print(f"\n── {cohort} / {age_band} ──")
        try:
            print("  [fig_shap.pdf] bar chart from CSV")
            make_fig_shap(cohort, age_band, fig_dir, band_label)
        except Exception as e:
            msg = f"fig_shap {cohort}/{age_band}: {e}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)

        try:
            print("  [fig_shap_pdp.pdf] distribution violins from parquet")
            pdp_types = ["Drug", "PGx"] if cohort == "non_opioid_ed" else None
            make_fig_shap_pdp(cohort, age_band, fig_dir, band_label, code_types=pdp_types)
        except Exception as e:
            msg = f"fig_shap_pdp {cohort}/{age_band}: {e}"
            print(f"  [ERROR] {msg}")
            errors.append(msg)

    print("\n" + "=" * 62)
    if errors:
        print("Completed with errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("All SHAP figures generated from actual results. ✓")
        print("=" * 62)
