#!/usr/bin/env python3
"""
Combined per-bin heatmaps: activity frequency, FP-Growth itemsets, and DTW sequence patterns
across all density bins (low / medium / high / extreme) in a single matrix view.

Mirrors the feature-importance cohort heatmap pattern (row_labels, column_labels, matrix)
so the dashboard can render it with the same Plotly component.

Outputs (per cohort/age_band):
  density/combined/bupar_activity_heatmap.json    -- BupaR: activity × bin frequency matrix
  density/combined/bupar_activity_heatmap.png     -- static PNG version
  density/combined/fpgrowth_itemset_heatmap.json  -- FP-Growth: itemset × bin support matrix
  density/combined/fpgrowth_itemset_heatmap.png
  density/combined/dtw_sequence_heatmap.json      -- DTW: code × bin sequence-position matrix
  density/combined/dtw_sequence_heatmap.png
  density/combined/bin_summary.json               -- n_patients per bin + escalation data

Usage:
    python create_combined_bin_heatmap.py --cohort opioid_ed --age-band 13-24
    python create_combined_bin_heatmap.py --cohort opioid_ed --age-band 13-24 --no-png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.pipeline_logger import setup_pipeline_logger  # noqa: E402

DENSITY_BINS = ("low", "medium", "high", "extreme")
BIN_COLORS = {"low": "#10b981", "medium": "#3b82f6", "high": "#f59e0b", "extreme": "#ef4444"}

DTW_VIZ_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "dtw"
BUPAR_VIZ_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar"
FPGROWTH_VIZ_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth"


# ---------------------------------------------------------------------------
# BupaR activity heatmap
# ---------------------------------------------------------------------------

def _load_bupar_activity_freqs(
    cohort_name: str, age_band: str, top_n: int = 30
) -> Optional[Dict[str, Any]]:
    """Load per-bin BupaR activity_frequency.json files and build a combined matrix."""
    age_band_fname = age_band.replace("-", "_")
    base = f"{cohort_name}_{age_band_fname}"
    bin_data: Dict[str, Dict[str, int]] = {}
    n_patients: Dict[str, int] = {}

    for bin_name in DENSITY_BINS:
        json_path = (
            BUPAR_VIZ_ROOT / cohort_name / age_band_fname
            / "density" / bin_name / "plots" / f"{base}_activity_frequency.json"
        )
        if not json_path.exists():
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                freq = json.load(f)
            n_patients[bin_name] = freq.get("n_patients", 0)
            data = freq.get("data", {})
            # data: {activity: [count_per_year...]} or {activity: [total]}
            for activity, counts in data.items():
                total = sum(counts) if isinstance(counts, list) else int(counts)
                bin_data.setdefault(activity, {})[bin_name] = total
        except Exception:
            continue

    if not bin_data:
        return None

    # Normalize by n_patients per bin (rate per patient), keep top_n by total
    normalized: Dict[str, Dict[str, float]] = {}
    for activity, by_bin in bin_data.items():
        row = {}
        for b in DENSITY_BINS:
            n = n_patients.get(b, 1) or 1
            row[b] = round(by_bin.get(b, 0) / n, 6)
        normalized[activity] = row

    # Sort by total across bins, take top_n
    sorted_activities = sorted(
        normalized.keys(),
        key=lambda a: sum(normalized[a].values()),
        reverse=True,
    )[:top_n]

    matrix = [[normalized[a].get(b, 0.0) for b in DENSITY_BINS] for a in sorted_activities]
    return {
        "cohort": cohort_name,
        "age_band": age_band,
        "heatmap_type": "bupar_activity_frequency",
        "row_labels": sorted_activities,
        "column_labels": list(DENSITY_BINS),
        "matrix": matrix,
        "metric": "rate_per_patient",
        "n_patients_per_bin": n_patients,
    }


# ---------------------------------------------------------------------------
# FP-Growth itemset heatmap
# ---------------------------------------------------------------------------

def _load_fpgrowth_itemset_supports(
    cohort_name: str, age_band: str, top_n: int = 30
) -> Optional[Dict[str, Any]]:
    """Load per-bin FP-Growth itemset JSON files and build a combined matrix of support values."""
    age_band_fname = age_band.replace("-", "_")
    item_types = ["drug_name", "icd_code"]
    bin_itemset_support: Dict[str, Dict[str, float]] = {}

    for bin_name in DENSITY_BINS:
        bin_dir = FPGROWTH_VIZ_ROOT / cohort_name / age_band_fname / "density" / bin_name
        if not bin_dir.exists():
            continue
        for item_type in item_types:
            json_path = bin_dir / f"{item_type}_itemsets.json"
            if not json_path.exists():
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    itemsets = json.load(f)
                if not isinstance(itemsets, list):
                    continue
                for row in itemsets:
                    items = row.get("itemsets", [])
                    support = float(row.get("support", 0))
                    if not items:
                        continue
                    label = " + ".join(sorted(str(i) for i in items))
                    existing = bin_itemset_support.setdefault(label, {})
                    existing[bin_name] = max(existing.get(bin_name, 0.0), support)
            except Exception:
                continue

    if not bin_itemset_support:
        return None

    # Sort by total support across bins, keep top_n
    sorted_itemsets = sorted(
        bin_itemset_support.keys(),
        key=lambda k: sum(bin_itemset_support[k].values()),
        reverse=True,
    )[:top_n]

    matrix = [
        [bin_itemset_support[k].get(b, 0.0) for b in DENSITY_BINS]
        for k in sorted_itemsets
    ]
    return {
        "cohort": cohort_name,
        "age_band": age_band,
        "heatmap_type": "fpgrowth_itemset_support",
        "row_labels": sorted_itemsets,
        "column_labels": list(DENSITY_BINS),
        "matrix": matrix,
        "metric": "support",
    }


# ---------------------------------------------------------------------------
# DTW sequence-code heatmap
# ---------------------------------------------------------------------------

def _load_dtw_sequence_heatmaps(
    cohort_name: str, age_band: str, top_n: int = 30
) -> Optional[Dict[str, Any]]:
    """Load per-bin DTW sequence_heatmap.json and build a combined code × bin matrix."""
    age_band_fname = age_band.replace("-", "_")
    bin_code_counts: Dict[str, Dict[str, int]] = {}
    n_patients: Dict[str, int] = {}

    for bin_name in DENSITY_BINS:
        json_path = (
            DTW_VIZ_ROOT / cohort_name / age_band_fname
            / "density" / bin_name / "sequence_heatmap.json"
        )
        if not json_path.exists():
            continue
        try:
            with open(json_path, encoding="utf-8") as f:
                heatmap = json.load(f)
            if heatmap.get("empty"):
                continue
            # sequence_heatmap has drug/icd/cpt sub-objects with codes/counts
            for section_key in ("drug", "icd", "cpt"):
                section = heatmap.get(section_key, {})
                if not isinstance(section, dict):
                    continue
                codes = section.get("codes", [])
                counts = section.get("counts", [])
                if not codes or not counts:
                    continue
                # counts may be per-position list or flat list; take sum across positions
                for code, cnt in zip(codes, counts):
                    total = sum(cnt) if isinstance(cnt, list) else int(cnt)
                    label = f"{section_key.upper()}:{code}"
                    bin_code_counts.setdefault(label, {})[bin_name] = (
                        bin_code_counts.get(label, {}).get(bin_name, 0) + total
                    )
            if "n_patients" in heatmap:
                n_patients[bin_name] = int(heatmap["n_patients"])
        except Exception:
            continue

    if not bin_code_counts:
        return None

    sorted_codes = sorted(
        bin_code_counts.keys(),
        key=lambda k: sum(bin_code_counts[k].values()),
        reverse=True,
    )[:top_n]

    # Normalize by n_patients if available
    def _val(code: str, bin_name: str) -> float:
        raw = bin_code_counts[code].get(bin_name, 0)
        n = n_patients.get(bin_name, 1) or 1
        return round(raw / n, 6)

    matrix = [[_val(c, b) for b in DENSITY_BINS] for c in sorted_codes]
    return {
        "cohort": cohort_name,
        "age_band": age_band,
        "heatmap_type": "dtw_sequence_code_frequency",
        "row_labels": sorted_codes,
        "column_labels": list(DENSITY_BINS),
        "matrix": matrix,
        "metric": "rate_per_patient",
        "n_patients_per_bin": n_patients,
    }


# ---------------------------------------------------------------------------
# PNG generation (optional)
# ---------------------------------------------------------------------------

def _write_heatmap_png(
    data: Dict[str, Any],
    out_path: Path,
    title: str,
    logger=None,
) -> bool:
    """Render a heatmap PNG using seaborn + matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
    except ImportError as e:
        if logger:
            logger.warning("matplotlib/seaborn not available for PNG heatmap: %s", e)
        return False

    row_labels = data["row_labels"]
    col_labels = data["column_labels"]
    matrix = data["matrix"]
    if not row_labels or not col_labels or not matrix:
        return False

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    n_rows, n_cols = len(row_labels), len(col_labels)
    fig_w = max(6, n_cols * 2.0)
    fig_h = max(6, n_rows * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd",
        annot=(n_rows <= 25),
        fmt=".3f" if data.get("metric") in ("support", "rate_per_patient") else ".0f",
        linewidths=0.3,
        linecolor="#e5e7eb",
        cbar_kws={"label": data.get("metric", "value"), "shrink": 0.6},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Event Density Bin", fontsize=10)
    ax.set_ylabel("Activity / Item", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Bin summary JSON
# ---------------------------------------------------------------------------

def _build_bin_summary(cohort_name: str, age_band: str) -> Dict[str, Any]:
    """Collect n_patients per bin from available data sources."""
    age_band_fname = age_band.replace("-", "_")
    base = f"{cohort_name}_{age_band_fname}"
    summary: Dict[str, Any] = {"cohort": cohort_name, "age_band": age_band, "n_patients_per_bin": {}}

    for bin_name in DENSITY_BINS:
        # Try BupaR activity_frequency.json for n_patients
        json_path = (
            BUPAR_VIZ_ROOT / cohort_name / age_band_fname
            / "density" / bin_name / "plots" / f"{base}_activity_frequency.json"
        )
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as f:
                    d = json.load(f)
                n = d.get("n_patients", 0)
                if n:
                    summary["n_patients_per_bin"][bin_name] = n
                    continue
            except Exception:
                pass
        # Fallback: DTW chart_data density_bin
        chart_path = (
            DTW_VIZ_ROOT / cohort_name / age_band_fname
            / "density" / bin_name / "chart_data.json"
        )
        if chart_path.exists():
            try:
                with open(chart_path, encoding="utf-8") as f:
                    d = json.load(f)
                n = d.get("n_patients", 0)
                if n:
                    summary["n_patients_per_bin"][bin_name] = n
            except Exception:
                pass

    # Load bin transition data if available
    transitions_path = (
        DTW_VIZ_ROOT / cohort_name / age_band_fname
        / "density" / "transitions" / "bin_transitions.json"
    )
    if transitions_path.exists():
        try:
            with open(transitions_path, encoding="utf-8") as f:
                tr = json.load(f)
            summary["escalation_rate"] = tr.get("escalation_rate")
            summary["de_escalation_rate"] = tr.get("de_escalation_rate")
            summary["stable_rate"] = tr.get("stable_rate")
            summary["n_patients_tracked"] = tr.get("n_patients_tracked")
            summary["per_year_distribution"] = tr.get("per_year_distribution")
        except Exception:
            pass

    return summary


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def _upload_combined_to_s3(
    combined_dir: Path,
    cohort_name: str,
    age_band: str,
    logger=None,
) -> None:
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() in ("1", "true", "yes"):
        return
    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dash_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    s3_base = f"{dash_prefix.rstrip('/')}/visualizations"
    try:
        import boto3 as _boto3
        _s3 = _boto3.client("s3")
        for p in sorted(combined_dir.rglob("*")):
            if not p.is_file():
                continue
            # Map local combined_dir to s3 path segments
            # local: .../dtw/{cohort}/{ab}/density/combined/... or bupar/...
            # s3: visualizations/dtw/{cohort}/{ab}/density/combined/...
            try:
                rel = p.relative_to(
                    REPO_ROOT / "10_risk_dashboard" / "visualizations"
                ).as_posix()
                key = f"{s3_base}/{rel}"
                _s3.put_object(
                    Bucket=s3_bucket, Key=key.replace(f"{s3_base}/", f"{s3_base}/"),
                    Body=p.read_bytes(),
                    ContentType="application/json" if p.suffix == ".json" else "image/png",
                )
                if logger:
                    logger.info("Uploaded combined heatmap: s3://%s/%s", s3_bucket, key)
            except Exception as e:
                if logger:
                    logger.warning("S3 upload failed for %s: %s", p.name, e)
    except Exception as e:
        if logger:
            logger.warning("Combined bin heatmap S3 upload failed: %s", e)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_combined_bin_heatmaps(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    top_n: int = 30,
    write_png: bool = True,
    force: bool = False,
    logger=None,
) -> Dict[str, Any]:
    """Build all combined bin heatmaps for one cohort/age_band. Returns dict of written paths."""
    def _log(level: str, msg: str, *args) -> None:
        if logger:
            getattr(logger, level)(msg, *args)
        else:
            print(f"[{level.upper()}] " + (msg % args if args else msg))

    age_band_fname = age_band.replace("-", "_")

    # One combined dir per visualization type
    combined_dirs = {
        "bupar": BUPAR_VIZ_ROOT / cohort_name / age_band_fname / "density" / "combined",
        "fpgrowth": FPGROWTH_VIZ_ROOT / cohort_name / age_band_fname / "density" / "combined",
        "dtw": DTW_VIZ_ROOT / cohort_name / age_band_fname / "density" / "combined",
    }
    written: Dict[str, str] = {}

    # ---- BupaR activity heatmap ----
    combined_dirs["bupar"].mkdir(parents=True, exist_ok=True)
    bupar_json = combined_dirs["bupar"] / "bupar_activity_heatmap.json"
    if force or not bupar_json.exists():
        bupar_data = _load_bupar_activity_freqs(cohort_name, age_band, top_n=top_n)
        if bupar_data:
            bupar_json.write_text(json.dumps(bupar_data, indent=2), encoding="utf-8")
            _log("info", "BupaR combined heatmap written: %s", bupar_json)
            written["bupar_json"] = str(bupar_json)
            if write_png:
                png_path = combined_dirs["bupar"] / "bupar_activity_heatmap.png"
                title = f"Activity Frequency by Density Bin — {cohort_name} / {age_band}"
                if _write_heatmap_png(bupar_data, png_path, title, logger=logger):
                    written["bupar_png"] = str(png_path)
        else:
            _log("info", "BupaR combined heatmap skipped: no per-bin activity_frequency.json found")
    else:
        _log("info", "BupaR combined heatmap exists; skipping (use --force)")

    # ---- FP-Growth itemset heatmap ----
    combined_dirs["fpgrowth"].mkdir(parents=True, exist_ok=True)
    fpgrowth_json = combined_dirs["fpgrowth"] / "fpgrowth_itemset_heatmap.json"
    if force or not fpgrowth_json.exists():
        fpgrowth_data = _load_fpgrowth_itemset_supports(cohort_name, age_band, top_n=top_n)
        if fpgrowth_data:
            fpgrowth_json.write_text(json.dumps(fpgrowth_data, indent=2), encoding="utf-8")
            _log("info", "FP-Growth combined heatmap written: %s", fpgrowth_json)
            written["fpgrowth_json"] = str(fpgrowth_json)
            if write_png:
                png_path = combined_dirs["fpgrowth"] / "fpgrowth_itemset_heatmap.png"
                title = f"Itemset Support by Density Bin — {cohort_name} / {age_band}"
                if _write_heatmap_png(fpgrowth_data, png_path, title, logger=logger):
                    written["fpgrowth_png"] = str(png_path)
        else:
            _log("info", "FP-Growth combined heatmap skipped: no per-bin itemset JSON found")
    else:
        _log("info", "FP-Growth combined heatmap exists; skipping (use --force)")

    # ---- DTW sequence heatmap ----
    combined_dirs["dtw"].mkdir(parents=True, exist_ok=True)
    dtw_json = combined_dirs["dtw"] / "dtw_sequence_heatmap.json"
    if force or not dtw_json.exists():
        dtw_data = _load_dtw_sequence_heatmaps(cohort_name, age_band, top_n=top_n)
        if dtw_data:
            dtw_json.write_text(json.dumps(dtw_data, indent=2), encoding="utf-8")
            _log("info", "DTW combined heatmap written: %s", dtw_json)
            written["dtw_json"] = str(dtw_json)
            if write_png:
                png_path = combined_dirs["dtw"] / "dtw_sequence_heatmap.png"
                title = f"Sequence Code Frequency by Density Bin — {cohort_name} / {age_band}"
                if _write_heatmap_png(dtw_data, png_path, title, logger=logger):
                    written["dtw_png"] = str(png_path)
        else:
            _log("info", "DTW combined heatmap skipped: no per-bin sequence_heatmap.json found")
    else:
        _log("info", "DTW combined heatmap exists; skipping (use --force)")

    # ---- Bin summary JSON (into dtw/combined/ as canonical location) ----
    summary_json = combined_dirs["dtw"] / "bin_summary.json"
    if force or not summary_json.exists():
        summary = _build_bin_summary(cohort_name, age_band)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log("info", "Bin summary written: %s", summary_json)
        written["bin_summary"] = str(summary_json)

    # ---- S3 upload ----
    for _, combined_dir in combined_dirs.items():
        _upload_combined_to_s3(combined_dir, cohort_name, age_band, logger=logger)

    _log("info", "Combined bin heatmaps complete: %d artifacts written", len(written))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined per-bin heatmaps for BupaR, FP-Growth, and DTW")
    parser.add_argument("--cohort", "--cohort-name", dest="cohort", required=True)
    parser.add_argument("--age-band", required=True)
    parser.add_argument("--top-n", type=int, default=30, help="Max rows in each heatmap (default: 30)")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG generation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    logger = setup_pipeline_logger(
        step_name="9_dashboard_visuals",
        cohort=args.cohort,
        age_band=args.age_band,
        script_name="create_combined_bin_heatmap",
    )

    written = build_combined_bin_heatmaps(
        project_root=project_root,
        cohort_name=args.cohort,
        age_band=args.age_band,
        top_n=args.top_n,
        write_png=not args.no_png,
        force=args.force,
        logger=logger.logger,
    )

    if written:
        for key, path in written.items():
            logger.info("  %s -> %s", key, path)
    else:
        logger.warning("No combined heatmap artifacts written for %s/%s", args.cohort, args.age_band)

    logger.log_summary()
    sys.exit(0)


if __name__ == "__main__":
    main()
