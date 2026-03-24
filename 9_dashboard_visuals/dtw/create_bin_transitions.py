#!/usr/bin/env python3
"""
Bin transition analysis: track patients moving between event-density bins across years.

For each cohort/age_band, computes per-patient n_event_bin for each year in the training
window (2016-2019), then identifies transitions (e.g. low→medium, medium→extreme) and
patients whose clinical utilization escalated or de-escalated over time.

Output: density/transitions/bin_transitions.json with:
  - transition_matrix: {from_bin: {to_bin: count}} across all consecutive year pairs
  - sankey_nodes / sankey_links: for dashboard Sankey diagram
  - escalation_rate: fraction of patients whose bin increased year-over-year
  - de_escalation_rate: fraction whose bin decreased
  - stable_rate: fraction whose bin stayed the same
  - top_escalation_drugs: most common drug codes among escalating patients
  - top_de_escalation_drugs: most common drug codes among de-escalating patients
  - per_year_distribution: {year: {bin: count}} for trend visualization

Usage:
    python create_bin_transitions.py --cohort opioid_ed --age-band 13-24
    python create_bin_transitions.py --cohort opioid_ed --age-band 13-24 --project-root /path/to/repo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.pipeline_logger import setup_pipeline_logger  # noqa: E402
from py_helpers.model_data_paths import resolve_model_events_paths  # noqa: E402
from py_helpers.event_density_utils import (  # noqa: E402
    DENSITY_BINS,
    assign_n_event_bins,
    load_thresholds,
)

TRAIN_YEARS = [2016, 2017, 2018, 2019]
BIN_ORDER = {b: i for i, b in enumerate(DENSITY_BINS)}  # low=0, medium=1, high=2, extreme=3


def _dtw_output_root(project_root: Path) -> Path:
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


def compute_bin_transitions(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    force: bool = False,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """Compute per-patient bin transitions across years and write output JSON."""
    def _log(level: str, msg: str, *args: Any) -> None:
        if logger is not None:
            getattr(logger, level)(msg, *args)
        else:
            print(f"[{level.upper()}] " + (msg % args if args else msg))

    age_band_fname = age_band.replace("-", "_")
    out_dir = _dtw_output_root(project_root) / cohort_name / age_band_fname / "density" / "transitions"
    out_path = out_dir / "bin_transitions.json"

    if not force and out_path.exists():
        _log("info", "Bin transitions already exist at %s; skipping (use --force)", out_path)
        return json.loads(out_path.read_text(encoding="utf-8"))

    # Load thresholds
    thresholds_path = (
        project_root / "6_final_model" / "outputs" / cohort_name / age_band_fname
        / "n_event_bin_thresholds.json"
    )
    if not thresholds_path.exists():
        _log("warning", "n_event_bin_thresholds.json not found at %s; cannot compute bin transitions", thresholds_path)
        return None
    thresholds = load_thresholds(thresholds_path)

    # Load model events
    paths = resolve_model_events_paths(project_root, cohort_name, age_band)
    if not paths:
        _log("warning", "model_events not found for %s/%s; cannot compute bin transitions", cohort_name, age_band)
        return None

    try:
        import duckdb
        con = duckdb.connect(":memory:")
        if len(paths) == 1:
            from_clause = f"read_parquet('{str(paths[0]).replace(chr(92), '/')}')"
        else:
            from_clause = (
                f"(SELECT * FROM read_parquet('{str(paths[0]).replace(chr(92), '/')}') "
                f"UNION ALL SELECT * FROM read_parquet('{str(paths[1]).replace(chr(92), '/')}'))"
            )
        year_list = ", ".join(str(y) for y in TRAIN_YEARS)
        df = con.execute(
            f"SELECT mi_person_key, event_year FROM {from_clause} "
            f"WHERE event_year IN ({year_list}) AND mi_person_key IS NOT NULL"
        ).df()
        con.close()
    except Exception as e:
        _log("warning", "Could not load model events for bin transitions: %s", e)
        return None

    if df.empty:
        _log("warning", "No rows found for %s/%s in TRAIN_YEARS", cohort_name, age_band)
        return None

    # Coerce event_year to int
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
    df = df.dropna(subset=["event_year"]).copy()
    df["event_year"] = df["event_year"].astype(int)
    df = df[df["event_year"].isin(TRAIN_YEARS)]

    # Count events per patient per year → assign bin
    events_per_patient_year = (
        df.groupby(["mi_person_key", "event_year"]).size().rename("n_events").reset_index()
    )
    events_per_patient_year["n_event_bin"] = assign_n_event_bins(
        events_per_patient_year.set_index("mi_person_key")["n_events"],
        thresholds,
    ).values
    _log("info", "Assigned bins for %d patient-year rows", len(events_per_patient_year))

    # Per-year distribution
    per_year_dist: Dict[str, Dict[str, int]] = {}
    for year in TRAIN_YEARS:
        yr_df = events_per_patient_year[events_per_patient_year["event_year"] == year]
        per_year_dist[str(year)] = {b: int((yr_df["n_event_bin"] == b).sum()) for b in DENSITY_BINS}

    # Build transition matrix from consecutive year pairs
    transition_matrix: Dict[str, Dict[str, int]] = {b: {b2: 0 for b2 in DENSITY_BINS} for b in DENSITY_BINS}
    transitions_list: List[Dict] = []

    pivot = events_per_patient_year.pivot(index="mi_person_key", columns="event_year", values="n_event_bin")
    years_present = sorted([y for y in TRAIN_YEARS if y in pivot.columns])

    for i in range(len(years_present) - 1):
        yr_from, yr_to = years_present[i], years_present[i + 1]
        pair = pivot[[yr_from, yr_to]].dropna()
        for _, row in pair.iterrows():
            b_from, b_to = str(row[yr_from]), str(row[yr_to])
            if b_from in transition_matrix and b_to in transition_matrix[b_from]:
                transition_matrix[b_from][b_to] += 1
            transitions_list.append({"from_bin": b_from, "to_bin": b_to,
                                      "from_year": yr_from, "to_year": yr_to})

    # Escalation / de-escalation / stable rates
    total_transitions = len(transitions_list)
    n_escalate = sum(1 for t in transitions_list if BIN_ORDER.get(t["to_bin"], 0) > BIN_ORDER.get(t["from_bin"], 0))
    n_deescalate = sum(1 for t in transitions_list if BIN_ORDER.get(t["to_bin"], 0) < BIN_ORDER.get(t["from_bin"], 0))
    n_stable = total_transitions - n_escalate - n_deescalate

    escalation_rate = round(n_escalate / total_transitions, 4) if total_transitions > 0 else 0.0
    de_escalation_rate = round(n_deescalate / total_transitions, 4) if total_transitions > 0 else 0.0
    stable_rate = round(n_stable / total_transitions, 4) if total_transitions > 0 else 0.0

    # Sankey nodes and links
    sankey_nodes = [{"id": f"{b}_{yr}", "label": f"{b} ({yr})", "bin": b, "year": yr}
                    for yr in years_present for b in DENSITY_BINS
                    if per_year_dist.get(str(yr), {}).get(b, 0) > 0]
    node_id_map = {n["id"]: idx for idx, n in enumerate(sankey_nodes)}

    sankey_links: List[Dict] = []
    for i in range(len(years_present) - 1):
        yr_from, yr_to = years_present[i], years_present[i + 1]
        for b_from in DENSITY_BINS:
            for b_to in DENSITY_BINS:
                count = transition_matrix[b_from].get(b_to, 0)
                if count == 0:
                    continue
                src_id = f"{b_from}_{yr_from}"
                tgt_id = f"{b_to}_{yr_to}"
                if src_id in node_id_map and tgt_id in node_id_map:
                    sankey_links.append({
                        "source": node_id_map[src_id],
                        "target": node_id_map[tgt_id],
                        "value": count,
                        "from_bin": b_from,
                        "to_bin": b_to,
                    })

    result = {
        "cohort": cohort_name,
        "age_band": age_band,
        "transition_matrix": transition_matrix,
        "sankey_nodes": sankey_nodes,
        "sankey_links": sankey_links,
        "escalation_rate": escalation_rate,
        "de_escalation_rate": de_escalation_rate,
        "stable_rate": stable_rate,
        "total_transitions": total_transitions,
        "n_patients_tracked": int(pivot.shape[0]),
        "years_present": years_present,
        "per_year_distribution": per_year_dist,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log("info", "Bin transitions written: %s (patients=%d, transitions=%d, escalation=%.1f%%)",
         out_path, result["n_patients_tracked"], total_transitions, escalation_rate * 100)

    # S3 upload
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() not in ("1", "true", "yes"):
        try:
            import boto3 as _boto3
            s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
            dash_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
            s3_key = f"{dash_prefix.rstrip('/')}/visualizations/dtw/{cohort_name}/{age_band}/density/transitions/bin_transitions.json"
            _boto3.client("s3").put_object(
                Bucket=s3_bucket, Key=s3_key,
                Body=out_path.read_bytes(), ContentType="application/json",
            )
            _log("info", "Bin transitions uploaded: s3://%s/%s", s3_bucket, s3_key)
        except Exception as e:
            _log("warning", "Bin transitions S3 upload failed: %s", e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute event-density bin transitions across years")
    parser.add_argument("--cohort", "--cohort-name", dest="cohort", required=True)
    parser.add_argument("--age-band", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    logger = setup_pipeline_logger(
        step_name="9_dtw",
        cohort=args.cohort,
        age_band=args.age_band,
        script_name="create_bin_transitions",
    )

    result = compute_bin_transitions(
        project_root=project_root,
        cohort_name=args.cohort,
        age_band=args.age_band,
        force=args.force,
        logger=logger.logger,
    )

    if result is None:
        logger.warning("Bin transitions not produced for %s/%s", args.cohort, args.age_band)
        sys.exit(1)

    logger.info(
        "Done: %d patients tracked, %d transitions, escalation=%.1f%%, de-escalation=%.1f%%, stable=%.1f%%",
        result["n_patients_tracked"],
        result["total_transitions"],
        result["escalation_rate"] * 100,
        result["de_escalation_rate"] * 100,
        result["stable_rate"] * 100,
    )
    logger.log_summary()
    sys.exit(0)


if __name__ == "__main__":
    main()
