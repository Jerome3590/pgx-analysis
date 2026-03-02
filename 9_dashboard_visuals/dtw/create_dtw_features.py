#!/usr/bin/env python3
"""
DTW alignment: compute distances to prototype trajectories and export common sequences (Step 2 of DTW workflow).

Reads the trajectory CSV produced by create_dtw_trajectories.py (dtw_features_{cohort}_{age_band}.csv),
encodes sequences as numeric series, selects prototype trajectories (evenly spaced by length),
computes DTW distance from each patient to each prototype using dtaidistance library, then:
- Augments the CSV with dtw_min_distance and dtw_distance_to_prototype_0..k
- Writes common_sequences.json with the prototype sequences (for dashboard/docs)

When event_density_bin is present, alignment is run per density bin (low/medium/high/extreme) and outputs
are written as sub-cohorts: one CSV and one common_sequences JSON per bin (no merge). Each patient is
compared only to prototypes from the same bin, which speeds up dtaidistance. Downstream (create_dtw_visuals)
loads by filter (or concatenates per-bin CSVs when building chart_data).
DTW alignment IS computed for dashboard analysis. Results used for visualization only (not model features).
Run after create_dtw_trajectories.py and before create_dtw_visuals.py.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from dtaidistance import dtw as dtw_lib
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

# Same pattern as BupaR/FP-Growth: use setup_pipeline_logger (repo root from py_helpers → project-level logs)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from py_helpers.fe_monitor import step_block  # noqa: E402
from py_helpers.pipeline_logger import setup_pipeline_logger  # noqa: E402


def _dtw_output_root(project_root: Path) -> Path:
    """Dashboard visualization outputs (step 10); creation code in 9_dashboard_visuals/dtw."""
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


def _read_table_parquet_or_csv(path_parquet: Path, path_csv: Path):
    """Load DataFrame from parquet if present, else CSV. Prefer Parquet + DuckDB when available."""
    if path_parquet.exists():
        if duckdb is not None:
            con = duckdb.connect(":memory:")
            try:
                return con.execute("SELECT * FROM read_parquet(?)", [str(path_parquet)]).df()
            finally:
                con.close()
        return pd.read_parquet(path_parquet)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    return None

# Bins must match create_dtw_trajectories.DENSITY_BINS for per-bin alignment
DENSITY_BINS = ("low", "medium", "high", "extreme")

_SKIP_TOKENS = frozenset({"nan", "none", "null", ""})


def _seq_to_tokens(seq_pattern_str: str) -> List[str]:
    """Parse seq_pattern_str into list of activity tokens (DRUG:X, ICD:Y, CPT:Z)."""
    if not seq_pattern_str or (isinstance(seq_pattern_str, float) and pd.isna(seq_pattern_str)):
        return []
    s = str(seq_pattern_str).strip()
    return [t.strip() for t in s.split("_") if t.strip() and t.strip().lower() not in _SKIP_TOKENS]


def _encode_trajectories(df: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[int, str], Dict[str, List[str]]]:
    """
    Build encoded trajectories (symbol -> int) and inverse map.
    Returns: (patient_id -> list of ints), (int -> symbol), (patient_id -> list of symbols)
    """
    all_items: set = set()
    raw_trajectories: Dict[str, List[str]] = {}
    if "mi_person_key" not in df.columns or "seq_pattern_str" not in df.columns:
        return {}, {}, {}
    for _, row in df.iterrows():
        pid = str(row["mi_person_key"])
        tokens = _seq_to_tokens(row.get("seq_pattern_str", ""))
        if not tokens:
            continue
        all_items.update(tokens)
        raw_trajectories[pid] = tokens
    unique_items = sorted(all_items)
    global_encoding = {item: idx for idx, item in enumerate(unique_items)}
    inv_encoding = {idx: item for item, idx in global_encoding.items()}
    encoded = {
        pid: [global_encoding[t] for t in traj]
        for pid, traj in raw_trajectories.items()
    }
    return encoded, inv_encoding, raw_trajectories


def _select_prototypes(
    encoded_trajectories: Dict[str, List[int]],
    n_prototypes: int,
) -> List[str]:
    """Select prototype patient IDs evenly spaced by trajectory length."""
    if not encoded_trajectories or n_prototypes <= 0:
        return []
    lengths = [(pid, len(traj)) for pid, traj in encoded_trajectories.items()]
    lengths.sort(key=lambda x: (x[1], x[0]))
    n_patients = len(lengths)
    if n_prototypes >= n_patients:
        return [x[0] for x in lengths]
    indices = [
        lengths[int(i * (n_patients - 1) / (n_prototypes - 1))][0]
        for i in range(n_prototypes)
    ]
    return indices


def _compute_dtw_for_patient(
    pid: str,
    encoded_traj: List[int],
    prototype_trajectories: Dict[str, List[int]],
    prototype_order: List[str],
) -> Optional[Dict[str, Any]]:
    """Compute DTW distance from one patient to each prototype."""
    if not encoded_traj or not DTW_AVAILABLE:
        return None
    s = np.array(encoded_traj, dtype=np.double)
    row: Dict[str, Any] = {"mi_person_key": pid}
    distances = []
    for proto_idx, proto_pid in enumerate(prototype_order):
        proto_traj = prototype_trajectories.get(proto_pid)
        if not proto_traj:
            row[f"dtw_distance_to_prototype_{proto_idx}"] = np.inf
            distances.append(np.inf)
            continue
        try:
            p = np.array(proto_traj, dtype=np.double)
            d = dtw_lib.distance(s, p)
            row[f"dtw_distance_to_prototype_{proto_idx}"] = float(d)
            distances.append(float(d))
        except Exception:
            row[f"dtw_distance_to_prototype_{proto_idx}"] = np.inf
            distances.append(np.inf)
    valid = [x for x in distances if np.isfinite(x)]
    row["dtw_min_distance"] = min(valid) if valid else np.inf
    row["dtw_max_distance"] = max(valid) if valid else np.inf
    row["dtw_mean_distance"] = float(np.mean(valid)) if valid else np.inf
    row["dtw_std_distance"] = float(np.std(valid)) if len(valid) > 1 else 0.0
    return row


def compute_dtw_distances(
    df: pd.DataFrame,
    n_prototypes: int = 5,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Compute DTW distances to prototype trajectories and common sequences metadata.

    Returns:
        Augmented DataFrame (with dtw_min_distance and dtw_distance_to_prototype_*),
        common_sequences dict (prototype_index -> seq_pattern_str list, prototype_patient_ids).
    """
    if not DTW_AVAILABLE:
        return df, None
    encoded, inv_encoding, raw_trajectories = _encode_trajectories(df)
    if not encoded:
        return df, None
    prototype_order = _select_prototypes(encoded, n_prototypes)
    if not prototype_order:
        return df, None
    prototype_trajectories = {pid: encoded[pid] for pid in prototype_order if pid in encoded}
    if not prototype_trajectories:
        return df, None

    # Compute distances for every patient in parallel (CPU-bound: use processes, not threads)
    distance_rows = []
    n_cpus = os.cpu_count() or 4
    max_workers = min(len(encoded), max(1, n_cpus), 32)  # Cap at 32 to avoid overwhelming the system

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _compute_dtw_for_patient,
                pid, encoded_traj, prototype_trajectories, prototype_order
            ): pid
            for pid, encoded_traj in encoded.items()
        }
        
        for future in as_completed(futures):
            row = future.result()
            if row:
                distance_rows.append(row)
    
    if not distance_rows:
        return df, None
    dist_df = pd.DataFrame(distance_rows)
    
    # Ensure mi_person_key dtype matches original df to avoid merge errors
    if "mi_person_key" in df.columns and "mi_person_key" in dist_df.columns:
        dist_df["mi_person_key"] = dist_df["mi_person_key"].astype(df["mi_person_key"].dtype)

    # Merge back into original df (preserve all columns; add/overwrite DTW columns)
    merge_cols = [c for c in dist_df.columns if c != "mi_person_key"]
    df = df.drop(columns=[c for c in merge_cols if c in df.columns], errors="ignore")
    df = df.merge(dist_df, on="mi_person_key", how="left")

    # Build common_sequences for export
    common_sequences = {
        "n_prototypes": len(prototype_order),
        "prototype_patient_ids": prototype_order,
        "prototype_sequences": [
            raw_trajectories.get(pid, [])
            for pid in prototype_order
        ],
        "description": "Prototype trajectories (evenly spaced by length); each patient's DTW distance to these captures alignment to common sequences.",
    }
    return df, common_sequences


def run_alignment(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    n_prototypes: int = 5,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Read trajectory CSV, run DTW alignment, write augmented CSV and common_sequences.json.
    Returns True if alignment was run (and dtw_min_distance filled), False if skipped or failed.
    """
    def log(level: str, msg: str, *args: Any) -> None:
        if logger is not None:
            getattr(logger, level)(msg, *args)
        else:
            prefix = "[%s] " % level.upper()
            print(prefix + (msg % args if args else msg))

    age_band_fname = age_band.replace("-", "_")
    fe_dir = _dtw_output_root(project_root) / "feature_engineering"
    base_name = f"dtw_features_{cohort_name}_{age_band_fname}"
    parquet_path = fe_dir / f"{base_name}.parquet"
    csv_path = fe_dir / f"{base_name}.csv"
    log("info", "DTW features output dir (EC2): %s", fe_dir)
    df = _read_table_parquet_or_csv(parquet_path, csv_path)
    if df is None:
        log("warning", "DTW features not found: %s or %s; run create_dtw_trajectories.py first.", parquet_path, csv_path)
        return False
    if df.empty or "seq_pattern_str" not in df.columns:
        log("warning", "CSV empty or missing seq_pattern_str; skipping alignment.")
        # Read trajectory_status for event/alignment counts; write empty common_sequences with message
        n_events_analyzed: Optional[int] = None
        status_path = fe_dir / f"trajectory_status_{cohort_name}_{age_band_fname}.json"
        if status_path.exists():
            try:
                with open(status_path, encoding="utf-8") as f:
                    status = json.load(f)
                n_events_analyzed = status.get("n_events_analyzed")
            except Exception:
                pass
        n_ev = n_events_analyzed if n_events_analyzed is not None else 0
        n_align = 0
        msg = f"Events analyzed: {n_ev}; alignments (trajectories) found: {n_align}. Alignment skipped (empty or invalid trajectories)."
        log("info", "Drug events analyzed: %s; alignments found: 0", n_ev)
        common_path = fe_dir / f"common_sequences_{cohort_name}_{age_band_fname}.json"
        empty_payload = {
            "message": msg,
            "n_events_analyzed": n_ev,
            "n_alignments_found": n_align,
            "prototypes": [],
        }
        with open(common_path, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f, indent=2)
        log("info", "Wrote empty common_sequences with message to %s", common_path)
        return False
    if not DTW_AVAILABLE:
        if logger:
            logger.error("dtaidistance is required for DTW alignment. Install with: pip install dtaidistance")
        else:
            print("[ERROR] dtaidistance is required for DTW alignment. Install with: pip install dtaidistance")
        sys.exit(1)

    # Per-bin (sub-cohort) alignment when event_density_bin present: write one CSV + common_sequences per bin; no merge
    if "event_density_bin" in df.columns and df["event_density_bin"].notna().any():
        bins_present = [b for b in DENSITY_BINS if (df["event_density_bin"] == b).any()]
        if bins_present:
            any_ok = False
            for bin_name in bins_present:
                df_bin = df.loc[df["event_density_bin"] == bin_name].copy()
                if len(df_bin) < 2:
                    log("info", "Skipping bin %s: too few trajectories (%d)", bin_name, len(df_bin))
                    continue
                log("info", "DTW alignment density=%s: %d patients, n_prototypes=%d", bin_name, len(df_bin), n_prototypes)
                df_out_bin, common_bin = compute_dtw_distances(df_bin, n_prototypes=n_prototypes)
                if common_bin is None:
                    continue
                bin_csv = fe_dir / f"dtw_features_{cohort_name}_{age_band_fname}_density_{bin_name}.csv"
                bin_common = fe_dir / f"common_sequences_{cohort_name}_{age_band_fname}_density_{bin_name}.json"
                bin_parquet = fe_dir / f"dtw_features_{cohort_name}_{age_band_fname}_density_{bin_name}.parquet"
                df_out_bin.to_parquet(bin_parquet, index=False)
                df_out_bin.to_csv(bin_csv, index=False)
                with open(bin_common, "w", encoding="utf-8") as f:
                    json.dump(common_bin, f, indent=2)
                log("info", "Wrote sub-cohort %s: %s, %s", bin_name, bin_csv.name, bin_common.name)
                any_ok = True
            if any_ok:
                return True
            log("warning", "No bin had enough trajectories for alignment.")

    # Global alignment (no event_density_bin or fallback)
    log("info", "DTW alignment: %d patients, n_prototypes=%d", len(df), n_prototypes)
    df_out, common_sequences = compute_dtw_distances(df, n_prototypes=n_prototypes)
    if common_sequences is None:
        log("warning", "No alignment computed (no encoded trajectories or prototypes).")
        common_path = fe_dir / f"common_sequences_{cohort_name}_{age_band_fname}.json"
        empty_payload = {
            "message": f"Events analyzed: {len(df)} trajectories; alignments found: 0. No encoded trajectories or prototypes.",
            "n_events_analyzed": len(df),
            "n_alignments_found": 0,
            "prototypes": [],
        }
        with open(common_path, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f, indent=2)
        log("info", "Wrote empty common_sequences to %s", common_path)
        return False

    df_out.to_parquet(parquet_path, index=False)
    df_out.to_csv(csv_path, index=False)
    log("info", "Wrote augmented parquet and CSV to %s / %s", parquet_path, csv_path)

    common_path = fe_dir / f"common_sequences_{cohort_name}_{age_band_fname}.json"
    with open(common_path, "w", encoding="utf-8") as f:
        json.dump(common_sequences, f, indent=2)
    log("info", "Wrote common sequences to %s", common_path)

    added_parquet = fe_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.parquet"
    added_path = fe_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    if added_path.exists() or added_parquet.exists():
        df_out.to_parquet(added_parquet, index=False)
        df_out.to_csv(added_path, index=False)
        log("info", "Updated %s and %s", added_parquet, added_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DTW alignment: distances to prototype trajectories and common sequences. Run after create_dtw_trajectories.py."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g. opioid_ed, non_opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g. 25-44)")
    parser.add_argument("--n-prototypes", type=int, default=5, help="Number of prototype trajectories (default: 5)")
    parser.add_argument("--force", action="store_true", help="Re-run even if CSV already has dtw_min_distance")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT, help="Project root")
    args = parser.parse_args()
    project_root = Path(args.project_root)
    pl = setup_pipeline_logger(
        step_name="5_dtw",
        cohort=args.cohort,
        age_band=args.age_band,
        script_name="create_dtw_features",
    )
    logger = pl.logger

    age_band_fname = args.age_band.replace("-", "_")
    fe_dir = _dtw_output_root(project_root) / "feature_engineering"
    parquet_path = fe_dir / f"dtw_features_{args.cohort}_{age_band_fname}.parquet"
    csv_path = fe_dir / f"dtw_features_{args.cohort}_{age_band_fname}.csv"
    df = _read_table_parquet_or_csv(parquet_path, csv_path)
    if df is None:
        logger.error("Not found: %s or %s. Run create_dtw_trajectories.py first.", parquet_path, csv_path)
        sys.exit(1)
    if not args.force:
        if "dtw_min_distance" in df.columns and df["dtw_min_distance"].notna().any():
            logger.info("Features already have DTW distances; skipping (use --force to re-run).")
            sys.exit(0)
        # Per-bin idempotency: if event_density_bin present and all density sub-cohorts exist, skip
        if "event_density_bin" in df.columns and df["event_density_bin"].notna().any():
            bins_present = [b for b in DENSITY_BINS if (df["event_density_bin"] == b).any()]
            if bins_present:
                base = f"dtw_features_{args.cohort}_{age_band_fname}"
                all_exist = all(
                    (fe_dir / f"{base}_density_{b}.parquet").exists() or (fe_dir / f"{base}_density_{b}.csv").exists()
                    for b in bins_present
                )
                if all_exist:
                    logger.info("Density sub-cohort outputs already exist; skipping (use --force to re-run).")
                    sys.exit(0)

    with step_block("5_dtw", "create_dtw_features", logger=logger):
        logger.info("Starting DTW alignment for %s / %s", args.cohort, args.age_band)
        ok = run_alignment(
            project_root=project_root,
            cohort_name=args.cohort,
            age_band=args.age_band,
            n_prototypes=args.n_prototypes,
            force=args.force,
            logger=logger,
        )
    if not ok:
        logger.warning(
            "DTW alignment skipped (empty or invalid trajectories); exiting 0 so pipeline continues. "
            "This is expected for cohort/age_band where only drug names exist (no ICD/CPT trajectories)."
        )
    pl.log_summary()
    sys.exit(0)


if __name__ == "__main__":
    main()
