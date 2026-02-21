#!/usr/bin/env python3
"""
DTW alignment: compute distances to prototype trajectories and export common sequences (Step 2 of DTW workflow).

Reads the trajectory CSV produced by create_dtw_trajectories.py (dtw_features_{cohort}_{age_band}.csv),
encodes sequences as numeric series, selects prototype trajectories (evenly spaced by length),
computes DTW distance from each patient to each prototype using dtaidistance library, then:
- Augments the CSV with dtw_min_distance and dtw_distance_to_prototype_0..k
- Writes common_sequences.json with the prototype sequences (for dashboard/docs)

DTW alignment IS computed for dashboard analysis. Results used for visualization only (not model features).
Run after create_dtw_trajectories.py and before create_dtw_visuals.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from dtaidistance import dtw as dtw_lib
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from py_helpers.fe_monitor import step_block  # noqa: E402


def _get_logger(cohort_name: str, age_band: str) -> tuple[logging.Logger, Path]:
    """Create a logger with both console and file handlers (same pattern as BupaR/FP-Growth)."""
    logs_dir = PROJECT_ROOT / "logs" / "dtw"
    logs_dir.mkdir(parents=True, exist_ok=True)
    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"dtw_{cohort_name}_{age_band_fname}.log"
    logger = logging.getLogger(f"dtw.{cohort_name}.{age_band_fname}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger, log_path


def _dtw_output_root(project_root: Path) -> Path:
    """Dashboard visualization outputs (step 10); creation code in 9_dashboard_visuals/dtw."""
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


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

    # Compute distances for every patient in parallel (maximize CPU utilization)
    distance_rows = []
    max_workers = min(len(encoded), 32)  # Cap at 32 to avoid overwhelming the system
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    fe_dir = _dtw_output_root(project_root) / "outputs" / "feature_engineering"
    csv_path = fe_dir / f"dtw_features_{cohort_name}_{age_band_fname}.csv"
    if not csv_path.exists():
        log("warning", "DTW features CSV not found: %s; run create_dtw_trajectories.py first.", csv_path)
        return False
    df = pd.read_csv(csv_path)
    if df.empty or "seq_pattern_str" not in df.columns:
        log("warning", "CSV empty or missing seq_pattern_str; skipping alignment.")
        return False
    if not DTW_AVAILABLE:
        if logger:
            logger.error("dtaidistance is required for DTW alignment. Install with: pip install dtaidistance")
        else:
            print("[ERROR] dtaidistance is required for DTW alignment. Install with: pip install dtaidistance")
        sys.exit(1)

    log("info", "DTW alignment: %d patients, n_prototypes=%d", len(df), n_prototypes)
    df_out, common_sequences = compute_dtw_distances(df, n_prototypes=n_prototypes)
    if common_sequences is None:
        log("warning", "No alignment computed (no encoded trajectories or prototypes).")
        return False

    df_out.to_csv(csv_path, index=False)
    log("info", "Wrote augmented CSV to %s", csv_path)

    # Write common_sequences.json next to the CSV (same directory for upload)
    common_path = fe_dir / f"common_sequences_{cohort_name}_{age_band_fname}.json"
    with open(common_path, "w", encoding="utf-8") as f:
        json.dump(common_sequences, f, indent=2)
    log("info", "Wrote common sequences to %s", common_path)

    # Also update dtw_added_features copy if present
    added_path = fe_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    if added_path.exists():
        df_out.to_csv(added_path, index=False)
        log("info", "Updated %s", added_path)
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
    logger, _ = _get_logger(args.cohort, args.age_band)

    age_band_fname = args.age_band.replace("-", "_")
    csv_path = _dtw_output_root(project_root) / "outputs" / "feature_engineering" / f"dtw_features_{args.cohort}_{age_band_fname}.csv"
    if not csv_path.exists():
        logger.error("Not found: %s. Run create_dtw_trajectories.py first.", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if "dtw_min_distance" in df.columns and df["dtw_min_distance"].notna().any() and not args.force:
        logger.info("CSV already has DTW distances; skipping (use --force to re-run).")
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
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
