#!/usr/bin/env python3
"""
DTW alignment: compute distances to prototype trajectories and export common sequences.

Reads the trajectory CSV produced by create_dtw_trajectories.py (dtw_features_{cohort}_{age_band}.csv),
encodes sequences as numeric series, selects prototype trajectories (evenly spaced by length),
computes DTW distance from each patient to each prototype using dtaidistance, then:
- Augments the CSV with dtw_min_distance and dtw_distance_to_prototype_0..k
- Writes common_sequences.json with the prototype sequences (for dashboard/docs)

Run after create_dtw_trajectories.py and before create_dtw_visuals.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from dtaidistance import dtw as dtw_lib
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    # Compute distances for every patient
    distance_rows = []
    for pid, encoded_traj in encoded.items():
        row = _compute_dtw_for_patient(
            pid, encoded_traj, prototype_trajectories, prototype_order
        )
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
) -> bool:
    """
    Read trajectory CSV, run DTW alignment, write augmented CSV and common_sequences.json.
    Returns True if alignment was run (and dtw_min_distance filled), False if skipped or failed.
    """
    age_band_fname = age_band.replace("-", "_")
    fe_dir = _dtw_output_root(project_root) / "outputs" / "feature_engineering"
    csv_path = fe_dir / f"dtw_features_{cohort_name}_{age_band_fname}.csv"
    if not csv_path.exists():
        print(f"[WARN] DTW features CSV not found: {csv_path}; run create_dtw_trajectories.py first.")
        return False
    df = pd.read_csv(csv_path)
    if df.empty or "seq_pattern_str" not in df.columns:
        print("[WARN] CSV empty or missing seq_pattern_str; skipping alignment.")
        return False
    if not DTW_AVAILABLE:
        print("[ERROR] dtaidistance is required for DTW alignment. Install with: pip install dtaidistance")
        sys.exit(1)

    print(f"[INFO] DTW alignment: {len(df)} patients, n_prototypes={n_prototypes}")
    df_out, common_sequences = compute_dtw_distances(df, n_prototypes=n_prototypes)
    if common_sequences is None:
        print("[WARN] No alignment computed (no encoded trajectories or prototypes).")
        return False

    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote augmented CSV to {csv_path}")

    # Write common_sequences.json next to the CSV (same directory for upload)
    common_path = fe_dir / f"common_sequences_{cohort_name}_{age_band_fname}.json"
    with open(common_path, "w", encoding="utf-8") as f:
        json.dump(common_sequences, f, indent=2)
    print(f"[INFO] Wrote common sequences to {common_path}")

    # Also update dtw_added_features copy if present
    added_path = fe_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    if added_path.exists():
        df_out.to_csv(added_path, index=False)
        print(f"[INFO] Updated {added_path}")
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

    age_band_fname = args.age_band.replace("-", "_")
    csv_path = _dtw_output_root(project_root) / "outputs" / "feature_engineering" / f"dtw_features_{args.cohort}_{age_band_fname}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}. Run create_dtw_trajectories.py first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if "dtw_min_distance" in df.columns and df["dtw_min_distance"].notna().any() and not args.force:
        print("[INFO] CSV already has DTW distances; skipping (use --force to re-run).")
        sys.exit(0)

    ok = run_alignment(
        project_root=project_root,
        cohort_name=args.cohort,
        age_band=args.age_band,
        n_prototypes=args.n_prototypes,
        force=args.force,
    )
    if not ok:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
