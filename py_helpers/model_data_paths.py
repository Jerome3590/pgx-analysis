"""
Resolve model_events.parquet path the same way BupaR R scripts do.

BupaR (create_bupar_outputs_*.R) **disabled** the Step 3b input_model_data path because
that parquet often lacks Step 4 columns (e.g. first_f1120_date / first_o11_p_date and
full ICD/CPT). DTW must follow the same rule or opioid_ed trajectories stay at 0 rows
while non_opioid_ed works (when 3b exists for opioid but not polypharmacy on disk).

Resolution order (matches BupaR):
1. Step 4: /mnt/nvme/4_model_data, PGX_DATA_ROOT/4_model_data, project 4_model_data
2. Step 3b: only if Step 4 missing AND parquet has a cohort target-date column

Where model_events are written (saved):
- Step 3b: 3b_feature_importance_eda/outputs/cohorts/input_model_data/cohort_name={slug}/...
- Step 4:  4_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet
  (or model_events_no_protocols.parquet). Built by 4_model_data/create_model_data.py.

This module resolves local paths only; S3 paths are not resolved here.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from py_helpers.constants import get_cohort_slug_by_cohort


def _target_date_column_candidates(cohort_name: str) -> Tuple[str, ...]:
    """Ordered target-date columns required for DTW/BupaR lookback (same as create_dtw_trajectories)."""
    base = cohort_name.replace("_extreme_density", "") if cohort_name.endswith("_extreme_density") else cohort_name
    if base == "opioid_ed":
        return ("first_f1120_date", "first_opioid_ed_date")
    if base == "non_opioid_ed":
        return ("first_o11_p_date", "first_ed_non_opioid_date", "first_opioid_ed_date")
    return ("event_date",)


def _parquet_column_names(path: Path) -> Optional[set]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return None


def _has_required_target_date(path: Path, cohort_name: str) -> bool:
    """Step 3b snapshots are invalid for DTW when Step 4 target-date columns are missing."""
    cols = _parquet_column_names(path)
    if not cols:
        return False
    if "target" not in cols:
        return False
    return any(c in cols for c in _target_date_column_candidates(cohort_name))


def _four_model_data_roots(project_root: Path) -> List[Path]:
    nvme_4 = Path("/mnt/nvme/4_model_data")
    data_root_env = os.environ.get("PGX_DATA_ROOT", "").strip()
    roots: List[Path] = [nvme_4]
    if data_root_env:
        roots.append(Path(data_root_env) / "4_model_data")
    roots.extend([project_root / "4_model_data", project_root / "4a_model_data"])
    return roots


def _model_events_in_hive_dir(base: Path, cohort_name: str, band: str) -> Optional[Path]:
    d = base / f"cohort_name={cohort_name}" / f"age_band={band}"
    for name in ("model_events_no_protocols.parquet", "model_events.parquet"):
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def confirm_paths_exist_with_listings(
    paths: List[Path],
    max_entries: int = 30,
) -> Tuple[bool, List[str]]:
    """
    Confirm each path exists (file, size > 0) and return listings of parent dir contents.
    Use before continuing so logs show "path exists with objects" before the script proceeds.
    Returns (all_exist, list of "path -> exists=True|False, size=N, parent contents: [...]").
    """
    result: List[str] = []
    all_ok = True
    for p in paths:
        path = Path(p)
        if not path.exists():
            result.append(f"{path} -> exists=False (missing)")
            all_ok = False
            continue
        if path.is_file():
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            if size <= 0:
                result.append(f"{path} -> exists=True size=0 (empty file)")
                all_ok = False
            else:
                result.append(f"{path} -> exists=True size={size}")
        else:
            result.append(f"{path} -> exists=True (directory)")
        parent = path.parent
        if parent.exists():
            try:
                entries = sorted(parent.iterdir())
                names = [e.name for e in entries[:max_entries]]
                if len(entries) > max_entries:
                    names.append(f"... and {len(entries) - max_entries} more")
                result.append(f"  parent contents: {names}")
            except OSError as e:
                result.append(f"  parent listdir error: {e}")
        else:
            result.append("  parent missing")
    return all_ok, result


def get_model_events_paths_checked(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> List[str]:
    """
    Return the ordered list of paths that resolve_model_events_path tries.
    Use when resolution fails so logs can record [ERROR_PARAMS] paths_checked
    for follow-on runs (e.g. fix path or create model_data).
    """
    project_root = Path(project_root).resolve()
    cohort_slug = get_cohort_slug_by_cohort(cohort_name)
    band_underscore = age_band.replace("-", "_") if "-" in age_band else age_band
    band_hyphen = age_band.replace("_", "-") if "_" in age_band else age_band
    bands_to_try = (band_underscore, band_hyphen) if band_underscore != band_hyphen else (age_band,)
    out: List[str] = []
    for root in _four_model_data_roots(project_root):
        for band in bands_to_try:
            for name in ("model_events_no_protocols.parquet", "model_events.parquet"):
                out.append(str(root / f"cohort_name={cohort_name}" / f"age_band={band}" / name))
    for band in bands_to_try:
        p = (
            project_root
            / "3b_feature_importance_eda"
            / "outputs"
            / "cohorts"
            / "input_model_data"
            / f"cohort_name={cohort_slug}"
            / f"age_band={band}"
            / "model_events.parquet"
        )
        out.append(str(p))
    return out


def get_path_check_listings(paths: List[str], max_entries: int = 30) -> List[str]:
    """
    For logging diagnostics: for each path (file path we checked), list the parent
    directory contents so logs show what actually exists at each location.
    Returns one string per path, e.g. "path -> parent contents: [a, b]" or "path -> parent missing".
    """
    result: List[str] = []
    for p in paths:
        path = Path(p)
        parent = path.parent
        if not parent.exists():
            result.append(f"{p} -> parent missing")
        else:
            try:
                entries = sorted(parent.iterdir())
                names = [e.name for e in entries[:max_entries]]
                if len(entries) > max_entries:
                    names.append(f"... and {len(entries) - max_entries} more")
                result.append(f"{p} -> parent contents: {names}")
            except OSError as e:
                result.append(f"{p} -> listdir error: {e}")
    return result


def resolve_model_events_paths(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> List[Path]:
    """
    Resolve model_events path(s). For 85-114, returns one path if partition 85-114 exists,
    or two paths [85-94, 95-114] when only sub-partitions exist (caller should UNION).
    Otherwise returns the same as resolve_model_events_path as a single-element list.
    """
    single = resolve_model_events_path(project_root, cohort_name, age_band)
    if single is not None:
        return [single]
    if age_band != "85-114":
        return []

    # 85-114: try union of 85-94 and 95-114 (same roots and naming as 4_model_data).
    import os
    nvme_4 = Path("/mnt/nvme/4_model_data")
    data_root_env = os.environ.get("PGX_DATA_ROOT", "").strip()
    candidates_4 = [nvme_4]
    if data_root_env:
        candidates_4.append(Path(data_root_env) / "4_model_data")
    candidates_4.extend([
        project_root / "4_model_data",
        project_root / "4a_model_data",
    ])

    def _file_in_dir(base: Path, band: str) -> Optional[Path]:
        d = base / f"cohort_name={cohort_name}" / f"age_band={band}"
        for name in ("model_events_no_protocols.parquet", "model_events.parquet"):
            p = d / name
            if p.exists():
                return p
        return None

    for root in candidates_4:
        if not root.exists():
            continue
        # Try 85-94 then 95-114 (hyphen and underscore).
        for b94 in ("85-94", "85_94"):
            for b114 in ("95-114", "95_114"):
                p94 = _file_in_dir(root, b94)
                p114 = _file_in_dir(root, b114)
                if p94 is not None and p114 is not None:
                    return [p94, p114]
    return []


def resolve_model_events_path(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> Optional[Path]:
    """
    Resolve model_events path: Step 4 (4_model_data) first, then 3b only if it has target-date columns.

    Matches create_bupar_outputs_opioid_ed.R / create_bupar_outputs_non_opioid_ed.R (3b path disabled there).

    Returns the first valid path, or None if none found.
    """
    project_root = Path(project_root).resolve()
    cohort_slug = get_cohort_slug_by_cohort(cohort_name)

    # EC2 uses underscore in partition names (age_band=75_84). Try underscore first, then hyphen.
    band_underscore = age_band.replace("-", "_") if "-" in age_band else age_band
    band_hyphen = age_band.replace("_", "-") if "_" in age_band else age_band
    bands_to_try = (band_underscore, band_hyphen) if band_underscore != band_hyphen else (age_band,)

    # 1) Step 4 model_data (canonical; same as BupaR R)
    for root in _four_model_data_roots(project_root):
        if not root.exists():
            continue
        for band in bands_to_try:
            p = _model_events_in_hive_dir(root, cohort_name, band)
            if p is not None:
                return p

    # 2) Legacy 3b only when Step 4 absent AND schema has target-date column
    for band in bands_to_try:
        path_3b = (
            project_root
            / "3b_feature_importance_eda"
            / "outputs"
            / "cohorts"
            / "input_model_data"
            / f"cohort_name={cohort_slug}"
            / f"age_band={band}"
            / "model_events.parquet"
        )
        if path_3b.exists() and path_3b.stat().st_size > 0 and _has_required_target_date(path_3b, cohort_name):
            return path_3b

    return None
