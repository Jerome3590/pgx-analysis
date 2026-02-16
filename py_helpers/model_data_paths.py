"""
Resolve model_events.parquet path the same way BupaR does.

Used by DTW and any other step that reads model_events, so we prefer 3b output
then 4_model_data (with same candidate roots and model_events_no_protocols preference).

Where model_events are written (saved):
- Step 3b: 3b_feature_importance_eda/outputs/cohorts/input_model_data/cohort_name={slug}/age_band={band}/model_events.parquet
  (slug = opioid | polypharmacy). Often synced to S3 gold/cohorts/input_model_data/...
- Step 4:  4_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet
  (or model_events_no_protocols.parquet). Built by 4_model_data/create_model_data.py.

This module resolves local paths only; S3 paths are not resolved here.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from py_helpers.constants import get_cohort_slug_by_cohort


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
    # 3b paths
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
    # 4_model_data roots (same order as resolve_model_events_path)
    nvme_4 = Path("/mnt/nvme/4_model_data")
    data_root_env = os.environ.get("PGX_DATA_ROOT", "").strip()
    candidates_4 = [nvme_4]
    if data_root_env:
        candidates_4.append(Path(data_root_env) / "4_model_data")
    candidates_4.extend([
        project_root / "4_model_data",
        project_root / "4a_model_data",
    ])
    for root in candidates_4:
        for band in bands_to_try:
            for name in ("model_events_no_protocols.parquet", "model_events.parquet"):
                out.append(str(root / f"cohort_name={cohort_name}" / f"age_band={band}" / name))
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
    Resolve model_events path: try 3b first, then 4_model_data (same logic as BupaR R scripts).

    - 3b: project_root/3b_feature_importance_eda/outputs/cohorts/input_model_data/cohort_name={slug}/age_band={age_band}/model_events.parquet
      where slug = "opioid" for opioid_ed, "polypharmacy" for non_opioid_ed.
    - 4_model_data: under PGX_DATA_ROOT/4_model_data, /mnt/nvme/4_model_data, or project_root/4_model_data;
      prefer model_events_no_protocols.parquet then model_events.parquet.

    Returns the first path that exists, or None if none found.
    """
    project_root = Path(project_root).resolve()
    cohort_slug = get_cohort_slug_by_cohort(cohort_name)

    # EC2 uses underscore in partition names (age_band=75_84). Try underscore first, then hyphen.
    band_underscore = age_band.replace("-", "_") if "-" in age_band else age_band
    band_hyphen = age_band.replace("_", "-") if "_" in age_band else age_band
    bands_to_try = (band_underscore, band_hyphen) if band_underscore != band_hyphen else (age_band,)

    # 1) Try 3b (same as BupaR)
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
        if path_3b.exists():
            return path_3b

    # 2) Fallback: 4_model_data. On EC2 data is on NVMe; try /mnt/nvme first, then PGX_DATA_ROOT, then project.
    nvme_4 = Path("/mnt/nvme/4_model_data")
    data_root_env = os.environ.get("PGX_DATA_ROOT", "").strip()
    candidates_4 = [nvme_4]
    if data_root_env:
        candidates_4.append(Path(data_root_env) / "4_model_data")
    candidates_4.extend([
        project_root / "4_model_data",
        project_root / "4a_model_data",
    ])
    def _check_dir(base: Path, band: str) -> Optional[Path]:
        d = base / f"cohort_name={cohort_name}" / f"age_band={band}"
        for name in ("model_events_no_protocols.parquet", "model_events.parquet"):
            p = d / name
            if p.exists():
                return p
        return None

    for root in candidates_4:
        if not root.exists():
            continue
        for band in bands_to_try:
            p = _check_dir(root, band)
            if p is not None:
                return p

    return None
