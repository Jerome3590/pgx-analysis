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

from pathlib import Path
from typing import Optional

from py_helpers.constants import get_cohort_slug_by_cohort


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
