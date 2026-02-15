"""
Load baseline aggregated feature importance summary.

Used by 2_feature_importance.ipynb to display a table of cohort/age_band with
row counts and sample features. Prefers local 3a outputs (and DATA_ROOT/S3 gold),
then falls back to pgx-repository (legacy) S3.
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from py_helpers.constants import REQUIRED_COHORTS

PGX_REPO_BUCKET = "pgx-repository"
PGX_REPO_FI_PREFIX = "pgx-analysis/3_feature_importance/outputs"


def _project_root() -> Path:
    """Project root (parent of 3a_feature_importance)."""
    return Path(__file__).resolve().parent.parent


def _age_band_to_fname(age_band: str) -> str:
    return age_band.replace("-", "_") if isinstance(age_band, str) else str(age_band)


def _load_aggregated_fi_local(cohort: str, age_band: str, project_root: Path) -> Optional[pd.DataFrame]:
    """Load one aggregated FI CSV from local 3a / DATA_ROOT / S3 (FileResolver). Returns None if not found."""
    try:
        from py_helpers.feature_importance_eda_utils import resolve_aggregated_fi_path
        path = resolve_aggregated_fi_path(cohort, age_band, project_root)
        if path is not None and path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def _load_aggregated_fi_from_pgx_repo(cohort: str, age_band_fname: str) -> Optional[pd.DataFrame]:
    """Load one aggregated FI CSV from pgx-repository (legacy). Returns None if not found or on error."""
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    s3_key = f"{PGX_REPO_FI_PREFIX}/{filename}"
    try:
        import boto3
        client = boto3.client("s3")
        obj = client.get_object(Bucket=PGX_REPO_BUCKET, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        return df
    except Exception:
        return None


def get_baseline_summary_df(project_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Build a summary DataFrame of aggregated feature importance for all cohort/age_band.

    For each (cohort, age_band) in REQUIRED_COHORTS, loads the aggregated FI CSV:
    1. From local 3a outputs / DATA_ROOT / S3 gold (resolve_aggregated_fi_path)
    2. Else from pgx-repository (legacy) S3

    Returns:
        DataFrame with columns: cohort, age_band, rows, unique_features, sample
    """
    root = project_root if project_root is not None else _project_root()
    rows: List[Tuple[str, str, int, int, str]] = []
    for cohort, age_bands in REQUIRED_COHORTS.items():
        for age_band in age_bands:
            age_band_fname = _age_band_to_fname(age_band)
            df = _load_aggregated_fi_local(cohort, age_band, root)
            if df is None:
                df = _load_aggregated_fi_from_pgx_repo(cohort, age_band_fname)
            if df is not None and "feature" in df.columns:
                n = len(df)
                features = df["feature"].astype(str).dropna().unique().tolist()
                sample = ", ".join(features[:3]) if features else ""
                rows.append((cohort, age_band, n, len(features), sample))
            else:
                rows.append((cohort, age_band, 0, 0, ""))
    return pd.DataFrame(
        rows,
        columns=["cohort", "age_band", "rows", "unique_features", "sample"],
    )
