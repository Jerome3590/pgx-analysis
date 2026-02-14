"""
Load baseline aggregated feature importance summary from pgx-repository (S3).

Used by 2_feature_importance.ipynb to display a table of cohort/age_band with
row counts and sample features from the historical baseline FI.
"""

import io
from typing import List, Optional, Tuple

import pandas as pd

from py_helpers.constants import REQUIRED_COHORTS

PGX_REPO_BUCKET = "pgx-repository"
PGX_REPO_FI_PREFIX = "pgx-analysis/3_feature_importance/outputs"


def _age_band_to_fname(age_band: str) -> str:
    return age_band.replace("-", "_") if isinstance(age_band, str) else str(age_band)


def _load_aggregated_fi_from_pgx_repo(cohort: str, age_band_fname: str) -> Optional[pd.DataFrame]:
    """Load one aggregated FI CSV from pgx-repository. Returns None if not found or on error."""
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


def get_baseline_summary_df() -> pd.DataFrame:
    """
    Build a summary DataFrame of baseline aggregated feature importance from pgx-repository.

    For each (cohort, age_band) in REQUIRED_COHORTS, tries to load the corresponding
    aggregated FI CSV from s3://pgx-repository/pgx-analysis/3_feature_importance/outputs/
    and reports rows, unique_features, and a small sample of feature names.

    Returns:
        DataFrame with columns: cohort, age_band, rows, unique_features, sample
    """
    rows: List[Tuple[str, str, int, int, str]] = []
    for cohort, age_bands in REQUIRED_COHORTS.items():
        for age_band in age_bands:
            age_band_fname = _age_band_to_fname(age_band)
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
