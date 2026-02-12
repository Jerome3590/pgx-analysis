#!/usr/bin/env python3
"""
Sync from S3 the model data and feature importance used by dashboard visuals (BupaR, DTW, FP-Growth).

Enables running [4_dashboard_visuals](4_dashboard_visuals.ipynb) or run_dashboard_visuals.py locally
without having run Steps 2–4 on this machine. Uses aws s3 sync (idempotent).

Syncs:
- s3://pgxdatalake/gold/cohorts_model_data/ -> get_model_data_root() (4_model_data)
  (model_events.parquet per cohort/age_band for DTW, BupaR, FP-Growth)
- s3://pgxdatalake/gold/feature_importance/ -> get_data_root()/gold/feature_importance
  (SHAP/FFA allowed codes; cohort_feature_importance.csv, etc.)

Usage:
  python 10_dashboard_visuals/sync_visualization_data_from_s3.py
  python 10_dashboard_visuals/sync_visualization_data_from_s3.py --profile my-aws-profile
  python 10_dashboard_visuals/sync_visualization_data_from_s3.py --model-data-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repo root for py_helpers (must be before py_helpers imports)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.env_utils import get_data_root, get_model_data_root  # noqa: E402
from py_helpers.workflow_sync_checkpoint import sync_s3_to_local  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

S3_BUCKET = "pgxdatalake"
MODEL_DATA_S3_PREFIX = f"s3://{S3_BUCKET}/gold/cohorts_model_data/"
FEATURE_IMPORTANCE_S3_PREFIX = f"s3://{S3_BUCKET}/gold/feature_importance/"


def run_sync(
    *,
    profile: str | None = None,
    model_data_only: bool = False,
    feature_importance_only: bool = False,
) -> bool:
    data_root = get_data_root()
    model_data_root = get_model_data_root()

    ok = True

    if not feature_importance_only:
        logger.info("Syncing model data (model_events) for dashboard visuals: %s -> %s", MODEL_DATA_S3_PREFIX, model_data_root)
        if not sync_s3_to_local(MODEL_DATA_S3_PREFIX, model_data_root, profile=profile):
            logger.warning("Model data sync failed")
            ok = False
        else:
            logger.info("Model data sync OK")

    if not model_data_only:
        fi_local = data_root / "gold" / "feature_importance"
        logger.info("Syncing feature importance (SHAP/FFA): %s -> %s", FEATURE_IMPORTANCE_S3_PREFIX, fi_local)
        if not sync_s3_to_local(FEATURE_IMPORTANCE_S3_PREFIX, fi_local, profile=profile):
            logger.warning("Feature importance sync failed")
            ok = False
        else:
            logger.info("Feature importance sync OK")

    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Sync from S3 the model data and feature importance used by dashboard visuals (BupaR, DTW, FP-Growth)."
    )
    ap.add_argument("--profile", default=None, help="AWS CLI profile (e.g. for local dev)")
    ap.add_argument("--model-data-only", action="store_true", help="Only sync gold/cohorts_model_data -> 4_model_data")
    ap.add_argument("--feature-importance-only", action="store_true", help="Only sync gold/feature_importance")
    args = ap.parse_args()

    if args.model_data_only and args.feature_importance_only:
        logger.error("Use only one of --model-data-only and --feature-importance-only")
        sys.exit(2)

    success = run_sync(
        profile=args.profile,
        model_data_only=args.model_data_only,
        feature_importance_only=args.feature_importance_only,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
