#!/usr/bin/env python3
"""
Check pipeline status via S3 objects (pgx-repository checkpoints and optional pgxdatalake outputs).

Reads:
- s3://pgx-repository/pipeline_checkpoints/{step}/{cohort}/{age_band}/checkpoint.json
  (steps: 4_model_data, 5_pgx_analysis, 6_final_model, 9_dashboard_metadata, 9_dashboard_visuals, etc.)
- Optionally: pgxdatalake gold/cohorts_model_data, gold/final_model (object counts per cohort/age_band)

Usage:
    python utility_scripts/check_pipeline_status_s3.py [--outputs] [--profile NAME]
    --outputs: also list pgxdatalake model data and final_model object counts
    --profile: AWS CLI profile (default: AWS_PROFILE or default)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import boto3
except ImportError:
    print("boto3 required: pip install boto3")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.constants import REQUIRED_COHORTS  # noqa: E402

_creds = REPO_ROOT.parent / "credentials"
if _creds.exists() and not os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(_creds)

REPO_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgx-repository")
DATALAKE_BUCKET = os.environ.get("PGX_DATALAKE_BUCKET", "pgxdatalake")
PIPELINE_CHECKPOINTS_PREFIX = "pipeline_checkpoints"


def _s3_list(s3_client, bucket: str, prefix: str, max_keys: int = 2000):
    """List object keys under prefix; return list of {Key, LastModified, Size}."""
    out = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append({
                "Key": obj["Key"],
                "LastModified": obj.get("LastModified"),
                "Size": obj.get("Size", 0),
            })
            if len(out) >= max_keys:
                return out
    return out


def run(profile: str | None, show_outputs: bool) -> None:
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    print("Pipeline status (S3 checkpoints and outputs)")
    print("=" * 70)
    print(f"Checkpoints bucket: s3://{REPO_BUCKET}/{PIPELINE_CHECKPOINTS_PREFIX}/")
    if show_outputs:
        print(f"Outputs bucket:     s3://{DATALAKE_BUCKET}/ (gold/cohorts_model_data, gold/final_model)")
    print()

    # ----- 1. Pipeline checkpoints (pgx-repository) -----
    print("1. Pipeline checkpoints (s3://{}/pipeline_checkpoints/)".format(REPO_BUCKET))
    print("-" * 70)
    all_objs = _s3_list(s3, REPO_BUCKET, PIPELINE_CHECKPOINTS_PREFIX + "/", max_keys=2000)
    checkpoint_files = [o for o in all_objs if o["Key"].endswith("checkpoint.json")]

    if not checkpoint_files:
        print("  No pipeline checkpoint files found.")
    else:
        # Parse: pipeline_checkpoints/{step}/{cohort}/{age_band}/checkpoint.json
        by_step: dict[str, list[tuple[str, str, datetime | None]]] = defaultdict(list)
        for o in checkpoint_files:
            rest = o["Key"][len(PIPELINE_CHECKPOINTS_PREFIX) + 1 : -len("/checkpoint.json")]
            parts = rest.split("/")
            step = parts[0] if len(parts) >= 1 else "?"
            cohort = parts[1] if len(parts) >= 2 else "?"
            age_band = (parts[2].replace("_", "-") if len(parts) >= 3 else "?")
            by_step[step].append((cohort, age_band, o.get("LastModified")))

        for step_name, entries in sorted(by_step.items()):
            # Sort by cohort, age_band
            entries.sort(key=lambda e: (e[0], e[1]))
            print("  Step: {}  ({} checkpoint(s))".format(step_name, len(entries)))
            for cohort, age_band, last_mod in entries[:12]:
                mt_str = last_mod.strftime("%Y-%m-%d %H:%M UTC") if last_mod else "—"
                print("    {}  {} / {}  {}".format(mt_str, cohort, age_band, ""))
            if len(entries) > 12:
                print("    ... and {} more".format(len(entries) - 12))
            print()

    # Expected (cohort, age_band) count for reference
    expected = sum(len(bands) for bands in REQUIRED_COHORTS.values())
    print("  Expected cohort/age_band combinations (REQUIRED_COHORTS): {}".format(expected))
    print("    opioid_ed: {}, non_opioid_ed: {}".format(
        len(REQUIRED_COHORTS["opioid_ed"]), len(REQUIRED_COHORTS["non_opioid_ed"])))
    print()

    # ----- 2. Optional: pgxdatalake outputs -----
    if show_outputs:
        print("2. Model data and final model outputs (s3://{}/)".format(DATALAKE_BUCKET))
        print("-" * 70)
        for prefix, label in [
            ("gold/cohorts_model_data/", "Model data (Step 4)"),
            ("gold/final_model/", "Final model (Step 6)"),
        ]:
            objs = _s3_list(s3, DATALAKE_BUCKET, prefix, max_keys=500)
            if not objs:
                print("  {}: no objects under {}".format(label, prefix))
            else:
                # Count by cohort/age_band from key pattern
                by_combo = defaultdict(int)
                for o in objs:
                    key = o["Key"]
                    after = key[len(prefix):].strip("/")
                    parts = after.split("/")
                    if len(parts) >= 2:
                        combo = "{} / {}".format(parts[0].replace("cohort_name=", ""), parts[1].replace("age_band=", ""))
                        by_combo[combo] += 1
                print("  {} ({} total objects, {} cohort/age_band combo(s))".format(
                    label, len(objs), len(by_combo)))
                for combo in sorted(by_combo.keys())[:10]:
                    print("    {}  {} file(s)".format(combo, by_combo[combo]))
                if len(by_combo) > 10:
                    print("    ... and {} more".format(len(by_combo) - 10))
            print()
    else:
        print("2. Outputs: (use --outputs to list pgxdatalake model_data and final_model)")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check pipeline status via S3 objects")
    ap.add_argument("--outputs", action="store_true", help="Also list pgxdatalake model_data and final_model")
    ap.add_argument("--profile", default=None, help="AWS profile name")
    args = ap.parse_args()
    run(profile=args.profile, show_outputs=args.outputs)
