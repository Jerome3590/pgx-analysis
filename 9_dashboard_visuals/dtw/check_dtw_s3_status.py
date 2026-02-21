#!/usr/bin/env python3
"""
Check S3 checkpoints and S3 logs for DTW workflow status.

Reports:
- pipeline_checkpoints (pgx-repository): 9_dashboard_* steps used by 5_build_and_deploy
- 6_dtw_checkpoint (pgx-repository): per-cohort/age_band checkpoints from create_dtw_visuals
- *_log (pgx-repository): extreme_density_extract_log, extreme_density_summarize_log
- DTW outputs in pgxdatalake: gold/feature_engineering/6_dtw/{cohort}/{age_band}/
- Optional: dashboard bucket DTW prefix (if S3_DASHBOARD_BUCKET/PREFIX set)

Usage:
    python check_dtw_s3_status.py [--logs] [--outputs] [--profile NAME]
    --logs: also list log objects under *_log prefixes
    --outputs: also list DTW output objects in pgxdatalake
    --profile: AWS CLI profile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import boto3
except ImportError:
    print("boto3 required: pip install boto3")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.constants import REQUIRED_COHORTS

# Optional: use project credentials when present
_creds = REPO_ROOT.parent / "credentials"
if _creds.exists() and not os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(_creds)

REPO_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgx-repository")
DATALAKE_BUCKET = os.environ.get("PGX_DATALAKE_BUCKET", "pgxdatalake")
PIPELINE_CHECKPOINTS_PREFIX = "pipeline_checkpoints"
DTW_CHECKPOINT_PREFIX = "5_dtw_checkpoint"
DTW_LOG_PREFIXES = ("extreme_density_extract_log", "extreme_density_summarize_log")
DTW_OUTPUTS_PREFIX = "gold/feature_engineering/5_dtw"


def _s3_list(s3_client, bucket: str, prefix: str, max_keys: int = 500):
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


def _s3_get_json(s3_client, bucket: str, key: str):
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return None


def run(profile: str | None, show_logs: bool, show_outputs: bool) -> None:
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3")

    print("DTW workflow – S3 checkpoints and logs")
    print("=" * 60)
    print(f"Bucket (checkpoints/logs): {REPO_BUCKET}")
    print(f"Bucket (DTW outputs):     {DATALAKE_BUCKET}")
    print()

    # ----- 1. Pipeline checkpoints (pipeline_checkpoints/) -----
    print("1. Pipeline checkpoints (s3://{}/pipeline_checkpoints/)".format(REPO_BUCKET))
    print("-" * 60)
    pipeline_objs = _s3_list(s3, REPO_BUCKET, PIPELINE_CHECKPOINTS_PREFIX + "/", max_keys=200)
    checkpoint_files = [o for o in pipeline_objs if o["Key"].endswith("checkpoint.json")]
    if not checkpoint_files:
        print("  No pipeline checkpoint files found.")
        print("  (DTW does not write these; 5_build_and_deploy uses 9_dashboard_models, 9_dashboard_cpic.)")
    else:
        by_step: dict[str, list[dict]] = {}
        for o in checkpoint_files:
            # key: pipeline_checkpoints/{step}/{cohort}/{age_band}/checkpoint.json
            parts = o["Key"].replace(PIPELINE_CHECKPOINTS_PREFIX + "/", "").split("/")
            step = parts[0] if len(parts) >= 1 else "?"
            by_step.setdefault(step, []).append(o)
        for step, objs in sorted(by_step.items()):
            print("  Step: {} ({} file(s))".format(step, len(objs)))
            for o in objs[:5]:
                mt = o.get("LastModified")
                mt_str = mt.strftime("%Y-%m-%d %H:%M UTC") if mt else "—"
                print("    {}  {}".format(mt_str, o["Key"]))
            if len(objs) > 5:
                print("    ... and {} more".format(len(objs) - 5))
    print()

    # ----- 2a. Dashboard visuals pipeline checkpoints (pipeline_checkpoints/9_dashboard_visuals/) -----
    print("2a. Dashboard visuals pipeline checkpoints (s3://{}/pipeline_checkpoints/9_dashboard_visuals/)".format(REPO_BUCKET))
    print("-" * 60)
    dtw_pipeline_prefix = "pipeline_checkpoints/9_dashboard_visuals/"
    dtw_pipeline_objs = _s3_list(s3, REPO_BUCKET, dtw_pipeline_prefix, max_keys=200)
    dtw_checkpoint_jsons = [o for o in dtw_pipeline_objs if o["Key"].endswith("checkpoint.json")]
    if not dtw_checkpoint_jsons:
        print("  No 9_dashboard_visuals checkpoint.json files found.")
        print("  (Written by create_dtw_visuals.py via save_step_checkpoint; used for idempotency.)")
    else:
        for o in sorted(dtw_checkpoint_jsons, key=lambda x: (x.get("LastModified") or datetime.min.replace(tzinfo=timezone.utc)), reverse=True):
            # key: pipeline_checkpoints/9_dashboard_visuals/{cohort}/{age_band}/checkpoint.json
            rest = o["Key"].replace(dtw_pipeline_prefix, "").replace("/checkpoint.json", "")
            parts = rest.split("/")
            combo = "{} / {}".format(parts[0], parts[1].replace("_", "-")) if len(parts) >= 2 else rest
            mt = o.get("LastModified")
            mt_str = mt.strftime("%Y-%m-%d %H:%M UTC") if mt else "—"
            print("  {}  {}".format(mt_str, combo))
        print("  Total: {} cohort/age_band combination(s)".format(len(dtw_checkpoint_jsons)))
    print()

    # ----- 2b. DTW artifact mirror (5_dtw_checkpoint/) — optional/legacy -----
    print("2b. DTW artifact mirror (s3://{}/5_dtw_checkpoint/) [optional]".format(REPO_BUCKET))
    print("-" * 60)
    dtw_cp_objs = _s3_list(s3, REPO_BUCKET, DTW_CHECKPOINT_PREFIX + "/", max_keys=200)
    if not dtw_cp_objs:
        print("  No 5_dtw_checkpoint objects found.")
        print("  (CSV mirror from create_dtw_visuals; see README_dtw_s3_checkpoints.md.)")
    else:
        by_combo: dict[str, list] = {}
        for o in dtw_cp_objs:
            rest = o["Key"].replace(DTW_CHECKPOINT_PREFIX + "/", "")
            parts = rest.split("/", 2)
            combo = "{} / {}".format(parts[0], parts[1]) if len(parts) >= 2 else rest
            by_combo.setdefault(combo, []).append(o)
        for combo, objs in sorted(by_combo.items()):
            mt = objs[0].get("LastModified")
            mt_str = mt.strftime("%Y-%m-%d %H:%M UTC") if mt else "—"
            print("  {}  {}  ({} file(s))".format(mt_str, combo, len(objs)))
        print("  Total: {} cohort/age_band combination(s)".format(len(by_combo)))
    print()

    # ----- 3. DTW-related logs (*_log/) -----
    if show_logs:
        print("3. DTW-related logs (s3://{}/..._log/)".format(REPO_BUCKET))
        print("-" * 60)
        for log_prefix in DTW_LOG_PREFIXES:
            objs = _s3_list(s3, REPO_BUCKET, log_prefix + "/", max_keys=100)
            if not objs:
                print("  {}: no objects".format(log_prefix))
            else:
                objs.sort(key=lambda x: (x.get("LastModified") or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
                print("  {}: {} file(s)".format(log_prefix, len(objs)))
                for o in objs[:5]:
                    mt = o.get("LastModified")
                    mt_str = mt.strftime("%Y-%m-%d %H:%M UTC") if mt else "—"
                    print("    {}  {}".format(mt_str, o["Key"]))
                if len(objs) > 5:
                    print("    ... and {} more".format(len(objs) - 5))
        print()
    else:
        print("3. DTW-related logs: (use --logs to list)")
        print()

    # ----- 4. DTW outputs in pgxdatalake -----
    out_objs = _s3_list(s3, DATALAKE_BUCKET, DTW_OUTPUTS_PREFIX + "/", max_keys=300)
    by_combo = {}
    if out_objs:
        for o in out_objs:
            rest = o["Key"].replace(DTW_OUTPUTS_PREFIX + "/", "")
            parts = rest.split("/", 2)
            combo = "{} / {}".format(parts[0], parts[1]) if len(parts) >= 2 else rest
            by_combo.setdefault(combo, []).append(o)

    if show_outputs:
        print("4. DTW outputs (s3://{}/{})".format(DATALAKE_BUCKET, DTW_OUTPUTS_PREFIX))
        print("-" * 60)
        if not by_combo:
            print("  No objects under {}.".format(DTW_OUTPUTS_PREFIX))
        else:
            for combo, objs in sorted(by_combo.items()):
                mt = max((x.get("LastModified") or datetime.min.replace(tzinfo=timezone.utc) for x in objs), default=None)
                mt_str = mt.strftime("%Y-%m-%d %H:%M UTC") if mt else "—"
                print("  {}  {}  ({} file(s))".format(mt_str, combo, len(objs)))
            print("  Total: {} cohort/age_band combination(s)".format(len(by_combo)))
        print()

    # ----- 5. DTW cohort completion times (when each finished; duration not stored in S3) -----
    print("5. DTW cohort completion times (when each finished)")
    print("-" * 60)
    if not by_combo:
        print("  No DTW outputs in pgxdatalake; no completion times.")
    else:
        now = datetime.now(timezone.utc)
        rows = []
        for combo, objs in sorted(by_combo.items()):
            mt = max((x.get("LastModified") for x in objs if x.get("LastModified")), default=None)
            if mt:
                if mt.tzinfo is None:
                    mt = mt.replace(tzinfo=timezone.utc)
                ago_sec = (now - mt).total_seconds()
                if ago_sec < 3600:
                    ago_str = "{:.0f}m ago".format(ago_sec / 60)
                elif ago_sec < 86400:
                    ago_str = "{:.1f}h ago".format(ago_sec / 3600)
                else:
                    ago_str = "{:.1f}d ago".format(ago_sec / 86400)
                mt_str = mt.strftime("%Y-%m-%d %H:%M UTC")
                rows.append((mt, combo, mt_str, ago_str))
            else:
                rows.append((datetime.min.replace(tzinfo=timezone.utc), combo, "—", "—"))
        rows.sort(key=lambda x: x[0], reverse=True)
        print("  {:20}  {}  {}".format("Cohort / age_band", "Completed (UTC)", "Finished"))
        for _, combo, mt_str, ago_str in rows:
            print("  {:20}  {:19}  {}".format(combo, mt_str, ago_str))
        print()
        print("  Note: Processing duration (e.g. minutes per cohort) is not stored in S3.")
        print("  Completion time is from output object LastModified.")

    # ----- 6. Duration estimates from previous run gaps (same-day completions) -----
    # Both cohorts use full set of age bands (from py_helpers.constants)
    REQUIRED_DTW = [(c, ab) for c, bands in REQUIRED_COHORTS.items() for ab in bands]
    combo_to_key = lambda c, ab: "{} / {}".format(c, ab)
    required_keys = {combo_to_key(c, ab) for c, ab in REQUIRED_DTW}
    # Completed required combos with completion time, sorted oldest first
    completed_required = []
    for combo, objs in (by_combo or {}).items():
        if combo not in required_keys:
            continue
        mt = max((x.get("LastModified") for x in objs if x.get("LastModified")), default=None)
        if mt:
            if mt.tzinfo is None:
                mt = mt.replace(tzinfo=timezone.utc)
            completed_required.append((mt, combo))
    completed_required.sort(key=lambda x: x[0])
    print("6. Duration estimates (inferred from same-day completion gaps)")
    print("-" * 60)
    if len(completed_required) < 2:
        print("  Need at least 2 completed required cohorts to infer duration from gaps.")
        print("  Typical range from past runs: ~30 min to ~1.5 h per cohort (depends on cohort size).")
    else:
        max_gap_sec = 4 * 3600  # treat gaps > 4h as different sessions
        gaps_min = []
        print("  {:20}  {}  {}".format("Cohort / age_band", "Completed (UTC)", "Est. duration (from gap)"))
        prev_mt = None
        for mt, combo in completed_required:
            mt_str = mt.strftime("%Y-%m-%d %H:%M UTC")
            if prev_mt is not None:
                gap_sec = (mt - prev_mt).total_seconds()
                if gap_sec <= max_gap_sec and gap_sec >= 0:
                    gap_min = gap_sec / 60
                    gaps_min.append(gap_min)
                    print("  {:20}  {:19}  ~{:.0f} min".format(combo, mt_str, gap_min))
                else:
                    print("  {:20}  {:19}  (session gap)".format(combo, mt_str))
            else:
                print("  {:20}  {:19}  —".format(combo, mt_str))
            prev_mt = mt
        if gaps_min:
            print()
            print("  Inferred range: {:.0f}-{:.0f} min per cohort (from same-day gaps).".format(min(gaps_min), max(gaps_min)))
        completed_set = {x[1] for x in completed_required}
        missing = [combo_to_key(c, ab) for c, ab in REQUIRED_DTW if combo_to_key(c, ab) not in completed_set]
        if missing and gaps_min:
            print("  Remaining ({}): est. ~{:.0f}-{:.0f} min each if run sequentially.".format(len(missing), min(gaps_min), max(gaps_min)))
    print()
    if not show_outputs:
        print("(Use --outputs to list DTW output objects in pgxdatalake.)")
        print()
    print("Done.")


def main():
    ap = argparse.ArgumentParser(description="Check S3 checkpoints and logs for DTW workflow")
    ap.add_argument("--logs", action="store_true", help="List log objects under *_log prefixes")
    ap.add_argument("--outputs", action="store_true", help="List DTW output objects in pgxdatalake")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile")
    args = ap.parse_args()
    run(profile=args.profile, show_logs=args.logs, show_outputs=args.outputs)


if __name__ == "__main__":
    main()
