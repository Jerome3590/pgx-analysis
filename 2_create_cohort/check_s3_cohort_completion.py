#!/usr/bin/env python3
r"""
Check S3 for cohort creation completion status with time durations.

Reads pipeline state from pgx-repository (pgx-pipeline-status/create_cohort/)
and optionally lists cohort parquets in pgxdatalake (gold/cohorts/) to report:
- Status (completed / running / failed)
- created_at, completed_at
- Duration (completed_at - created_at)

Usage:
    python check_s3_cohort_completion.py [--cohorts] [--outputs] [--profile NAME] [--repair] [--dry-run]
    --cohorts: only show pipeline state (default: both state + outputs summary)
    --outputs: also list each cohort parquet in pgxdatalake with LastModified and size
    --profile: AWS profile (default: AWS_PROFILE or default).
    --repair: for each 'running' entity, if cohort output exists in pgxdatalake, set state to completed in pgx-repository (run locally with profile that can write, e.g. mushin).
    --dry-run: with --repair, only print what would be repaired.
    Local: if C:\Projects\credentials exists, uses it (AWS_SHARED_CREDENTIALS_FILE).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use C:\Projects\credentials when present (local runs)
_creds_file = project_root.parent / "credentials"
if _creds_file.exists() and not os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(_creds_file)

STATE_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgx-repository")
STATE_PREFIX = "pgx-pipeline-status/create_cohort"
COHORT_BUCKET = os.environ.get("PGX_DATALAKE_BUCKET", "pgxdatalake")
COHORT_PREFIX = "gold/cohorts"
BUILD_LOGS_PREFIX = "build_logs/create_cohort"


def _parse_iso(s):
    if not s:
        return None
    try:
        # Handle Z suffix and +00:00
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s.replace("+00:00", ""))
    except Exception:
        return None


def _duration_seconds(created_at, completed_at):
    if not created_at or not completed_at:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)
    return (completed_at - created_at).total_seconds()


def _format_duration(seconds):
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def fetch_pipeline_states(s3_client):
    """List and fetch all create_cohort state.json from pgx-repository."""
    prefix = f"{STATE_PREFIX}/"
    states = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=STATE_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("state.json"):
                continue
            try:
                resp = s3_client.get_object(Bucket=STATE_BUCKET, Key=key)
                body = resp["Body"].read().decode("utf-8")
                data = json.loads(body)
                entity_id = key.replace(prefix, "").replace("/state.json", "").strip("/")
                data["_entity_id"] = entity_id
                data["_key"] = key
                data["_last_modified"] = obj.get("LastModified")
                states.append(data)
            except Exception as e:
                print(f"Warning: could not read {key}: {e}", file=sys.stderr)
    return states


def list_build_logs(s3_client, bucket=STATE_BUCKET, prefix=BUILD_LOGS_PREFIX, max_keys=100):
    """List recent build log objects (create_cohort) from pgx-repository."""
    results = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/", MaxKeys=max_keys):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".txt"):
                continue
            # key like build_logs/create_cohort/opioid_ed/25-44/2016/log_20260201_123456.txt
            parts = key.replace(prefix + "/", "").split("/")
            if len(parts) >= 4:
                cohort_name, band, year = parts[0], parts[1], parts[2]
            else:
                cohort_name = band = year = ""
            results.append({
                "cohort_name": cohort_name,
                "age_band": band,
                "event_year": year,
                "key": key,
                "last_modified": obj.get("LastModified"),
                "size": obj.get("Size", 0),
            })
    return results


def list_cohort_outputs(s3_client, bucket=COHORT_BUCKET, prefix=COHORT_PREFIX):
    """List cohort parquet keys with LastModified and size."""
    results = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("cohort.parquet"):
                continue
            # key like gold/cohorts/cohort_name=opioid_ed/event_year=2016/age_band=25-44/cohort.parquet
            parts = key.replace(prefix + "/", "").split("/")
            cohort_name = age_band = event_year = ""
            for p in parts:
                if p.startswith("cohort_name="):
                    cohort_name = p.replace("cohort_name=", "")
                elif p.startswith("event_year="):
                    event_year = p.replace("event_year=", "")
                elif p.startswith("age_band="):
                    age_band = p.replace("age_band=", "")
            results.append({
                "cohort_name": cohort_name,
                "event_year": event_year,
                "age_band": age_band,
                "key": key,
                "last_modified": obj.get("LastModified"),
                "size_bytes": obj.get("Size", 0),
            })
    return results


def _parse_entity_id(entity_id):
    """Parse entity_id into cohort, age_band, event_year. e.g. ed_non_opioid_85-114_2016 -> ('ed_non_opioid', '85-114', '2016')."""
    parts = entity_id.split("_")
    if len(parts) < 3:
        return None, None, None
    event_year = parts[-1]
    age_band = parts[-2]
    cohort = "_".join(parts[:-2])
    return cohort, age_band, event_year


def _cohort_output_exists(s3_client, cohort, age_band, event_year, bucket=COHORT_BUCKET):
    """Check if cohort parquet exists in pgxdatalake. cohort is e.g. opioid_ed or ed_non_opioid."""
    import py_helpers.s3_utils as s3_utils
    path = s3_utils.get_cohort_parquet_path(cohort, age_band, event_year, bucket_name=bucket)
    if path.startswith("s3://"):
        path = path[5:]
    buck, _, key = path.partition("/")
    try:
        s3_client.head_object(Bucket=buck, Key=key)
        return True
    except Exception:
        return False


def _repair_running_states(s3_client, states, bucket_state, bucket_cohorts, dry_run=False):
    """For each state with status 'running', if cohort output exists, set state to completed."""
    running = [s for s in states if s.get("status") == "running"]
    if not running:
        print("No 'running' states to repair.")
        return
    repaired = 0
    for s in running:
        entity_id = s.get("_entity_id", "")
        key = s.get("_key", "")
        cohort, age_band, event_year = _parse_entity_id(entity_id)
        if not cohort or not age_band or not event_year:
            print(f"  Skip {entity_id}: could not parse cohort/age_band/event_year")
            continue
        if cohort == "both":
            exists = (
                _cohort_output_exists(s3_client, "opioid_ed", age_band, event_year, bucket_cohorts)
                and _cohort_output_exists(s3_client, "ed_non_opioid", age_band, event_year, bucket_cohorts)
            )
        else:
            exists = _cohort_output_exists(s3_client, cohort, age_band, event_year, bucket_cohorts)
        if not exists:
            print(f"  Skip {entity_id}: output missing in pgxdatalake")
            continue
        if dry_run:
            print(f"  Would repair: {entity_id} -> completed")
            repaired += 1
            continue
        state = {k: v for k, v in s.items() if not k.startswith("_")}
        state["status"] = "completed"
        state["completed_at"] = datetime.utcnow().isoformat() + "Z"
        state["updated_at"] = state["completed_at"]
        state.setdefault("metadata", {})["repair_from_script"] = True
        try:
            s3_client.put_object(
                Bucket=bucket_state,
                Key=key,
                Body=json.dumps(state, indent=2),
                ContentType="application/json",
            )
            print(f"  Repaired: {entity_id} -> completed")
            repaired += 1
        except Exception as e:
            print(f"  Failed to write state for {entity_id}: {e}", file=sys.stderr)
    print(f"Repair: {repaired} state(s) updated." if not dry_run else f"Dry-run: {repaired} would be updated.")


def main():
    parser = argparse.ArgumentParser(
        description="Check S3 for cohort creation completion status with time durations"
    )
    parser.add_argument(
        "--cohorts",
        action="store_true",
        help="Only show pipeline state (default: state + outputs summary)",
    )
    parser.add_argument(
        "--outputs",
        action="store_true",
        help="List each cohort parquet in pgxdatalake with LastModified and size",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="List recent build logs from pgx-repository (create_cohort)",
    )
    parser.add_argument(
        "--bucket-state",
        default=STATE_BUCKET,
        help=f"S3 bucket for pipeline state (default: {STATE_BUCKET})",
    )
    parser.add_argument(
        "--bucket-cohorts",
        default=COHORT_BUCKET,
        help=f"S3 bucket for cohort parquets (default: {COHORT_BUCKET})",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="AWS profile for S3 (default: AWS_PROFILE or default profile)",
    )
    parser.add_argument("--repair", action="store_true",
        help="For each 'running' entity, if output exists in pgxdatalake, set state to completed in pgx-repository (run locally with profile that can write)")
    parser.add_argument("--dry-run", action="store_true", help="With --repair: only print what would be repaired")
    args = parser.parse_args()

    session_kw = {}
    if args.profile:
        session_kw["profile_name"] = args.profile
    session = boto3.Session(**session_kw)
    s3 = session.client("s3")

    # ----- Repair first if requested -----
    if getattr(args, "repair", False):
        states = fetch_pipeline_states(s3)
        print("REPAIR (running -> completed if output exists)")
        _repair_running_states(s3, states, args.bucket_state, args.bucket_cohorts, dry_run=getattr(args, "dry_run", False))
        if args.cohorts:
            return
        # Fall through to show state table

    # ----- Pipeline state (pgx-repository) -----
    print("=" * 80)
    print("COHORT PIPELINE STATE (pgx-repository)")
    print(f"Bucket: {args.bucket_state}  Prefix: {STATE_PREFIX}/")
    print("=" * 80)

    states = fetch_pipeline_states(s3)
    if not states:
        print("No pipeline state files found.")
    else:
        # Sort by entity_id for stable output
        states.sort(key=lambda x: x.get("_entity_id", ""))

        rows = []
        for s in states:
            entity_id = s.get("_entity_id", "?")
            status = s.get("status", "?")
            created_at = _parse_iso(s.get("created_at"))
            completed_at = _parse_iso(s.get("completed_at"))
            duration_sec = _duration_seconds(created_at, completed_at)
            created_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "—"
            completed_str = completed_at.strftime("%Y-%m-%d %H:%M") if completed_at else "—"
            rows.append({
                "entity_id": entity_id,
                "status": status,
                "created": created_str,
                "completed": completed_str,
                "duration": _format_duration(duration_sec),
            })

        # Print table
        max_id = max(len(r["entity_id"]) for r in rows)
        max_id = max(max_id, 24)
        fmt = f"{{:<{max_id}}}  {{:<10}}  {{:<16}}  {{:<16}}  {{:<10}}"
        print(fmt.format("ENTITY_ID", "STATUS", "CREATED", "COMPLETED", "DURATION"))
        print("-" * (max_id + 60))
        for r in rows:
            print(fmt.format(r["entity_id"], r["status"], r["created"], r["completed"], r["duration"]))

        # Summary
        completed = sum(r["status"] == "completed" for r in rows)
        failed = sum(r["status"] == "failed" for r in rows)
        running = sum(r["status"] == "running" for r in rows)
        print("-" * (max_id + 60))
        print(f"Total: {len(rows)}  completed: {completed}  failed: {failed}  running: {running}")

    # ----- Cohort outputs (pgxdatalake) -----
    if not args.cohorts:
        print()
        print("=" * 80)
        print("COHORT OUTPUTS (pgxdatalake)")
        print(f"Bucket: {args.bucket_cohorts}  Prefix: {COHORT_PREFIX}/")
        print("=" * 80)

        outputs = list_cohort_outputs(s3, bucket=args.bucket_cohorts)
        if not outputs:
            print("No cohort.parquet files found.")
        else:
            outputs.sort(key=lambda x: (x["cohort_name"], x["event_year"], x["age_band"]))
            if args.outputs:
                fmt = "{:<18} {:<6} {:<10}  {}  {:>12}"
                print(fmt.format("COHORT", "YEAR", "AGE_BAND", "LAST_MODIFIED", "SIZE_MB"))
                print("-" * 70)
                for o in outputs:
                    lm = o["last_modified"]
                    lm_str = lm.strftime("%Y-%m-%d %H:%M") if lm else "—"
                    size_mb = o["size_bytes"] / (1024 * 1024)
                    print(fmt.format(
                        o["cohort_name"], o["event_year"], o["age_band"],
                        lm_str, f"{size_mb:.2f}"
                    ))
            # Summary by cohort
            by_cohort = {}
            for o in outputs:
                c = o["cohort_name"]
                by_cohort[c] = by_cohort.get(c, 0) + 1
            print(f"Total parquet files: {len(outputs)}")
            for c, count in sorted(by_cohort.items()):
                print(f"  {c}: {count}")

    # ----- Build logs (pgx-repository) -----
    if args.logs:
        print()
        print("=" * 80)
        print("BUILD LOGS (pgx-repository)")
        print(f"Bucket: {args.bucket_state}  Prefix: {BUILD_LOGS_PREFIX}/")
        print("=" * 80)
        logs = list_build_logs(s3, bucket=args.bucket_state)
        if not logs:
            print("No build logs found.")
        else:
            logs.sort(key=lambda x: (x["last_modified"] or datetime.min.replace(tzinfo=timezone.utc), x["cohort_name"], x["age_band"], x["event_year"]), reverse=True)
            fmt = "{:<18} {:<8} {:<6}  {}  {}"
            print(fmt.format("COHORT", "AGE_BAND", "YEAR", "LAST_MODIFIED", "KEY"))
            print("-" * 90)
            for log in logs[:80]:
                lm = log["last_modified"]
                lm_str = lm.strftime("%Y-%m-%d %H:%M") if lm else "—"
                key_short = log["key"].split("/")[-1] if log["key"] else ""
                print(fmt.format(log["cohort_name"], log["age_band"], str(log["event_year"]), lm_str, key_short))
            if len(logs) > 80:
                print(f"... and {len(logs) - 80} more")
            print(f"Total log files: {len(logs)}")

    print()


if __name__ == "__main__":
    main()
