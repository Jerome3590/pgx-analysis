#!/usr/bin/env python3
"""
Upload dashboard visualization artifacts to S3 using the manifest as the single source of truth.

Reads 10_risk_dashboard/visualizations/dashboard_visual_objects.json and uploads ONLY the
files listed in each entry's static_files, resolved for every cohort/age_band combination.

Nothing outside the manifest static_files is uploaded, preventing S3 pollution with
intermediate files such as trajectory_status_*.json, dtw_model_events_diagnostics_*.json,
*.csv, *.parquet, Rplots.pdf, and other pipeline artifacts the dashboard never reads.

NOTE: Scenario Analysis artifacts (dashboard_data.json → scenario_data.json rename-on-upload)
are handled separately by upload_scenario_outputs_to_s3.py, which is still called by
5_build_and_deploy.py before this script.

Usage:
    python sync_visuals_to_s3.py
    python sync_visuals_to_s3.py --dry-run
    python sync_visuals_to_s3.py --strict     # exit 1 if any required file missing locally
    python sync_visuals_to_s3.py --tab "DTW Trajectories"   # restrict to one tab

Environment:
    S3_DASHBOARD_BUCKET  (default: jerome-dixon.io)
    S3_DASHBOARD_PREFIX  (default: vcu/pgx-risk-calculator)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "dashboard_visual_objects.json"

CONTENT_TYPES = {
    ".json": "application/json",
    ".html": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".css": "text/css",
    ".js": "application/javascript",
    ".svg": "image/svg+xml",
    ".csv": "text/csv",
}

# These dashboard tabs are handled by separate dedicated scripts; skip here to avoid double-upload.
_SKIP_TABS = {"Scenario Analysis (FFA/SHAP)"}


def _content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _get_cohorts_and_bands() -> List[Tuple[str, str]]:
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from py_helpers.constants import REQUIRED_COHORTS  # type: ignore[import]
        combos = []
        for cohort, bands in REQUIRED_COHORTS.items():
            for age_band in bands:
                combos.append((cohort, age_band))
        return combos
    except Exception:
        cohorts = ["opioid_ed", "non_opioid_ed"]
        bands = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
        return [(c, b) for c in cohorts for b in bands]


def _expand(template: str, *, cohort: str = "", age_band: str = "", age_band_fname: str = "", base: str = "") -> str:
    return (
        template
        .replace("{cohort}", cohort)
        .replace("{age_band}", age_band)
        .replace("{age_band_fname}", age_band_fname)
        .replace("{base}", base)
    )


def _resolve_local(
    repo_root: Path,
    entry: dict,
    expanded_static_file: str,
    cohort: str,
    age_band_fname: str,
) -> Optional[Path]:
    """
    Derive the local filesystem path for a (possibly already expanded) static_file entry.

    For directory-type entries:
      - The local base is repo_root / entry["path"] / cohort / age_band_fname
      - Any subdirectory embedded in s3_path AFTER {age_band} (e.g. /plots/) is appended to the base.
      - Then expanded_static_file is appended.
      - A fallback to data/<file> is tried when the file is not found at the top level (FP-Growth itemsets).

    For file_pattern / file entries:
      - The parent directory of the (expanded) entry["path"] is the local base.
    """
    path_type = entry.get("path_type", "directory")
    entry_path = entry.get("path", "")

    if path_type == "directory":
        s3_path = entry.get("s3_path", "")
        # Extract any fixed path component that appears after {age_band} in the s3_path template.
        # e.g. bupar s3_path ends with "/{age_band}/plots/" → after_age_band = "plots"
        after_age_band = ""
        if "{age_band}" in s3_path:
            raw = s3_path.split("{age_band}", 1)[1].strip("/")
            # Capture all fixed path components after {age_band} up to the first placeholder
            parts = raw.split("/")
            fixed_parts = []
            for part in parts:
                if not part or "{" in part:
                    break
                fixed_parts.append(part)
            after_age_band = "/".join(fixed_parts)  # e.g. "density/low/plots"

        local_base = repo_root / entry_path / cohort / age_band_fname
        if after_age_band:
            local_base = local_base / after_age_band

        candidate = local_base / expanded_static_file
        if candidate.exists():
            return candidate

        # Return non-existent candidate so caller can report it as missing
        return candidate

    elif path_type in ("file_pattern", "file"):
        path_expanded = _expand(entry_path, cohort=cohort, age_band_fname=age_band_fname)
        local_dir = (repo_root / path_expanded).parent
        return local_dir / expanded_static_file

    return None


def _resolve_s3_key(
    entry: dict,
    raw_static_file: str,
    cohort: str,
    age_band: str,
    base: str,
) -> str:
    """
    Build the full S3 key for a static_file entry.

    For directory entries: s3_key = s3_path_expanded.rstrip('/') + '/' + expanded_file
    For file/file_pattern entries: s3_key = dirname(s3_path_expanded) + '/' + expanded_file
    """
    s3_path = entry.get("s3_path", "")
    path_type = entry.get("path_type", "directory")
    expanded_file = _expand(raw_static_file, cohort=cohort, age_band=age_band, base=base)

    if path_type == "directory":
        s3_dir = _expand(s3_path.rstrip("/"), cohort=cohort, age_band=age_band, base=base)
        return f"{s3_dir}/{expanded_file}"

    elif path_type in ("file_pattern", "file"):
        s3_path_expanded = _expand(s3_path, cohort=cohort, age_band=age_band, base=base)
        s3_dir = str(PurePosixPath(s3_path_expanded).parent)
        return f"{s3_dir}/{expanded_file}"

    return ""


def build_upload_plan(
    manifest: dict,
    repo_root: Path,
    cohort_bands: List[Tuple[str, str]],
) -> List[Tuple[Path, str]]:
    """
    Return a deduplicated list of (local_path, s3_key) pairs derived strictly from the manifest.
    """
    plan: List[Tuple[Path, str]] = []
    seen_s3: set = set()
    cohorts_only = list(dict.fromkeys(c for c, _ in cohort_bands))

    for entry in manifest.get("visual_objects", []):
        tab = entry.get("dashboard_tab", "")
        if tab in _SKIP_TABS:
            continue

        path_type = entry.get("path_type", "directory")
        static_files: List[str] = entry.get("static_files") or []
        cohort_scope = entry.get("cohort_scope", "")

        if path_type == "directory":
            for cohort, age_band in cohort_bands:
                age_band_fname = age_band.replace("-", "_")
                base = f"{cohort}_{age_band_fname}"
                for sf in static_files:
                    expanded_sf = _expand(sf, cohort=cohort, age_band=age_band, age_band_fname=age_band_fname, base=base)
                    local = _resolve_local(repo_root, entry, expanded_sf, cohort, age_band_fname)
                    s3_key = _resolve_s3_key(entry, sf, cohort, age_band, base)
                    if local is None or not s3_key or s3_key in seen_s3:
                        continue
                    seen_s3.add(s3_key)
                    plan.append((local, s3_key))

        elif path_type == "file_pattern":
            # per_cohort scope: iterate cohorts without age_band
            if cohort_scope == "per_cohort":
                for cohort in cohorts_only:
                    for sf in static_files:
                        expanded_sf = _expand(sf, cohort=cohort)
                        local = _resolve_local(repo_root, entry, expanded_sf, cohort, "")
                        s3_key = _resolve_s3_key(entry, sf, cohort, "", "")
                        if local is None or not s3_key or s3_key in seen_s3:
                            continue
                        seen_s3.add(s3_key)
                        plan.append((local, s3_key))
            else:
                for cohort, age_band in cohort_bands:
                    age_band_fname = age_band.replace("-", "_")
                    base = f"{cohort}_{age_band_fname}"
                    for sf in static_files:
                        expanded_sf = _expand(sf, cohort=cohort, age_band=age_band, age_band_fname=age_band_fname, base=base)
                        local = _resolve_local(repo_root, entry, expanded_sf, cohort, age_band_fname)
                        s3_key = _resolve_s3_key(entry, sf, cohort, age_band, base)
                        if local is None or not s3_key or s3_key in seen_s3:
                            continue
                        seen_s3.add(s3_key)
                        plan.append((local, s3_key))

        elif path_type == "file":
            for sf in static_files:
                local = _resolve_local(repo_root, entry, sf, "", "")
                s3_key = _resolve_s3_key(entry, sf, "", "", "")
                if local is None or not s3_key or s3_key in seen_s3:
                    continue
                seen_s3.add(s3_key)
                plan.append((local, s3_key))

    return plan


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Upload dashboard visuals to S3 using manifest static_files only (no bulk sync)."
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be uploaded; do not upload")
    p.add_argument("--strict", action="store_true", help="Exit 1 if any locally-missing required file")
    p.add_argument("--tab", default=None, help="Restrict upload to a single dashboard tab name")
    p.add_argument("--bucket", default=os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io"))
    p.add_argument("--prefix", default=os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator"))
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    # Allow --prefix to override manifest s3_prefix
    manifest_prefix = manifest.get("s3_prefix", "").strip("/")
    effective_prefix = args.prefix.strip("/") or manifest_prefix

    # Patch all s3_paths in manifest if prefix differs from manifest s3_prefix
    if effective_prefix != manifest_prefix and manifest_prefix:
        for entry in manifest.get("visual_objects", []):
            if entry.get("s3_path", "").startswith(manifest_prefix):
                entry["s3_path"] = effective_prefix + entry["s3_path"][len(manifest_prefix):]

    # Tab filter
    if args.tab:
        manifest["visual_objects"] = [
            e for e in manifest.get("visual_objects", [])
            if e.get("dashboard_tab", "") == args.tab
        ]
        if not manifest["visual_objects"]:
            print(f"[WARN] No manifest entries for tab '{args.tab}'", file=sys.stderr)

    bucket = args.bucket
    cohort_bands = _get_cohorts_and_bands()
    plan = build_upload_plan(manifest, REPO_ROOT, cohort_bands)

    print("Dashboard manifest-only S3 upload")
    print("=" * 72)
    print(f"  Bucket  : s3://{bucket}")
    print(f"  Prefix  : {effective_prefix}")
    print(f"  Manifest: {args.manifest}")
    print(f"  Artifacts: {len(plan)} (from manifest static_files only)")
    if args.tab:
        print(f"  Tab filter: {args.tab}")
    if args.dry_run:
        print("  Mode    : DRY RUN (no uploads)")
    print()

    try:
        import boto3
        s3 = boto3.client("s3", region_name=args.region)
    except ImportError:
        print("[ERROR] boto3 not available; pip install boto3", file=sys.stderr)
        return 1

    uploaded = 0
    missing = 0
    failed = 0
    missing_list: List[str] = []

    for local_path, s3_key in plan:
        if not local_path.exists():
            rel = str(local_path.relative_to(REPO_ROOT)) if local_path.is_relative_to(REPO_ROOT) else str(local_path)
            print(f"  ~ missing: {rel}")
            missing += 1
            missing_list.append(str(local_path))
            continue

        ct = _content_type(local_path)
        rel = str(local_path.relative_to(REPO_ROOT)) if local_path.is_relative_to(REPO_ROOT) else str(local_path)

        if args.dry_run:
            print(f"  [DRY] {rel}  →  s3://{bucket}/{s3_key}")
            uploaded += 1
            continue

        try:
            s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs={"ContentType": ct})
            print(f"  ✓ {rel}  →  s3://{bucket}/{s3_key}")
            uploaded += 1
        except Exception as e:
            print(f"  ✗ FAILED {s3_key}: {e}", file=sys.stderr)
            failed += 1

    print()
    print("=" * 72)
    action = "Would upload" if args.dry_run else "Uploaded"
    print(f"{action}: {uploaded}  |  Missing locally: {missing}  |  Failed: {failed}")
    if missing_list:
        print(f"  ({missing} file(s) not yet produced by pipeline — run notebook 4 first)")
    if _SKIP_TABS:
        print(f"  Skipped tabs (separate scripts): {', '.join(sorted(_SKIP_TABS))}")
    if failed:
        return 2
    if args.strict and missing > 0:
        print("Exiting with code 1 (--strict).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
