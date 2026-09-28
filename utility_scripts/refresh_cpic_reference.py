#!/usr/bin/env python3
"""
Monthly official CPIC / ClinPGx (PharmGKB) reference refresh.

Idempotent: HEAD official files (and API count fingerprints), compare against
s3://pgxdatalake/gold/reference/cpic/manifest.json, skip when unchanged.

On change: download official tables, write Parquet via the sibling ingest, upload
versioned + current objects, update the manifest, mirror gene-drug pairs to
gold/dashboard/data/ so PREFER_S3 Lambda /pgx/card picks them up.

Does not invent clinical rows. No GitHub Actions writer. No fetch on user requests.

Usage (repo root):
    python utility_scripts/refresh_cpic_reference.py
    python utility_scripts/refresh_cpic_reference.py --check-only
    python utility_scripts/refresh_cpic_reference.py --pairs-only --force
    python utility_scripts/refresh_cpic_reference.py --no-email
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "10_risk_dashboard" / "data_preparation"))

from ingest_cpic_pharmgkb_reference import (  # noqa: E402
    CLINPGX_DOWNLOAD,
    CPIC_API,
    CPIC_PAIRS_XLSX_URL,
    CPIC_TABLES,
    PHARMGKB_ZIPS,
    S3_BUCKET,
    S3_DATA_PREFIX,
    S3_REF_CPIC,
    S3_REF_PHARMGKB,
    build_allele_rsid,
    documented_unavailable,
    http_get,
    http_head,
    ingest_cpic_tables,
    ingest_gene_drug_pairs,
    ingest_pharmgkb_zips,
    list_s3_prefix,
    write_manifest,
    upload_outputs,
)

try:
    from py_helpers.aws_utils import send_status_email_ses as _send_ses
except Exception:  # pragma: no cover — slim Lambda image without psutil/requests
    _send_ses = None

USER_AGENT = "PGx-Analysis/2.0 (monthly CPIC/ClinPGx reference refresh)"
MANIFEST_KEY = f"{S3_REF_CPIC}/manifest.json"
DEFAULT_KNOWLEDGE_VERSION = "cpic-gene-drug-pairs-dashboard"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stamp(retrieved_at: str) -> str:
    return retrieved_at.replace("-", "").replace(":", "").replace("+00:00", "Z")


def _send_status_email(subject: str, body_text: str) -> bool:
    if _send_ses is not None:
        try:
            return bool(_send_ses(subject, body_text))
        except Exception:
            pass
    try:
        import boto3

        source = os.environ.get("SES_SOURCE", "jerome@mushinsolutions.com")
        to_address = os.environ.get("SES_TO", "dixonrj@vcu.edu")
        ses = boto3.client("ses", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        resp = ses.send_email(
            Source=source,
            Destination={"ToAddresses": [to_address]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body_text, "Charset": "UTF-8"}},
            },
        )
        return bool(resp.get("MessageId"))
    except Exception:
        return False


def _load_s3_json(key: str) -> Optional[Dict[str, Any]]:
    import boto3
    from botocore.exceptions import ClientError

    try:
        obj = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def _api_fingerprint(endpoint: str) -> Dict[str, Any]:
    url = f"{CPIC_API}/{endpoint}?limit=1&offset=0"
    body, headers = http_get(url, headers={"Accept": "application/json", "Prefer": "count=exact"})
    content_range = headers.get("Content-Range") or headers.get("content-range") or ""
    total: Optional[int] = None
    if "/" in content_range:
        tail = content_range.rsplit("/", 1)[-1]
        if tail.isdigit():
            total = int(tail)
    etag = headers.get("ETag") or headers.get("etag")
    last_modified = headers.get("Last-Modified") or headers.get("last-modified")
    return {
        "name": endpoint,
        "kind": "cpic_api",
        "source_url": f"{CPIC_API}/{endpoint}",
        "etag": etag,
        "last_modified": last_modified,
        "row_count": total,
        "content_range": content_range or None,
        "probe_bytes": len(body or b""),
    }


def _file_fingerprint(url: str, name: str, kind: str) -> Dict[str, Any]:
    try:
        head = http_head(url)
        return {
            "name": name,
            "kind": kind,
            "source_url": url,
            "etag": head.get("etag"),
            "last_modified": head.get("last_modified"),
            "content_length": head.get("content_length"),
            "probed_via": head.get("probed_via") or "head",
        }
    except HTTPError as exc:
        return {
            "name": name,
            "kind": kind,
            "source_url": url,
            "error": f"HTTP {exc.code}",
        }


def collect_source_fingerprints(pairs_only: bool) -> List[Dict[str, Any]]:
    sources = [_file_fingerprint(CPIC_PAIRS_XLSX_URL, "cpic_gene-drug_pairs", "official_file")]
    if pairs_only:
        return sources
    for spec in PHARMGKB_ZIPS:
        url = f"{CLINPGX_DOWNLOAD}/{spec['filename']}"
        sources.append(_file_fingerprint(url, spec["parquet"].replace(".parquet", ""), "official_file"))
    for spec in CPIC_TABLES:
        try:
            sources.append(_api_fingerprint(spec["endpoint"]))
        except Exception as exc:
            sources.append(
                {
                    "name": spec["endpoint"],
                    "kind": "cpic_api",
                    "source_url": f"{CPIC_API}/{spec['endpoint']}",
                    "error": str(exc),
                }
            )
    return sources


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"')


def sources_unchanged(current: List[Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> bool:
    if not previous:
        return False
    prior_list = previous.get("sources") or []
    prior = {row.get("name"): row for row in prior_list if isinstance(row, dict)}
    if not prior:
        return False
    for src in current:
        name = src.get("name")
        old = prior.get(name)
        if not old:
            return False
        if src.get("error") or old.get("error"):
            return False
        matched = False
        if _norm(src.get("etag")) and _norm(src.get("etag")) == _norm(old.get("etag")):
            matched = True
        if _norm(src.get("content_sha256")) and _norm(src.get("content_sha256")) == _norm(old.get("content_sha256")):
            matched = True
        if (
            _norm(src.get("last_modified"))
            and _norm(src.get("last_modified")) == _norm(old.get("last_modified"))
            and src.get("kind") == "official_file"
        ):
            matched = True
        if src.get("kind") == "cpic_api":
            if src.get("row_count") is not None and src.get("row_count") == old.get("row_count"):
                if _norm(src.get("etag")) == _norm(old.get("etag")):
                    matched = True
        if not matched:
            return False
    return True


def knowledge_version_from_sources(sources: List[Dict[str, Any]], retrieved_at: str) -> str:
    for src in sources:
        if src.get("name") == "cpic_gene-drug_pairs" and src.get("last_modified"):
            try:
                dt = parsedate_to_datetime(str(src["last_modified"]))
                return f"cpic-pairs-{dt.date().isoformat()}"
            except (TypeError, ValueError, IndexError):
                pass
    return f"cpic-pairs-{retrieved_at[:10]}"


def copy_prefix_version(s3, prefix: str, stamp: str) -> List[str]:
    copied: List[str] = []
    token = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": S3_BUCKET, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/") or "/versions/" in key:
                continue
            dest = f"{prefix.rstrip('/')}/versions/{stamp}/{Path(key).name}"
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": key},
                Key=dest,
            )
            copied.append(dest)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return copied


def run_ingest(pairs_only: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tables: List[Dict[str, Any]] = []
    unavailable = documented_unavailable()
    ingest_gene_drug_pairs(tables)
    if not pairs_only:
        ingest_cpic_tables(tables)
        build_allele_rsid(tables)
        ingest_pharmgkb_zips(tables, unavailable)
    return tables, unavailable


def enhance_manifest(
    manifest_path: Path,
    *,
    sources: List[Dict[str, Any]],
    retrieved_at: str,
    knowledge_version: str,
    skipped: bool,
    pairs_only: bool,
    s3_objects: Optional[List[Dict[str, Any]]],
    version_prefix: Optional[str],
) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "retrieved_at": retrieved_at,
            "skipped": skipped,
            "pairs_only": pairs_only,
            "cpicKnowledgeVersion": knowledge_version,
            "sources": sources,
            "versioned_prefix": version_prefix,
            "refresh_script": "utility_scripts/refresh_cpic_reference.py",
        }
    )
    if s3_objects is not None:
        payload["s3_objects"] = s3_objects
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _tables_from_local_manifest() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from ingest_cpic_pharmgkb_reference import LOCAL_REF_DIR

    dest = LOCAL_REF_DIR / "manifest.json"
    if not dest.exists():
        raise FileNotFoundError(f"No local sibling snapshot at {dest}")
    payload = json.loads(dest.read_text(encoding="utf-8"))
    tables = payload.get("tables") or []
    unavailable = payload.get("not_available") or documented_unavailable()
    if not tables:
        raise RuntimeError(f"Local manifest {dest} has no tables")
    return tables, unavailable


def refresh(
    *,
    force: bool = False,
    pairs_only: bool = False,
    check_only: bool = False,
    send_email: bool = True,
    notify_unchanged: bool = True,
    upload_existing: bool = False,
) -> Dict[str, Any]:
    retrieved_at = _utc_now()
    print(f"Collecting official source fingerprints at {retrieved_at} ...")
    sources = collect_source_fingerprints(pairs_only)
    for src in sources:
        extra = src.get("etag") or src.get("last_modified") or src.get("error") or src.get("row_count")
        print(f"  {src.get('kind')}: {src.get('name')} -> {extra}")

    previous = _load_s3_json(MANIFEST_KEY)
    unchanged = (not force) and sources_unchanged(sources, previous)
    knowledge_version = knowledge_version_from_sources(
        sources, (previous or {}).get("retrieved_at") or retrieved_at
    )
    if previous and previous.get("cpicKnowledgeVersion") and unchanged:
        knowledge_version = str(previous["cpicKnowledgeVersion"])

    result: Dict[str, Any] = {
        "skipped": unchanged,
        "retrieved_at": retrieved_at,
        "cpicKnowledgeVersion": knowledge_version,
        "pairs_only": pairs_only,
        "manifest": f"s3://{S3_BUCKET}/{MANIFEST_KEY}",
        "sources": sources,
    }

    if unchanged:
        msg = (
            f"CPIC/PharmGKB official sources unchanged.\n"
            f"cpicKnowledgeVersion={knowledge_version}\n"
            f"manifest=s3://{S3_BUCKET}/{MANIFEST_KEY}\n"
            f"checked_at={retrieved_at}\n"
        )
        print(msg)
        if send_email and notify_unchanged:
            emailed = _send_status_email("[PGx] CPIC reference unchanged", msg)
            result["email_sent"] = emailed
        return result

    if check_only:
        result["action"] = "would_refresh"
        print("Official sources changed (or no manifest). --check-only: not downloading.")
        return result

    print(
        "Promoting existing local snapshot ..."
        if upload_existing
        else "Official sources changed (or --force). Running ingest ..."
    )
    try:
        if upload_existing:
            tables, unavailable = _tables_from_local_manifest()
        else:
            tables, unavailable = run_ingest(pairs_only)
        manifest_path = write_manifest(tables, unavailable, s3_objects=None)
        enhance_manifest(
            manifest_path,
            sources=sources,
            retrieved_at=retrieved_at,
            knowledge_version=knowledge_version,
            skipped=False,
            pairs_only=pairs_only,
            s3_objects=None,
            version_prefix=None,
        )
        print("Uploading current objects ...")
        upload_outputs(tables, manifest_path)
        import boto3

        s3 = boto3.client("s3")
        stamp = _stamp(retrieved_at)
        versioned = []
        versioned.extend(copy_prefix_version(s3, S3_REF_CPIC, stamp))
        if not pairs_only:
            versioned.extend(copy_prefix_version(s3, S3_REF_PHARMGKB, stamp))
        version_prefix = f"s3://{S3_BUCKET}/{S3_REF_CPIC}/versions/{stamp}/"
        s3_objects = list_s3_prefix()
        enhance_manifest(
            manifest_path,
            sources=sources,
            retrieved_at=retrieved_at,
            knowledge_version=knowledge_version,
            skipped=False,
            pairs_only=pairs_only,
            s3_objects=s3_objects,
            version_prefix=version_prefix,
        )
        s3.upload_file(
            str(manifest_path),
            S3_BUCKET,
            MANIFEST_KEY,
            ExtraArgs={"ContentType": "application/json"},
        )
        # Keep the versioned manifest in sync after the listing rewrite.
        s3.upload_file(
            str(manifest_path),
            S3_BUCKET,
            f"{S3_REF_CPIC}/versions/{stamp}/manifest.json",
            ExtraArgs={"ContentType": "application/json"},
        )
        result.update(
            {
                "action": "refreshed",
                "table_count": len(tables),
                "row_counts": {row["name"]: row["row_count"] for row in tables},
                "mirror_pairs": f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/cpic_gene-drug_pairs.parquet",
                "versioned_objects": len(versioned),
                "version_prefix": version_prefix,
            }
        )
        body = (
            f"CPIC/PharmGKB official reference refreshed.\n"
            f"cpicKnowledgeVersion={knowledge_version}\n"
            f"retrieved_at={retrieved_at}\n"
            f"tables={len(tables)}\n"
            f"pairs_mirror=s3://{S3_BUCKET}/{S3_DATA_PREFIX}/cpic_gene-drug_pairs.parquet\n"
            f"canonical=s3://{S3_BUCKET}/{S3_REF_CPIC}/\n"
            f"pharmgkb=s3://{S3_BUCKET}/{S3_REF_PHARMGKB}/\n"
        )
        for name, count in result["row_counts"].items():
            body += f"  {count:,}  {name}\n"
        print(body)
        if send_email:
            result["email_sent"] = _send_status_email("[PGx] CPIC reference refreshed", body)
        return result
    except Exception as exc:
        err = f"CPIC/PharmGKB refresh FAILED: {exc}\n{traceback.format_exc()}"
        print(err)
        if send_email:
            _send_status_email("[PGx] CPIC reference refresh FAILED", err)
        raise


def lambda_handler(event, context):  # noqa: ARG001
    event = event or {}
    if not os.environ.get("PGX_CPIC_WORK_DIR"):
        os.environ["PGX_CPIC_WORK_DIR"] = "/tmp/cpic_refresh"
    # Re-bind ingest paths if the module was imported before WORK_DIR was set.
    import ingest_cpic_pharmgkb_reference as ingest_mod

    work = Path(os.environ["PGX_CPIC_WORK_DIR"])
    ingest_mod.LOCAL_CPIC_DIR = work
    ingest_mod.SOURCE_EXCEL = work / "cpic_gene-drug_pairs.xlsx"
    ingest_mod.LOCAL_REF_DIR = work / "reference"
    ingest_mod.LOCAL_RAW_DIR = work / "reference" / "raw"
    work.mkdir(parents=True, exist_ok=True)
    ingest_mod.LOCAL_REF_DIR.mkdir(parents=True, exist_ok=True)
    ingest_mod.LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    result = refresh(
        force=bool(event.get("force")),
        pairs_only=bool(event.get("pairs_only")),
        check_only=bool(event.get("check_only")),
        send_email=event.get("send_email", True),
        notify_unchanged=event.get("notify_unchanged", True),
        upload_existing=bool(event.get("upload_existing")),
    )
    return {"statusCode": 200, "body": json.dumps(result, default=str)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Monthly official CPIC/PharmGKB reference refresh")
    parser.add_argument("--force", action="store_true", help="Re-download even if ETag/Last-Modified/sha256 match")
    parser.add_argument("--pairs-only", action="store_true", help="Refresh gene-drug pairs xlsx only")
    parser.add_argument("--check-only", action="store_true", help="HEAD/compare only; do not download or upload")
    parser.add_argument("--no-email", action="store_true", help="Do not send SES notifications")
    parser.add_argument("--no-notify-unchanged", action="store_true", help="Skip SES when sources are unchanged")
    parser.add_argument(
        "--upload-existing",
        action="store_true",
        help="Promote an existing local sibling snapshot to canonical S3 (no re-download)",
    )
    args = parser.parse_args()
    refresh(
        force=args.force or args.upload_existing,
        pairs_only=args.pairs_only,
        check_only=args.check_only,
        send_email=not args.no_email,
        notify_unchanged=not args.no_notify_unchanged,
        upload_existing=args.upload_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
