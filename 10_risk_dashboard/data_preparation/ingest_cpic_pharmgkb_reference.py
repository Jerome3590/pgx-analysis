#!/usr/bin/env python3
"""
Ingest official CPIC / ClinPGx (PharmGKB) reference tables to Parquet.

Writes local copies under 10_risk_dashboard/outputs/cpic/ (or $PGX_CPIC_WORK_DIR)
and uploads to the canonical prefixes:

  s3://pgxdatalake/gold/reference/cpic/        (CPIC tables + manifest.json)
  s3://pgxdatalake/gold/reference/pharmgkb/    (ClinPGx / PharmGKB tables)
  s3://pgxdatalake/gold/dashboard/data/        (pairs parquet/xlsx mirror for Lambda)

Monthly refresh: utility_scripts/refresh_cpic_reference.py (HEAD/manifest/SES).

Official sources only — no HTML guideline scraping, no invented mappings.

Usage (repo root):
    python 10_risk_dashboard/data_preparation/ingest_cpic_pharmgkb_reference.py
    python 10_risk_dashboard/data_preparation/ingest_cpic_pharmgkb_reference.py --no-upload
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import S3_BUCKET

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("duckdb is required for this ingest") from exc

USER_AGENT = "PGx-Analysis/2.0 (official CPIC/ClinPGx reference ingest)"
CPIC_API = "https://api.cpicpgx.org/v1"
CPIC_PAIRS_XLSX_URL = "https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx"
CLINPGX_DOWNLOAD = "https://api.clinpgx.org/v1/download/file/data"
CPIC_API_DOCS = "https://api.cpicpgx.org/"
CLINPGX_DOWNLOADS_PAGE = "https://www.clinpgx.org/downloads"
CPIC_API_AND_DB = "https://cpicpgx.org/api-and-database/"

_WORK_DIR = os.environ.get("PGX_CPIC_WORK_DIR", "").strip()
if _WORK_DIR:
    LOCAL_CPIC_DIR = Path(_WORK_DIR)
    SOURCE_EXCEL = LOCAL_CPIC_DIR / "cpic_gene-drug_pairs.xlsx"
else:
    LOCAL_CPIC_DIR = PROJECT_ROOT / "10_risk_dashboard" / "outputs" / "cpic"
    SOURCE_EXCEL = PROJECT_ROOT / "5_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
LOCAL_REF_DIR = LOCAL_CPIC_DIR / "reference"
LOCAL_RAW_DIR = LOCAL_REF_DIR / "raw"

# Canonical gold layout (monthly refresh + one-shot ingest share these).
S3_DATA_PREFIX = "gold/dashboard/data"
S3_REF_CPIC = "gold/reference/cpic"
S3_REF_PHARMGKB = "gold/reference/pharmgkb"
S3_CPIC_PREFIX = S3_REF_CPIC  # backward-compatible alias used by table URIs

# Official PostgREST tables that close phenotype / recommendation / rsid gaps.
CPIC_TABLES = (
    {
        "endpoint": "diplotype",
        "filename": "cpic_diplotype_phenotype.parquet",
        "description": "Official CPIC diplotype → phenotype (gene result) view",
    },
    {
        "endpoint": "recommendation_view",
        "filename": "cpic_recommendation.parquet",
        "description": "Official CPIC structured recommendations with drug and guideline names",
    },
    {
        "endpoint": "pair_view",
        "filename": "cpic_pair.parquet",
        "description": "Official CPIC gene–drug pairs with drug names and guideline URLs",
    },
    {
        "endpoint": "allele",
        "filename": "cpic_allele.parquet",
        "description": "Official CPIC allele clinical function assignments",
    },
    {
        "endpoint": "allele_definition",
        "filename": "cpic_allele_definition.parquet",
        "description": "Official CPIC named-allele definitions",
    },
    {
        "endpoint": "sequence_location",
        "filename": "cpic_sequence_location.parquet",
        "description": "Official CPIC variant locations including dbSNP rsIDs",
    },
    {
        "endpoint": "allele_location_value",
        "filename": "cpic_allele_location_value.parquet",
        "description": "Official CPIC allele-definition × sequence-location alleles",
    },
    {
        "endpoint": "gene_result",
        "filename": "cpic_gene_result.parquet",
        "description": "Official CPIC gene-level phenotype / activity-score terms",
    },
    {
        "endpoint": "gene_result_lookup",
        "filename": "cpic_gene_result_lookup.parquet",
        "description": "Official CPIC allele-function combination → phenotype lookup",
    },
    {
        "endpoint": "guideline",
        "filename": "cpic_guideline.parquet",
        "description": "Official CPIC guideline metadata and ClinPGx URLs",
    },
    {
        "endpoint": "gene",
        "filename": "cpic_gene.parquet",
        "description": "Official CPIC gene catalog",
    },
    {
        "endpoint": "drug",
        "filename": "cpic_drug.parquet",
        "description": "Official CPIC drug catalog with RxNorm / ClinPGx IDs",
    },
    {
        "endpoint": "file_artifact",
        "filename": "cpic_file_artifact.parquet",
        "description": "Official CPIC per-gene Excel supplement URLs on files.cpicpgx.org",
    },
    {
        "endpoint": "test_alert",
        "filename": "cpic_test_alert.parquet",
        "description": "Official CPIC CDS / test-alert recommendation text",
    },
)

# Official ClinPGx/PharmGKB bulk exports (Downloads page).
PHARMGKB_ZIPS = (
    {
        "filename": "variants.zip",
        "parquet": "pharmgkb_variants.parquet",
        "description": "ClinPGx annotated variants tracked in dbSNP (rsid catalog)",
    },
    {
        "filename": "clinicalVariants.zip",
        "parquet": "pharmgkb_clinical_variants.parquet",
        "description": "ClinPGx clinical variant–drug pairs and evidence levels",
    },
    {
        "filename": "guidelineAnnotations.json.zip",
        "parquet": "pharmgkb_guideline_annotations.parquet",
        "description": "ClinPGx structured clinical guideline annotation JSON",
    },
    {
        "filename": "clinicalAnnotations.zip",
        "parquet": "pharmgkb_clinical_annotations.parquet",
        "description": "ClinPGx clinical annotation summaries",
    },
)

PAGE_SIZE = 1000
REQUEST_PAUSE_SEC = 0.08
LICENSE_CPIC = (
    "CPIC / ClinPGx structured data via the official PostgREST API and files.cpicpgx.org. "
    "Cite the relevant CPIC guideline publications. See https://cpicpgx.org/ and "
    "https://api.cpicpgx.org/."
)
LICENSE_CLINPGX = (
    "ClinPGx / PharmGKB downloads (typically Creative Commons Attribution-ShareAlike 4.0). "
    "See LICENSE.txt inside each zip and https://www.clinpgx.org/downloads."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ssl_contexts() -> List[ssl.SSLContext]:
    contexts = [ssl.create_default_context()]
    contexts.append(ssl._create_unverified_context())  # noqa: SLF001 — match existing CPIC downloaders
    return contexts


def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 90) -> Tuple[bytes, Dict[str, str]]:
    merged = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        merged.update(headers)
    last_err: Optional[Exception] = None
    for ctx in _ssl_contexts():
        try:
            req = Request(url, headers=merged)
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                body = resp.read()
                resp_headers = {k: v for k, v in resp.headers.items()}
                return body, resp_headers
        except HTTPError as exc:
            if exc.code == 404:
                raise
            last_err = exc
        except (URLError, TimeoutError, ssl.SSLError) as exc:
            last_err = exc
            continue
    raise RuntimeError(f"GET failed for {url}: {last_err}")


def _header_ci(headers: Dict[str, str], name: str) -> Optional[str]:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def http_head(url: str, timeout: int = 60) -> Dict[str, Any]:
    """HEAD official file; fall back to a Range GET if HEAD is not allowed."""
    merged = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    last_err: Optional[Exception] = None
    for ctx in _ssl_contexts():
        try:
            req = Request(url, headers=merged, method="HEAD")
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                headers = {k: v for k, v in resp.headers.items()}
                return {
                    "url": url,
                    "status": getattr(resp, "status", 200),
                    "etag": _header_ci(headers, "ETag"),
                    "last_modified": _header_ci(headers, "Last-Modified"),
                    "content_length": _header_ci(headers, "Content-Length"),
                    "content_type": _header_ci(headers, "Content-Type"),
                }
        except HTTPError as exc:
            if exc.code in (403, 405, 501):
                break
            last_err = exc
        except (URLError, TimeoutError, ssl.SSLError) as exc:
            last_err = exc
            continue
    # Some CDNs reject HEAD; probe with a 1-byte range GET.
    try:
        body, headers = http_get(url, headers={"Range": "bytes=0-0"}, timeout=timeout)
        return {
            "url": url,
            "status": 206 if body is not None else 200,
            "etag": _header_ci(headers, "ETag"),
            "last_modified": _header_ci(headers, "Last-Modified"),
            "content_length": _header_ci(headers, "Content-Range") or _header_ci(headers, "Content-Length"),
            "content_type": _header_ci(headers, "Content-Type"),
            "probed_via": "range_get",
        }
    except Exception as exc:
        raise RuntimeError(f"HEAD failed for {url}: {last_err or exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_file(url: str, dest: Path) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    body, headers = http_get(url)
    dest.write_bytes(body)
    return {
        "url": url,
        "bytes": len(body),
        "content_type": _header_ci(headers, "Content-Type"),
        "etag": _header_ci(headers, "ETag"),
        "last_modified": _header_ci(headers, "Last-Modified"),
        "content_sha256": sha256_bytes(body),
        "path": str(dest),
    }


def sanitize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Serialize nested JSON so Parquet columns stay scalar."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {}
        for key, value in rec.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
            else:
                row[key] = value
        out.append(row)
    return out


def write_records_parquet(records: List[Dict[str, Any]], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if not records:
        return 0
    import pandas as pd

    df = pd.DataFrame(sanitize_records(records))
    con = duckdb.connect(":memory:")
    try:
        con.register("src", df)
        path_sql = str(dest.resolve().as_posix()).replace("'", "''")
        con.execute(f"COPY src TO '{path_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        count_result = con.execute("SELECT COUNT(*)::BIGINT FROM read_parquet(?)", [str(dest)]).fetchone()[0]
        return int(count_result) if count_result is not None else 0
    finally:
        con.close()


def parquet_count(path: Path) -> int:
    con = duckdb.connect(":memory:")
    try:
        count_result = con.execute("SELECT COUNT(*)::BIGINT FROM read_parquet(?)", [str(path)]).fetchone()[0]
        return int(count_result) if count_result is not None else 0
    finally:
        con.close()


def cpic_fetch_all(endpoint: str) -> Tuple[List[Dict[str, Any]], str]:
    """Page through an official CPIC PostgREST table. Returns (rows, source_url)."""
    source_url = f"{CPIC_API}/{endpoint}"
    rows: List[Dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None
    while True:
        url = f"{source_url}?limit={PAGE_SIZE}&offset={offset}"
        headers = {"Accept": "application/json"}
        if offset == 0:
            headers["Prefer"] = "count=exact"
        body, resp_headers = http_get(url, headers=headers)
        page = json.loads(body.decode("utf-8")) if body else []
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected payload from {url}: {type(page)}")
        if total is None:
            content_range = resp_headers.get("Content-Range") or resp_headers.get("content-range") or ""
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[-1]
                if tail.isdigit():
                    total = int(tail)
        rows.extend(page)
        print(f"  {endpoint}: fetched {len(rows):,}" + (f" / {total:,}" if total is not None else ""))
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        if offset > 2_000_000:
            raise RuntimeError(f"Safety stop paging {endpoint} at offset {offset}")
        time.sleep(REQUEST_PAUSE_SEC)
    return rows, source_url


def ingest_cpic_tables(manifest_tables: List[Dict[str, Any]]) -> None:
    for spec in CPIC_TABLES:
        endpoint = spec["endpoint"]
        dest = LOCAL_REF_DIR / spec["filename"]
        print(f"CPIC API {endpoint} -> {dest.name}")
        rows, source_url = cpic_fetch_all(endpoint)
        count = write_records_parquet(rows, dest)
        print(f"  wrote {count:,} rows")
        manifest_tables.append(
            {
                "name": dest.stem,
                "source_url": source_url,
                "source": "CPIC PostgREST API",
                "description": spec["description"],
                "local_path": str(dest),
                "s3_uri": f"s3://{S3_BUCKET}/{S3_CPIC_PREFIX}/{dest.name}",
                "row_count": count,
                "license": LICENSE_CPIC,
            }
        )


def build_allele_rsid(manifest_tables: List[Dict[str, Any]]) -> None:
    """Join official CPIC tables on documented keys; do not invent mappings."""
    dest = LOCAL_REF_DIR / "cpic_allele_rsid.parquet"
    definition = LOCAL_REF_DIR / "cpic_allele_definition.parquet"
    locations = LOCAL_REF_DIR / "cpic_sequence_location.parquet"
    loc_values = LOCAL_REF_DIR / "cpic_allele_location_value.parquet"
    alleles = LOCAL_REF_DIR / "cpic_allele.parquet"
    missing = [p.name for p in (definition, locations, loc_values) if not p.exists()]
    if missing:
        print(f"Skipping allele-rsid join; missing {missing}")
        return
    if dest.exists():
        dest.unlink()
    path_sql = str(dest.resolve().as_posix()).replace("'", "''")
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE allele_rsid AS
            SELECT
                ad.genesymbol,
                ad.name AS allele_name,
                ad.pharmvarid,
                sl.dbsnpid AS rsid,
                sl.name AS variant_name,
                sl.chromosomelocation,
                sl.genelocation,
                sl.proteinlocation,
                sl.position,
                alv.variantallele,
                ad.id AS allele_definition_id,
                sl.id AS location_id,
                a.clinicalfunctionalstatus,
                a.activityvalue
            FROM read_parquet(?) ad
            INNER JOIN read_parquet(?) alv
                ON CAST(ad.id AS VARCHAR) = CAST(alv.alleledefinitionid AS VARCHAR)
            INNER JOIN read_parquet(?) sl
                ON CAST(sl.id AS VARCHAR) = CAST(alv.locationid AS VARCHAR)
            LEFT JOIN read_parquet(?) a
                ON CAST(a.definitionid AS VARCHAR) = CAST(ad.id AS VARCHAR)
            WHERE sl.dbsnpid IS NOT NULL
              AND CAST(sl.dbsnpid AS VARCHAR) <> ''
            """,
            [str(definition), str(loc_values), str(locations), str(alleles)],
        )
        con.execute(f"COPY allele_rsid TO '{path_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        count_result = con.execute("SELECT COUNT(*)::BIGINT FROM allele_rsid").fetchone()[0]
        count = int(count_result) if count_result is not None else 0
    finally:
        con.close()
    print(f"Derived official join cpic_allele_rsid.parquet: {count:,} rows")
    manifest_tables.append(
        {
            "name": dest.stem,
            "source_url": f"{CPIC_API}/allele_definition + /allele_location_value + /sequence_location + /allele",
            "source": "CPIC PostgREST API (join on official foreign keys only)",
            "description": "Named allele → dbSNP rsid from official CPIC allele definition tables",
            "local_path": str(dest),
            "s3_uri": f"s3://{S3_BUCKET}/{S3_CPIC_PREFIX}/{dest.name}",
            "row_count": count,
            "license": LICENSE_CPIC,
        }
    )


def write_excel_parquet(xlsx_path: Path, dest: Path) -> int:
    import pandas as pd

    df = pd.read_excel(xlsx_path, engine="openpyxl")
    if dest.exists():
        dest.unlink()
    con = duckdb.connect(":memory:")
    try:
        con.register("src", df)
        path_sql = str(dest.resolve().as_posix()).replace("'", "''")
        con.execute(f"COPY src TO '{path_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        count_result = con.execute("SELECT COUNT(*)::BIGINT FROM read_parquet(?)", [str(dest)]).fetchone()[0]
        return int(count_result) if count_result is not None else 0
    finally:
        con.close()


def ingest_gene_drug_pairs(manifest_tables: List[Dict[str, Any]]) -> None:
    print(f"Downloading official CPIC gene-drug pairs: {CPIC_PAIRS_XLSX_URL}")
    SOURCE_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CPIC_DIR.mkdir(parents=True, exist_ok=True)
    meta = download_file(CPIC_PAIRS_XLSX_URL, SOURCE_EXCEL)
    dest_xlsx = LOCAL_CPIC_DIR / "cpic_gene-drug_pairs.xlsx"
    dest_xlsx.write_bytes(SOURCE_EXCEL.read_bytes())
    dest_parquet = LOCAL_CPIC_DIR / "cpic_gene-drug_pairs.parquet"
    count = write_excel_parquet(dest_xlsx, dest_parquet)
    ref_copy = LOCAL_REF_DIR / "cpic_gene-drug_pairs.parquet"
    if ref_copy.exists() or dest_parquet.exists():
        ref_copy.write_bytes(dest_parquet.read_bytes())
    print(f"  gene-drug pairs: {count:,} rows ({meta['bytes']:,} bytes)")
    manifest_tables.append(
        {
            "name": "cpic_gene-drug_pairs",
            "source_url": CPIC_PAIRS_XLSX_URL,
            "source": "CPIC files.cpicpgx.org current pair export",
            "description": "Official CPIC gene–drug pairs Excel (Lambda-compatible columns)",
            "local_path": str(dest_parquet),
            "s3_uri": f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/cpic_gene-drug_pairs.parquet",
            "s3_uri_canonical": f"s3://{S3_BUCKET}/{S3_REF_CPIC}/cpic_gene-drug_pairs.parquet",
            "s3_uri_xlsx": f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/cpic_gene-drug_pairs.xlsx",
            "row_count": count,
            "etag": meta.get("etag"),
            "last_modified": meta.get("last_modified"),
            "content_sha256": meta.get("content_sha256"),
            "license": LICENSE_CPIC,
        }
    )


def _flatten_guideline_json(obj: Any, filename: str) -> List[Dict[str, Any]]:
    """Keep official top-level fields; explode recommendation arrays when present."""
    if not isinstance(obj, dict):
        return [{"source_file": filename, "payload_json": json.dumps(obj, ensure_ascii=False)}]
    recs = obj.get("recommendations") or obj.get("recommendation")
    base = {
        "source_file": filename,
        "id": obj.get("id"),
        "name": obj.get("name"),
        "source": obj.get("source"),
        "guideline": obj.get("guideline"),
        "url": obj.get("url") or obj.get("href"),
        "related_genes": json.dumps(obj.get("relatedGenes") or obj.get("genes") or [], ensure_ascii=False),
        "related_chemicals": json.dumps(obj.get("relatedChemicals") or obj.get("chemicals") or [], ensure_ascii=False),
        "payload_json": json.dumps(obj, ensure_ascii=False),
    }
    if isinstance(recs, list) and recs:
        rows = []
        for rec in recs:
            row = dict(base)
            if isinstance(rec, dict):
                row["recommendation_json"] = json.dumps(rec, ensure_ascii=False)
                row["recommendation_text"] = rec.get("recommendation") or rec.get("text") or rec.get("implication")
            else:
                row["recommendation_json"] = json.dumps(rec, ensure_ascii=False)
                row["recommendation_text"] = str(rec)
            rows.append(row)
        return rows
    base["recommendation_json"] = None
    base["recommendation_text"] = None
    return [base]


def ingest_pharmgkb_zips(
    manifest_tables: List[Dict[str, Any]], unavailable: List[Dict[str, Any]]
) -> None:
    for spec in PHARMGKB_ZIPS:
        url = f"{CLINPGX_DOWNLOAD}/{spec['filename']}"
        dest_zip = LOCAL_RAW_DIR / spec["filename"]
        dest_parquet = LOCAL_REF_DIR / spec["parquet"]
        print(f"ClinPGx download {spec['filename']}")
        try:
            meta = download_file(url, dest_zip)
        except HTTPError as exc:
            unavailable.append(
                {
                    "what": spec["description"],
                    "official_url": url,
                    "reason": f"HTTP {exc.code} — official Downloads page lists this file; not retrieved",
                }
            )
            print(f"  unavailable HTTP {exc.code}: {url}")
            continue
        except Exception as exc:
            unavailable.append(
                {
                    "what": spec["description"],
                    "official_url": url,
                    "reason": f"Download failed: {exc}",
                }
            )
            print(f"  failed: {exc}")
            continue

        extract_dir = LOCAL_RAW_DIR / dest_zip.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest_zip) as zf:
            zf.extractall(extract_dir)
            names = zf.namelist()
        print(f"  extracted {len(names)} files ({meta['bytes']:,} bytes)")

        tsv_files = [
            p
            for p in extract_dir.rglob("*")
            if p.suffix.lower() in {".tsv", ".csv"} and p.name.upper() != "LICENSE.TXT"
        ]
        json_files = list(extract_dir.rglob("*.json"))
        count = 0
        if tsv_files:
            import pandas as pd

            con = duckdb.connect(":memory:")
            try:
                frames = []
                for tsv in tsv_files:
                    try:
                        frames.append(
                            con.execute(
                                "SELECT * FROM read_csv_auto(?, SAMPLE_SIZE=-1)",
                                [str(tsv)],
                            ).fetchdf()
                        )
                    except Exception:
                        frames.append(
                            con.execute(
                                "SELECT * FROM read_csv_auto(?, delim='\t', header=true, ignore_errors=true)",
                                [str(tsv)],
                            ).fetchdf()
                        )
                if not frames:
                    raise RuntimeError("no readable TSV in zip")
                df = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
                if dest_parquet.exists():
                    dest_parquet.unlink()
                con.register("src", df)
                path_sql = str(dest_parquet.resolve().as_posix()).replace("'", "''")
                con.execute(f"COPY src TO '{path_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
                count = parquet_count(dest_parquet)
            finally:
                con.close()
        elif json_files:
            records: List[Dict[str, Any]] = []
            for jf in json_files:
                try:
                    obj = json.loads(jf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                records.extend(_flatten_guideline_json(obj, jf.name))
            count = write_records_parquet(records, dest_parquet)
        else:
            unavailable.append(
                {
                    "what": spec["description"],
                    "official_url": url,
                    "reason": f"Zip downloaded but contained no TSV/JSON tables: {names[:12]}",
                }
            )
            print("  no tabular files in zip")
            continue

        print(f"  wrote {count:,} rows -> {dest_parquet.name}")
        manifest_tables.append(
            {
                "name": dest_parquet.stem,
                "source_url": url,
                "source": "ClinPGx / PharmGKB official Downloads export",
                "description": spec["description"],
                "local_path": str(dest_parquet),
                "s3_uri": f"s3://{S3_BUCKET}/{S3_REF_PHARMGKB}/{dest_parquet.name}",
                "row_count": count,
                "license": LICENSE_CLINPGX,
                "zip_bytes": meta["bytes"],
            }
        )


def upload_outputs(manifest_tables: List[Dict[str, Any]], manifest_path: Path) -> None:
    import boto3

    s3 = boto3.client("s3")
    pairs_parquet = LOCAL_CPIC_DIR / "cpic_gene-drug_pairs.parquet"
    pairs_xlsx = LOCAL_CPIC_DIR / "cpic_gene-drug_pairs.xlsx"
    extras = [
        (pairs_parquet, f"{S3_DATA_PREFIX}/cpic_gene-drug_pairs.parquet", "application/octet-stream"),
        (pairs_xlsx, f"{S3_DATA_PREFIX}/cpic_gene-drug_pairs.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (pairs_parquet, f"{S3_REF_CPIC}/cpic_gene-drug_pairs.parquet", "application/octet-stream"),
        (manifest_path, f"{S3_REF_CPIC}/manifest.json", "application/json"),
    ]
    for local, key, ctype in extras:
        if not local.exists():
            continue
        s3.upload_file(str(local), S3_BUCKET, key, ExtraArgs={"ContentType": ctype})
        print(f"  uploaded s3://{S3_BUCKET}/{key}")

    for path in sorted(LOCAL_REF_DIR.glob("*.parquet")):
        prefix = S3_REF_PHARMGKB if path.name.startswith("pharmgkb_") else S3_REF_CPIC
        key = f"{prefix}/{path.name}"
        s3.upload_file(str(path), S3_BUCKET, key, ExtraArgs={"ContentType": "application/octet-stream"})
        print(f"  uploaded s3://{S3_BUCKET}/{key}")


def list_s3_prefix() -> List[Dict[str, Any]]:
    import boto3

    s3 = boto3.client("s3")
    listed: List[Dict[str, Any]] = []
    for prefix in (S3_DATA_PREFIX + "/", S3_REF_CPIC + "/", S3_REF_PHARMGKB + "/"):
        token = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": S3_BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents") or []:
                key = obj["Key"]
                if key.endswith("/") or "/raw/" in key:
                    continue
                if prefix == S3_DATA_PREFIX + "/" and not Path(key).name.startswith("cpic"):
                    continue
                listed.append(
                    {
                        "key": key,
                        "bytes": int(obj.get("Size") or 0),
                        "last_modified": obj["LastModified"].astimezone(timezone.utc).isoformat(),
                    }
                )
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    # de-dupe by key
    by_key = {row["key"]: row for row in listed}
    return [by_key[k] for k in sorted(by_key)]


def write_manifest(
    tables: List[Dict[str, Any]],
    unavailable: List[Dict[str, Any]],
    s3_objects: Optional[List[Dict[str, Any]]],
) -> Path:
    LOCAL_REF_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "retrieved_at": _utc_now(),
        "s3_bucket": S3_BUCKET,
        "s3_prefix": f"s3://{S3_BUCKET}/{S3_REF_CPIC}/",
        "s3_prefix_pharmgkb": f"s3://{S3_BUCKET}/{S3_REF_PHARMGKB}/",
        "legacy_pairs_s3": f"s3://{S3_BUCKET}/{S3_DATA_PREFIX}/cpic_gene-drug_pairs.parquet",
        "local_dir": str(LOCAL_REF_DIR),
        "license_attribution": [LICENSE_CPIC, LICENSE_CLINPGX],
        "docs": {
            "cpic_api": CPIC_API_DOCS,
            "cpic_api_and_database": CPIC_API_AND_DB,
            "clinpgx_downloads": CLINPGX_DOWNLOADS_PAGE,
        },
        "later_dashboard_use": (
            "Lambda loads gold/dashboard/data/cpic_gene-drug_pairs.parquet (PREFER_S3). "
            "Official phenotype parquet is gold/reference/cpic/cpic_diplotype_phenotype.parquet "
            "when present; otherwise the hardcoded 10-gene table remains. "
            "Do not invent mappings beyond official columns."
        ),
        "tables": tables,
        "not_available": unavailable,
        "s3_objects": s3_objects or [],
    }
    dest = LOCAL_REF_DIR / "manifest.json"
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def documented_unavailable() -> List[Dict[str, Any]]:
    return [
        {
            "what": "CPIC /phenotype root endpoint (as cited in older 5_pgx_analysis README)",
            "official_url": "https://api.cpicpgx.org/v1/phenotype",
            "reason": (
                "No /phenotype table in the current PostgREST catalog. "
                "Official equivalent is GET https://api.cpicpgx.org/v1/diplotype "
                "(and gene_result / gene_result_lookup)."
            ),
        },
        {
            "what": "Unversioned api.cpicpgx.org/{table} paths",
            "official_url": "https://api.cpicpgx.org/",
            "reason": "API moved under /v1/; root now serves Swagger UI only.",
        },
        {
            "what": "HTML CPIC/ClinPGx guideline manuscripts as structured recommendation rows",
            "official_url": "https://www.clinpgx.org/guideline/",
            "reason": (
                "Guideline prose is not an official tabular export. Use API /recommendation_view "
                "and ClinPGx guidelineAnnotations.json.zip instead of scraping."
            ),
        },
        {
            "what": "CPIC allele_frequency table (population frequencies)",
            "official_url": "https://api.cpicpgx.org/v1/allele_frequency",
            "reason": (
                "Official (~388k rows) but not required for phenotype / rsid / recommendation gaps; "
                "left out of this ingest to keep the gold snapshot focused."
            ),
        },
    ]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest official CPIC/ClinPGx reference tables to Parquet + S3")
    parser.add_argument("--no-upload", action="store_true", help="Write local Parquet/manifest only")
    args = parser.parse_args()

    LOCAL_REF_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    tables: List[Dict[str, Any]] = []
    unavailable = documented_unavailable()

    ingest_gene_drug_pairs(tables)
    ingest_cpic_tables(tables)
    build_allele_rsid(tables)
    ingest_pharmgkb_zips(tables, unavailable)

    manifest_path = write_manifest(tables, unavailable, s3_objects=None)
    print(f"Wrote local manifest {manifest_path}")

    s3_objects: Optional[List[Dict[str, Any]]] = None
    if not args.no_upload:
        print("Uploading to S3...")
        upload_outputs(tables, manifest_path)
        s3_objects = list_s3_prefix()
        manifest_path = write_manifest(tables, unavailable, s3_objects)
        # re-upload manifest with object listing
        import boto3

        boto3.client("s3").upload_file(
            str(manifest_path),
            S3_BUCKET,
            f"{S3_REF_CPIC}/manifest.json",
            ExtraArgs={"ContentType": "application/json"},
        )
        print(f"Re-uploaded manifest s3://{S3_BUCKET}/{S3_REF_CPIC}/manifest.json")

    print("\n=== Local tables ===")
    for row in tables:
        print(f"  {row['row_count']:>10,}  {row['name']}  <- {row['source_url']}")
    print("\n=== Not ingested / not available as tables ===")
    for row in unavailable:
        print(f"  - {row['what']}: {row['official_url']}")
        print(f"    {row['reason']}")
    if s3_objects:
        print("\n=== S3 objects ===")
        for obj in s3_objects:
            print(f"  {obj['bytes']:>12,}  s3://{S3_BUCKET}/{obj['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
