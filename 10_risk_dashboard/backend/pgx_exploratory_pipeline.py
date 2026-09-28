"""
PGx card analysis pipeline for consumer raw DNA.

    User uploads Ancestry (or 23andMe / MyHeritage / unphased VCF)
        -> DuckDB writes Snappy Parquet
        -> SQL interval join against CPIC target Parquet
        -> PharmGKB / ClinPGx REST (literature and VIP)
        -> CPIC API star-allele map and associated drugs
        -> exploratory report and clinical-test referral

Unphased array rows never become a diplotype, metabolizer status, or dose change.
Missing target sites stay "Data Not Present in File".
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

CLINPGX_API = "https://api.clinpgx.org/v1"
CPIC_API = "https://api.cpicpgx.org/v1"
PHARMVAR_API = "https://www.pharmvar.org/api-service"
PIPELINE_ID = "pgx-exploratory-pipeline-v1"
MAX_API_GENES = 3
HTTP_TIMEOUT = 8
MIN_REQUEST_GAP = 0.55

CLINICAL_PGX_URL = "https://cpicpgx.org/guidelines/"
NSGC_URL = "https://findageneticcounselor.nsgc.org/"

REFERRAL_NEXT_STEP = (
    "Order a clinical pharmacogenomic (PGx) panel. A CLIA/CAP-certified test "
    "provides phased diplotype calling, copy-number validation, and actionable "
    "prescription guidance."
)

_last_http_at = 0.0
_http_cache: Dict[str, Any] = {}


def _norm_chrom(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text.upper()


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _sql_path(path: str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _chrom_label(value: Any) -> str:
    chrom = _norm_chrom(value)
    if chrom == "23":
        chrom = "X"
    elif chrom == "24":
        chrom = "Y"
    if not chrom:
        return ""
    return "chr" + chrom


def _connect_duckdb():
    import duckdb
    return duckdb.connect()


def rows_to_user_parquet(rows: Sequence[Mapping[str, Any]], dest_path: str) -> int:
    """Write parsed array rows to Snappy Parquet with the Ancestry interval columns."""
    con = _connect_duckdb()
    try:
        con.execute(
            """
            CREATE TABLE user_rows (
                rsid VARCHAR,
                chromosome VARCHAR,
                position BIGINT,
                genotype VARCHAR
            )
            """
        )
        payload = [
            (
                str(row.get("rsid") or ""),
                _norm_chrom(row.get("chromosome")),
                _as_int(row.get("position")),
                str(row.get("genotype") or ""),
            )
            for row in rows
            if row.get("rsid")
        ]
        if payload:
            con.executemany("INSERT INTO user_rows VALUES (?, ?, ?, ?)", payload)
        con.execute(
            f"""
            COPY (
                SELECT
                    rsid,
                    'chr' || CASE chromosome
                        WHEN '23' THEN 'X'
                        WHEN '24' THEN 'Y'
                        ELSE chromosome
                    END AS Chromosome,
                    position AS Start,
                    (position + 1) AS End,
                    genotype
                FROM user_rows
                WHERE position IS NOT NULL
                  AND chromosome IS NOT NULL
                  AND chromosome <> ''
            ) TO '{_sql_path(dest_path)}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
            """
        )
        count = con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(dest_path)}')"
        ).fetchone()
        return int(count[0] if count else 0)
    finally:
        con.close()


def ancestry_text_to_parquet(raw_txt_path: str, user_parquet_path: str) -> Dict[str, Any]:
    """Read an Ancestry-style text file and write user_dna.parquet with DuckDB."""
    sep = "\t"
    width = 0
    with open(raw_txt_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.count(",") > stripped.count("\t"):
                sep = ","
                width = len(stripped.split(","))
            else:
                width = len(stripped.split("\t"))
            break
    genotype_sql = "column3 || column4" if width >= 5 else "column3"
    con = _connect_duckdb()
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    column0 AS rsid,
                    'chr' || CASE upper(regexp_replace(column1, '^chr', '', 'i'))
                        WHEN '23' THEN 'X'
                        WHEN '24' THEN 'Y'
                        ELSE regexp_replace(column1, '^chr', '', 'i')
                    END AS Chromosome,
                    column2::BIGINT AS Start,
                    (column2::BIGINT + 1) AS End,
                    {genotype_sql} AS genotype
                FROM read_csv(
                    '{_sql_path(raw_txt_path)}',
                    sep='{sep}',
                    comment='#',
                    header=false,
                    all_varchar=true
                )
                WHERE column0 ILIKE 'rs%'
                  AND regexp_matches(column2, '^[0-9]+$')
            ) TO '{_sql_path(user_parquet_path)}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
            """
        )
        count = con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(user_parquet_path)}')"
        ).fetchone()
        return {
            "engine": "duckdb-parquet",
            "parquet": True,
            "rowCount": int(count[0] if count else 0),
            "columns": width,
        }
    finally:
        con.close()


def write_cpic_targets_parquet(intervals: Mapping[str, Mapping[str, Any]], dest_path: str) -> int:
    """CPIC gene intervals as Parquet: Chromosome, gene_start, gene_end, gene_symbol."""
    con = _connect_duckdb()
    try:
        con.execute(
            """
            CREATE TABLE cpic_targets (
                Chromosome VARCHAR,
                gene_start BIGINT,
                gene_end BIGINT,
                gene_symbol VARCHAR
            )
            """
        )
        payload = []
        for gene, interval in intervals.items():
            chrom = _chrom_label(interval.get("chrom") or interval.get("Chromosome"))
            start = _as_int(interval.get("start") if interval.get("start") is not None else interval.get("gene_start"))
            stop = _as_int(interval.get("stop") if interval.get("stop") is not None else interval.get("gene_end"))
            if not chrom or start is None or stop is None:
                continue
            lo, hi = (start, stop) if start <= stop else (stop, start)
            payload.append((chrom, lo, hi, str(gene).upper()))
        if payload:
            con.executemany("INSERT INTO cpic_targets VALUES (?, ?, ?, ?)", payload)
        con.execute(
            f"COPY cpic_targets TO '{_sql_path(dest_path)}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')"
        )
        return len(payload)
    finally:
        con.close()


def join_user_to_cpic_targets(user_parquet: str, cpic_parquet: str) -> List[Dict[str, Any]]:
    """Interval join of the user Parquet against the CPIC target Parquet."""
    con = _connect_duckdb()
    try:
        found = con.execute(
            f"""
            SELECT u.Chromosome, u.rsid, u.genotype, c.gene_symbol
            FROM read_parquet('{_sql_path(user_parquet)}') u
            JOIN read_parquet('{_sql_path(cpic_parquet)}') c
              ON u.Chromosome = c.Chromosome
             AND u.Start BETWEEN c.gene_start AND c.gene_end
            """
        ).fetchall()
    finally:
        con.close()
    return [
        {"chromosome": row[0], "rsid": row[1], "genotype": row[2], "gene": row[3]}
        for row in found
    ]


def ingest_genotype_rows(genotypes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """DuckDB writes the upload to ephemeral Snappy Parquet. Raw text is not retained."""
    import tempfile
    rows = []
    for item in genotypes or []:
        rsid = str(item.get("rsid") or "").strip()
        if not rsid:
            continue
        chrom = _norm_chrom(item.get("chromosome") or item.get("chrom") or item.get("chr"))
        if chrom == "23":
            chrom = "X"
        elif chrom == "24":
            chrom = "Y"
        rows.append({
            "rsid": rsid,
            "chromosome": chrom,
            "position": _as_int(item.get("position") or item.get("pos")),
            "genotype": str(item.get("genotype") or "").strip(),
        })
    result: Dict[str, Any] = {
        "engine": "duckdb-parquet",
        "status": "ok" if rows else "empty",
        "rowCount": len(rows),
        "parquet": False,
        "parquetPath": None,
        "rows": rows,
    }
    if not rows:
        return result
    handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    handle.close()
    try:
        written = rows_to_user_parquet(rows, handle.name)
        result["parquet"] = True
        result["parquetPath"] = handle.name
        result["parquetRows"] = written
    except ImportError:
        result["engine"] = "python-rows"
        result["note"] = "DuckDB is not installed in this runtime. Rows stayed in memory."
        try:
            os.remove(handle.name)
        except OSError:
            pass
    except Exception as exc:
        result["engine"] = "python-rows"
        result["note"] = f"DuckDB Parquet write failed ({exc}). Rows stayed in memory."
        try:
            os.remove(handle.name)
        except OSError:
            pass
    return result


def annotate_coordinates(genotypes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """VariantAnnotation-style step: keep rsid plus genomic coordinate from the file."""
    rows = []
    with_position = 0
    for item in genotypes or []:
        rsid = str(item.get("rsid") or item.get("id") or "").strip()
        if not rsid:
            continue
        chrom = _norm_chrom(item.get("chromosome") or item.get("chrom") or item.get("chr"))
        pos = _as_int(item.get("position") or item.get("pos"))
        genotype = str(item.get("genotype") or "").strip()
        row = {"rsid": rsid, "genotype": genotype, "chromosome": chrom or None, "position": pos}
        if chrom and pos is not None:
            with_position += 1
            row["coordinate"] = f"{chrom}:{pos}"
        else:
            row["coordinate"] = None
            row["coordinateStatus"] = "Coordinate not present in file"
        rows.append(row)
    return {
        "engine": "python-variant-annotation",
        "status": "ok" if rows else "empty",
        "rowCount": len(rows),
        "withPosition": with_position,
        "missingPosition": len(rows) - with_position,
        "rows": rows,
    }


def _point_in_interval(row: Mapping[str, Any], interval: Mapping[str, Any]) -> bool:
    chrom = _norm_chrom(row.get("chromosome"))
    pos = row.get("position")
    if not chrom or pos is None or not interval:
        return False
    target = _norm_chrom(interval.get("chrom"))
    start = _as_int(interval.get("start"))
    stop = _as_int(interval.get("stop"))
    if not target or start is None or stop is None:
        return False
    lo, hi = (start, stop) if start <= stop else (stop, start)
    return chrom == target and lo <= int(pos) <= hi


def coverage_check(
    annotated: Mapping[str, Any],
    gene_calls: Sequence[Mapping[str, Any]],
    gene_intervals: Optional[Mapping[str, Mapping[str, Any]]] = None,
    interval_hits: Optional[Mapping[str, int]] = None,
    engine: str = "duckdb-parquet",
) -> Dict[str, Any]:
    """Coverage from official sites, plus a DuckDB interval join when CPIC targets exist."""
    by_rsid = {str(row.get("rsid") or "").lower(): row for row in annotated.get("rows") or []}
    genes = []
    present = 0
    missing = 0
    for call in gene_calls or []:
        gene = str(call.get("gene") or "").upper()
        sites_out = []
        for site in call.get("observedSites") or []:
            rsid = str(site.get("rsid") or "")
            status = site.get("status")
            if status == "Data Not Present in File":
                missing += 1
            else:
                present += 1
            coord = by_rsid.get(rsid.lower()) or {}
            sites_out.append({
                "rsid": rsid,
                "genotype": site.get("genotype"),
                "status": status,
                "variantAlleleObserved": bool(site.get("variantAlleleObserved")),
                "coordinate": coord.get("coordinate"),
            })
        interval = (gene_intervals or {}).get(gene) or {}
        if interval_hits is not None:
            points_inside = int(interval_hits.get(gene, 0))
        else:
            points_inside = sum(1 for row in annotated.get("rows") or [] if _point_in_interval(row, interval))
        sites_present = sum(1 for site in sites_out if site["status"] != "Data Not Present in File")
        sites_missing = sum(1 for site in sites_out if site["status"] == "Data Not Present in File")
        total_sites = sites_present + sites_missing
        coverage_pct = int(round(100 * sites_present / total_sites)) if total_sites else 0
        genes.append({
            "gene": gene,
            "sites": sites_out,
            "sitesPresent": sites_present,
            "sitesMissing": sites_missing,
            "coveragePercent": coverage_pct,
            "coverageStatus": "Partial Panel (Microarray)" if sites_missing else "Measured target sites present",
            "coverageText": f"Gene Coverage: {coverage_pct}% of known {gene} variant locations present.",
            "unphased": True,
            "pointsInGeneInterval": points_inside,
            "intervalBuild": interval.get("build"),
        })
    return {
        "engine": engine,
        "status": "ok",
        "genesEvaluated": len(genes),
        "sitesPresent": present,
        "sitesMissing": missing,
        "genes": genes,
        "note": "Uncovered target rsids stay Data Not Present in File. They are not called *1.",
    }


def _default_http_get(url: str, headers: Optional[Mapping[str, str]] = None) -> Any:
    global _last_http_at
    if url in _http_cache:
        return _http_cache[url]
    wait = MIN_REQUEST_GAP - (time.monotonic() - _last_http_at)
    if wait > 0:
        time.sleep(wait)
    req_headers = {"User-Agent": "pgx-dashboard/1.0", "Accept": "application/json"}
    if headers:
        req_headers.update(dict(headers))
    request = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        _last_http_at = time.monotonic()
    _http_cache[url] = payload
    return payload


def _clinpgx_data(payload: Any) -> Any:
    if isinstance(payload, dict):
        return payload.get("data")
    return payload


def _first(payload: Any) -> Optional[Dict[str, Any]]:
    data = _clinpgx_data(payload)
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        return data
    return None


def _clip(text: Any, limit: int = 420) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "…"


def fetch_pharmgkb_evidence(
    genes: Sequence[str],
    hit_rsids: Mapping[str, Sequence[str]],
    http_get: Callable[..., Any],
) -> Dict[str, Any]:
    """PharmGKB REST stage. The live host is ClinPGx; api.pharmgkb.org was retired."""
    evidence = []
    errors = []
    for gene in list(genes)[:MAX_API_GENES]:
        record: Dict[str, Any] = {"gene": gene, "host": "api.clinpgx.org"}
        try:
            listed = _first(http_get(f"{CLINPGX_API}/data/gene?view=min&symbol={gene}"))
            gene_id = (listed or {}).get("id")
            if gene_id:
                detail = _first(http_get(f"{CLINPGX_API}/data/gene/{gene_id}?view=base")) or {}
                record["vipId"] = detail.get("vipId")
                record["vipTier"] = detail.get("vipTier")
                record["vipSummary"] = _clip(detail.get("vipSummary"))
                citation = detail.get("vipCitation")
                if isinstance(citation, dict):
                    record["vipCitation"] = {
                        "title": citation.get("title") or citation.get("name"),
                        "url": citation.get("_url") or citation.get("url"),
                    }
                elif citation:
                    record["vipCitation"] = {"title": _clip(citation, 180)}
                chrom = detail.get("chr")
                start = detail.get("chrStartPosB37")
                stop = detail.get("chrStopPosB37")
                if chrom and start and stop:
                    record["interval"] = {
                        "chrom": _norm_chrom(chrom),
                        "start": _as_int(start),
                        "stop": _as_int(stop),
                        "build": "GRCh37",
                    }
            rsids = list(hit_rsids.get(gene) or [])[:1]
            if rsids:
                variant = _first(http_get(f"{CLINPGX_API}/data/variant?view=base&symbol={rsids[0]}")) or {}
                record["variant"] = {
                    "rsid": rsids[0],
                    "id": variant.get("id"),
                    "clinicalSignificance": variant.get("clinicalSignificance"),
                    "url": f"https://www.clinpgx.org/variant/{rsids[0]}",
                }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{gene}: {exc}")
            record["error"] = str(exc)
        evidence.append(record)
    status = "ok" if evidence and not errors else ("partial" if evidence else "unavailable")
    if errors and evidence:
        status = "partial"
    return {
        "status": status if genes else "skipped",
        "host": "api.clinpgx.org",
        "formerHost": "api.pharmgkb.org",
        "note": "Literature and VIP text are evidence citations, not a diplotype or dose recommendation.",
        "genes": evidence,
        "errors": errors,
    }


def _cpic_names(candidates: Sequence[str]) -> str:
    cleaned = []
    for name in candidates:
        text = str(name or "").strip()
        if not text:
            continue
        if not text.startswith("*"):
            text = "*" + text
        cleaned.append(text)
    # PostgREST in.() needs quoted values when they contain *.
    inner = ",".join('"' + name.replace('"', "") + '"' for name in cleaned[:12])
    return f"in.({inner})" if inner else ""


def fetch_star_map_and_drugs(
    gene_calls: Sequence[Mapping[str, Any]],
    http_get: Callable[..., Any],
    pharmvar_api_key: Optional[str] = None,
    local_pairs: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> Dict[str, Any]:
    """CPIC gene–drug pairs plus star-allele names. PharmVar live calls need an API key."""
    genes_out = []
    errors = []
    key = pharmvar_api_key if pharmvar_api_key is not None else os.environ.get("PHARMVAR_API_KEY")
    targets = [call for call in gene_calls if call.get("clinicalNudge") or call.get("candidateAlleles")]
    for call in targets[:MAX_API_GENES]:
        gene = str(call.get("gene") or "").upper()
        candidates = [str(name) for name in (call.get("candidateAlleles") or [])]
        record: Dict[str, Any] = {
            "gene": gene,
            "candidateAlleles": candidates,
            "diplotypeAssigned": False,
            "pharmvarIds": [],
            "associatedDrugs": [],
        }
        try:
            pairs = http_get(
                f"{CPIC_API}/pair_view?genesymbol=eq.{gene}&select=genesymbol,drugname,cpiclevel&limit=12"
            )
            if isinstance(pairs, list):
                record["associatedDrugs"] = [
                    {"drug": row.get("drugname"), "cpicLevel": row.get("cpiclevel")}
                    for row in pairs
                    if row.get("drugname")
                ]
            name_filter = _cpic_names(candidates)
            if name_filter:
                encoded_names = urllib.parse.quote(name_filter, safe='(),."*')
                alleles = http_get(
                    f"{CPIC_API}/allele_definition?genesymbol=eq.{gene}&name={encoded_names}"
                    "&select=name,pharmvarid&limit=12"
                )
                if isinstance(alleles, list):
                    record["pharmvarIds"] = [
                        {"allele": row.get("name"), "pharmvarId": row.get("pharmvarid")}
                        for row in alleles
                        if row.get("name")
                    ]
            if key and candidates:
                allele_name = gene + candidates[0]
                try:
                    live = http_get(
                        f"{PHARMVAR_API}/alleles/{urllib.parse.quote(allele_name)}",
                        {"Authorization": f"Bearer {key}"},
                    )
                    record["pharmvarLive"] = {"status": "ok", "allele": allele_name, "received": bool(live)}
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                    record["pharmvarLive"] = {"status": "unavailable", "error": str(exc)}
            else:
                record["pharmvarLive"] = {
                    "status": "not-configured",
                    "detail": (
                        "Star-allele names were checked on the CPIC allele_definition table, "
                        "which carries PharmVar IDs. Live PharmVar was not called because PHARMVAR_API_KEY is not set."
                    ),
                }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{gene}: {exc}")
            record["error"] = str(exc)
        if not record["associatedDrugs"] and local_pairs:
            record["associatedDrugs"] = [
                {"drug": row.get("drug"), "cpicLevel": row.get("cpic_level") or row.get("cpicLevel")}
                for row in (local_pairs.get(gene) or [])[:12]
                if row.get("drug")
            ]
            record["drugSource"] = "local-cpic-pairs"
        else:
            record["drugSource"] = "cpic-api"
        genes_out.append(record)
    return {
        "status": "partial" if errors else "ok",
        "diplotypeAssigned": False,
        "genes": genes_out,
        "errors": errors,
        "boundary": "CPIC defines clinical guidelines for these genes when phased. No dose adjustment is recommended from this file.",
    }


def _local_drug_fallback(
    gene_calls: Sequence[Mapping[str, Any]],
    local_pairs: Optional[Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    genes_out = []
    for call in gene_calls or []:
        if not (call.get("clinicalNudge") or call.get("candidateAlleles")):
            continue
        gene = str(call.get("gene") or "").upper()
        drugs = [
            {"drug": row.get("drug"), "cpicLevel": row.get("cpic_level") or row.get("cpicLevel")}
            for row in ((local_pairs or {}).get(gene) or [])[:12]
            if row.get("drug")
        ]
        genes_out.append({
            "gene": gene,
            "candidateAlleles": list(call.get("candidateAlleles") or []),
            "diplotypeAssigned": False,
            "pharmvarIds": [],
            "associatedDrugs": drugs,
            "drugSource": "local-cpic-pairs",
            "pharmvarLive": {
                "status": "not-requested",
                "detail": "Live CPIC and PharmVar lookups were not requested.",
            },
        })
    return {
        "status": "local-only",
        "diplotypeAssigned": False,
        "genes": genes_out,
        "errors": [],
        "boundary": "CPIC defines clinical guidelines for these genes when phased. No dose adjustment is recommended from this file.",
    }


def build_findings(
    gene_calls: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    pharmgkb: Mapping[str, Any],
    star_map: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    coverage_by_gene = {row.get("gene"): row for row in coverage.get("genes") or []}
    evidence_by_gene = {row.get("gene"): row for row in pharmgkb.get("genes") or []}
    star_by_gene = {row.get("gene"): row for row in star_map.get("genes") or []}
    findings = []
    for call in gene_calls or []:
        gene = str(call.get("gene") or "").upper()
        covered = coverage_by_gene.get(gene) or {}
        hits = [site for site in (covered.get("sites") or []) if site.get("variantAlleleObserved")]
        evidence = evidence_by_gene.get(gene) or {}
        stars = star_by_gene.get(gene) or {}
        findings.append({
            "gene": gene,
            "summary": call.get("exploratorySummary"),
            "diplotype": None,
            "phenotype": None,
            "candidateAlleles": list(call.get("candidateAlleles") or []),
            "observedSites": covered.get("sites") or list(call.get("observedSites") or []),
            "clinicalNudge": bool(call.get("clinicalNudge")),
            "cnvMeasured": False,
            "coveragePercent": covered.get("coveragePercent"),
            "coverageStatus": covered.get("coverageStatus"),
            "coverageText": covered.get("coverageText"),
            "cpicBoundary": call.get("cpicBoundary"),
            "limitations": list(call.get("limitations") or []),
            "pharmgkb": {
                "vipTier": evidence.get("vipTier"),
                "vipCitation": evidence.get("vipCitation"),
                "variant": evidence.get("variant"),
            } if evidence else None,
            "associatedDrugs": stars.get("associatedDrugs") or [],
            "guidelineReferences": [
                f"CPIC guideline reference available for {row.get('drug')}."
                for row in (stars.get("associatedDrugs") or [])
                if row.get("drug")
            ],
            "pharmvarIds": stars.get("pharmvarIds") or [],
        })
    return findings


def clinical_nudges(findings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    nudges = []
    for finding in findings:
        if not finding.get("clinicalNudge"):
            continue
        hits = [site for site in (finding.get("observedSites") or []) if site.get("variantAlleleObserved")]
        rsids = [str(site.get("rsid")) for site in hits if site.get("rsid")]
        nudges.append({
            "gene": finding.get("gene"),
            "title": f"Exploratory finding: {finding.get('gene')} gene region",
            "status": (
                f"{len(rsids)} variant{'s' if len(rsids) != 1 else ''} detected in uploaded file"
                + (f" ({', '.join(rsids)})" if rsids else "")
            ),
            "summary": finding.get("summary"),
            "coverageText": finding.get("coverageText"),
            "coverageStatus": finding.get("coverageStatus"),
            "coveragePercent": finding.get("coveragePercent"),
            "limitation": (
                "Unphased consumer file. This upload cannot confirm whether these variants "
                "sit on the same chromosome or opposite chromosomes. CPIC guidelines require "
                "phased diplotype calling and copy-number validation."
            ),
            "nextStep": REFERRAL_NEXT_STEP,
            "clinicalAction": {
                "recommendation": "Follow-On Clinical Test Required",
                "reason": "Consumer array files cannot phase mutations or measure copy number variants.",
                "suggestedTest": "CLIA/CAP-Certified Pharmacogenomics Panel",
            },
            "observedRsids": rsids,
            "pharmgkbUrl": f"https://www.clinpgx.org/gene/{finding.get('gene')}",
            "links": [
                {"label": "Learn about clinical PGx testing", "url": CLINICAL_PGX_URL},
                {"label": "Find a genetic counselor", "url": NSGC_URL},
            ],
        })
    return nudges


def run_exploratory_pipeline(
    genotypes: Sequence[Mapping[str, Any]],
    gene_calls: Sequence[Mapping[str, Any]],
    local_pairs: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    fetch: bool = True,
    http_get: Optional[Callable[..., Any]] = None,
    pharmvar_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the card pipeline. fetch=False keeps the run local for tests."""
    ingested = ingest_genotype_rows(genotypes)
    annotated = annotate_coordinates(ingested.get("rows") or [])
    engine = ingested.get("engine") or "duckdb-parquet"
    cpic_parquet = None
    coverage = coverage_check(annotated, gene_calls, engine=engine)
    nudge_genes = [str(call.get("gene")) for call in gene_calls if call.get("clinicalNudge")]
    hit_rsids: Dict[str, List[str]] = {}
    for call in gene_calls:
        gene = str(call.get("gene") or "")
        hit_rsids[gene] = [
            str(site.get("rsid"))
            for site in (call.get("observedSites") or [])
            if site.get("variantAlleleObserved") and site.get("rsid")
        ]
    intervals: Dict[str, Dict[str, Any]] = {}
    if fetch and nudge_genes:
        getter = http_get or _default_http_get
        pharmgkb = fetch_pharmgkb_evidence(nudge_genes, hit_rsids, getter)
        for row in pharmgkb.get("genes") or []:
            if row.get("interval"):
                intervals[str(row.get("gene"))] = row["interval"]
        if intervals and ingested.get("parquetPath"):
            import tempfile
            handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
            handle.close()
            cpic_parquet = handle.name
            write_cpic_targets_parquet(intervals, cpic_parquet)
            joined = join_user_to_cpic_targets(ingested["parquetPath"], cpic_parquet)
            hits: Dict[str, int] = {}
            for row in joined:
                gene_name = str(row.get("gene") or "")
                hits[gene_name] = hits.get(gene_name, 0) + 1
            coverage = coverage_check(annotated, gene_calls, intervals, hits, engine="duckdb-parquet")
        elif intervals:
            coverage = coverage_check(annotated, gene_calls, intervals, engine=engine)
        star_map = fetch_star_map_and_drugs(gene_calls, getter, pharmvar_api_key, local_pairs)
    else:
        pharmgkb = {
            "status": "not-requested",
            "host": "api.clinpgx.org",
            "genes": [],
            "errors": [],
            "note": "Live PharmGKB / ClinPGx lookup was not requested.",
        }
        star_map = _local_drug_fallback(gene_calls, local_pairs)
    findings = build_findings(gene_calls, coverage, pharmgkb, star_map)
    nudges = clinical_nudges(findings)
    for path in (ingested.get("parquetPath"), cpic_parquet):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return {
        "id": PIPELINE_ID,
        "architecture": [
            "User uploads Ancestry raw text",
            "DuckDB ingestion to Snappy Parquet",
            "DuckDB interval join against CPIC target Parquet",
            "PharmGKB REST: VIP annotations and literature",
            "CPIC API: star-allele map check and associated drugs",
            "Exploratory report and clinical test referral",
        ],
        "technicalNotice": {
            "isPhased": False,
            "cnvSupported": False,
            "dataSource": "Direct-to-consumer array",
        },
        "stages": {
            "upload": {
                "status": ingested.get("status"),
                "rowCount": ingested.get("rowCount"),
                "engine": ingested.get("engine"),
                "parquet": ingested.get("parquet"),
                "note": ingested.get("note"),
            },
            "coordinates": {
                "status": annotated["status"],
                "engine": annotated["engine"],
                "withPosition": annotated["withPosition"],
                "missingPosition": annotated["missingPosition"],
            },
            "coverage": {
                "status": coverage["status"],
                "engine": coverage["engine"],
                "genesEvaluated": coverage["genesEvaluated"],
                "sitesPresent": coverage["sitesPresent"],
                "sitesMissing": coverage["sitesMissing"],
                "note": coverage["note"],
            },
            "pharmgkb": {
                "status": pharmgkb.get("status"),
                "host": pharmgkb.get("host"),
                "formerHost": pharmgkb.get("formerHost"),
                "note": pharmgkb.get("note"),
                "errors": pharmgkb.get("errors") or [],
            },
            "cpicPharmvar": {
                "status": star_map.get("status"),
                "diplotypeAssigned": False,
                "boundary": star_map.get("boundary"),
                "errors": star_map.get("errors") or [],
            },
            "report": {"status": "ok", "findingCount": len(findings), "nudgeCount": len(nudges)},
        },
        "findings": findings,
        "clinicalNudges": nudges,
        "referral": {
            "headline": "Order a clinical pharmacogenomic (PGx) panel",
            "detail": REFERRAL_NEXT_STEP,
            "doNotAlterMedications": (
                "Do not start, stop, or adjust the dosage of any prescription medication "
                "based on this exploratory report."
            ),
            "links": [
                {"label": "Learn about clinical PGx testing", "url": CLINICAL_PGX_URL},
                {"label": "Find a genetic counselor", "url": NSGC_URL},
            ],
        },
    }
