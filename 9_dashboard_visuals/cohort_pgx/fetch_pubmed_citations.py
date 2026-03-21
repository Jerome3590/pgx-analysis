#!/usr/bin/env python3
"""
Fetch PubMed citations for PGx cohort gene results (quality assurance / evidence support).

Follows the same search workflow as lit_review/lit_review.qmd:
  - Date range: last 5 years  (e.g. "2021:2026[PDAT]")
  - Two complementary queries per gene (mirrors search_pubmed_all pattern):
      1. Gene + pharmacogenomics MeSH  (general clinical evidence)
      2. Gene + cohort context keyword  (opioid / emergency department)
  - XML efetch to capture PMC IDs alongside title / authors / year / journal
  - BioC JSON full-text URL when PMC ID present:
      https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmc_id}/unicode

Results are saved as pubmed_citations.json in the same networks directory as
network_topology.html so sync_cohort_pgx_to_s3.py picks them up automatically.

NCBI rate limits (polite use):
  - Without API key : 3 req/s  → delay = 0.34 s
  - With API key    : 10 req/s → delay = 0.11 s

Usage:
    python fetch_pubmed_citations.py \\
        --cohort opioid_ed --age-band 25-44 \\
        --reports /path/to/reports/opioid_ed_25_44_vip_reports.json \\
        --output-dir /path/to/networks/opioid_ed/25_44 \\
        [--bin medium] [--ncbi-api-key <key>] [--force]
"""

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# NCBI E-utilities endpoints
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# PubMed article and PMC full-text URLs (matches lit_review download_pmc_article pattern)
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/"
PMC_BIOC_URL       = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmc_id}/unicode"

REQUEST_DELAY_NO_KEY  = 0.34   # 3 req/s  (NCBI polite limit without API key)
REQUEST_DELAY_WITH_KEY = 0.11  # 10 req/s (with API key)
LIT_REVIEW_YEARS      = 5      # date range: last N years (matches search_pubmed_all)
BATCH_SIZE            = 50     # XML fetch batch (matches search_pubmed_all batch_size)

# Cohort-specific context keywords (mirrors lit_review section 3 / PGx workflow)
_COHORT_CONTEXT: Dict[str, str] = {
    "opioid_ed":     "opioid",
    "non_opioid_ed": "emergency department",
}
_DEFAULT_CONTEXT = "pharmacogenomics"

OUTPUT_FILENAME = "pubmed_citations.json"


# ---------------------------------------------------------------------------
# Date range helper  (mirrors search_pubmed_all date filter)
# ---------------------------------------------------------------------------

def _date_range_filter() -> str:
    """Return NCBI PDAT filter for the last LIT_REVIEW_YEARS years."""
    current_year = datetime.now().year
    start_year   = current_year - LIT_REVIEW_YEARS
    return f"{start_year}:{current_year}[PDAT]"


# ---------------------------------------------------------------------------
# NCBI esearch  (mirrors entrez_search + use_history)
# ---------------------------------------------------------------------------

def _esearch(
    query: str,
    retmax: int,
    api_key: Optional[str],
    session: requests.Session,
    delay: float,
) -> List[str]:
    """
    Run esearch with date range and return list of PMIDs.
    Appends the 5-year PDAT filter exactly as search_pubmed_all() does.
    """
    dated_query = f"{query} AND {_date_range_filter()}"
    params: Dict[str, Any] = {
        "db":     "pubmed",
        "term":   dated_query,
        "retmax": retmax,
        "retmode": "json",
        "sort":   "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    time.sleep(delay)
    try:
        resp = session.get(NCBI_ESEARCH_URL, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        logger.warning("PubMed esearch failed: query=%r  error=%s", dated_query, exc)
        return []


# ---------------------------------------------------------------------------
# NCBI efetch XML  (mirrors entrez_fetch rettype="xml" + read_xml pattern)
# ---------------------------------------------------------------------------

def _efetch_xml(
    pmids: List[str],
    api_key: Optional[str],
    session: requests.Session,
    delay: float,
) -> ET.Element:
    """Fetch PubMed records as XML for a list of PMIDs."""
    params: Dict[str, Any] = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    time.sleep(delay)
    try:
        resp = session.get(NCBI_EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        return ET.fromstring(resp.text)
    except Exception as exc:
        logger.warning("PubMed efetch XML failed: pmids=%s  error=%s", pmids[:5], exc)
        return ET.Element("PubmedArticleSet")


# ---------------------------------------------------------------------------
# XML → citation dict  (mirrors article_details map_df in search_pubmed_all)
# ---------------------------------------------------------------------------

def _parse_article_xml(article: ET.Element) -> Optional[Dict[str, Any]]:
    """
    Extract title, authors, pubdate, pmc_id, pmid from one <PubmedArticle> node.
    Field selection mirrors the lit_review R code:
        title, authors (LastName), pubdate (Year), pmc_id (ArticleId[@IdType='pmc'])
    """
    def _txt(path: str) -> str:
        node = article.find(path)
        return (node.text or "").strip() if node is not None else ""

    title   = _txt(".//ArticleTitle")
    year    = _txt(".//PubDate/Year") or _txt(".//PubDate/MedlineDate")[:4]
    journal = _txt(".//Journal/Title") or _txt(".//Journal/ISOAbbreviation")

    # Authors: up to 3 LastNames (matches authors = paste(authors, collapse=", ") in R)
    last_names = [n.text.strip() for n in article.findall(".//Author/LastName") if n.text][:3]

    # PMC ID (ArticleId[@IdType='pmc']) — may have "PMC" prefix or not
    pmc_raw = ""
    pmid    = ""
    for aid in article.findall(".//ArticleId"):
        id_type = (aid.get("IdType") or "").lower()
        if id_type == "pmc":
            pmc_raw = (aid.text or "").strip()
        elif id_type == "pubmed":
            pmid = (aid.text or "").strip()

    # Normalise PMC ID: ensure "PMC" prefix (matches R mutate pmc_id logic)
    if pmc_raw:
        pmc_id = pmc_raw if pmc_raw.upper().startswith("PMC") else f"PMC{pmc_raw}"
    else:
        pmc_id = ""

    if not (title or pmid):
        return None

    citation: Dict[str, Any] = {
        "pmid":    pmid,
        "title":   title.rstrip(". "),
        "authors": last_names,
        "journal": journal,
        "year":    year,
        "url":     f"{PUBMED_ARTICLE_URL}{pmid}/" if pmid else "",
        "pmc_id":  pmc_id,
    }
    # Full-text BioC JSON URL when PMC ID available (matches download_pmc_article pattern)
    if pmc_id:
        citation["full_text_url"] = PMC_BIOC_URL.format(pmc_id=pmc_id)

    return citation


# ---------------------------------------------------------------------------
# Batch fetch + parse  (mirrors search_pubmed_all batched loop)
# ---------------------------------------------------------------------------

def _fetch_citations_for_pmids(
    pmids: List[str],
    api_key: Optional[str],
    session: requests.Session,
    delay: float,
) -> List[Dict[str, Any]]:
    """
    Fetch and parse citations for a list of PMIDs in batches of BATCH_SIZE.
    Returns list of citation dicts ordered by original PMID list.
    """
    results: List[Dict[str, Any]] = []
    for start in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[start : start + BATCH_SIZE]
        root  = _efetch_xml(batch, api_key, session, delay)
        for article in root.findall("PubmedArticle"):
            cit = _parse_article_xml(article)
            if cit:
                results.append(cit)
    return results


# ---------------------------------------------------------------------------
# Per-gene query  (two searches, mirrors search_pubmed_all call pattern)
# ---------------------------------------------------------------------------

def _query_gene(
    gene_symbol: str,
    cohort_name: str,
    api_key: Optional[str],
    session: requests.Session,
    delay: float,
    retmax_pgx: int,
    retmax_ctx: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run two date-ranged PubMed queries for one gene:
      1. pharmacogenomics: "{gene}"[Gene Name] AND pharmacogenomics[MeSH Terms]
      2. cohort_context  : "{gene}"[Gene Name] AND "{context_kw}"[Title/Abstract]

    Fetches XML for PMC IDs; deduplicates across queries.
    """
    context_kw = _COHORT_CONTEXT.get(cohort_name, _DEFAULT_CONTEXT)

    # Query 1: gene + pharmacogenomics (mirrors "pharmacovigilance pharmacogenomics" searches)
    q_pgx  = f'"{gene_symbol}"[Gene Name] AND pharmacogenomics[MeSH Terms]'
    pgx_pmids = _esearch(q_pgx, retmax_pgx, api_key, session, delay)

    # Query 2: gene + cohort context (mirrors opioid / emergency department PGx searches)
    q_ctx  = f'"{gene_symbol}"[Gene Name] AND "{context_kw}"[Title/Abstract]'
    ctx_pmids_raw = _esearch(q_ctx, retmax_ctx, api_key, session, delay)
    # Deduplicate: remove PMIDs already captured by query 1
    pgx_set   = set(pgx_pmids)
    ctx_pmids = [p for p in ctx_pmids_raw if p not in pgx_set]

    # Batch XML fetch for all unique PMIDs
    all_pmids = list(dict.fromkeys(pgx_pmids + ctx_pmids))
    all_cits  = _fetch_citations_for_pmids(all_pmids, api_key, session, delay)
    by_pmid   = {c["pmid"]: c for c in all_cits if c.get("pmid")}

    def _ordered(pmids: List[str]) -> List[Dict[str, Any]]:
        return [by_pmid[p] for p in pmids if p in by_pmid]

    return {
        "pharmacogenomics": _ordered(pgx_pmids),
        "cohort_context":   _ordered(ctx_pmids),
    }


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_pubmed_citations(
    reports_file: Path,
    cohort_name: str,
    age_band: str,
    output_dir: Path,
    bin_name: Optional[str] = None,
    api_key: Optional[str] = None,
    retmax_pgx: int = 5,
    retmax_ctx: int = 3,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Load VIP reports JSON, query PubMed for each gene, save citations JSON.

    Output path : output_dir / pubmed_citations.json
    Idempotent  : skips when file already exists unless force=True.
    S3 sync     : sync_cohort_pgx_to_s3.py picks up all files under networks/
                  via rglob("*"), so per-bin density/{bin}/pubmed_citations.json
                  is also synced automatically.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / OUTPUT_FILENAME

    if not force and output_file.exists():
        logger.info("Citations already exist: %s (--force to re-fetch).", output_file)
        try:
            with open(output_file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass

    if not reports_file.exists():
        logger.error("VIP reports file not found: %s", reports_file)
        return {}

    with open(reports_file, encoding="utf-8") as fh:
        reports = json.load(fh)

    if not isinstance(reports, list) or not reports:
        logger.warning("VIP reports empty or not a list: %s", reports_file)
        return {}

    delay   = REQUEST_DELAY_WITH_KEY if api_key else REQUEST_DELAY_NO_KEY
    session = requests.Session()
    session.headers["User-Agent"] = (
        "pgx-analysis/1.0 (pharmacogenomics research; "
        "contact: see https://pubmed.ncbi.nlm.nih.gov/help/#api)"
    )

    context_kw = _COHORT_CONTEXT.get(cohort_name, _DEFAULT_CONTEXT)
    citations:  Dict[str, Any] = {}
    total = len(reports)

    for idx, report in enumerate(reports, 1):
        gene = (report.get("gene_symbol") or "").strip()
        if not gene:
            continue
        logger.info("[%d/%d] PubMed query: %s ...", idx, total, gene)
        gene_cits = _query_gene(
            gene_symbol=gene,
            cohort_name=cohort_name,
            api_key=api_key,
            session=session,
            delay=delay,
            retmax_pgx=retmax_pgx,
            retmax_ctx=retmax_ctx,
        )
        n_pgx = len(gene_cits["pharmacogenomics"])
        n_ctx = len(gene_cits["cohort_context"])
        logger.info("  %s: %d PGx, %d cohort-context citations", gene, n_pgx, n_ctx)
        citations[gene] = {
            "gene_name":                report.get("gene_name") or "",
            "vip_url":                  report.get("vip_url") or "",
            "cpic_gene":                bool(report.get("cpic_gene")),
            "has_cpic_dosing_guideline": bool(report.get("has_cpic_dosing_guideline")),
            "pharmacogenomics":          gene_cits["pharmacogenomics"],
            "cohort_context":            gene_cits["cohort_context"],
        }

    current_year = datetime.now().year
    result: Dict[str, Any] = {
        "cohort":          cohort_name,
        "age_band":        age_band,
        "bin":             bin_name,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "date_range":      f"{current_year - LIT_REVIEW_YEARS}–{current_year}",
        "context_keyword": context_kw,
        "genes_queried":   len(citations),
        "citations":       citations,
    }

    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    total_pgx = sum(len(v["pharmacogenomics"]) for v in citations.values())
    total_ctx = sum(len(v["cohort_context"])   for v in citations.values())
    logger.info(
        "Saved %s  genes=%d  pgx=%d  ctx=%d",
        output_file, len(citations), total_pgx, total_ctx,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch PubMed citations for PGx cohort gene results (lit_review workflow)"
    )
    parser.add_argument("--cohort",    required=True, help="Cohort name (opioid_ed, non_opioid_ed)")
    parser.add_argument("--age-band",  required=True, help="Age band (e.g. 25-44)")
    parser.add_argument("--bin", dest="bin_name", default=None,
                        help="Event density bin (low/medium/high/extreme)")
    parser.add_argument("--reports",    type=Path, required=True,
                        help="Path to VIP reports JSON (from fetch_vip_reports.py)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for pubmed_citations.json "
                             "(same networks/{cohort}/{age_band}[/density/{bin}] dir "
                             "as build_network_topology.py output)")
    parser.add_argument("--retmax-pgx",     type=int, default=5,
                        help="Max results per gene – pharmacogenomics query (default 5)")
    parser.add_argument("--retmax-context", type=int, default=3,
                        help="Max results per gene – cohort-context query (default 3)")
    parser.add_argument("--ncbi-api-key", default=None,
                        help="NCBI E-utilities API key (raises rate limit to 10 req/s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even when output already exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = fetch_pubmed_citations(
        reports_file=args.reports,
        cohort_name=args.cohort,
        age_band=args.age_band,
        output_dir=args.output_dir,
        bin_name=args.bin_name,
        api_key=args.ncbi_api_key,
        retmax_pgx=args.retmax_pgx,
        retmax_ctx=args.retmax_context,
        force=args.force,
    )
    n = len(result.get("citations") or {})
    bin_label = f" [{args.bin_name}]" if args.bin_name else ""
    print(f"✓ {args.cohort}/{args.age_band}{bin_label}: {n} genes with PubMed citations")
    print(f"  Output: {args.output_dir / OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
