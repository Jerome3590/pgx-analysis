#!/usr/bin/env python3
"""
Fetch PharmGKB VIP gene reports for cohort analysis.

For each gene in a cohort's important features (from SHAP/FFA analysis),
fetch the full VIP report content from PharmGKB/ClinPGx for text analysis.

Uses PharmGKB REST API v1:
https://www.postman.com/pharmgkb/pharmgkb-api/documentation/g9rp4zr/pharmgkb-rest-api
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# API Configuration
PHARMGKB_API_BASE = "https://api.pharmgkb.org/v1"
CLINPGX_VIP_BASE = "https://www.clinpgx.org/vip/"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between API requests

# Minimum lengths to consider VIP summary "valid" (log warning below this)
MIN_VIP_SUMMARY_HTML_CHARS = 100
SAMPLE_TEXT_CHARS = 120  # chars of vip_summary_text to log as sample

# CPIC paths (relative to project root) for drug-name -> gene resolution
CPIC_DRUG_LIST_PATH = "5_pgx_analysis/data/cpic_drug_list.json"
DRUG_CPIC_MAPPING_GLOBAL_PATH = "5_pgx_analysis/outputs/global/drug_cpic_mapping_global.csv"


def _load_cpic_drug_to_genes(project_root: Path) -> Tuple[set, Dict[str, List[str]], List[Dict]]:
    """
    Load CPIC drug list and build known-gene set and drug-name -> genes mapping.
    Returns (known_genes, drug_to_genes, cpic_drug_list).
    drug_to_genes keys are uppercase drug names; values are lists of gene symbols.
    """
    known_genes: set = set()
    drug_to_genes: Dict[str, List[str]] = {}
    cpic_drug_list: List[Dict] = []

    cpic_path = project_root / CPIC_DRUG_LIST_PATH
    if not cpic_path.exists():
        logger.warning("CPIC drug list not found at %s; drug names will not be resolved to genes", cpic_path)
        return known_genes, drug_to_genes, cpic_drug_list

    try:
        with open(cpic_path, "r", encoding="utf-8") as f:
            cpic_drug_list = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load CPIC drug list from %s: %s", cpic_path, e)
        return known_genes, drug_to_genes, cpic_drug_list

    if not isinstance(cpic_drug_list, list):
        logger.warning("CPIC drug list root is not a list")
        return known_genes, drug_to_genes, cpic_drug_list

    for entry in cpic_drug_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("drugName") or ""
        if not name or not isinstance(name, str):
            continue
        genes = entry.get("genes")
        if isinstance(genes, list):
            gene_list = [str(g).strip() for g in genes if g]
        elif isinstance(genes, str) and genes.strip():
            gene_list = [genes.strip()]
        else:
            gene_list = []
        for g in gene_list:
            known_genes.add(g.upper())
        drug_to_genes[name.upper().strip()] = gene_list

    logger.info(
        "Loaded CPIC drug list: %d drugs, %d known gene symbols",
        len(drug_to_genes),
        len(known_genes),
    )
    return known_genes, drug_to_genes, cpic_drug_list


def _load_global_drug_mapping(project_root: Path) -> Optional[pd.DataFrame]:
    """Load global drug_name -> cpic_drug_name mapping CSV if present."""
    path = project_root / DRUG_CPIC_MAPPING_GLOBAL_PATH
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "drug_name" not in df.columns or "cpic_drug_name" not in df.columns:
            logger.warning("Global drug mapping missing drug_name or cpic_drug_name columns")
            return None
        logger.info("Loaded global drug-to-CPIC mapping from %s (%d rows)", path, len(df))
        return df
    except Exception as e:
        logger.warning("Could not load global drug mapping from %s: %s", path, e)
        return None


def _resolve_token_to_genes(
    token: str,
    known_genes: set,
    drug_to_genes: Dict[str, List[str]],
    global_mapping_df: Optional[pd.DataFrame],
    cpic_drug_list: List[Dict],
    project_root: Path,
    log: logging.Logger,
) -> List[str]:
    """
    Resolve a feature token (gene symbol or drug name) to a list of gene symbols for PharmGKB.
    Uses CPIC: known genes pass through; drug names are mapped to genes via CPIC list or global mapping.
    """
    if not token or not isinstance(token, str):
        return []
    upper = token.strip().upper()
    if not upper or len(upper) < 2:
        return []

    # 1) Token is a known pharmacogene symbol -> use as-is
    if upper in known_genes:
        return [upper]

    # 2) Exact match as CPIC drug name -> return its genes
    if upper in drug_to_genes:
        genes = drug_to_genes[upper]
        log.info("Resolved drug name to genes (CPIC exact): %s -> %s", token, genes)
        return genes

    # 3) Global mapping: feature token (drug name) -> cpic_drug_name -> genes
    if global_mapping_df is not None and not global_mapping_df.empty:
        match = global_mapping_df[
            global_mapping_df["drug_name"].astype(str).str.strip().str.upper() == upper
        ]
        if not match.empty:
            cpic_name = match.iloc[0].get("cpic_drug_name")
            if pd.notna(cpic_name) and cpic_name:
                cpic_upper = str(cpic_name).strip().upper()
                if cpic_upper in drug_to_genes:
                    genes = drug_to_genes[cpic_upper]
                    log.info(
                        "Resolved drug name to genes (global mapping): %s -> %s -> %s",
                        token,
                        cpic_name,
                        genes,
                    )
                    return genes

    # 4) Optional: fuzzy match via 5_pgx_analysis map_drugs_to_genes
    try:
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "5_pgx_analysis"))
        from map_drugs_to_genes import fuzzy_match_drug  # noqa: E402
        if cpic_drug_list:
            result = fuzzy_match_drug(token, cpic_drug_list, threshold=90)
            if result:
                matched_name, drug_dict, _ = result
                gene_list = drug_dict.get("genes")
                if isinstance(gene_list, list):
                    genes = [str(g).strip() for g in gene_list if g]
                elif isinstance(gene_list, str) and gene_list.strip():
                    genes = [gene_list.strip()]
                else:
                    genes = []
                if genes:
                    log.info(
                        "Resolved drug name to genes (CPIC fuzzy): %s -> %s -> %s",
                        token,
                        matched_name,
                        genes,
                    )
                    return genes
    except ImportError:
        pass
    except Exception as e:
        log.debug("Fuzzy match for %s failed: %s", token, e)

    # Not a known gene and not a resolved drug -> skip (do not pass to PharmGKB as gene symbol)
    log.debug("Token not resolved to genes (skipping): %s", token)
    return []


def _validate_pharmgkb_gene_data(data: Any, gene_symbol: str, log: Any) -> Tuple[bool, List[str]]:
    """
    Validate PharmGKB gene object (top-level 'data' or first element of 'data' list).
    Returns (ok, list of warning messages). ok=True if usable for report building.
    """
    warnings: List[str] = []
    if not isinstance(data, dict):
        log.warning("PharmGKB gene data is not a dict for gene=%s type=%s", gene_symbol, type(data).__name__)
        return False, [f"data type {type(data).__name__}"]

    # Expected keys (PharmGKB API v1 gene response)
    expected = {"id", "name", "vipSummary", "vipTier"}
    missing = expected - set(data.keys())
    if missing:
        warnings.append(f"missing keys: {sorted(missing)}")

    gene_id = data.get("id")
    if gene_id is None or (isinstance(gene_id, str) and not gene_id.strip()):
        warnings.append("id missing or empty")

    vip_summary = data.get("vipSummary")
    if vip_summary is not None and not isinstance(vip_summary, dict):
        warnings.append(f"vipSummary type {type(vip_summary).__name__} (expected dict)")
    elif isinstance(vip_summary, dict) and "html" not in vip_summary:
        warnings.append("vipSummary missing 'html' key")

    citation = data.get("vipCitation")
    if citation is not None and not isinstance(citation, dict):
        warnings.append(f"vipCitation type {type(citation).__name__} (expected dict)")

    for key, expected_type in (("cpicGene", bool), ("amp", bool), ("pharmVarGene", bool)):
        val = data.get(key)
        if val is not None and not isinstance(val, expected_type):
            warnings.append(f"{key} type {type(val).__name__} (expected {expected_type.__name__})")

    return True, warnings


class PharmGKBReportFetcher:
    """Fetch PharmGKB VIP gene reports for text analysis."""

    def __init__(self, base_url: str = PHARMGKB_API_BASE, logger_instance: Optional[logging.Logger] = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PGx-Analysis-Cohort/1.0"
        })
        self._log = logger_instance or logger

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request with rate limiting."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        time.sleep(REQUEST_DELAY)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            try:
                data = response.json()
            except (ValueError, TypeError) as e:
                self._log.error(
                    "PharmGKB API invalid JSON: url=%s status=%s error=%s body_preview=%s",
                    url, response.status_code, e, (response.text or "")[:200],
                )
                return {}
            if not isinstance(data, dict):
                self._log.warning("PharmGKB API response root is not dict: url=%s type=%s", url, type(data).__name__)
                return {}
            # Log valid JSON response shape (no full body) for debugging/audit
            keys = list(data.keys())
            data_preview = ""
            if "data" in data:
                val = data["data"]
                if isinstance(val, list):
                    data_preview = f"data=list(len={len(val)})"
                else:
                    data_preview = f"data={type(val).__name__}"
            self._log.info(
                "PharmGKB API response: url=%s params=%s status=%s keys=%s %s",
                url, params, response.status_code, keys, data_preview,
            )
            return data
        except requests.exceptions.RequestException as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            body = (resp.text[:500] if resp is not None else "") or str(e)
            self._log.error(
                "PharmGKB API request failed: url=%s params=%s error=%s status=%s body=%s",
                url, params, e, status, body,
                exc_info=False,
            )
            return {}
    
    def get_gene_report(self, gene_symbol: str) -> Dict:
        """
        Fetch comprehensive gene report from PharmGKB.

        The gene endpoint returns rich VIP data including:
        - vipSummary: HTML text with clinical guidelines, alleles, drug interactions
        - vipCitation: Full citation with authors, journal, DOI
        - CPIC/AMP status and tier information
        """
        endpoint = "/data/gene"
        gene_data = self._get(endpoint, params={"symbol": gene_symbol})

        if not gene_data:
            self._log.warning("PharmGKB returned empty response for gene=%s", gene_symbol)
            return {}
        if "data" not in gene_data:
            self._log.warning(
                "PharmGKB response missing 'data' for gene=%s keys=%s",
                gene_symbol, list(gene_data.keys()),
            )
            return {}

        data_result = gene_data["data"]

        # Handle list response (API returns list even for single gene)
        if isinstance(data_result, list):
            if not data_result:
                self._log.warning("PharmGKB response data=[] for gene=%s", gene_symbol)
                return {}
            first = data_result[0]
            if not isinstance(first, dict):
                self._log.warning(
                    "PharmGKB response data[0] is not dict for gene=%s type=%s",
                    gene_symbol, type(first).__name__,
                )
                return {}
            data = first
        elif isinstance(data_result, dict):
            data = data_result
        else:
            self._log.warning(
                "PharmGKB response 'data' is not list or dict for gene=%s type=%s",
                gene_symbol, type(data_result).__name__,
            )
            return {}

        # Validate gene object and log warnings
        ok, validation_warnings = _validate_pharmgkb_gene_data(data, gene_symbol, self._log)
        for w in validation_warnings:
            self._log.warning("PharmGKB validation gene=%s: %s", gene_symbol, w)
        if not ok:
            self._log.warning("PharmGKB gene data validation failed for gene=%s; returning partial report", gene_symbol)

        gene_id = data.get("id", "") or ""
        vip_id = data.get("vipId", "") or ""
        self._log.info(
            "PharmGKB gene report parsed: symbol=%s id=%s vipId=%s vipTier=%s has_vipSummary=%s",
            gene_symbol, gene_id, vip_id, data.get("vipTier"), "vipSummary" in data and isinstance(data.get("vipSummary"), dict),
        )

        # Extract VIP summary (contains rich HTML text)
        vip_summary_obj = data.get("vipSummary") if isinstance(data.get("vipSummary"), dict) else {}
        vip_summary_html = (vip_summary_obj.get("html") or "") if isinstance(vip_summary_obj.get("html"), str) else ""
        vip_summary_text = ""
        if vip_summary_html:
            soup = BeautifulSoup(vip_summary_html, "html.parser")
            vip_summary_text = soup.get_text(separator=" ", strip=True)

        gene_name = (data.get("name") or "") if isinstance(data.get("name"), str) else ""
        n_html = len(vip_summary_html)
        n_text = len(vip_summary_text)
        sample = (vip_summary_text or vip_summary_html or "")[:SAMPLE_TEXT_CHARS]
        if sample:
            sample = sample.replace("\n", " ").strip()
        self._log.info(
            "PharmGKB data elements: symbol=%s id_len=%s name_len=%s vipSummary_html_len=%s vipSummary_text_len=%s sample=%s",
            gene_symbol, len(str(gene_id)), len(gene_name), n_html, n_text, repr(sample) if sample else "",
        )
        if n_html > 0 and n_html < MIN_VIP_SUMMARY_HTML_CHARS:
            self._log.warning(
                "PharmGKB vipSummary very short for gene=%s: html_len=%s (min %s)",
                gene_symbol, n_html, MIN_VIP_SUMMARY_HTML_CHARS,
            )
        if not gene_id:
            self._log.warning("PharmGKB gene_id missing for symbol=%s", gene_symbol)
        
        # Extract citation information
        citation = data.get("vipCitation")
        citation_text = ""
        if isinstance(citation, dict):
            title = citation.get("title") or ""
            authors = citation.get("authors")
            author_str = ", ".join(authors[:3]) if isinstance(authors, list) and authors else ""
            journal = citation.get("journal") or ""
            year = citation.get("year") or ""
            citation_text = f"{title} {author_str} et al. {journal} {year}".strip()
        
        # Build comprehensive report
        report = {
            "gene_symbol": gene_symbol,
            "gene_id": gene_id,
            "gene_name": data.get("name", ""),
            "chromosome": data.get("chr", {}).get("name", "") if isinstance(data.get("chr"), dict) else "",
            "chromosome_location": f"{data.get('cbStart', '')}-{data.get('cbStop', '')}",
            "vip_id": vip_id,
            "vip_url": f"{CLINPGX_VIP_BASE}{vip_id}/overview" if vip_id else "",
            "vip_tier": data.get("vipTier", ""),
            
            # CPIC and AMP status
            "cpic_gene": data.get("cpicGene", False),
            "has_cpic_dosing_guideline": data.get("hasCpicDosingGuideline", False),
            "amp_gene": data.get("amp", False),
            "pharmvar_gene": data.get("pharmVarGene", False),
            
            # Rich text content for NLP
            "vip_summary_html": vip_summary_html,
            "vip_summary_text": vip_summary_text,
            "citation": citation,
            "citation_text": citation_text,
            
            # Allele information
            "allele_file": data.get("alleleFile", ""),
            "allele_type": data.get("alleleType", ""),
            "allele_function_source": data.get("alleleFunctionSource", ""),
            
            # Genomic coordinates
            "build_version": data.get("buildVersion", ""),
            "chr_start_b38": data.get("chrStartPosB38"),
            "chr_stop_b38": data.get("chrStopPosB38"),
            "strand": data.get("strand", ""),
            
            # Full raw data for reference
            "raw_gene_data": data
        }
        
        return report
    
    def fetch_clinpgx_vip_page(self, gene_id: str) -> str:
        """
        Fetch ClinPGx VIP page HTML for text extraction.
        
        Returns raw HTML content for further processing with BeautifulSoup.
        """
        vip_url = f"{CLINPGX_VIP_BASE}{gene_id}/overview"
        time.sleep(REQUEST_DELAY)
        
        try:
            response = requests.get(vip_url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            body = (resp.text[:300] if resp is not None else "") or str(e)
            self._log.warning(
                "ClinPGx VIP page fetch failed: url=%s gene_id=%s error=%s status=%s body=%s",
                vip_url, gene_id, e, status, body,
            )
            return ""
    
    def extract_vip_text(self, html_content: str) -> Dict[str, str]:
        """
        Extract text content from ClinPGx VIP page.
        
        Returns structured text sections for analysis.
        """
        if not html_content:
            return {}
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove scripts, styles, navigation
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Extract main content areas
        text_sections = {
            "overview": "",
            "clinical_annotations": "",
            "variant_annotations": "",
            "drug_labels": "",
            "full_text": soup.get_text(separator="\n", strip=True)
        }
        
        # Try to extract specific sections (structure may vary)
        # This is a basic extraction - may need refinement based on actual page structure
        for section_id in ["overview", "clinical-annotations", "variant-annotations", "drug-labels"]:
            section = soup.find(id=section_id) or soup.find(class_=section_id)
            if section:
                text_sections[section_id.replace("-", "_")] = section.get_text(separator="\n", strip=True)
        
        return text_sections


def load_cohort_genes(
    cohort_name: str,
    age_band: str,
    project_root: Path,
    top_n: int = 50,
    pipeline_logger: Optional[Any] = None,
) -> List[str]:
    """
    Load top N important genes for a cohort from SHAP/FFA analysis.

    Uses the same activity-type logic as BupaR: only **drug_name** codes from final feature
    importance are considered (via get_shap_ffa_important_codes(..., "drug_name")). Those tokens
    are drug names and are resolved to gene symbols via CPIC; only gene symbols are sent to
    PharmGKB. ICD/CPT codes are never passed to CPIC name match.
    """
    log = pipeline_logger.logger if pipeline_logger is not None and hasattr(pipeline_logger, "logger") else logger

    try:
        from py_helpers.shap_ffa_fpgrowth_utils import get_shap_ffa_important_codes
    except ImportError:
        log.warning("py_helpers.shap_ffa_fpgrowth_utils not available; cannot load drug_name codes for cohort genes.")
        return []

    # Same as BupaR: only drug_name codes from final (cohort) feature importance
    drug_names = get_shap_ffa_important_codes(
        cohort_name,
        age_band,
        "drug_name",
        top_n=top_n,
        project_root=project_root,
        data_root=None,
    )
    if not drug_names:
        log.warning(
            "No drug_name codes from feature importance for cohort=%s age_band=%s",
            cohort_name,
            age_band,
        )
        return []

    # Load CPIC drug -> genes for resolving drug names to genes
    known_genes, drug_to_genes, cpic_drug_list = _load_cpic_drug_to_genes(project_root)
    global_mapping_df = _load_global_drug_mapping(project_root)

    genes = set()
    for drug_name in drug_names:
        token = drug_name.strip().upper() if isinstance(drug_name, str) else ""
        if not token or len(token) < 2:
            continue
        resolved = _resolve_token_to_genes(
            token,
            known_genes,
            drug_to_genes,
            global_mapping_df,
            cpic_drug_list,
            project_root,
            log,
        )
        genes.update(resolved)

    log.info(
        "Resolved %d gene symbols from %d drug_name codes (BupaR activity-type logic) for %s/%s",
        len(genes),
        len(drug_names),
        cohort_name,
        age_band,
    )
    return sorted(genes)


def fetch_cohort_reports(
    cohort_name: str,
    age_band: str,
    project_root: Path,
    output_dir: Path,
    top_n: int = 50,
    include_vip_pages: bool = True,
    pipeline_logger: Optional[Any] = None,
    force: bool = False,
) -> Dict:
    """
    Fetch PharmGKB VIP reports for all important genes in a cohort.

    Idempotent: if reports file already exists, skips fetch unless force=True.

    Args:
        cohort_name: Cohort name (opioid_ed, non_opioid_ed)
        age_band: Age band (0-12, 13-24, etc.)
        project_root: Project root directory
        output_dir: Output directory for reports
        top_n: Number of top features to analyze
        include_vip_pages: Whether to fetch ClinPGx VIP page content
        pipeline_logger: Optional PipelineLogger for consistent logging
        force: If True, re-fetch even when reports file exists

    Returns:
        Dict with reports metadata
    """
    pl = pipeline_logger
    log = pl.logger if pl is not None and hasattr(pl, "logger") else logger
    output_dir.mkdir(parents=True, exist_ok=True)
    age_band_fname = age_band.replace("-", "_")
    reports_file = output_dir / f"{cohort_name}_{age_band_fname}_vip_reports.json"
    summary_file = output_dir / f"{cohort_name}_{age_band_fname}_vip_reports_summary.json"

    if not force and reports_file.exists():
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                log.info(
                    "Reports already exist for %s/%s (%d reports); skipping. Use --force to re-fetch.",
                    cohort_name,
                    age_band,
                    summary.get("reports_fetched", 0),
                )
                return summary
            except (json.JSONDecodeError, OSError):
                pass
        # No summary file: build minimal summary from reports file
        try:
            with open(reports_file, "r", encoding="utf-8") as f:
                reports = json.load(f)
            summary = {
                "cohort": cohort_name,
                "age_band": age_band,
                "genes_requested": len(reports),
                "reports_fetched": len(reports),
                "genes_with_vip_text": sum(1 for r in reports if "vip_text" in r),
                "genes": [r.get("gene_symbol", "") for r in reports if isinstance(r, dict)],
                "output_file": str(reports_file),
            }
            log.info(
                "Reports already exist for %s/%s (%d reports); skipping. Use --force to re-fetch.",
                cohort_name,
                age_band,
                len(reports),
            )
            return summary
        except (json.JSONDecodeError, OSError):
            pass

    if pl is not None and hasattr(pl, "info"):
        pl.info("=" * 80)
        pl.info("Fetching VIP reports for %s / %s", cohort_name, age_band)
        pl.info("=" * 80)
    else:
        log.info("Fetching VIP reports for %s / %s", cohort_name, age_band)

    genes = load_cohort_genes(
        cohort_name,
        age_band,
        project_root,
        top_n=top_n,
        pipeline_logger=pl,
    )

    if not genes:
        log.warning("No genes found for cohort=%s age_band=%s; exiting", cohort_name, age_band)
        return {"genes_found": 0, "reports_fetched": 0}

    if pl is not None and hasattr(pl, "info"):
        pl.info("Genes to fetch: %s", ", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""))
    else:
        log.info("Genes to fetch: %s", ", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""))

    fetcher = PharmGKBReportFetcher(logger_instance=log)
    reports = []

    for i, gene in enumerate(genes, 1):
        log.info("[%d/%d] Fetching %s...", i, len(genes), gene)

        report = fetcher.get_gene_report(gene)

        if not report:
            log.debug("PharmGKB returned no data for gene=%s", gene)
            continue

        if include_vip_pages and report.get("gene_id"):
            html_content = fetcher.fetch_clinpgx_vip_page(report["gene_id"])
            if html_content:
                text_sections = fetcher.extract_vip_text(html_content)
                report["vip_text"] = text_sections

        reports.append(report)
        log.info("  %s -> %s", gene, report.get("gene_name", "N/A"))

    with open(reports_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    if pl is not None and hasattr(pl, "info"):
        pl.info("Saved %d reports to %s", len(reports), reports_file)
    else:
        log.info("Saved %d reports to %s", len(reports), reports_file)

    summary = {
        "cohort": cohort_name,
        "age_band": age_band,
        "genes_requested": len(genes),
        "reports_fetched": len(reports),
        "genes_with_vip_text": sum(1 for r in reports if "vip_text" in r),
        "genes": [r["gene_symbol"] for r in reports],
        "output_file": str(reports_file)
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "VIP reports fetch complete: cohort=%s age_band=%s genes_requested=%s reports_fetched=%s genes_with_vip_text=%s",
        cohort_name, age_band, summary["genes_requested"], summary["reports_fetched"], summary["genes_with_vip_text"],
    )
    if summary["reports_fetched"] == 0 and summary["genes_requested"] > 0:
        log.warning("No reports fetched despite %s genes requested; check PharmGKB API errors above", summary["genes_requested"])
    if pl is not None and hasattr(pl, "info"):
        pl.info("Saved summary to %s", summary_file)

    return summary


def main():
    """Fetch VIP reports for a cohort. Uses same logging pattern as BupaR/DTW: logs to 9_dashboard_visuals/logs/cohort_pgx/."""
    parser = argparse.ArgumentParser(
        description="Fetch PharmGKB VIP reports for cohort PGx analysis"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name (opioid_ed, non_opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (0-12, 13-24, etc.)")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top features to analyze")
    parser.add_argument("--no-vip-pages", action="store_true", help="Skip fetching ClinPGx VIP pages")
    parser.add_argument("--force", action="store_true", help="Re-fetch even when reports file already exists (default: skip if exists)")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()

    project_root = args.project_root or Path(__file__).parent.parent.parent
    output_dir = args.output_dir or (project_root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "reports")

    # Same pattern as BupaR/DTW: pipeline logger → file under 9_dashboard_visuals/logs/cohort_pgx/
    sys.path.insert(0, str(project_root))
    from py_helpers.fe_monitor import function_block  # noqa: E402
    from py_helpers.pipeline_logger import setup_pipeline_logger  # noqa: E402
    pl = setup_pipeline_logger(
        step_name="cohort_pgx",
        cohort=args.cohort,
        age_band=args.age_band,
        script_name="fetch_vip_reports",
    )
    # Route this module's logger to the pipeline log file
    mod_logger = logging.getLogger(__name__)
    mod_logger.handlers.clear()
    mod_logger.setLevel(logging.DEBUG)
    mod_logger.propagate = False
    for h in pl.logger.handlers:
        mod_logger.addHandler(h)

    with function_block("cohort_pgx", "fetch_vip_reports", logger=pl.logger):
        pl.info("Logs: %s", pl.log_file_path)
        pl.info("Starting fetch_vip_reports for %s / %s", args.cohort, args.age_band)
        summary = fetch_cohort_reports(
            cohort_name=args.cohort,
            age_band=args.age_band,
            project_root=project_root,
            output_dir=output_dir,
            top_n=args.top_n,
            include_vip_pages=not args.no_vip_pages,
            pipeline_logger=pl,
            force=args.force,
        )
        pl.info("=" * 80)
        pl.info("PIPELINE STEP SUMMARY (fetch_vip_reports)")
        pl.info("=" * 80)
        for key, value in summary.items():
            if key != "genes":
                pl.info("  %s: %s", key, value)
        pl.info("Logs: %s", pl.log_file_path)
        pl.info("=" * 80)
    pl.log_summary()


if __name__ == "__main__":
    main()
