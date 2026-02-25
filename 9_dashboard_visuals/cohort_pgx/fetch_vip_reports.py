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
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

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


class PharmGKBReportFetcher:
    """Fetch PharmGKB VIP gene reports for text analysis."""
    
    def __init__(self, base_url: str = PHARMGKB_API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PGx-Analysis-Cohort/1.0"
        })
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request with rate limiting."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        time.sleep(REQUEST_DELAY)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Log valid JSON response shape (no full body) for debugging/audit
            if data is not None:
                keys = list(data.keys()) if isinstance(data, dict) else []
                data_preview = ""
                if isinstance(data, dict) and "data" in data:
                    val = data["data"]
                    if isinstance(val, list):
                        data_preview = f"data=list(len={len(val)})"
                    else:
                        data_preview = f"data={type(val).__name__}"
                logger.info(
                    "PharmGKB API response: url=%s params=%s status=%s keys=%s %s",
                    url, params, response.status_code, keys, data_preview,
                )
            return data
        except requests.exceptions.RequestException as e:
            # Log what was actually returned so EC2/CI can diagnose (throttling, 403, empty body, etc.)
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            body = (resp.text[:500] if resp is not None else "") or str(e)
            logger.error(
                "PharmGKB API request failed: url=%s params=%s error=%s status=%s body=%s",
                url, params, e, status, body,
                exc_info=False,
            )
            print(f"Error fetching {url}: {e}")
            if resp is not None:
                print(f"  Status: {resp.status_code}  Body: {resp.text[:300]}")
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
            logger.warning("PharmGKB returned empty response for gene=%s", gene_symbol)
            return {}
        if "data" not in gene_data:
            logger.warning(
                "PharmGKB response missing 'data' for gene=%s keys=%s",
                gene_symbol, list(gene_data.keys()),
            )
            return {}
        
        data_result = gene_data["data"]
        
        # Handle list response (API returns list even for single gene)
        if isinstance(data_result, list):
            if not data_result:
                logger.warning("PharmGKB response data=[] for gene=%s", gene_symbol)
                return {}
            data = data_result[0]  # Use first match
        else:
            data = data_result
        
        gene_id = data.get("id", "")
        vip_id = data.get("vipId", "")
        # Log valid gene payload we got from API (confirms parsing succeeded)
        logger.info(
            "PharmGKB gene report parsed: symbol=%s id=%s vipId=%s vipTier=%s has_vipSummary=%s",
            gene_symbol, gene_id, vip_id, data.get("vipTier"), "vipSummary" in data and isinstance(data.get("vipSummary"), dict),
        )
        
        # Extract VIP summary (contains rich HTML text)
        vip_summary_obj = data.get("vipSummary", {})
        vip_summary_html = ""
        vip_summary_text = ""
        if isinstance(vip_summary_obj, dict):
            vip_summary_html = vip_summary_obj.get("html", "") or ""
            # Convert HTML to plain text for NLP
            if vip_summary_html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(vip_summary_html, "html.parser")
                vip_summary_text = soup.get_text(separator=" ", strip=True)
        
        # Validate and log data element sizes + sample (so we can confirm valid content)
        gene_name = (data.get("name") or "") if isinstance(data.get("name"), str) else ""
        n_html = len(vip_summary_html)
        n_text = len(vip_summary_text)
        sample = (vip_summary_text or vip_summary_html or "")[:SAMPLE_TEXT_CHARS]
        if sample:
            sample = sample.replace("\n", " ").strip()
        logger.info(
            "PharmGKB data elements: symbol=%s id_len=%s name_len=%s vipSummary_html_len=%s vipSummary_text_len=%s sample=%s",
            gene_symbol, len(str(gene_id)), len(gene_name), n_html, n_text, repr(sample) if sample else "",
        )
        if n_html > 0 and n_html < MIN_VIP_SUMMARY_HTML_CHARS:
            logger.warning(
                "PharmGKB vipSummary very short for gene=%s: html_len=%s (min %s)",
                gene_symbol, n_html, MIN_VIP_SUMMARY_HTML_CHARS,
            )
        if not gene_id:
            logger.warning("PharmGKB gene_id missing for symbol=%s", gene_symbol)
        
        # Extract citation information
        citation = data.get("vipCitation", {})
        citation_text = ""
        if isinstance(citation, dict):
            title = citation.get("title", "")
            authors = citation.get("authors", [])
            journal = citation.get("journal", "")
            year = citation.get("year", "")
            citation_text = f"{title} {', '.join(authors[:3])} et al. {journal} {year}"
        
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
            logger.warning(
                "ClinPGx VIP page fetch failed: url=%s gene_id=%s error=%s status=%s body=%s",
                vip_url, gene_id, e, status, body,
            )
            print(f"Error fetching ClinPGx page {vip_url}: {e}")
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


def load_cohort_genes(cohort_name: str, age_band: str, project_root: Path, top_n: int = 50) -> List[str]:
    """
    Load top N important genes for a cohort from SHAP/FFA analysis.
    
    Returns list of gene symbols extracted from feature names (item_DRUG_<GENE>).
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try Step 3b feature importance first
    feature_importance_path = (
        project_root / "3a_feature_importance" / "outputs" / 
        cohort_name / age_band_fname / "cohort_feature_importance.csv"
    )
    
    # Fallback to notebook 3 combined importance
    if not feature_importance_path.exists():
        feature_importance_path = (
            project_root / "10_risk_dashboard" / "outputs" / 
            cohort_name / age_band_fname / "combined_importance.csv"
        )
    
    if not feature_importance_path.exists():
        logger.warning("No feature importance found for cohort=%s age_band=%s (tried %s)", cohort_name, age_band, feature_importance_path)
        print(f"Warning: No feature importance found for {cohort_name}/{age_band}")
        return []
    
    # Load feature importance
    df = pd.read_csv(feature_importance_path)
    
    # Get feature column
    feature_col = next((c for c in df.columns if "feature" in c.lower()), df.columns[0])
    importance_col = next((c for c in df.columns if "importance" in c.lower()), df.columns[1])
    
    # Sort by importance and get top N
    df = df.sort_values(importance_col, ascending=False).head(top_n)
    
    # Extract gene symbols from feature names
    # Pattern: item_DRUG_<GENE>, item_ICD_<CODE>, item_CPT_<CODE>
    # We want only DRUG features for PGx analysis
    genes = set()
    for feature in df[feature_col]:
        if isinstance(feature, str):
            # Extract gene from DRUG features
            if "DRUG" in feature.upper():
                parts = feature.split("_")
                if len(parts) >= 3:
                    # item_DRUG_GENENAME or DRUG:GENENAME
                    gene_part = parts[2] if parts[0].lower() == "item" else parts[1]
                    gene_part = gene_part.split(":")[1] if ":" in gene_part else gene_part
                    # Clean and validate gene symbol
                    gene = gene_part.strip().upper()
                    if gene and len(gene) >= 3 and gene.isalpha():
                        genes.add(gene)
    
    print(f"Extracted {len(genes)} genes from top {top_n} features for {cohort_name}/{age_band}")
    return sorted(genes)


def fetch_cohort_reports(
    cohort_name: str,
    age_band: str,
    project_root: Path,
    output_dir: Path,
    top_n: int = 50,
    include_vip_pages: bool = True
) -> Dict:
    """
    Fetch PharmGKB VIP reports for all important genes in a cohort.
    
    Args:
        cohort_name: Cohort name (opioid_ed, non_opioid_ed)
        age_band: Age band (0-12, 13-24, etc.)
        project_root: Project root directory
        output_dir: Output directory for reports
        top_n: Number of top features to analyze
        include_vip_pages: Whether to fetch ClinPGx VIP page content
    
    Returns:
        Dict with reports metadata
    """
    print(f"\n{'='*80}")
    print(f"Fetching VIP reports for {cohort_name} / {age_band}")
    print(f"{'='*80}\n")
    
    # Load cohort genes
    genes = load_cohort_genes(cohort_name, age_band, project_root, top_n=top_n)
    
    if not genes:
        print("No genes found. Exiting.")
        return {"genes_found": 0, "reports_fetched": 0}
    
    print(f"Genes to fetch: {', '.join(genes[:10])}{'...' if len(genes) > 10 else ''}")
    print()
    
    # Fetch reports
    fetcher = PharmGKBReportFetcher()
    reports = []
    
    for i, gene in enumerate(genes, 1):
        print(f"[{i}/{len(genes)}] Fetching {gene}...", end=" ")
        
        # Get gene report from API
        report = fetcher.get_gene_report(gene)
        
        if not report:
            logger.debug("PharmGKB returned no data for gene=%s (API may have failed or empty data)", gene)
            print("✗ Not found")
            continue
        
        # Optionally fetch ClinPGx VIP page content
        if include_vip_pages and report.get("gene_id"):
            html_content = fetcher.fetch_clinpgx_vip_page(report["gene_id"])
            if html_content:
                text_sections = fetcher.extract_vip_text(html_content)
                report["vip_text"] = text_sections
        
        reports.append(report)
        print(f"✓ {report.get('gene_name', 'N/A')}")
    
    # Save reports
    output_dir.mkdir(parents=True, exist_ok=True)
    age_band_fname = age_band.replace("-", "_")
    
    reports_file = output_dir / f"{cohort_name}_{age_band_fname}_vip_reports.json"
    with open(reports_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(reports)} reports to {reports_file}")
    
    # Save summary
    summary = {
        "cohort": cohort_name,
        "age_band": age_band,
        "genes_requested": len(genes),
        "reports_fetched": len(reports),
        "genes_with_vip_text": sum(1 for r in reports if "vip_text" in r),
        "genes": [r["gene_symbol"] for r in reports],
        "output_file": str(reports_file)
    }
    
    summary_file = output_dir / f"{cohort_name}_{age_band_fname}_vip_reports_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(
        "VIP reports fetch complete: cohort=%s age_band=%s genes_requested=%s reports_fetched=%s genes_with_vip_text=%s",
        cohort_name, age_band, summary["genes_requested"], summary["reports_fetched"], summary["genes_with_vip_text"],
    )
    if summary["reports_fetched"] == 0 and summary["genes_requested"] > 0:
        logger.warning("No reports fetched despite %s genes requested; check PharmGKB API errors above", summary["genes_requested"])
    print(f"✓ Saved summary to {summary_file}")
    
    return summary


def main():
    """Fetch VIP reports for a cohort."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Fetch PharmGKB VIP reports for cohort PGx analysis"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name (opioid_ed, non_opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (0-12, 13-24, etc.)")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top features to analyze")
    parser.add_argument("--no-vip-pages", action="store_true", help="Skip fetching ClinPGx VIP pages")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = args.project_root or Path(__file__).parent.parent.parent
    output_dir = args.output_dir or (project_root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "reports")
    
    # Fetch reports
    summary = fetch_cohort_reports(
        cohort_name=args.cohort,
        age_band=args.age_band,
        project_root=project_root,
        output_dir=output_dir,
        top_n=args.top_n,
        include_vip_pages=not args.no_vip_pages
    )
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for key, value in summary.items():
        if key != "genes":
            print(f"  {key}: {value}")
    print("="*80)


if __name__ == "__main__":
    main()
