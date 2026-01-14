#!/usr/bin/env python3
"""
Search PubMed for drug-gene relationships.

This script uses the NCBI Entrez API (via BioPython) to search PubMed
for pharmacogenomic drug-gene relationships.
"""

import sys
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Bio import Entrez
    PUBMED_AVAILABLE = True
except ImportError:
    PUBMED_AVAILABLE = False
    Entrez = None  # type: ignore
    print("Warning: BioPython not installed. Install with: pip install biopython")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set your email for Entrez API (required by NCBI) when available
if PUBMED_AVAILABLE and Entrez is not None:
    Entrez.email = "pgx-analysis@example.com"  # Update with your email


def search_pubmed_drug_gene(drug_name: str, gene_symbol: Optional[str] = None, 
                            max_results: int = 10) -> List[Dict]:
    """
    Search PubMed for drug-gene pharmacogenomic relationships.
    
    Parameters:
    -----------
    drug_name : str
        Drug name to search for
    gene_symbol : str, optional
        Specific gene symbol to search for (e.g., CYP2D6)
    max_results : int
        Maximum number of results to return
        
    Returns:
    --------
    List[Dict]
        List of PubMed article summaries with relevant information
    """
    if not PUBMED_AVAILABLE:
        logger.warning("BioPython not available, skipping PubMed search")
        return []
    
    try:
        # Build search query
        # Search for pharmacogenomics/pharmacogenetics + drug + gene
        if gene_symbol:
            query = f'("{drug_name}"[Title/Abstract] AND "{gene_symbol}"[Title/Abstract] AND (pharmacogenomic OR pharmacogenetic OR "pharmacogenomics"[MeSH Terms] OR "pharmacogenetics"[MeSH Terms]))'
        else:
            query = f'("{drug_name}"[Title/Abstract] AND (pharmacogenomic OR pharmacogenetic OR "pharmacogenomics"[MeSH Terms] OR "pharmacogenetics"[MeSH Terms]))'
        
        logger.debug(f"PubMed query: {query}")
        
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record.get("IdList", [])
        
        if not pmids:
            logger.debug(f"No PubMed results found for {drug_name}" + (f" and {gene_symbol}" if gene_symbol else ""))
            return []
        
        logger.info(f"Found {len(pmids)} PubMed articles for {drug_name}" + (f" and {gene_symbol}" if gene_symbol else ""))
        
        # Fetch article details
        handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
        summaries = Entrez.read(handle)
        handle.close()
        
        # Parse results
        results = []
        for summary in summaries:
            results.append({
                "pmid": summary.get("Id", ""),
                "title": summary.get("Title", ""),
                "authors": summary.get("AuthorList", []),
                "pub_date": summary.get("PubDate", ""),
                "journal": summary.get("Source", ""),
                "drug": drug_name,
                "gene": gene_symbol or "unknown"
            })
        
        # Rate limiting: be respectful to NCBI
        time.sleep(0.34)  # NCBI allows 3 requests per second
        
        return results
        
    except Exception as e:
        logger.warning(f"Error searching PubMed for {drug_name}: {e}")
        return []


def find_genes_for_drug_via_pubmed(drug_name: str, common_pgx_genes: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """
    Find pharmacogenomic genes associated with a drug by searching PubMed.
    
    This function searches PubMed for the drug with common pharmacogenomic genes
    to identify potential relationships.
    
    Parameters:
    -----------
    drug_name : str
        Drug name to search for
    common_pgx_genes : List[str], optional
        List of common pharmacogenomic genes to test (e.g., CYP2D6, CYP2C19)
        If None, uses a default list
        
    Returns:
    --------
    Dict[str, List[Dict]]
        Dictionary mapping gene symbols to lists of PubMed articles
    """
    if not PUBMED_AVAILABLE:
        return {}
    
    # Default list of common pharmacogenomic genes
    if common_pgx_genes is None:
        common_pgx_genes = [
            "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5",
            "CYP2B6", "CYP2C8", "CYP1A2", "CYP2E1",
            "TPMT", "DPYD", "UGT1A1", "NAT2",
            "SLCO1B1", "ABCB1", "ABCC2", "ABCG2",
            "VKORC1", "CYP4F2", "HLA-B", "HLA-A"
        ]
    
    gene_results = {}
    
    logger.info(f"Searching PubMed for {drug_name} with {len(common_pgx_genes)} common PGx genes...")
    
    for gene in common_pgx_genes:
        articles = search_pubmed_drug_gene(drug_name, gene_symbol=gene, max_results=5)
        if articles:
            gene_results[gene] = articles
            logger.info(f"Found {len(articles)} articles linking {drug_name} to {gene}")
    
    return gene_results


def main():
    """Test PubMed search functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search PubMed for drug-gene relationships")
    parser.add_argument("--drug", required=True, help="Drug name to search for")
    parser.add_argument("--gene", help="Specific gene symbol (optional)")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum results")
    
    args = parser.parse_args()
    
    if not PUBMED_AVAILABLE:
        print("Error: BioPython not installed. Install with: pip install biopython")
        return
    
    if args.gene:
        results = search_pubmed_drug_gene(args.drug, gene_symbol=args.gene, max_results=args.max_results)
    else:
        results = find_genes_for_drug_via_pubmed(args.drug)
        # Flatten results
        all_results = []
        for _, articles in results.items():
            all_results.extend(articles)
        results = all_results
    
    print(f"\nFound {len(results)} PubMed articles:")
    for i, article in enumerate(results[:args.max_results], 1):
        print(f"\n{i}. PMID: {article.get('pmid', 'N/A')}")
        print(f"   Title: {article.get('title', 'N/A')}")
        print(f"   Journal: {article.get('journal', 'N/A')}")
        print(f"   Date: {article.get('pub_date', 'N/A')}")
        if 'gene' in article:
            print(f"   Gene: {article['gene']}")


if __name__ == "__main__":
    main()

