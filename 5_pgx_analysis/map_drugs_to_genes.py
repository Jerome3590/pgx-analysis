#!/usr/bin/env python3
"""
Map drugs to pharmacogenomic genes using PharmGKB and CPIC data.

This script maps drugs identified in the analysis to relevant pharmacogenes
(e.g., CYP2D6, CYP2C19, TPMT, DPYD) based on established pharmacogenomic knowledge.
"""

import sys
import pandas as pd
from pathlib import Path
import json
import logging
import requests
from typing import List, Dict, Optional, Tuple
import time
from rapidfuzz import fuzz, process

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "5_pgx_analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "5_pgx_analysis"))

# Import PubMed search functionality
try:
    from search_pubmed_drug_gene import find_genes_for_drug_via_pubmed  # noqa: F401
    PUBMED_AVAILABLE = True
except ImportError:
    PUBMED_AVAILABLE = False
    find_genes_for_drug_via_pubmed = None  # type: ignore
    # logger not defined yet, will be set up below

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CPIC API base URL
CPIC_API_BASE = "https://api.cpicpgx.org"


def fetch_cpic_guidelines(drug_name: str) -> List[Dict]:
    """
    Fetch CPIC guidelines for a given drug name.
    
    Parameters:
    -----------
    drug_name : str
        Drug name to search for
        
    Returns:
    --------
    List[Dict]
        List of guideline records from CPIC API
    """
    try:
        # Search for drug in CPIC guidelines
        # CPIC API endpoint: /guideline?drugname={drug_name}
        url = f"{CPIC_API_BASE}/guideline"
        params = {"drugname": drug_name}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle both list and single dict responses
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return []
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching CPIC guidelines for {drug_name}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error fetching CPIC guidelines for {drug_name}: {e}")
        return []


def load_cpic_drug_list_from_file() -> List[Dict]:
    """
    Load CPIC drug list from saved JSON file, official Excel, or CPIC pairs CSV.
    
    Priority order:
    1. Saved JSON file (data/cpic_drug_list.json) - if already processed
    2. Official CPIC Excel file (cpic/cpic_gene-drug_pairs.xlsx) - PRIMARY SOURCE
       - Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
    3. CPIC pairs CSV (data/cpicPairs.csv) - fallback
    """
    drug_list_path = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpic_drug_list.json"
    cpic_excel_path = PROJECT_ROOT / "5_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
    cpic_pairs_path = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpicPairs.csv"
    
    # Try loading from JSON first
    if drug_list_path.exists():
        try:
            with open(drug_list_path, 'r') as f:
                drugs = json.load(f)
            logger.info(f"Loaded {len(drugs)} drugs from {drug_list_path}")
            return drugs
        except Exception as e:
            logger.warning(f"Error loading drug list from JSON: {e}")
    
    # Try official CPIC Excel file (build per-drug gene lists)
    if cpic_excel_path.exists():
        try:
            df = pd.read_excel(cpic_excel_path)
            # Standardize column names
            if 'Drug' not in df.columns:
                drug_cols = [col for col in df.columns if 'drug' in col.lower()]
                if drug_cols:
                    df = df.rename(columns={drug_cols[0]: 'Drug'})
            if 'Gene' not in df.columns:
                gene_cols = [col for col in df.columns if 'gene' in col.lower()]
                if gene_cols:
                    df = df.rename(columns={gene_cols[0]: 'Gene'})
            
            if 'Drug' in df.columns:
                # Build mapping: Drug -> unique list of Genes
                df_pairs = df[['Drug', 'Gene']].dropna()
                grouped = (
                    df_pairs.groupby('Drug')['Gene']
                    .apply(lambda s: sorted({str(x) for x in s}))
                    .reset_index()
                )
                drugs = [
                    {
                        "name": str(row["Drug"]),
                        "genes": row["Gene"],
                        "source": "cpic_gene-drug_pairs.xlsx",
                    }
                    for _, row in grouped.iterrows()
                ]
                logger.info(
                    "Loaded %d drugs with gene lists from official CPIC Excel file: %s",
                    len(drugs),
                    cpic_excel_path,
                )
                return drugs
        except Exception as e:
            logger.warning(f"Error loading drug list from Excel: {e}")
    
    # Fallback: load directly from CPIC pairs CSV
    if cpic_pairs_path.exists():
        try:
            df = pd.read_csv(cpic_pairs_path)
            unique_drugs = df['Drug'].dropna().unique()
            drugs = [{"name": str(drug), "source": "cpicPairs.csv"} for drug in unique_drugs]
            logger.info(f"Loaded {len(drugs)} drugs from {cpic_pairs_path}")
            return drugs
        except Exception as e:
            logger.warning(f"Error loading drug list from CSV: {e}")
    
    return []


def fetch_all_cpic_drugs() -> List[Dict]:
    """
    Fetch all drugs from CPIC API by extracting from guidelines.
    
    Returns:
    --------
    List[Dict]
        List of all drug records from CPIC API (extracted from guidelines)
    """
    try:
        # Try to get all guidelines and extract unique drugs
        url = f"{CPIC_API_BASE}/guideline"
        response = requests.get(url, timeout=60)  # May take longer for all guidelines
        
        if response.status_code == 200:
            guidelines = response.json()
            if isinstance(guidelines, list):
                # Extract unique drugs from guidelines
                drugs_dict = {}
                for guideline in guidelines:
                    if isinstance(guideline, dict):
                        drug_info = guideline.get("drug", {})
                        if isinstance(drug_info, dict):
                            drug_name = drug_info.get("name", "")
                            drug_id = drug_info.get("id", "")
                            if drug_name and drug_name not in drugs_dict:
                                drugs_dict[drug_name] = {
                                    "name": drug_name,
                                    "id": drug_id,
                                    "genes": []
                                }
                                # Extract genes from this guideline
                                gene_info = guideline.get("gene", {})
                                if isinstance(gene_info, dict):
                                    gene_symbol = gene_info.get("symbol", "")
                                    if gene_symbol:
                                        drugs_dict[drug_name]["genes"].append(gene_symbol)
                
                # Convert to list
                drugs_list = list(drugs_dict.values())
                logger.info(f"Extracted {len(drugs_list)} unique drugs from {len(guidelines)} guidelines")
                return drugs_list
        
        # Fallback: try /drug endpoint
        url = f"{CPIC_API_BASE}/drug"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        
        return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching all CPIC drugs: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error fetching all CPIC drugs: {e}")
        return []


def fuzzy_match_drug(drug_name: str, cpic_drug_list: List[Dict], threshold: int = 80) -> Optional[Tuple[str, Dict]]:
    """
    Use fuzzy matching to find the best CPIC drug match for a given drug name.
    
    Parameters:
    -----------
    drug_name : str
        Drug name to match
    cpic_drug_list : List[Dict]
        List of CPIC drug dictionaries (can be from API or file)
    threshold : int
        Minimum similarity score (0-100) to consider a match
        
    Returns:
    --------
    Optional[Tuple[str, Dict]]
        Tuple of (matched_cpic_name, drug_dict) if match found above threshold, else None
    """
    if not cpic_drug_list:
        return None
    
    # Extract drug names from CPIC list (handle both API format and file format)
    cpic_names = []
    cpic_name_to_dict = {}
    
    for drug in cpic_drug_list:
        if isinstance(drug, dict):
            # Handle different formats: API format vs file format
            name = drug.get("name", "") or drug.get("drugName", "") or drug.get("drug", "")
            if name:
                cpic_names.append(name)
                cpic_name_to_dict[name] = drug
    
    if not cpic_names:
        return None
    
    # Normalize drug name for better matching (remove common suffixes)
    normalized_drug = drug_name.upper().strip()
    # Remove common suffixes that might differ (in order of specificity)
    suffixes_to_remove = [
        " SODIUM PHOSPHATE", " SODIUM PHOSP",  # Handle truncated versions
        " HYDROCHLORIDE", " HCL",
        " SULFATE", " SULF",
        " PHOSPHATE", " PHOSP",
        " SODIUM",
        " ODT",  # Orally Disintegrating Tablet
        " B SULFATE/TRIME",  # Special case for combination drugs
    ]
    for suffix in suffixes_to_remove:
        if normalized_drug.endswith(suffix):
            normalized_drug = normalized_drug[:-len(suffix)].strip()
            break  # Only remove one suffix
    
    # Also try matching the base drug name (first word)
    base_drug = normalized_drug.split()[0] if normalized_drug.split() else normalized_drug
    
    # Use rapidfuzz to find best match - try both normalized and base name
    candidates = [normalized_drug, base_drug] if base_drug != normalized_drug else [normalized_drug]
    
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        result = process.extractOne(
            candidate,
            [n.upper() for n in cpic_names],
            scorer=fuzz.WRatio,  # Weighted ratio - good for partial matches
            score_cutoff=threshold
        )
        
        if result:
            matched_name_normalized, score, _ = result
            if score > best_score:
                best_score = score
                # Find original case name
                matched_name = None
                for name in cpic_names:
                    if name.upper() == matched_name_normalized:
                        matched_name = name
                        break
                
                if matched_name and matched_name in cpic_name_to_dict:
                    best_match = (matched_name, cpic_name_to_dict[matched_name], score)
    
    if best_match:
        matched_name, drug_dict, score = best_match
        logger.info(f"Fuzzy matched '{drug_name}' -> '{matched_name}' (score: {score:.1f})")
        return (matched_name, drug_dict)
    
    return None


def fetch_cpic_drug(drug_name: str) -> Optional[Dict]:
    """
    Fetch drug information from CPIC API.
    
    Parameters:
    -----------
    drug_name : str
        Drug name to search for
        
    Returns:
    --------
    Optional[Dict]
        Drug information from CPIC API
    """
    try:
        # CPIC API endpoint: /drug
        url = f"{CPIC_API_BASE}/drug"
        
        # Try with name parameter
        params = {"name": drug_name}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list) and len(data) > 0:
                    # Return first match
                    return data[0] if isinstance(data[0], dict) else None
            except ValueError:
                pass
        
        # Try getting all drugs and filtering
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            try:
                all_drugs = response.json()
                if isinstance(all_drugs, list):
                    drug_upper = drug_name.upper()
                    for drug in all_drugs:
                        if isinstance(drug, dict):
                            if drug_upper in str(drug.get("name", "")).upper():
                                return drug
            except ValueError:
                pass
        
        return None
            
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error fetching CPIC drug info for {drug_name}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching CPIC drug info for {drug_name}: {e}")
        return None


def fetch_cpic_gene(gene_symbol: str) -> Optional[Dict]:
    """
    Fetch gene information from CPIC API.
    
    Parameters:
    -----------
    gene_symbol : str
        Gene symbol (e.g., CYP2D6, CYP2C19)
        
    Returns:
    --------
    Optional[Dict]
        Gene information from CPIC API
    """
    try:
        # CPIC API endpoint: /gene?symbol={gene_symbol}
        url = f"{CPIC_API_BASE}/gene"
        params = {"symbol": gene_symbol}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data if isinstance(data, dict) else None
            
    except requests.exceptions.RequestException as e:
        logger.debug(f"Error fetching CPIC gene info for {gene_symbol}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching CPIC gene info for {gene_symbol}: {e}")
        return None


def map_drugs_to_genes(
    drug_names: List[str],
    output_path: Optional[Path] = None,
    rate_limit_delay: float = 0.5,
    use_fuzzy_matching: bool = True,
    fuzzy_threshold: int = 80,
    use_pubmed: bool = False,
    local_only: bool = False,
) -> pd.DataFrame:
    """
    Map a list of drug names to relevant pharmacogenes using CPIC API.
    
    Parameters:
    -----------
    drug_names : list
        List of drug names to map
    output_path : Path, optional
        Path to save the mappings CSV file
    rate_limit_delay : float
        Delay between API calls (seconds) to respect rate limits
    use_fuzzy_matching : bool
        Whether to use fuzzy matching to find CPIC drug names
    fuzzy_threshold : int
        Minimum similarity score (0-100) for fuzzy matching
    use_pubmed : bool
        Whether to search PubMed for drug-gene relationships when CPIC match not found
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: drug_name, cpic_drug_name, gene, relationship_type,
        evidence_level, clinical_significance, source, guideline_id,
        guideline_url, cpic_level
    """
    logger.info(f"Mapping {len(drug_names)} drugs to genes using CPIC API...")
    
    # Fetch CPIC drugs for fuzzy matching (from local Excel/JSON; avoid API in local_only mode)
    cpic_drug_list = []
    if use_fuzzy_matching:
        logger.info("Loading CPIC drug list for fuzzy matching...")
        cpic_drug_list = load_cpic_drug_list_from_file()

        # If file doesn't exist or is empty and we are not restricted to local-only,
        # we can fall back to the CPIC API.
        if not cpic_drug_list and not local_only:
            logger.info("Local CPIC drug list not found, fetching from CPIC API...")
            cpic_drug_list = fetch_all_cpic_drugs()

        logger.info(f"Loaded {len(cpic_drug_list)} drugs for fuzzy matching")
    
    mappings = []
    found_count = 0
    fuzzy_matched_count = 0
    
    for drug_name in drug_names:
        logger.debug(f"Processing drug: {drug_name}")
        
        # Try fuzzy matching first if enabled
        matched_drug_info = None
        matched_cpic_name = drug_name
        
        if use_fuzzy_matching and cpic_drug_list:
            fuzzy_match = fuzzy_match_drug(drug_name, cpic_drug_list, threshold=fuzzy_threshold)
            if fuzzy_match:
                matched_cpic_name, matched_drug_info = fuzzy_match
                fuzzy_matched_count += 1
                logger.info(f"Fuzzy matched '{drug_name}' -> '{matched_cpic_name}'")
        
        # Use the matched CPIC name for API queries
        search_name = matched_cpic_name if matched_cpic_name != drug_name else drug_name
        
        # If we have matched drug info from fuzzy matching, extract genes directly
        if matched_drug_info:
            # Extract gene associations from matched drug info
            gene_symbols = matched_drug_info.get("genes", [])
            if isinstance(gene_symbols, str):
                gene_symbols = [gene_symbols]
            elif not isinstance(gene_symbols, list):
                gene_symbols = []
            
            # If we have genes from the drug info, use them
            if gene_symbols:
                for gene_symbol in gene_symbols:
                    if gene_symbol:
                        mappings.append({
                            "drug_name": drug_name,  # Original name
                            "cpic_drug_name": matched_cpic_name,  # Matched CPIC name
                            "gene": gene_symbol,
                            "gene_name": "",
                            "relationship_type": "metabolism",
                            "evidence_level": "CPIC",
                            "clinical_significance": "",
                            "cpic_level": "",
                            "guideline_id": "",
                            "guideline_url": "",
                            "source": "CPIC_PAIRS_FUZZY_MATCHED"
                        })
                found_count += 1
                # Rate limiting
                time.sleep(rate_limit_delay)
                continue
        
        # Try PubMed search if enabled and no CPIC match found (skip in local-only mode)
        if use_pubmed and not local_only and PUBMED_AVAILABLE and not matched_drug_info:
            logger.info(f"Searching PubMed for {drug_name}...")
            try:
                pubmed_results = find_genes_for_drug_via_pubmed(drug_name)
                if pubmed_results:
                    for gene, articles in pubmed_results.items():
                        # Use the first article as evidence
                        article = articles[0] if articles else {}
                        mappings.append({
                            "drug_name": drug_name,
                            "cpic_drug_name": drug_name,  # No CPIC match
                            "gene": gene,
                            "gene_name": "",
                            "relationship_type": "metabolism",  # Default
                            "evidence_level": "PubMed",
                            "clinical_significance": article.get("title", "")[:200] if article else "",
                            "cpic_level": "",
                            "guideline_id": article.get("pmid", ""),
                            "guideline_url": f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', '')}" if article.get("pmid") else "",
                            "source": "PubMed"
                        })
                    if pubmed_results:
                        found_count += 1
                        logger.info(f"Found {len(pubmed_results)} gene associations via PubMed for {drug_name}")
                        time.sleep(rate_limit_delay)
                        continue
            except Exception as e:
                logger.warning(f"Error searching PubMed for {drug_name}: {e}")
        
        # Also try loading directly from CPIC pairs (Excel or CSV) if fuzzy match found
        # (skip this extra work when local_only is True and we already populated genes
        # from load_cpic_drug_list_from_file).
        cpic_excel_path = PROJECT_ROOT / "5_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
        cpic_pairs_path = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpicPairs.csv"
        
        cpic_df = None
        if cpic_excel_path.exists() and matched_cpic_name != drug_name:
            try:
                cpic_df = pd.read_excel(cpic_excel_path)
                # Standardize column names
                if 'Drug' not in cpic_df.columns:
                    drug_cols = [col for col in cpic_df.columns if 'drug' in col.lower()]
                    if drug_cols:
                        cpic_df = cpic_df.rename(columns={drug_cols[0]: 'Drug'})
                if 'Gene' not in cpic_df.columns:
                    gene_cols = [col for col in cpic_df.columns if 'gene' in col.lower()]
                    if gene_cols:
                        cpic_df = cpic_df.rename(columns={gene_cols[0]: 'Gene'})
            except Exception as e:
                logger.debug(f"Error loading from CPIC Excel: {e}")
        
        if cpic_df is None and cpic_pairs_path.exists() and matched_cpic_name != drug_name:
            try:
                cpic_df = pd.read_csv(cpic_pairs_path)
            except Exception as e:
                logger.debug(f"Error loading from CPIC CSV: {e}")
        
        if not local_only and cpic_df is not None and 'Drug' in cpic_df.columns:
            try:
                drug_rows = cpic_df[cpic_df['Drug'].astype(str).str.upper() == matched_cpic_name.upper()]
                if not drug_rows.empty:
                    for _, row in drug_rows.iterrows():
                        mappings.append({
                            "drug_name": drug_name,  # Original name
                            "cpic_drug_name": row['Drug'],  # CPIC name
                            "gene": row['Gene'],
                            "gene_name": "",
                            "relationship_type": "metabolism",
                            "evidence_level": "CPIC",
                            "clinical_significance": "",
                            "cpic_level": row.get('CPIC Level', ''),
                            "guideline_id": "",
                            "guideline_url": row.get('Guideline', ''),
                            "source": "CPIC_PAIRS_FUZZY_MATCHED"
                        })
                    found_count += 1
                    time.sleep(rate_limit_delay)
                    continue
            except Exception as e:
                logger.debug(f"Error loading from CPIC pairs CSV: {e}")
        
        # Fetch CPIC guidelines for this drug (using matched name if fuzzy matched),
        # unless we are in local-only mode.
        guidelines = [] if local_only else fetch_cpic_guidelines(search_name)
        
        if guidelines:
            found_count += 1
            for guideline in guidelines:
                # Extract relevant information from guideline
                guideline_id = guideline.get("id", "")
                guideline_url = guideline.get("url", "")
                cpic_level = guideline.get("cpicLevel", "")
                
                # Get drug and gene information
                drug_info = guideline.get("drug", {})
                gene_info = guideline.get("gene", {})
                
                if isinstance(drug_info, dict):
                    drug_display_name = drug_info.get("name", drug_name)
                else:
                    drug_display_name = drug_name
                
                if isinstance(gene_info, dict):
                    gene_symbol = gene_info.get("symbol", "")
                    gene_name = gene_info.get("name", "")
                else:
                    gene_symbol = ""
                    gene_name = ""
                
                # Determine relationship type from guideline
                relationship_type = "metabolism"  # Default
                if "metaboliz" in str(guideline).lower():
                    relationship_type = "metabolism"
                elif "transport" in str(guideline).lower():
                    relationship_type = "transport"
                elif "target" in str(guideline).lower():
                    relationship_type = "target"
                
                # Extract clinical significance
                clinical_significance = guideline.get("recommendation", "")
                if not clinical_significance:
                    clinical_significance = guideline.get("summary", "")
                
                mappings.append({
                    "drug_name": drug_name,  # Original name
                    "cpic_drug_name": drug_display_name,  # CPIC name (may differ)
                    "gene": gene_symbol,
                    "gene_name": gene_name,
                    "relationship_type": relationship_type,
                    "evidence_level": "CPIC",
                    "clinical_significance": clinical_significance[:200] if clinical_significance else "",
                    "cpic_level": cpic_level,
                    "guideline_id": guideline_id,
                    "guideline_url": guideline_url,
                    "source": "CPIC_API" if matched_cpic_name == drug_name else "CPIC_API_FUZZY_MATCHED"
                })
        else:
            # Try alternative search: fetch drug info directly
            drug_info = fetch_cpic_drug(drug_name)
            if drug_info:
                found_count += 1
                # Extract gene associations from drug info
                # Note: CPIC API structure may vary, adjust as needed
                gene_symbols = drug_info.get("genes", [])
                if isinstance(gene_symbols, str):
                    gene_symbols = [gene_symbols]
                
                for gene_symbol in gene_symbols:
                    mappings.append({
                        "drug_name": drug_name,  # Original name
                        "cpic_drug_name": drug_info.get("name", search_name),  # CPIC name
                        "gene": gene_symbol,
                        "gene_name": "",
                        "relationship_type": "metabolism",
                        "evidence_level": "CPIC",
                        "clinical_significance": "",
                        "cpic_level": "",
                        "guideline_id": "",
                        "guideline_url": "",
                        "source": "CPIC_API" if matched_cpic_name == drug_name else "CPIC_API_FUZZY_MATCHED"
                    })
        
        # Rate limiting: be respectful to the API
        time.sleep(rate_limit_delay)
    
    # Create DataFrame
    if mappings:
        mappings_df = pd.DataFrame(mappings)
        logger.info(f"Found CPIC guidelines for {found_count}/{len(drug_names)} drugs")
        if use_fuzzy_matching:
            logger.info(f"Fuzzy matched {fuzzy_matched_count} drugs")
        logger.info(f"Total drug-gene mappings: {len(mappings_df)}")
    else:
        logger.warning("No CPIC guidelines found for any drugs")
        mappings_df = pd.DataFrame(columns=[
            "drug_name", "cpic_drug_name", "gene", "gene_name", "relationship_type",
            "evidence_level", "clinical_significance", "cpic_level",
            "guideline_id", "guideline_url", "source"
        ])
    
    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mappings_df.to_csv(output_path, index=False)
        logger.info(f"Saved drug-gene mappings to {output_path}")
    
    return mappings_df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Map drugs to pharmacogenomic genes")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--drugs", nargs="+", help="List of drug names (optional)")
    parser.add_argument("--output", help="Output CSV path (optional)")
    parser.add_argument("--use-pubmed", action="store_true", help="Also search PubMed for drug-gene relationships")
    
    args = parser.parse_args()
    
    # If drugs not provided, try to extract from feature importance
    if not args.drugs:
        age_band_fname = args.age_band.replace("-", "_")
        # Prefer local feature_importance outputs; fall back to from_s3/by_cohort
        # Step 3 saves to outputs/{cohort}/{cohort}_{age_band}_aggregated_feature_importance.csv
        fi_outputs = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "outputs"
            / args.cohort
            / f"{args.cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
        # Also check legacy location (without cohort subdirectory)
        fi_outputs_legacy = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "outputs"
            / f"{args.cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
        fi_from_s3 = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "from_s3"
            / "by_cohort"
            / args.cohort
            / args.age_band
            / f"{args.cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )

        feature_importance_path = None
        if fi_outputs.exists():
            feature_importance_path = fi_outputs
        elif fi_outputs_legacy.exists():
            feature_importance_path = fi_outputs_legacy
        elif fi_from_s3.exists():
            feature_importance_path = fi_from_s3
        else:
            # Try downloading from S3 if not found locally
            try:
                import boto3
                from botocore.exceptions import ClientError
                s3_client = boto3.client("s3")
                s3_bucket = "pgxdatalake"
                
                # Try multiple S3 paths
                s3_paths_to_try = [
                    f"gold/feature_importance/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_aggregated_feature_importance.csv",
                    f"gold/feature_importance/aggregated/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_aggregated_feature_importance.csv",
                ]
                
                for s3_key in s3_paths_to_try:
                    try:
                        s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
                        # Download to local outputs directory (with cohort subdirectory)
                        fi_outputs.parent.mkdir(parents=True, exist_ok=True)
                        s3_client.download_file(s3_bucket, s3_key, str(fi_outputs))
                        logger.info(f"Downloaded feature importance from S3: s3://{s3_bucket}/{s3_key} -> {fi_outputs}")
                        feature_importance_path = fi_outputs
                        break
                    except ClientError:
                        continue
            except Exception as e:
                logger.debug(f"Could not download from S3: {e}")

        if feature_importance_path and feature_importance_path.exists():
            df_features = pd.read_csv(feature_importance_path)
            
            # Try to load FP-Growth drug itemsets to identify which items are drugs
            fpgrowth_itemsets_path = (
                PROJECT_ROOT
                / "5b_fpgrowth_analysis"
                / "outputs"
                / args.cohort
                / "target"
                / age_band_fname
                / "train"
                / "drug_name_itemsets_target_only.json"
            )
            
            drug_set = set()
            if fpgrowth_itemsets_path.exists():
                try:
                    with open(fpgrowth_itemsets_path, 'r') as f:
                        itemsets_data = json.load(f)
                    for row in itemsets_data:
                        for item in row.get("itemsets", []):
                            drug_set.add(item.upper())
                    logger.info(f"Loaded {len(drug_set)} drugs from FP-Growth itemsets")
                except Exception as e:
                    logger.warning(f"Error loading FP-Growth itemsets: {e}")
            
            # Extract drug names from features
            # Features are in format: item_AMOXICILLIN, item_F1120, etc.
            # Drugs are those that match items in FP-Growth drug itemsets
            if drug_set:
                drug_features = df_features[
                    df_features["feature"].str.replace("item_", "", regex=False).str.upper().isin(drug_set)
                ]
            else:
                # Fallback: identify drugs by pattern (uppercase, no numbers at start, not ICD/CPT patterns)
                # This is a heuristic - drugs are typically all uppercase letters/spaces
                drug_features = df_features[
                    ~df_features["feature"].str.match(r"^item_[A-Z]\d", na=False) &  # Not ICD codes (A-Z followed by digits)
                    ~df_features["feature"].str.match(r"^item_\d{5}", na=False) &    # Not CPT codes (5 digits)
                    df_features["feature"].str.contains(r"^item_[A-Z]{2,}", na=False)  # Starts with uppercase letters
                ]
            
            if len(drug_features) > 0:
                # Extract drug names (remove item_ prefix)
                drug_names = drug_features["feature"].str.replace("item_", "", regex=False).unique().tolist()
                args.drugs = drug_names
                logger.info(f"Extracted {len(args.drugs)} drugs from feature importance")
            else:
                logger.warning("No drug features found in feature importance file")
                logger.info(f"Sample features: {df_features['feature'].head(10).tolist()}")
                args.drugs = []
        else:
            logger.error(
                "Feature importance file not found at either outputs or from_s3 for "
                f"{args.cohort}, {args.age_band}"
            )
            logger.error(
                "Please provide --drugs argument or ensure feature importance step is complete"
            )
            return
    
    # Set output path
    if not args.output:
        args.output = (
            PROJECT_ROOT / "5_pgx_analysis" / "outputs" / args.cohort /
            args.age_band.replace("-", "_") /
            f"{args.cohort}_{args.age_band.replace('-', '_')}_drug_gene_mappings.csv"
        )
    
    # Map drugs to genes
    mappings = map_drugs_to_genes(
        drug_names=args.drugs,
        output_path=args.output,
        rate_limit_delay=0.0,
        use_fuzzy_matching=True,
        fuzzy_threshold=80,
        use_pubmed=False,
        local_only=True,
    )
    
    print(f"\nMapped {len(mappings)} drug-gene relationships")
    print(f"Unique drugs: {mappings['drug_name'].nunique()}")
    print(f"Unique genes: {mappings['gene'].nunique()}")
    print("\nTop genes by frequency:")
    print(mappings['gene'].value_counts().head(10))


if __name__ == "__main__":
    main()

