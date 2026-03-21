#!/usr/bin/env python3
"""
Fetch CPIC drug list (fallback script - use update_cpic_drug_list.py instead).

**Note:** The primary source is the official CPIC Excel file:
  - Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
  - Use `update_cpic_drug_list.py` to process the Excel file

This script is a fallback that attempts to fetch the CPIC drug list from:
1. CPIC API (if available)
2. CPIC website scraping
3. Static reference file

The drug list is saved as JSON for use in fuzzy matching.
"""

import sys
import json
import requests
from pathlib import Path
import logging
from bs4 import BeautifulSoup

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CPIC_API_BASE = "https://api.cpicpgx.org"
CPIC_WEBSITE = "https://cpicpgx.org"


def fetch_from_api() -> list:
    """Try to fetch drugs from CPIC API."""
    drugs = []
    
    # Try various API endpoints
    endpoints = [
        "/drug",
        "/guideline",
        "/genes-drugs"
    ]
    
    for endpoint in endpoints:
        try:
            url = f"{CPIC_API_BASE}{endpoint}"
            logger.info(f"Trying API endpoint: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    # Extract drug names
                    for item in data:
                        if isinstance(item, dict):
                            drug_name = item.get("name") or item.get("drugName") or item.get("drug")
                            if drug_name:
                                drugs.append({
                                    "name": drug_name,
                                    "source": f"API_{endpoint}",
                                    "raw": item
                                })
                elif isinstance(data, dict):
                    # Single drug or nested structure
                    drug_name = data.get("name") or data.get("drugName")
                    if drug_name:
                        drugs.append({
                            "name": drug_name,
                            "source": f"API_{endpoint}",
                            "raw": data
                        })
                
                if drugs:
                    logger.info(f"Found {len(drugs)} drugs from API endpoint {endpoint}")
                    return drugs
        except Exception as e:
            logger.debug(f"Error with endpoint {endpoint}: {e}")
            continue
    
    return []


def fetch_from_website() -> list:
    """Try to scrape drug list from CPIC website."""
    drugs = []
    
    try:
        # Try the genes-drugs page
        url = f"{CPIC_WEBSITE}/genes-drugs/"
        logger.info(f"Trying website: {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for drug names in various HTML structures
            # This is a heuristic approach - adjust based on actual page structure
            for tag in soup.find_all(['a', 'td', 'li', 'span']):
                text = tag.get_text(strip=True)
                # Simple heuristic: drug names are usually capitalized words
                if text and len(text) > 3 and text[0].isupper():
                    # Filter out common non-drug text
                    if not any(skip in text.lower() for skip in ['gene', 'guideline', 'download', 'page', 'next', 'previous']):
                        drugs.append({
                            "name": text,
                            "source": "website_scrape"
                        })
            
            if drugs:
                logger.info(f"Found {len(drugs)} potential drugs from website")
                # Deduplicate
                seen = set()
                unique_drugs = []
                for drug in drugs:
                    name_lower = drug["name"].lower()
                    if name_lower not in seen:
                        seen.add(name_lower)
                        unique_drugs.append(drug)
                return unique_drugs
    except Exception as e:
        logger.warning(f"Error scraping website: {e}")
    
    return []


def create_static_reference() -> list:
    """Create a static reference list of common CPIC drugs."""
    # Common drugs with CPIC guidelines (partial list)
    common_cpic_drugs = [
        "amitriptyline", "citalopram", "clomipramine", "desipramine", "doxepin",
        "imipramine", "nortriptyline", "paroxetine", "trimipramine",
        "carbamazepine", "phenytoin", "fosphenytoin",
        "warfarin", "acenocoumarol", "phenprocoumon",
        "codeine", "tramadol", "oxycodone", "hydrocodone",
        "tamoxifen", "toremifene",
        "azathioprine", "mercaptopurine", "thioguanine",
        "capecitabine", "fluorouracil", "tegafur",
        "abacavir", "allopurinol", "atazanavir", "boceprevir",
        "celecoxib", "clopidogrel", "fluvastatin", "metoprolol",
        "omeprazole", "pantoprazole", "propranolol", "simvastatin",
        "tacrolimus", "voriconazole"
    ]
    
    return [{"name": drug, "source": "static_reference"} for drug in common_cpic_drugs]


def main():
    """Fetch CPIC drug list and save to file."""
    output_path = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpic_drug_list.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    drugs = []
    
    # Try API first
    logger.info("Attempting to fetch CPIC drug list from API...")
    drugs = fetch_from_api()
    
    # Try website if API fails
    if not drugs:
        logger.info("API failed, trying website scraping...")
        drugs = fetch_from_website()
    
    # Use static reference as fallback
    if not drugs:
        logger.info("Using static reference list...")
        drugs = create_static_reference()
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(drugs, f, indent=2)
    
    logger.info(f"Saved {len(drugs)} drugs to {output_path}")
    
    # Print sample
    print(f"\nSample drugs ({min(10, len(drugs))} of {len(drugs)}):")
    for drug in drugs[:10]:
        print(f"  - {drug['name']} (source: {drug.get('source', 'unknown')})")
    
    return drugs


if __name__ == "__main__":
    main()

