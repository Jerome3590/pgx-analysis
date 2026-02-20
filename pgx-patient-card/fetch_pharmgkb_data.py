#!/usr/bin/env python3
"""
Fetch PharmGKB data using current API endpoints.

Migrated from R scripts (PGx.Rmd, Build_PGx_Database.Rmd) to Python.
Uses PharmGKB REST API v1 documented at:
https://www.postman.com/pharmgkb/pharmgkb-api/documentation/g9rp4zr/pharmgkb-rest-api

Updates:
- VIP URLs now point to clinpgx.org instead of pharmgkb.org
- Uses entity endpoints instead of deprecated /v1/site/vips
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from urllib.parse import urljoin


# API Configuration
PHARMGKB_API_BASE = "https://api.pharmgkb.org/v1"
CLINPGX_VIP_BASE = "https://www.clinpgx.org/vip/"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between API requests


class PharmGKBClient:
    """Client for PharmGKB REST API v1."""
    
    def __init__(self, base_url: str = PHARMGKB_API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PGx-Analysis/2.0 (Python)"
        })
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request with rate limiting."""
        url = urljoin(self.base_url, endpoint)
        time.sleep(REQUEST_DELAY)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}
    
    def get_gene_by_symbol(self, symbol: str) -> Dict:
        """
        Fetch gene data by gene symbol.
        
        Endpoint: GET /data/gene
        Params: symbol (e.g., CYP2D6)
        
        Returns gene metadata including PA ID, chromosome, variants, etc.
        """
        endpoint = "/data/gene"
        return self._get(endpoint, params={"symbol": symbol})
    
    def get_genes_batch(self, symbols: List[str]) -> List[Dict]:
        """Fetch multiple genes with rate limiting."""
        genes = []
        for symbol in symbols:
            gene_data = self.get_gene_by_symbol(symbol)
            if gene_data:
                genes.append(gene_data)
        return genes
    
    def get_vip_genes(self) -> List[Dict]:
        """
        Fetch all VIP (Very Important Pharmacogene) genes.
        
        Note: The old /v1/site/vips endpoint is deprecated.
        Current approach: Query known VIP genes or use /data/gene with filters.
        
        For now, returns a list of known VIP genes based on CPIC guidelines.
        In production, you may want to maintain a cached list or query all genes
        and filter by hasVarAnn=true or other VIP indicators.
        """
        # Known VIP genes from CPIC/PharmGKB
        vip_genes = [
            "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A5", "SLCO1B1",
            "DPYD", "TPMT", "UGT1A1", "VKORC1", "CFTR",
            "G6PD", "HLA-A", "HLA-B", "IFNL3", "NUDT15",
            "CYP4F2", "F5", "CACNA1S", "RYR1", "MT-RNR1"
        ]
        
        print(f"Fetching {len(vip_genes)} VIP genes from PharmGKB API...")
        genes_data = self.get_genes_batch(vip_genes)
        
        # Add ClinPGx VIP URL for each gene
        for gene in genes_data:
            if "data" in gene and "id" in gene["data"]:
                gene_id = gene["data"]["id"]  # PA ID
                gene["clinpgx_vip_url"] = f"{CLINPGX_VIP_BASE}{gene_id}/overview"
        
        return genes_data


def build_vip_dataframe(client: PharmGKBClient) -> List[Dict]:
    """
    Build VIP dataframe similar to R script vip_df.
    
    Returns list of dicts with:
    - gene: Gene symbol
    - gene_id: PharmGKB PA ID
    - vip_url: ClinPGx VIP URL
    - qr_filename: Filename for QR code image
    - summary: Gene summary text
    - chromosome: Chromosome location
    """
    vip_genes = client.get_vip_genes()
    
    vip_list = []
    for gene_data in vip_genes:
        if "data" not in gene_data:
            continue
        
        data = gene_data["data"]
        
        # Extract relevant fields
        vip_entry = {
            "gene": data.get("symbol", ""),
            "gene_id": data.get("id", ""),  # PA ID
            "vip_url": gene_data.get("clinpgx_vip_url", ""),
            "qr_filename": f"{data.get('symbol', 'unknown')}.png",
            "summary": data.get("name", ""),  # Full gene name
            "chromosome": data.get("chromosome", ""),
        }
        
        vip_list.append(vip_entry)
    
    return vip_list


def save_vip_data(vip_data: List[Dict], output_path: Path):
    """Save VIP data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vip_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(vip_data)} VIP genes to {output_path}")


def main():
    """Fetch PharmGKB VIP data and save to JSON."""
    # Output directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_file = data_dir / "pharmgkb_vip_genes.json"
    
    # Fetch data
    client = PharmGKBClient()
    vip_data = build_vip_dataframe(client)
    
    # Save to JSON
    save_vip_data(vip_data, output_file)
    
    # Print sample
    if vip_data:
        print("\nSample VIP gene:")
        print(json.dumps(vip_data[0], indent=2))
    
    return vip_data


if __name__ == "__main__":
    main()
