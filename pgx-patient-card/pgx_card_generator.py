#!/usr/bin/env python3
"""
Lambda-compatible PGx card generator using CPIC and PharmGKB data.

This module can be imported by Lambda (10_risk_dashboard/backend/lambda_function.py)
or used standalone. It provides the core PGx card generation logic without
depending on the R scripts.

Key differences from Lambda's current implementation:
1. Adds ClinPGx VIP URLs for genes (not just CPIC guideline URLs)
2. Supports QR code generation (if needed for exports)
3. Uses cached PharmGKB VIP data (JSON) to avoid API calls at runtime
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class PGxCardGenerator:
    """Generate PGx patient cards from genetic variants."""
    
    def __init__(
        self,
        cpic_data: Optional[pd.DataFrame] = None,
        vip_data: Optional[Dict[str, str]] = None
    ):
        """
        Initialize with CPIC and VIP data.
        
        Args:
            cpic_data: CPIC gene-drug pairs DataFrame
            vip_data: Dict mapping gene symbol -> ClinPGx VIP URL
        """
        self.cpic_data = cpic_data
        self.vip_data = vip_data or {}
    
    @classmethod
    def from_files(cls, cpic_excel_path: Path, vip_json_path: Optional[Path] = None):
        """Load from CPIC Excel and optional VIP JSON."""
        # Load CPIC data
        cpic_df = pd.read_excel(cpic_excel_path)
        
        # Load VIP data if available
        vip_dict = {}
        if vip_json_path and vip_json_path.exists():
            with open(vip_json_path, encoding="utf-8") as f:
                vip_list = json.load(f)
            vip_dict = {
                item["gene"].upper(): item["vip_url"]
                for item in vip_list
                if "gene" in item and "vip_url" in item
            }
        
        return cls(cpic_data=cpic_df, vip_data=vip_dict)
    
    def generate_card(
        self,
        variants: List[Dict[str, any]],
        timestamp: str,
        ip_address: str,
        patient_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate PGx card from genetic variants.
        
        Args:
            variants: List of dicts with 'gene' and 'variants' keys
            timestamp: Generation timestamp
            ip_address: Request IP address
            patient_id: Optional patient identifier
        
        Returns:
            PGx card data with genes, drugs, and VIP URLs
        """
        if self.cpic_data is None:
            raise ValueError("CPIC data not loaded")
        
        genes_processed = []
        drugs_found = []
        
        # Find CPIC gene column
        gene_col = next(
            (c for c in self.cpic_data.columns if "gene" in c.lower()),
            "Gene"
        )
        drug_col = next(
            (c for c in self.cpic_data.columns if "drug" in c.lower()),
            "Drug"
        )
        guideline_col = next(
            (c for c in self.cpic_data.columns if "guideline" in c.lower()),
            None
        )
        cpic_level_col = next(
            (c for c in self.cpic_data.columns if "cpic" in c.lower() and "level" in c.lower()),
            None
        )
        
        # Process variants
        for variant in variants:
            gene = variant.get("gene", "").upper().strip()
            variant_list = variant.get("variants", [])
            
            if not gene or not variant_list:
                continue
            
            # Store gene info
            gene_entry = {
                "gene": gene,
                "variants": variant_list,
                "allele_count": len([v for v in variant_list if v and v != "0"])
            }
            
            # Add ClinPGx VIP URL if available
            if gene in self.vip_data:
                gene_entry["vip_url"] = self.vip_data[gene]
            
            genes_processed.append(gene_entry)
            
            # Find drugs from CPIC data
            gene_cpic_data = self.cpic_data[
                self.cpic_data[gene_col].str.upper().str.strip() == gene
            ]
            
            for _, row in gene_cpic_data.iterrows():
                drug = row[drug_col]
                
                # Avoid duplicates
                if not any(d["drug"] == drug and d["gene"] == gene for d in drugs_found):
                    drug_entry = {
                        "gene": gene,
                        "drug": drug,
                        "guideline_url": row[guideline_col] if guideline_col and pd.notna(row[guideline_col]) else "",
                        "cpic_level": row[cpic_level_col] if cpic_level_col and pd.notna(row[cpic_level_col]) else "",
                    }
                    
                    # Add VIP URL for gene context
                    if gene in self.vip_data:
                        drug_entry["gene_vip_url"] = self.vip_data[gene]
                    
                    drugs_found.append(drug_entry)
        
        # Build result
        result = {
            "timestamp": timestamp,
            "ip_address": ip_address,
            "genes": genes_processed,
            "drugs": drugs_found
        }
        
        if patient_id:
            result["patient_id"] = patient_id
        
        return result


def update_lambda_with_vip_urls():
    """
    Helper to show how to update Lambda function with VIP URL support.
    
    The Lambda can load pharmgkb_vip_genes.json at startup and use it
    to augment the PGx card response with ClinPGx VIP URLs.
    """
    example_code = """
# Add to lambda_function.py at module level:

VIP_URL_CACHE = {}

def load_vip_urls():
    '''Load VIP URLs from container or S3.'''
    global VIP_URL_CACHE
    
    # Try container first
    vip_path = '/var/task/data/pharmgkb_vip_genes.json'
    if os.path.exists(vip_path):
        with open(vip_path) as f:
            vip_data = json.load(f)
        VIP_URL_CACHE = {
            item['gene'].upper(): item['vip_url']
            for item in vip_data
            if 'gene' in item and 'vip_url' in item
        }
        return
    
    # Fallback to S3
    try:
        s3_key = f'{METADATA_PREFIX}/pharmgkb_vip_genes.json'
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        vip_data = json.loads(obj['Body'].read().decode('utf-8'))
        VIP_URL_CACHE = {
            item['gene'].upper(): item['vip_url']
            for item in vip_data
            if 'gene' in item and 'vip_url' in item
        }
    except Exception as e:
        print(f'Warning: Could not load VIP URLs: {e}')

# Call at Lambda initialization
load_vip_urls()

# Update generate_pgx_card() to add VIP URLs:
def generate_pgx_card(variants, timestamp, ip_address, patient_id=None):
    # ... existing code ...
    
    for gene_entry in genes_processed:
        gene = gene_entry['gene']
        if gene in VIP_URL_CACHE:
            gene_entry['vip_url'] = VIP_URL_CACHE[gene]
    
    # ... rest of existing code ...
"""
    print(example_code)


if __name__ == "__main__":
    # Example usage
    print("PGx Card Generator - Lambda Compatible Module")
    print("=" * 60)
    print("\nThis module provides Lambda-compatible PGx card generation.")
    print("\nTo integrate with Lambda:")
    print("1. Run fetch_pharmgkb_data.py to generate VIP data")
    print("2. Copy pharmgkb_vip_genes.json to Lambda container data/")
    print("3. Import PGxCardGenerator in lambda_function.py")
    print("\nOr use the simpler approach of loading VIP URLs at startup:")
    update_lambda_with_vip_urls()
