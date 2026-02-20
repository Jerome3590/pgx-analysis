#!/usr/bin/env python3
"""
Test PharmGKB API to see what data we can actually retrieve.

This script tests all available endpoints to understand the data structure
and content available for VIP genes.
"""

import json
import time
import requests
from pprint import pprint


PHARMGKB_API_BASE = "https://api.pharmgkb.org/v1"
TEST_GENES = ["CYP2D6", "CYP2C19", "SLCO1B1"]  # Common VIP genes

# Rate limiting
REQUEST_DELAY = 0.5


def api_get(endpoint: str, params: dict = None):
    """Make GET request to PharmGKB API."""
    url = f"{PHARMGKB_API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    time.sleep(REQUEST_DELAY)
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Status: {e.response.status_code}")
            print(f"   Response: {e.response.text[:500]}")
        return None


def test_gene_endpoint(gene_symbol: str):
    """Test /data/gene endpoint."""
    print(f"\n{'='*80}")
    print(f"Testing Gene Endpoint: {gene_symbol}")
    print(f"{'='*80}")
    
    print(f"\n1. GET /data/gene?symbol={gene_symbol}")
    data = api_get("/data/gene", params={"symbol": gene_symbol})
    
    if data and "data" in data:
        gene_data = data["data"]
        
        # Handle if data is a list (multiple results) or single object
        if isinstance(gene_data, list):
            if not gene_data:
                print(f"❌ No genes found")
                return None
            print(f"✓ Found {len(gene_data)} gene(s), using first match")
            gene_data = gene_data[0]
        
        print(f"✓ Gene found: {gene_data.get('name')}")
        print(f"  ID: {gene_data.get('id')}")
        print(f"  Chromosome: {gene_data.get('chromosome')}")
        print(f"  VIP: {gene_data.get('vip', False)}")
        print(f"  Has variant annotation: {gene_data.get('hasVariantAnnotation', False)}")
        print(f"  Has CPIC guideline: {gene_data.get('hasCpicDosingGuideline', False)}")
        
        # Show available fields
        print(f"\n  Available fields in gene data:")
        for key in sorted(gene_data.keys()):
            value = gene_data[key]
            if isinstance(value, (list, dict)):
                print(f"    - {key}: {type(value).__name__} (length: {len(value) if isinstance(value, list) else 'N/A'})")
                # Show nested dict content for important fields
                if key in ["vipSummary", "vipCitation", "chr"] and isinstance(value, dict):
                    print(f"      Keys: {', '.join(value.keys())}")
            else:
                print(f"    - {key}: {value}")
        
        # Extract VIP summary text if available
        if "vipSummary" in gene_data and isinstance(gene_data["vipSummary"], dict):
            print(f"\n  VIP Summary Content:")
            vip_summary = gene_data["vipSummary"]
            for key, val in vip_summary.items():
                if isinstance(val, str) and val:
                    print(f"    {key}: {val[:200]}{'...' if len(val) > 200 else ''}")
        
        return gene_data.get('id')
    else:
        print(f"❌ No data returned")
        return None


def test_clinical_annotations(gene_id: str):
    """Test /data/clinicalAnnotation endpoint."""
    print(f"\n2. GET /data/clinicalAnnotation?geneId={gene_id}")
    data = api_get("/data/clinicalAnnotation", params={"geneId": gene_id})
    
    if data and "data" in data:
        annotations = data["data"]
        print(f"✓ Found {len(annotations)} clinical annotations")
        
        if annotations:
            print(f"\n  First annotation structure:")
            first = annotations[0]
            for key in sorted(first.keys()):
                value = first[key]
                if isinstance(value, (list, dict)):
                    print(f"    - {key}: {type(value).__name__} (length: {len(value) if isinstance(value, list) else 'N/A'})")
                else:
                    print(f"    - {key}: {str(value)[:100]}")
            
            print(f"\n  Sample annotation text:")
            if "annotationText" in first:
                print(f"    {first['annotationText'][:300]}...")
            elif "text" in first:
                print(f"    {first['text'][:300]}...")
        
        return annotations
    else:
        print(f"❌ No clinical annotations found")
        return []


def test_variant_annotations(gene_id: str):
    """Test /data/variantAnnotation endpoint."""
    print(f"\n3. GET /data/variantAnnotation?geneId={gene_id}")
    data = api_get("/data/variantAnnotation", params={"geneId": gene_id})
    
    if data and "data" in data:
        variants = data["data"]
        print(f"✓ Found {len(variants)} variant annotations")
        
        if variants:
            print(f"\n  First variant structure:")
            first = variants[0]
            for key in sorted(first.keys()):
                value = first[key]
                if isinstance(value, (list, dict)):
                    print(f"    - {key}: {type(value).__name__}")
                    if key == "variant" and isinstance(value, dict):
                        print(f"      Variant: {value.get('name', 'N/A')}")
                else:
                    print(f"    - {key}: {str(value)[:100]}")
        
        return variants
    else:
        print(f"❌ No variant annotations found")
        return []


def test_drug_labels(gene_id: str):
    """Test /data/drugLabel endpoint."""
    print(f"\n4. GET /data/drugLabel?geneId={gene_id}")
    data = api_get("/data/drugLabel", params={"geneId": gene_id})
    
    if data and "data" in data:
        labels = data["data"]
        print(f"✓ Found {len(labels)} drug labels")
        
        if labels:
            print(f"\n  First label structure:")
            first = labels[0]
            for key in sorted(first.keys()):
                value = first[key]
                if isinstance(value, (list, dict)):
                    print(f"    - {key}: {type(value).__name__} (length: {len(value) if isinstance(value, list) else 'N/A'})")
                else:
                    print(f"    - {key}: {str(value)[:100]}")
            
            if "textMarkdown" in first and first["textMarkdown"]:
                print(f"\n  Sample label text:")
                print(f"    {first['textMarkdown'][:300]}...")
        
        return labels
    else:
        print(f"❌ No drug labels found")
        return []


def test_related_chemicals(gene_id: str):
    """Test /data/chemical endpoint."""
    print(f"\n5. GET /data/chemical?relatedGeneId={gene_id}")
    data = api_get("/data/chemical", params={"relatedGeneId": gene_id})
    
    if data and "data" in data:
        chemicals = data["data"]
        print(f"✓ Found {len(chemicals)} related chemicals")
        
        if chemicals:
            print(f"\n  First few chemicals:")
            for i, chem in enumerate(chemicals[:5], 1):
                print(f"    {i}. {chem.get('name')} (ID: {chem.get('id')}, Type: {chem.get('type', 'N/A')})")
            
            print(f"\n  First chemical structure:")
            first = chemicals[0]
            for key in sorted(first.keys()):
                value = first[key]
                if isinstance(value, (list, dict)):
                    print(f"    - {key}: {type(value).__name__}")
                else:
                    print(f"    - {key}: {str(value)[:100]}")
        
        return chemicals
    else:
        print(f"❌ No related chemicals found")
        return []


def test_guidelines(gene_id: str):
    """Test /data/guidelineAnnotation endpoint."""
    print(f"\n6. GET /data/guidelineAnnotation?geneId={gene_id}")
    data = api_get("/data/guidelineAnnotation", params={"geneId": gene_id})
    
    if data and "data" in data:
        guidelines = data["data"]
        print(f"✓ Found {len(guidelines)} guidelines")
        
        if guidelines:
            print(f"\n  Guidelines:")
            for i, guide in enumerate(guidelines[:3], 1):
                print(f"    {i}. {guide.get('name', 'N/A')}")
                print(f"       Source: {guide.get('source', 'N/A')}")
                if "chemicals" in guide:
                    print(f"       Chemicals: {', '.join([c.get('name', '') for c in guide.get('chemicals', [])][:3])}")
        
        return guidelines
    else:
        print(f"❌ No guidelines found")
        return []


def save_sample_report(gene_symbol: str, gene_id: str, results: dict):
    """Save a complete sample report to JSON."""
    filename = f"test_report_{gene_symbol}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved complete report to {filename}")


def main():
    """Run API tests for multiple genes."""
    print("="*80)
    print("PharmGKB API Testing")
    print("="*80)
    print(f"\nTesting {len(TEST_GENES)} genes: {', '.join(TEST_GENES)}")
    print(f"API Base: {PHARMGKB_API_BASE}")
    
    all_gene_data = []
    
    for gene_symbol in TEST_GENES:
        # First get full gene data
        print(f"\n{'='*80}")
        print(f"Testing Gene Endpoint: {gene_symbol}")
        print(f"{'='*80}")
        
        print(f"\n1. GET /data/gene?symbol={gene_symbol}")
        gene_response = api_get("/data/gene", params={"symbol": gene_symbol})
        
        if gene_response and "data" in gene_response:
            gene_data_list = gene_response["data"]
            if isinstance(gene_data_list, list) and gene_data_list:
                gene_data = gene_data_list[0]
                all_gene_data.append(gene_data)
                gene_id = gene_data.get('id')
                
                # Show gene info
                print(f"✓ Found {len(gene_data_list)} gene(s), using first match")
                print(f"✓ Gene found: {gene_data.get('name')}")
                print(f"  ID: {gene_id}")
                print(f"  VIP ID: {gene_data.get('vipId')}")
                print(f"  CPIC Gene: {gene_data.get('cpicGene', False)}")
                print(f"  VIP Tier: {gene_data.get('vipTier', 'N/A')}")
                
                # Show all fields
                print(f"\n  Available fields:")
                for key in sorted(gene_data.keys()):
                    value = gene_data[key]
                    if isinstance(value, dict):
                        print(f"    - {key}: dict with keys: {', '.join(list(value.keys())[:10])}")
                        if key == "vipSummary" and value:
                            print(f"      VIP Summary fields: {list(value.keys())}")
                            for vip_key, vip_val in value.items():
                                if isinstance(vip_val, str) and vip_val:
                                    print(f"        {vip_key}: {vip_val[:150]}{'...' if len(vip_val) > 150 else ''}")
                    elif isinstance(value, list):
                        print(f"    - {key}: list with {len(value)} items")
                    else:
                        val_str = str(value)[:100]
                        print(f"    - {key}: {val_str}")
                
                # Try to get related annotations (these might not work with current API)
                results = {
                    "gene_symbol": gene_symbol,
                    "gene_id": gene_id,
                    "gene_data": gene_data,
                    "clinical_annotations": test_clinical_annotations(gene_id) if gene_id else [],
                    "variant_annotations": test_variant_annotations(gene_id) if gene_id else [],
                    "drug_labels": test_drug_labels(gene_id) if gene_id else [],
                    "related_chemicals": test_related_chemicals(gene_id) if gene_id else [],
                    "guidelines": test_guidelines(gene_id) if gene_id else []
                }
                
                # Save first gene's complete report
                if gene_symbol == TEST_GENES[0]:
                    save_sample_report(gene_symbol, gene_id, results)
        
        print(f"\n{'='*80}\n")
        time.sleep(1)  # Extra delay between genes
    
    # Save all gene data for analysis
    with open("all_gene_data.json", "w", encoding="utf-8") as f:
        json.dump(all_gene_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved all gene data to all_gene_data.json")
    
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print("\nKey findings:")
    print("1. Gene data includes vipSummary dict with detailed text")
    print("2. vipId can be used for ClinPGx URLs")
    print("3. Other endpoints (clinicalAnnotation, etc.) may not be queryable directly")
    print("4. VIP page HTML scraping may be needed for comprehensive reports")
    print("\nNext steps:")
    print("1. Review all_gene_data.json for complete VIP summary structure")
    print("2. Use vipSummary text content for NLP analysis")
    print("3. Fetch VIP HTML pages for additional context")


if __name__ == "__main__":
    main()
