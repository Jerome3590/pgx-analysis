#!/usr/bin/env python3
"""
Update CPIC drug list from official CPIC Excel file.

This script extracts unique drug names from the official CPIC Excel file
(cpic_gene-drug_pairs.xlsx) downloaded from the CPIC website and creates
a JSON file for fuzzy matching.

Primary source: Official CPIC Excel file from CPIC website
  - Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
  - Location: 5_pgx_analysis/cpic/cpic_gene-drug_pairs.xlsx

Fallback: CPIC pairs CSV file (data/cpicPairs.csv)
"""

import sys
import pandas as pd
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def update_cpic_drug_list():
    """
    Extract unique drugs from official CPIC Excel file and save to JSON.

    Primary source: Official CPIC Excel file from CPIC website
      - Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
      - Location: 5_pgx_analysis/cpic/cpic_gene-drug_pairs.xlsx

    Fallback: CPIC pairs CSV file (data/cpicPairs.csv)
    """

    # Try official CPIC Excel file first (primary source)
    cpic_excel_path = PROJECT_ROOT / "5_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
    cpic_pairs_path = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpicPairs.csv"
    output_path = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpic_drug_list.json"

    # Read CPIC pairs - prefer Excel file if available (PRIMARY SOURCE)
    source_name = None
    if cpic_excel_path.exists():
        print(f"Using official CPIC Excel file: {cpic_excel_path}")
        source_name = "cpic_gene-drug_pairs.xlsx"
        df = pd.read_excel(cpic_excel_path)
        # Standardize column names (may vary)
        if 'Drug' not in df.columns:
            # Try common variations
            drug_cols = [col for col in df.columns if 'drug' in col.lower() or 'medication' in col.lower()]
            if drug_cols:
                df = df.rename(columns={drug_cols[0]: 'Drug'})
        if 'Gene' not in df.columns:
            gene_cols = [col for col in df.columns if 'gene' in col.lower()]
            if gene_cols:
                df = df.rename(columns={gene_cols[0]: 'Gene'})
    elif cpic_pairs_path.exists():
        print(f"Using CPIC pairs CSV: {cpic_pairs_path}")
        source_name = "cpicPairs.csv"
        df = pd.read_csv(cpic_pairs_path)
    else:
        print(f"Error: CPIC pairs file not found at {cpic_excel_path} or {cpic_pairs_path}")
        return

    # Extract unique drugs
    unique_drugs = df['Drug'].unique().tolist()

    # Create drug list with metadata
    drugs_list = []
    for drug in sorted(unique_drugs):
        # Get all genes associated with this drug
        drug_rows = df[df['Drug'] == drug]
        genes = drug_rows['Gene'].unique().tolist()
        guidelines = drug_rows['Guideline'].unique().tolist()
        cpic_levels = drug_rows['CPIC Level'].unique().tolist()

        drugs_list.append({
            "name": drug,
            "source": source_name,
            "genes": genes,
            "guideline_count": len(guidelines),
            "cpic_levels": cpic_levels.tolist() if isinstance(cpic_levels, pd.Series) else cpic_levels
        })

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(drugs_list, f, indent=2)

    print(f"Extracted {len(drugs_list)} unique drugs from CPIC pairs")
    print(f"Saved to {output_path}")
    print("\nSample drugs (first 10):")
    for drug in drugs_list[:10]:
        print(f"  - {drug['name']} ({len(drug['genes'])} genes)")

    return drugs_list


if __name__ == "__main__":
    update_cpic_drug_list()

