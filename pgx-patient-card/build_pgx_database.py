#!/usr/bin/env python3
"""
Build PGx patient database combining CPIC and PharmGKB data.

Migrated from R scripts (Build_PGx_Database.Rmd) to Python.
Combines:
- CPIC gene-drug pairs (from Excel file)
- PharmGKB VIP gene data
- QR code mappings for patient cards

Output: Merged database ready for patient card generation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def load_cpic_data(cpic_excel_path: Path) -> pd.DataFrame:
    """
    Load CPIC gene-drug pairs from Excel file.
    
    Expected columns: Gene, Drug, Guideline, CPIC Level, etc.
    """
    if not cpic_excel_path.exists():
        raise FileNotFoundError(
            f"CPIC Excel file not found: {cpic_excel_path}\n"
            "Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx"
        )
    
    df = pd.read_excel(cpic_excel_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Expected columns (case-insensitive match)
    gene_col = next((c for c in df.columns if "gene" in c.lower()), None)
    drug_col = next((c for c in df.columns if "drug" in c.lower()), None)
    
    if not gene_col or not drug_col:
        raise ValueError(f"CPIC Excel missing required columns. Found: {list(df.columns)}")
    
    print(f"Loaded {len(df)} gene-drug pairs from CPIC")
    return df


def load_vip_data(vip_json_path: Path) -> pd.DataFrame:
    """Load PharmGKB VIP gene data from JSON."""
    if not vip_json_path.exists():
        raise FileNotFoundError(
            f"VIP data not found: {vip_json_path}\n"
            "Run fetch_pharmgkb_data.py first to generate VIP gene data."
        )
    
    with open(vip_json_path, encoding="utf-8") as f:
        vip_data = json.load(f)
    
    df = pd.DataFrame(vip_data)
    print(f"Loaded {len(df)} VIP genes from PharmGKB")
    return df


def load_qr_mappings(qr_json_path: Path) -> pd.DataFrame:
    """Load QR code mappings from JSON."""
    if not qr_json_path.exists():
        print(f"Warning: QR mappings not found at {qr_json_path}")
        print("Run generate_pgx_qr_codes.py to create QR codes")
        return pd.DataFrame()
    
    with open(qr_json_path, encoding="utf-8") as f:
        qr_data = json.load(f)
    
    df = pd.DataFrame(qr_data)
    print(f"Loaded {len(df)} QR code mappings")
    return df


def merge_pgx_database(
    cpic_df: pd.DataFrame,
    vip_df: pd.DataFrame,
    qr_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge CPIC, VIP, and QR code data into unified PGx database.
    
    Joins on gene symbol (case-insensitive).
    """
    # Standardize gene column names
    gene_col = next((c for c in cpic_df.columns if "gene" in c.lower()), "Gene")
    cpic_df = cpic_df.rename(columns={gene_col: "gene"})
    cpic_df["gene"] = cpic_df["gene"].str.upper().str.strip()
    
    vip_df["gene"] = vip_df["gene"].str.upper().str.strip()
    
    # Merge CPIC and VIP data
    merged = cpic_df.merge(
        vip_df,
        on="gene",
        how="left",
        suffixes=("_cpic", "_vip")
    )
    
    # Add QR code paths if available
    if qr_df is not None and not qr_df.empty:
        qr_df["gene"] = qr_df["gene"].str.upper().str.strip()
        merged = merged.merge(
            qr_df[["gene", "qr_path", "qr_filename"]],
            on="gene",
            how="left"
        )
    
    print(f"\nMerged database: {len(merged)} rows")
    print(f"Unique genes: {merged['gene'].nunique()}")
    print(f"Unique drugs: {merged['Drug'].nunique() if 'Drug' in merged.columns else 'N/A'}")
    
    return merged


def save_pgx_database(df: pd.DataFrame, output_dir: Path):
    """Save PGx database to multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / "pgx_database.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = output_dir / "pgx_database.json"
    df.to_json(json_path, orient="records", indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    # Save as Excel
    excel_path = output_dir / "pgx_database.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"✓ Saved Excel: {excel_path}")
    
    # Save summary statistics
    summary = {
        "total_rows": len(df),
        "unique_genes": int(df["gene"].nunique()),
        "unique_drugs": int(df["Drug"].nunique()) if "Drug" in df.columns else 0,
        "genes_with_vip_url": int(df["vip_url"].notna().sum()) if "vip_url" in df.columns else 0,
        "genes_with_qr_code": int(df["qr_path"].notna().sum()) if "qr_path" in df.columns else 0,
    }
    
    summary_path = output_dir / "pgx_database_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    return summary


def main():
    """Build complete PGx patient database."""
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = data_dir / "pgx_database"
    
    # Input files
    cpic_excel = data_dir / "cpic_gene-drug_pairs.xlsx"
    vip_json = data_dir / "pharmgkb_vip_genes.json"
    qr_json = data_dir / "qr_code_mappings.json"
    
    print("Building PGx patient database...\n")
    
    # Load data sources
    print("1. Loading CPIC data...")
    cpic_df = load_cpic_data(cpic_excel)
    
    print("\n2. Loading PharmGKB VIP data...")
    vip_df = load_vip_data(vip_json)
    
    print("\n3. Loading QR code mappings...")
    qr_df = load_qr_mappings(qr_json)
    
    # Merge databases
    print("\n4. Merging databases...")
    merged_df = merge_pgx_database(cpic_df, vip_df, qr_df if not qr_df.empty else None)
    
    # Save outputs
    print("\n5. Saving outputs...")
    summary = save_pgx_database(merged_df, output_dir)
    
    print("\n" + "="*60)
    print("PGx Database Summary:")
    print("="*60)
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("="*60)
    
    return merged_df


if __name__ == "__main__":
    main()
