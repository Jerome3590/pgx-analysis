#!/usr/bin/env python3
"""
One-shot utility to build / refresh the global PGx lookup tables from existing
per-cohort / per-age-band artifacts.

This script scans `5_pgx_analysis/outputs/**` for:
- `*_drug_gene_mappings.csv`
- `*_allele_frequencies.csv`

and writes consolidated, de-duplicated global tables:
- `5_pgx_analysis/outputs/global/pgx_drug_gene_mappings_global.csv`
- `5_pgx_analysis/outputs/global/pgx_allele_frequencies_global.csv`

You can run this once to pre-populate the global cache from prior CPIC/API work,
so subsequent PGx feature generation can reuse these lookups across all cohorts
and age bands.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def build_global_drug_gene_mappings() -> Path:
    base_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs"
    global_dir = base_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_path = global_dir / "pgx_drug_gene_mappings_global.csv"

    mapping_files = list(base_dir.rglob("*_drug_gene_mappings.csv"))
    # Exclude any existing global file from input set
    mapping_files = [p for p in mapping_files if p.name != global_path.name]

    if not mapping_files:
        print("[PGX] No existing *_drug_gene_mappings.csv artifacts found; skipping global mapping build.")
        return global_path

    dfs = []
    for p in mapping_files:
        try:
            df = pd.read_csv(p)
            df["source_file"] = str(p.relative_to(base_dir))
            dfs.append(df)
        except Exception as exc:
            print(f"[PGX] Warning: could not read {p}: {exc}")

    if not dfs:
        print("[PGX] No valid mapping CSVs could be read; skipping global mapping build.")
        return global_path

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate by drug_name + gene where possible
    if {"drug_name", "gene"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["drug_name", "gene"])
    else:
        combined = combined.drop_duplicates()

    combined.to_csv(global_path, index=False)
    print(f"[PGX] Wrote global drug-gene mapping table to {global_path} ({len(combined)} rows).")
    return global_path


def build_global_allele_frequencies() -> Path:
    base_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs"
    global_dir = base_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_path = global_dir / "pgx_allele_frequencies_global.csv"

    freq_files = list(base_dir.rglob("*_allele_frequencies.csv"))
    # Exclude any existing global file from input set
    freq_files = [p for p in freq_files if p.name != global_path.name]

    if not freq_files:
        print("[PGX] No existing *_allele_frequencies.csv artifacts found; skipping global allele build.")
        return global_path

    dfs = []
    for p in freq_files:
        try:
            df = pd.read_csv(p)
            df["source_file"] = str(p.relative_to(base_dir))
            dfs.append(df)
        except Exception as exc:
            print(f"[PGX] Warning: could not read {p}: {exc}")

    if not dfs:
        print("[PGX] No valid allele-frequency CSVs could be read; skipping global allele build.")
        return global_path

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate by gene + variant_id where possible
    subset_cols = [c for c in ["gene", "variant_id"] if c in combined.columns]
    if subset_cols:
        combined = combined.drop_duplicates(subset=subset_cols)
    else:
        combined = combined.drop_duplicates()

    combined.to_csv(global_path, index=False)
    print(f"[PGX] Wrote global allele-frequency table to {global_path} ({len(combined)} rows).")
    return global_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/refresh global PGx lookup tables from existing per-cohort artifacts."
    )
    _ = parser.parse_args()

    print("[PGX] Building global PGx caches from existing outputs...")
    mapping_path = build_global_drug_gene_mappings()
    freq_path = build_global_allele_frequencies()
    print("[PGX] Global PGx caches ready:")
    print(f"  - Drug-gene mappings: {mapping_path}")
    print(f"  - Allele frequencies: {freq_path}")


if __name__ == "__main__":
    main()

