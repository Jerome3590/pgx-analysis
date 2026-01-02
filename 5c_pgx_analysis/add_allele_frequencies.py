#!/usr/bin/env python3
"""
Add population allele frequencies to drug-gene mappings.

This script retrieves population-level allele frequencies for pharmacogenomic
variants from reference databases (1000 Genomes, gnomAD, etc.).
"""

import sys
import pandas as pd
from pathlib import Path
import logging
import requests
from typing import Dict, Optional
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CPIC API base URL
CPIC_API_BASE = "https://api.cpicpgx.org"


def map_race_to_ancestry_group(race_value: str) -> Optional[str]:
    """
    Map patient-reported race/ethnicity to population ancestry group.

    Note: This is an approximate mapping. Patient-reported race/ethnicity is NOT
    equivalent to genetic ancestry. For clinical use, actual genotyping is preferred.

    Parameters:
    -----------
    race_value : str
        Patient-reported race/ethnicity value

    Returns:
    --------
    Optional[str]
        Population ancestry group code (afr, amr, eas, eur, sas) or None
    """
    if pd.isna(race_value) or not race_value:
        return None

    race_lower = str(race_value).lower()

    # Mapping based on common race/ethnicity categories
    # This is approximate and should be used with caution
    if any(term in race_lower for term in ['african', 'black', 'afro']):
        return 'afr'
    elif any(term in race_lower for term in ['hispanic', 'latino', 'latinx', 'mexican', 'puerto rican']):
        return 'amr'
    elif any(term in race_lower for term in ['asian', 'chinese', 'japanese', 'korean', 'vietnamese', 'east asian']):
        return 'eas'
    elif any(term in race_lower for term in ['white', 'caucasian', 'european']):
        return 'eur'
    elif any(term in race_lower for term in ['south asian', 'indian', 'pakistani', 'bangladeshi']):
        return 'sas'

    return None


def fetch_cpic_allele_frequencies(gene_symbol: str) -> Optional[Dict]:
    """
    Fetch allele frequency data from CPIC API for a given gene.

    Parameters:
    -----------
    gene_symbol : str
        Gene symbol (e.g., CYP2D6, CYP2C19)

    Returns:
    --------
    Optional[Dict]
        Allele frequency data from CPIC API
    """
    try:
        # CPIC API endpoint: /allele
        url = f"{CPIC_API_BASE}/allele"

        # Try with genesymbol parameter
        params = {"genesymbol": gene_symbol}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    # Aggregate frequency data from multiple alleles
                    return aggregate_allele_frequencies(data, gene_symbol)
                elif isinstance(data, dict):
                    return data
            except ValueError:
                pass

        # Try getting all alleles and filtering
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            try:
                all_alleles = response.json()
                if isinstance(all_alleles, list):
                    # Filter by gene symbol
                    gene_upper = gene_symbol.upper()
                    filtered = [
                        a for a in all_alleles
                        if isinstance(a, dict) and (
                            gene_upper in str(a.get("genesymbol", "")).upper() or
                            gene_upper in str(a.get("gene", {})).upper()
                        )
                    ]
                    if filtered:
                        return aggregate_allele_frequencies(filtered, gene_symbol)
            except ValueError:
                pass

        return None

    except requests.exceptions.RequestException as e:
        logger.debug(f"Error fetching CPIC allele frequencies for {gene_symbol}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching CPIC allele frequencies for {gene_symbol}: {e}")
        return None


def aggregate_allele_frequencies(alleles: list, gene_symbol: str) -> Dict:
    """
    Aggregate allele frequency data from multiple alleles.

    Parameters:
    -----------
    alleles : list
        List of allele records from CPIC API
    gene_symbol : str
        Gene symbol

    Returns:
    --------
    Dict
        Aggregated frequency data
    """
    # Extract frequency information from alleles
    # Note: CPIC API structure may vary, adjust as needed
    aggregated = {
        "gene": gene_symbol,
        "allele_count": len(alleles),
        "variants": []
    }

    for allele in alleles:
        if isinstance(allele, dict):
            variant_info = {
                "allele_name": allele.get("name", ""),
                "allele_id": allele.get("id", ""),
                "function": allele.get("function", ""),
                "frequency": allele.get("frequency", {})
            }
            aggregated["variants"].append(variant_info)

    return aggregated


def add_allele_frequencies(drug_gene_mappings: pd.DataFrame, output_path: Optional[Path] = None,
                           rate_limit_delay: float = 0.5, use_patient_demographics: bool = False,
                           patient_demographics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add population allele frequencies to drug-gene mappings.

    **Important Notes on Frequency Assignment:**

    1. **Population-Level Frequencies**: Allele frequencies vary by population ancestry group
       (AFR, AMR, EAS, EUR, SAS). This is well-documented in pharmacogenomics.

    2. **Patient Demographics**: If `use_patient_demographics=True` and patient demographics
       are provided, the function will attempt to map patient-reported race/ethnicity to
       population ancestry groups and select the appropriate frequency.

    3. **Limitations**: Patient-reported race/ethnicity is NOT equivalent to genetic ancestry.
       This mapping is approximate and should be used with caution. For clinical use,
       actual genotyping is preferred.

    4. **Default Behavior**: By default, uses population-weighted average (average of all
       available population frequencies: AFR, AMR, EAS, EUR, SAS). This provides a more
       diverse representation than global frequency alone. Falls back to global frequency
       if population frequencies are unavailable.

    Parameters:
    -----------
    drug_gene_mappings : pd.DataFrame
        DataFrame with drug-gene mappings (must have 'gene' column).
        If using patient demographics, should have a patient identifier column.
    patient_demographics : pd.DataFrame, optional
        DataFrame with patient demographics. Must include:
        - Patient identifier column (to join with drug_gene_mappings)
        - Race/ethnicity column (column name should be specified or detected)
    output_path : Path, optional
        Path to save the enriched mappings CSV file
    rate_limit_delay : float
        Delay between API calls (seconds) to respect rate limits
    use_patient_demographics : bool
        If True and patient_demographics provided, will attempt to use
        patient race/ethnicity to select population-specific frequencies.
        Default: False (uses global frequency)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added allele frequency columns:
        - `allele_frequency_global`: Global average (default)
        - `allele_frequency_afr/amr/eas/eur/sas`: Population-specific frequencies
        - `allele_frequency_assigned`: The frequency actually used (based on method)
        - `frequency_assignment_method`: How the frequency was assigned
    """
    logger.info(f"Adding population allele frequencies for {len(drug_gene_mappings)} drug-gene pairs using CPIC API...")

    if use_patient_demographics and patient_demographics is not None:
        logger.warning("Using patient demographics for frequency assignment.")
        logger.warning("Note: Patient-reported race/ethnicity is approximate for genetic ancestry.")
        logger.warning("For clinical use, actual genotyping is preferred.")
    else:
        logger.info("Using population-weighted average as default (average of all available population frequencies).")
        logger.info("This provides a more diverse representation than global frequency alone.")

    # Get unique genes
    unique_genes = drug_gene_mappings['gene'].dropna().unique()
    logger.info(f"Fetching allele frequencies for {len(unique_genes)} unique genes...")

    # Create enriched DataFrame
    enriched_mappings = drug_gene_mappings.copy()

    # Add allele frequency columns
    enriched_mappings['variant_id'] = None
    enriched_mappings['allele_name'] = None
    enriched_mappings['allele_function'] = None
    enriched_mappings['allele_frequency_global'] = None  # Global average frequency
    enriched_mappings['allele_frequency_afr'] = None  # African/African American ancestry
    enriched_mappings['allele_frequency_amr'] = None  # Latino/Admixed American ancestry
    enriched_mappings['allele_frequency_eas'] = None  # East Asian ancestry
    enriched_mappings['allele_frequency_eur'] = None  # European ancestry
    enriched_mappings['allele_frequency_sas'] = None  # South Asian ancestry
    enriched_mappings['allele_frequency_assigned'] = None  # The frequency actually used
    enriched_mappings['frequency_source'] = None
    enriched_mappings['frequency_assignment_method'] = None  # How frequency was assigned

    # Prepare patient demographics mapping if provided
    patient_ancestry_map = {}
    if use_patient_demographics and patient_demographics is not None:
        # Try to detect race/ethnicity column
        race_cols = [col for col in patient_demographics.columns
                    if 'race' in col.lower() or 'ethnicity' in col.lower() or 'ethnic' in col.lower()]
        patient_id_cols = [col for col in patient_demographics.columns
                          if 'person' in col.lower() or 'patient' in col.lower() or 'id' in col.lower()]

        if race_cols and patient_id_cols:
            race_col = race_cols[0]
            patient_id_col = patient_id_cols[0]

            for _, row in patient_demographics.iterrows():
                patient_id = row[patient_id_col]
                race_value = row[race_col]
                ancestry_group = map_race_to_ancestry_group(race_value)
                if ancestry_group:
                    patient_ancestry_map[patient_id] = ancestry_group

            logger.info(f"Mapped {len(patient_ancestry_map)} patients to ancestry groups")
        else:
            logger.warning("Could not detect race/ethnicity or patient ID columns in demographics data")
            logger.warning("Falling back to global frequency")
            use_patient_demographics = False

    # Cache for gene frequencies to avoid duplicate API calls
    gene_frequency_cache = {}

    # Fill in frequencies for each gene
    for idx, row in enriched_mappings.iterrows():
        gene = row['gene']

        if pd.isna(gene) or not gene:
            continue

        # Check cache first
        if gene not in gene_frequency_cache:
            logger.debug(f"Fetching allele frequencies for gene: {gene}")
            freq_data = fetch_cpic_allele_frequencies(gene)
            gene_frequency_cache[gene] = freq_data
            time.sleep(rate_limit_delay)  # Rate limiting
        else:
            freq_data = gene_frequency_cache[gene]

        # Extract frequency information
        if freq_data:
            # Get the most common/important variant (first in list or most frequent)
            variants = freq_data.get('variants', [])
            if variants:
                # Use the first variant or find the most common one
                variant = variants[0]

                enriched_mappings.at[idx, 'variant_id'] = variant.get('allele_id', '')
                enriched_mappings.at[idx, 'allele_name'] = variant.get('allele_name', '')
                enriched_mappings.at[idx, 'allele_function'] = variant.get('function', '')

                # Extract frequency data
                frequency = variant.get('frequency', {})
                if isinstance(frequency, dict):
                    global_freq = frequency.get('global', None)
                    afr_freq = frequency.get('afr', frequency.get('african', None))
                    amr_freq = frequency.get('amr', frequency.get('latino', None))
                    eas_freq = frequency.get('eas', frequency.get('east_asian', None))
                    eur_freq = frequency.get('eur', frequency.get('european', None))
                    sas_freq = frequency.get('sas', frequency.get('south_asian', None))

                    enriched_mappings.at[idx, 'allele_frequency_global'] = global_freq
                    enriched_mappings.at[idx, 'allele_frequency_afr'] = afr_freq
                    enriched_mappings.at[idx, 'allele_frequency_amr'] = amr_freq
                    enriched_mappings.at[idx, 'allele_frequency_eas'] = eas_freq
                    enriched_mappings.at[idx, 'allele_frequency_eur'] = eur_freq
                    enriched_mappings.at[idx, 'allele_frequency_sas'] = sas_freq

                    # Assign frequency based on method
                    # Default: Use population-weighted average (average of all available population frequencies)
                    # This provides a more diverse representation than global alone
                    population_freqs = [f for f in [afr_freq, amr_freq, eas_freq, eur_freq, sas_freq] if f is not None]
                    if population_freqs:
                        # Use average of all available population frequencies
                        assigned_freq = sum(population_freqs) / len(population_freqs)
                        assignment_method = f'population_weighted_average_{len(population_freqs)}_populations'
                    else:
                        # Fallback to global if no population frequencies available
                        assigned_freq = global_freq
                        assignment_method = 'global_average_fallback'

                    # If using patient demographics, try to get patient-specific frequency
                    if use_patient_demographics:
                        # Try to find patient ID in the row
                        patient_id = None
                        for col in enriched_mappings.columns:
                            if 'person' in col.lower() or 'patient' in col.lower() or col == 'case_id':
                                patient_id = row.get(col)
                                break

                        if patient_id and patient_id in patient_ancestry_map:
                            ancestry_group = patient_ancestry_map[patient_id]
                            ancestry_freq_map = {
                                'afr': afr_freq,
                                'amr': amr_freq,
                                'eas': eas_freq,
                                'eur': eur_freq,
                                'sas': sas_freq
                            }

                            if ancestry_group in ancestry_freq_map and ancestry_freq_map[ancestry_group] is not None:
                                assigned_freq = ancestry_freq_map[ancestry_group]
                                assignment_method = f'patient_demographics_mapped_to_{ancestry_group}'
                            else:
                                # Fallback to global if population frequency not available
                                assignment_method = f'patient_demographics_mapped_to_{ancestry_group}_but_freq_unavailable_using_global'

                    enriched_mappings.at[idx, 'allele_frequency_assigned'] = assigned_freq
                    enriched_mappings.at[idx, 'frequency_assignment_method'] = assignment_method

                enriched_mappings.at[idx, 'frequency_source'] = 'CPIC_API'
            else:
                # No variants found, mark as missing
                enriched_mappings.at[idx, 'frequency_source'] = 'CPIC_API_no_data'
                enriched_mappings.at[idx, 'frequency_assignment_method'] = 'no_data_available'
        else:
            # API call failed or no data
            enriched_mappings.at[idx, 'frequency_source'] = 'CPIC_API_error'
            enriched_mappings.at[idx, 'frequency_assignment_method'] = 'api_error'

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_mappings.to_csv(output_path, index=False)
        logger.info(f"Saved allele frequencies to {output_path}")

    return enriched_mappings


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Add allele frequencies to drug-gene mappings")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--mappings", help="Path to drug-gene mappings CSV (optional)")
    parser.add_argument("--output", help="Output CSV path (optional)")

    args = parser.parse_args()

    # Load mappings
    if args.mappings:
        mappings_path = Path(args.mappings)
    else:
        mappings_path = (
            PROJECT_ROOT
            / "5c_pgx_analysis"
            / "outputs"
            / args.cohort
            / args.age_band.replace("-", "_")
            / f"{args.cohort}_{args.age_band.replace('-', '_')}_drug_gene_mappings.csv"
        )

    if not mappings_path.exists():
        logger.error(f"Mappings file not found at {mappings_path}")
        logger.error("Please run map_drugs_to_genes.py first or provide --mappings argument")
        return

    drug_gene_mappings = pd.read_csv(mappings_path)

    # Set output path
    if not args.output:
        args.output = (
            PROJECT_ROOT
            / "5c_pgx_analysis"
            / "outputs"
            / args.cohort
            / args.age_band.replace("-", "_")
            / f"{args.cohort}_{args.age_band.replace('-', '_')}_allele_frequencies.csv"
        )

    # Add allele frequencies
    enriched = add_allele_frequencies(
        drug_gene_mappings=drug_gene_mappings,
        output_path=args.output
    )

    print(f"\nAdded allele frequencies for {len(enriched)} variants")
    print(f"Genes with frequency data: {enriched['gene'].nunique()}")
    print("\nSample frequencies:")
    print(enriched[['drug_name', 'gene', 'variant_id', 'allele_frequency_global']].head(10))


if __name__ == "__main__":
    main()

