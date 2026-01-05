# Step 7: Pharmacogenomics (PGx) Analysis

**Purpose:** Map drugs to known pharmacogenomic gene relationships and incorporate population allele frequencies to enhance feature understanding and model interpretability.

**Status:** ✅ Active Development

**Primary Data Source:** The workflow uses the **Official CPIC Excel File** as the primary source:
- **Official CPIC Excel File**: `cpic/cpic_gene-drug_pairs.xlsx` - **PRIMARY SOURCE**
  - Official current gene-drug pairs file from CPIC website (573 pairs, 300 drugs, 121 genes)
  - **Download from:** https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
  - **Location:** `7_pgx_analysis/cpic/cpic_gene-drug_pairs.xlsx`
  - Updated regularly by CPIC - download the latest version periodically

**Additional Data Sources:**
- **PubMed Search**: Integrated to find additional drug-gene relationships (requires `biopython`)
- **Fuzzy Matching**: Uses `rapidfuzz` to match drug names to CPIC database
- **CPIC Pairs CSV** (`data/cpicPairs.csv`): Fallback source (261 drugs, 416 pairs from PGx-Patient-Card repo)
- **Bioconductor** (optional): Alternative data source for allele frequencies and drug-gene interactions
  - See `fetch_bioconductor_pgx_data.R` for R-based data fetching
  - Packages: `GenomicScores` (MAF data), `CTDquerier` (drug-gene interactions), `VariantFiltering` (variant filtering)
- **CPIC API** (`https://api.cpicpgx.org/`): Attempted but currently unavailable or has changed endpoints (returns 404)

**For production use:**
1. **Download the official CPIC Excel file** from the CPIC website and place it in `7_pgx_analysis/cpic/`
2. **Run `update_cpic_drug_list.py`** to process the Excel file and create the drug list JSON
3. Ensure `biopython` is installed for PubMed search functionality (`pip install biopython`)
4. Ensure `openpyxl` is installed for Excel reading (`pip install openpyxl`)
5. (Optional) Install Bioconductor packages for alternative allele frequency data: `Rscript -e "BiocManager::install(c('GenomicScores', 'CTDquerier', 'VariantFiltering'))"`

---

## Quick Start

```bash
# Run the PGx analysis pipeline
cd 7_pgx_analysis
jupyter notebook pgx_analysis_pipeline.ipynb

# Or run scripts directly
python map_drugs_to_genes.py --cohort opioid_ed --age_band 0-12 --use-pubmed
python add_allele_frequencies.py --cohort opioid_ed --age_band 0-12
python create_pgx_features_patient_level.py --cohort opioid_ed --age_band 0-12
python add_pgx_features_to_model_data.py --cohort-name opioid_ed --age-band 0-12
```

## Overview

This analysis step enriches drug features with pharmacogenomic information by:

1. **Drug-Gene Mapping**: Identifying genes associated with drug metabolism, transport, and targets
2. **Allele Frequency Integration**: Adding population-level allele frequencies for relevant genetic variants
3. **PGx Feature Enrichment**: Creating enhanced features that combine drug usage patterns with genetic predisposition information

## Methodology

### Data Sources

- **Drug-Gene Relationships**: 
  - **Official CPIC Excel File** (`cpic/cpic_gene-drug_pairs.xlsx`) - **PRIMARY SOURCE** (573 pairs, 300 drugs, 121 genes)
    - **Download from:** https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
    - Official gene-drug pairs file from CPIC website, updated regularly
    - Process using `update_cpic_drug_list.py` to create drug list JSON
  - **PubMed** (via NCBI Entrez API) - Secondary source for additional relationships
  - **CPIC Pairs CSV** (`data/cpicPairs.csv`) - Fallback (261 drugs, 416 pairs from PGx-Patient-Card repo)
  - **Bioconductor** (optional) - Alternative source via R packages:
    - `CTDquerier`: Chemical-gene interactions from Comparative Toxicogenomics Database
    - `pharmacoGx`: Pharmacogenomics datasets (if available)
  - **CPIC API** ([https://api.cpicpgx.org/](https://api.cpicpgx.org/)) - Attempted but currently unavailable
  - PharmGKB (Pharmacogenomics Knowledge Base) - Future integration
  - FDA PGx labeling information - Future integration

- **Allele Frequencies**:
  - **CPIC API** - Primary source (currently unavailable)
  - **Bioconductor** (alternative) - Via R packages:
    - `GenomicScores`: MAF data from 1000 Genomes Project, ExAC, gnomAD
    - `VariantFiltering`: Variant filtering with population frequency data
    - `AnnotationHub`: Access to various annotation databases
  
- **Allele Frequencies**:
  - **CPIC API** - Allele frequency data from CPIC database
  - 1000 Genomes Project population frequencies (fallback)
  - gnomAD (Genome Aggregation Database) allele frequencies (fallback)
  - Population-specific frequencies (when available)

### Process

1. **Drug Identification**: Extract unique drugs from feature importance and FP-Growth results
2. **Gene Mapping**: Map each drug to relevant pharmacogenes (e.g., CYP2D6, CYP2C19, TPMT, DPYD)
3. **Variant Identification**: Identify clinically relevant variants for each gene-drug pair
4. **Frequency Lookup**: Retrieve population allele frequencies for identified variants
5. **Feature Enrichment**: Create enriched features combining drug patterns with genetic risk

## Directory Structure

```
7_pgx_analysis/
├── README.md                          # This file
├── pgx_analysis_pipeline.ipynb        # Main orchestrator notebook
├── map_drugs_to_genes.py              # Drug-gene mapping script (CPIC + PubMed)
├── search_pubmed_drug_gene.py         # PubMed search for drug-gene relationships
├── add_allele_frequencies.py          # Allele frequency integration script
├── create_pgx_features_patient_level.py  # Patient-level PGx feature creation
├── add_pgx_features_to_model_data.py  # Merge PGx features into final dataset
├── update_cpic_drug_list.py           # Update CPIC drug list from pairs file
├── fetch_cpic_drug_list.py            # Fetch CPIC drug list (fallback)
├── fetch_bioconductor_pgx_data.R       # R script to fetch PGx data from Bioconductor
├── FEATURES_EXPLANATION.md            # Detailed feature documentation
├── WORKFLOW_USAGE.md                  # How drug-gene mappings are used
├── ALLELE_FREQUENCY_METHODOLOGY.md     # Allele frequency methodology
└── BIOCONDUCTOR_INTEGRATION.md        # Bioconductor integration guide
├── outputs/                           # Analysis outputs
│   ├── {cohort}/                      # Per-cohort outputs
│   │   └── {age_band}/                # Per age-band outputs
│   │       ├── drug_gene_mappings.csv
│   │       ├── allele_frequencies.csv
│   │       ├── pgx_enriched_features.csv
│   │       └── pgx_summary_stats.csv
│   └── plots/                         # Visualizations
│       └── {cohort}/
│           └── {age_band}/
│               ├── drug_gene_network.png
│               ├── allele_frequency_distribution.png
│               └── pgx_risk_heatmap.png
├── cpic/                              # CPIC reference data
│   └── cpic_gene-drug_pairs.xlsx     # Official CPIC gene-drug pairs (primary source)
└── data/                              # Reference data
    ├── cpicPairs.csv                  # CPIC pairs CSV (fallback)
    ├── cpic.csv                       # CPIC data CSV
    └── cpic_drug_list.json            # Extracted drug list for fuzzy matching
```

## Output Files

**For complete output paths documentation, see:** [`docs/README_analysis_workflow.md`](../docs/README_analysis_workflow.md#output-paths-summary)

The output paths summary has been migrated to the main pipeline documentation. See the link above for:
- Local file paths (prerequisite files, global cache, feature files)
- S3 output paths (primary location, global cache, legacy paths, checkpoints)
- File naming conventions
- Idempotency check information

## Input Dependencies

This step requires completion of:
- **Step 3**: Feature Importance (to identify important drugs)
- **Step 4**: FP-Growth Analysis (to identify frequent drug patterns)

## Workflow Integration

This step follows the standard analysis workflow pattern:

1. **Orchestrator Notebook**: `pgx_analysis_pipeline.ipynb` coordinates the analysis
2. **Supporting Scripts**: Python scripts handle specific tasks
3. **Output Structure**: Results saved to `outputs/{cohort}/{age_band}/` and `outputs/{cohort}/{age_band}/plots/`
4. **Sequential Execution**: Must complete Step 6 (DTW Analysis) before running this step

See `docs/README_output_structure.md` for the complete workflow framework.

## Key Features

- **Comprehensive Drug Coverage**: Maps all drugs identified in previous analysis steps
- **Evidence-Based**: Uses PharmGKB and CPIC evidence levels
- **Population-Aware**: Incorporates population-specific allele frequencies
- **Clinically Relevant**: Focuses on variants with established clinical significance

## Features Added to Model Data

### Overview

PGx features are **NOT directly based on patient race**. Instead, we create **multiple feature variants** (one for each population ancestry group) and let the **model determine** which features are most predictive.

### Feature Creation Process

1. **Patient Drug Exposures**: Extract drugs each patient took from `model_data`
2. **Drug → Gene Mapping**: Map each drug to pharmacogenomic genes (from CPIC/PubMed)
3. **Gene → Allele Frequencies**: Fetch allele frequencies for each gene (all populations)
4. **Patient-Level Aggregation**: Aggregate across all patient's drug-gene pairs

### Features Added (20 features per patient)

#### Global Frequency Features (3):
- `pgx_risk_global_mean`: Average allele frequency across all drug-gene pairs (global)
- `pgx_risk_global_max`: Maximum allele frequency across all drug-gene pairs (global)
- `pgx_risk_global_sum`: Sum of allele frequencies across all drug-gene pairs (global)

#### Population-Specific Features (15):
For each population (AFR, AMR, EAS, EUR, SAS), create 3 features:
- `pgx_risk_{population}_mean`: Average allele frequency for that population
- `pgx_risk_{population}_max`: Maximum allele frequency for that population
- `pgx_risk_{population}_sum`: Sum of allele frequencies for that population

**Example:**
- `pgx_risk_afr_mean`, `pgx_risk_afr_max`, `pgx_risk_afr_sum`
- `pgx_risk_amr_mean`, `pgx_risk_amr_max`, `pgx_risk_amr_sum`
- `pgx_risk_eas_mean`, `pgx_risk_eas_max`, `pgx_risk_eas_sum`
- `pgx_risk_eur_mean`, `pgx_risk_eur_max`, `pgx_risk_eur_sum`
- `pgx_risk_sas_mean`, `pgx_risk_sas_max`, `pgx_risk_sas_sum`

#### Count Features (2):
- `pgx_drugs_with_mappings`: Number of patient's drugs that have PGx mappings
- `pgx_genes_covered`: Number of unique PGx genes involved in patient's drug-gene pairs

### Complete Feature List

```
mi_person_key                    # Patient identifier
pgx_risk_global_mean            # Mean global frequency
pgx_risk_global_max             # Max global frequency
pgx_risk_global_sum             # Sum global frequency
pgx_risk_afr_mean               # Mean AFR frequency
pgx_risk_afr_max                # Max AFR frequency
pgx_risk_afr_sum                # Sum AFR frequency
pgx_risk_amr_mean               # Mean AMR frequency
pgx_risk_amr_max                # Max AMR frequency
pgx_risk_amr_sum                # Sum AMR frequency
pgx_risk_eas_mean               # Mean EAS frequency
pgx_risk_eas_max                # Max EAS frequency
pgx_risk_eas_sum                # Sum EAS frequency
pgx_risk_eur_mean               # Mean EUR frequency
pgx_risk_eur_max                # Max EUR frequency
pgx_risk_eur_sum                # Sum EUR frequency
pgx_risk_sas_mean               # Mean SAS frequency
pgx_risk_sas_max                # Max SAS frequency
pgx_risk_sas_sum                # Sum SAS frequency
pgx_drugs_with_mappings         # Count of drugs with PGx data
pgx_genes_covered               # Count of unique genes
```

### Important: Race is NOT Used for Feature Assignment

**What We Do NOT Do:**
- ❌ We do NOT assign frequencies based on patient-reported race
- ❌ We do NOT create a single "assigned" feature based on patient demographics
- ❌ We do NOT use patient race to select which frequency to use

**What We Do:**
- ✅ Create separate features for ALL populations (global, AFR, AMR, EAS, EUR, SAS)
- ✅ Let the model determine which features are most predictive
- ✅ Store all population frequencies for transparency

### Model Usage

During model training:
1. **Feature Selection**: Model can automatically select which population features improve predictions
2. **Feature Importance**: Shows which frequency variants (if any) are most informative
3. **No Race-Based Assignment**: Model evaluates all variants equally, regardless of patient demographics

### Example Feature Values

**Patient who took AMOXICILLIN + AZITHROMYCIN:**
```
mi_person_key: 12345
pgx_risk_global_mean: 0.18    # Average across 17 drug-gene pairs (global)
pgx_risk_global_max: 0.30     # Highest frequency variant (global)
pgx_risk_global_sum: 3.06     # Sum of all frequencies (global)

pgx_risk_eur_mean: 0.20       # Average across 17 drug-gene pairs (European)
pgx_risk_eur_max: 0.35       # Highest frequency variant (European)
pgx_risk_eur_sum: 3.40       # Sum of all frequencies (European)

pgx_risk_afr_mean: 0.15       # Average across 17 drug-gene pairs (African)
pgx_risk_afr_max: 0.25       # Highest frequency variant (African)
pgx_risk_afr_sum: 2.55       # Sum of all frequencies (African)

... (similar for AMR, EAS, SAS)

pgx_drugs_with_mappings: 2   # AMOXICILLIN, AZITHROMYCIN
pgx_genes_covered: 13        # Unique genes across both drugs
```

### Rationale

**Why Not Assign Based on Race?**
1. **Race ≠ Genetic Ancestry**: Patient-reported race is a social construct, not genetic ancestry
2. **Model-Driven**: Let the model discover which features are predictive
3. **Transparency**: All population frequencies are available for analysis
4. **Avoid Bias**: Don't make assumptions about which frequency to use

**Why Multiple Variants?**
1. **Population Differences**: Allele frequencies DO vary by population (well-documented)
2. **Model Selection**: Model can determine if population-specific features help
3. **Flexibility**: Can evaluate different approaches without assumptions

See `FEATURES_EXPLANATION.md` for detailed documentation.

