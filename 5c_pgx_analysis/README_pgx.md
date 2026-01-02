## PGx (Pharmacogenomics) Feature Engineering

This directory contains scripts to build pharmacogenomics-related features by linking cohort drug exposures to PGx-relevant genes, and aggregating those relationships to the patient level.

### Key Scripts

- `map_drugs_to_genes.py`  
  - Extracts distinct `drug_name` values from `4a_model_data` for a given `(cohort, age_band)`.  
  - Loads CPIC drug–gene relationship pairs (e.g., from `7_pgx_analysis/data/cpicPairs.csv`).  
  - Performs exact and (optionally) fuzzy matching to map cohort drugs to CPIC drugs and genes.  
  - Writes:
    - `7_pgx_analysis/outputs/drug_gene_mapping_{cohort}_{age_band}.csv` with columns such as:
      - `drug_name`, `cpic_drug`, `gene`, `match_score`, `source`.

- `create_pgx_features_patient_level.py`  
  - Consumes `drug_gene_mapping_{cohort}_{age_band}.csv` and `4a_model_data/.../model_events.parquet`.  
  - For each `mi_person_key`, computes PGx exposure features, for example:
    - `pgx_exposed_any`: indicator if patient was ever exposed to a mapped PGx-relevant drug.
    - `pgx_n_exposed_drugs`: count of distinct mapped drugs.
    - `pgx_n_genes_hit`: count of distinct genes touched by those exposures.  
  - Outputs:
    - `7_pgx_analysis/outputs/feature_engineering/pgx_features_{cohort}_{age_band}.csv`.

- `add_allele_frequencies.py`  
  - Utility to enrich variant-level PGx feature tables with population allele frequencies from external reference data.  
  - Inputs:
    - A variant-level PGx features CSV (e.g., from a VCF annotation step).
    - An allele frequency reference CSV (e.g., gnomAD-derived frequencies).  
  - Outputs:
    - A combined CSV with allele frequency columns appended; used upstream of patient-level PGx feature creation when variant data are available.

- `add_pgx_features_to_model_data.py`  
  - Final aggregation step for PGx features.  
  - Reads `pgx_features_{cohort}_{age_band}.csv` and writes:
    - `pgx_added_features_{cohort}_{age_band}.csv` under `7_pgx_analysis/outputs/feature_engineering/`.  
  - Also uploads to:
    - `s3://pgxdatalake/gold/feature_engineering/7_pgx/{cohort}/{age_band}/pgx_added_features_{cohort}_{age_band}.csv` (if AWS CLI is available).  
  - Output is joinable to the model matrix via `mi_person_key`.

- `create_pgx_features.py`  
  - Orchestration wrapper that runs:
    1. `map_drugs_to_genes.py` for the specified `(cohort, age_band)`.  
    2. `create_pgx_features_patient_level.py` to build patient-level PGx features.  
  - Produces `pgx_features_{cohort}_{age_band}.csv` in `7_pgx_analysis/outputs/feature_engineering/`.

### Inputs

- `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`  
  - Required by both `map_drugs_to_genes.py` and `create_pgx_features_patient_level.py`.
- `7_pgx_analysis/data/cpicPairs.csv` (or similar CPIC relationship file)  
  - Contains curated `drug`–`gene` pairs for mapping exposures.
- Optional variant-level PGx tables and allele frequency references for use with `add_allele_frequencies.py`.

### Outputs

All patient-level PGx features live under:

- `7_pgx_analysis/outputs/feature_engineering/`  
  - `pgx_features_{cohort}_{age_band}.csv`  
  - `pgx_added_features_{cohort}_{age_band}.csv`

These files include `mi_person_key` and are intended to be merged with other feature blocks (FP-Growth, BupaR, DTW) just prior to model training.

### Typical Workflow

From the project root:

```bash
python 7_pgx_analysis/create_pgx_features.py --cohort-name opioid_ed --age-band 0-12
python 7_pgx_analysis/add_pgx_features_to_model_data.py --cohort-name opioid_ed --age-band 0-12
```

This will:

1. Map cohort drug exposures to PGx genes using CPIC data.  
2. Create patient-level PGx exposure features.  
3. Produce `pgx_added_features_{cohort}_{age_band}.csv` ready to be joined into the final modeling dataset.

