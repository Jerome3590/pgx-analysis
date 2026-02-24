## Step 5: Pharmacogenomic (PGx) Analysis

This folder documents the PGx step that adds **CPIC drug counts only** to model data. No alleles or genotype data are used in this pipeline.

### Code Location

- **Primary module**: `5_pgx_analysis/`
  - **Main documentation**: `5_pgx_analysis/README.md`
  - Core scripts: `run_analysis.py`, `create_pgx_features_patient_level.py`, `add_pgx_features_to_model_data.py`, `build_global_drug_cpic_mapping.py`, `update_cpic_drug_list.py`

### Role in the Workflow

Step 5 runs **after** model data (Step 4) and adds two patient-level features:

- **CPIC drug identification:** Uses the global drug-to-CPIC mapping to identify which drugs have CPIC pharmacogenomic guidelines.
- **Patient-level counts:** `pgx_num_drugs` (total distinct drugs per patient) and `pgx_num_cpic_drugs` (count of CPIC drugs per patient).

**Alleles:** Not used in this step. Alleles are used in the **PGx card** (risk dashboard) when patients submit SNP data with alleles encoded.

### Inputs and Outputs

- **Inputs** (per cohort/age_band): Model events from Step 4; global drug-to-CPIC mapping (`5_pgx_analysis/outputs/global/drug_cpic_mapping_global.csv`).
- **Outputs:** `pgx_features_{cohort}_{age_band}.csv`, `pgx_added_features_{cohort}_{age_band}.csv` (merged into model data for Step 6).

### Related Documentation

- `5_pgx_analysis/README.md` – Full PGx workflow, data sources, and troubleshooting.
- `docs/README_analysis_workflow.md` – How PGx fits into the overall pipeline.
