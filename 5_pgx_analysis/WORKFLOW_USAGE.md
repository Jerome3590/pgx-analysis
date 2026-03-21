# PGx Step 5 — production workflow (drug → CPIC mapping)

This note complements **`README.md`**. Step 5 ships **two patient-level features**: `pgx_num_drugs` and `pgx_num_cpic_drugs`, using a **global APCD drug name → CPIC drug name** table.

## Production flow

1. **CPIC reference** — Official gene–drug pairs: `5_pgx_analysis/cpic/cpic_gene-drug_pairs.xlsx` (see CPIC site). `update_cpic_drug_list.py` builds `data/cpic_drug_list.json` for matchers.
2. **Global mapping** — `build_global_drug_cpic_mapping.py` reads cohort/aggregated feature-importance CSVs, fuzzy-matches drug tokens to CPIC names, writes `outputs/global/drug_cpic_mapping_global.csv`, and uploads to `s3://pgxdatalake/gold/pgx_features/global/drug_cpic_mapping_global.csv`.
3. **Features** — `create_pgx_features_patient_level.py` loads that mapping (local or S3), aggregates `model_events.parquet`, then `add_pgx_features_to_model_data.py` merges PGx columns for Step 6.

`map_drugs_to_genes.py` supports exploration and building the global table; it is not required for a standard count-only run once the global CSV exists.

## Lessons learned (operations)

- **Without** `drug_cpic_mapping_global.csv` (and failed S3 fetch), **`pgx_num_cpic_drugs` stays zero** for everyone. Bake the file into deploys or ensure the gold S3 object exists before EC2 runs.
- **Rebuild mapping** when feature-importance drug vocab shifts materially (`build_global_drug_cpic_mapping.py`, then re-upload).
- **Alleles / genotypes** are out of scope here; the **PGx card** (dashboard/Lambda) consumes CPIC tables for interactive genotype-guided content.
