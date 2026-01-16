## Step 5: Pharmacogenomic (PGx) Analysis

This folder documents the PGx feature-engineering step that maps drugs to pharmacogenes and integrates allele-frequency information into patient-level features.

### Code Location

- **Primary module**: `5_pgx_analysis/`
  - Orchestrator and usage: `5_pgx_analysis/WORKFLOW_USAGE.md` (if present)
  - Main documentation: `5_pgx_analysis/README.md`
  - Core scripts:
    - `map_drugs_to_genes.py`
    - `add_allele_frequencies.py`
    - `create_pgx_features_patient_level.py`
    - `add_pgx_features_to_model_data.py`

### Role in the Workflow

Step 5 runs **after** model data extraction and DTW protocol filtering (Steps 4a–4b) and focuses on pharmacogenomic enrichment:

- Uses important drugs from feature importance / FP-Growth to identify **clinically relevant drug–gene pairs** (CPIC-based).  
- Integrates **population allele frequencies** for key variants in those genes.  
- Produces **patient-level PGx features** (e.g., presence of high‑risk drug–gene combinations) and merges them into the model-ready feature tables.

### Inputs and Outputs

- **Inputs** (per `(cohort, age_band)`):
  - Model events data from Step 4a/4b (`model_events_no_protocols.parquet` preferred)
  - CPIC reference files under `5_pgx_analysis/data/` and `5_pgx_analysis/cpic/`.

- **Outputs**:
  - Intermediate PGx feature tables (e.g., `pgx_features_{cohort}_{age_band}.csv`).  
  - Final PGx-enriched features joined into the model dataset (e.g., `pgx_added_features_{cohort}_{age_band}.csv`).  
  - Optional visualizations of drug–gene networks and allele-frequency distributions.

### Related Documentation

- `5_pgx_analysis/README.md` – Detailed PGx workflow and methodology.  
- `5_pgx_analysis/FEATURES_EXPLANATION.md` – Explanation of PGx feature columns (if present).  
- `5_pgx_analysis/ALLELE_FREQUENCY_METHODOLOGY.md` – Allele frequency methodology (if present).  
- `docs/README_analysis_workflow.md` – How PGx integrates with the overall feature-engineering and modeling pipeline.  

