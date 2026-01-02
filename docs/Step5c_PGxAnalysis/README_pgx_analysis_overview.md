## Step 5c: Pharmacogenomic (PGx) Analysis

This folder documents the PGx feature-engineering step that maps drugs to pharmacogenes and integrates allele-frequency information into patient-level features.

### Code Location

- **Primary module**: `5c_pgx_analysis/`
  - Orchestrator and usage: `5c_pgx_analysis/WORKFLOW_USAGE.md`
  - Main documentation: `5c_pgx_analysis/README_pgx.md`
  - Core scripts:
    - `map_drugs_to_genes.py`
    - `add_allele_frequencies.py`
    - `create_pgx_features_patient_level.py`
    - `add_pgx_features_to_model_data.py`

### Role in the Workflow

Step 5c runs **after** core event-level features have been built (Steps 4–5b) and focuses on pharmacogenomic enrichment:

- Uses important drugs from feature importance / FP-Growth to identify **clinically relevant drug–gene pairs** (CPIC-based).  
- Integrates **population allele frequencies** for key variants in those genes.  
- Produces **patient-level PGx features** (e.g., presence of high‑risk drug–gene combinations) and merges them into the model-ready feature tables.

### Inputs and Outputs

- **Inputs** (per `(cohort, age_band)`):
  - Drug features and/or event-level data from earlier steps.
  - CPIC reference files under `5c_pgx_analysis/data/` and `5c_pgx_analysis/cpic/`.

- **Outputs**:
  - Intermediate PGx feature tables (e.g., `pgx_features_{cohort}_{age_band}.csv`).  
  - Final PGx-enriched features joined into the model dataset (e.g., `pgx_added_features_{cohort}_{age_band}.csv`).  
  - Optional visualizations of drug–gene networks and allele-frequency distributions.

### Related Documentation

- `5c_pgx_analysis/README_pgx.md` – Detailed PGx workflow and methodology.  
- `5c_pgx_analysis/FEATURES_EXPLANATION.md` – Explanation of PGx feature columns.  
- `5c_pgx_analysis/ALLELE_FREQUENCY_METHODOLOGY.md` – Allele frequency methodology.  
- `docs/README_analysis_workflow.md` – How PGx integrates with the overall feature-engineering and modeling pipeline.  

