# pgx-analysis

## Stack
Python 3.11, pandas, DuckDB, PyArrow, scikit-learn, XGBoost, CatBoost, SHAP, Plotly, AWS S3/Lambda/CloudFront, R/BupaR, Quarto/manuscript tooling.

## Purpose
End-to-end opioid and polypharmacy ED risk analysis pipeline. The repo builds APCD cohorts, screens features, adds PGx drug-count features, trains final models, runs SHAP/FFA/DTW/FP-Growth analyses, and builds the serverless PGx risk dashboard.

## Key Locations
| Path | Description |
|------|-------------|
| `py_helpers/` | Shared Python helpers for paths, constants, S3, dashboard visuals, SHAP/FFA utilities. |
| `utility_scripts/` | Project maintenance, audit, cleanup, and visualization helper scripts. |
| `2_create_cohort/` | Cohort creation and QA. |
| `4_model_data/` | Model-ready dataset construction. |
| `5_pgx_analysis/` | PGx feature engineering and CPIC drug-count workflow. |
| `6_final_model/` | Final model training and evaluation. |
| `7_shap_analysis/` | SHAP analysis. |
| `8_ffa_analysis/` | Formal Feature Attribution analysis. |
| `9_dashboard_visuals/` | Dashboard visual artifact generation. |
| `10_risk_dashboard/` | Frontend, backend Lambda, deployment assets. |
| `manuscript/` | CTS manuscript source, submission workflows, and validation scripts. |

## Corpus-First Rules
- Search local helpers and `utility_scripts/` before adding new utilities.
- Prefer existing S3/path/config conventions in `py_helpers/` and pipeline step directories.
- Keep project-specific notebook exclusions and generated artifact ignores intact; avoid broad rewrites of workflow notebooks.

## Project Metadata
- Project slug: `pgx-analysis`
- Notebook metadata bucket: `s3://mushin-solutions-project-metadata/notebooks/`
- Notebook output pointers resolve under: `s3://mushin-solutions-project-metadata/notebooks/pgx-analysis/`

## Notebook Outputs
The project has many large notebooks. Use project utility commands for notebook output sync when S3 output pointers are available:

```bash
python cursor_setup.py status
python cursor_setup.py push-outputs 5_pgx_analysis/pgx_cohort_runner.ipynb
python cursor_setup.py fetch-outputs 5_pgx_analysis/pgx_cohort_runner.ipynb
```

## Known Patterns
- Pipeline scripts generally accept cohort and age-band parameters and checkpoint large outputs through S3/local NVMe paths.
- PGx analysis currently uses CPIC drug-count features, not patient genotype/allele data, in the main modeling pipeline.
- Manuscript edits under `manuscript/cts/**` should follow the CTS rule in `manuscript/.cursor/rules/cts-submission-limits.mdc`.
