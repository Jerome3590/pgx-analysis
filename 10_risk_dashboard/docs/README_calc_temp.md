# PGx Risk Calculator Dashboard – Full Deployment Workflow

This document describes the **full risk calculator dashboard deployment workflow** for PGx, aligned with the PHTS calculator pattern. It starts from cohorts with aggregated feature importances and runs through Lambda/Docker deployment.

## Cohorts

The PGx dashboard uses two risk models (one per cohort):

| Cohort | Age bands | Description |
|--------|-----------|-------------|
| `opioid_ed` | Full set (0-12 through 85-114) | Opioid-related ED visit predictive model |
| `non_opioid_ed` (polypharmacy) | Full set (0-12 through 85-114) | Polypharmacy / adverse drug event model |

## Workflow overview

The workflow runs in this order:

1. **Verify inputs** – Feature importance (Step 3 or 3b) and final models (Step 6) for each cohort/age_band.
2. **Generate metadata** – Extract valid codes from feature importance for dashboard dropdowns (`data_preparation/generate_metadata.py`).
3. **Prepare models** – Package models and feature schemas from `6_final_model/outputs` (`data_preparation/prepare_models.py`).
4. **(Optional) Combine SHAP/FFA** – For the Causal Analysis tab (`data_preparation/combine_shap_ffa_results.py`).
5. **Prepare Lambda directory** – Assemble `lambda_dir` for Docker (`deployment/prepare_lambda_dir.py`).
6. **Verify & deploy** – Verify `lambda_dir`, then build Docker image and deploy (e.g. `deployment/docker_build.sh`).

## Notebook: `pgx_calculator_workflow.ipynb`

The notebook **pgx_calculator_workflow.ipynb** in this directory runs the full workflow interactively:

- **Step 0:** Verify cohorts and aggregated feature importances (Step 3/3b, Step 6).
- **Step 1:** Generate metadata (`generate_metadata.py --all`).
- **Step 2:** Prepare models (`prepare_models.py --all`).
- **Step 3 (optional):** Combine SHAP/FFA for causal tab.
- **Step 4:** Prepare Lambda directory (`prepare_lambda_dir.py`).
- **Step 5:** Verify Lambda directory (`prepare_lambda_dir.py --verify-only`).
- **Step 6:** Build and deploy (run `deployment/docker_build.sh` from shell; see docs).

Run the notebook from the project root or from `10_risk_dashboard` so that paths resolve correctly.

## Reference directories (PHTS)

- **PHTS calculator workflow:** `C:\Projects\phts\graft-loss\cohort_analysis\calculator\calculator_workflow.ipynb`
- **PHTS scripts:** `C:\Projects\phts\scripts` (e.g. `calculator_workflow_interactive.py`, `feature_importance_utils.py`, `model_utils.py`)
- **PHTS calculator module:** `C:\Projects\phts\graft-loss\cohort_analysis\calculator` (e.g. `calculator_features.py`, `prepare_lambda_dir_phts.py`, `risk_dashboard/`)

## Inputs (PGx)

- **Feature importance:**  
  - Step 3b: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`  
  - Step 3a (fallback): `3a_feature_importance/outputs/{cohort}/{cohort}_{age_band}_aggregated_feature_importance.csv` (layout may vary; see `3a_feature_importance/README.md`)
- **Final models:** `6_final_model/outputs/{cohort}/{age_band}/`
- **SHAP/FFA (optional):** Step 7 and Step 8 outputs (see `combine_shap_ffa_results.py`).

## Outputs

- **Metadata:** `10_risk_dashboard/outputs/metadata/`
- **Models:** `10_risk_dashboard/outputs/models/`
- **Lambda package:** `10_risk_dashboard/lambda_dir/` (or as configured in `prepare_lambda_dir.py`)

See **data_preparation/README.md** and **deployment/README.md** for script details and deployment steps.
