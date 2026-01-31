# Step 9: Results & Dashboard Documentation

This folder contains all documentation for the production-ready risk assessment dashboard, visualization system, and deployment pipeline.

## 📋 Documentation Index

### Main Documentation

- **[README_results_dashboard.md](README_results_dashboard.md)** - Complete dashboard system overview and architecture
- **[README_results_value_proposition.md](README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](README_results_deployment.md)** - Complete deployment guide (architecture, steps, security)
- **[README_results_dashboard_tabs.md](README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard_visualizations.md](README_results_dashboard_visualizations.md)** - Advanced visualization system (BupaR, FP-Growth, DTW)
- **[README_results_dashboard_deployment.md](README_results_dashboard_deployment.md)** - Deployment guide (incremental builds, Docker, Lambda)

### Prediction & Usage

- **[README_results_prediction.md](README_results_prediction.md)** - Detailed prediction workflow and technical details
- **[README_results_quickstart.md](README_results_quickstart.md)** - Quick start guide for making predictions

### Features

- **[README_results_pgx_card.md](README_results_pgx_card.md)** - PGx Patient Card feature documentation
- **[README_results_ensemble.md](README_results_ensemble.md)** - Ensemble model approach (CatBoost + XGBoost + XGBoost RF)
- **[README_results_model_weights.md](README_results_model_weights.md)** - Performance-based model weighting
- **[README_combined_ffa_shap_causal_analysis.md](README_combined_ffa_shap_causal_analysis.md)** - Combined FFA, SHAP, and causal analysis guide

### Deployment Guides

- **[README_results_deployment_ecr.md](README_results_deployment_ecr.md)** - Lambda ECR container deployment guide
- **[README_results_deployment_cpic.md](README_results_deployment_cpic.md)** - CPIC data deployment guide

### Reference

- **[README_results_storage.md](README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](README_results_age_bands.md)** - Supported age bands and mappings

## 🚀 Quick Start

1. **Overview**: Start with [README_results_dashboard.md](README_results_dashboard.md)
2. **Dashboard Tabs**: See [README_results_dashboard_tabs.md](README_results_dashboard_tabs.md) for tab organization and API endpoints
3. **Visualizations**: See [README_results_dashboard_visualizations.md](README_results_dashboard_visualizations.md) for BupaR/FP-Growth/DTW
4. **Deployment**: Follow [README_results_deployment.md](README_results_deployment.md) or [README_results_dashboard_deployment.md](README_results_dashboard_deployment.md)
5. **Usage**: See [README_results_quickstart.md](README_results_quickstart.md)

## 📚 Code Location

- **Primary module**: `9_risk_dashboard/`
  - Frontend: `index.html` - Tab-based dashboard interface
  - Backend: `lambda_function.py` - AWS Lambda API handler
  - Scripts:
    - `prepare_models.py` - Package models for deployment
    - `generate_metadata.py` - Extract valid codes for dropdowns
    - `prepare_cpic_data.py` - Prepare CPIC data for PGx cards
  - Deployment: `docker_build.sh` - Docker container build script

- **Workflow integration**: Run via `1_cohort_workflow.ipynb`, `2_feature_importance.ipynb`, `3_pgx_calculator_workflow.ipynb`; legacy scripts in `archived/utility_scripts/run_cohort_workflow.sh`
  - Step 9 runs automatically for each cohort/age_band
  - Prepares dashboard artifacts incrementally

## 🔑 Key Concepts

### Tab Organization

The dashboard is organized into four main tabs:

1. **Tab 1: Input & Selection** - User input form (age, drugs, ICDs, CPTs)
2. **Tab 2: Risk Score & Causal Analysis** - Risk score display, model breakdown, causal effects
3. **Tab 3: Risk Analysis Visualizations** - BupaR, FP-Growth, and DTW visualizations
4. **Tab 4: PGx Patient Card** - Pharmacogenomic card generator

### Visualization System

- **BupaR**: Process mining visualizations showing patient pathways and sequences
- **FP-Growth**: Frequent pattern analysis showing association rules and itemsets
- **DTW**: Trajectory clustering showing patient similarity and archetypes
- **Causal Analysis**: Impact of removing individual codes on risk score

### Incremental Deployment

- Dashboard can be built/deployed with partial data
- Missing cohorts are handled gracefully
- Visualizations filter based on user-selected codes
- Works with whatever models are available

## 🔄 Role in the Workflow

Step 9 runs **after** all model training and analysis steps (Steps 3-8) and provides:

- **Model Packaging**: Prepares trained models for production deployment
- **Metadata Generation**: Extracts valid codes from feature importance for dropdowns
- **Visualization Integration**: Combines BupaR, FP-Growth, and DTW outputs for dashboard display
- **Causal Analysis**: Integrates FFA causal effects for "what-if" analysis
- **Production Deployment**: Docker containerization and AWS Lambda deployment

## 📊 Inputs and Outputs

### Inputs (per `(cohort, age_band)`)

- **Models**: Trained CatBoost, XGBoost, and XGBoost RF models from Step 6
- **Feature Schemas**: Feature importance and encoding maps from Step 3
- **Visualization Data**:
  - BupaR process matrices and traces from Step 5a
  - FP-Growth association rules and itemsets from Step 5b
  - DTW trajectory clusters from Step 5d
- **Causal Data**: FFA causal effects from Step 8 (consensus with SHAP from Step 7)

### Outputs

- **Dashboard Artifacts**:
  - `10_results/models/{cohort}/{age_band}/` - Packaged models (joblib, JSON)
  - `10_results/metadata/metadata_{cohort}.json` - Valid codes for dropdowns
- **Deployment Artifacts**:
  - Docker container image (ECR)
  - Lambda function package
  - Static site files (S3)

## 📚 Related Documentation

- **Step 3**: See [`../Step3_FeatureImportance/`](../Step3_FeatureImportance/) for feature importance analysis
- **Feature Importance EDA**: See [`../../3b_feature_importance_eda/`](../../3b_feature_importance_eda/) for BupaR post-target analysis feature refinement
- **Step 6**: See [`../Step6_FinalModel/`](../Step6_FinalModel/) for model training
- **Step 7**: See [`../../7_shap_analysis/`](../../7_shap_analysis/) for SHAP analysis
- **Step 8**: See [`../Step8_FFA/`](../Step8_FFA/) for FFA causal analysis (uses SHAP to prioritize rules)
- **Main Index**: See [`../README.md`](../README.md) for complete documentation index
