# Documentation Index

This folder contains comprehensive documentation for the PGx analysis pipeline, organized by workflow step and topic.

## 📚 Documentation by Workflow Step

### Step 1-2: Data Pipeline & Cohort Creation
**Location**: [`Step1-2_DataPipeline/`](Step1-2_DataPipeline/)

- **[README_data_pipeline.md](Step1-2_DataPipeline/README_data_pipeline.md)** - Complete data pipeline architecture and optimization
- **[README_create_cohort.md](Step1-2_DataPipeline/README_create_cohort.md)** - Cohort creation guide
- **[README_preprocessing.md](Step1-2_DataPipeline/README_preprocessing.md)** - Data preprocessing steps
- **[README_s3_datalake.md](Step1-2_DataPipeline/README_s3_datalake.md)** - S3 data lake structure

### Step 3: Feature Importance Analysis
**Location**: [`Step3_FeatureImportance/`](Step3_FeatureImportance/)

- **[README_feature_importance.md](Step3_FeatureImportance/README_feature_importance.md)** - Feature importance analysis methodology
- **[README_feature_importance_visualization.md](Step3_FeatureImportance/README_feature_importance_visualization.md)** - Feature importance visualization guide

### Step 3b: Feature Refinement (BupaR Process Mining + Code Research)
**Code**: `3b_feature_importance_eda/`

- **BupaR Post-Target Analysis** – Uses process mining to identify post-target leakage features in aggregated importances
- **Code Research and Validation** – Researches and identifies non-value-added administrative/scheduling codes (actual event-level filtering happens in Step 4b)
- **Filter Post-Target Leakage** – Removes post-target leakage features from already-processed aggregated feature importance list
- **Note**: This is NOT a DTW filter - it filters aggregated feature importances using BupaR process mining and code research
- **Output**: Refined `cohort_feature_importance.csv` files that feed into Step 4a

### Step 4: Model Data & Filtering
**Code**: `4a_model_data/`, `4b_dtw_filter/`

- **[README_step4_overview.md](Step4_ModelData/README_step4_overview.md)** – Model-events extraction, DTW protocol filtering, and standardized extreme-density cohort split for all `(cohort, age_band)` combinations.
- **Step 4a**: Model-ready event datasets (target vs control) using refined features from Step 3b
- **Step 4b**: DTW protocol filtering to remove administrative codes, creating `model_events_no_protocols.parquet`

### Step 5: PGx Feature Engineering
**Location**: [`Step5_PGxAnalysis/`](Step5_PGxAnalysis/)

- **[README_pgx_analysis_overview.md](Step5_PGxAnalysis/README_pgx_analysis_overview.md)** – PGx mapping, allele frequencies, and PGx feature integration.

### Step 6: Final Model Development
**Location**: [`Step6_FinalModel/`](Step6_FinalModel/)

- **[README_final_model.md](Step6_FinalModel/README_final_model.md)** - Final model training and evaluation
- **[README_catboost.md](Step6_FinalModel/README_catboost.md)** - CatBoost model details
- **[README_xgboost.md](Step6_FinalModel/README_xgboost.md)** - XGBoost model details

### Step 7: SHAP Analysis
**Code**: `7_shap_analysis/`

- **SHAP Analysis** – Global and local SHAP values for CatBoost and XGBoost models (see `docs/README_analysis_workflow.md` for details)
- Used by Step 8 (FFA) to prioritize and filter rules

### Step 8: Formal Feature Attribution (FFA)
**Location**: [`Step8_FFA/`](Step8_FFA/)

- **[README_ffa_analysis.md](Step8_FFA/README_ffa_analysis.md)** - Complete FFA analysis framework overview
- **[README_ffa_causal_analysis.md](Step8_FFA/README_ffa_causal_analysis.md)** - Dual-approach causal analysis guide
- **[README_ffa_unified_schema.md](Step8_FFA/README_ffa_unified_schema.md)** - Unified schema for symbolic explainers
- **Note**: FFA analysis is performed only for XGBoost models. Uses SHAP importance from Step 7 to filter and prioritize rules.

### Step 9: Risk Dashboard
**Location**: [`Step9_RiskDashboard/`](Step9_RiskDashboard/)

**Main Documentation:**
- **[README_results_overview.md](Step9_RiskDashboard/README_results_overview.md)** - Complete documentation index and overview
- **[README_results_dashboard.md](Step9_RiskDashboard/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_dashboard_tabs.md](Step9_RiskDashboard/README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard_visualizations.md](Step9_RiskDashboard/README_results_dashboard_visualizations.md)** - Advanced visualization system (BupaR, FP-Growth, DTW)
- **[README_results_dashboard_deployment.md](Step9_RiskDashboard/README_results_dashboard_deployment.md)** - Deployment guide (incremental builds, Docker, Lambda)
- **[README_results_value_proposition.md](Step9_RiskDashboard/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](Step9_RiskDashboard/README_results_deployment.md)** - Complete deployment guide
- **[README_results_prediction.md](Step9_RiskDashboard/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](Step9_RiskDashboard/README_results_quickstart.md)** - Quick start guide for predictions

**Feature Documentation:**
- **[README_results_pgx_card.md](Step9_RiskDashboard/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](Step9_RiskDashboard/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](Step9_RiskDashboard/README_results_model_weights.md)** - Performance-based model weighting
- **[README_combined_ffa_shap_causal_analysis.md](Step9_RiskDashboard/README_combined_ffa_shap_causal_analysis.md)** - Combined FFA, SHAP, and causal analysis guide

**Deployment Guides:**
- **[README_results_deployment_ecr.md](Step9_RiskDashboard/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](Step9_RiskDashboard/README_results_deployment_cpic.md)** - CPIC data deployment

**Reference:**
- **[README_results_storage.md](Step9_RiskDashboard/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](Step9_RiskDashboard/README_results_age_bands.md)** - Supported age bands and mappings

**Main Documentation:**
- **[README_results_overview.md](Step9_RiskDashboard/README_results_overview.md)** - Complete documentation index and overview
- **[README_results_dashboard.md](Step9_RiskDashboard/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_dashboard_tabs.md](Step9_RiskDashboard/README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard_visualizations.md](Step9_RiskDashboard/README_results_dashboard_visualizations.md)** - Advanced visualization system (BupaR, FP-Growth, DTW)
- **[README_results_dashboard_deployment.md](Step9_RiskDashboard/README_results_dashboard_deployment.md)** - Deployment guide (incremental builds, Docker, Lambda)
- **[README_results_value_proposition.md](Step9_RiskDashboard/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](Step9_RiskDashboard/README_results_deployment.md)** - Complete deployment guide
- **[README_results_prediction.md](Step9_RiskDashboard/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](Step9_RiskDashboard/README_results_quickstart.md)** - Quick start guide for predictions

**Feature Documentation:**
- **[README_results_pgx_card.md](Step9_RiskDashboard/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](Step9_RiskDashboard/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](Step9_RiskDashboard/README_results_model_weights.md)** - Performance-based model weighting
- **[README_combined_ffa_shap_causal_analysis.md](Step9_RiskDashboard/README_combined_ffa_shap_causal_analysis.md)** - Combined FFA, SHAP, and causal analysis guide

**Deployment Guides:**
- **[README_results_deployment_ecr.md](Step9_RiskDashboard/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](Step9_RiskDashboard/README_results_deployment_cpic.md)** - CPIC data deployment

**Reference:**
- **[README_results_storage.md](Step9_RiskDashboard/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](Step9_RiskDashboard/README_results_age_bands.md)** - Supported age bands and mappings

## 🔄 Cross-Step Documentation

### Workflow & Analysis
**Location**: [`CrossStep_Workflow/`](CrossStep_Workflow/)

- **[README_analysis_workflow.md](CrossStep_Workflow/README_analysis_workflow.md)** - Complete analysis workflow (FPGrowth → CatBoost → BupaR)
- **[README_research_questions_mapping.md](CrossStep_Workflow/README_research_questions_mapping.md)** - Research questions to analysis methods mapping
- **[README_healthcare_outcomes.md](CrossStep_Workflow/README_healthcare_outcomes.md)** - Healthcare outcomes rationale for cohort design
- **[README_cross_ageband_analysis.md](CrossStep_Workflow/README_cross_ageband_analysis.md)** - Cross-age band analysis

### Visualization & Output
**Location**: [`CrossStep_Visualization/`](CrossStep_Visualization/)

- **[README_data_visualization.md](CrossStep_Visualization/README_data_visualization.md)** - Data visualization approaches
- **[README_output_structure.md](CrossStep_Visualization/README_output_structure.md)** - Output file structure

### Development & Testing
**Location**: [`CrossStep_Development/`](CrossStep_Development/)

- **[README_workflow_testing.md](CrossStep_Development/README_workflow_testing.md)** - Workflow testing and validation framework
- **[README_parallelization_pipeline.md](CrossStep_Development/README_parallelization_pipeline.md)** - Parallelization strategies
- **[README_local_notebook.md](CrossStep_Development/README_local_notebook.md)** - Local notebook development
- **[README_duckdb_dev.md](CrossStep_Development/README_duckdb_dev.md)** - DuckDB development notes
- **[README_target_leakage.md](CrossStep_Development/README_target_leakage.md)** - Target leakage prevention

## 📊 Presentations
**Location**: [`Presentations/`](Presentations/)

- **[3Min_Presentation_Quick_Reference.md](Presentations/3Min_Presentation_Quick_Reference.md)** - 3-minute presentation quick reference
- **[Pharmacy_Translational_Informatics_Presentation.md](Presentations/Pharmacy_Translational_Informatics_Presentation.md)** - Pharmacy translational informatics presentation

## 🚀 Quick Links

### Getting Started
1. Start with **[Step1-2_DataPipeline/README_data_pipeline.md](Step1-2_DataPipeline/README_data_pipeline.md)** for data pipeline overview
2. Review **[Step1-2_DataPipeline/README_create_cohort.md](Step1-2_DataPipeline/README_create_cohort.md)** for cohort creation
3. Follow **[CrossStep_Workflow/README_analysis_workflow.md](CrossStep_Workflow/README_analysis_workflow.md)** for analysis steps

### Deployment
1. **[Step9_RiskDashboard/README_results_deployment.md](Step9_RiskDashboard/README_results_deployment.md)** - Complete deployment guide (Step 9)
2. **[Step9_RiskDashboard/README_results_dashboard_deployment.md](Step9_RiskDashboard/README_results_dashboard_deployment.md)** - Dashboard deployment guide (incremental builds, Docker, Lambda)
3. **[Step9_RiskDashboard/README_results_deployment_ecr.md](Step9_RiskDashboard/README_results_deployment_ecr.md)** - ECR container deployment
4. **[Step9_RiskDashboard/README_results_dashboard.md](Step9_RiskDashboard/README_results_dashboard.md)** - Dashboard overview
5. **[Step9_RiskDashboard/README_results_dashboard_tabs.md](Step9_RiskDashboard/README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints

### Using the Dashboard
1. **[Step9_RiskDashboard/README_results_quickstart.md](Step9_RiskDashboard/README_results_quickstart.md)** - Quick start for predictions
2. **[Step9_RiskDashboard/README_results_prediction.md](Step9_RiskDashboard/README_results_prediction.md)** - Detailed prediction workflow
3. **[Step9_RiskDashboard/README_results_pgx_card.md](Step9_RiskDashboard/README_results_pgx_card.md)** - PGx card generation

## 📝 Documentation Structure

```
docs/
├── Step1-2_DataPipeline/      # Data pipeline & cohort creation
├── Step3_FeatureImportance/   # Step 3: Feature importance analysis
├── Step3b_FeatureRefinement/  # Step 3b: Feature refinement (BupaR post-target analysis) - NEW
├── Step4_ModelData/           # Step 4: Model data, DTW protocol filter, and extreme-density split
├── Step5_PGxAnalysis/        # Step 5: Pharmacogenomic (PGx) feature engineering
├── Step6_FinalModel/          # Step 6: Final model development
├── Step7_SHAP/                # Step 7: SHAP analysis
├── Step8_FFA/                 # Step 8: Formal Feature Attribution analysis
├── Step9_RiskDashboard/       # Step 9: Risk dashboard
├── CrossStep_Workflow/        # Cross-step workflow docs
├── CrossStep_Visualization/   # Visualization docs
├── CrossStep_Development/     # Development & testing docs
└── Presentations/             # Presentation materials
```

## 🔍 Finding Documentation

- **By Step**: Navigate to the appropriate `Step*_*/` folder
- **By Topic**: Check `CrossStep_*/` folders for cross-cutting topics
- **By Type**: 
  - Step folders contain step-specific documentation
  - CrossStep folders contain workflow-wide documentation
  - Presentations folder contains presentation materials
