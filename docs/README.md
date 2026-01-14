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

### Step 4: Model Data, DTW Protocol Filter & Extreme-Density Split
**Code**: `4a_model_data/`, `4b_dtw_filter/`, `5b_fpgrowth_analysis/extract_extreme_density_cohort.py`, `5b_fpgrowth_analysis/summarize_extreme_density_cohort.py`

- **[README_step4_overview.md](Step4_ModelData/README_step4_overview.md)** – Model-events extraction, DTW protocol filtering, and standardized extreme-density cohort split for all `(cohort, age_band)` combinations.

### Step 5: Pattern & Process Mining, PGx, and DTW Trajectories

- **Step 5a – Process Mining (BupaR)** – [`Step5a_BupaR/`](Step5a_BupaR/)
  - **[README_bupaR.md](Step5a_BupaR/README_bupaR.md)** – Process mining with BupaR.
- **Step 5b – FPGrowth Pattern Mining** – [`Step5b_FPGrowth/`](Step5b_FPGrowth/)
  - **[README_fpgrowth.md](Step5b_FPGrowth/README_fpgrowth.md)** – Frequent pattern mining with FPGrowth.
- **Step 5c – Pharmacogenomic (PGx) Analysis** – [`Step5c_PGxAnalysis/`](Step5c_PGxAnalysis/)
  - **[README_step5c_overview.md](Step5c_PGxAnalysis/README_step5c_overview.md)** – PGx mapping, allele frequencies, and PGx feature integration.
- **Step 5d – DTW Trajectory Features** – [`Step5d_DTW/`](Step5d_DTW/)
  - **[README_dtw_feature_extraction.md](Step5d_DTW/README_dtw_feature_extraction.md)** – Dynamic Time Warping trajectory analysis and feature extraction.

### Step 6: Final Model Development
**Location**: [`Step8_FinalModel/`](Step8_FinalModel/)

- **[README_final_model.md](Step8_FinalModel/README_final_model.md)** - Final model training and evaluation
- **[README_catboost.md](Step8_FinalModel/README_catboost.md)** - CatBoost model details

### Step 7: Feature Attribution Analysis (FFA)
**Location**: [`Step9_FFA/`](Step9_FFA/)

- **[README_ffa_analysis.md](Step9_FFA/README_ffa_analysis.md)** - Complete FFA analysis framework overview
- **[README_ffa_causal_analysis.md](Step9_FFA/README_ffa_causal_analysis.md)** - Dual-approach causal analysis guide
- **[README_ffa_unified_schema.md](Step9_FFA/README_ffa_unified_schema.md)** - Unified schema for symbolic explainers

### Step 8: SHAP & Combined SHAP+FFA
- **SHAP Analysis** – `8_shap_analysis/` (see `docs/README_analysis_workflow.md` for details)
- **Combined SHAP + FFA** – `9_combined_shap_ffa/` and [`Step10_Results/README_combined_ffa_shap_causal_analysis.md`](Step10_Results/README_combined_ffa_shap_causal_analysis.md)

### Step 10: Results & Dashboard
**Location**: [`Step10_Results/`](Step10_Results/)

**Main Documentation:**
- **[README_results_overview.md](Step10_Results/README_results_overview.md)** - Complete documentation index and overview
- **[README_results_dashboard.md](Step10_Results/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_dashboard_tabs.md](Step10_Results/README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard_visualizations.md](Step10_Results/README_results_dashboard_visualizations.md)** - Advanced visualization system (BupaR, FP-Growth, DTW)
- **[README_results_dashboard_deployment.md](Step10_Results/README_results_dashboard_deployment.md)** - Deployment guide (incremental builds, Docker, Lambda)
- **[README_results_value_proposition.md](Step10_Results/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](Step10_Results/README_results_deployment.md)** - Complete deployment guide
- **[README_results_prediction.md](Step10_Results/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](Step10_Results/README_results_quickstart.md)** - Quick start guide for predictions

**Feature Documentation:**
- **[README_results_pgx_card.md](Step10_Results/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](Step10_Results/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](Step10_Results/README_results_model_weights.md)** - Performance-based model weighting
- **[README_combined_ffa_shap_causal_analysis.md](Step10_Results/README_combined_ffa_shap_causal_analysis.md)** - Combined FFA, SHAP, and causal analysis guide

**Deployment Guides:**
- **[README_results_deployment_ecr.md](Step10_Results/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](Step10_Results/README_results_deployment_cpic.md)** - CPIC data deployment

**Reference:**
- **[README_results_storage.md](Step10_Results/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](Step10_Results/README_results_age_bands.md)** - Supported age bands and mappings

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
1. **[Step10_Results/README_results_deployment.md](Step10_Results/README_results_deployment.md)** - Complete deployment guide
2. **[Step10_Results/README_results_dashboard_deployment.md](Step10_Results/README_results_dashboard_deployment.md)** - Dashboard deployment guide (incremental builds, Docker, Lambda)
3. **[Step10_Results/README_results_deployment_ecr.md](Step10_Results/README_results_deployment_ecr.md)** - ECR container deployment
4. **[Step10_Results/README_results_dashboard.md](Step10_Results/README_results_dashboard.md)** - Dashboard overview
5. **[Step10_Results/README_results_dashboard_tabs.md](Step10_Results/README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints

### Using the Dashboard
1. **[Step10_Results/README_results_quickstart.md](Step10_Results/README_results_quickstart.md)** - Quick start for predictions
2. **[Step10_Results/README_results_prediction.md](Step10_Results/README_results_prediction.md)** - Detailed prediction workflow
3. **[Step10_Results/README_results_pgx_card.md](Step10_Results/README_results_pgx_card.md)** - PGx card generation

## 📝 Documentation Structure

```
docs/
├── Step1-2_DataPipeline/      # Data pipeline & cohort creation
├── Step3_FeatureImportance/   # Feature importance analysis
├── Step4_ModelData/           # Model data, DTW protocol filter, and extreme-density split
├── Step5a_BupaR/              # Process mining documentation
├── Step5b_FPGrowth/           # FPGrowth documentation (pattern mining layer)
├── Step5c_PGxAnalysis/        # Pharmacogenomic (PGx) feature engineering
├── Step5d_DTW/                # DTW trajectory analysis docs
├── Step8_FinalModel/          # Final model development
├── Step9_FFA/                 # Formal Feature Attribution analysis
├── Step10_Results/            # Results, combined SHAP+FFA, dashboard
├── CrossStep_Workflow/        # Cross-step workflow docs (includes Step 4a/4b/4c and extreme cohorts)
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
