# Documentation Index

This folder contains comprehensive documentation for the PGx analysis pipeline, organized by workflow step and topic.

## 📚 Documentation by Workflow Step

### Final Workflow (Phases 0–5)

Notebooks are broken up by **phase 0–5**. Run in order: **Phase 0** → **1** → **2** → **3** → **4** → **5**.

- **Phase 0:** `0_config_and_pipeline.ipynb` — Clear checkpoints, env checks. Run first for full E2E.
- **Phase 1:** `1_cohort_workflow.ipynb` (Steps 1-2): 1a APCD input, 1b event filter (aggregated FI + ICD/admin; target leakage removed in Step 4), 2 cohort creation. Uses S3 sync to NVMe and S3 checkpoints (idempotent).
- **Phase 2:** `2_feature_importance.ipynb` (Steps 3a-3c): 3a MC-CV feature importance, 3b BupaR/code research, 3c final update to features. Run after cohorts; sync gold/cohorts from S3.
- **Phase 3:** `3_model_train_shap_ffa.ipynb`: Model data → PGx → final model → SHAP → FFA → combine (no deploy). Sync 3a/3b/6 outputs from S3; checkpoint metadata/models prep.
- **Phase 3 sensitivity (CH4 R2):** `3_model_sensitivity.ipynb` → `6_final_model/run_sensitivity_util_free.py` (utilization-free refit; journal SSOT under `reports/CTS-2026-0235R2/`).
- **Phase 4:** `4_dashboard_visuals.ipynb` (Step 9): BupaR, DTW, FP-Growth (SHAP/FFA-driven). Alternative: `pgx_dashboard_visuals.py`.
- **Phase 5:** `5_build_and_deploy.ipynb` (Step 10): Lambda dir → Docker → ECR → Lambda → S3 frontend. Run once.

### Step 1-2: Data Pipeline & Cohort Creation
**Location**: [`Step1-2_DataPipeline/`](Step1-2_DataPipeline/), **Code**: `1a_apcd_input_data/`, `1b_apcd_event_filter/`, `2_create_cohort/`

- **[README_data_pipeline.md](Step1-2_DataPipeline/README_data_pipeline.md)** - Complete data pipeline architecture and optimization
- **[README_input_data_overview.md](Step1-2_DataPipeline/README_input_data_overview.md)** - APCD input data overview (Step 1a)
- **[README_create_cohort.md](Step1-2_DataPipeline/README_create_cohort.md)** - Cohort creation guide
- **[README_create_cohort_pipeline.md](Step1-2_DataPipeline/README_create_cohort_pipeline.md)** - Cohort pipeline phases and execution
- **[README_preprocessing.md](Step1-2_DataPipeline/README_preprocessing.md)** - Data preprocessing steps
- **[README_preprocessing_pipeline.md](Step1-2_DataPipeline/README_preprocessing_pipeline.md)** - Preprocessing pipeline detail
- **[README_s3_datalake.md](Step1-2_DataPipeline/README_s3_datalake.md)** - S3 data lake structure
- **[README_pipeline_state_tracking.md](Step1-2_DataPipeline/README_pipeline_state_tracking.md)** - Pipeline state and checkpoint tracking

### Step 3: Feature Importance Analysis
**Location**: [`Step3_FeatureImportance/`](Step3_FeatureImportance/)

- **[README_feature_importance_overview.md](Step3_FeatureImportance/README_feature_importance_overview.md)** - Feature importance overview and context
- **[README_feature_importance.md](Step3_FeatureImportance/README_feature_importance.md)** - Feature importance analysis methodology
- **[README_feature_importance_visualization.md](Step3_FeatureImportance/README_feature_importance_visualization.md)** - Feature importance visualization guide

### Feature Importance EDA: Feature Refinement (BupaR Process Mining + Code Research)
**Code**: `3b_feature_importance_eda/`

- **BupaR Post-Target Analysis** – Uses process mining to identify post-target leakage; Step 4 removes those events when building model data
- **Code Research and Validation** – Researches and identifies non-value-added administrative/scheduling codes (event-level filtering in **Step 1b**: `1b_apcd_event_filter`)
- **Output**: Refined `cohort_feature_importance.csv` files; Step 3c (2_feature_importance.ipynb) is the final update before Step 4

### Step 4: Model Data
**Code**: `4_model_data/` | **Location**: [`Step4_ModelData/`](Step4_ModelData/)

- **[README_model_data_overview.md](Step4_ModelData/README_model_data_overview.md)** – Model-events extraction for all `(cohort, age_band)` combinations
- **[README_feature_engineering_and_analysis.md](Step4_ModelData/README_feature_engineering_and_analysis.md)** – Feature engineering pipeline overview
- **[README_feature_encoding.md](Step4_ModelData/README_feature_encoding.md)** – Categorical encoding strategies
- **[README_model_data_and_extreme_split.md](Step4_ModelData/README_model_data_and_extreme_split.md)** – Extreme-density cohort split and model data schema
- Model-ready event datasets (target vs control) using refined features from Step 3c. Step 4 removes target leakage for case events (events before target date only). Event/ICD filtering runs in **Step 1b** (`1b_apcd_event_filter`).

### Step 5: PGx Feature Engineering
**Location**: [`Step5_PGxAnalysis/`](Step5_PGxAnalysis/)

- **[README_pgx_analysis_overview.md](Step5_PGxAnalysis/README_pgx_analysis_overview.md)** – PGx CPIC drug counts only (no alleles in this pipeline; alleles used in PGx card via patient-submitted SNP data).

### Step 6: Final Model Development
**Code**: `6_final_model/` | **Location**: [`Step6_FinalModel/`](Step6_FinalModel/)

- **[README_final_model.md](Step6_FinalModel/README_final_model.md)** - Final model training and evaluation
- **[README_catboost.md](Step6_FinalModel/README_catboost.md)** - CatBoost model details
- **[README_xgboost.md](Step6_FinalModel/README_xgboost.md)** - XGBoost model details

### Step 7: SHAP Analysis
**Code**: `7_shap_analysis/` | **Location**: [`Step7_SHAP/`](Step7_SHAP/)

- **[README_shap_analysis.md](Step7_SHAP/README_shap_analysis.md)** — Two-pass SHAP methodology (XGBoost `pred_contribs` + CatBoost `ShapValues`), outputs, per-bin support, S3 paths
- Used by Step 8 (FFA) to prioritize and filter rules; both XGBoost and CatBoost run regardless of model selection

### Step 8: Formal Feature Attribution (FFA)
**Location**: [`Step8_FFA/`](Step8_FFA/)

- **[README_ffa_analysis.md](Step8_FFA/README_ffa_analysis.md)** - Complete FFA analysis framework overview
- **[README_ffa_causal_analysis.md](Step8_FFA/README_ffa_causal_analysis.md)** - Dual-approach causal analysis guide
- **[README_ffa_unified_schema.md](Step8_FFA/README_ffa_unified_schema.md)** - Unified schema for symbolic explainers
- **Note**: FFA analysis is performed only for XGBoost models. Uses SHAP importance from Step 7 to filter and prioritize rules.

### Steps 9–10: Dashboard Visuals & Risk Dashboard
**Step 9 code**: `9_dashboard_visuals/` | **Step 10 code**: `10_risk_dashboard/` | **Docs**: [`Step9_RiskDashboard/`](Step9_RiskDashboard/)

**Main Documentation:**
- **[README_results_overview.md](Step9_RiskDashboard/README_results_overview.md)** - Complete documentation index and overview
- **[README_dashboard.md](Step9_RiskDashboard/README_dashboard.md)** - Dashboard overview (entry point)
- **[README_results_dashboard.md](Step9_RiskDashboard/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_dashboard_tabs.md](Step9_RiskDashboard/README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard_visualizations.md](Step9_RiskDashboard/README_results_dashboard_visualizations.md)** - Advanced visualization system (BupaR, FP-Growth, DTW)
- **[README_results_dashboard_deployment.md](Step9_RiskDashboard/README_results_dashboard_deployment.md)** - Deployment guide (incremental builds, Docker, Lambda)
- **[README_results_value_proposition.md](Step9_RiskDashboard/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](Step9_RiskDashboard/README_results_deployment.md)** - Complete deployment guide
- **[README_results_prediction.md](Step9_RiskDashboard/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](Step9_RiskDashboard/README_results_quickstart.md)** - Quick start guide for predictions

**Visualization Documentation (Step 9):**
- **[README_bupar_dashboard_visualizations.md](Step9_RiskDashboard/README_bupar_dashboard_visualizations.md)** - BupaR process mining visualizations
- **[README_dtw_dashboard_visualizations.md](Step9_RiskDashboard/README_dtw_dashboard_visualizations.md)** - DTW trajectory visualizations
- **[README_fpgrowth_dashboard_visualizations.md](Step9_RiskDashboard/README_fpgrowth_dashboard_visualizations.md)** - FP-Growth itemset visualizations

**Feature Documentation (Step 10):**
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

- **[README_analysis_workflow.md](CrossStep_Workflow/README_analysis_workflow.md)** - Canonical analysis workflow (Steps 1–10; five notebooks 1→2→3→4→5)
- **[README_research_questions_mapping.md](CrossStep_Workflow/README_research_questions_mapping.md)** - Research questions to analysis methods mapping
- **[README_healthcare_outcomes.md](CrossStep_Workflow/README_healthcare_outcomes.md)** - Healthcare outcomes rationale for cohort design
- **Cross-age band analysis** — archived; no longer part of active pipeline

### Visualization & Output
**Location**: [`CrossStep_Visualization/`](CrossStep_Visualization/)

- **[README_data_visualization.md](CrossStep_Visualization/README_data_visualization.md)** - Data visualization approaches
- **[README_data_visualizations.md](CrossStep_Development/README_data_visualizations.md)** - Pipeline visualization types, SHAP dependence plots (not PDP), network/DTW/FFA
- **[README_output_structure.md](CrossStep_Visualization/README_output_structure.md)** - Output file structure
- **Manuscript figure production:** [`../manuscript/infrastructure_setup/scripts/generate_figures/README.md`](../manuscript/infrastructure_setup/scripts/generate_figures/README.md)

### Development & Testing
**Location**: [`CrossStep_Development/`](CrossStep_Development/)

- **Workflow testing** (QA): see [`11_testing/DASHBOARD_VALIDATION.md`](../11_testing/DASHBOARD_VALIDATION.md) and [`11_testing/TEST_PLAN_FINAL_DASHBOARD.md`](../11_testing/TEST_PLAN_FINAL_DASHBOARD.md)
- **[README_parallelization_pipeline.md](CrossStep_Development/README_parallelization_pipeline.md)** - Parallelization strategies
- **[NotebookDevelopmentWorkflow.md](NotebookDevelopmentWorkflow.md)** - Dev vs production notebook workflow, Mermaid diagram, artifact setup cell, GitHub/S3 output conventions
- **[README_local_notebook.md](CrossStep_Development/README_local_notebook.md)** - Local notebook development
- **[README_duckdb_dev.md](CrossStep_Development/README_duckdb_dev.md)** - DuckDB development notes
- **[README_target_leakage.md](CrossStep_Development/README_target_leakage.md)** - Target leakage prevention
- **[README_logging.md](CrossStep_Development/README_logging.md)** - Pipeline logging architecture, EC2/S3 log locations, troubleshooting guide
- **[README_event_density_bins.md](CrossStep_Development/README_event_density_bins.md)** - n_event_bin architecture end-to-end (training → SHAP/FFA → Lambda)
- **[README_lessons_learned.md](CrossStep_Development/README_lessons_learned.md)** - Critical QA bugs, design decisions, final production workflow lessons (incl. Cursor notebook stability July 2026; remove intermediates)
- **[NotebookDevelopmentWorkflow.md](NotebookDevelopmentWorkflow.md)** - Final notebook / Cursor workflow (script-first; crash mitigations)
- **[DEPLOYMENT_LESSONS_LEARNED.md](../10_risk_dashboard/docs/DEPLOYMENT_LESSONS_LEARNED.md)** - Dashboard deployment lessons (tab display bugs, CloudFront/browser cache layers)
- **[README_target_leakage.md](../9_dashboard_visuals/fpgrowth/README_target_leakage.md)** - FP-Growth target leakage analysis (confirmed: FP-Growth features cause leakage; visualization only)

## 📊 Presentations
**Location**: [`Presentations/`](Presentations/)

- **[3Min_Presentation_Quick_Reference.md](Presentations/3Min_Presentation_Quick_Reference.md)** - 3-minute presentation quick reference
- **[Pharmacy_Translational_Informatics_Presentation.md](Presentations/Pharmacy_Translational_Informatics_Presentation.md)** - Pharmacy translational informatics presentation
- **[README_dissertation_defense_slides.md](Presentations/README_dissertation_defense_slides.md)** - Dissertation defense slide workflow, NotebookLM input staging scripts, CTS manuscript mapping

## 🚀 Quick Links

### Getting Started
1. Start with **[Step1-2_DataPipeline/README_data_pipeline.md](Step1-2_DataPipeline/README_data_pipeline.md)** for data pipeline overview
2. Review **[Step1-2_DataPipeline/README_create_cohort.md](Step1-2_DataPipeline/README_create_cohort.md)** for cohort creation
3. Follow **[CrossStep_Workflow/README_analysis_workflow.md](CrossStep_Workflow/README_analysis_workflow.md)** for analysis steps

### Deployment
1. **[Step9_RiskDashboard/README_results_deployment.md](Step9_RiskDashboard/README_results_deployment.md)** - Complete deployment guide (Step 10)
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
├── Step1-2_DataPipeline/      # Data pipeline & cohort creation (Steps 1a, 1b, 2)
├── Step3_FeatureImportance/   # Step 3a: Feature importance (MC-CV)
├── Step4_ModelData/           # Step 4: Model data (model_events.parquet)
├── Step5_PGxAnalysis/         # Step 5: Pharmacogenomic (PGx) feature engineering
├── Step6_FinalModel/          # Step 6: Final model development
├── Step7_SHAP/                # Step 7: SHAP analysis (XGBoost + CatBoost, two-pass)
├── Step8_FFA/                 # Step 8: Formal Feature Attribution analysis
├── Step9_RiskDashboard/       # Steps 9–10: Dashboard visuals (9_dashboard_visuals) & risk dashboard deploy (10_risk_dashboard)
├── CrossStep_Workflow/        # Cross-step workflow docs
├── CrossStep_Visualization/   # Visualization docs
├── CrossStep_Development/     # Development, logging, event density bins, lessons learned
└── Presentations/             # Presentation materials
```

**Shared utilities**: `py_helpers/README.md` — inventory of all shared Python utilities, active vs. stale helpers, archiving guidance

## Documentation and file naming conventions

- **READMEs**: `README.md` or `README_<lowercase_with_underscores>.md` (e.g. `README_model_data_overview.md`). No numbered suffixes (e.g. avoid `README2.md`). Exception: proper nouns such as `README_bupaR.md` (BupaR) may keep their capitalization.
- **Standalone / technical docs**: `UPPERCASE_WITH_UNDERSCORES.md` (e.g. `WORKFLOW_UPDATES.md`, `TIME_ESTIMATES.md`).
- **No spaces** in filenames; use underscores. Exceptions (e.g. `Presentations/`) may use Title_Case where needed.
- **References**: Use the exact filename case in links so they work on case-sensitive systems.

## 🔍 Finding Documentation

- **By Step**: Navigate to the appropriate `Step*_*/` folder
- **By Topic**: Check `CrossStep_*/` folders for cross-cutting topics
- **By Type**: 
  - Step folders contain step-specific documentation
  - CrossStep folders contain workflow-wide documentation
  - Presentations folder contains presentation materials
