# Combined Dissertation Outline and Project Workflow

---

## Phase 1: Foundation and Methodological Architecture

### Chapter 1: Introduction & Systematic Quantitative Literature Review (SQLR)
- **Clinical Challenge:** Opioid use disorder (OUD) in youth and polypharmacy in older adults, focusing on ED utilization.
- **Technical Challenge:** Complexities of APCD/EHR data, "warped time," and administrative noise. Need for Explainable AI (XAI) using SHAP and FFA.
- **SQLR Methodology:** PRISMA flowchart, inclusion/exclusion criteria, and knowledge gap analysis—demonstrating the lack of models combining XAI with PGx guidelines for personalized dosing.

### Chapter 2: Methodology & Data Engineering Architecture
- **Partition-First Architecture:** 10-step pipeline using DuckDB, S3 checkpoints, and parallelizable jobs.
- **Cohort Design & Temporal Validation:** Dual-target cohorts (Opioid vs. Non-Opioid), 5:1 control sampling, strict temporal validation (train: 2016–2018, test: 2019, exclude: 2020).
- **Target Leakage Prevention:** DTW, BupaR, FP-Growth for visualization only; event filtering before cohort creation.
- **Monte Carlo Cross-Validation:** Feature selection with CatBoost, XGBoost, and XGBoost RF.

#### Project Workflow Mapping
- **Step 1a:** APCD Input Data Processing (text → Parquet, cleaning, mapping)
- **Step 1b:** Event Filtering (ICD/admin codes, leakage removal)
- **Step 2:** Cohort Creation (target/control, age bands, QA)
- **Step 3a:** Feature Importance Screening (MC-CV, model aggregation)
- **Step 3b:** Feature Refinement (BupaR, clinical validation)
- **Step 3c:** Final Feature Update (leakage removal, QA)

---

## Phase 2: Core Research ("Manuscript" Chapters)

### Chapter 3: Predicting Opioid-Related ED Visits & Trajectory Mapping
- **Focus:** Opioid ED Cohort (F11.20)
- **Feature Space:** ICD/CPT codes, drug names/counts, CPIC counts
- **Exploratory Sequence Analysis:** BupaR process mining, DTW clustering
- **Per-Bin Modeling:** Four models by event density (low/medium/high/extreme)
- **Causal Attribution:** Consensus Filter (CatBoost SHAP + XGBoost rules + counterfactuals)

### Chapter 4: Polypharmacy, Drug Interactions, and Causal Rules
- **Focus:** Non-Opioid ED cohort (geriatric/high-cost)
- **Causal Calculator:** Feature space limited to drug names/counts, CPIC counts
- **Combinatorial Analysis:** FFA pipeline for feature pairs/triplets, Boolean logic rules
- **Synergy vs. Antagonism:** FP-Growth networks for drug-drug interactions

### Chapter 5: Translation to Practice – The PGx Risk Dashboard
- **Focus:** Clinical deployment and decision support
- **Serverless Architecture:** AWS Lambda, Docker, API Gateway, S3 frontend
- **Ensemble Risk Scoring:** PR-AUC mean, probability-weighted averaging
- **PGx Patient Card:** Genetic data separation, CPIC matching for dosing
- **Visual Context:** BupaR/DTW flows for context

#### Project Workflow Mapping
- **Step 4:** Model Data Preparation (event extraction, leakage removal)
- **Step 5:** PGx Feature Engineering (genetic-drug interactions)
- **Step 6:** Final Model Training (XGBoost, CatBoost, Ensemble, per-bin)
- **Step 7:** SHAP Analysis (global/local feature importance)
- **Step 8:** FFA Analysis (symbolic rule explanations)
- **Step 9:** Dashboard Visuals (BupaR, DTW, FP-Growth)
- **Step 10:** Build & Deploy (Lambda, Docker, S3, API Gateway)

---

## Phase 3: Synthesis

### Chapter 6: Conclusion
- **Synthesis:** Integrate findings from all manuscript chapters
- **Contributions:** XAI, APCD processing, and PGx advances in clinical predictive modeling
- **Limitations & Future Work:** Pipeline/data limitations, future research directions

---

## Appendix: Repository and Workflow Details

- **Repository Structure:**
  - 1a_apcd_input_data/: Data preprocessing
  - 1b_apcd_event_filter/: Event filtering
  - 2_create_cohort/: Cohort creation
  - 3a_feature_importance/: Feature importance
  - 3b_feature_importance_eda/: Feature refinement
  - 4_model_data/: Model-ready data
  - 5_pgx_analysis/: PGx feature engineering
  - 6_final_model/: Model training
  - 7_shap_analysis/: SHAP analysis
  - 8_ffa_analysis/: FFA analysis
  - 9_dashboard_visuals/: Visualizations
  - 10_risk_dashboard/: Dashboard & deployment
  - 11_testing/: Integration/smoke tests
  - py_helpers/, r_helpers/: Utilities
  - docs/: Documentation

- **Key Design Decisions:**
  - Partition-first, S3 checkpoints, per-bin models, strict leakage prevention, temporal validation

- **Documentation:**
  - See docs/ for pipeline, lessons learned, and FFA methodology

---

This combined outline integrates the dissertation structure with the technical workflow, providing a comprehensive roadmap for both the research narrative and the supporting data science pipeline.
