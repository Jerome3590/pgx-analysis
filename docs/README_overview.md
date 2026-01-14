# Overview

Project structure, components, and high-level workflow for the Prescription Drug Analysis pipeline.

## Project Structure

```mermaid
graph TD
    A[pgx_analysis] --> B[1_apcd_input_data]
    A --> C[2_create_cohort]
    A --> D[3_feature_importance]
    A --> E[4a_model_data]
    A --> E2[4b_dtw_filter]
    A --> H[5_pgx_analysis]
    A --> J[6_final_model_selection]
    A --> K[7_shap_analysis]
    A --> L[8_ffa_analysis]
    A --> N[9_risk_dashboard]
    A --> O[py_helpers]
    
    B --> B1[medical]
    B --> B2[pharmacy]
    B --> B3[0_txt_to_parquet.py]
    B --> B4[3_apcd_clean.py]
    B --> B5[drug_mappings]
    B --> B6[claim_mappings]
    
    C --> C1[0_create_cohort.py]
    C --> C2[2_step2_data_quality_qa.py]
    C --> C3[phases]
    C --> C4[table_mappings]
    
    D --> D1[run_mc_feature_importance.py]
    D --> D2[aggregated_feature_importance.csv]
    
    E --> E1[create_model_data.py]
    E --> E3[model_events.parquet]
    
    E2 --> E4[filter_protocol_events.py]
    E2 --> E5[model_events_no_protocols.parquet]
    
    H --> H1[run_analysis.py]
    H --> H2[pgx_added_features.csv]
    
    J --> J1[run_final_model.py]
    J --> J2[final_model outputs]
    
    K --> K1[run_full_ffa_analysis.py<br/>Uses SHAP to prioritize rules]
    
    L --> L1[run_shap_analysis.py]
    
    N --> N1[Risk Dashboard]
    N --> N2[bupaR_dashboard_visual]
    N --> N3[fpgrowth_dashboard_visual]
    N --> N4[dtw_dashboard_visual]
    
    O --> O1[common_imports.py]
    O --> O2[duckdb_utils.py]
    O --> O3[s3_utils.py]
    O --> O4[constants.py]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style E2 fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px
    style J fill:#fbb,stroke:#333,stroke-width:2px
    style K fill:#fbf,stroke:#333,stroke-width:2px
    style L fill:#fbf,stroke:#333,stroke-width:2px
    style N fill:#ffb,stroke:#333,stroke-width:2px
    style O fill:#bfb,stroke:#333,stroke-width:2px
    style P fill:#ddd,stroke:#333,stroke-width:1px
```

## High-Level Workflow

End-to-end workflow for feature discovery, noise reduction, and causal-oriented modeling using drug exposures, ICD/CPT codes, and classification outcomes.

### Overview

This project builds a classification model on a large, noisy healthcare dataset, then uses model-based feature importance plus pattern- and process-mining to derive a stable covariate set and interpretable tree ensembles for causal analyses.

**High-level phases:**

1. **Feature Screening** with a focused model ensemble (CatBoost, XGBoost boosted trees, XGBoost RF mode) + Monte Carlo cross-validation
2. **Structure Discovery** and noise reduction with FP-Growth, process mining (BupaR), and dynamic time warping (DTW)
3. **Final Model Development** combining features from all analysis methods for prediction and causal inference

### Cohorts and Age Bands in the Current Run

The pipeline is designed to support many cohort / age-band combinations, but the **current
modeling plan** focuses on a fixed grid where we train a **separate final model for each cell**:

- **Cohort 1 – Opioid ED (`opioid_ed`)**
  - Age bands modeled: **0–12** (smoke-test cohort), **13–24**, **25–44**, **45–54**, **55–64**

- **Cohort 2 – Polypharmacy ED (`non_opioid_ed`)**
  - Age bands modeled: **65–74**, **75–84**, **85–94**

For every `(cohort, age_band)` above we run:
- MC‑CV feature importance (`3_feature_importance/`) producing aggregated feature importances
- model‑ready event extraction (`4a_model_data/`) creating event-level cases + controls
- DTW-based protocol filtering (`4b_dtw_filter/`) to create `model_events_no_protocols.parquet`
- PGx feature engineering (`5_pgx_analysis/`) adding pharmacogenomics features
- final model training and export (`6_final_model_selection/`), producing **one model per cohort/age‑band** using aggregated features + PGx features (no encoding)
- post-model analysis: SHAP (`8_shap_analysis/`) followed by FFA (`7_ffa_analysis/`), which uses SHAP importance to filter rules. FFA rule selection: union of (1) first 100 matched rules, (2) random sample of 100 matched rules, and (3) all rules with SHAP > 0
- risk dashboard (`9_risk_dashboard/`) with BupaR/FP-Growth/DTW visuals (these analyses are now dashboard-only, not separate workflow steps)

### Workflow Pipeline

```mermaid
flowchart TD
    subgraph "Step 1-2: Data Preparation"
        A1[APCD Input Data] --> A2[Data Cleaning]
        A2 --> A3[Cohort Creation]
        A3 --> A4[Quality Assurance]
    end
    
    subgraph "Step 3: Feature Discovery"
        A4 --> B1[Monte Carlo CV]
        B1 --> B2[Aggregated Feature Importance<br/>CatBoost + XGBoost]
        B2 --> B3[Top Features Selection]
    end
    
    subgraph "Step 4: Model Data & Filtering"
        B3 --> C1[4a: Model Data Extraction<br/>Event-level Cases + Controls]
        C1 --> C2[4b: DTW Protocol Filtering<br/>Remove Administrative Codes]
    end
    
    subgraph "Step 5: PGx Feature Engineering"
        C2 --> D1[PGx Feature Engineering<br/>Drug-Gene Mappings<br/>Allele Frequencies]
    end
    
    subgraph "Step 6: Final Model Training"
        D1 --> E1[Feature Integration<br/>Aggregated Features + PGx]
        E1 --> E2[CatBoost Training]
        E1 --> E3[XGBoost Training]
        E2 --> E4[Model Selection & Evaluation]
        E3 --> E4
    end
    
    subgraph "Step 7-8: Post-Model Analysis"
        E4 --> F1[7: SHAP Analysis<br/>SHAP Values]
        E4 --> F2[8: FFA Analysis<br/>Formal Feature Attribution<br/>Rule selection: first 100 + random 100 + all SHAP > 0]
        F1 --> F2
    end
    
    subgraph "Step 9: Risk Dashboard"
        F2 --> G1[Risk Dashboard<br/>Model Deployment]
        G1 --> G2[Dashboard Visuals:<br/>BupaR/FP-Growth/DTW]
    end
    
    style A1 fill:#f9f,stroke:#333,stroke-width:2px
    style B2 fill:#bbf,stroke:#333,stroke-width:2px
    style D1 fill:#bfb,stroke:#333,stroke-width:2px
    style E4 fill:#fbb,stroke:#333,stroke-width:2px
    style F3 fill:#fbf,stroke:#333,stroke-width:2px
    style G1 fill:#ffb,stroke:#333,stroke-width:2px
    style D4 fill:#fbb,stroke:#333,stroke-width:2px
    style E2 fill:#fbb,stroke:#333,stroke-width:2px
    style E3 fill:#fbb,stroke:#333,stroke-width:2px
    style F3 fill:#f9f,stroke:#333,stroke-width:2px
```

## Repository Structure

```
pgx-analysis/
├── 1_apcd_input_data/          # APCD data preprocessing and cleaning (bronze → silver → gold)
├── 2_create_cohort/            # Cohort creation and QA
├── 3_feature_importance/       # MC-CV feature importance screening
├── 4a_model_data/              # Model-ready event datasets (target vs control)
├── 4b_dtw_filter/              # DTW-based protocol filtering (creates model_events_no_protocols.parquet used by downstream feature engineering)
├── 5_pgx_analysis/            # Pharmacogenomics (PGx) feature engineering
├── 6_final_model_selection/    # Final model development and evaluation
├── 7_ffa_analysis/             # Step 8: Formal feature attribution (FFA) analysis (uses SHAP to prioritize rules)
├── 8_shap_analysis/            # Step 7: SHAP-based post‑model analysis (distributional, per-feature and per-patient)
├── 9_risk_dashboard/           # Step 9: Risk calculator + dashboard, API, and deployment artifacts (Lambda + S3)
├── py_helpers/                 # Shared Python helper utilities
├── r_helpers/                  # Shared R helper utilities
└── docs/                       # Documentation
```

## Project Components

### Core Analysis Modules

**📊 1_apcd_input_data: APCD Data Processing**
- `0_txt_to_parquet.py` - Convert text files to Parquet format
- `3_apcd_clean.py` - Main data cleaning script
- `3a_clean_pharmacy.py` - Pharmacy data cleaning
- `3b_clean_medical.py` - Medical data cleaning
- `drug_mappings/` - Drug name standardization mappings (A-Z + medical supplies)
- `claim_mappings/` - ICD code mappings and classifications

**👥 2_create_cohort: Cohort Creation**
- `0_create_cohort.py` - Main cohort creation pipeline (orchestrator)
- `2_step2_data_quality_qa.py` - Cohort quality assurance and validation
- `phases/` - Individual pipeline phase implementations
- `table_mappings/` - Table mapping configurations

**📈 3_feature_importance: Feature Screening**
- `feature_importance_mc_cv.ipynb` - Monte Carlo CV feature importance analysis
- `feature_importance_mc_cv.R` - R script for MC-CV analysis
- `create_visualizations.R` - Visualization utilities
- Uses three core models for robust feature ranking: **CatBoost**, **XGBoost (boosted trees)**, and **XGBoost RF mode**

**📊 4b_dtw_filter: DTW Protocol Filtering**
- `filter_protocol_events.py` - DTW-derived protocol filtering to create `model_events_no_protocols.parquet`
- `dtw_cohort_analysis.py` / `dtw_trajectory_analysis.py` - Optional sequence similarity and trajectory development
- Patient clustering, similarity scoring, and time-window audit artifacts

**🤖 6_final_model_selection: Final Model Development**
- `run_final_model.py` - Model training, selection, and export (CatBoost / XGBoost)
- Model outputs include trained models, feature importance, and evaluation metrics
- `catboost_models/` - Trained model artifacts and metadata

**🎯 7_ffa_analysis: Feature Attribution**
- `catboost_axp_explainer.py` - CatBoost AXP (Approximate Explanations) analysis
- `ffa_analysis.py` - Feature Filtering and Analysis pipeline
- Tree export and causal inference

### Pipeline Architecture (Summary)

The cohort analysis pipeline uses a **modular orchestrator/executor design** on top of a **partition‑first DuckDB layer**:

- `2_create_cohort/create_cohort.py` orchestrates phases; individual step modules and SQL files implement the work.  
- All heavy work runs per `(age_band, event_year)` partition with S3‑backed checkpoints, so jobs are resumable and easy to parallelize.

For full operator‑level details (worker counts, DuckDB configuration, checkpoint layout, and performance tuning), see:

- `docs/CrossStep_Development/README_data_pipeline_architecture.md`  
- `docs/CrossStep_Development/README_data_pipeline.md`  

**🛠️ py_helpers: Utility Functions**
- `common_imports.py` - Common import statements and configurations
- `constants.py` - Global constants and configuration values
- `duckdb_utils.py` - DuckDB database utilities
- `s3_utils.py` - S3 storage utilities
- `logging_utils.py` - Logging configuration and utilities
- Additional utility modules for data processing, model training, and visualization

## Data and Variables

- **Unit of analysis**: Patient-episode or encounter
- **Outcome (Y)**: Binary classification target (e.g., opioid dependence, ED visit)
- **Treatments (A)**: Drug exposure indicators
- **Covariates (X)**:
  - ICD diagnosis codes (grouped/rolled up)
  - CPT procedure codes
  - Demographics and baseline attributes
- **Temporal info**: Timestamps for diagnoses, procedures, and drug administrations

**Separation:**
- Pre-treatment covariates (for confounding control)
- Treatment variables (drugs)
- Post-treatment variables (mediators/outcomes)

## Recent Enhancements

### Drug Event Explosion Strategy
- **Patient-Level → Drug-Level Transformation**: Each drug prescription becomes a separate row
- **Context Duplication**: Patient demographics and clinical data duplicated per drug event
- **Sequence Modeling Ready**: Enables FpGrowth, bupaR, DTW, and symbolic reasoning analysis
- **Temporal Tracking**: Maintains `days_to_ade` and `days_to_opioid_ed` relationships

### Cohort Exclusivity Enforcement
- **OPIOID_ED Priority**: Processes opioid_ed cohort first
- **Mutual Exclusivity**: Ensures no patient appears in both cohorts
- **Quality Assurance**: Validates cohort separation and logs metrics
- **Data Integrity**: Prevents data leakage between cohorts

### DTW and BupaR Integration

**DTW (Dynamic Time Warping)** and **BupaR (Process Mining)** serve different but complementary purposes in temporal sequence analysis:

| Aspect | DTW | BupaR |
|--------|-----|-------|
| **Scope** | Pairwise sequence comparison | Process discovery across many cases |
| **Output** | Distance metric | Process maps, flow diagrams |
| **Abstraction** | Low-level (raw sequences) | High-level (process patterns) |
| **Scalability** | O(n²) for each pair | Handles thousands of cases |
| **Interpretability** | "These sequences are X% similar" | "80% of patients follow path A→B→C" |

## Related Documentation

- [`README_data_pipeline.md`](README_data_pipeline.md) - Data processing and cohort creation
- [`README_analysis_workflow.md`](README_analysis_workflow.md) - Feature importance and analysis workflow
- [`README_data_visualizations.md`](README_data_visualizations.md) - Visualization approaches
- [`README_create_cohort_pipeline.md`](README_create_cohort_pipeline.md) - Comprehensive cohort creation guide

