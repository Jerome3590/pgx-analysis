# Analysis Workflow

Feature importance, pattern mining, and final model development for the Prescription Drug Analysis pipeline.

## Overview

The analysis workflow implements a multi-stage approach to feature discovery, noise reduction, model development, and interpretation:

1. **Feature Screening** with three core models (CatBoost, XGBoost boosted trees, XGBoost RF mode) + Monte Carlo cross-validation  
2. **Feature Refinement (Feature Importance EDA)** using BupaR post-target analysis to filter and refine aggregated feature importances, producing `cohort_feature_importance` files
3. **Model Data Extraction** into `4a_model_data/` (target vs control event datasets) using refined features from Feature Importance EDA  
4. **Event Filtering (Step 4b)** in `4b_event_filter` to create `model_events_no_protocols.parquet` that is then used as the **preferred input for all downstream feature engineering steps (PGx)**. Filters administrative codes (from 0_icd_cpt_check) and post-event leakage (from 1_bupaR).  
5. **PGx Feature Engineering (Step 5)** via `5_pgx_analysis/` adding pharmacogenomics features  
6. **Final Model Development** in `6_final_model_selection/`
   - **6a_feature_encoding** – cohort- and age-band-specific feature lookup tables and numeric drug codebooks (saved under `feature_encoding_outputs/`).  
   - **6b_final_model_selection** – final feature assembly, Monte Carlo CV, model training/export, and FFA-friendly JSON export.  
7. **SHAP-Based Distributional Analysis** via `7_shap_analysis` (global + local SHAP for both XGBoost and CatBoost, aligned with the final model feature set). Must run before Step 8 since FFA uses SHAP importance to filter and prioritize rules.
8. **Post‑Model Structural Analysis** via FFA (`8_ffa_analysis`):
   - **XGBoost FFA only**: FFA analysis is performed only for XGBoost models
   - **CatBoost FFA**: NOT performed due to CatBoost's complex hashing and CTR (Counter-based Target Statistics) for categorical variables
   - **CatBoost SHAP**: Used for feature importance filtering in XGBoost FFA (not for CatBoost FFA rule extraction)
   - **Rule Selection**: Uses SHAP importance from both XGBoost and CatBoost (from Step 7) to filter and prioritize rules for AXP computation. Rule selection uses a three-set union: (1) first 100 matched rules, (2) random sample of 100 matched rules, and (3) top 300 SHAP-filtered rules (or all rules above 10th percentile, whichever is larger). Causal analysis filters the final rule set based on causal importance scores.
9. **Risk Calculator + Dashboard Deployment** via `9_risk_dashboard` (Lambda-ready model packages and S3-hosted UI).

## Phase 1: Monte Carlo CV + Feature Importance

**Goal**: Robust, model-agnostic feature ranking on noisy, high-dimensional data, using
strict temporal validation and a small, focused model ensemble.

**Process (implemented in `3_feature_importance/`, `3b_feature_importance_eda/`, and `4a_model_data/`):**
1. **Monte Carlo Cross-Validation (MC‑CV)**  
   - Train on **2016–2018** and evaluate on a strict **2019 holdout** (no leakage).  
   - Default **10 splits** for feature-importance screening; many more (≈1000) for the final model.
2. **Model Training (per split)**  
   - Fit three core tree models:
     - **CatBoost** (categorical boosting)  
     - **XGBoost** (boosted trees)  
     - **XGBoost RF mode** (random-forest style XGBoost)
3. **Feature Importance (per model)**  
   - Compute **gain/Gini** importance for all tree-based features (XGBoost / XGBoost RF) and flag
     features with `gain_importance > 0`.  
   - Compute **permutation importance on the full feature set** for all models, optionally capping
     rows via `PGX_PERM_MAX_ROWS`.
4. **Aggregation Across Splits and Models**  
   - Aggregate permutation scores per feature across MC‑CV splits.  
   - Normalize within each model, scale by model performance (recall or inverse logloss), then
     aggregate across models (including a rare-variant XGBoost pass when available).
5. **Feature Refinement (Feature Importance EDA - `3b_feature_importance_eda/`)**  
   - **BupaR Post-Target Analysis**: Use process mining to analyze sequences before and after target event to identify post-target leakage features in aggregated importances
   - **Code Research and Validation**: Research and identify non-informative administrative/scheduling codes (actual event-level filtering happens in Step 4b)
   - **Filter and Refine**: Remove post-target leakage features from aggregated feature importance list and generate refined `cohort_feature_importance` files
   - **Note**: This is NOT a DTW filter - it uses BupaR process mining and code research to filter already-processed aggregated feature importances
   - Output: `cohort_feature_importance.csv` files that feed into Step 4a
6. **Model Data Extraction (`4a_model_data/`)**  
   - Use the refined `cohort_feature_importance` files from Feature Importance EDA to drive `4a_model_data` extraction
   - If Feature Importance EDA files are missing, Step 4a will error (no fallback to aggregated importances)

**Cohort Focus Strategy (Phase 1):**

Because full MC‑CV + permutation importance is computationally intensive but critical for
health‑grade robustness, we focus the heaviest analysis on two cohort groups:

- **Cohort Group 1 – Opioid ED (`opioid_ed`)**  
  - Age bands: **\<65** (e.g., 0–12 through 55–64).  
  - Feature space for MC‑CV: **drugs + ICD codes + CPT codes + event type**.

- **Cohort Group 2 – Polypharmacy ED visits (`non_opioid_ed`)**  
  - Age bands: **≥65** (e.g., 65–74 through 95–114).  
  - Feature space for MC‑CV feature importance: **drugs only** (polypharmacy focus).

Other cohort/age-band combinations may be explored with lighter settings (fewer splits or
reduced model sets), but **interpretation for publication and downstream causal analysis
is anchored on these two cohort groups**.

### Current Modeling Plan (Cohort / Age-Band Grid)

For the current analysis run, we fit a **separate end‑to‑end model (Steps 3–9)** for each of the following
`(cohort, age_band)` combinations:

- **Cohort 1 – Opioid ED (`opioid_ed`)**
  - **0–12** – smoke test / pipeline verification cohort
  - **13–24**
  - **25–44**
  - **45–54**
  - **55–64**

- **Cohort 2 – Polypharmacy ED (`non_opioid_ed`)**
  - **65–74**
  - **75–84**
  - **85–94**

Each of these nine cells in the grid will have its own:
- `3_feature_importance` run (MC‑CV + aggregation)
- `4a_model_data` extraction (`model_events.parquet` for target and control)
- `5_*` feature‑engineering passes (PGx as applicable; FP‑Growth and BupaR are dashboard-only visualizations; DTW is used in Step 9 for dashboard visualizations only)
- `6_final_model` training + evaluation (one final model per `(cohort, age_band)`).

### Model Data Extraction (Target vs Control)

After feature importance is computed for each `(cohort, age_band)` pair, we create a compact
**model-ready event dataset** that downstream methods consume:

- **Target cohort (opioid_ed)**:
  - Read `*_cohort_feature_importance.csv` from Feature Importance EDA (REQUIRED - no fallback to aggregated importances)
  - Strip the `item_` prefix to recover raw drug / ICD / procedure codes.
  - For each age band and event year, filter GOLD cohort events to **only those rows where at least one important item appears** in:
    - `drug_name`
    - `primary_icd_diagnosis_code` through `nine_icd_diagnosis_code`
    - `procedure_code`
  - Write filtered events to:
    - `4a_model_data/cohort_name=opioid_ed/age_band={band}/model_events.parquet`

- **Control cohort (non_opioid_ed)**:
  - Load full, unfiltered control cohort events for the same age band and years.
  - Sample control patients to maintain an approximate **5:1 control:target person-level ratio**.
  - Keep **all events** for sampled control patients (no feature filtering).
  - Write to:
    - `4a_model_data/cohort_name=non_opioid_ed/age_band={band}/model_events.parquet`

These paired `model_events.parquet` files provide a consistent, size-controlled input for BupaR post-target analysis in Feature Importance EDA and downstream feature engineering.

## Phase 2: PGx Feature Engineering (Step 5)

**Goal**: Add pharmacogenomics (PGx) features to the model-ready dataset.

### Step 5: PGx Feature Engineering (`5_pgx_analysis/`)

- Pharmacogenomics (PGx) analysis on important drugs
- Drug–gene mapping and allele-frequency-based risk features
- Patient-level PGx feature tables joined by `mi_person_key`

**Output**: PGx features integrated into the model-ready dataset

**Note**: 
- **Feature Importance EDA**: Uses BupaR post-target analysis for feature refinement (not DTW)
- **Step 4b**: Uses code-based filtering (administrative codes from 0_icd_cpt_check + post-event leakage from 1_bupaR) - creates `model_events_no_protocols.parquet`
- **Step 9**: Risk dashboard visualizations (BupaR process mining, FP-Growth patterns, DTW trajectories - visualization only)

## Phase 3: Final Model Development (`6_final_model_selection/`)

**Goal**: Integrate features from all analysis methods into final prediction model.

**Process**:
1. **Feature Integration**: Combine feature-importance–filtered `4a_model_data` and PGx features into a single patient-level table (e.g. via `6_final_model_selection/run_final_model.py` for a given `(cohort, age_band)`).
2. **Feature Schema**: Unified patient-level feature matrix (~185-750 features)
3. **Model Training**: CatBoost and Random Forest on integrated features
4. **Model Evaluation**: Performance metrics and feature importance analysis

**Output**: Trained models with interpretable feature sets

**Location**: `6_final_model_selection/`

## Enhanced Analysis Workflow Architecture

### Core Components

**1. FP-Growth Pattern Mining Layer**
- Implements market basket analysis on medication sequences to identify initial feature importances
- Identifies co-occurring prescriptions using minimum support thresholds (default: 0.05 for initial pattern discovery)
- Discovers significant event patterns that feed into both:
  - BupaR process mining for temporal analysis
  - CatBoost models for predictive modeling
- Filters patterns based on:
  - Minimum support threshold
  - Pattern frequency in positive vs negative samples
  - Clinical relevance of co-occurring events

**2. BupaR Process Mining Engine**
- Uses FP-Growth identified patterns to construct event logs using `mi_person_key` as case identifier
- Performs temporal analysis through process maps and trace alignment
- Identifies hospitalization precursor patterns
- Calculates throughput times between drug administrations
- Validates patterns through:
  - Process conformance checking
  - Trace alignment analysis
  - Performance metrics evaluation

**3. CatBoost Predictive Modeling**
- Incorporates FP-Growth discovered patterns as network features
- Uses Formal Feature Attribution (FFA) for feature importance analysis
- Implements temporal cross-validation for cohort-based forecasting
- Validates feature importance through:
  - Cross-validation stability
  - Statistical significance testing
  - Clinical relevance assessment

**4. FFA-based Importance Ranking**
- Uses FFA to rank features by their importance in predicting hospitalization risk
- Identifies top K important features based on:
  - Support and coverage thresholds
  - Statistical significance testing
  - Class-specific importance rankings
  - Cross-validation stability

## DTW Usage: Protocol Filtering and Dashboard Visualizations

**DTW (Dynamic Time Warping)** is used in two distinct contexts:

### 1. DTW Protocol Filtering (Step 4b)

**Purpose**: Identify and filter administrative/protocol codes from event data

**Location**: `4b_event_filter/`

**Output**: `model_events_no_protocols.parquet` - Event data with administrative/protocol codes removed

**Use Cases:**
1. **Protocol Identification**: Identify standard care protocols that both targets and controls follow
2. **Administrative Code Filtering**: Remove non-predictive administrative/scheduling codes
3. **Data Cleaning**: Create clean event dataset for downstream feature engineering (PGx)

**Note**: This filtering happens at the event level, creating `model_events_no_protocols.parquet` that is used as the preferred input for all downstream feature engineering steps.

### 2. DTW Dashboard Visualizations (Step 9)

**Purpose**: Trajectory analysis and visualization for dashboard exploration

**Location**: `9_risk_dashboard/visualizations/dtw/`

**Use Cases:**
1. **Patient Clustering**: Group patients with similar drug exposure histories
2. **Trajectory Visualization**: Interactive dashboard visualization of patient trajectories
3. **Outlier Detection**: Identify patients with unusual drug sequences
4. **Exploratory Analysis**: Visual exploration of sequence patterns (visualization only, not used as model features)

**Note**: DTW trajectory analysis in Step 9 is for dashboard visualizations only - these are not used as model features.

### BupaR: Process Discovery and Pathway Analysis

**Purpose:** Discover common process flows and temporal patterns across patient populations.

**Use Cases:**
1. **Feature Importance EDA**: Post-target analysis to identify leakage features (Feature Importance EDA)
2. **Process Flow Discovery**: Identify common pathways from drug exposure to outcomes (Step 9 dashboard)
3. **Temporal Pattern Analysis**: Understand timing relationships between events
4. **Pathway Comparison**: Compare process flows between target and control groups
5. **Dashboard Visualizations**: Process mining visualizations for interactive exploration (Step 9)

## Analysis Pipeline Overview

```mermaid
flowchart TD
    subgraph "Step 3: Feature Importance"
        A[Monte Carlo CV] --> B[Aggregated Feature Importance<br/>CatBoost + XGBoost]
        B --> C[Top Features Selection]
    end
    
    subgraph "Feature Importance EDA: Feature Refinement"
        C --> C1[BupaR Post-Target Analysis<br/>Identify Post-Target Leakage]
        C1 --> C2[Code Research & Validation<br/>Administrative Codes]
        C2 --> C3[Refined Cohort Feature Importance<br/>cohort_feature_importance.csv]
    end
    
    subgraph "Step 4: Model Data & Filtering"
        C3 --> D[4a: Model Data Extraction<br/>Event-level Cases + Controls]
        D --> E[4b: Protocol Filtering<br/>Remove Administrative Codes]
    end
    
    subgraph "Step 5: PGx Feature Engineering"
        E --> F[PGx Feature Engineering<br/>Drug-Gene Mappings<br/>Allele Frequencies]
    end
    
    subgraph "Step 6: Final Model Training"
        F --> G[Feature Integration<br/>Refined Features + PGx]
        G --> H[CatBoost Training]
        G --> I[XGBoost Training]
        G --> I2[XGBoost RF Training]
        H --> J[Model Selection & Evaluation<br/>Best Model<br/>Recall + AUC-PR]
        I --> J
        I2 --> J
    end
    
    subgraph "Step 7-8: Post-Model Analysis"
        J --> K[7: SHAP Analysis<br/>CatBoost + XGBoost<br/>SHAP Values]
        J --> L[8: FFA Analysis<br/>XGBoost Only<br/>Formal Feature Attribution<br/>Uses SHAP to prioritize rules]
        K --> L
    end
    
    subgraph "Step 9: Risk Dashboard"
        L --> N[Risk Dashboard<br/>Model Deployment]
        K --> N
        N --> N1[Frontend Dashboard<br/>Risk Assessment + PGx Cards]
        N --> N2[Backend API<br/>Lambda Function]
        N --> N3[Dashboard Visuals:<br/>Causal Analysis + DTW +<br/>FP-Growth + BupaR]
        N1 --> N4[Production Deployment<br/>S3 + API Gateway + Lambda]
        N2 --> N4
        N3 --> N4
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
    style J fill:#fbb,stroke:#333,stroke-width:2px
    style K fill:#fbf,stroke:#333,stroke-width:2px    %% SHAP Analysis
    style L fill:#fbf,stroke:#333,stroke-width:2px    %% FFA Analysis
    style N fill:#ffb,stroke:#333,stroke-width:2px    %% Risk Dashboard
    style N1 fill:#ffb,stroke:#333,stroke-width:2px    %% Frontend
    style N2 fill:#ffb,stroke:#333,stroke-width:2px    %% Backend
    style N3 fill:#ffb,stroke:#333,stroke-width:2px    %% Visualizations
    style N4 fill:#ffb,stroke:#333,stroke-width:2px    %% Deployment
```

## Key Insights

| Question | Analysis Method | Insights |
|----------|----------------|-----------|
| What itemsets are most common? | FP-Growth (visualization) | Frequent co-occurrence patterns for exploratory analysis |
| How do itemsets play out temporally? | BupaR (visualization) | Process flows and sequences for clinical interpretation |
| Which itemsets drive model predictions? | XGBoost FFA + CatBoost SHAP | Risk-influential patterns (XGBoost FFA with CatBoost SHAP filtering) |
| Are process-dominant paths aligned with risk? | BupaR vs. FFA | Pattern alignment analysis (visualization complements causal analysis) |

## Lessons Learned

### Why FP-Growth, BupaR, and DTW Are Visualization-Only

**FP-Growth**:
- **Target Leakage Risk**: Patterns mined from combined target+control data can encode target-specific information
- **Direct Leakage**: Rules may include target codes (e.g., F1120) as consequents, creating perfect target leakage
- **Solution**: Use FP-Growth for visualization and exploratory analysis only, not as model features

**BupaR**:
- **Complexity vs. Benefit**: Process mining features add significant complexity without sufficient predictive benefit over aggregated feature importances
- **Value in Exploration**: BupaR visualizations provide valuable clinical insights into patient pathways
- **Solution**: Use BupaR in Feature Importance EDA for feature refinement (post-target analysis) and in Step 9 for dashboard visualizations, but not as model features

**DTW**:
- **Protocol Filtering**: DTW excels at identifying standard care protocols that both targets and controls follow
- **Non-Predictive**: These protocols are non-predictive by design (they're standard care)
- **Solution**: Use DTW for protocol filtering (Step 4b) and trajectory visualization (Step 9), but not as model features

### Why Aggregated Features Are Used Directly (No Encoding)

**Initial Approach**: Feature encoding step to convert categorical features to numeric codes.

**Final Approach**: Use aggregated feature importances directly from Step 3.

**Key Insights**:
- **MC-CV Already Filters**: The Monte Carlo cross-validation process already identifies and ranks the most important features
- **Reduced Complexity**: Eliminating encoding reduces pipeline complexity and potential sources of error
- **Maintains Predictive Power**: Aggregated importances capture the essential signals without encoding overhead
- **PGx Adds Value**: PGx features provide additional pharmacogenomic information that complements aggregated features

**Result**: Simpler, more maintainable pipeline with equivalent or better predictive performance.

### Why SHAP Filters FFA Rules

**Approach**: Use SHAP importance from Step 7 to filter and prioritize rules for FFA (Step 8) AXP computation.

**Key Insights**:
- **Rule Explosion**: Without filtering, FFA can generate thousands of rules, many of which are noise
- **SHAP as Quality Filter**: SHAP importance identifies features that actually contribute to predictions
- **Three-Set Union**: Rule selection uses union of (1) first 100 matched rules, (2) random sample of 100 matched rules, and (3) top 300 SHAP-filtered rules
- **Causal Filtering**: Final rule set further filtered based on causal importance scores

**Result**: Focused, high-quality rule set for causal analysis that balances comprehensiveness with interpretability.

### Why CatBoost FFA Is Not Performed

**Technical Limitation**: CatBoost's complex hashing and CTR (Counter-based Target Statistics) transformations make symbolic rule extraction difficult.

**Design Philosophy**: This limitation functions as a deliberate quality control mechanism.

**Key Insights**:
- **Model Agreement**: Requiring features to be detected by CatBoost (SHAP) AND describable by XGBoost (symbolic rules) filters out model-specific artifacts
- **Robustness**: If CatBoost finds a signal that XGBoost cannot replicate, it may be an artifact of CatBoost's specific encoding
- **Clinical Actionability**: Features that can't be translated to symbolic rules are too opaque for clinical decision-making

**Result**: Higher-confidence features in causal analysis, with explicit logical verification possible.

## Output Paths Summary

### Step 5: PGx Feature Engineering Output Paths

#### Local File Paths

**Prerequisite Files (Cohort-Level, Shared Across Age Bands):**
- `5_pgx_analysis/outputs/{cohort}/{cohort}_drug_gene_mappings.csv` - Drug-to-gene mappings (cohort-level)
- `5_pgx_analysis/outputs/{cohort}/{cohort}_allele_frequencies.csv` - Allele frequencies (cohort-level)

**Global Cache Files (Shared Across All Cohorts):**
- `5_pgx_analysis/outputs/global/pgx_drug_gene_mappings_global.csv` - Global drug-gene mapping cache
- `5_pgx_analysis/outputs/global/pgx_allele_frequencies_global.csv` - Global allele frequency cache

**Feature Files (Age-Band Specific):**
- `5_pgx_analysis/outputs/feature_engineering/pgx_features_{cohort}_{age_band}.csv` - Intermediate patient-level PGx features
- `5_pgx_analysis/outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv` - Final PGx features ready for model training
- `5_feature_engineering/feature_engineering_outputs/7_pgx/{cohort}/{age_band}/pgx_features_{cohort}_{age_band}.csv` - Mirrored intermediate features
- `5_feature_engineering/feature_engineering_outputs/7_pgx/{cohort}/{age_band}/pgx_added_features_{cohort}_{age_band}.csv` - Mirrored final features

#### S3 Output Paths

**Primary S3 Location (`gold/pgx_features/`):**
- `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/pgx_added_features_{cohort}_{age_band}.csv` - Final PGx features
- `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/pgx_features_{cohort}_{age_band}.csv` - Intermediate PGx features
- `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/{cohort}_drug_gene_mappings.csv` - Drug-gene mappings (age-band specific copy)
- `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/{cohort}_allele_frequencies.csv` - Allele frequencies (age-band specific copy)

**Global Cache S3 Paths:**
- `s3://pgxdatalake/gold/pgx_features/global/pgx_drug_gene_mappings_global.csv` - Global drug-gene mapping cache
- `s3://pgxdatalake/gold/pgx_features/global/pgx_allele_frequencies_global.csv` - Global allele frequency cache

**Checkpoints:**
- `s3://pgx-repository/pipeline_checkpoints/5_pgx_analysis/{cohort}/{age_band}/checkpoint.json` - Step 5 completion checkpoint
- `s3://pgx-repository/7_pgx_log/{cohort}/{age_band}/pgx_{cohort}_{age_band}.log` - Step 5 execution logs

#### File Naming Conventions

- **Age band format**: `{age_band}` uses hyphens (e.g., `13-24`)
- **Filename format**: `{age_band_fname}` uses underscores (e.g., `13_24`)
- **Cohort-level files**: Shared across all age bands for a given cohort
- **Global cache files**: Shared across all cohorts and age bands

#### Idempotency Checks

The workflow checks for existing outputs in this order:
1. **Local files**: Checks cohort-level, then global cache
2. **S3 files**: Checks global cache → cohort-level paths
3. **Checkpoints**: Checks `pgx-repository` bucket for completion checkpoints

If outputs exist, the step is skipped. To force regeneration, use:
```bash
python utility_scripts/clear_pgx_step5_outputs.py --cohort {cohort} --age-band {age_band} --clear-local --clear-prerequisites
```

## Model Artifacts and Storage Structure

The pipeline generates and stores various model artifacts for each cohort and age band:

### Model Artifacts Structure
All model artifacts are stored in S3 with the following partition structure:
```
s3://{S3_BUCKET}/{artifact_type}/cohort_name={cohort}/age_band={band}/event_year={year}/
```

### Artifact Types and Contents

1. **Model Metrics and Info**
   - `model_metrics.json`: Performance metrics (AUC, accuracy, F1, precision, recall, Brier score, log loss)
   - `model_info.json`: Model metadata, feature names, and native feature importances

2. **SHAP Analysis**
   - `shap_values.parquet`: Raw SHAP values for feature importance analysis
   - `shap_plots/`: Directory containing SHAP value visualization plots for each class

3. **Cattail Analysis**
   - `cattail_plots/`: Directory containing Cattail distribution plots showing feature value distributions

4. **Causal Analysis**
   - `causal_summary.json`: Causal analysis results including feature effects and summary statistics

5. **Calibration Analysis**
   - `calibration_plots/`: Directory containing model calibration curves for each class

6. **Mirror Plots**
   - `mirror_plots/`: Directory containing feature importance mirror plots comparing classes

## Related Documentation

- [`README_overview.md`](README_overview.md) - Project structure and components
- [`README_data_pipeline.md`](README_data_pipeline.md) - Data processing and cohort creation
- [`README_data_visualizations.md`](README_data_visualizations.md) - Visualization approaches
- [`docs/README_feature_importance.md`](docs/README_feature_importance.md) - Feature importance analysis
- [`docs/README_fpgrowth.md`](docs/README_fpgrowth.md) - FP-Growth pattern mining
- [`docs/README_bupaR.md`](docs/README_bupaR.md) - Process mining with BupaR
- [`docs/README_dtw_feature_extraction.md`](docs/README_dtw_feature_extraction.md) - DTW trajectory analysis
- [`docs/README_final_model.md`](docs/README_final_model.md) - Final model development

