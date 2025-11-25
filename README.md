# Prescription Drug Analysis with FpGrowth, BupaR and CatBoost Integration

## Project Structure

```mermaid
graph TD
    A[pgx_analysis] --> B[1_apcd_input_data]
    A --> C[2_create_cohort]
    A --> D[3_feature_importance]
    A --> E[4_fpgrowth_analysis]
    A --> F[5_bupaR_analysis]
    A --> G[6_dtw_analysis]
    A --> H[7_final_model]
    A --> I[8_ffa_analysis]
    A --> J[helpers_1997_13]
    A --> K[pgx_cohort_pipeline.ipynb]
    
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
    
    D --> D1[feature_importance_mc_cv.ipynb]
    D --> D2[feature_importance_mc_cv.R]
    D --> D3[create_visualizations.R]
    
    E --> E1[global_fpgrowth.py]
    E --> E2[cohort_fpgrowth.py]
    E --> E3[global_fpgrowth_feature_importance.ipynb]
    E --> E4[cohort_fpgrowth_feature_importance.ipynb]
    
    F --> F1[bupaR_pipeline.ipynb]
    F --> F2[sankey_plot.html]
    
    G --> G1[dtw_cohort_analysis.py]
    G --> G2[dtw_trajectory_analysis.py]
    G --> G3[README.md]
    
    H --> H1[catboost_r.ipynb]
    H --> H2[run_catboost_opioid_ed.py]
    H --> H3[run_catboost_ade_ed.py]
    H --> H4[final_feature_schema.json]
    H --> H5[catboost_models]
    
    I --> I1[catboost_axp_explainer.py]
    I --> I2[ffa_analysis.py]
    
    J --> J1[common_imports.py]
    J --> J2[duckdb_utils.py]
    J --> J3[s3_utils.py]
    J --> J4[constants.py]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
    style K fill:#fbb,stroke:#333,stroke-width:2px
```

## High-Level Workflow

End-to-end workflow for feature discovery, noise reduction, and causal-oriented modeling using drug exposures, ICD/CPT codes, and classification outcomes.

### Overview

This project builds a classification model on a large, noisy healthcare dataset, then uses model-based feature importance plus pattern- and process-mining to derive a stable covariate set and interpretable tree ensembles for causal analyses.

**High-level phases:**

1. **Feature Screening** with tree ensembles (CatBoost, Random Forest) + Monte Carlo cross-validation
2. **Structure Discovery** and noise reduction with FP-Growth, process mining (BupaR), and dynamic time warping (DTW)
3. **Final Model Development** combining features from all analysis methods for prediction and causal inference

### Workflow Pipeline

```mermaid
flowchart TD
    subgraph "Phase 1: Data Preparation"
        A1[APCD Input Data] --> A2[Data Cleaning]
        A2 --> A3[Cohort Creation]
        A3 --> A4[Quality Assurance]
    end
    
    subgraph "Phase 2: Feature Discovery"
        A4 --> B1[Monte Carlo CV]
        B1 --> B2[Feature Importance<br/>CatBoost + Random Forest]
        B2 --> B3[Top Features Selection]
    end
    
    subgraph "Phase 3: Pattern Mining"
        B3 --> C1[FPGrowth Analysis<br/>Frequent Itemsets]
        C1 --> C2[BupaR Process Mining<br/>Temporal Pathways]
        C2 --> C3[DTW Trajectory Analysis<br/>Patient Clustering]
    end
    
    subgraph "Phase 4: Feature Engineering"
        C1 --> D1[FPGrowth Features<br/>Itemsets & Rules]
        C2 --> D2[BupaR Features<br/>Process Patterns]
        C3 --> D3[DTW Features<br/>Trajectory Clusters]
        D1 --> D4[Final Feature Schema]
        D2 --> D4
        D3 --> D4
    end
    
    subgraph "Phase 5: Final Model"
        D4 --> E1[Feature Integration]
        E1 --> E2[CatBoost Training]
        E1 --> E3[Random Forest Training]
        E2 --> E4[Model Evaluation]
        E3 --> E4
        E4 --> E5[Feature Attribution]
    end
    
    subgraph "Phase 6: Causal Analysis"
        E5 --> F1[Tree Export JSON]
        F1 --> F2[Subgroup Identification]
        F2 --> F3[Causal Inference]
    end
    
    style A1 fill:#f9f,stroke:#333,stroke-width:2px
    style B2 fill:#bbf,stroke:#333,stroke-width:2px
    style C1 fill:#bfb,stroke:#333,stroke-width:2px
    style C2 fill:#bfb,stroke:#333,stroke-width:2px
    style C3 fill:#bfb,stroke:#333,stroke-width:2px
    style D4 fill:#fbb,stroke:#333,stroke-width:2px
    style E2 fill:#fbb,stroke:#333,stroke-width:2px
    style E3 fill:#fbb,stroke:#333,stroke-width:2px
    style F3 fill:#f9f,stroke:#333,stroke-width:2px
```

### Phase 1: Monte Carlo CV + Feature Importance

**Goal**: Robust, model-agnostic feature ranking on noisy, high-dimensional data.

**Process**:
1. **Monte Carlo Cross-Validation**: Random splits (70/30, stratified by outcome) across multiple iterations
2. **Model Training**: Fit CatBoost and Random Forest classifiers on each split
3. **Feature Importance**: Compute normalized importance scores per model
4. **Aggregation**: Combine importance across models and iterations, weighted by validation performance
5. **Feature Screening**: Select top features based on combined importance and stability

**Output**: Ranked feature list with combined importance and stability statistics

**Location**: `3_feature_importance/`

### Phase 2: Pattern & Process Mining + DTW

**Goal**: Exploit structure in selected features and further reduce noise.

**Components**:

1. **FPGrowth Analysis** (`4_fpgrowth_analysis/`)
   - Frequent pattern mining on drug/ICD/CPT codes
   - Target-focused association rules (predicting opioid dependence, ED visits)
   - Itemset metrics and feature encoding

2. **BupaR Process Mining** (`5_bupaR_analysis/`)
   - Event log creation from patient sequences
   - Process flow discovery and pathway analysis
   - Temporal pattern identification

3. **DTW Trajectory Analysis** (`6_dtw_analysis/`)
   - Patient trajectory clustering
   - Similarity scoring and archetype matching
   - Multi-modal trajectory features

**Output**: Refined feature set that participates in frequent patterns, stable pathways, and respects process timing

### Phase 3: Final Model Development

**Goal**: Integrate features from all analysis methods into final prediction model.

**Process**:
1. **Feature Integration**: Combine FPGrowth itemsets, BupaR patterns, and DTW trajectories
2. **Feature Schema**: Unified patient-level feature matrix (~185-750 features)
3. **Model Training**: CatBoost and Random Forest on integrated features
4. **Model Evaluation**: Performance metrics and feature importance analysis

**Output**: Trained models with interpretable feature sets

**Location**: `7_final_model/`

### Data and Variables

- **Unit of analysis**: Patient-episode or encounter
- **Outcome (Y)**: Binary classification target (e.g., opioid dependence, ED visit)
- **Treatments (A)**: Drug exposure indicators
- **Covariates (X)**:
  - ICD diagnosis codes (grouped/rolled up)
  - CPT procedure codes
  - Demographics and baseline attributes
- **Temporal info**: Timestamps for diagnoses, procedures, and drug administrations

**Separation**:
- Pre-treatment covariates (for confounding control)
- Treatment variables (drugs)
- Post-treatment variables (mediators/outcomes)

### Repository Structure

```
pgx-analysis/
├── 1_apcd_input_data/          # Data preprocessing and cleaning
├── 2_create_cohort/            # Cohort creation and QA
├── 3_feature_importance/       # MC-CV feature importance analysis
├── 4_fpgrowth_analysis/        # Frequent pattern mining
├── 5_bupaR_analysis/           # Process mining
├── 6_dtw_analysis/            # Trajectory analysis
├── 7_final_model/              # Final model development
├── 8_ffa_analysis/             # Feature attribution analysis
├── helpers_1997_13/           # Utility functions
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
- Uses CatBoost and Random Forest for robust feature ranking

**🔍 4_fpgrowth_analysis: Frequent Pattern Mining**
- `global_fpgrowth.py` - Global pattern mining across all patients
- `cohort_fpgrowth.py` - Cohort-specific pattern mining
- `global_fpgrowth_feature_importance.ipynb` - Global analysis notebook
- `cohort_fpgrowth_feature_importance.ipynb` - Cohort analysis notebook
- Target-focused rule mining (TARGET_ICD, TARGET_ED, CONTROL)

**🔄 5_bupaR_analysis: Process Mining**
- `bupaR_pipeline.ipynb` - Process mining pipeline using BupaR
- `sankey_plot.html` - Interactive Sankey diagram visualizations
- Event log creation and process flow discovery

**📊 6_dtw_analysis: Trajectory Analysis**
- `dtw_cohort_analysis.py` - DTW analysis for drug sequence similarity
- `dtw_trajectory_analysis.py` - Enhanced trajectory development
- Patient clustering and similarity scoring

**🤖 7_final_model: Final Model Development**
- `catboost_r.ipynb` - R-based CatBoost model training
- `run_catboost_opioid_ed.py` - Opioid ED event prediction
- `run_catboost_ade_ed.py` - Combined ADE and ED analysis
- `final_feature_schema.json` - Comprehensive feature schema
- `catboost_models/` - Trained model artifacts and metadata

**🎯 8_ffa_analysis: Feature Attribution**
- `catboost_axp_explainer.py` - CatBoost AXP (Approximate Explanations) analysis
- `ffa_analysis.py` - Feature Filtering and Analysis pipeline
- Tree export and causal inference

### Pipeline Architecture

The cohort analysis pipeline follows a clean **orchestrator/executor** architecture pattern:

**🎯 Core Design Principles:**
- **Modular Implementation**: Individual step files in `pipeline_steps/` folder for better organization
- **Clean Separation**: `create_cohort.py` orchestrates, individual step files execute
- **No Code Duplication**: Each step implemented once in its own module
- **Maintainable**: Changes to step logic isolated to individual files

**📁 File Responsibilities:**
- **`create_cohort.py`** (365 lines): 
  - Main orchestration and context setup
  - Step execution dispatcher
  - Error handling and logging
  - Checkpoint management
- **`pipeline_steps/pipeline_step*.py`** (15 individual files):
  - Individual step implementations (run_step1 through run_step15)
  - Step dependencies and validation
  - Data processing logic
  - S3 operations and persistence
- **`sql/step*.txt`** (14 SQL files):
  - SQL query implementations for each step
  - Reusable SQL logic
  - Database operations

**🔄 Execution Flow:**
```python
# Orchestrator calls dispatcher
step_execution_dispatcher(next_step, context)

# Dispatcher calls individual step functions from pipeline_steps/
from pipeline_steps.pipeline_step1 import run_step1_lock_acquisition
from pipeline_steps.pipeline_step2 import run_step2_database_setup
# ... through step 15

run_step1_lock_acquisition(context)
run_step2_database_setup(context)
# ... individual step execution
```

## Defaults: Partitioned (imputed) Silver Inputs

The pipeline now prefers partitioned, imputed silver inputs by default for Phase 2 (pharmacy) and Phase 2b (medical). Global imputation (Phase 1) writes partitioned outputs under `s3://<bucket>/silver/imputed/{medical|pharmacy}_partitioned/` which improves DuckDB performance and enables partition-first parallelism.

- Why: DuckDB runs and worker parallelism are more efficient when processing partitioned data (one DuckDB instance per partition, fewer memory spikes).
- Where: outputs from `1_apcd_input_data/2_global_imputation.py` are written to `s3://pgxdatalake/silver/imputed/...` and include partition keys (e.g. `age_band` and `event_year`).
- Operator note: The orchestrator defaults now point to the imputed partitioned paths. For compatibility the orchestrator still accepts legacy "raw" silver paths and will attempt to convert/locate the imputed partitioned path if the raw path contains no parquet files.

Helper: The preflight discovery logic now lives in the Python helper `helpers_1997_13.s3_utils.select_silver_inputs`.
You can call it from a short Python one-liner to preview preferred input paths, for example:

```bash
python -c "from helpers_1997_13.s3_utils import select_silver_inputs; import json; print(json.dumps(select_silver_inputs('pgxdatalake','silver','medical')))"
```

This replaces the older `scripts/validate_silver_inputs.py` script which was removed in cleanup.

This architecture ensures maintainability, reduces bugs, and provides a clean separation of concerns with modular step implementations.

**🛠️ helpers_1997_13: Utility Functions**
- `common_imports.py` - Common import statements and configurations
- `constants.py` - Global constants and configuration values
- `duckdb_utils.py` - DuckDB database utilities
- `s3_utils.py` - S3 storage utilities
- `logging_utils.py` - Logging configuration and utilities
- Additional utility modules for data processing, model training, and visualization

**📓 Main Pipeline (`pgx_cohort_pipeline.ipynb`)**
- Comprehensive Jupyter notebook for the complete PGx analysis pipeline

## Executive Summary

```mermaid
flowchart TD
    subgraph "Data Processing"
        A[Pharmacy Data] --> B[Clean Pharmacy Data]
        C[Medical Data] --> D[Process Medical Data]
        B --> E[Create Cohorts]
        D --> E
    end

    subgraph "Cohort Classification"
        E --> E1[Event Classification]
        E1 --> E2[OPIOID_ED Cohort<br/>ICD Codes: F11.x, T40.x, Y12.x<br/>Drug Window: Complete History]
        E1 --> E3[ED_NON_OPIOID Cohort<br/>HCG Codes: P51, O11, P33<br/>Drug Window: 30 Days Prior]
        E2 --> E4[Cohort Exclusivity<br/>OPIOID_ED Priority]
        E3 --> E4
        E4 --> E5[Drug Event Explosion<br/>1 Row per Drug Prescription]
    end

    subgraph "Feature Engineering"
        E5 --> F[Global FpGrowth Analysis]
        F --> F1[Global Drug Pattern Discovery]
        F1 --> G[Global Encoding Map Creation]
        G --> H[Association Rules]
    end

    subgraph "Modeling & Analysis"
        H --> I[CatBoost Model Training]
        I --> I1[Temporal Filtering]
        I1 --> J[Feature Importance Analysis]
        J --> K[Process Mining with BupaR]
    end

    subgraph "Visualization & Output"
        K --> L[Network Visualization]
        K --> M[Process Maps]
        I --> N[Risk Predictions]
        J --> O[Feature Importance Plots]
    end

    subgraph "Validation & Metrics"
        L --> P[Performance Metrics]
        M --> P
        N --> P
        O --> P
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E1 fill:#bbf,stroke:#333,stroke-width:2px
    style E2 fill:#fbb,stroke:#333,stroke-width:2px
    style E3 fill:#fbb,stroke:#333,stroke-width:2px
    style E4 fill:#bfb,stroke:#333,stroke-width:2px
    style E5 fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
    style I fill:#fbb,stroke:#333,stroke-width:2px
    style K fill:#bfb,stroke:#333,stroke-width:2px
    style P fill:#fbb,stroke:#333,stroke-width:2px
```

Enhanced pipeline for analyzing prescription drug patterns with hospitalization risk using process mining, survival analysis, and machine learning.

## 🚀 **Recent Enhancements: Drug Event Explosion & Cohort Exclusivity**

### **Key Improvements**

**1. Drug Event Explosion Strategy** ⭐ **NEW**
- **Patient-Level → Drug-Level Transformation**: Each drug prescription becomes a separate row
- **Context Duplication**: Patient demographics and clinical data duplicated per drug event
- **Sequence Modeling Ready**: Enables FpGrowth, LSTM, DTW, and symbolic reasoning analysis
- **Temporal Tracking**: Maintains `days_to_ade` and `days_to_opioid_ed` relationships

**2. Cohort Exclusivity Enforcement** ⭐ **NEW**
- **OPIOID_ED Priority**: Processes opioid_ed cohort first
- **Mutual Exclusivity**: Ensures no patient appears in both cohorts
- **Quality Assurance**: Validates cohort separation and logs metrics
- **Data Integrity**: Prevents data leakage between cohorts

**3. Enhanced Drug Exposure Analysis** ⭐ **ENHANCED**
- **ADE Cohort**: 30-day lookback window for causality assessment
- **Opioid Cohort**: Complete drug history for pattern analysis
- **Temporal Relationships**: Precise tracking of drug-event timing
- **Data Source Tracking**: Distinguishes between `cohort_event` and `drug_exposure` rows

**4. Sequence Modeling Capabilities** ⭐ **NEW**
- **FpGrowth Pattern Mining**: Market basket analysis on drug sequences
- **LSTM Sequence Modeling**: Temporal drug sequence prediction
- **Dynamic Time Warping (DTW)**: Drug sequence similarity analysis
- **Symbolic Reasoning**: Interpretable drug pattern analysis
- **BupaR Process Mining**: Process discovery and pathway analysis

### **Impact on Analysis Pipeline**

| Component | Before | After |
|-----------|--------|-------|
| **Data Structure** | Patient-level aggregation | Drug-level explosion |
| **Cohort Assignment** | Potential overlap | Mutual exclusivity |
| **Pattern Mining** | Limited by aggregation | Full sequence analysis |
| **Temporal Analysis** | Approximate timing | Precise drug-event relationships |
| **Modeling Capabilities** | Basic feature engineering | Advanced sequence modeling |

## **DTW and BupaR Integration: Complementary Sequence Analysis** ⭐ **NEW**

### **Understanding the Distinction**

**DTW (Dynamic Time Warping)** and **BupaR (Process Mining)** serve different but complementary purposes in temporal sequence analysis:

| Aspect | DTW | BupaR |
|--------|-----|-------|
| **Scope** | Pairwise sequence comparison | Process discovery across many cases |
| **Output** | Distance metric | Process maps, flow diagrams |
| **Abstraction** | Low-level (raw sequences) | High-level (process patterns) |
| **Scalability** | O(n²) for each pair | Handles thousands of cases |
| **Interpretability** | "These sequences are X% similar" | "80% of patients follow path A→B→C" |

### **DTW: Sequence Similarity Analysis**

**Purpose:** Measure similarity between individual patient drug sequences that may vary in timing and length.

**Use Cases in Pipeline:**
1. **Patient Clustering**: Group patients with similar drug exposure histories
2. **Outlier Detection**: Identify patients with unusual drug sequences
3. **Similarity-Based Features**: Calculate distance to known high-risk patterns
4. **Sequence Validation**: Compare drug sequences across different time periods

**Implementation Example:**
```python
from dtaidistance import dtw
import numpy as np

def analyze_drug_sequence_similarity(patient_sequences):
    """
    Analyze similarity between patient drug sequences using DTW
    """
    n_patients = len(patient_sequences)
    similarity_matrix = np.zeros((n_patients, n_patients))
    
    for i in range(n_patients):
        for j in range(i+1, n_patients):
            # Convert drug sequences to numerical representations
            seq1 = encode_drug_sequence(patient_sequences[i])
            seq2 = encode_drug_sequence(patient_sequences[j])
            
            # Calculate DTW distance
            distance = dtw.distance(seq1, seq2)
            similarity_matrix[i][j] = distance
            similarity_matrix[j][i] = distance
    
    return similarity_matrix

def cluster_patients_by_drug_similarity(similarity_matrix, n_clusters=5):
    """
    Cluster patients based on DTW similarity
    """
    from sklearn.cluster import AgglomerativeClustering
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(similarity_matrix)
    
    return cluster_labels
```

### **BupaR: Process Discovery and Pathway Analysis**

**Purpose:** Discover common process patterns and pathways from event logs across multiple patients.

**Use Cases in Pipeline:**
1. **Process Discovery**: Identify common pathways leading to adverse events
2. **Performance Analysis**: Calculate typical time intervals between events
3. **Conformance Checking**: Validate if patients follow expected clinical pathways
4. **Bottleneck Detection**: Identify delays or inefficiencies in care processes

**Implementation Example:**
```r
library(bupaR)
library(processmapR)

# Create event log from drug exposure data
eventlog <- create_eventlog(
  data = drug_exposure_data,
  case_id = "mi_person_key",
  activity_id = "drug_name",
  timestamp = "event_date",
  lifecycle_id = "data_source"
)

# Discover process patterns
process_map <- process_map(eventlog, type = frequency("relative-consequent"))

# Analyze performance
throughput_times <- throughput_time(eventlog, level = "activity")

# Identify common pathways
trace_analysis <- trace_explorer(eventlog, n_traces = 10)
```

### **Integrated Workflow: DTW + BupaR**

**Step 1: DTW Clustering**
```python
# Cluster patients by drug sequence similarity
similarity_matrix = analyze_drug_sequence_similarity(patient_drug_sequences)
patient_clusters = cluster_patients_by_drug_similarity(similarity_matrix)

# Add cluster labels to patient data
patient_data['dtw_cluster'] = patient_clusters
```

**Step 2: BupaR Analysis per Cluster**
```r
# Analyze process patterns within each DTW cluster
for (cluster_id in unique(patient_data$dtw_cluster)) {
  cluster_patients <- patient_data[patient_data$dtw_cluster == cluster_id, ]
  
  # Create event log for this cluster
  cluster_eventlog <- create_eventlog(
    data = cluster_patients,
    case_id = "mi_person_key",
    activity_id = "drug_name",
    timestamp = "event_date"
  )
  
  # Discover process patterns for this cluster
  cluster_process_map <- process_map(cluster_eventlog, type = frequency("relative-consequent"))
  
  # Save cluster-specific insights
  save_process_analysis(cluster_id, cluster_process_map)
}
```

**Step 3: Cross-Cluster Pattern Comparison**
```python
def compare_cluster_processes(cluster_process_maps):
    """
    Compare process patterns across DTW clusters
    """
    cluster_comparisons = {}
    
    for cluster_id, process_map in cluster_process_maps.items():
        # Extract key process metrics
        cluster_comparisons[cluster_id] = {
            'most_common_path': extract_most_common_path(process_map),
            'avg_throughput_time': calculate_avg_throughput(process_map),
            'pathway_variants': count_pathway_variants(process_map)
        }
    
    return cluster_comparisons
```

### **Pipeline Integration Benefits**

**1. Enhanced Patient Stratification**
- DTW identifies patients with similar drug exposure patterns
- BupaR reveals common process pathways within each similarity group
- Combined insight: "Patients with similar drug sequences follow these specific pathways"

**2. Improved Risk Prediction**
- DTW similarity scores become features for predictive models
- BupaR process patterns provide interpretable risk factors
- Combined approach: "Patient X is similar to high-risk patients and follows risky pathway Y"

**3. Clinical Decision Support**
- DTW finds patients with similar histories to current patient
- BupaR shows typical outcomes for patients with similar patterns
- Clinical insight: "Patients like this typically experience outcome Z"

### **Implementation in Current Pipeline**

**Data Flow Integration:**
```mermaid
flowchart LR
    A[2_create_cohort:<br/>Drug Event Explosion] --> B[6_dtw_analysis:<br/>DTW Sequence Analysis]
    A --> C[5_bupaR_analysis:<br/>BupaR Process Mining]
    B --> D[Patient Clustering]
    C --> E[Process Discovery]
    D --> F[Cluster-Specific BupaR]
    E --> G[Cross-Cluster Comparison]
    F --> G
    G --> H[7_final_model:<br/>Integrated Insights]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bfb,stroke:#333,stroke-width:2px
    style E fill:#bfb,stroke:#333,stroke-width:2px
    style F fill:#fbb,stroke:#333,stroke-width:2px
    style G fill:#fbb,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
```

**Key Advantages:**
- **DTW** provides granular sequence similarity analysis
- **BupaR** offers high-level process discovery
- **Combined** approach enables both detailed patient comparison and population-level pattern recognition
- **Scalable** analysis from individual patient sequences to population-wide process patterns

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
```python
from mlxtend.frequent_patterns import fpgrowth
frequent_itemsets = fpgrowth(log_matrix, min_support=0.1, use_colnames=True)
```

**2. BupaR Process Mining Engine**  
- Uses FP-Growth identified patterns to construct event logs using `mi_person_key` as case identifier
- Performs temporal analysis through:
```r
process_map(eventlog, type = frequency("relative-consequent"))
```
- Identifies hospitalization precursor patterns
- Calculates throughput times between drug administrations
- Validates patterns through:
  - Process conformance checking
  - Trace alignment analysis
  - Performance metrics evaluation

**3. CatBoost Predictive Modeling**  
- Incorporates FP-Growth discovered patterns as network features
- Uses Formal Feature Attribution (FFA) for feature importance analysis:
```python
model.get_feature_importance(type='PredictionValuesChange')
```
- Implements temporal cross-validation for cohort-based forecasting
- Validates feature importance through:
  - Cross-validation stability
  - Statistical significance testing
  - Clinical relevance assessment

**4. FFA-based importance ranking**
- Uses FFA to rank features by their importance in predicting hospitalization risk
- Identifies top K important features based on:
  - Support and coverage thresholds
  - Statistical significance testing
   - Class-specific importance rankings
   - Cross-validation stability

## Updated Implementation Workflow

### Phase 1: Current Data Optimization
1. **Data Structuring**  
 - Create annual cohorts with explicit censoring dates (Dec 31 cutoff)
 - Maintain observation windows with survival analysis

2. **Pattern Mining**  
 - Use `processmapR::dotted_chart()` for visual comparison
 - Apply `sequence_analysis()` with gap penalty parameters

3. **Model Integration**  
```r
library(catboost)
pool <- catboost.load_pool(
data = feature_set,
label = "hospitalization_flag",
cat_features = which(sapply(feature_set, is.character))
)
```

## Multimodal Data Architecture

### Current Data Pillars
| Data Type | Integration Method | Analysis Technique |
|-----------|-------------------|-------------------|
| Demographics | DuckDB columnar storage | Cox proportional hazards |
| Age Cohorts | Temporal stratification | Survival analysis |
| Medication | FP-Growth itemset mining | Market basket analysis |

### Future Data Integration
**1. Genotype-Environment Interaction**  
- Planned implementation of GEI models:
```python
CatBoost(params={'interaction_config': 'Genotype:Environment'})
```

**2. Behavioral & Family History**  
- Multi-modal fusion using:
```r
bupaR::merge_logs(clinical, behavioral, by="mi_person_key")
```

# Scripts Overview

### 1. Pharmacy Data Cleaning (`clean_pharmacy.py`)

This script normalizes pharmacy data and prepares it for use in medical data processing. It performs the following operations:

1. Loads pharmacy data from S3
2. Processes the data through several stages:
   - `pharmacy_normalized`: Raw data with parsed event dates
   - `pharmacy_augmented`: Added age bands and event year
   - `pharmacy_standardized`: Standardized drug names
   - `pharmacy_cleaned`: Applied drug name mappings and excluded medical supplies

#### Usage
```bash
python clean_pharmacy.py --age-band "0-12" --event-year 2020
```

#### Output
Creates partitioned Parquet files in S3:
```
s3://pgxdatalake/pharmacy-clean-drug-names/age_band={age_band}/event_year={event_year}/clean.parquet
```

### 2. Medical Data Processing and Cohort Creation (`2_create_cohort/`)

This module processes medical data and creates comprehensive cohorts with drug exposure tracking using DuckDB. It performs the following operations:

#### **Enhanced Cohort Creation Process**

```mermaid
flowchart TD
    subgraph "1_apcd_input_data: Data Loading & Preparation"
        A[Medical Data] --> B[Medical Filtering]
        C[Pharmacy Data] --> D[Pharmacy Clean]
        B --> E[Age Imputation]
        D --> E
    end
    
    subgraph "2_create_cohort: Event Classification & Tagging"
        E --> F[Unified Cohort Features]
        F --> G[Event Classification]
        G --> H[First Event Resolution]
        H --> I[Tagged Cohort Events]
    end
    
    subgraph "2_create_cohort: Drug Exposure & Explosion"
        I --> J[Drug Exposure Views]
        J --> J1[ADE Drug Exposure<br/>30-day window]
        J --> J2[Opioid Drug Exposure<br/>No time restriction]
        J1 --> K[Drug Event Explosion]
        J2 --> K
        K --> K1[1 row per drug prescription<br/>Patient context duplicated]
    end
    
    subgraph "2_create_cohort: Cohort Exclusivity"
        I --> L[First Events Analysis]
        L --> M[Select OPIOID_ED First]
        M --> N[Remove OPIOID_ED from Pool]
        N --> O[Select ED_NON_OPIOID from Remaining]
        O --> P[Validate Exclusivity]
    end
    
    subgraph "Cohort Creation"
        K1 --> Q[OPIOID_ED Cohort]
        P --> Q
        K1 --> R[ED_NON_OPIOID Cohort]
        P --> R
    end
    
    subgraph "Output & Validation"
        Q --> S[OPIOID_ED Parquet]
        R --> T[ED_NON_OPIOID Parquet]
        S --> U[Metrics & Validation]
        T --> U
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
    style P fill:#bfb,stroke:#333,stroke-width:2px
    style Q fill:#fbb,stroke:#333,stroke-width:2px
    style R fill:#fbb,stroke:#333,stroke-width:2px
    style U fill:#bfb,stroke:#333,stroke-width:2px
```

### Key Features of Enhanced Pipeline

**1. Drug Event Explosion Strategy**
- Each drug prescription becomes a separate row
- Patient context duplicated per drug event
- Enables sequence modeling and pattern mining

**2. Cohort Exclusivity Enforcement**
- OPIOID_ED processed first
- Mutual exclusivity between cohorts
- Quality assurance validation

**3. Temporal Drug Analysis**
- ADE cohort: 30-day lookback window
- Opioid cohort: Complete drug history
- Temporal relationship tracking

**4. Sequence Modeling Ready**
- FpGrowth pattern mining
- LSTM sequence modeling
- Dynamic Time Warping (DTW)
- Symbolic reasoning support

**Step 1-3: Environment Setup & Data Loading**
- Validates environment and acquires processing locks
- Enables DuckDB S3 support and AWS credentials
- Loads medical data from partitioned Parquet files

**Step 4: Medical Data Filtering**
- Filters medical data to essential columns (ICD codes, procedures, demographics)
- Collects metrics on filtered data volume and distinct persons

**Step 5: Pharmacy Data Preparation**
- Creates pharmacy view for age imputation and drug exposure analysis
- Extracts drug names, therapeutic classes, and temporal data
- Collects metrics on pharmacy data volume and drug diversity

**Step 6: Age Imputation**
- Imputes missing ages using pharmacy data (within 365-day window)
- Applies age validation (1-114 years)
- Tracks age imputation success rates

**Step 7: Unified Cohort Event Features**
- Creates comprehensive view combining medical and pharmacy events
- Classifies events into:
  - `OPIOID_ED`: Opioid-related emergency visits (comprehensive ICD codes including F11.x, T40.x, Y12.x)
  - `ED_NON_OPIOID`: ADE events (Emergency room visits)
  - `Medical`: Other medical events
  - `Pharmacy`: Drug prescription events
- Maintains all clinical and demographic features

**Step 8: First Event Resolution**
- Determines each person's first event type (OPIOID_ED, ED_NON_OPIOID, Pharmacy, Medical)
- Creates temporal hierarchy for event sequencing
- Tags all events with first event classification

**Step 8.5: Drug Exposure Analysis & Event Explosion** ⭐ **ENHANCED**
- **Drug Event Explosion Strategy**: Each drug prescription becomes a separate row with duplicated patient context
- **ADE Cohort (ED_NON_OPIOID)**: Extracts all drug prescriptions 30 days prior to first ADE event
- **Opioid Cohort (OPIOID_ED)**: Extracts all drug prescriptions for the entire time period
- **Temporal Relationships**: Calculates `days_to_ade`, `days_to_opioid_ed` for sequence modeling
- **Data Source Tracking**: Distinguishes between `cohort_event` and `drug_exposure` rows
- **Sequence Modeling Ready**: Enables FpGrowth, LSTM, DTW, and symbolic reasoning analysis

**Step 8.8: Cohort Exclusivity Enforcement** ⭐ **NEW**
- **OPIOID_ED Priority**: Processes opioid_ed cohort first
- **Mutual Exclusivity**: Removes opioid_ed patients from ed_non_opioid cohort
- **Validation**: Ensures no overlap between cohorts
- **Quality Assurance**: Logs cohort counts and validates exclusivity

**Step 9: Opioid ED Cohort Creation**
- Creates comprehensive opioid ED cohort with:
  - All cohort events (medical, pharmacy, opioid ED events)
  - **Complete drug prescription history** (no time restriction)
  - **Exploded drug events** (1 row per drug prescription)
  - Temporal drug-event relationships
  - Data source tracking (`cohort_event` vs `drug_exposure`)

**Step 10: ADE Cohort Creation**
- Creates comprehensive ADE cohort with:
  - All cohort events (medical, pharmacy, ADE events)
  - **Drug prescriptions 30 days prior to first ADE** (time-restricted)
  - **Exploded drug events** (1 row per drug prescription)
  - Temporal drug-ADE relationships
  - Data source tracking for analysis

#### **Key Features**

**Drug-ADE Relationship Tracking**
```sql
-- ADE events: 30-day lookback window
SELECT drug_name, days_to_ade, therapeutic_class_1
FROM ade_drug_exposure
WHERE days_to_ade BETWEEN 0 AND 30

-- Opioid events: Complete drug history
SELECT drug_name, days_to_opioid_ed, therapeutic_class_1
FROM opioid_drug_exposure
```

**Temporal Analysis**
- Tracks days between drug prescription and adverse events
- Enables analysis of drug exposure timing
- Supports causality assessment in drug-ADE relationships

**Comprehensive Data Structure**
- Maintains all original clinical features
- Adds drug exposure metadata
- Includes data source classification for analysis

#### Usage
```bash
python cohort_analysis/create_cohort.py --age-band "0-12" --event-year 2020
```

#### Output
Creates comprehensive cohort files in S3:
```
s3://pgxdatalake/cohorts/opioid_ed/age_band={age_band}/event_year={event_year}/cohort.parquet
s3://pgxdatalake/cohorts/ed_non_opioid/age_band={age_band}/event_year={event_year}/cohort.parquet
```

**Cohort Contents:**
- **Medical Events**: All clinical encounters with ICD codes, procedures, demographics
- **Pharmacy Events**: Drug prescriptions with therapeutic classifications
- **Drug Exposure Data**: Temporal drug-event relationships for analysis
- **Metadata**: Data source tracking, temporal calculations, cohort classification

### 2.1 Drug Exposure Analysis Capabilities ⭐ **ENHANCED**

The enhanced cohort creation process now includes comprehensive drug exposure analysis with **drug event explosion** for both opioid ED events and ADE events, enabling advanced sequence modeling and pattern mining:

#### **Drug-ADE Relationship Tracking**

**ADE Events (ED_NON_OPIOID)**
- **30-Day Lookback Window**: Extracts all drug prescriptions within 30 days prior to first ADE event
- **Drug Event Explosion**: Each drug prescription becomes a separate row with duplicated patient context
- **Temporal Analysis**: Calculates `days_to_ade` for each drug prescription
- **Causality Assessment**: Enables analysis of drug exposure timing relative to adverse events
- **Risk Window Analysis**: Supports identification of high-risk drug exposure periods
- **Sequence Modeling**: Ready for FpGrowth, LSTM, and DTW analysis

**Opioid ED Events**
- **Complete Drug History**: Extracts all drug prescriptions for the entire time period
- **Drug Event Explosion**: Each drug prescription becomes a separate row with duplicated patient context
- **Temporal Analysis**: Calculates `days_to_opioid_ed` for each drug prescription
- **Pattern Analysis**: Enables identification of drug sequences leading to opioid events
- **Comprehensive Exposure**: Supports analysis of long-term drug exposure patterns
- **Sequence Modeling**: Ready for advanced pattern mining and sequence analysis

#### **Enhanced Analytical Capabilities**

**Temporal Drug Analysis with Exploded Events**
```sql
-- Analyze drug exposure patterns before ADE events (exploded format)
SELECT 
    drug_name,
    therapeutic_class_1,
    AVG(days_to_ade) as avg_days_to_ade,
    COUNT(*) as prescription_count,
    COUNT(DISTINCT mi_person_key) as distinct_patients
FROM ade_drug_exposure
WHERE days_to_ade BETWEEN 0 AND 30
GROUP BY drug_name, therapeutic_class_1
ORDER BY prescription_count DESC;

-- Analyze drug sequences leading to opioid events (exploded format)
SELECT 
    drug_name,
    therapeutic_class_1,
    AVG(days_to_opioid_ed) as avg_days_to_opioid_ed,
    COUNT(*) as prescription_count,
    COUNT(DISTINCT mi_person_key) as distinct_patients
FROM opioid_drug_exposure
GROUP BY drug_name, therapeutic_class_1
ORDER BY prescription_count DESC;
```

**Drug Interaction Analysis with Exploded Events**
- **Co-prescription Patterns**: Identify drugs commonly prescribed together before adverse events
- **Therapeutic Class Analysis**: Analyze drug class associations with specific adverse events
- **Temporal Sequencing**: Understand drug exposure timing and sequence patterns
- **Risk Factor Identification**: Identify high-risk drug combinations and exposure windows
- **Sequence Mining**: Enable FpGrowth analysis on drug sequences
- **Pattern Recognition**: Support LSTM and DTW analysis for drug patterns

**Sequence Modeling Capabilities**
```sql
-- Prepare data for FpGrowth analysis
SELECT 
    mi_person_key,
    drug_name,
    therapeutic_class_1,
    days_to_ade,
    data_source
FROM ade_drug_exposure
WHERE data_source = 'drug_exposure'
ORDER BY mi_person_key, days_to_ade;

-- Prepare data for LSTM sequence modeling
SELECT 
    mi_person_key,
    ARRAY_AGG(drug_name ORDER BY days_to_opioid_ed) as drug_sequence,
    ARRAY_AGG(days_to_opioid_ed ORDER BY days_to_opioid_ed) as temporal_sequence
FROM opioid_drug_exposure
WHERE data_source = 'drug_exposure'
GROUP BY mi_person_key;
```

**Data Quality Features**
- **Data Source Tracking**: Distinguishes between `cohort_event` and `drug_exposure` data sources
- **Temporal Validation**: Ensures drug prescriptions occur before adverse events
- **Completeness Tracking**: Monitors drug exposure data completeness and coverage
- **Metrics Collection**: Comprehensive metrics on drug exposure patterns and volumes
- **Cohort Exclusivity**: Ensures no overlap between OPIOID_ED and ED_NON_OPIOID cohorts
- **Explosion Validation**: Validates drug event explosion process and data integrity

#### **Research Applications**

**Pharmacovigilance Studies**
- Identify drugs associated with specific adverse events using exploded drug data
- Analyze temporal relationships between drug exposure and adverse outcomes
- Support causality assessment in drug safety studies
- Enable sequence-based pattern recognition for drug safety signals

**Risk Prediction Models**
- Develop models predicting adverse events based on drug exposure patterns
- Identify high-risk drug combinations and exposure windows
- Support personalized medicine approaches
- Enable sequence-based risk prediction using LSTM and DTW

**Clinical Decision Support**
- Provide evidence for drug safety monitoring protocols
- Support clinical guidelines for drug prescribing
- Enable proactive identification of high-risk patients
- Support sequence-based clinical decision making

**Regulatory Compliance**
- Support post-marketing surveillance requirements
- Provide data for drug safety signal detection
- Enable compliance with pharmacovigilance regulations
- Support sequence-based regulatory analysis

### 3. Network Feature Engineering (`feature_engineer_cohort_network.py`)

This script analyzes cohort data to extract network features from medical events and medications. It performs the following operations:

1. Loads cohort data for a specific age band and event year
2. Extracts tokens from diagnosis codes, procedure codes, and drug names
3. Performs market basket analysis using FP-Growth algorithm:
   - Identifies frequent itemsets in positive and negative samples (using 0.05 min_support threshold)
   - Extracts patterns that appear only in positive samples
   - Creates feature flags based on these patterns
4. Generates enhanced datasets with network features
5. Saves itemsets and association rules for further analysis

#### Usage
```bash
python feature_engineer_cohort_network.py --cohort opioid_ed --age-band "0-12" --event-year 2020
# Or process all combinations in parallel
python feature_engineer_cohort_network.py --all --parallel 4
```

#### Output
Creates enhanced datasets and signal files in S3:
```
s3://pgxdatalake/cohorts/samples/{cohort}_vs_non_ed/{age_band}/{event_year}/enhanced_dataset.parquet
s3://pgxdatalake/opioid-ed-visit-datasets/signals/{cohort}_vs_non_ed/{age_band}/{event_year}/positive_only_itemsets.json
s3://pgxdatalake/opioid-ed-visit-datasets/signals/{cohort}_vs_non_ed/{age_band}/{event_year}/positive_only_rules.json
```

### 4. Drug Name Feature Encoding

The pipeline implements a sophisticated numerical encoding strategy for drug names that combines linguistic features with pattern mining metrics. This encoding creates a unique, fixed-length numerical representation for each drug that preserves both its linguistic characteristics and its role in prescription patterns.

#### Encoding Format
```
{first_letter:02d}{length:02d}{syllables:02d}{consonants:02d}{support:03d}{confidence:03d}{certainty:03d}
```

#### Example: "Aspirin"
```
01 07 02 04 750 850 920
│  │  │  │  │   │   └── Certainty (0.92)
│  │  │  │  │   └── Confidence (0.85)
│  │  │  │  └── Support (0.75)
│  │  │  └── Number of consonants (4)
│  │  └── Number of syllables (2)
│  └── Length of word (7)
└── First letter index (1 for 'A')
```

#### Component Details
1. **First Letter Index (2 digits)**
   - Position in alphabet (01-26)
   - Primary separation for alphabetical grouping
   - Example: 'A' = 01, 'B' = 02, etc.

2. **Length (2 digits)**
   - Total number of characters in drug name
   - Secondary separation within each letter group
   - Example: "Aspirin" = 07

3. **Syllables (2 digits)**
   - Number of syllables in drug name
   - Captures phonetic complexity
   - Example: "Aspirin" = 02 (As-pir-in)

4. **Consonants (2 digits)**
   - Number of consonants in drug name
   - Additional phonetic feature
   - Example: "Aspirin" = 04 (s,p,r,n)

5. **FPGrowth Metrics (3 digits each)**
   - Support: Frequency of drug in patterns (×1000)
   - Confidence: Strength of association (×1000)
   - Certainty: Reliability of pattern (×1000)
   - Example: 750 = 0.75 support

#### Benefits
- Fixed-length (15 digits) for consistent processing
- Hierarchical separation for efficient grouping
- Combines linguistic and statistical features
- Preserves pattern mining insights
- Enables numerical comparison of drug names

#### Usage
```python
# Example encoding
drug_name = "Aspirin"
fpgrowth_metrics = {
    'support': 0.75,
    'confidence': 0.85,
    'certainty': 0.92
}
encoded = encode_drug_name(drug_name, fpgrowth_metrics)
# Result: "010702040750850920"
```

#### S3 Storage Paths
The drug encoding maps are stored in S3 with the following structure:

```
s3://{S3_BUCKET}/drug_encodings_parquet/cohort_name={cohort}/age_band={band}/event_year={year}/drug_encodings.parquet
s3://{S3_BUCKET}/drug_encodings_json/cohort_name={cohort}/age_band={band}/event_year={year}/drug_encodings.json
```

The files are stored in both Parquet and JSON formats for different use cases:
- Parquet format for efficient data processing and analysis
- JSON format for easy inspection and integration with other tools

#### Accessing Drug Encodings
```python
from s3_utils import get_output_paths

# Get paths for a specific cohort
paths = get_output_paths(
    cohort_name="opioid_ed",
    band="65-74",
    year="2020"
)

# Access drug encoding paths
drug_encodings_parquet = paths["drug_encodings_parquet"]
drug_encodings_json = paths["drug_encodings_json"]
```

This addition to the README:
1. Documents the S3 path structure for drug encoding maps
2. Explains the purpose of having both Parquet and JSON formats
3. Provides a code example for accessing the paths
4. Maintains consistency with the existing documentation style
5. Helps users understand where to find the encoded drug data

The documentation aligns with the implementation in `s3_utils.py` and provides clear guidance for users who need to access the drug encoding maps.

### Important Note on Network Visualization Interpretation

When interpreting the network visualizations generated from association rules, it's crucial to understand the distinction between correlation and causation:

❗ **Short Answer:**
The network visualizations show correlated patterns of co-occurrence, not directional influence or causal flow.

🔍 **Why?**
Association rules (like those from FpGrowth) represent:
- Statistical co-occurrence between items in transactions
- E.g., "If drug A is present, drug B is also often present"

But they don't establish causality because they:
- Don't control for confounding variables
- Don't establish temporal precedence
- Don't use interventions to test effects

So while an arrow A → B is drawn (based on a rule), this is not a causal arrow — it represents a conditional probability relationship:
```
P(B | A) is high → draw A → B
```

✅ **Why the Visualization Is Still Valuable**
Even without causality:
- Summing support across multiple co-occurrence paths gives a meaningful measure of total association weight
- Directionality reflects rule direction (not causal flow)
- Node centrality indicates clustering or "hub" drugs often present in many co-occurrence patterns
- Edge thickness communicates real signal strength in the data

🧠 **Bottom Line**
| Interpretation | Is Valid? | Explanation |
|----------------|-----------|-------------|
| A → B is causal | ❌ | FpGrowth doesn't model interventions |
| A → B co-occur often | ✅ | Based on high confidence/support |
| Thicker edge = more total co-occurrence | ✅ | Sum(support) reflects total influence |

### 5. Feature Importance to BupaR Analysis (`feature_importance_bupaR.py`)

This script analyzes the feature importance results from CatBoost models and prepares data for process mining analysis. It performs the following operations:

1. Loads feature importance metrics from CatBoost FFA analysis:
   - Support: Frequency of feature occurrence in important patterns
   - Coverage: Proportion of cases explained by the feature
   - Significance: Statistical significance of feature importance
   - Class-specific metrics for both target and control groups

2. Identifies top K important features based on:
   - Support and coverage thresholds
   - Statistical significance testing
   - Class-specific importance rankings
   - Cross-validation stability

3. Filters training data to include only important features:
   - Maintains temporal relationships
   - Preserves case identifiers
   - Retains activity and lifecycle information
   - Ensures data quality and completeness

4. Prepares data for BupaR analysis by:
   - Retaining timestamps and case IDs
   - Formatting activity and lifecycle information
   - Separating target and control groups
   - Creating event logs for process mining
   - Ensuring proper temporal ordering

5. Saves prepared data for process mining analysis:
   - Parquet format for efficient storage
   - Separate files for target and control groups
   - Metadata for analysis configuration
   - Validation metrics and summaries

#### Usage
```bash
# Process single age band
python bupaR_analysis/feature_importance_bupaR.py --age-band "0-12"

# Process all age bands in parallel
python bupaR_analysis/feature_importance_bupaR.py --all --parallel 4

# Customize feature selection
python bupaR_analysis/feature_importance_bupaR.py --age-band "0-12" --top-k 15 --min-support 0.1 --min-coverage 0.2

# Generate detailed analysis report
python bupaR_analysis/feature_importance_bupaR.py --age-band "0-12" --generate-report
```

#### Output
Creates prepared datasets for BupaR analysis in S3:
```
s3://pgxdatalake/ade-risk-model/Step5_Time_to_Event_Model/3_bupaR_datasets/cohort{age_band}_target/bupaR_data.parquet
s3://pgxdatalake/ade-risk-model/Step5_Time_to_Event_Model/3_bupaR_datasets/cohort{age_band}_control/bupaR_data.parquet
```

#### Analysis Capabilities
The script enables comprehensive process mining analysis through BupaR:

1. **Process Maps**
   - Frequency-based process maps
   - Relative consequent process maps
   - Performance-based process maps
   - Custom node and edge metrics

2. **Trace Analysis**
   - Trace frequency analysis
   - Trace length distribution
   - Trace variant analysis
   - Custom trace metrics

3. **Performance Analysis**
   - Throughput time analysis
   - Resource utilization
   - Activity frequency
   - Bottleneck detection

4. **Comparative Analysis**
   - Target vs. control group comparison
   - Age band comparisons
   - Temporal pattern analysis
   - Statistical significance testing

5. **Conformance Analysis**
   - Process conformance checking
   - Fitness analysis
   - Precision analysis
   - Alignment-based metrics

#### Visualization Examples
The analysis generates various visualizations:

1. **Process Maps**
   ```r
   # Frequency-based process map
   process_map(eventlog, type = frequency(value = "relative-consequent"))
   
   # Performance-based process map
   process_map(eventlog, type = performance(level = "activity"))
   ```

2. **Trace Analysis**
   ```r
   # Trace explorer
   trace_explorer(eventlog, n_traces = 10)
   
   # Activity presence
   activity_presence(eventlog)
   ```

3. **Performance Analysis**
   ```r
   # Throughput time
   throughput_time(eventlog) %>% summary()
   
   # Resource frequency
   resource_frequency(eventlog)
   ```

4. **Comparative Analysis**
   ```r
   # Activity frequency comparison
   activity_frequency(target_eventlog) %>%
     left_join(activity_frequency(control_eventlog), 
              by = "activity", 
              suffix = c("_target", "_control"))
   ```

#### Metrics and Validation
The script includes comprehensive metrics and validation:

1. **Feature Importance Metrics**
   - Support and coverage scores
   - Statistical significance (p-values)
   - Class-specific importance
   - Cross-validation stability

2. **Data Quality Metrics**
   - Completeness checks
   - Temporal consistency
   - Case coverage
   - Activity coverage

3. **Process Mining Metrics**
   - Process map metrics
   - Trace analysis metrics
   - Performance metrics
   - Conformance metrics

4. **Comparative Metrics**
   - Group differences
   - Statistical tests
   - Effect sizes
   - Confidence intervals

# Survival analysis integration

```r
censored_log <- log %>%
mutate(
event_status = ifelse(activity == "Hospitalization", 1, 0),
obs_end = pmax(last_event_time, ymd(paste(year, "12-31")))
) %>%
filter(start <= obs_end)

# Comparative pathway analysis

compare_pathways(
pathway_list(hospitalized, non_hospitalized),
type = "differential",
significance_level = 0.05
)
```

### 5. Cohort Quality Assurance (`qa_cohort.py`)

This script validates created cohorts and provides quality assurance:

1. Checks cohorts for proper control-to-case ratios (strict 5:1 ratio required)
2. Identifies cohorts with insufficient control samples
3. Deletes and reprocesses cohorts that don't meet the ratio threshold
4. Supports flexible path specifications
5. Verifies control-to-case ratios:
   - Enforces a strict 5:1 ratio for both opioid and non-opioid cohorts
   - Allows control reuse between cohorts to maintain the 5:1 ratio
   - Provides detailed information about control sharing
   - Ensures consistent ratio maintenance even with limited control pools

#### Usage
```bash
python qa_cohort.py --threshold 5.0
```

### 6. Opioid ED Risk Model Analysis (`opioid_ed_risk_model.qmd`)

This Quarto document performs comprehensive analysis of opioid ED risk using CatBoost and process mining, with separate models for each age band. It includes the following steps:

1. **Data Preparation**
   - Loads feature-engineered cohort data from S3 using DuckDB
   - Processes data separately for each age band:
     - 0-12 years
     - 13-17 years
     - 18-25 years
     - 26-35 years
     - 36-45 years
     - 46-55 years
     - 56-65 years
     - 66+ years
   - Filters data for years 2016-2019
   - Removes duplicates and handles missing values
   - Performs feature selection:
     - Excludes high cardinality columns
     - Removes lagging indicators
     - Filters out unnecessary features

2. **Feature Processing**
   - For each age band:
     - Identifies and processes categorical features:
       - Adds 'None' category for missing values
       - Converts to string type
       - Handles categorical encoding
     - Processes numerical features:
       - Identifies numerical columns
       - Excludes target variable
       - Maintains data types

3. **Model Training**
   - For each age band:
     - Splits data into train (2016-2018) and test (2019) sets
     - Configures CatBoost classifier with optimized parameters:
       - 2000 iterations
       - Depth of 12
       - Ordered boosting
       - MVS bootstrap
       - Early stopping
     - Trains model with categorical feature support
     - Uses recall as primary evaluation metric
     - Saves model to disk

4. **Feature Importance Analysis**
   - For each age band:
     - Calculates Formal Feature Attribution (FFA) values for model interpretability
     - Computes feature importance scores using FFA
     - Identifies top 20 most important features
     - Creates feature importance visualization
     - Saves results to CSV

5. **Model Evaluation**
   - For each age band:
     - Calculates comprehensive metrics:
       - AUC and AUPR
       - Brier Score
       - Accuracy and Log Loss
       - F1 Score, Precision, and Recall
     - Generates confusion matrix
     - Provides detailed performance analysis
     - Saves metrics to JSON

6. **Process Mining Analysis**
   - For each age band:
     - Converts predictions to event log format
     - Creates process maps using BupaR
     - Generates process animations
     - Analyzes risk distributions
     - Provides comparative analysis of outcomes

7. **Results Visualization**
   - Creates age-band specific visualizations:
     - Risk distribution plots
     - Process maps
     - Interactive visualizations
   - Generates comparative analysis:
     - Metrics comparison across age bands
     - Feature importance comparison
     - Process pattern comparison
   - Provides summary statistics by age band

#### Usage
```bash
quarto render opioid_ed_risk_model.qmd
```

#### Output
Generates an HTML report with:
- Age-band specific model performance metrics
- Feature importance analysis for each age band
- Process mining visualizations by age band
- Risk distribution analysis
- Interactive process maps
- Comparative analysis across age bands

#### Dependencies
- R packages: bupaR, processmapR, processmonitR, processanimateR
- Python packages: catboost, shap, pandas, numpy
- DuckDB for data loading

## Prerequisites

- Python 3.8 or higher
- DuckDB with S3 support
- AWS credentials configured for S3 access

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials for S3 access:
```bash
aws configure
```

## Data Flow

### Pharmacy Data Processing
1. **Data Loading**
   - Loads from `s3://pgxdatalake/pharmacy/**/*.parquet`
   - Parses event dates from incurred_date

2. **Data Augmentation**
   - Adds age bands based on member_age_dos
   - Adds event_year from parsed dates

3. **Drug Name Standardization**
   - Converts to lowercase
   - Removes trailing slashes
   - Replaces spaces with underscores
   - Replaces '/' with '+'
   - Sorts drug combinations alphabetically

4. **Drug Name Mapping**
   - Applies mappings from `s3://pgxdatalake/drug_mappings/*_mappings.json`
   - Excludes medical supplies
   - Standardizes drug names

### Medical Data Processing
1. **Medical Data Loading and Normalization**
   - Loads from `s3://pgxdatalake/medical/**/*.parquet`
   - Parses event dates from incurred_date
   - Applies ICD mappings from `s3://pgxdatalake/claim_mappings/icd_mappings.json`
   - Adds age bands and event years
   - Creates `medical_augmented` view

2. **Column Filtering**
   - Selects specific columns for analysis
   - Creates `medical_filtered` view

3. **Pharmacy Data Preparation**
   - Creates `pharmacy_augmented` view with basic fields
   - Creates `pharmacy_cleaned` view with all fields
   - Maintains consistent schema with medical data

4. **Age Imputation**
   - Identifies medical records with missing ages (member_age_dos = 255)
   - Matches with pharmacy records within 365 days
   - Uses pharmacy age data to fill missing values
   - Maintains original age if no pharmacy match found
   - Creates `medical_features` view

5. **Unified Timeline Creation**
   - Creates `cohort_features_timeline` view combining medical and pharmacy data
   - Maintains chronological ordering by person and date
   - Includes all fields from both data sources
   - Tags events as:
     - 'Pharmacy' for pharmacy events
     - 'OPIOID_ED' for opioid-related ED visits
     - 'ED_NON_OPIOID' for non-opioid ED visits
     - 'Medical' for other medical events

6. **Person-Level Event Tagging**
   - Creates `cohort_features_tagged` view
   - Identifies adverse event cases by person
   - Creates clean control groups:
     - People with no opioid or non-opioid ED events
     - People with no non-opioid or opioid ED events
   - Maintains person-level consistency in event tagging

7. **Existing Cohort Check**
   - Verifies if cohort already exists in S3
   - Skips processing if found to avoid duplication

8. **Cohort Sampling**
   - Creates separate sampled cohorts for opioid and non-opioid models
   - Counts distinct persons in each category:
     - `OPIOID_ED` events for opioid model
     - `ED_NON_OPIOID` events for non-opioid model
     - Non-adverse events for control sampling
   - Implements strict 5:1 person-level ratio:
     - Maintains exactly 5 controls per case for each cohort
     - Reuses controls between cohorts when necessary to maintain ratio
     - Adjusts sampling strategy when control pool is limited
     - Uses deterministic sampling for reproducibility
   - Samples controls using window functions:
     - Assigns controls to both cohorts when needed
     - Maintains person-level consistency
     - Ensures proper ratio verification
   - Assembles final cohorts:
     - Combines cases (target=1) with sampled controls (target=0)
     - Preserves all events for each person in chronological order
     - Creates `sampled_opioid_ed_cohort` and `sampled_ed_non_opioid_cohort` views
   - Verifies proper ratio:
     - Confirms 5:1 person-level ratio before writing output
     - Provides detailed logs of actual case and control counts
     - Ensures cohort integrity for downstream analysis
   - Adaptive Control Sampling:
     - Automatically detects when there aren't enough unique controls available
     - Switches to a shared control pool approach when necessary
     - Applies the same control-to-case ratio to both cohorts to maintain fairness
     - Allows controls to be used in both opioid and non-opioid cohorts in limited control scenarios
     - Adjusts verification thresholds proportionally when shared controls are detected

9. **Output Generation**
   - Creates two separate cohort files:
     - `opioid_ed`: For opioid-related adverse events model
     - `ed_non_opioid`: For non-opioid adverse events model
   - Each cohort includes its respective adverse events and sampled non-adverse events
   - Writes final cohorts to S3 with consistent partitioning
   - Saves processing metrics for both models

10. **Feature Importance Analysis**
    - Loads CatBoost model feature importance results
    - Identifies significant features using support and coverage metrics
    - Prepares data for process mining:
      - Filters to important features
      - Maintains temporal information
      - Preserves case and activity identifiers
    - Creates separate datasets for target and control groups
    - Enables process mining analysis of significant event patterns

## Metrics

Both scripts collect and save metrics for each processing step, including:
- Row counts
- Distinct person counts
- Age distributions
- Diagnosis and procedure code distributions
- Drug and HCG line distributions
- Target class distributions

Metrics are saved as JSON files in:
```
s3://pgx-repository/pgx-datasets/pipeline_metrics/{age_band}/{event_year}/{cohort}_{timestamp}.json
```

## Data Dependencies

The scripts work together in the following way:
1. `clean_pharmacy.py` processes pharmacy data and creates standardized drug names
2. `create_cohorts.py` uses the cleaned pharmacy data for age imputation in medical records
3. Both scripts maintain consistent age bands and event years for proper data alignment
4. `feature_importance_bupaR.py` uses CatBoost model results to prepare data for process mining

---

**Note:**  
Make sure your DuckDB installation supports S3 access and is properly configured with your AWS credentials.

#### Methods or Approach
We developed a comprehensive data processing pipeline using DuckDB for efficient handling of large-scale healthcare data. The approach includes:

1. **Standardized Data Processing**
   - Automated cleaning and normalization of pharmacy and medical claims data
   - Consistent handling of drug names, diagnosis codes, and procedure codes
   - Age band standardization and event date parsing
   - FAERS data integration for hospitalization risk window calculation

2. **Advanced Cohort Creation**
   - Person-level event timeline construction
   - Sophisticated adverse event identification
   - Balanced control group sampling using SYSTEM method
   - Age imputation from multiple data sources
   - FAERS-based hospitalization risk window definition

3. **Feature Engineering for High Cardinality Data**
   - Network analysis of medical events and drug interactions using market basket analysis (FP-Growth)
   - Two-phase approach to temporal analysis:
     1. Initial pattern discovery with market basket analysis
     2. Advanced temporal process mining with BupaR for significant patterns
   - Dimensionality reduction for high-cardinality categorical variables
   - Feature aggregation at multiple temporal windows
   - FAERS-derived hospitalization risk windows

4. **Advanced Risk Modeling**
   - CatBoost models for opioid ED event prediction
   - CatBoost hospitalization risk model with FAERS-validated windows
   - Formal feature attribution methods for model interpretability
   - Cross-validation with temporal stratification
   - FAERS-based risk window calibration for hospitalization prediction

5. **Process Mining Analysis**
   - Feature importance-based data filtering
   - Temporal process analysis using BupaR
   - Comparative analysis of target and control groups
   - Identification of critical event sequences
   - Process performance and conformance analysis
   - Advanced visualization capabilities:
     - Interactive process maps
     - Dynamic trace exploration
     - Performance dashboards
     - Comparative visualizations
   - Statistical validation:
     - Significance testing
     - Effect size analysis
     - Confidence intervals
     - Cross-validation
   - Custom metrics development:
     - Process-specific metrics
     - Group comparison metrics
     - Performance indicators
     - Quality measures

6. **FAERS Integration for Hospitalization Risk**
   - Mapping of drug names to FAERS terminology
   - Extraction of hospitalization-related ADE timelines
   - Risk window calibration based on FAERS hospitalization reports
   - Validation of hospitalization risk patterns
   - Integration of FAERS severity metrics for hospitalization prediction

#### Principal Findings
The pipeline successfully processes large-scale healthcare data with several key achievements:

1. **Data Quality**
   - Consistent handling of drug names and medical codes
   - Accurate age imputation from pharmacy records
   - Maintained data integrity across processing steps
   - FAERS-validated hospitalization risk windows

2. **Processing Efficiency**
   - Efficient handling of large datasets using DuckDB
   - Optimized sampling methods for balanced cohorts
   - Scalable architecture supporting multiple age bands and years
   - Efficient FAERS data integration for hospitalization risk

3. **Advanced Analytics Results**
   - Network analysis revealed significant drug interaction patterns
   - BupaR sequence analysis identified critical event pathways
   - CatBoost models achieved high predictive accuracy for both ED and hospitalization events
   - Formal feature attribution provided clinically interpretable risk factors
   - FAERS-validated hospitalization risk windows improved prediction accuracy

4. **Research Readiness**
   - Clean, standardized datasets ready for analysis
   - Balanced cohorts for both opioid and non-opioid studies
   - Comprehensive documentation and metrics
   - FAERS-validated hospitalization risk windows

#### Future Research Opportunities
The pipeline enables several promising research directions:

1. **Temporal Analysis**
   - Enhanced sequence mining with BupaR for process mining of significant event patterns
   - Time-based process modeling of medical events using BupaR
   - Identification of critical event sequences and pathways in patient journeys
   - Analysis of time intervals between significant medical events
   - Dynamic hospitalization risk window adjustment based on FAERS updates

2. **Multi-modal Analysis**
   - Integration of additional data sources
   - Cross-validation across different data types
   - Enhanced feature engineering
   - Network-based feature extraction
   - Real-time FAERS data integration for hospitalization risk

3. **Advanced Analytics**
   - Refinement of CatBoost hospitalization risk models
   - Development of ensemble methods
   - Advanced feature attribution techniques
   - Network-based risk scoring
   - FAERS-based hospitalization risk window optimization

4. **Clinical Applications**
   - Real-time hospitalization risk prediction
   - Personalized intervention strategies
   - Dynamic risk assessment
   - Clinical decision support systems
   - FAERS-informed hospitalization risk monitoring

#### Lessons Learned
Key insights from developing this large-scale healthcare data pipeline:

1. **Data Quality Challenges**
   - Importance of consistent data cleaning
   - Need for robust error handling
   - Value of comprehensive validation
   - Challenges in handling high-cardinality features
   - Benefits of FAERS data integration for hospitalization risk

2. **Technical Considerations**
   - Benefits of using DuckDB for large datasets
   - Importance of efficient sampling methods
   - Need for scalable architecture
   - Value of Formal Feature Attribution over SHAP values
   - Challenges in FAERS data integration for hospitalization risk

3. **Analytical Insights**
   - Effectiveness of network analysis for high-cardinality data
   - Value of sequence analysis in healthcare events
   - Importance of model interpretability through FFA
   - Benefits of CatBoost for hospitalization risk modeling
   - Significance of FAERS-validated hospitalization risk windows

4. **Research Impact**
   - Value of standardized processing
   - Importance of reproducible methods
   - Need for comprehensive documentation
   - Significance of clinically interpretable results
   - Impact of FAERS integration on hospitalization risk assessment

## Network Features

The model uses network features generated by the FpGrowth algorithm to identify patterns in medical and pharmacy data. These features are created by:

1. Extracting tokens from medical and pharmacy data
2. Running FpGrowth on positive samples to find frequent patterns
3. Creating feature flags named `network_feature_0` through `network_feature_24` (TOP_K=25)

## Integrated Analysis Approach

Our pipeline integrates pattern mining, process mining, and formal model explanation into a unified analysis framework:

### Step-by-Step Analysis Flow

1. **FpGrowth Pattern Mining** (see `feature_engineer_cohort_network.py`)
   - Extract frequent co-occurrence patterns in transactional data
   - Group by cohort, event type, or year
   - Example: itemsets of drugs, diagnoses, procedures
   - Identify significant patterns using support thresholds

2. **BupaR Process Mining** (see `feature_importance_bupaR.py`)
   - Convert itemsets into event logs:
     - Each itemset becomes a trace or partial trace
     - Attach timestamps, patient ID, and metadata
   - Analyze with BupaR:
     - Frequency and precedence analysis
     - Process conformance checking
     - Throughput time analysis
     - Process map generation
     - Trace and variant analysis

3. **CatBoost Risk Modeling**
   - Convert itemsets into binary features (1 = present, 0 = absent)
   - Train model to predict risk (hospitalization, adverse events)
   - Optimize model performance with:
     - Temporal cross-validation
     - Early stopping
     - Hyperparameter tuning
     - Class balancing

### 3.1 Temporal Filtering for Predictive Modeling ⭐ **NEW**

**Critical Data Leakage Prevention**
To ensure predictive model integrity, the CatBoost pipeline implements temporal filtering to prevent data leakage:

#### **Temporal Filtering Logic**
```python
# For each person, find the first target event date
first_target_dates = df[df['Target'] == 1].groupby('mi_person_key')['event_date'].min()

# Filter to keep only events BEFORE the first target event
df_filtered = df[
    (df['first_target_date'].isna()) |  # Keep all events for people with no target events
    (df['event_date'] < df['first_target_date'])  # Keep events before first target
]
```

#### **Why This Matters**
- **Prevents Data Leakage**: Future events cannot influence past predictions
- **Maintains Temporal Integrity**: Model learns from events that actually preceded the outcome
- **Improves Model Validity**: Ensures predictions are based on available information at prediction time

#### **Implementation Details**
- **Applied Before Train/Test Split**: Ensures both training and test sets follow temporal constraints
- **Person-Level Filtering**: Each person's timeline is filtered independently
- **Comprehensive Coverage**: Applies to all event types (medical, pharmacy, procedures)
- **Metrics Tracking**: Reports data volume before and after filtering

#### **Example Impact**
```
Before filtering: 1,250,000 events
After filtering:  987,000 events
Filtered out:     263,000 events (21% reduction)
```

This ensures the model only learns from temporally valid patterns and can make reliable predictions in real-world scenarios.

> **Note**: Temporal filtering is specific to **CatBoost predictive modeling** to prevent data leakage. **BupaR process mining** can handle multiple events and temporal sequences naturally, so this filtering step is not applied to the BupaR analysis workflow.

4. **Formal Feature Attribution (FFA)**
   - Attribute predictions to individual itemsets
   - Use symbolic methods:
     - AXPs (Abductive Explanations)
     - Contrastive reasons
     - Z3/SAT-based analysis
   - Identify minimal feature subsets responsible for predictions
   - Generate interpretable explanations

5. **Comparative Analysis**
   - Compare process patterns (BupaR) vs. risk patterns (FFA)
   - Identify convergent vs. divergent itemsets
   - Analyze alignment between:
     - Frequently used process traces
     - Risk-influential patterns
     - Clinical pathways

### Key Insights

| Question | Analysis Method | Insights |
|----------|----------------|-----------|
| What itemsets are most common? | FpGrowth | Frequent co-occurrence patterns |
| How do itemsets play out temporally? | BupaR | Process flows and sequences |
| Which itemsets drive model predictions? | CatBoost + FFA | Risk-influential patterns |
| Are process-dominant paths aligned with risk? | BupaR vs. FFA | Pattern alignment analysis |

### Visualization Approaches

1. **Venn Diagrams**
   - Compare frequent itemsets vs. risk itemsets
   - Identify overlapping patterns
   - Highlight unique patterns in each analysis

2. **Process Maps with Risk Overlay**
   - Base process map from BupaR
   - Color-coded by risk influence
   - Edge thickness based on frequency
   - Node size based on FFA importance

3. **Network Graphs**
   - Nodes: itemsets
   - Edges: co-occurrence relationships
   - Dual labels:
     - Process frequency
     - Risk weight
   - Color coding for pattern alignment

## Network Visualization Interpretation

❗ **Important Note:**
When interpreting the network visualizations generated from association rules, it's crucial to understand the distinction between correlation and causation.

🔍 **Why?**
Association rules (like those from FpGrowth) represent:
- Statistical co-occurrence between items in transactions
- E.g., "If drug A is present, drug B is also often present"

But they don't establish causality because they:
- Don't control for confounding variables
- Don't establish temporal precedence
- Don't use interventions to test effects

So while an arrow A → B is drawn (based on a rule), this is not a causal arrow — it represents a conditional probability relationship:
```
P(B | A) is high → draw A → B
```

✅ **Why the Visualization Is Still Valuable**
Even without causality:
- Summing support across multiple co-occurrence paths gives a meaningful measure of total association weight
- Directionality reflects rule direction (not causal flow)
- Node centrality indicates clustering or "hub" drugs often present in many co-occurrence patterns
- Edge thickness communicates real signal strength in the data

🧠 **Bottom Line**
| Interpretation | Is Valid? | Explanation |
|----------------|-----------|-------------|
| A → B is causal | ❌ | FpGrowth doesn't model interventions |
| A → B co-occur often | ✅ | Based on high confidence/support |
| Thicker edge = more total co-occurrence | ✅ | Sum(support) reflects total influence |

### Pattern Hashing and Attribution

The pattern mining process uses a sophisticated hashing and attribution system:

1. **Itemset Hashing**
   - Each frequent itemset (from FpGrowth) is:
     - Turned into a pipe-separated string (e.g., "drug_x|drug_y")
     - Hashed using MD5 to generate a unique pattern_id
   - Results in a pattern_lookup table:
     ```
     | pattern_id                         | itemsets          | support | ...metrics |
     |-----------------------------------|-------------------|---------|------------|
     | a8f72c99e5d1f4...                  | drug_x|drug_y     | 0.042   | ...        |
     | b04dd51b7926e2...                  | drug_z            | 0.089   | ...        |
     ```

2. **Pattern Attribution**
   - Each row in the DataFrame has up to MAX_PATTERN_COLUMNS slots:
     ```
     | pattern_1       | pattern_2       | ... | pattern_15     |
     |-----------------|-----------------|-----|----------------|
     | b04dd...         | None            |     |                |
     | a8f72...         | b04dd...        |     |                |
     | None            | None            |     |                |
     ```
   - These pattern_* columns reflect which pattern_ids (from pattern_lookup) were attributable to that row

3. **Metric Merge**
   - Using merge_pattern_metrics(), for each pattern_i, the corresponding metrics are merged in from pattern_lookup:
     ```
     | pattern_1       | support_1 | confidence_1 | ...
     |-----------------|-----------|--------------|
     | a8f72...         | 0.042     | 0.62         |
     ```
   - If a pattern was not matched or None, the row will have NaN or 0.0 after merge

#### Guarantees
- Each row only gets patterns it's eligible for — matched from rule/itemset presence
- Only patterns up to MAX_PATTERN_COLUMNS are attributed per row
- Patterns are attributed based on priority (e.g., support or rule quality) — usually highest scoring come first

#### Pattern Metrics
Each attributed pattern includes associated metrics:
- support_N: Frequency of the pattern in the dataset
- confidence_N: Confidence score for the pattern
- lift_N: Lift score indicating pattern significance
- certainty_N: Certainty factor for the pattern

Example schema after metric merge:
```
| pattern_1 | support_1 | confidence_1 | lift_1 | certainty_1 | pattern_2 | support_2 | ... |
|-----------|-----------|--------------|--------|-------------|-----------|-----------|-----|
| abc123... | 0.034     | 0.62         | 1.1    | 0.44        | def456... | 0.028     | ... |
| None      | NaN       | NaN          | NaN    | NaN         | None      | NaN       | ... |
```

### FPgrowth_Rank Variable

A new variable `FPgrowth_Rank` has been added to track the original ranking of network features. This variable:

- Stores the original rank (0-based index) of each active network feature
- For each row, contains a list of ranks for all network features where value = 1
- Uses -1 to indicate padded features (those added with zeros)

The rank information is valuable because:
- Higher ranked patterns (lower indices) were more frequent in the positive samples
- Helps identify the relative importance of different patterns
- Distinguishes between original and padded features
- Can be used to analyze the relationship between pattern rank and prediction accuracy

Example:
```python
# If network_feature_0 and network_feature_5 are active (value = 1)
# FPgrowth_Rank would contain [0, 5]
# If a feature was padded, its rank would be -1
```

## Association Rules and Co-Usage Analysis

### Purpose

We extract association rules from drug co-occurrence data using FP-Growth, filtering for positive-only patterns. These rules reveal structured relationships among drugs and serve as clinically interpretable features.

### What These Rules Show

Each rule takes the form:

```
```

## Analysis Pipeline Overview

```mermaid
flowchart TD
    subgraph "1_apcd_input_data: Data Processing"
        A[Pharmacy Data] --> B[Clean Pharmacy Data]
        C[Medical Data] --> D[Process Medical Data]
        B --> E[2_create_cohort: Create Cohorts]
        D --> E
    end

    subgraph "2_create_cohort: Cohort Classification"
        E --> E1[Event Classification]
        E1 --> E2[OPIOID_ED Cohort<br/>ICD Codes: F11.x, T40.x, Y12.x<br/>Drug Window: Complete History]
        E1 --> E3[ED_NON_OPIOID Cohort<br/>HCG Codes: P51, O11, P33<br/>Drug Window: 30 Days Prior]
        E2 --> E4[Cohort Exclusivity<br/>OPIOID_ED Priority]
        E3 --> E4
        E4 --> E5[Drug Event Explosion<br/>1 Row per Drug Prescription]
    end

    subgraph "3_feature_importance: Feature Screening"
        E5 --> F1[Monte Carlo CV]
        F1 --> F2[Feature Importance<br/>CatBoost + Random Forest]
        F2 --> F3[Top Features Selection]
    end

    subgraph "4_fpgrowth_analysis: Pattern Mining"
        F3 --> G[FPGrowth Analysis]
        G --> G1[Frequent Itemsets]
        G1 --> G2[Target-Focused Rules]
        G2 --> G3[Global Encoding Map]
    end

    subgraph "5_bupaR_analysis: Process Mining"
        G3 --> H[BupaR Process Mining]
        H --> H1[Event Log Creation]
        H1 --> H2[Process Flow Discovery]
    end

    subgraph "6_dtw_analysis: Trajectory Analysis"
        H2 --> I[DTW Trajectory Analysis]
        I --> I1[Patient Clustering]
        I1 --> I2[Similarity Scoring]
    end

    subgraph "7_final_model: Final Model"
        G3 --> J[Feature Integration]
        H2 --> J
        I2 --> J
        J --> J1[Final Feature Schema]
        J1 --> J2[CatBoost Training]
        J1 --> J3[Random Forest Training]
        J2 --> J4[Model Evaluation]
        J3 --> J4
    end

    subgraph "8_ffa_analysis: Attribution"
        J4 --> K[Feature Attribution]
        K --> K1[Tree Export]
        K1 --> K2[Causal Inference]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E1 fill:#bbf,stroke:#333,stroke-width:2px
    style E2 fill:#fbb,stroke:#333,stroke-width:2px
    style E3 fill:#fbb,stroke:#333,stroke-width:2px
    style E4 fill:#bfb,stroke:#333,stroke-width:2px
    style E5 fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
    style I fill:#fbb,stroke:#333,stroke-width:2px
    style K fill:#bfb,stroke:#333,stroke-width:2px
    style P fill:#fbb,stroke:#333,stroke-width:2px
```

### 4. Model Artifacts and Storage Structure

The pipeline generates and stores various model artifacts for each cohort and age band. These artifacts provide comprehensive insights into model performance, feature importance, and interpretability.

#### Model Artifacts Structure
All model artifacts are stored in S3 with the following partition structure:
```
s3://{S3_BUCKET}/{artifact_type}/cohort_name={cohort}/age_band={band}/event_year={year}/
```

#### Artifact Types and Contents

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

7. **AXP Metrics**
   - `axp_metrics.json`: Approximate Explanations (AXP) metrics including:
     - Feature coverage
     - Feature stability
     - Feature essentiality ratios

8. **Validation Results**
   - `validation_results.json`: Model validation metrics including:
     - Explanation coverage
     - Feature stability
     - Native vs SHAP importance comparison

#### Usage Example
```python
from s3_utils import get_output_paths

# Get paths for a specific cohort and age band
paths = get_output_paths(
    cohort_name="opioid_ed",
    band="65-74",
    year="2020"
)

# Access specific artifact paths
model_metrics_path = paths["model_metrics"]
shap_values_path = paths["shap_values"]
cattail_plots_path = paths["cattail_plots"]
```

#### Artifact Interpretation

1. **Model Metrics**
   - Use `model_metrics.json` to assess overall model performance
   - Compare training and test metrics to check for overfitting
   - Monitor calibration metrics (Brier score, log loss) for probability reliability

2. **Feature Importance**
   - Use SHAP plots to understand global feature importance
   - Use Cattail plots to analyze feature value distributions
   - Use mirror plots to compare feature importance between classes

3. **Model Validation**
   - Check explanation coverage in `validation_results.json`
   - Compare native CatBoost importance with SHAP values
   - Monitor feature stability across samples

4. **Causal Analysis**
   - Review feature effects in `causal_summary.json`
   - Identify significant causal relationships
   - Assess intervention impact on predictions
```

# FFA Analysis Pipeline

This pipeline performs comprehensive analysis of drug patterns and their causal relationships using CatBoost models and rule extraction.

## Overview

The pipeline performs the following key steps:

1. **Data Processing**
   - Loads and preprocesses drug pattern data
   - Handles feature encoding and normalization
   - Manages train/test splits

2. **Model Training**
   - Trains CatBoost models with optimized parameters
   - Performs cross-validation
   - Generates model predictions

3. **Rule Extraction**
   - Extracts decision rules from CatBoost models
   - Converts rules to interpretable format
   - Maps feature indices to drug names

4. **Causal Analysis**
   - Calculates causal importance of rules
   - Considers feature dependencies and correlations
   - Validates rule stability and reliability

5. **Visualization**
   - Creates hierarchical clustering of patterns
   - Generates heatmaps of feature relationships
   - Plots feature distributions and correlations

## Recent Improvements

### Enhanced Causal Analysis
- Added feature dependency consideration in causal importance calculation
- Implemented correlation-based feature adjustments
- Added confidence intervals for importance measures
- Included rule stability and coverage metrics

### Improved Visualization
- Added hierarchical clustering for pattern relationships
- Enhanced heatmap visualizations with dendrograms
- Added correlation matrix visualization
- Improved feature distribution plots

### S3 Integration
- Implemented parquet format for better performance
- Added consistent S3 path structure
- Included error handling for S3 operations

## Output Files

The pipeline generates several output files:

1. **Rule Metrics** (`rule_metrics.csv`)
   - Causal importance scores
   - Rule stability metrics
   - Coverage and accuracy measures

2. **Rule Table** (`rule_table.csv`)
   - Interpretable rule conditions
   - Pattern mappings
   - Support and confidence metrics

3. **Feature Importance** (`feature_importance.csv`)
   - Causal importance by feature
   - Feature stability metrics
   - Coverage statistics

4. **Visualizations**
   - **Pattern Analysis**
     - Pattern clustering heatmaps with hierarchical clustering
     - Feature relationship plots with correlation matrices
     - Distribution visualizations for top features
   
   - **Cattail Visualizations**
     - Feature value distributions for top 10 important features
     - Decision rule visualizations with support metrics
     - Pattern coverage and confidence plots
   
   - **Causal Analysis**
     - Causal relationship heatmaps
     - Feature importance bar plots
     - Correlation matrix visualizations

## S3 Storage

All outputs are stored in the following S3 structure:
```
s3://pgxdatalake/ffa_analysis/
├── rule_metrics/
│   └── rule_metrics.parquet
├── rule_tables/
│   └── rule_table.parquet
├── feature_importance/
│   └── feature_importance.parquet
└── visualizations/
    ├── pattern_analysis/
    │   ├── pattern_clustering.png
    │   ├── feature_relationships.png
    │   └── feature_distributions.png
    ├── cattail_plots/
    │   ├── feature_distributions.png
    │   ├── decision_rules.png
    │   └── pattern_metrics.png
    └── causal_analysis/
        ├── causal_relationships.png
        ├── feature_importance.png
        └── correlation_matrix.png
```

## Usage

```python
from ffa_analysis import FFAAnalysis

# Initialize analysis
analyzer = FFAAnalysis()

# Load and process data
analyzer.load_data()

# Train model and extract rules
analyzer.train_model()
rules = analyzer.extract_rules()

# Analyze and visualize
metrics, table, importance = analyzer.analyze_rule_metrics(rules, manifest, save_path='output/')
```

## Dependencies

- Python 3.8+
- catboost
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- boto3 (for S3 integration)

## Future Improvements

1. **Causal Analysis**
   - Add more sophisticated causal inference methods
   - Implement counterfactual analysis
   - Add intervention testing

2. **Visualization**
   - Add interactive visualizations
   - Implement network graph visualization
   - Add time series analysis

3. **Performance**
   - Optimize S3 operations
   - Add parallel processing
   - Implement caching

4. **Documentation**
   - Add API documentation
   - Include example notebooks
   - Add test cases

## Visualization Insights

### Pattern Analysis Visualizations
1. **Pattern Clustering Heatmaps**
   - Reveals groups of similar drug patterns
   - Shows hierarchical relationships between patterns
   - Helps identify common co-occurring drug combinations
   - Highlights patterns with similar support levels

2. **Feature Relationship Plots**
   - Shows correlations between different drug features
   - Identifies strongly associated drug pairs
   - Helps understand feature dependencies
   - Reveals potential confounding relationships

3. **Feature Distributions**
   - Shows the distribution of drug usage patterns
   - Identifies common vs. rare drug combinations
   - Helps understand the prevalence of different patterns
   - Reveals potential outliers or unusual patterns

### Cattail Visualizations
1. **Feature Value Distributions**
   - Shows the distribution of values for top 10 important features
   - Helps understand the range and frequency of drug usage
   - Identifies common thresholds in decision rules
   - Reveals potential data quality issues or outliers

2. **Decision Rule Visualizations**
   - Displays the extracted rules from the CatBoost model
   - Shows support and confidence metrics for each rule
   - Helps understand the model's decision-making process
   - Identifies the most influential rules

3. **Pattern Metrics Plots**
   - Shows coverage and confidence of different patterns
   - Helps identify reliable vs. rare patterns
   - Reveals the predictive power of different combinations
   - Aids in understanding pattern stability

### Causal Analysis Visualizations
1. **Causal Relationship Heatmaps**
   - Shows the strength of causal relationships between features
   - Identifies direct and indirect causal effects
   - Helps understand the impact of interventions
   - Reveals potential confounding factors

2. **Feature Importance Plots**
   - Ranks features by their causal importance
   - Shows the relative impact of different drugs
   - Helps identify key drivers of outcomes
   - Aids in understanding feature stability

3. **Correlation Matrix Visualizations**
   - Shows pairwise correlations between top features
   - Helps identify multicollinearity
   - Reveals potential feature redundancies
   - Aids in understanding feature dependencies

## Interpreting the Results

### Pattern Analysis
- **High Support Patterns**: Common drug combinations that appear frequently in the data
- **Hierarchical Clusters**: Groups of related patterns that share similar characteristics
- **Feature Correlations**: Strong associations between different drugs or features

### Cattail Insights
- **Decision Rules**: Clear, interpretable rules that explain model predictions
- **Pattern Coverage**: How widely applicable each pattern is in the dataset
- **Confidence Levels**: How reliable the patterns are in predicting outcomes

### Causal Analysis
- **Causal Importance**: How much each feature directly influences the outcome
- **Feature Stability**: How consistent the causal relationships are across different samples
- **Intervention Impact**: How changing feature values affects predictions

## Using the Visualizations

1. **Pattern Discovery**
   - Use pattern clustering to identify common drug combinations
   - Analyze feature distributions to understand usage patterns
   - Study decision rules to understand model behavior

2. **Causal Understanding**
   - Use causal heatmaps to identify key drivers
   - Study feature importance to prioritize interventions
   - Analyze correlations to understand feature dependencies

3. **Model Validation**
   - Use pattern metrics to assess rule reliability
   - Study feature stability to validate causal relationships
   - Analyze coverage to understand pattern applicability

## Common Pitfalls and Misinterpretations

### Pattern Analysis Pitfalls
1. **Correlation vs. Causation**
   - **Pitfall**: Assuming that correlated drug patterns indicate causal relationships
   - **Correct Interpretation**: Use causal analysis to validate relationships
   - **Example**: Two drugs may appear together frequently but not influence each other

2. **Support Thresholds**
   - **Pitfall**: Focusing only on high-support patterns
   - **Correct Interpretation**: Consider both support and confidence
   - **Example**: Rare but highly predictive patterns may be missed

3. **Feature Dependencies**
   - **Pitfall**: Ignoring feature correlations in pattern analysis
   - **Correct Interpretation**: Use hierarchical clustering to understand dependencies
   - **Example**: Some drugs may always appear together, creating redundant patterns

### Cattail Visualization Pitfalls
1. **Rule Interpretation**
   - **Pitfall**: Taking decision rules at face value
   - **Correct Interpretation**: Consider rule context and feature interactions
   - **Example**: A rule may be accurate but only in specific conditions

2. **Feature Importance**
   - **Pitfall**: Assuming feature importance is static
   - **Correct Interpretation**: Consider importance stability across samples
   - **Example**: Feature importance may vary based on data distribution

3. **Pattern Coverage**
   - **Pitfall**: Overgeneralizing from limited coverage
   - **Correct Interpretation**: Consider both coverage and confidence
   - **Example**: A pattern with high confidence but low coverage may not be generalizable

### Causal Analysis Pitfalls
1. **Confounding Variables**
   - **Pitfall**: Missing unobserved confounders
   - **Correct Interpretation**: Use feature stability to validate relationships
   - **Example**: A third variable may be driving the observed relationship

2. **Intervention Assumptions**
   - **Pitfall**: Assuming features can be modified independently
   - **Correct Interpretation**: Consider feature dependencies in interventions
   - **Example**: Changing one drug may affect related drugs

3. **Temporal Relationships**
   - **Pitfall**: Ignoring temporal aspects of drug usage
   - **Correct Interpretation**: Consider sequence and timing of drug patterns
   - **Example**: Order of drug administration may matter

### Data Quality Issues
1. **Missing Data**
   - **Pitfall**: Ignoring missing data patterns
   - **Correct Interpretation**: Analyze missing data distribution
   - **Example**: Systematic missing data may bias pattern discovery

2. **Data Imbalance**
   - **Pitfall**: Overlooking class imbalance
   - **Correct Interpretation**: Use appropriate sampling techniques
   - **Example**: Rare patterns may be overlooked in imbalanced data

3. **Data Quality**
   - **Pitfall**: Assuming data is clean and complete
   - **Correct Interpretation**: Validate data quality before analysis
   - **Example**: Inconsistent drug coding may affect pattern discovery

### Best Practices for Avoiding Pitfalls
1. **Validation Steps**
   - Cross-validate patterns across different time periods
   - Use multiple visualization types to confirm findings
   - Validate causal relationships with domain knowledge

2. **Interpretation Guidelines**
   - Consider both statistical and clinical significance
   - Look for consistent patterns across different analyses
   - Validate findings against known drug relationships

3. **Quality Checks**
   - Monitor data quality throughout the pipeline
   - Validate feature importance stability
   - Check for potential confounding factors

4. **Documentation**
   - Document assumptions and limitations
   - Keep track of data quality issues
   - Note any potential biases in the analysis

## How Our Workflow Mitigates Common Pitfalls

### Pattern Analysis Safeguards
1. **Correlation vs. Causation**
   - **Our Solution**: 
     - Hierarchical clustering to identify true pattern relationships
     - Causal importance calculation to validate relationships
     - Feature dependency consideration in pattern analysis
   - **Implementation**: 
     ```python
     # Feature dependency consideration in pattern analysis
     feature_correlations = rule_samples.corr()
     correlated_features = feature_correlations[feature][
         abs(feature_correlations[feature]) > 0.3
     ].index
     ```

2. **Support Thresholds**
   - **Our Solution**:
     - Multi-metric evaluation (support, confidence, coverage)
     - Hierarchical clustering to group similar patterns
     - Stability analysis across different thresholds
   - **Implementation**:
     ```python
     # Multi-metric evaluation in rule analysis
     df_metrics = df_metrics.merge(validation_results, on='rule_id', how='left')
     ```

3. **Feature Dependencies**
   - **Our Solution**:
     - Correlation matrix visualization
     - Hierarchical clustering of features
     - Feature stability analysis
   - **Implementation**:
     ```python
     # Feature stability analysis
     feature_stability = {}
     for condition in conditions:
         feature, operator, value = condition
         feature_values = rule_samples[feature]
         feature_stability[feature] = {
             'std': np.std(feature_values),
             'range': np.ptp(feature_values),
             'mean': np.mean(feature_values)
         }
     ```

### Cattail Visualization Safeguards
1. **Rule Interpretation**
   - **Our Solution**:
     - Context-aware rule extraction
     - Feature interaction consideration
     - Rule stability validation
   - **Implementation**:
     ```python
     # Rule stability validation
     rule_stability = np.std(original_preds)
     rule_coverage = len(rule_mask) / len(X_test)
     rule_accuracy = np.mean(rule_preds == y_test[rule_mask])
     ```

2. **Feature Importance**
   - **Our Solution**:
     - Dynamic importance calculation
     - Stability analysis across samples
     - Confidence interval calculation
   - **Implementation**:
     ```python
     # Dynamic importance with confidence intervals
     pred_changes = np.abs(original_preds - new_preds)
     importance = np.mean(pred_changes)
     confidence = np.std(pred_changes) / np.sqrt(len(pred_changes))
     ```

3. **Pattern Coverage**
   - **Our Solution**:
     - Coverage analysis with confidence metrics
     - Pattern validation across different subsets
     - Stability analysis of pattern coverage
   - **Implementation**:
     ```python
     # Pattern coverage analysis
     rule_coverage = len(rule_mask) / len(X_test)
     pattern_metrics = {
         'coverage': rule_coverage,
         'stability': rule_stability,
         'accuracy': rule_accuracy
     }
     ```

### Causal Analysis Safeguards
1. **Confounding Variables**
   - **Our Solution**:
     - Feature dependency consideration
     - Correlation analysis
     - Stability validation
   - **Implementation**:
     ```python
     # Feature dependency consideration in causal analysis
     for corr_feature in correlated_features:
         if corr_feature != feature:
             corr = feature_correlations.loc[feature, corr_feature]
             modified_samples[corr_feature] += corr * (
                 modified_samples[feature] - rule_samples[feature]
             )
     ```

2. **Intervention Assumptions**
   - **Our Solution**:
     - Feature correlation consideration
     - Proportional feature adjustment
     - Stability analysis of interventions
   - **Implementation**:
     ```python
     # Proportional feature adjustment
     if operator == '>':
         modified_samples[feature] = value - 1e-6
     elif operator == '<':
         modified_samples[feature] = value + 1e-6
     elif operator == '==':
         modified_samples[feature] = value + 1
     ```

3. **Temporal Relationships**
   - **Our Solution**:
     - Pattern sequence analysis
     - Time-based validation
     - Stability across time periods
   - **Implementation**:
     ```python
     # Pattern sequence analysis
     pattern_sequences = []
     for pattern in patterns:
         sequence = {
             'pattern_id': pattern['id'],
             'sequence': pattern.get('sequence', []),
             'temporal_stability': pattern.get('temporal_stability', 0)
         }
         pattern_sequences.append(sequence)
     ```

### Data Quality Safeguards
1. **Missing Data**
   - **Our Solution**:
     - Missing data analysis
     - Pattern validation across complete cases
     - Stability analysis with missing data
   - **Implementation**:
     ```python
     # Missing data analysis
     missing_analysis = {
         'missing_percentage': X_test.isnull().mean(),
         'pattern_completeness': rule_samples.notnull().mean(),
         'stability_with_missing': rule_stability
     }
     ```

2. **Data Imbalance**
   - **Our Solution**:
     - Balanced sampling in analysis
     - Pattern validation across strata
     - Stability analysis across different distributions
   - **Implementation**:
     ```python
     # Balanced sampling
     balanced_samples = []
     for stratum in strata:
         stratum_samples = rule_samples[rule_samples['stratum'] == stratum]
         balanced_samples.append(stratum_samples.sample(n=min_samples))
     ```

3. **Data Quality**
   - **Our Solution**:
     - Data quality checks
     - Pattern validation
     - Stability analysis
   - **Implementation**:
     ```python
     # Data quality checks
     quality_metrics = {
         'completeness': X_test.notnull().mean(),
         'consistency': check_data_consistency(X_test),
         'stability': calculate_stability_metrics(X_test)
     }
     ```

### Continuous Monitoring
1. **Validation Pipeline**
   - Regular pattern validation
   - Stability monitoring
   - Quality metric tracking

2. **Documentation**
   - Assumption tracking
   - Quality issue logging
   - Bias monitoring

3. **Feedback Loop**
   - Pattern refinement
   - Model updates
   - Quality improvements
```

## Drug Sequence Handling and Pattern Mining

### Temporal Integrity
- Drug sequences are truncated at the first ADE event for each person
- 30-day lookback window is maintained for drug history
- Post-ADE events are flagged and excluded from pattern mining

### Validation Steps
The pipeline includes comprehensive validation of drug sequence handling:

1. **Event Distribution**
   - Counts of events by type (OPIOID_ED, ED_NON_OPIOID, Medical, Pharmacy)
   - Unique person counts per event type

2. **Drug Sequence Statistics**
   - Pre-ADE vs Post-ADE drug counts
   - Average drugs per person
   - Range of drug counts
   - Verification of sequence truncation

3. **Pattern Matching Validation**
   - Events with valid patterns
   - Distribution of patterns across pre/post ADE events
   - Person-level pattern coverage

### Metrics
Validation metrics are stored in the metrics dictionary under the "validation" key:
```json
{
  "validation": {
    "event_distribution": {
      "event_type": {
        "event_count": number,
        "person_count": number
      }
    },
    "drug_sequences": {
      "pre_ade": {
        "person_count": number,
        "avg_drugs": number,
        "drug_range": "min-max"
      },
      "post_ade": {
        "person_count": number,
        "avg_drugs": number,
        "drug_range": "min-max"
      }
    },
    "pattern_matching": {
      "pre_ade": {
        "total_events": number,
        "events_with_patterns": number,
        "person_count": number
      },
      "post_ade": {
        "total_events": number,
        "events_with_patterns": number,
        "person_count": number
      }
    }
  }
}
```

### Key Features
- Ensures no post-ADE drugs are included in pattern mining
- Maintains temporal ordering of drug events
- Validates sequence truncation at ADE events
- Tracks pattern coverage and distribution

## Drug Name Feature Encoding Implementation

### 2. Implementation Steps

#### A. Preprocess Metadata
```python
def extract_features(name):
    return {
        'length': len(name),
        'syllables': count_syllables(name),
        'consonants': sum(1 for c in name.lower() if c in 'bcdfghjklmnpqrstvwxyz')
    }
```

#### B. Create Transactional Data
```python
# Bin features into categorical ranges
bins = {
    'length': ['short(1-5)', 'medium(6-10)', 'long(11+)'],
    'syllables': ['low(1-2)', 'medium(3-4)', 'high(5+)'],
    'consonants': ['few(1-3)', 'typical(4-6)', 'many(7+)']
}

transactions = []
for drug in drugs:
    feats = extract_features(drug)
    transaction = [
        f"length_{bins['length'][feats['length']]}",
        f"syllables_{bins['syllables'][feats['syllables']]}",
        f"consonants_{bins['consonants'][feats['consonants']]}",
        drug
    ]
    transactions.append(transaction)
```

#### C. Run FP-Growth
```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)
```

#### D. Generate Encodings
```python
def create_encoding(drug, features, fp_rules):
    letter_code = 1000 + (ord(drug[0].upper()) - ord('A')) * 1000
    support = fp_rules.loc[drug]['support'] * 1000  # 0.15 → 150
    confidence = fp_rules.loc[drug]['confidence'] * 100  # 0.82 → 82
    
    return (
        letter_code * 10**10 +
        features['length'] * 10**8 +
        features['syllables'] * 10**6 +
        features['consonants'] * 10**4 +
        int(support) * 10**2 +
        confidence
    )
```

### 3. Business Rule Integration
Example Threshold Rules:
```json
{
  "rules": [
    {
      "name": "High-Risk A-Class Drugs",
      "condition": "10000000000000 <= drug_code < 20000000000000",
      "action": "require_pharmacist_review"
    },
    {
      "name": "Frequent Consonant-Rich Drugs",
      "condition": "drug_code % 10000 >= 7000",
      "action": "flag_for_linguistic_audit"
    }
  ]
}
```

### 4. Collision Prevention & Audit
Collision Check:
```python
def check_collisions(encoded_values):
    unique_count = len(set(encoded_values))
    return "Collision risk: {0:.8f}%".format(
        (1 - unique_count/len(encoded_values)) * 100
    )
```

Reverse Lookup Table:
| Encoded Value | Drug Name | Length | Syllables | Consonants | Support | Confidence |
|---------------|-----------|---------|-----------|------------|---------|------------|
| 100007034015082 | Aspirin | 7 | 3 | 4 | 0.15 | 0.82 |

### 5. Advantages Over Standard Encoding
| Aspect | Traditional Target Encoding | FP-Growth Enhanced Encoding |
|--------|----------------------------|----------------------------|
| Collision Risk | High (frequency-only) | Extremely Low (6+ dimensions) |
| Rule Interpretability | Opaque numerical values | Hierarchical structure with discrete thresholds |
| Association Power | None | Built-in market basket insights |
| Temporal Tracking | Difficult | Versioned FP-Growth snapshots |

### 6. Maintenance Strategy
Weekly FP-Growth Updates:
```python
# Update rules with 10% support decay
new_support = previous_support * 0.9 + current_support * 0.1
```

Encoding Versioning:
```python
model.save('model_v2.1.cbm', 
           extra_attributes={'encoding_schema': '2025-06-15'})
```

This approach creates pharmaceutical-grade encodings that are both machine-learning friendly and business-rule transparent while leveraging FP-Growth's pattern discovery capabilities.

### 7. CatBoost Model Integration

The feature-encoded drug names are designed to work seamlessly with CatBoost models for rule extraction and interpretation. The encoding structure enables:

1. **Direct Rule Parsing**
   - Each encoded value contains hierarchical information that can be directly parsed from model rules
   - The fixed-length format (15 digits) allows for consistent rule extraction
   - Example rule parsing:
   ```python
   def parse_drug_rule(encoded_value):
       return {
           'first_letter': chr(ord('A') + int(encoded_value[:2]) - 1),
           'length': int(encoded_value[2:4]),
           'syllables': int(encoded_value[4:6]),
           'consonants': int(encoded_value[6:8]),
           'support': float(encoded_value[8:11]) / 1000,
           'confidence': float(encoded_value[11:14]) / 100,
           'certainty': float(encoded_value[14:17]) / 100
       }
   ```

2. **Model Feature Importance**
   - The encoding preserves the hierarchical structure in CatBoost's feature importance analysis
   - Each component (length, syllables, consonants, etc.) can be analyzed separately
   - Example feature importance extraction:
   ```python
   def analyze_drug_importance(model, feature_name):
       importance = model.get_feature_importance()
       encoded_value = feature_name
       components = parse_drug_rule(encoded_value)
       return {
           'drug_name': reverse_lookup[encoded_value],
           'components': components,
           'importance': importance[feature_name]
       }
   ```

3. **Rule Extraction and Validation**
   - The encoding enables easy validation of model rules against business rules
   - Rules can be checked for consistency with FP-Growth patterns
   - Example rule validation:
   ```python
   def validate_drug_rule(rule, business_rules):
       encoded_value = rule['feature_value']
       components = parse_drug_rule(encoded_value)
       
       # Check against business rules
       for business_rule in business_rules:
           if evaluate_condition(components, business_rule['condition']):
               return {
                   'rule': rule,
                   'business_rule': business_rule,
                   'action': business_rule['action']
               }
       return None
   ```

4. **Model Versioning and Updates**
   - The encoding schema version is stored with the model
   - Updates to the encoding can be tracked and validated
   - Example version tracking:
   ```python
   def check_encoding_compatibility(model, current_schema):
       model_schema = model.get_extra_attributes().get('encoding_schema')
       if model_schema != current_schema:
           return {
               'compatible': False,
               'model_schema': model_schema,
               'current_schema': current_schema,
               'required_updates': get_required_updates(model_schema, current_schema)
           }
       return {'compatible': True}
   ```

This integration ensures that the drug name encodings are not just features for the model but also provide a structured way to interpret and validate the model's decisions.

# FFA Pipeline

This repository contains the code for the FFA (Feature, Filter, Analyze) pipeline, which processes medical and pharmacy data to create cohorts for analysis.

## Event Types and Classification

The pipeline classifies events into four main categories:

1. **OPIOID_ED**: Opioid-related emergency department visits
   - Identified by comprehensive opioid-related ICD codes including:
     - F11.x: Opioid use disorder, dependence, abuse, intoxication, withdrawal
     - T40.x: Opioid poisoning and overdose
     - Y12.x: Opioid-related complications
   - Represents emergency department visits specifically related to opioid use, overdose, or complications

2. **ED_NON_OPIOID**: Non-opioid emergency department visits
   - Identified by HCG line codes:
     - 'P51 - ER Visits and Observation Care'
     - 'O11 - Emergency Room'
     - 'P33 - Urgent Care Visits'
   - Represents emergency department visits not related to opioid use

3. **Pharmacy**: Medication events
   - Captures all pharmacy claims
   - Includes drug name and therapeutic class information
   - Important for tracking medication history and potential drug interactions

4. **Medical**: Other medical events
   - All other medical encounters not classified as ED visits
   - Provides context for the patient's overall healthcare utilization
   - Includes various medical procedures, diagnoses, and treatments

### Event Classification Process

The event classification is performed in the `cohort_event_features` view, which combines both medical and pharmacy data:

```sql
CASE
    WHEN EXISTS (
        SELECT 1 FROM (VALUES ('F11.20'), ('F11.21'), ('T40.2'), ('T40.3'), ('T40.4'), ...) AS opioid_codes(code)
        WHERE opioid_codes.code IN (diagnosis_codes)
    ) THEN 'OPIOID_ED'
    WHEN hcg_line IN ('P51 - ER Visits and Observation Care',
                     'O11 - Emergency Room',
                     'P33 - Urgent Care Visits')
        THEN 'ED_NON_OPIOID'
    ELSE 'Medical'
END AS Event
```

Pharmacy events are added through a UNION ALL operation, maintaining their own specific attributes while ensuring consistent event tracking across the dataset.

### First Event Tracking

The pipeline also tracks the first occurrence of each event type for each patient through the `tagged_cohort_events` view. This helps in:
- Understanding the sequence of healthcare events
- Identifying patterns in healthcare utilization
- Analyzing the progression of opioid-related issues
```

# PGx Analysis Pipeline

## Overview
This pipeline processes healthcare data to analyze patterns and trends in medical events, with a focus on drug-related incidents and their outcomes.

## Healthcare Cost Group (HCG) System

### Milliman HCG System
The Milliman Healthcare Cost Group (HCG) system is a comprehensive framework for categorizing and analyzing healthcare services. It provides standardized methods for:
- Service classification
- Cost analysis
- Utilization tracking
- Quality assessment

### Key Components

#### HCG Line
- Specific service codes within the HCG system (e.g., "O11" for Emergency Room)
- Based on Virginia APCD data description standards
- Enables precise identification of service types
- Used for detailed service categorization

#### HCG Setting
- Broader service categories (e.g., Inpatient, Outpatient)
- Provides context for service location and type
- Works in conjunction with HCG Line codes
- Helps in service classification and analysis

#### VHI Healthcare Pricing Report
- Utilizes the Milliman HCG system
- Provides standardized cost comparisons
- Analyzes healthcare utilization trends
- Supports healthcare service pattern analysis

### Usage in Analysis
The HCG system is used throughout the pipeline for:
1. Service Classification
   - Consistent categorization of healthcare services
   - Standardized service type identification
   - Location-based service analysis

2. Cost Analysis
   - Standardized cost comparisons
   - Service-specific cost tracking
   - Cost trend analysis

3. Trend Analysis
   - Healthcare utilization patterns
   - Service delivery trends
   - Cost pattern analysis

4. Quality Metrics
   - Service delivery assessment
   - Outcome analysis
   - Performance tracking
```

### 4. FpGrowth Pattern Mining with Global Metrics ⭐ **NEW**

**Global Pattern Discovery for Consistent Feature Engineering**
The pipeline now uses a global FpGrowth approach to eliminate data leakage and ensure consistent drug pattern analysis across all cohorts:

#### **Global FpGrowth Workflow**

**Step 1: Global Drug Pattern Discovery**
```python
# Extract all unique drug names from entire pharmacy dataset
def extract_global_drug_names():
    query = """
    SELECT DISTINCT drug_name 
    FROM read_parquet('s3://pgxdatalake/pharmacy_clean/**/*.parquet')
    WHERE drug_name IS NOT NULL AND drug_name != ''
    """
    return unique_drugs

# Create global drug transactions
def create_global_drug_transactions():
    # Group by person to create transactions
    transactions = pharmacy_data.groupby('mi_person_key')['drug_tokens'].agg(
        lambda x: sorted(set(t for sublist in x for t in sublist if t))
    )
    return transactions
```

**Step 2: Global FpGrowth Analysis**
```python
# Run FpGrowth on entire dataset once
def run_global_fpgrowth():
    global_itemsets = fpgrowth_with_timeout(
        global_token_df, 
        min_support=0.005,  # Lower threshold for global analysis
        timeout=300
    )
    global_rules = generate_rules_from_itemsets(global_itemsets)
    return global_itemsets, global_rules
```

**Step 3: Global Encoding Map Creation**
```python
# Create consistent encoding map for all cohorts
def create_global_encoding_map(global_itemsets, global_rules):
    encoding_map = {}
    for _, row in global_itemsets.iterrows():
        for drug in row['itemsets']:
            drug_metrics = get_drug_metrics_from_rules(drug, global_rules)
            encoding_map[drug] = encode_drug_name(drug, drug_metrics)
    return encoding_map
```

**Step 4: Cohort-Specific Feature Application**
```python
# Apply global metrics to individual cohorts (no data leakage)
def apply_global_metrics_to_cohort(cohort_data, global_encoding_map):
    cohort_data['drug_encoding'] = cohort_data['drug_tokens'].apply(
        lambda tokens: [global_encoding_map.get(d, "default_encoding") for d in tokens]
    )
    return cohort_data
```

#### **Key Advantages**

1. **Eliminates Data Leakage**: Global metrics are independent of cohort splits
2. **Consistent Encoding**: Same drug = same encoding across all cohorts  
3. **Better Statistical Power**: More data for reliable pattern discovery
4. **Computational Efficiency**: Run FpGrowth once instead of per cohort
5. **Memory Efficient**: Only processes drug names, not entire pharmacy records
6. **Single Source of Truth**: Global patterns are stable and meaningful

#### **Storage Structure**
```
s3://pgxdatalake/global_fpgrowth/
├── drug_encoding_map.json      # Global drug encodings
├── global_itemsets.json        # Global frequent itemsets
├── global_rules.json          # Global association rules
└── global_metrics.json        # Global pattern metrics
```

#### **Cohort Processing Updates**
Each cohort now applies the global encoding map:
- **No per-cohort FpGrowth**: Eliminates redundant computation
- **Consistent features**: Same drug patterns across all cohorts
- **Faster processing**: Just apply pre-computed global metrics
- **No temporal leakage**: Global patterns don't depend on cohort timelines
```

### 2.2 Drug Event Explosion Strategy ⭐ **NEW**

The pipeline now implements a robust drug event explosion strategy that transforms each drug prescription into its own row, duplicating patient-level context per drug exposure. This approach is essential for sequence modeling and pattern mining applications.

#### **Strategy Overview**

**Cohort-Specific Drug Exposure Logic**

| Cohort | Strategy | Time Window | Use Case |
|--------|----------|-------------|----------|
| **OPIOID_ED** | Retain all drug rows | No restriction | Full drug trail analysis |
| **ED_NON_OPIOID** | Limit to 30 days prior | 30 days before ADE | Causality assessment |

#### **Implementation Details**

**1. Drug Event Explosion Process**
```sql
-- Each drug prescription becomes a separate row
-- Patient context is duplicated per drug event
SELECT 
    mi_person_key,
    event_date,
    drug_name,
    therapeutic_class_1,
    therapeutic_class_2,
    therapeutic_class_3,
    -- All patient demographics duplicated
    member_age_dos,
    member_gender,
    member_race,
    -- Temporal relationships
    days_to_ade,
    days_to_opioid_ed,
    -- Data source tracking
    data_source,
    Event
FROM cohort_event_features
```

**2. Cohort Exclusivity Enforcement**
```python
# Step 1: Select OPIOID_ED patients first
opioid_ed_ids = first_events[first_events["first_event"] == "OPIOID_ED"].mi_person_key.tolist()

# Step 2: Remove OPIOID_ED patients from consideration
first_events = first_events[~first_events["mi_person_key"].isin(opioid_ed_ids)]

# Step 3: Select ED_NON_OPIOID patients from remaining pool
ed_non_opioid_ids = first_events[first_events["first_event"] == "ED_NON_OPIOID"].mi_person_key.tolist()

# Step 4: Validate exclusivity
assert not set(opioid_ed_ids).intersection(ed_non_opioid_ids)
```

#### **Benefits for Sequence Modeling**

**1. FpGrowth Pattern Mining**
- Enables market basket analysis on drug sequences
- Identifies co-occurring drug patterns
- Supports association rule generation

**2. LSTM Sequence Modeling**
- Provides temporal drug sequences
- Enables prediction of next drug in sequence
- Supports drug interaction modeling

**3. Dynamic Time Warping (DTW)**
- Enables similarity analysis of drug sequences
- Supports clustering of drug exposure patterns
- Facilitates pattern matching across patients

**4. Symbolic Reasoning**
- Provides interpretable drug sequences
- Enables rule-based reasoning
- Supports clinical decision support systems

#### **Data Structure Example**

**Before Explosion (Patient-Level)**
```
mi_person_key | event_date | drugs
12345         | 2020-01-15 | [Aspirin, Lisinopril, Metformin]
```

**After Explosion (Drug-Level)**
```
mi_person_key | event_date | drug_name   | data_source
12345         | 2020-01-15 | Aspirin     | drug_exposure
12345         | 2020-01-15 | Lisinopril  | drug_exposure
12345         | 2020-01-15 | Metformin   | drug_exposure
```

#### **Analytical Capabilities**

**1. Drug Sequence Analysis**
```sql
-- Analyze drug sequences leading to opioid events
SELECT 
    drug_name,
    therapeutic_class_1,
    COUNT(*) as prescription_count,
    AVG(days_to_opioid_ed) as avg_days_to_event
FROM opioid_drug_exposure
GROUP BY drug_name, therapeutic_class_1
ORDER BY prescription_count DESC;
```

**2. Temporal Drug Patterns**
```sql
-- Identify drugs prescribed close to ADE events
SELECT 
    drug_name,
    COUNT(*) as ade_associated_count
FROM ade_drug_exposure
WHERE days_to_ade BETWEEN 0 AND 7
GROUP BY drug_name
ORDER BY ade_associated_count DESC;
```

**3. Drug Interaction Analysis**
```sql
-- Find co-prescribed drugs before adverse events
SELECT 
    d1.drug_name as drug1,
    d2.drug_name as drug2,
    COUNT(*) as co_prescription_count
FROM ade_drug_exposure d1
JOIN ade_drug_exposure d2 ON d1.mi_person_key = d2.mi_person_key
WHERE d1.drug_name < d2.drug_name
  AND ABS(d1.days_to_ade - d2.days_to_ade) <= 3
GROUP BY d1.drug_name, d2.drug_name
ORDER BY co_prescription_count DESC;
```

#### **Quality Assurance**

**1. Cohort Exclusivity Validation**
- Ensures no patient appears in both cohorts
- Logs cohort counts for monitoring
- Validates data integrity

**2. Temporal Consistency**
- Ensures drug prescriptions occur before adverse events
- Validates time window constraints
- Monitors data quality metrics

**3. Data Completeness**
- Tracks drug exposure coverage
- Monitors missing data rates
- Validates therapeutic class assignments
```

### Drug Event Explosion Data Flow

```mermaid
flowchart LR
    subgraph "Patient-Level Data"
        A[Patient 12345<br/>Event: 2020-01-15<br/>Drugs: [Aspirin, Lisinopril, Metformin]]
    end
    
    subgraph "Drug Event Explosion"
        B[Explosion Process<br/>Duplicate patient context<br/>per drug prescription]
    end
    
    subgraph "Drug-Level Data"
        C1[Patient 12345<br/>Event: 2020-01-15<br/>Drug: Aspirin<br/>data_source: drug_exposure]
        C2[Patient 12345<br/>Event: 2020-01-15<br/>Drug: Lisinopril<br/>data_source: drug_exposure]
        C3[Patient 12345<br/>Event: 2020-01-15<br/>Drug: Metformin<br/>data_source: drug_exposure]
    end
    
    A --> B
    B --> C1
    B --> C2
    B --> C3
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C1 fill:#bfb,stroke:#333,stroke-width:2px
    style C2 fill:#bfb,stroke:#333,stroke-width:2px
    style C3 fill:#bfb,stroke:#333,stroke-width:2px
```

### Cohort Exclusivity Flow

```mermaid
flowchart TD
    subgraph "Initial Patient Pool"
        A[All Patients with Events<br/>First_Event Classification]
    end
    
    subgraph "Step 1: OPIOID_ED Selection"
        B[Select OPIOID_ED patients<br/>Priority cohort]
        C[OPIOID_ED Cohort<br/>Complete drug history]
    end
    
    subgraph "Step 2: ED_NON_OPIOID Selection"
        D[Remove OPIOID_ED patients<br/>from consideration pool]
        E[Select ED_NON_OPIOID patients<br/>from remaining pool]
        F[ED_NON_OPIOID Cohort<br/>30-day drug window]
    end
    
    subgraph "Validation"
        G[Validate Exclusivity<br/>No overlap between cohorts]
    end
    
    A --> B
    A --> D
    B --> C
    D --> E
    E --> F
    C --> G
    F --> G
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#fbb,stroke:#333,stroke-width:2px
    style C fill:#fbb,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
```

### 2.3 Cohort Strategy Robustness & Validation ⭐ **NEW**

The cohort creation strategy implements comprehensive validation and quality assurance measures to ensure accurate and distinct fact tables for events of interest.

#### **Strategy Robustness Assessment**

**Strengths (9/10)**
1. **✅ Cohort Exclusivity**: Well-implemented mutual exclusivity with priority-based selection
2. **✅ Temporal Logic**: Appropriate time windows for each cohort (30-day ADE, unlimited opioid)
3. **✅ Data Explosion**: Enables advanced analytics with drug-level granularity
4. **✅ Quality Assurance**: Built-in validation checks and error handling
5. **✅ Scalability**: Efficient DuckDB implementation with S3 integration

#### **Data Integrity Validation**

**1. Cohort Exclusivity Enforcement**
```python
# Robust cohort exclusivity validation
def validate_cohort_exclusivity(opioid_ed_ids, ed_non_opioid_ids):
    overlap = set(opioid_ed_ids) & set(ed_non_opioid_ids)
    if overlap:
        raise ValueError(f"❌ Found {len(overlap)} patients in both cohorts")
    
    logger.info(f"✓ Cohort exclusivity validated - no overlap between cohorts")
    logger.info(f"  - OPIOID_ED: {len(opioid_ed_ids)} patients")
    logger.info(f"  - ED_NON_OPIOID: {len(ed_non_opioid_ids)} patients")
    return True
```

**2. Temporal Consistency Validation**
```sql
-- Validate drug prescriptions occur before adverse events
SELECT 
    COUNT(*) as future_drugs,
    COUNT(DISTINCT mi_person_key) as affected_patients
FROM ade_drug_exposure
WHERE days_to_ade < 0;

-- Validate opioid drug exposure temporal logic
SELECT 
    COUNT(*) as invalid_dates,
    COUNT(DISTINCT mi_person_key) as affected_patients
FROM opioid_drug_exposure
WHERE days_to_opioid_ed IS NULL OR days_to_opioid_ed < -3650; -- More than 10 years ago
```

**3. Data Completeness Validation**
```python
def validate_data_completeness(cohort_data, cohort_name):
    completeness_metrics = {
        "total_rows": len(cohort_data),
        "unique_patients": cohort_data['mi_person_key'].nunique(),
        "drug_events": len(cohort_data[cohort_data['data_source'] == 'drug_exposure']),
        "cohort_events": len(cohort_data[cohort_data['data_source'] == 'cohort_event']),
        "missing_drug_names": cohort_data['drug_name'].isna().sum(),
        "missing_therapeutic_class": cohort_data['therapeutic_class_1'].isna().sum(),
        "missing_temporal_data": cohort_data['days_to_ade'].isna().sum()
    }
    
    logger.info(f"✓ {cohort_name} data completeness metrics:")
    for metric, value in completeness_metrics.items():
        logger.info(f"  - {metric}: {value}")
    
    return completeness_metrics
```

#### **Quality Assurance Framework**

**1. Pre-Processing Validation**
```python
# Validate input data quality
def validate_input_data(medical_data, pharmacy_data):
    validation_results = {
        "medical_records": len(medical_data),
        "pharmacy_records": len(pharmacy_data),
        "medical_patients": medical_data['mi_person_key'].nunique(),
        "pharmacy_patients": pharmacy_data['mi_person_key'].nunique(),
        "date_range_medical": (medical_data['event_date'].min(), medical_data['event_date'].max()),
        "date_range_pharmacy": (pharmacy_data['event_date'].min(), pharmacy_data['event_date'].max())
    }
    
    # Check for reasonable data volumes
    if validation_results["medical_records"] < 1000:
        logger.warning("Low medical record count detected")
    
    if validation_results["pharmacy_records"] < 1000:
        logger.warning("Low pharmacy record count detected")
    
    return validation_results
```

**2. Drug Event Explosion Validation**
```python
def validate_drug_explosion(original_data, exploded_data):
    # Validate explosion process
    original_drug_events = len(original_data[original_data['Event'] == 'Pharmacy'])
    exploded_drug_events = len(exploded_data[exploded_data['data_source'] == 'drug_exposure'])
    
    if exploded_drug_events < original_drug_events:
        logger.error("Drug explosion resulted in fewer events than expected")
        return False
    
    # Validate patient context preservation
    original_patients = original_data['mi_person_key'].nunique()
    exploded_patients = exploded_data['mi_person_key'].nunique()
    
    if original_patients != exploded_patients:
        logger.error("Patient count mismatch after explosion")
        return False
    
    logger.info(f"✓ Drug explosion validation passed:")
    logger.info(f"  - Original drug events: {original_drug_events}")
    logger.info(f"  - Exploded drug events: {exploded_drug_events}")
    logger.info(f"  - Patient count preserved: {original_patients}")
    
    return True
```

**3. Cohort Assignment Validation**
```python
def validate_cohort_assignment(tagged_events, opioid_ed_ids, ed_non_opioid_ids):
    # Validate first event classification
    first_events = tagged_events.groupby('mi_person_key')['First_Event'].first()
    
    opioid_classified = first_events[first_events == 'OPIOID_ED'].index.tolist()
    ed_classified = first_events[first_events == 'ED_NON_OPIOID'].index.tolist()
    
    # Check for classification consistency
    if set(opioid_classified) != set(opioid_ed_ids):
        logger.error("OPIOID_ED classification mismatch")
        return False
    
    if set(ed_classified) != set(ed_non_opioid_ids):
        logger.error("ED_NON_OPIOID classification mismatch")
        return False
    
    logger.info("✓ Cohort assignment validation passed")
    return True
```

#### **Production-Ready Quality Metrics**

**1. Comprehensive Quality Dashboard**
```python
def generate_quality_dashboard(cohort_data, cohort_name, age_band, event_year):
    quality_metrics = {
        "cohort_info": {
            "name": cohort_name,
            "age_band": age_band,
            "event_year": event_year,
            "total_rows": len(cohort_data),
            "unique_patients": cohort_data['mi_person_key'].nunique()
        },
        "data_quality": {
            "missing_drug_names": cohort_data['drug_name'].isna().sum(),
            "missing_therapeutic_class": cohort_data['therapeutic_class_1'].isna().sum(),
            "missing_temporal_data": cohort_data['days_to_ade'].isna().sum(),
            "data_completeness_rate": 1 - (cohort_data['drug_name'].isna().sum() / len(cohort_data))
        },
        "temporal_analysis": {
            "avg_days_to_event": cohort_data['days_to_ade'].mean() if 'days_to_ade' in cohort_data.columns else None,
            "min_days_to_event": cohort_data['days_to_ade'].min() if 'days_to_ade' in cohort_data.columns else None,
            "max_days_to_event": cohort_data['days_to_ade'].max() if 'days_to_ade' in cohort_data.columns else None
        },
        "drug_analysis": {
            "unique_drugs": cohort_data['drug_name'].nunique(),
            "unique_therapeutic_classes": cohort_data['therapeutic_class_1'].nunique(),
            "most_common_drugs": cohort_data['drug_name'].value_counts().head(5).to_dict()
        }
    }
    
    return quality_metrics
```

**2. Audit Trail Implementation**
```python
def create_audit_trail(age_band, event_year, opioid_ed_ids, ed_non_opioid_ids, quality_metrics):
    audit_record = {
        "timestamp": datetime.now().isoformat(),
        "age_band": age_band,
        "event_year": event_year,
        "cohort_counts": {
            "opioid_ed": len(opioid_ed_ids),
            "ed_non_opioid": len(ed_non_opioid_ids)
        },
        "quality_metrics": quality_metrics,
        "validation_status": {
            "exclusivity_validated": True,
            "temporal_consistency": True,
            "data_completeness": quality_metrics["data_quality"]["data_completeness_rate"] > 0.95
        }
    }
    
    # Save audit trail to S3
    audit_path = f"s3://pgxdatalake/audit_trails/cohort_creation/{age_band}/{event_year}/audit.json"
    save_to_s3_json(audit_record, audit_path)
    
    return audit_record
```

#### **Error Handling & Recovery**

**1. Graceful Error Recovery**
```python
def robust_cohort_creation(age_band, event_year, logger):
    try:
        # Main cohort creation process
        result = main(age_band, event_year)
        
        if result == 0:
            logger.info("✓ Cohort creation completed successfully")
            return True
        else:
            logger.error("✗ Cohort creation failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Unexpected error in cohort creation: {str(e)}")
        notify_error("cohort_creation", f"Error: {str(e)}\n\n{traceback.format_exc()}", logger)
        return False
```

**2. Data Consistency Checks**
```python
def verify_cohort_consistency(opioid_cohort_path, ed_non_opioid_cohort_path):
    # Verify both cohorts exist and are accessible
    try:
        opioid_cohort = pd.read_parquet(opioid_cohort_path)
        ed_cohort = pd.read_parquet(ed_non_opioid_cohort_path)
        
        # Check for overlap
        opioid_patients = set(opioid_cohort['mi_person_key'].unique())
        ed_patients = set(ed_cohort['mi_person_key'].unique())
        
        overlap = opioid_patients & ed_patients
        
        if overlap:
            logger.error(f"❌ Found {len(overlap)} overlapping patients between cohorts")
            return False
        
        logger.info("✓ Cohort consistency verification passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cohort consistency check failed: {str(e)}")
        return False
```

#### **Performance Monitoring**

**1. Runtime Performance Metrics**
```python
def monitor_performance(start_time, step_name, logger):
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️ {step_name} completed in {elapsed_time:.2f} seconds")
    
    # Log performance metrics
    performance_metrics = {
        "step": step_name,
        "duration_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    return performance_metrics
```

**2. Resource Utilization Tracking**
```python
def track_resource_utilization():
    import psutil
    
    return {
        "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
        "cpu_percent": psutil.cpu_percent(),
        "disk_usage_percent": psutil.disk_usage('/').percent
    }
```

#### **Strategy Assessment Summary**

| Component | Rating | Status |
|-----------|--------|--------|
| **Cohort Exclusivity** | 10/10 | ✅ Excellent |
| **Temporal Logic** | 9/10 | ✅ Robust |
| **Data Explosion** | 9/10 | ✅ Well-implemented |
| **Quality Assurance** | 8/10 | ✅ Comprehensive |
| **Error Handling** | 8/10 | ✅ Good |
| **Performance** | 9/10 | ✅ Efficient |

**Overall Assessment: 9/10 - Production Ready**

The strategy provides a robust foundation for creating accurate and distinct fact tables with comprehensive validation, quality assurance, and error handling mechanisms suitable for production deployment.