---
editor_options: 
  markdown: 
    wrap: 72
---

# Prescription Drug Analysis with BupaR and CatBoost Integration

## Executive Summary

``` mermaid
flowchart TD
    subgraph "Data Processing"
        A[Pharmacy Data] --> B[Clean Pharmacy Data]
        C[Medical Data] --> D[Process Medical Data]
        B --> E[Create Cohorts]
        D --> E
    end

    subgraph "Data Filtering"
        E --> F1[Richmond Zip Codes]
        F1 --> F2[ICD/HCG Code Filtering]
        F2 --> F3[Age Band Filtering]
        F3 --> F4[Year Filtering 2016-2019]
    end

    subgraph "Cohort Creation"
        F4 --> G1[Opioid_ED Cohort]
        F4 --> G2[ED_Non_Opioid Cohort]
        G1 --> G3[Age Band 44-55]
        G2 --> G3
    end

    subgraph "Feature Engineering"
        G3 --> H[Network Feature Extraction]
        H --> I[FP-Growth Pattern Mining]
        I --> J[Association Rules]
    end

    subgraph "Modeling & Analysis"
        J --> K[CatBoost Model Training]
        K --> L[Feature Importance Analysis]
        L --> M[Process Mining with BupaR]
    end

    subgraph "Visualization & Output"
        M --> N[Network Visualization]
        M --> O[Process Maps]
        K --> P[Risk Predictions]
        L --> Q[Feature Importance Plots]
    end

    subgraph "Validation & Metrics"
        N --> R[Performance Metrics]
        O --> R
        P --> R
        Q --> R
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style F1 fill:#bbf,stroke:#333,stroke-width:2px
    style F2 fill:#bbf,stroke:#333,stroke-width:2px
    style F3 fill:#bbf,stroke:#333,stroke-width:2px
    style F4 fill:#bbf,stroke:#333,stroke-width:2px
    style G1 fill:#bfb,stroke:#333,stroke-width:2px
    style G2 fill:#bfb,stroke:#333,stroke-width:2px
    style G3 fill:#bfb,stroke:#333,stroke-width:2px
    style K fill:#fbb,stroke:#333,stroke-width:2px
    style M fill:#bfb,stroke:#333,stroke-width:2px
    style R fill:#fbb,stroke:#333,stroke-width:2px
```

Enhanced pipeline for analyzing prescription drug patterns with
hospitalization risk using process mining, survival analysis, and
machine learning, with initial focus on the 44-55 age band for early ADE
risk identification.

## Strategic Focus: 44-55 Age Band Analysis

### Rationale for Initial Focus

The pipeline initially focuses on the 44-55 age band for ADE analysis
based on several key factors:

1.  **Early Risk Identification**
    -   Enables identification of risk factors before individuals reach
        highest-risk older age groups
    -   Provides opportunity for preventive interventions
    -   Establishes baseline patterns for comparative analysis with
        older cohorts
2.  **Methodological Advantages**
    -   Sufficient event counts for robust modeling
    -   Clean separation from geriatric confounders
    -   Clear progression patterns for risk factor analysis
3.  **Clinical Relevance**
    -   ADE risk begins to increase in middle age
    -   Patterns identified can inform preventive strategies
    -   Findings can guide interventions before highest-risk periods

### Age Band Strategy

The pipeline processes data across multiple age bands, with special
attention to: - Primary focus: 44-55 age band for initial analysis -
Secondary analysis: 56-65 and 65+ age bands for validation - Comparative
analysis across age bands to understand risk progression

## Enhanced Analysis Workflow Architecture

### Core Components

**1. FP-Growth Pattern Mining Layer**\
- Implements market basket analysis on medication sequences to identify
initial feature importances - Identifies co-occurring prescriptions
using minimum support thresholds (default: 0.05 for initial pattern
discovery) - Discovers significant event patterns that feed into both: -
BupaR process mining for temporal analysis - CatBoost models for
predictive modeling - Filters patterns based on: - Minimum support
threshold - Pattern frequency in positive vs negative samples - Clinical
relevance of co-occurring events

``` python
from mlxtend.frequent_patterns import fpgrowth
frequent_itemsets = fpgrowth(log_matrix, min_support=0.1, use_colnames=True)
```

**2. BupaR Process Mining Engine**\
- Uses FP-Growth identified patterns to construct event logs using
`mi_person_key` as case identifier - Performs temporal analysis through:

``` r
process_map(eventlog, type = frequency("relative-consequent"))
```

-   Identifies hospitalization precursor patterns
-   Calculates throughput times between drug administrations
-   Validates patterns through:
    -   Process conformance checking
    -   Trace alignment analysis
    -   Performance metrics evaluation

**3. CatBoost Predictive Modeling**\
- Incorporates FP-Growth discovered patterns as network features - Uses
Formal Feature Attribution (FFA) for feature importance analysis:

``` python
model.get_feature_importance(type='PredictionValuesChange')
```

-   Implements temporal cross-validation for cohort-based forecasting
-   Validates feature importance through:
    -   Cross-validation stability
    -   Statistical significance testing
    -   Clinical relevance assessment

**4. FFA-based importance ranking** - Uses FFA to rank features by their
importance in predicting hospitalization risk - Identifies top K
important features based on: - Support and coverage thresholds -
Statistical significance testing - Class-specific importance rankings -
Cross-validation stability

## Updated Implementation Workflow

### Phase 1: Current Data Optimization

1.  **Data Structuring**\

-   Create annual cohorts with explicit censoring dates (Dec 31 cutoff)
-   Maintain observation windows with survival analysis

2.  **Pattern Mining**\

-   Use `processmapR::dotted_chart()` for visual comparison
-   Apply `sequence_analysis()` with gap penalty parameters

3.  **Model Integration**\

``` r
library(catboost)
pool <- catboost.load_pool(
data = feature_set,
label = "hospitalization_flag",
cat_features = which(sapply(feature_set, is.character))
)
```

## Multimodal Data Architecture

### Current Data Pillars

| Data Type    | Integration Method       | Analysis Technique       |
|--------------|--------------------------|--------------------------|
| Demographics | DuckDB columnar storage  | Cox proportional hazards |
| Age Cohorts  | Temporal stratification  | Survival analysis        |
| Medication   | FP-Growth itemset mining | Market basket analysis   |

### Future Data Integration

**1. Genotype-Environment Interaction**\
- Planned implementation of GEI models:

``` python
CatBoost(params={'interaction_config': 'Genotype:Environment'})
```

**2. Behavioral & Family History**\
- Multi-modal fusion using:

``` r
bupaR::merge_logs(clinical, behavioral, by="mi_person_key")
```

# Scripts Overview

### 1. Pharmacy Data Cleaning (`clean_pharmacy.py`)

This script normalizes pharmacy data and prepares it for use in medical
data processing. It performs the following operations:

1.  Loads pharmacy data from S3
2.  Processes the data through several stages:
    -   `pharmacy_normalized`: Raw data with parsed event dates
    -   `pharmacy_augmented`: Added age bands and event year
    -   `pharmacy_standardized`: Standardized drug names
    -   `pharmacy_cleaned`: Applied drug name mappings and excluded
        medical supplies

#### Usage

``` bash
python clean_pharmacy.py --age-band "0-12" --event-year 2020
```

#### Output

Creates partitioned Parquet files in S3:

```         
s3://pgxdatalake/pharmacy-clean-drug-names/age_band={age_band}/event_year={event_year}/clean.parquet
```

### 2. Medical Data Processing and Cohort Creation (`create_cohorts.py`)

This script processes medical data and creates cohorts using DuckDB. It
performs the following operations:

1.  Loads medical data from S3
2.  Processes the data through several stages:
    -   `medical`: Raw data with parsed event dates
    -   `medical_augmented`: Added age bands and event year
    -   `medical_normalized`: Renamed columns based on ICD mappings
    -   `medical_filtered`: Selected specific columns
    -   `medical_features`: Added age imputation from pharmacy data
3.  Creates a unified timeline view combining medical and pharmacy data
4.  Creates separate sampled cohorts for opioid and non-opioid models

#### Usage

``` bash
python create_cohorts.py --age-band "0-12" --event-year 2020 --cohort opioid_ed
```

#### Output

Creates partitioned Parquet files in S3:

```         
s3://pgxdatalake/cohorts/opioid_ed/age_band={age_band}/event_year={event_year}/cohort.parquet
s3://pgxdatalake/cohorts/ed_non_opioid/age_band={age_band}/event_year={event_year}/cohort.parquet
```

### 3. Network Feature Engineering (`feature_engineer_cohort_network.py`)

This script analyzes cohort data to extract network features from
medical events and medications. It performs the following operations:

1.  Loads cohort data for a specific age band and event year
2.  Extracts tokens from diagnosis codes, procedure codes, and drug
    names
3.  Performs market basket analysis using FP-Growth algorithm:
    -   Identifies frequent itemsets in positive and negative samples
        (using 0.05 min_support threshold)
    -   Extracts patterns that appear only in positive samples
    -   Creates feature flags based on these patterns
4.  Generates enhanced datasets with network features
5.  Saves itemsets and association rules for further analysis

#### Usage

``` bash
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

### Important Note on Network Visualization Interpretation

When interpreting the network visualizations generated from association
rules, it's crucial to understand the distinction between correlation
and causation:

❗ **Short Answer:** The network visualizations show correlated patterns
of co-occurrence, not directional influence or causal flow.

🔍 **Why?** Association rules (like those from FpGrowth) represent: -
Statistical co-occurrence between items in transactions - E.g., "If drug
A is present, drug B is also often present"

But they don't establish causality because they: - Don't control for
confounding variables - Don't establish temporal precedence - Don't use
interventions to test effects

So while an arrow A → B is drawn (based on a rule), this is not a causal
arrow — it represents a conditional probability relationship:

```         
P(B | A) is high → draw A → B
```

✅ **Why the Visualization Is Still Valuable** Even without causality: -
Summing support across multiple co-occurrence paths gives a meaningful
measure of total association weight - Directionality reflects rule
direction (not causal flow) - Node centrality indicates clustering or
"hub" drugs often present in many co-occurrence patterns - Edge
thickness communicates real signal strength in the data

🧠 **Bottom Line** \| Interpretation \| Is Valid? \| Explanation \|
\|----------------\|-----------\|-------------\| \| A → B is causal \|
❌ \| FpGrowth doesn't model interventions \| \| A → B co-occur often \|
✅ \| Based on high confidence/support \| \| Thicker edge = more total
co-occurrence \| ✅ \| Sum(support) reflects total influence \|

### 4. Feature Importance to BupaR Analysis (`feature_importance_bupaR.py`)

This script analyzes the feature importance results from CatBoost models
and prepares data for process mining analysis. It performs the following
operations:

1.  Loads feature importance metrics from CatBoost FFA analysis:
    -   Support: Frequency of feature occurrence in important patterns
    -   Coverage: Proportion of cases explained by the feature
    -   Significance: Statistical significance of feature importance
    -   Class-specific metrics for both target and control groups
2.  Identifies top K important features based on:
    -   Support and coverage thresholds
    -   Statistical significance testing
    -   Class-specific importance rankings
    -   Cross-validation stability
3.  Filters training data to include only important features:
    -   Maintains temporal relationships
    -   Preserves case identifiers
    -   Retains activity and lifecycle information
    -   Ensures data quality and completeness
4.  Prepares data for BupaR analysis by:
    -   Retaining timestamps and case IDs
    -   Formatting activity and lifecycle information
    -   Separating target and control groups
    -   Creating event logs for process mining
    -   Ensuring proper temporal ordering
5.  Saves prepared data for process mining analysis:
    -   Parquet format for efficient storage
    -   Separate files for target and control groups
    -   Metadata for analysis configuration
    -   Validation metrics and summaries

#### Usage

``` bash
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

1.  **Process Maps**
    -   Frequency-based process maps
    -   Relative consequent process maps
    -   Performance-based process maps
    -   Custom node and edge metrics
2.  **Trace Analysis**
    -   Trace frequency analysis
    -   Trace length distribution
    -   Trace variant analysis
    -   Custom trace metrics
3.  **Performance Analysis**
    -   Throughput time analysis
    -   Resource utilization
    -   Activity frequency
    -   Bottleneck detection
4.  **Comparative Analysis**
    -   Target vs. control group comparison
    -   Age band comparisons
    -   Temporal pattern analysis
    -   Statistical significance testing
5.  **Conformance Analysis**
    -   Process conformance checking
    -   Fitness analysis
    -   Precision analysis
    -   Alignment-based metrics

#### Visualization Examples

The analysis generates various visualizations:

1.  **Process Maps**

    ``` r
    # Frequency-based process map
    process_map(eventlog, type = frequency(value = "relative-consequent"))

    # Performance-based process map
    process_map(eventlog, type = performance(level = "activity"))
    ```

2.  **Trace Analysis**

    ``` r
    # Trace explorer
    trace_explorer(eventlog, n_traces = 10)

    # Activity presence
    activity_presence(eventlog)
    ```

3.  **Performance Analysis**

    ``` r
    # Throughput time
    throughput_time(eventlog) %>% summary()

    # Resource frequency
    resource_frequency(eventlog)
    ```

4.  **Comparative Analysis**

    ``` r
    # Activity frequency comparison
    activity_frequency(target_eventlog) %>%
      left_join(activity_frequency(control_eventlog), 
               by = "activity", 
               suffix = c("_target", "_control"))
    ```

#### Metrics and Validation

The script includes comprehensive metrics and validation:

1.  **Feature Importance Metrics**
    -   Support and coverage scores
    -   Statistical significance (p-values)
    -   Class-specific importance
    -   Cross-validation stability
2.  **Data Quality Metrics**
    -   Completeness checks
    -   Temporal consistency
    -   Case coverage
    -   Activity coverage
3.  **Process Mining Metrics**
    -   Process map metrics
    -   Trace analysis metrics
    -   Performance metrics
    -   Conformance metrics
4.  **Comparative Metrics**
    -   Group differences
    -   Statistical tests
    -   Effect sizes
    -   Confidence intervals

# Survival analysis integration

``` r
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

### 5. Cohort Verification and Reprocessing (`check_and_reprocess_cohorts.py`)

This script validates created cohorts and reprocesses them if necessary:

1.  Checks cohorts for proper control-to-case ratios (strict 5:1 ratio
    required)
2.  Identifies cohorts with insufficient control samples
3.  Deletes and reprocesses cohorts that don't meet the ratio threshold
4.  Supports flexible path specifications
5.  Verifies control-to-case ratios:
    -   Enforces a strict 5:1 ratio for both opioid and non-opioid
        cohorts
    -   Allows control reuse between cohorts to maintain the 5:1 ratio
    -   Provides detailed information about control sharing
    -   Ensures consistent ratio maintenance even with limited control
        pools

#### Usage

``` bash
python check_and_reprocess_cohorts.py --threshold 5.0
```

### 6. Opioid ED Risk Model Analysis (`opioid_ed_risk_model.qmd`)

This Quarto document performs comprehensive analysis of opioid ED risk
using CatBoost and process mining, with initial focus on the 44-55 age
band. It includes the following steps:

1.  **Data Preparation**
    -   Loads feature-engineered cohort data from S3 using DuckDB
    -   Primary focus on 44-55 age band for initial analysis
    -   Secondary analysis of other age bands:
        -   0-12 years
        -   13-17 years
        -   18-25 years
        -   26-35 years
        -   36-44 years
        -   44-55 years (primary focus)
        -   56-65 years
        -   66+ years
    -   Filters data for years 2016-2019
    -   Removes duplicates and handles missing values
    -   Performs feature selection:
        -   Excludes high cardinality columns
        -   Removes lagging indicators
        -   Filters out unnecessary features
2.  **Feature Processing**
    -   For each age band:
        -   Identifies and processes categorical features:
            -   Adds 'None' category for missing values
            -   Converts to string type
            -   Handles categorical encoding
        -   Processes numerical features:
            -   Identifies numerical columns
            -   Excludes target variable
            -   Maintains data types
3.  **Model Training**
    -   For each age band:
        -   Splits data into train (2016-2018) and test (2019) sets
        -   Configures CatBoost classifier with optimized parameters:
            -   2000 iterations
            -   Depth of 12
            -   Ordered boosting
            -   MVS bootstrap
            -   Early stopping
        -   Trains model with categorical feature support
        -   Uses recall as primary evaluation metric
        -   Saves model to disk
4.  **Feature Importance Analysis**
    -   For each age band:
        -   Calculates Formal Feature Attribution (FFA) values for model
            interpretability
        -   Computes feature importance scores using FFA
        -   Identifies top 20 most important features
        -   Creates feature importance visualization
        -   Saves results to CSV
5.  **Model Evaluation**
    -   For each age band:
        -   Calculates comprehensive metrics:
            -   AUC and AUPR
            -   Brier Score
            -   Accuracy and Log Loss
            -   F1 Score, Precision, and Recall
        -   Generates confusion matrix
        -   Provides detailed performance analysis
        -   Saves metrics to JSON
6.  **Process Mining Analysis**
    -   For each age band:
        -   Converts predictions to event log format
        -   Creates process maps using BupaR
        -   Generates process animations
        -   Analyzes risk distributions
        -   Provides comparative analysis of outcomes
7.  **Results Visualization**
    -   Creates age-band specific visualizations:
        -   Risk distribution plots
        -   Process maps
        -   Interactive visualizations
    -   Generates comparative analysis:
        -   Metrics comparison across age bands
        -   Feature importance comparison
        -   Process pattern comparison
    -   Provides summary statistics by age band

#### Usage

``` bash
quarto render opioid_ed_risk_model.qmd
```

#### Output

Generates an HTML report with: - Age-band specific model performance
metrics - Feature importance analysis for each age band - Process mining
visualizations by age band - Risk distribution analysis - Interactive
process maps - Comparative analysis across age bands

#### Dependencies

-   R packages: bupaR, processmapR, processmonitR, processanimateR
-   Python packages: catboost, shap, pandas, numpy
-   DuckDB for data loading

## Prerequisites

-   Python 3.8 or higher
-   DuckDB with S3 support
-   AWS credentials configured for S3 access

## Installation

1.  Install the required dependencies:

``` bash
pip install -r requirements.txt
```

2.  Configure AWS credentials for S3 access:

``` bash
aws configure
```

## Data Flow

### Pharmacy Data Processing

1.  **Data Loading**
    -   Loads from `s3://pgxdatalake/pharmacy/**/*.parquet`
    -   Parses event dates from incurred_date
2.  **Data Augmentation**
    -   Adds age bands based on member_age_dos
    -   Adds event_year from parsed dates
3.  **Drug Name Standardization**
    -   Converts to lowercase
    -   Removes trailing slashes
    -   Replaces spaces with underscores
    -   Replaces '/' with '+'
    -   Sorts drug combinations alphabetically
4.  **Drug Name Mapping**
    -   Applies mappings from
        `s3://pgxdatalake/drug_mappings/*_mappings.json`
    -   Excludes medical supplies
    -   Standardizes drug names

### Medical Data Processing

1.  **Medical Data Loading and Normalization**
    -   Loads from `s3://pgxdatalake/medical/**/*.parquet`
    -   Parses event dates from incurred_date
    -   Applies ICD mappings from
        `s3://pgxdatalake/claim_mappings/icd_mappings.json`
    -   Adds age bands and event years
    -   Creates `medical_augmented` view
2.  **Column Filtering**
    -   Selects specific columns for analysis
    -   Creates `medical_filtered` view
3.  **Pharmacy Data Preparation**
    -   Creates `pharmacy_augmented` view with basic fields
    -   Creates `pharmacy_cleaned` view with all fields
    -   Maintains consistent schema with medical data
4.  **Age Imputation**
    -   Identifies medical records with missing ages (member_age_dos =
        255) 
    -   Matches with pharmacy records within 365 days
    -   Uses pharmacy age data to fill missing values
    -   Maintains original age if no pharmacy match found
    -   Creates `medical_features` view
5.  **Unified Timeline Creation**
    -   Creates `cohort_features_timeline` view combining medical and
        pharmacy data
    -   Maintains chronological ordering by person and date
    -   Includes all fields from both data sources
    -   Tags events as:
        -   'Pharmacy' for pharmacy events
        -   'OPIOID_ED' for opioid-related ED visits
        -   'ED_NON_OPIOID' for non-opioid ED visits
        -   'Medical' for other medical events
6.  **Person-Level Event Tagging**
    -   Creates `cohort_features_tagged` view
    -   Identifies adverse event cases by person
    -   Creates clean control groups:
        -   People with no opioid or non-opioid ED events
        -   People with no non-opioid or opioid ED events
    -   Maintains person-level consistency in event tagging
7.  **Existing Cohort Check**
    -   Verifies if cohort already exists in S3
    -   Skips processing if found to avoid duplication
8.  **Cohort Sampling**
    -   Creates separate sampled cohorts for opioid and non-opioid
        models
    -   Counts distinct persons in each category:
        -   `OPIOID_ED` events for opioid model
        -   `ED_NON_OPIOID` events for non-opioid model
        -   Non-adverse events for control sampling
    -   Implements strict 5:1 person-level ratio:
        -   Maintains exactly 5 controls per case for each cohort
        -   Reuses controls between cohorts when necessary to maintain
            ratio
        -   Adjusts sampling strategy when control pool is limited
        -   Uses deterministic sampling for reproducibility
    -   Samples controls using window functions:
        -   Assigns controls to both cohorts when needed
        -   Maintains person-level consistency
        -   Ensures proper ratio verification
    -   Assembles final cohorts:
        -   Combines cases (target=1) with sampled controls (target=0)
        -   Preserves all events for each person in chronological order
        -   Creates `sampled_opioid_ed_cohort` and
            `sampled_ed_non_opioid_cohort` views
    -   Verifies proper ratio:
        -   Confirms 5:1 person-level ratio before writing output
        -   Provides detailed logs of actual case and control counts
        -   Ensures cohort integrity for downstream analysis
    -   Adaptive Control Sampling:
        -   Automatically detects when there aren't enough unique
            controls available
        -   Switches to a shared control pool approach when necessary
        -   Applies the same control-to-case ratio to both cohorts to
            maintain fairness
        -   Allows controls to be used in both opioid and non-opioid
            cohorts in limited control scenarios
        -   Adjusts verification thresholds proportionally when shared
            controls are detected
9.  **Output Generation**
    -   Creates two separate cohort files:
        -   `opioid_ed`: For opioid-related adverse events model
        -   `ed_non_opioid`: For non-opioid adverse events model
    -   Each cohort includes its respective adverse events and sampled
        non-adverse events
    -   Writes final cohorts to S3 with consistent partitioning
    -   Saves processing metrics for both models
10. **Feature Importance Analysis**
    -   Loads CatBoost model feature importance results
    -   Identifies significant features using support and coverage
        metrics
    -   Prepares data for process mining:
        -   Filters to important features
        -   Maintains temporal information
        -   Preserves case and activity identifiers
    -   Creates separate datasets for target and control groups
    -   Enables process mining analysis of significant event patterns

## Metrics

Both scripts collect and save metrics for each processing step,
including: - Row counts - Distinct person counts - Age distributions -
Diagnosis and procedure code distributions - Drug and HCG line
distributions - Target class distributions

Metrics are saved as JSON files in:

```         
s3://pgx-repository/pgx-datasets/pipeline_metrics/{age_band}/{event_year}/{cohort}_{timestamp}.json
```

## Data Dependencies

The scripts work together in the following way: 1. `clean_pharmacy.py`
processes pharmacy data and creates standardized drug names 2.
`create_cohorts.py` uses the cleaned pharmacy data for age imputation in
medical records 3. Both scripts maintain consistent age bands and event
years for proper data alignment 4. `feature_importance_bupaR.py` uses
CatBoost model results to prepare data for process mining

------------------------------------------------------------------------

**Note:**\
Make sure your DuckDB installation supports S3 access and is properly
configured with your AWS credentials.

#### Methods or Approach

We developed a comprehensive data processing pipeline using DuckDB for
efficient handling of large-scale healthcare data. The approach
includes:

1.  **Standardized Data Processing**
    -   Automated cleaning and normalization of pharmacy and medical
        claims data
    -   Consistent handling of drug names, diagnosis codes, and
        procedure codes
    -   Age band standardization and event date parsing
    -   FAERS data integration for hospitalization risk window
        calculation
2.  **Advanced Cohort Creation**
    -   Person-level event timeline construction
    -   Sophisticated adverse event identification
    -   Balanced control group sampling using SYSTEM method
    -   Age imputation from multiple data sources
    -   FAERS-based hospitalization risk window definition
3.  **Feature Engineering for High Cardinality Data**
    -   Network analysis of medical events and drug interactions using
        market basket analysis (FP-Growth)
    -   Two-phase approach to temporal analysis:
        1.  Initial pattern discovery with market basket analysis
        2.  Advanced temporal process mining with BupaR for significant
            patterns
    -   Dimensionality reduction for high-cardinality categorical
        variables
    -   Feature aggregation at multiple temporal windows
    -   FAERS-derived hospitalization risk windows
4.  **Advanced Risk Modeling**
    -   CatBoost models for opioid ED event prediction
    -   CatBoost hospitalization risk model with FAERS-validated windows
    -   Formal feature attribution methods for model interpretability
    -   Cross-validation with temporal stratification
    -   FAERS-based risk window calibration for hospitalization
        prediction
5.  **Process Mining Analysis**
    -   Feature importance-based data filtering
    -   Temporal process analysis using BupaR
    -   Comparative analysis of target and control groups
    -   Identification of critical event sequences
    -   Process performance and conformance analysis
    -   Advanced visualization capabilities:
        -   Interactive process maps
        -   Dynamic trace exploration
        -   Performance dashboards
        -   Comparative visualizations
    -   Statistical validation:
        -   Significance testing
        -   Effect size analysis
        -   Confidence intervals
        -   Cross-validation
    -   Custom metrics development:
        -   Process-specific metrics
        -   Group comparison metrics
        -   Performance indicators
        -   Quality measures
6.  **FAERS Integration for Hospitalization Risk**
    -   Mapping of drug names to FAERS terminology
    -   Extraction of hospitalization-related ADE timelines
    -   Risk window calibration based on FAERS hospitalization reports
    -   Validation of hospitalization risk patterns
    -   Integration of FAERS severity metrics for hospitalization
        prediction

#### Principal Findings

The pipeline successfully processes large-scale healthcare data with
several key achievements:

1.  **Data Quality**
    -   Consistent handling of drug names and medical codes
    -   Accurate age imputation from pharmacy records
    -   Maintained data integrity across processing steps
    -   FAERS-validated hospitalization risk windows
2.  **Processing Efficiency**
    -   Efficient handling of large datasets using DuckDB
    -   Optimized sampling methods for balanced cohorts
    -   Scalable architecture supporting multiple age bands and years
    -   Efficient FAERS data integration for hospitalization risk
3.  **Advanced Analytics Results**
    -   Network analysis revealed significant drug interaction patterns
    -   BupaR sequence analysis identified critical event pathways
    -   CatBoost models achieved high predictive accuracy for both ED
        and hospitalization events
    -   Formal feature attribution provided clinically interpretable
        risk factors
    -   FAERS-validated hospitalization risk windows improved prediction
        accuracy
4.  **Research Readiness**
    -   Clean, standardized datasets ready for analysis
    -   Balanced cohorts for both opioid and non-opioid studies
    -   Comprehensive documentation and metrics
    -   FAERS-validated hospitalization risk windows

#### Future Research Opportunities

The pipeline enables several promising research directions:

1.  **Age-Specific Analysis**
    -   Enhanced analysis of 44-55 age band patterns
    -   Comparative studies across age groups
    -   Longitudinal analysis of risk progression
    -   Age-specific intervention strategies
    -   Validation of findings in older cohorts
2.  **Multi-modal Analysis**
    -   Integration of additional data sources
    -   Cross-validation across different data types
    -   Enhanced feature engineering
    -   Network-based feature extraction
    -   Real-time FAERS data integration for hospitalization risk
3.  **Advanced Analytics**
    -   Refinement of CatBoost hospitalization risk models
    -   Development of ensemble methods
    -   Advanced feature attribution techniques
    -   Network-based risk scoring
    -   FAERS-based hospitalization risk window optimization
4.  **Clinical Applications**
    -   Real-time hospitalization risk prediction
    -   Personalized intervention strategies
    -   Dynamic risk assessment
    -   Clinical decision support systems
    -   FAERS-informed hospitalization risk monitoring

#### Lessons Learned

Key insights from developing this large-scale healthcare data pipeline:

1.  **Data Quality Challenges**
    -   Importance of consistent data cleaning
    -   Need for robust error handling
    -   Value of comprehensive validation
    -   Challenges in handling high-cardinality features
    -   Benefits of FAERS data integration for hospitalization risk
2.  **Technical Considerations**
    -   Benefits of using DuckDB for large datasets
    -   Importance of efficient sampling methods
    -   Need for scalable architecture
    -   Value of Formal Feature Attribution over SHAP values
    -   Challenges in FAERS data integration for hospitalization risk
3.  **Analytical Insights**
    -   Effectiveness of network analysis for high-cardinality data
    -   Value of sequence analysis in healthcare events
    -   Importance of model interpretability through FFA
    -   Benefits of CatBoost for hospitalization risk modeling
    -   Significance of FAERS-validated hospitalization risk windows
4.  **Research Impact**
    -   Value of standardized processing
    -   Importance of reproducible methods
    -   Need for comprehensive documentation
    -   Significance of clinically interpretable results
    -   Impact of FAERS integration on hospitalization risk assessment

## Network Features

The model uses network features generated by the FpGrowth algorithm to
identify patterns in medical and pharmacy data. These features are
created by:

1.  Extracting tokens from medical and pharmacy data
2.  Running FpGrowth on positive samples to find frequent patterns
3.  Creating feature flags named `network_feature_0` through
    `network_feature_24` (TOP_K=25)

## Integrated Analysis Approach

Our pipeline integrates pattern mining, process mining, and formal model
explanation into a unified analysis framework:

### Step-by-Step Analysis Flow

1.  **FpGrowth Pattern Mining** (see
    `feature_engineer_cohort_network.py`)
    -   Extract frequent co-occurrence patterns in transactional data
    -   Group by cohort, event type, or year
    -   Example: itemsets of drugs, diagnoses, procedures
    -   Identify significant patterns using support thresholds
2.  **BupaR Process Mining** (see `feature_importance_bupaR.py`)
    -   Convert itemsets into event logs:
        -   Each itemset becomes a trace or partial trace
        -   Attach timestamps, patient ID, and metadata
    -   Analyze with BupaR:
        -   Frequency and precedence analysis
        -   Process conformance checking
        -   Throughput time analysis
        -   Process map generation
        -   Trace and variant analysis
3.  **CatBoost Risk Modeling**
    -   Convert itemsets into binary features (1 = present, 0 = absent)
    -   Train model to predict risk (hospitalization, adverse events)
    -   Optimize model performance with:
        -   Temporal cross-validation
        -   Early stopping
        -   Hyperparameter tuning
        -   Class balancing
4.  **Formal Feature Attribution (FFA)**
    -   Attribute predictions to individual itemsets
    -   Use symbolic methods:
        -   AXPs (Abductive Explanations)
        -   Contrastive reasons
        -   Z3/SAT-based analysis
    -   Identify minimal feature subsets responsible for predictions
    -   Generate interpretable explanations
5.  **Comparative Analysis**
    -   Compare process patterns (BupaR) vs. risk patterns (FFA)
    -   Identify convergent vs. divergent itemsets
    -   Analyze alignment between:
        -   Frequently used process traces
        -   Risk-influential patterns
        -   Clinical pathways

### Key Insights

| Question | Analysis Method | Insights |
|----|----|----|
| What itemsets are most common? | FpGrowth | Frequent co-occurrence patterns |
| How do itemsets play out temporally? | BupaR | Process flows and sequences |
| Which itemsets drive model predictions? | CatBoost + FFA | Risk-influential patterns |
| Are process-dominant paths aligned with risk? | BupaR vs. FFA | Pattern alignment analysis |

### Visualization Approaches

1.  **Venn Diagrams**
    -   Compare frequent itemsets vs. risk itemsets
    -   Identify overlapping patterns
    -   Highlight unique patterns in each analysis
2.  **Process Maps with Risk Overlay**
    -   Base process map from BupaR
    -   Color-coded by risk influence
    -   Edge thickness based on frequency
    -   Node size based on FFA importance
3.  **Network Graphs**
    -   Nodes: itemsets
    -   Edges: co-occurrence relationships
    -   Dual labels:
        -   Process frequency
        -   Risk weight
    -   Color coding for pattern alignment

## Network Visualization Interpretation

❗ **Important Note:** When interpreting the network visualizations
generated from association rules, it's crucial to understand the
distinction between correlation and causation.

🔍 **Why?** Association rules (like those from FpGrowth) represent: -
Statistical co-occurrence between items in transactions - E.g., "If drug
A is present, drug B is also often present"

But they don't establish causality because they: - Don't control for
confounding variables - Don't establish temporal precedence - Don't use
interventions to test effects

So while an arrow A → B is drawn (based on a rule), this is not a causal
arrow — it represents a conditional probability relationship:

```         
P(B | A) is high → draw A → B
```

✅ **Why the Visualization Is Still Valuable** Even without causality: -
Summing support across multiple co-occurrence paths gives a meaningful
measure of total association weight - Directionality reflects rule
direction (not causal flow) - Node centrality indicates clustering or
"hub" drugs often present in many co-occurrence patterns - Edge
thickness communicates real signal strength in the data

🧠 **Bottom Line** \| Interpretation \| Is Valid? \| Explanation \|
\|----------------\|-----------\|-------------\| \| A → B is causal \|
❌ \| FpGrowth doesn't model interventions \| \| A → B co-occur often \|
✅ \| Based on high confidence/support \| \| Thicker edge = more total
co-occurrence \| ✅ \| Sum(support) reflects total influence \|

### Pattern Hashing and Attribution

The pattern mining process uses a sophisticated hashing and attribution
system:

1.  **Itemset Hashing**
    -   Each frequent itemset (from FpGrowth) is:

        -   Turned into a pipe-separated string (e.g., "drug_x\|drug_y")
        -   Hashed using MD5 to generate a unique pattern_id

    -   Results in a pattern_lookup table:

        ```         
        | pattern_id                         | itemsets          | support | ...metrics |
        |-----------------------------------|-------------------|---------|------------|
        | a8f72c99e5d1f4...                  | drug_x|drug_y     | 0.042   | ...        |
        | b04dd51b7926e2...                  | drug_z            | 0.089   | ...        |
        ```
2.  **Pattern Attribution**
    -   Each row in the DataFrame has up to MAX_PATTERN_COLUMNS slots:

        ```         
        | pattern_1       | pattern_2       | ... | pattern_15     |
        |-----------------|-----------------|-----|----------------|
        | b04dd...         | None            |     |                |
        | a8f72...         | b04dd...        |     |                |
        | None            | None            |     |                |
        ```

    -   These pattern\_\* columns reflect which pattern_ids (from
        pattern_lookup) were attributable to that row
3.  **Metric Merge**
    -   Using merge_pattern_metrics(), for each pattern_i, the
        corresponding metrics are merged in from pattern_lookup:

        ```         
        | pattern_1       | support_1 | confidence_1 | ...
        |-----------------|-----------|--------------|
        | a8f72...         | 0.042     | 0.62         |
        ```

    -   If a pattern was not matched or None, the row will have NaN or
        0.0 after merge

#### Guarantees

-   Each row only gets patterns it's eligible for — matched from
    rule/itemset presence
-   Only patterns up to MAX_PATTERN_COLUMNS are attributed per row
-   Patterns are attributed based on priority (e.g., support or rule
    quality) — usually highest scoring come first

#### Pattern Metrics

Each attributed pattern includes associated metrics: - support_N:
Frequency of the pattern in the dataset - confidence_N: Confidence score
for the pattern - lift_N: Lift score indicating pattern significance -
certainty_N: Certainty factor for the pattern

Example schema after metric merge:

```         
| pattern_1 | support_1 | confidence_1 | lift_1 | certainty_1 | pattern_2 | support_2 | ... |
|-----------|-----------|--------------|--------|-------------|-----------|-----------|-----|
| abc123... | 0.034     | 0.62         | 1.1    | 0.44        | def456... | 0.028     | ... |
| None      | NaN       | NaN          | NaN    | NaN         | None      | NaN       | ... |
```

### FPgrowth_Rank Variable

A new variable `FPgrowth_Rank` has been added to track the original
ranking of network features. This variable:

-   Stores the original rank (0-based index) of each active network
    feature
-   For each row, contains a list of ranks for all network features
    where value = 1
-   Uses -1 to indicate padded features (those added with zeros)

The rank information is valuable because: - Higher ranked patterns
(lower indices) were more frequent in the positive samples - Helps
identify the relative importance of different patterns - Distinguishes
between original and padded features - Can be used to analyze the
relationship between pattern rank and prediction accuracy

Example:

``` python
# If network_feature_0 and network_feature_5 are active (value = 1)
# FPgrowth_Rank would contain [0, 5]
# If a feature was padded, its rank would be -1
```

## Association Rules and Co-Usage Analysis

### Purpose

We extract association rules from drug co-occurrence data using
FP-Growth, filtering for positive-only patterns. These rules reveal
structured relationships among drugs and serve as clinically
interpretable features.

### What These Rules Show

Each rule takes the form:

```         
```

## Analysis Pipeline Overview

``` mermaid
flowchart TD
    subgraph "Data Processing"
        A[Pharmacy Data] --> B[Clean Pharmacy Data]
        C[Medical Data] --> D[Process Medical Data]
        B --> E[Create Cohorts]
        D --> E
    end

    subgraph "Data Filtering"
        E --> F1[Richmond Zip Codes]
        F1 --> F2[ICD/HCG Code Filtering]
        F2 --> F3[Age Band Filtering]
        F3 --> F4[Year Filtering 2016-2019]
    end

    subgraph "Cohort Creation"
        F4 --> G1[Opioid_ED Cohort]
        F4 --> G2[ED_Non_Opioid Cohort]
        G1 --> G3[Age Band 44-55]
        G2 --> G3
    end

    subgraph "Feature Engineering"
        G3 --> H[Network Feature Extraction]
        H --> I[FP-Growth Pattern Mining]
        I --> J[Association Rules]
    end

    subgraph "Modeling & Analysis"
        J --> K[CatBoost Model Training]
        K --> L[Feature Importance Analysis]
        L --> M[Process Mining with BupaR]
    end

    subgraph "Visualization & Output"
        M --> N[Network Visualization]
        M --> O[Process Maps]
        K --> P[Risk Predictions]
        L --> Q[Feature Importance Plots]
    end

    subgraph "Validation & Metrics"
        N --> R[Performance Metrics]
        O --> R
        P --> R
        Q --> R
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style F1 fill:#bbf,stroke:#333,stroke-width:2px
    style F2 fill:#bbf,stroke:#333,stroke-width:2px
    style F3 fill:#bbf,stroke:#333,stroke-width:2px
    style F4 fill:#bbf,stroke:#333,stroke-width:2px
    style G1 fill:#bfb,stroke:#333,stroke-width:2px
    style G2 fill:#bfb,stroke:#333,stroke-width:2px
    style G3 fill:#bfb,stroke:#333,stroke-width:2px
    style K fill:#fbb,stroke:#333,stroke-width:2px
    style M fill:#bfb,stroke:#333,stroke-width:2px
    style R fill:#fbb,stroke:#333,stroke-width:2px
```

## Cohort Generation Metrics

The following DuckDB queries track unique patient counts and transaction
counts through each step of the cohort generation pipeline:

``` sql
-- Create a metrics view for tracking counts through pipeline steps
CREATE OR REPLACE VIEW pipeline_metrics AS
WITH 
-- Initial data counts
initial_counts AS (
    SELECT 
        'Initial Data' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM medical_data
    UNION ALL
    SELECT 
        'Initial Pharmacy Data' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM pharmacy_data
),

-- Richmond zip code filter
richmond_counts AS (
    SELECT 
        'Richmond Zip Codes' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM medical_data
    WHERE zip_code IN (
        SELECT zip_code 
        FROM richmond_zip_codes
    )
),

-- ICD/HCG code filter
code_filtered_counts AS (
    SELECT 
        'ICD/HCG Code Filtered' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM medical_data
    WHERE (
        -- Opioid ED codes
        (icd_code IN (SELECT code FROM opioid_ed_codes) 
         OR hcg_code IN (SELECT code FROM opioid_ed_hcg_codes))
        OR
        -- Non-opioid ED codes
        (icd_code IN (SELECT code FROM non_opioid_ed_codes)
         OR hcg_code IN (SELECT code FROM non_opioid_ed_hcg_codes))
    )
),

-- Age band filter
age_filtered_counts AS (
    SELECT 
        'Age Band 44-55' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM medical_data
    WHERE member_age_dos BETWEEN 44 AND 55
),

-- Year filter
year_filtered_counts AS (
    SELECT 
        'Years 2016-2019' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM medical_data
    WHERE EXTRACT(YEAR FROM incurred_date) BETWEEN 2016 AND 2019
),

-- Final cohort counts
cohort_counts AS (
    SELECT 
        'Opioid_ED Cohort' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM opioid_ed_cohort
    UNION ALL
    SELECT 
        'ED_Non_Opioid Cohort' as pipeline_step,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        COUNT(*) as total_transactions
    FROM ed_non_opioid_cohort
)

-- Combine all metrics
SELECT * FROM initial_counts
UNION ALL
SELECT * FROM richmond_counts
UNION ALL
SELECT * FROM code_filtered_counts
UNION ALL
SELECT * FROM age_filtered_counts
UNION ALL
SELECT * FROM year_filtered_counts
UNION ALL
SELECT * FROM cohort_counts
ORDER BY 
    CASE pipeline_step
        WHEN 'Initial Data' THEN 1
        WHEN 'Initial Pharmacy Data' THEN 2
        WHEN 'Richmond Zip Codes' THEN 3
        WHEN 'ICD/HCG Code Filtered' THEN 4
        WHEN 'Age Band 44-55' THEN 5
        WHEN 'Years 2016-2019' THEN 6
        WHEN 'Opioid_ED Cohort' THEN 7
        WHEN 'ED_Non_Opioid Cohort' THEN 8
    END;
```

### Metrics Interpretation

The query above generates a comprehensive view of how the data is
filtered at each step of the pipeline. Key metrics to monitor:

1.  **Patient Retention**
    -   Track the percentage of unique patients retained at each step
    -   Identify steps with significant patient loss
    -   Ensure sufficient sample sizes for analysis
2.  **Transaction Volume**
    -   Monitor total transaction counts through the pipeline
    -   Identify steps with significant data reduction
    -   Ensure adequate event coverage for analysis
3.  **Cohort Balance**
    -   Compare patient and transaction counts between Opioid_ED and
        ED_Non_Opioid cohorts
    -   Verify 5:1 control-to-case ratio is maintained
    -   Ensure sufficient data for both cohorts

### Example Output

The query will produce a table with the following structure:

| Pipeline Step         | Unique Patients | Total Transactions |
|-----------------------|-----------------|--------------------|
| Initial Data          | [count]         | [count]            |
| Initial Pharmacy Data | [count]         | [count]            |
| Richmond Zip Codes    | [count]         | [count]            |
| ICD/HCG Code Filtered | [count]         | [count]            |
| Age Band 44-55        | [count]         | [count]            |
| Years 2016-2019       | [count]         | [count]            |
| Opioid_ED Cohort      | [count]         | [count]            |
| ED_Non_Opioid Cohort  | [count]         | [count]            |

This metrics view helps ensure data quality and sufficient sample sizes
throughout the pipeline.
