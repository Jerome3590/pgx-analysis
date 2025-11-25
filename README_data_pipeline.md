# Data Pipeline

Data processing, cohort creation, and data flow for the Prescription Drug Analysis pipeline.

## Overview

The data pipeline processes large-scale healthcare datasets from APCD (All-Payer Claims Database), performs data cleaning and normalization, creates analytical cohorts, and outputs structured data ready for analysis.

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

## Cohort Creation Pipeline

The cohort creation pipeline follows a modular architecture with four main phases:

### Phase 1: Data Preparation
- Loads and filters medical and pharmacy data from APCD gold tier
- Creates normalized views for downstream processing
- Applies data quality filters (valid age range, date range)

### Phase 2: Event Processing
- Creates unified event fact table combining medical and pharmacy events
- Implements classification logic for target identification
- Checks ALL 10 ICD diagnosis columns for comprehensive opioid patient identification

### Phase 3: Cohort Creation
- Creates OPIOID_ED and ED_NON_OPIOID cohorts
- Maintains 5:1 control-to-target ratio
- Ensures statistical independence (no control reuse within cohorts)
- Applies balanced temporal windows for ED_NON_OPIOID cohort

### Phase 4: Finalization
- Validates cohorts and saves to S3 in Parquet format
- Generates quality assurance metrics
- Provides detailed logging and error handling

## Key Features

### Drug Event Explosion Strategy ⭐ **NEW**
- **Patient-Level → Drug-Level Transformation**: Each drug prescription becomes a separate row
- **Context Duplication**: Patient demographics and clinical data duplicated per drug event
- **Sequence Modeling Ready**: Enables FpGrowth, bupaR, DTW, and symbolic reasoning analysis
- **Temporal Tracking**: Maintains `days_to_ade` and `days_to_opioid_ed` relationships

### Cohort Exclusivity Enforcement ⭐ **NEW**
- **OPIOID_ED Priority**: Processes opioid_ed cohort first
- **Mutual Exclusivity**: Ensures no patient appears in both cohorts
- **Quality Assurance**: Validates cohort separation and logs metrics
- **Data Integrity**: Prevents data leakage between cohorts

### Enhanced Drug Exposure Analysis ⭐ **ENHANCED**
- **ADE Cohort**: 30-day lookback window for causality assessment
- **Opioid Cohort**: Complete drug history for pattern analysis
- **Temporal Relationships**: Precise tracking of drug-event timing
- **Data Source Tracking**: Distinguishes between `cohort_event` and `drug_exposure` rows

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

## Related Documentation

- [`README_overview.md`](README_overview.md) - Project structure and components
- [`README_analysis_workflow.md`](README_analysis_workflow.md) - Feature importance and pattern mining
- [`docs/README_create_cohort.md`](docs/README_create_cohort.md) - Comprehensive cohort creation guide
- [`docs/README_data_pipeline.md`](docs/README_data_pipeline.md) - Detailed pipeline architecture

