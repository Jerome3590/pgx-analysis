# Step 3b: Feature Importance EDA and Refinement

## Overview

Step 3b performs additional exploratory data analysis on aggregated feature importances from Step 3, using:
1. **Administrative/Non-informative code filtering** (remove non-informative ICD/CPT codes from lookup table)
2. **BupaR post-target analysis** to identify pre/post F1120 ICD/CPT events (target leakage detection)
3. **Interactive code review and filtering** to refine feature selection

Based on this EDA, we filter and update the aggregated feature importances to produce refined `cohort_feature_importance` files that feed into Step 4a model data creation.

## Purpose

- **Identify post-target leakage**: Use BupaR to analyze sequences before and after the target event (F1120) to identify features that may leak future information
- **Filter non-value-added codes**: Remove administrative, scheduling, and non-medical codes that don't add predictive value
- **Apply safe feature filtering**: Exclude post-target leakage features while keeping all pre-target features to maximize information available to the algorithm
- **Refine feature importances**: Update aggregated feature importances based on EDA findings
- **Output refined features**: Generate `cohort_feature_importance` files for Step 4a

## Inputs

- **Aggregated feature importances** from Step 3:
  - `3_feature_importance/outputs/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv`
- **Model events data** (for BupaR/DTW analysis):
  - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`

## Outputs

### Local Files

**Primary Output:**
- `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_cohort_feature_importance.csv` - Refined feature importances for Step 4a

**Analysis Reports:**
- `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_bupar_post_target_analysis.csv` - BupaR post-target leakage analysis
- `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_feature_filtering_summary.json` - Filtering summary statistics

**Feature Filter Files:**
- `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_safe_feature_filter.json` - Safe feature filter (whitelist/blacklist)
- `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_post_target_filter.json` - Post-target leakage filter
- `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_pre_target_predictive_features.json` - Pre-target predictive features

**BupaR Feature Files:**
- `outputs/{cohort}/{age_band_fname}/features/*_bupar.csv` - BupaR process mining outputs (traces, patient features, time-to-F1120)
- See `1_bupaR/README.md` for complete file manifest

**Visualizations:**
- `outputs/{cohort}/{age_band_fname}/plots/*.png` - BupaR process mining visualizations
- See `OUTPUTS_AND_VISUALIZATIONS.md` for complete visualization documentation

### S3 Checkpoints

All outputs are automatically uploaded to S3 for checkpointing and downstream consumption:
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_bupar_post_target_analysis.csv`
- `s3://pgxdatalake/gold/bupar/{cohort}/{age_band}/*_bupar.csv` - BupaR feature files
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/*.png` - Visualizations

**Note**: Uploads are idempotent - files are only uploaded if they don't already exist in S3.

## Workflow

1. **Load aggregated feature importances** from Step 3
2. **Administrative/Non-informative Code Filtering** (`0_icd_cpt_check/`):
   - Load administrative codes from `4b_dtw_filter/administrative_codes_lookup.json`
   - Validate ICD/CPT codes by groups (ICD by chapter, CPT by range)
   - Remove non-informative ICD/CPT codes (administrative, scheduling, protocol codes)
   - See `0_icd_cpt_check/README.md` for detailed validation process
3. **BupaR Post-Target Analysis** (`1_bupaR/`):
   - Build BupaR event logs from `model_events.parquet`
   - Analyze sequences before and after target event (F1120)
   - Calculate pre-F1120 and post-F1120 ratios for each feature
   - Identify features that appear primarily post-target (>=80% post-F1120 ratio = potential leakage)
   - Generate comprehensive BupaR features and visualizations
   - Output: `{cohort}_{age_band}_bupar_post_target_analysis.csv`
   - See `1_bupaR/README.md` for detailed BupaR process mining documentation
4. **Create Safe Feature Filter**:
   - Exclude features with >=80% post-F1120 ratio (pure post-target leakage)
   - Keep ALL features with ANY pre-F1120 presence (maximize information)
   - Explicitly include F1120 for target creation
   - Output: `{cohort}_{age_band}_safe_feature_filter.json`
5. **Filter and Update Feature Importances**:
   - Apply safe feature filter (whitelist for cases, blacklist for controls)
   - Remove administrative/non-informative codes
   - Adjust importance scores based on EDA findings
   - Generate refined `cohort_feature_importance` files
6. **Save outputs locally and upload to S3**:
   - Save all outputs to local filesystem
   - Upload to S3 for checkpointing and Step 4a consumption
   - Save checkpoint metadata to S3

## Scripts

### Main Orchestration
- `run_step_3b.py` - Orchestration script to run all analyses in order
- `step3b_workflow.py` - Interactive workflow script (can be run as notebook or script)
- `step3b_interactive_analysis_cohort*.ipynb` - Cohort-specific interactive notebooks

### Analysis Scripts
- `run_bupar_post_target_analysis.py` - BupaR analysis for post-target leakage detection
- `create_bupar_post_target_analysis.py` - Creates post-target analysis CSV from BupaR outputs
- `filter_and_refine_features.py` - Main script to filter and refine feature importances

### Feature Filtering Scripts
- `create_safe_feature_filter_json.py` - Creates safe feature filter JSON (exclude leakage, keep pre-target)
- `create_post_target_filter_json.py` - Creates post-target leakage filter JSON
- `create_pre_target_predictive_features_json.py` - Creates pre-target predictive features JSON
- `analyze_leakage_features.py` - Analyzes and categorizes leakage features

### Validation Scripts
- `0_icd_cpt_check/analyze_code_groups.py` - Analyzes ICD/CPT codes by groups
- `0_icd_cpt_check/validate_icd_cpt_codes.py` - Interactive validation workflow
- `check_pre_post_f1120_events.py` - Verifies presence of pre/post F1120 events in data

### R Scripts (BupaR Process Mining)
- `1_bupaR/create_bupar_outputs_opioid_ed.R` - BupaR analysis for opioid_ed cohort
- `1_bupaR/create_bupar_outputs_non_opioid_ed.R` - BupaR analysis for non_opioid_ed cohort
- See `1_bupaR/README.md` for complete BupaR documentation

## Usage

```bash
# Run for a single cohort/age_band
python 3b_feature_importance_eda/run_step_3b.py --cohort opioid_ed --age-band 13-24

# Run for all cohorts
python 3b_feature_importance_eda/run_step_3b.py --all-cohorts
```

## Feature Filtering Strategy

We use a **safe feature filter** approach that:
1. **Excludes** post-target leakage features (>=80% post-F1120 ratio)
2. **Keeps** ALL features with ANY pre-F1120 presence (maximize information)
3. **Applies** different filtering for cases vs controls:
   - **Cases (target=1)**: Whitelist approach (only features from `all_features_to_keep`)
   - **Controls (target=0)**: Blacklist approach (exclude only post-target leakage features)

See `FEATURE_FILTERING_APPROACH.md` for detailed documentation.

## Directory Structure

```
3b_feature_importance_eda/
├── 0_icd_cpt_check/              # ICD/CPT code validation
│   ├── analyze_code_groups.py
│   ├── validate_icd_cpt_codes.py
│   ├── administrative_codes_lookup.json
│   └── README.md                 # Code validation documentation
├── 1_bupaR/                      # BupaR process mining analysis
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R
│   ├── create_plots.R
│   └── README.md                 # BupaR process mining documentation
├── outputs/                      # All outputs organized by cohort/age_band
│   ├── {cohort}/
│   │   └── {age_band_fname}/
│   │       ├── features/         # BupaR feature files
│   │       ├── plots/            # Visualization PNG files
│   │       └── *.csv, *.json     # Analysis results
├── step3b_workflow.py            # Main interactive workflow
├── step3b_interactive_analysis_cohort*.ipynb  # Cohort-specific notebooks
├── run_step_3b.py                # Orchestration script
├── filter_and_refine_features.py  # Feature filtering and refinement
└── README.md                     # This file
```

## Integration with Pipeline

- **Input**: Step 3 aggregated feature importances
  - `3_feature_importance/outputs/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv`
- **Output**: Refined `cohort_feature_importance` files
  - `outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_cohort_feature_importance.csv`
- **Consumed by**: Step 4a model data creation
  - Step 4a uses the refined feature importance to filter events and create model-ready data

## Additional Documentation

- **`EXECUTION_ORDER.md`**: Detailed execution order and rationale
- **`FEATURE_FILTERING_APPROACH.md`**: Safe feature filtering strategy
- **`OUTPUTS_AND_VISUALIZATIONS.md`**: Complete output file manifest and visualization documentation
- **`LEAKAGE_ANALYSIS_SUMMARY.md`**: Summary of identified leakage features
- **`0_icd_cpt_check/README.md`**: ICD/CPT code validation process
- **`1_bupaR/README.md`**: BupaR process mining documentation