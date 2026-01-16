# Step 3b Script Updates Summary

## Overview

The Step 3b scripts have been updated to use the working BupaR and DTW implementations from the `bupaR/` and `dtw/` folders that you added.

## Key Differences and Updates

### 1. BupaR Analysis (`run_bupar_post_target_analysis.py`)

#### Previous Implementation (Simplified)
- **Language**: Python-only
- **Functionality**: Simple post-target leakage analysis
- **What it did**: 
  - Analyzed features appearing after target event
  - Flagged potential leakage features
  - Generated basic CSV report
- **Limitations**: 
  - No comprehensive BupaR features
  - No R integration
  - No process mining capabilities
  - No visualizations

#### Updated Implementation (Working)
- **Language**: Python wrapper calling R scripts
- **Functionality**: Full BupaR process mining pipeline
- **What it does**:
  1. Calls `create_bupar_outputs_opioid_ed.R` or `create_bupar_outputs_non_opioid_ed.R`
  2. Builds BupaR event logs from `model_events.parquet`
  3. Runs pre- and post-F1120 sequence analyses
  4. Generates comprehensive features:
     - Pre-F1120 patient features
     - Post-F1120 patient features
     - Time-to-F1120 features
     - Trace sequences (top, rare, overall)
     - Process matrices
  5. Creates visualizations via R scripts
- **Outputs**:
  - `outputs/{cohort}/{age_band}/features/*_bupar.csv`: Per-patient features
  - `outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`: Final merged features
  - Visualizations in `outputs/{cohort}/{age_band}/plots/`

### 2. DTW Analysis (`run_dtw_trajectory_analysis.py`)

#### Previous Implementation (Simplified)
- **Language**: Python-only
- **Functionality**: Simple non-value-added code identification
- **What it did**:
  - Identified administrative/scheduling codes
  - Flagged Z codes and non-medical ICD codes
  - Generated basic CSV report
- **Limitations**:
  - No trajectory clustering
  - No DTW distance calculations
  - No prototype trajectory analysis
  - No comprehensive features

#### Updated Implementation (Working)
- **Language**: Python calling Python scripts
- **Functionality**: Full DTW trajectory analysis pipeline
- **What it does**:
  1. Calls `create_dtw_features.py` to create DTW features
  2. Extracts patient trajectories from model_events.parquet
  3. Computes DTW distances to prototype trajectories
  4. Creates comprehensive features:
     - Trajectory cluster memberships (drug, ICD, CPT)
     - DTW distances to prototypes
     - Trajectory characteristics (length, diversity, temporal properties)
     - Cluster properties (target rates, sizes)
     - Archetype matching scores
  5. Calls `add_dtw_features_to_model_data.py` to merge features
- **Outputs**:
  - `outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv`: DTW features
  - `outputs/feature_engineering/dtw_added_features_{cohort}_{age_band}.csv`: Final merged features
  - Trajectory analysis results in JSON/Parquet format

## Updated Script Structure

```
3b_feature_importance_eda/
├── bupaR/                                    # Working BupaR scripts (R-based)
│   ├── build_bupar_eventlogs.R              # Builds event logs
│   ├── create_bupar_outputs_opioid_ed.R     # Full BupaR analysis (opioid_ed)
│   ├── create_bupar_outputs_non_opioid_ed.R # Full BupaR analysis (non_opioid_ed)
│   ├── add_bupar_features_to_model_data.R   # Merges all BupaR features
│   └── ...
├── dtw/                                      # Working DTW scripts (Python-based)
│   ├── create_dtw_features.py               # Creates DTW features from trajectories
│   ├── dtw_trajectory_analysis.py          # Enhanced DTW analysis
│   ├── add_dtw_features_to_model_data.py    # Merges DTW features
│   └── ...
├── run_bupar_post_target_analysis.py        # ✅ UPDATED: Calls bupaR/ scripts
├── run_dtw_trajectory_analysis.py           # ✅ UPDATED: Calls dtw/ scripts
├── run_step_3b.py                           # Orchestration script
└── filter_and_refine_features.py            # Uses outputs from both analyses
```

## How the Updated Scripts Work

### BupaR Script Flow

```python
run_bupar_post_target_analysis.py
  ↓
  Finds Rscript executable
  ↓
  Calls: create_bupar_outputs_{cohort}.R {age_band}
  ↓
  R script:
    1. Builds event logs (build_bupar_eventlogs.R)
    2. Runs BupaR analysis
    3. Generates features and visualizations
    4. Saves outputs to outputs/{cohort}/{age_band}/
  ↓
  Python script verifies outputs were created
```

### DTW Script Flow

```python
run_dtw_trajectory_analysis.py
  ↓
  Calls: create_dtw_features.py --cohort {cohort} --age_band {age_band}
  ↓
  Creates DTW features from trajectories
  ↓
  Calls: add_dtw_features_to_model_data.py --cohort-name {cohort} --age-band {age_band}
  ↓
  Merges DTW features into final output
  ↓
  Saves to outputs/feature_engineering/
```

## Output Files

### BupaR Outputs
- **Location**: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/features/`
- **Files**:
  - `{cohort}_{age_band}_train_target_pre_f1120_patient_features_bupar.csv`
  - `{cohort}_{age_band}_train_target_post_f1120_patient_features_bupar.csv`
  - `{cohort}_{age_band}_train_target_time_to_f1120_features_bupar.csv`
  - `{cohort}_{age_band}_train_target_traces_bupar.csv`
  - `{cohort}_{age_band}_train_target_traces_top_bupar.csv`
  - `{cohort}_{age_band}_train_target_traces_rare_bupar.csv`
  - `{cohort}_{age_band}_train_target_process_matrix_bupar.csv`
- **Final merged**: `outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`

### DTW Outputs
- **Location**: `3b_feature_importance_eda/outputs/feature_engineering/`
- **Files**:
  - `dtw_features_{cohort}_{age_band}.csv`: DTW trajectory features
  - `dtw_added_features_{cohort}_{age_band}.csv`: Final merged DTW features

## Integration with Pipeline

1. **Step 3b runs**:
   - BupaR analysis → creates comprehensive BupaR features
   - DTW analysis → creates comprehensive DTW features
   - Filter and refine → uses both analyses to refine feature importances

2. **Step 4a consumes**:
   - `cohort_feature_importance.csv` from Step 3b
   - Uses refined features to filter `model_events.parquet`

## Requirements

### For BupaR:
- R installed and `Rscript` in PATH
- R packages: `bupaR`, `bupaverse`, `processmapR`, `edeaR`, `duckdb`, `arrow`, `dplyr`, `tidyr`, `jsonlite`, `readr`, `ggplot2`

### For DTW:
- Python packages: `dtaidistance`, `pandas`, `numpy`, `duckdb`, `sklearn`

## Testing

To test the updated scripts:

```bash
# Test BupaR
python 3b_feature_importance_eda/run_bupar_post_target_analysis.py \
    --cohort opioid_ed \
    --age-band 13-24

# Test DTW
python 3b_feature_importance_eda/run_dtw_trajectory_analysis.py \
    --cohort opioid_ed \
    --age-band 13-24

# Run full Step 3b
python 3b_feature_importance_eda/run_step_3b.py \
    --cohort opioid_ed \
    --age-band 13-24
```

## Migration Notes

- ✅ The simplified Python-only scripts are replaced with calls to working implementations
- ✅ R scripts are called via `Rscript` command
- ✅ Python scripts are called directly
- ✅ All outputs maintain the same format and S3 locations
- ✅ Backward compatibility: Scripts still accept same command-line arguments

## Next Steps

1. Test the updated scripts with a single cohort/age_band
2. Verify outputs match expected format
3. Run for all cohorts/age_bands
4. Verify Step 4a can consume the outputs correctly
