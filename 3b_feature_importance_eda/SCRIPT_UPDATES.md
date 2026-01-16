# Step 3b Script Updates: Using Working BupaR and DTW Implementations

## Summary of Changes

The Step 3b scripts have been updated to use the working BupaR and DTW implementations from the `bupaR/` and `dtw/` folders.

## Key Differences

### BupaR Analysis

**Previous (Simplified) Implementation:**
- `run_bupar_post_target_analysis.py`: Simple Python script that only analyzed post-target leakage
- Limited functionality: only identified features appearing after target event
- No R integration, no comprehensive BupaR features

**Updated (Working) Implementation:**
- Uses R scripts from `bupaR/` folder:
  - `build_bupar_eventlogs.R`: Builds BupaR-compatible event logs from model_events.parquet
  - `create_bupar_outputs_opioid_ed.R` / `create_bupar_outputs_non_opioid_ed.R`: Full BupaR analysis
  - `add_bupar_features_to_model_data.R`: Merges all BupaR features
- Creates comprehensive features:
  - Pre-F1120 patient features
  - Post-F1120 patient features  
  - Time-to-F1120 features
  - Trace sequences (top, rare, overall)
  - Process matrices
- Generates visualizations via R scripts
- Outputs: `bupaR_added_features_{cohort}_{age_band}.csv` ready for model training

### DTW Analysis

**Previous (Simplified) Implementation:**
- `run_dtw_trajectory_analysis.py`: Simple Python script that only identified non-value-added codes
- Limited functionality: only flagged administrative/scheduling codes
- No trajectory clustering, no DTW distance calculations

**Updated (Working) Implementation:**
- Uses Python scripts from `dtw/` folder:
  - `create_dtw_features.py`: Creates DTW trajectory features from patient sequences
  - `dtw_trajectory_analysis.py`: Enhanced DTW analysis with trajectory clustering
  - `add_dtw_features_to_model_data.py`: Merges DTW features
- Creates comprehensive features:
  - Trajectory cluster memberships (drug, ICD, CPT)
  - DTW distances to prototype trajectories
  - Trajectory characteristics (length, diversity, temporal properties)
  - Cluster properties (target rates, sizes)
  - Archetype matching scores
- Outputs: `dtw_added_features_{cohort}_{age_band}.csv` ready for model training

## Updated Scripts

### 1. `run_bupar_post_target_analysis.py`
**Now calls:** R scripts from `bupaR/` folder
- `build_bupar_eventlogs.R` → builds event logs
- `create_bupar_outputs_{cohort}.R` → runs full BupaR analysis
- `add_bupar_features_to_model_data.R` → merges features

### 2. `run_dtw_trajectory_analysis.py`
**Now calls:** Python scripts from `dtw/` folder
- `create_dtw_features.py` → creates DTW features from trajectories
- `add_dtw_features_to_model_data.py` → merges DTW features

### 3. `run_step_3b.py`
**Updated to:**
- Call the working implementations
- Handle R script execution for BupaR
- Integrate with existing workflow

## File Structure

```
3b_feature_importance_eda/
├── bupaR/                          # Working BupaR scripts (R-based)
│   ├── build_bupar_eventlogs.R
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R
│   ├── add_bupar_features_to_model_data.R
│   └── ...
├── dtw/                            # Working DTW scripts (Python-based)
│   ├── create_dtw_features.py
│   ├── dtw_trajectory_analysis.py
│   ├── add_dtw_features_to_model_data.py
│   └── ...
├── run_bupar_post_target_analysis.py  # Updated to call bupaR/ scripts
├── run_dtw_trajectory_analysis.py      # Updated to call dtw/ scripts
├── run_step_3b.py                      # Updated orchestration script
└── filter_and_refine_features.py       # Uses outputs from both analyses
```

## Output Files

### BupaR Outputs
- `outputs/{cohort}/{age_band}/features/*_bupar.csv`: Per-patient BupaR features
- `outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`: Final merged features

### DTW Outputs
- `outputs/{cohort}/{age_band}/trajectory_results_{item_type}.json`: DTW analysis results
- `outputs/feature_engineering/dtw_added_features_{cohort}_{age_band}.csv`: Final merged features

### Step 3b Final Output
- `outputs/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`: Refined feature importances

## Integration with Pipeline

1. **Step 3b runs:**
   - BupaR analysis → creates comprehensive BupaR features
   - DTW analysis → creates comprehensive DTW features
   - Filter and refine → uses both analyses to refine feature importances

2. **Step 4a consumes:**
   - `cohort_feature_importance.csv` from Step 3b
   - Uses refined features to filter model_events.parquet

## Migration Notes

- The simplified Python-only scripts are replaced with calls to the working implementations
- R scripts are called via `Rscript` command
- Python scripts are called directly
- All outputs maintain the same format and S3 locations
