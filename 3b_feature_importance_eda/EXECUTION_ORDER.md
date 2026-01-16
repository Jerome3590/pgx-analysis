# Step 3b Execution Order

## Overview

Step 3b has been updated to ensure proper execution order: **DTW runs first (Step 1), then BupaR (Step 2)**.

## Folder Naming Convention

Folders have been renamed to reflect execution order:
- `1_dtw/` - DTW analysis (runs first)
- `2_bupaR/` - BupaR analysis (runs second)

## Execution Order

The pipeline now executes in this order:

1. **DTW Trajectory Analysis** (`1_dtw/`)
   - Creates DTW features from patient sequences
   - Computes DTW distances to prototype trajectories
   - Generates trajectory cluster features
   - Outputs: `dtw_features_{cohort}_{age_band}.csv`

2. **BupaR Post-Target Analysis** (`2_bupaR/`)
   - Builds BupaR event logs from model_events.parquet
   - Runs pre- and post-F1120 sequence analyses
   - Generates comprehensive BupaR features
   - Outputs: Various BupaR feature CSVs

3. **Filter and Refine Features**
   - Combines outputs from both DTW and BupaR analyses
   - Filters features based on post-target leakage and non-value-added codes
   - Outputs: `cohort_feature_importance.csv`

4. **Create BupaR Visualizations**
   - Generates visualization plots from BupaR analysis
   - Copies plots to Step 3b outputs directory

## Updated Files

### Scripts Updated
- `run_step_3b.py`: Updated to run DTW first, then BupaR
- `run_dtw_trajectory_analysis.py`: Updated to reference `1_dtw/` folder
- `run_bupar_post_target_analysis.py`: Updated to reference `2_bupaR/` folder

### Folder Structure

```
3b_feature_importance_eda/
├── 1_dtw/                          # DTW analysis (Step 1)
│   ├── create_dtw_features.py
│   ├── add_dtw_features_to_model_data.py
│   └── ...
├── 2_bupaR/                        # BupaR analysis (Step 2)
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R
│   └── ...
├── run_dtw_trajectory_analysis.py   # Calls 1_dtw/ scripts
├── run_bupar_post_target_analysis.py # Calls 2_bupaR/ scripts
└── run_step_3b.py                   # Orchestrates execution order
```

## Rationale

DTW analysis runs first because:
- It processes raw patient trajectories from `model_events.parquet`
- It creates trajectory features that may be useful for BupaR analysis
- It has no dependencies on BupaR outputs

BupaR analysis runs second because:
- It may benefit from DTW trajectory features (if integrated)
- It performs process mining on the same `model_events.parquet` data
- Its outputs are combined with DTW outputs in the filtering step

## Running Step 3b

```bash
# Run for a single cohort/age band
python 3b_feature_importance_eda/run_step_3b.py --cohort opioid_ed --age-band 13-24

# The pipeline will:
# 1. Run DTW analysis (1_dtw/)
# 2. Run BupaR analysis (2_bupaR/)
# 3. Filter and refine features
# 4. Create visualizations
```
