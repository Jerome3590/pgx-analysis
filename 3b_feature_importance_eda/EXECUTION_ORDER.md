# Step 3b Execution Order

## Overview

Step 3b executes analyses in this order to properly filter features before model training:
1. **Administrative/Non-informative code filtering** (remove non-informative ICD/CPT codes)
2. **BupaR post-target analysis** (identify pre/post F1120 ICD/CPT events)
3. **DTW trajectory analysis** (trajectories, visualizations, additional non-informative events)
4. **Filter and refine** (combine all filtering results)

## Folder Naming Convention

Folders reflect execution order:
- `1_bupaR/` - BupaR analysis (runs second, after administrative filtering)
- `2_dtw/` - DTW analysis (runs third, after BupaR)

## Execution Order

The pipeline now executes in this order:

1. **Administrative/Non-informative Code Filtering**
   - Loads administrative codes from `4b_dtw_filter/administrative_codes_lookup.json`
   - Removes non-informative ICD/CPT codes (administrative, scheduling, protocol codes)
   - This filtering is applied in the `filter_and_refine_features.py` step
   - Note: Feature importances are already calculated in Step 3; we filter them here

2. **BupaR Post-Target Analysis** (`1_bupaR/`)
   - Builds BupaR event logs from model_events.parquet
   - Runs pre- and post-F1120 sequence analyses
   - Identifies ICD/CPT codes that appear primarily after F1120 (post-target leakage)
   - Generates comprehensive BupaR features
   - Outputs: `{cohort}_{age_band}_bupar_post_target_analysis.csv`

3. **DTW Trajectory Analysis** (`2_dtw/`)
   - Creates DTW features from patient sequences
   - Computes DTW distances to prototype trajectories
   - Generates trajectory cluster features
   - Data visualizations for target cohort
   - Identifies additional non-informative events through trajectory analysis
   - Outputs: `dtw_features_{cohort}_{age_band}.csv`, `{cohort}_{age_band}_dtw_trajectory_analysis.csv`

4. **Filter and Refine Features**
   - Combines outputs from administrative filtering, BupaR, and DTW analyses
   - Filters features based on:
     - Administrative/non-informative codes (from lookup table)
     - Post-target leakage (from BupaR)
     - Additional non-value-added codes (from DTW)
   - Outputs: `cohort_feature_importance.csv`

5. **Create BupaR Visualizations**
   - Generates visualization plots from BupaR analysis
   - Copies plots to Step 3b outputs directory

## Updated Files

### Scripts Updated
- `run_step_3b.py`: Updated to run BupaR second, then DTW third
- `run_dtw_trajectory_analysis.py`: Updated to reference `2_dtw/` folder
- `run_bupar_post_target_analysis.py`: Updated to reference `1_bupaR/` folder

### Folder Structure

```
3b_feature_importance_eda/
├── 1_bupaR/                        # BupaR analysis (Step 2, after admin filtering)
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R
│   └── ...
├── 2_dtw/                          # DTW analysis (Step 3, after BupaR)
│   ├── create_dtw_features.py
│   ├── add_dtw_features_to_model_data.py
│   └── ...
├── run_dtw_trajectory_analysis.py   # Calls 2_dtw/ scripts
├── run_bupar_post_target_analysis.py # Calls 1_bupaR/ scripts
└── run_step_3b.py                   # Orchestrates execution order
```

## Rationale

**Administrative code filtering runs first** because:
- Removes clearly non-informative codes before other analyses
- Uses pre-identified administrative codes from lookup table
- Reduces noise in subsequent BupaR and DTW analyses

**BupaR analysis runs second** because:
- Identifies pre vs post-F1120 events (critical for target leakage prevention)
- Must run before DTW to identify which events are post-target
- Outputs are used to filter post-target leakage features

**DTW analysis runs third** because:
- Uses pre-filtered data (after administrative and post-target filtering)
- Focuses on trajectory patterns and visualizations
- Identifies additional non-informative events through trajectory analysis
- Provides data visualizations for target cohort

## Running Step 3b

```bash
# Run for a single cohort/age band
python 3b_feature_importance_eda/run_step_3b.py --cohort opioid_ed --age-band 13-24

# The pipeline will:
# 1. Filter administrative/non-informative codes (from lookup table)
# 2. Run BupaR analysis (1_bupaR/) - identify pre/post F1120 events
# 3. Run DTW analysis (2_dtw/) - trajectories, visualizations, additional filtering
# 4. Filter and refine features (combine all filtering results)
# 5. Create visualizations
```
