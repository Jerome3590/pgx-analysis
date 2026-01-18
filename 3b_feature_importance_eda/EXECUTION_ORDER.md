# Step 3b Execution Order

## Overview

Step 3b executes analyses in this order to properly filter features before model training:
1. **Administrative/Non-informative code filtering** (remove non-informative ICD/CPT codes)
2. **BupaR post-target analysis** (identify pre/post F1120 ICD/CPT events)
3. **Filter and refine** (combine all filtering results)

## Folder Naming Convention

Folders reflect execution order:
- `0_icd_cpt_check/` - ICD/CPT code validation (runs first)
- `1_bupaR/` - BupaR analysis (runs second, after administrative filtering)

## Execution Order

The pipeline now executes in this order:

1. **Administrative/Non-informative Code Filtering** (`0_icd_cpt_check/`)
   - Validates ICD/CPT codes by groups (ICD by chapter, CPT by range)
   - Loads administrative codes from `4b_dtw_filter/administrative_codes_lookup.json`
   - Removes non-informative ICD/CPT codes (administrative, scheduling, protocol codes)
   - This filtering is applied in the `filter_and_refine_features.py` step
   - Note: Feature importances are already calculated in Step 3; we filter them here
   - See `0_icd_cpt_check/README.md` for detailed validation process

2. **BupaR Post-Target Analysis** (`1_bupaR/`)
   - Builds BupaR event logs from `model_events.parquet`
   - Runs pre- and post-F1120 sequence analyses
   - Calculates pre-F1120 and post-F1120 ratios for each feature
   - Identifies ICD/CPT codes that appear primarily after F1120 (>=80% post-F1120 ratio = post-target leakage)
   - Generates comprehensive BupaR features and visualizations
   - Outputs: `{cohort}_{age_band}_bupar_post_target_analysis.csv`
   - See `1_bupaR/README.md` for complete BupaR process mining documentation

3. **Create Safe Feature Filter**
   - Excludes features with >=80% post-F1120 ratio (pure post-target leakage)
   - Keeps ALL features with ANY pre-F1120 presence (maximize information)
   - Explicitly includes F1120 for target creation
   - Outputs: `{cohort}_{age_band}_safe_feature_filter.json`
   - See `FEATURE_FILTERING_APPROACH.md` for detailed strategy

4. **Filter and Refine Features**
   - Combines outputs from administrative filtering and BupaR analyses
   - Applies safe feature filter:
     - **Cases (target=1)**: Whitelist approach (only features from `all_features_to_keep`)
     - **Controls (target=0)**: Blacklist approach (exclude only post-target leakage features)
   - Filters features based on:
     - Administrative/non-informative codes (from lookup table)
     - Post-target leakage (from BupaR safe feature filter)
   - Outputs: `cohort_feature_importance.csv`

5. **Create BupaR Visualizations**
   - Generates visualization plots from BupaR analysis
   - Saves plots to `outputs/{cohort}/{age_band_fname}/plots/`
   - See `OUTPUTS_AND_VISUALIZATIONS.md` for complete visualization documentation

## Updated Files

### Scripts Updated
- `run_step_3b.py`: Updated to run BupaR analysis (DTW removed)
- `run_bupar_post_target_analysis.py`: Calls `1_bupaR/` scripts
- `create_bupar_post_target_analysis.py`: Creates post-target analysis CSV from BupaR outputs

### Folder Structure

```
3b_feature_importance_eda/
├── 0_icd_cpt_check/                 # ICD/CPT code validation (Step 1)
│   ├── analyze_code_groups.py
│   ├── validate_icd_cpt_codes.py
│   ├── administrative_codes_lookup.json
│   └── README.md
├── 1_bupaR/                         # BupaR analysis (Step 2, after admin filtering)
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R
│   ├── create_plots.R
│   └── README.md
├── run_bupar_post_target_analysis.py # Calls 1_bupaR/ scripts
├── create_bupar_post_target_analysis.py # Creates post-target analysis CSV
└── run_step_3b.py                   # Orchestrates execution order
```

## Rationale

**Administrative code filtering runs first** because:
- Removes clearly non-informative codes before other analyses
- Uses pre-identified administrative codes from lookup table
- Reduces noise in subsequent BupaR and DTW analyses

**BupaR analysis runs second** because:
- Identifies pre vs post-F1120 events (critical for target leakage prevention)
- Calculates pre-F1120 and post-F1120 ratios for each feature
- Outputs are used to create safe feature filter (exclude leakage, keep pre-target)
- Generates comprehensive process mining visualizations

## Running Step 3b

```bash
# Run for a single cohort/age band
python 3b_feature_importance_eda/run_step_3b.py --cohort opioid_ed --age-band 13-24

# The pipeline will:
# 1. Filter administrative/non-informative codes (from lookup table)
# 2. Run BupaR analysis (1_bupaR/) - identify pre/post F1120 events
# 3. Create safe feature filter - exclude leakage, keep pre-target features
# 4. Filter and refine features (combine administrative + BupaR filtering)
# 5. Create visualizations
```
