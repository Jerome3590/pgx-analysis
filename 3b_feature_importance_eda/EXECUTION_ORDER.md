# Feature Importance EDA Execution Order

## Overview

Feature Importance EDA executes analyses in this order to properly filter **already-processed aggregated feature importances** from Step 3:
1. **BupaR post-target analysis** (identify pre/post F1120 ICD/CPT events using process mining)
2. **Code research and validation** (identify non-informative ICD/CPT codes - actual event-level filtering happens in Step 4b)
3. **Filter and refine** (filter post-target leakage features from aggregated feature importance list)

**Note**: This is NOT a DTW filter. Feature Importance EDA uses BupaR process mining and code research to filter aggregated feature importances, not raw event data. DTW is used separately in Step 4b (protocol filtering of event data) and Step 9 (dashboard trajectory visualizations).

## Folder Naming Convention

Folders reflect execution order:
- `0_icd_cpt_check/` - ICD/CPT code validation (runs first)
- `1_bupaR/` - BupaR analysis (runs second, after administrative filtering)

## Execution Order

The pipeline now executes in this order:

1. **BupaR Post-Target Analysis** (`1_bupaR/`)
   - Builds BupaR event logs from `model_events.parquet`
   - Runs pre- and post-F1120 sequence analyses
   - Calculates pre-F1120 and post-F1120 ratios for each feature
   - Identifies ICD/CPT codes that appear primarily after F1120 (>=80% post-F1120 ratio = post-target leakage)
   - Generates comprehensive BupaR features and visualizations
   - Outputs: `{cohort}_{age_band}_bupar_post_target_analysis.csv`
   - See `1_bupaR/README_bupaR.md` for complete BupaR process mining documentation

2. **Code Research and Validation** (`0_icd_cpt_check/`)
   - Researches and validates ICD/CPT codes by groups (ICD by chapter, CPT by range)
   - Loads administrative codes from `4b_event_filter/administrative_codes_lookup.json` (codes identified in 0_icd_cpt_check)
   - Identifies non-informative ICD/CPT codes (administrative, scheduling, protocol codes)
   - **Note**: This step only validates and identifies codes - actual event-level filtering happens in Step 4b
   - The `filter_and_refine_features.py` step filters post-target leakage features from aggregated importances, not administrative codes
   - See `0_icd_cpt_check/README_icd_cpt_check.md` for detailed validation process

3. **Create Safe Feature Filter**
   - Excludes features with >=80% post-F1120 ratio (pure post-target leakage)
   - Keeps all features with any pre-target presence
   - Keeps ALL features with ANY pre-F1120 presence (maximize information)
   - Explicitly includes F1120 for target creation
   - Outputs: `{cohort}_{age_band}_safe_feature_filter.json`
   - See `FEATURE_FILTERING_APPROACH.md` for detailed strategy

4. **Filter and Refine Features**
   - Combines outputs from BupaR analysis and code research
   - Applies safe feature filter to aggregated feature importances:
     - **Cases (target=1)**: Whitelist approach (only features from `all_features_to_keep`)
     - **Controls (target=0)**: Blacklist approach (exclude only post-target leakage features)
   - Filters features from aggregated importance list based on:
     - Post-target leakage (from BupaR safe feature filter)
     - **Note**: Administrative codes are identified through code research but filtered at event level in Step 4b
   - Outputs: `cohort_feature_importance.csv` (refined aggregated feature importances)

5. **Create BupaR Visualizations**
   - Generates visualization plots from BupaR analysis
   - Saves plots to `outputs/{cohort}/{age_band_fname}/plots/`
   - See `OUTPUTS_AND_VISUALIZATIONS.md` for complete visualization documentation

## Updated Files

### Scripts Updated
- `run_feature_importance_eda.py`: Orchestrates Feature Importance EDA workflow (BupaR analysis and feature filtering)
- `run_bupar_post_target_analysis.py`: Calls `1_bupaR/` scripts
- `create_bupar_post_target_analysis.py`: Creates post-target analysis CSV from BupaR outputs

### Folder Structure

```
3b_feature_importance_eda/
├── 0_icd_cpt_check/                 # ICD/CPT code validation (Step 1)
│   ├── analyze_code_groups.py
│   ├── validate_icd_cpt_codes.py
│   ├── administrative_codes_lookup.json
│   └── README_icd_cpt_check.md
├── 1_bupaR/                         # BupaR analysis (Step 2, after admin filtering)
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R  (POLYPHARMACY COHORT)
│   ├── create_plots.R
│   └── README_bupaR.md
├── run_bupar_post_target_analysis.py # Calls 1_bupaR/ scripts
├── create_bupar_post_target_analysis.py # Creates post-target analysis CSV
└── run_feature_importance_eda.py                   # Orchestrates execution order
```

## Rationale

**BupaR analysis runs first** because:
- Identifies pre vs post-F1120 events (critical for target leakage prevention)
- Calculates pre-F1120 and post-F1120 ratios for each feature in aggregated importances
- Outputs are used to create safe feature filter (exclude leakage, keep pre-target)
- Generates comprehensive process mining visualizations

**Code research runs second** because:
- Validates and identifies administrative codes for reference
- Informs Step 4b event-level filtering (not used in Step 3b feature filtering)
- Provides documentation and validation of code classifications

## Running Feature Importance EDA

**⚠️ Important: Use Full Path to Python Jupyter Environment**

All scripts require the **full path to the Python jupyter environment** to ensure they use the correct Python interpreter and installed packages.

```bash
# EC2 Python jupyter environment path
/home/pgx3874/jupyter-env/bin/python3.11

# Run for a single cohort/age band (using full path - EC2)
/home/pgx3874/jupyter-env/bin/python3.11 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band 13-24

# Or if python is already in PATH and points to jupyter-env:
python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band 13-24

# To find your Python path (if different):
which python  # or: which python3
```

# The pipeline will:
# 1. Run BupaR analysis (1_bupaR/) - identify pre/post F1120 events in aggregated importances
# 2. Research and validate codes (0_icd_cpt_check/) - identify administrative codes (for Step 4b reference)
# 3. Create safe feature filter - exclude leakage, keep pre-target features
# 4. Filter and refine aggregated feature importances (filter post-target leakage from importance list)
# 5. Create visualizations
```
