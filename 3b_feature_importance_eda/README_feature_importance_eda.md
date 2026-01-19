# Feature Importance EDA and Refinement

## Overview

Feature Importance EDA performs additional exploratory data analysis on **already processed aggregated feature importances** from Step 3, using:
1. **BupaR post-target analysis** to identify pre/post F1120 ICD/CPT events (target leakage detection)
2. **Code research and validation** (identify and validate non-informative ICD/CPT codes from lookup table - actual event-level filtering happens in Step 4b)
3. **Interactive code review and filtering** to refine feature selection (filters post-target leakage features from feature importance list)

**Note**: This is NOT a DTW filter. Feature Importance EDA uses BupaR process mining and code research to filter already-processed aggregated feature importances, not raw event data. DTW is used separately in Step 4b (protocol filtering) and Step 9 (dashboard visualizations).

Based on this EDA, we filter and update the aggregated feature importances to produce refined `cohort_feature_importance` files that feed into Step 4a model data creation.

## Purpose

- **Identify post-target leakage**: Use BupaR process mining to analyze sequences before and after the target event (F1120) to identify features that may leak future information
- **Research and validate codes**: Identify and validate administrative, scheduling, and non-medical codes through code research (actual event-level filtering happens in Step 4b)
- **Apply safe feature filtering**: Exclude post-target leakage features from aggregated feature importance list while keeping all pre-target features to maximize information available to the algorithm
- **Refine feature importances**: Update already-processed aggregated feature importances based on BupaR and code research findings
- **Output refined features**: Generate `cohort_feature_importance` files for Step 4a

**Key Point**: Feature Importance EDA filters **aggregated feature importances** (already processed from Step 3), not raw event data. It uses BupaR process mining and code research, not DTW.

## Inputs

- **Aggregated feature importances** from Step 3:
  - `3_feature_importance/outputs/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv`
- **Model events data** (for BupaR analysis):
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
- See `1_bupaR/README_bupaR.md` for complete file manifest

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

1. **Load aggregated feature importances** from Step 3 (already processed feature importance scores)
2. **BupaR Post-Target Analysis** (`1_bupaR/`):
   - Build BupaR event logs from `model_events.parquet`
   - Analyze sequences before and after target event (F1120)
   - Calculate pre-F1120 and post-F1120 ratios for each feature
   - Identify features that appear primarily post-target (>=80% post-F1120 ratio = potential leakage)
   - Generate comprehensive BupaR features and visualizations
   - Output: `{cohort}_{age_band}_bupar_post_target_analysis.csv`
   - See `1_bupaR/README_bupaR.md` for detailed BupaR process mining documentation
3. **Code Research and Validation** (`0_icd_cpt_check/`):
   - Load administrative codes from `4b_dtw_filter/administrative_codes_lookup.json`
   - Research and validate ICD/CPT codes by groups (ICD by chapter, CPT by range)
   - Identify non-informative ICD/CPT codes (administrative, scheduling, protocol codes)
   - **Note**: This step only validates and identifies codes - actual event-level filtering happens in Step 4b
   - See `0_icd_cpt_check/README_icd_cpt_check.md` for detailed validation process
4. **Create Safe Feature Filter**:
   - Exclude features with >=80% post-F1120 ratio (pure post-target leakage)
   - Keep ALL features with ANY pre-F1120 presence (maximize information)
   - Explicitly include F1120 for target creation
   - Output: `{cohort}_{age_band}_safe_feature_filter.json`
5. **Filter and Update Feature Importances**:
   - Apply safe feature filter (whitelist for cases, blacklist for controls)
   - Filter post-target leakage features from aggregated feature importance list (based on BupaR analysis)
   - **Note**: Administrative codes are identified through code research but filtered at event level in Step 4b
   - Adjust importance scores based on BupaR and code research findings
   - Generate refined `cohort_feature_importance` files
6. **Save outputs locally and upload to S3**:
   - Save all outputs to local filesystem
   - Upload to S3 for checkpointing and Step 4a consumption
   - Save checkpoint metadata to S3

## Prerequisites

**⚠️ Important: Python/Jupyter Environment Path**

All scripts require the **full path to the Python jupyter environment** to ensure they use the correct Python interpreter and installed packages. This is especially important when:
- Running scripts from different directories
- Using cron jobs or automated workflows
- Running on EC2 instances with multiple Python environments

**Example (EC2 Environment):**
```bash
# EC2 Python jupyter environment path
/home/pgx3874/jupyter-env/bin/python3.11

# Use full path when running scripts
/home/pgx3874/jupyter-env/bin/python3.11 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band 13-24

# Or set as environment variable
export PYTHON_ENV="/home/pgx3874/jupyter-env/bin/python3.11"
$PYTHON_ENV 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band 13-24

# To find your Python path (if different):
which python  # or: which python3
```

**For R scripts:**
```bash
# Find Rscript path
which Rscript

# Use full path
/usr/local/bin/Rscript 3b_feature_importance_eda/1_bupaR/create_bupar_outputs_opioid_ed.R 13-24
```

**For Jupyter notebooks (EC2 Environment):**
```bash
# Use full path to jupyter (EC2)
/home/pgx3874/jupyter-env/bin/jupyter notebook --no-browser --port=8888

# Or if jupyter is in PATH:
jupyter notebook --no-browser --port=8888
```

## Scripts

### Main Orchestration
- `run_feature_importance_eda.py` - Orchestration script to run all analyses in order
- `feature_importance_eda_workflow.py` - Interactive workflow script (can be run as notebook or script)
- `feature_importance_eda_interactive_analysis_cohort*.ipynb` - Cohort-specific interactive notebooks

### Analysis Scripts
- `run_bupar_post_target_analysis.py` - BupaR analysis for post-target leakage detection
- `create_bupar_post_target_analysis.py` - Creates post-target analysis CSV from BupaR outputs
- `filter_and_refine_features.py` - Main script to filter and refine feature importances

### Feature Filtering Scripts
- `create_safe_feature_filter_json.py` - Creates safe feature filter JSON (exclude leakage, keep pre-target)

### Validation Scripts
- `0_icd_cpt_check/analyze_code_groups.py` - Analyzes ICD/CPT codes by groups
- `0_icd_cpt_check/validate_icd_cpt_codes.py` - Interactive validation workflow

### R Scripts (BupaR Process Mining)
- `1_bupaR/create_bupar_outputs_opioid_ed.R` - BupaR analysis for opioid_ed cohort
- `1_bupaR/create_bupar_outputs_non_opioid_ed.R` - BupaR analysis for non_opioid_ed cohort
- See `1_bupaR/README_bupaR.md` for complete BupaR documentation

## Usage

### Script-Based Execution

```bash
# Run for a single cohort/age_band
python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band 13-24

# Run for all cohorts
python 3b_feature_importance_eda/run_feature_importance_eda.py --all-cohorts

# Run multiple cohorts sequentially using shell script
bash 3b_feature_importance_eda/run_multiple_cohorts.sh
```

### Interactive Notebook Execution

You can run multiple Jupyter notebooks interactively at the same time! Each notebook runs in its own kernel, so they operate independently.

#### Quick Start

1. **Start Jupyter** (if not already running):
   ```bash
   cd /home/pgx3874/pgx-analysis
   /home/pgx3874/jupyter-env/bin/jupyter notebook --no-browser --port=8888
   
   # Or if jupyter is in PATH:
   jupyter notebook --no-browser --port=8888
   ```

2. **Open multiple notebooks** in separate browser tabs:
   - `feature_importance_eda_interactive_analysis_cohort5.ipynb` (non_opioid_ed / 65-74)
   - `feature_importance_eda_interactive_analysis_cohort6.ipynb` (non_opioid_ed / 75-84)
   - `feature_importance_eda_interactive_analysis_cohort7.ipynb` (non_opioid_ed / 85-94)

3. **Run cells independently** - Each notebook has its own kernel and can run cells independently.

#### Important Considerations

**Resource Usage:**
- Each notebook uses memory and CPU
- Running 3 notebooks simultaneously will use ~3x the resources
- Monitor with `htop` or `nvidia-smi` (if using GPU)

**File Conflicts:**
- **Output files**: Each notebook writes to cohort-specific directories:
  - `outputs/non_opioid_ed/65_74/`
  - `outputs/non_opioid_ed/75_84/`
  - `outputs/non_opioid_ed/85_94/`
- **No conflicts**: Different output directories prevent file conflicts
- **S3 uploads**: May happen simultaneously, but AWS handles this

**R Script Execution:**
- Each notebook calls R scripts via `subprocess`
- Multiple R processes can run simultaneously
- R scripts write to cohort-specific output directories (no conflicts)

**Control Cohort Creation:**
- ✅ **No conflicts**: Control cohorts are age-band specific
  - Cohort 5 (65-74) uses: `cohort_name=non_opioid_non_ed/age_band=65-74/model_events.parquet`
  - Cohort 6 (75-84) uses: `cohort_name=non_opioid_non_ed/age_band=75-84/model_events.parquet`
  - Cohort 7 (85-94) uses: `cohort_name=non_opioid_non_ed/age_band=85-94/model_events.parquet`
- Each notebook creates/uses its own age-band-specific control cohort file
- **Safe to run in parallel**: No file conflicts between different age bands

#### Best Practices

**Option 1: Parallel Execution (Recommended)**
✅ **Safe to run all 3 notebooks in parallel immediately** - Each uses its own age-band-specific control cohort:
- Cohort 5 (65-74) → `cohort_name=non_opioid_non_ed/age_band=65-74/model_events.parquet`
- Cohort 6 (75-84) → `cohort_name=non_opioid_non_ed/age_band=75-84/model_events.parquet`
- Cohort 7 (85-94) → `cohort_name=non_opioid_non_ed/age_band=85-94/model_events.parquet`

**No conflicts**: Each notebook creates/uses a separate control cohort file.

**Option 2: Sequential Execution (If Resource Constrained)**
If you have limited resources (memory/CPU), run sequentially:
1. Run cohort 5 first
2. Once complete, run cohort 6
3. Then run cohort 7

This ensures:
- Easier to monitor progress
- Less resource contention
- But not required for avoiding conflicts (each age band has its own control cohort)

**Option 3: Use tmux/screen for Multiple Sessions**
```bash
# Start tmux session
tmux new -s cohort5
# In tmux, start Jupyter (use full path to jupyter environment - EC2)
/home/pgx3874/jupyter-env/bin/jupyter notebook --no-browser --port=8888

# Create new tmux window for cohort 6
tmux new-window -t cohort5:1
/home/pgx3874/jupyter-env/bin/jupyter notebook --no-browser --port=8889

# Create new tmux window for cohort 7
tmux new-window -t cohort5:2
/home/pgx3874/jupyter-env/bin/jupyter notebook --no-browser --port=8890
```

Then access:
- Cohort 5: `http://your-ec2-ip:8888`
- Cohort 6: `http://your-ec2-ip:8889`
- Cohort 7: `http://your-ec2-ip:8890`

#### Monitoring

**Check Running Notebooks:**
```bash
# List all Jupyter processes
ps aux | grep jupyter

# Check which Python environment is being used
which python
which jupyter

# Check notebook kernels
jupyter kernelspec list
```

**Monitor Resources:**
```bash
# CPU and memory
htop

# Disk I/O
iostat -x 1

# Check output directories
ls -lh /home/pgx3874/pgx-analysis/3b_feature_importance_eda/outputs/non_opioid_ed/
```

#### Troubleshooting

**Issue: "Control cohort not found" errors**
**Solution**: Each notebook will automatically create its age-band-specific control cohort if it doesn't exist. If you want to pre-create them:
```bash
# Create control cohorts for all three age bands
python 4a_model_data/create_control_cohort_model_data.py --age-band 65-74 --sample-size 100000
python 4a_model_data/create_control_cohort_model_data.py --age-band 75-84 --sample-size 100000
python 4a_model_data/create_control_cohort_model_data.py --age-band 85-94 --sample-size 100000
```

**Note**: Each age band creates its own separate control cohort file, so you can run these in parallel too!

**Issue: Out of memory**
**Solution**: 
- Run notebooks sequentially instead of parallel
- Close other applications
- Consider using the script-based approach (`run_feature_importance_eda.py`) which is more memory-efficient

**Issue: R script conflicts**
**Solution**: R scripts write to cohort-specific directories, so no conflicts. If you see errors, check:
- Each notebook is using the correct cohort/age_band
- Output directories are separate

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
│   └── README_icd_cpt_check.md   # Code validation documentation
├── 1_bupaR/                      # BupaR process mining analysis
│   ├── create_bupar_outputs_opioid_ed.R
│   ├── create_bupar_outputs_non_opioid_ed.R
│   ├── create_plots.R
│   └── README_bupaR.md           # BupaR process mining documentation
├── outputs/                      # All outputs organized by cohort/age_band
│   ├── {cohort}/
│   │   └── {age_band_fname}/
│   │       ├── features/         # BupaR feature files
│   │       ├── plots/            # Visualization PNG files
│   │       └── *.csv, *.json     # Analysis results
├── feature_importance_eda_workflow.py            # Main interactive workflow
├── feature_importance_eda_interactive_analysis_cohort*.ipynb  # Cohort-specific notebooks
├── run_feature_importance_eda.py                # Orchestration script
├── filter_and_refine_features.py  # Feature filtering and refinement
└── README_feature_importance_eda.md              # This file
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
- **`0_icd_cpt_check/README_icd_cpt_check.md`**: ICD/CPT code validation process
- **`1_bupaR/README_bupaR.md`**: BupaR process mining documentation