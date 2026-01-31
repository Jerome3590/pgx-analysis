# Output Structure Framework

**Date:** December 9, 2025  
**Purpose:** Standardize output directory structure and sequential workflow across all analysis steps.

**Project alignment:** Step numbering and folder names match the main workflow: **1a_apcd_input_data**, **1b_apcd_event_filter**, **2_create_cohort**, **3a_feature_importance**, **3b_feature_importance_eda**, **4_model_data**, **5_pgx_analysis**, **6_final_model**, **7_shap_analysis**, **8_ffa_analysis**, **9_risk_dashboard**. Three workflow notebooks: 1_cohort_workflow.ipynb (1–2), 2_feature_importance.ipynb (3a–3b), 3_pgx_calculator_workflow.ipynb (4–9).

---

## Sequential Execution Workflow

**Critical Rule:** Each analysis step must be **fully completed** (all outputs and plots generated) before proceeding to the next step.

### Execution Order

Execution order matches the project's three workflow notebooks and step numbering:

```
Step 1a: APCD Input Data (1a_apcd_input_data)
  ↓
Step 1b: Event Filter – ICD/administrative (1b_apcd_event_filter)
  ↓
Step 2: Cohort Creation (2_create_cohort)
  ↓ (1_cohort_workflow.ipynb complete)
Step 3a: Feature Importance – MC-CV (3a_feature_importance)
  ↓
Step 3b: Feature Importance EDA – BupaR, code research (3b_feature_importance_eda)
  ↓ (2_feature_importance.ipynb complete; outputs refined cohort_feature_importance.csv)
Step 4: Model Data (4_model_data) – model_events.parquet from refined features
  ↓
Step 5: PGx Feature Engineering (5_pgx_analysis)
  ↓
Step 6: Final Model (6_final_model) – training and selection
  ↓
Step 7: SHAP Analysis (7_shap_analysis)
  ↓
Step 8: Formal Feature Attribution – FFA (8_ffa_analysis); uses SHAP to prioritize rules
  ↓
Step 9: Risk Dashboard (9_risk_dashboard)
  ↓ (Complete: outputs/ + visualizations/)
    - Frontend, Backend, Data Preparation, Deployment
    - Visualizations: BupaR, FP-Growth, DTW (dashboard-only, not used as model features)
```

### Completion Criteria

Before proceeding to the next step, verify:

1. ✅ **All data outputs exist** in `{step_folder}/outputs/`
   - Check for expected CSV, JSON, Parquet files
   - Verify file sizes are reasonable (not empty)
   - Confirm all cohorts/age bands processed

2. ✅ **All plots generated** in `{step_folder}/outputs/plots/`
   - Check for expected visualization files (PNG, PDF)
   - Verify plots are not corrupted
   - Confirm all visualizations are complete

3. ✅ **Results validated**
   - Review key outputs for reasonableness
   - Check for errors or warnings in logs
   - Verify S3 uploads completed (if applicable)

4. ✅ **Documentation updated**
   - Update step README with completion status
   - Document any issues or deviations
   - Note any manual interventions required

### Workflow Checklist

Use this checklist to track progress:

- [ ] **Step 3a: Feature Importance** (`3a_feature_importance/`)
  - [ ] All cohort/age-band combinations processed
  - [ ] Aggregated feature importance files generated
  - [ ] Individual model results saved
  - [ ] Results uploaded to S3 (if applicable)
  - [ ] **READY FOR STEP 3b** ✅

- [ ] **Step 3b: Feature Importance EDA** (`3b_feature_importance_eda/`)
  - [ ] BupaR post-target and code research completed
  - [ ] Refined cohort_feature_importance.csv generated per cohort/age_band
  - [ ] **READY FOR STEP 4** ✅

- [ ] **Step 4: Model Data** (`4_model_data/`)
  - [ ] Model events extracted for all cohorts/age bands using refined features
  - [ ] Results uploaded to S3 (if applicable)
  - [ ] **READY FOR STEP 5** ✅

- [ ] **Step 5: PGx Feature Engineering** (`5_pgx_analysis/`)
  - [ ] PGx features generated
  - [ ] Features integrated into model data
  - [ ] Results uploaded to S3 (if applicable)
  - [ ] **READY FOR STEP 6** ✅

- [ ] **Step 6: Final Model** (`6_final_model/`)
  - [ ] Feature integration completed
  - [ ] Models trained
  - [ ] Evaluation metrics computed
  - [ ] Results uploaded to S3 (if applicable)
  - [ ] **READY FOR STEP 7** ✅

- [ ] **Step 7: SHAP Analysis** (`7_shap_analysis/`)
  - [ ] SHAP values computed
  - [ ] Global importance calculated
  - [ ] Results uploaded to S3 (if applicable)
  - [ ] **READY FOR STEP 8** ✅

- [ ] **Step 8: Formal Feature Attribution** (`8_ffa_analysis/`)
  - [ ] Feature attribution computed (rule selection: first 100 + random 100 + all SHAP > 0)
  - [ ] Causal analysis completed
  - [ ] Results uploaded to S3 (if applicable)
  - [ ] **READY FOR STEP 9** ✅

- [ ] **Step 9: Risk Dashboard** (`9_risk_dashboard/`)
  - [ ] Models prepared for deployment (`outputs/models/`)
  - [ ] Metadata generated (`outputs/metadata/`)
  - [ ] CPIC data prepared (`outputs/cpic/`)
  - [ ] Frontend dashboard built (`frontend/`)
  - [ ] Backend API deployed (`backend/`)
  - [ ] Visualizations generated (`visualizations/`):
    - [ ] BupaR process mining visualizations (`visualizations/bupar/outputs/`)
    - [ ] FP-Growth pattern visualizations (`visualizations/fpgrowth/outputs/`)
    - [ ] DTW trajectory visualizations (`visualizations/dtw/outputs/`)
  - [ ] All outputs uploaded to S3 (if applicable)
  - [ ] **ANALYSIS COMPLETE** ✅

---

## Standard Analysis Step Structure

Each analytical step follows this consistent structure:

```
{analysis_folder}/
  ├── {analysis_name}_pipeline.ipynb          # Main Jupyter notebook (orchestrator)
  ├── {supporting_script_1}.py                # Supporting Python scripts
  ├── {supporting_script_2}.py                # Called by notebook
  ├── outputs/
  │   ├── {data_files}.csv
  │   ├── {data_files}.json
  │   ├── {data_files}.parquet
  │   └── plots/
  │       ├── {plot_files}.png
  │       ├── {plot_files}.jpg
  │       └── {plot_files}.pdf
  └── README.md
```

### Component Roles

1. **Main Jupyter Notebook**: 
   - Serves as the orchestrator and documentation
   - Contains high-level workflow, configuration, and results visualization
   - Calls supporting scripts to execute analysis logic
   - Example: `feature_importance_cohort_runner.ipynb`

2. **Supporting Scripts**:
   - Contain the actual implementation logic
   - Can be called from the notebook or run independently
   - Reusable and testable
   - Example: `run_cohort_1_0_12.py`, `cohort_fpgrowth.py`

3. **Output Directory**: Standardized location for all outputs (see below)

4. **README.md**: Documentation for the analysis step

### Directory Structure Rules

1. **Output Directory**: Each analysis step folder should have an `outputs/` subdirectory
   - Example: `3a_feature_importance/outputs/`
   - Example: `6_final_model/outputs/`
   - Example: `9_risk_dashboard/outputs/` (and `9_risk_dashboard/visualizations/{type}/outputs/` for dashboard visualizations)

2. **Plots Subdirectory**: All visualization files should be saved to `outputs/plots/`
   - Plots include: PNG, JPG, PDF, SVG files
   - Example: `3a_feature_importance/outputs/plots/top50_features.png`

3. **Data Files**: All data output files (CSV, JSON, Parquet, etc.) go directly in `outputs/`
   - Example: `3a_feature_importance/outputs/opioid_ed_0_12_aggregated_feature_importance.csv`

4. **Relative Paths**: When specifying `output_dir` in code, use paths relative to the analysis folder:
   - ✅ Good: `output_dir='3a_feature_importance/outputs'`
   - ✅ Good: `output_dir='outputs'` (if running from within the analysis folder)
   - ❌ Bad: `output_dir='/absolute/path/to/outputs'`
   - ❌ Bad: `output_dir='../outputs'` (outside analysis folder)

---

## Implementation Guidelines

### Python Scripts

```python
import os
from pathlib import Path

# Option 1: Specify full path relative to project root
output_dir = '3a_feature_importance/outputs'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Option 2: If running from within analysis folder
output_dir = 'outputs'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Save data files
data_file = os.path.join(output_dir, 'results.csv')
df.to_csv(data_file, index=False)

# Save plots
plot_file = os.path.join(plots_dir, 'visualization.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
```

### R Scripts

```r
# Set output directory
output_dir <- "outputs"
plots_dir <- file.path(output_dir, "plots")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

# Save data files
write_csv(df, file.path(output_dir, "results.csv"))

# Save plots
ggsave(
  filename = file.path(plots_dir, "visualization.png"),
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)
```

---

## Current Analysis Steps

### ✅ 3a_feature_importance
- **Notebook**: `feature_importance_cohort_runner.ipynb` (or use root `2_feature_importance.ipynb` for Steps 3a–3b)
- **Supporting Scripts**: `run_cohort_*.py`, `run_mc_feature_importance.py` (orchestrated by notebook)
- **Output Directory**: `3a_feature_importance/outputs/`
- **Plots Directory**: `3a_feature_importance/outputs/plots/` (auto-created)
- **Status**: Follows pattern ✅

### ✅ 9_risk_dashboard (Visualizations)
- **BupaR Visualizations**: `9_risk_dashboard/visualizations/bupar/`
  - **Scripts**: `run_analysis.py`, `create_bupar_outputs_*.R`, `create_plots.R`
  - **Output Directory**: `9_risk_dashboard/visualizations/bupar/outputs/`
  - **Status**: Dashboard visualization only (not used as features) ✅
- **FP-Growth Visualizations**: `9_risk_dashboard/visualizations/fpgrowth/`
  - **Scripts**: `run_analysis.py`, `create_plots.py`, `create_fpgrowth_features.py`
  - **Output Directory**: `9_risk_dashboard/visualizations/fpgrowth/outputs/`
  - **Status**: Dashboard visualization only (not used as features) ✅
- **DTW Visualizations**: `9_risk_dashboard/visualizations/dtw/`
  - **Scripts**: `create_dtw_visualizations.py`, `create_dtw_features.py`
  - **Output Directory**: `9_risk_dashboard/visualizations/dtw/outputs/`
  - **Status**: Dashboard visualization only (not used as features) ✅

### ⚠️ 8_ffa_analysis
- **Notebook**: `catboost_feature_attribution_analysis.ipynb`
- **Supporting Scripts**: `catboost_axp_explainer.py`, `ffa_analysis.py`
- **Output Directory**: Should use `8_ffa_analysis/outputs/`
- **Plots Directory**: Should use `8_ffa_analysis/outputs/plots/`
- **Status**: Needs verification

---

## S3 Upload Structure

When uploading to S3, maintain the same folder structure:

```
s3://pgxdatalake/gold/{analysis_name}/
  ├── {cohort}/
  │   ├── {age_band}/
  │   │   ├── {data_files}.csv
  │   │   └── plots/
  │   │       └── {plot_files}.png
```

**Example:**
```
s3://pgxdatalake/gold/feature_importance/
  └── opioid_ed/
      └── 0-12/
          ├── opioid_ed_0_12_aggregated_feature_importance.csv
          └── plots/
              └── opioid_ed_0_12_top50_features.png
```

---

## Benefits

1. **Consistency**: All analysis steps follow the same structure
2. **Organization**: Easy to find outputs for each analysis step
3. **Clean Separation**: Data files and plots are clearly separated
4. **Version Control**: Can easily `.gitignore` entire `outputs/` folders
5. **Portability**: Relative paths work across different environments

---

## Notebook-Script Pattern Guidelines

### Best Practices

1. **Notebook as Orchestrator**:
   - Notebook should contain high-level workflow and configuration
   - Use notebook cells to call supporting scripts with `!python script.py` or `subprocess`
   - Display results and visualizations in notebook cells
   - Document methodology and interpretation in markdown cells

2. **Scripts as Implementation**:
   - Scripts contain the actual analysis logic
   - Scripts should be executable independently (for testing/debugging)
   - Scripts should accept command-line arguments or configuration files
   - Scripts write results to `outputs/` directory

3. **Separation of Concerns**:
   - Notebook: Workflow, visualization, documentation
   - Scripts: Data processing, model training, computation
   - Helper modules: Reusable functions (in `py_helpers/` or `r_helpers/`)

### Example Pattern

**Notebook (`feature_importance_cohort_runner.ipynb`):**
```python
# Cell 1: Configuration
COHORT_NAME = "opioid_ed"
AGE_BAND = "0-12"
N_SPLITS = 25

# Cell 2: Run analysis
!python run_cohort_1_0_12.py

# Cell 3: Load and visualize results
import pandas as pd
results = pd.read_csv('outputs/opioid_ed_0_12_aggregated_feature_importance.csv')
# ... visualization code ...
```

**Supporting Script (`run_cohort_1_0_12.py`):**
```python
#!/usr/bin/env python3
from py_helpers.feature_importance_utils import run_cohort_analysis

result = run_cohort_analysis(
    cohort_name="opioid_ed",
    age_band="0-12",
    output_dir='3a_feature_importance/outputs'
)
```

## Step Dependencies

Each step may depend on outputs from previous steps:

| Step | Depends On | Key Inputs |
|------|------------|------------|
| **3a_feature_importance** | Cohort data (`2_create_cohort/`) | Cohort parquet (after 1a, 1b, 2) |
| **3b_feature_importance_eda** | Step 3a outputs | Aggregated feature importance |
| **4_model_data** | Step 3b outputs | Refined cohort_feature_importance.csv |
| **5_pgx_analysis** | Step 4 outputs | model_events.parquet |
| **6_final_model** | Step 5 outputs | PGx features + model events |
| **7_shap_analysis** | Step 6 outputs | Trained models, feature schema |
| **8_ffa_analysis** | Step 7 outputs | SHAP results, trained models |
| **9_risk_dashboard** | Steps 6–8 outputs | Models, SHAP, FFA, metadata |
| **9_risk_dashboard/visualizations** | Step 4 outputs | Model events (for BupaR/FP-Growth/DTW dashboard-only visualizations) |

**Important:** Always verify that prerequisite outputs exist before starting a new step.

## Validation Scripts

Consider creating validation scripts to check completion:

```python
# Example: validate_step_completion.py
import os
from pathlib import Path

def validate_step(step_folder, expected_outputs, expected_plots):
    """Validate that a step is complete before proceeding."""
    outputs_dir = Path(step_folder) / "outputs"
    plots_dir = outputs_dir / "plots"
    
    # Check outputs
    missing_outputs = []
    for output_file in expected_outputs:
        if not (outputs_dir / output_file).exists():
            missing_outputs.append(output_file)
    
    # Check plots
    missing_plots = []
    for plot_file in expected_plots:
        if not (plots_dir / plot_file).exists():
            missing_plots.append(plot_file)
    
    if missing_outputs or missing_plots:
        print(f"❌ Step incomplete:")
        if missing_outputs:
            print(f"  Missing outputs: {missing_outputs}")
        if missing_plots:
            print(f"  Missing plots: {missing_plots}")
        return False
    else:
        print(f"✅ Step complete: {step_folder}")
        return True

# Usage
if not validate_step("3a_feature_importance", 
                     ["opioid_ed_0_12_aggregated_feature_importance.csv"],
                     ["opioid_ed_0_12_top50_features.png"]):
    print("Cannot proceed to Step 4 until Step 3 is complete!")
    exit(1)
```

## Output Manifests

**Required:** Each analysis step's README must include an **Output Files Manifest** section that documents:

1. **Expected Data Files** (`outputs/`):
   - File naming patterns
   - Description of each file type
   - Required vs optional status
   - Example filenames

2. **Expected Visualization Files** (`outputs/plots/`):
   - Plot naming patterns
   - Description of each visualization
   - Required vs optional status
   - Example filenames

3. **Completion Checklist**:
   - Checklist items for verifying all outputs exist
   - S3 upload verification (if applicable)

**Purpose:**
- Track what files should be generated
- Verify completeness before proceeding to next step
- Document expected outputs for users
- Enable automated validation scripts

**Example Manifest Format:**
```markdown
## Output Files Manifest

### Expected Outputs Structure

#### Data Files (`outputs/`)
| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{pattern}.csv` | Description | ✅ Yes |

#### Visualization Files (`outputs/plots/`)
| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{pattern}.png` | Description | ✅ Yes |

### Completion Checklist
- [ ] All data files exist
- [ ] All plots generated
```

## Migration Checklist

For each analysis step:

- [ ] Ensure main Jupyter notebook exists as orchestrator
- [ ] Verify supporting scripts are called from notebook
- [ ] Create `outputs/` directory if it doesn't exist
- [ ] Create `outputs/plots/` directory for visualizations
- [ ] **Add Output Files Manifest to README** (required)
- [ ] Update code to use `{analysis_folder}/outputs` as `output_dir`
- [ ] Update plot saving code to use `{output_dir}/plots/`
- [ ] Update S3 upload paths to match structure
- [ ] Add validation checks for completion before proceeding
- [ ] Add `.gitignore` entry for `outputs/` if needed

---

## Questions or Issues?

See main project README or open an issue.

