# Feature Importance EDA: Outputs and Visualizations

## Overview

Feature Importance EDA creates comprehensive outputs including data files, feature engineering files, and visualizations from BupaR process mining analysis.

## Output Directory Structure

```
3b_feature_importance_eda/outputs/
├── {cohort}/
│   └── {age_band_fname}/
│       ├── features/                    # BupaR feature files
│       ├── plots/                       # Visualization PNG files
│       ├── {cohort}_{age_band_fname}_cohort_feature_importance.csv
│       └── {cohort}_{age_band_fname}_feature_filtering_summary.json
│       ├── {cohort}_{age_band_fname}_bupar_post_target_analysis.csv
│       └── {cohort}_{age_band_fname}_safe_feature_filter.json
```

## Data Outputs

### 1. Refined Feature Importance Files

**Location:** `outputs/{cohort}/{age_band_fname}/`

| File | Description | Columns |
|------|-------------|---------|
| `{cohort}_{age_band_fname}_cohort_feature_importance.csv` | **Primary output** - Refined feature importances for Step 4/6 (final model features) | `feature`, importance columns, `code_type`, `raw_code` |
| `{cohort}_{age_band_fname}_feature_filtering_summary.json` | Summary of filtering decisions | JSON with counts and filtering statistics |

**Features:**
- Feature names are sanitized (spaces/special chars → underscores)
- Filtered based on BupaR post-target analysis (safe feature filter)
- Filtered based on administrative/non-informative codes
- Sorted by importance score

### 2. BupaR Feature Files

**Location:** `outputs/{cohort}/{age_band_fname}/features/`

| File | Description | Use Case |
|------|-------------|----------|
| `{cohort}_{age_band}_train_target_pre_f1120_patient_features_bupar.csv` | Pre-F1120 per-patient features | Process mining features before target event |
| `{cohort}_{age_band}_train_target_post_f1120_patient_features_bupar.csv` | Post-F1120 per-patient features | Descriptive analysis (post-target leakage detection) |
| `{cohort}_{age_band}_train_target_time_to_f1120_features_bupar.csv` | Time-to-F1120 features | Temporal features (30d, 90d, 180d windows) |
| `{cohort}_{age_band}_train_target_traces_bupar.csv` | All trace sequences | Complete sequence patterns |
| `{cohort}_{age_band}_train_target_traces_top_bupar.csv` | Top (frequent) sequences | Most common patterns |
| `{cohort}_{age_band}_train_target_traces_rare_bupar.csv` | Rare (unique) sequences | Uncommon patterns |
| `{cohort}_{age_band}_train_target_pre_f1120_traces_top_bupar.csv` | Pre-F1120 top sequences | Frequent patterns before target |
| `{cohort}_{age_band}_train_target_pre_f1120_traces_rare_bupar.csv` | Pre-F1120 rare sequences | Rare patterns before target |
| `{cohort}_{age_band}_train_target_post_f1120_traces_bupar.csv` | Post-F1120 traces | Sequences after target |
| `{cohort}_{age_band}_train_target_post_f1120_traces_top_bupar.csv` | Post-F1120 top sequences | Frequent post-target patterns |
| `{cohort}_{age_band}_train_target_post_f1120_traces_rare_bupar.csv` | Post-F1120 rare sequences | Rare post-target patterns |
| `{cohort}_{age_band}_train_target_process_matrix_bupar.csv` | Process flow matrix | Activity transition frequencies |

**Note:** Some files may be empty if no events exist for that time period (e.g., pre-F1120 files for cohorts where all events occur after target).

## Visualizations

**Location:** `outputs/{cohort}/{age_band_fname}/plots/`

All visualizations are PNG files (300 DPI) created using ggplot2 in R.

### Overall Process Visualizations

| File | Description | Dimensions |
|------|-------------|------------|
| `{cohort}_{age_band}_overall_activity_frequency.png` | Bar chart of most frequent activities (top 30) | 12" × 10" |
| `{cohort}_{age_band}_activity_milestones_gantt.png` | Gantt chart showing activity timeline for sample patients (up to 30) | 16" × 12" |
| `{cohort}_{age_band}_activity_sequence_top.png` | Sequence plot highlighting top 10 activities | 16" × 12" |
| `{cohort}_{age_band}_gantt_icd.png` | ICD codes timeline (Gantt chart) | 18" × 12" |

### Pre-F1120 Visualizations

| File | Description | Dimensions |
|------|-------------|------------|
| `{cohort}_{age_band}_pre_f1120_activity_frequency.png` | Activity frequency before first F1120 event | 10" × 8" |
| `{cohort}_{age_band}_pre_f1120_gantt.png` | Pre-F1120 timeline (Gantt chart) | 14" × 10" |

**Note:** Pre-F1120 visualizations may be empty if no events occur before the target event.

### Post-F1120 Visualizations

| File | Description | Dimensions |
|------|-------------|------------|
| `{cohort}_{age_band}_post_f1120_activity_frequency.png` | Activity frequency after first F1120 event | 10" × 8" |
| `{cohort}_{age_band}_post_f1120_gantt.png` | Post-F1120 timeline (Gantt chart) | 14" × 10" |
| `{cohort}_{age_band}_post_f1120_gantt_icd.png` | Post-F1120 ICD codes timeline | 16" × 10" |

### Code Type-Specific Gantt Charts

The following visualizations are created conditionally (only if events of that type exist):

| File | Description | Dimensions |
|------|-------------|------------|
| `{cohort}_{age_band}_gantt_drugs.png` | Drug codes timeline (Gantt chart) | 18" × 12" |
| `{cohort}_{age_band}_gantt_cpt.png` | CPT codes timeline (Gantt chart) | 18" × 12" |
| `{cohort}_{age_band}_pre_f1120_gantt_drugs.png` | Pre-F1120 drug codes timeline | 16" × 10" |
| `{cohort}_{age_band}_pre_f1120_gantt_cpt.png` | Pre-F1120 CPT codes timeline | 16" × 10" |
| `{cohort}_{age_band}_post_f1120_gantt_drugs.png` | Post-F1120 drug codes timeline | 16" × 10" |
| `{cohort}_{age_band}_post_f1120_gantt_cpt.png` | Post-F1120 CPT codes timeline | 16" × 10" |


## S3 Upload Locations

All outputs are automatically uploaded to S3:

### Feature Importance Files
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_feature_filtering_summary.json`

### BupaR Features
- `s3://pgxdatalake/gold/bupar/{cohort}/{age_band}/{cohort}_{age_band}_train_target_*_bupar.csv`

### Visualizations
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/*.png`
  - BupaR visualizations: `*_activity_frequency.png`, `*_gantt*.png`, `*_sequence*.png`

## Example Output Summary

For `opioid_ed/13-24`:

### Data Files Created: 12
- 1 refined feature importance CSV
- 1 filtering summary JSON
- 1 post-target analysis CSV
- 1 safe feature filter JSON
- 10+ BupaR feature CSVs

### Visualizations Created: 9+ PNG files
- **BupaR Process Mining Visualizations:**
  - 1 overall activity frequency
  - 1 activity milestones Gantt
  - 1 activity sequence top
  - 1 overall ICD Gantt (if ICD events exist)
  - 1 pre-F1120 activity frequency (if pre-F1120 events exist)
  - 1 pre-F1120 Gantt (if pre-F1120 events exist)
  - 1 post-F1120 activity frequency
  - 1 post-F1120 Gantt
  - 1 post-F1120 ICD Gantt (if ICD events exist)
  - Additional code-type specific Gantt charts (drugs, CPT) if events exist

**Total Size:** ~1-2 MB (visualizations) + ~2-5 MB (data files, depending on cohort size)

## Troubleshooting: "No cohort_feature_importance.csv" / Script not finding files locally

**Why the script doesn't find files locally:** Step 4/6 and the notebook resolve `cohort_feature_importance` in this order: `3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}/`, then `DATA_ROOT/gold/feature_importance/`, then S3. If the file is missing, it's because the **Filter and refine** step (step 4 in the execution order) was never run for that cohort/age_band, or it failed (e.g. no aggregated FI, or `final_count == 0`).

**Correct file for final model features (Step 4 and Step 6):**

- **Path:** `3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_cohort_feature_importance.csv`
- **Example:** `3b_feature_importance_eda/outputs/opioid_ed/65_74/opioid_ed_65_74_cohort_feature_importance.csv`
- This is the **only** file Step 4 (model_events filter) and Step 6 (final model features) use; it is produced only by the filter-and-refine step.

**How to generate the missing file:** Run the filter-and-refine step for that cohort/age_band. From the project root:

```bash
python 3b_feature_importance_eda/2_filtering/filter_and_refine_features.py --cohort opioid_ed --age-band 65-74
```

Or run the full Step 3b workflow for that band (which runs BupaR, safe filter, then filter-and-refine):

```bash
python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band 65-74
```

Prerequisites: (1) Step 3a aggregated FI for that band (e.g. `3a_feature_importance/outputs/opioid_ed/opioid_ed_65_74_aggregated_feature_importance.csv`), (2) BupaR post-target CSV in `3b_feature_importance_eda/outputs/opioid_ed/65_74/`. The filter step writes the refined CSV and uploads it to S3.

## Usage in Downstream Steps

### Step 4a (Model Data Creation)
- **Primary Input:** `cohort_feature_importance.csv`
  - Used to filter events and features for model training
  - Feature names are sanitized (no spaces/special chars)

### Model Training
- **BupaR Features:** Available for feature engineering (if needed)
  - Pre-F1120 features: `*_pre_f1120_patient_features_bupar.csv`
  - Time-to-F1120 features: `*_time_to_f1120_features_bupar.csv`

### Analysis and Reporting
- **Visualizations:** Used for presentations and reports
- **Trace Files:** Used for sequence pattern analysis
- **Process Matrices:** Used for process flow analysis
