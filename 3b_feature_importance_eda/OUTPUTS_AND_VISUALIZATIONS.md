# Step 3b: Outputs and Visualizations

## Overview

Step 3b creates comprehensive outputs including data files, feature engineering files, and visualizations from BupaR process mining analysis.

## Output Directory Structure

```
3b_feature_importance_eda/outputs/
├── {cohort}/
│   └── {age_band_fname}/
│       ├── features/                    # BupaR feature files
│       ├── plots/                       # Visualization PNG files
│       ├── {cohort}_{age_band}_cohort_feature_importance.csv
│       └── {cohort}_{age_band}_feature_filtering_summary.json
└── feature_engineering/
    ├── dtw_features_{cohort}_{age_band}.csv
    └── dtw_added_features_{cohort}_{age_band}.csv
```

## Data Outputs

### 1. Refined Feature Importance Files

**Location:** `outputs/{cohort}/{age_band_fname}/`

| File | Description | Columns |
|------|-------------|---------|
| `{cohort}_{age_band}_cohort_feature_importance.csv` | **Primary output** - Refined feature importances for Step 4a | `feature`, `importance_normalized`, `importance_scaled` |
| `{cohort}_{age_band}_feature_filtering_summary.json` | Summary of filtering decisions | JSON with counts and filtering statistics |

**Features:**
- Feature names are sanitized (spaces/special chars → underscores)
- Filtered based on BupaR post-target analysis
- Filtered based on DTW non-value-added analysis
- Sorted by importance score

### 2. DTW Feature Engineering Files

**Location:** `outputs/feature_engineering/`

| File | Description | Columns |
|------|-------------|---------|
| `dtw_features_{cohort}_{age_band}.csv` | Intermediate DTW features | `mi_person_key` + DTW feature columns |
| `dtw_added_features_{cohort}_{age_band}.csv` | Final merged DTW features | `mi_person_key` + DTW feature columns |

**DTW Features Include:**
- Trajectory cluster memberships (drug, ICD, CPT)
- DTW distances to prototype trajectories
- Trajectory characteristics (length, diversity, temporal properties)
- Cluster properties (target rates, sizes)
- Archetype matching scores

**Example:** `dtw_features_opioid_ed_13_24.csv` contains 12 DTW features for 11,776 patients

### 3. BupaR Feature Files

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

## DTW Trajectory Visualizations

DTW visualizations are created automatically during DTW feature creation to visualize patient trajectory patterns.

### DTW Analysis Overview

| File | Description | Dimensions |
|------|-------------|------------|
| `dtw_trajectory_analysis_{cohort}_{age_band}.png` | **4-panel overview** of DTW trajectory analysis | 15" × 12" |

**Panel 1: Trajectory Length Distribution**
- Histogram of trajectory lengths (number of events per patient)
- Shows median trajectory length
- Purpose: Understand sequence length patterns

**Panel 2: Trajectory Diversity Distribution**
- Histogram of unique items per trajectory
- Shows median diversity
- Purpose: Understand how varied patient trajectories are

**Panel 3: Top 20 Most Common Items**
- Horizontal bar chart of most frequent items across all trajectories
- Shows top 20 items (drugs/ICD/CPT codes)
- Purpose: Identify common patterns in patient journeys

**Panel 4: DTW Distance Distribution**
- Histogram of minimum DTW distances to nearest prototype
- Shows median distance
- Purpose: Understand trajectory similarity patterns

### Sample Trajectory Timeline

| File | Description | Dimensions |
|------|-------------|------------|
| `dtw_sample_trajectories_{cohort}_{age_band}.png` | Sample patient trajectories visualized as timelines | 14" × variable |

**Content:**
- Shows 10 sample patient trajectories (shortest, median, longest, + random samples)
- Each trajectory displayed as a horizontal timeline
- Patient ID and trajectory length labeled
- First, middle, and last items labeled (for trajectories > 10 events)
- Purpose: Visualize actual patient journey patterns

**Note:** DTW visualizations require `matplotlib` and `seaborn`. If these are not available, visualizations will be skipped with a warning.

## S3 Upload Locations

All outputs are automatically uploaded to S3:

### Feature Importance Files
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_feature_filtering_summary.json`

### DTW Features
- `s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort}/{age_band}/dtw_features_{cohort}_{age_band}.csv`
- `s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort}/{age_band}/dtw_added_features_{cohort}_{age_band}.csv`

### BupaR Features
- `s3://pgxdatalake/gold/bupar/{cohort}/{age_band}/{cohort}_{age_band}_train_target_*_bupar.csv`

### Visualizations
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/*.png`
  - BupaR visualizations: `*_activity_frequency.png`, `*_gantt*.png`, `*_sequence*.png`
  - DTW visualizations: `dtw_trajectory_analysis_*.png`, `dtw_sample_trajectories_*.png`

## Example Output Summary

For `opioid_ed/13-24`:

### Data Files Created: 14
- 1 refined feature importance CSV
- 1 filtering summary JSON
- 2 DTW feature CSVs
- 10 BupaR feature CSVs

### Visualizations Created: 11 PNG files
- **BupaR (9 files):**
  - 1 overall activity frequency
  - 1 activity milestones Gantt
  - 1 activity sequence top
  - 1 overall ICD Gantt
  - 1 pre-F1120 activity frequency
  - 1 pre-F1120 Gantt
  - 1 post-F1120 activity frequency
  - 1 post-F1120 Gantt
  - 1 post-F1120 ICD Gantt
- **DTW (2 files):**
  - 1 trajectory analysis overview (4-panel)
  - 1 sample trajectory timeline

**Total Size:** ~1.5 MB (visualizations) + ~2 MB (data files)

## Usage in Downstream Steps

### Step 4a (Model Data Creation)
- **Primary Input:** `cohort_feature_importance.csv`
  - Used to filter events and features for model training
  - Feature names are sanitized (no spaces/special chars)

### Model Training
- **DTW Features:** `dtw_added_features_{cohort}_{age_band}.csv`
  - Merged with model data for training
- **BupaR Features:** Available for feature engineering (if needed)

### Analysis and Reporting
- **Visualizations:** Used for presentations and reports
- **Trace Files:** Used for sequence pattern analysis
- **Process Matrices:** Used for process flow analysis
