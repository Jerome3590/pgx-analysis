# BupaR Dashboard Visualizations

## Overview

BupaR process mining visualizations for the risk dashboard. These visualizations show patient pathways, activity sequences, and temporal patterns to complement risk predictions.

**⚠️ Important**: BupaR visualizations are for **exploratory analysis only**. BupaR is also used in Feature Importance EDA for feature refinement (post-target analysis), but BupaR features are **NOT** used in the final model.

## Purpose

BupaR visualizations help clinicians understand:
- **Patient Pathways**: How patients progress through sequences of drugs, diagnoses, and procedures
- **Temporal Patterns**: Timing relationships between events
- **Process Flows**: Common sequences and transitions in patient care
- **Pre/Post Target Analysis**: Pathways before and after target events (for opioid_ed cohort)

## Visualization Types

### 1. Activity Frequency Charts
- **File Pattern**: `{cohort}_{age_band}_overall_activity_frequency.png`
- **Description**: Bar chart showing frequency of each activity (drug, ICD, CPT code)
- **Use Case**: Identify most common activities in patient pathways

### 2. Gantt Charts
- **File Pattern**: `{cohort}_{age_band}_gantt.png`
- **Description**: Timeline visualization showing activity sequences per patient
- **Use Case**: Visualize temporal progression of patient care

### 3. Activity Sequence Charts
- **File Pattern**: `{cohort}_{age_band}_activity_sequence_top.png`
- **Description**: Bar chart of most frequent activity sequences
- **Use Case**: Identify common pathways patients follow

### 4. Pre/Post Target Visualizations (opioid_ed only)
- **Pre-F1120**: `{cohort}_{age_band}_pre_f1120_activity_frequency.png`, `pre_f1120_gantt.png`
- **Post-F1120**: `{cohort}_{age_band}_post_f1120_activity_frequency.png`, `post_f1120_gantt.png`
- **Description**: Separate visualizations for pathways before and after F1120 (opioid dependence) diagnosis
- **Use Case**: Understand how patient pathways change after diagnosis

### 5. Activity Milestones Gantt
- **File Pattern**: `{cohort}_{age_band}_activity_milestones_gantt.png`
- **Description**: Gantt chart highlighting key milestones in patient pathways
- **Use Case**: Identify critical transition points

## Scripts

### Main Analysis Scripts

**`create_bupar_visuals.py`** - Python orchestrator
- Creates BupaR visuals for specified cohort/age band
- Generates all visualization outputs
- Uploads to S3

**`create_bupar_outputs_opioid_ed.R`** - R script for opioid_ed cohort
- Generates pre/post F1120 visualizations
- Creates process mining outputs
- Generates all plot files

**`create_bupar_outputs_non_opioid_ed.R`** - R script for non_opioid_ed cohort
- Generates general process mining visualizations
- Creates activity frequency and sequence charts

**`create_plots.R`** - R plotting utilities
- Shared plotting functions
- Gantt chart generation
- Activity frequency visualization

## Output Structure

### Local Outputs

```
9_risk_dashboard/visualizations/bupar/outputs/
├── {cohort}/
│   └── {age_band}/
│       ├── features/          # Feature files (for reference, not used in model)
│       │   ├── {cohort}_{age_band}_train_target_pre_f1120_patient_features_bupar.csv
│       │   ├── {cohort}_{age_band}_train_target_post_f1120_patient_features_bupar.csv
│       │   └── ...
│       └── plots/             # Visualization files (for dashboard)
│           ├── {cohort}_{age_band}_overall_activity_frequency.png
│           ├── {cohort}_{age_band}_gantt.png
│           ├── {cohort}_{age_band}_activity_sequence_top.png
│           ├── {cohort}_{age_band}_pre_f1120_activity_frequency.png
│           ├── {cohort}_{age_band}_pre_f1120_gantt.png
│           ├── {cohort}_{age_band}_post_f1120_activity_frequency.png
│           ├── {cohort}_{age_band}_post_f1120_gantt.png
│           └── {cohort}_{age_band}_activity_milestones_gantt.png
```

### S3 Outputs

**S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`

All PNG visualization files are uploaded to S3 for dashboard access via Lambda API.

## Usage

### Running BupaR Visualizations

**For opioid_ed cohort:**
```bash
cd 9_risk_dashboard/visualizations/bupar
Rscript create_bupar_outputs_opioid_ed.R {age_band}
```

**For non_opioid_ed cohort:**
```bash
cd 9_risk_dashboard/visualizations/bupar
Rscript create_bupar_outputs_non_opioid_ed.R {age_band}
```

**Using Python orchestrator:**
```bash
cd 9_risk_dashboard/visualizations/bupar
python create_bupar_visuals.py --cohort-name {cohort} --age-band {age_band}
```

### Required Inputs

- **Model Events Data**: `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`
  - Event-level filtering is applied in Step 1b (`1b_apcd_event_filter`) before cohort creation.

### Output Verification

After running, verify:
- [ ] All PNG plot files generated in `outputs/{cohort}/{age_band}/plots/`
- [ ] Files follow naming convention: `{cohort}_{age_band}_{plot_type}.png`
- [ ] Files uploaded to S3 (if using orchestrator)
- [ ] Dashboard can access files via Lambda API

## Dashboard Integration

### API Endpoint

**`GET /visualizations/bupar`**
- Query params: `cohort`, `age_band`
- Returns: S3 paths to BupaR visualization images

### Frontend Display

Visualizations are displayed in the **BupaR Visualizations** tab of the dashboard:
- User selects cohort and age band
- Dashboard loads visualization images from S3
- Images displayed in organized panels

### Filtering

Visualizations can be filtered by user-selected codes (drugs, ICDs, CPTs):
- Server-side filtering in Lambda function
- Shows only pathways containing selected codes
- Updates visualization display dynamically

## Dependencies

- **R Packages**: `bupaR`, `processmapR`, `eventdataR`, `ggplot2`, `dplyr`
- **Python**: `pandas`, `boto3` (for S3 uploads)
- **Input Data**: Model events parquet files from Step 4 (`4_model_data/`)

## Notes

1. **Feature Engineering**: BupaR features are generated but **NOT** used in the final model. They are for visualization only.

2. **Feature Importance EDA Usage**: BupaR is also used in Feature Importance EDA (`3b_feature_importance_eda/`) for post-target analysis to identify leakage features. This is separate from dashboard visualizations.

3. **Event Filtering**: Event-level ICD/administrative code filtering runs in Step 1b (`1b_apcd_event_filter`); model data (Step 4) uses that filtered event set.

4. **Cohort-Specific**: `opioid_ed` cohort has additional pre/post F1120 visualizations that `non_opioid_ed` does not have.

## Related Documentation

- **[Dashboard Visualizations Overview](../../9_risk_dashboard/visualizations/README.md)** - General visualization documentation
- **[Dashboard API Documentation](README_results_dashboard_visualizations.md)** - Complete dashboard visualization guide
- **[Feature Importance EDA](../../3b_feature_importance_eda/)** - BupaR post-target analysis
