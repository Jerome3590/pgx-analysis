# DTW Dashboard Visualizations

## Overview

Dynamic Time Warping (DTW) trajectory visualizations for the risk dashboard. These visualizations show patient trajectory similarities, clustering patterns, and temporal sequences to complement risk predictions.

**⚠️ Important**: DTW visualizations are for **exploratory analysis only**. Event-level filtering is in Step 1b (`1b_apcd_event_filter`). DTW features are **NOT** used in the final model.

## Purpose

DTW visualizations help clinicians understand:
- **Trajectory Similarity**: How similar are patient drug exposure sequences
- **Patient Clustering**: Which patients follow similar pathways
- **Temporal Patterns**: Timing relationships in drug sequences
- **Representative Trajectories**: Typical patterns for different patient groups

## Visualization Types

### 1. Trajectory Analysis Overview
- **File Pattern**: `dtw_trajectory_analysis_{cohort}_{age_band}.png`
- **Description**: Overview visualization showing trajectory clustering and similarity patterns
- **Use Case**: Understand overall trajectory structure and patient groupings

### 2. Sample Trajectories
- **File Pattern**: `dtw_sample_trajectories_{cohort}_{age_band}.png`
- **Description**: Visualization of representative patient trajectories
- **Use Case**: See examples of typical patient pathways

### 3. Trajectory Metrics Chart
- **Data Format**: JSON metrics file
- **Description**: Bar chart showing trajectory metrics (cluster sizes, average distances, etc.)
- **Use Case**: Understand trajectory statistics and cluster characteristics

## Scripts

### Main Analysis Scripts

**`create_dtw_features.py`** - DTW feature extraction
- Computes DTW distances between patient sequences
- Performs trajectory clustering
- Generates trajectory features (for reference, not used in model)

**`create_dtw_visualizations.py`** - Visualization generator
- Generates trajectory analysis plots
- Creates sample trajectory visualizations
- Uploads outputs to S3

**`create_predictive_time_features.py`** - Time-based features
- Extracts temporal features from trajectories
- Generates time-to-event features (for reference, not used in model)

## Output Structure

### Local Outputs

```
9_risk_dashboard/visualizations/dtw/outputs/
├── feature_engineering/          # Feature files (for reference, not used in model)
│   └── predictive_time_features_{cohort}_{age_band}.csv
└── {cohort}/
    └── {age_band}/
        └── plots/                # Visualization files (for dashboard)
            ├── dtw_trajectory_analysis_{cohort}_{age_band}.png
            ├── dtw_sample_trajectories_{cohort}_{age_band}.png
            └── dtw_metrics_{cohort}_{age_band}.json
```

### S3 Outputs

**S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`

All visualization files (PNG and JSON) are uploaded to S3 for dashboard access via Lambda API.

## Usage

### Running DTW Visualizations

**Generate DTW features and visualizations:**
```bash
cd 9_risk_dashboard/visualizations/dtw
python create_dtw_features.py --cohort-name {cohort} --age-band {age_band}
python create_dtw_visualizations.py --cohort-name {cohort} --age-band {age_band}
```

**Example:**
```bash
python create_dtw_features.py --cohort-name opioid_ed --age-band 25-44
python create_dtw_visualizations.py --cohort-name opioid_ed --age-band 25-44
```

### Required Inputs

- **Model Events Data**: `4_model_data/` (or event-filter outputs) — e.g. `model_events_no_protocols.parquet` for DTW-filtered, or base `model_events.parquet`
  - Prefer DTW-filtered data (no protocols) when available for dashboard use
  - Falls back to base `model_events.parquet` if filtered version unavailable

### Output Verification

After running, verify:
- [ ] PNG visualization files generated in `outputs/{cohort}/{age_band}/plots/`
- [ ] JSON metrics file generated (if applicable)
- [ ] Files follow naming convention: `dtw_{plot_type}_{cohort}_{age_band}.{ext}`
- [ ] Files uploaded to S3 (if using orchestrator)

## Dashboard Integration

### API Endpoint

**`GET /visualizations/dtw`**
- Query params: `cohort`, `age_band`
- Returns: S3 paths to DTW visualization files (PNG images and JSON metrics)

### Frontend Display

Visualizations are displayed in the **DTW Trajectories** tab of the dashboard:
- User selects cohort and age band
- Dashboard loads visualization images from S3
- Images displayed in organized panels
- Metrics chart generated from JSON data using Plotly.js

### Filtering

Visualizations can be filtered by user-selected codes:
- Server-side filtering in Lambda function
- Shows only trajectories containing selected codes
- Updates visualization display dynamically

## DTW Usage in Pipeline

### Feature Importance EDA: Feature Refinement
- DTW trajectory analysis identifies non-value-added administrative/scheduling codes
- Used to filter features before model data extraction
- **Not** used as model features

### Event Filtering (Step 1b)
- DTW used to identify and remove standard care protocols
- Creates `model_events_no_protocols.parquet` for downstream steps
- Removes administrative codes that both targets and controls follow

### Step 9: Dashboard Visualizations
- DTW visualizations show trajectory similarities and clustering
- **Not** used as model features
- For exploratory analysis and clinical interpretation

## Dependencies

- **Python**: `pandas`, `numpy`, `scipy` (for DTW algorithm), `scikit-learn` (for clustering), `matplotlib`, `seaborn`, `boto3`
- **Input Data**: Model events parquet files from Step 4 (model data) or event-filter outputs (Step 1b)

## Notes

1. **Visualization Only**: DTW outputs are for visualization and exploratory analysis only, not model features.

2. **Event Filtering**: Event-level filtering runs in Step 1b (`1b_apcd_event_filter`). DTW visualizations show trajectory patterns for dashboard only; these are not predictive features.

3. **Sequence Comparison**: DTW compares patient drug exposure sequences to identify similar trajectories. This helps understand patient groupings but does not directly predict outcomes.

4. **Clustering**: Patient trajectories are clustered based on DTW distance. Visualizations show cluster representatives and statistics.

5. **Temporal Features**: Time-based features are extracted but not used in the final model. They are available for reference and exploratory analysis.

## Related Documentation

- **[Dashboard Visualizations Overview](../../9_risk_dashboard/visualizations/README.md)** - General visualization documentation
- **[Dashboard API Documentation](README_results_dashboard_visualizations.md)** - Complete dashboard visualization guide
- **[Event Filtering (Step 1b)](../../1b_apcd_event_filter/)** - ICD/administrative code filtering before cohort creation
- **[Feature Importance EDA](../../3b_feature_importance_eda/)** - BupaR post-target analysis for feature refinement
