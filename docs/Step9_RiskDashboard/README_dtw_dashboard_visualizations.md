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

### 1. Trajectory Cluster Plots (Plotly 3D / 1D)
- **File Patterns**:
  - **3D** (multi-axis cohorts, e.g. opioid_ed): `dtw_trajectory_cluster_3d_{cohort}_{age_band}.html` (and optional `.png`)
  - **1D** (polypharmacy cohort `non_opioid_ed` only): `dtw_trajectory_cluster_1d_{cohort}_{age_band}.html` (and optional `.png`)
- **Description**: Interactive Plotly scatter plots of trajectories. Each axis is the count of a code in the patient’s sequence (from `seq_pattern_str`). Points are colored by KMeans cluster. For 3D, axes are the top 3 codes by frequency; for polypharmacy, a single axis (top code count) is used.
- **Use Case**: Explore trajectory clusters by code counts; same pattern as FP-Growth visuals (HTML written to plots dir, uploaded to dashboard S3).
- **Script**: `create_dtw_plots.py` (invoked by `create_dtw_visuals.py`).

### 2. Trajectory Analysis Overview
- **File Pattern**: `dtw_trajectory_analysis_{cohort}_{age_band}.png`
- **Description**: Overview visualization showing trajectory clustering and similarity patterns
- **Use Case**: Understand overall trajectory structure and patient groupings

### 3. Sample Trajectories
- **File Pattern**: `dtw_sample_trajectories_{cohort}_{age_band}.png`
- **Description**: Visualization of representative patient trajectories
- **Use Case**: See examples of typical patient pathways

### 4. Trajectory Metrics Chart
- **Data Format**: JSON metrics file
- **Description**: Bar chart showing trajectory metrics (cluster sizes, average distances, etc.)
- **Use Case**: Understand trajectory statistics and cluster characteristics

## Scripts

### Main Analysis Scripts

**`create_dtw_features.py`** - DTW feature extraction
- Computes DTW distances between patient sequences
- Performs trajectory clustering
- Generates trajectory features and `seq_pattern_str` (for reference, not used in model)

**`create_dtw_visuals.py`** - Publish DTW visuals for the dashboard
- Copies DTW features to outputs, mirrors to feature_engineering_outputs, uploads to S3
- Builds chart data (routine vs no routine, high-risk trajectories) and uploads to dashboard S3
- Calls `create_dtw_plots.py` to generate 3D/1D trajectory cluster plots, then uploads plots (PNG and HTML) to the dashboard bucket

**`create_dtw_plots.py`** - Plotly trajectory cluster plots
- Builds per-patient code counts from `seq_pattern_str` in the DTW features CSV
- For non–polypharmacy cohorts: 3D Plotly scatter (axes = top 3 code counts), KMeans clusters
- For polypharmacy (`non_opioid_ed`): 1D plot (one axis = top code count)
- Writes HTML (and optional PNG) to `10d_dtw_dashboard_visual/outputs/{cohort}/{age_band}/plots/`
- CLI: `python create_dtw_plots.py --cohort-name {cohort} --age-band {age_band} [--n-clusters 5] [--force]`

**`create_predictive_time_features.py`** - Time-based features
- Extracts temporal features from trajectories
- Generates time-to-event features (for reference, not used in model)

## Output Structure

### Local Outputs

DTW visuals are written under `10d_dtw_dashboard_visual/outputs/` (and optionally mirrored to `5_feature_engineering/feature_engineering_outputs/6_dtw/`). Plots directory:

```
10d_dtw_dashboard_visual/outputs/
├── feature_engineering/          # DTW features CSV (create_dtw_features.py → create_dtw_visuals.py)
│   ├── dtw_features_{cohort}_{age_band}.csv
│   └── dtw_added_features_{cohort}_{age_band}.csv
└── {cohort}/
    └── {age_band}/
        └── plots/                # Visualization files (for dashboard)
            ├── dtw_trajectory_cluster_3d_{cohort}_{age_band}.html   # or 1d for non_opioid_ed
            ├── dtw_trajectory_cluster_3d_{cohort}_{age_band}.png   # optional
            ├── dtw_trajectory_analysis_{cohort}_{age_band}.png
            ├── dtw_sample_trajectories_{cohort}_{age_band}.png
            └── dtw_metrics_{cohort}_{age_band}.json
```

### S3 Outputs

**S3 Location**: Dashboard bucket under `{S3_DASHBOARD_PREFIX}/dtw/{cohort}/{age_band}/plots/` (same pattern as FP-Growth/BupaR).

All visualization files (PNG, HTML, and JSON) are uploaded to S3 for dashboard access via Lambda API.

## Usage

### Running DTW Visualizations

**Generate DTW features and visualizations:**
```bash
cd 10_risk_dashboard/visualizations/dtw
python create_dtw_features.py --cohort-name {cohort} --age-band {age_band}
python create_dtw_visuals.py --cohort-name {cohort} --age-band {age_band}
```

`create_dtw_visuals.py` runs the full publish pipeline (copy features, mirror, S3 upload, chart data, and trajectory cluster plots). To generate only the 3D/1D cluster plots:

```bash
python create_dtw_plots.py --cohort-name {cohort} --age-band {age_band} [--n-clusters 5] [--force]
```

**Example:**
```bash
python create_dtw_features.py --cohort-name opioid_ed --age-band 25-44
python create_dtw_visuals.py --cohort-name opioid_ed --age-band 25-44
```

### Required Inputs

- **Model Events Data**: `4_model_data/` (or event-filter outputs) — e.g. `model_events_no_protocols.parquet` for DTW-filtered, or base `model_events.parquet`
  - Prefer DTW-filtered data (no protocols) when available for dashboard use
  - Falls back to base `model_events.parquet` if filtered version unavailable

### Output Verification

After running, verify:
- [ ] Trajectory cluster plot(s) generated: `dtw_trajectory_cluster_3d_*.html` or `dtw_trajectory_cluster_1d_*.html` (polypharmacy) in `10d_dtw_dashboard_visual/outputs/{cohort}/{age_band}/plots/`
- [ ] PNG visualization files (and optional HTML) in plots dir
- [ ] JSON metrics/chart data as applicable
- [ ] Files follow naming convention: `dtw_{plot_type}_{cohort}_{age_band}.{ext}`
- [ ] Files uploaded to S3 (if using create_dtw_visuals)

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

- **Python**: `pandas`, `numpy`, `scipy` (for DTW algorithm), `scikit-learn` (for clustering and trajectory cluster plots), `plotly` (for 3D/1D cluster HTML), `matplotlib`, `seaborn`, `boto3`
- **Input Data**: Model events parquet files from Step 4 (model data) or event-filter outputs (Step 1b)

## Notes

1. **Visualization Only**: DTW outputs are for visualization and exploratory analysis only, not model features.

2. **Event Filtering**: Event-level filtering runs in Step 1b (`1b_apcd_event_filter`). DTW visualizations show trajectory patterns for dashboard only; these are not predictive features.

3. **Sequence Comparison**: DTW compares patient drug exposure sequences to identify similar trajectories. This helps understand patient groupings but does not directly predict outcomes.

4. **Clustering**: Patient trajectories are clustered based on DTW distance. Visualizations show cluster representatives and statistics.

5. **Temporal Features**: Time-based features are extracted but not used in the final model. They are available for reference and exploratory analysis.

## Related Documentation

- **[Dashboard Visualizations Overview](../../10_risk_dashboard/visualizations/README.md)** - General visualization documentation
- **[Dashboard API Documentation](README_results_dashboard_visualizations.md)** - Complete dashboard visualization guide
- **[Event Filtering (Step 1b)](../../1b_apcd_event_filter/)** - ICD/administrative code filtering before cohort creation
- **[Feature Importance EDA](../../3b_feature_importance_eda/)** - BupaR post-target analysis for feature refinement
