# DTW Dashboard Visualizations

## Overview

Dynamic Time Warping (DTW) trajectory visualizations for the risk dashboard. These visualizations show patient trajectory similarities, clustering patterns, and temporal sequences to complement risk predictions.

**DTW Alignment IS Computed**: DTW distances are computed using the dtaidistance library (create_dtw_features.py) to measure trajectory similarity. We do **not** use DTW for feature engineering due to target leakage concerns. We **do** use DTW **with feature importance** (SHAP/FFA allowed codes) for **analysis and answering research questions** (e.g. routine vs utilization, time-between aligned sequences) and dashboard display.

**⚠️ Important**: DTW is not used in the final model (target leakage). Event-level filtering is in Step 1b (`1b_apcd_event_filter`). DTW is used with feature importance for analysis and answering research questions.

**Time-between (N3):** We use DTW to get time-between and time-to-target for **aligned** sequences. This is more accurate than a straight comparison using BupaR: alignment makes intervals comparable across patients (like-with-like), whereas a straight BupaR aggregate of consecutive-event intervals mixes different stages and sequence lengths and is less interpretable for “what times between sequences lead to target outcomes?”

## Purpose

DTW visualizations help clinicians understand:
- **Trajectory Similarity**: How similar are patient medication claims-event sequences
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

### 3. Trajectory Metrics Chart
- **Data Format**: JSON metrics file
- **Description**: Bar chart showing trajectory metrics (cluster sizes, average distances, etc.)
- **Use Case**: Understand trajectory statistics and cluster characteristics

## Scripts

### Main Analysis Scripts

**`create_dtw_trajectories.py`** - Step 1: Trajectory extraction
- Extracts patient trajectories from model_data
- Generates seq_pattern_str (sequence of activity codes)
- Prepares data for DTW alignment (no distance computation in this step)

**`create_dtw_features.py`** - Step 2: DTW alignment and distance computation
- Computes DTW distances between patient sequences using dtaidistance library
- Selects prototype trajectories (evenly spaced by length)
- Augments CSV with dtw_min_distance and dtw_distance_to_prototype_* columns
- Generates common_sequences.json with prototype sequences
- DTW alignment IS computed for visualization analysis (not used as model features)

**`create_dtw_visuals.py`** - Publish DTW visuals for the dashboard
- Copies DTW features to outputs, mirrors to feature_engineering_outputs, uploads to S3
- Builds chart data (routine vs utilization, routine × medical utilization, high-risk trajectories) and uploads to dashboard S3
- Calls `create_dtw_plots.py` to generate 3D/1D trajectory cluster plots, then uploads plots (PNG and HTML) to the dashboard bucket

**`create_dtw_plots.py`** - Plotly trajectory cluster plots
- Builds per-patient code counts from `seq_pattern_str` in the DTW features CSV
- For non–polypharmacy cohorts: 3D Plotly scatter (axes = top 3 code counts), KMeans clusters
- For polypharmacy (`non_opioid_ed`): 1D plot (one axis = top code count)
- Writes HTML (and optional PNG) to `10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band}/plots/`
- CLI: `python create_dtw_plots.py --cohort-name {cohort} --age-band {age_band} [--n-clusters 5] [--force]`

**`create_predictive_time_features.py`** - Time-based features
- Extracts temporal features from trajectories
- Generates time-to-event features (for reference, not used in model)

## Output Structure

### Local Outputs

DTW visuals are written under `10_risk_dashboard/visualizations/dtw/outputs/`. Plots directory:

```
10_risk_dashboard/visualizations/dtw/outputs/
├── feature_engineering/          # DTW features CSV (create_dtw_features.py → create_dtw_visuals.py)
│   ├── dtw_features_{cohort}_{age_band}.csv
│   └── dtw_added_features_{cohort}_{age_band}.csv
└── {cohort}/
    └── {age_band}/
        └── plots/                # Visualization files (for dashboard)
            ├── dtw_trajectory_cluster_3d_{cohort}_{age_band}.html   # or 1d for non_opioid_ed
            ├── dtw_trajectory_cluster_3d_{cohort}_{age_band}.png   # optional
            ├── dtw_trajectory_analysis_{cohort}_{age_band}.png
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
- [ ] Trajectory cluster plot(s) generated: `dtw_trajectory_cluster_3d_*.html` or `dtw_trajectory_cluster_1d_*.html` (polypharmacy) in `10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band}/plots/`
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

**Event density (trajectory bin):** The DTW tab includes an **Event density** dropdown (All | Low | Medium | High | Extreme). It filters **Routine vs Utilization (Outcomes)**, **Routine vs Utilization (event counts)**, and **High-Risk vs Low-Risk Trajectories** by trajectory events-per-month bin. Data comes from `chart_data.json`: when the trajectory CSV has `event_density_bin` (from `create_dtw_trajectories.py`), `create_dtw_visuals.py` writes `event_density_bins` and per-bin series (`routine_comparison_by_density`, `routine_comparison_counts_by_density`, `high_risk_trajectories_by_density`). Changing the dropdown re-renders those three charts client-side from the already-loaded data; "All" uses the aggregate series. Bins align with FP-Growth (low/medium/high/extreme by percentiles).

**Code-based filtering (other tabs):** Visualizations on other tabs can be filtered by user-selected codes (e.g. Lambda returns filtered data for selected drugs/ICD/CPT). DTW tab filtering is density-only as above.

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
- For analysis, answering research questions, and clinical interpretation

## Dependencies

- **Python**: `pandas`, `numpy`, `scipy` (for DTW algorithm), `scikit-learn` (for clustering and trajectory cluster plots), `plotly` (for 3D/1D cluster HTML), `matplotlib`, `seaborn`, `boto3`
- **Input Data**: Model events parquet files from Step 4 (model data) or event-filter outputs (Step 1b)

## Notes

1. **DTW Alignment Computed**: DTW distances ARE computed using dtaidistance library (create_dtw_features.py). We do not use DTW for feature engineering (target leakage); we do use DTW with feature importance for analysis and answering research questions.

2. **Event Filtering**: Event-level filtering runs in Step 1b (`1b_apcd_event_filter`). DTW is not used for feature engineering (target leakage); it is used with feature importance for analysis and answering research questions and dashboard display.

3. **Sequence Comparison**: DTW compares patient medication claims-event sequences to identify similar trajectories. This helps understand patient groupings but does not directly predict outcomes.

4. **Clustering**: Patient trajectories are analyzed based on DTW distance to prototype sequences. Visualizations show representative trajectories and statistics.

5. **Temporal Features**: Time-based features are extracted but not used in the final model. They are available for analysis, answering research questions, and reference.

## Related Documentation

- **[Dashboard Visualizations Overview](../../10_risk_dashboard/visualizations/README.md)** - General visualization documentation
- **[Dashboard API Documentation](README_results_dashboard_visualizations.md)** - Complete dashboard visualization guide
- **[Event Filtering (Step 1b)](../../1b_apcd_event_filter/)** - ICD/administrative code filtering before cohort creation
- **[Feature Importance EDA](../../3b_feature_importance_eda/)** - BupaR post-target analysis for feature refinement
