# Visualizations

## Overview

Scripts to generate visualization files (images, HTML) for the dashboard visualization tabs.

## Directory Structure

```
visualizations/
├── dtw/           # DTW trajectory visualizations
├── fpgrowth/      # FP-Growth pattern visualizations
└── bupar/         # BupaR process mining visualizations
```

## DTW Visualizations (`dtw/`)

**Purpose**: Generate patient trajectory visualizations using Dynamic Time Warping.

**Scripts:**
- `create_dtw_features.py` - Extract DTW features and trajectories
- `create_dtw_visualizations.py` - Generate visualization images

**Outputs:**
- `dtw_trajectory_analysis_{cohort}_{age_band}.png` - Overview visualization
- `dtw_sample_trajectories_{cohort}_{age_band}.png` - Sample trajectories

**S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`

## FP-Growth Visualizations (`fpgrowth/`)

**Purpose**: Generate frequent pattern mining visualizations.

**Scripts:**
- `run_analysis.py` - Run FP-Growth analysis
- `create_plots.py` - Generate visualization images and HTML networks

**Outputs:**
- `*_top20_itemsets.png` - Top itemsets bar chart
- `*_itemset_support.png` - Support distribution
- `*_network.html` - Interactive co-occurrence network

**S3 Location**: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

## BupaR Visualizations (`bupar/`)

**Purpose**: Generate process mining visualizations.

**Scripts:**
- `run_analysis.py` - Run BupaR analysis
- `create_bupar_outputs_opioid_ed.R` - Generate visualizations for opioid_ed
- `create_bupar_outputs_non_opioid_ed.R` - Generate visualizations for non_opioid_ed

**Outputs:**
- `*_overall_activity_frequency.png` - Activity frequency chart
- `*_gantt.png` - Gantt chart
- `*_activity_sequence_top.png` - Top activity sequences

**S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`

## Usage

Each visualization type has its own README with specific usage instructions:

### Dashboard Visualization READMEs
- **[`bupar/README_dashboard.md`](bupar/README_dashboard.md)** - BupaR process mining dashboard visualizations
- **[`fpgrowth/README_dashboard.md`](fpgrowth/README_dashboard.md)** - FP-Growth pattern mining dashboard visualizations
- **[`dtw/README_dashboard.md`](dtw/README_dashboard.md)** - DTW trajectory dashboard visualizations

### Legacy Documentation (for reference)
- `bupar/README.md` - BupaR feature engineering documentation (legacy)
- `fpgrowth/README.md` - FP-Growth analysis documentation (legacy)
- `fpgrowth/README_visualization_only.md` - FP-Growth visualization-only rationale

## Integration with Dashboard

Visualizations are loaded via the Lambda API endpoints:
- `GET /visualizations/dtw`
- `GET /visualizations/fpgrowth`
- `GET /visualizations/bupar`

The frontend displays these visualizations in their respective tabs.
