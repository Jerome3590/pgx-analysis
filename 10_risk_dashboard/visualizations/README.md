# Visualizations (outputs only)

## Overview

This directory holds **data visualization outputs only** (plots, CSVs, JSON, HTML) for the risk dashboard tabs. It does **not** contain the code that creates these artifacts. Creation code lives in **`9_dashboard_visuals/`** (pipeline step 9), following the same pattern as `6_final_model`, `7_shap_analysis`, and `8_ffa_analysis`.

## Directory Structure

```
visualizations/
├── dtw/           # DTW outputs (features, plots, chart_data)
├── fpgrowth/      # FP-Growth outputs (itemsets, rules, plots, network HTML)
└── bupar/         # BupaR outputs (features, plots)
```

## DTW (`dtw/`)

**Purpose**: Patient trajectory visualizations using Dynamic Time Warping.

**Creation code (step 9):** `9_dashboard_visuals/dtw/` — e.g. `create_dtw_features.py`, `create_dtw_visuals.py`

**Outputs:**
- `dtw_trajectory_analysis_{cohort}_{age_band}.png` - Overview visualization
- `dtw_sample_trajectories_{cohort}_{age_band}.png` - Sample trajectories

**S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`

## FP-Growth (`fpgrowth/`)

**Purpose**: Frequent pattern mining visualizations.

**Creation code (step 9):** `9_dashboard_visuals/fpgrowth/` — e.g. `create_fpgrowth_visuals.py`, `create_plots.py`

**Outputs:**
- `*_top20_itemsets.png` - Top itemsets bar chart
- `*_itemset_support.png` - Support distribution
- `*_network.html` - Interactive co-occurrence network

**S3 Location**: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

## BupaR (`bupar/`)

**Purpose**: Process mining visualizations.

**Creation code (step 9):** `9_dashboard_visuals/bupar/` — e.g. `create_bupar_visuals.py`, `create_bupar_outputs_*_ed.R`

**Outputs:**
- `*_overall_activity_frequency.png` - Activity frequency chart
- `*_gantt.png` - Gantt chart
- `*_activity_sequence_top.png` - Top activity sequences

**S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`

## Usage

Run the dashboard visuals step from repo root: `python 9_dashboard_visuals/run_dashboard_visuals.py` (or use `4_dashboard_visuals.ipynb`). See `9_dashboard_visuals/README.md` and the READMEs under `9_dashboard_visuals/{bupar,dtw,fpgrowth}/` for creation and usage.

## Integration with Dashboard

Visualizations are loaded via the Lambda API endpoints:
- `GET /visualizations/dtw`
- `GET /visualizations/fpgrowth`
- `GET /visualizations/bupar`

The frontend displays these visualizations in their respective tabs.
