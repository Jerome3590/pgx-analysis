# Visualizations (outputs only)

## Overview

This directory holds **data visualization outputs** (plots, CSVs, JSON, HTML) for the risk dashboard tabs when step 9 is run. It does **not** contain the code that creates these artifacts; creation code lives in **`9_dashboard_visuals/`** (pipeline step 9).

**Outputs are not in the repo.** They are generated on EC2 (or locally when running the dashboard visuals step) and uploaded to the dashboard S3 bucket. The repo only contains READMEs and docs; `*/outputs/` under this tree is in `.gitignore`.

## Directory Structure

```
visualizations/
├── dtw/           # DTW outputs (features, plots, chart_data)
├── fpgrowth/      # FP-Growth outputs (itemsets, rules, plots, network HTML)
└── bupar/         # BupaR outputs (features, plots)
```

## DTW (`dtw/`)

**Purpose**: Patient trajectory visualizations using Dynamic Time Warping.

**Creation code (step 9):** `9_dashboard_visuals/dtw/` — `create_dtw_trajectories.py` (features CSV, including N3 time-between metrics), then `create_dtw_visuals.py` (plots and chart_data)

**Outputs:**
- `dtw_trajectory_cluster_3d_{cohort}_{age_band}.html` / `.png` (or `1d` for non_opioid_ed) — trajectory cluster Plotly
- `dtw_trajectory_analysis_{cohort}_{age_band}.png` — overview (optional)
- `dtw_sample_trajectories_{cohort}_{age_band}.png` — sample trajectories (optional)
- `chart_data.json` at parent of `plots/`

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/dtw/{cohort}/{age_band}/`

## FP-Growth (`fpgrowth/`)

**Purpose**: Frequent pattern mining visualizations.

**Creation code (step 9):** `9_dashboard_visuals/fpgrowth/` — e.g. `create_fpgrowth_visuals.py`, `create_plots.py`

**Outputs:**
- `{cohort}_{age_band}_{item_type}_combined_top_itemsets.png` — top itemsets
- `*_itemsets_interactive.html` — interactive itemsets
- `*_target_rules_network.png` / `*_target_rules_network.html`, `*_network_interactive.html` (item_type: drug_name, icd_code, cpt_code, medical_code)

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/fpgrowth/{cohort}/{age_band}/plots/`

## BupaR (`bupar/`)

**Purpose**: Process mining visualizations (pathways, trace patterns, activity frequency).

**Creation code (step 9):** `9_dashboard_visuals/bupar/` — e.g. `create_bupar_visuals.py`, `create_bupar_outputs_*_ed.R`

**Outputs (final):**
- `*_overall_activity_frequency.png` — activity frequency (static)
- `*_activity_frequency_interactive.html` — same with year dropdown (requires `plots/lib/`)
- `*_trace_explorer_pre_f1120.png` (opioid_ed) / `*_trace_explorer_pre_hcg.png` (non_opioid_ed) — pre-target trace patterns
- `*_trace_explorer_interactive.html` — pre-target trace explorer with year dropdown (requires `plots/lib/`)
- `*_pre_f1120_activity_frequency.png` (opioid_ed only)
- `*_performance_spectrum.png` (optional, psmineR)
- `*_frequency_map.png` (optional)

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/bupar/{cohort}/{age_band}/plots/` (full tree including `lib/`)

## Usage

Run the dashboard visuals step from repo root: `python 9_dashboard_visuals/run_dashboard_visuals.py` (or use `4_dashboard_visuals.ipynb`). See `9_dashboard_visuals/README.md` and `10_risk_dashboard/docs/README_visualization_plan.md` for workflow and outputs. After a sync that fails with WinError 5 on rename, run `9_dashboard_visuals/cleanup_aws_temp_files.py` to remove AWS CLI temp files in `bupar/outputs/.../plots/`.

## Integration with Dashboard

Visualizations are loaded via the Lambda API endpoints:
- `GET /visualizations/dtw`
- `GET /visualizations/fpgrowth`
- `GET /visualizations/bupar`

The frontend displays these visualizations in their respective tabs.
