# Visualizations (outputs only)

## Overview

This directory holds **data visualization outputs** (plots, CSVs, JSON, HTML) for the risk dashboard tabs when step 9 is run. It does **not** contain the code that creates these artifacts; creation code lives in **`9_dashboard_visuals/`** (pipeline step 9).

**Outputs are not in the repo.** They are generated on EC2 (or locally when running the dashboard visuals step) and uploaded to the dashboard S3 bucket. The repo only contains READMEs and docs; `*/outputs/` under this tree is in `.gitignore`.

**Data pattern:** Prefer **JSON** for dashboard visuals so Lambda can return inline data and the frontend can render with Plotly. **Exception:** FP-Growth network and PGx Cohort network are built on EC2 and served as HTML only. See `10_risk_dashboard/docs/VISUALIZATION_DATA_PATTERN.md`.

## Directory Structure

```
visualizations/
├── dtw/           # DTW outputs (features, plots, chart_data)
├── fpgrowth/      # FP-Growth outputs (itemsets, rules, plots, network HTML)
└── bupar/         # BupaR outputs (features, plots)
```

## DTW (`dtw/`)

**Purpose**: Patient trajectory visualizations using Dynamic Time Warping. **Dashboard sequence heatmap shows drug slice only.**

**Creation code (step 9):** `9_dashboard_visuals/dtw/` — `create_dtw_trajectories.py` (features CSV, including N3 time-between metrics), then `create_dtw_visuals.py` (plots and chart_data)

**Outputs used by dashboard:**
- `chart_data.json`, `sequence_heatmap.json` (dashboard uses **drug** slice only for the common-sequences heatmap)
- Overview and sample trajectory images when present

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/dtw/{cohort}/{age_band}/`

## FP-Growth (`fpgrowth/`)

**Purpose**: Frequent pattern mining visualizations (**drug names only**; research focus on drug sequences/combinations).

**Creation code (step 9):** `9_dashboard_visuals/fpgrowth/` — e.g. `create_fpgrowth_visuals.py`, `create_plots.py`. Pipeline runs only `drug_name` for both cohorts.

**Outputs used by dashboard:**
- `{cohort}_{age_band}_drug_name_combined_top_itemsets.png` — top drug itemsets
- `*_drug_name_*_itemsets_interactive.html`, `*_combined_rules_network.html` — network and itemsets
- `.../data/drug_name_itemsets.json` — itemsets JSON for client-side Plotly

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/fpgrowth/{cohort}/{age_band}/plots/` and `.../data/`

## BupaR (`bupar/`)

**Purpose**: Process mining visualizations (pathways, trace patterns, activity frequency). **Dashboard shows Drug × Drug process matrix only.**

**Creation code (step 9):** `9_dashboard_visuals/bupar/` — e.g. `create_bupar_visuals.py`, `create_bupar_outputs_*_ed.R`

**Outputs used by dashboard:**
- `*_overall_activity_frequency.png`, `*_activity_frequency_interactive.html` — activity frequency
- `*_trace_explorer_pre_f1120.png` (opioid_ed) / `*_trace_explorer_pre_hcg.png` (non_opioid_ed), `*_trace_explorer_interactive.html`
- `*_process_matrix_drug_drug.png` — **Drug × Drug flows only** (other type-pair PNGs are not used by the dashboard)
- `*_frequency_map.png` (optional)

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/bupar/{cohort}/{age_band}/plots/` (full tree including `lib/`)

## Usage

Run the dashboard visuals step from repo root: `python 9_dashboard_visuals/run_dashboard_visuals.py` (or use `4_dashboard_visuals.ipynb`). See `9_dashboard_visuals/README.md` and `10_risk_dashboard/docs/README_visualization_plan.md` for workflow and outputs. After a sync that fails with WinError 5 on rename, run `9_dashboard_visuals/cleanup_aws_temp_files.py` to remove AWS CLI temp files in `bupar/outputs/.../plots/`.

## Integration with Dashboard

Visualizations are loaded via the Lambda API endpoints:
- `GET /visualizations/dtw`
- `GET /visualizations/fpgrowth`
- `GET /visualizations/bupar`

The frontend displays these visualizations in their respective tabs. The pipeline produces only these artifacts (final production for research-question exploration).
