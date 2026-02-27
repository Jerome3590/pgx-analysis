# DTW visualizations – S3 mapping

This doc maps **Lambda expectations**, **pipeline uploads**, and **Step 6 promote** for the DTW Trajectories tab so you can verify or fix what’s in S3.

## Expected S3 layout (final, what Lambda uses)

**Prefix:** `{S3_DASHBOARD_PREFIX}/visualizations/dtw/`  
Example: `vcu/pgx-risk-calculator/visualizations/dtw/`

For each `{cohort}/{age_band}` (age_band with **hyphen**, e.g. `13-24`):

| S3 key (under prefix above) | Required | Description |
|-----------------------------|----------|-------------|
| `{cohort}/{age_band}/chart_data.json` | ✓ | Routine vs no routine, high-risk trajectories, pathway patterns (JSON inline in API) |
| `{cohort}/{age_band}/sequence_heatmap.json` | ✓ | Common-sequences heatmap (JSON inline in API) |
| `{cohort}/{age_band}/plots/` | optional | Trajectory overview PNG/HTML; `trajectory_overview_plot.json` preferred for Plotly |

**Lambda** (`handle_visualizations_dtw`): reads `chart_data.json` and `sequence_heatmap.json` from these keys and returns them inline; optionally returns URLs for `plots/` assets.

## DTW tab panels ↔ S3 files (index.html)

The DTW tab in `10_risk_dashboard/frontend/index.html` has two sub-tabs and several panels. Each panel needs data from the API; the API gets that from S3. If a file is missing, the panel shows a placeholder message.

| Panel | API payload key (from Lambda) | S3 file / source |
|-------|--------------------------------|------------------|
| **Overview & Trajectories** | | |
| Trajectory overview | `trajectory_overview_plot` (preferred) or `overview_image` | `plots/trajectory_overview_plot.json` or `plots/dtw_trajectory_analysis_{cohort}_{age_band_fname}.png` |
| Sample trajectories | `trajectory_overview_plot` (reused) or `sample_trajectories_image` | same as above or `plots/dtw_sample_trajectories_{cohort}_{age_band_fname}.png` |
| Trajectory metrics (bar) | `metrics` (Lambda derives from chart_data) | from `chart_data.json` |
| High-risk vs low-risk | `chart_data.high_risk_trajectories` (or `_by_density`) | `chart_data.json` |
| Times between sequences (N3) | `chart_data.times_between_sequences` | `chart_data.json` |
| Time to target | `chart_data.time_to_target_sequences` | `chart_data.json` |
| Target pathway patterns | `chart_data.target_pathway_patterns` | `chart_data.json` |
| **Sequence heatmap (drugs)** | `sequence_heatmap` (e.g. `sequence_heatmap.drug`) | `sequence_heatmap.json` |
| **Routine vs No Routine** | | |
| Routine vs no routine (outcomes) | `chart_data.routine_comparison` (or `_by_density`) | `chart_data.json` |
| Routine counts (medical/prescription events) | `chart_data.routine_comparison_counts` (or `_by_density`) | `chart_data.json` |

**Summary – required vs optional**

- **Required for the tab to show real content (no placeholders):**
  - `{cohort}/{age_band}/chart_data.json` – provides routine_comparison, high_risk_trajectories, times_between_sequences, target_pathway_patterns, etc.
  - `{cohort}/{age_band}/sequence_heatmap.json` – provides the sequence heatmap (drug slice).
- **Optional (overview/sample panels):**
  - `{cohort}/{age_band}/plots/trajectory_overview_plot.json` – Plotly overview (preferred).
  - `{cohort}/{age_band}/plots/dtw_trajectory_analysis_*.png`, `dtw_sample_trajectories_*.png` – image fallbacks.
  - `{cohort}/{age_band}/plots/dtw_trajectory_cluster_interactive_*.html` – interactive HTML (optional).

If you only have `chart_data.json` for one combo and `sequence_heatmap.json` for another, **both** panels that need chart_data will show placeholders for the combo that’s missing it, and the heatmap will show a placeholder for the combo missing sequence_heatmap. You need **both** files for **each** cohort/age_band you want to support.

## Pipeline mapping

| Step | Where | S3 destination |
|------|--------|----------------|
| **Notebook 4** (with `S3_VISUALIZATIONS_BUILDS=1`) | `9_dashboard_visuals/dtw/create_dtw_visuals.py` | `visualizations/dtw/builds/{cohort}/{age_band}/chart_data.json`, `sequence_heatmap.json`, `.../plots/*` |
| **Notebook 5 Step 6** | Promote | `aws s3 sync .../visualizations/dtw/builds/ .../visualizations/dtw/` → final layout above |

- **Local outputs** (before upload): `10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band_fname}/`  
  - `chart_data.json`, `sequence_heatmap.json`, `plots/*.png`, `plots/*.html`, `plots/*.json`  
- **Upload**: When `S3_VISUALIZATIONS_BUILDS=1`, uploads go to `.../dtw/builds/...`. When unset, uploads go directly to `.../dtw/{cohort}/{age_band}/` (final).

## Current S3 vs expected (checklist)

If you only see:

- `visualizations/dtw/opioid_ed/13-24/chart_data.json`
- `visualizations/dtw/non_opioid_ed/13-24/sequence_heatmap.json`

then:

| Cohort       | Age band | chart_data.json | sequence_heatmap.json | plots/ |
|-------------|----------|-----------------|------------------------|--------|
| opioid_ed   | 13-24    | ✓               | ✗ missing              | ?      |
| non_opioid_ed | 13-24  | ✗ missing       | ✓                      | ?      |
| (other cohorts/age_bands) | — | ✗ | ✗ | ✗ |

So:

1. **Per cohort/age_band** you should have **both** `chart_data.json` and `sequence_heatmap.json` for the DTW tab to work fully.
2. **Coverage**: Only 13-24 is present; other age bands (e.g. 25-44, 45-54, …) need a pipeline run if you want them in the dashboard.

## How to fix

1. **Re-run DTW for the combos you care about**  
   In **notebook 4**, ensure the DTW step runs for each `(cohort, age_band)` you want (e.g. opioid_ed 13-24, non_opioid_ed 13-24, and any other bands).  
   - Script: `9_dashboard_visuals/dtw/create_dtw_visuals.py` with `--cohort-name` and `--age-band`.  
   - With `S3_VISUALIZATIONS_BUILDS=1`, files go to `.../dtw/builds/...`.

2. **Promote builds → final**  
   In **notebook 5**, run **Step 6**. It runs:
   - `aws s3 sync s3://{bucket}/{prefix}/visualizations/dtw/builds/ s3://{bucket}/{prefix}/visualizations/dtw/`
   so that the final paths above are populated.

3. **If you upload without builds**  
   If notebook 4 is run **without** `S3_VISUALIZATIONS_BUILDS=1`, uploads go straight to `.../visualizations/dtw/{cohort}/{age_band}/`. In that case Step 6 promote does nothing for DTW; what you have in S3 is exactly what the pipeline uploaded. Then fix is to re-run the DTW step for each cohort/age_band so both `chart_data.json` and `sequence_heatmap.json` (and optionally plots) are uploaded.

## Quick verification

- **Lambda**: For a given `cohort` and `age_band`, GET `/visualizations/dtw?cohort=...&age_band=...` returns 200 and includes `chart_data` and `sequence_heatmap` when both S3 objects exist.
- **S3**: List final prefix:
  - `aws s3 ls s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/dtw/ --recursive`
