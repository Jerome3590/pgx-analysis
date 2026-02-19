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

## Visualization Types (final artifacts)

### 1. Activity frequency
- **Static**: `{cohort}_{age_band_fname}_overall_activity_frequency.png` — bar chart of top activities (drug, ICD, CPT).
- **Interactive**: `{cohort}_{age_band_fname}_activity_frequency_interactive.html` — same with year dropdown (Plotly). Requires `plots/lib/` deployed with the HTML.

### 2. Trace explorer (pre-target only)
- **Static**: `{cohort}_{age_band_fname}_trace_explorer_pre_f1120.png` (opioid_ed) or `_trace_explorer_pre_hcg.png` (non_opioid_ed) — top trace patterns before target.
- **Interactive**: `{cohort}_{age_band_fname}_trace_explorer_interactive.html` — pre-target only, year dropdown. Requires `plots/lib/`.

### 3. Pre-target activity frequency (opioid_ed only)
- **File**: `{cohort}_{age_band_fname}_pre_f1120_activity_frequency.png` — activity frequency before F1120.

### 4. Performance spectrum (optional)
- **File**: `{cohort}_{age_band_fname}_performance_spectrum.png` — aggregated activity trace (psmineR). Skipped if psmineR not installed.

### 5. Frequency map (optional)
- **File**: `{cohort}_{age_band_fname}_frequency_map.png` — process map frequency view. Skipped if processmapR::export_map not available.

**Not produced:** overall trace_explorer.png, process_matrix, gantt, post-F1120 visuals, activity_milestones_gantt.

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
10_risk_dashboard/visualizations/bupar/outputs/
├── {cohort}/
│   └── {age_band_fname}/
│       ├── features/          # Feature files (for reference, not used in model)
│       │   ├── *_train_target_pre_f1120_patient_features_bupar.csv  (opioid_ed) / *_pre_hcg_* (non_opioid_ed)
│       │   └── ...
│       └── plots/             # Visualization files (for dashboard)
│           ├── {cohort}_{age_band_fname}_overall_activity_frequency.png
│           ├── {cohort}_{age_band_fname}_activity_frequency_interactive.html
│           ├── {cohort}_{age_band_fname}_trace_explorer_pre_f1120.png  (opioid_ed) / _trace_explorer_pre_hcg.png (non_opioid_ed)
│           ├── {cohort}_{age_band_fname}_trace_explorer_interactive.html
│           ├── {cohort}_{age_band_fname}_pre_f1120_activity_frequency.png   (opioid_ed only)
│           ├── {cohort}_{age_band_fname}_performance_spectrum.png            (optional)
│           ├── {cohort}_{age_band_fname}_frequency_map.png                   (optional)
│           └── lib/             # Dependencies for interactive HTML (must deploy with HTML)
```

### S3 Outputs

**S3 Location**: `{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/bupar/{cohort}/{age_band}/plots/` (e.g. `jerome-dixon.io` / `vcu/pgx-risk-calculator`).

All PNG, HTML, and `plots/lib/` are uploaded so interactive HTMLs load correctly.

## Usage

### Running BupaR Visualizations

**Using Python orchestrator (recommended, from repo root):**
```bash
python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --cohort {cohort} --age-band {age_band}
```
Or run BupaR only: `python 9_dashboard_visuals/bupar/create_bupar_visuals.py --cohort-name {cohort} --age-band {age_band}`.

**R only (from repo root):**
```bash
Rscript 9_dashboard_visuals/bupar/create_bupar_outputs_opioid_ed.R {age_band}
# or
Rscript 9_dashboard_visuals/bupar/create_bupar_outputs_non_opioid_ed.R {age_band}
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

- **[Dashboard Visualizations Overview](../../10_risk_dashboard/visualizations/README.md)** - General visualization documentation
- **[Dashboard API Documentation](README_results_dashboard_visualizations.md)** - Complete dashboard visualization guide
- **[Feature Importance EDA](../../3b_feature_importance_eda/)** - BupaR post-target analysis
