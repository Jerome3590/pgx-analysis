# DTW dashboard visualizations – status

**Last updated:** 2025-02  
**Pipeline step:** 9 (`9_dashboard_visuals`)  
**Creation code:** `9_dashboard_visuals/dtw/`  
**Outputs only:** `10_risk_dashboard/visualizations/dtw/`

---

## Summary

| Item | Status |
|------|--------|
| **Pipeline integration** | Step 9; runs from `4_dashboard_visuals.ipynb` or `python 9_dashboard_visuals/run_dashboard_visuals.py` after model data + SHAP/FFA exist |
| **Cohort / target focus** | Per-cohort, per–age-band; SHAP/FFA important codes; target-aligned trajectories (events before first target event for cases) |
| **Idempotency** | Local output existence + `pipeline_checkpoints/9_dashboard_visuals/{cohort}/{age_band}/checkpoint.json`; `--force` to re-run |
| **S3 / dashboard** | Features CSV → pgxdatalake + optional 6_dtw_checkpoint mirror; plots + chart_data → dashboard bucket |
| **Status script** | `python 9_dashboard_visuals/dtw/check_dtw_s3_status.py` (optionally `--logs`, `--outputs`, `--profile NAME`) |

---

## Flow (per cohort / age_band)

1. **create_dtw_features.py**  
   - **Inputs:** `4_model_data` (or 3b) `model_events`; SHAP/FFA allowed codes from `get_shap_ffa_allowed_codes_combined(cohort_name, age_band, top_n=500)`.  
   - **Logic:** Target-event pathway: anchor = `first_opioid_ed_date` (opioid_ed) or `first_ed_non_opioid_date` (non_opioid_ed). For target patients, only events *before* anchor; optional `max_lookback_months` (default 24). Trajectories restricted to SHAP/FFA codes when available; fallback to all events if filter yields 0 rows.  
   - **Outputs:** `10_risk_dashboard/visualizations/dtw/outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv` (and research outputs under `outputs/research/` if used).

2. **create_dtw_visuals.py**  
   - **Inputs:** DTW features CSV from step 1.  
   - **Actions:** Copies to `dtw_added_features_{cohort}_{age_band}.csv` in same feature_engineering dir; uploads CSV to `s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort}/{age_band}/`; calls `save_step_checkpoint("9_dashboard_visuals", cohort_name, age_band, ...)`; optionally mirrors CSV to `s3://pgx-repository/6_dtw_checkpoint/`; runs **create_dtw_plots.py** (3D/1D cluster plots); uploads plots to dashboard bucket `{S3_DASHBOARD_PREFIX}/dtw/{cohort}/{age_band}/plots/`; builds chart data (routine vs no routine, high-risk trajectories) and uploads to dashboard S3.  
   - **Outputs:** Local plots under `10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band}/plots/`; dashboard bucket: plots + chart_data.

---

## By-cohort behavior

- **SHAP/FFA:** `get_shap_ffa_allowed_codes_combined(cohort_name, age_band, ...)` — each (cohort, age_band) uses its own important-codes set.  
- **Target date:** opioid_ed → `first_opioid_ed_date`; non_opioid_ed → `first_ed_non_opioid_date`.  
- **Model data:** `resolve_model_events_path(project_root, cohort_name, age_band)` — per cohort/age_band.  
- **Notebook / runner:** Loops over `(cohort_name, age_band)`; each combination runs create_dtw_features then create_dtw_visuals once.

---

## Output locations

| Location | Contents |
|----------|----------|
| `10_risk_dashboard/visualizations/dtw/outputs/feature_engineering/` | `dtw_features_*.csv`, `dtw_added_features_*.csv` |
| `10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band}/plots/` | PNG/HTML from create_dtw_plots |
| `s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort}/{age_band}/` | DTW features CSV |
| `s3://pgx-repository/pipeline_checkpoints/9_dashboard_visuals/{cohort}/{age_band}/checkpoint.json` | Step 9 checkpoint (idempotency) |
| `s3://pgx-repository/6_dtw_checkpoint/{cohort}/{age_band}/` | Optional CSV mirror |
| Dashboard bucket `{S3_DASHBOARD_PREFIX}/dtw/{cohort}/{age_band}/plots/` | Plot files for frontend |
| Dashboard bucket (chart_data) | Prebuilt routine vs no routine / trajectory risk chart data |

---

## Scripts in 9_dashboard_visuals/dtw

| Script | Purpose |
|--------|---------|
| **create_dtw_features.py** | Build trajectories (SHAP/FFA, target-aligned), DTW distances, feature CSV |
| **create_dtw_visuals.py** | Publish CSV, checkpoint, plots, chart_data to S3/dashboard |
| **create_dtw_plots.py** | 3D/1D trajectory cluster plots (Plotly) |
| **check_dtw_s3_status.py** | Report pipeline_checkpoints/9_dashboard_visuals, 6_dtw_checkpoint, pgxdatalake DTW outputs, optional logs |
| **barycenter_reporting.py** | Barycenter/prototype reporting (optional) |
| **create_predictive_time_features.py** | Predictive time-window features (optional) |
| **extract_extreme_density_cohort.py** | Extreme-density cohort extraction (optional) |
| **summarize_extreme_density_cohort.py** | Extreme-density summary (optional) |

---

## Prerequisites

- **4_model_data** (or 3b) with `model_events` and target date column (`first_opioid_ed_date` / `first_ed_non_opioid_date`) for leakage-safe cutoffs.  
- **SHAP/FFA** outputs (Steps 7/8) for SHAP/FFA filtering; if missing or empty, DTW falls back to all events.  
- **dtaidistance** (DTW library); **duckdb**; **py_helpers** (model_data_paths, shap_ffa_fpgrowth_utils, checkpoint_utils).  
- **1b_apcd_event_filter/administrative_codes_lookup.json** for `admin_icd_event_count` (routine vs no routine).

---

## Checking status

- **Local:** Output CSV and plots under `10_risk_dashboard/visualizations/dtw/outputs/`.  
- **S3:**  
  `python 9_dashboard_visuals/dtw/check_dtw_s3_status.py`  
  Use `--outputs` for pgxdatalake DTW objects, `--logs` for DTW-related logs.  
- **Idempotency:** Checkpoints in `pipeline_checkpoints/9_dashboard_visuals/`; create_dtw_visuals also skips when local output CSV exists unless `--force`.
