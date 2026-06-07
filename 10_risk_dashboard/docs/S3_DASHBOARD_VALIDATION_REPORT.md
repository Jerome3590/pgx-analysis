# S3 Dashboard Folder Validation Report

**Bucket:** `jerome-dixon.io`  
**Prefix:** `vcu/pgx-risk-calculator`  
**Reference:** [README_dashboard_visual_artifact_paths.md](README_dashboard_visual_artifact_paths.md)  
**Date:** 2026-02-27

---

## 1. S3 contents (recursive list under prefix)

| S3 key pattern | Present | Notes |
|----------------|---------|--------|
| **Top level** | | |
| `index.html`, `README.md`, `dashboard_index_template.html` | ✓ | Frontend entry points |
| **metadata/** | | |
| `metadata/model_performance_metrics.json` | ✓ | Documentation tab |
| `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json` | ✓ | Drugs/ICD/CPT dropdowns (same-origin) |
| **visualizations/** | | |
| `visualizations/dashboard_visual_objects.json` | ✓ | Manifest for checklist |
| **visualizations/scenario/** | | |
| `visualizations/scenario/{cohort}/{age_band}/scenario_data.json` | ✓ | Both cohorts, all 8 age bands (hyphen: 0-12 … 85-114) |
| **visualizations/cohort_pgx/** | | |
| `visualizations/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` | ✓ | Both cohorts, all age bands; supporting files (CSV, JSON) also present |
| **visualizations/feature_importance/** | | |
| `visualizations/feature_importance/opioid_ed/aggregated_fi_heatmap.png` | ✓ | |
| `visualizations/feature_importance/opioid_ed/aggregated_fi_heatmap.json` | ✓ | |
| `visualizations/feature_importance/non_opioid_ed/aggregated_fi_heatmap.png` | ✓ | |
| `visualizations/feature_importance/non_opioid_ed/aggregated_fi_heatmap.json` | ✓ | |
| `visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png` | ✓ | |
| `visualizations/feature_importance/combined/aggregated_fi_heatmap.json` | ✓ | |
| **visualizations/bupar/** | | |
| `visualizations/bupar/{cohort}/{age_band}/plots/*` | ✗ **MISSING** | No `bupar/` under `visualizations/` in S3 |
| **visualizations/dtw/** | | |
| `visualizations/dtw/{cohort}/{age_band}/chart_data.json`, `sequence_heatmap.json`, `plots/*` | ✗ **MISSING** | No `dtw/` under `visualizations/` in S3 |
| **visualizations/fpgrowth/** | | |
| `visualizations/fpgrowth/{cohort}/{age_band}/plots/*`, `data/*` | ✗ **MISSING** | No `fpgrowth/` under `visualizations/` in S3 |

---

## 2. Mapping document vs S3

Per **README_dashboard_visual_artifact_paths.md**:

- **Scenario:** EC2 → `upload_scenario_outputs_to_s3.py` → S3 `visualizations/scenario/{cohort}/{age_band}/scenario_data.json` (hyphen). **Match:** S3 has these keys.
- **Cohort PGx:** EC2 → `sync_cohort_pgx_to_s3.py` → S3 `visualizations/cohort_pgx/networks/{cohort}/{age_band}/*`. **Match:** S3 has these keys.
- **Feature importance:** Notebook 5 Step 6 → S3 `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png|.json` and `.../combined/...`. **Match:** S3 has these keys.
- **BupaR:** Step 6 sync from local `.../bupar/{cohort}/{age_band}/plots/` → S3 `visualizations/bupar/{cohort}/{age_band}/plots/*`. **Mismatch:** No `visualizations/bupar/` in S3.
- **DTW:** Step 6 sync from local `.../dtw/{cohort}/{age_band}/` (skip `feature_engineering/`) → S3 `visualizations/dtw/{cohort}/{age_band}/`. **Mismatch:** No `visualizations/dtw/` in S3.
- **FP-Growth:** Step 6 sync from local `.../fpgrowth/{cohort}/{age_band}/` → S3 `visualizations/fpgrowth/{cohort}/{age_band}/plots/`, `data/`. **Mismatch:** No `visualizations/fpgrowth/` in S3.

**Age band format:** Mapping requires hyphen in S3 (e.g. `25-44`). Scenario and cohort_pgx keys in S3 use hyphen. BupaR/DTW/FP-Growth are absent so not applicable.

---

## 3. Frontend calls vs S3 / API

| Tab / feature | Frontend behavior | Expected data source | Validation |
|---------------|-------------------|----------------------|------------|
| **Metadata (Drugs, ICD, CPT)** | `staticJsonPath('metadata/${cohort}.json')` then fallback `GET /metadata?cohort=` | Same-origin `metadata/*.json` or API | ✓ S3 has `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json` |
| **Documentation (metrics)** | `fetch('metadata/model_performance_metrics.json')` then fallback `GET /metrics` | Same-origin `metadata/model_performance_metrics.json` | ✓ S3 has it |
| **Feature importance** | `staticJsonPath('visualizations/feature_importance/...')` then fallback `GET /visualizations/feature_importance?cohort=` | Same-origin under prefix or API (Lambda returns S3 URLs) | ✓ S3 has `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.json` and combined |
| **Scenario analysis** | `GET /visualizations/scenario?cohort=&age_band=` then `fetch(data.causal_data_url)` | Lambda returns `causal_data_url` → S3 `visualizations/scenario/{cohort}/{age_band}/scenario_data.json` | ✓ S3 has these objects; Lambda uses hyphen in key |
| **DTW** | `GET /visualizations/dtw?cohort=&age_band=` then `fetch(chart_data_url)`, `fetch(sequence_heatmap_url)` | Lambda returns URLs to S3 `visualizations/dtw/{cohort}/{age_band}/chart_data.json`, `sequence_heatmap.json` | ✗ S3 has no `visualizations/dtw/` → API will return empty or 404-style behavior for those URLs |
| **FP-Growth** | `GET /visualizations/fpgrowth?cohort=&age_band=&item_type=drug_name`; network via `GET /visualizations/fpgrowth/network_html?cohort=&age_band=` | Lambda returns URLs to S3 `visualizations/fpgrowth/{cohort}/{age_band}/plots/`, `data/` | ✗ S3 has no `visualizations/fpgrowth/` → same as DTW |
| **BupaR** | `GET /visualizations/bupar?cohort=&age_band=` and `GET /visualizations/bupar/activity_frequency?cohort=&age_band=` | Lambda returns URLs to S3 `visualizations/bupar/{cohort}/{age_band}/plots/*` | ✗ S3 has no `visualizations/bupar/` → same as DTW |
| **PGx Cohort** | `GET /visualizations/cohort_pgx?cohort=&age_band=` then load `data.network_topology_url` | Lambda returns `network_topology_url` → S3 `visualizations/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` | ✓ S3 has these objects |
| **Manifest** | `fetch(staticJsonPath('visualizations/dashboard_visual_objects.json'))` | Same-origin `visualizations/dashboard_visual_objects.json` | ✓ S3 has it |

---

## 4. Summary

- **Aligned with mapping and frontend:**  
  Top-level (index, README, template), `metadata/`, `visualizations/scenario/`, `visualizations/cohort_pgx/`, `visualizations/feature_importance/`, and `visualizations/dashboard_visual_objects.json` are present and match the mapping doc and how the frontend (and Lambda) use them.

- **Missing in S3 (mapping and API expect them):**  
  - `visualizations/bupar/{cohort}/{age_band}/plots/*`  
  - `visualizations/dtw/{cohort}/{age_band}/` (e.g. `chart_data.json`, `sequence_heatmap.json`, `plots/`)  
  - `visualizations/fpgrowth/{cohort}/{age_band}/plots/` and `data/`  

  So **BupaR**, **DTW**, and **FP-Growth** tabs will not have assets in S3 until Step 6 (or equivalent) runs after the corresponding notebook 4 outputs exist under `10_risk_dashboard/visualizations/bupar/`, `.../dtw/`, and `.../fpgrowth/` (no `outputs` subdir; causal-style paths), and syncs them to the dashboard bucket under `visualizations/bupar/`, `visualizations/dtw/`, and `visualizations/fpgrowth/`.

- **Path-style URL:**  
  Lambda builds asset URLs with `_dashboard_s3_url(key)` as `https://s3.{region}.amazonaws.com/{bucket}/{key}`. Keys under the prefix (e.g. `vcu/pgx-risk-calculator/visualizations/...`) match this and the mapping doc.

---

## 5. Notebook 4: where visuals are saved (local)

Notebook 4 sets `SKIP_DASHBOARD_S3_UPLOAD=1`, so it **writes local only**; notebook 5 Step 6 is the single place that syncs to S3.

| Viz | Local save path (repo root) | Script / step |
|-----|-----------------------------|----------------|
| **BupaR** | `10_risk_dashboard/visualizations/bupar/{cohort}/{age_band_fname}/plots/` | `9_dashboard_visuals/bupar/create_bupar_visuals.py` (notebook 4 BupaR cell) |
| **DTW** | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/` (chart_data.json, sequence_heatmap.json, plots/) | `create_dtw_trajectories.py` → `create_dtw_features.py` → `create_dtw_visuals.py` (notebook 4 DTW cell) |
| **FP-Growth** | `10_risk_dashboard/visualizations/fpgrowth/{cohort}/{age_band_fname}/plots/`, `.../data/` | `9_dashboard_visuals/fpgrowth/create_plots.py` via create_fpgrowth_visuals (notebook 4 FP-Growth cell) |
| **Cohort PGx** | `10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/` | `build_network_topology.py` (notebook 4 Cohort PGx) |
| **Scenario** | `10_risk_dashboard/visualizations/scenario/{cohort}/{age_band_fname}/dashboard_data.json` | `combine_shap_ffa_results.py`; upload via `upload_scenario_outputs_to_s3.py` (or Step 6) |
| **Feature importance** | `3a_feature_importance/outputs/{cohort}/plots/{cohort}_aggregated_fi_heatmap.png`, `3a_feature_importance/outputs/plots/combined_cohorts_feature_importance_heatmap.png` | Notebook 4 FI heatmaps cell; Step 6 uploads from these paths |

**Age band in paths:** Local/EC2 uses **underscore** (e.g. `25_44`); S3 uses **hyphen** (e.g. `25-44`). Step 6 sync maps `age_band_fname` → `age_band` when building S3 keys.

---

## 6. S3 “builds” folder check

The docs describe an optional flow where notebook 4 uploads to **builds** (`visualizations/{bupar,dtw,fpgrowth,cohort_pgx}/builds/`) and Step 6 promotes builds → final. With `SKIP_DASHBOARD_S3_UPLOAD=1`, notebook 4 does **not** upload to S3 at all.

**S3 listing (2026-02-27):**

- `aws s3 ls s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/` shows only: `causal/`, `cohort_pgx/`, `feature_importance/`, `dashboard_visual_objects.json`.
- No `visualizations/bupar/`, `visualizations/dtw/`, or `visualizations/fpgrowth/` (neither final nor builds).
- No keys containing `builds` under the dashboard prefix.

**Conclusion:** The **builds** folder is **not present** on S3 for BupaR, DTW, or FP-Growth. Those tabs are populated only when (1) notebook 4 has been run (so local outputs exist under `10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth}/` with no `outputs` subdir) and (2) notebook 5 Step 6 has been run to sync those local dirs to **final** `visualizations/{bupar,dtw,fpgrowth}/` (Step 6 syncs local → final directly; it does not use a builds path when using the current idempotent setup).

---

## 7. Recommended next steps

1. **BupaR / DTW / FP-Growth:** On EC2 (or wherever notebook 4 runs), ensure BupaR, DTW, and FP-Growth pipelines have produced outputs under `10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth}/{cohort}/{age_band_fname}/`. Then run **Notebook 5 Step 6** again so the “Single source of truth” sync block runs `aws s3 sync` for those three trees to `s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/{bupar,dtw,fpgrowth}/`.
2. **Re-run validation:** After sync, run `aws s3 ls s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/ --recursive` and confirm `bupar/`, `dtw/`, and `fpgrowth/` appear and match the mapping (hyphen age bands, expected filenames).
3. **Optional:** Add a small script or notebook cell that (1) lists the dashboard prefix, (2) checks for required keys from the mapping doc, and (3) prints this validation table for CI or pre-release checks.
