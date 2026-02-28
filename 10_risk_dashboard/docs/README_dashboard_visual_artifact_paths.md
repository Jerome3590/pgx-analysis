# Dashboard tab & visual → data artifact → EC2 path → S3 path

This README documents the mapping from **dashboard tab** and **visual (heading)** to **data artifact** (file or API payload), **EC2 file path** (where the pipeline writes it), and **S3 object path** (path-style, where the dashboard loads it from).

**Path-style S3 only.** All dashboard HTML and images use path-style URLs:  
`https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{object_key}`  
Do not use virtual-hosted style. Bucket = `S3_DASHBOARD_BUCKET` (e.g. `jerome-dixon.io`), prefix = `S3_DASHBOARD_PREFIX` (e.g. `vcu/pgx-risk-calculator`).

**EC2 paths** are relative to repo root on EC2 (e.g. `/home/pgx3874/pgx-analysis`). **Age bands:** EC2/file paths use underscore (e.g. `25_44`); **S3 paths use hyphen** (e.g. `25-44`). **S3 object key** = prefix + path below (no leading slash).

**`10_risk_dashboard/outputs/`** contains only **models**, **cpic**, and **metadata** (for Lambda/container and same-origin deploy). All visualization artifacts (causal, DTW, FP-Growth, BupaR, feature importance, cohort_pgx) live under **`10_risk_dashboard/visualizations/`** or their step folders (e.g. `3a_feature_importance/` for FI source).

**Related:** [RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md), [DASHBOARD_TABS.md](DASHBOARD_TABS.md).

---

## Canonical EC2 write locations (verify artifacts are here)

All paths below are relative to **repo root** (`project_root`). If outputs are not in these locations, ensure every script is invoked with the same `--project-root` (or equivalent) as the notebook’s `REPO_ROOT`. The path check script reads from these same paths: `10_risk_dashboard/data_preparation/check_dashboard_artifact_paths.py`.

| Script / step | Writes to (under repo root) |
|---------------|-----------------------------|
| **create_dtw_trajectories.py** | `10_risk_dashboard/visualizations/dtw/feature_engineering/dtw_features_{cohort}_{age_band_fname}.csv` |
| **create_dtw_features.py** | Same dir: `dtw_features_*_density_{bin}.csv`, `common_sequences_*_density_{bin}.json` (or single `dtw_features_*.csv` + `common_sequences_*.json` when no density bins) |
| **create_dtw_visuals.py** | Reads from `.../dtw/feature_engineering/`; writes `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/chart_data.json`, `sequence_heatmap.json`, `plots/*` |
| **create_bupar_visuals.py** | `10_risk_dashboard/visualizations/bupar/{cohort}/{age_band_fname}/plots/*` |
| **create_plots.py** (FP-Growth) | `10_risk_dashboard/visualizations/fpgrowth/{cohort}/{age_band_fname}/plots/`, `.../data/` |
| **upload_causal_outputs_to_s3.py** (source) | Reads from `10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json` (written by combine_shap_ffa_results / causal pipeline) |
| **build_network_topology.py** (Cohort PGx) | `10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/` |

**DTW:** Notebook 4 must pass `--project-root str(REPO_ROOT)` to all three DTW steps (trajectories, features, visuals) so they use the same root. The check script expects `chart_data.json` and `sequence_heatmap.json` under `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/`.

---

## Path alignment: EC2 → upload/sync → S3 → Lambda

All visualization artifacts use the **same S3 location** and are **mapped consistently** from EC2 to Lambda:

| Visualization | EC2 location (underscore age band) | Upload/sync script | S3 key under prefix (hyphen age band) | Lambda (same key) |
|---------------|------------------------------------|--------------------|----------------------------------------|-------------------|
| **Causal** | `10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json` | `upload_causal_outputs_to_s3.py` | `visualizations/causal/{cohort}/{age_band}/causal_data.json` | ✓ `handle_visualizations_causal` |
| **DTW** | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/` (chart_data.json, sequence_heatmap.json, plots/) | `create_dtw_visuals.py` (upload helpers) | `visualizations/dtw/{cohort}/{age_band}/chart_data.json`, `sequence_heatmap.json`, `visualizations/dtw/{cohort}/{age_band}/plots/*` | ✓ `handle_visualizations_dtw` |
| **FP-Growth** | `10_risk_dashboard/visualizations/fpgrowth/{cohort}/{age_band_fname}/plots/` (and data/) | `create_plots.py` → `create_all_fpgrowth_plots` (s3_prefix = prefix/visualizations/fpgrowth) | `visualizations/fpgrowth/{cohort}/{age_band}/plots/*`, `visualizations/fpgrowth/{cohort}/{age_band}/data/*` | ✓ `handle_visualizations_fpgrowth` |
| **BupaR** | `10_risk_dashboard/visualizations/bupar/{cohort}/{age_band_fname}/plots/` | `create_bupar_visuals.py` → `upload_bupar_plots_to_dashboard_s3` | `visualizations/bupar/{cohort}/{age_band}/plots/*` | ✓ `handle_visualizations_bupar` |
| **Feature importance** | `3a_feature_importance/{cohort}/plots/` (and `plots/`, `combined/`) | `pgx_dashboard_visuals.py` (when DEPLOY_FRONTEND=1) or notebook 5 deploy | `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png` or `.json`, `visualizations/feature_importance/combined/aggregated_fi_heatmap.json`, `visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png` | ✓ `handle_visualizations_feature_importance` |
| **Cohort PGx** | `10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/` | `sync_cohort_pgx_to_s3.py` or `build_network_topology.py` upload | `visualizations/cohort_pgx/networks/{cohort}/{age_band}/*` | ✓ `handle_visualizations_cohort_pgx` |

**S3 layout (bucket prefix = e.g. `vcu/pgx-risk-calculator/`):**

- **`visualizations/`** — All dashboard visualization data (causal, dtw, fpgrowth, bupar, feature_importance, cohort_pgx). Sync and upload scripts write here; Lambda reads from here.
- **Top level** — Frontend (index.html, assets), `metadata/` (cohort JSON, model_performance_metrics), and optional **backups/**, **testing/**, **builds/** for non-production use.

---

## Files available on S3 after Step 6 sync

After notebook 5 **Step 6** (or `pgx_dashboard_visuals.py` with deploy), the following objects are available under the dashboard prefix. The **frontend** loads these via same-origin (path relative to index.html) or via the Lambda API fallback. All paths below are **object keys under the prefix** (e.g. `vcu/pgx-risk-calculator/`); age bands use **hyphen** (e.g. `25-44`).

| Category | S3 object keys (under prefix) |
|----------|-------------------------------|
| **Manifest** | `visualizations/dashboard_visual_objects.json` |
| **Metadata** | `metadata/model_performance_metrics.json`, `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json` |
| **Feature importance** | `visualizations/feature_importance/opioid_ed/aggregated_fi_heatmap.png`, `.json`; `visualizations/feature_importance/non_opioid_ed/aggregated_fi_heatmap.png`, `.json`; `visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png`; `visualizations/feature_importance/combined/aggregated_fi_heatmap.json` |
| **Causal** | `visualizations/causal/{cohort}/{age_band}/causal_data.json` (e.g. `opioid_ed/0-12`, `non_opioid_ed/25-44`) |
| **BupaR** | `visualizations/bupar/{cohort}/{age_band}/plots/*` — e.g. `{base}_activity_frequency.json`, `{base}_pre_target_activity_frequency.json`, `{base}_post_target_activity_frequency.json`, `{base}_trace_explorer_plot.json`, `{base}_process_matrix_drug_drug.json`, `{base}_process_matrix_drug_drug.png`, `{base}_activity_sequence_top.json`, `{base}_activity_sequence_top.png`, `{base}_overall_activity_frequency.png`, trace explorer PNGs, etc. (`{base}` = `{cohort}_{age_band}` with underscore, e.g. `opioid_ed_25_44`) |
| **DTW** | `visualizations/dtw/{cohort}/{age_band}/chart_data.json`, `sequence_heatmap.json`; `visualizations/dtw/{cohort}/{age_band}/plots/trajectory_overview_plot.json`, `*.html` |
| **FP-Growth** | `visualizations/fpgrowth/{cohort}/{age_band}/drug_name_itemsets.json`, `drug_name_rules.json`, `drug_name_encoding_map.json`, `drug_name_metrics.json`, etc.; `visualizations/fpgrowth/{cohort}/{age_band}/plots/{cohort}_{age_band}_combined_rules_network.html`, `{cohort}_{age_band}_drug_name_combined_top_itemsets.png` |
| **Cohort PGx** | `visualizations/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` (and related JSON/CSV in that dir) |
| **Frontend** | `index.html` (and any assets) |

These paths match what the **frontend** expects when using **static-first** loading (see [STATIC_FIRST_JSON.md](STATIC_FIRST_JSON.md)). The manifest `visualizations/dashboard_visual_objects.json` is the single source of truth for tab → static file mapping; the frontend builds URLs from it when available.

**Notes:**

- **Same prefix everywhere:** Lambda and every upload/sync script use `S3_DASHBOARD_PREFIX` (e.g. `vcu/pgx-risk-calculator`). Object key = that prefix + the path in the table (e.g. `vcu/pgx-risk-calculator/visualizations/dtw/opioid_ed/25-44/sequence_heatmap.json`).
- **Age band:** Upload scripts always send S3 keys with **hyphen** (e.g. `25-44`). Lambda builds URLs from `age_band` in the request (already hyphen). EC2 and local filenames use **underscore** (e.g. `25_44`) where the path is a directory or filename.
- **Metadata:** Not under the dashboard prefix for path-style assets. Lambda loads from container `/var/task/metadata/` or S3 `gold/dashboard/metadata/`. For static fallback the frontend may request same-origin `metadata/{cohort}.json`; that requires deploying metadata under the frontend prefix (e.g. `prefix/metadata/`) or using `?metadata=api` to use the API instead.

---

## Feature Importance

| Tab | Visual (heading) | Data artifact | EC2 file path | S3 object key (path-style) |
|-----|-------------------|---------------|---------------|-----------------------------|
| Feature Importance | Feature Importance by Age Band | `aggregated_fi_heatmap.png` or `.json` | `3a_feature_importance/{cohort}/aggregated_fi_heatmap.png` or `3a_feature_importance/plots/combined_cohorts_feature_importance_heatmap.png` | `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png` or `visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png` (combined: `visualizations/feature_importance/combined/aggregated_fi_heatmap.json` when present) |

---

## Causal Analysis

| Tab | Visual (heading) | Data artifact | EC2 file path | S3 object key (path-style) |
|-----|-------------------|---------------|---------------|-----------------------------|
| Causal Analysis | Top Causal Factors (FFA) | `dashboard_data.json` → Lambda `chart_data.causal_factors` | `10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json` | `visualizations/causal/{cohort}/{age_band}/causal_data.json` |
| Causal Analysis | SHAP Feature Importance | `dashboard_data.json` → Lambda `chart_data.shap_importance` | (same) | (same) |
| Causal Analysis | Feature Interactions | `dashboard_data.json` → Lambda `chart_data.feature_interactions` | (same) | (same) |
| Causal Analysis | Effect on outcome (by feature) | `dashboard_data.json` → Lambda `chart_data` (radar) | (same) | (same) |

---

## BupaR Process Mining (drug-specific visuals)

`{base}` = `{cohort}_{age_band_fname}` (e.g. `opioid_ed_25_44`). Pre suffix = `pre_f1120` (opioid_ed) or `pre_hcg` (non_opioid_ed).

| Tab | Visual (heading) | Data artifact | EC2 file path | S3 object key (path-style) |
|-----|-------------------|---------------|---------------|-----------------------------|
| BupaR Process Mining | Sequences to Target Outcomes (drugs) | `{base}_activity_sequence_top.png` | `10_risk_dashboard/visualizations/bupar/{cohort}/{age_band_fname}/plots/{base}_activity_sequence_top.png` | `bupar/{cohort}/{age_band}/plots/{base}_activity_sequence_top.png` |
| BupaR Process Mining | Overall Activity Frequency (drugs) | `{base}_activity_frequency.json` (+ optional PNG/HTML) | `.../plots/{base}_activity_frequency.json`, `{base}_overall_activity_frequency.png`, `{base}_activity_frequency_interactive.html` | `bupar/{cohort}/{age_band}/plots/{base}_activity_frequency.json`, `.../plots/{base}_overall_activity_frequency.png`, `.../plots/{base}_activity_frequency_interactive.html` |
| BupaR Process Mining | Pre-Target Activity Frequency (drugs) | `{base}_pre_target_activity_frequency.json`, `{base}_{pre}_activity_frequency.png` | `.../plots/{base}_pre_target_activity_frequency.json`, `.../plots/{base}_{pre}_activity_frequency.png` | `bupar/{cohort}/{age_band}/plots/{base}_pre_target_activity_frequency.json`, `.../plots/{base}_{pre}_activity_frequency.png` |
| BupaR Process Mining | Post-Target Activity Frequency (drugs) | `{base}_post_target_activity_frequency.json` | `.../plots/{base}_post_target_activity_frequency.json` | `bupar/{cohort}/{age_band}/plots/{base}_post_target_activity_frequency.json` |
| BupaR Process Mining | Trace Explorer (top 20 traces, drugs) | `{base}_trace_explorer_plot.json` or `{base}_trace_explorer_interactive.html` | `.../plots/{base}_trace_explorer_plot.json`, `.../plots/{base}_trace_explorer_interactive.html` | `bupar/{cohort}/{age_band}/plots/{base}_trace_explorer_plot.json`, `.../plots/{base}_trace_explorer_interactive.html` |
| BupaR Process Mining | Trace Explorer Pre-Target (drugs) | `{base}_trace_explorer_{pre}.png` | `.../plots/{base}_trace_explorer_{pre}.png` | `bupar/{cohort}/{age_band}/plots/{base}_trace_explorer_{pre}.png` |
| BupaR Process Mining | Process Matrix (Drug × Drug) | `{base}_process_matrix_drug_drug.png` or `.json` | `.../plots/{base}_process_matrix_drug_drug.png`, `.../plots/{base}_process_matrix_drug_drug.json` | `bupar/{cohort}/{age_band}/plots/{base}_process_matrix_drug_drug.png`, `.../plots/{base}_process_matrix_drug_drug.json` |
| BupaR Process Mining | (Interactive HTML deps) | `lib/*` (Plotly etc.) | `.../plots/lib/*` | `bupar/{cohort}/{age_band}/plots/lib/*` |

---

## DTW Trajectories

| Tab | Visual (heading) | Data artifact | EC2 file path | S3 object key (path-style) |
|-----|-------------------|---------------|---------------|-----------------------------|
| DTW Trajectories | Trajectory Analysis Overview (drugs) | Trajectory cluster plot image | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/plots/*.png` | `dtw/{cohort}/{age_band}/plots/*.png` |
| DTW Trajectories | Sample Trajectories (drugs) | (same) | (same) | (same) |
| DTW Trajectories | Trajectory Metrics | `chart_data.json` (metrics) | `.../chart_data.json` | `dtw/{cohort}/{age_band}/chart_data.json` |
| DTW Trajectories | High-Risk vs Low-Risk Trajectories (drugs) | `chart_data.json` → `high_risk_trajectories` | (same) | (same) |
| DTW Trajectories | Times Between Sequences (N3) | `chart_data.json` → `times_between_sequences`, `time_to_target_sequences` | (same) | (same) |
| DTW Trajectories | Target Pathway Patterns (drugs) | `chart_data.json` → `target_pathway_patterns` | (same) | (same) |
| DTW Trajectories | Common Sequences Heatmap (Drugs only) | `sequence_heatmap.json` | `.../sequence_heatmap.json` | `dtw/{cohort}/{age_band}/sequence_heatmap.json` |
| DTW Trajectories | Routine vs No Routine (Outcomes) | `chart_data.json` → `routine_comparison` | `.../chart_data.json` | `dtw/{cohort}/{age_band}/chart_data.json` |
| DTW Trajectories | (Routine vs No Routine event counts) | `chart_data.json` → `routine_comparison_counts` | (same) | (same) |
| DTW Trajectories | Event density filter | `chart_data.json` → `event_density_bins`, `routine_comparison_by_density`, `routine_comparison_counts_by_density`, `high_risk_trajectories_by_density` | (same) | (same) |

**Event density:** When the trajectory CSV has `event_density_bin` (from `create_dtw_trajectories.py`), chart_data includes the keys above so the dashboard can filter Routine vs No Routine and High-Risk charts by bin (All | Low | Medium | High | Extreme). See `10_risk_dashboard/visualizations/dtw/README.md`.

**DTW run order:** `create_dtw_features` writes to `feature_engineering/` (CSV + common_sequences). **`create_dtw_visuals`** reads that CSV, builds `chart_data.json` and `sequence_heatmap.json`, writes them under `{cohort}/{age_band_fname}/`, and uploads to S3. The artifact path check expects these JSONs in that output dir; run `create_dtw_visuals` per cohort/age_band after feature engineering.

---

## FP-Growth Patterns (drug only)

| Tab | Visual (heading) | Data artifact | EC2 file path | S3 object key (path-style) |
|-----|-------------------|---------------|---------------|-----------------------------|
| FP-Growth Patterns | Top Itemsets | `drug_name_itemsets.json` (JSON first, Plotly), `{base}_drug_name_combined_top_itemsets.png` (fallback) | `.../drug_name_itemsets.json`, `.../plots/{base}_drug_name_combined_top_itemsets.png` | `fpgrowth/{cohort}/{age_band}/drug_name_itemsets.json`, `fpgrowth/{cohort}/{age_band}/plots/{base}_*.png` |
| FP-Growth Patterns | Itemset Support Distribution | Same: `drug_name_itemsets.json` first (Plotly), PNG fallback | (same) | (same) |
| FP-Growth Patterns | Drug Association Network | `{base}_combined_rules_network.html` | `.../plots/{cohort}_{age_band_fname}_combined_rules_network.html` | `fpgrowth/{cohort}/{age_band}/plots/{cohort}_{age_band_fname}_combined_rules_network.html` |

---

## PGx Cohort

| Tab | Visual (heading) | Data artifact | EC2 file path | S3 object key (path-style) |
|-----|-------------------|---------------|---------------|-----------------------------|
| PGx Cohort | Gene–Drug–Phenotype Network Topology | `network_topology.html` | `10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html` | `cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` |

---

## Risk Assessment, Drugs, ICD Codes, CPT Codes, PGx Card, Documentation

These tabs use **Lambda/container or same-origin JSON**, not per-visual S3 object paths:

| Tab | Visual (heading) | Data source | EC2 path (if applicable) | S3 / API |
|-----|-------------------|-------------|---------------------------|---------|
| Risk Assessment | (score, band, model breakdown) | Models in container; `POST /risk` | `10_risk_dashboard/outputs/models/` (or bundled in Lambda image) | — |
| Drugs | (drug list, chips) | `GET /metadata` | `10_risk_dashboard/outputs/metadata/metadata_{cohort}.json` | `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json` |
| ICD Codes | (ICD list, chips) | (same) | (same) | (same) |
| CPT Codes | (CPT list, chips) | (same) | (same) | (same) |
| PGx Patient Card | PGx Patient Card | CPIC data in container | `10_risk_dashboard/outputs/cpic/` | — |
| Documentation | (tabs overview, RQ table, etc.) | Model performance metrics | `10_risk_dashboard/outputs/metadata/model_performance_metrics.json` | `metadata/model_performance_metrics.json` |

---

## Full path-style URL template

```
https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{object_key}
```

Example (PGx Cohort network):

- **Object key:** `cohort_pgx/networks/non_opioid_ed/55-64/network_topology.html` (S3 uses hyphen in age_band)
- **Full URL:** `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/cohort_pgx/networks/non_opioid_ed/55-64/network_topology.html`

Example (BupaR process matrix):

- **Object key:** `bupar/opioid_ed/25-44/plots/opioid_ed_25_44_process_matrix_drug_drug.png`
- **Full URL:** `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/bupar/opioid_ed/25-44/plots/opioid_ed_25_44_process_matrix_drug_drug.png`

Lambda builds these URLs via `_dashboard_s3_url(key)` where `key` is the object key under the bucket (prefix + path as in the table).
