# Research Questions → Artifacts (Canonical)

**Purpose:** Each visual and data artifact we **save and use** is tied directly to a research question. Only these artifacts are produced and retained for the dashboard. All other pipeline outputs are documented as archived (see [ARCHIVED_ARTIFACTS_NO_LONGER_USED.md](ARCHIVED_ARTIFACTS_NO_LONGER_USED.md)).

**Related:** [README_visualization_plan.md](README_visualization_plan.md), [DASHBOARD_TABS.md](DASHBOARD_TABS.md).

---

## Research questions and artifact mapping

| ID | Research question | Tab(s) | Artifacts we keep and use |
|----|-------------------|--------|----------------------------|
| **N1** | Routine vs no routine appointments → outcomes? (How do routine screenings reduce extreme outcomes?) | DTW Trajectories | `chart_data.json`: `routine_comparison`, `routine_comparison_counts`. Trajectory overview image (drug-only), sample trajectories image when present. |
| **N2** | What sequences lead to target outcomes? | BupaR Process Mining | Sequences to target: `*_activity_sequence_top.png`. Pre-target activity: `*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json`; `*_overall_activity_frequency.png` (optional fallback). Trace explorer: `*_trace_explorer_plot.json` or `*_trace_explorer_interactive.html`, `*_trace_explorer_pre_f1120.png` / `*_trace_explorer_pre_hcg.png`. |
| **N3** | What times between sequences lead to target outcomes? | DTW Trajectories | `chart_data.json`: `times_between_sequences`, `time_to_target_sequences` (when present). DTW overview/sample images for trajectory context. |
| **N4** | Drug connections → target? (Risk-predictive co-occurrence) | FP-Growth Patterns | `*_combined_rules_network.html` (drug association network). `*_drug_name_combined_top_itemsets.png`. `.../data/drug_name_itemsets.json` (client Plotly). |
| **N5** | What features drive outcome and how do they relate? | Causal Analysis, Feature Importance | **Causal:** `dashboard_data.json` → `causal_data`, `chart_data` (causal_factors, shap_importance, feature_interactions, radar). S3: `causal/{cohort}/{age_band_fname}/causal_data.json`. **Feature Importance:** `aggregated_fi_heatmap.png`, `aggregated_fi_heatmap.json` (per cohort or combined). |
| **N6** | What drug combinations drive polypharmacy ED? | Causal Analysis, BupaR | **Causal:** Same as N5 (drug-focused factors). **BupaR:** Drug × Drug process matrix: `*_process_matrix_drug_drug.png`, `*_process_matrix_drug_drug.json` (when present). Sequences and pre-target activity (same as N2). |

**Cohort-level (RQ1/RQ2):** Risk Assessment, Drugs, ICD, CPT, Causal, and the above tabs together address RQ1 (polypharmacy) and RQ2 (opioid ED). No separate artifact list; they use the same tabs and metadata (e.g. `metadata_{cohort}.json`, models).

---

## Per-tab artifact list (production only)

### Risk Assessment
- **Data:** Ensemble models (container or S3), `GET /metadata` (drugs, icd_codes, cpt_codes). No visualization artifacts; score and band from `POST /risk`.

### Feature Importance (N5)
- **Keep:** `feature_importance/{cohort}/aggregated_fi_heatmap.png`, `.json`; `feature_importance/combined_cohorts_feature_importance_heatmap.png` (and JSON when present).
- **API:** `GET /visualizations/feature_importance?cohort=`

### Causal Analysis (N5, N6)
- **Keep:** `causal/{cohort}/{age_band_fname}/causal_data.json` (from `dashboard_data.json` / combine SHAP+FFA). Lambda returns `chart_data` (causal_factors, shap_importance, feature_interactions, whatif).
- **API:** `GET /visualizations/causal?cohort=&age_band=`

### BupaR Process Mining (N2, N6)
- **Keep (drug-specific only):**
  - `*_activity_sequence_top.png` — Sequences to target.
  - `*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json` — Activity frequency bar charts.
  - `*_overall_activity_frequency.png`, `*_activity_frequency_interactive.html` — Optional fallbacks.
  - `*_trace_explorer_pre_f1120.png` / `*_trace_explorer_pre_hcg.png`, `*_trace_explorer_interactive.html`, `*_trace_explorer_plot.json` — Trace explorer.
  - `*_process_matrix_drug_drug.png`, `*_process_matrix_drug_drug.json` — Drug × Drug process matrix only.
- **Location:** `bupar/{cohort}/{age_band}/plots/`
- **API:** `GET /visualizations/bupar`, `GET /visualizations/bupar/activity_frequency`

### DTW Trajectories (N1, N3)
- **Keep:** `chart_data.json` (routine_comparison, routine_comparison_counts, high_risk_trajectories, times_between_sequences, time_to_target_sequences, target_pathway_patterns). `sequence_heatmap.json` (drug slice). Overview and sample trajectory images when generated.
- **Location:** `dtw/{cohort}/{age_band}/`
- **API:** `GET /visualizations/dtw?cohort=&age_band=`

### FP-Growth Patterns (N4)
- **Keep:** `*_combined_rules_network.html`, `*_drug_name_combined_top_itemsets.png`, `.../data/drug_name_itemsets.json`. Drug names only; no ICD/CPT itemset artifacts used.
- **Location:** `fpgrowth/{cohort}/{age_band}/plots/`, `.../data/`
- **API:** `GET /visualizations/fpgrowth`, `GET /visualizations/fpgrowth/network_html`

### PGx Cohort
- **Keep:** `cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html`
- **API:** `GET /visualizations/cohort_pgx?cohort=&age_band=`

---

## Pipeline and deployment

- **Step 9 (dashboard visuals):** BupaR, DTW, and FP-Growth pipelines produce and upload **only** the artifacts listed above (and their dependencies, e.g. `allowed_codes_shap_ffa_*.json`, `plots/lib/` for BupaR HTML).
- **BupaR upload:** `9_dashboard_visuals/bupar/create_bupar_visuals.py` uploads only RQ artifact filenames (allowlist in `_bupar_rq_artifact_basenames`); archived files (e.g. `process_matrix.png`, `frequency_map.png`) are not uploaded.
- **Lambda:** Serves only RQ artifact keys; see `10_risk_dashboard/backend/lambda_function.py` (BupaR candidates list trimmed to RQ-only).
- **FP-Growth:** Both cohorts use `drug_name` only (`cohort_fpgrowth.COHORT_ITEM_TYPES`); no ICD/CPT artifacts produced for dashboard.
- **DTW:** Uploads only `chart_data.json`, `sequence_heatmap.json`, and trajectory plot images; CSVs not uploaded.
- **Frontend:** Displays only these visuals; see `10_risk_dashboard/frontend/index.html` for panel keys and API usage.

When adding a new visual or artifact, add it to this table and the per-tab list, tie it to at least one research question, and update the BupaR allowlist or Lambda keys if applicable.
