# Validate dashboard frontend (index.html) updates

Use this README to **track dashboard updates** and to **validate** changes to **`10_risk_dashboard/frontend/index.html`** so the dashboard stays aligned with paths, research questions, and backend.

**References:** [README_dashboard_visual_artifact_paths.md](README_dashboard_visual_artifact_paths.md), [RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md), [ARCHIVED_ARTIFACTS_NO_LONGER_USED.md](ARCHIVED_ARTIFACTS_NO_LONGER_USED.md).

---

## Dashboard updates log

Record notable changes here (date, scope, and brief description). Run the checklist below when editing the frontend.

| Date       | Scope / tab              | Change |
|------------|--------------------------|--------|
| 2025-02-25 | Production finalization  | Removed legacy orphaned "Feature Interactions" tab (`#interactions-tab`). Interactions remain only as panel inside Causal Analysis tab. |
| 2025-02-25 | Validation README        | Added per-tab and main-page sections; this updates log for tracking. |

---

## Main page (shell)

- [ ] **Title:** `<h1>` is "PGx Risk Assessment Dashboard"; `<title>` matches.
- [ ] **Cohort tabs:** Two buttons—Opioid ED (`data-cohort="opioid_ed"`, `id="cohort-tab-opioid-ed"`) and Polypharmacy (`data-cohort="non_opioid_ed"`, `id="cohort-tab-polypharmacy"`). Values match partition names. Only one active at a time.
- [ ] **Primary tab bar:** Risk Assessment, Drugs, ICD Codes, CPT Codes, PGx Card, Documentation. `data-tab` values: `risk-assessment`, `drugs`, `icd-codes`, `cpt-codes`, `pgx-card`, `documentation`. Default active: `risk-assessment`.
- [ ] **Secondary tab bar (visualizations):** Feature Importance, Causal Analysis, BupaR Process Mining, DTW Trajectories, FP-Growth Patterns, PGx Cohort. `data-tab` values: `feature-importance-visualizations`, `causal-analysis`, `bupar-visualizations`, `dtw-visualizations`, `fpgrowth-visualizations`, `cohort-pgx-visualizations`.
- [ ] **Tab content IDs:** Each tab content div has `id="{data-tab}-tab"` and class `tab-content`; `switchTab()` receives the same string as `data-tab`.
- [ ] **API_BASE:** Config (and optional `?apiBase=` override) points to correct Lambda/API Gateway URL for the environment.
- [ ] **CDN scripts:** Plotly and Chart.js (or other) script tags present and versioned as expected.

---

## Risk Assessment tab (`risk-assessment-tab`)

- [ ] **Subtitle:** States Opioid ED vs Polypharmacy; cohort set by tabs above; age selects age band; age band 0-12 excluded; model indicator span present.
- [ ] **Controls:** Age input (min 13, max 114); codes summary and "Edit codes" (switches to Drugs tab); Calculate Risk Score, Compare Scenarios, Reset buttons.
- [ ] **Elements:** `#age`, `#codes-summary-text`, `#btnEditCodes`, `#btnRisk`, `#btnComparison`, `#btnReset`, `#status`, `#risk-display`, `#risk-score`, `#risk-band`, `#model-info`, `#model-indicator`.
- [ ] **Panels:** "Model Breakdown" (`#model-chart`), "Risk Distribution" (`#risk-dist-chart`), "Scenario Comparison" (`#comparison-mode`, `#comparison-scenarios`).
- [ ] **API:** Uses cohort from active cohort tab; `POST /risk` with age and codes; metadata from `GET /metadata` (cohort + age band).

---

## Drugs tab (`drugs-tab`)

- [ ] **Subtitle:** Explains drugs feed Risk Assessment; filtered by age band; search + multi-select (Ctrl/Cmd+click).
- [ ] **Controls:** Search input (`#drug-search`), multi-select (`#drugs`), selected drugs display (`#selected-drugs-display`).
- [ ] **Labels:** "Drug Names (search below, then select from list)", "Selected Drugs:".
- [ ] **API:** Metadata loads drugs for current cohort and age band; selections stored for risk calculation.

---

## ICD Codes tab (`icd-codes-tab`)

- [ ] **Subtitle:** Explains ICD codes feed Risk Assessment; filtered by age band; used for Opioid ED cohort; search + multi-select.
- [ ] **Controls:** Search input (`#icd-search`), multi-select (`#icds`), selected display (`#selected-icds-display`).
- [ ] **Labels:** "ICD Codes (search below, then select from list)", "Selected ICD Codes:".
- [ ] **Behavior:** Tab visibility may depend on cohort (e.g. hidden for Polypharmacy); metadata filtered by cohort/age band.

---

## CPT Codes tab (`cpt-codes-tab`)

- [ ] **Subtitle:** Explains CPT codes feed Risk Assessment; filtered by age band; used for Opioid ED cohort; search + multi-select.
- [ ] **Controls:** Search input (`#cpt-search`), multi-select (`#cpts`), selected display (`#selected-cpts-display`).
- [ ] **Labels:** "CPT Codes (search below, then select from list)", "Selected CPT Codes:".
- [ ] **Behavior:** Tab visibility may depend on cohort; metadata filtered by cohort/age band.

---

## PGx Card tab (`pgx-card-tab`)

- [ ] **Subtitle:** PGx card from ancestry/SNP input; CPIC guidelines; privacy note (anonymous, Patient ID optional, timestamp/IP recorded).
- [ ] **Controls:** Patient ID (optional), SNP textarea (`#snp-input`), file upload (`#snp-file`), Generate PGx Card, Reset.
- [ ] **Labels:** "Patient ID (Optional - Not Necessary)", "SNP Data Input", "Or Upload File"; format example (Gene,Variant1,Variant2).
- [ ] **Card output:** Sections Genes Tested, Drugs Requiring Dosing Modifications, Gene Details (`#pgx-genes-list`, `#pgx-drugs-list`, `#pgx-gene-details`). Privacy copy in card header.
- [ ] **API:** Card generation endpoint; CPIC data from backend/container.

---

## Documentation tab (`documentation-tab`)

- [ ] **Subtitle:** "How to use the PGx Risk Assessment Dashboard. All content stays on this page."
- [ ] **Sections (h2):** Overview, Tabs, Research questions and visualizations, Workflow, Model performance and at-risk identification, Feature importance sources for visuals, Technical notes.
- [ ] **Overview:** F1120 Opioid ED and Polypharmacy; age bands 13–114; cohort switch and age band selection.
- [ ] **Tabs list:** Matches actual tab order and names (Risk Assessment, Drugs, ICD Codes, CPT Codes, Causal, DTW, FP-Growth, BupaR, Feature Importance, PGx Patient Card, Documentation).
- [ ] **RQ table:** N1–N6 and RQ1/RQ2 with correct tab and visual names; N5 includes Feature Importance and Causal; N4 drug-only; N3 DTW; etc.
- [ ] **Model metrics container:** `#doc-metrics-container`; loads `model_performance_metrics.json` (path-style S3 or API).
- [ ] **Feature importance sources:** BupaR/DTW use SHAP/FFA combined; FP-Growth uses final feature importance list. Copy matches RESEARCH_QUESTIONS_ARTIFACTS.

---

## Feature Importance tab (`feature-importance-visualizations-tab`)

- [ ] **Single dropdown ("View:")** with exactly three options: `opioid_ed` (Opioid ED), `non_opioid_ed` (Polypharmacy), `combined` (Combined cohorts). No per–age-band options; one cohort or combined view at a time.
- [ ] **One visual at a time:** Single panel (`#fi-single-panel`) showing either Plotly heatmap (from JSON) or image fallback. No multiple heatmaps on the same tab.
- [ ] **Heading:** Panel title "Feature Importance by Age Band" (and "— Opioid ED" / "— Polypharmacy" / "— Combined cohorts" when loaded). Matches path README visual heading.
- [ ] **Subtitle/copy:** States that user chooses Opioid ED, Polypharmacy, or Combined to view one heatmap at maximum size; heatmaps from Step 3a (Monte Carlo CV); rows = top features, columns = age bands.
- [ ] **API:** `GET /visualizations/feature_importance?cohort=` with `cohort` = `opioid_ed` | `non_opioid_ed` | `combined`. Response: `heatmap_data` (JSON for Plotly) and/or `heatmap_url` (image fallback).
- [ ] **JSON-first loading:** Prefer `heatmap_data` and render with Plotly; if missing or invalid, fall back to `heatmap_url` image. Status message reflects "loaded (image)" vs "loaded" (chart).
- [ ] **Element IDs:** `fi-cohort`, `btnLoadFeatureImportance`, `fi-status`, `fi-single-panel`, `fi-panel-title`, `fi-heatmap-container`, `fi-heatmap-chart`, `fi-heatmap-image` used consistently by script.

---

## Causal Analysis tab (`causal-analysis-tab`)

- [ ] **Subtitle:** Research focus (features drive outcome; drug combinations); uses same cohort, age, and code selections as Risk Assessment.
- [ ] **Controls:** What-if scenario input (`#causal-whatif-codes`), Load Causal Analysis, Clear filters. Cohort/age from Risk Assessment context.
- [ ] **Headings (path README):** Top Causal Factors (FFA), SHAP Feature Importance, **Feature Interactions** (panel within this tab, not a separate tab), Effect on outcome (by feature). Interactions come from Lambda `chart_data.feature_interactions` (causal_data.json).
- [ ] **API:** `GET /visualizations/causal?cohort=&age_band=`; response `chart_data` (causal_factors, shap_importance, feature_interactions, radar/whatif). Path README: `causal/{cohort}/{age_band_fname}/causal_data.json`.
- [ ] **Element IDs / chart containers:** Match script (e.g. causal factors chart, SHAP chart, `#interactions-chart` for Feature Interactions, radar).

---

## BupaR Process Mining tab (`bupar-visualizations-tab`)

- [ ] **Subtitle:** "Drug-specific visuals" (cohort/age band and show/hide apply to drug-only panels); "Event-to-target" uses **all activity types** (Drug, ICD, CPT) and is separate.
- [ ] **Labels:** "Cohort (drug-specific visuals only)", "Age band (drug-specific visuals only)", "Show visuals (drug-specific only)" with legend/copy that show/hide applies to drug-only panels only; event-to-target not controlled here.
- [ ] **Section label:** "Drug-specific visuals" with copy that event-to-target BupaR visuals are kept separate.
- [ ] **Headings (RQ artifacts only):** Sequences to Target Outcomes (drugs), Overall Activity Frequency (drugs), Pre-Target Activity Frequency (drugs), Post-Target Activity Frequency (drugs), Trace Explorer (top 20 traces, drugs), Trace Explorer Pre-Target (drugs), Process Matrix (Drug × Drug). No panels for archived artifacts (e.g. combined process_matrix, frequency_map).
- [ ] **API:** `GET /visualizations/bupar` (and activity_frequency etc.); request only RQ artifact keys; no trace_explorer_image, process_matrix_image, frequency_map_image.
- [ ] **S3/iframe URLs:** Path-style only; keys match README_dashboard_visual_artifact_paths.

---

## DTW Trajectories tab (`dtw-visualizations-tab`)

- [ ] **Subtitle:** Research focus (drug trajectories; routine screenings vs outcomes); cohort/age band select; sub-tabs: Overview & Trajectories (drugs), Routine vs No Routine.
- [ ] **Controls:** Cohort, Age Band dropdowns; Load DTW Visualizations. Sub-tabs: "Overview & Trajectories (drugs)", "Routine vs No Routine".
- [ ] **Overview sub-panel headings:** Trajectory Analysis Overview (drugs), Sample Trajectories (drugs), Trajectory Metrics, High-Risk vs Low-Risk Trajectories (drugs), Times Between Sequences (N3), Target Pathway Patterns (drugs), Common Sequences Heatmap (Drugs only).
- [ ] **Routine sub-panel:** "Routine vs No Routine (Outcomes)" with copy about admin ICD and outcome rate; routine comparison counts container.
- [ ] **API / data:** `chart_data.json` (routine_comparison, routine_comparison_counts, high_risk_trajectories, times_between_sequences, time_to_target_sequences, target_pathway_patterns); `sequence_heatmap.json`; overview/sample images. Path README and RQ (N1, N3) match.

---

## FP-Growth Patterns tab (`fpgrowth-visualizations-tab`)

- [ ] **Subtitle:** Research focus (drug sequences/combinations → target); drug names only; itemsets and association rules from pharmacy events.
- [ ] **Controls:** Cohort, Age Band; Load FP-Growth Visualizations. Status: "Drug-name itemsets and network only."
- [ ] **Headings (path README):** Top Itemsets, Itemset Support Distribution, Drug Association Network. Drug-only; no ICD/CPT itemset panels.
- [ ] **Containers:** `#fpgrowth-itemsets-image`, `#fpgrowth-support-image`, `#fpgrowth-network-container` (e.g. iframe for combined_rules_network.html).
- [ ] **API:** Request keys and paths match path README and RQ (N4); drug_name only.

---

## PGx Cohort tab (`cohort-pgx-visualizations-tab`)

- [ ] **Subtitle:** Gene–drug–phenotype network topology; top PGx genes (SHAP/FFA) and PharmGKB VIP; cohort and age band select, then load.
- [ ] **Controls:** Cohort, Age Band dropdowns; Load PGx Cohort Network.
- [ ] **Heading:** "Gene–Drug–Phenotype Network Topology" (matches path README).
- [ ] **Iframe:** `#cohort-pgx-iframe` loads `network_topology.html`; URL path-style S3: `cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html`.
- [ ] **API:** `GET /visualizations/cohort_pgx?cohort=&age_band=` or direct S3 path; age_band format (e.g. `25_44`) consistent with backend.

---

## URLs and API (global)

- [ ] **Path-style S3 only:** All iframe/image URLs for S3 use path-style:  
  `https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{key}`. No virtual-hosted style.
- [ ] **Metadata endpoints:** References to `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json`, `metadata/model_performance_metrics.json` match backend and path README.
- [ ] **Visualization API keys:** BupaR and other handlers request only RQ artifact keys; no archived keys (e.g. trace_explorer_image, process_matrix_image, frequency_map_image).

---

## JSON-first and loading behavior

- [ ] Where docs specify JSON-first (Feature Importance heatmap_data, BupaR activity frequency JSON), frontend prefers JSON and falls back to PNG/HTML as documented.
- [ ] Cohort/age_band query params and dropdown values match backend (e.g. age band `25_44` in URLs vs `25-44` in labels).

---

## After frontend changes

- [ ] **Artifact path check:** Run `10_risk_dashboard/data_preparation/check_dashboard_artifact_paths.py` (e.g. from notebook 5 before Step 6 / Lambda deploy) to confirm required paths exist.
- [ ] **Deploy:** Re-sync frontend to S3 (notebook 5 Step 6); if Lambda response shape changed, update and deploy Lambda first.
- [ ] **Optional:** Run any frontend or dashboard tests in `10_risk_dashboard/tests/` or `11_testing/` if present.

---

## Quick reference

| Doc | Use |
|-----|-----|
| README_dashboard_visual_artifact_paths.md | Tab name, visual heading, EC2 path, S3 key (path-style) |
| RESEARCH_QUESTIONS_ARTIFACTS.md | Which artifacts we keep; per-tab list; pipeline/Lambda alignment |
| ARCHIVED_ARTIFACTS_NO_LONGER_USED.md | Do not add panels or API requests for these |
