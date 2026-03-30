# Research Questions → Artifacts (Canonical)

**Purpose:** Each visual and data artifact we **save and use** is tied directly to a research question. Only these artifacts are produced and retained for the dashboard. All other pipeline outputs are documented as archived (see [ARCHIVED_ARTIFACTS_NO_LONGER_USED.md](ARCHIVED_ARTIFACTS_NO_LONGER_USED.md)).

**Related:** [README_visualization_plan.md](README_visualization_plan.md). **Clinical OODA commentary (per visual):** [§ Visuals, research questions, and clinical OODA loop](#visuals-research-questions-and-clinical-ooda-loop) below.

---

## Research questions and artifact mapping

| ID | Research question | Tab(s) | Artifacts we keep and use |
|----|-------------------|--------|----------------------------|
| **N1** | Routine vs utilization appointments → outcomes? (How do routine screenings reduce extreme outcomes?) | DTW Trajectories | `chart_data.json`: `routine_comparison`, `routine_comparison_counts`, `routine_by_medical_utilization`. Trajectory overview image (drug-only), sample trajectories image when present. |
| **N2** | What sequences lead to target outcomes? | BupaR Process Mining | Sequences to target: `*_activity_sequence_top.png`. Pre-target activity: `*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json`; `*_overall_activity_frequency.png` (optional fallback). Trace explorer: `*_trace_explorer_plot.json` or `*_trace_explorer_interactive.html`, `*_trace_explorer_pre_f1120.png` / `*_trace_explorer_pre_hcg.png`. |
| **N3** | What times between sequences lead to target outcomes? | DTW Trajectories | DTW provides time-between and time-to-target for **aligned** sequences—more accurate than a straight BupaR comparison because alignment makes intervals comparable across patients (like-with-like); BupaR straight aggregate mixes stages. `chart_data.json`: `times_between_sequences`, `time_to_target_sequences` (when present). DTW overview/sample images for trajectory context. |
| **N4** | Drug connections → target? (Risk-predictive co-occurrence) | FP-Growth Patterns | `*_combined_rules_network.html` (drug association network). `*_drug_name_combined_top_itemsets.png`. `.../data/drug_name_itemsets.json` (client Plotly). |
| **N5** | What features drive outcome and how do they relate? | Causal Analysis, Feature Importance | **Causal:** `dashboard_data.json` → `causal_data`, `chart_data` (causal_factors, shap_importance, feature_interactions, radar). S3: `visualizations/causal/{cohort}/{age_band}/causal_data.json` (age_band with hyphen). **Feature Importance:** `aggregated_fi_heatmap.png`, `aggregated_fi_heatmap.json` (per cohort or combined). S3: `visualizations/feature_importance/{cohort}/...`, `visualizations/feature_importance/combined/...`. |
| **N6** | What drug combinations drive polypharmacy ED? | Causal Analysis, BupaR | **Causal:** Same as N5 (drug-focused factors). **BupaR:** Drug × Drug process matrix: `*_process_matrix_drug_drug.png`, `*_process_matrix_drug_drug.json` (when present). Sequences and pre-target activity (same as N2). |

**Cohort-level (RQ1/RQ2):** Risk Assessment, Drugs, ICD, CPT, Causal, and the above tabs together address RQ1 (polypharmacy) and RQ2 (opioid ED). No separate artifact list; they use the same tabs and metadata (e.g. `metadata_{cohort}.json`, models).

---

## Per-tab artifact list (production only)

### Risk Assessment
- **Data:** Ensemble models (container or S3), `GET /metadata` (drugs, icd_codes, cpt_codes). No visualization artifacts; score and band from `POST /risk`.

### Feature Importance (N5)
- **Keep:** `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png`, `.json`; `visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png` (and JSON when present).
- **API:** `GET /visualizations/feature_importance?cohort=`

### Causal Analysis (N5, N6)
- **Keep:** `visualizations/causal/{cohort}/{age_band}/causal_data.json` (S3 path uses hyphen; EC2 has `10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/`). Lambda returns `chart_data` (causal_factors, shap_importance, feature_interactions, whatif).
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
- **Keep:** `chart_data.json` (routine_comparison, routine_comparison_counts, routine_by_medical_utilization, high_risk_trajectories, times_between_sequences, time_to_target_sequences, target_pathway_patterns). `sequence_heatmap.json` (drug slice). Overview and sample trajectory images when generated.
- **Location:** `dtw/{cohort}/{age_band}/`
- **API:** `GET /visualizations/dtw?cohort=&age_band=`

### FP-Growth Patterns (N4)
- **Keep:** `*_combined_rules_network.html`, `*_drug_name_combined_top_itemsets.png`, `.../data/drug_name_itemsets.json`. Drug names only; no ICD/CPT itemset artifacts used.
- **Location:** `fpgrowth/{cohort}/{age_band}/plots/`, `.../data/`
- **API:** `GET /visualizations/fpgrowth`, `GET /visualizations/fpgrowth/network_html`

### PGx Cohort
- **Keep:** `cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` (S3 path uses hyphen; EC2 dirs use `age_band_fname`)
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

---

## Visuals, research questions, and clinical OODA loop {#visuals-research-questions-and-clinical-ooda-loop}

**Purpose:** For each saved artifact, document (1) which **research question(s)** it supports (N1–N6 and/or cohort-level **RQ1** polypharmacy / **RQ2** opioid ED), and (2) how it fits the **clinical OODA loop** used in [`10_risk_dashboard/README.md`](../README.md): **Observe** (surface risk and context), **Orient** (interpret drivers against guidelines and population evidence), **Decide** (choose tests, deprescribing, or referrals), **Act** (document changes, order PGx, adjust therapy). Density-bin routing (`n_event_bin`) is part of **Observe**—it matches the patient to the same strata the models were trained on.

**Clinical OODA (short):**

| Phase | Dashboard role |
|-------|----------------|
| **Observe** | Risk Assessment (score, band, model agreement); optional inputs (age, drugs, ICD, CPT); exploratory tabs show *why* similar patients experienced outcomes in training data. |
| **Orient** | Feature Importance, Causal Analysis, BupaR, DTW, FP-Growth, PGx Cohort: connect individual risk to **interpretable** pathways, timing, co-occurrence, and gene–drug evidence. |
| **Decide** | Causal / What-If style reasoning (risk deltas), FP-Growth hubs (deprescribing leverage), DTW windows (intervention timing), PGx Card (CPIC level A/B). |
| **Act** | PGx Patient Card PDF; prescribing / test-order actions implied by CPIC; follow-up encounters refresh **Observe** with new claims. |

---

### Risk Assessment (cohort-level **RQ1**, **RQ2**)

| Element | RQ | OODA / comment |
|---------|----|----------------|
| **Ensemble risk score + band + model agreement** | RQ1, RQ2 | **Observe:** Quantifies probability of cohort-specific outcome under the same density-bin models used in research. **Orient:** Places the patient in a calibrated risk band for shared decision-making. |
| **POST /risk** feature vector (age, drugs, ICD, CPT, optional counts) | RQ1, RQ2 | **Observe:** Encodes the clinical story in the same feature space as Chapters 3–4. **Decide:** Inputs are the modifiable targets for What-If and deprescribing. |
| **What-If / scenario comparison** (when enabled) | RQ1, RQ2 | **Decide** → **Act:** Counterfactual Δ risk supports tapering, substitution, or referral before ordering irreversible steps. |

---

### Feature Importance (heatmap) (**N5**; supports **RQ1**, **RQ2**)

| Artifact | RQ | OODA / comment |
|----------|----|----------------|
| `aggregated_fi_heatmap.png` / `.json` (per cohort or combined) | **N5** | **Orient:** Shows which code families dominate the *global* model across age—aligns clinician expectations with population-level drivers. **RQ1/RQ2:** Grounds polypharmacy vs opioid cohort differences in the same SHAP-derived view used in research. |

---

### Causal Analysis (`causal_data.json` / dashboard_data) (**N5**, **N6**; **RQ1**, **RQ2**)

| Element | RQ | OODA / comment |
|---------|----|----------------|
| **Causal factors + SHAP bars + interactions** | **N5**, **N6** | **Orient:** Separates FFA-structured drivers from correlation. **Decide:** Highlights **which** drugs/diagnoses to address first for polypharmacy harm (**N6**). |
| **Radar / top-factor panels** (when shown) | **N5** | **Orient:** Compact view of multi-feature responsibility for **RQ2** opioid and **RQ1** polypharmacy endpoints. |

---

### BupaR Process Mining artifacts (**N2**, **N3**, **N6**)

| Artifact | RQ | OODA / comment |
|----------|----|----------------|
| `*_activity_sequence_top.png` | **N2** | **Orient:** Answers “what sequence of care preceded the outcome for similar patients?”— **Act:** suggests where to intervene earlier in the pathway. |
| `*_activity_frequency.json`, pre/post target | **N2**, **N3** | **Orient:** Volume and ordering of activities before/around target—supports **N3** timing questions at aggregate level. |
| `*_overall_activity_frequency.png` | **N2** | **Observe / Orient:** Fallback intensity view when JSON is not loaded. |
| `*_trace_explorer_*`, `*_trace_explorer_plot.json` | **N2** | **Orient:** Explores dominant traces consistent with dissertation process-mining RQs. |
| `*_process_matrix_drug_drug.*` | **N6** | **Orient → Decide:** Drug × drug flow between activities—supports deprescribing **pairs** that co-travel in pathways. |

---

### DTW Trajectories (`chart_data.json`, images) (**N1**, **N3**; **RQ2**)

| Element | RQ | OODA / comment |
|---------|----|----------------|
| `routine_comparison`, `routine_comparison_counts`, `routine_by_medical_utilization` | **N1** | **Orient:** Links **routine vs utilization** coding patterns to outcomes—**Decide:** distinguishes under-management vs over-utilization. |
| `times_between_sequences`, `time_to_target_sequences` | **N3** | **Orient → Decide:** Warped alignment makes inter-event times comparable—**Act:** defines **when** windows for PDMP, PT, or MH referral still help (**RQ2**). |
| Trajectory overview / sample PNGs | **N1**, **N3** | **Observe / Orient:** Visual archetypes connect to Chapter 3 Rapid-Onset vs Chronic-Escalation narrative. |

---

### FP-Growth Patterns (**N4**; supports **RQ1**)

| Artifact | RQ | OODA / comment |
|----------|----|----------------|
| `*_drug_name_combined_top_itemsets.png`, `drug_name_itemsets.json` | **N4** | **Orient:** Shows **lifted** co-prescription sets among model-important drugs—**Decide:** prioritizes combinations to break. |
| `*_target_rules_network.png` / `.html` | **N4** | **Orient:** Network view of **high-risk associations**—**Act:** identifies hub drugs that sever multiple edges if stopped. |

---

### PGx Cohort (`network_topology.html`)

| Artifact | RQ | OODA / comment |
|----------|----|----------------|
| Gene–drug–literature topology | **RQ1** (PGx layer) | **Orient:** Places cohort risk in **pharmacogenomic** context—**Decide** → **Act:** motivates test ordering aligned with CPIC. |

---

### PGx Patient Card (Tab; CPIC snapshot in container)

| Element | RQ | OODA / comment |
|---------|----|----------------|
| **SNP → phenotype → CPIC** | **RQ1** (precision layer) | **Decide → Act:** Converts **Orient** evidence into guideline-ready dosing statements—closes the loop from population models to individual allele-informed therapy. |

---

### Documentation / metadata (`model_performance_metrics.json`, cohort metadata)

| Element | RQ | OODA / comment |
|---------|----|----------------|
| **Holdout metrics, cohort definitions** | **RQ1**, **RQ2** | **Observe:** Transparency for governance and committee review; supports trust in **Observe** phase scores. |

---

**Maintainers:** When adding or renaming an artifact, update the tables in [README_visualization_plan.md](README_visualization_plan.md) and this section, and ensure [`dashboard_visual_objects.json`](../visualizations/dashboard_visual_objects.json) stays aligned with the Lambda handlers in `backend/lambda_function.py`.
