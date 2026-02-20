# PGx Risk Dashboard: Visualization Plan

**Reference:** [PGx Risk Calculator](https://jerome-dixon.io.s3.us-east-1.amazonaws.com/vcu/pgx-risk-calculator/index.html)

This document is the **single source of truth** for the final production workflow, data visualization outputs, and research-question alignment. Prefer **full-dataset, filter-to-features**: pipeline produces cohort/age_band (and item_type where applicable) outputs; dashboard and Lambda only filter.

**See also:** [README_dashboard_visuals_review.md](README_dashboard_visuals_review.md) (this folder) for a detailed RQ-alignment and DTW-optimization review and lessons learned; that doc references this plan as the canonical workflow.

---

## Final production workflow

| Step | Command / script | Purpose |
|------|-------------------|--------|
| **Sync (optional)** | `python 9_dashboard_visuals/sync_visualization_data_from_s3.py` | Pull model data and feature importance from S3 so BupaR/DTW/FP-Growth have inputs. |
| **Dashboard visuals** | `python 9_dashboard_visuals/run_dashboard_visuals.py` | BupaR → DTW trajectories → DTW visuals → FP-Growth for all cohort/age_band combinations. Use `--no-sync` if data is already local; `--force` to re-run; `--cohort X --age-band Y` to restrict. |
| **Quick DTW test** | `python 9_dashboard_visuals/run_dtw_test_one_age_band.py --age-band 25-44` | Run DTW (trajectories + visuals) for one age band and both cohorts (opioid_ed, non_opioid_ed). Requires allowed_codes and model_events. |

**Prerequisite:** SHAP/FFA combined allowed codes for each (cohort, age_band). Created on EC2 or via `sync_visualization_data_from_s3.py --allowed-codes-only`. See `9_dashboard_visuals/README.md`.

---

## Tab order and outputs

| # | Tab | Purpose | Key outputs |
|---|-----|---------|-------------|
| 1 | **Causal Analysis** | Features driving outcome; relations; drug combinations → polypharmacy ED | FFA + SHAP importance, feature interactions, radar (optional). Data: Lambda `/causal/importance`, S3 gold/ffa_analysis, gold/shap_analysis. |
| 2 | **BupaR Process Mining** | Sequences to target (N2); times between sequences (N3 optional) | Activity frequency (overall, pre-target), trace explorer (aggregated), activity sequence top. Static PNG + interactive HTML (year dropdown). No Gantt (see `9_dashboard_visuals/bupar/ARCHIVE_GANTT_REMOVAL.md`). |
| 3 | **DTW Trajectories** | Routine vs no routine (N1); times between sequences (N3) | Trajectory cluster plots (1D/3D), routine vs no routine chart, high-risk trajectories, times-between-sequences (N3), time-to-target (N3), target pathway patterns. chart_data.json. |
| 4 | **FP-Growth Patterns** | ICD/CPT/Drug connections → target (N4) | Co-occurrence network, top itemsets, support distribution. Item type = Drug / ICD / CPT. |

**Creation code:** All visualization creation lives in **`9_dashboard_visuals/`** (step 9). Outputs are written under **`10_risk_dashboard/visualizations/`** and uploaded to the dashboard S3 bucket. See `10_risk_dashboard/visualizations/README.md` for directory layout and script names per tab.

---

## Research questions → tabs

| ID | Question | Tab | Visuals |
|----|----------|-----|---------|
| **N1** | Routine vs no routine appointments → outcomes? | DTW | Routine vs no routine (admin ICD), high-risk trajectories, trajectory overview. |
| **N2** | What sequences lead to target outcomes? | BupaR | Sequences to target, pre-target activity frequency, trace explorer (aggregated). |
| **N3** | What times between sequences lead to target outcomes? | DTW, BupaR | DTW: times-between-sequences and time-to-target charts (by routine bucket). BupaR: optional future time-between summary. |
| **N4** | ICD/CPT/Drug connections → target? | FP-Growth | Co-occurrence network, itemsets (filter by item type). |
| **N5** | What features drive outcome and how do they relate? | Causal | FFA, SHAP, feature interactions, radar (recommended). |
| **N6** | What drug combinations drive polypharmacy ED? | Causal + BupaR | Causal drug factors; BupaR sequences / pre-target activity. |

Original RQ1/RQ2 (cohort-level questions) are covered by the same tabs and risk assessment; see `docs/CrossStep_Workflow/README_research_questions_mapping.md` for full mapping.

---

## Data pattern

- **SHAP/FFA-driven:** BupaR, DTW, and FP-Growth (and Causal when no user filter) use **model-important features** (SHAP/FFA from Step 7/8). Event logs, trajectories, and itemsets are restricted to those codes so visuals align with what drives model results.
- **Filterability:** Risk Assessment and Causal use the user's selected drugs/ICD/CPT when provided. BupaR and DTW are cohort/age_band only (filtering done at pipeline time). FP-Growth: cohort, age band, item type. See `README_IMPLEMENTATION_PLAN_TAB_VISUALIZATIONS.md` for API details.

---

## Implementation notes

- **Causal:** Radar chart (top 5–8 features) can be built in frontend from causal_factors + shap_importance.
- **BupaR:** Trace explorer is **aggregated activity frequency** (one bar per activity, ordered by frequency, aligned to N2/N6). Gantt not produced.
- **DTW:** Two-step pipeline: `create_dtw_trajectories.py` (features CSV including N3 time-between metrics) then `create_dtw_visuals.py` (plots and chart_data.json).
- **FP-Growth:** Keep item type selector; co-occurrence network is the main visual. Network HTML loaded via iframe or fetch.

---

## Related docs

| Doc | Use |
|-----|-----|
| **`9_dashboard_visuals/README.md`** | Step 9 pipeline, run commands, quick DTW test. |
| **`10_risk_dashboard/visualizations/README.md`** | Output directories (bupar, dtw, fpgrowth) and creation script names. |
| **`docs/Step9_RiskDashboard/README_bupar_dashboard_visualizations.md`** | BupaR file names and S3 layout. |
| **`10_risk_dashboard/docs/README_IMPLEMENTATION_PLAN_TAB_VISUALIZATIONS.md`** | Per-tab implementation, API, checklists. |
| **`10_risk_dashboard/docs/README_CALCULATOR_WORKFLOW.md`** | Deployment workflow (metadata, models, Lambda, deploy). |
| **`10_risk_dashboard/docs/TAB_ARCHITECTURE_PHTS_VS_PGX.md`** | Tab layout comparison (PHTS vs PGx). |
| **`archived/dashboard_docs/README.md`** | Index of archived planning/historical docs and lessons learned. |
