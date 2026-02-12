# Implementation Plan: Data Visualizations by Tab

This document is an **implementation plan** for each dashboard tab’s data visualizations: data sources, API usage, visual elements, implementation status, and verification checklists.

**Related:**  
- [VISUALIZATION_PLAN.md](VISUALIZATION_PLAN.md) – Research questions → tabs and recommended visuals  
- [Backend README](../backend/README.md) – API endpoints and responses  
- [Main dashboard README](../README.md) – Data sources and deployment

---

## How to Use This Document

- **Implementing a tab:** Use the tab’s “Data sources” and “API” sections, then work through the “Visuals” and “Checklist.”
- **Verifying a tab:** Use the “Checklist” and “Status” for each visual.
- **Adding a new visual:** Add a row under the tab’s “Visuals” table and extend the checklist.

**Data pattern:** Prefer **full-dataset, filter-to-features**: pipeline produces cohort/age_band (and item_type where applicable) outputs; dashboard and Lambda only filter. Avoid running heavy analysis in Lambda.

**SHAP/FFA-driven visualizations:** BupaR, DTW, FP-Growth, and Causal visuals are driven by **model-important features** (SHAP and/or FFA from Step 7 / Step 8). The **original dataset** (model_data / event parquet) is filtered to those items before running process mining, trajectory analysis, or pattern mining; Causal tab defaults to top SHAP/FFA features when no user selection. This makes the visuals **meaningful to what is driving model results**—sequences, trajectories, itemsets, and causal charts focus on the same drugs/ICDs/CPTs the models use for prediction. See `py_helpers/shap_ffa_fpgrowth_utils.py`, BupaR `allowed_codes_shap_ffa_*.json`, and per-tab notes below.

---

## Filterability by user selections (Drug / ICD / CPT)

Not all tabs use the user’s selected drugs, ICDs, or CPTs from the Drugs / ICD Codes / CPT Codes selectors. Summary:

| Tab | Filterable by selected codes? | How |
|-----|-------------------------------|-----|
| **Risk Assessment** | ✅ Yes | Risk score uses selected `drugs`, `icds`, `cpts` in `POST /risk`. |
| **Causal Analysis** | ✅ Yes (or SHAP/FFA default) | Optional query params `drugs`, `icds`, `cpts`. When present, causal and SHAP charts show only those features. When absent, Lambda restricts to **top 500 SHAP/FFA important features** (same importance-driven view as other tabs). |
| **Drugs / ICD / CPT tabs** | N/A | These tabs are where the user makes the selections; they don’t “filter” another dataset. |
| **BupaR Process Mining** | ❌ No (SHAP/FFA at pipeline) | Cohort and age band. Pipeline filters event log to **SHAP/FFA important codes** (see `allowed_codes_shap_ffa_*.json`); no per-request code list. |
| **DTW Trajectories** | ❌ No (SHAP/FFA at pipeline) | Cohort and age band. Pipeline filters trajectories to **SHAP/FFA important codes** in `create_dtw_features.py`; no per-request code list. |
| **FP-Growth Patterns** | ✅ By cohort, age band, item type | Filters by **cohort**, **age band**, and **item type** (Drug Names, ICD Codes, CPT Codes, Medical Codes), not by the user’s specific selected code list. Optional future: pass selected codes to highlight or filter itemsets. |

**To get Causal charts filtered to your selections:** Select drugs and/or ICD/CPT codes in the Drugs and ICD/CPT tabs, then open the Causal Analysis tab and click “Load Causal Analysis.” The status line will show “(filtered to your selected codes)” when a filter was applied.

---

## Tab 1: Risk Assessment

**Purpose:** Compute and display risk score for F1120 Opioid ED (13–64) or Polypharmacy (65–114) from selected drugs, ICDs, and CPTs.

| Item | Detail |
|------|--------|
| **API** | `POST /risk` (body: `cohort`, `age_band`, `drugs[]`, `icds[]`, `cpts[]`) |
| **Data sources** | Ensemble models (CatBoost, XGBoost, XGBoost RF) from container or S3 `gold/dashboard/models/` |
| **Frontend elements** | Risk score numeric, risk band badge, model breakdown (optional) |

### Visuals / outputs

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| Risk score (0–1) | Numeric | Lambda `POST /risk` → `risk_score` | ✅ Implemented |
| Risk band (Low / Medium / High) | Badge | Lambda → `risk_band` | ✅ Implemented |
| Model indicator (cohort/age) | Badge | Client from age/cohort | ✅ Implemented |

### Implementation checklist

- [ ] Metadata loaded so cohort/age_band selectors and code lists work (`GET /metadata`).
- [ ] Risk request sends selected `drugs`, `icds`, `cpts` for chosen `cohort` and `age_band`.
- [ ] Score and band update after “Calculate Risk” (and on scenario change if applicable).
- [ ] Ages 95–114 show 85–94 band; 0–12 excluded (per docs).

---

## Tab 2: Drugs

**Purpose:** Let users select drugs for risk calculation; show selected drugs as chips.

| Item | Detail |
|------|--------|
| **API** | `GET /metadata?cohort=...` → `drugs` list |
| **Data sources** | `gold/dashboard/metadata/metadata_{cohort}.json` (or container) |
| **Frontend elements** | Multi-select (or search), chips for selected drugs, remove-one control |

### Visuals / outputs

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| Drug dropdown/list | Multi-select | `metadata.drugs` | ✅ Implemented |
| Selected drugs chips | Chips + remove | Client state | ✅ Implemented |

### Implementation checklist

- [ ] Drugs list populated from metadata for selected cohort.
- [ ] Selections sync with Risk Assessment payload and chips update immediately.

---

## Tab 3: ICD Codes

**Purpose:** Let users select ICD codes for risk calculation; show selected codes as chips.

| Item | Detail |
|------|--------|
| **API** | `GET /metadata?cohort=...` → `icd_codes` (or equivalent) |
| **Data sources** | Same metadata as Drugs |
| **Frontend elements** | Multi-select, chips for selected ICDs, remove-one |

### Visuals / outputs

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| ICD dropdown/list | Multi-select | `metadata.icd_codes` | ✅ Implemented |
| Selected ICD chips | Chips + remove | Client state | ✅ Implemented |

### Implementation checklist

- [ ] ICD list populated from metadata; selections and chips stay in sync with risk request.

---

## Tab 4: CPT Codes

**Purpose:** Let users select CPT codes for risk calculation; show selected codes as chips.

| Item | Detail |
|------|--------|
| **API** | `GET /metadata?cohort=...` → `cpt_codes` (or equivalent) |
| **Data sources** | Same metadata as Drugs/ICD |
| **Frontend elements** | Multi-select, chips for selected CPTs, remove-one |

### Visuals / outputs

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| CPT dropdown/list | Multi-select | `metadata.cpt_codes` | ✅ Implemented |
| Selected CPT chips | Chips + remove | Client state | ✅ Implemented |

### Implementation checklist

- [ ] CPT list and chips work like Drugs/ICD and are included in risk payload.

---

## Tab 5: PGx Card

**Purpose:** Generate a PGx patient card from gene/variant input.

| Item | Detail |
|------|--------|
| **API** | `POST /pgx/card` (body: e.g. `variants: [{ gene, variants[] }]`) |
| **Data sources** | CPIC/clinical logic in Lambda; may use bundled Excel/JSON |
| **Frontend elements** | Form for gene/variants, card layout (drug–gene interactions, recommendations) |

### Visuals / outputs

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| PGx card content | Card / sections | Lambda `POST /pgx/card` | ✅ Implemented |
| Gene/variant form | Form | Client | ✅ Implemented |

### Implementation checklist

- [ ] Card request sends correct body; response rendered in PGx Card tab.
- [ ] Drug–gene interactions and recommendations clearly shown.

---

## Tab 6: Causal Analysis (visualization tab)

**Purpose:** Show what features drive the target outcome and how they relate; support “what drug combinations drive polypharmacy ED?”

| Item | Detail |
|------|--------|
| **API** | `GET /visualizations/causal?cohort=...&age_band=...` |
| **Data sources** | FFA: `gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet`; SHAP: `gold/shap_analysis/{cohort}/{age_band}/*_shap_global_importance_xgboost.csv` |
| **Frontend trigger** | “Load Causal Analysis” with cohort + age band selected |

### Visuals

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| Top Causal Factors (FFA) | Bar chart (Plotly) | `causal_factors[]` (feature, importance) | ✅ Implemented |
| SHAP Feature Importance | Bar chart (Plotly) | `shap_importance[]` (feature, importance) | ✅ Implemented |
| Feature Interactions | Panel | `interactions` (optional) | ⚠️ Placeholder / optional |
| Feature Relations (Radar) | Radar (Plotly) | Derived from `causal_factors` (top N, normalized) | ✅ Implemented |
| Drug-combo emphasis | Text / link to BupaR | Causal factors + BupaR tab | 📋 Optional: dedicated subsection |

### Implementation checklist

- [ ] Causal and SHAP data loaded in Lambda from S3 for selected cohort/age_band.
- [ ] Response includes `causal_factors` and `shap_importance`; frontend renders both bar charts.
- [ ] Radar chart built from top N causal factors (normalized); empty state handled.
- [ ] If “Feature Interactions” is used: backend serves interaction data or frontend shows “Not available.”
- [ ] Optional: small “Drug combinations” block that highlights drug-related causal features or links to BupaR.

---

## Tab 7: BupaR Process Mining (visualization tab)

**Purpose:** Show sequences and times that lead to target outcomes (research questions N2, N3).

| Item | Detail |
|------|--------|
| **API** | `GET /visualizations/bupar?cohort=...&age_band=...` |
| **Data sources** | S3 `gold/feature_importance/{cohort}/{age_band}/plots/`: PNGs for activity frequency, Gantt, sequences, milestones Pipeline uses **SHAP/FFA allowed codes** when `allowed_codes_shap_ffa_*.json` exists. Optional: `gold/bupar/{cohort}/{age_band}/` CSVs for time-between summary. |
| **Frontend trigger** | “Load BupaR Visualizations” with cohort + age band |

### Visuals

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| Sequences to Target Outcomes | Image | `sequence_image` (S3 path) | ✅ Implemented |
| Times Between Sequences (Pre-Target Gantt) | Image | `pre_target_gantt_image` | ✅ Implemented |
| Overall Activity Frequency | Image | `activity_frequency_image` | ✅ Implemented |
| Pre-Target Activity Frequency | Image | `pre_target_frequency_image` | ✅ Implemented |
| Post-Target Activity Frequency | Image | `post_target_frequency_image` | ✅ Implemented |
| Gantt (Overall / Post-Target) | Image | `gantt_image`, `post_target_gantt_image` | ✅ Implemented |
| Activity Milestones Gantt | Image | `milestones_image` | ✅ Implemented |
| Time-between summary (e.g. bar chart) | Chart (optional) | Pipeline CSV → Lambda or frontend | 📋 Optional |

### Implementation checklist

- [ ] Pipeline writes BupaR PNGs to `gold/feature_importance/{cohort}/{age_band}/plots/` with expected names (see Lambda handler).
- [ ] Lambda returns all image S3 paths; frontend resolves URLs (presigned or proxy) and displays images.
- [ ] Panel titles match plan: “Sequences to Target Outcomes,” “Times Between Sequences (Pre-Target Gantt).”
- [ ] Optional: time-between summary (e.g. time from last drug/ICD/CPT to target by sequence type) from pipeline + Lambda or frontend.

---

## Tab 8: DTW Trajectories (visualization tab)

**Purpose:** Show trajectory patterns, routine vs non-routine proxy, and high-risk vs low-risk trajectory archetypes (research questions N1, RQ2).

| Item | Detail |
|------|--------|
| **API** | `GET /visualizations/dtw?cohort=...&age_band=...` |
| **Data sources** | Images: `gold/feature_importance/{cohort}/{age_band}/plots/` (`dtw_trajectory_analysis_*.png`, `dtw_sample_trajectories_*.png`). Chart data: `gold/feature_engineering/6_dtw/{cohort}/{age_band}/dtw_features_*.csv`. Pipeline (`create_dtw_features.py`) restricts trajectories to **SHAP/FFA important codes** when available. |
| **Frontend trigger** | “Load DTW Visualizations” with cohort + age band |

### Visuals

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| Trajectory Analysis Overview | Image | `overview_image` (S3 path) | ✅ Implemented |
| Sample Trajectories | Image | `sample_trajectories_image` | ✅ Implemented |
| Trajectory Metrics | Bar chart (Plotly) | `metrics` (object) | ✅ Implemented (if backend sends metrics) |
| Routine vs No Routine (Outcomes) | Bar chart (Plotly) or image | `routine_comparison` (x, y) or `routine_comparison_image` | ✅ Implemented (chart from DTW CSV) |
| High-Risk vs Low-Risk Trajectories | Bar chart (Plotly) or image | `high_risk_trajectories` (x, y) or `high_risk_trajectories_image` | ✅ Implemented (chart from DTW CSV) |

### Implementation checklist

- [ ] DTW feature CSV exists in S3 for each cohort/age_band (from `create_dtw_features.py` → `gold/feature_engineering/6_dtw/`).
- [ ] Lambda loads CSV and returns `routine_comparison` (outcome rate by trajectory intensity: Low/Medium/High) and `high_risk_trajectories` (outcome rate by quartile of DTW distance/length).
- [ ] Frontend renders both bar charts when present; shows placeholder message when data missing.
- [ ] Optional: pipeline generates overview/sample trajectory PNGs and uploads to `gold/feature_importance/.../plots/`; frontend displays them (with presigned or proxy URLs).

---

## Tab 9: FP-Growth Patterns (visualization tab)

**Purpose:** Show connections between ICD, CPT, and drugs that associate with target outcome (research question N4). Exploratory only (not model features). FP-Growth runs on the **original dataset** restricted to items identified by **SHAP/FFA** (Step 7 / Step 8); see `visualizations/fpgrowth/README_visualization_only.md` and `py_helpers/shap_ffa_fpgrowth_utils.py`.

| Item | Detail |
|------|--------|
| **API** | `GET /visualizations/fpgrowth?cohort=...&age_band=...&item_type=...` |
| **Data sources** | S3 `gold/fpgrowth/{cohort}/{age_band}/plots/`: `*_top20_itemsets.png`, `*_itemset_support.png`, `*_network.html` (item_type: drug_name, icd_code, cpt_code, medical_code). |
| **Frontend trigger** | “Load FP-Growth Visualizations” with cohort, age band, and item type |

### Visuals

| Visual | Type | Data from | Status |
|--------|------|-----------|--------|
| Top Itemsets | Image | `itemsets_image` (S3 path) | ✅ Implemented |
| Itemset Support Distribution | Image | `support_image` | ✅ Implemented |
| ICD/CPT/Drug Connections (Co-occurrence Network) | HTML (iframe or inject) | `network_html` (S3 path or content) | ✅ Implemented (container present; URL resolution may need proxy/presigned) |

### Implementation checklist

- [ ] Pipeline writes itemsets PNGs and network HTML to `gold/fpgrowth/{cohort}/{age_band}/plots/` with naming convention `{cohort}_{age_band_fname}_train_{item_type}_*`.
- [ ] Lambda returns `itemsets_image`, `support_image`, `network_html` for selected `item_type`; frontend loads and displays (images and network via iframe or fetched HTML).
- [ ] Cohort and age band selectors drive request (same as other visualization tabs); item type selector (Drug Names, ICD Codes, CPT Codes, Medical Codes) selects which code type to visualize.
- [ ] Note in UI that FP-Growth is exploratory only (not used in model), per README_visualization_only.

---

## Tab 10: Documentation

**Purpose:** In-app help and usage instructions.

| Item | Detail |
|------|--------|
| **API** | None |
| **Data sources** | Static content in `index.html` (or loaded from a doc file). |
| **Frontend elements** | Text/sections, no dynamic visuals. |

### Implementation checklist

- [ ] Documentation tab content is up to date (overview, how to use risk, codes, visualization tabs).
- [ ] Links to external docs (e.g. VISUALIZATION_PLAN, backend README) work if present.

---

## Cross-Cutting: Image and HTML URLs

Visualization tabs that show **images** or **network HTML** from S3 currently receive **S3 object paths** (e.g. `s3://bucket/key`) or HTTP URLs. Browsers cannot load `s3://` directly. Ensure one of:

- **Presigned URLs:** Lambda or a separate endpoint returns short-lived HTTPS URLs for each image/HTML object.
- **Proxy:** Backend (or API Gateway + Lambda) proxies GET requests to S3 and returns the object body with correct Content-Type.
- **Public read:** S3 objects are public and frontend uses `https://bucket.s3.region.amazonaws.com/key` (not recommended for sensitive data).

Update this section when the chosen approach is implemented.

---

## Summary: Visualization API and S3 Paths

| Tab | API endpoint | Key S3/data paths | Filter by drug/ICD/CPT? |
|-----|--------------|-------------------|-------------------------|
| Causal Analysis | `GET /visualizations/causal?cohort=&age_band=[&drugs=&icds=&cpts=]` | `gold/ffa_analysis/`, `gold/shap_analysis/` | ✅ Optional `drugs`, `icds`, `cpts` (comma-separated) |
| BupaR | `GET /visualizations/bupar?cohort=&age_band=` | `gold/feature_importance/{cohort}/{age_band}/plots/` (BupaR PNGs) | ❌ No |
| DTW | `GET /visualizations/dtw?cohort=&age_band=` | `gold/feature_importance/.../plots/` (PNGs), `gold/feature_engineering/6_dtw/` (CSV) | ❌ No |
| FP-Growth | `GET /visualizations/fpgrowth?cohort=&age_band=&item_type=` | `gold/fpgrowth/{cohort}/{age_band}/plots/` | ✅ By cohort, age band, and item type (drug vs ICD vs CPT) |

---

**Last updated:** From dashboard and Lambda state as of implementation plan creation. Align with [VISUALIZATION_PLAN.md](VISUALIZATION_PLAN.md) and backend README for research-question mapping and response shapes.
