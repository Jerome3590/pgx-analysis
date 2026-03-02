# Step 9: Dashboard Visualizations

## Overview

The dashboard includes three advanced visualization systems that complement the risk score by providing insights into patient pathways, frequent patterns, and trajectory similarities. All visualizations are filtered based on user-selected codes (drugs, ICDs, CPTs).

## Risk Calculation

The risk score is the **estimated probability of the target outcome** (e.g. opioid-related ED or polypharmacy) in the **2019 holdout population** context. It is computed by the backend API (`POST /risk`) and drives the risk band and interpretation shown in the dashboard.

### When baseline vs when model

| User input | Result |
|------------|--------|
| **No** Drug, ICD, or CPT codes | **Baseline risk:** the actual 2019 outcome rate for that cohort/age_band (from `risk_distribution_2019.json`). The UI shows “Baseline risk (2019 holdout population). Add Drug, ICD, or CPT codes to see personalized risk.” |
| **Any** Drug, ICD, or CPT code provided | **Model risk:** the predicted probability from the **best model** for that cohort/age_band (single model with weight 1.0). The UI shows “Risk from &lt;model_used&gt; (best model for this cohort/age).” |

So risk stays **calibrated to the 2019 population**: with no codes you see the population rate; with codes you see the model’s personalized estimate in that same context.

### Best model per cohort/age_band

For each (cohort, age_band) we use **one** best model (CatBoost, XGBoost, or XGBoost RF), chosen by composite score (PR-AUC + normalized log loss) in MC-CV. The API runs only that model and returns `model_used` (e.g. `"xgboost"`) when the score is from the model (not baseline).

### Risk bands (Low / Medium / High)

Risk bands are derived from **configurable thresholds**:

- **Source:** `risk_distribution_2019.json` per cohort/age_band can include `risk_band_thresholds` (e.g. 33rd and 67th percentiles of 2019 predicted probabilities: `low_medium`, `medium_high`).
- **Logic:** Score &lt; low_medium → Low; low_medium ≤ score &lt; medium_high → Medium; score ≥ medium_high → High. If thresholds are missing, the API falls back to default values (e.g. 0.2, 0.5).

### Comparison (scenarios)

`POST /risk/comparison` compares a **base** scenario to one or more **scenarios**. The same rule applies: if the base or a scenario has **no** Drug/ICD/CPT codes, that scenario’s score is the **baseline_risk** for that cohort/age_band; otherwise it is the model’s predicted probability.

### Code validation and interpretation

- **codes_used / codes_unknown:** The API validates provided drugs, ICDs, and CPTs against the model’s feature schema. It returns which codes were **used** by the model (affected the score) and which were **unknown** (not in the model; did not affect the score). The frontend can show “Not in model (did not affect score): …” for unknown codes.
- **interpretation:** The API returns a short **interpretation** string (e.g. “Estimated probability of target outcome (2019 holdout context)…”; when not baseline, “Risk from &lt;model_used&gt; (best model for this cohort/age). …”). The dashboard displays this in the Risk Assessment tab.

### Data flow: risk → visuals

1. User selects cohort, age band, and optionally drugs/ICDs/CPTs.
2. **Risk** is calculated (baseline or model) and shown with band, interpretation, and model/code info.
3. **Visualizations** (BupaR, FP-Growth, DTW, Causal) load for the same cohort/age_band; where applicable they are filtered or contextualized by the user’s selected codes, so risk and visuals stay aligned.

For API details (request/response shapes, endpoints), see `10_risk_dashboard/backend/README.md` and `10_risk_dashboard/data_preparation/README.md` (risk distribution and model preparation).

### Risk calculation robustness

- **Baseline when no codes:** If the user sends no Drug/ICD/CPT codes and `risk_distribution_2019.json` has `baseline_risk`, the API returns that value and does not load models. If the file is missing or has no `baseline_risk`, the API falls back to running the best model with an empty code set (all item features 0), so a score is still returned.
- **Best model only:** Only the single best model per cohort/age_band is run (weight 1.0). If that model fails to load or predict, the request returns 500 with a clear error (no silent fallback to a different model).
- **Input handling:** The API normalizes `drugs`/`icds`/`cpts` to lists (handles `null` or missing keys). Invalid JSON body returns 400 with a clear message. Cohort and age_band are validated via `determine_cohort_and_age_band` when inferred from age; dashboard-style requests use explicit cohort/age_band.
- **Feature schema and codes:** Unknown codes (not in the model’s feature schema) do not affect the score; the API returns `codes_used` and `codes_unknown` so the UI can show what was used. Missing or empty feature schema is handled (default empty features, defaults applied where defined).
- **Risk bands:** If `risk_band_thresholds` are missing from the 2019 distribution, the API uses default thresholds (e.g. 0.2, 0.5) so a band is always returned.
- **Comparison:** Same baseline-vs-model rule and input normalization apply to `POST /risk/comparison`; base and each scenario are evaluated consistently.
- **Limitations:** No retry or fallback if the best model fails. No allowlist for cohort/age_band strings when provided explicitly (assume dashboard sends valid values). Probability outputs are clamped to [0, 1] for XGBoost raw outputs; model weights are normalized so zero weights fall back to equal weights or simple average.

### Is the risk calculation approach correct?

- **Target and train/test:** Models are trained on 2016–2018 data with a temporal holdout; the **2019 test set is never used for training**. The target is the same binary outcome (e.g. opioid-related ED or polypharmacy) that the dashboard describes. So the quantity being predicted is the right one.
- **Baseline (no codes):** When the user supplies no Drug/ICD/CPT codes, we return **baseline_risk** = mean(target) in the **2019 test set** for that cohort/age_band. That is the **unconditional** outcome rate in the holdout population. Conceptually this is correct: “average risk in this group when we don’t condition on any codes.”
- **With codes:** When the user supplies codes, we return the **model’s predicted probability** P(outcome | features) from the best model for that cohort/age_band. The model was trained on 2016–2018 and applied to the same feature schema; the number is an **estimated probability of the same outcome** in the same cohort/age_band. So we are comparing like with like (unconditional 2019 rate vs model-based conditional probability).
- **No target leakage:** The outcome code (e.g. F1120 for opioid ED) is **excluded from inputs** in the Lambda feature builder and in code validation, so the model never sees the target as a feature. Correct.
- **Feature alignment:** For the final model we **do not calculate** trajectory, sequence, or itemset features in feature engineering. We only build: **# events** (n_events), **# drugs / # CPIC drugs** (from PGx, e.g. pgx_num_drugs, pgx_num_cpic_drugs), and **item_*** binary indicators (drug/ICD/CPT from aggregated feature importance). BupaR, FP-Growth, and DTW are used for dashboard visualizations only, not for model training. At inference the dashboard sends only **cohort, age_band, and selected drugs/ICDs/CPTs** (and optionally age). The API does **not** require n_events, pgx_num_drugs, or pgx_num_cpic_drugs as inputs: those are filled from **schema defaults** (training medians) so the model gets consistent values without the user having to supply them.
- **Calibration:** The model is **not** recalibrated on 2019. So absolute probabilities when the user adds codes may not be perfectly calibrated to 2019 rates; relative ordering and risk bands should still be meaningful. If needed, a future step could add 2019-based recalibration (e.g. Platt scaling or isotonic regression on the 2019 test set).
- **Best model:** The single best model per cohort/age_band is chosen by a composite score (e.g. PR-AUC + normalized log loss) on the same MC-CV splits used for model selection. Using that one model for scoring is consistent with the pipeline and avoids mixing in weaker models.

**Summary:** The approach is **methodologically correct**: same outcome, no leakage (including no trajectory/sequence/itemset in training), sensible baseline, and conditional risk from the chosen model. Remaining caveat: no explicit 2019 recalibration.

## Visualization Types

### 1. BupaR Process Mining

**Purpose**: Analyze patient pathways and sequences

**Visualizations**:
- **Sankey Diagrams**: Flow diagrams showing transitions between activities (drugs, ICDs, CPTs)
- **Process Matrices**: Transition frequency matrices showing common pathways
- **Trace Frequency Charts**: Bar charts of most frequent patient sequences

**Data Source**: `s3://pgxdatalake/gold/bupar/{cohort}/{age_band}/`

**Filtering**: Shows only pathways containing user-selected codes

**Example Use Cases**:
- "What pathways do patients with Drug X typically follow?"
- "How do patients progress from initial diagnosis to outcome?"
- "What are the most common sequences involving these codes?"

### 2. FP-Growth Frequent Patterns

**Purpose**: Discover association rules and frequent itemsets

**Visualizations**:
- **Association Rules Network**: Sankey diagram showing rules between codes
  - Nodes: Individual codes (drugs, ICDs, CPTs)
  - Edges: Association rules (antecedents → consequents)
  - Edge width: Confidence of the rule
- **Frequent Itemsets Bar Chart**: Top frequent code combinations

**Data Source**: `s3://pgxdatalake/gold/fpgrowth/cohort/cohort_name={cohort}/age_band={age_band}/`

**Filtering**: Shows only rules/itemsets containing user-selected codes

**Example Use Cases**:
- "What codes are frequently associated with Drug X?"
- "What patterns predict high risk?"
- "What are the most common code combinations?"

### 3. DTW Trajectory Clusters

**Purpose**: Analyze patient trajectory similarity and clustering

**Visualizations**:
- **Cluster Size Distribution**: Bar chart showing number of patients per cluster
- **Average DTW Distance**: Line chart showing average distance within clusters
- **Patient Trajectory Timelines**: Timeline visualization of representative trajectories

**Data Source**: `s3://pgxdatalake/gold/dtw_trajectories/{cohort}/{age_band}/`

**Filtering**: Shows only clusters containing trajectories with user-selected codes

**Example Use Cases**:
- "Which trajectory clusters contain patients with these codes?"
- "How similar are patient trajectories?"
- "What are the representative patterns for this patient group?"

## Integration with Risk Score

See **[Risk Calculation](#risk-calculation)** above for how the score is computed (baseline vs model, best model, bands, comparison, and code validation). All visualizations complement the risk score by:

1. **Contextualizing Risk**: Showing how selected codes appear in patient pathways
2. **Pattern Discovery**: Revealing associations and sequences involving selected codes
3. **Trajectory Analysis**: Identifying which patient clusters contain similar patterns
4. **Causal Insights**: Supporting causal analysis by showing pathway impacts

## Data Flow

```
User Selects Codes → Risk Score Calculated → Visualizations Load
                                              ↓
                    Filter by Selected Codes → Display Filtered Visualizations
```

## Technical Implementation

### Frontend

- **Library**: Plotly.js (v2.27.0)
- **Rendering**: Dynamic chart generation based on API responses
- **Filtering**: Client-side filtering of visualization data
- **Lazy Loading**: Visualizations load when Tab 3 is opened

### Backend

- **Endpoints**: `GET /visualizations/{type}` (bupar, fpgrowth, dtw)
- **Filtering**: Server-side filtering based on selected codes
- **Data Sources**: S3 buckets for each visualization type
- **Error Handling**: Graceful degradation if data unavailable

### Data Filtering Logic

1. **BupaR**: Filter process matrices/traces containing selected codes
2. **FP-Growth**: Filter rules/itemsets intersecting with selected codes
3. **DTW**: Filter clusters containing trajectories with selected codes

## Related Documentation

- **[README_results_dashboard_tabs.md](README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard.md](README_results_dashboard.md)** - Complete dashboard system overview
- **Risk calculation (backend):** `10_risk_dashboard/backend/README.md` - API endpoints, baseline vs model, best model, comparison
- **Risk data preparation:** `10_risk_dashboard/data_preparation/README.md` - risk_distribution_2019 (baseline_risk, risk_band_thresholds), prepare_models (best model per cohort/age_band)
- **[README_bupar_dashboard_visualizations.md](README_bupar_dashboard_visualizations.md)** - BupaR process mining dashboard visualizations
- **[README_fpgrowth_dashboard_visualizations.md](README_fpgrowth_dashboard_visualizations.md)** - FP-Growth pattern mining dashboard visualizations
- **[README_dtw_dashboard_visualizations.md](README_dtw_dashboard_visualizations.md)** - DTW trajectory dashboard visualizations
- **Visualization Implementation**: See `10_risk_dashboard/visualizations/` for BupaR, FP-Growth, and DTW visualization generation scripts
- **Feature Refinement**: See `3b_feature_importance_eda/` for BupaR post-target analysis used in Feature Importance EDA

