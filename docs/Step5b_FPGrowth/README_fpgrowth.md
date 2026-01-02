# FP-Growth Analysis (Detailed Documentation)

This document provides detailed documentation for the FP-Growth analysis pipeline used in `pgx-analysis`. A shorter summary and output manifest live in `5_fpgrowth_analysis/README_fpgrowth.md`.

---

## Goals

- Discover frequent co-occurring patterns of clinical items (drugs, ICD codes, CPT codes) within each cohort / age-band.
- Derive association rules that describe directional relationships between clinical events.
- Produce patient-level features capturing membership in high-value itemsets and rules, for use in downstream models.

The analysis is run per `(cohort, age_band, split_type, event_year)` combination and writes JSON outputs plus visualizations under `5_fpgrowth_analysis/outputs/`.

---

## Inputs

- **Model data** from `4a_model_data`:
  - **Preferred**: `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet` (DTW-filtered, protocol events removed)
  - **Fallback**: `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (if DTW filter not run)
  - Contains event-level data with `mi_person_key`, `event_date`, `drug_name`, ICD diagnosis columns, `procedure_code`, and `target` label.
  - **Note**: FP-Growth scripts automatically prefer `model_events_no_protocols.parquet` if available. This ensures itemsets and association rules only capture useful signals (non-protocol events), improving the quality of discovered patterns.
- **Configuration** in `5_fpgrowth_analysis/cohort_fpgrowth.py` and `global_fpgrowth.py`:
  - `MIN_SUPPORT`, `MIN_CONFIDENCE`, `MIN_ITEMSET_LIFT`
  - `ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code', 'medical_code']`
  - `DENSITY_BINS = ['low', 'medium', 'high', 'extreme']` for transaction size stratification.

---

## Core Scripts

- `5_fpgrowth_analysis/cohort_fpgrowth.py`
  - Runs FP-Growth per cohort, age band, item type, and year.
  - **Automatically prefers DTW-filtered data** (`model_events_no_protocols.parquet`) if available, ensuring itemsets and rules only capture useful signals (non-protocol events).
  - Reads events from `4a_model_data` (via DuckDB) and builds patient-level transactions.
  - Bins patients by transaction density (low/medium/high/extreme) to control memory usage.
  - Runs FP-Growth separately per density bin and merges results.
  - Filters itemsets by lift (`MIN_ITEMSET_LIFT`) to drop trivial/common patterns.
  - Generates association rules and saves:
    - `{item_type}_itemsets.json`
    - `{item_type}_rules.json`
    - `{item_type}_metrics.json`
    - `{item_type}_encoding_map.json`

- `5_fpgrowth_analysis/global_fpgrowth.py`
  - Optional global analysis across cohorts (if needed).
  - Same output structure, but aggregating across broader populations.

- `4_fpgrowth_analysis/create_fpgrowth_features.py`
  - Loads JSON itemsets/rules and model_data.
  - Builds per-patient feature matrix:
    - `_match` indicators for top N itemsets/rules per item type.
    - Counts and summary metrics (e.g., `*_itemsets_matched_count`, `*_itemsets_max_support`).
  - Writes `fpgrowth_features_{cohort}_{age_band}.csv` under `4_fpgrowth_analysis/outputs/feature_engineering/`.

- `4_fpgrowth_analysis/add_fpgrowth_features_to_model_data.py`
  - Performs final aggregation / reshaping of FP-Growth features.
  - Writes `fpgrowth_added_features_{cohort}_{age_band}.csv` for merging into the final model dataset.

---

## Output Structure

See `5_fpgrowth_analysis/README_fpgrowth.md` for the full manifest. In brief:

- **JSON data files** under `5_fpgrowth_analysis/outputs/{cohort}/{split_type}/{age_band}/{year}/`:
  - `{item_type}_itemsets.json`
  - `{item_type}_rules.json`
  - `{item_type}_metrics.json`
  - `{item_type}_encoding_map.json`
- **Plots** under `5_fpgrowth_analysis/outputs/{cohort}/{age_band}/plots/` and mirrored to S3:
  - Top itemsets, support histograms, size distributions, support vs size.
  - HTML network visualizations for co-occurrence and rules.
- **Feature engineering artifacts** under `4_fpgrowth_analysis/outputs/feature_engineering/`:
  - `fpgrowth_features_{cohort}_{age_band}.csv`
  - `fpgrowth_added_features_{cohort}_{age_band}.csv`

All patient-level feature files include `mi_person_key` for joining with other feature blocks and the final model dataset.

---

## Design Decisions

### Transaction Density Binning

Event-level data can be extremely dense for some patients. To reduce memory pressure and avoid bias from very complex trajectories:

- Patients are assigned a `Transaction_Density` bin based on transaction size percentiles:
  - `low`, `medium`, `high`, `extreme`
- FP-Growth is run separately within each bin.
- Itemsets are then merged across bins, de-duplicated, and filtered by lift.

This allows the analysis to capture patterns among both simple and complex histories without being dominated by a small set of high-utilizers.

### Lift-Based Itemset Filtering

Support alone often yields many uninteresting itemsets (e.g., common codes that appear everywhere). To focus on meaningful patterns, itemsets are filtered by lift:

- **Lift** ≈ `support(itemset) / product(support(item_i))`
- Itemsets with lift close to 1 are near-independent (not interesting).
- We enforce `lift >= MIN_ITEMSET_LIFT` (e.g., 1.1) before rule generation.

This significantly reduces noise and improves interpretability of both itemsets and downstream rules.

### Target-Focused Rules (Optional)

In some configurations, rules are constrained to be target-focused:

- Antecedent: clinical pattern (drugs, ICD, CPT).
- Consequent: target indicator (e.g., opioid dependence or ED visit category).

This keeps rule sets small and directly relevant to risk modeling.

---

## Running the Analysis

From the project root:

1. Ensure `4a_model_data` exists for the desired `(cohort, age_band)`:
   - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`

2. Run cohort-level FP-Growth for a single cohort/age-band/year using a helper such as `5_fpgrowth_analysis/run_single_cohort_fpgrowth.py` (if present) or by calling `process_single_cohort` from `cohort_fpgrowth.py` in a small driver.

3. Generate patient-level features:
   - `python 4_fpgrowth_analysis/create_fpgrowth_features.py --cohort {cohort} --age_band {age_band}`

4. Aggregate to final FP-Growth feature file:
   - `python 4_fpgrowth_analysis/add_fpgrowth_features_to_model_data.py --cohort-name {cohort} --age-band {age_band}`

This produces `fpgrowth_added_features_{cohort}_{age_band}.csv`, ready to merge with BupaR, DTW, and PGx feature blocks in the final model pipeline.

