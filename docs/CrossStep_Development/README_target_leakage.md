# Target Leakage Prevention and Predictive Features

## Overview

This document outlines our approach to preventing **target leakage** in the final model and ensuring all features are **truly predictive** (i.e., available at prediction time without knowledge of the target outcome).

**Production pipeline:** Feature engineering for the final model **does not** build trajectory, sequence, or itemset columns. We only build **n_events**, **item_*** (drug/ICD/CPT from feature importance; **drug-only** for `non_opioid_ed`), **PGx counts** (e.g. pgx_num_drugs, pgx_num_cpic_drugs), and other schema features. `remove_target_leakage.py` / `run_final_model.py` still strip trajectory/sequence/itemset names **defensively** if those columns ever appear. FPGrowth and BupaR feed **dashboard visualizations** only; DTW supports **protocol filtering** and visuals, not the training matrix.

**Temporal validation:** Train on **2016–2018**, report metrics on **2019** holdout only; **2020** excluded (COVID). See `3a_feature_importance/README.md` and `6_final_model/README.md`.

## Cohort Target Dates (Step 2 → Step 4)

| Cohort | Target event | Cohort parquet column | `model_events` column |
|--------|--------------|----------------------|------------------------|
| `opioid_ed` | Opioid-related ED / F11.20 anchor | `first_opioid_ed_date` | `first_f1120_date` |
| `non_opioid_ed` | Earliest qualifying HCG ED (O11, P51b, P33) + drug–ED window | `first_ed_non_opioid_date` | `first_o11_p_date` |

**Index QA (Step 2):** `python 2_create_cohort/qa_index_date_uniqueness.py` verifies one target index date per `mi_person_key` per partition (prevents duplicate target rows that could inflate leakage checks). See `2_create_cohort/README.md` § Target Leakage and Downstream Modeling.

## Important Distinction: Feature Engineering vs. Final Model

**For Feature Engineering and Exploratory Analysis (BupaR / dashboards):**
- ✅ Target dates and target ICD codes (e.g. **F1120** for `opioid_ed`) **must** appear in event logs to:
  - Identify when the target event occurs
  - Split events into pre-target and post-target for process-mining analysis
  - Flag post-target codes in Step 3b (`LEAKAGE_ANALYSIS_SUMMARY.md`)

**For Final Model Training:**
- ❌ Target ICD codes (F1120, etc.) and target-date columns **must not** be model features
- ❌ **ALL events on or after the target date** are dropped for **cases** in Step 4 (`event_date < target_date`)
- ❌ **No post-event features** (`post_*`) or **time-to-target** features (`time_to_*`, `*_30d` before target, etc.)
- ✅ Production uses **`n_events`** (count of rows in `model_events` per patient) and **`n_event_bin_ordinal`** in gradient boosting—not legacy BupaR `pre_n_*` columns (exploratory only; not built in Step 6)

**Key principle:** At scoring time, the model must not require knowledge of whether/when the index ED (or F1120) will occur. Case event rows in `model_events` are strictly pre-target.

### Where Event-Level Leakage Is Removed (Pipeline)

1. **Step 2 (`2_create_cohort`):** Defines index date per target; optional QA via `qa_index_date_uniqueness.py`.
2. **Step 3b:** BupaR post-target analysis → drop leakage-prone items from `cohort_feature_importance.csv`.
3. **Step 4 (`4_model_data/create_model_data.py`):** For **cases**, `event_date <` target date (`first_f1120_date` or `first_o11_p_date` / source `first_ed_non_opioid_date`). Events on or after the target date are dropped.
4. **Step 6:** `remove_target_leakage.py` drops target-date columns, `post_*`, DTW, trajectory/itemset names; `prepare_train_test_s3.py` enforces 2016–2018 train / 2019 test.

### Case vs Control Event Windows in `model_events` (Step 4)

This asymmetry is **intentional** but affects interpretation of `n_events`:

| | Case (`target = 1`) | Control (`target = 0`) |
|--|---------------------|----------------------|
| Event time filter | `event_date <` target date | No target-date truncation (partition-year gold extract) |
| Item filter | Feature-importance items only | All items except 3b post-target code blacklist |
| `n_events` in Step 6 | Pre-target, FI-filtered row count | Full extract row count per patient |

Utilization-density bins and partition-first training mitigate control–case volume differences. See `4_model_data/README_model_data.md` § Case vs control event windows.

## Post-Target Audit (Step 3b): Feature Importances vs. Target Population

Step 3a permutation / MC-CV feature importance (`3a_feature_importance/run_mc_feature_importance.py`) ranks **item-level** candidates (`item_drug_*`, `item_icd_*`, `item_cpt_*`). High importance alone does not prove a code is usable at prediction time. Step 3b **audits each candidate against event timing in the target case population** before any code enters `cohort_feature_importance.csv`.

### Workflow (3b)

| Step | Script / output | Role |
|------|-----------------|------|
| 1 | Step 3a → `{cohort}_{age_band}_aggregated_feature_importance.csv` | Initial ranked feature list |
| 2 | `3b_feature_importance_eda/1_bupaR/create_bupar_post_target_analysis.py` → `{cohort}_{age_band}_bupar_post_target_analysis.csv` | **Pre/post index timing audit on `target = 1` only** |
| 3 | `2_filtering/create_safe_feature_filter_json.py` → `{cohort}_{age_band}_safe_feature_filter.json` | Whitelist: exclude ≥80% post-index; keep any meaningful pre-index presence |
| 4 | `2_filtering/filter_and_refine_features.py` → `{cohort}_{age_band}_cohort_feature_importance.csv` | Refined FI for Step 4 / 6 |

Orchestration: `3b_feature_importance_eda/run_feature_importance_eda.py`. See `3b_feature_importance_eda/README_feature_importance_eda.md`, `EXECUTION_ORDER.md`, `FEATURE_FILTERING_APPROACH.md`.

### Timing audit logic (`analyze_post_target_leakage_from_events`)

For each **target patient** (`target = 1`):

1. Resolve **index date** (`first_ed_non_opioid_date` / F1120 anchor from cohort or `model_events`).
2. For every event carrying a candidate code (drug; or ICD/CPT for `opioid_ed`), label **`pre`** if `event_date < index`, else **`post`**.
3. Per feature (`item_*` name), compute `pre_count`, `post_count`, `post_target_ratio`, `pre_target_ratio`.
4. Flag **`is_post_target_leakage = 1`** when `post_target_ratio ≥ 0.8` (default) and `total_count ≥ 5`.
5. Flag **`is_pre_target_predictive = 1`** when `pre_target_ratio ≥ 0.8`.

Example opioid_ed finding: **348 / 1,029** features were ≥80% post-F1120 (treatment meds, drug screens, repeat F1120, etc.) — documented in `3b_feature_importance_eda/LEAKAGE_ANALYSIS_SUMMARY.md`. Those codes are removed from the refined feature list; they are **not** leakage that survives into production `item_*` columns.

`non_opioid_ed` runs the same logic on **drugs only**; polypharmacy `model_events` for cases is often pre-index by construction, but the audit still runs before refinement.

### Step 6 re-check

`6_final_model/remove_target_leakage.py` validates each retained **`item_*`** against `model_events`: if any target patient has that code **on or after** the index date, the column is dropped. This is a second guard after Step 3b.

### Audit conclusion: `n_events` is not item-level leakage

| Signal | Step 3b pre/post audit? | In production GBT matrix? |
|--------|-------------------------|---------------------------|
| `item_*` drug / ICD / CPT | Yes | Yes (refined whitelist only) |
| Post-index treatment / monitoring codes | Flagged and removed | No |
| **`n_events`** (row count in `model_events`) | **No** (not an `item_*` in the ratio loop) | Built in Step 6; trees use **`n_event_bin_ordinal`** |

After the audit pipeline, **no item-level feature with persistent post-index dominance remains in the training matrix**. The main **residual confound** is **utilization volume** (`n_events` → density bins), which can reflect true healthcare intensity **or**:

- **Imperfect VHI de-identified linkage** (`mi_person_key`): duplicate or fragmented member records across payers/time inflating counts.
- A small subset of **high-utilization repeat presenters** (frequent ED/claims) increasing pre-index row counts without a single leaked drug code.

That is **documentation / linkage confounding**, not target-date leakage from a specific `item_*`. Mitigation: utilization-density strata, partition-first training, and explicit Limitations (CH_4); exploratory unstratified SHAP may still rank raw `n_events` highly—production does not feed raw counts to trees.

---

## What is Target Leakage?

**Target leakage** occurs when features include information that would not be available at the time of prediction. These features artificially inflate model performance but fail in real-world deployment because they require knowledge of future outcomes.

### Common Types of Target Leakage

1. **Post-event features**: Features calculated AFTER the target event has occurred
2. **Time-to-target features**: Features that reference the target event date/time
3. **Target-referenced time windows**: Event counts within X days before the target event

---

## Removed Features (Target Leakage)

### 1. Post-Event Features ❌

**Removed:**
- `post_n_events` - Number of events after target event
- `post_n_drug_events` - Number of drug events after target event
- `post_n_icd_events` - Number of ICD events after target event
- `post_n_cpt_events` - Number of CPT events after target event
- `post_n_unique_activities` - Unique activities after target event

**Why removed:** These features are calculated using events that occur AFTER the target event (e.g., F1120 opioid-related ED visit). At prediction time, we don't know if/when the target event will occur, so these features are not available.

**Source:** `5_bupaR_analysis/create_bupar_outputs_opioid_ed.R` (post-F1120 features)

---

### 2. Time-to-Target Features ❌

**Removed:**
- `time_to_F1120_days` - Days from first event to target event
- `target_time` - Timestamp of target event
- `first_time` - Timestamp of first event (used in time-to-target calculations)

**Why removed:** These features explicitly reference the target event date. At prediction time, we don't know when (or if) the target event will occur.

**Source:** `5_bupaR_analysis/create_bupar_outputs_opioid_ed.R` (time-to-F1120 features)

---

### 3. Target-Referenced Time Windows ❌

**Removed:**
- `n_events_30d` - Events in 30 days before target
- `n_events_90d` - Events in 90 days before target
- `n_events_180d` - Events in 180 days before target
- `n_drug_events_30d` - Drug events in 30 days before target
- `n_drug_events_90d` - Drug events in 90 days before target
- `n_drug_events_180d` - Drug events in 180 days before target
- `n_icd_events_30d` - ICD events in 30 days before target
- `n_icd_events_90d` - ICD events in 90 days before target
- `n_icd_events_180d` - ICD events in 180 days before target
- `n_cpt_events_30d` - CPT events in 30 days before target
- `n_cpt_events_90d` - CPT events in 90 days before target
- `n_cpt_events_180d` - CPT events in 180 days before target

**Why removed:** These features count events within a fixed time window BEFORE the target event. To calculate these, we need to know when the target event occurs, which is not available at prediction time.

**Source:** `5_bupaR_analysis/create_bupar_outputs_opioid_ed.R` (time-windowed counts)

---

### 4. DTW Distance Features ❌

**Removed:**
- `combined_dtw_distance_to_prototype_0` through `combined_dtw_distance_to_prototype_4`
- `combined_dtw_min_distance`
- `combined_dtw_max_distance`
- `combined_dtw_mean_distance`
- `combined_dtw_std_distance`
- `combined_trajectory_length` (if calculated using target event as endpoint)
- `combined_trajectory_diversity` (if calculated using target event as endpoint)

**Why removed:** The original DTW implementation calculated distances to prototype trajectories that were derived from target patients' trajectories ending at the target event. This creates leakage because:
1. Prototypes are based on target event timing
2. Trajectories are truncated at the target event
3. Control patients don't have target events, so DTW features can't be calculated consistently

**Source:** `6_dtw_analysis/create_dtw_features.py`

---

## Kept Features (Predictive)

### 1. Production utilization features ✅ (`n_events` / `n_event_bin_ordinal`)

**Built in Step 6 (`build_final_cohort_model_features.py`):**
- `n_events` — `COUNT(*)` of rows in `model_events.parquet` per `mi_person_key` (cases: pre-target, FI-filtered events from Step 4)
- `n_event_bin` / `n_event_bin_ordinal` — utilization-density strata (P25/P50/P95 on training cases); **ordinal bin is what gradient boosting uses**; raw `n_events` is dropped before training

**Not used in production:** Legacy BupaR `pre_n_*` columns (`pre_n_events`, `pre_n_drug_events`, …). Those require the target date to define “pre” and appear only in exploratory BupaR exports (`5_bupaR_analysis/`). Step 6 does not merge them into `final_features.parquet`.

**Source:** `6_final_model/build_final_cohort_model_features.py`; density bins — `docs/CrossStep_Development/README_event_density_bins.md`

---

### 2. Time Interval Features ✅

**Kept:**
- `drug_interval_mean` - Mean time (days) between consecutive drug events
- `drug_interval_median` - Median time between consecutive drug events
- `drug_interval_std` - Standard deviation of drug event intervals
- `drug_interval_min` - Minimum interval between drug events
- `drug_interval_max` - Maximum interval between drug events
- `drug_interval_count` - Number of intervals (n_drug_events - 1)
- `icd_interval_mean`, `icd_interval_median`, `icd_interval_std`, `icd_interval_min`, `icd_interval_max`, `icd_interval_count`
- `cpt_interval_mean`, `cpt_interval_median`, `cpt_interval_std`, `cpt_interval_min`, `cpt_interval_max`, `cpt_interval_count`

**Why kept:** These features measure the **time intervals between consecutive events** of the same type. They are:
- **Predictive**: Can be calculated from any sequence of events without knowledge of target event
- **Available for controls**: Control patients have events, so intervals can be calculated
- **Temporally meaningful**: Capture patterns in event frequency and regularity
- **Key time window info**: Essential for understanding temporal patterns between events

**Example:**
- Patient A: Drug events on days 0, 5, 12, 20 → intervals: [5, 7, 8] → mean = 6.67 days
- Patient B: Drug events on days 0, 30, 60, 90 → intervals: [30, 30, 30] → mean = 30 days

**Source:** `6_dtw_analysis/create_predictive_time_features.py`

---

### 2b. Sequence Features

**In the current pipeline:** Feature engineering **does not produce** sequence (or trajectory/itemset) columns for the final model. BupaR is used for dashboard visualizations only. The following would be kept *if* we built them (we do not):

- `is_top_sequence`, `is_rare_sequence`, sequence frequency features (from pre-F1120 events only)

**Why they would be predictive if used:** Based on pre-target event sequences; available for controls with a reference date. See `5_bupaR_analysis/` for BupaR outputs (visualization only).

---

### 3. FP-Growth Features

**In the current pipeline:** Feature engineering **does not produce** itemset (or trajectory/sequence) columns for the final model. FP-Growth is used for dashboard visualizations only. The following would be kept *if* we built them (we do not):

- Itemset match/support features, rule match/confidence/lift features, summary counts

**Why they would be predictive if used:** Based on pre-F1120 co-occurrence patterns; available for controls. See `10_risk_dashboard/visualizations/fpgrowth/` for FP-Growth outputs (visualization only).

---

### 4. PGx Features ✅

**Kept:**
- All pharmacogenomic features (allele frequencies, risk scores)
- Gene coverage and drug mapping counts

**Why kept:** PGx features are based on:
- **Patient genetics** (static, available at any time)
- **Medication claims events** (historical pharmacy/medical lines before prediction; not dose/PK exposure)
- **Population allele frequencies** (reference data, not patient-specific)

**Source:** `5_pgx_analysis/create_pgx_features_patient_level.py`

---

## Feature Engineering Approach

### For Target Patients (current pipeline)

1. **Pre-event / count features**: n_events and related counts from events before target event date
2. **Item features**: item_* (drug/ICD/CPT) from feature importance
3. **PGx**: pgx_num_drugs, pgx_num_cpic_drugs; n_drugs from PGx step
4. We **do not** build trajectory, sequence, or itemset features for the model; FPGrowth/BupaR are visualization-only

### For Control Patients

Same methodology with a **reference date** (e.g., first event, cohort entry) instead of target event date.

**Key principle:** Control patients should have features calculated using the **same methodology** as target patients, just with a different reference point (not target event date).

---

## Implementation Details

### Script: `6_final_model/remove_target_leakage.py`

This script identifies and removes target leakage features from the final feature table:

```python
# Removed feature categories:
1. Post-event features (post_*)
2. Time-to-target features (time_to_*, target_time, first_time)
3. Target-referenced time windows (*_30d, *_90d, *_180d, excluding intervals)
4. DTW distance features (dtw_distance_to_prototype_*)
```

### Script: `6_dtw_analysis/create_predictive_time_features.py`

This script creates predictive time interval features:

```python
# Features created:
- drug_interval_* (mean, median, std, min, max, count)
- icd_interval_* (mean, median, std, min, max, count)
- cpt_interval_* (mean, median, std, min, max, count)
```

### Script: `6_final_model/build_final_cohort_model_features.py`

This script builds the final feature table. Feature engineering **never generates** trajectory, sequence, or itemset columns.

```python
# Features actually produced:
- n_events, item_* (drug/ICD/CPT from feature importance), PGx counts, demographics, etc.

# NOT produced (and removed defensively if ever present):
- Trajectory/sequence/itemset features (we do not build these)
- Post-event features (post_*)
- Time-to-target features (time_to_*)
- Target-referenced time windows (*_30d, *_90d, *_180d)
- DTW distance features (dtw_distance_to_prototype_*)
- F1120 code itself (excluded from all feature calculations)
```

---

## Validation Checklist

Before training the final model, verify:

- [ ] No features reference `target_time` or `first_time`
- [ ] No features use `post_*` prefix
- [ ] No features use `time_to_*` prefix
- [ ] No time-window features (`*_30d`, `*_90d`, `*_180d`) except intervals
- [ ] No DTW distance features to prototypes
- [ ] All features can be calculated for control patients
- [ ] All features use the same calculation method for targets and controls
- [ ] Time intervals are calculated between consecutive events (not to target)

---

## Future Improvements

### 1. Replace Pre-Event Features with Fixed Lookback Windows

**Current:** `pre_n_events` = events before target event  
**Proposed:** `n_events_12m` = events in 12 months before reference date

This makes features truly predictive by using a fixed lookback period from a reference date (e.g., cohort entry, first event) rather than the target event date.

### 2. Add Rolling Window Features

For real-time prediction, calculate features using rolling windows:
- `n_events_last_30d` - Events in last 30 days from current date
- `n_events_last_90d` - Events in last 90 days from current date
- `n_events_last_180d` - Events in last 180 days from current date

### 3. Temporal Sequence Features

Add features that capture temporal patterns without referencing target:
- **Event sequence patterns**: Common sequences of drug → ICD → CPT
- **Temporal density**: Events per month over observation period
- **Sequence diversity**: Number of unique event sequences

---

## References

- **Target Leakage Detection**: `6_final_model/remove_target_leakage.py`
- **Predictive Time Features**: `6_dtw_analysis/create_predictive_time_features.py`
- **Feature Table Building**: `6_final_model/build_final_cohort_model_features.py`
- **Removed Features List**: `6_final_model/outputs/{cohort}/{age_band}/{cohort}_{age_band}_removed_leakage_features.txt`

---

**Last Updated:** June 2, 2026 (Step 3b post-target audit vs. target population; `n_events`/VHI linkage conclusion)

