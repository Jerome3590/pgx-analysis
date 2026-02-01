# Target Leakage Prevention and Predictive Features

## Overview

This document outlines our approach to preventing **target leakage** in the final model and ensuring all features are **truly predictive** (i.e., available at prediction time without knowledge of the target outcome).

## Important Distinction: Feature Engineering vs. Final Model

**For Feature Engineering and Exploratory Analysis:**
- ✅ **F1120 MUST be included** in the data to:
  - Identify when the target event occurs
  - Split events into pre-F1120 and post-F1120 for analysis
  - Understand patterns leading up to F1120
  - Create pre-event and post-event features for exploration

**For Final Model Training:**
- ❌ **F1120 MUST be excluded** from all features
- ❌ **ALL events AFTER F1120 MUST be excluded** (everything past the target code)
- ✅ **Only events BEFORE F1120** (not including F1120) are used
- ❌ **No post-event features** are included
- ❌ **No time-to-target features** are included

**Key Principle:** For final model training, we exclude everything past F1120. Only events that occur BEFORE the target code are used for feature calculation.

This ensures that at prediction time, we don't need to know if/when F1120 will occur.

### Where Event-Level Leakage Is Removed (Pipeline)

Step 4 (model data) removes target leakage when building `model_events.parquet`: for **case events**, only events **strictly before** the target date are kept (`event_date < first_opioid_ed_date` or `first_ed_non_opioid_date`). Events on or after the target date are dropped. Implemented in **Step 4** (`4_model_data/create_model_data.py`). Step 3b (e.g. BupaR post-target analysis) identifies leakage; Step 4 applies the filter when constructing model data.

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

### 1. Pre-Event Features ✅

**Kept:**
- `pre_n_events` - Number of events before target event
- `pre_n_drug_events` - Number of drug events before target event
- `pre_n_icd_events` - Number of ICD events before target event
- `pre_n_cpt_events` - Number of CPT events before target event
- `pre_n_unique_activities` - Unique activities before target event

**Why kept:** These features are calculated using events that occur BEFORE the target event. However, **note**: These features still require knowledge of the target event date to define the "pre" period. For a fully predictive model, we should use:
- **Fixed lookback windows** from a reference date (e.g., first event, cohort entry date)
- **Rolling windows** from the current date (for real-time prediction)

**Current status:** Kept for initial model, but should be replaced with fixed lookback windows for production.

**Source:** `5_bupaR_analysis/create_bupar_outputs_opioid_ed.R` (pre-F1120 features)

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

### 2b. Sequence Features ✅

**Kept:**
- `is_top_sequence` - Binary indicator if patient has a top-frequency sequence (from pre-F1120 events)
- `is_rare_sequence` - Binary indicator if patient has a rare sequence (from pre-F1120 events)
- Sequence frequency features (if calculated from pre-F1120 events only)

**Why kept:** These features capture **important sequence patterns** from events BEFORE F1120:
- **Predictive**: Based on pre-target event sequences
- **Pattern-based**: Capture common and rare event sequences leading up to target
- **Available for controls**: Can be calculated for control patients using reference dates

**Source:** `5_bupaR_analysis/create_sequence_features.py`, `5_bupaR_analysis/add_bupar_features_to_model_data.py`

---

### 3. FP-Growth Features ✅

**Kept:**
- **Itemset features**: Binary indicators for top N itemsets (e.g., `drug_name_itemset_0_match`, `icd_code_itemset_5_match`)
- **Itemset support scores**: Support values for matched itemsets (e.g., `drug_name_itemset_0_support`)
- **Rule features**: Binary indicators for top N association rules (e.g., `drug_name_rule_0_match`)
- **Rule confidence/lift scores**: Confidence and lift values for matched rules (e.g., `drug_name_rule_0_confidence`, `drug_name_rule_0_lift`)
- **Summary features**: 
  - `*_itemsets_matched_count` - Count of matched itemsets per patient
  - `*_itemsets_max_support` - Maximum support among matched itemsets
  - `*_rules_matched_count` - Count of matched rules per patient
  - `*_rules_max_confidence` - Maximum confidence among matched rules
  - `*_rules_max_lift` - Maximum lift among matched rules

**Why kept:** FP-Growth features capture **important rules and itemsets** from pre-F1120 events:
- **Predictive**: Based on co-occurrence patterns, not target event timing
- **Available for controls**: Can be calculated for any patient with events
- **Pattern-based**: Capture associations between drugs, diagnoses, and procedures
- **Important patterns**: Top itemsets and rules represent frequent and meaningful associations

**Example:**
- Patient has itemset `[AMOXICILLIN, F10.10]` → `drug_name_itemset_2_match = 1`, `drug_name_itemset_2_support = 0.15`
- Patient matches rule `AMOXICILLIN → F10.10` → `drug_name_rule_5_match = 1`, `drug_name_rule_5_confidence = 0.8`

**Source:** `9_risk_dashboard/visualizations/fpgrowth/create_fpgrowth_features.py`

---

### 4. PGx Features ✅

**Kept:**
- All pharmacogenomic features (allele frequencies, risk scores)
- Gene coverage and drug mapping counts

**Why kept:** PGx features are based on:
- **Patient genetics** (static, available at any time)
- **Drug exposure** (historical, available before prediction)
- **Population allele frequencies** (reference data, not patient-specific)

**Source:** `7_pgx_analysis/create_pgx_features_patient_level.py`

---

## Feature Engineering Approach

### For Target Patients

1. **Pre-event features**: Calculate from events before target event date
2. **Time intervals**: Calculate from all available events (no target reference)
3. **FP-Growth**: Match patient events to frequent itemsets/rules
4. **PGx**: Map patient drugs to gene-allele frequencies

### For Control Patients

1. **Pre-event features**: Calculate from events before a **reference date** (e.g., first event, cohort entry)
2. **Time intervals**: Calculate from all available events (same as targets)
3. **FP-Growth**: Match patient events to frequent itemsets/rules (same as targets)
4. **PGx**: Map patient drugs to gene-allele frequencies (same as targets)

**Key principle:** Control patients should have features calculated using the **same methodology** as target patients, just with a different reference point (not target event date).

---

## Implementation Details

### Script: `8_final_model/remove_target_leakage.py`

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

### Script: `8_final_model/build_final_cohort_model_features.py`

This script builds the final feature table **without** leakage features:

```python
# Merged features (PRESERVED):
1. Pre-event features (pre_*)
2. Predictive time intervals (drug_interval_*, icd_interval_*, cpt_interval_*)
3. FP-Growth features:
   - Important itemsets (top N itemsets with match indicators and support scores)
   - Important rules (top N rules with match indicators, confidence, and lift scores)
   - Summary statistics (matched counts, max support/confidence/lift)
4. Sequence features (is_top_sequence, is_rare_sequence)
5. PGx features (allele frequencies, risk scores)

# NOT merged (REMOVED):
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

- **Target Leakage Detection**: `8_final_model/remove_target_leakage.py`
- **Predictive Time Features**: `6_dtw_analysis/create_predictive_time_features.py`
- **Feature Table Building**: `8_final_model/build_final_cohort_model_features.py`
- **Removed Features List**: `8_final_model/outputs/{cohort}/{age_band}/{cohort}_{age_band}_removed_leakage_features.txt`

---

**Last Updated:** December 9, 2025

