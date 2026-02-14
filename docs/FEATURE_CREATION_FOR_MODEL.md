# How We Create Features for the Model

This document summarizes the pipeline from raw data to the feature set used for model training, and how we keep it correct and consistent.

## Single source of truth: Step 3b refined feature list

**The feature set used for the model is defined by Step 3b’s refined cohort_feature_importance (leakage-filtered).** All downstream steps that need “which codes/features go into the model” use this same source:

| Step | What it uses | Purpose |
|------|----------------|----------|
| **Step 4** (create_model_data) | Step 3b `*_cohort_feature_importance.csv` | Filters event-level data so **model_events.parquet** contains only events whose codes are in the refined list. |
| **Step 6** (run_final_model, build_final_cohort_model_features) | Step 3b `*_cohort_feature_importance.csv` | Builds binary (and count) item features from the **same** list so training features align with model_events. |

We do **not** use Step 3a aggregated_feature_importance for the final model feature set. Step 3a feeds into Step 3b; Step 3b applies leakage filtering and refinement and writes the canonical list.

---

## Pipeline overview

### 1. Step 3a – Aggregated feature importance

- **Input:** Cohort data (Step 2 cohort.parquet).
- **Output:** `{cohort}_{age_band_fname}_aggregated_feature_importance.csv` (many ICD/CPT/drug features with importance scores).
- **Role:** Provides a broad, importance-ranked feature set. For new cohorts without a baseline in pgx-repository, baseline is built from cohort-derived ICD/CPT/drug codes (never n_events only). We never run or write single-feature aggregated FI.

### 2. Step 3b – Refine and filter (canonical feature list)

- **Input:** Step 3a aggregated FI + BupaR post-target analysis (and optional safe_feature_filter.json).
- **Logic:** Removes post-target leakage, target-family codes (e.g. F11 for opioid_ed), applies importance threshold, normalizes feature names (e.g. `item_icd_F1120`).
- **Output:** `{cohort}_{age_band_fname}_cohort_feature_importance.csv` — **this is the list used for the model.**
- **Locations:** `3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}/`, S3 `gold/feature_importance/{cohort}/{age_band}/`.

### 3. Step 4 – Model data (event-level)

- **Input:** Event-level data + Step 3b cohort_feature_importance CSVs.
- **Logic:** `get_important_items(agg_csv)` reads the refined CSV (prefers `raw_code`, else derives from `feature`) and filters events so **model_events.parquet** only contains events whose codes are in that list.
- **Output:** model_events.parquet (and related artifacts) keyed by (cohort, age_band). No fallback to aggregated_feature_importance; Step 3b must have run.

### 4. Step 6 – Final model features (patient-level)

- **Input:** model_events (or model_data) + Step 3b cohort_feature_importance.
- **Logic:**
  - **run_final_model.py:** `_load_aggregated_feature_importance_codes()` loads Step 3b refined CSV and builds binary (and count) item features from those codes.
  - **build_final_cohort_model_features.py:** `load_cohort_feature_importance()` loads the same Step 3b CSV and builds item features from it.
- **Output:** Patient-level feature table (mi_person_key, target, item_* binary/count columns, PGx columns) used for training.

---

## Correctness checks

1. **Single source:** Step 4 and Step 6 both use **Step 3b cohort_feature_importance** (via `get_important_items` / `load_cohort_feature_importance` or `_load_aggregated_feature_importance_codes`). No step uses Step 3a aggregated FI to define the model feature set.
2. **No n_events-only:** Step 3a never returns or writes a single-feature (n_events-only) aggregated FI; baseline for new cohorts uses cohort-derived ICD/CPT/drug list.
3. **Empty guard:** Loaders for both 3a and 3b refined FI reject empty files (ValueError/FileNotFoundError) so we never build model features from an empty list.
4. **Naming:** Step 3b outputs canonical feature names (e.g. `item_icd_F1120`, `item_cpt_80307`, `item_drug_SUBOXONE`); Step 4 and Step 6 use the same list so event filtering and feature construction align.

---

## File locations (refined list for model)

- **Local:** `3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_cohort_feature_importance.csv`
- **S3:** `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band_fname}_cohort_feature_importance.csv`

Shared resolution is in `py_helpers.feature_importance_eda_utils`: `resolve_cohort_fi_path()`, `load_cohort_feature_importance()`.
