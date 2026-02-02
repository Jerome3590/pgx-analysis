# Feature Importance: Current Output vs Original

## Current S3 output (actual)

**Source:** Step 3a `run_mc_feature_importance.py` using **cohort data only** (cohort.parquet).  
Patient-level table: one row per person with `target = MAX(is_target_case)` and **one feature: `n_events`** (event count per person).

**Example aggregated CSV** (opioid_ed/25-44):

```csv
feature,scaled_importance_mean,scaled_importance_std,scaled_importance_count,importance_mean,importance_std,recall_mean,logloss_mean,auc_mean,pr_auc_mean
n_events,1.0,0.0,25,1.0,0.0,0.205,0.449,0.748,0.445
```

- **Columns:** `feature`, `scaled_importance_mean`, `scaled_importance_std`, `scaled_importance_count`, `importance_mean`, `importance_std`, `recall_mean`, `logloss_mean`, `auc_mean`, `pr_auc_mean`
- **Rows:** One feature only: `n_events`

---

## Original / legacy format (README)

**Source:** Legacy pipeline (R/Python) using **event-level or item-level** data (e.g. model_events or event-level features).  
Many features: drug names, ICD codes, CPT codes.

**Documented columns:**

| Column                  | Description                          |
|-------------------------|--------------------------------------|
| `rank`                  | Final rank by importance_scaled      |
| `feature`               | Drug, ICD, or CPT (e.g. HYDROCODONE-ACETAMINOPHEN, F11.20) |
| `importance_normalized` | Sum of normalized importances       |
| `importance_scaled`      | Sum of Recall-scaled importances      |
| `n_models`               | Number of models including feature   |
| `models`                 | e.g. "catboost, xgboost, xgboost_rf" |
| `mc_cv_recall_mean`     | Average Recall across models         |
| `mc_cv_recall_std`      | Recall std dev                       |

**Example rows:** Many features (drugs, ICDs, CPTs), e.g. `HYDROCODONE-ACETAMINOPHEN`, `TRAMADOL HCL`, `F11.20`.

---

## Comparison summary

| Aspect        | Current (cohort-only)     | Original (event/item-level)     |
|---------------|---------------------------|----------------------------------|
| **Input data**| cohort.parquet only       | model_events or event-level     |
| **Features**  | One: `n_events`           | Many: drugs, ICDs, CPTs         |
| **CSV columns** | scaled_importance_mean, recall_mean, logloss_mean, auc_mean, pr_auc_mean | rank, importance_normalized, importance_scaled, n_models, models, mc_cv_recall_mean, mc_cv_recall_std |
| **Downstream (Step 4)** | Step 4 reads `feature` and uses values as drug/ICD/CPT codes to filter case events. With only `n_events`, Step 4 would get one “item” that does not match event-level columns. | Step 4 gets a list of drug/ICD/CPT codes and filters case events by those items. |

**Conclusion:** Current results are **correct for the current design** (cohort-only, one patient-level feature), but they are **not equivalent to the original** in content or schema. The original produced many item-level features used by Step 4 to filter events; the current run produces a single feature `n_events`, which is not used as a filter code. To get original-like, many-feature rankings and Step 4 compatibility, the pipeline would need either:

1. **Use model_events again** (Step 4 output) as input to Step 3a so event-level columns (drug_name, ICD, procedure_code) can be used as features, or  
2. **Build more features from cohort** (e.g. pivot event-level drug/ICD/CPT into patient-level counts or indicators) and keep using cohort-only data.

---

## Correct flow (feature columns and target)

| Step | Description |
|------|--------------|
| **Historical aggregated feature importances (baseline)** | Live in **pgx-repository** (read-only). Used to filter cohort features **after cohorts are built** (1b event filter). |
| **After cohorts built** | 1b event filter uses baseline FI to keep only events whose codes appear in baseline. |
| **Second feature importances** | Start from **baseline aggregated FI** (not the original full item-level set). Load **historical aggregated FI from pgx-repository** → minus admin/Z codes → use that list as features (~**11K features**) → build patient-level feature matrix from cohort.parquet → run MC CV → **always save to pgxdatalake**. So the second pass has **all columns from aggregated feature importances minus the admin Z codes**; feature set is much smaller than the original (~11K). |
| **Final model train features** | Final model training (Step 6) uses **second-pass feature importances from pgxdatalake** for train features (build_final_cohort_model_features / run_final_model). |
| **Target** | Unchanged: `target = MAX(is_target_case)` from cohort (0 = control, 1 = case). Validated so both classes are present. |
