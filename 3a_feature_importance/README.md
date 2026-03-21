# Feature Importance Analysis

**Date:** November 25, 2025  
**Project:** PGx Analysis - Feature Importance with Monte Carlo Cross-Validation  
**Notebook:** `feature_importance_mc_cv.ipynb`

**Output Structure:** This analysis follows the standard output structure framework. See [`docs/README_output_structure.md`](../docs/README_output_structure.md) for details.

---

## Table of Contents

1. [Overview](#overview)
   - [Background: Categorical features as count columns](#background-categorical-features-as-count-columns)
2. [Quick Start](#quick-start)
   - [Local Testing](#local-testing-5-splits-5-minutes)
   - [Production Run](#production-run-100-splits-1-2-hours-on-ec2)
   - [Parallel Execution](#parallel-execution-default)
   - [Single Cohort Execution](#single-cohort-execution-optional)
3. [Methodology](#methodology)
4. [Aggregation Method](#aggregation-method)
5. [Output Files](#output-files)
6. [Visualization](#visualization)
7. [Filtering Administrative and Non-Informative Codes](#filtering-administrative-and-non-informative-codes)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [EC2 Configuration and Optimizations](#ec2-configuration-and-optimizations)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This project calculates scaled feature importance for predicting opioid dependence using:
- **Core Tree Models:** CatBoost, XGBoost (boosted trees), XGBoost RF mode
- **Validation:** Monte Carlo Cross-Validation (typically 10–50 splits; **current large-cohort runs use 10 splits** with strict temporal validation)
- **Scaling:** Permutation-based importance weighted by model performance (Recall, or inverse LogLoss when configured)
- **Aggregation:** Union across **all features with non-zero gain** (`gain_importance > 0`) and their permutation-based importances, aggregated across models

### Temporal Validation Strategy

**Important:** This analysis uses a strict temporal validation approach to avoid data leakage and COVID-19 impact:

- **Training Data:** Years 2016-2018 (combined)
- **Test Data:** Year 2019 (holdout set, never used for training)
- **Excluded:** Year 2020 (COVID-19 pandemic year)

**Rationale:**
1. **Prevents Data Leakage:** 2019 data is never seen during training, ensuring true temporal validation
2. **Maintains Temporal Order:** Train on past data, test on future data
3. **Avoids COVID Impact:** 2020 excluded due to pandemic-related changes in healthcare patterns
4. **Consistent with Final Model:** Feature importance results generalize to final model which also trains on 2016-2018 and tests on 2019

**MC-CV Implementation:**
- Each MC-CV split samples a different subset from the 2016-2018 training data
- All splits evaluate on the same 2019 test set
- This provides robust feature importance estimates while maintaining temporal integrity

### Flow and downstream use

- **Historical baseline FI** lives in **pgx-repository** (read-only); used to filter cohort features after cohorts are built (1b event filter).
- **Second pass** (default): load historical FI from pgx-repository → minus admin/Z codes → build feature matrix from cohort → run MC CV. **If baseline is missing** in pgx-repository for that cohort/age_band, the script runs a **baseline pass first** (permutation feature importance on cohort-derived features), then uses that result for the second pass. Use `--no-run-baseline-if-missing` to skip this and use n_events only.
- **Second-pass feature importances are always saved to pgxdatalake** (`gold/feature_importance/{cohort}/{age_band}/`). We never write second-pass results to pgx-repository so the historical baseline is never overwritten.
- **Final model train features:** Step 4 (model data) and Step 6 (final model) use these second-pass feature importances from pgxdatalake for train features.

### Key Features

✅ **Monte Carlo Cross-Validation** – Up to 1000 random train/test splits  
✅ **Stratified Sampling** – Maintains target distribution  
✅ **Parallel Processing** – Fast execution (30 workers on EC2)  
✅ **Quality Weighting** – Features scaled by model performance (Recall)  
✅ **Model Consensus** – Union-based aggregation rewards agreement  
✅ **Publication-Ready Plots** – 4 visualization types with S3 upload

### Background: Categorical features as count columns

Event-level data has categorical columns (e.g. drug name, ICD code, procedure code). In this pipeline we convert them to **one numeric column per distinct code**, where each cell is the **event count** for that patient for that code (0, 1, 2, …). All three models (CatBoost, XGBoost, XGBoost RF) then receive the same numeric count matrix; CatBoost is **not** given raw categorical columns.

**Why this approach is correct and efficient here**

- **Semantics:** The natural summary is “how many times did this code appear?” — counts are the right representation for utilization/frequency.
- **Interpretability:** Feature importance is directly “which codes matter”; no extra encoding layer.
- **Efficiency:** One pass over events (unpivot → filter → groupby patient + code → pivot) fits DuckDB/SQL and produces a sparse matrix that tree models handle well.
- **No target leakage:** We only use event-level codes and aggregate to counts; no post-event or target-time information in the matrix.

**Why categoricals aren’t always handled this way**

- **Cardinality:** With ~11K codes we get ~11K columns, which is fine for trees. With millions of categories (e.g. user IDs), one column per category is usually impractical; people use embeddings, hashing, or target-style encodings instead.
- **Meaning:** Counts make sense when the variable is event-like (“how often X happened”). For purely nominal categories (e.g. region, color) there is no “count” per row — then one-hot, label encoding, or target/embedding encodings are more common.
- **Model type:** Trees work well with many sparse numeric columns. Linear models and some others prefer fewer dimensions (e.g. one-hot or a small set of embedding dimensions) and may need regularization or dimension reduction when there are many columns.

**Summary:** Using one column per code with event counts is a straightforward and appropriate choice when (1) the variable is event-like and count is meaningful, (2) cardinality is manageable for the tools, and (3) models are tree-based and handle many sparse numeric features well — as in this feature-importance pipeline.

### Understanding Permutation Importance vs. Row-Level Analysis

**Important:** This analysis uses **permutation importance**, which measures **average feature effects** across all patients. It does **NOT** preserve row-level associations or tell you which specific drug combinations matter for specific patients.

**What Permutation Importance Does:**
- Measures average effect: "On average, does shuffling feature X affect model performance?"
- Breaks row associations: Shuffling breaks the connection between specific drug combinations and specific patients
- Works well with sampling: Using `PGX_PERM_MAX_ROWS=50000` for speed doesn't significantly impact average-effect measurements

**What Permutation Importance Cannot Do:**
- ❌ Identify which specific drug combinations drive outcomes for specific patients
- ❌ Tell you "Patient 12345 with drugs [A, B] had outcome Y"
- ❌ Preserve row-level associations

**For Row-Level Analysis, Use:**
- **SHAP Values** (Step 8): Patient-specific feature contributions
- **FFA Analysis** (Step 9): Rule-based patient explanations
- **FPGrowth** (Step 4): Frequent drug combination patterns

**Best Approach:**
1. Use permutation importance (with sampling) for fast feature ranking
2. Use SHAP/FFA/FPGrowth on full dataset for patient-specific insights
3. Combine both approaches for comprehensive understanding

**See:** [`docs/Step3_FeatureImportance/README_feature_importance.md`](../docs/Step3_FeatureImportance/README_feature_importance.md#understanding-permutation-importance-vs-row-level-analysis) for detailed explanation.

---

## Quick Start

### Local Testing (5 splits, ~5 minutes)

```r
# In feature_importance_mc_cv.ipynb
DEBUG_MODE <- TRUE
COHORT_NAME <- "opioid_ed"
AGE_BAND <- "25-44"
EVENT_YEAR <- 2016

# Run all cells
```

### Production Run (100 splits, ~1-2 hours on EC2)

```r
DEBUG_MODE <- FALSE
N_SPLITS <- 100  # or 1000 for publication

# Set up EC2:
# - x2iedn.8xlarge (32 cores, 1TB RAM)
# - Data in /mnt/nvme/cohorts/
# - Auto-shutdown enabled
```

### Parallel Execution (Default)

The notebook (Cell 5) runs all combinations defined in `COHORT_NAMES` × `AGE_BANDS` in parallel. Each task processes one cohort/age-band combination using the `run_cohort_analysis()` function.

**Configuration:**
- Automatically handles multiple cohorts and age-bands
- Idempotent: Skips already-processed combinations (checks local files and S3)
- Nested parallelism: Optimizes worker allocation between task-level and MC-CV level

### Single Cohort Execution (Optional)

If you want to run a single cohort/age-band combination instead of parallel execution, you can call `run_cohort_analysis()` directly after sourcing the helper functions:

```r
# Source helper functions first (from Cell 2)
source(file.path(helpers_dir, "constants.R"))
source(file.path(helpers_dir, "logging_utils.R"))
source(file.path(helpers_dir, "metrics.R"))
source(file.path(helpers_dir, "model_helpers.R"))
source(file.path(helpers_dir, "mc_cv_helpers.R"))
source(file.path(helpers_dir, "run_cohort_analysis.R"))

# Set configuration (from Cell 3)
DEBUG_MODE <- FALSE
N_SPLITS <- 200
TEST_SIZE <- 0.2
TRAIN_PROP <- 1 - TEST_SIZE
SCALING_METRIC <- "recall"
N_WORKERS <- 30  # Adjust based on available cores

MODEL_PARAMS <- list(
  catboost = list(
    iterations = 100,
    learning_rate = 0.1,
    depth = 6,
    verbose = 0L,
    random_seed = 42
  ),
  random_forest = list(
    ntree = 100,
    mtry = NULL,
    nodesize = 1,
    maxnodes = NULL
  )
)

# Run single cohort/age-band analysis
result <- run_cohort_analysis(
  cohort_name = "opioid_ed",
  age_band = "25-44",
  event_year = 2016,
  n_splits = N_SPLITS,
  train_prop = TRAIN_PROP,
  n_workers = N_WORKERS,
  scaling_metric = SCALING_METRIC,
  model_params = MODEL_PARAMS,
  debug_mode = DEBUG_MODE
)

# Check results
if (result$status == "success") {
  cat(sprintf("✓ Analysis complete. Features: %d\n", nrow(result$aggregated)))
  cat(sprintf("Output file: %s\n", result$output_file))
} else {
  cat(sprintf("✗ Analysis failed: %s\n", result$error))
}
```

### Command Line (Python equivalent - future)

```bash
Rscript feature_importance_mc_cv.R \
  --cohort opioid_ed \
  --age-band 25-44 \
  --year 2016 \
  --splits 100
```

---

## Methodology

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Cohort Data (parquet)                               │
│    - Drugs, ICD codes, CPT codes                            │
│    - Target: is_target_case (opioid dependence)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Engineering                                      │
│    - Patient-level aggregation                              │
│    - CatBoost: Categorical factors                          │
│    - XGBoost / XGBoost RF: Binary 0/1                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Monte Carlo Cross-Validation (100–1000 splits)           │
│    ┌──────────────────────────────┐                         │
│    │   Core Models (3 total):     │                         │
│    │   - CatBoost                 │                         │
│    │   - XGBoost (boosted trees)  │                         │
│    │   - XGBoost RF mode          │                         │
│    │                              │                         │
│    │  Per split:                  │                         │
│    │  - Train (80%)               │                         │
│    │  - Test (20%)                │                         │
│    │  - Recall                    │                         │
│    │  - Feature imp               │                         │
│    └──────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Union-Based Aggregation                                  │
│    - Combine **all features** from:                         │
│        • CatBoost                                           │
│        • XGBoost                                            │
│        • XGBoost RF                                         │
│        • XGBoost (rare-variant scan, target cohort only)    │
│    - Keep permutation-based scores for every feature        │
│    - Annotate XGBoost-family features with gain>0           │
│    - Normalize + scale by model performance                 │
│    - Rank by aggregated, scaled importance                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Outputs                                                  │
│    - Aggregated CSV (final rankings)                        │
│    - Visualizations (4 plots)                               │
│    - S3 upload                                              │
└─────────────────────────────────────────────────────────────┘
```

### Models

**Core Tree Models:**

1. **CatBoost:**
   - Handles categorical features natively
   - Feature format: Each column is a factor with item name as level
   - Importance: Permutation-based (PredictionValuesChange)

2. **XGBoost (boosted trees):**
   - Gradient boosting with tree-based learners
   - Feature format: Binary 0/1 encoding (one-hot)
   - Importance: **combined gain + permutation**:
     - **Gain screen (annotation):** Compute XGBoost’s built-in tree importance (gain / Gini) for **all features** and flag features with `gain_importance > 0` (i.e., ever used in a split). This ensures that **all potentially meaningful signals (including rare variants) are explicitly tracked**.
     - **Permutation importance (primary score):** Run permutation-based importance on the **full feature set** (optionally capped in rows via `PGX_PERM_MAX_ROWS`). Every non-constant feature receives a permutation score; the gain flag is used for interpretation, not as a filter.

3. **XGBoost RF Mode:**
   - Random-forest style configuration of XGBoost
   - Feature format: Binary 0/1 encoding (one-hot)
   - Importance: Same **gain>0 + full-feature permutation** scheme as boosted XGBoost

All three core models use **permutation-based importance** for fair comparison; XGBoost and XGBoost RF additionally expose gain-based annotations (`gain_gt_zero`) to show which features were ever used in tree splits.

---

## Aggregation Method

### Step-by-Step Process

#### 1. Per-Model MC-CV + Importance

For each model type:

- **CatBoost, XGBoost, XGBoost RF** (core ensemble, all cohorts/age bands)
- **XGBoost Rare** (second-pass rare-variant scan, target cohort only)

we:

- Run **MC-CV** (currently 10 splits for feature importance) with 2016–2018 → 2019 temporal validation.
- Compute **permutation-based feature importance** for every non-constant feature.
- For XGBoost / XGBoost RF, also compute:
  - `gain_importance` (tree-based gain / Gini) for **all features**
  - `gain_gt_zero` flag indicating `gain_importance > 0` (ever used in a split)

Each per-model CSV contains, at minimum:

- `feature`
- `importance_mean` (permutation-based, averaged across splits)
- `recall_mean`, `logloss_mean`
- For XGBoost-family models: `gain_importance`, `gain_gt_zero`

#### 2. Normalize Within Each Model

For each model separately:

- Normalize `importance_mean` to \[0, 1\] across that model’s own features, producing `importance_normalized_model`.

This makes the **shape of each model’s importance distribution comparable**, independent of its raw scale.

#### 3. Scale by Model Performance

Within each model:

- If `scaling_metric == 'recall'`:
  - Compute `importance_scaled_by_model = importance_normalized_model × recall_mean`.
- If `scaling_metric == 'logloss'`:
  - Compute `importance_scaled_by_model = importance_normalized_model × (1 / logloss_mean)` (lower logloss → larger weight).

This step **weights features by how strong the model is overall**.

#### 4. Union Across Models and Aggregate

- Take the **union of all features** that appear in any of:
  - CatBoost, XGBoost, XGBoost RF, XGBoost Rare.
- For each feature, aggregate across models:
  - `importance_scaled_by_model_sum` = sum of `importance_scaled_by_model` over models where the feature appears.
  - `importance_normalized_sum` = sum of per-model normalized importances.
  - `n_models` = number of distinct models contributing to that feature.
  - `importance_scaled_mean` = `importance_scaled_by_model_sum / n_models`.

We then **renormalize** `importance_scaled_mean` to \[0, 1\] across all features to obtain a global `importance_normalized`.

#### 5. Final Scaling by Best Model

- Identify the **best-performing model** by the chosen `scaling_metric`:
  - If `recall`: model with highest `recall_mean`.
  - If `logloss`: model with lowest `logloss_mean` (then invert so lower is better).
- Compute a final scaling factor:

  - `scale_factor = best_recall` if scaling on recall, or  
  - `scale_factor = 1 / best_logloss` if scaling on logloss.

- Define the final **aggregated score**:

```python
importance_scaled = importance_normalized * scale_factor
```

The final aggregated CSV therefore contains:

- `feature`
- `importance_normalized` (0–1, aggregated across models)
- `importance_scaled` (normalized × best-model performance)
- `n_models` (how many models contributed non-zero signal for this feature)

For XGBoost-family models, you can also join back the `gain_gt_zero` flags to explicitly see **which features had gain>0 plus their permutation-based scores**, matching the “gain>0 + permutation” workflow.

### Why This Approach?

#### ✅ Advantages

1. **Model Agreement Rewarded**
   - Features important in **both** models get higher scores (summed importances)
   - Reduces risk of model-specific artifacts

2. **Quality Weighting**
   - Features are scaled by model performance (Recall)
   - Better-performing models contribute more to final scores

3. **Permutation-Based Importance**
   - Uses true feature importance (not tree-based Gini/gain)
   - Measures actual predictive contribution

4. **Top-N Focus**
   - Only considers top 50 from each model
   - Filters out noise from low-importance features
   - Computationally efficient

5. **No Arbitrary Weights**
   - No manual weighting of models required
   - Performance-based scaling is automatic and objective

#### Considerations

1. **Summing Favors Overlap**
   - Features in both models will typically rank higher
   - This is intentional (consensus is valuable)
   - But model-specific features can still rank high if importance × recall is large

2. **Top-50 Cutoff**
   - Features ranked 51+ in all models are excluded
   - Ensure cutoff is appropriate for your use case
   - Can be adjusted in code

3. **Recall as Quality Metric**
   - Appropriate for imbalanced classification (opioid dependence)
   - For other use cases, consider alternative metrics (F1, AUC-ROC)

### Example Calculation

**Scenario:**
- CatBoost top 50: Feature "HYDROCODONE-ACETAMINOPHEN"
  - `importance_normalized` = 0.95
  - `mc_cv_recall` = 0.82
  - `importance_scaled` = 0.95 × 0.82 = **0.779**

- Random Forest top 50: Same feature "HYDROCODONE-ACETAMINOPHEN"
  - `importance_normalized` = 0.88
  - `mc_cv_recall` = 0.80
  - `importance_scaled` = 0.88 × 0.80 = **0.704**

**Final Aggregated Value:**
- `importance_normalized` = 0.95 + 0.88 = **1.83**
- `importance_scaled` = 0.779 + 0.704 = **1.483** ← Used for ranking
- `n_models` = 2
- `models` = "catboost, random_forest"

**Interpretation:** This feature is highly important in both models and performs well, earning a high final score.

### Comparison to Alternative Methods

| Method | Formula | Problem |
|--------|---------|---------|
| **Averaging** ❌ | `(cb + rf) / 2` | Treats poor and good models equally |
| **Concatenation** ❌ | `union(all_features)` | No weighting, noisy features dilute results |
| **Intersection** ❌ | `intersect(cb, rf)` | Too restrictive, misses model-specific features |
| **Union + Sum + QW** ✅ | `sum(cb_scaled, rf_scaled)` | Rewards agreement, weights by performance |

---

## Feature Engineering Decisions

### FP-Growth Itemset Features: Match vs Support

**Decision:** We keep `_match` features (binary indicators) and remove individual `_support` features, while retaining aggregate `itemsets_max_support` features.

#### Analysis

**Feature Definitions:**
- `{item_type}_itemset_{idx}_match`: Binary indicator (0/1) - whether patient has this itemset
- `{item_type}_itemset_{idx}_support`: `_match * support_value` - support value if matched, 0 otherwise
- `{item_type}_itemsets_max_support`: Maximum support across all matched itemsets (aggregate feature)

**Findings from Feature Importance Analysis:**

1. **High Correlation:** `_support` features are mathematically derived from `_match` features:
   - `_support = _match * support_value`
   - When `_match = 0`, `_support = 0`
   - When `_match = 1`, `_support = support_value`
   - This creates near-perfect correlation between the two feature types

2. **Similar Importance Patterns:**
   - Both feature types show similar importance rankings
   - Mean importance: Match features (4.08) vs Support features (3.60)
   - Count of significant features: Match (81) vs Support (80)

3. **Match Features More Prevalent:**
   - Match features consistently rank higher in importance
   - Binary indicators capture the primary signal (presence/absence)
   - Support values add minimal additional predictive power

**Rationale for Keeping Match Only:**

1. **Simplicity:** Binary indicators are easier to interpret and use
2. **Redundancy Reduction:** Removing highly correlated features reduces multicollinearity
3. **Model Efficiency:** Fewer features = faster training and inference
4. **Signal Preservation:** Match features capture the primary predictive signal

**Exception: Aggregate Support Features**

We retain `itemsets_max_support` features because:
- They aggregate across multiple itemsets, providing a different signal
- They capture the "best" itemset support for each patient
- They are not redundant with individual match features
- They may provide additional predictive value beyond binary indicators

**Implementation:**

- **Removed:** Individual `{item_type}_itemset_{idx}_support` features
- **Kept:** Individual `{item_type}_itemset_{idx}_match` features
- **Kept:** Aggregate `{item_type}_itemsets_max_support` features
- **Kept:** Count features like `{item_type}_itemsets_matched_count`

This decision applies to all FP-Growth itemset features (drug_name, icd_code, cpt_code, medical_code).

---

## Output Files Manifest

### Expected Outputs Structure

For each `(cohort, age_band)` combination, the following files should be generated:

#### Data Files (`outputs/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{cohort}_{age_band}_aggregated_feature_importance.csv` | Final aggregated feature rankings | ✅ Yes |
| `{cohort}_{age_band}_catboost_feature_importance.csv` | CatBoost model results | ✅ Yes |
| `{cohort}_{age_band}_xgboost_feature_importance.csv` | XGBoost model results | ✅ Yes |
| `{cohort}_{age_band}_xgboost_rf_feature_importance.csv` | XGBoost RF model results | ✅ Yes |
| `{cohort}_{age_band}_constant_features.csv` | List of constant features removed | ✅ Yes |

**Example Files:**
- `opioid_ed_0_12_aggregated_feature_importance.csv`
- `opioid_ed_0_12_catboost_feature_importance.csv`
- `opioid_ed_0_12_xgboost_feature_importance.csv`
- `opioid_ed_0_12_xgboost_rf_feature_importance.csv`
- `opioid_ed_0_12_constant_features.csv`

#### Visualization Files (`outputs/plots/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{cohort}_{age_band}_{year}_top50_features.png` | Top 50 features bar chart | ✅ Yes |
| `{cohort}_{age_band}_{year}_top50_with_recall.png` | Top 50 with recall confidence | ✅ Yes |
| `{cohort}_{age_band}_{year}_normalized_vs_scaled.png` | Normalized vs scaled comparison | ✅ Yes |
| `{cohort}_{age_band}_{year}_category_distribution.png` | Feature category breakdown | ✅ Yes |

**Note:** Visualizations are generated using the Python script `py_helpers/create_feature_importance_visualizations.py` for consistency with the rest of the Python-based workflow. The R script (`r_helpers/create_visualizations.R`) is maintained for backward compatibility but is no longer the primary visualization tool.

**Example Files:**
- `opioid_ed_0_12_top50_features.png`
- `opioid_ed_0_12_top50_with_recall.png`
- `opioid_ed_0_12_normalized_vs_scaled.png`
- `opioid_ed_0_12_category_distribution.png`

**Note:** Visualization files are generated by the R visualization script (`r_helpers/create_visualizations.R`) and may need to be run separately after the main analysis completes.

### Completion Checklist

For each cohort/age-band combination:

- [ ] Aggregated feature importance CSV exists
- [ ] All three model result CSVs exist (CatBoost, XGBoost, XGBoost RF)
- [ ] Constant features CSV exists
- [ ] All four visualization plots exist in `outputs/plots/`
- [ ] Files uploaded to S3 (if applicable)

---

## Output Files

### 1. Aggregated Feature Importance CSV

**Location:**
- Local: `outputs/{cohort}_{age}_{year}_feature_importance_aggregated.csv`
- S3: `s3://pgxdatalake/gold/feature_importance/cohort_name={cohort}/age_band={age}/event_year={year}/`

**Columns:**

| Column | Description | Range |
|--------|-------------|-------|
| `rank` | Final rank by `importance_scaled` | 1, 2, 3, ... |
| `feature` | Feature name (drug, ICD, CPT) | String |
| `importance_normalized` | Sum of normalized importances | 0.0 – 3.0 |
| `importance_scaled` | Sum of Recall-scaled importances | 0.0 – ~2.4 |
| `n_models` | Number of models including feature | 1 to 3 |
| `models` | Which models | Comma-separated list (e.g., "catboost, xgboost, xgboost_rf") |
| `mc_cv_recall_mean` | Average Recall across models | 0.0 – 1.0 |
| `mc_cv_recall_std` | Recall std dev | 0.0 – 1.0 |

**Key Metric:** `importance_scaled` - Used for final ranking and visualization.

**Example:**
```csv
rank,feature,importance_normalized,importance_scaled,n_models,models,mc_cv_recall_mean,mc_cv_recall_std
1,HYDROCODONE-ACETAMINOPHEN,2.45,1.98,3,"catboost, xgboost, xgboost_rf",0.84,0.012
2,TRAMADOL HCL,1.80,1.50,2,"catboost, xgboost",0.83,0.016
3,F11.20,0.92,0.76,1,"catboost",0.82,0.015
```

### 2. Per-Model CSVs

**Files:**
- `{cohort}_{age}_{year}_catboost_feature_importance.csv`
- `{cohort}_{age}_{year}_random_forest_feature_importance.csv`

**Purpose:** Debugging, model comparison, reproducibility

---

## Visualization

Four publication-ready plots are automatically generated:

### 1. Top 50 Features (Bar Chart)
- **File:** `{cohort}_{age}_{year}_top50_features.png`
- **Size:** 12" × 14"
- **Shows:** Scaled importance, ranked

### 2. Top 50 with Recall Confidence
- **File:** `{cohort}_{age}_{year}_top50_with_recall.png`
- **Size:** 12" × 14"
- **Color:** Orange (lower Recall) → Dark Blue (higher Recall)
- **Shows:** Importance + model quality

### 3. Normalized vs Recall-Scaled (Top 50)
- **File:** `{cohort}_{age}_{year}_normalized_vs_scaled.png`
- **Size:** 12" × 14"
- **Shows:** Impact of quality weighting (side-by-side comparison)

### 4. Feature Category Distribution
- **File:** `{cohort}_{age}_{year}_category_distribution.png`
- **Size:** 12" × 10"
- **Shows:** Drug / ICD / CPT breakdown of top features

**Location:**
- Local: `3a_feature_importance/outputs/{cohort}/plots/` (and `PGX_FEATURE_IMPORTANCE_OUTPUTS/{cohort}/plots/` when the env var is set)
- S3: `s3://pgxdatalake/gold/feature_importance/cohort_name={cohort}/age_band={age}/event_year={year}/plots/`

**Note:** The `plots/` subdirectory is automatically created when running the analysis. All visualization files are saved here following the standard output structure framework.

---

## Filtering Administrative and Non-Informative Codes

After generating aggregated feature importance, administrative and non-informative codes should be filtered out before proceeding to model data creation (Step 4a). This filtering is performed in **Step 3b: Feature Importance EDA**.

### Administrative ICD Z Codes

**Not all Z codes are administrative.** ICD-10 Chapter 21 (Z00-Z99) contains both administrative codes and clinically important codes:

#### Administrative Z Codes (Filtered)

The following Z codes are classified as administrative and are filtered out:

- **Z00.00** - Encounter for general adult medical examination without abnormal findings
- **Z00.01** - Encounter for general adult medical examination with abnormal findings  
- **Z00.121** - Encounter for routine child health examination with abnormal findings
- **Z00.129** - Encounter for routine child health examination without abnormal findings

**Note:** These codes represent routine administrative examinations rather than clinical diagnoses and do not add predictive value for the target outcome.

#### Medical Z Codes (Kept)

Most Z codes are **medical** and should be kept as features, including:

- **Z11-Z13** - Screening codes (preventive care)
- **Z55-Z65, Z59** - Social determinants of health (clinical context)
- **Z34-Z39** - Encounter for maternal care
- **Z40-Z53** - Encounter for other specific health care
- **Z80-Z99** - Family history, personal history, etc.
- **All other Z codes** - Classified as medical by default

### Filtering Process

The filtering process in Step 3b includes:

1. **DTW Analysis** - Identifies administrative/non-informative ICD/CPT codes based on trajectory patterns
2. **BupaR Analysis** - Identifies post-target leakage features (codes appearing after target event)
3. **Administrative Codes Lookup** - Uses pre-identified administrative codes from `4b_dtw_filter/administrative_codes_lookup.json`
4. **Manual Review** - Allows manual addition/removal of codes based on domain expertise

### Example: Z Codes in Feature Importance

For the `opioid_ed` cohort, age band `13-24`:
- **Total Z codes:** 386
- **Administrative Z codes (filtered):** 4 (Z00.00, Z00.01, Z00.121, Z00.129)
- **Medical Z codes (kept):** 382

**Important:** Even though administrative Z codes may have high feature importance scores (e.g., Z00.129 with importance 0.265), they should still be filtered as they represent administrative encounters rather than clinical risk factors.

### Code Group Analysis: ICD and CPT Codes by Letter/Range

Analysis of feature importance data shows the distribution of administrative vs informative codes across ICD-10 chapters and CPT code ranges.

#### ICD Codes by Chapter (ICD-10)

**Analysis Results for `opioid_ed` cohort, age band `13-24`:**

| Chapter | Letter | Description | Total Codes | Administrative | Informative | Classification |
|---------|--------|-------------|-------------|----------------|-------------|----------------|
| 1 | A | Certain infectious and parasitic diseases | 114 | 0 | 114 | Informative |
| 1 (cont.) | B | Certain infectious and parasitic diseases (continued) | 77 | 0 | 77 | Informative |
| 2 | C | Neoplasms | 53 | 0 | 53 | Informative |
| 3 | D | Diseases of blood and immune mechanism | 124 | 0 | 124 | Informative |
| 4 | E | Endocrine, nutritional and metabolic diseases | 122 | 0 | 122 | Informative |
| 5 | F | Mental, behavioral and neurodevelopmental disorders | 292 | 0 | 292 | Informative |
| 6 | G | Diseases of the nervous system | 241 | 0 | 241 | Informative |
| 7 | H | Diseases of the eye and adnexa | 424 | 0 | 424 | Informative |
| 9 | I | Diseases of the circulatory system | 138 | 0 | 138 | Informative |
| 10 | J | Diseases of the respiratory system | 317 | 0 | 317 | Informative |
| 11 | K | Diseases of the digestive system | 221 | 0 | 221 | Informative |
| 12 | L | Diseases of the skin and subcutaneous tissue | 240 | 0 | 240 | Informative |
| 13 | M | Diseases of the musculoskeletal system and connective tissue | 453 | 0 | 453 | Informative |
| 14 | N | Diseases of the genitourinary system | 180 | 0 | 180 | Informative |
| 15 | O | Pregnancy, childbirth and the puerperium | 400 | 0 | 400 | Informative |
| 16 | P | Certain conditions originating in the perinatal period | 17 | 0 | 17 | Informative |
| 17 | Q | Congenital malformations, deformations and chromosomal abnormalities | 122 | 0 | 122 | Informative |
| 18 | R | Symptoms, signs and abnormal clinical and laboratory findings | 293 | 0 | 293 | Informative |
| 19 | S | Injury, poisoning and certain other consequences of external causes | 163 | 0 | 163 | Informative |
| 19 (cont.) | T | Injury, poisoning and certain other consequences of external causes (continued) | 34 | 0 | 34 | Informative |
| 22 | U | Codes for special purposes | 1 | 0 | 1 | Informative |
| 20 | V | External causes of morbidity | 108 | 0 | 108 | Informative |
| 21 | **Z** | **Factors influencing health status and contact with health services** | **353** | **4** | **349** | **Mixed** |

**Key Findings:**
- **All ICD chapters A-Y are 100% informative** - No administrative codes identified
- **Only Z chapter contains administrative codes** - 4 out of 353 codes (1.1%) are administrative
- **The 4 administrative Z codes** are routine examination codes (Z00.00, Z00.01, Z00.121, Z00.129)
- **349 Z codes are informative** and should be kept as features

#### CPT Codes by Range

**Analysis Results for `opioid_ed` cohort, age band `13-24`:**

| Range | Description | Total Codes | Administrative | Informative | Classification |
|-------|-------------|-------------|----------------|-------------|----------------|
| 00000-00999 | Anesthesia | 47 | 0 | 47 | Informative |
| 01000-01999 | Anesthesia (continued) | 39 | 0 | 39 | Informative |
| 10000-19999 | Surgery - Integumentary System | 122 | 0 | 122 | Informative |
| 20000-29999 | Surgery - Musculoskeletal System | 276 | 0 | 276 | Informative |
| 30000-39999 | Surgery - Respiratory, Cardiovascular, Hemic/Lymphatic | 184 | 0 | 184 | Informative |
| 40000-49999 | Surgery - Digestive System | 118 | 0 | 118 | Informative |
| 50000-59999 | Surgery - Urinary, Male Genital, Female Genital, Maternity | 119 | 0 | 119 | Informative |
| 60000-69999 | Surgery - Endocrine, Nervous System | 171 | 0 | 171 | Informative |
| 70000-79999 | Radiology | 361 | 0 | 361 | Informative |
| 80000-89999 | Pathology and Laboratory | 754 | 0 | 754 | Informative |
| **90000-99999** | **Medicine, Evaluation and Management, Miscellaneous** | **561** | **1** | **560** | **Mixed** |

**Key Findings:**
- **All CPT ranges 00000-89999 are 100% informative** - No administrative codes identified
- **Only 90000-99999 range contains administrative codes** - 1 out of 561 codes (0.2%) is administrative
- **The administrative CPT codes** are post-operative follow-up codes (99024, 99025, 99026, 99027)
- **560 codes in 90000-99999 range are informative** and should be kept as features

### Summary

**Overall Code Classification:**
- **ICD Codes:** 6,249 total codes, 4 administrative (0.06%), 6,245 informative (99.94%)
- **CPT Codes:** 2,773 total codes, 1 administrative (0.04%), 2,772 informative (99.96%)

**Administrative Codes Identified:**
- **ICD:** Z00.00, Z00.01, Z00.121, Z00.129 (routine examination codes)
- **CPT:** 99024, 99025, 99026, 99027 (post-operative follow-up codes)

**Conclusion:** The vast majority of ICD and CPT codes in feature importance are informative and should be kept. Only a small number of administrative codes (routine examinations and post-operative follow-ups) are filtered out.

### Configuration

Administrative codes are maintained in:
- **Lookup Table:** `4b_dtw_filter/administrative_codes_lookup.json`
- **Cohort-Specific Config:** `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_manual_filtering_config.json`
- **Code Group Analysis:** `3b_feature_importance_eda/outputs/{cohort}/{age_band}/code_group_analysis.json`

**See:** `3b_feature_importance_eda/README_feature_importance_eda.md` for details on the filtering workflow.

---

## Usage Examples

### 1. Feature Selection for Downstream ML

```r
# Load aggregated results
features <- read_csv("opioid_ed_25-44_2016_feature_importance_aggregated.csv")

# Strategy 1: Top N features
top_features <- features %>% head(20) %>% pull(feature)

# Strategy 2: Features in both models (high consensus)
consensus_features <- features %>% 
  filter(n_models == 2) %>% 
  head(20) %>% 
  pull(feature)

# Strategy 3: Threshold by importance
important_features <- features %>%
  filter(importance_scaled > 0.5) %>%
  pull(feature)

# Use in CatBoost
train_pool <- catboost.load_pool(
  data = patient_data %>% select(all_of(top_features)),
  label = patient_data$target
)
```

### 2. Compare Cohorts

```r
# Load both cohorts
opioid <- read_csv("opioid_ed_25-44_2016_feature_importance_aggregated.csv")
non_opioid <- read_csv("non_opioid_ed_25-44_2016_feature_importance_aggregated.csv")

# Find common features
common_features <- intersect(
  head(opioid, 50)$feature,
  head(non_opioid, 50)$feature
)

# Find opioid-specific features
opioid_specific <- setdiff(
  head(opioid, 50)$feature,
  head(non_opioid, 50)$feature
)
```

### 3. Validate Model Quality

```r
features <- read_csv("opioid_ed_25-44_2016_feature_importance_aggregated.csv")

# Check model overlap (all 3 core models)
overlap_pct <- 100 * sum(features$n_models == 3) / nrow(features)
cat(sprintf("Model overlap (all 3 models): %.1f%%\n", overlap_pct))

# Check Recall values
cat(sprintf("Mean Recall: %.3f ± %.3f\n",
            mean(features$mc_cv_recall_mean),
            mean(features$mc_cv_recall_std)))

# Top features should make clinical sense
head(features, 20) %>% select(rank, feature, importance_scaled, n_models)
```

---

## Best Practices

### 1. Data Quality

✅ **Do:**
- Remove NA target values before MC-CV
- Verify target distribution (check for class imbalance)
- Check for patient-level target consistency

❌ **Don't:**
- Use event-level data (must aggregate to patient-level)
- Include features that leak target information
- Run without stratified sampling

### 2. Computational Resources

**For DEBUG_MODE = TRUE (5 splits):**
- Any machine (4+ cores)
- ~5 minutes
- Good for testing

**For 100 splits (development):**
- EC2 x2iedn.8xlarge (32 cores, 1TB RAM)
- ~1-2 hours
- Recommended for development

**For 1000 splits (publication):**
- Same EC2 instance
- ~10-20 hours
- Use for final results only

### 3. Feature Count

**Too Many Features (>20k):**
- Consider pre-filtering (e.g., min frequency)
- Use larger `future.globals.maxSize`
- May require more RAM

**Too Few Features (<100):**
- Results may be unstable
- Consider including more data or feature types

### 4. Model Interpretation

✅ **Do:**
- Look at `n_models` column (2 = high confidence)
- Check if top features make clinical sense
- Review Recall values (should be reasonable, e.g., >0.6)

❌ **Don't:**
- Use features ranked 100+ without inspection
- Ignore model-specific features (n_models = 1)
- Trust results without domain validation

---

## EC2 Configuration and Optimizations

### Instance Specifications

**Recommended EC2 Instance:**
- **Type:** `x2iedn.8xlarge` (or equivalent)
- **CPU:** 32 cores
- **RAM:** 1TB
- **Storage:** NVMe SSD (for fast data access)
- **Data Location:** `/mnt/nvme/cohorts/` (or set via `LOCAL_DATA_PATH` environment variable)

### Optional: Writing Step 3 outputs to NVMe

By default, Step 3a writes to `3a_feature_importance/outputs/{cohort}/` under the project root. On EC2 you can send these outputs to NVMe for faster I/O:

```bash
export PGX_FEATURE_IMPORTANCE_OUTPUTS=/mnt/nvme/3a_feature_importance/outputs
```

Then run the Step 3a scripts as usual; CSVs and `plots/` will be written under that path. Downstream steps (3b, Step 4 model data, etc.) expect the same layout; if you use NVMe, either symlink `3a_feature_importance/outputs` to the NVMe path or set `PGX_FEATURE_IMPORTANCE_OUTPUTS` consistently when reading.

### Parallel Processing Configuration

The feature importance pipeline uses a two-level parallelization strategy optimized for 32-core EC2 instances:

#### 1. MC-CV Worker Configuration

**Workers:** 8 workers (configurable in `run_cohort_*.py` scripts)

```python
# In run_cohort_*.py
N_WORKERS = max(1, multiprocessing.cpu_count() - 24)
# On 32-core system: 32 - 24 = 8 workers
```

**Rationale:**
- Leaves 24 cores free for system processes, other tasks, and overhead
- 8 workers process MC-CV splits in parallel
- Each worker handles one split at a time

#### 2. Model Thread Configuration

**Per-Model Threads:** 4 threads per model

**Configuration:**
- **CatBoost:** `thread_count: 4` (in `feature_importance_model_utils.py`)
- **XGBoost:** `n_jobs: 4` (in `feature_importance_model_utils.py`)
- **XGBoost RF:** `n_jobs: 4` (in `feature_importance_model_utils.py`)

**Total CPU Usage:**
- 8 workers × 4 threads = 32 cores fully utilized
- No oversubscription or thread contention

#### 3. Feature Matrix Building Optimization

**Batching Strategy:** Columns are processed in batches to reduce joblib overhead

**Configuration:**
- **Workers:** `min(16, max(1, multiprocessing.cpu_count() - 2))` (16 workers on 32-core system, capped to avoid oversubscription)
- **Batch Size:** Automatically calculated as `items_per_worker * 4` (~4 batches per worker)
- **Purpose:** Reduces process spawning overhead by processing multiple columns per worker

**Example:** For 4,962 features with 16 workers:
- Batch size: ~77 columns per batch
- Total batches: ~65 batches
- Each worker processes multiple batches sequentially

**Verification:**
Check logs for:
```
Feature matrix parallel workers: 16 (CPU count: 32), batch size: 77, batches: 65
```

### Monitoring Parallelization

#### Check Worker Count

```bash
# Count Python processes
# - During feature matrix building: ~16+ processes (one per worker)
# - During MC-CV execution: ~8 processes (one per MC-CV worker)
ps aux | grep python3.11 | grep -v grep | wc -l

# Check threads per process
ps -p $(pgrep -f "run_cohort") -o pid,pcpu,pmem,nlwp,cmd
```

#### Check CPU Utilization

```bash
# Per-core CPU usage (should see 8-16 cores active at 50-80%)
mpstat -P ALL 1 5

# Overall CPU usage
top -bn1 | grep "^%Cpu"
```

**Expected Behavior:**
- **Feature Matrix Building:** 16 workers active, 8-16 cores at 50-80% CPU (with batching, single-threaded per worker)
- **MC-CV Training:** 8 workers active, 8 cores at 80-100% CPU (each worker uses 4 threads internally)
- **Idle:** Most cores idle during I/O or single-threaded operations

#### Troubleshooting Low CPU Usage

**If only 1-2 cores are active:**

1. **Check if batching is working:**
   ```bash
   # Look for batch size in logs
   grep "batch size" /path/to/log/file
   ```

2. **Verify joblib is spawning workers:**
   ```bash
   # Should see ~16 Python processes during feature matrix building
   # Should see ~8 Python processes during MC-CV execution
   ps aux | grep python3.11 | grep -v grep | wc -l
   ```

3. **Check for bottlenecks:**
   ```bash
   # Memory usage
   free -h
   
   # I/O wait
   iostat -x 1 3
   ```

**Common Issues:**
- **Too many workers:** Feature matrix workers are capped at 16; if memory is constrained, reduce further
- **Too few workers:** Check that `multiprocessing.cpu_count()` returns correct value (should be 32 on EC2)
- **I/O bound:** Feature matrix building may be limited by disk speed

### Model Configuration

**Estimator Settings (for 32-core EC2):**

```python
MODEL_PARAMS = {
    'catboost': {
        'iterations': 500,  # CatBoost processes fast, can use more iterations
        'learning_rate': 0.1,
        'depth': 6,
        'thread_count': 4,  # Set in feature_importance_model_utils.py
    },
    'xgboost': {
        'n_estimators': 250,  # Balanced for speed/quality
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_jobs': 4,  # Set in feature_importance_model_utils.py
    },
    'xgboost_rf': {
        'n_estimators': 250,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_jobs': 4,  # Set in feature_importance_model_utils.py
    },
}
```

### CatBoost and Rare Variants

CatBoost is the primary model used for **high-cardinality, sparse healthcare features** (e.g., ICD/CPT codes, medications) and has built-in mechanisms that handle **rare variants** in a statistically principled way.

**How CatBoost encodes rare categories:**

- CatBoost uses **ordered target statistics** (a form of target encoding) for categorical features instead of simple one-hot encoding.
- For each category, it computes a smoothed estimate of the target rate using a **prior + data-driven update**:
  - Common categories are driven mostly by their observed outcomes.
  - Very rare categories are **shrunk toward the global mean**, which reduces overfitting to noise while still allowing strong, consistent rare signals to stand out.
- The ordered construction (permutation-based) prevents **target leakage** while preserving signal from low-frequency codes.

**Implications for feature importance:**

- Rare variants are **not dropped**; they are encoded and can drive splits when they have a meaningful association with the outcome.
- Extremely rare, noisy categories are intentionally **regularized**, so they do not dominate feature importance purely due to chance.
- In the aggregated, multi-model feature-importance analysis, CatBoost plays a key role in:
  - Capturing signal from **high-cardinality categorical structure**, including rare but real patterns.
  - Providing a complementary view to XGBoost / XGBoost RF, which operate on binary 0/1 encodings.

### Random Forest / XGBoost RF and Complementary Signals

In prior work, we observed that **random forest–style models** sometimes surface clinically plausible features that **do not appear in the top ranks of boosted models** (including CatBoost).

**Why we include XGBoost RF in the core ensemble:**

- Random forest / XGBoost RF rely on **bagging + randomized splits**, which makes them:
  - More tolerant of **idiosyncratic but real patterns** that appear strongly in some subsamples but are not globally dominant.
  - Sometimes better at surfacing additional, **model-specific but clinically meaningful** predictors.
- By taking the **union of top features across CatBoost, XGBoost, and XGBoost RF**, and tracking `n_models`:
  - Features found by **all three models** are treated as highly robust.
  - Features found **only by RF/XGBoost RF** are still retained as candidates, instead of being silently discarded because CatBoost did not select them.
- This design choice explicitly reflects the empirical finding that **random-forest style models can identify important features that boosting alone may miss**, and we want our final feature-importance results to capture that broader signal space.

### XGBoost Tree Method and Rare Variants

By design, **XGBoost in this feature-importance pipeline uses the exact tree-growing method** (i.e., we do **not** enable `tree_method="hist"`), even though histogram-based trees are substantially faster on large tabular datasets.

**Rationale:**

- Healthcare data naturally contains many **rare but clinically important variants** (e.g., infrequent CPT/ICD codes or medication patterns).
- Histogram-based methods group feature values into bins before evaluating splits. This can make it harder for the model to **isolate very low-frequency patterns**, because rare values may be merged into bins dominated by common values.
- Using the exact method preserves the **full split resolution** on these sparse, high-cardinality features, which is important for our goal of **maximizing visibility of rare yet meaningful predictors** in the feature-importance analysis.

**Trade-off (explicitly accepted):**

- **Pros:** Better ability to detect and rank rare variants in XGBoost / XGBoost RF feature importance.
- **Cons:** **Longer runtime** (e.g., ~11–12 hours for large cohorts such as `opioid_ed, 25–44` with 25 MC-CV splits and 3 core models).

We explicitly **accept this runtime cost** in order to:

- Preserve sensitivity to rare variants.
- Keep XGBoost / XGBoost RF as a complementary lens to CatBoost in the **robust, multi-model feature-importance ensemble**.

If future use cases prioritize throughput over rare-variant sensitivity, `tree_method="hist"` (with a larger `max_bin`) can be considered as an alternative configuration, but that is **not** the default for the primary publication-quality runs described here.
```

### Performance Expectations

**Large Cohort (e.g., opioid_ed, 25-44 age band):**
- **Patients:** ~78,000 training, ~50,000 test
- **Features:** ~5,000 (after pruning)
- **MC-CV Splits:** 25-100
- **Expected Time:** 1.5-2 hours

**Breakdown:**
- Data loading: ~10 seconds
- Feature engineering: ~20 seconds
- Feature matrix building: ~15-20 minutes (with batching)
- MC-CV execution: ~60-90 minutes (25 splits × 3 models)
- Aggregation: ~2-3 minutes

**Smaller Cohorts:**
- Proportionally faster
- Feature matrix building scales with number of features
- MC-CV time scales with number of splits

### Opioid_ed Age-Band Size and Runtime Scaling (N_SPLITS = 25)

Using the **raw cohort parquet files** in `data/cohorts_F1120/cohort_name=opioid_ed/` (train = 2016–2018, test = 2019), we can summarize **both event workload and distinct patients** per age band:

- **Event-level row counts (workload):**
  - **0–12**: train = 2,186, test = 1,936  
  - **13–24**: train = 435,982, test = 176,151  
  - **25–44**: train = 4,651,487, test = 3,044,733  
  - **45–54**: train = 2,770,352, test = 1,382,862  
  - **55–64**: train = 3,231,509, test = 1,392,618  
  - **65–74**: train = 2,857,618, test = 1,015,348  
  - **75–84**: train = 1,227,068, test = 370,364  
  - **85–94**: train = 274,315, test = 96,795  
  - **95–114**: train = 10,918, test = 2,754  

- **Distinct patients:**
  - **0–12**: train = 78, test = 66  
  - **13–24**: train = 9,834, test = 3,840  
  - **25–44**: train = 78,296, test = 50,400  
  - **45–54**: train = 32,070, test = 16,950  
  - **55–64**: train = 31,507, test = 14,898  
  - **65–74**: train = 23,356, test = 9,150  
  - **75–84**: train = 8,477, test = 2,976  
  - **85–94**: train = 1,878, test = 726  
  - **95–114**: train = 77, test = 24  

If we take `opioid_ed 25–44` as the **baseline** (factor = 1.0 for `(train + test)` event rows), the **relative size factors** for event workload are:

- **0–12**: ≈ 0.001×  
- **13–24**: ≈ 0.08×  
- **25–44**: 1.00× (baseline)  
- **45–54**: ≈ 0.54×  
- **55–64**: ≈ 0.60×  
- **65–74**: ≈ 0.50×  
- **75–84**: ≈ 0.21×  
- **85–94**: ≈ 0.05×  
- **95–114**: ≈ 0.002×  

Since MC‑CV + permutation importance cost is dominated by the number of **rows** processed per split, **wall-clock runtime for a fixed configuration (25 splits, 3 models, exact XGBoost)** scales roughly with these factors. Concretely, if `opioid_ed 25–44` takes **~11–12 hours**, then:

- **13–24** should be on the order of **~1 hour** (0.08×).  
- **45–54 / 55–64 / 65–74** should each be around **~5–7 hours** (0.5–0.6×).  
- **75–84** should be **~2–3 hours** (0.2×).  
- **0–12, 85–94, 95–114** should complete in **minutes to well under an hour**, given their tiny relative size, even though they still contain clinically meaningful patient cohorts.

### Non_Opioid_ed (Polypharmacy) Age-Band Size and Runtime Scaling (N_SPLITS = 25)

For the polypharmacy ED cohort (`cohort_name=non_opioid_ed`), which is the primary focus for **older age bands** in this feature-importance analysis, the cohort parquet files in `data/cohorts_F1120/cohort_name=non_opioid_ed/` have substantially larger event workloads and patient counts:

- **Event-level row counts (workload), train = 2016–2018, test = 2019:**
  - **0–12**: train = 32,482,174, test = 13,095,946  
  - **13–24**: train = 30,064,091, test = 12,717,593  
  - **25–44**: train = 70,326,824, test = 29,280,711  
  - **45–54**: train = 52,120,942, test = 20,750,036  
  - **55–64**: train = 71,132,187, test = 29,816,173  
  - **65–74**: train = 135,465,040, test = 50,047,383  
  - **75–84**: train = 87,267,781, test = 32,780,611  
  - **85–94**: train = 35,670,313, test = 12,278,221  
  - **95–114**: train = 3,219,193, test = 1,185,156  

- **Distinct patients, train = 2016–2018, test = 2019:**
  - **0–12**: train = 1,215,320, test = 870,021  
  - **13–24**: train = 954,442, test = 696,076  
  - **25–44**: train = 1,542,990, test = 1,168,512  
  - **45–54**: train = 831,967, test = 630,978  
  - **55–64**: train = 884,664, test = 713,998  
  - **65–74**: train = 919,654, test = 766,298  
  - **75–84**: train = 462,222, test = 391,003  
  - **85–94**: train = 181,679, test = 136,146  
  - **95–114**: train = 21,546, test = 14,729  

If we take `non_opioid_ed 65–74` as the **baseline** (largest `(train + test)` event workload), the **relative event workload factors** are:

- **0–12**: ≈ 0.25×  
- **13–24**: ≈ 0.23×  
- **25–44**: ≈ 0.54×  
- **45–54**: ≈ 0.39×  
- **55–64**: ≈ 0.54×  
- **65–74**: 1.00× (baseline)  
- **75–84**: ≈ 0.65×  
- **85–94**: ≈ 0.26×  
- **95–114**: ≈ 0.02×  

For the same MC‑CV configuration (25 splits, 3 tree models, exact XGBoost), this implies that **non_opioid_ed 65–74 is the heaviest polypharmacy age band** by event workload, with **55–64 and 75–84** in a similar runtime regime, and younger / extreme-age bands contributing a smaller fraction of total compute despite very large patient counts.

### Performance Monitoring and Expected Behavior

When running on EC2 (32 cores, 1TB RAM), you should observe the following performance characteristics:

#### Expected System Metrics

**During MC-CV Execution (8 workers active):**

```bash
# CPU Usage
%Cpu(s): 70.4 us,  0.8 sy,  0.0 ni, 28.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st

# Load Average
load average: 15.88, 16.40, 16.79

# Memory Usage (per process)
MiB Mem: ~70GB per Python process (7% of 1TB total)
```

**Process Characteristics:**

- **8 Python processes** running in parallel (one per MC-CV worker)
- Each process using **200-370% CPU** (indicating 4 threads per process)
- Each process using **~70GB RAM** (7% of total system memory)
- **Total CPU utilization:** ~70% (good utilization without oversubscription)
- **Load average:** ~16 (reasonable for 32 cores under heavy computation)

#### Configuration Breakdown

**Worker Configuration:**
- **MC-CV Workers:** 8 workers (`multiprocessing.cpu_count() - 24 = 8`)
- **Per-Model Threads:** 4 threads per model (CatBoost/XGBoost `thread_count=4` / `n_jobs=4`)
- **Total Threads:** 8 workers × 4 threads = 32 threads (matches 32 cores)

**What's Happening:**
- 8 parallel MC-CV workers processing different splits simultaneously
- Each worker trains 3 models sequentially per split (CatBoost, XGBoost, XGBoost RF)
- Each model uses 4 threads internally, explaining the 200-370% CPU per process
- The 70% CPU usage indicates healthy utilization with some I/O wait (data loading, model serialization)

#### Runtime Estimates

**For Large Cohort (opioid_ed, 25-44 age band, 25 MC-CV splits):**

- **Total Runtime:** ~11-12 hours
- **Breakdown:**
  - Feature matrix building: ~2.5-3 minutes (one-time, parallelized)
  - MC-CV execution: ~11 hours
    - 25 splits × 3 models = 75 model training tasks
    - ~8-9 minutes per model training task
    - Parallelized across 8 workers

**Status Indicators:**

✅ **Good Performance:**
- 8 Python processes visible in `top`
- CPU usage 60-80%
- Load average 12-20
- Memory usage stable (~70GB per process)
- Processes running for expected duration

⚠️ **Potential Issues:**

**During MC-CV Execution (should see 8 processes):**
- Only 1-2 processes visible → Check `N_WORKERS` configuration (should be 8 on 32-core system)
- CPU usage <50% → May indicate I/O bottleneck or insufficient parallelization
- Load average >32 → Oversubscription, reduce `N_WORKERS` or model thread counts
- Memory usage growing → Potential memory leak

**During Feature Matrix Building (should see ~16 processes):**
- Only 1-2 processes visible → Check `n_workers_matrix` configuration (should be 16 on 32-core system)
- CPU usage <30% → May indicate I/O bottleneck or insufficient parallelization
- Load average >32 → Oversubscription, check if cap at 16 is working correctly
- Memory usage growing → Potential memory leak

**General:**
- Processes stuck at same CPU% for >30 minutes → May indicate deadlock or infinite loop
- Memory usage >200GB → May indicate memory leak or insufficient feature pruning

#### Monitoring Commands

```bash
# Check running processes
ps aux | grep python3.11 | grep run_cohort | grep -v grep

# Check CPU and memory per process
ps -p $(pgrep -f "run_cohort") -o pid,pcpu,pmem,nlwp,cmd

# Check overall CPU usage
top -bn1 | grep "^%Cpu"

# Check per-core CPU usage
grep "cpu[0-9]" /proc/stat | awk '{print $1": "$2+$3+$4+$5+$6+$7+$8+$9+$10+$11}'

# Monitor load average
uptime
```

### Best Practices for EC2

1. **Monitor Resource Usage:**
   - Use `htop` or `top` to watch CPU/memory
   - Check logs for parallelization messages
   - Verify all cores are being utilized

2. **Memory Management:**
   - Large cohorts may use 10-20GB RAM
   - Ensure sufficient swap space if needed
   - Monitor for memory leaks in long-running jobs

3. **Auto-Shutdown:**
   - Set up EC2 auto-shutdown after job completion
   - Save costs by stopping instance when not in use

4. **Logging:**
   - Logs are automatically saved to S3
   - Check logs for parallelization confirmation
   - Monitor for errors or warnings

---

## References

- **Permutation Importance:** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- **Monte Carlo Cross-Validation:** Picard, R. R., & Cook, R. D. (1984). Cross-validation of regression models. JASA.
- **Model Ensembling:** Dietterich, T. G. (2000). Ensemble methods in machine learning. MCS 2000.

---

## Related Documentation

- **Main Notebook:** `feature_importance_mc_cv.ipynb`
- **Visualization Script:** `create_visualizations.R`
- **S3 Output Structure:** `S3_OUTPUT_STRUCTURE.md`
- **rsample Bug:** `docs/RSAMPLE_BUG_WORKAROUND.md`

---

**Questions or Issues?** See main project README or open an issue.

