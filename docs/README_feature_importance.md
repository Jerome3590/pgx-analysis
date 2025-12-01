# Feature Importance Analysis

**Date:** November 29, 2025  
**Project:** PGx Analysis - Feature Importance with Monte Carlo Cross-Validation  
**Notebook:** `feature_importance_mc_cv.ipynb`

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
   - [Local Testing](#local-testing-5-splits-5-minutes)
   - [Production Run](#production-run-100-splits-1-2-hours-on-ec2)
   - [Parallel Execution](#parallel-execution-default)
   - [Single Cohort Execution](#single-cohort-execution-optional)
3. [Methodology](#methodology)
4. [Aggregation Method](#aggregation-method)
5. [Output Files](#output-files)
6. [Visualization](#visualization)
7. [Cross-Age-Band Analysis](#cross-age-band-analysis)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This project calculates scaled feature importance for predicting opioid dependence using:
- **Models (core ensemble):** CatBoost, XGBoost (boosted trees), XGBoost RF mode
- **Validation:** Monte Carlo Cross-Validation (100–1000 splits) with temporal validation
- **Scaling:** Permutation-based importance weighted by model Recall
- **Aggregation:** Union of top features from each model with summed importances

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

### Key Features

✅ **Monte Carlo Cross-Validation** – Up to 1000 random train/test splits  
✅ **Temporal Validation** – Train on 2016-2018, test on 2019 (avoids COVID year 2020)  
✅ **Stratified Sampling** – Maintains target distribution  
✅ **Parallel Processing** – Fast execution with **conservative worker counts** (see below)  
✅ **Quality Weighting** – Features scaled by model performance (Recall)  
✅ **Model Consensus** – Union-based aggregation rewards agreement  
✅ **Multiple Models** – Tree ensembles (CatBoost, RF, XGBoost, LightGBM, ExtraTrees) and linear models (LogisticRegression, LinearSVC, ElasticNet, LASSO)  
✅ **Publication-Ready Plots** – 4 visualization types with S3 upload

---

## Quick Start

### Local Testing (5 splits, ~5 minutes)

```python
# In feature_importance_mc_cv_python.ipynb
DEBUG_MODE = True
TRAIN_YEARS = [2016, 2017, 2018]  # Training data years
TEST_YEAR = 2019  # Test data year (never used for training)

# Run all cells
```

**Note:** The Python notebook uses temporal validation (train on 2016-2018, test on 2019). The R notebook (`feature_importance_mc_cv.ipynb`) uses single-year splits and should be updated to match this strategy.

## Prerequisites (on EC2)

From your EC2 instance:

```bash
cd /home/ec2-user/pgx-analysis   # or your actual clone path

# 1) Ensure cohort parquet data is local on NVMe
aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ /mnt/nvme/cohorts/ \
  --exclude "*.log" --exclude "*.json"

# 2) Activate your analysis environment (example)
conda activate pgx-analysis   # or source your venv

# 3) Start Jupyter
jupyter notebook
```

Then open this notebook: `3_feature_importance/run_feature_importance_cohorts.ipynb`.

### Production Run (25–50 splits on EC2)

```python
DEBUG_MODE = False
N_SPLITS = 25  # current default for cohort-level feature screening
TRAIN_YEARS = [2016, 2017, 2018]  # Training data years
TEST_YEAR = 2019  # Test data year (never used for training)

# Set up EC2:
# - x2iedn.8xlarge (32 cores, 1TB RAM)
# - Data in /mnt/nvme/cohorts/
# - Auto-shutdown enabled
```

**Temporal Validation:** Each MC-CV split samples from 2016-2018 training data, but all splits evaluate on the same 2019 test set. This ensures robust feature importance estimates while maintaining temporal integrity.

**Splits Sensitivity Check:** For the opioid_ed cohort we explicitly compared runs with **10, 20, 30, and 50 MC‑CV splits** and observed no material changes in the leading feature rankings—only minor reordering in the long tail. Based on this, we use **25 splits** as the default for **feature screening** (to filter out weak/noisy features and reduce model complexity for downstream FP‑Growth, bupaR, and DTW). For the **final ensemble and FFA analyses**, we reserve higher split counts (e.g., 50+) when tighter confidence bands on feature importance are required.

**Feature-Matrix Pruning (Python MC‑CV helpers):** To keep MC‑CV tractable on large cohorts, we prune ultra‑rare items *before* building feature matrices. For each `(cohort, age_band)`:

- Build a patient‑item table from drugs, ICD codes, CPT/procedure codes, and event types (excluding non‑informative tokens like `"pharmacy"`, `"medical"`, and the target code `F1120`).
- Count, for each item, how many **distinct training patients** (2016–2018) have that item.
- Keep only items that appear in **at least 25 patients**; drop items below that threshold from both CatBoost and RF/XGBoost feature spaces.

This reduces the raw feature count from \~30k+ items per cohort/age band to a few thousand (e.g., 11k → ~1.1k for `opioid_ed 13–24`, 32k → ~5k for `opioid_ed 25–44`), dramatically shrinking the CatBoost and RF/XGBoost matrices while preserving clinically meaningful, sufficiently frequent codes.

### Parallel Execution (Default)

The notebook (Cell 5) runs all combinations defined in `COHORT_NAMES` × `AGE_BANDS` in parallel. Each task processes one cohort/age-band combination using the `run_cohort_analysis()` function.

**Configuration:**
- Automatically handles multiple cohorts and age-bands
- Idempotent: Skips already-processed combinations (checks local files and S3)
- Nested parallelism: Optimizes worker allocation between task-level and MC-CV level
- Cross-age-band aggregation: Only runs when all combinations are complete

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
│ 1. Load Cohort Data (parquet)                              │
│    - Training: Years 2016-2018 (combined)                  │
│    - Test: Year 2019 (holdout, never used for training)    │
│    - Excluded: Year 2020 (COVID-19 pandemic)               │
│    - Drugs, ICD codes, CPT codes                           │
│    - Target: is_target_case (opioid dependence)            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Engineering                                      │
│    - Patient-level aggregation                             │
│    - CatBoost: Categorical factors                         │
│    - Random Forest/XGBoost: Binary 0/1                     │
│    - Consistent feature space across train/test             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Monte Carlo Cross-Validation (100–1000 splits)          │
│    - Each split samples from 2016-2018 training data        │
│    - All splits evaluate on same 2019 test set              │
│    ┌────────────────────┐  ┌────────────────────┐         │
│    │   Multiple Models  │  │  Multiple Models   │         │
│    │   (CatBoost, RF,   │  │  (XGBoost, LGBM,   │         │
│    │    XGBoost, etc.)  │  │   ExtraTrees, etc.)│         │
│    │                    │  │                    │         │
│    │  Per split:        │  │  Per split:        │         │
│    │  - Train: Sample   │  │  - Train: Sample   │         │
│    │    from 2016-2018  │  │    from 2016-2018  │         │
│    │  - Test: Always    │  │  - Test: Always    │         │
│    │    2019 (same)     │  │    2019 (same)     │         │
│    │  - Recall          │  │  - Recall          │         │
│    │  - Feature imp     │  │  - Feature imp     │         │
│    └────────────────────┘  └────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Union-Based Aggregation                                  │
│    - Top 50 from CatBoost                                   │
│    - Top 50 from Random Forest                              │
│    - Scale by Recall                                        │
│    - SUM where overlap                                      │
│    - Rank by summed importance                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Outputs                                                   │
│    - Aggregated CSV (final rankings)                        │
│    - Visualizations (4 plots)                               │
│    - S3 upload                                              │
└─────────────────────────────────────────────────────────────┘
```

### Models

**CatBoost:**
- Handles categorical features natively
- Feature format: Each column is a factor with item name as level
- Importance: Permutation-based (PredictionValuesChange)
- For feature-importance **screening runs**, we allow CatBoost to use a somewhat larger tree budget (more iterations) than XGBoost / XGBoost RF. This is acceptable because the goal here is robust feature discovery, not a strict algorithm bake-off.

**XGBoost (boosted trees):**
- Gradient boosting with tree-based learners
- Feature format: Binary 0/1 encoding
- Importance: Permutation-based

**XGBoost RF mode:**
- Random-forest style XGBoost configuration
- Feature format: Binary 0/1 encoding
- Importance: Permutation-based

All three models use **permutation-based importance** for fair comparison. In the **final model evaluation and deployment pipeline** (see `7_final_model/`), we align key hyperparameters such as the number of trees (e.g., `iterations` / `n_estimators`) across CatBoost and XGBoost-family models when we want an apples-to-apples comparison of model performance. Here, in the feature-importance stage, we prioritize stability and coverage of signal for downstream PGx engineering, accepting small asymmetries in tree counts between models to keep wall-clock and memory usage manageable.

### Runtime and Performance Considerations

Because this pipeline is designed for **health data** and downstream causal/clinical interpretation, the Monte Carlo feature importance step is intentionally **computationally heavy**:

- **Data scale (per cohort/age_band):**
  - \~10,000–15,000 patients in the 2016–2018 training window
  - \~3,000–5,000 patients in the 2019 holdout
  - \~10,000–15,000 binary/categorical features after feature engineering
- **MC-CV configuration:**
  - `N_SPLITS = 50` Monte Carlo splits per model (per cohort, per age band)
  - For each split:
    - Train on an 80% sample of **2016–2018** patients
    - Evaluate on the **full 2019 holdout** (no subsampling)
- **Permutation importance:**
  - For every split and every model, we run **permutation-based importance** on the **entire 2019 holdout**
  - With `n_repeats = 5` and \~10k features, this implies on the order of:
    - 10,000 features × 5 permutations = 50,000 perturbations
    - Each perturbation requires a full prediction over all 2019 patients
  - This is repeated across **50 splits × 3 models (CatBoost, XGBoost, XGBoost RF)**.

In practice this means:

- It is **normal** to see logs like:
  - `[Parallel(n_jobs=30)]: Done   3 out of 50 | elapsed: 338.1min ...`
  - This reflects the fact that each XGBoost split with full permutation importance can take **tens of minutes** on large cohorts, even with 30 workers.
- The cost is buying you:
  - Stable **mean recall/log-loss** and standard deviations across many temporal resamples
  - Feature importance estimates that are **robust to train-sample perturbations**
  - Consensus importance across three strong models with different inductive biases

If you need to reduce runtime while retaining robustness, the **safest levers** are:

- Keep `N_SPLITS = 50`, but:
  - Reduce permutation `n_repeats` from 5 → 2–3 in `get_permutation_importance`, or
  - Pre-filter to a smaller candidate feature set (e.g., minimum frequency) **after** verifying this does not materially change the top features.
- Keep all three models, but use lighter MC-CV settings:
  - e.g., `n_estimators: 200` for XGBoost / XGBoost RF during feature importance,
  - Retain larger `n_estimators` only for the **final model training** in `7_final_model/`.

We do **not** recommend dropping to a single model or a handful of splits for publication‑grade health analyses; the current defaults intentionally favor **stability and reproducibility** over wall-clock speed.

#### Concrete Timing Example

An example from a real run (single cohort/age_band, XGBoost MC‑CV only):

- **Start of MC‑CV (XGBoost):**
  - `2025-11-28 06:08:27,123 - INFO - Running MC-CV for xgboost...`
- **Mid-run joblib progress logs:**
  - `[Parallel(n_jobs=30)]: Done   3 out of  50 | elapsed: 338.1min remaining: 5296.9min`
  - `[Parallel(n_jobs=30)]: Done  27 out of  50 | elapsed: 384.9min remaining: 327.9min`
- **Interpretation:**
  - By 27 completed splits, total elapsed time was \~385 minutes (\~6.4 hours).
  - Estimated remaining time for the last 23 splits was \~328 minutes (\~5.5 hours).
  - Total expected wall-clock time for **50 XGBoost splits with full permutation importance** on this cohort was \~12 hours (from \~06:08 to \~18:00 on the same day).

These numbers are **in line with expectations** for:

- Large per-patient feature matrices (\~10k+ features),
- 50 Monte Carlo splits,
- Full permutation importance over all features on the entire 2019 holdout,
- Three-core-model ensemble (CatBoost + XGBoost + XGBoost RF) repeated across multiple cohorts and age bands.

### Cohort Focus for Full MC-CV Runs

To keep runtimes tractable while preserving robustness for health analyses, we **focus full MC‑CV + permutation importance on specific cohort groups** rather than the entire cohort × age-band grid:

- **Cohort Group 1 – Opioid ED focus (`opioid_ed`)**
  - Primary goal: detailed feature discovery around opioid‑related ED visits.
  - Age-band “cohorts” 1–5 (you can interpret these as the younger to mid‑age bands in `AGE_BANDS`).
  - These receive the *full* 3‑model MC‑CV treatment (CatBoost, XGBoost, XGBoost RF) with 50 splits and full permutation importance.

- **Cohort Group 2 – Polypharmacy ED visits (non‑opioid ED focus)**
  - Primary goal: detailed feature discovery around polypharmacy‑related ED patterns in older adults.
  - Age-band “cohorts” 6–8 (you can interpret these as the older age bands in `AGE_BANDS`).
  - These also receive full MC‑CV + permutation importance, but are conceptually treated as a separate analysis program.

Other `(cohort, age_band)` combinations can still be run with lighter settings (fewer splits, fewer models, or restricted feature sets), but the **publication‑grade, health‑critical analysis is concentrated in these two cohort groups**.

### Opioid_ed Age-Band Size and Expected Runtime (N_SPLITS = 25)

For the `opioid_ed` cohort using **2016–2018 as training** and **2019 as test**, the underlying cohort parquet files in `data/cohorts_F1120/cohort_name=opioid_ed/` give us both **event workload** and **distinct patient counts**:

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

Taking `opioid_ed 25–44` as a **baseline** (factor = 1.0 for `(train + test)` event rows), the **relative size factors** are approximately:

- **0–12**: ≈ 0.001×  
- **13–24**: ≈ 0.08×  
- **25–44**: 1.00× (baseline)  
- **45–54**: ≈ 0.54×  
- **55–64**: ≈ 0.60×  
- **65–74**: ≈ 0.50×  
- **75–84**: ≈ 0.21×  
- **85–94**: ≈ 0.05×  
- **95–114**: ≈ 0.002×  

Since MC‑CV + permutation importance cost is dominated by the number of **rows** processed per split, **wall‑clock runtime for a fixed configuration (25 splits, 3 models, exact XGBoost)** scales roughly with these factors. If `opioid_ed 25–44` takes **~11–12 hours**, then:

- **13–24** is expected to take **~1 hour** (0.08×).  
- **45–54 / 55–64 / 65–74** are expected to take **~5–7 hours** each (0.5–0.6×).  
- **75–84** is expected to take **~2–3 hours** (0.2×).  
- **0–12, 85–94, 95–114** should complete in **minutes to well under an hour**, even though they still contain clinically meaningful patient cohorts.

---

## Aggregation Method

### Step-by-Step Process

#### 1. Train Models with MC-CV

For each model type (CatBoost, Random Forest):
- Create 100–1000 stratified Monte Carlo cross-validation splits
- Train model on each training set (80%)
- Evaluate Recall on each test set (20%)
- Extract **permutation-based feature importance** for each split
- Aggregate across splits to get:
  - `importance_normalized` - Feature importance normalized to [0, 1]
  - `mc_cv_recall_mean` - Mean Recall across all splits
  - `mc_cv_recall_std` - Standard deviation of Recall

#### 2. Select Top 50 from Each Model

For each model:
- Rank features by `importance_normalized` (permutation-based)
- Select **top 50 features**
- Scale importance by MC-CV Recall:

```r
importance_scaled = importance_normalized × mc_cv_recall_mean
```

#### 3. Union of Features

- Take the **union** of top 50 from CatBoost and top 50 from Random Forest
- Results in up to 100 features (if no overlap) or as few as 50 (if complete overlap)

#### 4. Sum Importances for Overlapping Features

For each feature in the union:

**If feature appears in both models:**
- Sum the scaled importances: `importance_scaled = catboost_scaled + rf_scaled`
- Sum the normalized importances: `importance_normalized = catboost_norm + rf_norm`
- Average the Recall: `mc_cv_recall_mean = (catboost_recall + rf_recall) / 2`

**If feature appears in only one model:**
- Use that model's values directly

#### 5. Rank by Summed Scaled Importance

- Sort features by `importance_scaled` (descending)
- Assign rank (1 = most important)

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

#### ⚠️ Considerations

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
| `importance_normalized` | Sum of normalized importances | 0.0 – 2.0 |
| `importance_scaled` | Sum of Recall-scaled importances | 0.0 – ~1.6 |
| `n_models` | Number of models including feature | 1 or 2 |
| `models` | Which models | "catboost", "random_forest", or both |
| `mc_cv_recall_mean` | Average Recall across models | 0.0 – 1.0 |
| `mc_cv_recall_std` | Recall std dev | 0.0 – 1.0 |

**Key Metric:** `importance_scaled` - Used for final ranking and visualization.

**Example:**
```csv
rank,feature,importance_normalized,importance_scaled,n_models,models,mc_cv_recall_mean,mc_cv_recall_std
1,HYDROCODONE-ACETAMINOPHEN,1.8234,1.5012,2,"catboost, random_forest",0.8234,0.0156
2,TRAMADOL HCL,1.6821,1.3856,2,"catboost, random_forest",0.8234,0.0156
3,F11.20,0.9234,0.7603,1,catboost,0.8234,0.0156
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
- Local: `outputs/plots/`
- S3: `s3://pgxdatalake/gold/feature_importance/cohort_name={cohort}/age_band={age}/event_year={year}/plots/`

---

## Cross-Age-Band Analysis

After running feature importance for multiple age bands, create comparison heatmaps:

```r
source("create_cross_ageband_heatmap.R")

create_ageband_heatmap(
  cohort_name = "opioid_ed",
  event_year = 2016,
  age_bands = c("13-24", "25-44", "45-54", "55-64", "65-74"),
  top_n = 50
)
```

**Outputs:**
- Heatmap: Features × Age bands (color = importance)
- Summary CSV: Variability metrics (CV, consistency)
- Insights: Universal vs age-specific features

**Use Cases:**
- Identify universal risk factors (low CV)
- Find age-specific features (high CV)
- Decide between age-agnostic vs age-stratified models

**See:** `README_CROSS_AGEBAND_ANALYSIS.md` for details

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

# Check model overlap
overlap_pct <- 100 * sum(features$n_models == 2) / nrow(features)
cat(sprintf("Model overlap: %.1f%%\n", overlap_pct))

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

**For DEBUG_MODE = TRUE (5 splits, Python MC‑CV helpers):**
- Any machine (4+ cores)
- A few minutes per cohort/age band
- Good for functional tests

**For 25–50 splits (development / screening, Python MC‑CV helpers):**
- EC2 x2iedn.8xlarge (32 vCPUs, 1TB RAM) or comparable high‑RAM instance.
- Python `run_cohort_*.py` scripts currently set:
  - `N_SPLITS = 25` for cohort‑level feature screening.
  - `N_WORKERS = max(1, multiprocessing.cpu_count() - 12)` so that, on a 20‑vCPU machine, **8 workers** are used for MC‑CV (`20 - 12 = 8`) to reduce OOM risk.
- Expect **1–3 hours** per heavy cohort/age band under these settings, depending on feature count and prevalence.

**For 50+ splits (final, publication‑grade runs):**
- Same class of instance, but consider further reducing `N_WORKERS` or tightening the item‑frequency threshold if memory pressure is high.

### 3. Feature Count

**Too Many Features (>20k):**
- The Python MC‑CV pipeline now **automatically pre‑filters** very rare items by requiring that each drug / ICD / CPT / event token appear in **≥ 25 training patients** before it is included as a feature.
- This keeps the effective feature count in the **2k–5k** range per cohort/age band instead of 20k+ and significantly reduces CatBoost / XGBoost memory footprint.

**Too Few Features (<100):**
- Results may be unstable
- Consider including more data or feature types

### 4. Model Interpretation

✅ **Do:**
- Look at `n_models` column (2 = high confidence)
- Check if top features make clinical sense
- Review Recall values (should be reasonable, e.g., >0.6)
- Compare across age bands for consistency

❌ **Don't:**
- Use features ranked 100+ without inspection
- Ignore model-specific features (n_models = 1)
- Trust results without domain validation

---

## Troubleshooting

### Issue: "test_idx is empty after removing NAs"

**Cause:** `rsample::mc_cv()` bug with NA targets

**Fix:** Already implemented - NA targets removed before MC-CV. See `docs/RSAMPLE_BUG_WORKAROUND.md`

### Issue: "future.globals.maxSize exceeded"

**Cause:** Feature matrix too large for parallel processing

**Fix:**
```r
options(future.globals.maxSize = 97 * 1024^3)  # 97 GB
```

### Issue: Low Recall values (<0.5)

**Possible causes:**
- Severe class imbalance
- Features don't predict target well
- Model hyperparameters need tuning

**Actions:**
- Check target distribution
- Review feature engineering
- Try different model parameters

### Issue: No overlap between models (all n_models = 1)

**Possible causes:**
- Models finding different patterns (may be valid)
- Different feature representations (CatBoost vs RF)
- Very noisy data

**Actions:**
- Review per-model CSVs
- Check if features make sense
- Consider using only one model

### Issue: OOM error during execution

**Causes:**
- Too many features
- Too many workers
- Insufficient RAM

**Fixes:**
```r
# Reduce workers
N_WORKERS <- 15  # instead of 30

# Reduce splits for testing
N_SPLITS <- 50  # instead of 100

# Use larger instance
# x2iedn.16xlarge (64 cores, 2TB RAM)
```

---

## References

- **Permutation Importance:** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- **Monte Carlo Cross-Validation:** Picard, R. R., & Cook, R. D. (1984). Cross-validation of regression models. JASA.
- **Model Ensembling:** Dietterich, T. G. (2000). Ensemble methods in machine learning. MCS 2000.

---

## Related Documentation

- **Main Notebook:** `feature_importance_mc_cv.ipynb`
- **Visualization Script:** `create_visualizations.R`
- **Cross-Age-Band Analysis:** `README_CROSS_AGEBAND_ANALYSIS.md`
- **S3 Output Structure:** `S3_OUTPUT_STRUCTURE.md`
- **rsample Bug:** `docs/RSAMPLE_BUG_WORKAROUND.md`

---

**Questions or Issues?** See main project README or open an issue.

