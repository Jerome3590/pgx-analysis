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
│ 1. Load Cohort Data (parquet)                               │
│    - Training: Years 2016-2018 (combined)                   │
│    - Test: Year 2019 (holdout, never used for training)     │
│    - Excluded: Year 2020 (COVID-19 pandemic)                │
│    - Drugs, ICD codes, CPT codes                            │
│    - Target: is_target_case (opioid dependence)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Engineering                                      │
│    - Patient-level aggregation                              │
│    - CatBoost: Categorical factors                          │
│    - Random Forest/XGBoost: Binary 0/1                      │
│    - Consistent feature space across train/test             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Monte Carlo Cross-Validation (100–1000 splits)           │
│    - Each split samples from 2016-2018 training data        │
│    - All splits evaluate on same 2019 test set              │
│    ┌────────────────────┐  ┌────────────────────┐           │
│    │   Multiple Models  │  │  Multiple Models   │           │
│    │   (CatBoost, RF,   │  │  (XGBoost, LGBM,   │           │
│    │    XGBoost, etc.)  │  │   ExtraTrees, etc.)│           │
│    │                    │  │                    │           │
│    │  Per split:        │  │  Per split:        │           │
│    │  - Train: Sample   │  │  - Train: Sample   │           │
│    │    from 2016-2018  │  │    from 2016-2018  │           │
│    │  - Test: Always    │  │  - Test: Always    │           │
│    │    2019 (same)     │  │    2019 (same)     │           │
│    │  - Recall          │  │  - Recall          │           │
│    │  - Feature imp     │  │  - Feature imp     │           │
│    └────────────────────┘  └────────────────────┘           │
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

**CatBoost:**
- Handles categorical features natively
- Feature format: Each column is a factor with item name as level
- Importance: Permutation-based (PredictionValuesChange)
- For feature-importance **screening runs**, we allow CatBoost to use a somewhat larger tree budget (more iterations) than XGBoost / XGBoost RF. This is acceptable because the goal here is robust feature discovery, not a strict algorithm bake-off.

**XGBoost (boosted trees):**
- Gradient boosting with tree-based learners
- Feature format: Binary 0/1 encoding
- Importance: **two components**:
  - **Gain screen (annotation):** XGBoost’s built-in tree importance (gain / Gini) is computed for **all features** and used to flag which features have `gain_importance > 0` (ever used in a split). This gives a fast, model-internal signal that captures even rare variants.
  - **Permutation importance (primary score):** We run permutation-based importance on the **full feature set** (evaluation rows optionally capped via `PGX_PERM_MAX_ROWS`). Every non-constant feature receives a permutation score; the gain flag is used for interpretation, not as a hard filter.

**XGBoost RF mode:**
- Random-forest style XGBoost configuration
- Feature format: Binary 0/1 encoding
- Importance: Same **gain>0 + full-feature permutation** scheme as boosted XGBoost.

All three models use **permutation-based importance** for fair comparison, but the **workload is shaped differently per model**:

- CatBoost: full permutation importance over its feature set (categorical `item_*` columns), using CatBoost’s internal `PredictionValuesChange` implementation.
- XGBoost / XGBoost RF: gain-based screen over the full feature space, plus permutation importance over the **entire dense feature matrix** (with evaluation rows optionally capped via `PGX_PERM_MAX_ROWS`). The gain-based signals (`gain_importance > 0`) are used as annotations and for rare‑variant reasoning; they no longer restrict which features receive permutation scores.

In the **final model evaluation and deployment pipeline** (see `7_final_model/`), we align key hyperparameters such as the number of trees (e.g., `iterations` / `n_estimators`) across CatBoost and XGBoost-family models when we want an apples-to-apples comparison of model performance. Here, in the feature-importance stage, we prioritize stability and coverage of signal for downstream PGx engineering, accepting small asymmetries in tree counts and feature screens between models to keep wall-clock and memory usage manageable.

### Runtime and Performance Considerations

Because this pipeline is designed for **health data** and downstream causal/clinical interpretation, the Monte Carlo feature importance step is intentionally **computationally heavy**:

- **Data scale (per cohort/age_band):**
  - \~10,000–15,000 patients in the 2016–2018 training window
  - \~3,000–5,000 patients in the 2019 holdout
  - \~10,000–15,000 binary/categorical features after feature engineering

- **Current feature-importance MC-CV defaults:**
  - `N_SPLITS = 10` Monte Carlo splits per model (per cohort, per age band) for the feature-importance workflow.
  - For each split: train on an 80% sample of **2016–2018** patients and evaluate on the **full 2019 holdout** (or a capped subset if `PGX_PERM_MAX_ROWS` is set).
- **MC-CV configuration:**
  - `N_SPLITS = 50` Monte Carlo splits per model (per cohort, per age band)
  - For each split:
    - Train on an 80% sample of **2016–2018** patients
    - Evaluate on the **full 2019 holdout** (no subsampling)
- **Permutation importance:**
  - **CatBoost:** full permutation importance on the entire 2019 holdout.
  - **XGBoost / XGBoost RF:** permutation importance is run only on the **top‑K features by gain**, where `K` is controlled by `PGX_XGB_PERM_TOP_K` (default 2000). This substantially reduces the number of perturbations.
  - For permutation on any model, the effective evaluation set size can be capped via `PGX_PERM_MAX_ROWS` (e.g., 10k rows) to prevent pathological runs on extremely large cohorts, trading a small loss in precision for large gains in runtime and memory safety.
  - **Current implementation note:** In the code, XGBoost / XGBoost RF now run permutation importance over the **full feature set**, with gain-based scores retained only as annotations (`gain_importance > 0`) and for rare‑variant reasoning. The older `PGX_XGB_PERM_TOP_K`-based truncation path is deprecated in favor of **gain>0 + full-feature permutation**.

In practice this means:

- On a **well-provisioned EC2 instance** (e.g., 32 cores, 1 TB RAM), with `N_SPLITS = 25–50` and full 2019 holdout:
  - It is **normal** to see logs like:
    - `[Parallel(n_jobs=30)]: Done   3 out of 50 | elapsed: 338.1min ...`
    - This reflects the fact that each XGBoost split with full permutation importance can take **tens of minutes** on large cohorts, even with 30 workers.
  - The cost is buying you:
    - Stable **mean recall/log-loss** and standard deviations across many temporal resamples
    - Feature importance estimates that are **robust to train-sample perturbations**
    - Consensus importance across three strong models with different inductive biases (CatBoost, XGBoost, XGBoost RF, plus the rare-variant XGBoost model).

- On a **local workstation** with fewer cores / less RAM (e.g., Windows + single GPU), running full-holdout permutation for large cohorts (like `opioid_ed 25–44`) with `N_SPLITS = 10` can still take on the order of **several days** if both XGBoost and XGBoost RF are recomputed (e.g., ~6–7 hours per split × 10 splits × 2 models). For this reason, we typically:
  - Use **more aggressive caps** locally (`PGX_PERM_MAX_ROWS` and lower `n_repeats`) for iteration, and
  - Reserve **full-holdout, higher-split runs** for EC2.

If you need to reduce runtime while retaining robustness, the **safest levers** are:

- Keep `N_SPLITS` fixed, but:
  - Permutation `n_repeats` is set to 3 (default) for optimal speed/accuracy balance, and/or
  - Lower `PGX_XGB_PERM_TOP_K` (e.g., 2000 → 1000) to shrink the XGBoost/XGBoost‑RF permutation set, and/or
  - Lower `PGX_PERM_MAX_ROWS` (e.g., from 50k full holdout → 5k–10k subsample) so permutation operates on a capped evaluation set.
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

In more recent configurations (especially on constrained local/GPU workstations), we additionally:

- Reduce `N_SPLITS` for exploratory or local runs (e.g., 10 splits instead of 25–50).
- Use `PGX_XGB_PERM_TOP_K` and `PGX_PERM_MAX_ROWS` to keep XGBoost permutation passes tractable while preserving robustness for the top‑ranked features.

### Non_Opioid_ed (Polypharmacy) Age-Band Size and Expected Runtime (N_SPLITS = 25)

For the **polypharmacy ED cohort** (`cohort_name=non_opioid_ed`), which is the primary focus for **older age bands**, the cohort parquet files in `data/cohorts_F1120/cohort_name=non_opioid_ed/` show substantially larger event workloads and patient counts:

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

Taking `non_opioid_ed 65–74` as a **baseline** (largest `(train + test)` event workload), the **relative event workload factors** are approximately:

- **0–12**: ≈ 0.25×  
- **13–24**: ≈ 0.23×  
- **25–44**: ≈ 0.54×  
- **45–54**: ≈ 0.39×  
- **55–64**: ≈ 0.54×  
- **65–74**: 1.00× (baseline)  
- **75–84**: ≈ 0.65×  
- **85–94**: ≈ 0.26×  
- **95–114**: ≈ 0.02×  

For the same MC‑CV configuration (25 splits, 3 models, exact XGBoost), this implies that **non_opioid_ed 65–74 is the heaviest polypharmacy age band** by event workload, with **55–64 and 75–84** in a similar runtime regime, and younger / extreme-age bands contributing a smaller fraction of total compute despite very large underlying patient counts.

### Rare-Variant Second-Pass Scan (Target Cohort Only)

To improve sensitivity for **rare but potentially important variants** without exploding the cost of the full MC‑CV, the workflow includes a **second, XGBoost-only pass on the target cohort (2019 holdout)**:

- Identify **rare candidate features** on the holdout feature matrix (`test_data_rf`), using binary `item_*` columns:
  - A feature is considered "rare" if it appears in between `PGX_RARE_MIN_PATIENTS` and `PGX_RARE_MAX_PATIENTS` holdout patients (defaults: 5 and 25, configurable).
- Build a **slim rare-feature matrix** for 2019 only and perform a stratified train/eval split within that cohort (e.g., 50/50).
- Train an **XGBoost model on rare features only**, reusing the main XGBoost hyperparameters.
- Run **permutation importance on all rare features** (no gain-based top‑K screen here), using the eval subset:
  - This allows higher-fidelity measurement for rare variants on the target cohort while keeping the problem small.
- Save and upload a dedicated artifact:
  - `{cohort_name}_{age_band_fname}_xgboost_rare_feature_importance.csv`
  - Stored under `gold/feature_importance/{cohort_name}/{age_band}/` in S3.

This second pass complements the main MC‑CV ensemble and is **included in the final aggregation**:

- The main pass (CatBoost, XGBoost, XGBoost RF) gives **robust, cross-model importance for common and moderately frequent features** under temporal resampling.
- The rare-variant pass adds an additional XGBoost-based model (`xgboost_rare`) focused on **rare signals in the target cohort**, which is treated as a fourth model in aggregation.

### Methodological Constraints and Limitations

Even with the above design, there are important **constraints and limitations** to keep in mind:

- **Extremely rare or tiny-effect features can still be missed**
  - If only a handful of patients carry a feature and its effect size is small, both gain and permutation importance can be too noisy to separate it from random variation.
  - The rare-variant pass improves sensitivity for this regime but does not guarantee recovery of *all* true rare effects.

- **Gain-based screening influences which features get permutation importance (XGBoost / XGBoost RF)**
  - XGBoost’s built-in importance (gain / Gini) is used to select the top‑K features for permutation.
  - This can **under‑prioritize some features**, particularly very sparse or weakly used ones, which then receive permutation importance `0` in the XGBoost/XGBoost‑RF results (though they may still be scored by CatBoost).

- **Row subsampling for permutation trades precision for tractability**
  - `PGX_PERM_MAX_ROWS` caps the number of evaluation rows used in permutation importance (e.g., 10k–20k instead of the full 2019 cohort).
  - This keeps memory and runtime under control, but slightly reduces precision and stability for borderline features compared to using the full holdout.

- **Correlated features can share or mask signal**
  - Permutation importance can under‑estimate the importance of features that are highly correlated with others, because shuffling one correlated partner may leave sufficient signal in the remaining partners.
  - This affects both the main MC‑CV pass and the rare‑variant pass.

- **Model- and configuration-dependence**
  - Rare‑variant importance in the second pass is **XGBoost-only** and depends on:
    - The internal train/eval split within 2019,
    - The rare-feature thresholds (`PGX_RARE_MIN_PATIENTS`, `PGX_RARE_MAX_PATIENTS`),
    - XGBoost hyperparameters.
  - Different reasonable settings may yield slightly different rankings, especially for borderline or highly correlated features.

- **Finite compute and environment-specific knobs**
  - Multiple environment variables (`PGX_XGB_PERM_TOP_K`, `PGX_PERM_MAX_ROWS`, `PGX_RARE_MIN_PATIENTS`, `PGX_RARE_MAX_PATIENTS`) and OS-specific parallelism choices (Windows vs Linux/EC2) make the pipeline sensitive to configuration.
  - For publication-grade analyses, we recommend **fixing and documenting** these settings so runs are reproducible and comparable across environments.

In practice, this framework is **very strong for discovering moderate and strong true signals** and reasonably sensitive to many rare but impactful variants, but like any empirical feature-importance pipeline, it should be viewed as a **powerful discovery tool, not an oracle**. Downstream clinical/causal interpretation and validation remain essential.

---

## Understanding Permutation Importance vs. Row-Level Analysis

### Critical Point: What Permutation Importance Actually Does

**Permutation importance does NOT preserve row-level associations.**

When you run permutation importance:

```python
# For each feature, it does this:
1. Save original column: [drug_A, drug_B, drug_C, drug_A, ...]
2. Randomly shuffle: [drug_C, drug_A, drug_B, drug_A, ...]  # BREAKS row associations!
3. Make predictions on shuffled data
4. Compare performance drop
```

**Key insight**: The shuffling **breaks the connection** between specific drug combinations and specific patients. It's measuring **average effect**, not **which specific combinations matter for which patients**.

### What Permutation Importance Can and Cannot Tell You

#### Even with Full Dataset:

❌ **Permutation importance CANNOT tell you:**
- "Patient 12345 with drugs [AMOXICILLIN, AZITHROMYCIN] had outcome Y"
- "The combination of Drug A + Drug B in row X drives the outcome"
- "Which specific drug combinations matter for which specific patients"

✅ **Permutation importance CAN tell you:**
- "On average, does shuffling AMOXICILLIN affect model performance?"
- "Is AMOXICILLIN important overall?"
- "Which features matter most on average across all patients"

#### With Sampling (`PGX_PERM_MAX_ROWS`):

The same limitation applies - you're still measuring average effects, just with a sample.

**However**, sampling could affect:
- **Rare combinations**: If a drug combination only appears in 100 rows out of 766K, sampling might miss it
- **Statistical power**: Less data = wider confidence intervals

**For permutation importance specifically:**
- **No significant impact** - You're measuring average effects anyway
- Rare combinations might be missed, but they'd be hard to detect even with full data
- Top features will still be identified correctly

### What You Actually Need for Row-Level Analysis

If you want to know **which specific drug combinations drive outcomes for specific patients**, you need:

#### 1. **SHAP Values** (Row-Level Feature Importance)

```python
# SHAP tells you: "For patient X, how much did drug A contribute to the prediction?"
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# shap_values[i, j] = contribution of feature j to prediction for patient i
```

**Use case**: Patient-specific feature contributions

#### 2. **Actual Predictions + Feature Values**

```python
# Look at high-risk patients and their actual drug combinations
high_risk_patients = X_test[y_pred_proba > 0.8]
# Then examine their actual drug combinations
```

**Use case**: Identify patterns in high-risk patients

#### 3. **FPGrowth Pattern Mining** (Already Available)

Your codebase has FPGrowth analysis that finds **frequent drug combinations**:
- `4_fpgrowth_analysis/` - Finds frequent itemsets
- Identifies which drug combinations are associated with outcomes
- Preserves row-level associations

**Use case**: Discover frequent drug combination patterns

#### 4. **FFA Analysis** (Step 9)

- **Anchored Explanations (AXP)**: Rule-based explanations per patient
- **Causal Analysis**: Intervention effects for specific patients
- See [`../Step9_FFA/`](../Step9_FFA/) for details

**Use case**: Patient-specific rule-based explanations

### Recommendation: Combining Approaches

**Best approach for comprehensive analysis:**

1. **Use `PGX_PERM_MAX_ROWS=50000` for fast permutation importance** (feature ranking)
   - Identifies which drugs/features matter overall
   - Sampling doesn't significantly impact average-effect measurements
   - Fast and efficient for feature screening

2. **Use FPGrowth/SHAP/FFA on full dataset for row-level combination analysis**
   - FPGrowth: Frequent drug combinations
   - SHAP: Patient-specific feature contributions
   - FFA: Rule-based patient explanations

3. **Combine both approaches for complete understanding**
   - Permutation importance → Which features matter overall
   - Row-level methods → Which combinations matter for specific patients

### When to Use Each Method

| Method | Use Case | Question Answered | Dataset Size |
|--------|----------|-------------------|--------------|
| **Permutation Importance** | Feature ranking, screening | "Which features matter on average?" | Can use sampling (`PGX_PERM_MAX_ROWS`) |
| **SHAP** | Patient-specific explanations | "How much did feature X contribute for patient Y?" | Full dataset recommended |
| **FPGrowth** | Pattern discovery | "Which drug combinations are frequent?" | Full dataset recommended |
| **FFA** | Rule-based explanations | "Which rules/conditions led to this prediction?" | Full dataset recommended |

### Bottom Line

**Permutation importance is the right tool for feature ranking, but not for row-level analysis.**

- **Permutation importance**: Measures average effects (works fine with sampling)
- **Row-level analysis**: Requires SHAP, FPGrowth, FFA, or examining actual predictions (use full dataset)

**Workflow:**
1. Use permutation importance with sampling for fast feature screening
2. Use row-level methods (SHAP/FFA/FPGrowth) on full dataset for patient-specific insights
3. Combine both approaches for comprehensive understanding

---

## Aggregation Method

### Step-by-Step Process

#### 1. Train Models with MC-CV

For each model type (CatBoost, XGBoost, XGBoost RF):
- Create multiple stratified Monte Carlo cross-validation splits
- Train the model on each training set (e.g., 80% of 2016–2018)
- Evaluate Recall (or LogLoss) on the 2019 holdout for each split
- Extract **permutation-based feature importance** for each split
- Aggregate across splits (per model) to get:
  - `importance_mean` (per-feature mean importance across splits)
  - `scaled_importance_mean` (per-feature importance scaled by the split-level metric)
  - `recall_mean`, `logloss_mean` (model-level performance)

#### 2. Rare-Variant XGBoost Model

- Build a **rare-feature-only XGBoost model** on the 2019 holdout and compute its own permutation-based importance (as described above).
- Treat this as an additional model (`xgboost_rare`) with its own:
  - `importance_mean`
  - `scaled_importance_mean`
  - `recall_mean` / `logloss_mean`

#### 3. Combine Per-Model Results

- Collect all per-model DataFrames into `all_results` with a `method` label:
  - `catboost`, `xgboost`, `xgboost_rf`, and `xgboost_rare` (when available).
- For each model:
  - Normalize `importance_mean` to `[0, 1]` within that model → `importance_normalized`.
  - Scale by model performance (`recall_mean` or inverse `logloss_mean`) → `importance_scaled_by_model`.

#### 4. Aggregate Across Models (Union of Features)

- Concatenate all per-model records into a single table.
- Group by `feature` and compute:
  - `importance_scaled_by_model_sum` = sum of scaled importances across models
  - `importance_normalized_sum` = sum of normalized importances across models
  - `n_models` = number of distinct models that contributed to this feature
- Convert the sum to a **mean contribution per model**:

```text
importance_scaled_mean = importance_scaled_by_model_sum / n_models
```

- Renormalize `importance_scaled_mean` to `[0, 1]` across all features → final `importance_normalized`.
- Scale by the **best model performance** (`best_performance` from CatBoost / XGBoost / XGBoost RF / XGBoost rare) to obtain the final `importance_scaled`:

```text
importance_scaled = importance_normalized × best_performance
```

- This ensures:
  - Features supported by **more models** (higher `n_models`) tend to have more stable scores.
  - Rare-variant signals from `xgboost_rare` are included on equal footing with the main ensemble.

---

## Output Files

### 1. Aggregated Feature Importance CSV

**Location:**
- Local: `3_feature_importance/outputs/{cohort}_{age_band}_aggregated_feature_importance.csv`
- S3: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv`

**Columns:**

| Column | Description | Range |
|--------|-------------|-------|
| `feature` | Feature name (drug, ICD, CPT) | String |
| `importance_normalized` | Final normalized importance (0–1) | 0.0 – 1.0 |
| `importance_scaled` | Final scaled importance (normalized × best model performance) | 0.0 – ~1.0 |
| `n_models` | Number of models contributing (CatBoost, XGBoost, XGBoost RF, XGBoost rare) | 1–4 |

**Key Metric:** `importance_scaled` - Used for final ranking and visualization.

### 2. Per-Model CSVs

**Files:**
- `{cohort}_{age}_{year}_catboost_feature_importance.csv`
- `{cohort}_{age}_{year}_random_forest_feature_importance.csv`

**Purpose:** Debugging, model comparison, reproducibility

---

## Visualization

**See [`docs/README_feature_importance_visualization.md`](README_feature_importance_visualization.md) for comprehensive visualization guide including cross-platform usage, multiple execution methods, and troubleshooting.**

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

- **Main Notebook:** `3_feature_importance/feature_importance_cohort_runner.ipynb`
- **Visualization Guide:** [`docs/README_feature_importance_visualization.md`](README_feature_importance_visualization.md)
- **Visualization Script:** `py_helpers/create_feature_importance_visualizations.py` (Python, recommended)
- **Legacy Visualization Script:** `r_helpers/create_visualizations.R` (R, maintained for backward compatibility)
- **Cross-Age-Band Analysis:** [`docs/README_cross_ageband_analysis.md`](README_cross_ageband_analysis.md)
- **Workflow Testing:** [`docs/README_workflow_testing.md`](README_workflow_testing.md)
- **S3 Output Structure:** `S3_OUTPUT_STRUCTURE.md`
- **rsample Bug:** `docs/RSAMPLE_BUG_WORKAROUND.md`

---

**Questions or Issues?** See main project README or open an issue.

