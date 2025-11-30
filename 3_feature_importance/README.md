# Feature Importance Analysis

**Date:** November 25, 2025  
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
10. [EC2 Configuration and Optimizations](#ec2-configuration-and-optimizations)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This project calculates scaled feature importance for predicting opioid dependence using:
- **Core Tree Models:** CatBoost, XGBoost (boosted trees), XGBoost RF mode
- **Validation:** Monte Carlo Cross-Validation (100–1000 splits) with temporal validation
- **Scaling:** Permutation-based importance weighted by model Recall
- **Aggregation:** Union of top 50 features from each core model with summed importances

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
✅ **Stratified Sampling** – Maintains target distribution  
✅ **Parallel Processing** – Fast execution (30 workers on EC2)  
✅ **Quality Weighting** – Features scaled by model performance (Recall)  
✅ **Model Consensus** – Union-based aggregation rewards agreement  
✅ **Publication-Ready Plots** – 4 visualization types with S3 upload

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
│    - Top 50 from CatBoost                                   │
│    - Top 50 from Random Forest                              │
│    - Scale by Recall                                        │
│    - SUM where overlap                                      │
│    - Rank by summed importance                              │
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
   - Importance: Permutation-based

3. **XGBoost RF Mode:**
   - Random-forest style configuration of XGBoost
   - Feature format: Binary 0/1 encoding (one-hot)
   - Importance: Permutation-based

All three core models use **permutation-based importance** for fair comparison.

---

## Aggregation Method

### Step-by-Step Process

#### 1. Train Models with MC-CV

For each **core** model type (CatBoost, XGBoost, XGBoost RF):
- Create 100–1000 stratified Monte Carlo cross-validation splits
- Train model on each training set (80% sampled from 2016-2018)
- Evaluate Recall on 2019 test set (temporal validation)
- Extract **permutation-based feature importance** for each split
- Aggregate across splits to get:
  - `importance_normalized` - Feature importance normalized to [0, 1]
  - `mc_cv_recall_mean` - Mean Recall across all splits
  - `mc_cv_recall_std` - Standard deviation of Recall

#### 2. Select Top 50 from Each Model

For each model:
- Rank features by `importance_normalized` (permutation-based or coefficient-based)
- Select **top 50 features**
- Scale importance by MC-CV Recall:

```python
importance_scaled = importance_normalized × mc_cv_recall_mean
```

#### 3. Union of Features

- Take the **union** of top 50 from all 3 core models
- Results in up to 150 features (if no overlap) or as few as 50 (if complete overlap)

#### 4. Sum Importances for Overlapping Features

For each feature in the union:

**If feature appears in multiple models:**
- Sum the scaled importances: `importance_scaled = sum(all_model_scaled_importances)`
- Sum the normalized importances: `importance_normalized = sum(all_model_normalized_importances)`
- Average the Recall: `mc_cv_recall_mean = mean(all_model_recalls)`
- Track which models: `models = "catboost, xgboost, xgboost_rf"` (subset depending on which models selected it)

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
- Compare across age bands for consistency

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
- **Workers:** `multiprocessing.cpu_count() - 2` (30 workers on 32-core system)
- **Batch Size:** Automatically calculated as `items_per_worker * 4`
- **Purpose:** Reduces process spawning overhead by processing multiple columns per worker

**Example:** For 4,962 features with 30 workers:
- Batch size: ~41 columns per batch
- Total batches: ~121 batches
- Each worker processes multiple batches sequentially

**Verification:**
Check logs for:
```
Feature matrix parallel workers: 30 (CPU count: 32), batch size: 41, batches: 121
```

### Monitoring Parallelization

#### Check Worker Count

```bash
# Count Python processes (should see ~30+ during feature matrix building)
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
- **Feature Matrix Building:** 8-16 cores active at 50-80% (with batching)
- **MC-CV Training:** 8 cores active at 80-100% (one per worker)
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
   # Should see 30+ Python processes during feature matrix building
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
- **Too many workers:** Reduce `n_workers_matrix` if memory is constrained
- **Too few workers:** Increase batch size or reduce `multiprocessing.cpu_count() - 2`
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
- **Cross-Age-Band Analysis:** `README_CROSS_AGEBAND_ANALYSIS.md`
- **S3 Output Structure:** `S3_OUTPUT_STRUCTURE.md`
- **rsample Bug:** `docs/RSAMPLE_BUG_WORKAROUND.md`

---

**Questions or Issues?** See main project README or open an issue.

