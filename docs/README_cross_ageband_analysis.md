# Cross-Age-Band Feature Importance Analysis

**Purpose:** Compare how feature importance changes across different age bands within a cohort

**Script:** `create_cross_ageband_heatmap.R`

---

## Overview

After running feature importance analysis for multiple age bands of the same cohort, this script creates:

1. **Heatmap** - Visual comparison of features across age bands
2. **Summary Statistics** - Quantitative analysis of feature variability
3. **Key Insights** - Identifies consistent vs age-specific features

---

## Prerequisites

Must have completed feature importance analysis for **at least 2 age bands** of the same cohort:

```
outputs/
├── opioid_ed_13-24_2016_feature_importance_aggregated.csv
├── opioid_ed_25-44_2016_feature_importance_aggregated.csv
├── opioid_ed_45-54_2016_feature_importance_aggregated.csv
├── opioid_ed_55-64_2016_feature_importance_aggregated.csv
└── opioid_ed_65-74_2016_feature_importance_aggregated.csv
```

---

## Usage

### Option 1: Standalone Execution

```bash
# From 3_feature_importance directory
Rscript create_cross_ageband_heatmap.R

# Or with environment variables
COHORT_NAME=opioid_ed EVENT_YEAR=2016 Rscript create_cross_ageband_heatmap.R
```

### Option 2: Source in R/Notebook

```r
source("create_cross_ageband_heatmap.R")

# Create heatmap for opioid_ed cohort
create_ageband_heatmap(
  cohort_name = "opioid_ed",
  event_year = 2016,
  age_bands = c("13-24", "25-44", "45-54", "55-64", "65-74", "75+"),
  output_dir = "outputs",
  s3_upload = TRUE,
  top_n = 50  # Number of top features to include
)

# Compare non_opioid_ed cohort
create_ageband_heatmap(
  cohort_name = "non_opioid_ed",
  event_year = 2016,
  age_bands = c("13-24", "25-44", "45-54", "55-64"),
  s3_upload = TRUE
)
```

---

## Outputs

### 1. Heatmap (PNG)

**Filename:** `{cohort}_{year}_ageband_heatmap_top50.png`

**Location:**
- Local: `outputs/plots/`
- S3: `s3://pgxdatalake/gold/feature_importance/cohort_name={cohort}/cross_ageband_analysis/`

**Visual Elements:**
- **Rows:** Top 50 features (union across all age bands)
- **Columns:** Age bands (13-24, 25-44, 45-54, etc.)
- **Color:**
  - White = Feature not in top 50 for that age band
  - Orange = Medium importance
  - Dark Blue = High importance

**Interpretation:**
- **Horizontal dark blue streak** = Feature important across ALL age bands (consistent)
- **Single dark blue cell** = Feature specific to ONE age band (age-specific)
- **Gradient across cells** = Feature importance changes with age (trend)

### 2. Summary Statistics (CSV)

**Filename:** `{cohort}_{year}_ageband_summary_top50.csv`

**Columns:**
- `feature` - Feature name
- `n_age_bands` - Number of age bands where feature appears in top 50
- `mean_importance` - Average scaled importance across age bands
- `sd_importance` - Standard deviation (variability)
- `min_importance` / `max_importance` - Range
- `range_importance` - Difference between max and min
- `cv_importance` - Coefficient of variation (% variability)

**Key Metrics:**
- **Low CV (<20%)** = Consistent across age bands
- **High CV (>50%)** = Highly age-specific
- **n_age_bands = 1** = Unique to one age band

### 3. Console Insights

```
Key Insights:
================================================================================

Most Consistent Features Across Age Bands (Low Variability):
  1. HYDROCODONE-ACETAMINOPHEN (CV=8.3%, present in 7 age bands)
  2. F11.20 (CV=12.1%, present in 7 age bands)
  3. TRAMADOL HCL (CV=15.4%, present in 6 age bands)

Most Variable Features Across Age Bands (High Age-Specificity):
  1. ADHD MEDICATION (CV=87.2%, range=0.023-0.421)
  2. WELL CHILD VISIT (CV=73.5%, range=0.010-0.378)
  3. NURSING HOME CARE (CV=69.8%, range=0.005-0.312)

Age Band-Specific Features (appear in only 1 age band): 15 features
  Top 5:
    1. PEDIATRIC VACCINE (importance=0.234, age band=0-12)
    2. MENOPAUSE TREATMENT (importance=0.189, age band=55-64)
    3. COLLEGE HEALTH VISIT (importance=0.156, age band=13-24)
```

---

## Use Cases

### 1. Identify Universal Risk Factors

Features with **low CV** and **high n_age_bands**:
- Important across all ages
- Use in age-agnostic models
- Priority for population-wide interventions

```r
# From summary CSV
universal_features <- summary %>%
  filter(cv_importance < 20, n_age_bands >= 5) %>%
  arrange(desc(mean_importance))
```

### 2. Find Age-Specific Features

Features with **n_age_bands = 1**:
- Unique risk factors for specific age groups
- Use for age-targeted interventions
- Build age-specific models

```r
# From summary CSV
age_specific <- summary %>%
  filter(n_age_bands == 1) %>%
  arrange(desc(mean_importance))
```

### 3. Track Feature Trends

Features with **high CV** but **present across multiple bands**:
- Importance changes with age
- May indicate disease progression
- Useful for temporal analysis

```r
# From summary CSV
trending_features <- summary %>%
  filter(cv_importance > 50, n_age_bands >= 4) %>%
  arrange(desc(mean_importance))
```

### 4. Model Selection Strategy

```r
# Age-agnostic model (all ages combined)
universal_features <- filter(summary, cv_importance < 25)
use_features <- universal_features$feature

# Age-stratified models (separate model per age band)
# Use all features from individual age band analysis
# Benefits: Captures age-specific patterns

# Hybrid approach
# Use universal features + age-specific features per band
```

---

## S3 Output Structure

```
s3://pgxdatalake/gold/feature_importance/
├── cohort_name=opioid_ed/
│   ├── cross_ageband_analysis/
│   │   ├── opioid_ed_2016_ageband_heatmap_top50.png
│   │   ├── opioid_ed_2016_ageband_summary_top50.csv
│   │   ├── opioid_ed_2017_ageband_heatmap_top50.png
│   │   └── opioid_ed_2017_ageband_summary_top50.csv
│   ├── age_band=13-24/
│   │   └── ...
│   └── age_band=25-44/
│       └── ...
└── cohort_name=non_opioid_ed/
    └── cross_ageband_analysis/
        └── ...
```

---

## Example Workflow

### Step 1: Run Individual Age Band Analyses

```r
# Run for each age band
for (age_band in c("13-24", "25-44", "45-54", "55-64", "65-74")) {
  # Set parameters in notebook
  AGE_BAND <- age_band
  COHORT_NAME <- "opioid_ed"
  EVENT_YEAR <- 2016
  
  # Run feature_importance_mc_cv.ipynb
  # This creates: opioid_ed_{age_band}_2016_feature_importance_aggregated.csv
}
```

### Step 2: Run Cross-Age-Band Analysis

```r
source("create_cross_ageband_heatmap.R")

results <- create_ageband_heatmap(
  cohort_name = "opioid_ed",
  event_year = 2016,
  age_bands = c("13-24", "25-44", "45-54", "55-64", "65-74"),
  top_n = 50
)

# Review outputs
# - Heatmap: outputs/plots/opioid_ed_2016_ageband_heatmap_top50.png
# - Summary: outputs/plots/opioid_ed_2016_ageband_summary_top50.csv
```

### Step 3: Interpret Results

```r
# Load summary
summary <- read_csv("outputs/plots/opioid_ed_2016_ageband_summary_top50.csv")

# Find features to use in age-agnostic model
universal <- summary %>%
  filter(cv_importance < 20, n_age_bands >= 4) %>%
  pull(feature)

cat(sprintf("Universal features for age-agnostic model: %d\n", length(universal)))

# Find features requiring age-stratified modeling
age_specific_count <- sum(summary$n_age_bands == 1)
cat(sprintf("Age-specific features requiring stratification: %d\n", age_specific_count))
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cohort_name` | string | required | Cohort identifier (e.g., "opioid_ed") |
| `event_year` | integer | required | Event year (e.g., 2016) |
| `age_bands` | vector | `c("0-12", "13-24", ...)` | Age bands to compare |
| `output_dir` | string | `"outputs"` | Directory with aggregated CSV files |
| `s3_upload` | boolean | `TRUE` | Whether to upload to S3 |
| `top_n` | integer | `50` | Number of top features to include |

---

## Notes

- **Minimum requirement:** 2 age bands (but 4+ recommended for meaningful trends)
- **Missing age bands:** Automatically skipped with warning
- **Feature selection:** Takes union of top N from each age band
- **Zero values:** Features not in top N for an age band show as white (0 importance)
- **Ordering:** Features ordered by average importance across age bands

---

## Troubleshooting

### Error: "Need at least 2 age bands"

**Cause:** Only found 1 or 0 aggregated CSV files

**Solution:**
1. Check that CSV files exist in `outputs/` directory
2. Verify filename pattern: `{cohort}_{age}_{year}_feature_importance_aggregated.csv`
3. Run feature importance analysis for more age bands

### Heatmap shows mostly white cells

**Cause:** Features are very age-specific (different top 50 for each age band)

**Interpretation:** This is actually informative! Suggests high age-specificity

**Action:** Consider:
- Increasing `top_n` parameter to capture more features
- Building age-stratified models instead of age-agnostic models

### No insights printed

**Cause:** Insufficient overlap in features

**Check:** Review the `n_age_bands` column in summary CSV

---

## Future Enhancements

- [ ] Add temporal comparison (same cohort/age band across years)
- [ ] Cross-cohort comparison (opioid_ed vs non_opioid_ed)
- [ ] Interactive heatmap with plotly
- [ ] Clustering analysis to group similar age bands
- [ ] Statistical tests for significance of differences

---

**Questions or Issues?** See main project README or feature importance documentation.

