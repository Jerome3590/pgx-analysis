# Feature Importance S3 Output Structure

**Date:** November 25, 2025  
**Location:** `s3://pgxdatalake/gold/feature_importance/`

---

## S3 Folder Structure

```
s3://pgxdatalake/gold/feature_importance/
├── cohort_name=opioid_ed/
│   ├── age_band=13-24/
│   │   ├── event_year=2016/
│   │   │   ├── opioid_ed_13-24_2016_feature_importance_aggregated.csv
│   │   │   └── plots/
│   │   │       ├── opioid_ed_13-24_2016_top50_features.png
│   │   │       ├── opioid_ed_13-24_2016_top50_with_recall.png
│   │   │       ├── opioid_ed_13-24_2016_normalized_vs_scaled.png
│   │   │       └── opioid_ed_13-24_2016_category_distribution.png
│   │   ├── event_year=2017/
│   │   │   ├── opioid_ed_13-24_2017_feature_importance_aggregated.csv
│   │   │   └── plots/
│   │   │       └── ...
│   │   └── ...
│   ├── age_band=25-44/
│   │   ├── event_year=2016/
│   │   │   ├── opioid_ed_25-44_2016_feature_importance_aggregated.csv
│   │   │   └── plots/
│   │   │       └── ...
│   │   └── ...
│   └── ...
└── cohort_name=non_opioid_ed/
    └── (same structure)
```

---

## File Contents

### Aggregated Feature Importance CSV

**Filename:** `{cohort}_{age_band}_{year}_feature_importance_aggregated.csv`

**Aggregation Method:**
1. Extract top 50 features from CatBoost (by normalized permutation importance)
2. Extract top 50 features from Random Forest (by normalized permutation importance)
3. Take union (up to 100 features if no overlap)
4. Scale each feature's importance by its MC-CV Recall (quality weighting)
5. **SUM** scaled importances where features appear in both models
6. Rank by summed scaled importance

**Columns:**
- `rank` - Feature rank by scaled importance (1 = most important)
- `feature` - Feature name (drug, ICD code, or CPT code)
- `importance_normalized` - Sum of normalized permutation importances from models
- `importance_scaled` - **Sum of Recall-scaled importances (FINAL METRIC)**
- `n_models` - Number of models including this feature (1 or 2)
- `models` - Which models included it (e.g., "catboost, random_forest")
- `mc_cv_recall_mean` - Average feature Recall across models
- `mc_cv_recall_std` - Standard deviation of Recall

**Example:**
```csv
rank,feature,importance_normalized,importance_scaled,n_models,models,mc_cv_recall_mean,mc_cv_recall_std
1,HYDROCODONE-ACETAMINOPHEN,1.8234,1.5012,2,"catboost, random_forest",0.8234,0.0156
2,TRAMADOL HCL,1.6821,1.3856,2,"catboost, random_forest",0.8234,0.0156
3,F11.20,0.9234,0.7603,1,catboost,0.8234,0.0156
4,OXYCODONE HCL,0.8567,0.7054,1,random_forest,0.8234,0.0156
...
```

**Note:** Features appearing in both models (n_models=2) have higher summed importance.

### Visualization Plots (PNG)

**Location:** `plots/` subfolder

**Files:**
1. **`{cohort}_{age}_{year}_top50_features.png`** (12" x 14")
   - Bar chart of top 50 features
   - Sorted by scaled importance
   - Shows full feature names

2. **`{cohort}_{age}_{year}_top50_with_recall.png`** (12" x 14")
   - Top 50 features with color gradient (Orange → Dark Blue)
   - Color represents MC-CV Recall quality (with 95% CI)
     - Orange = Lower Recall (less confident)
     - Dark Blue = Higher Recall (more confident)
   - Height represents scaled importance
   - Shows both importance magnitude and model confidence

3. **`{cohort}_{age}_{year}_normalized_vs_scaled.png`** (12" x 14")
   - Side-by-side comparison of top 50 features
   - Gray bars: normalized importance
   - Blue bars: Recall-scaled importance
   - Shows impact of quality weighting on feature rankings

4. **`{cohort}_{age}_{year}_category_distribution.png`** (10" x 6")
   - Bar chart showing feature category counts
   - Categories: Drug Name, ICD Code, CPT Code
   - Based on top 50 features

---

## Data Generation

### Source Notebook
**File:** `feature_importance_mc_cv.ipynb`

### Process
1. Load cohort data from `/mnt/nvme/cohorts/` or local path
2. Extract patient-level features (drugs, ICDs, CPTs)
3. Run Monte Carlo Cross-Validation:
   - 100 random train/test splits (80/20)
   - Train CatBoost and Random Forest models
   - Calculate feature importance per split
   - Evaluate Recall on test set
4. Aggregate results across splits and models
5. Normalize and scale by MC-CV Recall
6. Save top features to S3

### Configuration
- **MC-CV Splits:** 100 (configurable: `N_SPLITS`)
- **Train/Test Ratio:** 80/20 (configurable: `TRAIN_PROP`)
- **Models:** CatBoost + Random Forest
- **Scaling Metric:** Recall (or LogLoss)

---

## Accessing Data

### Python (boto3)
```python
import boto3
import pandas as pd
from io import StringIO

s3 = boto3.client('s3')

# Download feature importance
key = 'gold/feature_importance/cohort_name=opioid_ed/age_band=25-44/event_year=2016/opioid_ed_25-44_2016_feature_importance_aggregated.csv'
obj = s3.get_object(Bucket='pgxdatalake', Key=key)
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Get top 50 features
top50 = df.head(50)
```

### R (aws.s3)
```r
library(aws.s3)
library(readr)

# Download feature importance
s3_path <- "s3://pgxdatalake/gold/feature_importance/cohort_name=opioid_ed/age_band=25-44/event_year=2016/opioid_ed_25-44_2016_feature_importance_aggregated.csv"
df <- read_csv(s3_path)

# Get top 50 features
top50 <- head(df, 50)
```

### AWS CLI
```bash
# Download specific cohort data
aws s3 cp s3://pgxdatalake/gold/feature_importance/cohort_name=opioid_ed/age_band=25-44/event_year=2016/opioid_ed_25-44_2016_feature_importance_aggregated.csv ./

# Download all opioid_ed results (including plots)
aws s3 sync s3://pgxdatalake/gold/feature_importance/cohort_name=opioid_ed/ ./opioid_ed/

# Download only plots for a specific cohort
aws s3 sync s3://pgxdatalake/gold/feature_importance/cohort_name=opioid_ed/age_band=25-44/event_year=2016/plots/ ./plots/

# List available cohorts
aws s3 ls s3://pgxdatalake/gold/feature_importance/
```

---

## Use Cases

### 1. Model Development
```python
# Load top features for model training
features_df = load_from_s3('...feature_importance_aggregated.csv')
top_features = features_df.head(50)['feature'].tolist()

# Use for feature selection in CatBoost
model = CatBoostClassifier()
model.fit(X[top_features], y)
```

### 2. Cross-Cohort Comparison
```python
# Compare feature importance across cohorts
opioid_features = load_from_s3('.../opioid_ed/.../feature_importance_aggregated.csv')
non_opioid_features = load_from_s3('.../non_opioid_ed/.../feature_importance_aggregated.csv')

# Find cohort-specific features
opioid_specific = set(opioid_features['feature']) - set(non_opioid_features['feature'])
```

### 3. Temporal Analysis
```python
# Compare features across years
features_2016 = load_from_s3('.../event_year=2016/...')
features_2017 = load_from_s3('.../event_year=2017/...')
features_2018 = load_from_s3('.../event_year=2018/...')

# Track feature importance over time
```

---

## Quality Metrics

### Included in Results
- **Recall Mean:** Average Recall across 100 MC-CV splits
- **Recall Std:** Standard deviation (measures consistency)
- **95% CI:** Narrow confidence intervals indicate robust estimates

### Interpretation
- **High Recall + Low Std:** Reliable, consistent feature
- **High Recall + High Std:** Important but variable across splits
- **Importance Scaled:** Adjusts for model performance quality

---

## Cross-Age-Band Analysis

After running feature importance for multiple age bands, create comparison heatmaps:

```
s3://pgxdatalake/gold/feature_importance/cohort_name=opioid_ed/cross_ageband_analysis/
├── opioid_ed_2016_ageband_heatmap_top50.png      # Heatmap: features × age bands
└── opioid_ed_2016_ageband_summary_top50.csv      # Statistics: variability, consistency
```

**Script:** `create_cross_ageband_heatmap.R`

**Outputs:**
- Heatmap showing how features change across age bands
- Summary statistics (CV, range, consistency metrics)
- Identifies universal vs age-specific features

**See:** `README_CROSS_AGEBAND_ANALYSIS.md` for details

---

## Related Files

### Local Outputs
- `outputs/{cohort}_{age}_{year}_catboost_feature_importance.csv` - CatBoost only
- `outputs/{cohort}_{age}_{year}_random_forest_feature_importance.csv` - RF only
- `outputs/{cohort}_{age}_{year}_feature_importance_aggregated.csv` - Combined

### Documentation
- `README_feature_importance.md` - Complete analysis documentation
- `feature_importance_mc_cv.ipynb` - Analysis notebook
- `create_visualizations.R` - Per-cohort visualization script
- `create_cross_ageband_heatmap.R` - Cross-age-band comparison
- `README_CROSS_AGEBAND_ANALYSIS.md` - Cross-age-band analysis guide
- `docs/RSAMPLE_BUG_WORKAROUND.md` - Known issues and fixes

---

## Update History

- **2025-11-25:** Initial S3 upload implementation
- **Frequency:** Updated per cohort analysis run
- **Retention:** Permanent (gold layer)

---

## Notes

- Results are **deterministic** given same data and random seed
- **100 splits** provide narrow confidence intervals
- **Recall-scaled** importance reflects both feature relevance and model quality
- Use **top 50 features** for downstream ML models (avoids overfitting)

---

**Questions or Issues?**  
See `3_feature_importance/README.md` for methodology details.

