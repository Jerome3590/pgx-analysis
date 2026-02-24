# Final Model Development - PGx Analysis

This module (`6_final_model`) hosts the final prediction model pipeline combining features from FPGrowth, BupaR, and DTW analyses for patient-level classification.

## Overview

The final model integrates two complementary analysis methods to create comprehensive patient-level features:

1. **FPGrowth** - Frequent pattern mining (itemsets, association rules)
2. **BupaR** - Process mining (sequence patterns, temporal flows)

**Important:** DTW (Dynamic Time Warping) is used for **protocol filtering** (preprocessing) to remove standard care patterns, **not as features** in the final model. Sequence information comes from **BupaR**, not DTW.

### Temporal Validation Strategy

**Important:** The final model uses a strict temporal validation approach consistent with feature importance analysis:

- **Training Data:** Years 2016-2018 (full training set)
- **Test Data:** Year 2019 (holdout set, never used for training)
- **Excluded:** Year 2020 (COVID-19 pandemic year)

**Rationale:**
1. **Prevents Data Leakage:** 2019 data is never seen during training, ensuring true temporal validation
2. **Maintains Temporal Order:** Train on past data, test on future data
3. **Avoids COVID Impact:** 2020 excluded due to pandemic-related changes in healthcare patterns
4. **Consistent with Feature Importance:** Same train/test split as feature importance analysis ensures selected features generalize well

**Note:** This validation strategy matches the feature importance analysis pipeline, ensuring that features identified as important during MC-CV will perform well in the final model.

## Goals

- Build cohort-level prediction models for target outcomes (opioid dependence, ED visits)
- Integrate features from FPGrowth and BupaR analyses
- Use DTW for protocol filtering (preprocessing) to remove standard care patterns
- Standardize feature extraction across pharmacy (drug_name) and medical (ICD/CPT) domains
- Produce model explanations to guide feature reduction and clinical review

## Feature Schema

The complete feature schema is defined in `final_feature_schema.json` (JSON Schema Draft 7).

### Feature Categories

| Category | Feature Count | Description |
|----------|---------------|-------------|
| **FPGrowth** | ~100-500 | Frequent itemsets, association rules, itemset metrics |
| **BupaR** | ~50-200 | Process flow patterns, sequence features, temporal metrics |
| **PGx** | 2 | CPIC drug counts only (`pgx_num_drugs`, `pgx_num_cpic_drugs`); alleles used in PGx card via patient-submitted SNP data |
| **Pre-event counts** | ~5-10 | Event counts before target (drugs, ICDs, CPTs, unique activities) |
| **Demographics** | ~10-15 | Age, gender, race, location, payer |
| **Temporal** | ~5-10 | Event dates, temporal windows, seasonality |
| **Total** | **~185-750** | Patient-level features for classification |

**Note:** DTW features are **NOT included** in the final model. DTW is used for protocol filtering (preprocessing) only.

### Key Features

#### FPGrowth Features
- **Frequent Itemsets**: Binary features for each frequent itemset (drugs, ICD codes, CPT codes)
- **Association Rules**: Rule matching counts, confidence, and lift metrics
  - `rules_target_icd_match`: Number of opioid dependence prediction rules matched
  - `rules_target_ed_match`: Number of ED visit prediction rules matched
  - `max_rule_confidence_target_icd`: Maximum confidence of matched rules
  - `max_rule_lift_target_icd`: Maximum lift of matched rules
- **Itemset Metrics**: Aggregated statistics (total unique items, avg support)
- **Drug Encoding**: Global drug encoding for CatBoost categorical features

#### BupaR Features
- **Process Flow**: Path length, unique activities, path diversity
- **Temporal**: Throughput time, waiting time, active time, avg time between activities
- **Activity Frequencies**: Counts for each activity type
- **Sequence Patterns**: Repetition indicators, complexity measures, common pattern matching
  - Top sequences: Frequent sequence patterns (e.g., `overall_is_top_sequence`, `overall_top_sequence_frequency`)
  - Rare sequences: Uncommon sequence patterns (e.g., `overall_is_rare_sequence`, `overall_rare_sequence_frequency`)
  - Sequence categories: Classification of patient sequences
- **Drug Sequences**: Sequence length, drug switches, concurrent drugs

**Note:** BupaR provides all sequence information for the final model. DTW is not used for sequence features.

#### DTW Role: Protocol Filtering (Not Features)

**DTW is used for preprocessing/filtering, NOT as features in the final model.**

- **Purpose**: Identify and filter out protocol-like events (standard care patterns)
- **Method**: Uses time windows between consecutive events to identify protocol events (< 7 days apart)
- **Output**: Filtered `model_events_no_protocols.parquet` for cleaner feature engineering
- **Rationale**: DTW captures standard care protocols that both targets and controls follow, representing noise rather than predictive signal
- **Sequence Information**: All sequence features come from **BupaR**, not DTW

See `6_dtw_analysis/DTW_ROLE.md` and `6_dtw_analysis/PROTOCOL_FILTERING.md` for details.

## Data Inputs

### Base Cohort Data
- Gold cohort partitions: `s3://pgxdatalake/gold/cohorts/{cohort_name}/{age_band}/{event_year}/`

### FPGrowth Features
- **Source**: `s3://pgxdatalake/gold/fpgrowth/global/{item_type}/`
- **Files**: 
  - `rules_TARGET_ICD.json` - Opioid dependence prediction rules
  - `rules_TARGET_ED.json` - ED visit prediction rules
  - `rules_CONTROL.json` - Baseline/control rules
  - `frequent_itemsets.parquet` - Frequent itemsets

### BupaR Features
- **Source**: `s3://pgxdatalake/gold/bupar/{cohort_name}/{age_band}/{event_year}/`
- **Files**:
  - `process_flow_features.parquet`
  - `sequence_patterns.parquet`
  - `activity_frequencies.parquet`

### DTW Protocol Filtering (Preprocessing)
- **Purpose**: Filter protocol-like events before feature engineering
- **Script**: `6_dtw_analysis/filter_protocol_events.py`
- **Output**: `model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet`
- **Note**: DTW features are **NOT** included in the final model. DTW is used only for preprocessing.

## Step 6 Pipeline Overview

Step 6 for each `(cohort, age_band)` now has two main sub-steps:

1. **6a – Feature encoding artifacts (per cohort)**
   - Build cohort/age-band specific lookup tables and codebooks:
     - `6_final_model/create_feature_lookup.py`
       - Produces `{cohort}_{age_band_fname}_feature_lookup.csv` under:
         - `6_final_model/outputs/{cohort}/{age_band_fname}/`
         - `feature_encoding_outputs/{cohort}/{age_band_fname}/`
       - Maps **numeric feature indices** to:
         - `feature_name`
         - `group` (FPGrowth/BupaR/DTW/PGx/drug_name/ICD/CPT structural, etc.)
         - `description`
         - For FP-Growth itemsets: `itemset_type` and `itemset_items` (actual drug/ICD/CPT/medical codes).
     - `6_final_model/create_drug_codebook.py`
       - Produces `{cohort}_{age_band_fname}_drug_codebook.csv` under the same locations.
       - One row per distinct `drug_name` observed in `4a_model_data`, including:
         - `drug_id`, `drug_name_raw`, `drug_name_normalized`
         - Full numeric encoding vector from `encode_drug_name_series` (length, phonetic, positional, entropy/run metrics, etc.).
   - These artifacts are **per cohort and age band**, matching FP-Growth frequency statistics and event distributions. They are used by:
     - SHAP analysis (to interpret drug/code-related features).
     - FFA and symbolic rule extraction (to map feature indices and itemsets back to human-readable codes and drugs).

2. **6b – Final feature assembly and model selection**
   - Implemented in `6b_final_model_selection/run_final_model.py`:
     - Load event-level model data from `4a_model_data`, including protocol-filtered variants.
     - Merge FP-Growth, BupaR, DTW, and PGx patient-level features.
     - Apply target-leakage removal rules (post-event features, time-to-target, DTW-derived features, etc.).
     - Restrict to numeric features and run Monte-Carlo CV for:
       - XGBoost (GPU if available).
       - CatBoost with `grow_policy="SymmetricTree"` (oblivious trees).
       - Simple ensemble (average of XGBoost + CatBoost probabilities).
     - Export:
       - Leakage-filtered final feature table for FFA.
       - FFA-friendly model JSONs (XGBoost + CatBoost).
       - XGBoost feature importance CSV.

**Important Notes:**
- **DTW features are removed** during target leakage removal and are not used directly as model inputs.
- **Sequence information comes from BupaR**, not DTW.
- **DTW is used for protocol filtering** (preprocessing) to remove standard care patterns and reduce noise.

## Model Training and Selection

Final model development uses the **same three-model ensemble** as feature importance:

- **CatBoost** (gradient boosting on categorical features)
- **XGBoost (boosted trees)**
- **XGBoost RF mode** (random forest-style XGBoost)

**CatBoost tree structure (explainability):**

- For both feature importance and final modeling, CatBoost is explicitly configured with `grow_policy="SymmetricTree"`, forcing **oblivious (symmetric) trees**.
- This slightly constrains raw predictive flexibility, but:
  - Makes tree structure **regular and shallow**, which is much easier to convert into a unified **path DataFrame** and then into symbolic rules for FFA.
  - Ensures CatBoost’s trees are compatible with the same JSON → DataFrame → rules framework used by the XGBoost explainer.
- Empirically, this only causes a **minor change in MC‑CV metrics** while greatly improving the stability and interpretability of downstream FFA and causal analysis.

These models are compared with **Monte Carlo Cross-Validation (MC-CV)** on the training window (2016–2018),
then the best-performing base model is further tuned and calibrated before being evaluated on a strict 2019 holdout.

### MC-CV Split Strategy (Feature Importance vs Final Model)

- **Feature importance stage (3_feature_importance):**
  - Uses **`N_SPLITS = 10`** MC-CV splits per model (CatBoost, XGBoost, XGBoost RF) to keep the heavy permutation-importance workload tractable while still providing stable estimates of feature importance.
  - These runs define the **feature set and relative importance rankings** that feed into this final model module.

- **Final model stage (this module):**
  - Uses a **much larger number of MC-CV splits (target \~`N_SPLITS = 1000`)** on the selected feature set to obtain **highly stable performance estimates and uncertainty bounds** for publication-grade reporting.
  - The temporal structure is identical: each split trains on an 80% sample of **2016–2018** and is evaluated on **2019**, but now with **no permutation importance overhead**, focusing purely on predictive performance and calibration.

See `final_model.ipynb` for the full Python workflow:

- MC-CV performance comparison and model selection by mean Recall
- Optuna hyperparameter tuning on 2016–2018
- Temporal probability calibration (train on 2016–2017, calibrate on 2018)
- Final model export (joblib + native formats) locally and to S3 `gold/final_model/.../event_year=train/models/`

## Notebooks and Scripts

- `final_model.ipynb`: MC-CV comparison, Optuna tuning, temporal calibration, and final model export.
- `build_final_cohort_model_features.py`: Builds the final feature table from `model_data`, FP-Growth, BupaR, and PGx features.
  - For `non_opioid_ed` cohort: Filters to drug-only item features (excludes ICD/CPT codes for polypharmacy analysis)
- `remove_target_leakage.py`: Removes target leakage features including DTW features (DTW is for filtering, not features).
  - Validates item_* features against event data to detect post-target leakage
  - Removes non-predictive markers (SUBOXONE, BUPRENORPHINE, F1123)
  - For `non_opioid_ed` cohort: Removes any ICD/CPT features that may have slipped through
- `prepare_train_test_s3.py`: Splits final feature table into train/test sets using temporal validation and uploads to S3.
- `train_final_model.py`: Trains final model with MC-CV comparison across CatBoost, XGBoost, and XGBoost RF.
- `analyze_trigger_features.py`: Analyzes trajectory and pre-event features for triggering/thresholding, calculates cohort-specific percentiles and suggested thresholds.
- `extract_final_feature_importance.py`: Extracts, aggregates, and scales feature importances from the final trained model.
- `create_model_plots.py`: Creates visualization plots for final model feature importance analysis (same 4 plots as feature importance step).

## S3 Data Organization

### Train/Test Datasets

For distributed training or additional compute resources, train and test datasets are saved to S3:

**Local Structure:**
```
8_final_model/inputs/{cohort}/{age_band}/
├── model_train/
│   ├── final_features.parquet    # Training data (2016-2018)
│   └── metadata.json              # Dataset metadata
└── model_test/
    ├── final_features.parquet     # Test data (2019)
    └── metadata.json               # Dataset metadata
```

**S3 Structure:**
```
s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/
└── inputs/
    ├── model_train/
    │   ├── final_features.parquet    # Training data (2016-2018)
    │   └── metadata.json              # Dataset metadata
    └── model_test/
        ├── final_features.parquet     # Test data (2019)
        └── metadata.json               # Dataset metadata
```

**Temporal Split:**
- **Train:** Patients with max event year 2016-2018
- **Test:** Patients with max event year 2019

**Usage:**
```bash
# Prepare and upload train/test datasets
python 8_final_model/prepare_train_test_s3.py --cohort-name opioid_ed --age-band 0-12

# Load from local inputs folder (recommended)
import pandas as pd

train_df = pd.read_parquet('8_final_model/inputs/opioid_ed/0_12/model_train/final_features.parquet')
test_df = pd.read_parquet('8_final_model/inputs/opioid_ed/0_12/model_test/final_features.parquet')

# Or load from S3 inputs location (for distributed training)
import s3fs
s3 = s3fs.S3FileSystem()
train_df = pd.read_parquet('s3://pgxdatalake/gold/final_model/opioid_ed/0-12/inputs/model_train/final_features.parquet', filesystem=s3)
test_df = pd.read_parquet('s3://pgxdatalake/gold/final_model/opioid_ed/0-12/inputs/model_test/final_features.parquet', filesystem=s3)

# Or load from local inputs folder
train_df = pd.read_parquet('8_final_model/inputs/opioid_ed/0_12/model_train/final_features.parquet')
test_df = pd.read_parquet('8_final_model/inputs/opioid_ed/0_12/model_test/final_features.parquet')
```

## Model Visualizations

The final model uses the same visualization plots as the feature importance analysis step for consistency and comparison.

### Creating Visualizations

After extracting feature importances with `extract_final_feature_importance.py`, create visualizations:

```bash
# Create all visualization plots
python 8_final_model/create_model_plots.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --event-year 2019
```

**Output Location:**
- **Local**: `8_final_model/outputs/{cohort}/{age_band}/plots/`
- **S3**: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/plots/`

### Visualization Plots

The script creates **6 publication-quality plots** plus **2 mapping visualizations**:

1. **Top 50 Features Bar Chart** (`*_top50_features.png`)
   - Horizontal bar chart of top 50 features by scaled importance
   - Shows relative importance rankings

2. **Top 50 Features with Recall Confidence** (`*_top50_with_recall.png`)
   - Same as Plot 1, but includes recall confidence intervals (if available)
   - Shows model performance context

3. **Normalized vs Scaled Importance Comparison** (`*_normalized_vs_scaled.png`)
   - Side-by-side comparison of normalized vs recall-scaled importance
   - Shows impact of model quality weighting on feature rankings

4. **Feature Categories Distribution** (`*_category_distribution.png`)
   - Bar chart showing distribution of feature types in top 50
   - Categories: FP-Growth itemsets, BupaR features, PGx features, Pre-event counts, etc.

5. **Drug Sequence Frequency Chart** (`*_drug_sequence_frequency.png`)
   - Horizontal bar chart showing top 20 most frequent drug sequences (pre-F1120)
   - Extracted from BupaR traces
   - Shows drug sequences as "Drug1 → Drug2 → Drug3" format
   - Frequency represents number of patients with that sequence

6. **Drug/CPT Sequence to Target Frequency Chart** (`*_drug_cpt_sequence_frequency.png`)
   - Horizontal bar chart showing top 20 most frequent drug/CPT sequences leading to target (pre-F1120)
   - Extracted from BupaR traces
   - Shows combined drug and CPT sequences as "DRUG: Drug1 → CPT: Code1 → DRUG: Drug2" format
   - Only includes sequences with both drugs and CPTs
   - Frequency represents number of patients with that sequence before F1120

7. **Sankey Diagram: Sequence & Itemset → Feature Mapping** (`*_sequence_feature_mapping_sankey.html`)
   - Interactive Sankey diagram showing **parallel feature engineering flows**:
     - **Left side (parallel)**: 
       - BupaR drug sequences (top sequences from patient traces)
       - FP-Growth drug itemsets (frequent co-occurring drug sets)
     - **Right side**: Final feature importances
       - BupaR sequence features (e.g., `overall_is_top_sequence`, `overall_is_rare_sequence`)
       - FP-Growth itemset features (e.g., `drug_name_itemset_6_match`, `drug_name_itemset_6_support`)
   - **Alignment with Feature Engineering**:
     - ✅ **BupaR Sequences** → **BupaR Sequence Features** (direct mapping: if patient sequence matches top/rare sequence, they get the feature)
     - ✅ **FP-Growth Itemsets** → **FP-Growth Itemset Features** (direct mapping: if patient has the itemset, they get the feature)
     - Both feature types flow to Final Features (they are the final features)
   - Link thickness represents mapping strength (frequency × importance)
   - Interactive HTML file - hover over nodes/links for details
   - **Requires Plotly**: Install with `pip install plotly`

8. **Mapping Table: Sequence & Itemset → Feature** (`*_sequence_feature_mapping_table.csv` and `.html`)
   - CSV and HTML tables showing detailed mappings aligned with feature engineering:
     - **BupaR Sequences** → **BupaR Sequence Features** (e.g., `overall_is_top_sequence`)
     - **FP-Growth Itemsets** → **FP-Growth Itemset Features** (e.g., `drug_name_itemset_X_match`)
   - Columns: Source, Source Name, Source Frequency, Feature Type, Feature Name, Feature Rank, Feature Importance
   - Sorted by Feature Rank and Source type
   - Shows which sequences/itemsets contribute to which final features
   - **Note**: BupaR sequence features may have 0 importance in some cohorts (not predictive), but they are still included in the final feature set

### Usage Examples

```python
# From Python script or notebook
from py_helpers.create_feature_importance_visualizations import create_feature_importance_plots

plot_files = create_feature_importance_plots(
    aggregated_file='8_final_model/outputs/opioid_ed/0_12/opioid_ed_0_12_final_feature_importance_aggregated_scaled.csv',
    output_dir='8_final_model/outputs/opioid_ed/0_12',
    cohort_name='opioid_ed',
    age_band='0-12',
    event_year=2019,
    s3_upload=True
)
```

### Plot Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG
- **Style**: Seaborn whitegrid style
- **Cross-platform**: Works on both Linux EC2 (headless) and Windows

**Note**: These plots use the same visualization function as the feature importance step (`py_helpers/create_feature_importance_visualizations.py`), ensuring consistency across the analysis workflow.

## Feature Validation

### Missing Values
- **Categorical**: Use "unknown" or mode imputation
- **Continuous**: Use median or mean imputation
- **Binary**: Use 0 (absence) for missing

### Feature Scaling
- **CatBoost**: No scaling needed (handles categoricals natively)
- **Random Forest**: No scaling needed (tree-based)
- **Logistic Regression**: Standardize continuous features

### Expected Feature Importance
- **High importance**: 
  - FPGrowth: `rules_target_icd_match`, `max_rule_lift_target_icd`, itemset match counts
  - BupaR: Sequence features (`overall_is_top_sequence`, `overall_is_rare_sequence`), pre-event counts
  - PGx: `pgx_genes_covered`, `pgx_drugs_with_mappings`
  - Pre-event counts: `pre_n_events`, `pre_n_unique_activities`
- **DTW features**: **NOT included** - DTW is used for protocol filtering only

## Using Features for Triggering/Thresholding

The final model features can be used to create **predictive triggers** for real-time patient risk assessment. This section documents which features are suitable for triggering and provides guidance on threshold selection.

### Trigger-Ready Features

#### 1. Trajectory Features (Filtered by FP-Growth Itemsets)

**Features:**
- `combined_trajectory_length`: Number of events in patient trajectory (filtered by important codes)
- `combined_trajectory_diversity`: Number of unique items in trajectory (filtered)

**Characteristics:**
- **Source**: DTW analysis pipeline (trajectory characteristics only, not DTW distances)
- **Filtering**: Only includes events/codes identified as important by FP-Growth analysis
- **Use Case**: Trigger on patients with **complex trajectories involving important codes**
- **Advantage**: Focused on quality (important codes) rather than quantity (all events)
- **Scaling**: Better for larger cohorts (filtered = less noise)

**Feature Importance Ranking:**
- `combined_trajectory_diversity`: Rank #22 (9.57% scaled importance)
- `combined_trajectory_length`: Rank #36 (8.15% scaled importance)

**Suggested Thresholds (Percentile-Based):**
- **Medium Risk** (>75th percentile): `trajectory_length > 24`, `trajectory_diversity > 11`
- **High Risk** (>90th percentile): `trajectory_length > 49`, `trajectory_diversity > 17`
- **Very High Risk** (>95th percentile): `trajectory_length > 68`, `trajectory_diversity > 19`

*Note: Thresholds are cohort/age-band specific. Use `analyze_trigger_features.py` to calculate cohort-specific percentiles.*

#### 2. Pre-Event Count Features (All Events Before F1120)

**Features:**
- `pre_n_events`: Total events before F1120 (**#1 feature, 100% importance**)
- `pre_n_unique_activities`: Unique activities before F1120 (**#2 feature, 85% importance**)
- `pre_n_icd_events`: ICD events before F1120 (#3 feature, 83% importance)
- `pre_n_cpt_events`: CPT events before F1120 (#4 feature, 74% importance)
- `pre_n_drug_events`: Drug events before F1120

**Characteristics:**
- **Source**: BupaR analysis (all pre-F1120 events)
- **Filtering**: Includes **ALL events** before target, not filtered
- **Use Case**: Trigger on patients with **high overall healthcare utilization**
- **Advantage**: Captures total event volume, not just important codes
- **Best For**: Identifying patients with high healthcare engagement

**Suggested Thresholds (Percentile-Based):**
- **Medium Risk** (>75th percentile): `pre_n_events > 23`, `pre_n_unique_activities > 11`
- **High Risk** (>90th percentile): `pre_n_events > 35`, `pre_n_unique_activities > 17`
- **Very High Risk** (>95th percentile): `pre_n_events > 58`, `pre_n_unique_activities > 24`

*Note: Thresholds are cohort/age-band specific. Use `analyze_trigger_features.py` to calculate cohort-specific percentiles.*

### Key Differences: Trajectory vs Pre-Event Features

| Feature Type | What It Measures | Best For | Correlation |
|-------------|------------------|----------|-------------|
| **Trajectory** | Events/codes filtered by FP-Growth (important only) | **Quality**: Patients with many important events | Low correlation with pre-event features |
| **Pre-Event** | All events before F1120 | **Quantity**: Patients with high overall utilization | - |

**Important:** These features capture **different patterns** (negative correlation ~-0.30), so combining them provides complementary signals.

### Best Practices for Triggering

1. **Use Percentile-Based Thresholds**
   - Avoid absolute values (e.g., "> 50 events")
   - Use percentiles (e.g., "> 90th percentile") for cohort-specific adaptation
   - Recalculate thresholds per cohort/age band

2. **Combine Multiple Features**
   ```python
   # Example: Multi-feature trigger
   IF (trajectory_length > 90th_percentile) 
      AND (pre_n_events > 75th_percentile) 
   THEN flag_high_risk()
   ```

3. **Use Trajectory for Quality, Pre-Event for Quantity**
   - Trajectory features = important codes (quality signal)
   - Pre-event features = total volume (quantity signal)
   - Combining both captures both dimensions

4. **Age-Band-Specific Thresholds**
   - Different age bands have different event distributions
   - Calculate thresholds separately per age band
   - Use `analyze_trigger_features.py` to generate cohort-specific thresholds

5. **For Larger Cohorts**
   - Trajectory features scale better (filtered = less noise)
   - Pre-event features may have more variance in larger cohorts
   - Consider using trajectory features as primary triggers for larger cohorts

### Analysis Tool

Use the provided analysis script to calculate cohort-specific thresholds:

```bash
# Analyze trigger features for a specific cohort/age band
python 8_final_model/analyze_trigger_features.py \
    --cohort-name opioid_ed \
    --age-band 0-12
```

**Output:**
- Feature distributions (min, max, mean, median, percentiles)
- Target vs control comparisons
- Suggested thresholds (75th, 90th, 95th percentiles)
- Correlation analysis between trajectory and pre-event features
- Feature importance rankings

### Example Trigger Implementation

```python
import pandas as pd
import numpy as np

def calculate_trigger_thresholds(df, feature_col, percentiles=[0.75, 0.90, 0.95]):
    """Calculate percentile-based thresholds for a feature."""
    thresholds = {}
    for p in percentiles:
        thresholds[f'{int(p*100)}th'] = df[feature_col].quantile(p)
    return thresholds

def flag_high_risk_patients(df, cohort_name, age_band):
    """Flag high-risk patients using trajectory and pre-event features."""
    # Load cohort-specific thresholds (calculated from training data)
    # For production, these should be pre-calculated and stored
    
    # Example thresholds (cohort-specific)
    traj_length_threshold = df['combined_trajectory_length'].quantile(0.90)
    pre_events_threshold = df['pre_n_events'].quantile(0.90)
    
    # Multi-feature trigger
    high_risk = (
        (df['combined_trajectory_length'] > traj_length_threshold) |
        (df['pre_n_events'] > pre_events_threshold)
    )
    
    return high_risk

# Usage
feature_df = pd.read_csv('final_features_no_leakage.csv')
high_risk_patients = flag_high_risk_patients(feature_df, 'opioid_ed', '0-12')
```

### Important Notes

- **Trajectory features are NOT DTW distance features**: They are simple trajectory characteristics (length, diversity) calculated from filtered trajectories. DTW distance features were removed.
- **Thresholds are cohort-specific**: Always calculate thresholds from training data for the specific cohort/age band.
- **Use training data for thresholds**: Calculate thresholds from training set (2016-2018), apply to test/production data.
- **Monitor threshold performance**: Track false positive/negative rates and adjust thresholds based on clinical feedback.

## Important Notes

1. **DTW Role**: DTW is used for **protocol filtering** (preprocessing), **NOT as features**. DTW features are removed during target leakage removal.
2. **Sequence Information**: All sequence features come from **BupaR**, not DTW. BupaR provides top/rare sequence patterns, sequence frequencies, and sequence categories.
3. **Protocol Filtering**: Use `model_events_no_protocols.parquet` (created by `6_dtw_analysis/filter_protocol_events.py`) for cleaner feature engineering.
4. **Feature Count**: Actual feature count varies based on:
   - Number of frequent itemsets discovered (FPGrowth)
   - Number of sequence patterns discovered (BupaR)
   - Number of PGx drug-gene mappings
5. **Target Leakage Removal**: The `remove_target_leakage.py` script automatically removes:
   - Post-event features
   - Time-to-target features
   - DTW features (DTW is for filtering, not features)
   - F1120 code itself
   - Post-target drug/ICD/CPT events (validates against event data)
   - Non-predictive markers (SUBOXONE, BUPRENORPHINE, F1123)

6. **Drug name column exclusions**: The following values are excluded from the drug name feature set for model training (see `DRUG_NAMES_EXCLUDED_MODEL_TRAINING` in `py_helpers.constants` and `1b_apcd_event_filter/README_administrative_codes_lookup.md`): **Narcan**, **Unknown**, **Fentanyl**, **1036F**, **T401XA1**. 1036F is a CPT Category II tracking code (tobacco non-user), not a drug; T401XA1 is an ICD-10-CM poisoning diagnosis code (4-aminophenol/acetaminophen, initial encounter), not a drug.

7. **Cohort-Specific Feature Filtering**:
   - **non_opioid_ed (Polypharmacy) Cohort**: Only drug events are included as item features
     - Excludes ICD codes (`item_icd_*`)
     - Excludes CPT codes (`item_cpt_*`)
     - Includes only drug features (`item_drug_*`)
     - Rationale: Polypharmacy analysis focuses on drug interactions and medication patterns
   - **opioid_ed Cohort**: Includes all event types (drugs, ICD codes, CPT codes)

## TODOs

- [ ] Implement feature engineering pipeline script
- [ ] Create feature extraction utilities for FPGrowth, BupaR, DTW
- [ ] Feature importance exploration: identify which features most strongly predict target outcomes
- [ ] Use model-based importance and SHAP summaries to filter to manageable feature set
- [ ] Post-model: revisit ICD/CPT/Drug heatmaps with top features only

## References

- **Feature Schema**: `final_feature_schema.json` - Complete JSON Schema definition
- **FPGrowth Analysis**: `../4_fpgrowth_analysis/`
- **BupaR Analysis**: `../5_bupaR_analysis/`
- **DTW Analysis**: `../6_dtw_analysis/DTW_ROLE.md` - DTW is used for protocol filtering, not features
- **Protocol Filtering**: `../6_dtw_analysis/PROTOCOL_FILTERING.md` - How DTW time windows filter protocol events
- **Notebook Integration**: See `../docs/notebook_calls.md` for calls and patterns


