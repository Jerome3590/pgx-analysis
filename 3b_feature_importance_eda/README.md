# Step 3b: Feature Importance EDA and Refinement

## Overview

Step 3b performs additional exploratory data analysis on aggregated feature importances from Step 3, using:
1. **BupaR analysis** for transactions past the target event
2. **DTW analysis** for trajectories and non-value-added ICD/CPT codes

Based on this EDA, we filter and update the aggregated feature importances to produce refined `cohort_feature_importance` files that feed into Step 4a model data creation.

## Purpose

- **Identify post-target leakage**: Use BupaR to analyze sequences after the target event to identify features that may leak future information
- **Filter non-value-added codes**: Use DTW trajectory analysis to identify administrative, scheduling, and non-medical codes that don't add predictive value
- **Refine feature importances**: Update aggregated feature importances based on EDA findings
- **Output refined features**: Generate `cohort_feature_importance` files for Step 4a

## Inputs

- **Aggregated feature importances** from Step 3:
  - `3_feature_importance/outputs/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv`
- **Model events data** (for BupaR/DTW analysis):
  - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`

## Outputs

### Local Files
- **Refined feature importances**:
  - `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
- **EDA reports**:
  - `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_bupar_post_target_analysis.csv`
  - `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_dtw_trajectory_analysis.csv`
  - `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_feature_filtering_summary.json`

### S3 Checkpoints
All outputs are automatically uploaded to S3 for checkpointing and downstream consumption:
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_bupar_post_target_analysis.csv`
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_dtw_trajectory_analysis.csv`
- `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_feature_filtering_summary.json`

**Note**: Uploads are idempotent - files are only uploaded if they don't already exist in S3.

## Workflow

1. **Load aggregated feature importances** from Step 3
2. **BupaR Post-Target Analysis**:
   - Analyze sequences after target event
   - Identify features that appear primarily post-target (potential leakage)
   - Flag features for filtering
3. **DTW Trajectory Analysis**:
   - Analyze trajectories for non-value-added codes
   - Identify administrative, scheduling, and non-medical codes
   - Flag features for filtering
4. **Filter and Update Feature Importances**:
   - Remove flagged features
   - Adjust importance scores based on EDA findings
   - Generate refined `cohort_feature_importance` files
5. **Save outputs locally and upload to S3**:
   - Save all outputs to local filesystem
   - Upload to S3 for checkpointing and Step 4a consumption
   - Save checkpoint metadata to S3

## Scripts

- `run_bupar_post_target_analysis.py` - BupaR analysis for post-target transactions
- `run_dtw_trajectory_analysis.py` - DTW analysis for trajectories and non-value-added codes
- `filter_and_refine_features.py` - Main script to filter and refine feature importances
- `run_step_3b.py` - Orchestration script to run all analyses

## Usage

```bash
# Run for a single cohort/age_band
python 3b_feature_importance_eda/run_step_3b.py --cohort opioid_ed --age-band 13-24

# Run for all cohorts
python 3b_feature_importance_eda/run_step_3b.py --all-cohorts
```

## Integration with Pipeline

- **Input**: Step 3 aggregated feature importances
- **Output**: Refined `cohort_feature_importance` files
- **Consumed by**: Step 4a model data creation
