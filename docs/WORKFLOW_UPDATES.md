# Workflow Pipeline Updates

## Overview

The workflow has been streamlined to focus on PGx analysis as the primary feature engineering step, with other feature engineering methods (BupaR, DTW, FP-Growth) moved to the risk dashboard for visualizations only.

## New Pipeline Steps

### Step 3: Feature Importance
- **Purpose**: Check for completed aggregated feature importances
- **Script**: `3_feature_importance/run_mc_feature_importance.py`
- **Output**: `{cohort}_{age_band}_aggregated_feature_importance.csv`
- **Note**: Idempotent - skips if results already exist

### Step 4b: DTW Protocol Filtering
- **Purpose**: Filter administrative, scheduling, and non-medical related codes
- **Keep**: All surgeries
- **Script**: `4b_dtw_filter/filter_protocol_events.py`
- **Output**: Filtered model events data

### Step 5c: PGx Feature Engineering
- **Purpose**: Generate PGx features (ONLY feature engineering step)
- **Script**: `5c_pgx_analysis/run_analysis.py`
- **Output**: PGx feature tables
- **Note**: BupaR, DTW, and FP-Growth are no longer used as features, only for dashboard visualizations

### Step 6: Final Model Training
- **Purpose**: Train final models using aggregated features + PGx features
- **Script**: `6b_final_model_selection/run_final_model.py`
- **Key Changes**:
  - Uses aggregated feature importances directly (no encoding)
  - Only combines with PGx features (no BupaR, DTW, FP-Growth features)
  - Trains both CatBoost and XGBoost (XGBoost vs XGBoost RF)
  - Model selection based on **recall** and **AUC-PR**
  - Outputs:
    - Best CatBoost model binary (for SHAP analysis)
    - Best XGBoost model JSON (for FFA analysis)

### Step 7: SHAP Analysis
- **Purpose**: SHAP value analysis using best CatBoost model
- **Script**: `8_shap_analysis/run_shap_analysis.py`
- **Input**: Best CatBoost model binary from Step 6
- **Output**: SHAP global importance scores (used by Step 8)

### Step 8: FFA Analysis
- **Purpose**: Feature Forward Analysis using best XGBoost model
- **Script**: `7_ffa_analysis/run_full_ffa_analysis.py`
- **Input**: Best XGBoost model JSON from Step 6 + SHAP importance from Step 7
- **Note**: Uses SHAP importance to prioritize rules for AXP computation. Consensus between SHAP and FFA is reflected in FFA's causal importance scores.

### Step 9: Risk Dashboard Preparation
- **Purpose**: Integrate BupaR, DTW, and FP-Growth visualizations with causal analysis
- **Location**: `9_risk_dashboard/`
- **Note**: These methods are now used for visualization only, not as model features. Dashboard deployment is also handled here.

## Removed Steps

The following steps are no longer part of the main pipeline (moved to dashboard visualizations):
- Step 5a: BupaR Process Mining (now dashboard only)
- Step 5b: FP-Growth Analysis (now dashboard only)
- Step 5d: DTW Trajectory Analysis (now dashboard only)

## Model Selection Criteria

The final model step now uses:
- **Primary metrics**: Recall and AUC-PR (Average Precision)
- **Model comparison**: XGBoost vs XGBoost RF
- **Selection**: Best model based on combined recall and AUC-PR performance

## Feature Engineering Changes

### Before
- Multiple feature engineering steps: BupaR, FP-Growth, DTW, PGx
- All features combined for final model
- Feature encoding required

### After
- Single feature engineering step: PGx only
- Aggregated feature importances used directly (no encoding)
- Only PGx features added to aggregated features
- BupaR, DTW, FP-Growth used for visualizations only

