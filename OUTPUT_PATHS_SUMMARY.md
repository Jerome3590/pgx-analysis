# Workflow Step Output Paths Summary

This document summarizes where each workflow step writes its outputs after the folder structure updates.

## Step Output Locations

### Step 3: Feature Importance
**Folder**: `3_feature_importance/`  
**Output Location**: `3_feature_importance/outputs/{cohort}/`  
**Key Files**:
- `{cohort}_{age_band}_aggregated_feature_importance.csv`

### Step 4a: Model Data Extraction
**Folder**: `4a_model_data/`  
**Output Location**: `4a_model_data/cohort_name={cohort}/age_band={age_band}/`  
**Key Files**:
- `model_events.parquet` (cases + controls)

### Step 4b: DTW Protocol Filtering
**Folder**: `4b_dtw_filter/`  
**Output Location**: `4b_dtw_filter/outputs/`  
**Key Files**:
- `model_events_no_protocols.parquet`
- `protocol_summary_*.csv`

### Step 5: PGx Feature Engineering
**Folder**: `5_pgx_analysis/` (renamed from `5c_pgx_analysis`)  
**Output Location**: `5_pgx_analysis/outputs/`  
**Key Files**:
- `outputs/global/` - Global PGx cache (shared across cohorts)
  - `pgx_drug_gene_mappings_global.csv`
  - `pgx_allele_frequencies_global.csv`
- `outputs/{cohort}/` - Cohort-specific outputs
  - `{cohort}_drug_gene_mappings.csv`
  - `{cohort}_allele_frequencies.csv`
- `outputs/{cohort}/{age_band}/` - Age-band specific features
  - `pgx_added_features_{cohort}_{age_band}.csv`

### Step 6: Final Model Training
**Folder**: `6_final_model_selection/` (renamed from `6b_final_model_selection`)  
**Output Location**: `6_final_model/outputs/{cohort}/{age_band_fname}/`  
**Key Files**:
- `{cohort}_{age_band_fname}_train_final_features_no_leakage.csv`
- `{cohort}_{age_band_fname}_best_xgboost_model.json`
- `{cohort}_{age_band_fname}_best_catboost_model.cbm`
- `{cohort}_{age_band_fname}_xgboost_feature_importance.csv`
- `models/` subdirectory with additional model artifacts

**Note**: Script is in `6_final_model_selection/` but outputs go to `6_final_model/outputs/`

### Step 7: FFA Analysis
**Folder**: `7_ffa_analysis/`  
**Output Location**: `7_ffa_analysis/outputs/{cohort}/{age_band_fname}/`  
**Key Files**:
- `xgboost/axp_explanations.csv`
- `xgboost/feature_importance_axp.csv`

### Step 8: SHAP Analysis
**Folder**: `8_shap_analysis/`  
**Output Location**: `8_shap_analysis/outputs/{cohort}/{age_band_fname}/`  
**Key Files**:
- `{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv`
- `{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet`

### Step 9: Combined SHAP + FFA
**Folder**: `9_combined_shap_ffa/`  
**Output Location**: `9_combined_shap_ffa/outputs/{cohort}/{age_band_fname}/`  
**Key Files**:
- Combined analysis outputs

### Step 10: Risk Dashboard
**Folder**: `10_risk_dashboard/`  
**Output Location**: `10_risk_dashboard/` (various subdirectories)  
**Key Files**:
- `models/{cohort}/{age_band}/` - Packaged models for deployment
- `metadata/metadata_{cohort}.json` - Valid codes for dropdowns
- Visualization data in `bupaR_dashboard_visual/`, `fpgrowth_dashboard_visual/`, `dtw_dashboard_visual/`

## S3 Output Locations

All steps also upload outputs to S3 under:
- **Checkpoints**: `s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/`
- **Outputs**: `s3://pgxdatalake/gold/{step_name}/` (varies by step)

## Verification

All output paths have been updated to use:
- ✅ `5_pgx_analysis` (instead of `5c_pgx_analysis`)
- ✅ `6_final_model_selection` (instead of `6b_final_model_selection`)
- ✅ Outputs write to correct folder locations
- ✅ Workflow scripts reference updated paths
- ✅ Checkpoint system uses updated step names

