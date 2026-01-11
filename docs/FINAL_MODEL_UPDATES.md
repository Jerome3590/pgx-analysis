# Final Model Script Updates Required

## Summary of Changes Needed

The `6b_final_model_selection/run_final_model.py` script needs significant updates to match the new workflow requirements.

## Key Changes

### 1. Feature Loading (COMPLETED)
- ✅ Removed FP-Growth, BupaR, DTW feature loading
- ✅ Only loads PGx features now
- ✅ Keeps patient-level aggregated features (drug/ICD/CPT encodings)

### 2. Model Training (NEEDS UPDATE)
- **Current**: Trains XGBoost and CatBoost only
- **Required**: Train XGBoost, XGBoost RF, and CatBoost
- **Required**: Track metrics separately for XGBoost and XGBoost RF

### 3. Model Selection (NEEDS UPDATE)
- **Current**: No explicit model selection
- **Required**: Compare XGBoost vs XGBoost RF using:
  - **Primary**: Recall (mean across MC runs)
  - **Secondary**: AUC-PR (mean across MC runs)
- **Required**: Select best XGBoost variant (XGBoost or XGBoost RF)
- **Required**: Select best CatBoost (only one variant, but track for consistency)

### 4. Model Export (NEEDS UPDATE)
- **Current**: Exports XGBoost JSON and CatBoost JSON
- **Required**: Export **best** CatBoost model as **binary** (`.cbm` file for SHAP)
- **Required**: Export **best** XGBoost model as **JSON** (for FFA)
- **Required**: Save model selection metadata (which variant was selected and why)

## Implementation Details

### Model Training Loop Changes

In the MC-CV loop, need to:
1. Train XGBoost (boosting)
2. Train XGBoost RF (random forest)
3. Train CatBoost
4. Track metrics for all three separately

### Model Selection Logic

After MC-CV:
```python
# Compare XGBoost variants
xgb_recall_mean = np.mean(metrics["xgb"]["recall"])
xgb_pr_auc_mean = np.mean(metrics["xgb"]["pr_auc"])
xgb_rf_recall_mean = np.mean(metrics["xgb_rf"]["recall"])
xgb_rf_pr_auc_mean = np.mean(metrics["xgb_rf"]["pr_auc"])

# Select best XGBoost variant
# Primary: Recall, Secondary: AUC-PR
if xgb_recall_mean > xgb_rf_recall_mean:
    best_xgb_variant = "xgb"
elif xgb_recall_mean < xgb_rf_recall_mean:
    best_xgb_variant = "xgb_rf"
else:
    # Tie on recall, use AUC-PR
    if xgb_pr_auc_mean >= xgb_rf_pr_auc_mean:
        best_xgb_variant = "xgb"
    else:
        best_xgb_variant = "xgb_rf"
```

### Model Export Changes

1. **Best CatBoost Binary**:
   - Train final CatBoost on full data
   - Save as `.cbm` file: `{cohort}_{age_band}_best_catboost_model.cbm`
   - This will be used by SHAP analysis

2. **Best XGBoost JSON**:
   - Train final best XGBoost variant on full data
   - Export as JSON: `{cohort}_{age_band}_best_xgboost_model.json`
   - This will be used by FFA analysis

3. **Model Selection Metadata**:
   - Save JSON with selection rationale:
     ```json
     {
       "best_xgb_variant": "xgb" or "xgb_rf",
       "xgb_recall_mean": 0.85,
       "xgb_pr_auc_mean": 0.78,
       "xgb_rf_recall_mean": 0.83,
       "xgb_rf_pr_auc_mean": 0.76,
       "selection_reason": "XGBoost selected due to higher recall (0.85 vs 0.83)"
     }
     ```

## Files to Update

1. `6b_final_model_selection/run_final_model.py` - Main script updates
2. `8_shap_analysis/run_shap_analysis.py` - Update to use best CatBoost binary
3. `7_ffa_analysis/run_full_ffa_analysis.py` - Update to use best XGBoost JSON

## Testing Checklist

- [ ] XGBoost RF training works correctly
- [ ] Metrics tracked separately for XGBoost and XGBoost RF
- [ ] Model selection logic works correctly
- [ ] Best CatBoost binary saved correctly
- [ ] Best XGBoost JSON saved correctly
- [ ] Model selection metadata saved
- [ ] SHAP analysis can load best CatBoost binary
- [ ] FFA analysis can load best XGBoost JSON

