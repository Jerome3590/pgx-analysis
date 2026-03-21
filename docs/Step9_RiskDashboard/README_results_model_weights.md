# Model Performance-Based Weighting

## Overview

The ensemble uses **performance-based weights** calculated from Monte Carlo Cross-Validation (MC-CV) results from **Step 6** (`6_final_model`). This ensures that better-performing models contribute more to the final prediction.

## Weight Calculation Process

### Step 1: Extract MC-CV Results

During model preparation (`prepare_models.py`), the script reads:
```
6_final_model/outputs/{cohort}/{age_band}/models/{cohort}_{age_band}_mc_cv_results.csv
```

This CSV contains performance metrics for each model across multiple MC-CV splits:
- `logloss`: Lower is better
- `pr_auc`: Higher is better (Precision-Recall AUC)

### Step 2: Calculate Composite Scores

For each model, calculate a composite score:

```python
mean_logloss = mc_cv_results['logloss'].mean()
mean_pr_auc = mc_cv_results['pr_auc'].mean()

# Normalize logloss: 1 / (1 + logloss) - higher is better
normalized_logloss = 1 / (1 + mean_logloss)

# PR-AUC is already in [0, 1], higher is better
normalized_pr_auc = mean_pr_auc

# Composite score: equal weight to both metrics
composite_score = 0.5 × normalized_pr_auc + 0.5 × normalized_logloss
```

**Example** (from `opioid_ed` / `0-12`):
- CatBoost: logloss=1.0023, PR-AUC=0.4590 → composite_score=0.4796
- XGBoost: logloss=1.0069, PR-AUC=0.4500 → composite_score=0.4745
- XGBoost RF: logloss=0.7042, PR-AUC=0.4921 → composite_score=0.5441

### Step 3: Normalize to Weights

Normalize composite scores to sum to 1.0:

```python
total_composite = sum(composite_scores.values())
weights = {
    model: composite_score / total_composite
    for model, composite_score in composite_scores.items()
}
```

**Example** (from above):
- CatBoost: 0.4796 / 1.4982 = **0.320**
- XGBoost: 0.4745 / 1.4982 = **0.317**
- XGBoost RF: 0.5441 / 1.4982 = **0.363** (highest weight - best performance)

### Step 4: Store in Feature Schema

Weights are stored in `feature_schema.json`:

```json
{
  "features": [...],
  "defaults": {...},
  "model_weights": {
    "catboost": 0.320,
    "xgboost": 0.317,
    "xgboost_rf": 0.363
  }
}
```

## Usage in Lambda Function

### Loading Weights

```python
feature_schema = load_feature_schema(cohort, age_band)
model_weights = feature_schema.get('model_weights', {
    'catboost': 1.0/3,
    'xgboost': 1.0/3,
    'xgboost_rf': 1.0/3
})
```

### Applying Weights

```python
# Only use weights for models that succeeded
available_weights = {m: model_weights.get(m, 0.0) for m in predictions.keys()}
total_weight = sum(available_weights.values())

# Weighted average
ensemble_score = sum(
    predictions[m] × available_weights[m]
    for m in predictions.keys()
) / total_weight
```

## Benefits

1. **Performance-Driven**: Better models get higher weights
2. **Automatic**: Weights calculated from actual validation performance
3. **Robust**: Falls back to equal weights if MC-CV data unavailable
4. **Transparent**: Weights included in API response

## Fallback Behavior

If MC-CV results are not available:
- Equal weights are used: `1/3` for each model
- Warning is logged
- API response indicates `weights_source: 'equal_fallback'`

## Example API Response

```json
{
  "risk_score": 0.65,
  "risk_band": "high",
  "model_breakdown": {
    "catboost": 0.64,
    "xgboost": 0.66,
    "xgboost_rf": 0.65
  },
  "ensemble_info": {
    "method": "performance_weighted_average",
    "models_used": 3,
    "models_failed": [],
    "weights": {
      "catboost": 0.320,
      "xgboost": 0.317,
      "xgboost_rf": 0.363
    },
    "weights_source": "mc_cv_performance"
  }
}
```

## Verification

To verify weights are being used correctly:

1. Check `feature_schema.json` includes `model_weights`
2. Check Lambda logs show: `Using model weights: {...}`
3. Check API response includes `weights_source: 'mc_cv_performance'`
4. Verify weights sum to ~1.0 (allowing for rounding)

## Updating Weights

If models are retrained and MC-CV results change:

1. Re-run `prepare_models.py` to recalculate weights
2. Rebuild Docker container image
3. Update Lambda function with new image
4. Weights will automatically update

## References

- MC-CV Results: `6_final_model/outputs/{cohort}/{age_band}/models/{cohort}_{age_band}_mc_cv_results.csv`
- Model Summary: `6_final_model/outputs/{cohort}/{age_band}/models/{cohort}_{age_band}_model_summary.txt`
- Feature Schema (after `10_risk_dashboard/data_preparation/prepare_models.py`): `10_risk_dashboard/outputs/models/{cohort}/{age_band}/feature_schema.json` (also uploaded under `s3://pgxdatalake/gold/dashboard/models/...` for Lambda)

