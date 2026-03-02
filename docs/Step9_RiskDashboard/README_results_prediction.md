# Prediction Workflow: How New Predictions Work

## Overview

Yes! You can use the previously trained models with **just the inputs you care about** (ICD codes, CPT codes, drug names, and age). The system automatically handles missing features by using default values.

## Complete Prediction Flow

### Step 1: User Inputs

**What you provide:**
```json
{
  "age": 35,
  "drugs": ["AMOXICILLIN", "METHYLPHENIDATE HYDROCHLO"],
  "icds": ["R51", "G89"],  // Note: F1120 is excluded (it's the target, not an input)
  "cpts": ["80305", "99213"]
}
```

**What you DON'T need to provide (and are not in the model):**
- Feature engineering **never generates** trajectory, sequence, or itemset features; the schema has no such columns.
- n_events, n_drugs, pgx_num_drugs, pgx_num_cpic_drugs can be omitted; the API fills them from schema defaults if not provided.

### Step 2: Feature Vector Building

The `build_feature_vector()` function creates a complete feature vector:

#### 2.1 Initialize All Features to Zero
```python
# All features from training set initialized to 0.0
# Schema includes: age, item_* (drug/ICD/CPT), n_events, n_drugs, pgx_*; no trajectory/sequence/itemset (never generated)
features = {
    'age': 0.0,
    'item_AMOXICILLIN': 0.0,
    'item_R51': 0.0,
    'item_G89': 0.0,
    'item_80305': 0.0,
    'n_events': 0.0,
    'n_drugs': 0.0,
    'pgx_num_drugs': 0.0,
    'pgx_num_cpic_drugs': 0.0,
    # ... all other item_* and schema features
}
```

#### 2.2 Set User-Provided Values
```python
# Age (always provided)
features['age'] = 35.0

# Drug names → item_{DRUG_NAME} = 1.0
features['item_AMOXICILLIN'] = 1.0
features['item_METHYLPHENIDATE HYDROCHLO'] = 1.0

# ICD codes → item_{ICD_CODE} = 1.0
features['item_R51'] = 1.0
features['item_G89'] = 1.0

# CPT codes → item_{CPT_CODE} = 1.0
features['item_80305'] = 1.0
features['item_99213'] = 1.0
```

#### 2.3 Fill Missing Features with Defaults
```python
# Non-item features (n_events, n_drugs, pgx_*, etc.) use schema defaults when not provided
defaults = feature_schema.get('defaults', {})

# Example defaults (from training data medians):
features['n_events'] = defaults.get('n_events', 8.0)
features['n_drugs'] = defaults.get('n_drugs', 3.0)
features['pgx_num_drugs'] = defaults.get('pgx_num_drugs', 0.0)
features['pgx_num_cpic_drugs'] = defaults.get('pgx_num_cpic_drugs', 0.0)
# ... etc for all non-item features
```

**Key Point**: The model schema does **not** include trajectory/sequence/itemset features (feature engineering never generates them). Missing non-item features get **default values from the schema** (e.g. training medians).

### Step 3: Model Prediction

#### 3.1 Load Models (Cached)
```python
# Models loaded once, cached in Lambda memory
catboost_model = load_model(cohort, age_band, 'catboost')
xgboost_model = load_model(cohort, age_band, 'xgboost')
xgboost_rf_model = load_model(cohort, age_band, 'xgboost_rf')
```

#### 3.2 Run Predictions
```python
# Each model predicts probability
catboost_prob = catboost_model.predict_proba(feature_vector)[0][1]  # e.g., 0.64
xgboost_prob = xgboost_model.predict_proba(feature_vector)[0][1]    # e.g., 0.66
xgboost_rf_prob = xgboost_rf_model.predict_proba(feature_vector)[0][1]  # e.g., 0.65
```

**Important**: Models were trained on **complete feature vectors** with all features. When you provide partial inputs:
- Your provided features (ICD/CPT/drugs) are set to 1.0
- Missing features use training data medians
- Models handle this naturally (they've seen similar patterns during training)

### Step 4: Ensemble Combination

#### 4.1 Load Performance-Based Weights
```python
# Weights from MC-CV results (e.g., from opioid_ed/0_12)
weights = {
    'catboost': 0.320,
    'xgboost': 0.317,
    'xgboost_rf': 0.363  # Best performer
}
```

#### 4.2 Calculate Weighted Average
```python
ensemble_score = (
    catboost_prob × 0.320 +
    xgboost_prob × 0.317 +
    xgboost_rf_prob × 0.363
)
# Example: (0.64 × 0.320) + (0.66 × 0.317) + (0.65 × 0.363) = 0.651
```

### Step 5: Return Results

```json
{
  "risk_score": 0.651,
  "risk_band": "high",
  "model_breakdown": {
    "catboost": 0.64,
    "xgboost": 0.66,
    "xgboost_rf": 0.65
  },
  "ensemble_info": {
    "method": "performance_weighted_average",
    "weights": {
      "catboost": 0.320,
      "xgboost": 0.317,
      "xgboost_rf": 0.363
    }
  }
}
```

## Example Scenarios

### Scenario 1: Minimal Input (Just Age + One Drug)
```json
{
  "age": 45,
  "drugs": ["AMOXICILLIN"],
  "icds": [],
  "cpts": []
}
```

**What happens:**
- `item_AMOXICILLIN` = 1.0
- All other item features = 0.0
- n_events, n_drugs, pgx_* etc. from schema defaults
- Models predict based on: age + one drug + typical patient profile

### Scenario 2: Complete Input (Age + Multiple Codes)
```json
{
  "age": 35,
  "drugs": ["AMOXICILLIN", "METHYLPHENIDATE"],
  "icds": ["R51", "G89"],
  "cpts": ["80305", "99213", "99284"]
}
```

**What happens:**
- All provided codes set to 1.0
- n_events, n_drugs, pgx_* use schema defaults when not provided
- Models predict based on: age + multiple codes + typical patient profile

### Scenario 3: Polypharmacy (Age 65+, Drugs Only)
```json
{
  "age": 72,
  "drugs": ["WARFARIN", "METFORMIN", "LISINOPRIL"],
  "icds": [],  // Not used for polypharmacy
  "cpts": []   // Not used for polypharmacy
}
```

**What happens:**
- Automatically uses `non_opioid_ed` cohort
- Age band: `65-74`
- Only drug features matter
- n_events, n_drugs, pgx_* use schema defaults when not provided

## Key Points

### ✅ What Works
1. **Partial inputs are fine**: You only need ICD/CPT/drug codes you care about
2. **Missing features handled**: Schema defaults fill in n_events, n_drugs, pgx_*, etc. (no trajectory/sequence/itemset in schema)
3. **Models are robust**: Trained on diverse patterns, handle partial inputs well
4. **Age determines model**: Automatically selects correct cohort/age_band

### ⚠️ Important Considerations

1. **Feature Matching**: 
   - Codes must match training data format (e.g., `F1120` not `F11.20`)
   - Drug names must match exactly (case-insensitive matching)
   - Codes not in training data are ignored (no error)

2. **Default Values**:
   - Non-item features (n_events, n_drugs, pgx_*, etc.) use **schema defaults** (e.g. training medians)
   - Feature engineering never produces trajectory/sequence/itemset; the schema has no such columns.
   - Predictions reflect: **your inputs + typical patient profile**

3. **Model Limitations**:
   - Models trained on historical data (2016-2018)
   - Predictions assume similar patterns hold
   - No patient-specific history (n_events/n_drugs etc. use defaults when not provided)

## Code Flow Diagram

```
User Input
  ↓
  age: 35
  drugs: ["AMOXICILLIN"]
  icds: ["R51", "G89"]
  cpts: ["80305"]
  ↓
build_feature_vector()
  ↓
  Initialize all features = 0.0
  Set age = 35.0
  Set item_AMOXICILLIN = 1.0
  Set item_R51 = 1.0, item_G89 = 1.0
  Set item_80305 = 1.0
  Fill non-item features (n_events, n_drugs, pgx_*, etc.) = schema defaults
  ↓
Complete Feature Vector (e.g., 500 features)
  ↓
predict_risk() → Ensemble
  ↓
  CatBoost.predict_proba() → 0.64
  XGBoost.predict_proba() → 0.66
  XGBoost RF.predict_proba() → 0.65
  ↓
  Weighted Average:
  (0.64 × 0.320) + (0.66 × 0.317) + (0.65 × 0.363) = 0.651
  ↓
Return: risk_score = 0.651 (65.1%)
```

## API Usage Example

### Request
```bash
curl -X POST https://your-api.execute-api.us-east-1.amazonaws.com/prod/risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "drugs": ["AMOXICILLIN"],
    "icds": ["R51", "G89"],
    "cpts": ["80305"]
  }'
```

### Response
```json
{
  "risk_score": 0.651,
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
  },
  "age_band_used": "25-44",
  "cohort_used": "opioid_ed"
}
```

## Summary

**Yes, you can use just ICD/CPT/drug_name inputs!**

The system:
1. ✅ Takes your inputs (age, ICD codes, CPT codes, drug names)
2. ✅ Builds complete feature vector with defaults for missing features
3. ✅ Runs all three models (CatBoost, XGBoost, XGBoost RF)
4. ✅ Combines predictions using performance-based weights
5. ✅ Returns ensemble risk score

**You don't need (and the schema doesn't include):**
- Trajectory/sequence/itemset features (feature engineering never generates these)
- Patient history beyond what you supply (n_events, n_drugs, pgx_* can use schema defaults)

**The API fills missing non-item features from the schema (e.g. training medians).**

