# Quick Start: Making Predictions

## Simple Answer

**Yes!** You can use the trained models with **just**:
- Age
- ICD codes (optional)
- CPT codes (optional)  
- Drug names (optional)

Everything else is handled automatically.

## Minimal Example

```python
# What you provide
inputs = {
    "age": 35,
    "drugs": ["AMOXICILLIN"],
    "icds": ["R51", "G89"],  # Note: F1120 is excluded (it's the target, not an input)
    "cpts": ["80305"]
}

# What the system does automatically:
# 1. Determines cohort: opioid_ed (age 13-64)
# 2. Determines age_band: 25-44
# 3. Builds complete feature vector (500+ features)
#    - Your inputs → set to 1.0
#    - Missing features → use training data medians
# 4. Runs all 3 models
# 5. Combines with performance-based weights
# 6. Returns risk score

# Result
{
  "risk_score": 0.65,  # 65% risk
  "risk_band": "high"
}
```

## What Gets Set Automatically

### Your Inputs → Set to 1.0
```python
item_AMOXICILLIN = 1.0      # From your drugs list
item_R51 = 1.0              # From your icds list
item_G89 = 1.0               # From your icds list
item_80305 = 1.0            # From your cpts list
age = 35.0                  # From your age input
```

### Missing Features → Use Training Medians
```python
trajectory_length = 15.0     # Median from training data
pre_n_events = 8.0          # Median from training data
itemset_14_match = 0.0       # Median from training data
# ... all other features get median/default values
```

## Why This Works

1. **Models are trained on complete data**: They've seen patterns with all features
2. **Missing features use realistic defaults**: Medians represent "typical" patients
3. **Your inputs are the key signals**: ICD/CPT/drug codes drive the prediction
4. **Ensemble is robust**: Three models provide consensus

## Real-World Use Cases

### Use Case 1: Check Risk for Specific Codes
```json
{
  "age": 45,
  "icds": ["R51", "G89"],  // Note: F1120 is excluded (it's the target, not an input)
  "drugs": [],
  "cpts": []
}
```
**Result**: Risk score for F1120 opioid ED visit based on provided inputs + typical patient profile

### Use Case 2: Compare Drug Combinations
```json
// Scenario A
{"age": 35, "drugs": ["AMOXICILLIN"]}

// Scenario B  
{"age": 35, "drugs": ["AMOXICILLIN", "METHYLPHENIDATE"]}
```
**Result**: Compare risk scores to see impact of adding second drug

### Use Case 3: Polypharmacy Check (Age 65-114)
```json
{
  "age": 72,
  "drugs": ["WARFARIN", "METFORMIN", "LISINOPRIL", "ATORVASTATIN"]
}
```
**Result**: Automatically uses polypharmacy model, returns polypharmacy risk

**Note**: 
- Age band 0-12 is excluded due to small cohort size
- Ages 95-114 are mapped to age band 85-94 (uses 85-94 model)

## API Call Example

```bash
curl -X POST https://api.example.com/risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "drugs": ["AMOXICILLIN"],
    "icds": ["R51", "G89"],  // Note: F1120 is excluded (it's the target, not an input)
    "cpts": ["80305"]
  }'
```

**Response:**
```json
{
  "risk_score": 0.651,
  "risk_band": "high",
  "model_breakdown": {
    "catboost": 0.64,
    "xgboost": 0.66,
    "xgboost_rf": 0.65
  }
}
```

## Important Notes

### ✅ What You Need
- **Age** (required): 13-114 (determines which model to use)
  - Ages 13-64: Opioid ED Risk
  - Ages 65-114: Polypharmacy Risk (ages 95-114 use 85-94 model)
- **At least one code** (recommended): ICD, CPT, or drug name

**Note**: 
- Age band 0-12 is excluded due to small cohort size
- Ages 95-114 are treated as age band 85-94 (shown on dashboard)

### ⚠️ Code Format
- **Must match training data**: `F1120` not `F11.20`
- **Case-insensitive**: `amoxicillin` = `AMOXICILLIN`
- **Unknown codes ignored**: No error, just not used

### 📊 What Defaults Mean
- **Not zero-risk**: Defaults represent typical patients
- **Your inputs matter most**: ICD/CPT/drug codes drive prediction
- **Defaults fill gaps**: Trajectory features you don't have

## Summary

**You provide**: Age + codes you care about  
**System provides**: Complete feature vector + predictions  
**You get**: Risk score from ensemble of 3 models

**It's that simple!** 🎯

