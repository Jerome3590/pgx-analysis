# Target-Focused FPGrowth: Predictive Rule Mining

**Last Updated:** November 24, 2025  
**Version:** 4.0 (Target-Focused)

---

## 🎯 Overview

This module implements **Target-Focused FPGrowth** - a predictive analytics approach that discovers patterns leading to specific outcomes, not just any co-occurrences.

### The Paradigm Shift

**❌ Old Approach (Descriptive)**:  
Generate ALL possible association rules  
→ Results: 100,000+ rules like `{Aspirin} → {Ibuprofen}` (so what?)

**✅ New Approach (Predictive)**:  
Generate ONLY rules that predict target outcomes  
→ Results: 50-1,000 rules like `{Gabapentin, Tramadol} → {OPIOID_DEPENDENCE}` (actionable!)

---

## 🎯 Target Outcomes

### Target 1: Opioid Dependence (ICD Codes)

**Codes**: F11.20, F11.21, F11.22, F11.23, F11.24, F11.25, F11.29  
**Marker**: `TARGET_ICD:OPIOID_DEPENDENCE`

**Example Rules**:
```json
{
  "antecedents": ["Gabapentin", "Tramadol", "Hydrocodone"],
  "consequents": ["TARGET_ICD:OPIOID_DEPENDENCE"],
  "support": 0.08,
  "confidence": 0.72,
  "lift": 4.5
}
```
**Interpretation**: "Patients on this medication combo have 72% chance of developing opioid dependence (4.5x higher than baseline)"

### Target 2: Emergency Department Visits (HCG Line)

**HCG Lines**:  
- `P51 - ER Visits and Observation Care`
- `O11 - Emergency Room`
- `P33 - Urgent Care Visits`

**Marker**: `TARGET_ED:EMERGENCY_DEPT`

**Example Rules**:
```json
{
  "antecedents": ["99213", "J0670", "99285"],
  "consequents": ["TARGET_ED:EMERGENCY_DEPT"],
  "support": 0.12,
  "confidence": 0.68,
  "lift": 3.8
}
```
**Interpretation**: "This care pattern leads to 68% ED return rate (3.8x baseline)"

---

## 📁 Output Files (3 Types)

### 1. `rules_TARGET_ICD.json` - Opioid Dependence Predictors

Contains rules where **consequent = TARGET_ICD:OPIOID_DEPENDENCE**

**Use Cases**:
- Identify high-risk medication combinations
- Early warning system for providers
- Feature engineering for opioid risk prediction models
- Process mining: pathways TO dependence

### 2. `rules_TARGET_ED.json` - ED Visit Predictors

Contains rules where **consequent = TARGET_ED:EMERGENCY_DEPT**

**Use Cases**:
- Identify care patterns leading to ED visits
- Improve discharge planning protocols
- Predict ED return risk
- Process mining: pathways TO ED

### 3. `rules_CONTROL.json` - Baseline Patterns

Contains rules where **consequent is NOT a target**

**Use Cases**:
- Identify normal/safe healthcare utilization patterns
- Protective factors (in control but not target)
- Comparative analysis (target vs control)
- Baseline for risk stratification

---

## 🔬 How It Works

### Step 1: Add Target Markers

For each patient, append special items to their transaction:

```python
# Original transaction
patient_123 = ['Gabapentin', 'Tramadol', 'Hydrocodone', 'Lisinopril']

# Add targets if applicable
if patient_has_opioid_code(patient_123):
    patient_123.append('TARGET_ICD:OPIOID_DEPENDENCE')

if patient_has_ed_visit(patient_123):
    patient_123.append('TARGET_ED:EMERGENCY_DEPT')

# Final transaction
# ['Gabapentin', 'Tramadol', 'Hydrocodone', 'Lisinopril', 'TARGET_ICD:OPIOID_DEPENDENCE']
```

### Step 2: Generate ALL Rules (as before)

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules

# FP-Growth finds frequent itemsets
itemsets = fpgrowth(transactions, min_support=0.05)

# Generate ALL rules
all_rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
```

### Step 3: Split by Target Type

```python
# Target rules: consequent contains target marker
target_mask = all_rules['consequents'].apply(
    lambda x: any(item.startswith(('TARGET_ICD:', 'TARGET_ED:')) for item in x)
)

rules_target = all_rules[target_mask]      # Predictive rules
rules_control = all_rules[~target_mask]    # Baseline rules

# Further split target rules by outcome type
rules_icd = rules_target[rules_target['consequents'].apply(
    lambda x: any('TARGET_ICD:' in str(item) for item in x)
)]

rules_ed = rules_target[rules_target['consequents'].apply(
    lambda x: any('TARGET_ED:' in str(item) for item in x)
)]
```

### Step 4: Save Separately

```python
# Save each type to separate file
save_json(rules_icd, 's3://.../rules_TARGET_ICD.json')
save_json(rules_ed, 's3://.../rules_TARGET_ED.json')
save_json(rules_control, 's3://.../rules_CONTROL.json')
```

---

## 💡 Usage Examples

### 1. Identify High-Risk Patterns

```python
import pandas as pd

# Load opioid risk rules
opioid_rules = pd.read_json('s3://.../rules_TARGET_ICD.json')

# Find highest-risk combinations
top_risk = opioid_rules.nlargest(20, 'lift')

print("Top 20 Opioid Risk Patterns:")
for _, rule in top_risk.iterrows():
    print(f"  {rule['antecedents']} → {rule['consequents']}")
    print(f"    Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.1f}")
```

### 2. Comparative Analysis (Target vs Control)

```python
# Load both
target_rules = pd.read_json('s3://.../rules_TARGET_ICD.json')
control_rules = pd.read_json('s3://.../rules_CONTROL.json')

# Get all items from each
def extract_items(rules_df):
    items = set()
    for _, rule in rules_df.iterrows():
        items.update(rule['antecedents'])
        items.update(rule['consequents'])
    return items

target_items = extract_items(target_rules)
control_items = extract_items(control_rules)

# Risk factors: in target but NOT in control
risk_factors = target_items - control_items
print(f"High-risk medications: {risk_factors}")

# Protective factors: in control but NOT in target
protective = control_items - target_items
print(f"Potentially protective: {protective}")
```

### 3. Feature Engineering for CatBoost

```python
def patient_matches_rule(patient_meds, antecedents):
    """Check if patient has all items in antecedents."""
    return all(med in patient_meds for med in antecedents)

# Load opioid risk rules
opioid_rules = pd.read_json('s3://.../rules_TARGET_ICD.json')

# Create binary features for top 50 rules
for idx, rule in opioid_rules.head(50).iterrows():
    feature_name = f"opioid_risk_rule_{idx}"
    
    # For each patient, check if they match this risky pattern
    df[feature_name] = df['medications'].apply(
        lambda meds: patient_matches_rule(meds, rule['antecedents'])
    )

# Now use these features in CatBoost
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.fit(df[feature_cols], df['target'])
```

### 4. BupaR Process Mining (R)

```r
library(jsonlite)
library(bupaR)
library(processmapR)

# Load target rules
opioid_rules <- fromJSON("s3://.../rules_TARGET_ICD.json")

# Create process map showing pathways TO opioid dependence
# (Not random co-occurrences, but actual pathways to bad outcome!)
process_map <- create_process_map(opioid_rules, 
                                  type = "frequency",
                                  rankdir = "LR")

# Visualize high-risk pathways
plot(process_map)

# Compare with control pathways
control_rules <- fromJSON("s3://.../rules_CONTROL.json")
control_map <- create_process_map(control_rules)

# Where do pathways diverge?
compare_process_maps(process_map, control_map)
```

---

## 🎯 Quality-Focused Parameters

### Why High Thresholds?

We use **higher confidence thresholds** than traditional FPGrowth:

**Traditional**: `min_confidence = 0.01` (1%) → 100,000+ weak rules  
**Target-Focused**: `min_confidence = 0.5` (50%) → 50-1,000 strong rules

### Configuration

```python
# Global analysis (5.7M patients)
MIN_SUPPORT = 0.01       # 1% = 57,000 occurrences
MIN_CONFIDENCE = 0.4     # 40% confidence
MIN_CONFIDENCE_CPT = 0.5 # 50% for procedures
MAX_RULES_PER_ITEM_TYPE = 5000  # Top 5000 by lift

# Cohort analysis (per cohort)
MIN_SUPPORT = 0.05       # 5% within cohort
MIN_CONFIDENCE = 0.5     # 50% confidence
MIN_CONFIDENCE_CPT = 0.6 # 60% for procedures
MAX_RULES_PER_COHORT = 1000  # Top 1000 by lift
```

**Rationale**:
- ✅ **Meaningful patterns only**: 50%+ confidence = strong associations
- ✅ **Clinically actionable**: Can actually intervene on high-confidence patterns
- ✅ **Interpretable**: Can review 1,000 rules, not 100,000
- ✅ **Top by lift**: Most important patterns ranked first

---

## 📊 Expected Results

### Rule Counts

| Cohort Type | TARGET_ICD | TARGET_ED | CONTROL | Total |
|-------------|------------|-----------|---------|-------|
| **Global** (5.7M patients) | ~1,000-2,000 | ~800-1,500 | ~2,000-3,000 | ~5,000 |
| **Cohort** (per age/year) | ~50-400 | ~50-300 | ~200-500 | ~1,000 |

### File Sizes

| File | Typical Size | Load Time |
|------|--------------|-----------|
| `rules_TARGET_ICD.json` | 50-500 KB | <1 sec |
| `rules_TARGET_ED.json` | 40-400 KB | <1 sec |
| `rules_CONTROL.json` | 100-800 KB | <1 sec |

**Compare to old approach**: 500 MB+ files that couldn't load!

---

## 🔍 Quality Metrics

Each `summary.json` includes:

```json
{
  "total_rules": 850,
  "rules_by_target": {
    "TARGET_ICD": 320,
    "TARGET_ED": 280,
    "CONTROL": 250
  },
  "min_support": 0.05,
  "min_confidence": 0.5,
  "max_rules_limit": 1000,
  "rules_truncated": false,
  "target_focused": true,
  "target_icd_codes": ["F11.20", "F11.21", ...],
  "target_hcg_lines": ["P51 - ER Visits and Observation Care", ...]
}
```

---

## 🎉 Why This Matters

### For Clinicians
- **Risk Stratification**: Identify high-risk patients early
- **Intervention Points**: Know which patterns to disrupt
- **Clinical Decision Support**: Alert on risky combinations

### For Researchers
- **Comparative Analysis**: Target vs control differences
- **Protective Factors**: What patterns DON'T lead to bad outcomes
- **Process Mining**: Map pathways TO outcomes (not random associations)

### For ML Engineers
- **Better Features**: Predictive patterns as features
- **Reduced Dimensionality**: 1,000 rules vs 100,000+
- **Interpretable**: Can explain why model predicts risk

### For Healthcare Systems
- **Quality Improvement**: Identify problematic care patterns
- **Cost Reduction**: Prevent ED visits and complications
- **Population Health**: Risk stratify entire populations

---

## 📚 References

### Implementation Details
- `global_fpgrowth_feature_importance.ipynb` - Global analysis notebook
- `cohort_fpgrowth_feature_importance.ipynb` - Cohort analysis notebook
- `global_fpgrowth.py` - Global analysis script
- `cohort_fpgrowth.py` - Cohort analysis script

### Related Documentation
- `README.md` - Quick start guide
- `README_fprgrowth.md` - FPGrowth algorithm details
- `README_bupaR.md` - Process mining with BupaR
- `README_catboost.md` - ML integration

---

**Questions?** Review the main `README.md` or check the notebooks for detailed implementation.
