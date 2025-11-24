# FPGrowth Analysis - Target-Focused Rule Mining

**Last Updated:** November 24, 2025  
**Pipeline Version:** 4.0 (Target-Focused)

---

## Overview

This directory contains **Target-Focused FP-Growth analysis** tools for discovering patterns that **predict specific outcomes** in patient healthcare data. This is predictive analytics, not just descriptive statistics.

### What's New: Target-Focused Mining 🎯

Instead of finding ALL possible associations, we generate rules that **predict target outcomes**:

**Target 1: Opioid Dependence** (ICD codes F11.20-F11.29)  
**Target 2: ED Visits** (HCG Line codes: P51, O11, P33)

### Three Types of Rules Generated:

1. **`rules_TARGET_ICD.json`** - Patterns that predict opioid dependence
   - Example: `{Gabapentin, Tramadol, Hydrocodone} → {OPIOID_DEPENDENCE}` (72% confidence, 4.5x lift)

2. **`rules_TARGET_ED.json`** - Patterns that predict ED visits
   - Example: `{99213: Office Visit, J0670: Morphine} → {ED_VISIT}` (68% confidence, 3.8x lift)

3. **`rules_CONTROL.json`** - Patterns that DON'T predict targets (baseline/protective)
   - Example: `{Lisinopril, Metoprolol} → {Aspirin}` (standard cardiac care)

### Why This Matters:

- ✅ **Actionable Insights**: Know what patterns lead to bad outcomes
- ✅ **Comparative Analysis**: Target vs Control differences
- ✅ **Risk Prediction**: Use rules as features for CatBoost
- ✅ **Process Mining**: Pathways TO target outcomes (BupaR)
- ✅ **Clinical Utility**: Identify high-risk medication combinations

---

## Files

### 📊 Jupyter Notebooks (Interactive)

1. **`global_fpgrowth_feature_importance.ipynb`**
   - Analyzes patterns across ALL patients and cohorts
   - Processes: drug_name, icd_code, cpt_code
   - Output: `s3://pgxdatalake/gold/fpgrowth/global/{item_type}/`
   - Runtime: 2-3 hours (sequential processing)

2. **`cohort_fpgrowth_feature_importance.ipynb`**
   - Analyzes cohort-specific patterns (90 cohorts x 3 item types = 270 jobs)
   - Output: `s3://pgxdatalake/gold/fpgrowth/cohort/{item_type}/cohort_name={cohort}/age_band={age}/event_year={year}/`
   - Runtime: 3-5 hours (parallel processing)

### 🐍 Python Scripts (Production)

1. **`global_fpgrowth.py`**
   - Same functionality as global notebook
   - Better for production and memory management
   - Run: `python global_fpgrowth.py`

2. **`cohort_fpgrowth.py`**
   - Same functionality as cohort notebook
   - Parallel processing with configurable workers
   - Run: `python cohort_fpgrowth.py`

### 📖 Documentation

- **`README.md`** (this file) - Quick start guide
- **`README_local_notebook.md`** - Comprehensive local setup guide
- **`FpGrowth_README.md`** - FP-Growth algorithm details
- **`FPGrowth_Filtering_README.md`** - Filtering and optimization
- **`Parallelization_README.md`** - Parallel processing guide
- **`BupaR_README.md`** - BupaR process mining (R-based)
- **`CatBoost_README.md`** - CatBoost ML integration

---

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install mlxtend duckdb pandas boto3 jupyter

# Sync data from S3
aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ data/gold/cohorts_F1120/
```

### 2. Configuration (Quality-Focused Parameters)

Edit parameters in scripts/notebooks as needed:

```python
# Quality-focused thresholds (not quantity!)
MIN_SUPPORT = 0.01       # Global: 1% (57K patients), Cohort: 5%
MIN_CONFIDENCE = 0.4     # Global: 40%, Cohort: 50% (strong associations only)
MIN_CONFIDENCE_CPT = 0.5 # Even higher for procedures (50-60%)

# Target-focused mining (ENABLED by default)
TARGET_FOCUSED = True
TARGET_ICD_CODES = ['F11.20', 'F11.21', 'F11.22', 'F11.23', 'F11.24', 'F11.25', 'F11.29']
TARGET_HCG_LINES = ['P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits']

# Rule limits (top rules by lift)
MAX_RULES_PER_COHORT = 1000  # Cohort: 1000 rules max
MAX_RULES_PER_ITEM_TYPE = 5000  # Global: 5000 rules max

MAX_WORKERS = 2  # Parallel workers (reduced for memory stability)
```

### 3. Run Analysis

**Option A: Python Scripts (Recommended for Production)**

```bash
# Run global analysis
cd /path/to/pgx-analysis
python 3_fpgrowth_analysis/global_fpgrowth.py

# Run cohort analysis
python 3_fpgrowth_analysis/cohort_fpgrowth.py
```

**Option B: Jupyter Notebooks (Recommended for Exploration)**

```bash
# Start Jupyter
cd /path/to/pgx-analysis/3_fpgrowth_analysis
jupyter notebook

# Open and run:
# - global_fpgrowth_feature_importance.ipynb
# - cohort_fpgrowth_feature_importance.ipynb
```

**Option C: Command Line Notebook Execution**

```bash
# Execute global notebook
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=14400 \
  --output executed_global.ipynb \
  3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb

# Execute cohort notebook
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=18000 \
  --output executed_cohort.ipynb \
  3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb
```

---

## Input Data

### Source
- **Local Path**: `data/gold/cohorts_F1120/`
- **S3 Path**: `s3://pgxdatalake/gold/cohorts_F1120/`
- **Format**: Hive-partitioned Parquet files
- **Size**: ~7 GB, 947 million events

### Structure
```
cohorts_F1120/
├── cohort_name=opioid_ed/
│   ├── event_year=2016/
│   │   ├── age_band=0-12/
│   │   │   └── cohort.parquet
│   │   ├── age_band=13-24/
│   │   │   └── cohort.parquet
│   │   └── ...
│   └── ...
└── cohort_name=non_opioid_ed/
    └── ...
```

---

## Output Structure (Target-Focused)

### Global Analysis
```
s3://pgxdatalake/gold/fpgrowth/global/
├── drug_name/
│   ├── encoding_map.json          # Feature encodings for ML
│   ├── itemsets.json              # Frequent drug combinations
│   ├── rules_TARGET_ICD.json      # ← NEW: Rules predicting opioid dependence
│   ├── rules_TARGET_ED.json       # ← NEW: Rules predicting ED visits
│   ├── rules_CONTROL.json         # ← NEW: Non-target patterns (baseline)
│   └── metrics.json               # Processing statistics + target counts
├── icd_code/
│   └── (same files)
└── cpt_code/
    └── (same files)
```

### Cohort Analysis
```
s3://pgxdatalake/gold/fpgrowth/cohort/
├── drug_name/
│   └── cohort_name=opioid_ed/
│       └── age_band=25-44/
│           └── event_year=2017/
│               ├── itemsets.json
│               ├── rules_TARGET_ICD.json    # ← Opioid rules only
│               ├── rules_TARGET_ED.json     # ← ED rules only
│               ├── rules_CONTROL.json       # ← Control patterns
│               └── summary.json             # Includes rules_by_target counts
├── icd_code/
│   └── (same structure)
└── cpt_code/
    └── (same structure)
```

### File Contents

**`rules_TARGET_ICD.json`** - Rules predicting opioid dependence:
```json
[
  {
    "antecedents": ["Gabapentin", "Tramadol", "Hydrocodone"],
    "consequents": ["TARGET_ICD:OPIOID_DEPENDENCE"],
    "support": 0.08,
    "confidence": 0.72,
    "lift": 4.5
  }
]
```

**`rules_TARGET_ED.json`** - Rules predicting ED visits:
```json
[
  {
    "antecedents": ["99213", "J0670", "99285"],
    "consequents": ["TARGET_ED:EMERGENCY_DEPT"],
    "support": 0.12,
    "confidence": 0.68,
    "lift": 3.8
  }
]
```

**`rules_CONTROL.json`** - Non-target patterns (baseline care):
```json
[
  {
    "antecedents": ["Lisinopril", "Metoprolol"],
    "consequents": ["Aspirin"],
    "support": 0.15,
    "confidence": 0.68,
    "lift": 2.3
  }
]
```

---

## Item Types

### 1. drug_name (Pharmacy Events)
- **Source**: `drug_name` column from pharmacy events
- **Example items**: "AMOXICILLIN", "IBUPROFEN", "ATORVASTATIN"
- **Purpose**: Identify common drug combinations

### 2. icd_code (Medical Diagnosis Codes)
- **Source**: All ICD diagnosis columns (primary through five)
- **Example items**: "F11.20", "Z79.891", "E11.9"
- **Purpose**: Identify diagnosis patterns and comorbidities

### 3. cpt_code (Medical Procedure Codes)
- **Source**: `procedure_code` column from medical events
- **Example items**: "99213", "80053", "36415"
- **Purpose**: Identify procedure patterns and service bundles

---

## Performance

### Global Analysis
| Stage | Duration | Memory |
|-------|----------|--------|
| Data Loading | 30-60s | 2 GB |
| Transaction Creation | 1-2 min | 5 GB |
| FP-Growth (per item type) | 30-60 min | 15 GB |
| **Total (all 3 types)** | **2-3 hours** | **15-20 GB peak** |

### Cohort Analysis
| Configuration | Duration | Memory |
|---------------|----------|--------|
| 5 workers (recommended) | 3-5 hours | 20-30 GB |
| 10 workers (high-end) | 2-3 hours | 40-50 GB |

---

## Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. **Increase MIN_SUPPORT** (reduces itemset size):
   ```python
   MIN_SUPPORT = 0.02  # From 0.01
   ```

2. **Reduce MAX_WORKERS** (for cohort analysis):
   ```python
   MAX_WORKERS = 3  # From 5
   ```

3. **Process one item type at a time**:
   ```python
   ITEM_TYPES = ['drug_name']  # Instead of all 3
   ```

4. **Use Python scripts instead of notebooks** (better memory cleanup)

---

## Troubleshooting

### Issue: "No module named 'mlxtend'"
```bash
pip install mlxtend
```

### Issue: "Local data path does not exist"
```bash
aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ data/gold/cohorts_F1120/
```

### Issue: "0 records found" or "No data"
- Check event_type case sensitivity (should be lowercase: 'pharmacy', 'medical')
- Verify data sync completed successfully
- Check DuckDB query syntax

### Issue: Process killed (OOM)
- Follow memory optimization steps above
- Close other applications
- Consider running on EC2 with more RAM

---

## Using Target-Focused Results

### 1. Load Target-Specific Rules

```python
import pandas as pd

# Load opioid dependence predictors
opioid_rules = pd.read_json('s3://pgxdatalake/gold/fpgrowth/global/drug_name/rules_TARGET_ICD.json')
print(f"Opioid rules: {len(opioid_rules)}")

# Load ED visit predictors
ed_rules = pd.read_json('s3://pgxdatalake/gold/fpgrowth/global/drug_name/rules_TARGET_ED.json')
print(f"ED rules: {len(ed_rules)}")

# Load control patterns (baseline)
control_rules = pd.read_json('s3://pgxdatalake/gold/fpgrowth/global/drug_name/rules_CONTROL.json')
print(f"Control rules: {len(control_rules)}")

# Find highest-risk patterns
top_risk = opioid_rules.nlargest(10, 'lift')
print(top_risk[['antecedents', 'consequents', 'lift', 'confidence']])
```

### 2. Comparative Analysis (Target vs Control)

```python
# Get all drugs in target rules
target_drugs = set()
for _, rule in opioid_rules.iterrows():
    target_drugs.update(rule['antecedents'])

# Get all drugs in control rules
control_drugs = set()
for _, rule in control_rules.iterrows():
    control_drugs.update(rule['antecedents'])

# Risk factors: in target but not control
risk_factors = target_drugs - control_drugs
print(f"High-risk medications: {risk_factors}")

# Protective factors: in control but not target
protective = control_drugs - target_drugs
print(f"Potentially protective: {protective}")
```

### 3. Feature Engineering for CatBoost

```python
def patient_matches_rule(patient_meds, rule_antecedents):
    """Check if patient has all antecedents in rule."""
    return all(med in patient_meds for med in rule_antecedents)

# Create risk features from opioid rules
for idx, rule in opioid_rules.head(20).iterrows():
    feature_name = f"opioid_risk_rule_{idx}"
    df[feature_name] = df['medications'].apply(
        lambda meds: patient_matches_rule(meds, rule['antecedents'])
    )

# Create protective features from control rules
for idx, rule in control_rules.head(20).iterrows():
    feature_name = f"control_pattern_{idx}"
    df[feature_name] = df['medications'].apply(
        lambda meds: patient_matches_rule(meds, rule['antecedents'])
    )
```

### 4. BupaR Process Mining (R Example)

```r
# Load target-focused rules for process maps
library(jsonlite)
library(bupaR)

# Opioid pathway analysis
opioid_rules <- fromJSON("s3://pgxdatalake/gold/fpgrowth/cohort/drug_name/.../rules_TARGET_ICD.json")

# Create process map showing pathways TO opioid dependence
# (Not random co-occurrences!)
process_map <- create_process_map(opioid_rules)

# Compare with control pathways
control_rules <- fromJSON("s3://pgxdatalake/gold/fpgrowth/cohort/drug_name/.../rules_CONTROL.json")
control_map <- create_process_map(control_rules)

# Identify divergence points
compare_processes(process_map, control_map)
```

---

## Key Implementation Details

### Target Detection

The notebooks automatically add target markers to patient transactions:

```python
# For each patient, add special items:
if patient_has_opioid_icd_code:
    transaction.append('TARGET_ICD:OPIOID_DEPENDENCE')

if patient_has_ed_visit:  # HCG Line in [P51, O11, P33]
    transaction.append('TARGET_ED:EMERGENCY_DEPT')
```

Example patient transaction:
```python
# Before
['Gabapentin', 'Tramadol', 'Hydrocodone', 'Lisinopril']

# After (if they developed opioid dependence)
['Gabapentin', 'Tramadol', 'Hydrocodone', 'Lisinopril', 'TARGET_ICD:OPIOID_DEPENDENCE']
```

### Rule Filtering

After generating ALL rules, we split them:

```python
# Target rules: consequent contains TARGET_ICD: or TARGET_ED:
target_mask = rules['consequents'].apply(
    lambda x: any(item.startswith(('TARGET_ICD:', 'TARGET_ED:')) for item in x)
)

rules_target = rules[target_mask]      # Predictive rules
rules_control = rules[~target_mask]    # Baseline rules
```

### Why This Works

**Traditional FPGrowth**: `{Aspirin} → {Ibuprofen}` (so what?)  
**Target-Focused**: `{Gabapentin, Tramadol} → {OPIOID_DEPENDENCE}` (actionable!)

- **Drastically fewer rules**: 1,000 target rules vs 100,000+ all rules
- **Clinically meaningful**: Every rule predicts an outcome
- **Comparative**: Can compare target vs control patterns

---

## Next Steps

1. **Verify Outputs**: Check S3 for `rules_TARGET_ICD.json`, `rules_TARGET_ED.json`, `rules_CONTROL.json`
2. **Download Results**: `aws s3 sync s3://pgxdatalake/gold/fpgrowth/ data/gold/fpgrowth/`
3. **Compare Targets**: Analyze differences between opioid and ED predictors
4. **Risk Stratification**: Use rules to identify high-risk patient patterns
5. **Process Mining**: Map pathways TO target outcomes (not random associations)
6. **Feature Engineering**: Create rule-based features for CatBoost

---

## Support

For detailed configuration and troubleshooting, see:
- `README_local_notebook.md` - Comprehensive setup guide
- `FpGrowth_README.md` - Algorithm details
- Main project documentation in `docs/`

---

**Questions or Issues?** Review the troubleshooting section above and check logs for specific error messages.


