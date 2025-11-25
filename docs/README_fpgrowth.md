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

## The Paradigm Shift

**❌ Old Approach (Descriptive)**:  
Generate ALL possible association rules  
→ Results: 100,000+ rules like `{Aspirin} → {Ibuprofen}` (so what?)

**✅ New Approach (Predictive)**:  
Generate ONLY rules that predict target outcomes  
→ Results: 50-1,000 rules like `{Gabapentin, Tramadol} → {OPIOID_DEPENDENCE}` (actionable!)

---

## Target Outcomes

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

- **`docs/README_fpgrowth.md`** (this file) - Comprehensive guide
- **`README_local_notebook.md`** - Comprehensive local setup guide
- **`README_fprgrowth.md`** - FP-Growth algorithm details
- **`README_parallelization.md`** - Parallel processing guide
- **`docs/README_bupaR.md`** - BupaR process mining (R-based)
- **`README_catboost.md`** - CatBoost ML integration

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

**Rationale for High Thresholds**:
- ✅ **Meaningful patterns only**: 50%+ confidence = strong associations
- ✅ **Clinically actionable**: Can actually intervene on high-confidence patterns
- ✅ **Interpretable**: Can review 1,000 rules, not 100,000
- ✅ **Top by lift**: Most important patterns ranked first

### 3. Run Analysis

**Option A: Python Scripts (Recommended for Production)**

```bash
# Run global analysis
cd /path/to/pgx-analysis
python 4_fpgrowth_analysis/global_fpgrowth.py

# Run cohort analysis
python 4_fpgrowth_analysis/cohort_fpgrowth.py
```

**Option B: Jupyter Notebooks (Recommended for Exploration)**

```bash
# Start Jupyter
cd /path/to/pgx-analysis/4_fpgrowth_analysis
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
  4_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb

# Execute cohort notebook
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=18000 \
  --output executed_cohort.ipynb \
  4_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb
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

### Expected Results

#### Rule Counts

| Cohort Type | TARGET_ICD | TARGET_ED | CONTROL | Total |
|-------------|------------|-----------|---------|-------|
| **Global** (5.7M patients) | ~1,000-2,000 | ~800-1,500 | ~2,000-3,000 | ~5,000 |
| **Cohort** (per age/year) | ~50-400 | ~50-300 | ~200-500 | ~1,000 |

#### File Sizes

| File | Typical Size | Load Time |
|------|--------------|-----------|
| `rules_TARGET_ICD.json` | 50-500 KB | <1 sec |
| `rules_TARGET_ED.json` | 40-400 KB | <1 sec |
| `rules_CONTROL.json` | 100-800 KB | <1 sec |

**Compare to old approach**: 500 MB+ files that couldn't load!

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

## Technical Implementation

### Core Functions

#### Global Pipeline Functions:
- `run_fpgrowth_global(logger)` - Main global pipeline
- `extract_global_drug_names(logger)` - Extract all unique drugs
- `create_global_drug_transactions(logger)` - Create patient-level transactions
- `create_global_encoding_map(logger, itemsets, rules)` - Generate encoding map
- `save_global_fpgrowth_results(logger, itemsets, rules, encoding_map)` - Save to S3

#### By-Cohort Functions:
- `feature_engineer(cohort_name, age_band, event_year, paths, logger)` - Process single cohort
- `process_cohort_feature_engineer(...)` - Orchestrate cohort processing

#### Utility Functions:
- `create_fpgrowth_logger(name)` - Logger with same pattern as create_cohort.py
- `load_global_encoding_map(logger)` - Load saved encoding map
- `get_drug_metrics_from_rules(drug, rules, logger)` - Extract drug-specific metrics

### Dependencies

#### Required Libraries:
```python
# Core ML libraries
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Project utilities
from helpers_1997_13.common_imports import s3_client, get_logger
from helpers_1997_13.duckdb_utils import get_duckdb_connection, execute_duckdb_query
from helpers_1997_13.s3_utils import save_to_s3_json, save_to_s3_parquet
```

#### Installation:
```bash
pip install mlxtend pandas numpy duckdb boto3
```

### Pipeline Workflows

#### Global FP-Growth Workflow:
1. **Extract** → Get all unique drug names from pharmacy dataset
2. **Transform** → Create patient-level drug transactions
3. **Encode** → Apply TransactionEncoder for FP-Growth input format
4. **Mine** → Run FP-Growth algorithm (min_support=0.01)
5. **Rules** → Generate association rules (min_confidence=0.4)
6. **Split** → Separate rules by target type (TARGET_ICD, TARGET_ED, CONTROL)
7. **Encode** → Create drug encoding map with FP-Growth metrics
8. **Save** → Store results to S3 for downstream ML models

#### By-Cohort Workflow:
1. **Discover** → List available cohort paths from S3
2. **Check** → Verify which cohorts need processing
3. **Process** → Run FP-Growth for each cohort individually
4. **Extract** → Generate cohort-specific drug tokens
5. **Mine** → Apply FP-Growth with fallback support levels
6. **Build Features** → For each cohort, build all drug metrics/features from scratch using only that cohort's data
7. **Split** → Separate rules by target type
8. **Visualize** → Create network graphs for drug associations
9. **Store** → Save cohort-specific results partitioned by cohort/age/year

### Key Differences: Global vs By-Cohort

| Aspect | Global FP-Growth | By-Cohort FP-Growth |
|--------|------------------|-------------------|
| **Data Scope** | Entire pharmacy dataset (5.7M patients) | Individual cohorts (~10K-100K patients) |
| **Purpose** | ML feature engineering | Process mining analysis |
| **Support Threshold** | 0.01 (lower for coverage) | 0.05 (higher for significance) |
| **Confidence Threshold** | 0.4 (40%) | 0.5 (50%) |
| **Output Format** | Universal encoding map | Cohort-specific patterns |
| **Use Case** | CatBoost consistent features | BupaR pathway analysis |
| **Granularity** | Population-level patterns | Cohort-specific patterns |
| **Processing** | Single large job | Multiple parallel jobs |
| **Rule Limit** | 5,000 per item type | 1,000 per cohort |

### Cohort-Specific Feature Generation

**⚠️ Important**: All cohort-specific features, metrics, and encodings are built from scratch for each cohort.

- No global drug names or metrics are reused for cohort-specific outputs
- Each cohort's drug features, metrics, and encodings are generated using only the drugs present in that cohort's data
- This ensures true independence and correctness for downstream process mining and ML

---

## How It Works

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

### Why This Works

**Traditional FPGrowth**: `{Aspirin} → {Ibuprofen}` (so what?)  
**Target-Focused**: `{Gabapentin, Tramadol} → {OPIOID_DEPENDENCE}` (actionable!)

- **Drastically fewer rules**: 1,000 target rules vs 100,000+ all rules
- **Clinically meaningful**: Every rule predicts an outcome
- **Comparative**: Can compare target vs control patterns

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

## Usage Examples

### 1. Identify High-Risk Patterns

```python
import pandas as pd

# Load opioid risk rules
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

## Integration with Downstream Analysis

### CatBoost Feature Engineering Integration

#### Loading Global Encoding Map

```python
# Load the global drug encoding map for consistent features across all cohorts
from helpers_1997_13.s3_utils import load_from_s3_json

def load_global_encoding_map(cohort_name, age_band, event_year):
    """Load global encoding map for CatBoost feature engineering"""
    encoding_path = f"s3://pgxdatalake/gold/fpgrowth/global/drug_name/encoding_map.json"
    encoding_map = load_from_s3_json(encoding_path)
    return encoding_map

# Load encoding map
encoding_map = load_global_encoding_map("opioid_ed", "65-74", 2020)
print(f"Loaded encoding map with {len(encoding_map)} drugs")
```

#### Transform Patient Drug Lists

```python
def encode_patient_drugs(drug_list, encoding_map):
    """Transform patient drug lists into numerical features for CatBoost"""
    return [encoding_map.get(drug, "X000000000000000") for drug in drug_list]

# Apply to patient data
df['drug_encodings'] = df['drug_list'].apply(
    lambda drugs: encode_patient_drugs(drugs, encoding_map)
)

# Create feature columns for CatBoost
df['encoded_drug_features'] = df['drug_encodings'].apply(
    lambda encodings: ','.join(encodings) if encodings else ''
)
```

#### CatBoost Integration Example

```python
from catboost import CatBoostClassifier, Pool

# Use encoded drug features as categorical features
categorical_features = ['encoded_drug_features', 'age_band', 'cohort_type']

# Create CatBoost training pool
train_pool = Pool(
    X_train, 
    y_train, 
    cat_features=categorical_features
)

# Train model with drug pattern features
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    verbose=100
)

model.fit(train_pool)
```

### BupaR Process Mining Integration

#### Loading Cohort-Specific Results

```python
import json
from helpers_1997_13.s3_utils import load_from_s3_json

def load_cohort_fpgrowth_results(cohort_name, age_band, event_year, item_type='drug_name'):
    """Load cohort-specific FP-Growth results for process mining"""
    
    base_path = (
        f"s3://pgxdatalake/gold/fpgrowth/cohort/{item_type}/"
        f"cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}/"
    )
    
    # Load itemsets
    itemsets = load_from_s3_json(f"{base_path}itemsets.json")
    
    # Load association rules (target-focused)
    rules_target_icd = load_from_s3_json(f"{base_path}rules_TARGET_ICD.json")
    rules_target_ed = load_from_s3_json(f"{base_path}rules_TARGET_ED.json")
    rules_control = load_from_s3_json(f"{base_path}rules_CONTROL.json")
    
    return itemsets, {
        'target_icd': rules_target_icd,
        'target_ed': rules_target_ed,
        'control': rules_control
    }

# Example usage
itemsets, rules = load_cohort_fpgrowth_results("ed_non_opioid", "65-74", "2020")
print(f"Loaded {len(itemsets)} itemsets and {len(rules['target_icd'])} target ICD rules")
```

#### Process Flow Analysis

```python
def analyze_drug_pathways(rules_data):
    """Analyze drug prescription pathways for process mining"""
    
    pathways = []
    for rule in rules_data:
        if rule.get('confidence', 0) > 0.7:  # High confidence rules
            antecedents = rule.get('antecedents', [])
            consequents = rule.get('consequents', [])
            
            pathway = {
                'from_drugs': antecedents,
                'to_drugs': consequents,
                'confidence': rule.get('confidence', 0),
                'support': rule.get('support', 0),
                'lift': rule.get('lift', 0)
            }
            pathways.append(pathway)
    
    return pathways

# Analyze pathways for process mining
pathways = analyze_drug_pathways(rules['target_icd'])
print(f"Found {len(pathways)} high-confidence drug pathways")
```

#### BupaR Event Log Creation

```python
def create_bupar_event_log(itemsets, rules, cohort_data):
    """Create event log for BupaR process mining from FP-Growth results"""
    
    event_log = []
    
    for patient_id, patient_data in cohort_data.groupby('mi_person_key'):
        drug_sequence = patient_data.sort_values('event_date')['drug_name'].tolist()
        
        # Use FP-Growth patterns to identify process stages
        for i, drug in enumerate(drug_sequence):
            # Find relevant patterns
            relevant_patterns = [
                rule for rule in rules['target_icd'] 
                if drug in rule.get('antecedents', []) or drug in rule.get('consequents', [])
            ]
            
            event = {
                'case_id': patient_id,
                'activity': drug,
                'timestamp': patient_data.iloc[i]['event_date'],
                'pattern_support': max([p.get('support', 0) for p in relevant_patterns] or [0]),
                'sequence_position': i + 1
            }
            event_log.append(event)
    
    return pd.DataFrame(event_log)

# Create event log for BupaR
event_log = create_bupar_event_log(itemsets, rules, cohort_df)
```

### Integration Validation

#### Validate Results Availability

```python
def validate_fpgrowth_integration():
    """Validate that FP-Growth results are ready for downstream use"""
    
    validation_results = {
        'global_ready': False,
        'cohort_ready': False,
        'errors': []
    }
    
    try:
        # Check global encoding map
        encoding_map = load_global_encoding_map("opioid_ed", "65-74", 2020)
        validation_results['global_ready'] = len(encoding_map) > 0
        print(f"✓ Global encoding map: {len(encoding_map)} drugs")
        
    except Exception as e:
        validation_results['errors'].append(f"Global encoding error: {e}")
        print(f"✗ Global encoding map error: {e}")
    
    try:
        # Check cohort results
        itemsets, rules = load_cohort_fpgrowth_results("ed_non_opioid", "65-74", "2020")
        validation_results['cohort_ready'] = len(itemsets) > 0
        print(f"✓ Cohort results: {len(itemsets)} itemsets available")
        
    except Exception as e:
        validation_results['errors'].append(f"Cohort results error: {e}")
        print(f"✗ Cohort results error: {e}")
    
    return validation_results

# Run validation
validation = validate_fpgrowth_integration()
print(f"Integration ready: {validation['global_ready'] and validation['cohort_ready']}")
```

### Pipeline Integration Checklist

#### Before CatBoost Analysis
- [ ] Global FP-Growth completed successfully
- [ ] Global encoding map accessible
- [ ] Drug features transformed using encoding map
- [ ] Categorical features properly configured in CatBoost

#### Before BupaR Analysis
- [ ] By-cohort FP-Growth completed for target cohorts
- [ ] Cohort-specific itemsets and rules accessible
- [ ] Event logs created with FP-Growth pattern enrichment
- [ ] Process flow analysis configured

---

## Quality Metrics

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

## Why This Matters

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

## Next Steps

1. **Verify Outputs**: Check S3 for `rules_TARGET_ICD.json`, `rules_TARGET_ED.json`, `rules_CONTROL.json`
2. **Download Results**: `aws s3 sync s3://pgxdatalake/gold/fpgrowth/ data/gold/fpgrowth/`
3. **Compare Targets**: Analyze differences between opioid and ED predictors
4. **Risk Stratification**: Use rules to identify high-risk patient patterns
5. **Process Mining**: Map pathways TO target outcomes (not random associations)
6. **Feature Engineering**: Create rule-based features for CatBoost

---

## Parallelization & Multiprocessing

### Sequential Cohort Processing (Recommended for Large Jobs)

To avoid overloading the system, you can process one cohort at a time, using all available processes for that cohort, then move to the next cohort after completion. This ensures you never exceed your process limit and makes logs/outputs easier to manage.

**Pattern:**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def build_jobs_for_cohort(cohort):
    # Return a list of job dicts for all age_band/event_year for this cohort
    ...

for cohort in ["opioid_ed", "non_opioid_ed"]:
    jobs = build_jobs_for_cohort(cohort)
    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(run_single_cohort, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(f"Cohort {cohort} job complete: {result}")
    print(f"All jobs for {cohort} complete.")
```

**Benefits:**
- Never more than 30 jobs running at once
- No resource contention between cohorts
- Easier to debug and monitor

For details on parallel and SQS-based execution, see [`README_parallelization.md`](./README_parallelization.md).

---

## Prerequisites

### Data Requirements:
- **Cohort Data**: `s3://pgxdatalake/gold/cohorts_F1120/**/*.parquet`
- **Required Columns**: `mi_person_key`, `drug_name`, `event_date`, `is_target_case`
- **Format**: Hive-partitioned Parquet files

### System Requirements:
- **Python**: 3.8+
- **Memory**: 8GB+ RAM for global analysis, 20-30GB for cohort analysis
- **Storage**: S3 write access to `pgxdatalake` bucket
- **Network**: AWS S3 connectivity

### Environment Setup:
```bash
# Ensure project root is in Python path
export PYTHONPATH=/path/to/pgx-analysis:$PYTHONPATH

# Install dependencies
pip install mlxtend pandas numpy duckdb boto3 jupyter
```

---

## References

### Implementation Details
- `global_fpgrowth_feature_importance.ipynb` - Global analysis notebook
- `cohort_fpgrowth_feature_importance.ipynb` - Cohort analysis notebook
- `global_fpgrowth.py` - Global analysis script
- `cohort_fpgrowth.py` - Cohort analysis script

### Related Documentation
- `README_local_notebook.md` - Comprehensive local setup guide
- `README_parallelization.md` - Parallel processing guide
- `docs/README_bupaR.md` - Process mining with BupaR
- `README_catboost.md` - ML integration

### External References
- **FP-Growth Algorithm**: [MLxtend Documentation](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
- **Association Rules**: [MLxtend Association Rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)

---

**Questions or Issues?** Review the troubleshooting section above and check logs for specific error messages.

*Last Updated: November 24, 2025*  
*Module Version: 4.0 (Target-Focused)*  
*Compatible with: pgx_analysis pipeline v4+*

