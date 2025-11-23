# FPGrowth Initial Filtering for Cohort Analysis

## Overview

This module provides **FPGrowth-based initial filtering** for drugs, ICD codes, and CPT codes before downstream analysis (CatBoost, BupaR). It discovers frequent patterns and association rules that can be used to:

1. **Filter features** before CatBoost modeling
2. **Identify patterns** for BupaR process mining
3. **Discover associations** for predictive modeling

## 🎯 Purpose

### For ED_NON_OPIOID Cohort
- **Drug Window Analysis**: Finds frequent drug patterns within the 30-day lookback window
- **Temporal Drug Associations**: Identifies which drugs co-occur before non-opioid ED visits
- **Feature Filtering**: Reduces drug feature space for CatBoost by focusing on frequent patterns

### For OPIOID_ED Cohort
- **ICD Code Patterns**: Discovers frequent diagnosis code combinations that predict opioid ED events
- **CPT Code Patterns**: Identifies procedure code patterns associated with opioid ED visits
- **Predictive Features**: Creates filtered feature sets for CatBoost modeling

## 📁 Module Structure

```
3_fpgrowth_analysis/
├── run_fpgrowth_cohort_filtering.py  # Main filtering pipeline
├── FPGrowth_Filtering_README.md      # This documentation
└── (uses existing fpgrowth_utils.py)
```

## 🚀 Quick Start

### Basic Usage

```bash
# Process all cohorts and item types
python run_fpgrowth_cohort_filtering.py

# Process specific cohort
python run_fpgrowth_cohort_filtering.py --cohort ED_NON_OPIOID

# Process specific item type
python run_fpgrowth_cohort_filtering.py --cohort OPIOID_ED --item-type icd

# Process specific age band and year
python run_fpgrowth_cohort_filtering.py --cohort ED_NON_OPIOID --item-type drug --age-band "65-74" --event-year 2020
```

### Cohort-Specific Item Types

| Cohort | Available Item Types | Notes |
|--------|---------------------|-------|
| **ED_NON_OPIOID** | `drug` | Only drugs within 30-day window |
| **OPIOID_ED** | `icd`, `cpt` | All ICD codes and CPT codes |

## 📊 Output Structure

Results are saved to S3 with the following structure:

```
s3://pgxdatalake/fpgrowth_features/
├── cohort_name=ed_non_opioid/
│   ├── age_band=65-74/
│   │   └── event_year=2020/
│   │       ├── itemsets_drug.json      # Frequent drug itemsets
│   │       ├── rules_drug.json         # Drug association rules
│   │       └── encoding_drug.parquet   # Drug encoding map
│   └── ...
└── cohort_name=opioid_ed/
    ├── age_band=65-74/
    │   └── event_year=2020/
    │       ├── itemsets_icd.json       # Frequent ICD itemsets
    │       ├── rules_icd.json          # ICD association rules
    │       ├── encoding_icd.parquet     # ICD encoding map
    │       ├── itemsets_cpt.json       # Frequent CPT itemsets
    │       ├── rules_cpt.json          # CPT association rules
    │       └── encoding_cpt.parquet    # CPT encoding map
    └── ...
```

## 🔧 Integration with Downstream Analysis

### 1. CatBoost Feature Filtering

Use FPGrowth results to filter features before CatBoost modeling:

```python
import json
from helpers_1997_13.s3_utils import load_from_s3_json

# Load frequent itemsets
itemsets_path = "s3://pgxdatalake/fpgrowth_features/cohort_name=ed_non_opioid/age_band=65-74/event_year=2020/itemsets_drug.json"
itemsets = load_from_s3_json(itemsets_path)

# Extract top frequent drugs
top_drugs = set()
for itemset in itemsets[:50]:  # Top 50 itemsets
    items = itemset.get('itemsets', [])
    for item in items:
        drug_name = item.replace('drug_', '')
        top_drugs.add(drug_name)

# Filter cohort data to only include top frequent drugs
filtered_df = cohort_df[cohort_df['drug_name'].isin(top_drugs)]
```

### 2. BupaR Process Mining

Use association rules for process flow analysis:

```python
# Load association rules
rules_path = "s3://pgxdatalake/fpgrowth_features/cohort_name=opioid_ed/age_band=65-74/event_year=2020/rules_icd.json"
rules = load_from_s3_json(rules_path)

# Filter high-confidence rules
high_conf_rules = [
    r for r in rules 
    if r.get('confidence', 0) > 0.7 and r.get('lift', 0) > 1.5
]

# Use for BupaR event log creation
# Rules indicate likely transitions: ICD_A -> ICD_B -> OPIOID_ED
```

### 3. Feature Importance Pre-filtering

Use support/confidence metrics to prioritize features:

```python
import pandas as pd
from helpers_1997_13.s3_utils import load_from_s3_parquet

# Load encoding map with metrics
encoding_path = "s3://pgxdatalake/fpgrowth_features/cohort_name=opioid_ed/age_band=65-74/event_year=2020/encoding_icd.parquet"
encoding_df = load_from_s3_parquet(encoding_path)

# Filter to high-support items
high_support_items = encoding_df[
    encoding_df['support'] > 0.05
]['item_name'].tolist()

# Use these items as features in CatBoost
```

## 📈 Analysis Workflow

### Complete Pipeline

```python
# Step 1: Run FPGrowth filtering
# python run_fpgrowth_cohort_filtering.py --cohort ED_NON_OPIOID --item-type drug

# Step 2: Load results for CatBoost
from run_fpgrowth_cohort_filtering import load_cohort_data
import pandas as pd

# Load filtered drug patterns
itemsets = load_from_s3_json("s3://.../itemsets_drug.json")
rules = load_from_s3_json("s3://.../rules_drug.json")

# Step 3: Use in CatBoost
# Filter features based on FPGrowth results
# Train CatBoost model with filtered features

# Step 4: Use in BupaR
# Create event log with FPGrowth pattern enrichment
# Analyze process flows with association rules
```

## ⚙️ Configuration

### Default Parameters

```python
MIN_SUPPORT_THRESHOLD = 0.05      # Minimum support for itemsets
MIN_CONFIDENCE_MEDIUM = 0.3       # Minimum confidence for rules
TOP_K = 30                        # Top K itemsets to extract
TIMEOUT_SECONDS = 300             # Timeout per FP-Growth run
```

### Custom Parameters

```bash
python run_fpgrowth_cohort_filtering.py \
    --cohort OPIOID_ED \
    --item-type icd \
    --min-support-threshold 0.03 \
    --timeout-seconds 600
```

## 🔍 Key Features

### ED_NON_OPIOID Drug Filtering

- **30-Day Window**: Only includes drugs within 30 days before target event
- **Balanced Windows**: Applies same temporal logic to targets and controls
- **Temporal Patterns**: Discovers drug sequences leading to ED visits

### OPIOID_ED Code Filtering

- **ICD Patterns**: Finds diagnosis code combinations predictive of opioid ED
- **CPT Patterns**: Identifies procedure patterns associated with opioid ED
- **Full History**: Includes all historical codes (no temporal filtering)

## 📋 Prerequisites

### Data Requirements
- **Cohort Data**: `s3://pgxdatalake/cohort_clean/**/*.parquet`
- **Required Columns**: 
  - `mi_person_key`, `drug_name` (for drugs)
  - `primary_icd_diagnosis_code` (for ICD)
  - `procedure_code` (for CPT)
  - `days_to_target_event` (for ED_NON_OPIOID drugs)

### System Requirements
- Python 3.8+
- mlxtend library
- DuckDB with S3 extension
- AWS S3 access

## 🛠️ Usage Examples

### Example 1: Filter Drugs for ED_NON_OPIOID

```bash
# Run FPGrowth for drugs
python run_fpgrowth_cohort_filtering.py \
    --cohort ED_NON_OPIOID \
    --item-type drug \
    --age-band "65-74" \
    --event-year 2020

# Results saved to:
# s3://pgxdatalake/fpgrowth_features/cohort_name=ed_non_opioid/age_band=65-74/event_year=2020/itemsets_drug.json
```

### Example 2: Filter ICD Codes for OPIOID_ED

```bash
# Run FPGrowth for ICD codes
python run_fpgrowth_cohort_filtering.py \
    --cohort OPIOID_ED \
    --item-type icd \
    --age-band "65-74" \
    --event-year 2020

# Results saved to:
# s3://pgxdatalake/fpgrowth_features/cohort_name=opioid_ed/age_band=65-74/event_year=2020/itemsets_icd.json
```

### Example 3: Process All Cohorts and Types

```bash
# Process everything
python run_fpgrowth_cohort_filtering.py

# This will:
# - Process ED_NON_OPIOID: drugs
# - Process OPIOID_ED: icd, cpt
# - Process all age bands and years
```

## 🔗 Integration Points

### Before CatBoost Analysis
1. Run FPGrowth filtering: `python run_fpgrowth_cohort_filtering.py`
2. Load frequent itemsets/rules
3. Filter cohort features based on FPGrowth results
4. Train CatBoost with filtered features

### Before BupaR Analysis
1. Run FPGrowth filtering for relevant item types
2. Load association rules
3. Create event log enriched with FPGrowth patterns
4. Analyze process flows with pattern-based transitions

## 📚 References

- **FP-Growth Algorithm**: [MLxtend Documentation](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
- **Association Rules**: [MLxtend Association Rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- **Cohort Creation**: See `docs/Create_Cohort_README.md`
- **CatBoost Integration**: See `3_fpgrowth_analysis/CatBoost_README.md`
- **BupaR Integration**: See `3_fpgrowth_analysis/BupaR_README.md`

---

*Last Updated: January 2025*
*Module Version: 1.0*
*Compatible with: pgx_analysis pipeline v2+*

