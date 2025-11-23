# FPGrowth Analysis

**Last Updated:** November 23, 2025  
**Pipeline Version:** 3.0

---

## Overview

This directory contains FP-Growth analysis tools for discovering frequent patterns and associations in patient healthcare data. FP-Growth identifies:
- **Frequent itemsets**: Items that commonly appear together
- **Association rules**: Relationships between items (e.g., "if X then Y")
- **Feature importance**: Support and confidence scores for ML feature engineering

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

### 2. Configuration

Edit parameters in scripts/notebooks as needed:

```python
MIN_SUPPORT = 0.01      # Items must appear in 1% of patients
MIN_CONFIDENCE = 0.01   # Rules must have 1% confidence
MAX_WORKERS = 5         # Parallel workers (cohort analysis only)
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

## Output Structure

### Global Analysis
```
s3://pgxdatalake/gold/fpgrowth/global/
├── drug_name/
│   ├── encoding_map.json      # Feature encodings for ML
│   ├── itemsets.json          # Frequent drug combinations
│   ├── rules.json             # Association rules
│   └── metrics.json           # Processing statistics
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
│               ├── encoding_map.json
│               ├── itemsets.json
│               ├── rules.json
│               └── metrics.json
├── icd_code/
│   └── (same structure)
└── cpt_code/
    └── (same structure)
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

## Using Results in ML Models

### Load Encoding Map
```python
import json

# Load global encoding map
with open('data/gold/fpgrowth/global/drug_name/encoding_map.json') as f:
    drug_encodings = json.load(f)

# Apply to features
df['drug_support'] = df['drug_name'].map(lambda x: drug_encodings.get(x, {}).get('support', 0))
df['drug_rank'] = df['drug_name'].map(lambda x: drug_encodings.get(x, {}).get('rank', 999999))
```

### Load Association Rules
```python
import pandas as pd

# Load rules
rules = pd.read_json('data/gold/fpgrowth/global/drug_name/rules.json')

# Find high-lift rules
top_rules = rules.nlargest(10, 'lift')
print(top_rules[['antecedents', 'consequents', 'lift', 'confidence']])
```

---

## Next Steps

1. **Verify Outputs**: Check S3 for generated JSON files
2. **Download Results**: `aws s3 sync s3://pgxdatalake/gold/fpgrowth/ data/gold/fpgrowth/`
3. **Integrate with ML**: Use encoding maps and rules in CatBoost/neural network models
4. **Process Mining**: Use `BupaR` for detailed patient journey analysis

---

## Support

For detailed configuration and troubleshooting, see:
- `README_local_notebook.md` - Comprehensive setup guide
- `FpGrowth_README.md` - Algorithm details
- Main project documentation in `docs/`

---

**Questions or Issues?** Review the troubleshooting section above and check logs for specific error messages.


