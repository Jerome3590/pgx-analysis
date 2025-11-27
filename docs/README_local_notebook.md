# Local Notebook Configuration & Setup

**Last Updated:** November 22, 2025  
**Pipeline Version:** 3.0  
**Environment:** Windows 10/11 with HP Omen Laptop

---

## Overview

This guide covers the setup and configuration for running FPGrowth analysis notebooks locally on Windows. The analysis processes **947 million patient event records** across two notebooks:

1. **Global FPGrowth** (`global_fpgrowth_feature_importance.ipynb`)
2. **Cohort FPGrowth** (`cohort_fpgrowth_feature_importance.ipynb`)

---

## Hardware Configuration

### Tested Configuration (HP Omen Laptop)

```
CPU:    14 cores (recommended minimum: 8 cores)
RAM:    32 GB (recommended minimum: 16 GB)
GPU:    NVIDIA GPU (optional - not used for FPGrowth)
Disk:   SSD with 50+ GB free space
```

### Why GPU Isn't Used

**FP-Growth is CPU-bound**, not GPU-accelerated:
- The `mlxtend` library (FP-Growth implementation) is pure Python/NumPy
- No GPU-accelerated FPGrowth libraries exist for Python
- GPU would only help for later ML training (CatBoost, neural networks)

### CPU vs GPU Trade-offs

| Task | Engine | Hardware | Runtime |
|------|--------|----------|---------|
| **Data Loading** | DuckDB | CPU | Fast (seconds) |
| **FP-Growth** | mlxtend | CPU | Slow (hours) |
| **CatBoost Training** | CatBoost | CPU/GPU | Medium (minutes-hours) |
| **Neural Networks** | PyTorch | GPU | Fast with GPU |

---

## Software Requirements

### Core Dependencies

```bash
# Python 3.11+ (tested with Python 3.12)
python --version

# Required packages
pip install jupyter nbconvert
pip install mlxtend>=0.23.4  # FP-Growth algorithm
pip install duckdb           # Fast analytics database
pip install pandas numpy     # Data manipulation
pip install boto3            # AWS S3 access

# Optional monitoring
pip install psutil           # System monitoring
```

### Verify Installation

```python
# Check all packages
python -c "
import mlxtend
import duckdb
import pandas as pd
import boto3
print(f'✓ mlxtend {mlxtend.__version__}')
print(f'✓ duckdb {duckdb.__version__}')
print(f'✓ pandas {pd.__version__}')
print(f'✓ boto3 {boto3.__version__}')
"
```

---

## Data Configuration

### Local Data Structure

```
C:\Projects\pgx-analysis\
└── data\
    └── cohorts_F1120\
        ├── cohort_name=opioid_ed\
        │   ├── event_year=2016\
        │   │   ├── age_band=0-12\
        │   │   │   └── cohort.parquet
        │   │   ├── age_band=13-24\
        │   │   │   └── cohort.parquet
        │   │   └── ...
        │   └── ...
        └── cohort_name=non_opioid_ed\
            └── ...
```

### Data Statistics

- **Total Records:** 947,940,930 events
- **Medical Events:** 894,118,024 (94.3%)
- **Pharmacy Events:** 53,822,906 (5.7%)
- **Unique Drugs:** ~53M (one per pharmacy event)
- **Unique ICD Codes:** ~10K-50K (estimated)
- **Unique CPT Codes:** ~5K-10K (estimated)

### Sync from S3

```bash
# Initial sync (one-time, ~7 GB)
aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ data/cohorts_F1120/ \
  --exclude "*.log" \
  --exclude "*.json"

# Verify sync
du -sh data/cohorts_F1120/
# Expected: ~7.0 GB
```

### Python Dependencies (pip install)

Create/activate your environment, then install the required Python libraries:

```bash
python -m pip install --upgrade pip

python -m pip install \
  numpy pandas scipy scikit-learn \
  xgboost lightgbm catboost \
  duckdb pyarrow \
  boto3 botocore tenacity certifi urllib3 requests \
  mlxtend networkx matplotlib seaborn jinja2 \
  joblib tqdm psutil ipython jupyter
```

---

## Notebook Configuration

### Global FPGrowth Parameters

```python
# FP-Growth algorithm parameters
MIN_SUPPORT = 0.005      # Items must appear in 0.5% of patients
MIN_CONFIDENCE = 0.01    # Rules must have 1% confidence

# Item types to process
ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code']

# Output location
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/global"

# Local data path (for Windows local development)
LOCAL_DATA_PATH = "C:\\Projects\\pgx-analysis\\data\\cohorts_F1120"
```

### Cohort FPGrowth Parameters

```python
# Same MIN_SUPPORT, MIN_CONFIDENCE, ITEM_TYPES

# Parallel processing
MAX_WORKERS = 10  # For 14-core CPU (cores * 0.7)

# Output per cohort
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/cohort"
```

### Performance Tuning

**For Global Analysis:**
- Single-threaded (processes all data together)
- Memory: ~10-20 GB peak during FP-Growth
- Runtime: 1-2 hours per item type

**For Cohort Analysis:**
- Multi-threaded (10 workers for 90 cohorts)
- Memory: ~2-4 GB per worker
- Runtime: 3-5 hours total

---

## Running the Notebooks

### Method 1: Jupyter Notebook (Interactive)

```bash
# Start Jupyter
cd C:\Projects\pgx-analysis\3_fpgrowth_analysis
jupyter notebook

# Open in browser:
# - global_fpgrowth_feature_importance.ipynb
# - cohort_fpgrowth_feature_importance.ipynb

# Run cells sequentially
```

### Method 2: Command Line Execution (Automated)

```bash
# Global analysis
cd C:\Projects\pgx-analysis
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=14400 \
  --output 3_fpgrowth_analysis/executed_global_fpgrowth.ipynb \
  3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb

# Cohort analysis
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=18000 \
  --output 3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb \
  3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb
```

### Method 3: PowerShell Quick Start (Windows)

```powershell
# Run as Administrator
cd C:\Projects\pgx-analysis\3_fpgrowth_analysis
.\QUICK_START.ps1
```

---

## Output Structure

### Global FPGrowth Outputs

```
s3://pgxdatalake/gold/fpgrowth/global/
├── drug_name/
│   ├── encoding_map.json      # Universal drug encodings
│   ├── itemsets.json          # Frequent drug combinations
│   ├── rules.json             # Association rules
│   └── metrics.json           # Processing statistics
├── icd_code/
│   └── (same files)
└── cpt_code/
    └── (same files)
```

### Cohort FPGrowth Outputs

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

## Monitoring Progress

### Check Running Processes

```bash
# Find Jupyter processes
ps aux | grep jupyter | grep -v grep

# Check CPU/Memory usage
python -c "
import psutil
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
    if 'python' in proc.info['name'].lower():
        print(f\"{proc.info['name']} (PID {proc.info['pid']}): CPU {proc.info['cpu_percent']:.1f}%, Memory {proc.info['memory_percent']:.1f}%\")
"
```

### Check S3 Outputs

```bash
# List global outputs
aws s3 ls s3://pgxdatalake/gold/fpgrowth/global/ --recursive

# List cohort outputs
aws s3 ls s3://pgxdatalake/gold/fpgrowth/cohort/ --recursive | head -20

# Count completed cohorts
aws s3 ls s3://pgxdatalake/gold/fpgrowth/cohort/ --recursive | grep "metrics.json" | wc -l
```

### Monitor Log Files

```bash
# Global execution log
tail -f 3_fpgrowth_analysis/run_fixed.log

# Cohort execution log
tail -f 3_fpgrowth_analysis/cohort_execution.log
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError: No module named 'mlxtend'

**Solution:**
```powershell
# Run PowerShell as Administrator
python -m pip install mlxtend --upgrade
```

### Issue 2: Output buffering (no logs visible)

**Cause:** Jupyter buffers output until cells complete  
**Solution:** Check CPU usage to verify it's running:
```bash
ps aux | grep python | grep -v grep
```

### Issue 3: Out of Memory

**Symptoms:** Process killed, system freezes  
**Solution:** Reduce `MIN_SUPPORT` to decrease itemset size:
```python
MIN_SUPPORT = 0.01  # Increase from 0.005
```

### Issue 4: Event type not found (0 records)

**Cause:** Case sensitivity in SQL queries  
**Solution:** Use lowercase event types:
```python
# Correct
WHERE event_type = 'pharmacy'  # lowercase

# Wrong
WHERE event_type = 'PHARMACY'  # uppercase
```

### Issue 5: Slow DuckDB queries

**Solution:** Check file paths and hive partitioning:
```python
# Correct pattern
parquet_pattern = "C:\\Projects\\pgx-analysis\\data\\gold\\cohorts_F1120\\**\\cohort.parquet"

# Use hive_partitioning=1
read_parquet('{pattern}', hive_partitioning=1)
```

---

## Performance Expectations

### Global FPGrowth (per item type)

| Stage | Duration | CPU | Memory |
|-------|----------|-----|--------|
| Data Loading | 10-30s | 20% | 2 GB |
| Transaction Creation | 30-60s | 40% | 5 GB |
| Encoding | 1-2 min | 80% | 10 GB |
| FP-Growth | 30-60 min | 50% | 15 GB |
| Association Rules | 5-10 min | 60% | 10 GB |
| Save to S3 | 10-30s | 10% | 5 GB |

**Total per item type:** 40-75 minutes  
**Total for all 3 types:** 2-4 hours

### Cohort FPGrowth (90 cohorts, 10 workers)

| Stage | Duration | CPU | Memory |
|-------|----------|-----|--------|
| Setup | 1 min | 10% | 1 GB |
| Parallel Processing | 3-5 hours | 70-90% | 20-40 GB |
| Finalization | 5 min | 10% | 5 GB |

**Total:** 3-5 hours

---

## DuckDB Architecture

### Why DuckDB?

**Advantages:**
- ✅ **In-memory:** No database server needed
- ✅ **Parquet native:** Reads Hive-partitioned data directly
- ✅ **Fast:** Vectorized execution, columnar storage
- ✅ **SQL interface:** Easy to query and filter
- ✅ **Parallel:** Multi-threaded by default

**Use Cases:**
- Reading 947M rows from Parquet files
- Filtering medical/pharmacy events
- Grouping by patient for transactions
- Extracting unique items (drugs, ICD, CPT codes)

### DuckDB Configuration

```python
from helpers_1997_13.duckdb_utils import get_duckdb_connection

# Single-threaded (for multiprocessing compatibility)
con = get_duckdb_connection(logger=logger)
# Returns: DuckDB with 1 thread per worker

# Query example
result = con.execute("""
    SELECT DISTINCT drug_name
    FROM read_parquet('data/**/*.parquet', hive_partitioning=1)
    WHERE event_type = 'pharmacy'
      AND drug_name IS NOT NULL
""").fetchdf()
```

---

## Next Steps After Completion

### 1. Verify Outputs

```bash
# Check global outputs
aws s3 ls s3://pgxdatalake/gold/fpgrowth/global/drug_name/
aws s3 ls s3://pgxdatalake/gold/fpgrowth/global/icd_code/
aws s3 ls s3://pgxdatalake/gold/fpgrowth/global/cpt_code/

# Expected files per item type:
# - encoding_map.json
# - itemsets.json
# - rules.json
# - metrics.json
```

### 2. Download Results Locally

```bash
# Sync results for local analysis
aws s3 sync s3://pgxdatalake/gold/fpgrowth/ data/gold/fpgrowth/
```

### 3. Use in ML Models

```python
import json

# Load global encoding map
with open('data/gold/fpgrowth/global/drug_name/encoding_map.json') as f:
    drug_encodings = json.load(f)

# Apply to CatBoost features
df['drug_encoded'] = df['drug_name'].map(drug_encodings)
```

### 4. Run Cohort Analysis

After global analysis completes, run cohort-specific analysis:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=18000 \
  --output 3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb \
  3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb
```

---

## AWS Configuration

### Required Environment Variables

```bash
# Set in .bashrc or PowerShell profile
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"

# Or use AWS CLI
aws configure
```

### S3 Bucket Access

- **Bucket:** `pgxdatalake`
- **Input Path:** `s3://pgxdatalake/gold/cohorts_F1120/`
- **Output Path:** `s3://pgxdatalake/gold/fpgrowth/`
- **Required Permissions:** `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`

---

## References

- **FPGrowth Algorithm:** [mlxtend documentation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
- **DuckDB:** [DuckDB Python documentation](https://duckdb.org/docs/api/python/overview)
- **Project Pipeline:** See `docs/Analysis_Workflow_README.md`
- **Cohort Creation:** See `docs/README_create_cohort.md`

---

## Support

**For issues or questions:**
1. Check logs: `tail -f 3_fpgrowth_analysis/*.log`
2. Verify hardware: `python 3_fpgrowth_analysis/check_hardware.py`
3. Review this README
4. Check main project documentation

**Common Issues:** See Troubleshooting section above


