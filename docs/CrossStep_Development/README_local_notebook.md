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

### Process Health Check Script

For long-running analysis scripts (e.g., feature importance analysis), use the process health check script to monitor progress without relying on log files:

```bash
# Run health check (checks for run_cohort_2_65_74.py by default)
./check_process_health.sh

# Or specify a different script pattern
ps aux | grep "your_script_name" | grep -v grep
```

**What it checks:**
1. **Main process status**: PID, CPU, memory, runtime, state
2. **Child processes**: Number of parallel workers and their status
3. **All Python processes**: Complete list of related processes
4. **CPU usage breakdown**: Total and per-process CPU usage
5. **Memory usage**: Total memory and system memory status
6. **File system activity**: Recent output file updates
7. **System load**: Current system load average
8. **Disk I/O**: Disk activity (if `iostat` available)
9. **File descriptors**: Open file descriptor count
10. **Thread count**: Thread usage per process
11. **Process health assessment**: Identifies stuck or inactive processes

**Example Output:**
```
==========================================
Process Health Check - Mon Dec 18 14:30:00 EST 2025
==========================================

1. MAIN PROCESS STATUS:
   PID: 12345
   CPU: 2.5% | MEM: 15.3% | VSZ: 12345678 | RSS: 2345678 | Runtime: 02:15:30 | State: S

2. CHILD PROCESSES (Parallel Workers):
   Number of child processes: 28
   Child process details:
     PID 12346: CPU=45.2% MEM=2.1% Runtime=02:10:15 State=S
     PID 12347: CPU=42.8% MEM=2.0% Runtime=02:10:12 State=S
     ...
   Active children (CPU > 0.1%): 28 / 28

3. ALL PYTHON PROCESSES (including workers):
   PID 12345  CPU  2.5% MEM 15.3% Runtime 02:15:30 python run_cohort_2_65_74.py
   ...

4. CPU USAGE BREAKDOWN:
   Total CPU across all processes: 1250.5%
   Main process CPU: 2.5%

5. MEMORY USAGE:
   Total memory across all processes: 45.2%
   Main process memory: 15.3%
   System memory:
     Mem:   32Gi total,  18Gi used,  14Gi free,   2Gi buff/cache
     Swap:  16Gi total,   0Gi used,  16Gi free

6. FILE SYSTEM ACTIVITY:
   ✓ XGBoost output file exists:
     File: non_opioid_ed_65_74_xgboost_feature_importance.csv
     Size: 2.3M
     Last modified: 2025-12-18 14:25:30
     Status: ✓ Recently updated (< 1 hour ago)
   Files modified in last hour: 3

7. SYSTEM LOAD:
   load average: 12.5, 11.8, 10.2

8. DISK I/O:
   [iostat output if available]

9. PROCESS FILE DESCRIPTORS:
   Open file descriptors: 156

10. THREAD COUNT:
   Main process threads: 8
   Total threads (all processes): 224

11. PROCESS HEALTH ASSESSMENT:
   ✓ Main process is active (CPU: 2.5%)
   ✓ Worker processes are active (28/28 using CPU)
   Process runtime: 02:15:30

==========================================
SUMMARY:
  Main PID: 12345
  Child processes: 28
  Total CPU: 1250.5%
  Total Memory: 45.2%
  Output file: non_opioid_ed_65_74_xgboost_feature_importance.csv (2.3M)
==========================================
```

**Customizing for Different Scripts:**

To monitor a different script, edit `check_process_health.sh` and change the process pattern:

```bash
# Line 12: Change the grep pattern
MAIN_PID=$(ps aux | grep "your_script_name" | grep -v grep | awk '{print $2}')

# Line 47: Update the process filter
ps aux | grep -E "python.*your_script_name|python.*related_pattern" | grep -v grep
```

**Using with Feature Importance Analysis:**

```bash
# Monitor feature importance analysis
./check_process_health.sh

# Check every 5 minutes
watch -n 300 ./check_process_health.sh

# Save output to file
./check_process_health.sh >> process_health.log 2>&1
```

### Check Running Processes (Python)

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

# Feature importance logs
tail -f 3_feature_importance/logs/*.txt
```

### Parallel Process Logging

For parallel analysis jobs (e.g., Feature Importance Analysis with MC-CV), the pipeline implements **multiprocessing-safe logging** to ensure logs from worker processes are visible and correctly attributed.

#### How It Works

**Main Process:**
- Creates a unique log file with timestamp: `feature_importance_{cohort}_{age_band}_{year}_{timestamp}.log`
- Logs are written to both console (stdout) and file simultaneously
- Uses `AutoFlushHandler` to ensure immediate visibility in console

**Worker Processes:**
- Each worker process writes to the **same log file** as the main process (append mode)
- All log messages include a `[Worker-ProcessName-PID]` prefix for identification
- Example: `[Worker-LokyProcess-27-268558] 2025-12-22 18:29:30,840 - INFO - ...`

#### Log Message Format

```
[Worker-ProcessName-PID] YYYY-MM-DD HH:MM:SS,mmm - LEVEL - Message
```

**Components:**
- `Worker-ProcessName-PID`: Identifies which worker process generated the log
  - Process name: `LokyProcess`, `ForkPoolWorker`, etc. (depends on backend)
  - PID: Process ID for tracking individual workers
- Timestamp: Standard Python logging timestamp
- Level: `INFO`, `WARNING`, `ERROR`, etc.
- Message: The actual log message

#### Example Log Output

```bash
# Main process logs
2025-12-22 18:22:06,749 - INFO - --- Running MC-CV for xgboost (25 splits) ---
2025-12-22 18:22:06,750 - INFO - Using 28 parallel workers for MC-CV splits

# Worker process logs (with identification)
[Worker-LokyProcess-18-268549] 2025-12-22 18:22:10,123 - INFO - [MC-CV] Split 0 (xgboost): training model on 3065192 samples × 1519 features
[Worker-LokyProcess-27-268558] 2025-12-22 18:22:10,456 - INFO - [MC-CV] Split 1 (xgboost): training model on 3065192 samples × 1519 features
[Worker-LokyProcess-18-268549] 2025-12-22 18:31:46,789 - INFO - [MC-CV] Split 0 (xgboost): training completed in 576.3s
[Worker-LokyProcess-18-268549] 2025-12-22 18:33:35,012 - INFO - [MC-CV] Split 0 (xgboost): prediction completed in 108.2s
[Worker-LokyProcess-18-268549] 2025-12-22 18:29:30,840 - INFO - Permutation importance: baseline score=0.042741 on 766298 rows × 1519 features
```

#### Monitoring Parallel Jobs

**Real-Time Monitoring:**
```bash
# Watch logs in real-time (all workers visible)
tail -f 3_feature_importance/logs/feature_importance_*.log

# Filter by specific worker
tail -f 3_feature_importance/logs/feature_importance_*.log | grep "Worker-LokyProcess-27"

# Count active workers
tail -f 3_feature_importance/logs/feature_importance_*.log | grep -o "Worker-[^]]*" | sort -u | wc -l
```

**Progress Tracking:**
```bash
# Count completed splits
grep "training completed" 3_feature_importance/logs/feature_importance_*.log | wc -l

# Check for errors
grep -i error 3_feature_importance/logs/feature_importance_*.log

# View timing information
grep "training completed\|prediction completed" 3_feature_importance/logs/feature_importance_*.log
```

#### Benefits

1. **Visibility**: See progress from all workers in real-time
2. **Debugging**: Identify which worker encountered an issue
3. **Performance Analysis**: Track timing for each worker and split
4. **Error Isolation**: Worker-specific error messages help isolate problems

#### Technical Details

- **File Locking**: Python's logging module handles file locking automatically for multi-process writes
- **Append Mode**: Workers write in append mode to avoid overwriting each other's logs
- **Auto-Flush**: Console handler flushes immediately for real-time visibility
- **Backend Compatibility**: Works with both `loky` (process-based) and `threading` backends

For more details, see the [Parallelization Pipeline README](../CrossStep_Development/README_parallelization_pipeline.md#worker-process-logging).

### Additional Monitoring Scripts

#### EC2 Health & Script Monitoring

For comprehensive EC2 instance and script health checks:

```bash
# Basic usage (checks default log pattern)
./monitor_ec2_script.sh

# Specify custom log file pattern
./monitor_ec2_script.sh "logs/*feature*.txt"
```

**What it checks:**
1. Python process status (running/stopped)
2. System resources (CPU, memory, disk)
3. Recent log activity (last 5 minutes)
4. Output file status
5. EC2 instance metadata (if running on EC2)
6. Process health (stuck processes detection)
7. Summary and recommendations

**Example Output:**
```
==========================================
EC2 Health & Script Monitoring
==========================================

1. Checking Python processes...
   ✓ Python script is running:
     user 12345 2.5 15.3 ... python run_cohort_2_65_74.py

2. System Resources:
   CPU Usage: %Cpu(s): 12.5 us, 2.3 sy, ...
   Memory Usage: Mem: 32Gi total, 18Gi used, 14Gi free
   Disk Usage: Used: 50G / 200G (25%)

3. Log File Activity (most recent: logs/feature_importance_...txt):
   Last modified: 2025-12-18 14:30:00
   File size: 2.3M
   Last 10 log lines:
     [2025-12-18 14:30:00] Completed split 15/25
     ...
   ✓ Log file was modified in the last 5 minutes

4. Output Files Check:
   ✓ non_opioid_ed_65_74_xgboost_feature_importance.csv - modified: 2025-12-18 14:25:00

5. Network Connectivity:
   ✓ Running on EC2 instance: i-1234567890abcdef0
   Instance type: m5.8xlarge

6. Process Health Check:
   PID 12345: CPU=2.5%, MEM=15.3%, Runtime=02:15:30
     ✓ Process is active (CPU usage > 0.1%)

==========================================
Summary & Recommendations:
==========================================
✓ Script appears to be running normally
  → Continue monitoring log file for progress
  → Expected completion: 25 MC-CV splits for xgboost
```

#### Real-Time Process Monitoring

For continuous real-time monitoring (updates every 5 seconds):

```bash
# Start real-time monitoring
./monitor_processes_realtime.sh

# Press Ctrl+C to stop
```

**Features:**
- Updates every 5 seconds
- Shows main process and all child processes
- Displays CPU, memory, runtime for each process
- Shows system resources (memory, load average)
- Tracks output file status
- Clear screen for easy reading

**Example Output:**
```
==========================================
Real-time Process Monitor - 14:30:15
==========================================

MAIN PROCESS:
  PID: 12345 | CPU: 2.5% | MEM: 15.3% | Runtime: 02:15:30 | State: S

CHILD PROCESSES: 28
  PID 12346  CPU 45.2% MEM  2.1% Runtime 02:10:15
  PID 12347  CPU 42.8% MEM  2.0% Runtime 02:10:12
  ...
  Active workers: 28 / 28

SYSTEM RESOURCES:
  Memory: 18Gi / 32Gi (56.25%)
  Load:  12.5, 11.8, 10.2

OUTPUT FILE:
  non_opioid_ed_65_74_xgboost_feature_importance.csv: 2.3M (modified: 14:25:30)

Press Ctrl+C to stop monitoring
```

#### Quick Progress Check

For a quick one-time status check:

```bash
./CHECK_PROGRESS_NOW.sh
```

**What it shows:**
1. Process status (PID, CPU, MEM, start time)
2. Latest log file and modification time
3. Completed splits count
4. Current phase (permutation importance, parallel progress)
5. Recent log activity (last 20 lines)
6. Error count and recent errors
7. Output file status

**Example Output:**
```
==========================================
Script Progress Check
==========================================

1. PROCESS STATUS:
   PID: 12345, CPU: 2.5%, MEM: 15.3%, Started: Dec 18 12:15

2. LOG FILE:
   Latest: logs/feature_importance_...txt
   Modified: 2025-12-18 14:30:00
   Size: 2.3M
   Status: ✓ Recently active

3. COMPLETED SPLITS:
   Total: 15 / 25
   Last completed:
     [2025-12-18 14:30:00] Completed split 15/25

4. CURRENT PHASE:
   Latest progress:
     Permutation importance progress: 500/1519 features (32.9%)

5. RECENT LOG ACTIVITY (last 20 lines):
   [shows last 20 log lines]

6. ERRORS:
   Count: 0

7. OUTPUT FILES:
   XGBoost output exists: .../non_opioid_ed_65_74_xgboost_...
   Last modified: 2025-12-18 14:25:00
   Size: 2.3M

==========================================
✓ STATUS: Script running - 15/25 splits completed
==========================================
```

#### Progress Estimation

For time estimation and progress tracking:

```bash
./estimate_progress.sh
```

**What it estimates:**
1. Worker runtimes (shows longest running worker)
2. Output file status and time since last update
3. Active vs. finished workers
4. Time estimation based on runtime
5. Completion indicators (row count checks)

**Example Output:**
```
==========================================
Progress Estimation
==========================================

1. WORKER RUNTIMES:
   Worker runtimes:
   02:10:15
   02:10:12
   ...
   Longest running worker: 02:10:15

2. OUTPUT FILE STATUS:
   File: non_opioid_ed_65_74_xgboost_feature_importance.csv
   Size: 2.3M
   Last modified: 2025-12-18 14:25:00
   Time since last update: 5 minutes
   Status: ✓ File is being updated

3. WORKER STATUS:
   Active workers (CPU > 0.1%): 28 / 28

4. TIME ESTIMATION:
   Main process runtime: 02:15:30
   Note: With 20 workers and 25 splits:
   - First batch: 20 splits (currently running)
   - Second batch: 5 splits (will start after first batch completes)

5. COMPLETION INDICATORS:
   Progress: ~15 splits completed (estimated from row count)

==========================================
SUMMARY:
  Workers running: 28 / 28
  Runtime: ~2 hours 15 minutes
  Status: ✓ All workers active and computing
==========================================
```

#### Python-Based Status Check (Cross-Platform)

For cross-platform monitoring using Python:

```bash
# One-time check
python check_script_status.py

# Continuous monitoring (updates every 60 seconds)
python check_script_status.py --watch

# Custom interval (every 30 seconds)
python check_script_status.py --watch --interval 30
```

**Features:**
- Cross-platform (works on Windows, Linux, Mac)
- Finds Python processes automatically
- Checks log file activity
- Shows recent log lines
- Checks output files
- Counts completed splits
- Continuous monitoring mode

**Example Output:**
```
==========================================
Script Status Check
==========================================

Process Status:
  PID: 12345
  CPU: 2.5%
  Memory: 15.3%
  Command: python run_cohort_2_65_74.py

Log File:
  Latest: logs/feature_importance_...txt
  Modified: 2025-12-18 14:30:00 (2 minutes ago)
  Size: 2.3 MB
  Status: ✓ Active (modified recently)

Recent Log Activity:
  [shows last 20 lines]

Output Files:
  ✓ non_opioid_ed_65_74_xgboost_feature_importance.csv
    Modified: 2025-12-18 14:25:00
    Size: 2.3 MB

Completed Splits: 15 / 25

==========================================
Status: ✓ Running normally
==========================================
```

### Creating Custom Monitoring Scripts

You can create custom monitoring scripts based on these patterns:

**Bash Script Template:**
```bash
#!/bin/bash
# Custom monitoring script

MAIN_PID=$(ps aux | grep "your_script_name" | grep -v grep | awk '{print $2}')

if [ -z "$MAIN_PID" ]; then
    echo "Process not found!"
    exit 1
fi

# Check process status
ps -p $MAIN_PID -o pid,%cpu,%mem,etime,cmd

# Check child processes
CHILD_COUNT=$(pgrep -P $MAIN_PID 2>/dev/null | wc -l)
echo "Child processes: $CHILD_COUNT"

# Check log files
LATEST_LOG=$(ls -t logs/*.txt 2>/dev/null | head -1)
if [ ! -z "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    tail -20 "$LATEST_LOG"
fi
```

**Python Script Template:**
```python
#!/usr/bin/env python3
import subprocess
import glob
import os
from pathlib import Path

# Find process
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'your_script_name' in line and 'grep' not in line:
        print(f"Found: {line}")

# Find latest log
logs = glob.glob('logs/*.txt')
if logs:
    latest = max(logs, key=os.path.getmtime)
    print(f"Latest log: {latest}")
    with open(latest) as f:
        print("".join(f.readlines()[-20:]))
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


