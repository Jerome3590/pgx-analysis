# Running Multiple FFA Analysis Cohorts in Parallel

## Summary

**Yes, the FFA analysis can handle two cohorts running simultaneously** without conflicts. Each cohort uses completely separate file paths and resources.

## Isolation Mechanisms

### 1. **Cohort-Specific Output Directories**
- **Path structure**: `8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/`
- **Example**: 
  - Cohort 1: `8_ffa_analysis/outputs/opioid_ed/13-24/xgboost/`
  - Cohort 2: `8_ffa_analysis/outputs/non_opioid_ed/75-84/xgboost/`
- **No conflicts**: Each cohort writes to its own directory

### 2. **Cohort-Specific Input Files**
- **Model JSONs**: `6_final_model/outputs/{cohort}/{age_band}/final_model_json/`
- **Data files**: `6_final_model/outputs/{cohort}/{age_band}/inputs/model_train/`
- **SHAP files**: `7_shap_analysis/outputs/{cohort}/{age_band}/`
- **No conflicts**: Each cohort reads from its own input directory

### 3. **Unique Log Files**
- **Path**: `8_ffa_analysis/logs/ffa_analysis_{timestamp}.log`
- **Format**: Includes timestamp (e.g., `ffa_analysis_20260113_143022.log`)
- **No conflicts**: Each run gets a unique log file

### 4. **No Shared State**
- **No file locks**: No locking mechanisms that would block concurrent execution
- **No global singletons**: Each script instance is independent
- **Temp files**: Uses `tempfile.NamedTemporaryFile` with unique names (OS-managed)

### 5. **Independent Process Pools**
- Each script instance creates its own `ProcessPoolExecutor`
- Workers are isolated per instance
- No shared worker pool that could cause conflicts

## Resource Considerations

### CPU Usage
**Current configuration:**
- **Per instance**: Up to 28 workers (leaves 4 cores free)
- **Two instances**: 28 × 2 = 56 workers on 32 cores
- **Issue**: Oversubscription (56 workers > 32 cores)

**Recommendation for parallel execution:**
- **Option 1**: Reduce `n_jobs` per instance to 14 workers each (14 × 2 = 28 total)
- **Option 2**: Use 12 workers each (12 × 2 = 24 total, leaves 8 cores free)
- **Option 3**: Keep 28 workers each but expect context switching overhead (may still be faster than sequential)

### Memory Usage
- **Available**: 1TB RAM
- **Current usage**: ~4.5GB per instance (estimated)
- **Two instances**: ~9GB total
- **Risk**: Low - plenty of headroom

### Disk I/O
- **Separate directories**: Each cohort writes to different paths
- **Risk**: Low - minimal contention
- **Consideration**: Both instances may read from S3 simultaneously (should be fine)

## Recommended Configuration for Parallel Execution

### Option 1: Balanced (Recommended)
Run two cohorts with reduced workers per instance:

```bash
# Terminal 1
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name opioid_ed \
  --age-band 13-24

# Terminal 2 (adjust n_jobs in script or via environment)
# Temporarily set n_jobs to 14 in ANALYSIS_CONFIG
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 75-84
```

**Modify script temporarily** (or add CLI flag):
```python
# In run_full_ffa_analysis.py, before main():
import os
if os.environ.get('FFA_REDUCED_WORKERS') == 'true':
    ANALYSIS_CONFIG['n_jobs'] = min(14, max(1, get_sklearn_n_jobs()))
```

### Option 2: Full Parallelism (May Cause Context Switching)
Run both with full 28 workers each:
- **Pros**: Maximum parallelism per cohort
- **Cons**: Context switching overhead (56 workers on 32 cores)
- **Result**: May still be faster than sequential, but not optimal

### Option 3: Sequential (Safest)
Run one cohort at a time:
- **Pros**: No resource contention
- **Cons**: Takes longer overall
- **Result**: ~30-60 minutes per cohort sequentially

## Monitoring Parallel Execution

### Check CPU Usage
```bash
# Should see ~24-28 cores utilized if running one instance
# Should see all 32 cores utilized if running two instances (with context switching)
top -p $(pgrep -f run_full_ffa_analysis | tr '\n' ',' | sed 's/,$//')
```

### Check Process Count
```bash
# Should see ~28-30 processes per instance
ps aux | grep run_full_ffa_analysis | wc -l
```

### Check Memory Usage
```bash
# Monitor memory usage
free -h
# Or per-process
ps aux | grep run_full_ffa_analysis | awk '{sum+=$6} END {print sum/1024 " MB"}'
```

### Check Disk I/O
```bash
# Monitor disk I/O
iostat -x 5
```

## Example: Running Two Cohorts in Parallel

### Terminal 1
```bash
cd /home/pgx3874/pgx-analysis
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name opioid_ed \
  --age-band 55-64 \
  --model-type xgboost
```

### Terminal 2
```bash
cd /home/pgx3874/pgx-analysis
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74 \
  --model-type xgboost
```

## Potential Issues and Mitigations

### Issue 1: CPU Oversubscription
**Symptom**: High context switching, slower than expected
**Mitigation**: Reduce `n_jobs` to 14 per instance (or add CLI flag)

### Issue 2: Memory Pressure
**Symptom**: OOM errors, swapping
**Mitigation**: Unlikely with 1TB RAM, but monitor memory usage

### Issue 3: Disk I/O Contention
**Symptom**: Slow disk writes
**Mitigation**: Separate directories minimize contention; monitor with `iostat`

### Issue 4: S3 Rate Limiting
**Symptom**: S3 download failures or throttling
**Mitigation**: S3 handles concurrent requests well; unlikely to be an issue

## Conclusion

**Yes, you can run two cohorts simultaneously.** The script is designed with proper isolation:

✅ **Separate output directories**  
✅ **Separate input files**  
✅ **Unique log files**  
✅ **No shared state or locks**  
✅ **Independent process pools**

**Recommendation**: For optimal performance, reduce `n_jobs` to 14 per instance when running two cohorts in parallel (14 × 2 = 28 workers total, optimal for 32-core system).
