# Pipeline Time Estimates (New Workflow)

**System:** EC2 with 32 cores, 1TB RAM  
**Date:** January 2026  
**Workflow:** Steps 3 → 4a → 4b → 5c → 6 → 7 → 8 → 9

---

## Per-Cohort Time Estimates (Sequential Execution)

Based on actual benchmarks and EC2 performance characteristics:

| Cohort | Age Band | Events | Step 4a | Step 4b | Step 5c | Step 6 | Step 7 | Step 8 | Step 9 | **Total** |
|--------|----------|--------|---------|---------|---------|--------|--------|--------|--------|-----------|
| **opioid_ed** | 13-24 | ~600K | 10-15 min | 15-20 min | 10-15 min | 30-45 min | 1-1.5 hrs | 1-1.5 hrs | 10-15 min | **~4-5 hours** |
| **opioid_ed** | 25-44 | ~4.6M | 30-45 min | 30-45 min | 20-30 min | 1-2 hrs | 2-3 hrs | 2-3 hrs | 20-30 min | **~7-10 hours** |
| **opioid_ed** | 45-54 | ~1.5M | 20-30 min | 20-30 min | 15-20 min | 45-60 min | 1.5-2 hrs | 1.5-2 hrs | 15-20 min | **~5-7 hours** |
| **opioid_ed** | 55-64 | ~3.2M | 30-45 min | 30-45 min | 20-30 min | 1-2 hrs | 2-3 hrs | 2-3 hrs | 20-30 min | **~7-10 hours** |
| **non_opioid_ed** | 65-74 | ~2.9M | 30-45 min | 30-45 min | 20-30 min | 1-2 hrs | 2-3 hrs | 2-3 hrs | 20-30 min | **~7-10 hours** |
| **non_opioid_ed** | 75-84 | ~1.2M | 20-30 min | 20-30 min | 15-20 min | 45-60 min | 1.5-2 hrs | 1.5-2 hrs | 15-20 min | **~5-7 hours** |
| **non_opioid_ed** | 85-94 | ~275K | 10-15 min | 10-15 min | 10-15 min | 30-45 min | 45-60 min | 45-60 min | 10-15 min | **~3-4 hours** |

**Note:** Step 3 (Feature Importance) is already complete for all cohorts (~6.5-7.5 hours per cohort when run).

---

## Step-by-Step Breakdown

### Step 4a: Model Data Creation (Cases + Controls)
- **Small cohorts** (275K events): 10-15 minutes
- **Medium cohorts** (600K-1.5M events): 20-30 minutes
- **Large cohorts** (2.9M-4.6M events): 30-45 minutes
- **Bottleneck:** DuckDB queries on gold medical/pharmacy data, S3 sync

### Step 4b: DTW Protocol Filtering
- **Small cohorts**: 10-15 minutes
- **Medium cohorts**: 15-30 minutes
- **Large cohorts**: 30-45 minutes
- **Bottleneck:** Interval calculations, protocol event detection

### Step 5c: PGx Feature Engineering
- **All cohorts**: 10-30 minutes (relatively fast)
- **Bottleneck:** Drug-gene mapping, allele frequency lookups

### Step 6: Final Model Training
- **Small cohorts**: 30-45 minutes
- **Medium cohorts**: 45-60 minutes
- **Large cohorts**: 1-2 hours
- **Bottleneck:** CatBoost/XGBoost training with cross-validation, model selection

### Step 7: FFA Analysis
- **Small cohorts**: 45-60 minutes
- **Medium cohorts**: 1.5-2 hours
- **Large cohorts**: 2-3 hours
- **Bottleneck:** XGBoost AXP explanation generation (computationally intensive)

### Step 8: SHAP Analysis
- **Small cohorts**: 45-60 minutes
- **Medium cohorts**: 1.5-2 hours
- **Large cohorts**: 2-3 hours
- **Bottleneck:** SHAP value computation for all samples (memory-intensive)

### Step 9: Combined SHAP/FFA Analysis
- **All cohorts**: 10-30 minutes (mostly aggregation/combination)
- **Bottleneck:** Feature consensus calculation

---

## Parallel Execution Strategy

With 32 cores and 1TB RAM, you can run **multiple cohorts in parallel**:

### Recommended Parallelization:
- **3-4 cohorts simultaneously** (depending on cohort size)
- Each cohort uses ~8-12 cores internally
- Total memory usage: ~200-300GB for 3-4 cohorts

### Example Parallel Execution:
```bash
# Terminal 1: Small cohort (85-94)
nohup ./run_cohort_workflow.sh non_opioid_ed 85-94 > logs/non_opioid_ed_85-94.log 2>&1 &

# Terminal 2: Medium cohort (13-24)
nohup ./run_cohort_workflow.sh opioid_ed 13-24 > logs/opioid_ed_13-24.log 2>&1 &

# Terminal 3: Medium cohort (75-84)
nohup ./run_cohort_workflow.sh non_opioid_ed 75-84 > logs/non_opioid_ed_75-84.log 2>&1 &

# Terminal 4: Large cohort (25-44) - wait for one to finish first
nohup ./run_cohort_workflow.sh opioid_ed 25-44 > logs/opioid_ed_25-44.log 2>&1 &
```

---

## Total Time Estimates

### Sequential (One at a Time):
- **All 7 cohorts**: ~38-47 hours total
- **Fastest path**: Start with small cohorts (85-94, 13-24, 75-84) = ~12-16 hours
- **Then large cohorts** (25-44, 55-64, 65-74, 45-54) = ~26-31 hours

### Parallel (3-4 cohorts simultaneously):
- **Batch 1** (3 small/medium): ~5-7 hours (wall time)
- **Batch 2** (3 medium/large): ~7-10 hours (wall time)
- **Batch 3** (1 large): ~7-10 hours (wall time)
- **Total wall time**: ~19-27 hours

### Optimal Strategy (4 cohorts parallel):
- **Round 1**: 85-94, 13-24, 75-84, 45-54 → ~5-7 hours
- **Round 2**: 25-44, 55-64, 65-74 → ~7-10 hours
- **Total**: ~12-17 hours wall time

---

## Factors Affecting Runtime

1. **Data Size**: Primary driver (event count)
2. **S3 I/O**: Minimal impact - all data is local on NVMe (no download throttling)
3. **Memory**: 1TB allows multiple cohorts
4. **CPU**: 32 cores allows good parallelization
5. **Checkpointing**: S3 checkpoints add ~1-2 min overhead per step (uploads only)
6. **Local NVMe**: Fast I/O, no network bottleneck for reads

---

## Current Status (from S3 Checkpoints)

- ✅ **Step 4a**: All 7 cohorts have outputs (may need regeneration with controls)
- ⚠️ **Step 5c**: 3 cohorts have PGx features (opioid_ed: 13-24, 25-44, 45-54)
- ❌ **Steps 4b, 6, 7, 8, 9**: Not started

**Remaining work:**
- Regenerate Step 4a outputs with controls (if needed): ~2-4 hours total
- Complete Step 4b for all cohorts: ~2-3 hours total
- Complete Step 5c for remaining 4 cohorts: ~1-2 hours total
- Complete Steps 6-9 for all cohorts: ~25-35 hours total (sequential) or ~12-17 hours (parallel)

---

## Monitoring Progress

Use the checkpoint status script:
```bash
python utility_scripts/check_s3_checkpoints.py
```

This shows which steps are complete for each cohort.

