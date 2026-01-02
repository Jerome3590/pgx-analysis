# EC2 Runtime Estimates (32 cores, 1TB RAM, Full Parallelization)

**System:** EC2 with 32 cores, 1TB RAM  
**Strategy:** Maximum parallelization (all cohorts + all independent steps)

---

## Key Assumptions

1. **Worker Configuration:**
   - 28 workers (leaving 4 cores for system/OS overhead)
   - 2 threads per XGBoost model (28 workers × 2 = 56 threads on 32 cores)
   - 1TB RAM allows running multiple cohorts simultaneously

2. **Parallelization Strategy:**
   - **Multiple cohorts in parallel:** Run all 5 pending cohorts simultaneously
   - **Independent steps in parallel:** Steps 5a, 5c, 5d can run simultaneously after 5b
   - **Steps 7 and 8 in parallel:** Can run simultaneously after Step 6

3. **EC2 Performance vs Local:**
   - 2-3x faster for CPU-bound tasks (better CPU, faster I/O)
   - S3 access is much faster (same region)
   - No Windows overhead

---

## Per-Cohort Sequential Time (Steps 4b-8) on EC2

Based on actual benchmarks and EC2 optimizations:

| Cohort | Events (train) | Step 4b | Step 4c | Step 5b | Step 5a | Step 5c | Step 5d | Step 6 | Step 7 | Step 8 | **Total Sequential** |
|--------|----------------|---------|---------|---------|---------|---------|---------|--------|--------|--------|---------------------|
| **opioid_ed 0-12** | 2K | 2 min | 2 min | 5 min | 3 min | 2 min | 3 min | 5 min | 10 min | 10 min | **~42 min** |
| **opioid_ed 55-64** | 3.2M | 15 min | 5 min | 1.5-2 hrs | 30 min | 10 min | 30 min | 30 min | 1-1.5 hrs | 1-1.5 hrs | **~5-7 hours** |
| **non_opioid_ed 65-74** | 2.9M | 15 min | 5 min | 1.5-2 hrs | 30 min | 10 min | 30 min | 30 min | 1-1.5 hrs | 1-1.5 hrs | **~5-7 hours** |
| **non_opioid_ed 75-84** | 1.2M | 10 min | 5 min | 1-1.5 hrs | 20 min | 8 min | 20 min | 20 min | 45 min | 45 min | **~3.5-4.5 hours** |
| **non_opioid_ed 85-94** | 274K | 5 min | 3 min | 30-45 min | 10 min | 5 min | 10 min | 10 min | 20 min | 20 min | **~1.5-2 hours** |

**Note:** These are sequential times with internal parallelization (28 workers for MC-CV, etc.)

---

## Parallel Execution Strategy

### Strategy 1: All Cohorts in Parallel (Maximum Throughput)

**Run all 5 cohorts simultaneously, each in its own process/terminal:**

```
Terminal 1: opioid_ed 0-12 (Steps 4b-8)
Terminal 2: opioid_ed 55-64 (Steps 4b-8)
Terminal 3: non_opioid_ed 65-74 (Steps 4b-8)
Terminal 4: non_opioid_ed 75-84 (Steps 4b-8)
Terminal 5: non_opioid_ed 85-94 (Steps 4b-8)
```

**Total Wall Time:** ~7 hours (bottleneck is the longest cohort: 55-64 or 65-74)

**Resource Usage:**
- CPU: ~140 workers total (28 per cohort × 5 cohorts) - **OVERSUBSCRIBED**
- RAM: ~500GB total (100GB per cohort estimate) - **WITHIN LIMIT**

**Problem:** CPU oversubscription will slow things down.

---

### Strategy 2: Optimal Parallelization (Recommended)

**Phase 1: Data Preparation (Parallel)**
- Run Steps 4b and 4c for all 5 cohorts in parallel
- **Time:** ~20 minutes (bottleneck: 15 min for large cohorts)

**Phase 2: Feature Engineering (Sequential by cohort, parallel within)**
- Run Steps 5b, then 5a/5c/5d in parallel for each cohort
- **Option A:** Run cohorts sequentially (one at a time)
  - Total: 42 min + 7 hrs + 7 hrs + 4.5 hrs + 2 hrs = **~21 hours**
- **Option B:** Run 2-3 cohorts in parallel (balance CPU/RAM)
  - Run: (0-12 + 85-94) in parallel, then (55-64 + 75-84) in parallel, then 65-74 alone
  - Total: ~7 hours (bottleneck: 55-64 or 65-74)

**Phase 3: Modeling & Analysis (Parallel where possible)**
- Run Step 6 for all cohorts in parallel (if RAM allows)
- Run Steps 7 and 8 in parallel for each cohort
- **Time:** ~1.5-2 hours (bottleneck: FFA/SHAP for large cohorts)

**Total Wall Time (Option B):** ~10-12 hours

---

### Strategy 3: Maximum Parallelization (Aggressive)

**Run 2-3 cohorts simultaneously throughout:**

**Batch 1 (Small cohorts):**
- `opioid_ed 0-12` + `non_opioid_ed 85-94` in parallel
- **Time:** ~2 hours (bottleneck: 85-94)

**Batch 2 (Medium cohorts):**
- `non_opioid_ed 75-84` alone (or with 0-12 if it finishes early)
- **Time:** ~4.5 hours

**Batch 3 (Large cohorts - run 2 in parallel):**
- `opioid_ed 55-64` + `non_opioid_ed 65-74` in parallel
- **Time:** ~7 hours (bottleneck: both are similar size)

**Total Wall Time:** ~7-8 hours

**Resource Usage:**
- CPU: ~56 workers (28 per cohort × 2 cohorts) - **OPTIMAL**
- RAM: ~200GB (100GB per cohort × 2) - **WELL WITHIN LIMIT**

---

## Recommended Execution Plan

### Day 1: Setup & Small Cohorts (2-3 hours)

```bash
# Terminal 1: Small cohort 1
python 4b_dtw_filter/filter_protocol_events.py --cohort-name opioid_ed --age-band 0-12
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name opioid_ed --age-band 0-12
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name opioid_ed --age-band 0-12
# ... continue with Steps 5a, 5c, 5d, 6, 7, 8

# Terminal 2: Small cohort 2
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 85-94
# ... continue with all steps
```

**Expected:** Both complete in ~2 hours

### Day 1-2: Medium Cohort (4-5 hours)

```bash
# Terminal 1: Medium cohort
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 75-84
# ... continue with all steps
```

**Expected:** Completes in ~4.5 hours

### Day 2: Large Cohorts in Parallel (7-8 hours)

```bash
# Terminal 1: Large cohort 1
python 4b_dtw_filter/filter_protocol_events.py --cohort-name opioid_ed --age-band 55-64
# ... continue with all steps

# Terminal 2: Large cohort 2
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 65-74
# ... continue with all steps
```

**Expected:** Both complete in ~7-8 hours (running in parallel)

---

## Total Wall Time Summary

| Strategy | Total Time | Resource Usage | Notes |
|----------|------------|----------------|-------|
| **Sequential (one at a time)** | ~21 hours | Low (28 workers) | Safest, slowest |
| **2 cohorts in parallel** | **~10-12 hours** | **Optimal (56 workers)** | **RECOMMENDED** |
| **All 5 cohorts in parallel** | ~7-8 hours | High (140 workers) | CPU oversubscription, may be slower |

---

## Step-by-Step Parallelization Opportunities

### Steps That Can Run in Parallel (After Dependencies)

1. **After Step 5b completes:**
   - Steps 5a, 5c, 5d can run in parallel (they all depend on 5b's itemsets)
   - **Time savings:** ~1-2 hours per cohort

2. **After Step 6 completes:**
   - Steps 7 and 8 can run in parallel (they both depend on final model)
   - **Time savings:** ~1-2 hours per cohort

3. **Multiple cohorts:**
   - All cohorts can run Steps 4b and 4c in parallel
   - Small cohorts can run alongside large ones

---

## Memory Considerations

**Per-Cohort Memory Usage (estimated):**
- Step 4b/4c: ~10-20GB
- Step 5b (FP-Growth): ~50-100GB (memory-intensive)
- Step 5a (BupaR): ~20-30GB
- Step 5c (PGx): ~10-20GB
- Step 5d (DTW): ~30-50GB
- Step 6 (Final Model): ~50-100GB
- Step 7 (FFA): ~50-100GB
- Step 8 (SHAP): ~50-100GB

**Peak Memory (single cohort):** ~100-150GB

**With 2 cohorts in parallel:** ~200-300GB (well within 1TB limit)

**With all 5 cohorts in parallel:** ~500-750GB (still within limit, but CPU oversubscription)

---

## Final Recommendation

**Best Strategy: Run 2 cohorts in parallel**

1. **Start with small cohorts:** `opioid_ed 0-12` + `non_opioid_ed 85-94` (~2 hours)
2. **Then medium cohort:** `non_opioid_ed 75-84` alone (~4.5 hours)
3. **Finally large cohorts:** `opioid_ed 55-64` + `non_opioid_ed 65-74` in parallel (~7-8 hours)

**Total Wall Time: ~10-12 hours**

**Alternative (if you want to start and walk away):**
- Run all 5 cohorts in parallel
- **Total Wall Time: ~7-8 hours** (but may be slower due to CPU oversubscription)

---

## Quick Start Commands

### Run All Cohorts in Parallel (Aggressive)

```bash
# Terminal 1
python 4b_dtw_filter/filter_protocol_events.py --cohort-name opioid_ed --age-band 0-12 && \
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name opioid_ed --age-band 0-12 && \
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name opioid_ed --age-band 0-12

# Terminal 2
python 4b_dtw_filter/filter_protocol_events.py --cohort-name opioid_ed --age-band 55-64 && \
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name opioid_ed --age-band 55-64 && \
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name opioid_ed --age-band 55-64

# Terminal 3
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 65-74 && \
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name non_opioid_ed --age-band 65-74 && \
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name non_opioid_ed --age-band 65-74

# Terminal 4
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 75-84 && \
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name non_opioid_ed --age-band 75-84 && \
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name non_opioid_ed --age-band 75-84

# Terminal 5
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 85-94 && \
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name non_opioid_ed --age-band 85-94 && \
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name non_opioid_ed --age-band 85-94
```

**Or use the cohort runner notebooks configured for all cohorts!**

---

**Last Updated:** 2025-01-02  
**System:** EC2 32-core, 1TB RAM
