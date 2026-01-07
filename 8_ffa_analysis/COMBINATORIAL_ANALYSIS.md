# FFA Analysis: Rule Combinations vs Feature Combinations

## Summary

**The issue is NOT rule combinations** - those are well-controlled (~300-500 max).  
**The issue IS feature combinations** - exponential growth in causal/interaction analysis.

---

## Rule Combinations (NOT the problem) ✅

**Location:** `base_symbolic_explainer.py:_compute_axp()`

**How it works:**
1. Takes **first 100 matched rules** (from potentially thousands)
2. Takes **random sample of 100 rules** (for diversity)
3. Takes **top 300 SHAP-filtered rules** OR all above 10th percentile (whichever is larger)
4. **Union of all three** = ~300-500 unique rules maximum

**Why it's controlled:**
- Hard limits: 100 + 100 + 300 = 500 rules max
- SHAP filtering reduces from potentially 10,000+ rules to 300
- AXP computation runs on this limited set

**Computation:** O(n) where n ≤ 500 rules per instance

---

## Feature Combinations (THE PROBLEM) ⚠️

### 1. Single-Feature Causal Analysis

**Location:** `run_full_ffa_analysis.py:perform_causal_analysis()`

**Combinatorial Growth:**
```
For N features:
- Each feature requires 2 explanation runs (original + modified)
- Each explanation processes M samples (default: 50-100)
- Total: N × 2 × M explanation instances

Example with 100 features, 50 samples:
- 100 features × 2 runs × 50 samples = 10,000 explanation instances
- If each takes 0.1s: 1,000 seconds = 16.7 minutes
- If each takes 1s: 10,000 seconds = 2.8 hours
```

**Current Status:** ✅ **FIXED**
- Reduced sample size: 100 → 50
- Added time limit: 1 hour max
- Added progress logging

---

### 2. Multi-Feature Interaction Analysis (EXPLOSIVE) ⚠️

**Location:** `run_full_ffa_analysis.py:perform_multi_feature_causal_analysis()`

**Combinatorial Growth:**
```
For K top features, testing interactions of size 2 and 3:

2-way combinations: C(K, 2) = K × (K-1) / 2
3-way combinations: C(K, 3) = K × (K-1) × (K-2) / 6

Each combination requires 2 explanation runs (original + modified)
Each explanation processes M samples

Total combinations:
- K=10: C(10,2) + C(10,3) = 45 + 120 = 165 combinations
- K=20: C(20,2) + C(20,3) = 190 + 1,140 = 1,330 combinations
- K=30: C(30,2) + C(30,3) = 435 + 4,060 = 4,495 combinations

Total explanation runs:
- K=10: 165 × 2 = 330 runs
- K=20: 1,330 × 2 = 2,660 runs
- K=30: 4,495 × 2 = 8,990 runs

Total explanation instances (with 50 samples):
- K=10: 330 × 50 = 16,500 instances
- K=20: 2,660 × 50 = 133,000 instances
- K=30: 8,990 × 50 = 449,500 instances
```

**Time Estimates (assuming 0.1s per instance):**
- K=10: 1,650 seconds = **27 minutes**
- K=20: 13,300 seconds = **3.7 hours**
- K=30: 44,950 seconds = **12.5 hours**

**Current Status:** ✅ **PARTIALLY FIXED**
- Reduced `interaction_top_k`: 20 → 10
- Added `max_interaction_combinations`: 100 hard limit
- But still exponential growth if K increases

---

## The Real Bottleneck

**It's not the number of rules** - those are capped at ~500 per instance.  
**It's the number of features** being analyzed in causal/interaction analysis.

### Why Feature Combinations Explode:

1. **Single-feature analysis:** Linear growth (N features)
   - ✅ Manageable with limits

2. **Multi-feature interactions:** **Exponential growth** (C(N,k))
   - ⚠️ Can explode quickly
   - C(20,2) = 190
   - C(20,3) = 1,140
   - C(30,3) = 4,060

3. **Each combination requires full explanation runs**
   - Each explanation processes 50-100 samples
   - Each sample requires AXP computation over ~300-500 rules
   - **Multiplicative effect**

---

## Current Fixes Applied

### ✅ Single-Feature Causal Analysis
- Reduced `causal_sample_size`: 100 → 50
- Added `max_causal_time`: 3600s (1 hour)
- Added progress logging
- **Result:** ~50% faster, time-bounded

### ✅ Multi-Feature Interaction Analysis
- Reduced `interaction_top_k`: 20 → 10
- Added `max_interaction_combinations`: 100 hard limit
- Reduced `interaction_sample_size`: 100 → 50
- **Result:** Limits worst case, but still exponential

---

## Recommendations

### Option 1: Further Reduce Feature Counts
```python
'interaction_top_k': 5,  # Instead of 10
'causal_sample_size': 20,  # Instead of 50
```

### Option 2: Skip Multi-Feature Analysis Entirely
```python
'enable_interaction_analysis': False,  # Already default
```

### Option 3: Use Sampling for Combinations
Instead of testing all combinations, randomly sample:
```python
# Sample 50 random combinations instead of all
if len(feature_combinations) > 50:
    feature_combinations = random.sample(feature_combinations, 50)
```

### Option 4: Early Exit Based on Time
```python
if elapsed_time > max_time:
    logger.warning("Stopping interaction analysis due to time limit")
    break
```

---

## Conclusion

**Rule combinations are NOT the problem** - they're well-controlled at ~300-500 max.

**Feature combinations ARE the problem** - exponential growth in interaction analysis:
- C(10,2) + C(10,3) = 165 combinations ✅ Manageable
- C(20,2) + C(20,3) = 1,330 combinations ⚠️ Slow
- C(30,2) + C(30,3) = 4,495 combinations ❌ Very slow

**The fixes applied reduce the explosion, but the fundamental issue is the exponential nature of combinations.**

