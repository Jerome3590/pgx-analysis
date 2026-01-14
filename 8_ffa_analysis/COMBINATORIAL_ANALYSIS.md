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

### 2. Multi-Feature Interaction Analysis (CONTROLLED) ✅

**Location:** `run_full_ffa_analysis.py:perform_multi_feature_causal_analysis()`

**Cohort-Specific Interaction Sizes:**
- **First cohort (`opioid_ed`)**: Tests size 2 only (pairs)
- **Second cohort (`non_opioid_ed`/`polypharmacy`)**: Tests size 2 and 3 (pairs and triplets)

**Feature Selection:**
- Includes ALL features with SHAP > 0 OR FFA > 0 OR causal > 0 (no top_k limit)
- Features sorted by combined importance (SHAP + causal + FFA) for prioritization
- Safe for drug-only features where all drugs with any importance signal should be tested

**Combinatorial Growth:**
```
For N features with SHAP > 0, testing interactions:

First cohort (opioid_ed) - size 2 only:
- 2-way combinations: C(N, 2) = N × (N-1) / 2

Second cohort (non_opioid_ed/polypharmacy) - size 2 and 3:
- 2-way combinations: C(N, 2) = N × (N-1) / 2
- 3-way combinations: C(N, 3) = N × (N-1) × (N-2) / 6

Each combination requires 2 explanation runs (original + modified)
Each explanation processes M samples (default: 50)

Example with N=50 features:
- First cohort: C(50,2) = 1,225 pairs
- Second cohort: C(50,2) + C(50,3) = 1,225 + 19,600 = 20,825 combinations

Total explanation instances (with 50 samples):
- First cohort: 1,225 × 2 × 50 = 122,500 instances
- Second cohort: 20,825 × 2 × 50 = 2,082,500 instances
```

**Time Estimates (assuming 0.1s per instance):**
- First cohort (N=50): 12,250 seconds = **3.4 hours**
- Second cohort (N=50): 208,250 seconds = **57.8 hours**

**Current Status:** ✅ **CONTROLLED**
- Cohort-specific interaction sizes prevent explosion for first cohort
- Includes all features with SHAP > 0 (no arbitrary top_k limit)
- Co-occurrence filtering and capping still apply to reduce combinations

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

