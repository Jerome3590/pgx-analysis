# Polypharmacy Cohort Interaction Analysis - What's Happening

## Log Analysis

### 1. **14 Workers Being Used** ✅
```
Using parallel processing with 14 workers for 6 instances
```
- **Why**: You're running two cohorts in parallel with `--n-jobs 14` each (optimal for 32 cores)
- **This is correct** - 14 workers per cohort = 28 total workers

### 2. **Only 6 Instances Being Tested** ⚠️
```
Completed 6/6 instances
```
- **Expected**: `interaction_sample_size = 50` (from config)
- **Actual**: Only 6 instances
- **Why**: After filtering for co-occurrence (features must appear together), only 6 instances remain
- **This is normal** for polypharmacy cohort where:
  - Many features are rare (drugs, ICD codes)
  - Co-occurrence filter requires features to appear together
  - With `min_cooccur_support = 5` for pairs and `min_cooccur_support_triplet = 3` for triplets, very few instances meet the criteria

### 3. **36 Million Combinations for Size 3** ⚠️ **COMBINATORIAL EXPLOSION**
```
Filtered 36361101 combinations to 36361101 based on SHAP importance > 0.0
(All features in combinations have SHAP > 0.0)
```

**The Problem:**
- **36,361,101 combinations** = C(n, 3) where n ≈ 4,000 features
- **No filtering happened** - all features have SHAP > 0, so they all pass through
- **This will take FOREVER** - even with 14 workers, testing 36M combinations × 6 instances = 218M explanation calls

**Math:**
- If n features have SHAP > 0, then C(n, 3) = n × (n-1) × (n-2) / 6
- 36,361,101 ≈ C(4,000, 3) = 4,000 × 3,999 × 3,998 / 6 ≈ 10.6 billion... wait, that's not right
- Actually: C(400, 3) = 400 × 399 × 398 / 6 ≈ 10.6 million
- C(500, 3) = 500 × 499 × 498 / 6 ≈ 20.7 million
- C(600, 3) = 600 × 599 × 598 / 6 ≈ 35.8 million ✓ **This matches!**

**So you have ~600 features with SHAP > 0**

## What Should Happen Next

The code has **pruning stages** that should reduce this:

### Stage 3: Co-occurrence Pruning
- Filters combinations where features don't co-occur enough
- For size 3: requires at least 3 instances where all 3 features are present
- **This should drastically reduce 36M → much smaller number**

### Stage 4: Capping
- `max_combinations_per_size = 1000` (from config)
- **Should cap at 1,000 combinations per size**
- But this happens AFTER co-occurrence pruning

## Why It's Taking So Long

1. **36M combinations generated** (before pruning)
2. **Co-occurrence check** must iterate through all 36M combinations
3. **Each check** requires scanning the 6-instance sample
4. **This is the bottleneck** - the pruning logic itself is slow!

## Expected Behavior

After co-occurrence pruning, you should see:
```
Filtered 36361101 combinations to [much smaller number] based on co-occurrence
```

Then after capping:
```
Capping size-3 combinations: [number] -> 1000
```

## The Real Issue

**The co-occurrence pruning is happening, but it's slow** because:
1. It must check all 36M combinations
2. Each check requires scanning the sample data
3. This is single-threaded (not parallelized)

## What You're Seeing

1. ✅ **14 workers**: Correct (running two cohorts in parallel)
2. ⚠️ **6 instances**: Normal (after co-occurrence filtering)
3. ⚠️ **36M combinations**: Expected (before pruning)
4. ⏳ **Currently**: Iterating through 36M combinations to check co-occurrence (this is slow!)

## Timeline Estimate

- **Size 2 interactions**: 23 combinations, ~7 minutes total ✅ (completed)
- **Size 3 interactions**: 
  - **Co-occurrence pruning**: Currently running (checking 36M combinations)
  - **After pruning**: Should reduce to manageable number (maybe 100-1000)
  - **Testing**: ~18 seconds per combination × remaining combinations

**Estimated time for size 3:**
- Pruning: 10-30 minutes (checking 36M combinations)
- Testing: Depends on how many pass pruning (could be 1-2 hours if many pass)

## Recommendations

### Option 1: Wait It Out
- The pruning will eventually complete
- After pruning, testing should be faster
- Total time: 1-3 hours for size 3

### Option 2: Increase SHAP Threshold
- Set `min_individual_shap_threshold > 0.0` to filter features earlier
- Reduces initial combination count before co-occurrence check
- Example: `min_individual_shap_threshold = 0.001` might reduce 600 → 200 features

### Option 3: Increase Co-occurrence Threshold
- Increase `min_cooccur_support_triplet` from 3 to 5 or 10
- Reduces combinations that pass pruning
- Faster testing phase

### Option 4: Reduce Max Combinations Cap
- Already set to 1000, but could reduce to 500 or 100
- Tests fewer combinations, faster completion

## Summary

**What's happening:**
1. ✅ Size 2 completed (23 combinations, 7 minutes)
2. ⏳ Size 3: Currently pruning 36M combinations (slow, single-threaded)
3. ⏳ After pruning: Will test remaining combinations (faster, parallelized)

**Why it's slow:**
- 36M combinations must be checked for co-occurrence
- This check is single-threaded
- After pruning, testing will be faster (parallelized with 14 workers)

**This is normal** - the pruning phase is computationally expensive but necessary to avoid testing billions of combinations.
