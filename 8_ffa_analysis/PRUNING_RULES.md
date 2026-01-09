# Feature Pruning Rules: Detailed Implementation Guide

This document provides detailed pruning rules for each stage of the FFA pipeline, with specific implementation guidance.

---

## A) Univariate Causal Pruning (Stage 2.5)

**Location:** After `calculate_feature_importance()`, before `perform_causal_analysis()`

**Insertion point:** Between lines 2090-2092 in `run_full_analysis_for_model()`

### Rule 1: Feature Existence & Model Relevance ✅ IMPLEMENTED

**Current implementation:**
- `get_model_features_for_causal_analysis()` filters to model-relevant features
- Includes: `item_*` (drugs/ICDs), `pgx_*`, `n_events`
- Code: Line 995 in `perform_causal_analysis()`

**Status:** ✅ Already working correctly

---

### Rule 2: Binary Prevalence Filter ⚠️ NOT IMPLEMENTED

**What it does:**
- For binary features in `remove_only` mode: Require `#(x=1) ≥ min_present_support`
- For binary features in `add_only` mode: Require `#(x=0) ≥ min_absent_support`
- Prevents testing features with insufficient intervenable instances

**Recommended defaults:**
- `min_present_support = 10` (for sample_size=50)
- `min_present_support = 30` (for sample_size=1000)
- Scale with sample size: `min_present_support = max(5, sample_size // 50)`

**Implementation:**
```python
def prune_by_prevalence(X_class, available_features, binary_intervention_mode='remove_only', min_support=10):
    """Prune features with insufficient prevalence for intervention."""
    pruned_features = []
    
    for feat_name in available_features:
        unique_vals = X_class[feat_name].unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})
        
        if is_binary:
            if binary_intervention_mode == 'remove_only':
                support = (X_class[feat_name] == 1).sum()
                if support >= min_support:
                    pruned_features.append(feat_name)
            elif binary_intervention_mode == 'add_only':
                support = (X_class[feat_name] == 0).sum()
                if support >= min_support:
                    pruned_features.append(feat_name)
            else:  # flip or both
                pruned_features.append(feat_name)  # No prevalence filter for flip
        else:
            # Continuous features: no prevalence filter
            pruned_features.append(feat_name)
    
    return pruned_features
```

**Where to add:** Before `perform_causal_analysis()` call, filter `available_features`

---

### Rule 3: AXP Coverage Filter ⚠️ NOT IMPLEMENTED

**What it does:**
- Require `coverage ≥ min_axp_coverage` (e.g., 0.01 = 1% of explanations)
- Ensures feature appears in enough AXPs to be meaningful
- Already computed in `calculate_feature_importance()` as `coverage` column

**Recommended defaults:**
- `min_axp_coverage = 0.01` (1% of explanations)
- `min_axp_coverage = 0.05` (5% of explanations) for stricter filtering

**Implementation:**
```python
def prune_by_axp_coverage(feature_importance_df, available_features, min_coverage=0.01):
    """Prune features with insufficient AXP coverage."""
    if feature_importance_df.empty:
        return available_features
    
    coverage_map = dict(zip(
        feature_importance_df['feature'],
        feature_importance_df['coverage']
    ))
    
    pruned_features = [
        f for f in available_features
        if coverage_map.get(f, 0.0) >= min_coverage
    ]
    
    return pruned_features
```

**Where to add:** After `calculate_feature_importance()`, before `perform_causal_analysis()`

---

### Rule 4: Importance-Union Filter ⚠️ PARTIALLY IMPLEMENTED

**What it does:**
- Only test features with `SHAP > 0 OR FFA > 0` (or both)
- Ensures feature has some importance signal before expensive intervention testing

**Current implementation:**
- Only checks `FFA importance > 0` (line 1001)
- Missing: SHAP union check

**Recommended implementation:**
```python
def prune_by_importance_union(available_features, shap_map, feature_importance_df, 
                             min_shap=0.0, min_ffa=0.0):
    """Prune features without SHAP OR FFA importance."""
    pruned_features = []
    
    # Build FFA importance map
    ffa_map = {}
    if not feature_importance_df.empty:
        ffa_map = dict(zip(
            feature_importance_df['feature'],
            feature_importance_df['importance']
        ))
    
    for feat_name in available_features:
        shap_importance = shap_map.get(feat_name, 0.0)
        ffa_importance = ffa_map.get(feat_name, 0.0)
        
        # Keep if SHAP > threshold OR FFA > threshold
        if shap_importance > min_shap or ffa_importance > min_ffa:
            pruned_features.append(feat_name)
    
    return pruned_features
```

**Where to add:** After `load_shap_importance()`, before `perform_causal_analysis()`

---

## B) Interaction Candidate Pruning (Stage 3)

**Location:** Inside `perform_multi_feature_causal_analysis()`, before generating combinations

### Rule 5: Co-Occurrence Support ⚠️ NOT IMPLEMENTED

**What it does:**
- For pair (A,B), require `#(A=1 & B=1) ≥ min_cooccur_support` (for `remove_only` mode)
- For pair (A,B), require `#(A=0 & B=0) ≥ min_cooccur_support` (for `add_only` mode)
- Prevents testing combinations with insufficient co-occurrence

**Recommended defaults:**
- `min_cooccur_support = 5` (for pairs)
- `min_cooccur_support = 3` (for triplets)

**Implementation:**
```python
def prune_by_cooccurrence(feature_combinations, X_sample, binary_intervention_mode='remove_only', 
                         min_cooccur=5):
    """Prune combinations with insufficient co-occurrence."""
    pruned_combos = []
    
    for combo in feature_combinations:
        # Identify binary features in combination
        binary_feats = []
        for feat_name in combo:
            unique_vals = X_sample[feat_name].unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                binary_feats.append(feat_name)
        
        if not binary_feats:
            # No binary features, no co-occurrence filter
            pruned_combos.append(combo)
            continue
        
        # Check co-occurrence based on mode
        if binary_intervention_mode == 'remove_only':
            # Require all binary features = 1
            cooccur_mask = pd.Series(True, index=X_sample.index)
            for feat_name in binary_feats:
                cooccur_mask = cooccur_mask & (X_sample[feat_name] == 1)
            cooccur_count = cooccur_mask.sum()
        elif binary_intervention_mode == 'add_only':
            # Require all binary features = 0
            cooccur_mask = pd.Series(True, index=X_sample.index)
            for feat_name in binary_feats:
                cooccur_mask = cooccur_mask & (X_sample[feat_name] == 0)
            cooccur_count = cooccur_mask.sum()
        else:  # flip
            # No co-occurrence filter for flip mode
            pruned_combos.append(combo)
            continue
        
        if cooccur_count >= min_cooccur:
            pruned_combos.append(combo)
    
    return pruned_combos
```

**Where to add:** After SHAP filtering (line 1466), before testing combinations

---

### Rule 6: Cap Combinations Per Size ⚠️ NOT IMPLEMENTED

**What it does:**
- Limit number of combinations tested per interaction size
- Even after SHAP filtering, combinations can explode
- Use top-K by combined SHAP score

**Recommended defaults:**
- `max_combinations_per_size = 1000` (for pairs)
- `max_combinations_per_size = 100` (for triplets)

**Implementation:**
```python
# After SHAP filtering and scoring (line 1451)
if len(filtered_combinations) > max_combinations_per_size:
    # Take top-K by combined SHAP score
    filtered_combinations = filtered_combinations[:max_combinations_per_size]
    logger.info(f"  Capped combinations to top {max_combinations_per_size} by SHAP score")
```

**Where to add:** After SHAP filtering (line 1466), before interaction testing loop

---

### Rule 7: Binary Intervention Consistency ✅ IMPLEMENTED

**What it does:**
- Interaction analysis uses same `binary_intervention_mode` as univariate
- Ensures consistency between univariate and interaction results

**Current implementation:**
- ✅ Line 1559: `mode = ANALYSIS_CONFIG.get('binary_intervention_mode', 'remove_only')`
- ✅ Lines 1566-1599: Applies mode consistently to all binary features in combination

**Status:** ✅ Already working correctly

---

## C) Runtime Pruning (Stage 4)

**Location:** Inside interaction testing loop, during intervention evaluation

### Rule 8: Early Stopping ✅ IMPLEMENTED

**What it does:**
- Checks first N instances for zero changes before full computation
- Skips full explanation generation if zero changes detected early
- Saves compute on obviously non-interactive pairs

**Implementation:**
- Location: Lines 1790-1841 in `perform_multi_feature_causal_analysis()`
- Checks first `early_stopping_n` instances (default: 10)
- Only applies when sample size > 2*early_stopping_n
- Falls back to full computation if early check fails
- Still records zero-effect results for completeness

**Configuration:**
- `enable_early_stopping`: True (default)
- `early_stopping_n`: 10 (default)

---

### Rule 9: CI-Based Termination ⚠️ NOT IMPLEMENTED

**What it does:**
- If bootstrap CI computed and lower bound < threshold, skip
- Requires bootstrap implementation first

**Status:** Bootstrap CI not yet implemented

---

## Implementation Status

1. ✅ **COMPLETE:** Rule 2 (Prevalence filter) - Prevents testing features with insufficient data
2. ✅ **COMPLETE:** Rule 3 (AXP coverage) - Uses already-computed coverage metric
3. ✅ **COMPLETE:** Rule 4 (Importance-union) - Completes existing partial implementation
4. ✅ **COMPLETE:** Rule 5 (Co-occurrence) - Critical for interaction efficiency
5. ✅ **COMPLETE:** Rule 6 (Cap combinations) - Prevents combinatorial explosion
6. ✅ **COMPLETE:** Rule 8 (Early stopping) - Optimization for efficiency
7. ⚠️ **PENDING:** Rule 9 (CI termination) - Requires bootstrap implementation

---

## Configuration Parameters

Add to `ANALYSIS_CONFIG`:

```python
ANALYSIS_CONFIG = {
    # ... existing config ...
    
    # Univariate pruning (Stage 2.5)
    'min_present_support': 10,  # Minimum # instances with feature=1 for removal mode
    'min_absent_support': 10,   # Minimum # instances with feature=0 for addition mode
    'min_axp_coverage': 0.01,   # Minimum AXP coverage (1% of explanations)
    'min_shap_for_causal': 0.0, # Minimum SHAP importance for causal testing
    'min_ffa_for_causal': 0.0,  # Minimum FFA importance for causal testing
    
    # Interaction pruning (Stage 3)
    'min_cooccur_support': 5,   # Minimum co-occurrence for pairs
    'min_cooccur_support_triplet': 3,  # Minimum co-occurrence for triplets
    'max_combinations_per_size': 1000,  # Cap on combinations per size
    
    # Runtime pruning (Stage 4)
    'early_stopping_n': 10,     # Check first N instances for early stopping
    'min_ci_lower_bound': 0.01, # Minimum CI lower bound (if bootstrap computed)
}
```

---

## Summary

| Rule | Stage | Status | Priority | Location |
|------|-------|--------|----------|----------|
| 1. Feature relevance | 2.5 | ✅ Implemented | - | Line 995 |
| 2. Prevalence filter | 2.5 | ✅ **COMPLETE** | HIGH | Lines 821-920 |
| 3. AXP coverage | 2.5 | ✅ **COMPLETE** | HIGH | Lines 821-920 |
| 4. Importance-union | 2.5 | ✅ **COMPLETE** | MEDIUM | Lines 821-920 |
| 5. Co-occurrence | 3 | ✅ **COMPLETE** | MEDIUM | Lines 1583-1626 |
| 6. Cap combinations | 3 | ✅ **COMPLETE** | MEDIUM | Lines 1628-1633 |
| 7. Binary mode consistency | 3 | ✅ Implemented | - | Line 1577 |
| 8. Early stopping | 4 | ✅ **COMPLETE** | LOW | Lines 1790-1841 |
| 9. CI termination | 4 | ⚠️ Pending | LOW | Requires bootstrap |
