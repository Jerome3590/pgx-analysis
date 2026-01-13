# SHAP Filtering Analysis: Are We Missing Rules?

## Summary

**Yes, we are potentially missing rules by filtering to SHAP values.** However, we have partial coverage through Sets 1 and 2. The reference FFA implementation does not use SHAP filtering at all.

## Current Implementation

### Rule Selection Strategy (3 Sets Union)

**Location:** `base_symbolic_explainer.py:_explain_instance_worker()` (lines 130-163)

1. **Set 1: First 100 matched rules** (from ALL matched rules)
   - ✅ Includes rules with SHAP = 0
   - ✅ Includes rules with missing SHAP values
   - ⚠️ Limited to first 100 (order-dependent)

2. **Set 2: Random 100 matched rules** (from ALL matched rules)
   - ✅ Includes rules with SHAP = 0
   - ✅ Includes rules with missing SHAP values
   - ✅ Provides diversity through random sampling

3. **Set 3: SHAP-filtered rules** (Top 300 OR 10th percentile)
   - ❌ **EXCLUDES rules with SHAP score = 0** (line 145)
   - ❌ **EXCLUDES rules with missing SHAP values** (score = 0)
   - ⚠️ Only includes rules where all features have SHAP > 0

**Final Rule Set:** Union of all three sets (line 163)

## The Problem

### Issue 1: Rules with SHAP = 0 are Excluded from Set 3

**Code:** Line 145 in `base_symbolic_explainer.py`
```python
rule_scores = [(rid, score) for rid, score in rule_scores if score > 0]
```

**Impact:**
- Rules containing features with SHAP = 0 are excluded from Set 3
- These rules might still be important for FFA (they appear in explanations)
- They're only included if they happen to be in Set 1 or Set 2

### Issue 2: Rules with Missing SHAP Values

**Code:** Lines 119-127 in `base_symbolic_explainer.py`
```python
for feat_name in features_in_rule:
    if feat_name in instance_shap_values:
        score += abs(instance_shap_values[feat_name])
    else:
        # Feature not in SHAP values - score remains 0
        pass
```

**Impact:**
- If a feature doesn't exist in SHAP values, the rule gets score = 0
- These rules are excluded from Set 3
- They might be important rules that weren't evaluated by SHAP

### Issue 3: Coverage Depends on Order and Randomness

**Scenario:** If we have 10,000 matched rules:
- Set 1: First 100 (order-dependent, might miss important rules)
- Set 2: Random 100 (random, might miss important rules)
- Set 3: Only ~200 rules with SHAP > 0 (if most rules have SHAP = 0)

**Result:** We're only guaranteed coverage of ~300-400 rules out of potentially thousands.

## Comparison with Reference Implementation

**Reference:** [https://github.com/ffattr/ffa](https://github.com/ffattr/ffa)

The reference implementation:
- ✅ Enumerates **ALL** AXPs from **ALL** rules
- ✅ Does **NOT** use SHAP filtering
- ✅ Computes feature attribution from **complete** AXP enumeration

**Our Implementation:**
- ⚠️ Filters rules using SHAP before AXP computation
- ⚠️ Only considers ~300-500 rules per instance (out of potentially thousands)
- ✅ Computes feature attribution from AXP frequency (correct methodology)

## Are Missing Rules Important?

### Arguments FOR Missing Rules Being Important:

1. **FFA Methodology:** Feature attribution should be based on ALL explanations, not just SHAP-important ones
2. **Completeness:** A rule might have SHAP = 0 but still appear frequently in AXPs (making it important for FFA)
3. **Feature Coverage:** Some features might not have SHAP values but are still part of the model's logic

### Arguments AGAINST Missing Rules Being Important:

1. **SHAP = 0:** If all features in a rule have SHAP = 0, the rule likely doesn't contribute to predictions
2. **Efficiency:** Including all rules would be computationally expensive (potentially 10,000+ rules per instance)
3. **Noise Reduction:** Filtering to SHAP > 0 might reduce noise from irrelevant rules

## Current Mitigations

### Partial Coverage Through Sets 1 and 2

- Set 1 and Set 2 include rules regardless of SHAP score
- This provides some coverage of SHAP = 0 rules
- However, coverage is limited (200 rules max) and non-deterministic

### Feature Coverage Check

**Location:** `base_symbolic_explainer.py:_filter_rules_by_shap()` (lines 534-585)

- Ensures all features with SHAP > 0 are represented in the filtered rule set
- Adds rules containing missing features if needed
- **But:** Only works for features with SHAP > 0

## Recommendations

### Option 1: Remove SHAP Filtering from Set 3 (Most Complete)

**Change:** Include ALL matched rules in Set 3, not just SHAP > 0

**Pros:**
- ✅ Complete coverage of all rules
- ✅ Matches reference implementation methodology
- ✅ No risk of missing important rules

**Cons:**
- ⚠️ Increased computation (potentially 10,000+ rules per instance)
- ⚠️ May include noisy/irrelevant rules

**Implementation:**
```python
# Instead of filtering by score > 0, include all rules
rule_scores = [(rid, score_rule_by_shap(rid)) for rid in matched]
# Sort by score but include all (even score = 0)
rule_scores.sort(key=lambda x: x[1], reverse=True)
# Take top 300 (or all if < 300)
top_300 = [rid for rid, score in rule_scores[:300]]
```

### Option 2: Add Fallback for SHAP = 0 Rules (Balanced)

**Change:** Include rules with SHAP = 0 if they're not already covered by Sets 1 and 2

**Pros:**
- ✅ Maintains efficiency (still limits to ~300-500 rules)
- ✅ Ensures coverage of SHAP = 0 rules
- ✅ Balances completeness with performance

**Cons:**
- ⚠️ More complex logic
- ⚠️ Still might miss some rules

**Implementation:**
```python
# After computing Set 3, check for missing rules
covered_rules = set(first_rules) | set(random_rules) | set(shap_filtered_matched)
missing_rules = set(matched) - covered_rules
# Add up to 100 missing rules with SHAP = 0
shap_zero_rules = [rid for rid in missing_rules if score_rule_by_shap(rid) == 0]
if shap_zero_rules:
    additional_rules = shap_zero_rules[:100]
    combined_rule_ids = list(covered_rules | set(additional_rules))
```

### Option 3: Make SHAP Filtering Optional (Flexible)

**Change:** Add configuration option to disable SHAP filtering

**Pros:**
- ✅ Allows users to choose completeness vs efficiency
- ✅ Can run both versions and compare results
- ✅ Maintains backward compatibility

**Cons:**
- ⚠️ Adds complexity to configuration
- ⚠️ Requires running analysis twice for comparison

**Implementation:**
```python
# Add to ANALYSIS_CONFIG
'use_shap_filtering': True,  # Set to False to include all rules

# In rule selection:
if ANALYSIS_CONFIG.get('use_shap_filtering', True):
    rule_scores = [(rid, score) for rid, score in rule_scores if score > 0]
else:
    # Include all rules, sorted by SHAP score
    rule_scores.sort(key=lambda x: x[1], reverse=True)
```

## Recommended Approach

**Recommendation: Option 2 (Add Fallback for SHAP = 0 Rules)**

This balances completeness with efficiency:
- Maintains current performance (limits to ~300-500 rules)
- Ensures coverage of SHAP = 0 rules through Sets 1, 2, and fallback
- Provides deterministic coverage (not just random)

## Testing Strategy

To validate if we're missing important rules:

1. **Compare Feature Importance:**
   - Run with current SHAP filtering
   - Run with Option 2 (fallback for SHAP = 0)
   - Compare feature importance rankings

2. **Check AXP Coverage:**
   - Count how many unique rules appear in AXPs
   - Compare with total matched rules
   - Identify if important rules are missing

3. **Validate Against Reference:**
   - If possible, run reference implementation on same data
   - Compare feature importance rankings
   - Identify discrepancies

## Conclusion

**Yes, we are potentially missing rules by filtering to SHAP values.** However:

- ✅ We have partial coverage through Sets 1 and 2
- ✅ Feature attribution methodology is correct (based on AXP frequency)
- ⚠️ We might miss rules that are important for FFA but have SHAP = 0

**Recommendation:** Implement Option 2 (fallback for SHAP = 0 rules) to ensure better coverage while maintaining efficiency.
