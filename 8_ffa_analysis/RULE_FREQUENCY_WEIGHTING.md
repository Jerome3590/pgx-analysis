# Rule Frequency Weighting for Feature Importance

## Summary

**Yes, the reference implementation implicitly weights feature importance by rule frequency.** When a rule appears frequently (matches many instances), it contributes to AXPs for all those instances, so features in that rule appear in AXPs more often. **We currently do this implicitly** for rules that pass our filter, but we might miss frequent rules that were filtered out.

## How Rule Frequency Affects Feature Importance

### Reference Implementation (All Rules)

**Process:**
1. Uses ALL matched rules for each instance
2. Computes AXP from all rules
3. Counts feature appearances in AXPs

**Implicit Weighting:**
- If Rule A appears in 1000 instances → contributes to 1000 AXPs
- Features in Rule A appear in 1000 AXPs
- Feature importance = count / total_explanations
- **Frequent rules implicitly weight features higher**

**Example:**
```
Rule A (frequent, matches 1000 instances): Contains features [X, Y]
Rule B (rare, matches 10 instances): Contains features [X, Z]

Feature X appears in:
- 1000 AXPs (from Rule A)
- 10 AXPs (from Rule B)
Total: 1010 appearances

Feature Y appears in:
- 1000 AXPs (from Rule A)
Total: 1000 appearances

Feature Z appears in:
- 10 AXPs (from Rule B)
Total: 10 appearances

Importance (normalized by total explanations):
- Feature X: 1010 / 1000 = 1.01 (most important)
- Feature Y: 1000 / 1000 = 1.00 (important)
- Feature Z: 10 / 1000 = 0.01 (less important)
```

### Our Implementation (Filtered Rules)

**Current Process:**
1. Filters rules using SHAP (only ~300-500 rules per instance)
2. Computes AXP from filtered rules
3. Counts feature appearances in AXPs

**Implicit Weighting (for included rules):**
- ✅ Frequent rules that pass filter → contribute to many AXPs
- ✅ Features in frequent rules → appear in AXPs more often
- ✅ **Implicit weighting works for included rules**

**Missing Weighting (for excluded rules):**
- ⚠️ We limit to ~300-500 rules total (union of Sets 1-4)
- ⚠️ If a frequent rule doesn't make it into any set → excluded
- ⚠️ Features in excluded frequent rules → don't appear in AXPs
- ⚠️ **Missing implicit weighting from frequent rules that don't make the cut**

**Why rules might be excluded:**
- Not in first 100 matched rules (Set 1)
- Not randomly selected (Set 2) 
- Not in top 300 SHAP or 10th percentile (Set 3)
- Not in fallback SHAP=0 rules (Set 4)
- **Even if a rule is frequent, it might not make it into any set**

**Example:**
```
Rule A (frequent, matches 1000 instances, SHAP=0.5): Contains [X, Y] → INCLUDED
Rule B (rare, matches 10 instances, SHAP=0.8): Contains [X, Z] → INCLUDED
Rule C (frequent, matches 800 instances, SHAP=0.0): Contains [Y, W] → FILTERED OUT

Feature X appears in:
- 1000 AXPs (from Rule A)
- 10 AXPs (from Rule B)
Total: 1010 appearances ✅

Feature Y appears in:
- 1000 AXPs (from Rule A)
- 0 AXPs (from Rule C - filtered out) ❌ MISSING 800!
Total: 1000 appearances (should be 1800)

Feature W appears in:
- 0 AXPs (from Rule C - filtered out) ❌ MISSING 800!
Total: 0 appearances (should be 800)
```

## The Problem

**We're missing feature importance from frequent rules that don't make it into our ~300-500 rule limit.**

Even though we use SHAP to **prioritize** rules (not filter them out), we still limit the total number of rules to ~300-500. If a rule is frequent (matches many instances) but doesn't make it into any of the 4 sets (first 100, random 100, top SHAP, or fallback), it won't contribute to AXPs, and features in that rule won't get frequency weighting. This could bias our importance scores away from frequent patterns that don't happen to be selected.

## Solution: Explicit Rule Frequency Weighting

### Option 1: Pre-compute Rule Frequencies and Weight Features

**Approach:** Count how often each rule matches across the dataset, then weight features by rule frequency when computing importance.

**Implementation:**
```python
def calculate_feature_importance_with_rule_frequency(
    df_axps: pd.DataFrame, 
    explainer: Any,
    X: pd.DataFrame,
    y: np.ndarray
) -> pd.DataFrame:
    """
    Calculate feature importance weighted by rule frequency.
    
    For each feature, count:
    1. How often it appears in AXPs (current method)
    2. How often rules containing it match instances (rule frequency weighting)
    
    Combine both for final importance.
    """
    # Step 1: Pre-compute rule frequencies across dataset
    rule_frequencies = defaultdict(int)
    for idx in range(len(X)):
        instance = X.iloc[idx].values if isinstance(X, pd.DataFrame) else X[idx]
        predicted_class = y[idx]
        matched_rules = explainer._satisfied_rules(instance, predicted_class)
        for rule_id in matched_rules:
            rule_frequencies[rule_id] += 1
    
    # Step 2: Build feature-to-rules mapping
    feature_to_rules = defaultdict(set)
    for rule_id in range(len(explainer.rule_clauses)):
        clause = explainer.rule_clauses[rule_id]
        features_in_rule = set()
        for lit in clause:
            feat_idx, _, _ = explainer.id_condition_map[lit]
            feat_name = explainer.feature_names.get(feat_idx, f"f{feat_idx}")
            features_in_rule.add(feat_name)
        for feat_name in features_in_rule:
            feature_to_rules[feat_name].add(rule_id)
    
    # Step 3: Calculate feature importance from AXPs (current method)
    feature_counts_axp = Counter()
    for axp in df_axps["axp"].dropna():
        # Parse and count features in AXP
        # ... (existing logic)
        for feature in parsed_features:
            feature_counts_axp[feature] += 1
    
    # Step 4: Calculate rule frequency contribution for each feature
    feature_rule_frequencies = {}
    for feat_name, rule_ids in feature_to_rules.items():
        # Sum frequencies of all rules containing this feature
        total_frequency = sum(rule_frequencies.get(rule_id, 0) for rule_id in rule_ids)
        feature_rule_frequencies[feat_name] = total_frequency
    
    # Step 5: Combine AXP frequency and rule frequency
    total_explanations = len(df_axps)
    max_rule_freq = max(feature_rule_frequencies.values()) if feature_rule_frequencies else 1
    
    importance_df = []
    for feat_name in set(feature_counts_axp.keys()) | set(feature_rule_frequencies.keys()):
        axp_count = feature_counts_axp.get(feat_name, 0)
        rule_freq = feature_rule_frequencies.get(feat_name, 0)
        
        # Normalize rule frequency (0-1 scale)
        normalized_rule_freq = rule_freq / max_rule_freq if max_rule_freq > 0 else 0
        
        # Combine: 70% AXP frequency, 30% rule frequency
        # (AXP frequency is more direct, but rule frequency captures filtered rules)
        combined_importance = 0.7 * (axp_count / total_explanations) + 0.3 * normalized_rule_freq
        
        importance_df.append({
            'feature': feat_name,
            'axp_count': axp_count,
            'axp_importance': axp_count / total_explanations,
            'rule_frequency': rule_freq,
            'rule_frequency_normalized': normalized_rule_freq,
            'importance': combined_importance,
            'coverage': len(feature_to_instances[feat_name]) / total_explanations
        })
    
    return pd.DataFrame(importance_df)
```

**Pros:**
- ✅ Captures importance from frequent rules even if filtered
- ✅ Balances AXP frequency (direct) with rule frequency (indirect)
- ✅ More aligned with reference implementation

**Cons:**
- ⚠️ Requires pre-computation of rule frequencies (one pass over dataset)
- ⚠️ More complex calculation
- ⚠️ Need to tune weighting (70/30 split)

### Option 2: Track Rule IDs in AXP Generation

**Approach:** Store which rules contributed to each AXP, then weight features by rule frequency.

**Implementation:**
```python
# Modify explain_dataset to return rule IDs
def explain_instance_with_rules(instance, predicted_class):
    matched_rules = explainer._satisfied_rules(instance, predicted_class)
    # ... filter rules ...
    axp_literals = explainer._compute_axp(combined_rule_ids)
    return {
        'axp': axp_literals,
        'matched_rules': matched_rules,  # NEW: Track which rules matched
        'rules_used_for_axp': combined_rule_ids  # NEW: Track which rules used
    }

# Then weight features by rule frequency
def calculate_feature_importance_weighted(df_axps_with_rules):
    # Pre-compute rule frequencies
    rule_frequencies = count_rule_frequencies(df_axps_with_rules)
    
    # For each feature, weight by frequency of rules containing it
    feature_importance = {}
    for feat_name in all_features:
        # Count AXP appearances (current)
        axp_count = count_feature_in_axps(feat_name, df_axps_with_rules)
        
        # Weight by rule frequency
        rules_with_feature = get_rules_containing_feature(feat_name)
        rule_freq_weight = sum(rule_frequencies.get(rule_id, 0) for rule_id in rules_with_feature)
        
        # Combine
        importance = 0.7 * axp_count + 0.3 * rule_freq_weight
        feature_importance[feat_name] = importance
```

**Pros:**
- ✅ More precise (knows exactly which rules contributed)
- ✅ Can weight by actual rules used, not just all rules

**Cons:**
- ⚠️ Requires modifying AXP generation to track rules
- ⚠️ More complex data structure

### Option 3: Use Rule Frequency for Rule Selection (Instead of Feature Importance)

**Approach:** Use rule frequency to select rules for AXP computation, then compute feature importance normally.

**Implementation:**
```python
# Pre-compute rule frequencies
rule_frequencies = defaultdict(int)
for instance in X:
    matched_rules = explainer._satisfied_rules(instance, predicted_class)
    for rule_id in matched_rules:
        rule_frequencies[rule_id] += 1

# Add frequent rules as Set 5
frequent_rules = sorted(rule_frequencies.items(), key=lambda x: x[1], reverse=True)
top_100_frequent = [rule_id for rule_id, freq in frequent_rules[:100]]

# Include in rule selection
combined_rule_ids = list(
    set(first_rules) | 
    set(random_rules) | 
    set(shap_filtered_matched) | 
    set(fallback_rules) |
    set(top_100_frequent)  # NEW: Frequent rules
)
```

**Pros:**
- ✅ Ensures frequent rules are included in AXP computation
- ✅ Features in frequent rules will appear in AXPs naturally
- ✅ Simpler than weighting feature importance

**Cons:**
- ⚠️ Still might miss frequent rules if they're filtered out elsewhere
- ⚠️ Doesn't explicitly weight by frequency

## Recommendation

**Recommendation: Option 3 (Add Frequent Rules to Rule Selection)**

This is the simplest and most aligned with the reference implementation:
1. Pre-compute rule frequencies across the dataset (one-time cost)
2. Add top 100 frequent rules as Set 5 in rule selection
3. Features in frequent rules will naturally appear in AXPs more often
4. Feature importance will implicitly weight by rule frequency

**Why this works:**
- Frequent rules contribute to AXPs for all instances they match
- Features in frequent rules appear in AXPs more often
- Implicit weighting happens naturally (same as reference)
- No need to modify feature importance calculation

## Current Status

**We ARE implicitly weighting by rule frequency** for rules that make it into our ~300-500 rule set. The issue is that we might exclude frequent rules that don't happen to be selected by any of the 4 sets, missing their contribution.

**Solution:** Add frequent rules to rule selection (Set 5) to ensure they're included, then feature importance will naturally weight by frequency.

## Conclusion

- ✅ **Current approach:** Implicitly weights by rule frequency for included rules
- ⚠️ **Missing:** Frequent rules that don't make it into our ~300-500 rule limit don't contribute
- 💡 **Solution:** Add frequent rules to rule selection (Set 5) to ensure coverage
- ✅ **Result:** Feature importance will naturally weight by rule frequency (same as reference)

**Key Point:** We use SHAP to **prioritize** rules (not filter them out), but we still limit the total number of rules. Adding frequent rules as Set 5 ensures frequent patterns are captured, matching the reference implementation's implicit weighting.
