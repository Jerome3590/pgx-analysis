# Rule Selection Methodology: Frequency vs SHAP

## Summary

**The reference FFA implementation uses ALL matched rules** for AXP computation (no filtering). **We use SHAP-based filtering** to limit rules for efficiency. We do NOT currently use frequency-based rule selection, but we could add it as an alternative or complement to SHAP filtering.

## Reference Implementation Approach

Based on the reference repository (https://github.com/ffattr/ffa):

### Rule Selection Strategy
- ✅ **Uses ALL matched rules** for each instance
- ✅ **No filtering** before AXP computation
- ✅ **Complete enumeration** of abductive explanations
- ⚠️ **Computationally expensive** (potentially 10,000+ rules per instance)

### Feature Attribution
- ✅ Computes feature importance from **frequency of feature appearance in AXPs**
- ✅ Counts how often each feature appears across all AXPs
- ✅ `importance = count / total_explanations`

## Our Current Implementation

### Rule Selection Strategy (Per Instance)

**Location:** `base_symbolic_explainer.py:_explain_instance_worker()` (lines 100-174)

We use a **4-set union approach** to limit rules:

1. **Set 1: First 100 matched rules** (order-based)
   - From ALL matched rules for this instance
   - No frequency consideration

2. **Set 2: Random 100 matched rules** (diversity)
   - From ALL matched rules for this instance
   - No frequency consideration

3. **Set 3: Top 300 SHAP-filtered rules** (importance-based)
   - Rules scored by SHAP importance of their features
   - Top 300 OR 10th percentile (whichever is larger)
   - **NOT frequency-based**

4. **Set 4: Fallback SHAP=0 rules** (completeness)
   - Up to 100 rules with SHAP = 0
   - Ensures coverage of rules not captured by Sets 1-3

**Final Rule Set:** Union of all four sets (~300-500 rules max)

### What We DON'T Do

❌ **We do NOT track rule frequency** (how often each rule matches across instances)
❌ **We do NOT select rules based on frequency** (most common rules first)
❌ **We do NOT use global rule statistics** (only instance-specific matching)

### What We DO Use Frequency For

✅ **Feature importance** - We count how often features appear in AXPs (correct FFA methodology)
✅ **Feature coverage** - We track how many instances have each feature in their AXPs

## The Difference: Rule Selection vs Feature Attribution

### Rule Selection (Before AXP Computation)
- **Reference:** Uses ALL matched rules (no filtering)
- **Ours:** Uses SHAP-filtered subset (~300-500 rules)
- **Frequency:** Reference doesn't use frequency for rule selection either

### Feature Attribution (After AXP Computation)
- **Reference:** Counts feature frequency in AXPs
- **Ours:** Counts feature frequency in AXPs ✅ **SAME METHODOLOGY**

## Could We Add Frequency-Based Rule Selection?

### Option 1: Pre-compute Rule Frequencies

**Approach:** Count how often each rule matches across the dataset, then prioritize frequent rules

**Implementation:**
```python
# Pre-compute rule frequencies across dataset
rule_frequencies = defaultdict(int)
for instance in X:
    matched_rules = explainer._satisfied_rules(instance, predicted_class)
    for rule_id in matched_rules:
        rule_frequencies[rule_id] += 1

# Sort rules by frequency
frequent_rules = sorted(rule_frequencies.items(), key=lambda x: x[1], reverse=True)
top_frequent_rules = [rule_id for rule_id, freq in frequent_rules[:300]]
```

**Pros:**
- ✅ Captures rules that match many instances (common patterns)
- ✅ Could complement SHAP filtering (frequency + importance)
- ✅ More aligned with "frequency-based" interpretation

**Cons:**
- ⚠️ Requires pre-computation pass over entire dataset
- ⚠️ Adds computational overhead
- ⚠️ May favor common but less important rules

### Option 2: Hybrid Frequency + SHAP

**Approach:** Combine frequency and SHAP scores for rule selection

**Implementation:**
```python
# Score rules by combined frequency + SHAP
def score_rule_hybrid(rule_id, frequency, shap_score):
    # Normalize frequency (0-1 scale)
    max_freq = max(rule_frequencies.values()) if rule_frequencies else 1
    normalized_freq = frequency / max_freq
    
    # Combine: 50% frequency, 50% SHAP
    combined_score = 0.5 * normalized_freq + 0.5 * shap_score
    return combined_score
```

**Pros:**
- ✅ Balances common patterns (frequency) with importance (SHAP)
- ✅ More robust rule selection
- ✅ Captures both common and important rules

**Cons:**
- ⚠️ More complex scoring logic
- ⚠️ Requires tuning weights (50/50 split)

### Option 3: Frequency-Based Set (Add as Set 5)

**Approach:** Add frequency-based rules as a fifth set in our union

**Implementation:**
```python
# Set 5: Top 100 most frequent rules (across dataset)
# Pre-compute frequencies (could cache across instances)
frequent_rules = get_top_frequent_rules(matched, top_k=100)
combined_rule_ids = list(
    set(first_rules) | 
    set(random_rules) | 
    set(shap_filtered_matched) | 
    set(fallback_rules) |
    set(frequent_rules)  # NEW: Frequency-based
)
```

**Pros:**
- ✅ Adds frequency dimension without removing existing logic
- ✅ Ensures common patterns are captured
- ✅ Minimal changes to existing code

**Cons:**
- ⚠️ Requires frequency computation
- ⚠️ May add redundant rules (already in other sets)

## Recommendation

### Current Status: ✅ **Adequate**

Our current approach (SHAP-based filtering) is **functionally correct** for FFA:
- ✅ Feature attribution is based on AXP frequency (correct methodology)
- ✅ Rule selection is for efficiency (not part of core FFA methodology)
- ✅ We capture diverse rules through Sets 1, 2, and 4

### Potential Enhancement: Add Frequency-Based Set

**Recommendation:** Add frequency-based rule selection as **Set 5** to complement existing sets:

1. **Pre-compute rule frequencies** once per dataset (cacheable)
2. **Add top 100 frequent rules** to the union
3. **Benefits:**
   - Ensures common patterns are captured
   - Complements SHAP filtering (frequency + importance)
   - Minimal performance impact (one-time computation)

**Implementation Priority:** Medium (nice-to-have, not critical)

## Key Insight

**The reference implementation doesn't use frequency for rule selection either** - it uses ALL rules. Frequency is used for **feature attribution** (counting feature appearances in AXPs), which we already do correctly.

Our SHAP-based filtering is an **optimization** to make FFA computationally tractable, not a deviation from the methodology. Adding frequency-based selection would be an **enhancement** to ensure we capture common patterns, but it's not required to match the reference implementation.

## Rule Frequency and Feature Importance

### How Rule Frequency Affects Feature Importance

**Reference Implementation:**
- Uses ALL rules for each instance
- If Rule A appears in 1000 instances → contributes to 1000 AXPs
- Features in Rule A appear in 1000 AXPs
- **Feature importance is implicitly weighted by rule frequency**

**Our Implementation:**
- Filters rules (only ~300-500 per instance)
- If Rule A is frequent but filtered out → features don't appear in AXPs
- **Missing implicit weighting from frequent but filtered rules**

**Current Status:**
- ✅ We implicitly weight by rule frequency for **included** rules
- ⚠️ We miss weighting from **filtered** frequent rules
- 💡 **Solution:** Add frequent rules to rule selection (Set 5) to ensure coverage

### Implicit vs Explicit Weighting

**Implicit Weighting (Current):**
- Frequent rules → contribute to more AXPs → features appear more often
- Works automatically for included rules
- Missing for filtered rules

**Explicit Weighting (Not Needed):**
- We don't need to explicitly weight feature importance by rule frequency
- Counting feature appearances in AXPs already captures rule frequency (for included rules)
- The issue is rule selection, not feature importance calculation

## Conclusion

- ✅ **Feature attribution:** We use frequency correctly (count feature appearances in AXPs)
- ✅ **Rule selection:** We use SHAP filtering for efficiency (reference uses all rules)
- ⚠️ **Missing:** Frequent rules that are filtered out don't contribute to feature importance
- 💡 **Enhancement opportunity:** Add frequency-based rule selection as Set 5 for better coverage

**Key Insight:** The reference implementation implicitly weights feature importance by rule frequency because frequent rules contribute to more AXPs. We do the same for included rules, but we might miss frequent rules that are filtered out. Adding frequent rules to rule selection (Set 5) will ensure we capture this implicit weighting.

The core FFA methodology (feature attribution from AXP frequency) is correctly implemented. Rule selection is an optimization layer, and adding frequency-based selection would improve coverage of common patterns and ensure proper rule frequency weighting.
