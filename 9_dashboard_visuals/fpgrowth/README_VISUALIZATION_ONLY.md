# FP-Growth: Visualization and Exploratory Analysis Only

**Date:** 2026-01-03  
**Status:** Active - FP-Growth is for visualization/exploratory analysis, NOT for model features

---

## ⚠️ Important: Target Leakage Risk

**FP-Growth features are NOT included in the final model** due to target leakage concerns.

**Status**: ✅ **CONFIRMED** - Target leakage has been verified through code analysis. See `TARGET_LEAKAGE_ANALYSIS.md` for detailed evidence.

### Why FP-Growth Features Cause Target Leakage

1. **Pattern Mining from Combined Data**: FP-Growth mines frequent itemsets and association rules from the **combined target + control dataset**.

2. **Target Information Encoding**: Patterns discovered from the combined dataset can encode target-specific information:
   - Itemsets that are frequent in target patients but rare in controls
   - Association rules that predict target outcomes
   - Patterns that implicitly capture target class distributions

3. **Feature Creation**: When these patterns are converted to features (binary indicators, match counts, confidence scores), they carry target information from the training data into the feature set.

4. **Data Leakage**: Using these features in the model would allow the model to "see" target information through the pattern-based features, creating artificial predictive power that doesn't generalize.

### Example of Direct Leakage (Most Critical)

Consider an FP-Growth rule discovered from combined data:
- **Rule**: `{DRUG_A, DRUG_B} → {F1120}` (confidence: 0.85)
- **Feature**: `icd_code_rule_0_match` (binary indicator if patient has DRUG_A, DRUG_B, **AND F1120**)
- **Problem**: This feature **directly checks for the target code F1120**! This is perfect target leakage.

**Code Evidence**: The `match_rule` function checks for BOTH antecedents AND consequents:
```python
def match_rule(patient_items, antecedents, consequents):
    return antecedents_set.issubset(patient_items) and consequents_set.issubset(patient_items)
```

Since target codes (like F1120) are included in ICD transactions, they can appear as consequents in rules, creating direct leakage.

---

## FP-Growth Use Cases

### ✅ Recommended Uses

1. **Visualization and Exploratory Analysis**
   - Network visualizations showing co-occurrence patterns
   - Understanding which items frequently appear together
   - Identifying potential risk patterns for clinical review

2. **Risk Dashboard Integration**
   - Interactive network visualizations in the risk dashboard
   - Causal analysis visualization (combined with FFA/SHAP results)
   - Patient trajectory visualization

3. **Clinical Hypothesis Generation**
   - Identifying drug combinations of interest
   - Discovering diagnostic code patterns
   - Understanding treatment sequences

4. **Target-Only Analysis**
   - FP-Growth on target patients only (separate from controls)
   - Understanding patterns specific to target cohort
   - Clinical pathway analysis

### ❌ NOT Recommended Uses

1. **Model Features**: Do NOT use FP-Growth features in the final model
2. **Feature Engineering**: Do NOT create features from FP-Growth patterns
3. **Predictive Modeling**: Do NOT use FP-Growth patterns directly for prediction

---

## FP-Growth Outputs

### Visualization Files

All FP-Growth outputs are available for visualization and exploratory analysis:

**Network Visualizations:**
- `{cohort}_{age_band}_{item_type}_network.html`: Interactive co-occurrence networks
- `{cohort}_{age_band}_{item_type}_rules_network.html`: Interactive association rules networks

**Statistical Plots:**
- `{cohort}_{age_band}_{item_type}_top{top_n}_itemsets.png`: Top itemsets bar chart
- `{cohort}_{age_band}_{item_type}_itemset_support.png`: Support distribution
- `{cohort}_{age_band}_{item_type}_rule_confidence.png`: Rule confidence distribution

**Location:**
- Local: `10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band}/plots/`
- S3: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

### Data Files (For Analysis Only)

**Itemsets and Rules:**
- `{item_type}_itemsets.json`: Frequent itemsets (for visualization)
- `{item_type}_rules.json`: Association rules (for visualization)
- `{item_type}_metrics.json`: Itemset metrics (support, confidence, lift)

**Location:**
- Local: `10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{split_type}/{age_band}/{year}/`
- S3: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{split_type}/{age_band}/{year}/`

---

## Integration with Risk Dashboard

### Network Visualization in Dashboard

FP-Growth network visualizations can be integrated into the risk dashboard for:

1. **Causal Analysis Visualization**
   - Show FP-Growth patterns alongside FFA/SHAP results
   - Highlight patterns that align with causal features
   - Visualize patient trajectories through the network

2. **Interactive Exploration**
   - Allow users to explore co-occurrence patterns
   - Filter by centrality, support, confidence
   - Export network views for clinical review

3. **Patient-Specific Context**
   - Show which FP-Growth patterns a patient matches
   - Visualize patient's position in the network
   - Compare patient patterns to target cohort patterns

### Implementation Example

```python
# In risk dashboard
from pathlib import Path

# Load FP-Growth network visualization
fpgrowth_network_path = Path(
    "10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band}/plots/"
    "{cohort}_{age_band}_{item_type}_network.html"
)

# Embed in dashboard
dashboard.add_network_visualization(
    fpgrowth_network_path,
    title="Drug Co-occurrence Patterns",
    description="Interactive network showing frequent drug combinations"
)

# Combine with FFA/SHAP results
dashboard.add_causal_analysis_section(
    ffa_results=ffa_importance_df,
    shap_results=shap_importance_df,
    fpgrowth_network=fpgrowth_network_path,
    title="Causal Analysis with Pattern Visualization"
)
```

---

## Workflow Updates

### Final Model Building

**Before (Incorrect):**
```python
# build_final_cohort_model_features.py
fpgrowth_df = pd.read_csv(fpgrowth_features_path)
merged = merged.merge(fpgrowth_df, on="mi_person_key", how="left")
```

**After (Correct):**
```python
# build_final_cohort_model_features.py
# FP-Growth features excluded - visualization only
# Use FP-Growth network visualizations in dashboard instead
```

### Feature Engineering Checklist

- [x] Run FP-Growth analysis for visualization
- [x] Generate network visualizations
- [x] Review patterns for clinical insights
- [ ] **DO NOT** add FP-Growth features to final model
- [ ] **DO** integrate network visualizations into risk dashboard

---

## Related Documentation

- `10_risk_dashboard/visualizations/fpgrowth/README.md`: FP-Growth analysis documentation
- `6_final_model/README_STREAMLINED_WORKFLOW.md`: Final model workflow (FP-Growth excluded)
- `10_risk_dashboard/README.md`: Risk dashboard integration guide

---

**Last Updated:** 2026-01-03
