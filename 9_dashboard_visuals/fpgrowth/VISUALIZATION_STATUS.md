# FP-Growth Visualization Status

**Date:** December 9, 2025  
**Cohort:** opioid_ed, Age Band: 0-12

---

## Current Status

### ✅ Data Available for Visualization

**Location:** `4_fpgrowth_analysis/outputs/opioid_ed/combined/0_12/train/`

- ✅ **Itemsets:** 114 drug_name itemsets (support range: 0.09-0.18)
- ✅ **Rules:** 0 rules (min_confidence threshold may be too high for small cohort)
- ✅ **Metrics:** Available in JSON format
- ✅ **Encoding Maps:** Available

**Item Types Available:**
- `drug_name` - 114 itemsets
- `icd_code` - Itemsets available
- `cpt_code` - Itemsets available  
- `medical_code` - Itemsets available

---

## Documented Visualizations (Not Yet Implemented)

According to `4_fpgrowth_analysis/README.md`, the following visualizations are **optional**:

### 1. Itemset Support Distribution
- **File Pattern:** `{cohort}_{age_band}_itemset_support.png`
- **Description:** Histogram showing distribution of support values across itemsets
- **Status:** ❌ Not implemented

### 2. Rule Confidence Distribution
- **File Pattern:** `{cohort}_{age_band}_rule_confidence.png`
- **Description:** Histogram showing distribution of confidence values across rules
- **Status:** ❌ Not implemented (also no rules generated for this cohort)

### 3. Top N Frequent Itemsets
- **File Pattern:** `{cohort}_{age_band}_top_itemsets.png`
- **Description:** Bar chart showing top N itemsets by support
- **Status:** ❌ Not implemented

---

## Available Visualization Function

### Network Visualization (Implemented but Not Used)

**Function:** `py_helpers/visualization_utils.py::create_network_visualization()`

**Capabilities:**
- Creates interactive network graphs from association rules
- Uses NetworkX for graph construction
- Uses Cytoscape.js for interactive HTML rendering
- Computes node centrality
- Visualizes rule relationships (antecedents → consequents)
- Edge thickness based on support/confidence

**Status:** ✅ Function exists but not integrated into FP-Growth workflow

**Usage Example:**
```python
from py_helpers.visualization_utils import create_network_visualization
import pandas as pd
import json

# Load rules
with open('4_fpgrowth_analysis/outputs/opioid_ed/combined/0_12/train/drug_name_rules.json') as f:
    rules_data = json.load(f)

# Convert to DataFrame (if rules exist)
rules_df = pd.DataFrame(rules_data)

# Create network visualization
result = create_network_visualization(
    rules_df=rules_df,
    title="Drug Association Rules - Opioid ED, Age 0-12",
    cohort_name="opioid_ed",
    age_band="0-12",
    event_year="train"
)
```

**Note:** For cohort 0-12, no rules were generated (likely due to small cohort size and high confidence threshold).

---

## Visualization Gaps

### Missing Components

1. **No Standard Plot Script**
   - No Python script equivalent to `create_feature_importance_visualizations.py`
   - No automated plot generation after FP-Growth analysis

2. **No Plot Directory Structure**
   - `4_fpgrowth_analysis/outputs/plots/` directory doesn't exist
   - No visualization files generated

3. **Notebook Integration**
   - `cohort_fpgrowth_feature_importance.ipynb` mentions visualizations
   - May not actually execute visualization code
   - Network visualization import exists but may not be called

4. **No Cross-Platform Support**
   - No visualization script with Linux EC2 / Windows compatibility
   - No consistent visualization framework like feature importance

---

## Recommendations

### Option 1: Create Python Visualization Script (Recommended)

Create `py_helpers/create_fpgrowth_visualizations.py` similar to feature importance visualizations:

**Plots to Generate:**
1. **Itemset Support Distribution** - Histogram of support values
2. **Top N Itemsets Bar Chart** - Top itemsets by support
3. **Itemset Size Distribution** - Distribution of itemset sizes (1-item, 2-item, etc.)
4. **Support vs Itemset Size** - Scatter plot showing relationship
5. **Item Type Comparison** - Compare itemsets across drug_name, icd_code, cpt_code, medical_code

**Benefits:**
- Consistent with feature importance visualization approach
- Cross-platform compatible
- Can be called from notebooks or scripts
- Automated generation after FP-Growth analysis

### Option 2: Integrate Network Visualization

Add network visualization generation to FP-Growth workflow:

**When to Generate:**
- After rules are generated
- For cohorts with sufficient rules (> 10 rules)
- Optionally for itemsets (co-occurrence networks)

**Output:**
- Interactive HTML network graphs
- Static PNG versions for reports
- Upload to S3

### Option 3: Notebook-Based Visualizations

Enhance `cohort_fpgrowth_feature_importance.ipynb` to:
- Generate all standard plots inline
- Create network visualizations
- Export plots to `outputs/plots/` directory

---

## Current Data Summary

### Drug Name Itemsets (0-12, train)
- **Total Itemsets:** 114
- **Support Range:** 0.0909 - 0.1818
- **Example:** `['AZITHROMYCIN', 'AMOXICILLIN']` (support: 0.18)

### Association Rules
- **Total Rules:** 0
- **Reason:** Likely due to high `min_confidence` threshold (0.5) and small cohort size
- **Recommendation:** Lower threshold or use itemsets for visualization

### Metrics Available
- Unique items: 105
- Total transactions: 50
- Frequent itemsets: 114
- Processing time: 0.97 seconds

---

## Next Steps

1. **Create Visualization Script** (Priority: High)
   - Similar to `create_feature_importance_visualizations.py`
   - Generate standard plots from itemsets/rules JSON files
   - Cross-platform compatible

2. **Lower Confidence Threshold** (If Rules Needed)
   - For small cohorts like 0-12, consider lowering `min_confidence`
   - Or visualize itemsets directly instead of rules

3. **Integrate Network Visualization**
   - Add network graph generation to workflow
   - Generate for cohorts with sufficient data

4. **Update Workflow**
   - Add visualization step after FP-Growth analysis
   - Similar to feature importance workflow pattern

---

**Related Documentation:**
- [`docs/README_fpgrowth.md`](../docs/README_fpgrowth.md) - FP-Growth analysis guide
- [`py_helpers/visualization_utils.py`](../py_helpers/visualization_utils.py) - Network visualization function
- [`docs/README_feature_importance_visualization.md`](../docs/README_feature_importance_visualization.md) - Feature importance visualization pattern

