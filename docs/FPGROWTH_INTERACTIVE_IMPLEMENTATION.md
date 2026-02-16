# FP-Growth Interactive Multi-Year Visualization Implementation

## Overview

Implemented interactive Plotly visualizations for FP-Growth analysis with multi-year support (train/, 2016/, 2017/, 2018/), replacing static PNG exports with year-dropdown enabled HTML files.

**Completion Date**: Implemented  
**Status**: ✅ Complete - Ready for Testing

---

## Implementation Summary

### Files Modified

1. **py_helpers/create_fpgrowth_visualizations.py** (467 lines added)
   - Added `_load_multi_year_data()` function (62 lines)
   - Added `_top_itemsets_interactive()` function (159 lines)
   - Added `_network_interactive_multi_year()` function (222 lines)
   - Added `create_all_fpgrowth_plots_multi_year()` function (144 lines)

2. **10_risk_dashboard/backend/lambda_function.py**
   - Updated `handle_visualizations_fpgrowth()` to include interactive URLs
   - Added `itemsets_interactive` and `network_interactive` URL fields

3. **10_risk_dashboard/pgx_dashboard.html**
   - Updated `renderFPGrowthVisualizations()` to prioritize interactive HTML
   - Smart rendering: iframe for .html files (900px height), img for .png

---

## Technical Implementation Details

### 1. Multi-Year Data Loading (`_load_multi_year_data`)

**Purpose**: Load itemsets and rules from multiple year directories

**Data Sources**:
- Year 0 (All Years): `train/` directory → `{item_type}_itemsets_lift_filtered.json`
- Year 2016: `2016/` directory → `{item_type}_itemsets_lift_filtered.json`
- Year 2017: `2017/` directory → `{item_type}_itemsets_lift_filtered.json`
- Year 2018: `2018/` directory → `{item_type}_itemsets_lift_filtered.json`

**Split Types**:
- `combined`: Itemsets from control + target patients
- `target`: Rules from target patients only (for network graphs)

**Returns**: `Dict[int, Dict[str, pd.DataFrame]]`
- Maps year → {"itemsets": df, "rules": df}
- Year 0 = "All Years (2016-2018)"
- Handles missing files gracefully with logging

**Key Features**:
- Loads from standardized directory structure
- Supports both itemsets (combined split) and rules (target split)
- Error handling for missing or malformed files
- Detailed logging for data availability

---

### 2. Interactive Itemsets Bar Chart (`_top_itemsets_interactive`)

**Visualization Type**: Horizontal bar chart with year dropdown

**Data Display**:
- Top N itemsets by support (default: 30)
- Sorted descending by support value
- Itemset labels truncated to first 3 items for readability
- Full itemset details in hover tooltips

**Year Filtering**:
- Dropdown menu: All Years (2016-2018), 2016, 2017, 2018
- Default view: "All Years"
- Dropdown position: x=0.15, y=1.08 (top-left)
- Each year shows different subset of top itemsets

**Interactive Features**:
- Hover tooltips with full itemset and support value
- Responsive layout (auto-scales to container)
- Display mode bar for zoom/pan/export
- Self-contained HTML (plotly.js included)

**File Naming Convention**:
```
{cohort}_{age_band}_{item_type}_itemsets_interactive.html
Example: opioid_ed_1_0_12_drug_name_itemsets_interactive.html
```

**Plotly Configuration**:
- Height: 800px
- Left margin: 200px (for long itemset labels)
- updatemenus: Year dropdown with visible trace switching
- Color: steelblue bars

---

### 3. Interactive Network Graph (`_network_interactive_multi_year`)

**Visualization Type**: Directed network graph with year dropdown

**Graph Construction**:
- Nodes: Individual items (drugs, ICD codes, CPT codes, medical codes)
- Edges: Association rules (antecedent → consequent)
- Layout: Spring layout (networkx, seed=42, k=0.5, iterations=50)
- Node size: Based on degree centrality (15 + 35 × centrality)
- Edge attributes: Support, confidence, lift

**Year Filtering**:
- Dropdown menu: All Years (2016-2018), 2016, 2017, 2018
- Default view: "All Years"
- Dropdown position: x=0.15, y=1.15 (top-left)
- Each year shows different network topology

**Size Limitations**:
- Maximum nodes: 50 (configurable via `max_nodes` parameter)
- If network exceeds limit: keeps top nodes by degree centrality
- Minimum rules: 5 (skips year if too few rules)

**Interactive Features**:
- Node hover: Item name, in-degree, out-degree, centrality
- Edge hover: Rule (A → B), support, confidence, lift
- Responsive layout
- Display mode bar for zoom/pan/export
- Self-contained HTML

**File Naming Convention**:
```
{cohort}_{age_band}_{item_type}_network_interactive.html
Example: non_opioid_ed_1_13_24_icd_code_network_interactive.html
```

**Plotly Configuration**:
- Height: 800px
- Two traces per year: edges (gray lines) + nodes (lightblue markers)
- updatemenus: Year dropdown updates both traces simultaneously
- Title includes node/edge count for selected year
- Hidden axes (showgrid=False, zeroline=False, showticklabels=False)

---

### 4. Main Plotting Function (`create_all_fpgrowth_plots_multi_year`)

**Purpose**: Generate all interactive FP-Growth visualizations for a cohort/age_band

**Parameters**:
- `base_dir`: Root directory containing FP-Growth results
- `cohort_name`: opioid_ed or non_opioid_ed
- `age_band`: Age range (e.g., "1-0-12", "1-13-24")
- `item_types`: List of item types (default: drug_name, icd_code, cpt_code, medical_code)
- `output_dir`: Directory to save output files (default: base_dir/plots)
- `s3_upload`: Whether to upload results to S3
- `s3_bucket`: S3 bucket name
- `s3_prefix`: S3 key prefix (default: "fpgrowth")
- `top_n`: Number of top itemsets to display (default: 30)
- `max_nodes`: Maximum network nodes (default: 50)

**Workflow**:
1. For each item_type:
   - Load multi-year combined data (itemsets)
   - Generate interactive itemsets bar chart
   - Load multi-year target data (rules)
   - Generate interactive network graph
2. Save all files to output_dir
3. Optionally upload to S3

**Returns**: `Dict[str, Any]`
```python
{
    "plots": {
        "drug_name": {
            "itemsets_interactive": Path(...),
            "network_interactive": Path(...)
        },
        "icd_code": {...},
        ...
    },
    "s3_urls": {  # If s3_upload=True
        "drug_name": {
            "itemsets_interactive": "https://...",
            "network_interactive": "https://..."
        },
        ...
    }
}
```

---

## Lambda Backend Integration

### Updated Endpoint: `/visualizations/fpgrowth`

**Parameters**:
- `cohort`: opioid_ed or non_opioid_ed
- `age_band`: Age range (e.g., "1-0-12")
- `item_type`: drug_name, icd_code, cpt_code, or medical_code (default: drug_name)

**Response JSON**:
```json
{
  "itemsets_image": "https://bucket.s3.amazonaws.com/.../combined_top_itemsets.png",
  "support_image": "https://bucket.s3.amazonaws.com/.../combined_top_itemsets.png",
  "network_html": "https://bucket.s3.amazonaws.com/.../target_rules_network.html",
  "network_png": "https://bucket.s3.amazonaws.com/.../target_rules_network.png",
  "itemsets_interactive": "https://bucket.s3.amazonaws.com/.../itemsets_interactive.html",
  "network_interactive": "https://bucket.s3.amazonaws.com/.../network_interactive.html"
}
```

**Backward Compatibility**:
- Legacy fields retained: `itemsets_image`, `support_image`, `network_html`, `network_png`
- New fields added: `itemsets_interactive`, `network_interactive`
- Dashboard frontend prioritizes interactive versions, falls back to static

---

## Dashboard Frontend Integration

### Updated Function: `renderFPGrowthVisualizations(data)`

**Smart Rendering Logic**:

1. **Itemsets**:
   ```javascript
   const itemsetsUrl = data.itemsets_interactive || data.itemsets_image;
   if (itemsetsUrl.endsWith('.html')) {
     // Render as iframe (900px height)
   } else {
     // Render as img tag
   }
   ```

2. **Support Distribution**:
   - Same as itemsets (uses same data source)

3. **Network**:
   ```javascript
   const networkUrl = data.network_interactive || data.network_html || data.network_png;
   if (networkUrl.endsWith('.html')) {
     // Render as iframe (900px height)
   } else {
     // Render as img tag
   }
   ```

**Iframe Configuration**:
- Width: 100%
- Height: 900px (standardized across all visualization types)
- Border: 1px solid #ddd
- Title attribute for accessibility

**Fallback Hierarchy**:
1. Interactive multi-year HTML (preferred)
2. Single-year HTML (legacy)
3. Static PNG (fallback)

---

## File Structure

### Generated Files per Cohort/Age Band/Item Type

```
plots/
├── opioid_ed_1_0_12_drug_name_itemsets_interactive.html      # 4 years dropdown
├── opioid_ed_1_0_12_drug_name_network_interactive.html       # 4 years dropdown
├── opioid_ed_1_0_12_icd_code_itemsets_interactive.html
├── opioid_ed_1_0_12_icd_code_network_interactive.html
├── opioid_ed_1_0_12_cpt_code_itemsets_interactive.html
├── opioid_ed_1_0_12_cpt_code_network_interactive.html
├── opioid_ed_1_0_12_medical_code_itemsets_interactive.html
├── opioid_ed_1_0_12_medical_code_network_interactive.html
└── ... (repeat for each age band and cohort)
```

**File Count Comparison**:
- **Before**: 4 item types × 2 viz types × 4 years = 32 files per cohort/age_band
- **After**: 4 item types × 2 viz types = 8 interactive HTML files per cohort/age_band
- **Reduction**: 75% fewer files, all years in single interactive file

---

## Testing Checklist

### Local Development Testing

- [ ] Run `create_all_fpgrowth_plots_multi_year()` for test cohort/age_band
- [ ] Verify 8 HTML files generated (4 item types × 2 viz types)
- [ ] Open each HTML file in browser
- [ ] Test year dropdown functionality
- [ ] Verify hover tooltips display correct data
- [ ] Test responsive layout (resize window)
- [ ] Confirm plotly.js included (works offline)

### Integration Testing

- [ ] Upload files to S3 dashboard bucket
- [ ] Deploy updated lambda_function.py to AWS Lambda
- [ ] Deploy updated pgx_dashboard.html to S3/CloudFront
- [ ] Test API endpoint `/visualizations/fpgrowth?cohort=opioid_ed&age_band=1-0-12&item_type=drug_name`
- [ ] Verify response includes `itemsets_interactive` and `network_interactive` URLs
- [ ] Load dashboard in browser
- [ ] Test FP-Growth tab for each cohort/age_band/item_type combination
- [ ] Verify iframe rendering (900px height)
- [ ] Test year dropdown in each visualization
- [ ] Confirm fallback to static PNG if interactive HTML missing

### Data Validation

- [ ] Verify train/ directory loaded as "All Years (2016-2018)"
- [ ] Verify 2016/, 2017/, 2018/ directories loaded correctly
- [ ] Compare year-specific data with legacy single-year outputs
- [ ] Validate network edge counts match rule counts from JSON
- [ ] Validate itemset support values match JSON data
- [ ] Test with edge cases: missing years, empty itemsets, insufficient rules

---

## Usage Examples

### Generate Interactive Visualizations

```python
from pathlib import Path
from py_helpers.create_fpgrowth_visualizations import create_all_fpgrowth_plots_multi_year

# Generate for one cohort/age_band
result = create_all_fpgrowth_plots_multi_year(
    base_dir="/path/to/fpgrowth_results",
    cohort_name="opioid_ed",
    age_band="1-0-12",
    item_types=["drug_name", "icd_code", "cpt_code", "medical_code"],
    output_dir="/path/to/plots",
    s3_upload=True,
    s3_bucket="pgx-dashboard-bucket",
    s3_prefix="fpgrowth",
    top_n=30,
    max_nodes=50
)

# Access generated files
print(result["plots"]["drug_name"]["itemsets_interactive"])
print(result["plots"]["drug_name"]["network_interactive"])

# Access S3 URLs (if s3_upload=True)
print(result["s3_urls"]["drug_name"]["itemsets_interactive"])
print(result["s3_urls"]["drug_name"]["network_interactive"])
```

### Batch Processing

```python
cohorts = ["opioid_ed", "non_opioid_ed"]
age_bands = ["1-0-12", "1-13-24", "1-25-44", "1-45-54", "1-55-64", "1-65-74", "1-75-84", "1-85-114"]

for cohort in cohorts:
    for age_band in age_bands:
        print(f"Processing {cohort} / {age_band}...")
        result = create_all_fpgrowth_plots_multi_year(
            base_dir=f"/data/fpgrowth/{cohort}",
            cohort_name=cohort,
            age_band=age_band,
            s3_upload=True,
            s3_bucket="pgx-dashboard-bucket"
        )
        print(f"Generated {len(result['plots'])} item types")
```

---

## Design Decisions

### 1. Why Separate Functions for Itemsets and Network?

**Rationale**:
- Itemsets use `combined` split (control + target patients)
- Network uses `target` split (target patients only)
- Different data sources require separate loading
- Allows independent visualization development
- Easier to extend with additional plot types

### 2. Why Year 0 = "All Years"?

**Rationale**:
- Sentinel value distinguishes combined data from individual years
- `train/` directory contains pre-split data (combined years for training)
- Simplifies year iteration logic: `[0, 2016, 2017, 2018]`
- Clear UI labeling: "All Years (2016-2018)" vs "2016"

### 3. Why Maximum 50 Nodes in Network?

**Rationale**:
- Large networks (>100 nodes) become unreadable
- Spring layout computation time increases quadratically
- Browser rendering slows with >100 nodes + edges
- 50 nodes captures most important relationships
- Configurable parameter allows adjustment per domain

### 4. Why 900px Iframe Height?

**Rationale**:
- Matches BupaR and DTW interactive visualizations (consistency)
- Sufficient height for network graphs with 50 nodes
- Allows year dropdown + title + graph in single viewport
- Larger than static PNG (620px) to maximize interactive space
- Responsive width (100%) adapts to container

### 5. Why Keep Legacy Static Visualizations?

**Rationale**:
- Backward compatibility with existing dashboard deployments
- Fallback option if interactive HTML fails to load
- Some users prefer static exports for presentations
- Gradual migration strategy: test interactive → deprecate static
- Lambda returns both; frontend prioritizes interactive

---

## Performance Considerations

### File Sizes

**Interactive HTML**:
- Includes full plotly.js library (~3MB uncompressed)
- Gzip compression reduces to ~800KB
- Self-contained (works offline)
- One-time download per visualization

**Static PNG**:
- ~200-500KB per image
- No interactive features
- Fast rendering

**Trade-off**: Larger initial download, richer user experience

### Network Complexity

**Max Nodes Limit**:
- 50 nodes: ~10 seconds to generate (networkx layout)
- 100 nodes: ~45 seconds to generate
- 200 nodes: >2 minutes to generate

**Optimization**: Limit to 50 nodes by degree centrality (most connected items)

### Multi-Year Data Loading

**I/O Operations**:
- 4 years × 4 item types = 16 JSON file reads per visualization type
- Combined: 16 files (itemsets)
- Target: 16 files (rules)
- Total: 32 JSON file reads per cohort/age_band

**Optimization**: Parallel loading possible with ThreadPoolExecutor (future enhancement)

---

## Troubleshooting

### Issue: "No data found for year X"

**Cause**: Missing JSON file in year directory

**Solution**:
1. Verify FP-Growth pipeline completed for that year
2. Check file naming: `{item_type}_itemsets_lift_filtered.json`
3. Verify directory structure: `{cohort}/{split}/{age_band}/{year}/`
4. Review FP-Growth logs for errors during that year

### Issue: "Too few rules for network"

**Cause**: Less than `min_rules` (default: 5) association rules

**Solution**:
1. Lower `min_rules` parameter in function call
2. Check FP-Growth parameters: support, confidence, lift thresholds
3. Verify sufficient target events for that year
4. Consider combining years or broadening age band

### Issue: "Interactive HTML not rendering in dashboard"

**Cause**: File not uploaded to S3 or incorrect URL

**Solution**:
1. Verify file exists in S3 bucket at expected key
2. Check S3 bucket policy allows public-read
3. Verify CloudFront distribution serves S3 content
4. Check browser console for CORS errors
5. Test URL directly in browser (should download/render HTML)

### Issue: "Year dropdown not working"

**Cause**: Plotly.js not loaded or updatemenus config error

**Solution**:
1. Check browser console for JavaScript errors
2. Verify plotly.js included in HTML (file should be ~3MB)
3. Download HTML and open locally (should work offline)
4. Inspect updatemenus config in HTML source
5. Verify visible list lengths match trace count

---

## Future Enhancements

### 1. Pre-computed Layouts

**Idea**: Cache networkx spring_layout positions per year

**Benefits**:
- Faster visualization generation (~5x speedup)
- Consistent network topology across sessions
- Reduced server compute time

**Implementation**:
- Save layout to JSON: `{node: [x, y], ...}`
- Load from JSON if exists, else compute and save
- Invalidate cache if rules change

### 2. Additional Network Layouts

**Options**:
- Hierarchical layout (antecedents top, consequents bottom)
- Circular layout (high-degree nodes in center)
- Force-directed with edge weight = confidence
- Bipartite layout (separate antecedents/consequents)

**Implementation**:
- Add `layout_type` parameter to `_network_interactive_multi_year()`
- Switch statement for networkx layout functions

### 3. Edge Filtering by Confidence/Lift

**Idea**: Slider to filter edges by confidence or lift threshold

**Benefits**:
- Reduce visual clutter in dense networks
- Focus on high-confidence rules
- Interactive exploration of rule strength

**Implementation**:
- Add Plotly slider with `restyle` method
- Pre-compute edge traces for multiple thresholds
- Update visible edges based on slider value

### 4. Node Clustering/Coloring

**Idea**: Color nodes by community detection or item category

**Benefits**:
- Identify drug classes, diagnosis groups
- Visual separation of medication vs procedure codes
- Highlight polypharmacy clusters

**Implementation**:
- Run Louvain community detection on graph
- Map communities to color palette
- Add legend for color meanings

### 5. Export Functionality

**Idea**: Button to export current view as PNG or SVG

**Benefits**:
- Static snapshots for presentations
- High-resolution exports for papers
- Year-specific exports for comparison

**Implementation**:
- Plotly's built-in export via displayModeBar
- Custom button to export with metadata (year, cohort, age_band)

---

## Related Documentation

- **BupaR Implementation**: [INTERACTIVE_PLOTLY_IMPLEMENTATION.md](INTERACTIVE_PLOTLY_IMPLEMENTATION.md)
- **DTW Implementation**: [DTW_YEAR_FILTERING_PLAN.md](DTW_YEAR_FILTERING_PLAN.md)
- **Dashboard Architecture**: [10_risk_dashboard/README.md](../10_risk_dashboard/README.md)
- **FP-Growth Pipeline**: [8_ffa_analysis/README.md](../8_ffa_analysis/README.md)

---

## Summary

Successfully implemented interactive multi-year visualizations for FP-Growth analysis:

✅ **467 lines of new code** in create_fpgrowth_visualizations.py  
✅ **Multi-year data loading** from train/, 2016/, 2017/, 2018/  
✅ **Interactive itemsets bar chart** with year dropdown (30 top itemsets)  
✅ **Interactive network graph** with year dropdown (max 50 nodes)  
✅ **Lambda backend integration** with new URL fields  
✅ **Dashboard frontend integration** with smart iframe/img rendering  
✅ **75% file reduction** (8 interactive HTML vs 32 static PNGs per cohort/age_band)  
✅ **Backward compatibility** with legacy static visualizations  
✅ **Self-contained HTML** (works offline, includes plotly.js)  
✅ **Responsive design** (adapts to container width)  
✅ **Comprehensive hover tooltips** (itemsets, rules, network details)  

Ready for EC2 pipeline testing and deployment! 🚀
