# FP-Growth Dashboard Visualizations

## Overview

FP-Growth frequent pattern mining visualizations for the risk dashboard. Items are restricted to **SHAP/FFA allowed codes** (model-salient features), and mining runs on the **target cohort** only. Per FFA and SHAP, the resulting rules **predict risk**: they surface co-occurrence among features the model ranks as important for the outcome. These visualizations complement risk predictions by showing which risk-relevant codes tend to appear together.

**Note:** We do **not** use FP-Growth for feature engineering due to target leakage concerns. We **do** use FP-Growth **with feature importance** (SHAP/FFA allowed codes) for **analysis and answering research questions** (e.g. code connections and itemsets) and dashboard display. Rules describe co-occurrence in the target population and are not contrastive (we do not compare support in target vs control).

## Purpose

FP-Growth visualizations help clinicians understand:
- **Risk-relevant co-occurrence**: Which drugs, diagnoses, and procedures (that drive model risk) frequently occur together in the target population
- **Association Rules**: Relationships between codes (antecedent → consequent) among SHAP/FFA-important features
- **Co-occurrence Networks**: Visual representation of code relationships
- **Pattern Strength**: Support, confidence, and lift metrics for patterns

## Visualization Types

### 1. Top Itemsets Bar Chart
- **File Pattern**: `{cohort}_{age_band}_{item_type}_top{top_n}_itemsets.png`
- **Description**: Bar chart showing top N frequent itemsets by support
- **Use Case**: Identify most common code combinations

### 2. Itemset Support Distribution
- **File Pattern**: `{cohort}_{age_band}_{item_type}_itemset_support.png`
- **Description**: Histogram showing distribution of itemset support values
- **Use Case**: Understand support distribution across all itemsets

### 3. Interactive Co-occurrence Network
- **File Pattern**: `{cohort}_{age_band}_{item_type}_network.html`
- **Description**: Interactive Cytoscape.js network visualization
- **Features**:
  - Nodes: Individual codes (drugs, ICDs, CPTs)
  - Edges: Co-occurrence relationships
  - Node size: Degree centrality
  - Edge width: Support value
  - Interactive filters: Centrality threshold, support threshold, max nodes
- **Use Case**: Explore code relationships interactively

### 4. Association Rules Network (Optional)
- **File Pattern**: `{cohort}_{age_band}_{item_type}_rules_network.html`
- **Description**: Interactive network showing association rules
- **Features**:
  - Directed edges: Antecedent → Consequent
  - Edge width: Rule confidence
  - Filters: Confidence threshold, support threshold
- **Use Case**: Understand predictive relationships between codes

### 5. Rule Confidence Distribution (Optional)
- **File Pattern**: `{cohort}_{age_band}_{item_type}_rule_confidence.png`
- **Description**: Histogram of rule confidence values
- **Use Case**: Understand strength of association rules

## Scripts

### Main Analysis Scripts

**`create_fpgrowth_visuals.py`** - Main orchestrator
- Creates FP-Growth visuals for specified cohort/age band
- Generates itemsets and rules JSON files
- Calls visualization generation scripts

**`create_plots.py`** - Visualization generator
- Generates PNG plots (itemsets, support distribution)
- Creates interactive HTML network visualizations
- Uploads outputs to S3

**`cohort_fpgrowth.py`** - FP-Growth analysis engine
- Implements FP-Growth algorithm
- Generates itemsets and rules
- Computes support, confidence, lift metrics

## Output Structure

### Local Outputs

```
10_risk_dashboard/visualizations/fpgrowth/outputs/
├── {cohort}/
│   ├── {split_type}/              # combined or target
│   │   └── {age_band}/
│   │       └── {year}/            # train or test
│   │           ├── {item_type}_itemsets.json
│   │           ├── {item_type}_rules.json
│   │           ├── {item_type}_metrics.json
│   │           └── {item_type}_encoding_map.json
│   └── {age_band}/
│       └── plots/                 # Visualization files (for dashboard)
│           ├── {cohort}_{age_band}_{item_type}_top20_itemsets.png
│           ├── {cohort}_{age_band}_{item_type}_itemset_support.png
│           ├── {cohort}_{age_band}_{item_type}_network.html
│           └── {cohort}_{age_band}_{item_type}_rules_network.html
```

### S3 Outputs

**S3 Location**: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

All visualization files (PNG and HTML) are uploaded to S3 for dashboard access via Lambda API.

## Usage

### Running FP-Growth Visualizations

**Basic usage:**
```bash
cd 10_risk_dashboard/visualizations/fpgrowth
python create_fpgrowth_visuals.py --cohort-name {cohort} --age-band {age_band}
```

**Example:**
```bash
python create_fpgrowth_visuals.py --cohort-name opioid_ed --age-band 25-44
```

**Generate visualizations only (if data already exists):**
```bash
python create_plots.py --cohort-name {cohort} --age-band {age_band} --item-type {item_type}
```

### Required Inputs

- **Model Events Data**: `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`
  - Event-level filtering is in Step 1b (`1b_apcd_event_filter`).

### Output Verification

After running, verify:
- [ ] JSON data files generated in `outputs/{cohort}/{split_type}/{age_band}/{year}/`
- [ ] PNG plot files generated in `outputs/{cohort}/{age_band}/plots/`
- [ ] HTML network files generated in `outputs/{cohort}/{age_band}/plots/`
- [ ] Files follow naming convention: `{cohort}_{age_band}_{item_type}_{plot_type}.{ext}`
- [ ] Files uploaded to S3 (if using orchestrator)

## Dashboard Integration

### API Endpoint

**`GET /visualizations/fpgrowth`**
- Query params: `cohort`, `age_band`, `item_type`
- Returns: S3 paths to FP-Growth visualization files (PNG and HTML)

### Frontend Display

Visualizations are displayed in the **FP-Growth Visualizations** tab of the dashboard:
- User selects cohort, age band, and item type
- Dashboard loads visualization images and HTML networks from S3
- Interactive HTML networks embedded in iframe or div
- PNG images displayed in organized panels

### Filtering

Visualizations can be filtered by user-selected codes:
- Server-side filtering in Lambda function
- Shows only rules/itemsets containing selected codes
- Updates network visualization dynamically

### Interactive Network Features

The HTML network visualizations include:
- **Zoom/Pan**: Mouse wheel zoom, drag to pan
- **Hover Tooltips**: Show code details on hover
- **Filter Controls**:
  - Node Centrality threshold (≥ 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
  - Edge Support threshold
  - Edge Confidence threshold (rules networks)
  - Max Nodes limit (20, 50, 100, 200, or All)
  - Reset Filters button
- **Export**: PNG export functionality

## Target Leakage Note

**⚠️ Important**: FP-Growth outputs are **not** added to model data (no feature engineering from itemsets/rules). Mining uses **target-only** data and **SHAP/FFA allowed codes** only, so patterns describe co-occurrence among model-important features in the target population. For details on why FP-Growth is visualization-only, see `10_risk_dashboard/visualizations/fpgrowth/README_VISUALIZATION_ONLY.md` (if present).

## Dependencies

- **Python**: `pandas`, `numpy`, `mlxtend` (for FP-Growth), `networkx`, `boto3`
- **JavaScript**: Cytoscape.js (embedded in HTML networks)
- **Input Data**: Model events parquet files from Step 4 (`4_model_data/`)

## Notes

1. **Visualization only (not feature engineering)**: FP-Growth itemsets and rules are computed for dashboard visualization. Results are not added to model data. Rules are risk-predictive in the sense that they describe co-occurrence among SHAP/FFA-important features in the target population.

2. **Population**: Mining uses **target cohort** only (no contrastive target vs control support comparison).

3. **Item types**: Separate analyses and graphs per type; user selects in dashboard:
   - `drug_name`: Drug prescriptions (both opioid_ed and non_opioid_ed)
   - `icd_code`, `cpt_code`: ICD/CPT codes (opioid_ed only)

4. **Network Size**: Large networks can be computationally expensive. Use filter controls to manage visualization size.

## Related Documentation

- **[Dashboard Visualizations Overview](../../10_risk_dashboard/visualizations/README.md)** - General visualization documentation
- **[Dashboard API Documentation](README_results_dashboard_visualizations.md)** - Complete dashboard visualization guide
- **[FP-Growth Visualization Only](../../10_risk_dashboard/visualizations/fpgrowth/README_visualization_only.md)** - Target leakage analysis and visualization-only rationale
