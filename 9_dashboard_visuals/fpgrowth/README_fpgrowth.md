# FPGrowth Analysis

**Documentation pointer**: See `docs/README_fpgrowth.md` for complete documentation. This file summarizes the expected outputs and how they are used in the downstream pipeline.

---

## Output Files Manifest

### Expected Outputs Structure

For each `(cohort, age_band, split_type)` combination, the following files should be generated.

#### Data Files (`outputs/{cohort}/{split_type}/{age_band}/{year}/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{item_type}_itemsets.json` | Frequent itemsets for each item type | Yes |
| `{item_type}_rules.json` | Association rules for each item type | Yes |
| `{item_type}_metrics.json` | Itemset metrics (support, confidence, lift) | Yes |
| `{item_type}_encoding_map.json` | Feature encoding map for itemsets | Yes |

**Item Types**: `drug_name`, `icd_code`, `cpt_code`, `medical_code`  
**Split Types**: `combined`, `target`

**Example Files**:
- `outputs/opioid_ed/combined/0_12/train/drug_name_itemsets.json`
- `outputs/opioid_ed/combined/0_12/train/drug_name_rules.json`
- `outputs/opioid_ed/combined/0_12/train/drug_name_metrics.json`
- `outputs/opioid_ed/combined/0_12/train/drug_name_encoding_map.json`
- `outputs/opioid_ed/target/0_12/train/drug_name_itemsets_target_only.json`
- `outputs/opioid_ed/target/0_12/train/drug_name_rules_target_only.json`

#### Visualization Files (`outputs/{cohort}/{age_band}/plots/`)

Directory organization uses `cohort` and `age_band` for structure (consistent with feature importance analysis).

- `outputs/{cohort}/{age_band}/plots/` – all visualization files for a cohort/age-band combination
- `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/` – uploaded plots

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{cohort}_{age_band}_{event_year}_{item_type}_top{top_n}_itemsets.png` | Top N itemsets bar chart | Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_itemset_support.png` | Itemset support distribution histogram | Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_itemset_size.png` | Itemset size distribution | Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_support_vs_size.png` | Support vs itemset size scatter plot | Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_network.html` | Interactive co-occurrence network (Cytoscape.js) with filters | Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_rules_network.html` | Interactive association rules network with filters (if rules available) | Optional |
| `{cohort}_{age_band}_{event_year}_{item_type}_rule_confidence.png` | Rule confidence distribution (if rules available) | Optional |
| `{cohort}_{age_band}_{event_year}_{item_type}_top_rules.png` | Top N rules by confidence (if rules available) | Optional |

**Item Types**: `drug_name`, `icd_code`, `cpt_code`, `medical_code`

HTML network plots use Cytoscape.js and include:
- Co-occurrence networks (itemsets)
- Association rule networks (directed)
- Filter controls (node centrality, support, edge confidence, max nodes)
- Interactive zoom/pan, tooltips, PNG export

#### Feature Engineering Files (`outputs/feature_engineering/`)

| File Pattern | Description | Required | Created By |
|--------------|-------------|----------|------------|
| `fpgrowth_features_{cohort}_{age_band}.csv` | FP-Growth features (itemset/rule indicators) | Yes | `create_fpgrowth_features.py` |
| `fpgrowth_added_features_{cohort}_{age_band}.csv` | Final merged FP-Growth features for dashboard visualization only (not added to model data) | Yes | `add_fpgrowth_features_to_model_data.py` |

S3 locations:
- `s3://pgxdatalake/gold/feature_engineering/4_fpgrowth/{cohort}/{age_band}/fpgrowth_features_{cohort}_{age_band}.csv`
- `s3://pgxdatalake/gold/feature_engineering/4_fpgrowth/{cohort}/{age_band}/fpgrowth_added_features_{cohort}_{age_band}.csv`

Format: CSV with `mi_person_key`; used by dashboard visuals only. We do not add FP-Growth or DTW features to model data.

---

## Workflow Overview

1. **Itemsets and Rules**  
   `10_risk_dashboard/visualizations/fpgrowth/cohort_fpgrowth.py` and `global_fpgrowth.py` run FP-Growth over model events (from `4a_model_data`) and generate itemsets, rules, metrics, and encoding maps, split by item type and split type (`combined`, `target`).
   
   **Note**: FP-Growth scripts automatically prefer DTW-filtered data (`model_events_no_protocols.parquet`) if available. This ensures itemsets and association rules only capture useful signals (non-protocol events), improving the quality of discovered patterns. See `4b_dtw_filter/DTW_ROLE.md` for details on DTW protocol filtering.

2. **Feature Creation**  
   `10_risk_dashboard/visualizations/fpgrowth/create_fpgrowth_features.py` converts itemsets/rules into patient-level features (NOTE: These features are NOT used in the final model due to target leakage - visualization only):
   - Binary indicators for top N itemsets and rules (`*_match` columns)
   - Count of matched itemsets/rules
   - Aggregate support/confidence metrics (e.g., `*_itemsets_max_support`, `*_rules_max_confidence`)

3. **Feature Aggregation**  
   `10_risk_dashboard/visualizations/fpgrowth/add_fpgrowth_features_to_model_data.py` writes final merged features to `fpgrowth_added_features_{cohort}_{age_band}.csv` (NOTE: These features are NOT used in the final model - visualization only).

---

## Completion Checklist

For each `(cohort, age_band)`:

- **Data Files**
  - [ ] All item types present (`drug_name`, `icd_code`, `cpt_code`, `medical_code`)
  - [ ] Both `combined` and `target` split types processed
  - [ ] `itemsets`, `rules`, `metrics`, and `encoding_map` JSONs exist

- **Visualizations**
  - [ ] PNG plots for top itemsets, support histograms, size distributions, support vs size
  - [ ] HTML co-occurrence networks for all item types
  - [ ] HTML rule networks when rules are available
  - [ ] All plots uploaded to S3 under `gold/fpgrowth/{cohort}/{age_band}/plots/` (if applicable)

- **Feature Engineering**
  - [ ] `fpgrowth_features_{cohort}_{age_band}.csv` present under `4_fpgrowth_analysis/outputs/feature_engineering/`
  - [ ] `fpgrowth_added_features_{cohort}_{age_band}.csv` present (dashboard only; not added to model data)
  - [ ] (Optional) Copies uploaded to S3 `gold/feature_engineering/4_fpgrowth/...`

