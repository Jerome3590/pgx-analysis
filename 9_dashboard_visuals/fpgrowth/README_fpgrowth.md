# FPGrowth Analysis

**Documentation pointer**: See `docs/README_fpgrowth.md` for complete documentation. This file summarizes the expected outputs and how they are used in the downstream pipeline.

---

## Output Files Manifest

### Expected Outputs Structure

For each `(cohort, age_band)` combination, visualization artifacts use **cohort then age_band only**: `outputs/{cohort}/{age_band_fname}/`.

#### Data Files (`outputs/{cohort}/{age_band_fname}/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{item_type}_itemsets.json` | Frequent itemsets (combined) | Yes |
| `{item_type}_rules.json` | Association rules (combined) | Yes |
| `{item_type}_itemsets_target_only.json` | Target-only itemsets | Yes |
| `{item_type}_rules_target_only.json` | Target-only rules | Yes |
| `{item_type}_metrics.json`, `{item_type}_encoding_map.json` | Metrics and encoding | Yes |

**Item Types**: `drug_name`, `icd_code`, `cpt_code`, `medical_code`

**Example Files**:
- `outputs/opioid_ed/0_12/drug_name_itemsets.json`
- `outputs/opioid_ed/0_12/drug_name_itemsets_target_only.json`
- `outputs/opioid_ed/0_12/drug_name_rules_target_only.json`

#### Visualization Files (`outputs/{cohort}/{age_band_fname}/plots/`)

- `outputs/{cohort}/{age_band_fname}/plots/` – all visualization files for a cohort/age-band combination
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

**Outputs:** Itemsets/rules JSON and plots under `outputs/{cohort}/{age_band_fname}/` and `.../plots/`. Visualization artifacts use cohort then age_band only. We do **not** use or create `outputs/feature_engineering/`; FP-Growth features are not added to model data.

---

## Workflow Overview

1. **Itemsets and Rules**  
   `9_dashboard_visuals/fpgrowth/cohort_fpgrowth.py` and `global_fpgrowth.py` run FP-Growth over model events (from `4_model_data` / `4a_model_data`) and generate itemsets, rules, metrics, and encoding maps for all item types. Outputs go to `10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band_fname}/` (visualization artifacts = cohort then age_band only).
   
   **Note**: FP-Growth scripts automatically prefer DTW-filtered data (`model_events_no_protocols.parquet`) if available. This ensures itemsets and association rules only capture useful signals (non-protocol events), improving the quality of discovered patterns. See `4b_dtw_filter/DTW_ROLE.md` for details on DTW protocol filtering.

2. **Plots**  
   `create_fpgrowth_visuals.py` / `create_plots.py` build PNG and HTML from itemsets/rules JSON and write under `outputs/{cohort}/{age_band}/plots/`. No feature-engineering CSVs are created.

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

- **No feature_engineering:** We do not create or use `outputs/feature_engineering/` for FP-Growth.

