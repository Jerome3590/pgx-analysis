# S3 contents verification (dashboard prefix)

**Bucket:** `jerome-dixon.io`  
**Prefix:** `vcu/pgx-risk-calculator`  
**Generated:** From `aws s3 ls s3://jerome-dixon.io/vcu/pgx-risk-calculator/ --recursive`

This report verifies that S3 object keys under the dashboard prefix match what the **manifest** (`visualizations/dashboard_visual_objects.json`) and **frontend** expect.

---

## 1. Top-level and metadata

| Expected (manifest / README) | S3 present | Notes |
|-----------------------------|------------|--------|
| `index.html` | ✓ | Frontend entry point |
| `metadata/model_performance_metrics.json` | ✓ | |
| `metadata/opioid_ed.json` | ✓ | |
| `metadata/non_opioid_ed.json` | ✓ | |

---

## 2. Manifest

| Expected | S3 present |
|----------|------------|
| `visualizations/dashboard_visual_objects.json` | ✓ |

---

## 3. Feature importance (manifest: per cohort + combined)

| Expected pattern | S3 present | Example key |
|------------------|------------|-------------|
| `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png` | ✓ | `.../opioid_ed/aggregated_fi_heatmap.png`, `.../non_opioid_ed/aggregated_fi_heatmap.png` |
| `visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.json` | ✓ | (same) |
| `visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png` | ✓ | |
| `visualizations/feature_importance/combined/aggregated_fi_heatmap.json` | ✓ | |

**static_files in manifest:** `aggregated_fi_heatmap.json`, `aggregated_fi_heatmap.png` (per cohort); `combined/aggregated_fi_heatmap.json`, `combined_cohorts_feature_importance_heatmap.png` (combined). **Match:** ✓

---

## 4. Causal (manifest: causal_data.json per cohort/age_band)

| Expected pattern | S3 present | Count |
|------------------|------------|--------|
| `visualizations/causal/{cohort}/{age_band}/causal_data.json` | ✓ | 16 (opioid_ed × 8 + non_opioid_ed × 8) |

Age bands on S3 use **hyphen** (e.g. `0-12`, `13-24`, `25-44`). **Match:** ✓

---

## 5. BupaR (manifest: plots under bupar/{cohort}/{age_band}/plots/)

| Expected | S3 present | Notes |
|----------|------------|--------|
| `visualizations/bupar/{cohort}/{age_band}/plots/` | ✓ | All cohort/age_band combos present |
| `{base}_activity_frequency.json` | ✓ | `{base}` = e.g. `non_opioid_ed_13_24` (underscore) |
| `{base}_pre_target_activity_frequency.json` | ✓ | |
| `{base}_post_target_activity_frequency.json` | ✓ | |
| `{base}_trace_explorer_plot.json` | ✓ | |
| `{base}_process_matrix_drug_drug.json` | ✓ | |
| `{base}_activity_sequence_top.json` | ✓ | |
| PNGs (process_matrix, overall_activity_frequency, trace_explorer_*) | ✓ | |

**static_files in manifest:** All listed JSONs; frontend may also load PNGs. **Match:** ✓

---

## 6. DTW (manifest: chart_data.json, sequence_heatmap.json at cohort/age_band)

| Expected | S3 present | Notes |
|----------|------------|--------|
| `visualizations/dtw/{cohort}/{age_band}/chart_data.json` | ✓ | All cohort/age_band |
| `visualizations/dtw/{cohort}/{age_band}/sequence_heatmap.json` | ✓ | |
| `visualizations/dtw/{cohort}/{age_band}/plots/trajectory_overview_plot.json` | ✓ | Where present |
| `visualizations/dtw/{cohort}/{age_band}/plots/*.html` | ✓ | e.g. dtw_trajectory_cluster_*_*.html |

**static_files in manifest:** `chart_data.json`, `sequence_heatmap.json`. **Match:** ✓

---

## 7. FP-Growth (manifest: root JSONs + plots; frontend expects itemsets at root)

| Expected | S3 present | Notes |
|----------|------------|--------|
| `visualizations/fpgrowth/{cohort}/{age_band}/drug_name_itemsets.json` | ✓ | **At root** (no `data/`) — matches frontend |
| `visualizations/fpgrowth/{cohort}/{age_band}/drug_name_rules.json` | ✓ | |
| `visualizations/fpgrowth/{cohort}/{age_band}/drug_name_encoding_map.json` | ✓ | |
| `visualizations/fpgrowth/{cohort}/{age_band}/drug_name_metrics.json` | ✓ | |
| `visualizations/fpgrowth/{cohort}/{age_band}/plots/{base}_combined_rules_network.html` | ✓ | `{base}` = e.g. `non_opioid_ed_13_24` |
| `visualizations/fpgrowth/{cohort}/{age_band}/plots/{base}_drug_name_combined_top_itemsets.png` | ✓ | |

**Manifest s3_path:** `vcu/pgx-risk-calculator/visualizations/fpgrowth/{cohort}/{age_band}/` (no trailing `/plots/`). **static_files:** `drug_name_itemsets.json`, `plots/{base}_combined_rules_network.html`, `plots/{base}_drug_name_combined_top_itemsets.png`. **Match:** ✓

---

## 8. Cohort PGx (manifest: network_topology.html per cohort/age_band)

| Expected | S3 present | Notes |
|----------|------------|--------|
| `visualizations/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` | ✓ | All cohort/age_band |
| Same dir: network_edges.csv, network_nodes.csv, network_stats.json, etc. | ✓ | |

**static_files in manifest:** `network_topology.html`. **Match:** ✓

---

## 9. Summary

| Category | Manifest / frontend expectation | S3 layout | Match |
|----------|--------------------------------|-----------|--------|
| Manifest | `visualizations/dashboard_visual_objects.json` | Present | ✓ |
| Metadata | `metadata/*.json` | Present | ✓ |
| Feature importance | Per-cohort + combined PNG/JSON | Present | ✓ |
| Causal | `visualizations/causal/{cohort}/{age_band}/causal_data.json` | 16 files | ✓ |
| BupaR | `visualizations/bupar/{cohort}/{age_band}/plots/{base}_*` | Present | ✓ |
| DTW | `visualizations/dtw/{cohort}/{age_band}/chart_data.json`, `sequence_heatmap.json`, `plots/*` | Present | ✓ |
| FP-Growth | `drug_name_itemsets.json` at **root** of fpgrowth/{cohort}/{age_band}/; plots under `plots/` | Present at root; no `data/` | ✓ |
| Cohort PGx | `visualizations/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` | Present | ✓ |
| Frontend | `index.html` | Present | ✓ |

**Conclusion:** S3 contents under `vcu/pgx-risk-calculator/` match the manifest and frontend expectations. FP-Growth itemsets are at the root of each cohort/age_band folder (not under `data/`), and all manifest `static_files` patterns have corresponding objects on S3.

---

## Optional: Re-run listing

To refresh this verification, run:

```bash
aws s3 ls s3://jerome-dixon.io/vcu/pgx-risk-calculator/ --recursive
```

Then compare keys to the paths in `10_risk_dashboard/visualizations/dashboard_visual_objects.json` and `10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md`.
