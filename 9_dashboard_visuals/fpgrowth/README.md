# FPGrowth Analysis

📖 **Documentation**: See [`docs/README_fpgrowth.md`](/docs/README_fpgrowth.md) for complete documentation.

**Feature importance source:** FP-Growth uses the **same SHAP/FFA combined allowed codes file** as BupaR and DTW (required prerequisite; see [9_dashboard_visuals/README.md](../README.md#feature-importance-sources-for-visuals)).

---

## How to run

**Recommended (all cohorts and age bands):** Run FP-Growth as part of the dashboard workflow from the repo root. Default is all cohorts and all age bands, with one worker per (cohort, age_band) combo (capped by CPU count). No dry run.

```bash
# From repo root: sync (optional) then BupaR → DTW → FP-Growth for all combos
python 9_dashboard_visuals/run_dashboard_visuals.py

# Skip S3 sync (data already local)
python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync
```

Optional flags: `--cohort` / `--age-band` (repeatable) to limit combos; `--workers N` / `--fpgrowth-workers N` to override parallelism; `--force` to re-run even when outputs exist. See [9_dashboard_visuals/README.md](../README.md) for full workflow and prerequisites (allowed-codes files, fail-fast).

**Single (cohort, age_band):** Run the FP-Growth visuals script for one combination (itemsets + plots):

```bash
python 9_dashboard_visuals/fpgrowth/create_fpgrowth_visuals.py --cohort-name opioid_ed --age-band 0_12
```

**Batch (EC2 / direct):** Run `cohort_fpgrowth.py` directly for all cohorts and age bands. Default is no dry run; parallelism is one core per (cohort, age_band), capped by CPU count.

```bash
python 9_dashboard_visuals/fpgrowth/cohort_fpgrowth.py
```

Config in `cohort_fpgrowth.py`: `DRY_RUN = False` (full run); `COHORTS_TO_PROCESS` and `AGE_BANDS` define the set; `MAX_WORKERS` is set from cohort×age_band count and CPU count.

---

## Parameters

FP-Growth parameters in `cohort_fpgrowth.py` are **tuned for SHAP/FFA pre-filtered data**. Since we only work with important features (top 500 codes from feature importance), we use lower thresholds to generate meaningful rules without risk of spurious associations:

```python
MIN_SUPPORT = 0.03            # 3% support (finds patterns in ~3% of patients)
MIN_CONFIDENCE = 0.4          # 40% confidence for rules
MIN_ITEMSET_LIFT = 1.02       # 2% lift threshold (permissive since data pre-filtered)
```

**CPT-specific (higher to prevent memory exhaustion):**
```python
MIN_SUPPORT_CPT = 0.10        # 10% support for CPT codes
MIN_CONFIDENCE_CPT = 0.5      # 50% confidence for CPT rules
```

**Rationale:**
- **Pre-filtered data**: Working only with SHAP/FFA important features (not all codes), so lower thresholds are safe
- **MIN_ITEMSET_LIFT = 1.02**: Permissive lift filter (2% above independence) allows multi-item itemsets needed for rule generation
- **Lower = more rules**: After filtering to important features, we want to capture meaningful co-occurrence patterns even if less frequent

**Transaction Density:**
- Data split into 4 bins: `low` (P25), `medium` (P50), `high` (P75), `extreme` (P95+)
- Extreme density uses adjusted support: `max(MIN_SUPPORT * 0.5, 0.01)` (at least 1%)

---

## Output Files Manifest

### Expected Outputs Structure

For each `(cohort, age_band, split_type)` combination, the following files should be generated:

#### Data Files (`outputs/{cohort}/{split_type}/{age_band}/{year}/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{item_type}_itemsets.json` | Frequent itemsets for each item type | ✅ Yes |
| `{item_type}_rules.json` | Association rules for each item type | ✅ Yes |
| `{item_type}_metrics.json` | Itemset metrics (support, confidence, lift) | ✅ Yes |
| `{item_type}_encoding_map.json` | Feature encoding map for itemsets | ✅ Yes |

**Item Types:** `drug_name`, `icd_code`, `cpt_code`, `medical_code`

**Folder structure:** Visualization artifacts use cohort then age_band only: `outputs/{cohort}/{age_band_fname}/` (no combined/target/train subdirs).

**Example Files:**
- `outputs/opioid_ed/0_12/drug_name_itemsets.json`
- `outputs/opioid_ed/0_12/drug_name_rules.json`
- `outputs/opioid_ed/0_12/drug_name_itemsets_target_only.json`
- `outputs/opioid_ed/0_12/drug_name_rules_target_only.json`
- `outputs/opioid_ed/0_12/plots/` (PNG/HTML)

#### Visualization Files (`outputs/{cohort}/{age_band}/plots/`)

**Directory Organization:** Uses `cohort` and `age_band` for directory structure (consistent with feature importance analysis).

**Local Structure:**
- `outputs/{cohort}/{age_band}/plots/` - All visualization files for a cohort/age-band combination

**S3 Structure:**
- `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/` - Uploaded plots

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{cohort}_{age_band}_{event_year}_{item_type}_top{top_n}_itemsets.png` | Top N itemsets bar chart | ✅ Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_itemset_support.png` | Itemset support distribution histogram | ✅ Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_itemset_size.png` | Itemset size distribution | ✅ Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_support_vs_size.png` | Support vs itemset size scatter plot | ✅ Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_network.html` | Interactive co-occurrence network (Cytoscape.js) with filters | ✅ Yes |
| `{cohort}_{age_band}_{event_year}_{item_type}_rules_network.html` | Interactive association rules network with filters (if rules available) | ⚠️ Optional |
| `{cohort}_{age_band}_{event_year}_{item_type}_rule_confidence.png` | Rule confidence distribution (if rules available) | ⚠️ Optional |
| `{cohort}_{age_band}_{event_year}_{item_type}_top_rules.png` | Top N rules by confidence (if rules available) | ⚠️ Optional |

**Item Types:** `drug_name`, `icd_code`, `cpt_code`, `medical_code`

**HTML Network Plots:**
- Interactive network visualizations using Cytoscape.js
- **Co-occurrence networks**: Show which items frequently appear together in itemsets
- **Association rules networks**: Show directed relationships (antecedent → consequent)
- **Interactive Features**:
  - Zoom, pan, hover tooltips, PNG export
  - **Filter Controls**:
    - Node Centrality threshold (≥ 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
    - Node Support threshold (co-occurrence networks only)
    - Edge Support threshold
    - Edge Confidence threshold (rules networks only)
    - Max Nodes limit (20, 50, 100, 200, or All)
    - Reset Filters button
- **Visual Encoding**:
  - Node size represents degree centrality
  - Edge width represents support/confidence
  - Filters dynamically update the network view for manageable exploration

**Note:** All visualization files are generated by `py_helpers/create_fpgrowth_visualizations.py` and uploaded to S3 at `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

### Completion Checklist

For each cohort/age-band combination:

**Data Files:**
- [x] All item type files exist (`drug_name`, `icd_code`, `cpt_code`, `medical_code`)
- [x] Outputs under cohort/age_band only (no combined/target/train subdirs)
- [x] All file types generated (`itemsets`, `rules`, `metrics`, `encoding_map`)
- [x] Files organized in `outputs/{cohort}/{split_type}/{age_band}/{year}/`

**Visualization Files:**
- [x] All PNG plots generated for each item type (top itemsets, support distribution, size distribution, support vs size)
- [x] HTML network plots generated (co-occurrence networks for all item types)
- [x] HTML rules network plots generated (if rules available)
- [x] All plots organized in `outputs/{cohort}/{age_band}/plots/`
- [x] Network plots include filter controls (node centrality, edge support, max nodes)
- [x] Files uploaded to S3 at `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/` (if applicable)

**Verification:**
- [x] Network plots are interactive and manageable for large datasets
- [x] All plots follow naming convention: `{cohort}_{age_band}_{event_year}_{item_type}_{plot_type}.{ext}`
- [x] Directory structure uses cohort/age_band organization (consistent with feature importance)

**Outputs only:** This step produces itemsets/rules JSON and plots under `outputs/{cohort}/{age_band_fname}/` (and `.../plots/`). Visualization artifacts use cohort then age_band only. We do **not** use or create `outputs/feature_engineering/`; FP-Growth features are not added to model data.

---
