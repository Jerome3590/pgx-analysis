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
### Special logic for opioid_ed cohort

For the `opioid_ed` cohort, only events from the **current year** are used for FP-Growth rule mining (i.e., only events where `event_year == current year` are included in the transaction set for mining frequent itemsets and rules). However, the **multi-year stability filter** is still applied: any rule pattern must exist in at least 2 of the 4 years (2016–2019) to be retained. This reduces spurious rules while ensuring patterns are stable across years.

FP-Growth parameters in `cohort_fpgrowth.py` are **very permissive** since data is **pre-filtered to SHAP/FFA important features only**. Working with only the top ~500 most important codes (not all codes) allows us to use minimal thresholds without risk of spurious associations:

```python
MIN_SUPPORT = 0.01            # 1% support (find rare but meaningful patterns)
MIN_CONFIDENCE = 0.2          # 20% confidence (permissive - capture weak associations)
MIN_ITEMSET_LIFT = 1.0        # No lift filtering (accept all patterns since features pre-curated)
```

**CPT-specific (also permissive):**
```python
MIN_SUPPORT_CPT = 0.05        # 5% support for CPT codes
MIN_CONFIDENCE_CPT = 0.3      # 30% confidence for CPT rules
```

**Rationale:**
- **Pre-filtered to important features**: Working only with SHAP/FFA top ~500 codes (not all 50K+ codes), so aggressive lowering is safe
- **MIN_ITEMSET_LIFT = 1.0**: No lift filtering (lift=1.0 means independence) - we accept all patterns since features are already curated
- **Very low confidence (0.2)**: Captures even weak co-occurrence patterns between important features that may be clinically meaningful
- **Very low support (0.01)**: Finds rare but potentially important combinations (e.g., drug-drug interactions affecting 1% of patients)
- **More rules, more insights**: Since data is pre-filtered by ML feature importance, permissive thresholds maximize discovery of meaningful associations without noise

**Transaction Density:**
- Data split into 4 bins: `low` (P25), `medium` (P50), `high` (P75), `extreme` (P95+)
- Extreme density uses adjusted support: `max(MIN_SUPPORT * 0.5, 0.01)` (at least 1%)

**Rule persistence across years (business rule):**
- We use a **4-year** training window (2016–2019). Rules must not be driven by rare or single-year patterns.
- **Rule patterns must exist in at least 2 of the 4 years.** After association rules are generated, we filter out any rule whose (antecedent ∪ consequent) pattern does not appear in at least 2 distinct calendar years in the data. This avoids rules that depend on a single year or on rare `mi_person_key` patients whose pattern does not recur.
- **Applies to all cohorts** (e.g. `opioid_ed`, `non_opioid_ed`): the same logic runs for every cohort and age band when using the aggregated `train` run.
- Implemented in `filter_rules_by_year_support()`; controlled by `MIN_YEARS_FOR_RULE = 2` in `cohort_fpgrowth.py`. Applied only for the aggregated `train` run (single-year runs are not filtered by year count).

**EC2 / capacity setup:**
- **DuckDB threads:** Each item-type connection uses **3 threads** (`DUCKDB_THREADS = 3` in `cohort_fpgrowth.py`). Per (cohort, age_band), item types (e.g. drug_name, icd_code, cpt_code for opioid_ed) run in parallel; each type gets its own graph network and the user selects which to view.
- **Item-type parallelism:** Within each (cohort, age_band), item types are run with **ProcessPoolExecutor** (see `run_single_cohort_fpgrowth.py`), so each item type runs in its own process and can use a full core for Python/pandas/mlxtend (avoids GIL limits that kept utilization low with threads).
- **FP-Growth workers:** The dashboard workflow runs FP-Growth with **all (cohort, age_band) combinations in parallel** by default (max EC2 capacity). Override with `--fpgrowth-workers N` to cap parallelism, e.g. `python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --fpgrowth-workers 8`.

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

**Item types:** `drug_name` (both cohorts); `icd_code`, `cpt_code` (opioid_ed only). Each type has a **separate graph network**; the dashboard lets the user select which type to view (Drug / ICD / CPT).

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

**Item types:** Drug, ICD, CPT (per cohort; non_opioid_ed = drug only). One network per type; user selects which graph to view.

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
- [x] Item type files exist (drug_name for both; icd_code, cpt_code for opioid_ed)
- [x] Outputs under cohort/age_band only (no combined/target/train subdirs)
- [x] All file types generated (`itemsets`, `rules`, `metrics`, `encoding_map`)
- [x] Files organized in `outputs/{cohort}/{split_type}/{age_band}/{year}/`

**Visualization Files:**
- [x] PNG plots generated per item type (top itemsets, support distribution, size distribution, support vs size)
- [x] HTML network plots generated per item type (separate graph per Drug / ICD / CPT; user selects which to view)
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
