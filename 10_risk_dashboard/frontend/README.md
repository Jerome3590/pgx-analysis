# Frontend Dashboard

## Overview

The frontend dashboard is a single-page application (SPA) built with vanilla HTML, CSS, and JavaScript. It provides an interactive interface for risk assessment and PGx patient card generation.

## Files

- **`index.html`** - Main dashboard HTML file with all tabs and JavaScript
- **`assets/`** - Static assets (CSS, JavaScript, images) - currently inline in HTML

## Tabs

Live names match `index.html` and [DASHBOARD_USE_CASES.md](../docs/DASHBOARD_USE_CASES.md).

**Cohort:** Opioid ED · Polypharmacy

**Primary:** User Guide · Risk Assessment · Drugs · ICD Codes · CPT Codes · **PGx Card**

**Visualizations:** Feature Importance · Scenario Analysis (FFA/SHAP) · BupaR Process Mining · DTW Trajectories · FP-Growth Patterns · Drug Networks · PGx Cohort

1. **User Guide** — Training table (video / slides / audio per use case) plus how to use, research-question coverage, model performance
2. **Risk Assessment** — Calculate risk scores for opioid ED visits or polypharmacy (age 13–114)
3. **Drugs / ICD Codes / CPT Codes** — Code selection (ICD/CPT hidden on Polypharmacy)
4. **PGx Card** — Claims radar (**Load Cohort PGx Profile**) and **Generate PGx results**. Consumer array files (AncestryDNA, 23andMe, MyHeritage, unphased VCF) are parsed in the browser. The server uses DuckDB and Snappy Parquet to report detected variants and gene coverage, then a clinical-test referral. It does not assign a diplotype or a dose from those files. Lab `Gene,*allele` lines still use the official phenotype table.
5. **Feature Importance** — Population feature-importance heatmap by age band
6. **Scenario Analysis (FFA/SHAP)** — FFA interaction factors and SHAP importance (does not recalculate ensemble risk)
7. **BupaR Process Mining** — Process flows, activity sequences, and Drug × Drug process matrix
8. **DTW Trajectories** — Patient trajectory patterns
9. **FP-Growth Patterns** — Drug-name itemsets and association rules
10. **Drug Networks** — Interactive FI-filtered drug association graph (Cytoscape HTML)
11. **PGx Cohort** — Population gene–drug–phenotype topology

## Dependencies

- **Plotly.js** (CDN) - For interactive charts
- **Chart.js** (CDN) - For additional visualizations (if needed)

## API Integration

The frontend communicates with the Lambda backend via API Gateway:
- Base URL: Configured in `index.html` (`API_BASE` constant)
- Endpoints: See `../backend/README.md` for API documentation

## Artifact usage (manifest and S3)

The dashboard loads visualization data **static-first** from S3 using paths defined in the **manifest** (`visualizations/dashboard_visual_objects.json`). The frontend fetches the manifest once, then builds URLs for each tab’s artifacts from `s3_path` and `static_files`. API is used as fallback when static requests 404 or for risk/metadata.

The manifest is the single source of truth for **all data visual requirements**: it defines **metadata_files** (model_performance_metrics.json, cohort metadata for Documentation tab and dropdowns) and **visual_objects** (per-tab `s3_path` and `static_files`). FP-Growth includes `plots/empty_state.json` for empty-state when no rules; all tabs are fully enumerated.

| Tab | Primary artifacts | Behavior |
|-----|-------------------|----------|
| **BupaR Process Mining** | `{base}_trace_explorer_plot.json`, `{base}_pre_target_activity_frequency.json`, `{base}_process_matrix_drug_drug.json`, `{base}_activity_sequence_top.json`, etc. | **JSON + Plotly first** for Trace Explorer, Trace Explorer Pre-Target, Process Matrix (Drug × Drug), Sequences to Target, and activity frequency charts. PNG used only as fallback when JSON is missing. Manifest lists all under `visualizations/bupar/{cohort}/{age_band}/plots/`. |
| **DTW Trajectories** | `chart_data.json`, `sequence_heatmap.json`, `plots/trajectory_overview_plot.json`, `plots/dtw_trajectory_analysis_{base}.png`, `plots/dtw_trajectory_cluster_1d/3d_{base}.html` | **JSON-first:** chart_data, sequence_heatmap, and trajectory_overview_plot fetched from static (manifest); overview uses trajectory_overview_plot JSON (Plotly) when present, then overview PNG (urls[3]), then interactive HTML. Sample Trajectories visual removed. API fallback when static did not provide data. |
| **FP-Growth** | `drug_name_itemsets.json`, `plots/{base}_combined_rules_network.html`, `plots/{base}_drug_name_combined_top_itemsets.png` | Manifest drives itemsets and network URLs; empty_state.json when pipeline produced no rules. |
| **Scenario Analysis (FFA/SHAP)** | `scenario_data.json` per cohort/age_band | Static path from manifest; API fallback. |
| **Feature Importance** | `aggregated_fi_heatmap.json` / `.png` per cohort; combined heatmap | Manifest paths; API fallback. |
| **PGx Cohort** | `network_topology.html` | Manifest path under `visualizations/cohort_pgx/networks/{cohort}/{age_band}/`. |
| **Drug Networks** | Cytoscape HTML under `visualizations/cytoscape/{cohort}/{age_band}/` | Same FI-gated FP-Growth rules as the FP-Growth tab; not Plotly. |

All asset URLs use **path-style** S3 (same-origin or `https://s3.{region}.amazonaws.com/{bucket}/{prefix}/...`). See [README_dashboard_visual_artifact_paths.md](../docs/README_dashboard_visual_artifact_paths.md) and [README_dashboard_validation.md](../../README_dashboard_validation.md).

## Validating frontend updates

When changing `index.html`, use the checklist in **[README_dashboard_validation.md](../../README_dashboard_validation.md) (project root)** to ensure tabs, visual headings, BupaR copy, S3 path-style URLs, and API usage stay aligned with the path mapping and research-question artifacts.

## Deployment

The frontend is deployed as a static website on S3:
- Build: No build step required (vanilla HTML/JS)
- Deploy: Upload `index.html` to S3 bucket
- CDN: Can be served via CloudFront for better performance
