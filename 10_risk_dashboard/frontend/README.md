# Frontend Dashboard

## Overview

The frontend dashboard is a single-page application (SPA) built with vanilla HTML, CSS, and JavaScript. It provides an interactive interface for risk assessment and PGx patient card generation.

## Files

- **`index.html`** - Main dashboard HTML file with all tabs and JavaScript
- **`assets/`** - Static assets (CSS, JavaScript, images) - currently inline in HTML

## Tabs

1. **Risk Assessment** - Calculate risk scores for opioid ED visits or polypharmacy
2. **Causal Analysis** - Explore FFA causal factors and SHAP importance
3. **DTW Trajectories** - View patient trajectory patterns
4. **FP-Growth Patterns** - Explore drug-name itemsets and association rules (drugs only)
5. **BupaR Process Mining** - View process flows, activity sequences, and Drug × Drug process matrix
6. **PGx Patient Card** - Generate pharmacogenomic cards

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
| **DTW Trajectories** | `chart_data.json`, `sequence_heatmap.json`, `plots/trajectory_overview_plot.json` | Static URLs from manifest; overview/sample panels use trajectory_overview_plot JSON for Plotly when present. API returns 200 with message/empty when objects are missing on S3. |
| **FP-Growth** | `drug_name_itemsets.json`, `plots/{base}_combined_rules_network.html`, `plots/{base}_drug_name_combined_top_itemsets.png` | Manifest drives itemsets and network URLs; empty_state.json when pipeline produced no rules. |
| **Causal Analysis** | `causal_data.json` per cohort/age_band | Static path from manifest; API fallback. |
| **Feature Importance** | `aggregated_fi_heatmap.json` / `.png` per cohort; combined heatmap | Manifest paths; API fallback. |
| **PGx Cohort** | `network_topology.html` | Manifest path under `visualizations/cohort_pgx/networks/{cohort}/{age_band}/`. |

All asset URLs use **path-style** S3 (same-origin or `https://s3.{region}.amazonaws.com/{bucket}/{prefix}/...`). See [README_dashboard_visual_artifact_paths.md](../docs/README_dashboard_visual_artifact_paths.md) and [README_dashboard_validation.md](../../README_dashboard_validation.md).

## Validating frontend updates

When changing `index.html`, use the checklist in **[README_dashboard_validation.md](../../README_dashboard_validation.md) (project root)** to ensure tabs, visual headings, BupaR copy, S3 path-style URLs, and API usage stay aligned with the path mapping and research-question artifacts.

## Deployment

The frontend is deployed as a static website on S3:
- Build: No build step required (vanilla HTML/JS)
- Deploy: Upload `index.html` to S3 bucket
- CDN: Can be served via CloudFront for better performance
