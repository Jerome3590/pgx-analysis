# Dashboard visualization data pattern

**Rule:** Use **JSON as much as possible** for dashboard visuals. Pipeline exports JSON → upload to S3 → Lambda loads and returns in API response → frontend renders from JSON (Plotly or chart) with fallback to image/iframe when JSON is missing.

**Exception:** **Network plots** (FP-Growth network, PGx Cohort network) are processed on **EC2** and served as pre-built HTML only. No JSON conversion for these; they stay HTML/iframe.

---

## Default pattern (JSON-first)

| Step | Responsibility |
|------|----------------|
| **Pipeline (EC2 / step 9)** | Export visualization data as JSON (e.g. Plotly figure, heatmap data, chart_data). Write alongside or instead of PNG/HTML where feasible. |
| **Upload** | Include `*.json` in the set of files synced to the dashboard S3 prefix for that visual. |
| **Lambda** | For each visualization endpoint, try to load the corresponding JSON from S3; when present, add parsed object to the API response (e.g. `heatmap_data`, `chart_data`, `trace_explorer_plot`). Continue to return image/HTML URLs for fallback. |
| **Frontend** | Prefer inline JSON from the API: if the response contains the JSON object (e.g. `data.heatmap_data`, `data.chart_data`), render with Plotly or build the chart from the data. If not, fall back to image URL or iframe URL. |

This gives consistent control and flexibility: one API call, optional inline data, and a clear fallback when JSON is not yet generated.

---

## Exception: EC2 network plots (HTML only)

These visuals are **not** converted to JSON; they are built on EC2 and served as HTML:

| Tab | Visual | Delivery |
|-----|--------|----------|
| **FP-Growth Patterns** | Network plot (drug association rules) | Pre-built HTML on S3; Lambda returns URL; frontend uses iframe or network HTML proxy. |
| **PGx Cohort** | Network topology + drug network figure pack | Pre-built HTML on S3; Lambda returns `network_topology_url`; frontend uses iframe. Figure pack uses static PNG previews with links to interactive HTML under `visualizations/cohort_pgx/figure_pack/`. |

Pipeline for these: run on EC2 → write HTML → upload to dashboard bucket → Lambda returns URL → frontend displays in iframe.

---

## Per-tab summary

| Tab | JSON (inline from API) | Fallback / other |
|-----|------------------------|------------------|
| **Feature Importance** | `heatmap_data` (aggregated FI heatmap) | `heatmap_url` (PNG) |
| **Causal Analysis** | `causal_data` (raw JSON) + `chart_data` (Lambda-filtered: causal_factors, shap_importance, whatif, feature_interactions) | Same pattern as Feature Importance: Lambda loads JSON, applies drugs/icds/cpts/whatif filters, returns chart_data for bar charts, radar, and interactions. |
| **BupaR Process Mining** | `trace_explorer_plot`, `process_matrix_drug_drug`, activity_frequency (separate endpoint) | Image/iframe URLs for sequence, trace pre, process matrix combined, frequency map |
| **DTW Trajectories** | `chart_data`, `sequence_heatmap`, `trajectory_overview_plot` | Image URLs for overview/sample when JSON missing |
| **FP-Growth Patterns** | `itemsets_data` (itemsets JSON for client Plotly) | **Network: HTML only** (EC2). Itemsets PNG/HTML URLs. |
| **PGx Cohort** | `figure_pack` URL map (optional API fallback) | **Network: HTML only** (EC2). `network_topology_url`; figure pack PNG/HTML loaded static-first. |

---

## References

- **Lambda:** `10_risk_dashboard/backend/lambda_function.py` — each `handle_visualizations_*` loads JSON from S3 when present and adds to payload.
- **Frontend:** `10_risk_dashboard/frontend/index.html` — per-tab logic prefers `data.<json_key>` then falls back to URL.
- **Pipeline:** `9_dashboard_visuals/` — scripts write JSON (e.g. `*_trace_explorer_plot.json`, `chart_data.json`, `aggregated_fi_heatmap.json`) and upload with plots.
- **S3 layout:** `10_risk_dashboard/docs/DASHBOARD_TABS.md`.
