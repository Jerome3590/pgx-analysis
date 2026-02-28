# EC2 visualization save paths (summary)

Paths are under repo root on EC2 (e.g. `/home/pgx3874/pgx-analysis`). **Age bands:** EC2 uses **underscore** (e.g. `25_44`); S3 uses **hyphen** (e.g. `25-44`).

---

## Top level: `10_risk_dashboard/visualizations/`

```
visualizations/
├── PLOTS_VS_CODE_ANOMALIES.md
├── README.md
├── bupar/
├── causal/
├── cohort_pgx/
├── dashboard_visual_objects.json
├── dtw/
└── fpgrowth/
```

---

## BupaR

**Base path:** `10_risk_dashboard/visualizations/bupar/`

- **Cohort-level JSONs** (in `bupar/` root):  
  `allowed_codes_shap_ffa_{cohort}_{age_band}.json`  
  (e.g. `allowed_codes_shap_ffa_non_opioid_ed_25_44.json`, `allowed_codes_shap_ffa_opioid_ed_25_44.json`)

- **Cohorts:** `non_opioid_ed`, `opioid_ed`

- **Age bands:** `13_24`, `25_44`, `45_54`, `55_64`, `65_74`, `75_84`, `85_114` (and `0_12` if present)

- **Per cohort/age_band:** `{cohort}/{age_band}/` contains:
  - **features/** — CSVs (e.g. `{cohort}_{age_band}_train_target_pre_hcg_patient_features_bupar.csv`, `*_traces_bupar.csv`)
  - **plots/** — all dashboard assets for that cohort/age_band

**Plots dir** (`.../bupar/{cohort}/{age_band}/plots/`) — `{base}` = `{cohort}_{age_band}` (e.g. `non_opioid_ed_25_44`):

| File pattern | Purpose |
|--------------|---------|
| `{base}_activity_frequency.json` | Overall activity frequency (static-first) |
| `{base}_pre_target_activity_frequency.json` | Pre-target activity frequency |
| `{base}_post_target_activity_frequency.json` | Post-target activity frequency |
| `{base}_overall_activity_frequency.png` | Overall frequency image |
| `{base}_post_hcg_activity_frequency.png` | Post-target image |
| `{base}_activity_sequence_top.json`, `.png` | Sequence top |
| `{base}_process_matrix.png` | Process matrix |
| `{base}_process_matrix_drug_drug.json`, `.png` | Drug×drug matrix |
| `{base}_trace_explorer_plot.json` | Trace explorer data |
| `{base}_trace_explorer_pre_hcg.png`, `_post_hcg.png` | Trace explorer images |
| `{base}_Rplots.pdf` | R summary (optional) |

**Example full path:**  
`10_risk_dashboard/visualizations/bupar/non_opioid_ed/25_44/plots/non_opioid_ed_25_44_activity_frequency.json`

---

## Other visualization types (EC2 write locations)

| Viz | EC2 path (underscore age_band) | Key artifacts |
|-----|--------------------------------|----------------|
| **Causal** | `visualizations/causal/{cohort}/{age_band_fname}/` | `dashboard_data.json` (uploaded to S3 as `causal_data.json`) |
| **DTW** | `visualizations/dtw/{cohort}/{age_band_fname}/` (and `dtw/feature_engineering/` for CSVs) | `chart_data.json`, `sequence_heatmap.json`, `plots/*.png` |
| **FP-Growth** | `visualizations/fpgrowth/{cohort}/{age_band_fname}/` | `plots/*`, `data/*` (e.g. `drug_name_itemsets.json`, `empty_state.json`) |
| **Cohort PGx** | `visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/` | `network_topology.html`, supporting CSV/JSON |

---

## Sync to S3 (Step 6)

- **BupaR:** `visualizations/bupar/` (per cohort/age_band/plots/) → S3 `visualizations/bupar/{cohort}/{age_band}/plots/` (age_band → hyphen).
- **DTW:** `visualizations/dtw/` (per cohort/age_band; skip `feature_engineering/`) → S3 `visualizations/dtw/{cohort}/{age_band}/`.
- **FP-Growth:** `visualizations/fpgrowth/` → S3 `visualizations/fpgrowth/{cohort}/{age_band}/`.
- **Cohort PGx:** `visualizations/cohort_pgx/` → S3 `visualizations/cohort_pgx/networks/{cohort}/{age_band}/`.
- **Causal:** Upload script reads `dashboard_data.json` from `visualizations/causal/{cohort}/{age_band_fname}/` → S3 `visualizations/causal/{cohort}/{age_band}/causal_data.json`.

See [README_dashboard_visual_artifact_paths.md](README_dashboard_visual_artifact_paths.md) and [S3_DASHBOARD_VALIDATION_REPORT.md](S3_DASHBOARD_VALIDATION_REPORT.md) for full mapping and validation.
