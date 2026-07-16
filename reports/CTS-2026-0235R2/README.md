# CTS-2026-0235R2 — utilization-free stress-test artifacts

Public (GitHub-tracked) SSOT for the CH4 CTS revision **CTS-2026-0235R2** util-free stress test (Supplementary Table S5). Aggregate util-free refits withdraw utilization tempo/density structure entirely; density-stratified ensembles remain the primary analysis.

| Field | Value |
|:------|:------|
| Journal MS ID | **CTS-2026-0235R2** |
| Cohort | `non_opioid_ed` (polypharmacy) |
| Driver | root `3_model_sensitivity.ipynb` |
| Runner | `6_final_model/run_sensitivity_util_free.py` |
| This folder | Numbers + per-band comparison tables (no fitted model binaries) |

## Rollup files (Table S5)

- `sensitivity_auprc_by_age_band.csv`
- `sensitivity_summary_all_bands.json`
- `sensitivity_summary.json` (legacy 65–74 focal mirror)

## Per-band folders

`0_12/` … `85_114/` — dropped-feature lists, top drug SHAP, published-pair IE persistence.

## Rebuild

```powershell
python 6_final_model/run_sensitivity_util_free.py
```

Writes rollups and per-band tables into this directory (`reports/CTS-2026-0235R2/`).

Manuscript DOCX Supplement embeds Table S5 via `manuscript/templates/make_supp_tables.py` (reads from this path).
