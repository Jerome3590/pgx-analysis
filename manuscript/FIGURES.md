# Figures & Visuals — Master Inventory

> All figures confirmed generated and cited in QMD source as of 2026-03-29.  
> See **[METRICS.md](METRICS.md)** for data values behind figures.  
> See **[README.md](README.md)** for build instructions.

---

## Complete Figure Inventory

| Chapter | File | QMD ref label | In QMD ✓ |
|:--------|:-----|:--------------|:--------:|
| CH_1 | `figures/ch01/fig_prisma.pdf` | `fig-prisma` | ✅ |
| CH_1 | `figures/ch01/fig_ml_methods.pdf` | `fig-ml-methods` | ✅ |
| CH_1 | `figures/ch01/fig_evidence_map.pdf` | `fig-evidence-map` | ✅ |
| CH_2 | `figures/ch02/fig_architecture.pdf` | `fig-architecture` | ✅ |
| CH_2 | `figures/ch02/fig_attrition.pdf` | `fig-attrition` | ✅ |
| CH_2 | `figures/ch02/fig_consensus.pdf` | `fig-consensus` | ✅ |
| CH_3 | `figures/ch03/fig_attrition.pdf` | `fig-attrition` | ✅ |
| CH_3 | `figures/ch03/fig_curves.pdf` | `fig-curves` | ✅ |
| CH_3 | `figures/ch03/fig_shap.pdf` | `fig-shap` | ✅ |
| CH_3 | `figures/ch03/fig_shap_pdp.pdf` | `fig-shap-pdp` | ✅ |
| CH_3 | `figures/ch03/fig_trajectories.pdf` | `fig-trajectories` | ✅ |
| CH_3 | `figures/ch03/fig_dtw_pathways.pdf` | `fig-dtw-pathways` | ✅ |
| CH_3 | `figures/ch03/fig_trajectories_heatmap.pdf` | `fig-trajectories-heatmap` | ✅ |
| CH_4 | `figures/ch04/fig_network.pdf` | `fig-network` | ✅ |
| CH_4 | `figures/ch04/fig_ir.pdf` | `fig-ir` | ✅ |
| CH_4 | `figures/ch04/fig_zcode.pdf` | `fig-zcode` | ✅ |
| CH_4 | `figures/ch04/fig_shap.pdf` | `fig-shap` | ✅ |
| CH_4 | `figures/ch04/fig_shap_pdp.pdf` | `fig-shap-pdp` | ✅ |
| CH_5 | `figures/ch05/fig_architecture.pdf` | `fig-architecture` | ✅ |
| CH_5 | `figures/ch05/fig_dashboard.pdf` | `fig-dashboard` | ✅ |
| CH_5 | `figures/ch05/fig_imputation.pdf` | `fig-imputation` | ✅ |
| CH_5 | `figures/ch05/fig_latency.pdf` | `fig-latency` | ✅ |
| CH_6 | `figures/ch06/fig_scenario.pdf` | `fig-scenario` | ✅ |
| CH_6 | `figures/ch02/fig_consensus.pdf` (cross-ref) | `fig-consensus` | ✅ |
| CH_6 | `figures/ch06/pgx_dashboard_architecture.pdf` | `fig-dashboard-arch` | ✅ |

---

## Generation

### Python-generated figures (CH_3, CH_4, CH_5)

```powershell
# From repo root — regenerate all data-driven figures after retrain
python manuscript/generate_figures.py

# Chapter-specific
python manuscript/generate_figures_ch3.py   # SHAP beeswarm, PDP, DTW heatmap, trajectory clusters
python manuscript/generate_figures_ch4.py   # FP-Growth network, IR rankings, Z-code violin
python manuscript/generate_figures_ch5.py   # Architecture diagram, latency histograms, imputation
```

Data inputs read from `manuscript/scripts/` extraction JSON outputs:
- `shap_top_features.json` → `fig_shap.pdf`, `fig_shap_pdp.pdf`
- `dtw_manuscript_summary.json` → `fig_trajectories.pdf`, `fig_trajectories_heatmap.pdf`, `fig_dtw_pathways.pdf`
- `visual_manuscript_data.json` → `fig_network.pdf`, `fig_ir.pdf`, `fig_zcode.pdf`, `fig_curves.pdf`
- CloudWatch results → `fig_latency.pdf`

### TikZ-compiled figures (CH_6)

```powershell
# Compile fig_scenario_standalone.tex → fig_scenario.pdf
pdflatex -interaction=nonstopmode figures\ch06\fig_scenario_standalone.tex
mv fig_scenario_standalone.pdf figures\ch06\fig_scenario.pdf
```

Source: `figures/ch06/fig_scenario_standalone.tex`  
Packages required: `tikz`, `amsmath` (do **not** add `microtype` — conflicts with standalone CM fonts)

### Static / pre-generated figures (CH_1, CH_2)

- CH_1: PRISMA flow diagram — `prisma2020` R package or prisma-statement.org; export PDF
- CH_2: Architecture diagram — draw.io / Lucidchart; export PDF
- CH_2: `fig_consensus.pdf` — compiled from `figures/ch02/fig_consensus_standalone.tex`
- CH_2: `fig_attrition.pdf` — CONSORT-style; R ggplot2 or draw.io

---

## Post-Retrain Figure Checklist

After `generate_figures.py` runs on new extraction JSONs:

- [ ] `fig_shap.pdf` (CH_3 & CH_4) — top-20 Consensus-Causal features updated
- [ ] `fig_shap_pdp.pdf` (CH_3 & CH_4) — partial dependence plots updated
- [ ] `fig_curves.pdf` (CH_3 & CH_4) — PR curves + calibration diagrams updated
- [ ] `fig_trajectories.pdf` (CH_3) — DTW cluster assignments updated
- [ ] `fig_trajectories_heatmap.pdf` (CH_3) — cluster heatmap updated
- [ ] `fig_dtw_pathways.pdf` (CH_3) — top drug sequence pathways updated
- [ ] `fig_ir.pdf` (CH_4) — FFA Intervention Rate rankings updated
- [ ] `fig_network.pdf` (CH_4) — FP-Growth network updated
- [ ] `fig_latency.pdf` (CH_5) — Lambda latency histogram updated
- [ ] Rebuild all PDFs: `.\build.ps1`

---

## Figure Count by Journal (submission limits)

| Chapter | Journal | Figures | Limit | Status |
|:--------|:--------|:-------:|:-----:|:------:|
| CH_1 | MDPI JPM (review) | 3 | ≤ 8 | ✅ |
| CH_2 | CPT:PSP | 3 | ≤ 5 | ✅ |
| CH_3 | CTS | 7 | ≤ 5 | ⚠️ consolidate 2 |
| CH_4 | CPT:PSP | 5 | ≤ 5 | ✅ |
| CH_5 | MDPI JPM (article) | 4 | ≤ 7 | ✅ |
| CH_6 | dissertation | 3 | — | ✅ |

> ⚠️ **CH_3** exceeds CTS limit of 5 — consider merging `fig_dtw_pathways` + `fig_trajectories_heatmap` into one composite panel, or moving one to supplementary.
