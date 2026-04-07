# Metrics & Tables — Pipeline Results Reference

> Confirmed values from S3 logs, CloudWatch, and extraction scripts.  
> See **[README.md](README.md)** for placeholder status and post-retrain checklist.  
> See **[FIGURES.md](FIGURES.md)** for figure generation from these values.

---

## Cohort Counts

Source: `s3://pgx-repository/5_pgx_analysis_log/{cohort}/{ab}/pgx_{cohort}_{ab_snake}.log`  
("Created 2 PGx features for N patients" — N = total patients cases + controls, training 2016–2018)

| Cohort | Age Band | Cases | Controls | Total |
|:-------|:---------|------:|---------:|------:|
| opioid_ed | 13–24 | 1,630 | 12,080 | 13,710 |
| opioid_ed | 25–44 | 12,753 | 94,635 | 107,388 |
| opioid_ed | 45–54 | 5,984 | 37,655 | 43,639 |
| opioid_ed | 55–64 | 6,343 | 36,270 | 42,613 |
| non_opioid_ed | 65–74 | 801 | 10,770 | 11,571 |
| non_opioid_ed | 75–84 | 213 | 2,980 | 3,193 |
| non_opioid_ed | 85–114 | 168 | 2,355 | 2,523 |

**Manuscript grand totals:**  
opioid_ed (13–64): **26,710 cases / 180,640 controls**  
non_opioid_ed (65–114): **1,182 cases / 16,105 controls**  
`opioid_ed/0-12` excluded (893 cases — below training threshold)

APCD overall: ~4.2M unique patients, 380M claims; years 2016–2019 (2020 excluded — COVID).

---

## Model Performance

Source: `s3://pgxdatalake/gold/final_model/{cohort}/{ab}/bin_models/low/{cohort}_{ab_snake}_model_metrics_summary.csv`  
Temporal holdout: 2019. LOW event-density bin reported (primary manuscript tables).

| Cohort | Band | Selected Model | AUROC | PR-AUC | Recall | Brier | ICI |
|:-------|:-----|:---------------|------:|-------:|-------:|------:|-----:|
| opioid_ed | 13–24 | XGBoost | 0.957 | 0.840 | 0.648 | 0.008 | 0.164 |
| opioid_ed | 25–44 | XGBoost | 0.979 | 0.935 | 0.799 | 0.013 | 0.108 |
| opioid_ed | 45–54 | Ensemble | 0.987 | 0.955 | 0.816 | 0.007 | 0.154 |
| opioid_ed | 55–64 | Ensemble | 0.991 | 0.974 | 0.874 | 0.051 | 0.138 |
| non_opioid_ed | 65–74 | CatBoost | 0.996 | 0.984 | 0.977 | 0.008 | 0.071 |
| non_opioid_ed | 75–84 | Ensemble | 0.999 | 0.997 | 0.973 | 0.007 | 0.290 |
| non_opioid_ed | 85–114 | Ensemble | 0.997 | 0.992 | 0.971 | 0.007 | 0.212 |

Brier: `brier_ici_results.json` (catboost_per_bin, 2019 holdout).  
MCCV: 25-run 80/20 stratified splits on 2016–2018 training data.

---

## Consensus-Causal Feature Counts

Source: `s3://pgxdatalake/gold/bupar/allowed_codes/allowed_codes_shap_ffa_{cohort}_{ab_snake}.json`  
Features passing BOTH SHAP rank ≥ Q75 AND FFA support ≥ 0.05, confidence ≥ 0.70.  
Script: `scripts/get_consensus_counts.py`

| Cohort | Band | Consensus-Causal Features |
|:-------|:-----|-------------------------:|
| opioid_ed | 13–24 | 384 |
| opioid_ed | 25–44 | 498 |
| opioid_ed | 45–54 | 498 |
| opioid_ed | 55–64 | 498 |
| non_opioid_ed | 65–74 | 89 |
| non_opioid_ed | 75–84 | 33 |
| non_opioid_ed | 85–114 | 29 |

Overall: Consensus features ≈ 49% of total. SHAP-only: 252. FFA-only: 33.

---

## Top Drugs by Band

Source: `s3://pgxdatalake/gold/dashboard/metadata/metadata_{cohort}.json`  
Keys: `codes[band].drugs[].display` sorted by `importance` (0–1 normalized).

| Cohort | Band | #1 Drug | #2 Drug | #3 Drug |
|:-------|:-----|:--------|:--------|:--------|
| opioid_ed | 13–24 | Buprenorphine-Naloxone | Gabapentin | Naltrexone |
| opioid_ed | 25–44 | Buprenorphine-Naloxone | Gabapentin | Quetiapine |
| opioid_ed | 45–54 | Buprenorphine-Naloxone | Clonidine | Gabapentin |
| opioid_ed | 55–64 | Buprenorphine-Naloxone | Oxycodone HCl | Gabapentin |
| non_opioid_ed | 65–74 | Gabapentin | Fluzone HD | Gavilyte-C |
| non_opioid_ed | 75–84 | Losartan | Pravastatin | Furosemide |
| non_opioid_ed | 85–114 | Amlodipine | Furosemide | Potassium Chloride ER |

---

## Lambda Benchmarks

Source: CloudWatch log group `/aws/lambda/pgx-risk-calculator`  
Query: `REPORT` lines for Duration; `INIT_REPORT` lines for Init Duration.

| Metric | Value | SD | n samples |
|:-------|------:|---:|----------:|
| Warm inference latency | 6 ms | 1 ms | 18 REPORT lines |
| Cold-start (container init) | 2,100 ms | 250 ms | 4 INIT_REPORT lines |

1 outlier excluded (3,532 ms — likely image pull, not steady-state).  
Manuscript targets: cold-start < 500 ms; warm < 100 ms.

```bash
aws logs filter-log-events \
  --log-group-name /aws/lambda/pgx-risk-calculator \
  --filter-pattern "REPORT" \
  --start-time {epoch_ms} \
  --query "events[*].message" --output text
```

---

## Placeholder Tracker

### ✅ Resolved

| Placeholder | Value applied | Chapters |
|:-----------|:--------------|:---------|
| `[IRB-XXXX]` | HM20022300 | CH_3, CH_4 |
| `[Funding statement]` | "This research received no external funding." | All |
| `[https://github.com/[repo]]` | https://github.com/Jerome3590/pgx-analysis | CH_2, CH_5 |
| `[version/date]` CPIC snapshot | March 2026 | CH_5 |
| `[Month Year]` defense | 1 June 2026 (planned) | CH_6 |
| `[CRD-XXXXXX]` | CRD420261354089 | CH_1 |
| `[Chair]`, `[Member 1–3]` | See committee in README.md | CH_6 |
| Data Availability statement | VHI DUA canonical language | CH_2–CH_4, CH_6 |

### ✅ Confirmed Post-Retrain

| Item | Chapter(s) | Source |
|:-----|:-----------|:-------|
| Brier / ICI per cohort × band | CH_3, CH_4 | `data/brier_ici_results.json` |
| FFA pair/triplet counts (115 pairs; 5,021 triplets) | CH_4 | `data/ffa_ie_ci.json` |
| FFA IE scores table (top 5 pairs with 95% CI) | CH_4 | `data/ffa_ie_ci.json` |
| IR scores (simvastatin/furosemide/alprazolam) | CH_4 | `data/ffa_manuscript_data.json` |
| DTW archetypes (Rapid-Onset n=5,481/21%; Chronic-Escalation n=21,229/79%) | CH_3 | chapter text |
| SHAP rank #1 `pgx_num_drugs` (mean\|SHAP\|=1.22); rank #2 `pgx_num_cpic_drugs` (0.63); gabapentin (0.23) | CH_3 | `data/shap_top_features.json` opioid_ed/25-44/low |
| PGx coverage data generated (opioid_ed 13–24: 74.2%; 25–44: 81.9%; 45–54: 85.9%; 55–64: 85.8%) | CH_5 | `data/pgx_coverage.json` |

### ⏳ Still Needed

| Placeholder | Chapter(s) | Source file | Notes |
|:-----------|:-----------|:------------|:------|
| FP-Growth top rule (support, confidence) | CH_3 | `data/visual_manuscript_data.json` → opioid_ed/25-44 medium/high bin | low bin has 0 rules |
| CH_5 PGx coverage table | CH_5 | `pgx_coverage.json` | Add table to results section |
| CRediT author contributions | CH_1–CH_5 | Manual — MDPI/Wiley required field | ✍️ |
| Lambda benchmarks (post-redeploy) | CH_5 | CloudWatch post `prepare_models.py` redeploy | Manual |

---

## Post-Retrain Update Checklist

After EC2 Notebooks 3 + 4 complete and local extraction scripts run:

- [x] **CH_3 Table 2** — AUROC/PR-AUC/Brier/ICI confirmed (ICI range corrected to 0.108–0.164)
- [x] **CH_3 abstract** — PR-AUC 0.840–0.979; AUROC 0.957–0.992 confirmed
- [x] **CH_3 DTW cluster N/%** — Rapid-Onset n=5,481/21%; Chronic-Escalation n=21,229/79% confirmed
- [x] **CH_4 DDI pair/triplet counts** — 115 synergistic pairs; 5,021 high-risk triplets confirmed
- [x] **CH_4 tbl-ddi IE scores table** — top 5 pairs with 95% CI from `ffa_ie_ci.json` confirmed
- [x] **CH_4 IR scores** — simvastatin/furosemide/alprazolam IR values confirmed
- [x] **CH_3 SHAP values** — `pgx_num_drugs` rank #1 (1.22), `pgx_num_cpic_drugs` rank #2 (0.63), gabapentin rank #3 (0.23); chapter updated
- [ ] **CH_3 FP-Growth rule** — opioid_ed/25-44/low has 0 rules; check medium/high bins in `data/visual_manuscript_data.json`
- [x] **CH_5 PGx coverage table** — `data/pgx_coverage.json` + table in CH_5 (`{#tbl-pgx-coverage}`)
- [x] **CH_5 benchmark tables** — Synthetic `{#tbl-benchmarks}` + CloudWatch ops `{#tbl-benchmarks-cw}`; snapshot **`2026-03-31T16:46:25Z`** post-deploy (`manuscript/infrastructure_setup/cloudwatch/benchmark_snapshot.json`, `lambda_timing*_20260331.txt`)
- [ ] Regenerate all figures: `python manuscript/infrastructure_setup/scripts/generate_figures/generate_figures.py`
- [ ] Rebuild all PDFs: `cd manuscript; .\build.ps1`

---

## Pipeline Run Order

### EC2 — Strict sequence

```
Notebook 3  →  run_final_model.py (6_final_model/)
               Output: bin_models/ → s3://pgxdatalake/gold/final_model/
               Verify: bin_models/low/models/ exists for all cohort/band combos

Notebook 4  →  4_dashboard_visuals.ipynb (9_dashboard_visuals/)
               Scripts: bupar/, dtw/, fpgrowth/; also 8_ffa_analysis/
               Output: manuscript_checkpoints/ → s3://pgxdatalake/gold/manuscript_checkpoints/
               Verify: no RuntimeError on n_event_bin_thresholds.json (needs Notebook 3 first)

Notebook 5  →  prepare_models.py + Lambda build/deploy (10_risk_dashboard/)
               Output: Docker image to ECR; cold-start test via POST /risk
```

> ⚠️ `ffa_causal_factors.csv` is written to EC2 local disk only (`8_ffa_analysis/outputs/`).  
> Must be manually copied to S3 before running `extract_ffa_manuscript.py`.

### Local — After EC2 completes

```powershell
cd c:\Projects\pgx-analysis\manuscript\scripts

python compute_brier_ici.py          # → brier_ici_results.json
python extract_ffa_manuscript.py     # → ffa_manuscript_data.json, ffa_ie_ci.json
python extract_visual_manuscript.py  # → visual_manuscript_data.json, pgx_coverage.json, shap_top_features.json

cd c:\Projects\pgx-analysis\manuscript
python generate_figures.py           # regenerate CH_3/CH_4/CH_5 figures
.\build.ps1                          # rebuild all PDFs
```

---

## S3 Data Sources

### Bucket: `pgxdatalake` (primary model outputs)

| Path pattern | Contents |
|:-------------|:---------|
| `gold/final_model/{cohort}/{ab}/bin_models/{bin}/` | Per-bin model `.joblib`, calibration, FI CSV |
| `gold/dashboard/metadata/model_performance_metrics.json` | All model metrics snapshot (2026-03-25) |
| `gold/dashboard/metadata/metadata_{cohort}.json` | Top features per band (drugs / ICD / CPT) |
| `gold/shap_analysis/{cohort}/{ab}/` | SHAP global importance CSV + sample values parquet |
| `gold/fpgrowth/cohort/drug_name/cohort_name={cohort}/age_band={ab}/` | FP-Growth co-occurrence results |
| `gold/pgx_features/{cohort}/{ab}/` | `pgx_num_drugs`, `pgx_num_cpic_drugs` per patient |
| `gold/bupar/allowed_codes/allowed_codes_shap_ffa_{cohort}_{ab}.json` | Consensus-Causal feature lists |
| `gold/manuscript_checkpoints/{type}/{cohort}/{ab}/{bin}/` | Notebook 4 extraction JSONs |

Note: one row per claim event in cohort parquet — use `5_pgx_analysis_log` for patient counts, not row counts.

### Bucket: `pgx-repository` (pipeline logs)

| Path prefix | Contents |
|:------------|:---------|
| `5_pgx_analysis_log/{cohort}/{ab}/` | **Authoritative patient counts** ("Created 2 PGx features for N patients") |
| `6_final_model_log/` | Training runtime, bin durations, S3 upload confirmation |
| `7_shap_analysis_log/` | SHAP n_background, n_eval per bin |
| `8_ffa_analysis_log/` | FFA rules log — **outputs on EC2 local disk only, not in S3** |
| `9_dtw_log/` | DTW logs — FAILED 2026-03-29 (missing `first_opioid_ed_date` column) |
| `9_fpgrowth_log/` | FP-Growth visualization logs |
| `pipeline_checkpoints/{step}/{cohort}/{ab}/checkpoint.json` | Step completion status + output S3 paths |

### CloudWatch

Log group: `/aws/lambda/pgx-risk-calculator`  
Filter: `REPORT` lines (warm latency), `INIT_REPORT` lines (cold-start).
