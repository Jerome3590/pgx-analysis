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

> ⚠️ **Pre-retrain values.** Update after Notebook 3 reruns without `n_events` feature. Run `scripts/compute_brier_ici.py`.

| Cohort | Band | Selected Model | ROC-AUC | PR-AUC | Recall | LogLoss |
|:-------|:-----|:---------------|--------:|-------:|-------:|--------:|
| opioid_ed | 13–24 | CatBoost | 0.937 | 0.835 | 0.550 | 0.252 |
| opioid_ed | 25–44 | Ensemble | 0.961 | 0.889 | 0.671 | 0.207 |
| opioid_ed | 45–54 | Ensemble | 0.960 | 0.896 | 0.679 | 0.209 |
| opioid_ed | 55–64 | Ensemble | 0.966 | 0.916 | 0.728 | 0.213 |
| non_opioid_ed | 65–74 | CatBoost | 0.996 | 0.984 | 0.977 | 0.064 |
| non_opioid_ed | 75–84 | Ensemble | 0.999 | 0.997 | 0.973 | 0.043 |
| non_opioid_ed | 85–114 | Ensemble | 0.997 | 0.992 | 0.971 | 0.081 |

Brier score range (pre-retrain): 0.0070–0.0509 (opioid_ed); ICI range: 0.1084–0.1635.  
MCCV: 50+ random 80/20 stratified splits on 2016–2018 training data.

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

### ⏳ Still Needed (post-retrain)

| Placeholder | Chapter(s) | Source file | Notes |
|:-----------|:-----------|:------------|:------|
| Brier score / ICI per cohort × band | CH_3, CH_4 | `scripts/compute_brier_ici.py` → `brier_ici_results.json` | Run after Notebook 3 |
| SHAP top-10 feature names + mean\|SHAP\| | CH_3, CH_4, CH_5 | `scripts/extract_visual_manuscript.py` → `shap_top_features.json` | |
| FP-Growth top rule (support, confidence) | CH_3, CH_4 | `visual_manuscript_data.json` → `fpgrowth.top_rules[0]` (opioid_ed/25-44/low) | |
| DTW cluster N / % / median months | CH_3, CH_6 | `dtw_manuscript_summary.json` → `archetypes_by_dtw_quartile` | DTW failed 2026-03-29; rerun after fix |
| FFA pair/triplet counts + IE/IR scores | CH_4 | `ffa_ie_ci.json`, `ffa_manuscript_data.json` | EC2 local files — copy to S3 first |
| PGx feature coverage % | CH_5 | `pgx_coverage.json` | |
| CRediT author contributions | CH_1–CH_5 | Manual — MDPI/Wiley required field | ✍️ |
| Lambda benchmarks (post-redeploy) | CH_5 | CloudWatch post `prepare_models.py` redeploy | Manual |

---

## Post-Retrain Update Checklist

After EC2 Notebooks 3 + 4 complete and local extraction scripts run:

- [ ] **CH_3 Table 2** — AUROC/PR-AUC/Brier/ICI from `brier_ici_results.json`
- [ ] **CH_3 abstract** — update mean PR-AUC (currently 0.88) and AUROC (currently 0.96) averages
- [ ] **CH_3 SHAP values** (~line 301) — top features + mean|SHAP| from `shap_top_features.json`
- [ ] **CH_3 FP-Growth rule** — top association rule from `visual_manuscript_data.json`
- [ ] **CH_3 DTW cluster N/%** — Rapid-Onset / Chronic-Escalation sizes from `dtw_manuscript_summary.json`
- [ ] **CH_4 DDI pair/triplet counts** (~line 313) — from `ffa_ie_ci.json`
- [ ] **CH_4 tbl-ddi IE scores table** — from `ffa_ie_ci.json`
- [ ] **CH_4 IR scores** (~line 359) — from `ffa_manuscript_data.json`
- [ ] **CH_5 benchmark table** — Lambda latency from CloudWatch post-redeploy
- [ ] Regenerate all figures: `python manuscript/generate_figures.py`
- [ ] Rebuild all PDFs: `.\build.ps1`

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
