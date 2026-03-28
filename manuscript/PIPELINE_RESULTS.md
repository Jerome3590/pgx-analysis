# Pipeline Results — S3 Log Audit & Manuscript Placeholder Resolution

**Audited:** 2026-03-28  
**Source buckets:** `s3://pgx-repository` · `s3://pgxdatalake`

---

## S3 Log Architecture Map

### Architecture Generations

| Era | Compute | Date Range | Canonical? |
|:----|:--------|:-----------|:----------:|
| EMR | AWS EMR (Spark) | ~2025 | No |
| EC2/DuckDB v1 | EC2 + DuckDB/Parquet | 2026-03-21 | No |
| EC2/DuckDB v2 | EC2 + DuckDB/Parquet | 2026-03-23/24 | **Yes** |

### Log Folder Inventory (`s3://pgx-repository/`)

| Folder | Description | Date Range | Use |
|:-------|:------------|:-----------|:----|
| `pgx-analysis/7_final_model/catboost_models/` | EMR-era CatBoost models | 2025 | Historical only |
| `final_model_log/` | EC2 v1 model training logs | 2026-03-21 | Superseded |
| `6_final_model_log/` | EC2 v2 model training logs | 2026-03-23/24 | **Canonical** |
| `7_shap_analysis_log/` | SHAP per bin/cohort/age | 2026-03-23/24 | Canonical |
| `8_ffa_analysis_log/` | FFA per bin/cohort/age (4.7 MiB each) | 2026-03-24 | Canonical |
| `9_cohort_pgx_log/` | PGx network topology + VIP reports | 2026-03-24/25 | Canonical |
| `cohort_pgx_log/` | PGx network topology (older runs) | 2026-02-24 to 03-20 | Superseded |
| `5_pgx_analysis_log/` | PGx feature attachment logs (~52 KiB) | 2026-03-24 | Canonical |
| `5_pgx_analysis_checkpoint/` | PGx added-features CSV per cohort | 2026-03-24 | Canonical |
| `pipeline_checkpoints/` | Structured JSON checkpoints with metadata | 2026-02-18 to 03-24 | Canonical |
| `4_model_data_log/` | Model data build logs | — | Reference |
| `4_bupar_log/` | BupaR process mining logs | — | Reference |
| `5_dtw_log/` | DTW trajectory logs | — | Reference |
| `6_fpgrowth_log/` | FP-Growth association rules | — | Reference |
| `9_dtw_log/` / `9_fpgrowth_log/` | Latest DTW / FP-Growth | — | Canonical |

### Gold Outputs (`s3://pgxdatalake/gold/`)

| Path | Description |
|:-----|:------------|
| `gold/final_model/{cohort}/{ab}/` | Model + FI CSVs + selection metadata JSON |
| `gold/dashboard/metadata/model_performance_metrics.json` | All model metrics (2026-03-25) |
| `gold/dashboard/metadata/metadata_{cohort}.json` | Full dashboard metadata per cohort |
| `gold/dashboard/models/{cohort}/{ab}/` | Deployed bin models + calibration |
| `gold/shap_analysis/` | SHAP outputs |
| `gold/fpgrowth/` | FP-Growth results |
| `gold/dtw_filter/` | DTW trajectory results |
| `gold/pgx_features/` | PGx feature outputs |

---

## Cohort Counts

**Source:** `s3://pgx-repository/pipeline_checkpoints/4_model_data/{cohort}/{ab}/checkpoint.json`  
**Completed:** 2026-03-22/23

### Opioid ED (`opioid_ed`) — RQ1, CH_3

| Age Band | Cases (N) | Controls (N) | Total |
|:---------|----------:|-------------:|------:|
| 0–12 | 893 | 11,822 | 12,715 |
| 13–24 | 52,629 | 1,069,444 | 1,122,073 |
| 25–44 | 569,096 | 12,126,611 | 12,695,707 |
| 45–54 | 413,630 | 7,046,805 | 7,460,435 |
| 55–64 | 480,246 | 8,982,552 | 9,462,798 |
| 65–74 | 397,743 | 11,499,180 | 11,896,923 |
| 75–84 | 153,082 | 5,399,679 | 5,552,761 |
| 85–114 | 38,808 | 1,426,827 | 1,465,635 |
| **Total** | **2,106,127** | **47,562,920** | **49,669,047** |

### Non-Opioid / Polypharmacy ED (`non_opioid_ed`) — RQ2, CH_4

| Age Band | Cases (N) | Controls (N) | Total |
|:---------|----------:|-------------:|------:|
| 0–12 | 27,742 | 21,406,127 | 21,433,869 |
| 13–24 | 11,956 | 13,231,511 | 13,243,467 |
| 25–44 | 8,433 | 14,251,319 | 14,259,752 |
| 45–54 | 3,310 | 8,111,787 | 8,115,097 |
| 55–64 | 2,563 | 7,936,927 | 7,939,490 |
| 65–74 | 827 | 4,615,442 | 4,616,269 |
| 75–84 | 224 | 1,599,856 | 1,600,080 |
| 85–114 | 193 | 1,370,947 | 1,371,140 |
| **Total** | **55,248** | **72,523,916** | **72,579,164** |

---

## Model Performance Metrics

**Source:** `s3://pgxdatalake/gold/dashboard/metadata/model_performance_metrics.json`  
**Generated:** 2026-03-25 · 25-run MCCV (Monte Carlo Cross-Validation) · Optuna hyperparameter optimization

### Opioid ED (`opioid_ed`)

| Age Band | Selected Model | AUC-PR | ROC-AUC | Recall |
|:---------|:--------------|-------:|--------:|-------:|
| 0–12 | — | — | — | — |
| 13–24 | XGBoost | 0.8401 | 0.9572 | 0.6478 |
| 25–44 | XGBoost | 0.9351 | 0.9788 | 0.7987 |
| 45–54 | Ensemble | 0.9547 | 0.9865 | 0.8160 |
| 55–64 | Ensemble | 0.9737 | 0.9908 | 0.8740 |
| 65–74 | Ensemble | 0.9786 | 0.9924 | 0.8673 |
| 75–84 | Ensemble | 0.9678 | 0.9904 | 0.8103 |
| 85–114 | Ensemble | 0.9009 | 0.9670 | 0.5524 |

> **Note:** `opioid_ed/0-12` model metadata not found in gold outputs — only 893 cases,
> likely below minimum threshold for stable Optuna training. Treat as excluded age band.
> Confirmed: `s3://pgxdatalake/gold/final_model/opioid_ed/0-12/` contains only `inputs/` prefix.

### Non-Opioid / Polypharmacy ED (`non_opioid_ed`)

| Age Band | Selected Model | AUC-PR | ROC-AUC | Recall |
|:---------|:--------------|-------:|--------:|-------:|
| 0–12 | Ensemble | 0.9268 | 0.9729 | 0.9831 |
| 13–24 | Ensemble | 0.9081 | 0.9769 | 0.9351 |
| 25–44 | CatBoost | 0.9105 | 0.9790 | 0.9330 |
| 45–54 | CatBoost | 0.9331 | 0.9838 | 0.9185 |
| 55–64 | CatBoost | 0.9496 | 0.9879 | 0.9366 |
| 65–74 | CatBoost | 0.9840 | 0.9958 | 0.9772 |
| 75–84 | Ensemble | 0.9966 | 0.9989 | 0.9725 |
| 85–114 | Ensemble | 0.9915 | 0.9968 | 0.9710 |

> **Note:** `non_opioid_ed` medium/high/extreme bins have deployed model files but no
> `model_selection_metadata.json` or `mc_cv_results.csv` — insufficient cases in those
> density bins to complete 25-run MCCV. Per-bin metrics available for `low` bin only.

---

## Per-Bin Model Metrics

**Bins:** `low` · `medium` · `high` · `extreme` (n_event_bin density quartiles)  
**Source:** `s3://pgxdatalake/gold/final_model/{cohort}/{ab}/bin_models/{bin}/{cohort}_{ab}_model_selection_metadata.json`  
**Training:** 25-run MCCV + Optuna per bin

### Opioid ED — Per Bin

| Age Band | Bin | Model | AUC-PR | ROC-AUC | Recall |
|:---------|:----|:------|-------:|--------:|-------:|
| 13–24 | low | CatBoost | 0.8347 | 0.9367 | 0.5502 |
| 13–24 | medium | XGBoost | 0.8401 | 0.9572 | 0.6478 |
| 13–24 | high | Ensemble | 0.8897 | 0.9825 | 0.6385 |
| 13–24 | extreme | XGBoost | 0.6325 | 0.9109 | 0.0000 |
| 25–44 | low | Ensemble | 0.8891 | 0.9597 | 0.6706 |
| 25–44 | medium | XGBoost | 0.9351 | 0.9788 | 0.7987 |
| 25–44 | high | Ensemble | 0.9675 | 0.9950 | 0.8446 |
| 25–44 | extreme | XGBoost | 0.8584 | 0.9910 | 0.5770 |
| 45–54 | low | Ensemble | 0.8958 | 0.9587 | 0.6787 |
| 45–54 | medium | Ensemble | 0.9547 | 0.9865 | 0.8160 |
| 45–54 | high | Ensemble | 0.9823 | 0.9974 | 0.8774 |
| 45–54 | extreme | XGBoost | 0.8437 | 0.9852 | 0.4000 |
| 55–64 | low | Ensemble | 0.9155 | 0.9645 | 0.7277 |
| 55–64 | medium | Ensemble | 0.9737 | 0.9908 | 0.8740 |
| 55–64 | high | XGBoost | 0.9839 | 0.9976 | 0.8967 |
| 55–64 | extreme | — | — | — | — |
| 65–74 | low | Ensemble | 0.9618 | 0.9732 | 0.8513 |
| 65–74 | medium | Ensemble | 0.9786 | 0.9924 | 0.8673 |
| 65–74 | high | XGBoost | 0.9318 | 0.9949 | 0.6864 |
| 65–74 | extreme | — | — | — | — |
| 75–84 | low | Ensemble | 0.9682 | 0.9726 | 0.8741 |
| 75–84 | medium | Ensemble | 0.9678 | 0.9904 | 0.8103 |
| 75–84 | high | XGBoost | 0.6974 | 0.9696 | 0.3238 |
| 75–84 | extreme | — | — | — | — |
| 85–114 | low | Ensemble | 0.9350 | 0.9333 | 0.8113 |
| 85–114 | medium | Ensemble | 0.9009 | 0.9670 | 0.5524 |
| 85–114 | high | XGBoost | 0.3023 | 0.8034 | 0.0400 |
| 85–114 | extreme | — | — | — | — |

> **Pattern:** Performance peaks at `medium` and `high` bins for most age bands.
> `extreme` bin absent for age ≥ 55 (too few extreme-density patients).
> `high` bin degrades for oldest cohorts (75–84, 85–114) — sparse event density.

### Non-Opioid ED — Per Bin

> Only `low` bin has full MCCV metadata across all age bands. `medium`/`high`/`extreme`
> bins have deployed model artifacts but no evaluation metadata — insufficient cases.

| Age Band | Bin | Model | AUC-PR | ROC-AUC | Recall |
|:---------|:----|:------|-------:|--------:|-------:|
| 0–12 | low | Ensemble | 0.9268 | 0.9729 | 0.9831 |
| 13–24 | low | Ensemble | 0.9081 | 0.9769 | 0.9351 |
| 25–44 | low | CatBoost | 0.9105 | 0.9790 | 0.9330 |
| 45–54 | low | CatBoost | 0.9331 | 0.9838 | 0.9185 |
| 55–64 | low | CatBoost | 0.9496 | 0.9879 | 0.9366 |
| 65–74 | low | CatBoost | 0.9840 | 0.9958 | 0.9772 |
| 75–84 | low | Ensemble | 0.9966 | 0.9989 | 0.9725 |
| 85–114 | low | Ensemble | 0.9915 | 0.9968 | 0.9710 |
| all | medium/high/extreme | — | — | — | — |

---

## Outstanding — Still Needed for Manuscript

### Lambda Inference Benchmarks

Not available in S3. Must be retrieved from **AWS CloudWatch Logs**.

```powershell
# Retrieve Lambda performance metrics from CloudWatch
aws logs filter-log-events `
  --log-group-name "/aws/lambda/pgx-risk-dashboard" `
  --filter-pattern "REPORT" `
  --start-time (Get-Date).AddDays(-30).ToUniversalTime() `
  | ConvertFrom-Json | Select-Object -ExpandProperty events `
  | ForEach-Object { $_.message } `
  | Select-String "Duration|Billed|Memory"
```

Alternatively, use Lambda console → Monitor → View CloudWatch Logs → filter for `REPORT` lines.  
Report format: `REPORT RequestId: ... Duration: XXX.XX ms   Billed Duration: XXX ms   Memory Size: XXXX MB   Max Memory Used: XXX MB`

### opioid_ed / 0–12 Model Metrics

Only 893 cases — likely excluded from training. Confirm by checking:

```powershell
aws s3 ls s3://pgxdatalake/gold/final_model/opioid_ed/0-12/ --human-readable
```

If absent, note in manuscript: *"The 0–12 age band was excluded from opioid ED prediction modeling due to insufficient case count (N = 893)."*

---

## Extraction Commands for Remaining Pipeline Results

### FP-Growth Top Drug Associations

```powershell
aws s3 ls s3://pgxdatalake/gold/fpgrowth/ --recursive --human-readable | head -30
```

### DTW Trajectory Cluster Counts

```powershell
aws s3 ls s3://pgxdatalake/gold/dtw_filter/ --recursive --human-readable | head -30
```

### PGx Feature Coverage (% patients with PGx data)

```powershell
# Read the pgx added features CSV for a representative cohort
aws s3 cp s3://pgx-repository/5_pgx_analysis_checkpoint/opioid_ed/55-64/pgx_added_features_opioid_ed_55_64.csv - | python -c "import sys,csv; r=list(csv.reader(sys.stdin)); print(f'rows={len(r)-1}')"
```

### SHAP Top Features per Cohort

```powershell
aws s3 ls s3://pgxdatalake/gold/shap_analysis/ --recursive --human-readable | head -20
```

---

## Manuscript Placeholder → Data Mapping

| Placeholder | Chapter | Data Source | Status |
|:-----------|:--------|:------------|:------:|
| Total opioid ED cohort N | CH_2, CH_3, CH_6 | `4_model_data` checkpoints | ✅ 2,106,127 cases |
| Total polypharmacy ED cohort N | CH_2, CH_4, CH_6 | `4_model_data` checkpoints | ✅ 55,248 cases |
| Opioid ED AUROC by age band | CH_3 | `model_performance_metrics.json` | ✅ 0.9572–0.9924 |
| Polypharmacy ED AUROC by age band | CH_4 | `model_performance_metrics.json` | ✅ 0.9729–0.9989 |
| Opioid ED AUC-PR by age band | CH_3 | `model_performance_metrics.json` | ✅ 0.8401–0.9786 |
| Polypharmacy ED AUC-PR by age band | CH_4 | `model_performance_metrics.json` | ✅ 0.9081–0.9966 |
| Lambda inference latency `[XXX ms]` | CH_5, CH_6 | CloudWatch Logs | ⏳ Pending |
| opioid_ed/0-12 model metrics | CH_3 | gold/final_model | ❌ Not found — exclude |
| FP-Growth top associations | CH_3, CH_4 | gold/fpgrowth | ⏳ Extract |
| DTW cluster N / trajectory counts | CH_3, CH_4 | gold/dtw_filter | ⏳ Extract |
| PGx feature coverage % | CH_5 | 5_pgx_analysis_checkpoint | ⏳ Extract |
| SHAP top-10 features | CH_3, CH_4, CH_5 | gold/shap_analysis | ⏳ Extract |
