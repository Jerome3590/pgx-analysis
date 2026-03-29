# CH_1 Metrics README
**Chapter:** A Partition-First Data Architecture for Large-Scale Pharmacoepidemiological Modeling  
**Manuscript:** `CH_1/ch01_bmic.qmd` → MDPI JPM

---

## Metrics Summary

| Metric | Value | Calculation | Script |
|--------|-------|-------------|--------|
| Throughput improvement | 15× | Wall-clock time: partition-first vs monolithic baseline on same EC2 instance | Architecture benchmark; reported from EC2 pipeline run logs |
| Worker scalability | Linear across 32-core, 1 TB RAM | Elapsed time per age-band × year partition across parallel DuckDB workers | EC2 CloudWatch run logs |
| MCCV splits | 50+ random splits | Random train/test splits on 2016–2018 data with stratified sampling | `6_final_model/run_final_model.py` (`n_runs` param) |
| Feature rank stability | Median recall-weighted rank ≥ 25th pct | Rank of each feature across 50+ MCCV splits; retain if median rank ≥ Q25 | `6_final_model/build_final_cohort_model_features.py` |
| Pipeline stages | 5 (Bronze → Silver → Gold → MCCV → Ensemble) | Sequential data lake zones + modeling steps | `2_create_cohort/`, `5_feature_engineering/`, `6_final_model/` |
| APCD coverage | ~4.2M unique patients, 380M claims | Distinct `mi_person_key` count from `medical_raw.medical_partitioned` | `scripts/apcd_total_count2.py` |
| Pandemic exclusion | 2020 excluded | Hard-coded year filter: `event_year IN (2016,2017,2018,2019)` | `2_create_cohort/phases/phase1_data_extraction.py` |

---

## Detailed Metric Definitions

### 1. Throughput Improvement (15×)
- **Definition:** Ratio of records processed per wall-clock second between partition-first and monolithic architectures.
- **Calculation:** `throughput_ratio = (records/sec_partition) / (records/sec_monolithic)`
- **Source:** EC2 instance run logs comparing single-threaded DuckDB vs Age Band × Year parallel workers.
- **Script reference:** Architecture benchmark logs; not recomputed by a standalone script.

### 2. MCCV Stability
- **Definition:** Coefficient of variation (CV) of AUROC and PR-AUC across 50+ MCCV splits.
- **Calculation:** `CV = SD(metric) / mean(metric)` across splits; lower CV = more stable.
- **Script reference:** `6_final_model/run_final_model.py` — `train_and_evaluate()` function, `n_runs` argument.

### 3. Consensus Filter
- **Definition:** Feature passes Consensus Filter iff SHAP rank ≥ 75th percentile **AND** FFA causal_responsibility ≥ 0.05 with rule confidence ≥ 0.70.
- **Calculation:** Boolean AND of SHAP threshold and FFA threshold.
- **Script reference:** `scripts/consensus_filter_final.py`; counts output to manuscript in `scripts/get_consensus_counts.py`.

### 4. Data Lake Architecture
- **Bronze zone:** Raw fixed-width APCD files uploaded to `s3://pgxdatalake/bronze/`
- **Silver zone:** DuckDB-processed partitioned parquet at `s3://pgxdatalake/silver/`
- **Gold zone:** Final analytic cohorts at `s3://pgxdatalake/gold/`
- **Script reference:** `2_create_cohort/phases/` (phase1–phase4); `5_feature_engineering/`

### 5. Total APCD Patient Count (6,929,576)
- **Definition:** Distinct `mi_person_key` values in `medical_raw.medical_partitioned` for years 2016–2019.
- **Calculation:** `SELECT COUNT(DISTINCT mi_person_key) FROM medical_raw.medical_partitioned WHERE CAST(event_year AS INTEGER) BETWEEN 2016 AND 2019`
- **Script reference:** `scripts/apcd_total_count2.py`

---

## Data Sources
| Source | Location |
|--------|----------|
| Virginia APCD raw | `s3://pgxdatalake/bronze/` |
| Silver partitioned | `s3://pgxdatalake/silver/` |
| Gold cohorts | `s3://pgxdatalake/gold/cohorts_model_data/` |
| Athena catalog | AWS Glue: `medical_raw.medical_partitioned` |
