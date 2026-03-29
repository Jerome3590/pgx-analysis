# CH_3 Metrics README
**Chapter:** Opioid-Use Disorder ED Risk Prediction with Temporal Trajectory Analysis  
**Manuscript:** `CH_3/ch03_cts.qmd` → CTS (Wiley)

---

## Metrics Summary

| Metric | Value | Calculation | Script |
|--------|-------|-------------|--------|
| Total APCD patients | 6,929,576 | `COUNT(DISTINCT mi_person_key)` from `medical_raw.medical_partitioned`, years 2016–2019 | `scripts/apcd_total_count2.py` |
| Training cohort — cases | 26,710 | Unique `mi_person_key` with `target=1` in train final_features (opioid_ed, all bands) | `scripts/count_train_rows.py` → `cohort_counts_train.json` |
| Training cohort — controls | 180,640 | Unique `mi_person_key` with `target=0` in train final_features (opioid_ed, all bands) | `scripts/count_train_rows.py` → `cohort_counts_train.json` |
| Brier score | 0.0070–0.0509 (opioid_ed) | `mean((y_true − y_prob)²)` on 2019 holdout | `scripts/compute_brier_ici.py` → `brier_ici_results.json` |
| ICI | 0.1084–0.1635 (opioid_ed) | Mean absolute calibration deviation (10-bin) on 2019 holdout | `scripts/compute_brier_ici.py` → `brier_ici_results.json` |
| DTW cluster 1 (Rapid-Onset) | n=5,481 (21% of cases) | Span from first to last pre-index event < 9 months; extrapolated to full 26,710 cases | `scripts/dtw_thresholds.py` |
| DTW cluster 2 (Chronic-Escalation) | n=21,229 (79% of cases) | Span ≥ 9 months; extrapolated to full 26,710 cases | `scripts/dtw_thresholds.py` |
| DTW threshold | 9 months (274 days) | Elbow criterion on span distribution (p25=297 days ≈ 9.8 months) | `scripts/dtw_thresholds.py` |

---

## Detailed Metric Definitions

### 1. Total APCD Patient Count (6,929,576)
- **Definition:** Unique patients with any APCD medical claim, years 2016–2019.
- **Calculation:**
  ```sql
  SELECT COUNT(DISTINCT mi_person_key) AS n_patients
  FROM medical_raw.medical_partitioned
  WHERE CAST(event_year AS INTEGER) BETWEEN 2016 AND 2019
  ```
- **Script reference:** `scripts/apcd_total_count2.py`; Athena WorkGroup: `APCD`.

### 2. Opioid ED Training Cohort Counts
- **Definition:** Matched case-control cohort from 2016–2018 APCD.
  - **Cases:** Adults with ≥1 opioid-related ED visit (ICD-10 F11.xx, T40.xx); one index event per patient.
  - **Controls:** 5:1 matched on age band (±2 years) and sex; no opioid codes.
- **Calculation:** `COUNT(mi_person_key) WHERE target=1/0` in train CSV.
- **Script reference:** `scripts/count_train_rows.py` → `cohort_counts_train.json[opioid_ed]`.

### 3. Brier Score (opioid_ed)
- **Definition:** Mean squared probability error on 2019 temporal holdout. Per-bin models aggregated.
- **Calculation:**
  ```python
  brier = sklearn.metrics.brier_score_loss(y_true, y_prob)
  ```
  Run over all event-density bins (low/medium/high/extreme), combined.
- **Script reference:** `scripts/compute_brier_ici.py` → `brier_ici_results.json[opioid_ed][band]["brier"]`.

### 4. ICI — Integrated Calibration Index
- **Definition:** Reliability of predicted probabilities; 0 = perfect calibration.
- **Calculation:**
  ```python
  frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
  ICI = mean(abs(frac_pos - mean_pred))
  ```
- **Script reference:** `scripts/compute_brier_ici.py` `ici()` function → `brier_ici_results.json[opioid_ed][band]["ici"]`.

### 5. DTW Cluster Sizes
- **Definition:** k=2 temporal trajectory clusters identified by DTW distance matrix + Ward-linkage hierarchical clustering on 12-month pre-index event sequences.
- **Cluster 1 — Rapid-Onset:** First-to-last event span < 9 months (p25 elbow).
- **Cluster 2 — Chronic-Escalation:** First-to-last event span ≥ 9 months.
- **Calculation:**
  1. Load `event_intervals_{cohort}_{band}.parquet` from `gold/dtw_filter/opioid_ed/{band}/`.
  2. Per case patient: `span_days = max(event_date) - min(event_date)`.
  3. Classify: `span_days < 9*30.4` → Rapid-Onset.
  4. Proportions from DTW-matched cases (82% coverage); extrapolated to full 26,710.
- **Script reference:** `scripts/dtw_thresholds.py`; individual bands: `scripts/dtw_cluster_sizes.py`, `scripts/dtw_13_24.py`.

### 6. MCCV Feature Screening
- **Definition:** 50+ random train/test splits on 2016–2018; features retained at median recall-weighted rank ≥ Q25.
- **Script reference:** `6_final_model/build_final_cohort_model_features.py`.

---

## Data Sources
| Source | Location |
|--------|----------|
| Full APCD (Athena) | `medical_raw.medical_partitioned` (Glue catalog) |
| Training cohort CSV | `gold/final_model/opioid_ed/{band}/{cohort}_{ab}_train_final_features_no_leakage.csv` |
| Test cohort parquet | `gold/final_model/opioid_ed/{band}/inputs/model_test/final_features.parquet` |
| DTW event intervals | `gold/dtw_filter/opioid_ed/{band}/event_intervals_opioid_ed_{ab}.parquet` |
| Per-bin models | `gold/final_model/opioid_ed/{band}/bin_models/{bin}/catboost_model.cbm` |
| Cohort counts | `cohort_counts_train.json`, `cohort_counts_test.json` (manuscript root) |
| Brier/ICI | `brier_ici_results.json` (manuscript root) |

## Cohort Age Bands (opioid_ed)
0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114
