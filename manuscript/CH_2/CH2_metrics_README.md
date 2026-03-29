# CH_2 Metrics README
**Chapter:** Partition-First Architecture and Causal Consensus Filter for Scalable Pharmacoepidemiological Modeling  
**Manuscript:** `CH_2/ch02_psp.qmd` → CPT:PSP (Wiley)

---

## Metrics Summary

| Metric | Value | Calculation | Script |
|--------|-------|-------------|--------|
| AUROC (opioid_ed) | Per band, 2019 holdout | Area under ROC curve from ensemble predictions on model_test parquet | `scripts/compute_brier_ici.py` |
| PR-AUC (opioid_ed) | Per band, 2019 holdout | Area under precision-recall curve from ensemble predictions | `scripts/compute_brier_ici.py` |
| Brier score | 0.0070–0.0509 (opioid_ed) | `mean((y_true - y_prob)²)` over model_test holdout | `scripts/compute_brier_ici.py` → `brier_ici_results.json` |
| ICI | 0.1084–0.1635 (opioid_ed) | Mean absolute deviation of observed vs predicted proportions (10-bin calibration curve) | `scripts/compute_brier_ici.py` → `brier_ici_results.json` |
| MCCV splits | 50+ | Random 80/20 stratified splits on 2016–2018 training data | `6_final_model/run_final_model.py` (`n_runs` param) |
| Consensus features | 49% of total | Features selected by BOTH SHAP (rank ≥ Q75) AND FFA (CR ≥ 0.05, conf ≥ 0.70) | `scripts/consensus_filter_final.py` |
| SHAP-only features | 252 | Features above SHAP threshold but below FFA threshold | `scripts/consensus_filter_final.py` |
| FFA-only features | 33 | Features above FFA threshold but below SHAP threshold | `scripts/consensus_filter_final.py` |

---

## Detailed Metric Definitions

### 1. AUROC — Area Under ROC Curve
- **Definition:** Probability that the model ranks a random positive case above a random negative control.
- **Calculation:** `sklearn.metrics.roc_auc_score(y_true, y_prob)` on 2019 test holdout.
- **Stratification:** Computed per age band; ensemble aggregates across all event-density bins (low/medium/high/extreme).
- **Script reference:** `scripts/compute_brier_ici.py` lines 140–149; output to `brier_ici_results.json`.

### 2. PR-AUC — Precision-Recall AUC
- **Definition:** Area under the precision-recall curve; preferred metric for imbalanced classification.
- **Calculation:** `sklearn.metrics.average_precision_score(y_true, y_prob)` on 2019 holdout.
- **Stratification:** Per band; 5:1 case:control ratio addressed with `scale_pos_weight` (XGBoost) and `class_weight` (CatBoost).
- **Script reference:** `6_final_model/run_final_model.py` → `train_and_evaluate()`.

### 3. Brier Score
- **Definition:** Mean squared error between predicted probability and true outcome. Range [0,1]; lower = better calibrated.
- **Calculation:** `sklearn.metrics.brier_score_loss(y_true, y_prob)`
- **Script reference:** `scripts/compute_brier_ici.py` line 142; stored in `brier_ici_results.json[cohort][band]["brier"]`.

### 4. ICI — Integrated Calibration Index
- **Definition:** Mean absolute deviation between observed event rates and mean predicted probabilities across 10 equal-width bins.
- **Calculation:**
  ```python
  frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
  ICI = mean(abs(frac_pos - mean_pred))
  ```
- **Script reference:** `scripts/compute_brier_ici.py` lines 29–35 (`ici()` function); stored in `brier_ici_results.json[cohort][band]["ici"]`.

### 5. Consensus Filter — Feature Counts
- **Definition:** Three-way partition of model features into SHAP-only, FFA-only, and Consensus (both).
- **Thresholds:**
  - SHAP: feature rank ≥ 75th percentile across MCCV splits
  - FFA: `causal_responsibility ≥ 0.05` AND `rule_confidence ≥ 0.70`
- **Counts (opioid_ed, all bands combined):**
  - SHAP-only: **252** features
  - FFA-only: **33** features
  - Consensus (both): **49%** of total retained features
- **Script reference:** `scripts/consensus_filter_final.py`; counts reported by `scripts/get_consensus_counts.py`.

### 6. MCCV Feature Selection
- **Definition:** Feature stability score = fraction of MCCV splits in which the feature ranks ≥ 25th percentile.
- **Calculation:** `stability = sum(rank_i >= Q25) / n_splits` for each feature `i`.
- **Script reference:** `6_final_model/build_final_cohort_model_features.py` → `screen_features_mccv()`.

---

## Data Sources
| Source | Location |
|--------|----------|
| Training features | `gold/final_model/{cohort}/{band}/{cohort}_{ab}_train_final_features_no_leakage.csv` |
| Test features | `gold/final_model/{cohort}/{band}/inputs/model_test/final_features.parquet` |
| Per-bin models | `gold/final_model/{cohort}/{band}/bin_models/{bin}/catboost_model.cbm` |
| Brier/ICI results | `brier_ici_results.json` (manuscript root) |
| FFA causal factors | `gold/ffa_analysis/{cohort}/{band_ab}/bin_models/low/ffa_causal_factors.csv` |
| SHAP values | `gold/shap_analysis/{cohort}/{band_ab}/bin_models/{bin}/shap_values.parquet` |

## Cohorts and Age Bands
- **opioid_ed:** 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114
- **non_opioid_ed:** 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114
