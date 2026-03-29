# CH_4 Metrics README
**Chapter:** Polypharmacy Synergy Detection and Causal Deprescribing Prioritization in Geriatric Patients  
**Manuscript:** `CH_4/ch04_psp.qmd` → CPT:PSP (Wiley)

---

## Metrics Summary

| Metric | Value | Calculation | Script |
|--------|-------|-------------|--------|
| Training cohort — cases | 1,182 | Unique `target=1` patients (non_opioid_ed, bands 65-74/75-84/85-114) | `scripts/count_train_rows.py` → `cohort_counts_train.json` |
| Training cohort — controls | 16,105 | Unique `target=0` patients (non_opioid_ed, same bands) | `scripts/count_train_rows.py` → `cohort_counts_train.json` |
| Brier score | 0.0070–0.0079 (non_opioid_ed) | `mean((y_true − y_prob)²)` on 2019 holdout | `scripts/compute_brier_ici.py` → `brier_ici_results.json` |
| ICI | 0.0707–0.2896 (non_opioid_ed) | Mean absolute calibration deviation (10-bin) | `scripts/compute_brier_ici.py` → `brier_ici_results.json` |
| PR-AUC | 0.991 ± 0.007 | Area under precision-recall curve, mean ± SD across bands | `6_final_model/run_final_model.py` |
| Synergistic DDI pairs | 115 | FFA pairs with IE > 1.0 AND bootstrap 95% CI > 0 (1,000 resamples) | `scripts/ffa_synergy_pairs.py` |
| High-risk triplets | 5,021 | Triplets with lift-based IE > 1.0 across geriatric bands | `scripts/ffa_synergy_pairs.py` |
| IR rank correlation (ρ) | 0.53–0.68 | Spearman ρ of IR scores across the three geriatric bands | `scripts/ir_rank_corr.py` |
| Z-code IQR — cases | 0.08 (0.00–0.29) | Median (Q25–Q75) of Z-code proportion in 30-day pre-index window | `scripts/zcode_30day.py` |
| Z-code IQR — controls | 0.09 (0.04–0.18) | Median (Q25–Q75) of Z-code proportion in 30-day pre-index window | `scripts/zcode_30day.py` |
| Z-code OR | 2.10 (95% CI 1.51–2.94) | Logistic regression OR per Z-code proportion quartile, adjusted for n_events + band | `scripts/zcode_30day.py` |

---

## Detailed Metric Definitions

### 1. Non-Opioid ED Cohort Counts
- **Definition:** Adults aged 65–114 years with ≥1 non-opioid ADE-related ED visit (HCG O11/P51), excluding opioid codes. Controls matched 5:1 within age band (±2 years) and sex.
- **Calculation:** `COUNT(DISTINCT mi_person_key) WHERE target=1/0` in train parquet.
- **Script reference:** `scripts/count_train_rows.py` → `cohort_counts_train.json[non_opioid_ed]`.

### 2. Brier Score and ICI (non_opioid_ed)
- **Brier:** `sklearn.metrics.brier_score_loss(y_true, y_prob)` on 2019 holdout, per-bin models aggregated.
- **ICI:** `mean(abs(frac_pos - mean_pred))` from 10-bin calibration curve.
- **Script reference:** `scripts/compute_brier_ici.py` → `brier_ici_results.json[non_opioid_ed]`.

### 3. FFA Interaction Effect (IE)
- **Definition:** Lift-based pairwise synergy score measuring co-occurrence of drug pair in ADE rules relative to independence baseline.
- **Calculation:**
  ```
  IE(f_i, f_j) = E[p̂(f_i=1, f_j=1)] - E[p̂(f_i=1, f_j=0)]
               - E[p̂(f_i=0, f_j=1)] + E[p̂(f_i=0, f_j=0)]
  ```
  IE > 0 = synergy; bootstrap 95% CI from 1,000 resamples of 2019 holdout.
- **Script reference:** `scripts/ffa_synergy_pairs.py`; source data: `gold/ffa_analysis/non_opioid_ed/{band_ab}/bin_models/{bin}/ffa_causal_factors.csv`.

### 4. Intervention Rate (IR)
- **Definition:** Expected risk reduction if drug `f_i` is removed (counterfactual substitution with training median).
- **Calculation:**
  ```
  IR(f_i) = (1/n) × Σ_k [p̂_k - p̂_k(f_i → median)]
  ```
  Normalized by population-level case prevalence. Range [0,1].
- **Script reference:** `scripts/extract_ffa_manuscript.py` (reads `causal_responsibility` column from FFA CSV); `scripts/ir_rank_corr.py` for cross-band correlation.

### 5. IR Rank Correlation (ρ = 0.53–0.68)
- **Definition:** Spearman rank correlation of per-drug IR scores across the three geriatric age bands (65-74, 75-84, 85-114).
- **Calculation:**
  ```python
  from scipy.stats import spearmanr
  rho, pval = spearmanr(ir_band1, ir_band2)
  ```
  Averaged across available FFA bins (low/medium/high/extreme) per band before correlating.
- **Script reference:** `scripts/ir_rank_corr.py` lines 63–78.

### 6. Z-Code Proportion
- **Definition:** Fraction of claims in the 30-day pre-index window with ICD-10 Z-code (Z00–Z99) in any diagnosis position. Proxy for routine clinical monitoring.
- **Calculation:**
  ```python
  z_prop = z_code_claims / total_claims
  ```
  Computed from `gold/cohorts_model_data/cohort_name=non_opioid_ed/age_band={b}/model_events.parquet`,
  filtering to events within `(first_o11_p_date - 30, first_o11_p_date)`.
- **Script reference:** `scripts/zcode_30day.py`.

### 7. Z-Code Logistic Regression OR
- **Definition:** Odds ratio for ADE ED visit per Z-code proportion quartile increment, adjusted for `n_events` and age band.
- **Model:** `logit(P(target=1)) = β₀ + β₁·z_q + β₂·n_events + Σ β_band`
  where `z_q` = quartile (1–4) of Z-code proportion.
- **Calculation:** MLE via `scipy.optimize.minimize` (L-BFGS-B), Fisher information SE for CI.
  `OR = exp(β₁)`, `95% CI = exp(β₁ ± 1.96·SE)`
- **Script reference:** `scripts/zcode_30day.py` lines 93–142; full Athena-based version: `scripts/zcode_athena.py`.

---

## Data Sources
| Source | Location |
|--------|----------|
| Model events (cases + controls) | `gold/cohorts_model_data/cohort_name=non_opioid_ed/age_band={b}/model_events.parquet` |
| Training features | `gold/final_model/non_opioid_ed/{band}/inputs/model_train/final_features.parquet` |
| Test features | `gold/final_model/non_opioid_ed/{band}/inputs/model_test/final_features.parquet` |
| FFA causal factors | `gold/ffa_analysis/non_opioid_ed/{band_ab}/bin_models/{bin}/ffa_causal_factors.csv` |
| Per-bin models | `gold/final_model/non_opioid_ed/{band}/bin_models/{bin}/catboost_model.cbm` |
| Brier/ICI | `brier_ici_results.json` (manuscript root) |
| FFA summary | `ffa_manuscript_data.json` (manuscript root) |

## Cohort Age Bands (non_opioid_ed — study bands)
65-74, 75-84, 85-114
