# FFA Analysis - Remaining Cohorts

Analysis of cohorts: opioid_ed/55-64, non_opioid_ed/65-74, non_opioid_ed/75-84, non_opioid_ed/85-94


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

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.000000 | n_events|pgx_num_drugs |
| 2 | 1.000000 | -1.000000 | item_drug_GABAPENTIN|n_events |
| 3 | 1.000000 | -1.000000 | item_icd_M545|n_events |
| 4 | 1.000000 | -1.000000 | item_icd_G894|n_events |
| 5 | 1.000000 | -1.000000 | item_icd_E785|n_events |
| 6 | 1.000000 | -1.000000 | item_drug_LISINOPRIL|n_events |
| 7 | 1.000000 | -1.000000 | item_drug_LEVOTHYROXINE_SODIUM|n_events |
| 8 | 1.000000 | -1.000000 | item_drug_FLUTICASONE_PROPIONATE|n_events |
| 9 | 1.000000 | -1.000000 | item_drug_CEPHALEXIN|n_events |
| 10 | 1.000000 | -1.000000 | item_drug_Unknown|n_events |
| 11 | 1.000000 | -1.000000 | item_drug_SULFAMETHOXAZOLE_TRIMETHO|n_events |
| 12 | 1.000000 | -1.000000 | item_drug_PANTOPRAZOLE_SODIUM|n_events |
| 13 | 1.000000 | -1.000000 | item_drug_METFORMIN_HYDROCHLORIDE|n_events |
| 14 | 1.000000 | -1.000000 | item_icd_R918|n_events |
| 15 | 1.000000 | -1.000000 | item_drug_OXYCODONE_ACETAMINOPHEN|n_events |
| 16 | 1.000000 | -1.000000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO|n_events |
| 17 | 1.000000 | -1.000000 | item_drug_HYDROCODONE_BITARTRATE_AC|n_events |
| 18 | 1.000000 | -1.000000 | item_drug_OMEPRAZOLE|n_events |
| 19 | 1.000000 | -1.000000 | item_drug_IBUPROFEN|n_events |
| 20 | 1.000000 | -1.000000 | item_drug_AMLODIPINE_BESYLATE|n_events |

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

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.940000 | item_drug_LISINOPRIL|n_events|pgx_num_drugs |
| 2 | 1.000000 | -1.940000 | item_drug_GABAPENTIN|item_drug_LISINOPRIL|n_events |
| 3 | 1.000000 | -1.940000 | item_drug_ATORVASTATIN_CALCIUM|n_events|pgx_num_drugs |
| 4 | 1.000000 | -1.940000 | item_drug_GABAPENTIN|n_events|pgx_num_drugs |
| 5 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE|n_events|pgx_num_drugs |
| 6 | 1.000000 | -1.940000 | n_events|pgx_num_cpic_drugs|pgx_num_drugs |
| 7 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE|item_drug_ATORVASTATIN_CALCIUM|n_events |
| 8 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE|item_drug_LISINOPRIL|n_events |
| 9 | 1.000000 | -0.940000 | item_drug_LEVOTHYROXINE_SODIUM|item_drug_LISINOPRIL|n_events |
| 10 | 1.000000 | -0.940000 | item_drug_ATORVASTATIN_CALCIUM|item_drug_Unknown|n_events |
| 11 | 1.000000 | -0.940000 | item_drug_OXYCODONE_HYDROCHLORIDE|n_events|pgx_num_drugs |
| 12 | 1.000000 | -0.940000 | item_drug_LISINOPRIL|item_drug_TAMSULOSIN_HYDROCHLORIDE|n_events |
| 13 | 1.000000 | -0.940000 | item_drug_HYDROCHLOROTHIAZIDE|item_drug_LISINOPRIL|n_events |
| 14 | 1.000000 | -0.940000 | item_drug_CYCLOBENZAPRINE_HYDROCHLO|n_events|pgx_num_drugs |
| 15 | 1.000000 | -0.940000 | item_drug_SPIRONOLACTONE|n_events|pgx_num_drugs |
| 16 | 1.000000 | -0.940000 | item_drug_DOXYCYCLINE_HYCLATE|n_events|pgx_num_drugs |
| 17 | 1.000000 | -0.940000 | item_drug_VITAMIN_D|n_events|pgx_num_drugs |
| 18 | 1.000000 | -0.940000 | item_drug_TAMSULOSIN_HYDROCHLORIDE|n_events|pgx_num_drugs |
| 19 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE|n_events|pgx_num_cpic_drugs |
| 20 | 1.000000 | -0.940000 | item_drug_ROSUVASTATIN_CALCIUM|n_events|pgx_num_drugs |

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

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.940000 | item_drug_LOSARTAN_POTASSIUM|n_events|pgx_num_drugs |
| 2 | 1.000000 | -1.940000 | item_drug_ATORVASTATIN_CALCIUM|item_drug_LISINOPRIL|n_events |
| 3 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE|n_events|pgx_num_drugs |
| 4 | 1.000000 | -1.940000 | item_drug_PREDNISONE|n_events|pgx_num_drugs |
| 5 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE|n_events|pgx_num_drugs |
| 6 | 1.000000 | -1.940000 | item_drug_GABAPENTIN|n_events|pgx_num_drugs |
| 7 | 1.000000 | -0.940000 | item_drug_TAMSULOSIN_HYDROCHLORIDE|n_events|pgx_num_drugs |
| 8 | 1.000000 | -0.940000 | item_drug_LEVOTHYROXINE_SODIUM|item_drug_WARFARIN_SODIUM|n_events |
| 9 | 1.000000 | -0.940000 | item_drug_ROSUVASTATIN_CALCIUM|n_events|pgx_num_drugs |
| 10 | 1.000000 | -0.940000 | item_drug_SIMVASTATIN|n_events|pgx_num_drugs |
| 11 | 1.000000 | -0.940000 | item_drug_CLOPIDOGREL|n_events|pgx_num_drugs |
| 12 | 1.000000 | -0.940000 | item_drug_HYDROCODONE_BITARTRATE_AC|n_events|pgx_num_drugs |
| 13 | 1.000000 | -0.940000 | item_drug_ELIQUIS|n_events|pgx_num_drugs |
| 14 | 1.000000 | -0.940000 | item_drug_CIPROFLOXACIN_HYDROCHLORI|n_events|pgx_num_drugs |
| 15 | 1.000000 | -0.940000 | item_drug_OMEPRAZOLE|n_events|pgx_num_drugs |
| 16 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE|item_drug_PANTOPRAZOLE_SODIUM|n_events |
| 17 | 1.000000 | -0.940000 | item_drug_GABAPENTIN|item_drug_HYDROCODONE_BITARTRATE_AC|n_events |
| 18 | 1.000000 | -0.940000 | item_drug_FINASTERIDE|n_events|pgx_num_drugs |
| 19 | 1.000000 | -0.940000 | item_drug_CEFDINIR|n_events|pgx_num_drugs |
| 20 | 1.000000 | -0.940000 | item_drug_VALSARTAN|n_events|pgx_num_drugs |

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

### Top 20 Interactions

| Rank | Combined Causal | Interaction Effect | Features |
|------|----------------|-------------------|----------|
| 1 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE|item_drug_LEVOTHYROXINE_SODIUM|n_events |
| 2 | 1.000000 | -1.940000 | item_drug_LEVOTHYROXINE_SODIUM|item_drug_OMEPRAZOLE|n_events |
| 3 | 1.000000 | -1.940000 | item_drug_METOPROLOL_SUCCINATE_ER|n_events|pgx_num_drugs |
| 4 | 1.000000 | -1.940000 | item_drug_LOSARTAN_POTASSIUM|n_events|pgx_num_drugs |
| 5 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE|n_events|pgx_num_cpic_drugs |
| 6 | 1.000000 | -1.940000 | item_drug_OMEPRAZOLE|n_events|pgx_num_drugs |
| 7 | 1.000000 | -1.940000 | item_drug_FUROSEMIDE|n_events|pgx_num_drugs |
| 8 | 1.000000 | -1.940000 | item_drug_AMLODIPINE_BESYLATE|n_events|pgx_num_drugs |
| 9 | 1.000000 | -1.940000 | n_events|pgx_num_cpic_drugs|pgx_num_drugs |
| 10 | 1.000000 | -1.940000 | item_drug_ATORVASTATIN_CALCIUM|item_drug_FUROSEMIDE|n_events |
| 11 | 1.000000 | -1.940000 | item_drug_OMEPRAZOLE|n_events|pgx_num_cpic_drugs |
| 12 | 1.000000 | -0.940000 | item_drug_FUROSEMIDE|item_drug_PREDNISONE|n_events |
| 13 | 1.000000 | -0.940000 | item_drug_PANTOPRAZOLE_SODIUM|n_events|pgx_num_drugs |
| 14 | 1.000000 | -0.940000 | item_drug_AMOXICILLIN_CLAVULANATE_P|n_events|pgx_num_drugs |
| 15 | 1.000000 | -0.940000 | item_drug_SERTRALINE_HCL|n_events|pgx_num_drugs |
| 16 | 1.000000 | -0.940000 | item_drug_HYDROCODONE_BITARTRATE_AC|n_events|pgx_num_drugs |
| 17 | 1.000000 | -0.940000 | item_drug_ELIQUIS|n_events|pgx_num_drugs |
| 18 | 1.000000 | -0.940000 | item_drug_MIRTAZAPINE|n_events|pgx_num_drugs |
| 19 | 1.000000 | -0.940000 | item_drug_PREDNISONE|n_events|pgx_num_drugs |
| 20 | 1.000000 | -0.940000 | item_drug_ISOSORBIDE_MONONITRATE_ER|n_events|pgx_num_drugs |

---