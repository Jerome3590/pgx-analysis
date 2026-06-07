# Scenario SHAP/FFA Audit Summary

- Combined files analyzed: **63**
- Ranked feature rows: **7876**
- Top-20 rows: **1252**

## Top-20 feature family counts

- **drug**: 822
- **cpt**: 197
- **pgx**: 126
- **icd**: 69
- **utilization_temporal**: 38

## Most recurrent top-20 features

- **pgx_num_drugs** (pgx): 60 bins, avg combined 0.9584
- **pgx_num_cpic_drugs** (pgx): 58 bins, avg combined 0.1715
- **item_drug_AMOXICILLIN** (drug): 51 bins, avg combined 0.0317
- **item_drug_PREDNISONE** (drug): 45 bins, avg combined 0.0279
- **item_drug_AZITHROMYCIN** (drug): 44 bins, avg combined 0.0328
- **item_drug_CEPHALEXIN** (drug): 41 bins, avg combined 0.0373
- **item_drug_OMEPRAZOLE** (drug): 40 bins, avg combined 0.0189
- **item_drug_ALPRAZOLAM** (drug): 32 bins, avg combined 0.0386
- **item_drug_GABAPENTIN** (drug): 30 bins, avg combined 0.1210
- **item_drug_LISINOPRIL** (drug): 27 bins, avg combined 0.0302
- **item_drug_IBUPROFEN** (drug): 26 bins, avg combined 0.0147
- **item_drug_METRONIDAZOLE** (drug): 26 bins, avg combined 0.0143
- **item_drug_NAPROXEN** (drug): 22 bins, avg combined 0.0064
- **item_drug_FUROSEMIDE** (drug): 21 bins, avg combined 0.0633
- **item_drug_SIMVASTATIN** (drug): 21 bins, avg combined 0.0211
- **item_drug_BENZONATATE** (drug): 20 bins, avg combined 0.0152
- **item_drug_MELOXICAM** (drug): 20 bins, avg combined 0.0030
- **item_drug_LORAZEPAM** (drug): 18 bins, avg combined 0.0386
- **item_drug_FLUCONAZOLE** (drug): 18 bins, avg combined 0.0095
- **item_drug_HYDROCHLOROTHIAZIDE** (drug): 18 bins, avg combined 0.0063

## Physical therapy-like CPT trace

- PT-like rows across all ranks: **124**
- Bins containing PT-like rows: **16**
- PT-like rows in top 20: **3**

### Highest-ranked / highest-score PT-like rows

- **item_cpt_97110**: opioid_ed/55_64/extreme, rank 34, combined 0.1081, SHAP 0.2018, FFA 0.0144
- **item_cpt_97112**: opioid_ed/65_74/extreme, rank 17, combined 0.1051, SHAP 0.1902, FFA 0.0200
- **item_cpt_97162**: opioid_ed/65_74/extreme, rank 27, combined 0.0871, SHAP 0.1628, FFA 0.0113
- **item_cpt_97140**: opioid_ed/65_74/extreme, rank 39, combined 0.0698, SHAP 0.1283, FFA 0.0114
- **item_cpt_97530**: opioid_ed/55_64/extreme, rank 50, combined 0.0698, SHAP 0.1339, FFA 0.0057
- **item_cpt_97112**: opioid_ed/65_74/high, rank 13, combined 0.0694, SHAP 0.1257, FFA 0.0132
- **item_cpt_97140**: opioid_ed/65_74/high, rank 21, combined 0.0578, SHAP 0.1062, FFA 0.0094
- **item_cpt_97110**: opioid_ed/55_64/high, rank 29, combined 0.0572, SHAP 0.1067, FFA 0.0076
- **item_cpt_97110**: opioid_ed/85_114/extreme, rank 21, combined 0.0468, SHAP 0.0853, FFA 0.0083
- **item_cpt_97110**: opioid_ed/85_114/high, rank 23, combined 0.0442, SHAP 0.0805, FFA 0.0079
- **item_cpt_97162**: opioid_ed/65_74/high, rank 32, combined 0.0403, SHAP 0.0754, FFA 0.0052
- **item_cpt_97530**: opioid_ed/65_74/extreme, rank 78, combined 0.0351, SHAP 0.0685, FFA 0.0017
- **item_cpt_97110**: opioid_ed/65_74/extreme, rank 87, combined 0.0307, SHAP 0.0568, FFA 0.0047
- **item_cpt_97112**: opioid_ed/55_64/extreme, rank 91, combined 0.0293, SHAP 0.0571, FFA 0.0015
- **item_cpt_97116**: opioid_ed/85_114/extreme, rank 28, combined 0.0293, SHAP 0.0565, FFA 0.0021

## Interpretation note

The current regenerated scenario outputs support PT-related utilization as a secondary, age-stratified opioid_ed signal, not as a broad top-ranked absence-of-PT rule. The current features encode CPT presence rather than explicit absence such as no_physical_therapy.