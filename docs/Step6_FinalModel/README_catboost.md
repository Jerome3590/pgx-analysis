# CatBoost Feature Engineering Documentation

## Overview
This document describes how to use the outputs of the FP-Growth global pipeline as input features for CatBoost machine learning models.

---

## 1. Input Format from FP-Growth


The main input is a table with the following columns:

| drug_name         | global_drug_encoded_name | support | trend | ...additional features... |
|-------------------|-------------------------|---------|-------|--------------------------|
| ACETAMINOPHEN     | X000000000000001        | 0.85    | 0.02  | ...                      |
| IBUPROFEN         | X000000000000002        | 0.67    | -0.01 | ...                      |

- **drug_name**: The original drug name as found in the patient data.
- **global_drug_encoded_name**: The unique encoding assigned to each drug by the FP-Growth pipeline.
- **support, trend, ...**: Additional FP-Growth metrics/features.

- **Source:** `pgx_pipeline/fpgrowth_analysis/global_itemsets/` and `pgx_pipeline/fpgrowth_analysis/processed_itemsets/`
- **How to use:** Join this table to each patient's drug list on `drug_name` to create feature vectors, then use `global_drug_encoded_name` for encoding.

---

## 2. Feature Engineering Steps


1. **Load the global encoding map:**
   - Use the output from FP-Growth to map each `drug_name` to its `global_drug_encoded_name` and associated features.
2. **Transform patient drug lists:**
   - For each patient, join their drug list on `drug_name` to retrieve encodings and features.
3. **Create feature columns:**
   - Aggregate or concatenate `global_drug_encoded_name` values (and optionally other features) for CatBoost input.

---

## 3. Example Workflow

```python
from fpgrowth_analysis.run_fpgrowth import load_global_encoding_map, create_fpgrowth_logger

logger = create_fpgrowth_logger()
encoding_map = load_global_encoding_map(logger)

df['encoded_drug_features'] = df['drug_encodings'].apply(lambda encodings: ','.join(encodings) if encodings else '')
def encode_patient_drugs(drug_list, encoding_map):
    """
    For each drug in the patient's list, return the encoded name (or a default if missing).
    """
    return [encoding_map.get(drug, "X000000000000000") for drug in drug_list]

# Optionally, also keep the original drug names for reference or additional features
df['drug_names'] = df['drug_list']
df['drug_encodings'] = df['drug_list'].apply(lambda drugs: encode_patient_drugs(drugs, encoding_map))
df['encoded_drug_features'] = df['drug_encodings'].apply(lambda encodings: ','.join(encodings) if encodings else '')
```

---

# Additional Drug Name Features (Linguistic Feature Engineering)
For each drug name, you can compute features such as:
- Number of vowels
- Number of consonants
- Name length
- Starting character/index
These can be aggregated (mean, sum, etc) for each patient as extra features for CatBoost.

```python
import re

def count_vowels(name):
    return len(re.findall(r'[AEIOUaeiou]', name))

def count_consonants(name):
    return len(re.findall(r'[BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz]', name))

def get_name_length(name):
    return len(name)

def get_starts_with(name):
    return name[0] if name else ''

# Add linguistic features for each drug in the patient's list
df['drug_vowel_counts'] = df['drug_list'].apply(lambda drugs: [count_vowels(d) for d in drugs])
df['drug_consonant_counts'] = df['drug_list'].apply(lambda drugs: [count_consonants(d) for d in drugs])
df['drug_name_lengths'] = df['drug_list'].apply(lambda drugs: [get_name_length(d) for d in drugs])
df['drug_starts_with'] = df['drug_list'].apply(lambda drugs: [get_starts_with(d) for d in drugs])

# Optionally, aggregate these features (e.g., mean, sum, max) for use in CatBoost
df['mean_drug_name_length'] = df['drug_name_lengths'].apply(lambda x: sum(x)/len(x) if x else 0)
df['total_vowels'] = df['drug_vowel_counts'].apply(lambda x: sum(x) if x else 0)
df['total_consonants'] = df['drug_consonant_counts'].apply(lambda x: sum(x) if x else 0)
```

---

## 4. CatBoost Integration

- Use `encoded_drug_features` as a categorical feature in CatBoost.
- Combine with other features (age, cohort, etc.) as needed.

```python
from catboost import CatBoostClassifier, Pool

categorical_features = ['encoded_drug_features', 'age_band', 'cohort_type']
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, cat_features=categorical_features, verbose=100)
model.fit(train_pool)
```

---


## 5. Output Layout for CatBoost (Wide Table)



Each row: one patient
Columns:
- `drug_names`: List of original drug names for the patient
- `encoded_drug_features`: Concatenated string of encoded drug names (for CatBoost)
- Additional engineered features (support, trend, etc. if used)

**Best Practice:**
- Join the patient's drug list to the wide encoding table (one row per drug) to retrieve all features (encoding, linguistics, patterns, FP-Growth metrics).
- Aggregate or concatenate as needed for CatBoost input.

Example:

| patient_id | drug_names                | encoded_drug_features         | ... |
|------------|---------------------------|-------------------------------|-----|
| 1001       | [ACETAMINOPHEN, IBUPROFEN]| X000000000000001,X000000000000002 | ... |

---


## Drug Pattern Lookup Table from Itemsets (Wide Table)

The pipeline also creates a drug pattern lookup table from frequent itemsets, which can be used for additional feature engineering.

- **Location:** `pgx_pipeline/fpgrowth_analysis/processed_itemsets/itemset_lookup_table.json`
- **Structure:** Maps each unique itemset (combination of drugs) to a unique pattern ID, e.g. `Drug_Pattern001`, `Drug_Pattern002`, ...
- **Example:**

```json
{
  "ACETAMINOPHEN": "Drug_Pattern001",
  "ACETAMINOPHEN,IBUPROFEN": "Drug_Pattern002",
  "ACETAMINOPHEN,ASPIRIN": "Drug_Pattern003"
}
```


### How to Use
- For each patient's drug list, match the set (or subsets) to the lookup table to assign pattern IDs as features.
- These pattern IDs can be used as categorical features in CatBoost, or aggregated (e.g., count of patterns per patient).
- Always join to the wide encoding table for additional features.

### Example Usage

```python
import json

# Load the lookup table
with open('pgx_pipeline/fpgrowth_analysis/processed_itemsets/itemset_lookup_table.json') as f:
    pattern_lookup = json.load(f)

def get_pattern_ids(drug_list, pattern_lookup):
    # Join drugs alphabetically for matching
    key = ','.join(sorted(drug_list))
    return pattern_lookup.get(key, None)

# Assign pattern IDs to each patient
# (Assume df['drug_list'] is a list of drugs for each patient)
df['drug_pattern_id'] = df['drug_list'].apply(lambda drugs: get_pattern_ids(drugs, pattern_lookup))
```

---


## Final Feature Layout: FP-Growth to CatBoost (Wide Table)

The following features are recommended for each patient and each drug_name, as input to the CatBoost model:


| Column              | Type      | Description                                      |
|---------------------|-----------|--------------------------------------------------|
| mi_person_key       | string/int| Patient identifier                               |
| drug_name           | string    | Original drug name                               |
| encoded_drug_name   | int/str   | Encoded drug name (21-char string or integer)    |
| drug_pattern        | string    | Pattern ID from itemset lookup (e.g. hash/label) |
| first_letter_index  | int       | Index of first letter (A=1, ..., Z=26)           |
| length              | int       | Length of drug name                              |
| syllables           | int       | Number of syllables in drug name                 |
| consonants          | int       | Number of consonants in drug name                |
| support             | float     | FP-Growth support metric                         |
| confidence          | float     | FP-Growth confidence metric                      |
| certainty           | float     | FP-Growth certainty metric                       |

**Wide Table:**
- One row per drug, with all features and summary metrics.
- Join to patient drug lists for feature engineering.

- `encoded_drug_name` can be stored as a string (21-char) or parsed to integer features using the parse function.
- `drug_pattern` is a categorical label or hash from the itemset lookup table.
- All linguistic features (first_letter_index, length, syllables, consonants) can be extracted from the encoding or computed directly.
- Support, confidence, and certainty are typically floats (0-1).

### Example (per drug per patient):

| mi_person_key | drug_name    | encoded_drug_name     | drug_pattern      | first_letter_index | length | syllables | consonants | support | confidence | certainty |
|---------------|-------------|----------------------|-------------------|-------------------|--------|-----------|------------|---------|------------|-----------|
| 1001          | ACETAMINOPHEN| 001120507085092078    | Drug_Pattern001   | 1                 | 12     | 5         | 7          | 0.85    | 0.92       | 0.78      |
| 1001          | IBUPROFEN    | 009080404067081070    | Drug_Pattern002   | 9                 | 8      | 4         | 4          | 0.67    | 0.81       | 0.70      |


*This document is focused on CatBoost feature engineering. For FP-Growth logic and outputs, see `FpGROWTH_README.md`.*
