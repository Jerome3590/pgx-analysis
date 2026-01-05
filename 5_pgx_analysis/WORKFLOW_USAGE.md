# How Drug-Gene Mappings Are Used in PGx Analysis

## Complete Workflow Overview

The drug-gene mappings created by `map_drugs_to_genes.py` flow through a multi-step pipeline to create patient-level features for machine learning models.

## Step-by-Step Usage

### Step 1: Drug-Gene Mapping (`map_drugs_to_genes.py`)
**Input:**
- Drugs from FP-Growth itemsets (e.g., AMOXICILLIN, AZITHROMYCIN, ONDANSETRON)
- CPIC pairs database (261 drugs, 416 drug-gene pairs)
- PubMed search (optional, with `--use-pubmed`)

**Output:**
- `drug_gene_mappings.csv` with columns:
  - `drug_name`: Original drug name
  - `cpic_drug_name`: Matched CPIC name (if fuzzy matched)
  - `gene`: Associated pharmacogene (e.g., CYP2D6, CYP2C19)
  - `evidence_level`: CPIC or PubMed
  - `guideline_url`: Link to CPIC guideline or PubMed article
  - `source`: CPIC_PAIRS_FUZZY_MATCHED or PubMed

**Example Output:**
```
drug_name,cpic_drug_name,gene,evidence_level,source
AMOXICILLIN,AMOXICILLIN,CYP2C19,PubMed,PubMed
AZITHROMYCIN,AZITHROMYCIN,CYP3A4,PubMed,PubMed
ONDANSETRON ODT,ondansetron,CYP2D6,CPIC,CPIC_PAIRS_FUZZY_MATCHED
```

---

### Step 2: Allele Frequency Integration (`add_allele_frequencies.py`)
**Input:**
- Drug-gene mappings from Step 1
- CPIC API (or fallback sources) for allele frequencies

**Process:**
- For each gene in the mappings, fetch population-level allele frequencies
- Store frequencies for all populations: Global, AFR, AMR, EAS, EUR, SAS

**Output:**
- `allele_frequencies.csv` with columns:
  - `gene`: Gene symbol
  - `allele_frequency_global`: Global average frequency
  - `allele_frequency_afr`: African/African American ancestry frequency
  - `allele_frequency_amr`: Latino/Admixed American ancestry frequency
  - `allele_frequency_eas`: East Asian ancestry frequency
  - `allele_frequency_eur`: European ancestry frequency
  - `allele_frequency_sas`: South Asian ancestry frequency
  - `frequency_source`: CPIC_API or other source

**Example Output:**
```
gene,allele_frequency_global,allele_frequency_afr,allele_frequency_eur,...
CYP2D6,0.15,0.12,0.18,...
CYP3A4,0.25,0.20,0.30,...
```

---

### Step 3: Patient-Level Feature Creation (`create_pgx_features_patient_level.py`)
**Input:**
- Drug-gene mappings (Step 1)
- Allele frequencies (Step 2)
- Patient drug exposures from `model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`

**Process:**
1. **Extract Patient Drug Exposures:**
   ```sql
   SELECT DISTINCT mi_person_key, drug_name
   FROM model_events
   WHERE target = 1 AND drug_name IS NOT NULL
   ```

2. **Merge with Drug-Gene Mappings:**
   - Join patient drugs → drug-gene mappings → genes
   - Each patient-drug-gene combination gets allele frequencies

3. **Aggregate to Patient Level:**
   - For each patient, aggregate across all their drug-gene pairs:
     - Mean, max, sum of allele frequencies (per population)
     - Count of drugs with PGx mappings
     - Count of unique genes covered

**Output:**
- `pgx_features_{cohort}_{age_band}.csv` with patient-level features:
  - `mi_person_key`: Patient identifier
  - `pgx_risk_global_mean/max/sum`: Aggregated global frequency metrics
  - `pgx_risk_afr_mean/max/sum`: Aggregated AFR frequency metrics
  - `pgx_risk_amr_mean/max/sum`: Aggregated AMR frequency metrics
  - `pgx_risk_eas_mean/max/sum`: Aggregated EAS frequency metrics
  - `pgx_risk_eur_mean/max/sum`: Aggregated EUR frequency metrics
  - `pgx_risk_sas_mean/max/sum`: Aggregated SAS frequency metrics
  - `pgx_drugs_with_mappings`: Number of patient's drugs with PGx data
  - `pgx_genes_covered`: Number of unique PGx genes involved

**Example Output:**
```
mi_person_key,pgx_risk_global_mean,pgx_risk_global_max,pgx_drugs_with_mappings,pgx_genes_covered
12345,0.18,0.25,2,5
67890,0.15,0.20,1,3
```

---

### Step 4: Feature Merging (`add_pgx_features_to_model_data.py`)
**Input:**
- PGx features from Step 3
- Other feature sets (FP-Growth, BupaR, DTW features)

**Process:**
- Merge PGx features with other feature sets using `mi_person_key`
- Create final feature matrix ready for model training

**Output:**
- `pgx_added_features_{cohort}_{age_band}.csv`
- Ready to join with `model_data` for final model training

---

## How Features Are Used in Models

### Feature Variants for Model Selection

The pipeline creates **multiple feature variants** to let the model determine which approach is most predictive:

1. **Global Frequency Features** (`pgx_risk_global_*`):
   - Uses global average allele frequencies
   - Baseline approach, population-agnostic

2. **Population-Specific Features** (`pgx_risk_afr/amr/eas/eur/sas_*`):
   - Uses population-specific frequencies
   - Model can evaluate if population-specific features improve predictions

3. **Aggregation Methods** (`mean`, `max`, `sum`):
   - Different ways to aggregate across multiple drug-gene pairs
   - Model can select which aggregation is most informative

### Model Training Usage

During model training:
- **Feature Importance**: Model will show which PGx features (if any) improve predictions
- **Feature Selection**: Model can automatically select the most predictive frequency variant
- **Interpretability**: Features link back to specific drugs, genes, and evidence sources

### Example Model Features

For a patient who took AMOXICILLIN and AZITHROMYCIN:

```
Patient Features:
- pgx_risk_global_mean: 0.18 (average across all drug-gene pairs)
- pgx_risk_global_max: 0.25 (highest frequency variant)
- pgx_risk_eur_mean: 0.20 (European-specific average)
- pgx_drugs_with_mappings: 2 (AMOXICILLIN, AZITHROMYCIN)
- pgx_genes_covered: 13 (CYP2C19, CYP3A4, CYP2D6, etc.)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Drug-Gene Mapping                                   │
│ Input: FP-Growth drugs (9 drugs)                            │
│ Output: 19 drug-gene pairs (3 drugs → 13 genes)            │
│ Sources: CPIC (2 pairs) + PubMed (17 pairs)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Allele Frequency Integration                         │
│ Input: 13 unique genes from Step 1                          │
│ Output: Population frequencies for each gene                │
│ (Global, AFR, AMR, EAS, EUR, SAS)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Patient-Level Feature Creation                      │
│ Input:                                                      │
│  - Drug-gene mappings (Step 1)                             │
│  - Allele frequencies (Step 2)                              │
│  - Patient drug exposures (model_data)                      │
│ Output: Patient-level PGx risk features                     │
│ (per patient: mean/max/sum across drug-gene pairs)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Feature Merging                                     │
│ Input: PGx features + other feature sets                    │
│ Output: Final feature matrix ready for model training        │
└─────────────────────────────────────────────────────────────┘
```

---

## Current Status for Cohort 1, Age Band 0-12

✅ **Step 1 Complete**: 19 drug-gene mappings
- 3 drugs mapped (AMOXICILLIN, AZITHROMYCIN, ONDANSETRON ODT)
- 13 unique genes identified
- Sources: CPIC (2) + PubMed (17)

⏳ **Step 2 Pending**: Allele frequency integration
- Need to run `add_allele_frequencies.py`

⏳ **Step 3 Pending**: Patient-level feature creation
- Need to run `create_pgx_features_patient_level.py`

⏳ **Step 4 Pending**: Feature merging
- Need to run `add_pgx_features_to_model_data.py`

---

## Key Benefits

1. **Comprehensive Coverage**: Combines curated CPIC guidelines with broader PubMed literature
2. **Population-Aware**: Provides multiple frequency variants for model evaluation
3. **Evidence-Based**: Each mapping includes source (CPIC/PubMed) and references
4. **Model-Driven**: Multiple feature variants let the model select the most predictive approach
5. **Interpretable**: Features trace back to specific drugs, genes, and evidence sources

