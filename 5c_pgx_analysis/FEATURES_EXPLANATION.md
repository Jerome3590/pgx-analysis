# PGx Features Added to Model Data

## Overview

PGx features are **NOT directly based on patient race**. Instead, we create **multiple feature variants** (one for each population ancestry group) and let the **model determine** which features are most predictive.

## Feature Creation Process

### Step 1: Patient Drug Exposures
For each patient, we extract their drug exposures from `model_data`:
```
Patient 12345: [AMOXICILLIN, AZITHROMYCIN]
Patient 67890: [ONDANSETRON ODT]
```

### Step 2: Drug → Gene Mapping
Each drug is mapped to pharmacogenomic genes:
```
AMOXICILLIN → [CYP2C19, CYP2C9, CYP3A4, CYP2C8, HLA-B]
AZITHROMYCIN → [CYP2D6, CYP2C9, CYP3A4, CYP3A5, CYP2B6, CYP2C8, UGT1A1, SLCO1B1, ABCB1, ABCC2, VKORC1, HLA-B]
ONDANSETRON ODT → [CYP2D6, ABCB1]
```

### Step 3: Gene → Allele Frequencies
For each gene, we fetch allele frequencies for **all populations**:
```
CYP2D6:
  - allele_frequency_global: 0.15
  - allele_frequency_afr: 0.12
  - allele_frequency_amr: 0.14
  - allele_frequency_eas: 0.10
  - allele_frequency_eur: 0.18
  - allele_frequency_sas: 0.13

CYP3A4:
  - allele_frequency_global: 0.25
  - allele_frequency_afr: 0.20
  - allele_frequency_amr: 0.22
  - allele_frequency_eas: 0.18
  - allele_frequency_eur: 0.30
  - allele_frequency_sas: 0.24
```

### Step 4: Patient-Level Aggregation
For each patient, we aggregate across all their drug-gene pairs:

**Example for Patient 12345:**
- Drugs: AMOXICILLIN, AZITHROMYCIN
- Drug-gene pairs: 17 total (5 + 12)
- For each population, calculate: **mean**, **max**, **sum** of allele frequencies

## Features Added to Model Data

### Feature Names (20 total features per patient)

#### Global Frequency Features (3 features):
- `pgx_risk_global_mean`: Average allele frequency across all drug-gene pairs (global)
- `pgx_risk_global_max`: Maximum allele frequency across all drug-gene pairs (global)
- `pgx_risk_global_sum`: Sum of allele frequencies across all drug-gene pairs (global)

#### Population-Specific Features (18 features):
For each population (AFR, AMR, EAS, EUR, SAS), create 3 features:
- `pgx_risk_{population}_mean`: Average allele frequency for that population
- `pgx_risk_{population}_max`: Maximum allele frequency for that population
- `pgx_risk_{population}_sum`: Sum of allele frequencies for that population

**Example:**
- `pgx_risk_afr_mean`, `pgx_risk_afr_max`, `pgx_risk_afr_sum`
- `pgx_risk_amr_mean`, `pgx_risk_amr_max`, `pgx_risk_amr_sum`
- `pgx_risk_eas_mean`, `pgx_risk_eas_max`, `pgx_risk_eas_sum`
- `pgx_risk_eur_mean`, `pgx_risk_eur_max`, `pgx_risk_eur_sum`
- `pgx_risk_sas_mean`, `pgx_risk_sas_max`, `pgx_risk_sas_sum`

#### Count Features (2 features):
- `pgx_drugs_with_mappings`: Number of patient's drugs that have PGx mappings
- `pgx_genes_covered`: Number of unique PGx genes involved in patient's drug-gene pairs

### Complete Feature List

```
mi_person_key                    # Patient identifier
pgx_risk_global_mean            # Mean global frequency
pgx_risk_global_max             # Max global frequency
pgx_risk_global_sum             # Sum global frequency
pgx_risk_afr_mean               # Mean AFR frequency
pgx_risk_afr_max                # Max AFR frequency
pgx_risk_afr_sum                # Sum AFR frequency
pgx_risk_amr_mean               # Mean AMR frequency
pgx_risk_amr_max                # Max AMR frequency
pgx_risk_amr_sum                # Sum AMR frequency
pgx_risk_eas_mean               # Mean EAS frequency
pgx_risk_eas_max                # Max EAS frequency
pgx_risk_eas_sum                # Sum EAS frequency
pgx_risk_eur_mean               # Mean EUR frequency
pgx_risk_eur_max                # Max EUR frequency
pgx_risk_eur_sum                # Sum EUR frequency
pgx_risk_sas_mean               # Mean SAS frequency
pgx_risk_sas_max                # Max SAS frequency
pgx_risk_sas_sum                # Sum SAS frequency
pgx_drugs_with_mappings         # Count of drugs with PGx data
pgx_genes_covered               # Count of unique genes
```

## Important: Race is NOT Used for Feature Assignment

### What We Do NOT Do:
❌ **We do NOT assign frequencies based on patient-reported race**
❌ **We do NOT create a single "assigned" feature based on patient demographics**
❌ **We do NOT use patient race to select which frequency to use**

### What We Do:
✅ **Create separate features for ALL populations** (global, AFR, AMR, EAS, EUR, SAS)
✅ **Let the model determine which features are most predictive**
✅ **Store all population frequencies for transparency**

## Example Feature Values

### Patient 12345 (took AMOXICILLIN + AZITHROMYCIN):

```
mi_person_key: 12345
pgx_risk_global_mean: 0.18    # Average across 17 drug-gene pairs (global)
pgx_risk_global_max: 0.30     # Highest frequency variant (global)
pgx_risk_global_sum: 3.06     # Sum of all frequencies (global)

pgx_risk_eur_mean: 0.20       # Average across 17 drug-gene pairs (European)
pgx_risk_eur_max: 0.35       # Highest frequency variant (European)
pgx_risk_eur_sum: 3.40        # Sum of all frequencies (European)

pgx_risk_afr_mean: 0.15       # Average across 17 drug-gene pairs (African)
pgx_risk_afr_max: 0.25       # Highest frequency variant (African)
pgx_risk_afr_sum: 2.55       # Sum of all frequencies (African)

... (similar for AMR, EAS, SAS)

pgx_drugs_with_mappings: 2   # AMOXICILLIN, AZITHROMYCIN
pgx_genes_covered: 13        # Unique genes across both drugs
```

## Model Usage

During model training:

1. **Feature Selection**: Model can automatically select which population features improve predictions
2. **Feature Importance**: Shows which frequency variants (if any) are most informative
3. **No Race-Based Assignment**: Model evaluates all variants equally, regardless of patient demographics

### Example Model Behavior:

If the model finds that `pgx_risk_eur_mean` is highly predictive, this means:
- European population frequencies happen to correlate with outcomes
- This is a **data-driven finding**, not an assumption
- The model discovered this relationship, we didn't assign it based on patient race

## Rationale

### Why Not Assign Based on Race?

1. **Race ≠ Genetic Ancestry**: Patient-reported race is a social construct, not genetic ancestry
2. **Model-Driven**: Let the model discover which features are predictive
3. **Transparency**: All population frequencies are available for analysis
4. **Avoid Bias**: Don't make assumptions about which frequency to use

### Why Multiple Variants?

1. **Population Differences**: Allele frequencies DO vary by population (well-documented)
2. **Model Selection**: Model can determine if population-specific features help
3. **Flexibility**: Can evaluate different approaches without assumptions

## Summary

**Features Added**: 20 features per patient
- 3 global frequency features (mean, max, sum)
- 18 population-specific features (3 per population × 6 populations)
- 2 count features (drugs with mappings, genes covered)

**Race Usage**: ❌ **NOT used** for feature assignment
- All population variants are created
- Model determines which are predictive
- No race-based assumptions

**Final Output**: `pgx_added_features_{cohort}_{age_band}.csv`
- Ready to join with `model_data` using `mi_person_key`
- All features are numeric (float)
- Missing values filled with 0.0

