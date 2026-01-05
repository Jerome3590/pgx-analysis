# Allele Frequency Feature Creation Methodology

## Overview

This document explains how allele frequency features are created in the PGx analysis step. We create **multiple feature variants** and let the **model/algorithm determine** which frequency approach is most predictive.

## Current Implementation

### What We Do

1. **Fetch Population-Level Frequencies**: We retrieve allele frequencies from the CPIC API for different population ancestry groups:
   - **Global**: Average frequency across all populations
   - **AFR**: African/African American ancestry
   - **AMR**: Latino/Admixed American ancestry
   - **EAS**: East Asian ancestry
   - **EUR**: European ancestry
   - **SAS**: South Asian ancestry

2. **Create Multiple Feature Variants**: We create separate features for each frequency approach:
   - `pgx_risk_global`: Uses global average (baseline)
   - `pgx_risk_afr/amr/eas/eur/sas`: Uses population-specific frequencies
   - `pgx_risk_assigned`: Uses patient demographics if available, else global

3. **Let Model Determine Best Approach**: All feature variants are included, and the model/algorithm evaluates which improves predictions

4. **Store All Frequencies**: All population frequencies are stored for transparency and analysis

### Model-Driven Selection

✅ **Create multiple feature variants**  
✅ **Include both global and population-specific options**  
✅ **Let feature importance/model selection determine best approach**  
✅ **Use patient demographics to create population-specific features (optional)**  
✅ **Document which features the model selects**

## Rationale

### 1. Race/Ethnicity Categories Are Social Constructs

- Race and ethnicity categories in healthcare data are social constructs, not genetic categories
- They are poor proxies for genetic ancestry
- Using them for genetic predictions can perpetuate health disparities

### 2. Population Ancestry Groups ≠ Patient Demographics

- The AFR/AMR/EAS/EUR/SAS categories in genomic databases represent population ancestry groups
- These are based on genetic clustering from reference populations (e.g., 1000 Genomes, gnomAD)
- They do not correspond directly to patient-reported race/ethnicity categories

### 3. Clinical Best Practice

- Clinical pharmacogenomics uses **actual patient genotyping**, not population averages
- Population frequencies inform risk stratification but are not substitutes for genotyping
- CPIC guidelines recommend genotyping for clinical decision-making

### 4. Gender/Sex Is Not Relevant for Most PGx Variants

- Most pharmacogenomic variants are autosomal (not sex-linked)
- Gender/sex is typically not a factor in PGx allele frequencies
- Exceptions (e.g., X-linked genes) are rare and would require specific handling

## Data Structure

### Output Columns

The `allele_frequencies.csv` file contains:

```csv
drug_name,gene,variant_id,allele_name,allele_frequency_global,allele_frequency_afr,allele_frequency_amr,allele_frequency_eas,allele_frequency_eur,allele_frequency_sas,frequency_source,frequency_assignment_method
```

- **`allele_frequency_global`**: Default frequency to use (global average)
- **`allele_frequency_*`**: Population-specific frequencies (for reference only)
- **`frequency_assignment_method`**: Always `population_level_all_stored` (documents that we store all, don't assign)

### Usage in Downstream Analysis

If downstream analysis needs to use population-specific frequencies:

1. **Use Global Frequency**: Recommended default for most analyses
2. **Use Population Frequencies**: Only if you have validated genetic ancestry data (not demographic categories)
3. **Use Actual Genotyping**: Best practice for clinical applications

## Example Feature Creation

```python
# Create multiple feature variants for model evaluation
features['pgx_risk_global'] = drug_importance * allele_frequency_global
features['pgx_risk_afr'] = drug_importance * allele_frequency_afr
features['pgx_risk_amr'] = drug_importance * allele_frequency_amr
features['pgx_risk_eas'] = drug_importance * allele_frequency_eas
features['pgx_risk_eur'] = drug_importance * allele_frequency_eur
features['pgx_risk_sas'] = drug_importance * allele_frequency_sas

# If patient demographics available, create assigned feature
if patient_demographics:
    features['pgx_risk_assigned'] = drug_importance * assigned_frequency
else:
    features['pgx_risk_assigned'] = features['pgx_risk_global']

# Model will evaluate which features improve predictions
# Feature importance will show which frequency approach is most informative
```

## References

- CPIC Guidelines: https://cpicpgx.org/
- CPIC API: https://api.cpicpgx.org/
- NIH Statement on Race: https://www.genome.gov/about-genomics/fact-sheets/Genetics-vs-Genomics
- ASHG Statement on Ancestry: https://www.ashg.org/publications-news/press-releases/ancestry-testing-statement/

## Future Considerations

If individual patient genotyping data becomes available:
- Replace population frequencies with actual patient genotypes
- Use genotype-based phenotype predictions (e.g., CYP2D6 metabolizer status)
- Follow CPIC guidelines for genotype-to-phenotype translation

