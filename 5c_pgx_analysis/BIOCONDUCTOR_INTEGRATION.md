# Bioconductor Integration for PGx Analysis

This document describes how to use Bioconductor R packages as an alternative data source for pharmacogenomics (PGx) analysis.

## Overview

Bioconductor provides several packages that can supplement or replace CPIC API data for:
1. **Allele Frequencies**: Population-level minor allele frequencies (MAFs)
2. **Drug-Gene Interactions**: Chemical-gene relationships
3. **Variant Annotations**: Genetic variant filtering and annotation

## Available Bioconductor Packages

### 1. GenomicScores
**Purpose**: Access to genomic scores including MAF data from major population databases.

**Data Sources**:
- 1000 Genomes Project
- ExAC (Exome Aggregation Consortium)
- gnomAD (Genome Aggregation Database)

**Installation**:
```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GenomicScores")
```

**Usage Example**:
```r
library(GenomicScores)
# Get MAF database
gsco <- getGScores("MafDb.1Kgenomes.phase3.hs37d5")
# Query variants (requires GRanges object with genomic coordinates)
scores <- gscores(gsco, gr)
```

**Reference**: https://bioconductor.org/packages/GenomicScores/

### 2. CTDquerier
**Purpose**: Extract and analyze data from the Comparative Toxicogenomics Database (CTD), including chemical-gene interactions.

**Installation**:
```r
BiocManager::install("CTDquerier")
```

**Usage Example**:
```r
library(CTDquerier)
# Get genes associated with a chemical
drug_genes <- get_genes(drugs = c("aspirin", "warfarin"))
# Get interactions
interactions <- get_interactions(chemicals = drugs, genes = genes)
```

**Reference**: https://bioconductor.org/packages/CTDquerier/

### 3. VariantFiltering
**Purpose**: Filter genetic variants based on criteria including population allele frequencies.

**Installation**:
```r
BiocManager::install("VariantFiltering")
```

**Usage Example**:
```r
library(VariantFiltering)
# Filter variants by MAF threshold
vf <- VariantFiltering(vcf_file, ...)
filtered <- filterVariants(vf, mafThreshold = 0.01)
```

**Reference**: https://bioconductor.org/packages/VariantFiltering/

### 4. AnnotationHub
**Purpose**: Access to various annotation databases and resources.

**Installation**:
```r
BiocManager::install("AnnotationHub")
```

**Usage Example**:
```r
library(AnnotationHub)
ah <- AnnotationHub()
# Search for resources
query(ah, c("allele", "frequency", "population"))
```

**Reference**: https://bioconductor.org/packages/AnnotationHub/

## Integration Workflow

### Step 1: Install Bioconductor Packages

```bash
# Install BiocManager if not already installed
Rscript -e "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')"

# Install required packages
Rscript -e "BiocManager::install(c('GenomicScores', 'CTDquerier', 'VariantFiltering', 'AnnotationHub'), update = FALSE, ask = FALSE)"
```

### Step 2: Fetch Data Using R Script

```bash
# Fetch allele frequencies for specific genes
Rscript 7_pgx_analysis/fetch_bioconductor_pgx_data.R \
    --cohort opioid_ed \
    --age-band 0-12 \
    --genes "CYP2D6,CYP2C19,CYP2C9" \
    --output-dir outputs/opioid_ed/0_12

# Fetch drug-gene relationships
Rscript 7_pgx_analysis/fetch_bioconductor_pgx_data.R \
    --cohort opioid_ed \
    --age-band 0-12 \
    --drugs "warfarin,clopidogrel,codeine" \
    --output-dir outputs/opioid_ed/0_12
```

### Step 3: Convert to CPIC-Compatible Format

The R script outputs CSV files that need to be converted to match the CPIC data format used by the Python workflow:

**Expected Format for Allele Frequencies**:
```csv
gene,variant_id,allele_frequency_global,allele_frequency_afr,allele_frequency_amr,allele_frequency_eas,allele_frequency_eur,allele_frequency_sas,source
CYP2D6,rs1065852,0.15,0.10,0.12,0.18,0.20,0.14,GenomicScores
```

**Expected Format for Drug-Gene Mappings**:
```csv
drug_name,gene,relationship_type,evidence_level,source
warfarin,CYP2C9,metabolism,1A,CTDquerier
```

### Step 4: Integrate with Python Workflow

The Python scripts (`add_allele_frequencies.py`, `map_drugs_to_genes.py`) can be modified to read Bioconductor-generated CSV files as an alternative to CPIC API data.

## Advantages of Bioconductor

1. **Comprehensive Data**: Access to multiple population databases (1000 Genomes, ExAC, gnomAD)
2. **Standardized Format**: Consistent data structures across packages
3. **Active Maintenance**: Regularly updated with new data releases
4. **R Integration**: Native R support for bioinformatics workflows

## Limitations

1. **Gene Coordinates Required**: GenomicScores requires genomic coordinates (GRanges) for variants
2. **Drug Name Normalization**: CTDquerier requires chemical name normalization
3. **VCF Files**: VariantFiltering requires VCF format input
4. **R Dependency**: Requires R/Bioconductor installation separate from Python workflow

## Current Status

The `fetch_bioconductor_pgx_data.R` script provides a framework for Bioconductor integration but requires:
1. Implementation of specific data fetching logic for your use case
2. Gene coordinate mapping for variant queries
3. Drug name normalization for CTD queries
4. Data format conversion to match CPIC structure

## References

- Bioconductor: https://bioconductor.org/
- GenomicScores: https://bioconductor.org/packages/GenomicScores/
- CTDquerier: https://bioconductor.org/packages/CTDquerier/
- VariantFiltering: https://bioconductor.org/packages/VariantFiltering/
- AnnotationHub: https://bioconductor.org/packages/AnnotationHub/

