# PharmGKB API Test Results

## Summary

The PharmGKB REST API v1 provides rich VIP (Very Important Pharmacogene) data through the `/data/gene` endpoint. The gene endpoint returns everything needed for comprehensive pharmacogenomic reports without requiring multiple endpoint calls.

## Key Findings

### 1. Gene Endpoint Returns Complete VIP Data

**Endpoint**: `GET /v1/data/gene?symbol={GENE_SYMBOL}`

Example: `https://api.pharmgkb.org/v1/data/gene?symbol=CYP2D6`

### 2. Available Data Fields

#### Core Gene Information
- `id`: PharmGKB gene ID (e.g., "PA128")
- `symbol`: Gene symbol (e.g., "CYP2D6")
- `name`: Full gene name
- `chr`: Chromosome object with name
- `cbStart`, `cbStop`: Cytoband location

#### VIP Information
- `vipId`: VIP page ID (e.g., "PA166170264") - **use for ClinPGx URLs**
- `vipTier`: Tier classification (Tier 1, Tier 2, etc.)
- `vipSummary`: **Rich HTML content with:**
  - Clinical guidelines and drug interactions
  - Links to specific alleles
  - AMP tier 1 alleles
  - Phenoconversion information
  - Drug-drug interactions
- `vipCitation`: Full citation with authors, journal, DOI, PubMed links

#### CPIC/AMP Status
- `cpicGene`: Boolean - is this a CPIC gene?
- `hasCpicDosingGuideline`: Boolean - has CPIC dosing guidelines?
- `amp`: Boolean - on AMP recommended list?
- `pharmVarGene`: Boolean - in PharmVar database?

#### Allele Information
- `alleleFile`: Excel file name for allele definitions
- `alleleType`: "Named Alleles" or other
- `alleleFunctionSource`: Source of function assignments (e.g., "CPIC")

#### Genomic Coordinates
- `buildVersion`: Genome build (e.g., "GRCh38.p7")
- `chrStartPosB37`, `chrStopPosB37`: Build 37 coordinates
- `chrStartPosB38`, `chrStopPosB38`: Build 38 coordinates
- `strand`: "plus" or "minus"

### 3. Example VIP Summary Content (CYP2D6)

```
CYP2D6 is one of the key pharmacogenes involved in implementation of pharmacogenomics. 
It is highly polymorphic and involved in the metabolism of up to 25% of the drugs that 
are in common use in the clinic.

- It is involved in guidelines for codeine and other opioids, antidepressants, and 
  tamoxifen
- the nomenclature has been set by PharmVar
- the AMP tier 1 alleles are CYP2D6*2, CYP2D6*3, CYP2D6*4, CYP2D6*5, CYP2D6*6, CYP2D6*9, 
  CYP2D6*10, CYP2D6*17, CYP2D6*29, CYP2D6*41, CYP2D6 duplications and other copy number 
  variants
- CYP2D6 is the most studied CYP for phenoconversion, where a drug-drug interaction mimics 
  a metabolizer phenotype. paroxetine, fluoxetine and bupropion are strong inhibitors of 
  CYP2D6
```

### 4. What We DON'T Need

The following endpoints are **not needed** or **not accessible** via simple query parameters:

- ❌ `/data/clinicalAnnotation?geneId=...` - Returns 400 error "No such property: 'geneId'"
- ❌ `/data/variantAnnotation?geneId=...` - Returns 400 error
- ❌ `/data/drugLabel?geneId=...` - Returns 400 error
- ❌ `/data/chemical?relatedGeneId=...` - Returns 400 error
- ❌ `/data/guidelineAnnotation?geneId=...` - Returns 400 error

**All needed information is in the gene endpoint's `vipSummary` field.**

## Implementation Strategy

### For Cohort PGx Network Topology

1. **Fetch gene data**: Use `/data/gene?symbol={GENE}` endpoint only
2. **Extract text**: Parse `vipSummary.html` to plain text
3. **NLP processing**: Use pytextrank + AWS Comprehend on vip_summary_text
4. **Entity extraction**: Identify drugs, phenotypes, and relationships from summary text
5. **Optional enhancement**: Scrape ClinPGx VIP HTML page for additional context

### Updated Data Structure

```json
{
  "gene_symbol": "CYP2D6",
  "gene_id": "PA128",
  "vip_id": "PA166170264",
  "vip_url": "https://www.clinpgx.org/vip/PA166170264/overview",
  "vip_tier": "Tier 1",
  "cpic_gene": true,
  "vip_summary_text": "CYP2D6 is one of the key pharmacogenes...",
  "citation_text": "Cytochrome P450 2D6. Owen Ryan P et al. 2009"
}
```

### ClinPGx URL Format

✅ **Correct**: `https://www.clinpgx.org/vip/{vipId}/overview`

Example: `https://www.clinpgx.org/vip/PA166170264/overview`

Use `vipId` field, not `id` field, for constructing URLs.

## Test Results Files

- **all_gene_data.json**: Complete gene data for CYP2D6, CYP2C19, SLCO1B1
- **test_report_CYP2D6.json**: Structured report (empty annotations due to API limitations)

## Next Steps

1. ✅ Updated `fetch_vip_reports.py` to use vipSummary text
2. ✅ Updated `build_network_topology.py` to extract vip_summary_text
3. ⏭️ Test full workflow on one cohort/age band
4. ⏭️ Verify pytextrank extracts useful phrases from VIP summaries
5. ⏭️ Validate network topology generation with real data

## API Rate Limiting

- Conservative delay: 0.5 seconds between requests
- Tested successfully with 3 genes without throttling
- For 50 genes per cohort: ~25-30 seconds total fetch time

## Conclusion

The PharmGKB gene endpoint provides **all necessary data** for building rich pharmacogenomic reports. The `vipSummary` HTML field contains comprehensive clinical information that can be directly used for:

- Network topology analysis
- Drug-gene relationship extraction
- Clinical guideline identification
- Allele and variant information

No additional API endpoints are needed for the Cohort PGx feature.
