# Cohort PGx Network Features

## Enhanced Features Summary

This document summarizes the rich multi-layer network features added to the Cohort PGx module.

## Data Layers

### 1. Gene Nodes (Tier-Based Coloring)

**PharmGKB Tier Classification**:
- **Tier 1** (Red) - Most clinically important
  - AMP Tier 1 alleles
  - CPIC guidelines with Levels A/B evidence
  - FDA-approved PGx labels
  - Examples: CYP2D6, CYP2C19, SLCO1B1, TPMT
  
- **Tier 2** (Orange) - Well-established
  - CPIC guidelines with emerging evidence
  - Multiple published studies
  - Examples: CYP2C9, CYP3A5, VKORC1
  
- **Tier 3** (Yellow-Orange) - Emerging evidence
  - Research implications
  - Limited clinical data
  
- **Unknown** (Gray)
  - No tier classification available

**Metadata Included**:
- CPIC guideline status (✓ CPIC)
- AMP tier annotation (✓ AMP)
- Node size = number of connections (degree centrality)
- Direct links to ClinPGx VIP pages

### 2. Drug Nodes

**Extraction Methods**:
- Pattern matching from VIP summary text
- Drug name normalization (generic + brand names)
- AWS Comprehend entity recognition (optional)

**Relationships**:
- **Gene → Drug** ("metabolizes"): Pharmacokinetic relationships
  - Edge weight = mention frequency in VIP text
  - Normalized 0-1 scale (max 10 mentions)
  - Example: CYP2D6 → Codeine (weight 0.8)

- **Drug ↔ Drug** (multiple types):
  - **Inhibition**: Drug1 inhibits metabolism of Drug2
  - **Induction**: Drug1 induces metabolism of Drug2
  - **Metabolic**: General metabolic interaction
  - **Combination**: Synergistic/additive effects
  - **Enhancement**: Drug1 enhances effect of Drug2
  - Edge weight = evidence text length (proxy for detail)

**Example Drug-Drug Interactions**:
- Paroxetine inhibits CYP2D6 → affects codeine metabolism
- Fluoxetine inhibits CYP2D6 → phenoconversion
- Rifampin induces CYP3A4 → reduces drug efficacy

### 3. Phenotype Nodes (Adverse Events)

**Extraction Strategy**:

Pattern matching for adverse event mentions:
- "risk of [phenotype]"
- "adverse events: [phenotype]"
- "[phenotype] toxicity/reaction"

**Common Adverse Events Detected**:
- Bleeding, Thrombosis
- Nausea, Vomiting, Diarrhea, Constipation
- Respiratory Depression, Sedation
- Liver Toxicity, Nephrotoxicity, Cardiotoxicity
- Myopathy, Neuropathy
- QT Prolongation
- Serotonin Syndrome
- Stevens-Johnson Syndrome
- And 15+ more...

**Relationships**:
- **Gene → Phenotype** ("affects_risk"): Clinical outcome associations
  - Edge weight = mention frequency
  - Example: SLCO1B1 → Myopathy (weight 0.7)

### 4. Gene-Gene Relationships

**Co-metabolizes Edges**:
- Genes that share drug targets
- Indicates pathway overlap
- Example: CYP2D6 ↔ CYP2C19 (both metabolize antidepressants)

## Interactive Filtering System

### Filter Options

**Dropdown menu** (top-left of visualization):

1. **Show All**
   - Complete multi-layer network
   - All nodes and edges visible
   - Use for: Understanding full complexity

2. **Genes Only**
   - Gene nodes + gene-gene connections
   - Hide drugs and phenotypes
   - Use for: Pathway analysis, identifying gene clusters

3. **Genes + Drugs**
   - Gene and drug nodes
   - Gene→Drug edges ("metabolizes")
   - Gene↔Gene edges ("co_metabolizes")
   - Use for: Pharmacokinetic analysis, drug selection

4. **Genes + Phenotypes**
   - Gene and phenotype nodes
   - Gene→Phenotype edges ("affects_risk")
   - Use for: Adverse event risk assessment, clinical outcomes

5. **Drug-Drug Interactions**
   - Drug nodes only
   - Drug↔Drug interaction edges
   - Use for: Polypharmacy safety, drug combination analysis

6. **Tier 1 Only**
   - Focus on Tier 1 genes (most clinically actionable)
   - All edges connected to Tier 1 genes
   - Use for: Clinical implementation, PGx testing prioritization

## Visual Encoding

### Node Properties

- **Shape**:
  - Circle = Gene
  - Diamond = Drug
  - Square = Phenotype

- **Color**:
  - Genes: Tier-based (red/orange/yellow/gray)
  - Drugs: Cyan
  - Phenotypes: Mint green

- **Size**: Proportional to node degree (number of connections)
  - Large nodes = hubs (affect many drugs/phenotypes)
  - Small nodes = specific interactions

### Edge Properties

- **Width**: Evidence strength
  - Thick = high mention frequency or detailed evidence
  - Thin = low mention frequency or limited evidence

- **Color**: Relationship type
  - Gray: Gene→Drug metabolism
  - Pink: Gene→Phenotype risk
  - Blue: Gene↔Gene co-metabolism
  - Purple/Red/Green/Gold: Drug-drug interactions

### Hover Text

Rich metadata on hover:

**Genes**:
```
GENE: CYP2D6
Tier: Tier 1
✓ CPIC ✓ AMP
Connections: 24
```

**Drugs**:
```
DRUG: Codeine
Connections: 3
```

**Phenotypes**:
```
PHENOTYPE: Respiratory Depression
Adverse Event
Associated Genes: 2
```

**Edges**:
```
CYP2D6 → Codeine
metabolizes
Weight: 0.82
```

## Evidence Weighting

### Gene→Drug/Phenotype Edges

**Calculation**:
```python
weight = min(mention_count, 10) / 10.0
```

- Counts how many times drug/phenotype mentioned in VIP text
- Normalized to 0-1 scale
- Max 10 mentions = weight 1.0

**Interpretation**:
- weight ≥ 0.7: Strong evidence (multiple mentions)
- weight 0.4-0.6: Moderate evidence
- weight < 0.4: Limited mentions, may be incidental

### Drug-Drug Interaction Edges

**Calculation**:
```python
weight = min(len(evidence_text) / 100.0, 1.0)
```

- Based on evidence snippet length
- Longer text = more detailed description
- Normalized to 0-1 scale

**Interpretation**:
- weight ≥ 0.8: Detailed interaction description
- weight 0.5-0.7: Moderate detail
- weight < 0.5: Brief mention

## Use Cases

### Clinical Applications

1. **Pre-prescription Review**
   - Filter: "Genes + Drugs"
   - Check if patient's high-risk genes (red nodes) affect proposed medication
   - Example: Patient with CYP2D6 poor metabolizer → avoid codeine

2. **Adverse Event Investigation**
   - Filter: "Genes + Phenotypes"
   - Identify genes associated with observed adverse event
   - Example: Myopathy event → check SLCO1B1 status

3. **Polypharmacy Safety**
   - Filter: "Drug-Drug Interactions"
   - Review pharmacokinetic interactions in current regimen
   - Example: Paroxetine + codeine → CYP2D6 inhibition

4. **PGx Testing Prioritization**
   - Filter: "Tier 1 Only"
   - Focus on genes with CPIC guidelines
   - Identify high-impact genes for testing budget

### Research Applications

1. **Cohort Comparison**
   - Compare network topology across cohorts
   - Identify cohort-specific gene-drug patterns
   - Example: Opioid ED vs Non-opioid ED pharmacogenomic differences

2. **Hub Gene Discovery**
   - Look for large nodes (high degree centrality)
   - Identify genes affecting many drugs/phenotypes
   - Prioritize for mechanistic studies

3. **Pathway Analysis**
   - Filter: "Genes Only"
   - Visualize gene-gene co-metabolism relationships
   - Identify pathway overlap and redundancy

4. **Age-Stratified Analysis**
   - Compare networks across age bands
   - Identify age-specific PGx patterns
   - Guide age-appropriate testing strategies

## Output Files Reference

### network_nodes.csv
```csv
id,type,label,degree,tier,cpic_gene,amp,url
CYP2D6,gene,CYP2D6,24,Tier 1,True,True,https://www.clinpgx.org/vip/...
Codeine,drug,Codeine,3,,,
Respiratory Depression,phenotype,Respiratory Depression,2,,,
```

### network_edges.csv
```csv
source,target,relation,weight,mentions,evidence
CYP2D6,Codeine,metabolizes,0.82,8 mentions,
CYP2D6,Respiratory Depression,affects_risk,0.60,6 mentions,
Paroxetine,Codeine,inhibition,0.85,,Paroxetine inhibits CYP2D6 metabolism of codeine...
```

### drug_interactions.csv
```csv
drug1,drug2,interaction_type,evidence
Paroxetine,Fluoxetine,inhibition,Both inhibit CYP2D6 leading to increased drug exposure...
Rifampin,Simvastatin,induction,Rifampin induces CYP3A4 reducing simvastatin efficacy...
```

### network_stats.json
```json
{
  "nodes_total": 127,
  "edges_total": 342,
  "genes": 50,
  "drugs": 68,
  "phenotypes": 9,
  "cpic_genes": 24,
  "drug_drug_interactions": 17,
  "density": 0.043,
  "avg_degree": 5.4,
  "gene_tiers": {
    "Tier 1": 18,
    "Tier 2": 22,
    "Tier 3": 8,
    "Unknown": 2
  }
}
```

### gene_metadata.json
```json
{
  "gene_tiers": {
    "CYP2D6": "Tier 1",
    "CYP2C19": "Tier 1",
    "SLCO1B1": "Tier 1",
    "CYP3A5": "Tier 2"
  },
  "cpic_genes": [
    "CYP2D6",
    "CYP2C19",
    "SLCO1B1",
    "TPMT",
    "DPYD"
  ]
}
```

## Future Enhancements

Potential additions:

1. **Allele-Specific Networks**
   - Show which specific alleles drive drug-phenotype associations
   - Example: CYP2D6*4 → poor metabolizer → respiratory depression risk

2. **Evidence Strength Tiers**
   - Categorize by PharmGKB evidence levels
   - CPIC Level A/B vs Level C/D

3. **FDA Label Integration**
   - Highlight drugs with FDA PGx labels
   - Link to label sections

4. **Patient-Specific Filtering**
   - Upload patient genotype
   - Highlight genes with risk alleles
   - Show only drugs affected by patient's variants

5. **3D Network Layout**
   - Use plotly 3D scatter for complex networks
   - Layer separation (genes, drugs, phenotypes in different planes)

6. **Temporal Evolution**
   - Animate network changes across PharmGKB updates
   - Show emerging evidence over time
