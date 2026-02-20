# Patient-Specific PGx Network

Enhanced PGx Patient Card with interactive multi-layer network visualization personalized to each patient's genotype and medication history.

## Overview

This module extends the PGx Patient Card to include rich network topology similar to Cohort PGx, but **filtered and personalized** based on:
- Patient's specific genetic variants
- Risk level of each variant (high/moderate/normal)
- Current medication list
- CPIC guideline recommendations

## Key Differences from Cohort PGx

| Feature | Cohort PGx | Patient PGx |
|---------|------------|-------------|
| **Genes** | Top N from SHAP/FFA | Only patient's tested genes |
| **Coloring** | PharmGKB tier (1/2/3) | Patient risk level (high/moderate/normal) |
| **Drugs** | All drugs in VIP | Only drugs affected by patient's variants |
| **Edges** | Mention frequency | Risk-weighted by patient phenotype |
| **Filters** | Show All, Tier 1, etc. | High Risk Only, Current Meds, Actionable Drugs |
| **Alerts** | None | Drug-drug interaction warnings |
| **Focus** | Research/cohort patterns | Clinical action/patient safety |

## Features

### 1. Risk-Based Gene Coloring

**Red (High Risk)**:
- Poor metabolizers for critical drugs
- Ultra-rapid metabolizers for prodrugs
- High-impact variants (e.g., TPMT deficiency)
- Example: CYP2D6 *4/*4 (poor metabolizer) → avoid codeine

**Orange (Moderate Risk)**:
- Intermediate metabolizers
- Rapid metabolizers
- Moderate impact on drug exposure
- Example: CYP2C19 *2/*17 (intermediate) → adjust clopidogrel

**Green (Normal)**:
- Normal/extensive metabolizers
- No dosing changes typically needed
- Example: CYP2D6 *1/*1 (normal)

**Gray (Unknown)**:
- Insufficient data to classify
- Variants of uncertain significance (VUS)

### 2. Medication-Aware Network

**Current Medications** (bright cyan diamonds):
- Drugs patient is currently taking
- Highlighted with thicker borders
- Connected to patient's risk genes
- Priority for drug-drug interaction checks

**Actionable Drugs** (larger nodes):
- Drugs affected by patient's high/moderate risk variants
- Require dosing adjustment or monitoring
- May need alternative drug selection

**Other Drugs** (muted cyan):
- Drugs mentioned in VIP reports
- Patient has normal metabolism
- Lower priority for intervention

### 3. Personalized Risk Associations

**High-Risk Phenotypes** (pink squares):
- Adverse events patient is at increased risk for
- Based on patient's high-risk variants
- Example: SLCO1B1 variant → myopathy risk with statins

**Edge Thickness**:
- Thicker edges = higher patient risk
- High risk variants: weight 1.0 (thick)
- Moderate risk: weight 0.6 (medium)
- Normal: weight 0.3 (thin)

### 4. Drug-Drug Interaction Alerts

**Red Edges with ⚠️**:
- Interactions between patient's current medications
- Extracted from VIP reports
- Require clinical review
- Example: Paroxetine + codeine → CYP2D6 inhibition

### 5. CPIC Guideline Integration

If CPIC data provided:
- Edge annotations with CPIC level (A/B/C/D)
- Guideline recommendations in hover text
- Level A (dark red): Prescribing change recommended
- Level B (orange-red): Change may be considered

## Interactive Filters

Dropdown menu (top-left) for different clinical views:

### Show All
- Complete patient network
- All genes, drugs, phenotypes, interactions
- **Use for**: Comprehensive review

### High Risk Only
- Only high-risk genes (red nodes)
- Drugs affected by high-risk variants
- **Use for**: Priority clinical actions

### Current Medications
- Patient's current medication list
- Genes affecting those medications
- **Use for**: Medication review, drug-drug interactions

### Actionable Drugs
- Drugs requiring dosing adjustments
- High/moderate risk associations
- **Use for**: Prescribing guidance, formulary selection

### Adverse Event Risks
- Genes with phenotype associations
- Risk phenotypes patient is susceptible to
- **Use for**: Adverse event prevention, monitoring plans

## Usage

### Command Line

```bash
# Basic usage
python patient_network_builder.py \
  --variants patient_123_variants.json \
  --vip-reports vip_reports.json \
  --output-dir outputs/patient_123 \
  --patient-id "Patient_123"

# With current medications
python patient_network_builder.py \
  --variants patient_123_variants.json \
  --vip-reports vip_reports.json \
  --output-dir outputs/patient_123 \
  --patient-id "Patient_123" \
  --medications codeine simvastatin clopidogrel omeprazole

# With CPIC guidelines
python patient_network_builder.py \
  --variants patient_123_variants.json \
  --vip-reports vip_reports.json \
  --output-dir outputs/patient_123 \
  --patient-id "Patient_123" \
  --medications codeine simvastatin \
  --cpic-data ../data/cpic_gene_drug_pairs.xlsx
```

### Input: Patient Variants JSON

```json
[
  {
    "gene": "CYP2D6",
    "variant": "*4/*4",
    "diplotype": "*4/*4",
    "phenotype": "Poor Metabolizer",
    "impact": "High",
    "function": "No function"
  },
  {
    "gene": "SLCO1B1",
    "variant": "rs4149056 T/T",
    "phenotype": "Decreased Function",
    "impact": "High"
  },
  {
    "gene": "CYP2C19",
    "variant": "*2/*17",
    "phenotype": "Intermediate Metabolizer",
    "impact": "Moderate"
  }
]
```

### Input: VIP Reports JSON

Should contain PharmGKB VIP reports fetched using `fetch_vip_reports.py` from Cohort PGx module.

```json
{
  "CYP2D6": {
    "gene_symbol": "CYP2D6",
    "vip_summary_text": "CYP2D6 is involved in metabolism of 25% of drugs...",
    "vip_tier": "Tier 1",
    "cpic_gene": true,
    "amp": true,
    "vip_url": "https://www.clinpgx.org/vip/PA166170325/overview"
  },
  "SLCO1B1": {
    "gene_symbol": "SLCO1B1",
    "vip_summary_text": "SLCO1B1 transports statins...",
    "vip_tier": "Tier 1",
    "cpic_gene": true,
    "vip_url": "https://www.clinpgx.org/vip/PA134865839/overview"
  }
}
```

### Output Files

```
outputs/patient_123/
├── Patient_123_pgx_network.html          # Interactive visualization
├── Patient_123_actionable_drugs.csv      # Drugs requiring action
├── Patient_123_risk_phenotypes.csv       # Adverse events at risk
├── Patient_123_drug_interactions.csv     # Drug-drug interactions
└── Patient_123_pgx_summary.json          # Complete summary
```

## Integration with Lambda/Dashboard

### Step 1: Fetch VIP Reports (One-Time)

Use Cohort PGx module to fetch VIP reports for all PGx genes:

```bash
python 9_dashboard_visuals/cohort_pgx/fetch_vip_reports.py \
  --cohort all_pgx_genes \
  --age-band all \
  --top-n 200 \
  --output-dir pgx-patient-card/data
```

This creates a comprehensive VIP report cache for patient card generation.

### Step 2: Add Lambda Endpoint

In `10_risk_dashboard/backend/lambda_function.py`:

```python
from pathlib import Path
import sys

# Add patient network builder to path
sys.path.append(str(Path(__file__).parent.parent.parent / "pgx-patient-card"))
from patient_network_builder import PatientPGxNetworkBuilder

@app.post("/pgx/patient-network")
def generate_patient_network(
    variants: List[Dict],
    patient_id: str = "Patient",
    medications: Optional[List[str]] = None
):
    """Generate patient-specific PGx network."""
    
    # Load cached VIP reports
    vip_file = Path("/opt/ml/model/vip_reports_cache.json")  # Lambda layer
    with open(vip_file) as f:
        vip_reports = json.load(f)
    
    # Load CPIC data
    cpic_file = Path("/opt/ml/model/cpic_gene_drug_pairs.xlsx")
    cpic_data = pd.read_excel(cpic_file)
    
    # Build network
    builder = PatientPGxNetworkBuilder(
        patient_variants=variants,
        vip_reports=vip_reports,
        current_medications=medications,
        cpic_data=cpic_data,
        use_comprehend=False  # Use pytextrank only in Lambda
    )
    
    graph = builder.build_patient_network()
    
    # Generate visualization
    output_dir = Path("/tmp/pgx_networks")
    output_dir.mkdir(exist_ok=True)
    
    viz_file = output_dir / f"{patient_id}_network.html"
    builder.create_interactive_visualization(viz_file, patient_id)
    
    # Upload to S3
    s3_key = f"patient_networks/{patient_id}_network.html"
    s3_client.upload_file(str(viz_file), DASHBOARD_BUCKET, s3_key)
    
    # Generate reports
    builder.export_patient_report(output_dir, patient_id)
    
    # Return URLs and summary
    return {
        "network_url": f"https://{DASHBOARD_BUCKET}/{s3_key}",
        "actionable_drugs": list(builder.actionable_drugs),
        "high_risk_genes": [
            g for g, info in builder.patient_genes.items()
            if info["risk_level"] == "high_risk"
        ],
        "drug_interactions": len(builder.drug_interactions),
        "risk_phenotypes": list(builder.high_risk_phenotypes)
    }
```

### Step 3: Frontend Integration

In `index.html`, enhance PGx Card tab:

```html
<div class="tab-pane fade" id="pgx-card-tab">
  <h3>PGx Patient Card</h3>
  
  <!-- Traditional card view -->
  <div id="pgx-card-traditional">
    <!-- Existing card generator -->
  </div>
  
  <!-- Network visualization -->
  <div id="pgx-card-network" class="mt-4">
    <h4>Interactive PGx Network</h4>
    <iframe id="patient-network-iframe" 
            style="width:100%; height:900px; border:1px solid #ddd;">
    </iframe>
  </div>
  
  <!-- Actionable items -->
  <div id="pgx-actionable-items" class="mt-3">
    <h5>Actionable Items</h5>
    <div id="high-risk-genes"></div>
    <div id="actionable-drugs"></div>
    <div id="drug-interactions"></div>
  </div>
</div>
```

JavaScript:

```javascript
async function generatePatientNetwork(variants, patientId, medications) {
  const response = await fetch('/pgx/patient-network', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      variants: variants,
      patient_id: patientId,
      medications: medications
    })
  });
  
  const data = await response.json();
  
  // Load network visualization
  document.getElementById('patient-network-iframe').src = data.network_url;
  
  // Display actionable items
  displayHighRiskGenes(data.high_risk_genes);
  displayActionableDrugs(data.actionable_drugs);
  displayDrugInteractions(data.drug_interactions);
}
```

## Clinical Use Cases

### 1. Pre-prescription Review

**Scenario**: Physician considering codeine for patient with CYP2D6 *4/*4

**Workflow**:
1. Upload patient genetic data
2. Select "Current Medications" filter
3. Add "codeine" to medication list
4. **Result**: Red high-risk node CYP2D6 → codeine edge, thick line
5. **Action**: Avoid codeine, select alternative (morphine direct-acting)

### 2. Statin Safety Check

**Scenario**: Patient on simvastatin with SLCO1B1 variant

**Workflow**:
1. Load patient network with simvastatin in current meds
2. Filter "Adverse Event Risks"
3. **Result**: SLCO1B1 (orange/red) → Myopathy (pink) edge visible
4. **Action**: Monitor CPK levels, consider lower dose or alternative statin

### 3. Polypharmacy Review

**Scenario**: Patient on multiple CYP2D6 substrates/inhibitors

**Workflow**:
1. Enter all current medications
2. Select "Current Medications" filter
3. **Result**: Red alert edges between drugs (e.g., paroxetine ↔ codeine)
4. **Action**: Adjust doses, select alternatives, monitor closely

### 4. Adverse Event Investigation

**Scenario**: Patient experiencing respiratory depression on opioids

**Workflow**:
1. Load patient network
2. Filter "Adverse Event Risks"
3. **Result**: CYP2D6 poor metabolizer → Respiratory Depression
4. **Action**: Confirm CYP2D6 status, switch to non-CYP2D6 opioid

## Comparison: Traditional Card vs Network

### Traditional PGx Card
- **Format**: Static table/document
- **Content**: Gene → Variant → Recommendation
- **Relationships**: Implicit, user must connect dots
- **Medications**: Separate list
- **Interactions**: Not shown
- **Use**: Quick reference, print/export

### Network Visualization
- **Format**: Interactive graph
- **Content**: Genes + Drugs + Phenotypes interconnected
- **Relationships**: Explicit edges with evidence weights
- **Medications**: Highlighted nodes, integrated view
- **Interactions**: Red alert edges
- **Use**: Clinical decision support, comprehensive review

### Recommended Approach
Use **both**:
1. **Traditional card** for documentation, quick lookup, patient handout
2. **Network** for prescribing decisions, polypharmacy review, adverse event investigation

## Technical Requirements

### Dependencies

```bash
pip install pandas networkx plotly beautifulsoup4 spacy pytextrank openpyxl
python -m spacy download en_core_web_sm
```

### Optional: AWS Comprehend
```bash
pip install boto3
aws configure  # Set credentials
```

### Data Requirements

1. **Patient variants** (JSON):
   - gene, variant, phenotype, impact
   - From genetic testing report

2. **VIP reports** (JSON):
   - PharmGKB VIP summaries
   - Fetch using `fetch_vip_reports.py`

3. **CPIC data** (Excel, optional):
   - Gene-drug pairs with guidelines
   - Available from CPIC website

4. **Current medications** (list, optional):
   - Enhances drug-drug interaction detection
   - Highlights relevant nodes

## Troubleshooting

### No network generated
- **Error**: "No VIP report for gene X"
- **Solution**: Fetch VIP reports for missing genes:
  ```bash
  python 9_dashboard_visuals/cohort_pgx/fetch_vip_reports.py \
    --genes CYP2D6 CYP2C19 SLCO1B1 \
    --output-dir pgx-patient-card/data
  ```

### Empty visualization
- **Issue**: Patient has no high-risk variants
- **Solution**: Network will be small, focus on current medications filter

### Missing drug-drug interactions
- **Issue**: Interactions not detected
- **Solution**: Interactions only found if mentioned in VIP text. Add manual interaction database for comprehensive coverage.

### Slow generation
- **Issue**: Large networks take time
- **Solution**: 
  - Disable Comprehend: `--no-comprehend`
  - Limit to patient's genes only (automatic)
  - Pre-compute VIP reports, don't fetch at runtime

## Future Enhancements

1. **Allele-Specific Annotations**
   - Show which specific alleles drive recommendations
   - *4 vs *5 vs *10 allele effects

2. **Severity Scoring**
   - Quantitative risk scores (1-10)
   - Combine phenotype + drug + evidence strength

3. **Alternative Drug Suggestions**
   - "Safe alternatives" node group
   - Drugs not affected by patient's variants

4. **Temporal Monitoring**
   - Track network changes over time
   - Show how medication changes affect risk profile

5. **Pathway Visualization**
   - Group genes by metabolic pathway
   - Show pathway-level redundancy

6. **Mobile/Tablet Optimization**
   - Responsive design
   - Touch-friendly filters
   - Simplified view for small screens

