# PGx Patient Card Integration

## Overview

The PGx Patient Card feature has been integrated into the dashboard as a second tab. Users can enter their ancestry report with SNPs and receive a personalized PGx card showing potential drug interactions based on genetic variants.

## Features

### Tabbed Interface
- **Risk Assessment Tab**: Original risk scoring functionality
- **PGx Patient Card Tab**: New PGx card generation

### PGx Card Input
- **SNP Data Input**: Text area for entering gene variants
  - No personal identification required
  - Card is anonymous and generic
  - Format: `Gene,Variant1,Variant2,Variant3` (one per line)
  - Example: `CYP2D6,*1,*2`
- **File Upload**: Support for CSV, Excel, or text files

### PGx Card Output
- **Card Metadata**: Timestamp and IP address (for tracking, not identification)
- **Genes Tested**: List of genes with variants
- **Drugs Requiring Modifications**: Drugs associated with genetic variants
- **Gene Details**: Detailed variant information
- **CPIC Guidelines**: Links to Clinical Pharmacogenomics Implementation Consortium guidelines

**Privacy**: The card is anonymous and generic. Users can add their own identification information as needed.

## API Endpoint

### POST `/pgx/card`

**Request Body:**
```json
{
  "variants": [
    {
      "gene": "CYP2D6",
      "variants": ["*1", "*2"]
    },
    {
      "gene": "CYP2C19",
      "variants": ["*1", "*17"]
    }
  ]
}
```

**Response:**
```json
{
  "timestamp": "2024-01-15 14:30:00 UTC",
  "ip_address": "192.168.1.1",
  "genes": [
    {
      "gene": "CYP2D6",
      "variants": ["*1", "*2"],
      "allele_count": 2
    }
  ],
  "drugs": [
    {
      "gene": "CYP2D6",
      "drug": "codeine",
      "guideline_url": "https://cpicpgx.org/guidelines/...",
      "cpic_level": "A",
      "fda_label": "Actionable PGx"
    }
  ]
}
```

## CPIC Data Integration

The system uses CPIC (Clinical Pharmacogenomics Implementation Consortium) data to match genes to drugs:

- **Data Source**: Master Excel file `cpic_gene-drug_pairs.xlsx` from `5_pgx_analysis/cpic/` (see `10_risk_dashboard/data_preparation/prepare_cpic_data.py` to stage copies for Lambda)
  - Official CPIC gene-drug pairs file (573 pairs, 300 drugs, 121 genes)
  - Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
- **Storage**: 
  - Primary: Container filesystem (`/var/task/data/cpic_gene-drug_pairs.xlsx`)
  - Fallback: S3 (`gold/dashboard/data/cpic_gene-drug_pairs.xlsx`; Parquet may be used when present)
- **Format**: Excel file with columns: Gene, Drug, Guideline, CPIC Level, FDA Label, etc.

## Implementation Details

### Frontend (index.html)
- Tab switching functionality
- SNP input form with file upload
- PGx card display with gene and drug information
- Error handling and status messages

### Backend (lambda_function.py)
- `handle_pgx_card()`: Route handler for PGx card requests
- `generate_pgx_card()`: Processes variants and matches to drugs
- `load_cpic_data()`: Loads CPIC data from container or S3

## Deployment Notes

1. **CPIC Data File**: Ensure `cpic_gene-drug_pairs.xlsx` is included in:
   - Docker container at `/var/task/data/cpic_gene-drug_pairs.xlsx`
   - Or uploaded to S3 at `gold/dashboard/metadata/cpic_gene-drug_pairs.xlsx`

2. **API Gateway**: Add route for `/pgx/card` POST method

3. **CORS**: Already configured for cross-origin requests

## Privacy & Anonymity

The PGx Card is designed to be **anonymous and generic**:
- **No personal identification** is required or stored
- **No patient ID** fields
- Card includes only:
  - Timestamp (when card was generated)
  - IP address (for tracking purposes, not for identification)
- Users can add their own identification information to the printed/saved card as needed

## Usage Example

1. Navigate to "PGx Patient Card" tab
2. Enter SNP data:
   ```
   CYP2D6,*1,*2
   CYP2C19,*1,*17
   CYP2C9,*1,*2
   ```
4. Click "Generate PGx Card"
5. View results showing:
   - Genes tested
   - Drugs requiring dosing modifications
   - CPIC guideline links

## Future Enhancements

- Support for more file formats (23andMe, Ancestry.com exports)
- QR code generation for genes
- Export card as PDF
- Integration with pharmacy systems
- Drug interaction severity scoring

