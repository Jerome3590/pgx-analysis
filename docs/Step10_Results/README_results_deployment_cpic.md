# CPIC Data Deployment Guide

## Overview

The PGx Patient Card feature uses the **master Excel file** from `7_pgx_analysis` as the primary data source for gene-drug interactions.

## Master Excel File

- **Source**: `7_pgx_analysis/cpic/cpic_gene-drug_pairs.xlsx`
- **Official Download**: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
- **Content**: 573 gene-drug pairs, 300 drugs, 121 genes
- **Format**: Excel (.xlsx) file with columns:
  - Gene
  - Drug
  - Guideline (URL)
  - CPIC Level
  - CPIC Level Status
  - PharmGKB Level of Evidence
  - PGx on FDA Label
  - CPIC Publications (PMID)

## Preparation Steps

### 1. Prepare CPIC Data

Run the preparation script to copy the master Excel file:

```bash
cd 10_results
python prepare_cpic_data.py
```

This will:
- Copy `cpic_gene-drug_pairs.xlsx` from `7_pgx_analysis/cpic/` to `10_results/data/`

### 2. Verify Files

Check that the file is in place:

```bash
ls -lh 10_results/data/
# Should show:
# - cpic_gene-drug_pairs.xlsx
```

### 3. Docker Build

The Dockerfile automatically includes the data directory:

```dockerfile
COPY data/ ${LAMBDA_TASK_ROOT}/data/
```

The file will be available in the container at:
- `/var/task/data/cpic_gene-drug_pairs.xlsx`

### 4. S3 Backup (Optional)

For redundancy, upload to S3:

```bash
aws s3 cp 10_results/data/cpic_gene-drug_pairs.xlsx \
  s3://pgxdatalake/gold/dashboard/metadata/cpic_gene-drug_pairs.xlsx
```

## Loading Priority

The Lambda function loads CPIC data in this order:

1. **Container Excel** (`/var/task/data/cpic_gene-drug_pairs.xlsx`) - **PRIMARY**
2. **S3 Excel** (`gold/dashboard/metadata/cpic_gene-drug_pairs.xlsx`) - Fallback

## Dependencies

The Lambda function requires `openpyxl` for Excel reading:

```txt
openpyxl>=3.1.0
```

This is already included in `requirements.txt` and will be installed during Docker build.

## Data Updates

To update the CPIC data:

1. Download the latest Excel file from CPIC:
   ```bash
   wget https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx \
     -O 7_pgx_analysis/cpic/cpic_gene-drug_pairs.xlsx
   ```

2. Run the preparation script:
   ```bash
   cd 10_results
   python prepare_cpic_data.py
   ```

3. Rebuild and redeploy the Docker container

## Verification

Test the PGx Card endpoint:

```bash
curl -X POST https://YOUR_API.execute-api.REGION.amazonaws.com/prod/pgx/card \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "TEST001",
    "variants": [
      {"gene": "CYP2D6", "variants": ["*1", "*2"]},
      {"gene": "CYP2C19", "variants": ["*1", "*17"]}
    ]
  }'
```

Expected response includes:
- Patient ID
- List of genes with variants
- List of drugs requiring modifications
- CPIC guideline URLs

## Troubleshooting

### Excel file not found
- Check that `prepare_cpic_data.py` ran successfully
- Verify file exists in `10_results/data/`
- Check Docker build logs for COPY errors

### pandas/openpyxl import errors
- Ensure `openpyxl>=3.1.0` is in `requirements.txt`
- Check Docker build logs for pip install errors

### Column detection issues
- The function auto-detects column names (case-insensitive)
- Check Excel file structure matches expected format
- Review Lambda logs for column detection messages

### Excel file loading errors
- Check Lambda logs for specific error messages
- Verify pandas and openpyxl are installed correctly
- Ensure Excel file is not corrupted

