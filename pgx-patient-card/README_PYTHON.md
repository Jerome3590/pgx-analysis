# PGx Patient Card - Python Migration

**Migrated from R (2020-2021) to Python (2026)** to support:
- Updated PharmGKB API endpoints
- ClinPGx VIP URLs (https://www.clinpgx.org)
- Lambda integration
- Jupyter notebook workflow

## Overview

The PGx Patient Card system generates personalized pharmacogenomic cards showing:
- Genes tested and variants detected
- Drugs requiring dosing modifications
- CPIC guideline links
- QR codes linking to ClinPGx VIP pages

## Migration Summary

### What Changed

1. **PharmGKB API Updates**
   - Old: `https://api.pharmgkb.org/v1/site/vips` (deprecated)
   - New: `https://api.pharmgkb.org/v1/data/gene?symbol=...` (documented in Postman)
   - Reference: https://www.postman.com/pharmgkb/pharmgkb-api/documentation/g9rp4zr/pharmgkb-rest-api

2. **VIP URL Migration**
   - Old: `https://www.pharmgkb.org/vip/{PA_ID}`
   - New: `https://www.clinpgx.org/vip/{PA_ID}/overview`

3. **CPIC Data Source**
   - Still valid: `https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx`
   - Verified: February 2026 (31,145 bytes, last modified Feb 5, 2026)

### Old R Scripts (Deprecated)

The following R scripts are **deprecated** but kept for reference:
- `PGx.Rmd` - Original PGx card generator
- `Build_PGx_Database.Rmd` - Database builder
- `PGx-Card-Visualization.Rmd` - Card visualization
- `PGx_dataViz.R` - Data visualization functions
- `PGx_formatData.R` - Data formatting functions

**Do not use** - these scripts use deprecated API endpoints and outdated URLs.

## New Python Scripts

### 1. `fetch_pharmgkb_data.py`
Fetch PharmGKB VIP gene data using current API.

**Usage:**
```bash
python fetch_pharmgkb_data.py
```

**Output:**
- `data/pharmgkb_vip_genes.json` - VIP gene metadata with ClinPGx URLs

**What it does:**
- Fetches VIP genes (CYP2D6, CYP2C19, etc.) from PharmGKB API
- Adds ClinPGx VIP URLs for each gene
- Saves to JSON for QR code generation and Lambda

### 2. `generate_pgx_qr_codes.py`
Generate QR codes for ClinPGx VIP pages.

**Usage:**
```bash
python generate_pgx_qr_codes.py
```

**Prerequisites:**
- Run `fetch_pharmgkb_data.py` first
- Install: `pip install qrcode[pil]`

**Output:**
- `qr_codes/{GENE}.png` - QR code images (200x200 px)
- `data/qr_code_mappings.json` - Gene -> QR path mappings

**What it does:**
- Loads VIP data from `pharmgkb_vip_genes.json`
- Generates QR codes pointing to ClinPGx VIP URLs
- Saves PNGs for inclusion in patient cards

### 3. `build_pgx_database.py`
Build unified PGx database from CPIC and PharmGKB sources.

**Usage:**
```bash
python build_pgx_database.py
```

**Prerequisites:**
- Download CPIC Excel: `wget https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx -O data/cpic_gene-drug_pairs.xlsx`
- Run `fetch_pharmgkb_data.py`
- Run `generate_pgx_qr_codes.py` (optional)

**Output:**
- `data/pgx_database/pgx_database.csv` - Merged database (CSV)
- `data/pgx_database/pgx_database.json` - Merged database (JSON)
- `data/pgx_database/pgx_database.xlsx` - Merged database (Excel)
- `data/pgx_database/pgx_database_summary.json` - Summary stats

**What it does:**
- Merges CPIC gene-drug pairs with PharmGKB VIP data
- Adds QR code paths for each gene
- Exports in multiple formats

### 4. `pgx_card_generator.py`
Lambda-compatible PGx card generator module.

**Usage:**
```python
from pgx_card_generator import PGxCardGenerator

# Load data
generator = PGxCardGenerator.from_files(
    cpic_excel_path=Path("data/cpic_gene-drug_pairs.xlsx"),
    vip_json_path=Path("data/pharmgkb_vip_genes.json")
)

# Generate card
card = generator.generate_card(
    variants=[
        {"gene": "CYP2D6", "variants": ["*1", "*2"]},
        {"gene": "CYP2C19", "variants": ["*1", "*17"]}
    ],
    timestamp="2026-02-20 12:00:00 UTC",
    ip_address="192.168.1.1",
    patient_id="ABC123"  # optional
)
```

**What it does:**
- Matches variants to CPIC gene-drug pairs
- Adds ClinPGx VIP URLs for genes
- Returns structured card data for dashboard/export

## Installation

### Python Dependencies

```bash
pip install pandas openpyxl requests qrcode[pil] pillow
```

Or add to `requirements.txt`:
```txt
pandas>=2.0.0
openpyxl>=3.1.0
requests>=2.31.0
qrcode[pil]>=7.4.2
Pillow>=10.0.0
```

### Download CPIC Data

```bash
mkdir -p pgx-patient-card/data
cd pgx-patient-card/data
wget https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
```

## Full Workflow

### Initial Setup (Run Once)

```bash
cd pgx-patient-card

# 1. Download CPIC data
wget https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx \
  -O data/cpic_gene-drug_pairs.xlsx

# 2. Fetch PharmGKB VIP data
python fetch_pharmgkb_data.py

# 3. Generate QR codes
python generate_pgx_qr_codes.py

# 4. Build unified database
python build_pgx_database.py
```

**Outputs:**
- `data/pharmgkb_vip_genes.json` (20 VIP genes)
- `qr_codes/*.png` (20 QR code images)
- `data/qr_code_mappings.json`
- `data/pgx_database/` (merged CSV/JSON/XLSX)

### Jupyter Notebook Integration

See `4_dashboard_visuals.ipynb` for notebook cells that call these scripts.

### Lambda Integration

The Lambda function in `10_risk_dashboard/backend/lambda_function.py` already has PGx card generation via `POST /pgx/card`. To add VIP URLs:

1. Copy `data/pharmgkb_vip_genes.json` to Lambda container:
   ```dockerfile
   COPY pgx-patient-card/data/pharmgkb_vip_genes.json ${LAMBDA_TASK_ROOT}/data/
   ```

2. Load at Lambda startup:
   ```python
   VIP_URL_CACHE = {}
   
   def load_vip_urls():
       vip_path = '/var/task/data/pharmgkb_vip_genes.json'
       if os.path.exists(vip_path):
           with open(vip_path) as f:
               vip_data = json.load(f)
           VIP_URL_CACHE = {
               item['gene'].upper(): item['vip_url']
               for item in vip_data
           }
   
   load_vip_urls()  # Call at module level
   ```

3. Add VIP URLs in `generate_pgx_card()`:
   ```python
   for gene_entry in genes_processed:
       gene = gene_entry['gene']
       if gene in VIP_URL_CACHE:
           gene_entry['vip_url'] = VIP_URL_CACHE[gene]
   ```

## API Endpoints

### PharmGKB (Current)

- **API Docs**: https://www.postman.com/pharmgkb/pharmgkb-api/documentation/g9rp4zr/pharmgkb-rest-api
- **Gene endpoint**: `GET https://api.pharmgkb.org/v1/data/gene?symbol={GENE}`
- **VIP pages**: `https://www.clinpgx.org/vip/{PA_ID}/overview`

### CPIC

- **Gene-drug pairs**: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx
- **Prioritization docs**: https://cpicpgx.org/prioritization/

## Files Structure

```
pgx-patient-card/
├── data/
│   ├── cpic_gene-drug_pairs.xlsx          # Downloaded from CPIC
│   ├── pharmgkb_vip_genes.json            # Generated by fetch_pharmgkb_data.py
│   ├── qr_code_mappings.json              # Generated by generate_pgx_qr_codes.py
│   └── pgx_database/
│       ├── pgx_database.csv               # Unified database
│       ├── pgx_database.json
│       ├── pgx_database.xlsx
│       └── pgx_database_summary.json
├── qr_codes/
│   ├── CYP2D6.png                         # QR codes for each VIP gene
│   ├── CYP2C19.png
│   └── ...
├── fetch_pharmgkb_data.py                 # Fetch VIP data from PharmGKB API
├── generate_pgx_qr_codes.py               # Generate QR codes
├── build_pgx_database.py                  # Build unified database
├── pgx_card_generator.py                  # Lambda-compatible generator
└── README_PYTHON.md                       # This file
```

## Deprecation Timeline

- **2020-2021**: R scripts created using PharmGKB API v1 `/site/vips`
- **2024-2025**: PharmGKB migrates content to ClinPGx platform
- **2026 Feb**: Python migration complete
- **Future**: R scripts may be removed (kept for reference)

## Maintenance

To update VIP gene data:
```bash
python fetch_pharmgkb_data.py  # Fetches latest from PharmGKB API
python generate_pgx_qr_codes.py  # Regenerates QR codes
```

To update CPIC data:
```bash
wget https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx \
  -O data/cpic_gene-drug_pairs.xlsx
python build_pgx_database.py  # Rebuild database
```

## Questions?

- PharmGKB API: https://www.postman.com/pharmgkb/pharmgkb-api/
- CPIC: https://cpicpgx.org/
- ClinPGx: https://www.clinpgx.org/
