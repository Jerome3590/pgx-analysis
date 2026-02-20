# Archived PGx Patient Card Files

This folder contains **deprecated** files from the original R-based PGx Patient Card system (2020-2021).

## Reason for Archival

These files are no longer used due to:
1. **API endpoint changes** - PharmGKB deprecated old endpoints
2. **VIP URL migration** - Moved from pharmgkb.org to clinpgx.org
3. **Platform migration** - Migrated from R to Python for Lambda integration
4. **Workflow updates** - Integrated with Jupyter notebooks and AWS dashboard

## Archived Files

### R Scripts (Deprecated)
- `PGx.Rmd` - Original PGx card generator using old PharmGKB API
- `Build_PGx_Database.Rmd` - Database builder with deprecated endpoints
- `PGx-Card-Visualization.Rmd` - Card visualization with old URLs
- `PGx_Flextable.Rmd` - FlexTable formatting functions
- `PGx_dataViz.R` - Data visualization helper functions
- `PGx_formatData.R` - Data formatting helper functions
- `card_layout.R` - Card layout functions
- `base64_images.R` - Base64 image encoding functions

### Sample Outputs
- `Sample_Results_Card.pptx` - Example output (no longer representative)

## Current Active Files

See parent directory for active Python-based implementation:
- `patient_network_builder.py` - New multi-layer network visualization
- `pgx_card_generator.py` - Lambda-compatible card generator
- `fetch_pharmgkb_data.py` - Current API data fetcher
- `generate_pgx_qr_codes.py` - QR code generator for ClinPGx VIP URLs
- `build_pgx_database.py` - Python database builder
- `README_PYTHON.md` - Current documentation

## Migration Date

**Archived**: February 20, 2026
**Migration**: R (2020-2021) → Python (2026)

---

**Note**: These files are kept for historical reference only. Do not use for production PGx card generation.
