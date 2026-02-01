# Lake Formation & Glue Scripts — Moved

**These scripts have been moved to `aws-pgx-setup/` and organized by AWS service.**

| Former location (utility_scripts/) | New location |
|------------------------------------|--------------|
| **Lake Formation** | **aws-pgx-setup/lake_formation/** |
| lakeformation_set_iam_only_defaults.py | aws-pgx-setup/lake_formation/lakeformation_set_iam_only_defaults.py |
| lakeformation_grant_crawler_on_database.py | aws-pgx-setup/lake_formation/lakeformation_grant_crawler_on_database.py |
| lakeformation_grant_drop_tables.py | aws-pgx-setup/lake_formation/lakeformation_grant_drop_tables.py |
| **Glue** | **aws-pgx-setup/glue/** |
| glue_delete_tables.py | aws-pgx-setup/glue/glue_delete_tables.py |
| glue_create_databases.py | aws-pgx-setup/glue/glue_create_databases.py |
| test_glue_lakeformation_permissions.py | aws-pgx-setup/glue/test_glue_lakeformation_permissions.py |
| test_glue_crawler_permissions.py | aws-pgx-setup/glue/test_glue_crawler_permissions.py |
| validate_pharmacy_row_coverage.py | aws-pgx-setup/glue/validate_pharmacy_row_coverage.py |
| validate_medical_row_coverage.py | aws-pgx-setup/glue/validate_medical_row_coverage.py |

**Documentation:**

- **Lake Formation:** [aws-pgx-setup/lake_formation/README.md](../aws-pgx-setup/lake_formation/README.md)
- **Glue:** [aws-pgx-setup/glue/README.md](../aws-pgx-setup/glue/README.md)

Run scripts from the **repo root** (pgx-analysis), e.g.:

```bash
python aws-pgx-setup/lake_formation/lakeformation_set_iam_only_defaults.py --credentials-dir /mnt/c/Projects
python aws-pgx-setup/glue/test_glue_crawler_permissions.py --credentials-dir /mnt/c/Projects --dataset pharmacy
```

**Reference (cross-account Glue/Lake Formation):** See `C:\Projects\vr_sling_analytics\aws\iam` for lf_*.json, glue_*.json, trust policies.
