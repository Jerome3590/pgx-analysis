S3 zones and canonical paths

This document describes the canonical S3 zones used by the pipeline and how to construct
paths for bronze, silver and gold, plus how we track pipeline state and run summaries.

Zones
-----
- bronze: raw parsed records from ingestion (source-like)
  - s3://<bucket>/bronze/<data_type>/...
- silver: imputed and partitioned data used by downstream transforms
  - s3://<bucket>/silver/imputed/<data_type>/age_band=<band>/event_year=<year>/
- gold: final outputs
  - s3://<bucket>/gold/<dataset_name>/age_band=<band>/event_year=<year>/

Notes
-----
- Write bronze once from raw ingest, write silver during global-imputation, then write gold in the cleaning steps. Overwrite gold in the second pass after target-code normalization.

Pipeline state and run summaries
-------------------------------
- Checkpoints and state (authoritative; for resume/retry):
  - Bucket: `pgx-repository`
  - Prefix: `pgx-pipeline-status/<pipeline>/<entity>/...`
  - Written by: `PipelineState` and `GlobalPipelineTracker`
- Aggregated run summaries (for BI/dashboards):
  - Bucket: `<bucket>` (e.g., `pgxdatalake`)
  - Prefix: `pgx_pipeline/<script_name>/run_id=<run_id>/summary.json`
  - Common fields: `tx`, `run_id`, `start_time`, `end_time`, `duration_sec`, `status`, `status_code`, `totals`
- Transaction codes:
  - `txt_to_parquet` → `txtpq`
  - `reprocess_txt_to_parquet` → `repro`
  - `global_imputation` → `glimp`
  - `clean_pharmacy` → `clnph`
  - `clean_medical` → `clnmd`

Helpers
-------
Use the helpers in `helpers_1997_13.s3_utils` to construct canonical paths:

- `zone_root(zone)` -> s3://<bucket>/<zone>/
- `build_zone_path(zone, data_type, age_band, event_year)` -> canonical partition path
- `convert_raw_to_imputed_path(raw_path, data_type, ...)` -> convenience for converting older silver raw prefixes

Backup and promotion
--------------------
- Always backup silver and gold before replacing them.
- Use the `backup_s3_prefix(src_prefix, backup_root)` helper to copy objects under a prefix to a backup root with a timestamped subprefix.

Examples
--------
Backup a silver partition and run a minimal two-pass pipeline (no brass, overwrite gold in pass 2):

```bash
# backup
python - <<'PY'
from helpers_1997_13.s3_utils import backup_s3_prefix
backup_s3_prefix('s3://pgxdatalake/silver/imputed/medical/age_band=65-74/event_year=2019/', 's3://pgxdatalake-backups/')
PY

# smoke-run the pipeline (from repo root)
python 1_apcd_input_data/2_global_imputation.py --pharmacy-input s3://pgxdatalake/bronze/pharmacy/*.parquet --medical-input s3://pgxdatalake/bronze/medical/*.parquet --output-root s3://pgxdatalake/silver/imputed --limit 1
python 1_apcd_input_data/3_apcd_clean.py --job pharmacy --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet --output-root s3://pgxdatalake/gold/pharmacy --limit 1
python 1_apcd_input_data/3_apcd_clean.py --job medical --raw-medical s3://pgxdatalake/silver/medical/*.parquet --output-root s3://pgxdatalake/gold/medical --limit 1
# optional exploration of target codes
python 1_apcd_input_data/6_target_frequency_analysis.py --limit 1
# pass 2: normalize target codes in gold (overwrite in place)
python 1_apcd_input_data/7_update_codes.py --resume --no-merge --years "2019" --workers-medical 4 --threads 1 --chunked --chunk-rows 250000
```

Notes
-----
- For large-scale migrations consider using AWS CLI `aws s3 cp --recursive` or `aws s3 sync` but wrap with tmp prefixes to enforce atomic promotion semantics.
- Prefer using the repo helpers so path formatting remains consistent across scripts.
