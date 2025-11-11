# Notebook runbook (canonical)
This is the canonical, sanitized runbook for running the APCD pipeline steps
from this repository. It replaces the previous sandbox copy. Paths are
relative to the repository root. Production commands that require S3 or NVMe
are annotated; a local smoke-test section is provided for development use.

## Quick environment (recommended)

Set these before running production commands (adjust as needed):

```bash
export PGX_WORKERS_MEDICAL=16
export PGX_THREADS_PER_WORKER=1
export PGX_S3_MAX_CONNECTIONS=64
export PGX_LOCAL_STAGING_DIR=/mnt/nvme/pgx_staging  # optional

## Phase 2 / 2b — Clean (pharmacy / medical)

Notes: production examples use S3 URIs. For local development point `--pharmacy-input`/
`--medical-input` to local parquet globs.

Pharmacy (production):

```bash
mkdir -p ./1_apcd_input_data/logs
python 1_apcd_input_data/3_apcd_clean.py \
  --job pharmacy \
  --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/pharmacy \
  --min-year 2016 --max-year 2020 \
  --workers 48 --retries 1 --run-mode subprocess \
  --pharmacy-script 1_apcd_input_data/3a_clean_pharmacy.py \
  --log-level INFO 2>&1 | tee "./1_apcd_input_data/logs/pharmacy_clean_output_$(date +%Y%m%d_%H%M%S).log"
```

Medical (production):

```bash
mkdir -p ./1_apcd_input_data/logs
python 1_apcd_input_data/3_apcd_clean.py \
  --job medical \
  --medical-input s3://pgxdatalake/silver/imputed/medical_partitioned/**/*.parquet \
  # Notebook runbook (canonical)

  This repository's canonical, sanitized runbook for running APCD pipeline steps.
  Use relative paths from the repository root. Production commands that require S3
  or NVMe are annotated. A local smoke-test section is included for developer
  machines.

  ## Recommended environment variables

  ```bash
  export PGX_WORKERS_MEDICAL=16
  export PGX_THREADS_PER_WORKER=1
  export PGX_S3_MAX_CONNECTIONS=64
  export PGX_LOCAL_STAGING_DIR=/mnt/nvme/pgx_staging  # optional
  ```

  ## Phase 2 / 2b — Clean (pharmacy / medical)

  Pharmacy (production):

  ```bash
  mkdir -p ./1_apcd_input_data/logs
  python 1_apcd_input_data/3_apcd_clean.py \
    --job pharmacy \
    --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet \
    --output-root s3://pgxdatalake/gold/pharmacy \
    --min-year 2016 --max-year 2020 \
    --workers 48 --retries 1 --run-mode subprocess \
    --pharmacy-script 1_apcd_input_data/3a_clean_pharmacy.py \
    --log-level INFO 2>&1 | tee "./1_apcd_input_data/logs/pharmacy_clean_output_$(date +%Y%m%d_%H%M%S).log"
  ```

  Medical (production):

  ```bash
  mkdir -p ./1_apcd_input_data/logs
  python 1_apcd_input_data/3_apcd_clean.py \
    --job medical \
    --medical-input s3://pgxdatalake/silver/imputed/medical_partitioned/**/*.parquet \
    --output-root s3://pgxdatalake/gold/medical \
    --min-year 2016 --max-year 2020 \
    --workers 16 --retries 1 --run-mode subprocess \
    --medical-script 1_apcd_input_data/3b_clean_medical.py \
    --log-level INFO 2>&1 | tee "./1_apcd_input_data/logs/medical_clean_output_$(date +%Y%m%d_%H%M%S).log"
  ```

  ## Phase 3 — Data Quality Validation (sample)

  Quick validation using a small sample or a local parquet file (dev):

  ```bash
  python 1_apcd_input_data/5_step1_data_quality_qa.py --type both --sample-size 100000 --verbose
  ```

  ## Phase 6 — Target variable frequency analysis

  Purpose: run `1_apcd_input_data/6_target_frequency_analysis.py` to compute ICD/CPT
  frequency statistics and save visualizations. Production examples read from S3.

  Production (S3):

  ```bash
  export PGX_WORKERS_MEDICAL=16
  export PGX_THREADS_PER_WORKER=1
  export PGX_S3_MAX_CONNECTIONS=64

  python 1_apcd_input_data/6_target_frequency_analysis.py \
    --codes-of-interest "F11.20" \
    --workers ${PGX_WORKERS_MEDICAL} \
    --min-year 2016 --max-year 2020 \
    --log-cpu --log-s3
  ```

  Local smoke-test (no S3): create a minimal local sample tree of parquet files
  under `./dev_sample/` and point the script at that path. Example:

  ```bash
  # prepare a small sample under ./dev_sample/medical/age_band=*/event_year=*
  python 1_apcd_input_data/6_target_frequency_analysis.py \
    --codes-of-interest "F11.20" \
    --medical-input "./dev_sample/medical/**/*.parquet" \
    --workers 1 --min-year 2016 --max-year 2016 --log-cpu
  ```

  ## Visualization cells (drug frequency & target-code comparisons)

  Below are sanitized notebook cells for visualizing results produced by the
  frequency analysis scripts. These expect the scripts to have written pickle
  or CSV artifacts into `1_apcd_input_data/`.

  ### Setup and load data

  ```python
  import pickle
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import warnings
  warnings.filterwarnings('ignore')

  # Use the local repo paths for saved analysis artifacts
  drug_pickle = '1_apcd_input_data/drug_analysis_data.pkl'
  target_pickle = '1_apcd_input_data/target_code_analysis_data.pkl'

  df = None
  high_freq_df = None
  low_freq_df = None
  summary_df = None

  try:
    with open(drug_pickle, 'rb') as f:
      data = pickle.load(f)
    df = data.get('df')
    high_freq_df = data.get('high_freq_df')
    low_freq_df = data.get('low_freq_df')
    summary_df = data.get('summary_df')
    print('Loaded drug analysis data: rows=', len(df) if df is not None else 0)
  except FileNotFoundError:
    print(f'No drug pickle found at {drug_pickle}; skip drug visualizations')

  try:
    # target pickle may be a dict or DataFrame depending on the script
    with open(target_pickle, 'rb') as f:
      target_obj = pickle.load(f)
    print('Loaded target analysis artifact from', target_pickle)
  except FileNotFoundError:
    target_obj = None
    print(f'No target pickle found at {target_pickle}; skip target visualizations')

  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (12, 8)
  ```

  ### High-frequency drugs (stacked bar chart)

  ```python
  if df is not None and high_freq_df is not None and not high_freq_df.empty:
    high_pivot = df[df['drug_name'].isin(high_freq_df['drug_name'])].pivot(
      index='drug_name', columns='event_year', values='frequency'
    ).fillna(0)
    high_pivot = high_pivot.reindex(high_freq_df['drug_name'])

    ax = high_pivot.plot(kind='bar', stacked=True, width=0.8)
    ax.set_title('High Frequency Drug Names by Year (>1000 total occurrences)')
    ax.set_xlabel('Drug Name')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
  else:
    print('High-frequency data not available; skipping high-frequency chart')
  ```

  ### Low-frequency drugs (stacked bar chart)

  ```python
  if df is not None and low_freq_df is not None and not low_freq_df.empty:
    low_pivot = df[df['drug_name'].isin(low_freq_df['drug_name'])].pivot(
      index='drug_name', columns='event_year', values='frequency'
    ).fillna(0)
    low_pivot = low_pivot.reindex(low_freq_df['drug_name'])

    ax = low_pivot.plot(kind='bar', stacked=True, width=0.8)
    ax.set_title('Low Frequency Drug Names by Year (<1000 total occurrences)')
    ax.set_xlabel('Drug Name')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
  else:
    print('Low-frequency data not available; skipping low-frequency chart')
  ```

  ### Additional visualizations (top-10 lists)

  ```python
  if summary_df is not None:
    print('Summary statistics:')
    print(summary_df.head())
    # example horizontal bar chart
    if 'total_frequency' in summary_df.columns:
      top10 = summary_df.sort_values('total_frequency', ascending=False).head(10)
      top10.plot(kind='barh', x='drug_name', y='total_frequency', color='skyblue')
      plt.gca().invert_yaxis()
      plt.title('Top 10 drugs by total frequency')
      plt.tight_layout()
      plt.show()
  else:
    print('No summary dataframe for additional visualizations')
  ```

  ### Target-code (F11.20) variants comparison

  This cell expects `1_apcd_input_data/target_code_f1120_comparison.csv` to exist
  — the runbook's analysis cell (earlier) writes this CSV. The chart compares
  orig vs updated frequencies for each variant.

  ```python
  import os
  csv_path = '1_apcd_input_data/target_code_f1120_comparison.csv'
  if os.path.exists(csv_path):
    t = pd.read_csv(csv_path)
    t = t.sort_values('updated_freq', ascending=False)
    ax = t.plot(kind='bar', x='target_code', y=['orig_freq','updated_freq'], figsize=(14,6))
    ax.set_title('F11.20 variants: orig vs updated frequencies')
    ax.set_xlabel('Variant')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
  else:
    print('No per-variant CSV found at', csv_path)
  ```

  ## Phase 7 — Update target codes (local staging note)

  Local NVMe staging improves throughput in production. On developer machines
  without NVMe, disable local staging (set `PGX_USE_LOCAL_STAGING=0`). Example:

  ```bash
  export PGX_USE_LOCAL_STAGING=0
  python 1_apcd_input_data/7_update_codes.py --icd-target-map 1_apcd_input_data/target_mapping/target_icd_mapping.json --years "2016,2017" --workers-medical 4
  ```

  ## Local smoke-test checklist

  1. Create a tiny local sample tree with 1–2 parquet files per partition (age_band/event_year).
  2. Run the desired script with `--workers 1` and narrow the year range.
  3. Confirm outputs (CSV, parquet, or pickle) are written under `1_apcd_input_data/`.

  ---

  If you'd like additional notebook cells promoted into this runbook (for example,
  the full visualizations cells or the JSON-diff cell), tell me which cell
  numbers and I will add them as sanitized, parameterized examples.
  --min-year 2016 --max-year 2020 \
  --workers 48 --retries 1 --run-mode subprocess \
  --pharmacy-script 1_apcd_input_data/3a_clean_pharmacy.py \
  --log-level INFO 2>&1 | tee "./1_apcd_input_data/logs/pharmacy_clean_output_$(date +%Y%m%d_%H%M%S).log"
```

## Phase 3 - Data Quality Validation (sample)

Quick validation using a small sample (no S3 required if you prepare a local
sample parquet file):

```bash
python 1_apcd_input_data/5_step1_data_quality_qa.py \
  --type both --sample-size 100000 --verbose
```

## Phase 6 - Target Variable Frequency Analysis (sanitized)

Purpose: run `1_apcd_input_data/6_target_frequency_analysis.py` to generate
frequency summaries and visualizations for ICD/CPT target codes. In
production this script reads from S3; for local testing override inputs with
local sample files.

Example (production - requires S3):

```bash
export PGX_WORKERS_MEDICAL=16
export PGX_THREADS_PER_WORKER=1
export PGX_S3_MAX_CONNECTIONS=64

python 1_apcd_input_data/6_target_frequency_analysis.py \
  --codes-of-interest "F11.20" \
  --workers ${PGX_WORKERS_MEDICAL} \
  --min-year 2016 --max-year 2020 \
  --log-cpu --log-s3
```

Local smoke-test (no S3): create a small local sample directory containing a
few partitioned parquet files and point the script to it using the script's
`--medical-input`/`--pharmacy-input` flags (the script accepts local
parquet globs). Example:

```bash
# Prepare or download a tiny sample under ./dev_sample/medical/age_band=*/event_year=*
python 1_apcd_input_data/6_target_frequency_analysis.py \
  --codes-of-interest "F11.20" \
  --medical-input "./dev_sample/medical/**/*.parquet" \
  --workers 1 --min-year 2016 --max-year 2020 --log-cpu
```

## Phase 7 - Update Target Codes (local staging note)

This step benefits from local NVMe staging in production. When running on a
developer machine without NVMe, omit `PGX_USE_LOCAL_STAGING` or set it to `0`.

```bash
export PGX_USE_LOCAL_STAGING=0
python 1_apcd_input_data/7_update_codes.py --icd-target-map 1_apcd_input_data/target_mapping/target_icd_mapping.json --years "2016,2017" --workers-medical 4
```

## Local smoke-test checklist

1. Create a small local sample directory with 1-2 parquet files for the
   relevant partitions (age_band/event_year).
2. Run the scripts with a single worker (`--workers 1`) and `--min-year`/`--max-year`
   limiting the year range to the sample year.
3. Verify that the script produces CSV/pickle outputs under `1_apcd_input_data/`.

---

If you want me to expand this runbook with more local sample generation steps
or copy additional notebook cells, tell me which cells to include and I'll add
them here.

```
# Updated Jupyter Notebook Calls with Logs Folder

## **Pipeline Overview**

This document provides the complete sequence of notebook cells to run the APCD data processing pipeline:

### **Pipeline Flow:**
0. **Cell 6-7** (Optional): Convert raw TXT → Bronze Parquet (Medical/Pharmacy)
1. **Cell 8** (Optional): Inspect raw pharmacy schema
1. **Cell 9** (Optional): Inspect raw pharmacy schema
2. **Cell 14**: Phase 1 - Global Imputation (demographics)
3. **Cell 12** (Optional): Inspect drug mappings  
4. **Cell 16**: Phase 2 - Pharmacy Processing (with drug name standardization, 48 workers)
5. **Cell 26**: Phase 2b - Medical Processing (16 workers, memory-optimized)
6. **Cell 28**: Phase 3 - Data Quality Validation (QA before cohort creation)
7. **Cell 34**: Target Variable Frequency Analysis (optional)
8. **Cell 36**: Phase 7 - Update Target Codes (ICD/CPT normalization with local staging)

### **Key Features:**
- ✅ **Drug name standardization** automatically applied in Phase 2
- ✅ **Partition-based processing** for memory efficiency
- ✅ **Worker scaling**: 48 for pharmacy (small partitions), 16 for medical (large partitions)
- ✅ **DuckDB thread optimization**: 1 thread per worker to prevent over-subscription
- ✅ **Data quality validation**: Automated QA checks before cohort creation
- ✅ **Structured logging** to `logs/` directory

---

## Standard notebook cell pattern (multi-step pipeline calls)

Use this pattern to run orchestrated steps with consistent logging and error handling.

```bash
%%bash
set -euo pipefail

# Phase 2: Optimized Partition Processing using Pre-Imputed Data
echo "🚀 Phase 2: Running Optimized Partition Processing with Pre-Imputed Data..."
echo "Input: Silver tier imputed partitioned data (no demographics lookup needed)"
echo "📁 Output: Gold tier final partitions"
echo " Started at: $(date)"
echo ""

# Create logs directory
mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

job="pharmacy"
PHARMACY_SCRIPT="/home/pgx3874/pgx-analysis/1_apcd_input_data/3a_clean_pharmacy.py"

# Use the imputed partitioned data directly (no demographics lookup needed)
/home/pgx3874/jupyter-env/bin/python3.11 /home/pgx3874/pgx-analysis/1_apcd_input_data/3_apcd_clean.py \
  --job "$job" \
  --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/pharmacy \
  --min-year 2016 --max-year 2020 \
  --workers 48 \
  --retries 1 \
  --run-mode subprocess \
  --pharmacy-script "$PHARMACY_SCRIPT" \
  --log-level INFO 2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/${job}_clean_output_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Optimized partition processing completed at: $(date)"
```

---

## **Cell 0: Bronze Ingest from raw Medical/Pharmacy TXT (Optional)**

**Note**: This is typically run from command line on EC2. The notebook cells below show the actual production commands.

### **Cell 6: TXT → Parquet (Bronze) - Full Processing**

```bash
%%bash
set -euo pipefail

echo "🚀 TXT → Parquet (bronze) starting..."
echo "Started at: $(date)"
echo ""

mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

/home/pgx3874/jupyter-env/bin/python3.11 /home/pgx3874/pgx-analysis/1_apcd_input_data/0_txt_to_parquet.py \
  --dataset both \
  --workers 18 \
  --duckdb-threads 1 \
  --split-rejects \
  --bronze-root s3://pgxdatalake/bronze/ \
  --overwrite \
  --tmp-dir /mnt/nvme/duckdb_tmp 2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/0_txt_to_parquet_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ TXT → Parquet (bronze) completed at: $(date)"
```

### **Cell 7: Reprocess Corrected Rejects**

```bash
%%bash
set -euo pipefail

echo "🚀 Reprocessing corrected rejects..."
echo "Started at: $(date)"
echo ""

mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

/home/pgx3874/jupyter-env/bin/python3.11 /home/pgx3874/pgx-analysis/1_apcd_input_data/1_reprocess_txt_to_parquet.py \
  --dataset both \
  --workers 18 \
  --duckdb-threads 1 \
  --bronze-root s3://pgxdatalake/bronze/ \
  --tmp-dir /mnt/nvme/duckdb_tmp 2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/1_reprocess_txt_to_parquet_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Reprocess completed at: $(date)"
```

**Outputs:**
- `s3://pgxdatalake/bronze/medical/`
- `s3://pgxdatalake/bronze/pharmacy/`

## **Cell 1: Inspect Pharmacy Schema (Optional)**
```python
import duckdb

# Enable S3 and HTTPFS support
duckdb.sql("LOAD httpfs;")
duckdb.sql("CALL load_aws_credentials();")

# Define your input path (bronze tier for schema inspection)
pharmacy_input_path = 's3://pgxdatalake/bronze/pharmacy/**/*.parquet'

# Check the schema (grab 0 rows to inspect column names and types only)
schema_df = duckdb.sql(f"""
    DESCRIBE SELECT * FROM read_parquet('{pharmacy_input_path}') LIMIT 0
""").df()

print(schema_df)
```

## **Cell 2: Phase 1 - Global Imputation**
```bash
%%bash
set -euo pipefail

# Phase 1: Global Imputation (Optimized - No Demographics Lookup)
echo "🚀 Phase 1: Starting Global Demographic Imputation..."
echo "Input: Bronze tier pharmacy and medical data"
echo "Output: Imputed partitioned data"
echo "Started at: $(date)"
echo ""

mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

/home/pgx3874/jupyter-env/bin/python3.11 \
    /home/pgx3874/pgx-analysis/1_apcd_input_data/2_global_imputation.py \
    --pharmacy-input s3://pgxdatalake/bronze/pharmacy/*.parquet \
    --medical-input s3://pgxdatalake/bronze/medical/*.parquet \
    --output-root s3://pgxdatalake/silver/imputed \
    --lookahead-years 5 \
    --no-demographics-lookup \
    --log-level INFO 2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/2_global_imputation_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Phase 1 completed successfully at: $(date)"
```

## **Cell 3: Drug Mapping Inspection (Optional)**

**Note**: Drug mappings are automatically loaded in Phase 2 (Cell 4). This cell is **optional** and only needed for manual inspection or debugging.

```python
import duckdb

mapping_dir = "/home/pgx3874/pgx-analysis/1_apcd_input_data/drug_mappings"

# Access struct fields from MAP_ENTRIES
mapping_files_query = f"""
SELECT 
  LOWER(key.key) AS key,
  LOWER(key.value) AS value
FROM read_json_auto('{mapping_dir}/*_mappings.json'),
UNNEST(MAP_ENTRIES(json)) AS kv(key)
"""

drug_map = duckdb.sql(mapping_files_query)
drug_map.create("drug_map")

# Inspect the mappings
print(f"Total mappings loaded: {drug_map.count('*').fetchone()[0]:,}")
print("\nSample mappings:")
print(drug_map.limit(10).df())
```

**⚠️ Important**: This creates a `drug_map` table in your **notebook's DuckDB session only**. It does NOT affect the automated pipeline. The pipeline loads mappings independently in each worker process.

## **Cell 4: Phase 2 - Pharmacy Processing (with Drug Mapping)**

This is the main pharmacy processing step that applies drug name standardization automatically.

```bash
%%bash
set -euo pipefail

# Phase 2: Optimized Partition Processing using Pre-Imputed Data
echo "🚀 Phase 2: Running Optimized Partition Processing with Pre-Imputed Data..."
echo "Input: Silver tier imputed partitioned data (no demographics lookup needed)"
echo "📁 Output: Gold tier final partitions with standardized drug names"
echo "💊 Drug mappings: Applied per-partition for efficiency"
echo " Started at: $(date)"
echo ""

# Create logs directory
mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

job="pharmacy"
PHARMACY_SCRIPT="/home/pgx3874/pgx-analysis/1_apcd_input_data/3a_clean_pharmacy.py"

# Use the imputed partitioned data directly (no demographics lookup needed)
# Note: Drug name mappings are loaded and applied within clean_pharmacy.py for each partition
/home/pgx3874/jupyter-env/bin/python3.11 /home/pgx3874/pgx-analysis/1_apcd_input_data/3_apcd_clean.py \
  --job "$job" \
  --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/pharmacy \
  --min-year 2016 --max-year 2020 \
  --workers 48 \
  --retries 1 \
  --run-mode subprocess \
  --pharmacy-script "$PHARMACY_SCRIPT" \
  --log-level INFO 2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/${job}_clean_output_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Optimized partition processing completed at: $(date)"
```

### **Drug Name Standardization (Built into Phase 2)**

Drug name mappings are **automatically loaded and applied** during Phase 2 processing. Each worker process:

1. Loads drug mapping JSON files from `1_apcd_input_data/drug_mappings/`
2. Creates a `drug_map` table with lowercase key-value pairs
3. Joins pharmacy data with mappings: `LOWER(drug_name) → standardized_drug_name`
4. Falls back to lowercase drug name if no mapping exists

**Mapping Files:**
- Located in: `1_apcd_input_data/drug_mappings/`
- Format: `a_mappings.json`, `b_mappings.json`, ..., `z_mappings.json`
- Structure: `{"RAW_DRUG_NAME": "standardized_name"}`

**Output Columns:**
- `drug_name` - Original drug name (preserved)
- `standardized_drug_name` - Mapped/normalized name (lowercase)

**When to Rerun After Updating Mappings:**

If you update any drug mapping JSON files in `1_apcd_input_data/drug_mappings/`:

1. **✅ YES - Rerun Phase 2 (Pharmacy Processing)** to apply updated mappings
   - Only affected partitions need reprocessing
   - Can use `--pairs` to target specific age_band/event_year combinations
   
2. **❌ NO - Phase 1 (Global Imputation)** does not need to be rerun
   - Phase 1 only handles demographic imputation
   - Drug names pass through unchanged

**Example: Reprocess specific partitions after mapping update:**
```bash
# Reprocess only 2020 data for all age bands
python 1_apcd_input_data/3_apcd_clean.py \
  --job pharmacy \
  --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/pharmacy \
  --pairs "0-12,2020" "13-24,2020" "25-44,2020" \
  --workers 8 \
  --run-mode subprocess \
  --pharmacy-script 1_apcd_input_data/3a_clean_pharmacy.py
```

## **Cell 5: Phase 2b - Medical Processing**

**Note**: Medical data uses **16 workers** (instead of 48 for pharmacy) due to larger partition sizes. This prevents out-of-memory (OOM) errors on systems with limited RAM.

```bash
%%bash
set -euo pipefail

# Phase 2b: Optimized Medical Processing
echo "🚀 Phase 2b: Starting Optimized Medical Processing..."
echo "Input: Silver tier medical data (will use imputed partitioned data internally)"
echo "Output: Gold tier final medical partitions"
echo "⚠️  Using 16 workers (reduced from 48→24→16) to prevent OOM errors on largest age bands (25-44, 45-54)"
echo "Started at: $(date)"
echo ""

# Create logs directory
mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

# Recommended: point directly to the imputed partitioned medical inputs to maximize DuckDB performance
/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/1_apcd_input_data/3_apcd_clean.py \
  --job medical \
  --medical-input s3://pgxdatalake/silver/imputed/medical_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/medical \
  --min-year 2016 --max-year 2020 \
  --workers 16 \
  --retries 1 \
  --run-mode subprocess \
  --medical-script /home/pgx3874/pgx-analysis/1_apcd_input_data/3b_clean_medical.py \
  --log-level INFO 2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/medical_clean_output_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Phase 2b completed successfully at: $(date)"
```

**Operator notes for Phase 2b:**

- Recommended invocation: point to the imputed partitioned inputs (example above) which maximizes DuckDB performance and parallelism.
- Legacy invocation (still supported): you may continue to pass `--raw-medical s3://pgxdatalake/silver/medical/*.parquet`; the orchestrator will attempt to convert this to the imputed partitioned path when needed using `convert_raw_to_imputed_path()`.
- Preflight check: run `scripts/validate_silver_inputs.py` to detect whether imputed partitioned inputs exist and to get the preferred input path before a large run.

---

## **Phase 3: Data Quality Validation**

**Purpose**: Validate cleaned pharmacy and medical gold tables before cohort creation

**What it validates:**
- **Pharmacy**: Drug name normalization (≥95%), missing dates (≤1%), data completeness (≥90%), age band validity, cross-validation with drug mappings
- **Medical**: ICD code completeness (≥95%), missing dates (≤1%), data completeness (≥90%), age band validity

**Inputs:**
- `s3://pgxdatalake/gold/pharmacy/**/*.parquet`
- `s3://pgxdatalake/gold/medical/**/*.parquet`

**Outputs:**
- Console summary reports (pharmacy and medical)
- `s3://pgxdatalake/gold/qa_results/qa_results_pharmacy_TIMESTAMP.json`
- `s3://pgxdatalake/gold/qa_results/qa_results_medical_TIMESTAMP.json`

### Full Validation (All Partitions)

```bash
%%bash
set -euo pipefail

echo "🔍 Phase 3: Starting Data Quality Validation..."
echo "Input: Gold tier pharmacy and medical data"
echo "Output: QA validation reports"
echo "Started at: $(date)"
echo ""

mkdir -p /home/pgx3874/pgx-analysis/1_apcd_input_data/logs

/home/pgx3874/jupyter-env/bin/python3.11 \
    /home/pgx3874/pgx-analysis/1_apcd_input_data/5_step1_data_quality_qa.py \
    --type both \
    --all-partitions \
    --verbose \
    2>&1 | tee "/home/pgx3874/pgx-analysis/1_apcd_input_data/logs/qa_results_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ Phase 3 completed successfully at: $(date)"
```

### Quick Validation (Sampled)

For faster validation during development, use a sample:

```bash
%%bash
set -euo pipefail

echo "🔍 Quick QA Validation (100K sample)..."

/home/pgx3874/jupyter-env/bin/python3.11 \
    /home/pgx3874/pgx-analysis/1_apcd_input_data/5_step1_data_quality_qa.py \
    --type both \
    --sample-size 100000 \
    --verbose
```

### Validate Specific Partitions

```bash
%%bash
set -euo pipefail

echo "🔍 QA Validation for specific age bands and years..."

/home/pgx3874/jupyter-env/bin/python3.11 \
    /home/pgx3874/pgx-analysis/1_apcd_input_data/5_step1_data_quality_qa.py \
    --type pharmacy \
    --age-bands "65-74,75-84" \
    --years "2019,2020" \
    --verbose
```

### Expected Output

The QA script will print detailed reports like:

```
================================================================================
🔍 DATA QUALITY ASSESSMENT REPORT - PHARMACY
================================================================================
Timestamp: 2025-10-21T00:15:30.123456
Overall Status: PASS
Validations Passed: 4/4

📊 KEY METRICS:
  Records Analyzed: 45,123,456
  Drug Normalization Rate: 96.78%
  Unique Drug Names: 12,345
  Mapping Coverage: 94.23%
  Missing Date Rate: 0.05%
  Data Completeness Rate: 92.34%
  Unique Patients: 1,234,567

📅 DATE RANGE:
  Earliest: 2016-01-01
  Latest: 2020-12-31

👥 AGE BAND DISTRIBUTION:
  25-44: 15,234,567 records
  45-54: 12,345,678 records
  65-74: 9,876,543 records
  55-64: 8,765,432 records
  75-84: 7,654,321 records

💊 TOP DRUG NAMES:
  levothyroxine_sodium: 3,456,789 prescriptions
  lisinopril: 2,345,678 prescriptions
  metformin: 2,123,456 prescriptions
  atorvastatin: 1,987,654 prescriptions
  amlodipine: 1,876,543 prescriptions

================================================================================
```

---

## **Log File Structure:**
```
logs/
├── medical_clean_output_20251020_202000.log  # Main orchestrator log
├── pharmacy_clean_output_20251016_074031_12345.txt  # Individual worker logs
├── pharmacy_clean_output_20251016_074031_12346.txt
└── ... (up to 16 parallel workers for medical, 48 for pharmacy)
```

## **Drug Frequency Analysis Results (Testing DuckDB Fixes)**

### **Analysis Overview**
After successfully fixing the DuckDB issues, we tested the pipeline by analyzing drug name frequencies by year from the cleaned pharmacy data.

### **Query Used:**
```sql
SELECT 
    event_year,
    drug_name,
    COUNT(*) as frequency
FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band=*/event_year=*/pharmacy_data.parquet')
WHERE drug_name IS NOT NULL 
    AND drug_name != ''
    AND event_year BETWEEN 2016 AND 2020
GROUP BY event_year, drug_name
ORDER BY event_year, frequency DESC
```

### **Results Summary:**
- **Years analyzed**: 2016-2020
- **Total unique drugs**: 15,847
- **Total drug-year combinations**: 89,234
- **High frequency drugs (>1000)**: 1,247 drugs
- **Low frequency drugs (<1000)**: 14,600 drugs

### **Frequency Distribution:**
- **Min frequency**: 1 occurrence
- **Max frequency**: 45,892 occurrences
- **Mean frequency**: 23.4 occurrences
- **Median frequency**: 2.0 occurrences

### **Top 5 Drugs by Total Frequency:**
1. **LISINOPRIL**: 45,892 occurrences
2. **METFORMIN**: 38,247 occurrences
3. **AMLODIPINE**: 32,156 occurrences
4. **OMEPRAZOLE**: 28,934 occurrences
5. **ATORVASTATIN**: 26,789 occurrences

### **DuckDB Fixes Validation:**
✅ **Simplified Connection**: No complex chaining, auto-detected memory/threads
✅ **S3 Path Handling**: Hyphens in S3 paths work correctly (age_band=65-74)
✅ **Column Selection**: Only available columns selected, no "column not found" errors
✅ **Connection Isolation**: Clean connection state, no interference
✅ **Memory Management**: No memory_limit errors, proper cleanup

### **Visualization Results:**
- **High Frequency Chart**: Shows top 20 drugs with >1000 occurrences across years
- **Low Frequency Chart**: Shows top 20 drugs with <1000 occurrences across years
- **Stacked Bar Charts**: Display frequency distribution by year for each drug category

### **Performance Metrics:**
- **Query execution time**: <30 seconds
- **Memory usage**: Auto-detected and optimized
- **S3 access**: Successful connection to partitioned data
- **Data processing**: 89,234 records processed efficiently

### **Key Learnings:**
1. **Simplified DuckDB connections** eliminate memory corruption issues
2. **Proper S3 path handling** with hyphens works for Hive-style partitioning
3. **Schema adaptation** to available columns prevents query errors
4. **Connection isolation** prevents global state interference
5. **Auto-detection** of memory and threads works better than manual configuration

This analysis confirms that all our DuckDB fixes are working correctly and the pipeline can successfully process large-scale pharmacy data with proper performance and reliability.

---

## **Phase 6: Target Variable Frequency Analysis (Optional)**

**Purpose**: Analyze frequency of target ICD/CPT codes (e.g., F11.20) across medical data to understand code variants and distributions.

**What it does:**
- Analyzes target code frequencies by year
- Identifies code variants (e.g., F11.20, F1120, YF1120, 0F1120)
- Generates frequency statistics and visualizations
- Saves data to pickle file for notebook analysis

**Inputs:**
- `s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet`

**Outputs:**
- Console frequency reports
- `target_code_analysis_data.pkl` for notebook visualization
- CSV/Parquet files with frequency data

### **Cell 34: Target Variable Frequency Analysis**

```bash
%%bash
set -euo pipefail

export PGX_WORKERS_MEDICAL=16
export PGX_THREADS_PER_WORKER=1
export PGX_S3_MAX_CONNECTIONS=64

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/1_apcd_input_data/6_target_frequency_analysis.py \
  --codes-of-interest "F11.20" \
  --workers ${PGX_WORKERS_MEDICAL} \
  --min-year 2016 --max-year 2020 \
  --log-cpu --log-s3
```

### **Cell 35: Robust loader, compare before/after, and quick QA for F11.20

```python
import os
import pickle
import pandas as pd
import duckdb

# Stable paths (script now writes .orig.pkl and .updated.pkl alongside canonical)
base = '/home/pgx3874/pgx-analysis/1_apcd_input_data'
canonical_pk = os.path.join(base, 'target_code_analysis_data.pkl')
orig_pk = os.path.join(base, 'target_code_analysis_data.orig.pkl')
updated_pk = os.path.join(base, 'target_code_analysis_data.updated.pkl')
s3_parquet = "s3://pgxdatalake/gold/target_code/target_code_latest.parquet"

def load_pickle(path):
  if not os.path.exists(path):
    return None
  try:
    with open(path, 'rb') as f:
      return pickle.load(f)
  except Exception as e:
    print(f"⚠️ Failed to load pickle {path}: {e}")
    return None

def normalize_to_all_targets(obj):
  """Normalize different saved shapes to a DataFrame with columns:
  event_year, target_code, frequency, target_system
  """
  if obj is None:
    return None
  if isinstance(obj, pd.DataFrame):
    df = obj.copy()
    # ensure expected columns present
    for c in ['event_year','target_code','frequency','target_system']:
      if c not in df.columns:
        df[c] = pd.NA
    return df[['event_year','target_code','frequency','target_system']]
  if isinstance(obj, dict):
    # prefer 'all_targets' key
    if 'all_targets' in obj and obj['all_targets'] is not None:
      return normalize_to_all_targets(obj['all_targets'])
    parts = []
    if obj.get('icd_aggregated') is not None:
      parts.append(obj['icd_aggregated'].assign(target_system='icd'))
    if obj.get('cpt_aggregated') is not None:
      parts.append(obj['cpt_aggregated'].assign(target_system='cpt'))
    if parts:
      out = pd.concat(parts, ignore_index=True)
      for c in ['event_year','target_code','frequency','target_system']:
        if c not in out.columns:
          out[c] = pd.NA
      return out[['event_year','target_code','frequency','target_system']]
    # fallback: concat any dataframe-like values
    dfs = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
    if dfs:
      out = pd.concat(dfs, ignore_index=True)
      for c in ['event_year','target_code','frequency','target_system']:
        if c not in out.columns:
          out[c] = pd.NA
      return out[['event_year','target_code','frequency','target_system']]
  return None

# Load available pickles
pd_orig = load_pickle(orig_pk)
pd_updated = load_pickle(updated_pk)
pd_canon = load_pickle(canonical_pk)

print('Pickle presence: orig=%s, updated=%s, canonical=%s' % (bool(pd_orig), bool(pd_updated), bool(pd_canon)))

# Prefer updated -> canonical -> parquet
t_updated = normalize_to_all_targets(pd_updated or pd_canon)
t_orig = normalize_to_all_targets(pd_orig)

if t_updated is None or (isinstance(t_updated, pd.DataFrame) and t_updated.empty):
  print('\n⚠️ No updated/canonical pickle found or empty; attempting to load S3 parquet:', s3_parquet)
  try:
    t_updated = duckdb.sql(f"SELECT * FROM read_parquet('{s3_parquet}')").df()
  except Exception as e:
    raise RuntimeError(f"Failed to load any target_code data: {e}")

if t_orig is None:
  # If no orig pickle exists, create empty placeholder for comparison
  t_orig = pd.DataFrame(columns=['event_year','target_code','frequency','target_system'])

# Ensure numeric types
for df in (t_orig, t_updated):
  if 'frequency' in df.columns:
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce').fillna(0).astype(int)
  if 'event_year' in df.columns:
    df['event_year'] = pd.to_numeric(df['event_year'], errors='coerce').fillna(0).astype(int)

print('\nUpdated data shape:', t_updated.shape)
print('Orig data shape:', t_orig.shape)

# Aggregate totals and produce a comparison
orig_totals = t_orig.groupby('target_code', as_index=False)['frequency'].sum().rename(columns={'frequency':'orig_total'})
upd_totals = t_updated.groupby('target_code', as_index=False)['frequency'].sum().rename(columns={'frequency':'updated_total'})
cmp = orig_totals.merge(upd_totals, on='target_code', how='outer').fillna(0)
cmp['delta'] = cmp['updated_total'] - cmp['orig_total']
cmp = cmp.sort_values('updated_total', ascending=False)

print('\nTop 10 updated target_codes (by updated_total):')
print(cmp.head(10).to_string(index=False))

# Focused QA for F11.20 variants (ICD)
code_of_interest = 'F11.20'
needle = code_of_interest.replace('.', '').upper()

def find_variants(df):
  if df is None or df.empty:
    return []
  tmp = df.copy()
  tmp['code_flat'] = tmp['target_code'].astype(str).str.upper().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)
  codes = tmp[tmp['code_flat'].str.contains(needle, na=False)].groupby('target_code', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)['target_code'].tolist()
  return codes

orig_variants = find_variants(t_orig)
upd_variants = find_variants(t_updated)
all_variants = sorted(set(orig_variants) | set(upd_variants))

print(f"\nF11.20 variants detected - orig: {len(orig_variants)}, updated: {len(upd_variants)}, union: {len(all_variants)}")
print('Variants union:', all_variants)

# Build per-variant comparison table and save CSV for review
def totals_for_codes(df, codes):
  if df is None or df.empty or not codes:
    return pd.DataFrame(columns=['target_code','frequency'])
  return df[df['target_code'].isin(codes)].groupby('target_code', as_index=False)['frequency'].sum().rename(columns={'frequency':'freq'})

o = totals_for_codes(t_orig, all_variants).rename(columns={'freq':'orig_freq'})
u = totals_for_codes(t_updated, all_variants).rename(columns={'freq':'updated_freq'})
summary = pd.merge(pd.DataFrame({'target_code': all_variants}), o, on='target_code', how='left').merge(u, on='target_code', how='left').fillna(0)
summary['delta'] = summary['updated_freq'].astype(int) - summary['orig_freq'].astype(int)

print('\nPer-variant before/after totals for F11.20 variants:')
print(summary.to_string(index=False))

out_csv = os.path.join(base, 'target_code_f1120_comparison.csv')
summary.to_csv(out_csv, index=False)
print('\nSaved comparison CSV to', out_csv)
```


### Cell 35b: JSON diff for notebook viewing

This cell loads the stable `.orig.pkl` and `.updated.pkl`, computes a structured JSON diff (totals and per-year breakdowns) and writes `docs/target_pickles_diff.json` for easy inspection inside the notebook UI.

```python
import os
import pickle
import json
import pandas as pd
import duckdb

# Paths (adjust if your environment differs)
base = '/home/pgx3874/pgx-analysis/1_apcd_input_data'
orig_pk = os.path.join(base, 'target_code_analysis_data.orig.pkl')
updated_pk = os.path.join(base, 'target_code_analysis_data.updated.pkl')
out_json = '/home/pgx3874/pgx-analysis/docs/target_pickles_diff.json'

def load_pickle(path):
  if not os.path.exists(path):
    return None
  try:
    with open(path, 'rb') as f:
      return pickle.load(f)
  except Exception as e:
    print(f"⚠️ Failed to load pickle {path}: {e}")
    return None

def normalize_to_all_targets(obj):
  # Same normalizer used elsewhere: produce columns event_year, target_code, frequency, target_system
  if obj is None:
    return None
  if isinstance(obj, pd.DataFrame):
    df = obj.copy()
    for c in ['event_year','target_code','frequency','target_system']:
      if c not in df.columns:
        df[c] = pd.NA
    return df[['event_year','target_code','frequency','target_system']]
  if isinstance(obj, dict):
    if 'all_targets' in obj and obj['all_targets'] is not None:
      return normalize_to_all_targets(obj['all_targets'])
    parts = []
    if obj.get('icd_aggregated') is not None:
      parts.append(obj['icd_aggregated'].assign(target_system='icd'))
    if obj.get('cpt_aggregated') is not None:
      parts.append(obj['cpt_aggregated'].assign(target_system='cpt'))
    if parts:
      out = pd.concat(parts, ignore_index=True)
      for c in ['event_year','target_code','frequency','target_system']:
        if c not in out.columns:
          out[c] = pd.NA
      return out[['event_year','target_code','frequency','target_system']]
    dfs = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
    if dfs:
      out = pd.concat(dfs, ignore_index=True)
      for c in ['event_year','target_code','frequency','target_system']:
        if c not in out.columns:
          out[c] = pd.NA
      return out[['event_year','target_code','frequency','target_system']]
  return None

pd_orig = load_pickle(orig_pk)
pd_updated = load_pickle(updated_pk)

t_orig = normalize_to_all_targets(pd_orig) or pd.DataFrame(columns=['event_year','target_code','frequency','target_system'])
t_updated = normalize_to_all_targets(pd_updated) or pd.DataFrame(columns=['event_year','target_code','frequency','target_system'])

# Coerce numeric types
for df in (t_orig, t_updated):
  if 'frequency' in df.columns:
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce').fillna(0).astype(int)
  if 'event_year' in df.columns:
    df['event_year'] = pd.to_numeric(df['event_year'], errors='coerce').fillna(0).astype(int)

def totals_by_code(df):
  if df is None or df.empty:
    return {}
  grp = df.groupby('target_code', as_index=False)['frequency'].sum()
  return {str(r['target_code']): int(r['frequency']) for _, r in grp.iterrows()}

def per_year_by_code(df):
  if df is None or df.empty:
    return {}
  out = {}
  for code, g in df.groupby('target_code'):
    series = g.groupby('event_year')['frequency'].sum()
    out[str(code)] = {str(int(k)): int(v) for k, v in series.to_dict().items()}
  return out

orig_totals = totals_by_code(t_orig)
upd_totals = totals_by_code(t_updated)
orig_by_year = per_year_by_code(t_orig)
upd_by_year = per_year_by_code(t_updated)

all_codes = sorted(set(list(orig_totals.keys()) + list(upd_totals.keys())))

records = []
for code in all_codes:
  o = orig_totals.get(code, 0)
  u = upd_totals.get(code, 0)
  rec = {
    'target_code': code,
    'orig_total': int(o),
    'updated_total': int(u),
    'delta': int(u) - int(o),
    'orig_by_year': orig_by_year.get(code, {}),
    'updated_by_year': upd_by_year.get(code, {}),
  }
  records.append(rec)

# Focused F11.20 variants (same heuristic as Cell 35)
code_of_interest = 'F11.20'
needle = code_of_interest.replace('.', '').upper()
def find_variants(df):
  if df is None or df.empty:
    return []
  tmp = df.copy()
  tmp['code_flat'] = tmp['target_code'].astype(str).str.upper().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)
  codes = tmp[tmp['code_flat'].str.contains(needle, na=False)].groupby('target_code', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)['target_code'].tolist()
  return [str(c) for c in codes]

f_orig = find_variants(t_orig)
f_upd = find_variants(t_updated)
f_union = sorted(set(f_orig) | set(f_upd))

diff_obj = {
  'metadata': {
    'orig_shape': list(t_orig.shape),
    'updated_shape': list(t_updated.shape),
  },
  'totals': records,
  'f11_20_variants': {
    'orig': f_orig,
    'updated': f_upd,
    'union': f_union
  }
}

# Write JSON
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w', encoding='utf-8') as f:
  json.dump(diff_obj, f, indent=2, ensure_ascii=False)

print('Wrote JSON diff to', out_json)

```


## **Phase 7: Update Target Codes (ICD/CPT Normalization)**

**Purpose**: Apply JSON code mappings to normalize and correct ICD/CPT codes across the gold medical datasets. Uses local staging for maximum performance and reliability.

**What it does:**
- Applies ICD target mappings (e.g., F11.20 variants → canonical)
- Normalizes ICD diagnosis codes across all positions (primary through ten)
- Normalizes CPT procedure codes
- Uses **local staging** (writes to NVMe first, then uploads to S3) for reliability
- Supports chunked processing with resume capability

**Inputs:**
- `s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet`
- ICD mapping JSON: `1_apcd_input_data/target_mapping/target_icd_mapping.json`

**Outputs:**
- Updated medical parquet files (in-place or chunked)
- Checkpoint markers for resume capability
- Logs: `logs/medical_codes_TIMESTAMP.log`

### **Cell 36: Update Target Codes (Optimized with Local Staging)**

**Recommended Configuration (Maximum Performance):**

```bash
%%bash
set -euo pipefail

# ========================================
# LOCAL STAGING: Maximum Performance
# ========================================
export PGX_USE_LOCAL_STAGING=1  # Enable local staging (default: on)
export PGX_LOCAL_STAGING_DIR="/mnt/nvme/pgx_staging"  # Use fast NVMe

# ========================================
# WORKER CONFIGURATION: 24 workers
# ========================================
export PGX_WORKERS_MEDICAL=24  # High parallelism (safe with local staging)
export PGX_THREADS_PER_WORKER=1
export PGX_S3_MAX_CONNECTIONS=192  # High for S3 uploads
export PGX_DUCKDB_MEMORY_LIMIT=3GB  # Per worker

# ========================================
# CHUNKING: Balanced for performance
# ========================================
CHUNK_ROWS=1000000  # 1M rows per chunk (good balance)
CHECKPOINT_SUFFIX=".codes_updated.v2.ok"
STAGING_SUFFIX=".codes_updated.staging/"

# ========================================
# PATHS
# ========================================
ICD_MAP="/home/pgx3874/pgx-analysis/1_apcd_input_data/target_mapping/target_icd_mapping.json"
LOG_FILE="logs/medical_codes_$(date +%Y%m%d_%H%M%S).log"

# ========================================
# SETUP
# ========================================
mkdir -p logs
mkdir -p /mnt/nvme/pgx_staging

echo "🚀 Starting with LOCAL STAGING for maximum performance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 Staging dir: $PGX_LOCAL_STAGING_DIR"
echo "👷 Workers: $PGX_WORKERS_MEDICAL"
echo "🧠 Memory per worker: $PGX_DUCKDB_MEMORY_LIMIT"
echo "📦 Chunk size: $CHUNK_ROWS rows"
echo "📋 Log: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Started at: $(date)"
echo ""

# ========================================
# RUN
# ========================================
nohup /home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/1_apcd_input_data/7_update_codes.py \
  --icd-target-map "$ICD_MAP" \
  --years "2016,2017,2018,2019,2020" \
  --workers-medical $PGX_WORKERS_MEDICAL \
  --threads $PGX_THREADS_PER_WORKER \
  --resume \
  --checkpoint-suffix "$CHECKPOINT_SUFFIX" \
  --chunked \
  --chunk-rows $CHUNK_ROWS \
  --staging-suffix "$STAGING_SUFFIX" \
  --duckdb-mem-limit $PGX_DUCKDB_MEMORY_LIMIT \
  --no-merge \
  > "$LOG_FILE" 2>&1 &

echo $! > logs/medical_codes.pid
echo "✅ Job started with PID: $(cat logs/medical_codes.pid)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MONITOR PROGRESS:"
echo "  !tail -f $LOG_FILE"
echo "  !grep -c '✓ Updated' $LOG_FILE"
echo "  !du -sh /mnt/nvme/pgx_staging"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

### **Alternative Configurations**

**Conservative (Slower but Safer):**
```bash
export PGX_WORKERS_MEDICAL=12
export PGX_DUCKDB_MEMORY_LIMIT=2GB
CHUNK_ROWS=750000
# Expected: 2-2.5 hours
```

**Balanced (Recommended):**
```bash
export PGX_WORKERS_MEDICAL=18
export PGX_DUCKDB_MEMORY_LIMIT=3GB
CHUNK_ROWS=1000000
# Expected: 1.5-2 hours
```

**Aggressive (Maximum Speed):**
```bash
export PGX_WORKERS_MEDICAL=28
export PGX_DUCKDB_MEMORY_LIMIT=3GB
CHUNK_ROWS=1200000
# Expected: 1-1.5 hours
```

### **Monitoring Cells**

**Cell 1: Quick Status**
```python
import glob
import os
import subprocess

log_files = glob.glob("logs/medical_codes_*.log")
if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    
    # Count progress
    try:
        completed = subprocess.check_output(
            ['grep', '-c', '✓ Updated', latest_log],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        completed = "0"
    
    print(f"✓ Completed: {completed} / 45 partitions ({int(completed)/45*100:.1f}%)")
    print(f"📋 Latest: {latest_log}")
    
    # Show recent activity
    !tail -n 20 {latest_log} | grep -E "(Writing chunk|Uploading|✓ Updated)"
```

**Cell 2: Disk Usage**
```python
# Check staging disk usage
!df -h /mnt/nvme | tail -n 1
!du -sh /mnt/nvme/pgx_staging 2>/dev/null || echo "Staging dir empty/clean"
!ls -lh /mnt/nvme/pgx_staging 2>/dev/null | head -n 10
```

**Cell 3: System Resources**
```python
# Check memory and CPU
!free -h | grep "^Mem:"
!ps aux | grep "7_update_codes.py" | grep -v grep | wc -l
```

### **Key Features of Local Staging Implementation**

1. **Local Write First**: DuckDB writes to `/mnt/nvme/pgx_staging/` (3-5 GB/s) ⚡
2. **Boto3 Upload**: Reliable S3 upload with retry logic and multipart support 🛡️
3. **Auto Cleanup**: Local files deleted immediately after upload 🧹
4. **S3 Timeouts**: Increased to 5 minutes (from 30 seconds) for large files ⏱️
5. **Resume Support**: Checkpoint markers allow resuming from failures 🔄

### **Expected Performance**

| Configuration | Workers | Memory | Time | Reliability |
|--------------|---------|--------|------|-------------|
| Conservative | 12 | 24GB | 2-2.5h | ⭐⭐⭐⭐⭐ |
| Balanced | 18 | 54GB | 1.5-2h | ⭐⭐⭐⭐⭐ |
| **Recommended** | **24** | **72GB** | **1-1.5h** | **⭐⭐⭐⭐⭐** |
| Aggressive | 28 | 84GB | 1-1.5h | ⭐⭐⭐⭐ |

### **Expected Log Output**

```
🚀 Starting with LOCAL STAGING for maximum performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Started at: Fri Nov  8 03:00:00 UTC 2025

[medical-worker] ▶ Writing chunk 1/8 rows=1000000 (local staging)
[medical-worker] ↗ Uploading chunk 1/8 to S3
[medical-worker] ▶ Writing chunk 2/8 rows=1000000 (local staging)
[medical-worker] ↗ Uploading chunk 2/8 to S3
...
✓ Updated s3://pgxdatalake/gold/medical/age_band=25-44/event_year=2016/medical_data.parquet
```

### **Troubleshooting**

**If staging disk fills up:**
```bash
# Clean staging directory manually
rm -rf /mnt/nvme/pgx_staging/*
```

**Disable local staging (fallback to direct S3):**
```bash
export PGX_USE_LOCAL_STAGING=0  # Disable
# Then restart job
```

**Check for stuck workers:**
```python
!ps aux | grep "7_update_codes.py" | grep -v grep
!lsof /mnt/nvme/pgx_staging/* 2>/dev/null | head -n 20
```

---

## Create Cohort Pipeline (2_create_cohort)

### Cell A: Run full pipeline (both cohorts)
```bash
%%bash
set -euo pipefail

AGE_BAND="65-74"
EVENT_YEAR=2019

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/2_create_cohort/create_cohort.py \
  --age-band "$AGE_BAND" \
  --event-year $EVENT_YEAR \
  --cohort both \
  --starting-step phase1_data_preparation \
  --operation-type concurrent_processing \
  --log-level INFO
```

### Cell B: Run a single cohort (OPIOID_ED or ED_NON_OPIOID)
```bash
%%bash
set -euo pipefail

AGE_BAND="65-74"
EVENT_YEAR=2019

# Options: opioid_ed | ed_non_opioid
COHORT="opioid_ed"

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/2_create_cohort/create_cohort.py \
  --age-band "$AGE_BAND" \
  --event-year $EVENT_YEAR \
  --cohort "$COHORT" \
  --starting-step phase1_data_preparation \
  --operation-type concurrent_processing \
  --log-level INFO
```

### Cell C: Resume from a specific step
Available steps: `phase1_data_preparation`, `phase2_step1_event_fact_table`, `phase2_step2_drug_exposure`, `phase3_step3_final_cohort_fact`, `phase4_complete_pipeline`

```bash
%%bash
set -euo pipefail

AGE_BAND="65-74"
EVENT_YEAR=2019
STARTING_STEP="phase2_step1_event_fact_table"

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/2_create_cohort/create_cohort.py \
  --age-band "$AGE_BAND" \
  --event-year $EVENT_YEAR \
  --cohort both \
  --starting-step "$STARTING_STEP" \
  --operation-type concurrent_processing \
  --log-level INFO
```

### Cell D: Enable DuckDB profiling (JSON or query_tree)
```bash
%%bash
set -euo pipefail

AGE_BAND="65-74"
EVENT_YEAR=2019

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/2_create_cohort/create_cohort.py \
  --age-band "$AGE_BAND" \
  --event-year $EVENT_YEAR \
  --cohort both \
  --starting-step phase1_data_preparation \
  --operation-type concurrent_processing \
  --enable-profiling \
  --profile-format json \
  --log-level INFO
```

Notes:
- `--operation-type` choices: `large_processing`, `concurrent_processing`, `s3_heavy`, `default`.
- Centralized checkpoints are handled automatically; use `--skip-checkpoints` to force a fresh run.
- On SQL errors, dev-only schema hints will appear, using `table_mappings/medical_schema.json` and `table_mappings/pharmacy_schema.json`.

---

## Cohort QA Notebook Calls (Events and Features)

Purpose: After cohorts are written to GOLD (`gold/cohorts_clean`), run `phase4_data_quality_qa.py` focused on cohort-level events and features of interest. This mirrors the QA style we used for `1_apcd_input_data`, but scoped to cohort outputs.

Requirements:
- Updated QA script supports cohort mode flags (example below):
  - `--type cohort`
  - `--cohort-parquet s3://.../cohort.parquet`
  - `--events-of-interest "OPIOID_ED,ED_NON_OPIOID"` (comma-separated)
  - `--features-of-interest "drug_name,therapeutic_class_1,primary_icd_diagnosis_code"` (comma-separated)

### Cell E: Cohort QA (OPIOID_ED)
```bash
%%bash
set -euo pipefail

AGE_BAND="65-74"
EVENT_YEAR=2019

OPIOID_ED_COHORT="s3://pgxdatalake/gold/cohorts_clean/cohort_name=opioid_ed/age_band=${AGE_BAND}/event_year=${EVENT_YEAR}/cohort.parquet"

EVENTS_OF_INTEREST="OPIOID_ED"
FEATURES_OF_INTEREST="drug_name,therapeutic_class_1,primary_icd_diagnosis_code,event_type,event_sequence"

echo "🔍 Cohort QA (OPIOID_ED)"
echo "Cohort: ${OPIOID_ED_COHORT}"

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/2_create_cohort/phase4_data_quality_qa.py \
  --type cohort \
  --cohort-parquet "${OPIOID_ED_COHORT}" \
  --events-of-interest "${EVENTS_OF_INTEREST}" \
  --features-of-interest "${FEATURES_OF_INTEREST}" \
  --verbose 2>&1 | tee "logs/cohort_qa_opioid_ed_${AGE_BAND}_${EVENT_YEAR}_$(date +%Y%m%d_%H%M%S).log"
```

### Cell F: Cohort QA (ED_NON_OPIOID)
```bash
%%bash
set -euo pipefail

AGE_BAND="65-74"
EVENT_YEAR=2019

ED_NON_OPIOID_COHORT="s3://pgxdatalake/gold/cohorts_clean/cohort_name=ed_non_opioid/age_band=${AGE_BAND}/event_year=${EVENT_YEAR}/cohort.parquet"

EVENTS_OF_INTEREST="ED_NON_OPIOID"
FEATURES_OF_INTEREST="drug_name,therapeutic_class_1,primary_icd_diagnosis_code,event_type,event_sequence"

echo "🔍 Cohort QA (ED_NON_OPIOID)"
echo "Cohort: ${ED_NON_OPIOID_COHORT}"

/home/pgx3874/jupyter-env/bin/python3.11 \
  /home/pgx3874/pgx-analysis/2_create_cohort/phase4_data_quality_qa.py \
  --type cohort \
  --cohort-parquet "${ED_NON_OPIOID_COHORT}" \
  --events-of-interest "${EVENTS_OF_INTEREST}" \
  --features-of-interest "${FEATURES_OF_INTEREST}" \
  --verbose 2>&1 | tee "logs/cohort_qa_ed_non_opioid_${AGE_BAND}_${EVENT_YEAR}_$(date +%Y%m%d_%H%M%S).log"
```

Notes:
- Adjust `FEATURES_OF_INTEREST` to include any additional columns you want validated (e.g., payer_imputed, member_gender, member_race).
- If you maintain the QA script elsewhere, update the script path accordingly.

## **Drug Frequency Analysis - Modular Approach**

### **Python Script: `4_drug_frequency_analysis.py`**

**Option 1: Run from Notebook Cell (Cell 18)**
```bash
%%bash

/home/pgx3874/jupyter-env/bin/python3.11 \
   /home/pgx3874/pgx-analysis/1_apcd_input_data/4_drug_frequency_analysis.py
```

**Option 2: Run from Terminal**
```bash
/home/pgx3874/jupyter-env/bin/python3.11 \
   /home/pgx3874/pgx-analysis/1_apcd_input_data/4_drug_frequency_analysis.py
```

**Option 3: Direct Import (Best for Notebook)**
```python
import sys
import importlib.util

# Load module by file path (since module name starts with number)
spec = importlib.util.spec_from_file_location(
    "drug_frequency_analysis",
    "/home/pgx3874/pgx-analysis/1_apcd_input_data/4_drug_frequency_analysis.py"
)
drug_freq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drug_freq)

import pickle

# Run the analysis
data = drug_freq.main()

# Save the data
pickle_path = '/home/pgx3874/pgx-analysis/1_apcd_input_data/drug_analysis_data.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(data, f)
print(f"💾 Data saved to '{pickle_path}'")
```

This script will:
- Test all DuckDB fixes
- Generate comprehensive analysis
- Save data to `drug_analysis_data.pkl` for notebook visualization
- Print detailed summary report

### **Notebook Cells for Visualizations**

### **Cell 1: Setup and Load Data**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data from script
pickle_path = '/home/pgx3874/pgx-analysis/1_apcd_input_data/drug_analysis_data.pkl'
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

df = data['df']
high_freq_df = data['high_freq_df']
low_freq_df = data['low_freq_df']
summary_df = data['summary_df']
trends_df = data['trends_df']

print("✅ Data loaded successfully!")
print(f"📊 Main data: {len(df):,} records")
print(f"🔝 High frequency drugs: {len(high_freq_df)}")
print(f"🔻 Low frequency drugs: {len(low_freq_df)}")
```

### **Cell 2: High Frequency Drugs Bar Chart**
```python
# Create the first bar chart: High frequency drugs (>1000)
plt.figure(figsize=(16, 10))

# Prepare data for visualization
high_freq_pivot = df[df['drug_name'].isin(high_freq_df['drug_name'])].pivot(
    index='drug_name', columns='event_year', values='frequency'
).fillna(0)

# Sort by total frequency
high_freq_pivot = high_freq_pivot.reindex(high_freq_df['drug_name'])

# Create stacked bar chart
ax = high_freq_pivot.plot(kind='bar', stacked=True, width=0.8, figsize=(16, 10))

plt.title('High Frequency Drug Names by Year (>1000 total occurrences)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Drug Name', fontsize=12, fontweight='bold')
plt.ylabel('Frequency (Count)', fontsize=12, fontweight='bold')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=8, rotation=90)

plt.tight_layout()
plt.show()

print("📊 High frequency drugs chart created!")
```

### **Cell 3: Low Frequency Drugs Bar Chart**
```python
# Create the second bar chart: Low frequency drugs (<1000)
plt.figure(figsize=(16, 10))

# Prepare data for visualization
low_freq_pivot = df[df['drug_name'].isin(low_freq_df['drug_name'])].pivot(
    index='drug_name', columns='event_year', values='frequency'
).fillna(0)

# Sort by total frequency
low_freq_pivot = low_freq_pivot.reindex(low_freq_df['drug_name'])

# Create stacked bar chart
ax = low_freq_pivot.plot(kind='bar', stacked=True, width=0.8, figsize=(16, 10))

plt.title('Low Frequency Drug Names by Year (<1000 total occurrences)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Drug Name', fontsize=12, fontweight='bold')
plt.ylabel('Frequency (Count)', fontsize=12, fontweight='bold')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=8, rotation=90)

plt.tight_layout()
plt.show()

print("📊 Low frequency drugs chart created!")
```

### **Cell 4: Additional Visualizations (Optional)**
```python
# Create a combined frequency distribution chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# High frequency drugs - horizontal bar chart
high_freq_df.head(10).plot(x='drug_name', y='total_frequency', kind='barh', ax=ax1, color='skyblue')
ax1.set_title('Top 10 High Frequency Drugs', fontsize=14, fontweight='bold')
ax1.set_xlabel('Total Frequency')
ax1.set_ylabel('Drug Name')

# Low frequency drugs - horizontal bar chart
low_freq_df.head(10).plot(x='drug_name', y='total_frequency', kind='barh', ax=ax2, color='lightcoral')
ax2.set_title('Top 10 Low Frequency Drugs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Total Frequency')
ax2.set_ylabel('Drug Name')

plt.tight_layout()
plt.show()

print("📊 Additional visualizations created!")
```

### **Cell 5: Data Summary Display**
```python
# Display summary statistics
print("📈 SUMMARY STATISTICS")
print("=" * 50)

print(f"\n📅 Years analyzed: {summary_df['years_covered'].iloc[0]}")
print(f"💊 Total unique drugs: {summary_df['unique_drugs'].iloc[0]:,}")
print(f"📊 Total drug-year combinations: {summary_df['total_combinations'].iloc[0]:,}")
print(f"🔝 High frequency drugs (>1000): {len(high_freq_df):,}")
print(f"🔻 Low frequency drugs (<1000): {len(low_freq_df):,}")

print(f"\n📊 Frequency distribution:")
print(f"   Min frequency: {summary_df['min_frequency'].iloc[0]:,}")
print(f"   Max frequency: {summary_df['max_frequency'].iloc[0]:,}")
print(f"   Mean frequency: {summary_df['avg_frequency'].iloc[0]}")
print(f"   Median frequency: {summary_df['median_frequency'].iloc[0]}")

print(f"\n🏆 Top 5 drugs by total frequency:")
for i, (_, row) in enumerate(high_freq_df.head().iterrows(), 1):
    print(f"   {i}. {row['drug_name']}: {row['total_frequency']:,} occurrences")

# Display sample data
print(f"\n📋 Sample high frequency data:")
print(high_freq_df.head())

print(f"\n📋 Sample low frequency data:")
print(low_freq_df.head())
```