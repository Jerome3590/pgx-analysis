````markdown
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

... (archived content)

````
