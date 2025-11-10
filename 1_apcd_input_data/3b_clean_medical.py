#!/usr/bin/env python3
"""
Simplified Medical Data Cleaning Pipeline - Version 1997 + 13
Based on clean_pharmacy.py template (removes complex DuckDB chaining to fix memory_limit issues)
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import json
import boto3

try:
    import psutil
except ImportError:
    psutil = None  # psutil is optional for memory logging

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.logging_utils import (
    setup_logging,
    save_logs_to_s3,
    save_logs_checkpoint,
    save_logs_immediate,
)
from helpers_1997_13.data_utils import validate_data_for_blank_strings
from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
from helpers_1997_13.s3_utils import (
    s3_directory_exists_with_files,
    validate_s3_input_paths,
    convert_raw_to_imputed_path,
    delete_s3_parquet_files,
)
from helpers_1997_13.constants import S3_BUCKET


def build_optimized_pipeline(
    age_band: str,
    event_year: int,
    medical_input: str,
    demographics_lookup: str,
    output_root: str,
    conn,
    logger,
    log_buffer,
    lookahead_years: int = 5,
    resume: bool = True,
):
    """Build optimized pipeline using pre-imputed demographics"""

    # Create unique identifiers for this process to avoid conflicts
    process_id = os.getpid()
    # Note: Keep hyphens in age_band for Hive-style partitioning consistency
    # Only sanitize for run_id logging (not for S3 paths)
    safe_run_id = age_band.replace("-", "_")
    run_id = f"{safe_run_id}_{event_year}_{process_id}"  # Unique ID for this run (underscores for DuckDB table names)

    logger.info(f"📊 Processing: {age_band}/{event_year}")
    logger.info(f"📊 Process ID: {process_id}")
    logger.info(f"📊 Medical input: {medical_input}")
    logger.info(f"📊 Lookahead years: {lookahead_years}")

    # Determine if we're using pre-imputed data or need demographics lookup
    using_imputed_data = demographics_lookup is None
    logger.info(f"📊 Using pre-imputed data: {using_imputed_data}")
    if demographics_lookup:
        logger.info(f"📊 mi_person_key demographics lookup: {demographics_lookup}")
    else:
        logger.info(
            f"📊 Using pre-imputed partitioned data (no demographics lookup needed)"
        )

    # Save initial checkpoint
    save_logs_checkpoint(
        log_buffer,
        "medical_optimized",
        age_band,
        event_year,
        "step0_pipeline_started",
        logger=logger,
    )

    # Idempotent resume: if partition already exists with consistent filename, skip work
    if resume:
        try:
            # Check for the specific medical_data.parquet file (consistent filename for Glue/Athena)
            output_file = f"{output_root}/age_band={age_band}/event_year={event_year}/medical_data.parquet"
            if output_file.startswith("s3://"):
                # S3 check
                from urllib.parse import urlparse
                import boto3

                parsed = urlparse(output_file)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                try:
                    s3 = boto3.client("s3")
                    s3.head_object(Bucket=bucket, Key=key)
                    logger.info(
                        f"✅ Output already exists for {age_band}/{event_year}; skipping per resume=true: {output_file}"
                    )
                    return {
                        "output_path": output_file,
                        "final_count": None,
                        "final_patients": None,
                    }
                except s3.exceptions.ClientError:
                    # File doesn't exist, continue with processing
                    pass
            else:
                # Local check
                if os.path.exists(output_file):
                    logger.info(
                        f"✅ Output already exists for {age_band}/{event_year}; skipping per resume=true: {output_file}"
                    )
                    return {
                        "output_path": output_file,
                        "final_count": None,
                        "final_patients": None,
                    }
        except Exception as e:
            logger.warning(f"⚠️ Resume check failed (continuing): {e}")

    # Step 1: Load medical data (optimized for partitioned data if available)
    logger.info("📊 Step 1: Loading medical data...")
    
    # Log memory status before loading
    try:
        if psutil:
            mem = psutil.virtual_memory()
            logger.info(f"💾 Memory before Step 1: {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
    except Exception:
        pass

    # Derive imputed partitioned path from medical_input using centralized utility
    # Convert: s3://pgxdatalake/silver/medical/*.parquet
    # To:      s3://pgxdatalake/silver/imputed/medical_partitioned/age_band={age_band}/event_year={event_year}
    partitioned_path = convert_raw_to_imputed_path(
        medical_input, "medical", age_band, event_year
    )
    logger.info(f"📊 Checking for imputed partitioned data at: {partitioned_path}")

    # Validate that the partitioned data actually exists before trying to read it
    if not s3_directory_exists_with_files(partitioned_path, file_pattern="*.parquet"):
        error_msg = f"❌ No parquet files found at {partitioned_path}. Run global_imputation.py first!"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("✅ Using imputed partitioned data - optimized loading")

    # Load imputed partitioned data (standardized schema from global_imputation)
    conn.sql(
        f"""
        CREATE OR REPLACE TABLE medical_filtered_{run_id} AS
        SELECT
                CAST(mi_person_key AS VARCHAR) AS mi_person_key,
                age_imputed AS member_age_dos,
                zip_imputed AS member_zip_code_dos,
                county_imputed AS member_county_dos,
                CASE
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 0  AND 12  THEN '0-12'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 13 AND 24  THEN '13-24'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 25 AND 44  THEN '25-44'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 45 AND 54  THEN '45-54'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 55 AND 64  THEN '55-64'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 65 AND 74  THEN '65-74'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 75 AND 84  THEN '75-84'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 85 AND 94  THEN '85-94'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 95 AND 114 THEN '95-114'
                    ELSE 'Other'
                END AS member_age_band_dos,
                gender_imputed AS member_gender,
                race_imputed AS member_race,
                payer_imputed AS payer_type,
                cchg_label, cchg_grouping, hcg_setting, hcg_line, hcg_detail,
                place_of_service, admit_type, primary_icd_diagnosis_code, primary_icd_rollup,
                primary_icd_ccs_level_1, primary_icd_ccs_level_2, primary_icd_ccs_level_3,
                two_icd_diagnosis_code,
                two_icd_rollup,
                three_icd_diagnosis_code,
                three_icd_rollup,
                four_icd_diagnosis_code,
                four_icd_rollup,
                five_icd_diagnosis_code,
                five_icd_rollup,
                six_icd_diagnosis_code,
                six_icd_rollup,
                seven_icd_diagnosis_code,
                seven_icd_rollup,
                eight_icd_diagnosis_code,
                eight_icd_rollup,
                nine_icd_diagnosis_code,
                nine_icd_rollup,
                ten_icd_diagnosis_code,
                ten_icd_diagnosis_rollup,
                procedure_code, procedure_name,
                procedure_family_1, procedure_family_2, procedure_family_3,
                two_icd_procedure_code,
                three_icd_procedure_code,
                four_icd_procedure_code,
                five_icd_procedure_code,
                six_icd_procedure_code,
                seven_icd_procedure_code,
                eight_icd_procedure_code,
                nine_icd_procedure_code,
                ten_icd_procedure_code,
                cpt_mod_1_code, cpt_mod_2_code, billing_provider_name, billing_provider_zip,
                billing_provider_county, billing_provider_state, service_provider_name,
                claim_id,
                service_provider_zip, service_provider_county, service_provider_state,
                TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') AS event_date,
                SUBSTR(CAST(incurred_date AS VARCHAR), 1, 4) AS event_year -- keep as string for joins
            FROM read_parquet('{partitioned_path}/**/*.parquet')
            WHERE incurred_date IS NOT NULL
                AND CAST(incurred_date AS VARCHAR) <> ''
                AND regexp_matches(CAST(incurred_date AS VARCHAR), '^[0-9]{{8}}$')
                AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
                AND event_date IS NOT NULL
        """
    )
    logger.info("📊 Loaded imputed data from partitioned source")
    
    # Log memory after loading
    try:
        if psutil:
            mem = psutil.virtual_memory()
            logger.info(f"💾 Memory after loading: {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
    except Exception:
        pass

    logger.info("📊 Counting initial records...")
    initial_count = conn.sql(
        f"SELECT COUNT(*) FROM medical_filtered_{run_id}"
    ).fetchone()[0]
    logger.info("📊 Counting initial patients...")
    initial_patients = conn.sql(
        f"SELECT COUNT(DISTINCT mi_person_key) FROM medical_filtered_{run_id}"
    ).fetchone()[0]

    logger.info(f"📊 Initial medical records (year {event_year}): {initial_count:,}")
    logger.info(f"📊 Initial medical patients: {initial_patients:,}")

    # Step 2: Apply age band filtering
    logger.info("📊 Step 2: Applying age band filtering...")

    # Age band filtering is already done in partitioned data, but let's verify
    age_band_count = conn.sql(
        f"""
        SELECT COUNT(*) 
        FROM medical_filtered_{run_id} 
        WHERE member_age_band_dos = '{age_band}'
    """
    ).fetchone()[0]

    logger.info(f"📊 Records matching age band {age_band}: {age_band_count:,}")

    # Step 3: Apply year filtering and date validation
    logger.info("📊 Step 3: Applying year filtering and date validation...")

    year_count = conn.sql(
        f"""
        SELECT COUNT(*) 
        FROM medical_filtered_{run_id} 
        WHERE event_year = '{event_year}'
          AND EXTRACT(YEAR FROM event_date) > 1900 
          AND EXTRACT(YEAR FROM event_date) < 2100
    """
    ).fetchone()[0]

    logger.info(
        f"📊 Records matching year {event_year} with valid dates: {year_count:,}"
    )

    # Step 4: Compute data quality levels (optimized - filter first, then compute)
    logger.info("📊 Step 4: Computing data quality levels...")

    # Create enriched table with quality tracking (optimized: filter first, then compute quality)
    conn.sql(
        f"""
        CREATE OR REPLACE TABLE medical_enriched_{run_id} AS
        SELECT
            *,
            -- Data quality computation (optimized with single expression)
            (CASE WHEN member_age_dos IS NULL THEN 1 ELSE 0 END +
             CASE WHEN member_gender IS NULL THEN 1 ELSE 0 END +
             CASE WHEN member_race IS NULL THEN 1 ELSE 0 END +
             CASE WHEN member_zip_code_dos IS NULL THEN 1 ELSE 0 END +
             CASE WHEN member_county_dos IS NULL THEN 1 ELSE 0 END +
             CASE WHEN payer_type IS NULL THEN 1 ELSE 0 END) AS total_missing_fields,
            CASE 
                WHEN member_age_dos IS NOT NULL 
                    AND member_gender IS NOT NULL 
                    AND member_race IS NOT NULL 
                    AND payer_type IS NOT NULL 
                THEN 'complete'
                ELSE 'partial'
            END AS data_quality_level
        FROM (
            -- OPTIMIZATION: Filter to target partition FIRST before computing quality
            SELECT * 
            FROM medical_filtered_{run_id}
            WHERE member_age_band_dos = '{age_band}' 
                AND event_year = '{event_year}'
                AND member_age_band_dos <> 'Other'
                AND EXTRACT(YEAR FROM event_date) > 1900 
                AND EXTRACT(YEAR FROM event_date) < 2100
        ) AS filtered_partition
        """
    )

    enriched_count = conn.sql(
        f"SELECT COUNT(*) FROM medical_enriched_{run_id}"
    ).fetchone()[0]
    logger.info(f"📊 Enriched medical records: {enriched_count:,}")
    
    # Log memory after quality computation
    try:
        if psutil:
            mem = psutil.virtual_memory()
            logger.info(f"💾 Memory after Step 4 (quality): {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
    except Exception:
        pass

    # Step 5: Final data preparation
    logger.info("📊 Step 5: Final data preparation...")

    # Create final table with all necessary columns (consistent partitioning for Glue/Athena)
    conn.sql(
        f"""
        CREATE OR REPLACE TABLE medical_final_{run_id} AS
        SELECT
            -- Partition columns for Athena/Hive layout
            member_age_band_dos AS age_band,
            CAST(event_year AS INTEGER) AS event_year,
            -- Core fields
            mi_person_key, member_age_dos, member_zip_code_dos,
            member_county_dos, member_age_band_dos, member_gender,
            member_race, event_date, incurred_date,
            -- Medical-specific fields
            cchg_label, cchg_grouping, hcg_setting, hcg_line, hcg_detail, place_of_service, admit_type,
            primary_icd_diagnosis_code, primary_icd_rollup, primary_icd_ccs_level_1,
            primary_icd_ccs_level_2, primary_icd_ccs_level_3, two_icd_diagnosis_code, two_icd_rollup,
            three_icd_diagnosis_code, three_icd_rollup, four_icd_diagnosis_code, four_icd_rollup,
            five_icd_diagnosis_code, five_icd_rollup, six_icd_diagnosis_code, six_icd_rollup,
            seven_icd_diagnosis_code, seven_icd_rollup, eight_icd_diagnosis_code, eight_icd_rollup,
            nine_icd_diagnosis_code, nine_icd_rollup, ten_icd_diagnosis_code, ten_icd_diagnosis_rollup,
            procedure_code, procedure_name, procedure_family_1, procedure_family_2, procedure_family_3,
            two_icd_procedure_code, three_icd_procedure_code, four_icd_procedure_code, five_icd_procedure_code,
            six_icd_procedure_code, seven_icd_procedure_code, eight_icd_procedure_code,
            nine_icd_procedure_code, ten_icd_procedure_code, cpt_mod_1_code, cpt_mod_2_code,
            billing_provider_name, billing_provider_zip, billing_provider_county, billing_provider_state,
            service_provider_name, service_provider_zip, service_provider_county, service_provider_state,
            -- Insurance
            claim_id, payer_type,
            -- Quality tracking
            total_missing_fields, data_quality_level
        FROM medical_enriched_{run_id}
        """
    )

    final_count = conn.sql(f"SELECT COUNT(*) FROM medical_final_{run_id}").fetchone()[0]
    final_patients = conn.sql(
        f"SELECT COUNT(DISTINCT mi_person_key) FROM medical_final_{run_id}"
    ).fetchone()[0]

    # Log data quality statistics
    quality_stats = conn.sql(
        f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN data_quality_level = 'complete' THEN 1 END) as complete_records,
            COUNT(CASE WHEN data_quality_level = 'partial' THEN 1 END) as partial_records,
            ROUND(COUNT(CASE WHEN data_quality_level = 'complete' THEN 1 END) * 100.0 / COUNT(*), 2) as completeness_rate
        FROM medical_final_{run_id}
        """
    ).fetchone()

    logger.info(f"📊 Final medical records: {final_count:,}")
    logger.info(f"📊 Final medical patients: {final_patients:,}")
    logger.info(
        f"📊 Data quality: Complete={quality_stats[1]:,} ({quality_stats[3]}%), Partial={quality_stats[2]:,}"
    )

    # Step 6: Write directly to S3 (memory-efficient, no intermediate copy)
    logger.info("📊 Step 6: Writing to S3...")
    
    # Log memory before write
    try:
        if psutil:
            mem = psutil.virtual_memory()
            logger.info(f"💾 Memory before write: {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
    except Exception:
        pass

    # Strict behavior: pre-delete all Parquet files in gold partition (S3 only)
    gold_partition = f"{output_root}/age_band={age_band}/event_year={event_year}"
    delete_s3_parquet_files(gold_partition, logger)

    # Validate data for blank strings before writing
    try:
        validate_data_for_blank_strings(f"medical_final_{run_id}", logger, conn)
    except Exception as val_e:
        logger.warning(f"⚠️ Blank string validation warning: {val_e}")

    # Write directly to gold partition (memory-efficient: no ORDER BY, no intermediate copy)
    gold_partition = f"{output_root}/age_band={age_band}/event_year={event_year}"
    gold_file = f"{gold_partition}/medical_data.parquet"
    
    logger.info(f"Writing Parquet (direct) → {gold_file}")
    
    conn.sql(
        f"""
            COPY (
                SELECT * EXCLUDE (age_band, event_year) FROM medical_final_{run_id}
            )
        TO '{gold_file}' 
        (FORMAT PARQUET, ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE true)
        """
    )
    
    # COPY command is synchronous - S3 write is complete at this point
    # Verify the write succeeded by checking row count
    actual_output = f"{output_root}/age_band={age_band}/event_year={event_year}/medical_data.parquet"
    logger.info(f"✓ S3 write complete (synchronous): {actual_output}")
    logger.info(f"✓ Wrote {final_count:,} records, {final_patients:,} patients")

    # Save final checkpoint after successful completion
    save_logs_checkpoint(
        log_buffer,
        "medical_optimized",
        age_band,
        event_year,
        "step6_data_written_success",
        logger=logger,
    )

    # Clean up temporary views and tables
    logger.info("📊 Cleaning up temporary objects...")
    try:
        conn.sql(f"DROP TABLE IF EXISTS medical_filtered_{run_id}")
        conn.sql(f"DROP TABLE IF EXISTS medical_enriched_{run_id}")
        conn.sql(f"DROP TABLE IF EXISTS medical_final_{run_id}")
        logger.info("✅ Temporary objects cleaned up")
        
        # Log final memory status
        try:
            if psutil:
                mem = psutil.virtual_memory()
                logger.info(f"💾 Memory after cleanup: {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"⚠️ Could not clean up some temporary objects: {e}")

    return {
        "output_path": actual_output,
        "final_count": final_count,
        "final_patients": final_patients,
    }


def main():
    parser = argparse.ArgumentParser(description="Simplified Medical Data Cleaning")
    parser.add_argument("--age-band", required=True, help="Age band to process")
    parser.add_argument(
        "--event-year", type=int, required=True, help="Event year to process"
    )
    parser.add_argument(
        "--medical-input", required=True, help="S3 path to pre-imputed medical data"
    )
    parser.add_argument(
        "--demographics-lookup",
        help="S3 path to mi_person_key demographics lookup table (optional for optimized mode)",
    )
    parser.add_argument(
        "--output-root", required=True, help="S3 root path for output files"
    )
    parser.add_argument(
        "--lookahead-years", type=int, default=5, help="Lookahead years for analysis"
    )
    parser.add_argument("--tmp-dir", help="Temporary directory for DuckDB")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable idempotent resume (always re-run)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--aggregate-root", default="s3://pgxdatalake/pgx_pipeline/", help="S3 root for aggregated run summaries (for BI)")

    args = parser.parse_args()

    # Debug logging for parsed arguments
    print(
        f"DEBUG: Parsed arguments - age_band: {args.age_band}, event_year: {args.event_year}"
    )

    # Version logging
    print("🔧 Using Version 1997 + 13 - Simplified Medical Clean Pipeline")
    print(
        "🚀 SIMPLIFIED - Removed complex DuckDB chaining to fix memory_limit issues (based on pharmacy template)"
    )

    # Setup logging first
    run_id = time.strftime("%Y%m%d-%H%M%S")
    logger, log_buffer = setup_logging(
        "medical_optimized", args.age_band, args.event_year
    )

    # Log pipeline configuration
    logger.info("🚀 Starting Simplified Medical Pipeline")
    logger.info("=" * 80)
    logger.info(f"📊 Processing: {args.age_band}/{args.event_year}")
    logger.info(f"📊 Medical input: {args.medical_input}")
    if args.demographics_lookup:
        logger.info(f"📊 Demographics lookup: {args.demographics_lookup}")
    else:
        logger.info(f"📊 Using pre-imputed data (no demographics lookup needed)")
    logger.info(f"📊 Output root: {args.output_root}")
    logger.info(f"📊 Lookahead years: {args.lookahead_years}")
    logger.info("📊 Threads: Auto-detected, Memory: Auto-detected")
    logger.info(f"📊 Log level: {args.log_level}")
    logger.info("=" * 80)

    # NOTE: Skipping raw silver path validation since this pipeline uses pre-imputed partitioned data
    # The actual imputed partitioned paths are validated within build_optimized_pipeline()
    logger.info("ℹ️  Using pre-imputed partitioned data (validation happens per-partition)")
    logger.info(f"📊 Raw medical input path (for reference): {args.medical_input}")

    # Create simplified DuckDB connection
    conn = create_simple_duckdb_connection(logger, args.tmp_dir)

    try:
        run_started = time.time()
        agg = {
            "tx": "clnmd",
            "run_id": run_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_started)),
            "age_band": args.age_band,
            "event_year": args.event_year,
        }
        # Write directly to gold location
        results = build_optimized_pipeline(
            args.age_band,
            args.event_year,
            args.medical_input,
            args.demographics_lookup,
            args.output_root,
            conn,
            logger,
            log_buffer,
            args.lookahead_years,
            resume=not args.no_resume,
        )
        # Data is now written directly to gold location, no need for copy logic
        logger.info(
            f"✅ Pipeline completed successfully for {args.age_band}/{args.event_year}"
        )

        # Save logs to S3 on success
        try:
            save_logs_to_s3(
                log_buffer,
                "medical_optimized",
                args.age_band,
                args.event_year,
                "apcd_input_data",
                logger=logger,
            )
            logger.info("✅ Logs saved to S3")
        except Exception as e:
            logger.warning(f"⚠️ Could not save logs to S3: {e}")

        print(f"✅ SUCCESS: Processed {args.age_band}/{args.event_year}")
        if results["final_count"] is not None and results["final_patients"] is not None:
            print(
                f"📊 Final count: {results['final_count']:,} records, {results['final_patients']:,} patients"
            )
        else:
            print("📊 Final count: Skipped (data already exists)")
        print(f"📁 Output: {results['output_path']}")

        # Aggregated summary (success)
        try:
            run_finished = time.time()
            agg.update({
                "status": "success",
                "status_code": 0,
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_finished)),
                "duration_sec": round(run_finished - run_started, 3),
                "output_path": results.get("output_path"),
                "final_count": results.get("final_count"),
                "final_patients": results.get("final_patients"),
            })
            bucket, root = args.aggregate_root.replace("s3://", ""), ""
            if "/" in bucket:
                bucket, root = bucket.split("/", 1)
            key = f"{root.rstrip('/')}/clean_medical/run_id={run_id}/summary.json" if root else f"clean_medical/run_id={run_id}/summary.json"
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{bucket}/{key}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save aggregated summary: {e}")

    except MemoryError as e:
        logger.error(f"❌ OUT OF MEMORY: Pipeline killed by memory exhaustion")
        logger.error(f"❌ Error: {e}", exc_info=True)
        try:
            if psutil:
                mem = psutil.virtual_memory()
                logger.error(f"💾 Memory at failure: {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
        except:
            pass
        
        # Save error logs to S3
        try:
            save_logs_immediate(
                log_buffer,
                "medical_optimized",
                args.age_band,
                args.event_year,
                reason="out_of_memory",
            )
            logger.info("✅ Error logs saved to S3")
        except Exception as save_e:
            logger.warning(f"⚠️ Could not save error logs to S3: {save_e}")
        
        print(f"❌ FAILED: OUT OF MEMORY - {e}")
        # Aggregated summary (OOM)
        try:
            run_finished = time.time()
            agg.update({
                "status": "error",
                "status_code": 137,
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_finished)),
                "duration_sec": round(run_finished - run_started, 3),
                "error": str(e),
            })
            bucket, root = args.aggregate_root.replace("s3://", ""), ""
            if "/" in bucket:
                bucket, root = bucket.split("/", 1)
            key = f"{root.rstrip('/')}/clean_medical/run_id={run_id}/summary.json" if root else f"clean_medical/run_id={run_id}/summary.json"
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{bucket}/{key}")
        except Exception as e2:
            logger.warning(f"⚠️ Could not save aggregated summary: {e2}")
        sys.exit(137)  # 128 + 9 (SIGKILL)
    except Exception as e:
        logger.error(f"❌ Optimized pipeline failed: {e}", exc_info=True)
        
        # Log memory at failure
        try:
            if psutil:
                mem = psutil.virtual_memory()
                logger.error(f"💾 Memory at failure: {mem.percent}% used, {mem.available / (1024**3):.1f}GB available")
        except:
            pass

        # Save error logs to S3
        try:
            save_logs_immediate(
                log_buffer,
                "medical_optimized",
                args.age_band,
                args.event_year,
                "apcd_input_data",
                logger=logger,
                reason="error",
            )
            logger.info("✅ Error logs saved to S3")
        except Exception as save_e:
            logger.warning(f"⚠️ Could not save error logs to S3: {save_e}")

        print(f"❌ FAILED: {e}")
        # Aggregated summary (error)
        try:
            run_finished = time.time()
            agg.update({
                "status": "error",
                "status_code": 1,
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_finished)),
                "duration_sec": round(run_finished - run_started, 3),
                "error": str(e),
            })
            bucket, root = args.aggregate_root.replace("s3://", ""), ""
            if "/" in bucket:
                bucket, root = bucket.split("/", 1)
            key = f"{root.rstrip('/')}/clean_medical/run_id={run_id}/summary.json" if root else f"clean_medical/run_id={run_id}/summary.json"
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{bucket}/{key}")
        except Exception as e2:
            logger.warning(f"⚠️ Could not save aggregated summary: {e2}")
        sys.exit(1)

    finally:
        # Clean up DuckDB connection
        try:
            conn.close()
            logger.info("🧹 DuckDB connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Could not close DuckDB connection: {e}")


if __name__ == "__main__":
    main()
