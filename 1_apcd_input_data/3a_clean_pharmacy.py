#!/usr/bin/env python3
"""
Simplified Pharmacy Data Cleaning Pipeline - Version 1997 + 12
Removed complex DuckDB chaining to fix memory_limit issues
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import json
import boto3

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3, save_logs_checkpoint, save_logs_immediate
from helpers_1997_13.data_utils import validate_data_for_blank_strings
from helpers_1997_13.s3_utils import s3_directory_exists_with_files
from helpers_1997_13.s3_utils import s3_delete_prefix

def create_simple_duckdb_connection(logger, tmp_dir=None):
    """Create a simple DuckDB connection without complex chaining"""
    import duckdb
    
    try:
        # Create basic connection
        conn = duckdb.connect(database=':memory:')
        
        # Basic S3 setup only
        conn.sql("INSTALL httpfs; LOAD httpfs;")
        conn.sql("INSTALL aws; LOAD aws;")
        conn.sql("CALL load_aws_credentials();")
        conn.sql("SET s3_region='us-east-1'")
        conn.sql("SET s3_url_style='path'")
        
        # Set temp directory if provided
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
            conn.sql(f"SET temp_directory = '{tmp_dir}'")
        
        # Let DuckDB handle memory and threads automatically - NO manual settings
        logger.info("✅ Simple DuckDB connection created - auto memory/threads")
        return conn
        
    except Exception as e:
        logger.error(f"❌ Failed to create DuckDB connection: {e}")
        raise

def build_optimized_pipeline(age_band: str, event_year: int, pharmacy_input: str, 
                           demographics_lookup: str, output_root: str, conn, logger, log_buffer,
                           resume: bool = True):
    """Build optimized pipeline using pre-imputed demographics"""
    
    # Create unique identifiers for this process to avoid conflicts
    process_id = os.getpid()
    # Note: Keep hyphens in age_band for Hive-style partitioning consistency
    # Only sanitize for run_id logging (not for S3 paths)
    safe_run_id = age_band.replace('-', '_')
    run_id = f"{safe_run_id}_{event_year}_{process_id}" # Unique ID for this run (underscores for DuckDB table names)
    
    logger.info(f"📊 Processing: {age_band}/{event_year}")
    logger.info(f"📊 Process ID: {process_id}")
    logger.info(f"📊 Pharmacy input: {pharmacy_input}")
    # Determine if we're using pre-imputed data or need demographics lookup
    using_imputed_data = demographics_lookup is None
    logger.info(f"📊 Using pre-imputed data: {using_imputed_data}")
    if demographics_lookup:
        logger.info(f"📊 mi_person_key demographics lookup: {demographics_lookup}")
    else:
        logger.info(f"📊 Using pre-imputed partitioned data (no demographics lookup needed)")
    
    # Save initial checkpoint
    save_logs_checkpoint(log_buffer, "pharmacy_optimized", age_band, event_year, "step0_pipeline_started", logger=logger)
    
    # Idempotent resume: if partition already exists with consistent filename, skip work
    if resume:
        try:
            # Check for the specific pharmacy_data.parquet file (consistent filename for Glue/Athena)
            output_file = f"{output_root}/age_band={age_band}/event_year={event_year}/pharmacy_data.parquet"
            if output_file.startswith("s3://"):
                # S3 check
                from urllib.parse import urlparse
                import boto3
                parsed = urlparse(output_file)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                try:
                    s3 = boto3.client("s3")
                    s3.head_object(Bucket=bucket, Key=key)
                    logger.info(f"✅ Output already exists for {age_band}/{event_year}; skipping per resume=true: {output_file}")
                    return {
                        "output_path": output_file,
                        "final_count": None,
                        "final_patients": None
                    }
                except s3.exceptions.ClientError:
                    # File doesn't exist, continue with processing
                    pass
            else:
                # Local check
                if os.path.exists(output_file):
                    logger.info(f"✅ Output already exists for {age_band}/{event_year}; skipping per resume=true: {output_file}")
                    return {
                        "output_path": output_file,
                        "final_count": None,
                        "final_patients": None
                    }
        except Exception as e:
            logger.warning(f"⚠️ Resume check failed (continuing): {e}")
    
    # Step 1: Load pharmacy data (optimized for partitioned data if available)
    logger.info("📊 Step 1: Loading pharmacy data...")
    
    # Check if imputed partitioned data exists first (silver tier - intermediate data)
    # Note: age_band keeps hyphens - DuckDB PARTITION_BY writes values as-is
    partitioned_path = f"s3://pgxdatalake/silver/imputed/pharmacy_partitioned/age_band={age_band}/event_year={event_year}"
    logger.info(f"📊 Checking for imputed partitioned data at: {partitioned_path}")
    
    using_imputed_data = False
    
    # Use imputed partitioned data directly (discovery already confirmed it exists)
    logger.info("✅ Using imputed partitioned data - optimized loading")
    
    # Use imputed partitioned data (much more efficient)
    conn.sql(f"""
        CREATE OR REPLACE TABLE pharmacy_filtered_{run_id} AS
        SELECT *
        FROM read_parquet('{partitioned_path}/**/*.parquet')
    """)
    logger.info("📊 Loaded imputed data from partitioned source")
    using_imputed_data = True
    
    logger.info("📊 Counting initial records...")
    initial_count = conn.sql(f"SELECT COUNT(*) FROM pharmacy_filtered_{run_id}").fetchone()[0]
    logger.info("📊 Counting initial patients...")
    initial_patients = conn.sql(f"SELECT COUNT(DISTINCT mi_person_key) FROM pharmacy_filtered_{run_id}").fetchone()[0]
    
    logger.info(f"📊 Initial pharmacy records (year {event_year}): {initial_count:,}")
    logger.info(f"📊 Initial pharmacy patients: {initial_patients:,}")
    
    # Step 2: Apply age band filtering (if needed)
    logger.info("📊 Step 2: Applying age band filtering...")
    
    # Age band filtering is already done in partitioned data, but let's verify
    age_band_count = conn.sql(f"""
        SELECT COUNT(*) 
        FROM pharmacy_filtered_{run_id} 
        WHERE age_band = '{age_band}'
    """).fetchone()[0]
    
    logger.info(f"📊 Records matching age band {age_band}: {age_band_count:,}")
    
    # Step 3: Apply year filtering
    logger.info("📊 Step 3: Applying year filtering...")
    
    year_count = conn.sql(f"""
        SELECT COUNT(*) 
        FROM pharmacy_filtered_{run_id} 
        WHERE event_year = {event_year}
    """).fetchone()[0]
    
    logger.info(f"📊 Records matching year {event_year}: {year_count:,}")
    
    # Step 3b: Load drug name mappings
    logger.info("📊 Step 3b: Loading drug name mappings...")
    
    # Determine drug_mappings directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_dir = os.path.join(script_dir, "drug_mappings")
    
    try:
        # Create drug_map table from JSON files
        conn.sql(f"""
            CREATE OR REPLACE TABLE drug_map_{run_id} AS
            SELECT 
                LOWER(key.key) AS key,
                LOWER(key.value) AS value
            FROM read_json_auto('{mapping_dir}/*_mappings.json'),
            UNNEST(MAP_ENTRIES(json)) AS kv(key)
        """)
        
        mapping_count = conn.sql(f"SELECT COUNT(*) FROM drug_map_{run_id}").fetchone()[0]
        logger.info(f"📊 Loaded {mapping_count:,} drug name mappings")
    except Exception as e:
        logger.warning(f"⚠️ Could not load drug mappings: {e}")
        logger.warning("⚠️ Proceeding without drug name standardization")
        # Create empty mapping table as fallback
        conn.sql(f"""
            CREATE OR REPLACE TABLE drug_map_{run_id} (
                key VARCHAR,
                value VARCHAR
            )
        """)
    
    # Step 4: Final data preparation with standardized drug names
    logger.info("📊 Step 4: Final data preparation with drug name standardization...")
    
    # Create final table with drug name mapping applied
    conn.sql(f"""
        CREATE OR REPLACE TABLE pharmacy_final_{run_id} AS
        SELECT 
            p.mi_person_key,
            p.age_band,
            p.event_year,
            p.drug_name,
            COALESCE(dm.value, LOWER(p.drug_name)) AS standardized_drug_name,
            p.incurred_date,
            p.gender_source,
            p.age_source
        FROM pharmacy_filtered_{run_id} p
        LEFT JOIN drug_map_{run_id} dm ON LOWER(p.drug_name) = dm.key
        WHERE p.age_band = '{age_band}' 
        AND p.event_year = {event_year}
    """)
    
    final_count = conn.sql(f"SELECT COUNT(*) FROM pharmacy_final_{run_id}").fetchone()[0]
    final_patients = conn.sql(f"SELECT COUNT(DISTINCT mi_person_key) FROM pharmacy_final_{run_id}").fetchone()[0]
    
    logger.info(f"📊 Final pharmacy records: {final_count:,}")
    logger.info(f"📊 Final pharmacy patients: {final_patients:,}")
    
    # Step 5: Write directly to S3 (memory-efficient, no intermediate copy)
    logger.info("📊 Step 5: Writing to S3...")
    
    # Strict behavior: pre-delete all Parquet files in gold partition (S3 only)
    gold_partition = f"{output_root}/age_band={age_band}/event_year={event_year}"
    if gold_partition.startswith("s3://"):
        try:
            from urllib.parse import urlparse
            import boto3
            
            def s3_path_to_bucket_key(s3_path):
                parsed = urlparse(s3_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                return bucket, key
            
            s3 = boto3.client("s3")
            bucket_gold, key_gold_prefix = s3_path_to_bucket_key(gold_partition)
            resp = s3.list_objects_v2(Bucket=bucket_gold, Prefix=key_gold_prefix)
            deleted_count = 0
            for obj in resp.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    s3.delete_object(Bucket=bucket_gold, Key=obj['Key'])
                    deleted_count += 1
            if deleted_count > 0:
                logger.info(f"→ Pre-deleted {deleted_count} Parquet files under {gold_partition}")
        except Exception as del_e:
            logger.warning(f"⚠️ Pre-delete encountered an issue (will still attempt write): {del_e}")
    
    # Write directly to gold partition (memory-efficient: no ORDER BY, no intermediate copy)
    gold_file = f"{gold_partition}/pharmacy_data.parquet"
    logger.info(f"Writing Parquet (direct) → {gold_file}")
    
    conn.sql(f"""
        COPY (
            SELECT * EXCLUDE (age_band, event_year) FROM pharmacy_final_{run_id}
        )
        TO '{gold_file}'
        (FORMAT PARQUET, ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE true)
    """)
    
    # COPY command is synchronous - S3 write is complete at this point
    actual_output = f"{output_root}/age_band={age_band}/event_year={event_year}/pharmacy_data.parquet"
    logger.info(f"✓ S3 write complete (synchronous): {actual_output}")
    logger.info(f"✓ Wrote {final_count:,} records, {final_patients:,} patients")
    
    # Save final checkpoint after successful completion
    save_logs_checkpoint(log_buffer, "pharmacy_optimized", age_band, event_year, "step5_data_written_success", logger=logger)
    
    # Clean up final views to free memory
    conn.sql(f"DROP TABLE IF EXISTS pharmacy_final_{run_id}")
    logger.info("🧹 Cleaned up final views to free memory")
    
    # Clean up temporary views and tables
    logger.info("📊 Cleaning up temporary objects...")
    try:
        conn.sql(f"DROP TABLE IF EXISTS pharmacy_base_{run_id}")
        conn.sql(f"DROP TABLE IF EXISTS pharmacy_with_age_bands_{run_id}")
        conn.sql(f"DROP TABLE IF EXISTS pharmacy_filtered_{run_id}")
        conn.sql(f"DROP TABLE IF EXISTS pharmacy_final_{run_id}")
        conn.sql(f"DROP TABLE IF EXISTS drug_map_{run_id}")
        logger.info("✅ Temporary objects cleaned up")
    except Exception as e:
        logger.warning(f"⚠️ Could not clean up some temporary objects: {e}")
    
    return {
        "output_path": actual_output,
        "final_count": final_count,
        "final_patients": final_patients
    }

def main():
    parser = argparse.ArgumentParser(description="Simplified Pharmacy Data Cleaning")
    parser.add_argument("--age-band", required=True, help="Age band to process")
    parser.add_argument("--event-year", type=int, required=True, help="Event year to process")
    parser.add_argument("--pharmacy-input", required=True, help="S3 path to pre-imputed pharmacy data (mi_person_key imputed)")
    parser.add_argument("--demographics-lookup", help="S3 path to mi_person_key demographics lookup table (optional for optimized mode)")
    parser.add_argument("--output-root", required=True, help="S3 root path for output files")
    parser.add_argument("--tmp-dir", help="Temporary directory for DuckDB")
    parser.add_argument("--no-resume", action="store_true", help="Disable idempotent resume (always re-run)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--aggregate-root", default="s3://pgxdatalake/pgx_pipeline/", help="S3 root for aggregated run summaries (for BI)")
    
    args = parser.parse_args()
    
    # Debug logging for parsed arguments
    print(f"DEBUG: Parsed arguments - age_band: {args.age_band}, event_year: {args.event_year}")
    
    # Version logging
    print("🔧 Using Version 1997 + 12 - Simplified Pharmacy Clean Pipeline")
    print("🚀 SIMPLIFIED - Removed complex DuckDB chaining to fix memory_limit issues")
    
    # Setup logging first
    run_id = time.strftime("%Y%m%d-%H%M%S")
    logger, log_buffer = setup_logging("pharmacy_optimized", args.age_band, args.event_year)
    
    # Log pipeline configuration
    logger.info("🚀 Starting Simplified Pharmacy Pipeline")
    logger.info("=" * 80)
    logger.info(f"📊 Processing: {args.age_band}/{args.event_year}")
    logger.info(f"📊 Pharmacy input: {args.pharmacy_input}")
    if args.demographics_lookup:
        logger.info(f"📊 Demographics lookup: {args.demographics_lookup}")
    else:
        logger.info(f"📊 Using pre-imputed data (no demographics lookup needed)")
    logger.info(f"📊 Output root: {args.output_root}")
    logger.info("📊 Threads: Auto-detected, Memory: Auto-detected")
    logger.info(f"📊 Log level: {args.log_level}")
    logger.info("=" * 80)
    
    # Create simplified DuckDB connection
    conn = create_simple_duckdb_connection(logger, args.tmp_dir)
    
    try:
        run_started = time.time()
        agg = {
            "tx": "clnph",
            "run_id": run_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_started)),
            "age_band": args.age_band,
            "event_year": args.event_year,
        }
        # Write directly to gold location
        results = build_optimized_pipeline(
            args.age_band,
            args.event_year,
            args.pharmacy_input,
            args.demographics_lookup,
            args.output_root,
            conn,
            logger,
            log_buffer
        )
        # Data is now written directly to gold location, no need for copy logic
        logger.info(f"✅ Pipeline completed successfully for {args.age_band}/{args.event_year}")
        
        # Save logs to S3 on success
        try:
            save_logs_to_s3(log_buffer, "pharmacy_optimized", args.age_band, args.event_year, logger=logger)
            logger.info("✅ Logs saved to S3")
        except Exception as e:
            logger.warning(f"⚠️ Could not save logs to S3: {e}")
        
        print(f"✅ SUCCESS: Processed {args.age_band}/{args.event_year}")
        if results['final_count'] is not None and results['final_patients'] is not None:
            print(f"📊 Final count: {results['final_count']:,} records, {results['final_patients']:,} patients")
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
            key = f"{root.rstrip('/')}/clean_pharmacy/run_id={run_id}/summary.json" if root else f"clean_pharmacy/run_id={run_id}/summary.json"
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{bucket}/{key}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save aggregated summary: {e}")
        
    except Exception as e:
        logger.error(f"❌ Optimized pipeline failed: {e}")
        
        # Save error logs to S3
        try:
            save_logs_immediate(log_buffer, "pharmacy_optimized", args.age_band, args.event_year, "error", logger=logger)
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
            key = f"{root.rstrip('/')}/clean_pharmacy/run_id={run_id}/summary.json" if root else f"clean_pharmacy/run_id={run_id}/summary.json"
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
