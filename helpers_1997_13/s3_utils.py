import pandas as pd
import json
import io
import os
import sys
import time
import logging
import re
import threading
import boto3
import duckdb
import certifi
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
import pyarrow.parquet as pq


# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.common_imports import s3_client
from helpers_1997_13.data_utils import convert_json_serializable
from helpers_1997_13.constants import S3_BUCKET
from botocore.config import Config


# ---- S3 Retry Helper ----

def validate_s3_input_paths(pharmacy_input: str, medical_input: str, logger):
    """Validate that S3 input paths exist and contain expected data.
    
    Args:
        pharmacy_input: Pharmacy dataset path (e.g., s3://bucket/silver/pharmacy/**/*.parquet)
        medical_input: Medical dataset path (e.g., s3://bucket/silver/medical/**/*.parquet)
        logger: Logger instance
    """
    try:
        # Use the enhanced validation function that properly handles wildcards
        validation_results = validate_input_dataset_paths(pharmacy_input, medical_input, logger)
        
        # Log additional details if validation failed
        if not validation_results["pharmacy"]:
            logger.warning(f"Pharmacy validation details: {validation_results['details'].get('pharmacy', 'Unknown')}")
        if not validation_results["medical"]:
            logger.warning(f"Medical validation details: {validation_results['details'].get('medical', 'Unknown')}")
            
        return validation_results
            
    except Exception as e:
        logger.warning(f"Could not validate S3 input paths: {e}")
        # Fallback to simple existence check
        logger.info("→ Falling back to simple path validation...")
        
        results = {"pharmacy": False, "medical": False, "details": {}}
        
        if not s3_exists(pharmacy_input.replace("/**/*.parquet", "")):
            logger.warning(f"Pharmacy input path may not exist: {pharmacy_input}")
            results["details"]["pharmacy"] = "Path validation failed"
        else:
            logger.info(f"✓ Pharmacy input path validated: {pharmacy_input}")
            results["pharmacy"] = True
            results["details"]["pharmacy"] = "Path exists"
            
        if not s3_exists(medical_input.replace("/**/*.parquet", "")):
            logger.warning(f"Medical input path may not exist: {medical_input}")
            results["details"]["medical"] = "Path validation failed"
        else:
            logger.info(f"✓ Medical input path validated: {medical_input}")
            results["medical"] = True
            results["details"]["medical"] = "Path exists"
            
        return results


def validate_s3_input_paths_for_project(pharmacy_input: str, medical_input: str, logger):
    """Convenience wrapper for validate_s3_input_paths that matches the original interface."""
    return validate_s3_input_paths(pharmacy_input, medical_input, logger)


def validate_s3_output_path(output_path: str, age_band: str, event_year: int, logger):
    """Validate that the S3 output path is correctly configured for the gold directory."""
    try:
        # Check if path points to gold directory
        if not output_path.startswith(f"s3://{S3_BUCKET}/gold/"):
            logger.warning(f"Output path may not be in gold directory: {output_path}")
            logger.warning(f"Expected format: s3://{S3_BUCKET}/gold/medical_clean/age_band=<band>/event_year=<year>/cleaned_medical.parquet")
        
        # Parse S3 path components
        try:
            bucket, key = _parse_s3_path_components(output_path)
            logger.info(f"✓ S3 output validated - Bucket: {bucket}, Key: {key}")
            
            # Ensure it's the medical_clean directory with correct partitioning
            if "medical_clean" not in key:
                logger.warning(f"Output path may not be medical_clean directory: {key}")
            if f"age_band={age_band}" not in key:
                logger.warning(f"Output path missing age_band partition: {key}")  
            if f"event_year={event_year}" not in key:
                logger.warning(f"Output path missing event_year partition: {key}")
                
        except Exception as e:
            logger.error(f"Could not parse S3 path {output_path}: {e}")
            
        # Log the partition structure
        logger.info(f"Writing medical data to gold directory:")
        logger.info(f"  Full path: {output_path}")
        logger.info(f"  Partition: age_band={age_band}, event_year={event_year}")
        
    except Exception as e:
        logger.warning(f"Could not validate S3 output path: {e}")


def convert_raw_to_imputed_path(raw_path: str, data_type: str, age_band: str = None, event_year: int = None) -> str:
    """Convert raw silver path to imputed partitioned path.
    
    Centralizes the path conversion logic used across the pipeline to transform raw silver paths
    into their corresponding imputed partitioned paths.
    
    Args:
        raw_path: Raw silver path (e.g., 's3://pgxdatalake/silver/medical/*.parquet')
        data_type: 'medical' or 'pharmacy'
        age_band: Optional age band for full partitioned path (e.g., '65-74')
        event_year: Optional event year for full partitioned path (e.g., 2019)
        
    Returns:
        str: Imputed path, optionally with partitions
        
    Examples:
        >>> convert_raw_to_imputed_path('s3://pgxdatalake/silver/medical/*.parquet', 'medical')
        's3://pgxdatalake/silver/imputed/medical_partitioned'
        
        >>> convert_raw_to_imputed_path('s3://pgxdatalake/silver/medical/*.parquet', 'medical', '65-74', 2019)
        's3://pgxdatalake/silver/imputed/medical_partitioned/age_band=65-74/event_year=2019'
    """
    if data_type not in ['medical', 'pharmacy']:
        raise ValueError(f"data_type must be 'medical' or 'pharmacy', got: {data_type}")
    
    # Convert raw path to imputed base path
    if f'silver/{data_type}' in raw_path:
        imputed_base = raw_path.replace(f'silver/{data_type}', f'silver/imputed/{data_type}_partitioned')
        # Remove glob patterns
        imputed_base = imputed_base.replace('/*.parquet', '').replace('/**/*.parquet', '')
    else:
        # Fallback if path structure is different
        imputed_base = f"s3://{S3_BUCKET}/silver/imputed/{data_type}_partitioned"
    
    # Add partition keys if provided
    if age_band is not None and event_year is not None:
        return f"{imputed_base}/age_band={age_band}/event_year={event_year}"
    elif age_band is not None:
        return f"{imputed_base}/age_band={age_band}"
    elif event_year is not None:
        return f"{imputed_base}/event_year={event_year}"
    else:
        return imputed_base


# ---- Zone helpers: bronze, brass, silver, gold ----
def zone_root(zone: str) -> str:
    """Return the s3 uri root for a named zone under the project bucket.

    zone: one of 'bronze','brass','silver','gold'
    """
    zone = zone.strip().lower()
    if zone not in ("bronze", "brass", "silver", "gold"):
        raise ValueError(f"unknown zone: {zone}")
    return f"s3://{S3_BUCKET}/{zone}/"


def build_zone_path(zone: str, data_type: str = None, age_band: str = None, event_year: Union[int, str] = None, suffix: str = "") -> str:
    """Construct a canonical path inside a zone.

    Examples:
        build_zone_path('brass', 'medical', '65-74', 2019)
        -> s3://pgxdatalake/brass/medical/age_band=65-74/event_year=2019

    If data_type is provided, common subpaths are used (e.g., imputed for silver).
    """
    root = zone_root(zone)
    parts = [root.rstrip('/')]

    if zone == 'silver' and data_type:
        # silver imputed layout
        parts.append(f"imputed/{data_type}")
    elif data_type:
        parts.append(data_type)

    if age_band is not None:
        parts.append(f"age_band={age_band}")
    if event_year is not None:
        parts.append(f"event_year={event_year}")
    if suffix:
        parts.append(suffix.strip('/'))

    return '/'.join(parts) + ('/' if not suffix or suffix.endswith('/') else '')


def backup_s3_prefix(src_prefix: str, backup_root: str, logger: Optional[logging.Logger] = None) -> str:
    """Copy all objects under src_prefix into backup_root/<timestamp>/ preserving keys.

    Returns the backup prefix used.
    """
    import boto3 as _boto3
    s3 = _boto3.client('s3')
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    if not backup_root.endswith('/'):
        backup_root = backup_root + '/'
    backup_prefix = f"{backup_root}backup_{ts}/"

    # parse src
    try:
        src_bucket, src_key = _parse_s3_path_components(src_prefix)
    except Exception:
        # assume prefix like s3://bucket/path/
        src_bucket, src_key = parse_s3_path(src_prefix)

    paginator = s3.get_paginator('list_objects_v2')
    copied = 0
    for page in paginator.paginate(Bucket=src_bucket, Prefix=src_key):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel = key[len(src_key):].lstrip('/')
            dest_key = (backup_prefix + rel).lstrip('/')
            dest_bucket = backup_root.replace('s3://', '').split('/', 1)[0]
            copy_source = {'Bucket': src_bucket, 'Key': key}
            try:
                s3.copy_object(Bucket=dest_bucket, CopySource=copy_source, Key=dest_key)
                copied += 1
            except Exception as e:
                if logger:
                    logger.error(f"Failed to copy {key} to backup {dest_key}: {e}")
    if logger:
        logger.info(f"Backed up {copied} objects from {src_prefix} to {backup_prefix}")
    return backup_prefix

def sanitize_for_s3_key(value):
    """Sanitize a value for use in S3 key names.

    Replaces problematic characters with underscores:
    - Spaces, slashes, backslashes
    - Special characters that could cause issues
    - Multiple consecutive underscores are collapsed to single underscore
    """
    if value is None:
        return "unknown"

    # Convert to string and replace problematic characters
    sanitized = str(value).strip()
    sanitized = re.sub(r'[\\/ \t\n\r]', '_', sanitized)  # Replace spaces, slashes, tabs, newlines
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)  # Replace any other non-alphanumeric chars except -_.
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    sanitized = sanitized.strip('_')  # Remove leading/trailing underscores

    # Ensure we have a valid value
    if not sanitized:
        return "unknown"

    return sanitized


def validate_s3_source_paths(age_band, event_year, logger):
    """Validate that S3 source paths exist and contain expected data."""
    logger.info(f"→ Validating S3 source paths for {age_band}/{event_year}...")
    
    try:
        # Check medical/pharmacy sources in SILVER zone
        medical_path = f"s3://pgxdatalake/silver/medical/age_band={age_band}/event_year={event_year}/"
        pharmacy_path = f"s3://pgxdatalake/silver/pharmacy/age_band={age_band}/event_year={event_year}/"
        
        logger.info(f"→ Checking medical source: {medical_path}")
        logger.info(f"→ Checking pharmacy source: {pharmacy_path}")
        
        # List objects in these paths
        try:
            medical_objects = s3_client.list_objects_v2(
                Bucket="pgxdatalake",
                Prefix=f"silver/medical/age_band={age_band}/event_year={event_year}/"
            )
            
            pharmacy_objects = s3_client.list_objects_v2(
                Bucket="pgxdatalake",
                Prefix=f"silver/pharmacy/age_band={age_band}/event_year={event_year}/"
            )
            
            medical_count = len(medical_objects.get('Contents', []))
            pharmacy_count = len(pharmacy_objects.get('Contents', []))
            
            logger.info(f"→ Medical source files: {medical_count}")
            logger.info(f"→ Pharmacy source files: {pharmacy_count}")
            
            if medical_count == 0:
                logger.error(f"→ ERROR: No medical data files found at {medical_path}")
                
                # Check what age_bands are actually available
                available_medical = s3_client.list_objects_v2(
                    Bucket="pgxdatalake",
                    Prefix="silver/medical/",
                    Delimiter="/"
                )
                
                logger.error("→ Available medical age_bands:")
                for prefix in available_medical.get('CommonPrefixes', []):
                    logger.error(f"→   {prefix['Prefix']}")
            
            if pharmacy_count == 0:
                logger.error(f"→ ERROR: No pharmacy data files found at {pharmacy_path}")
                
                # Check what age_bands are actually available
                available_pharmacy = s3_client.list_objects_v2(
                    Bucket="pgxdatalake",
                    Prefix="silver/pharmacy/",
                    Delimiter="/"
                )
                
                logger.error("→ Available pharmacy age_bands:")
                for prefix in available_pharmacy.get('CommonPrefixes', []):
                    logger.error(f"→   {prefix['Prefix']}")
            
            if medical_count == 0 and pharmacy_count == 0:
                logger.error(f"→ CRITICAL: No source data found for {age_band}/{event_year}")
                return False
            
            logger.info(f"→ ✓ S3 source validation completed")
            return True
        except Exception as s3_e:
            logger.error(f"→ ERROR checking S3 paths: {str(s3_e)}")
            return False
            
    except Exception as e:
        logger.error(f"→ ERROR during S3 validation: {str(e)}")
        return False


def _sanitize_age_band(age_band: str) -> str:
    """Helper function to sanitize age_band for S3 paths."""
    return age_band.replace(" ", "_").replace("/", "_")

def _parse_s3_path_components(s3_path: str) -> tuple[str, str]:
    """Helper function to parse S3 path into bucket and key."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    bucket_key = s3_path.replace("s3://", "")
    parts = bucket_key.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 path format: {s3_path}")
    return parts[0], parts[1]

# ===== Path Parsing and Validation Functions =====
def parse_path_params(s3_path: str) -> dict:

    try:
        # Strip s3:// and split path
        path_parts = s3_path.replace("s3://", "").split("/")

        parsed = {}
        for part in path_parts:
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key] = value

        # Optional: cast event_year to int if needed
        if "event_year" in parsed:
            parsed["event_year"] = str(parsed["event_year"])  # or int() if expected

        return parsed
    except Exception as e:
        print(f"✗ Error parsing S3 path: {e}")
        return {}


def extract_column_names(s3_path):
    try:
        table = pq.read_table(s3_path)
        return table.schema.names
    except Exception as e:
        logging.error(f"Failed to extract columns from {s3_path}: {e}")
        return []

def parse_s3_path(s3_uri):
    """Helper to split s3://bucket/key path"""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_uri}")
    parts = s3_uri[5:].split("/", 1)
    return parts[0], parts[1]


# ---- Light-weight discovery helpers (migrated from scripts/validate_silver_inputs.py)
def s3_prefix_exists(bucket: str, prefix: str, min_keys: int = 1, s3: Optional[Any] = None) -> bool:
    """Return True if prefix contains at least `min_keys` objects.

    This is a minimal utility used by discovery code. It prefers the shared
    `s3_client` if none is provided.
    """
    _s3 = s3 or s3_client or boto3.client("s3", config=Config(retries={"max_attempts": 3}))
    paginator = _s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    count = 0
    try:
        for page in paginator.paginate(**kwargs):
            if "Contents" in page:
                count += len(page["Contents"])
                if count >= min_keys:
                    return True
    except Exception:
        # Mirror previous script behavior: treat errors as prefix missing
        return False
    return False


def select_silver_inputs(bucket: str, base_prefix: str, job: str) -> Dict[str, Optional[str]]:
    """Select preferred silver input S3 URIs for `medical` / `pharmacy` jobs.

    Returns a dict with keys `raw_medical` and `raw_pharmacy` mapping to
    s3://.../*.parquet strings or None if not found.
    """
    result = {"raw_medical": None, "raw_pharmacy": None}

    if job in ("medical", "both"):
        pref = f"{base_prefix}/imputed/medical_partitioned/"
        if s3_prefix_exists(bucket, pref):
            result["raw_medical"] = f"s3://{bucket}/{pref}*.parquet"
        else:
            fallback = f"{base_prefix}/medical/"
            if s3_prefix_exists(bucket, fallback):
                result["raw_medical"] = f"s3://{bucket}/{fallback}*.parquet"

    if job in ("pharmacy", "both"):
        pref = f"{base_prefix}/imputed/pharmacy_partitioned/"
        if s3_prefix_exists(bucket, pref):
            result["raw_pharmacy"] = f"s3://{bucket}/{pref}*.parquet"
        else:
            fallback = f"{base_prefix}/pharmacy/"
            if s3_prefix_exists(bucket, fallback):
                result["raw_pharmacy"] = f"s3://{bucket}/{fallback}*.parquet"

    return result


def check_previous_orchestrator_logs(job: str, repo_bucket: str = "pgx-repository") -> bool:
    """Return True if previous orchestrator logs exist for `job` under
    `pgx-repository/build_logs/apcd_input_data/orchestrator/{job}/`.
    """
    prefix = f"build_logs/apcd_input_data/orchestrator/{job}/"
    return s3_prefix_exists(repo_bucket, prefix)



def validate_output_paths(paths, logger):
    """Validate that none of the expected output files already exist in S3."""
    logger.info("Validating output paths")
    try:
        all_paths_valid = True
        for key, path in paths.items():
            try:
                bucket, s3_key = parse_s3_path(path)
                try:
                    s3_client.head_object(Bucket=bucket, Key=s3_key)
                    logger.warning(f"Output file already exists: {path}")
                    return False
                except s3_client.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        logger.info(f"  → {key}: {path} (does not exist - will create)")
                    else:
                        raise
            except Exception as e:
                logger.warning(f"⚠ Warning: Error checking path {path}: {str(e)}")
                all_paths_valid = False
                continue

        if not all_paths_valid:
            logger.warning("⚠ Warning: Some paths could not be validated, but will attempt to continue")

        logger.info("All output paths are valid")
        return True
    except Exception as e:
        logger.error(f"Error during output path validation: {str(e)}")
        logger.error(f"Paths being validated: {paths}")
        return False


# ===== File Operations =====
def s3_exists(s3_path: str, bucket_name: Optional[str] = None, region: str = "us-east-1") -> bool:
    try:
        parsed_bucket, key = _parse_s3_path_components(s3_path)
        bucket = bucket_name if bucket_name else parsed_bucket

        s3_client.head_object(Bucket=bucket, Key=key)
        return True

    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["404", "403"]:
            return False
        raise


def s3_delete_object_if_exists(s3_path: str, logger: Optional[logging.Logger] = None, wait: bool = True, max_wait_seconds: int = 60) -> bool:
    """Delete a single S3 object if it exists. Optionally wait until deletion is observable.

    Returns True if the object existed and was deleted (or didn't exist), False on non-fatal issues.
    Raises on unexpected AWS errors.
    """
    try:
        bucket, key = _parse_s3_path_components(s3_path)
        # Check existence
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            exists = True
        except s3_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ["404", "403"]:
                exists = False
            else:
                raise

        if not exists:
            if logger:
                logger.info(f"✓ No existing object to delete: {s3_path}")
            return True

        # Delete
        s3_client.delete_object(Bucket=bucket, Key=key)
        if logger:
            logger.info(f"→ Deleted existing object: {s3_path}")

        if not wait:
            return True

        # Wait for deletion consistency
        import time as _time
        deadline = _time.time() + max_wait_seconds
        while _time.time() < deadline:
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                # still exists
                _time.sleep(1)
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ["404", "403"]:
                    if logger:
                        logger.info(f"✓ Deletion confirmed: {s3_path}")
                    return True
                else:
                    raise
        if logger:
            logger.warning(f"⚠ Timed out waiting for deletion to propagate: {s3_path}")
        return False
    except Exception as e:
        if logger:
            logger.error(f"✗ Error deleting object {s3_path}: {str(e)}")
        raise


def s3_delete_prefix(prefix_path: str, logger: Optional[logging.Logger] = None, wait: bool = True, max_wait_seconds: int = 120) -> int:
    """Delete all objects under an S3 prefix (directory-like). Returns number of deleted objects.

    This blocks until the prefix lists empty if wait=True.
    """
    try:
        bucket, key_prefix = _parse_s3_path_components(prefix_path)
        if key_prefix and not key_prefix.endswith('/'):
            key_prefix = key_prefix + '/'

        deleted_total = 0
        continuation_token = None
        while True:
            list_kwargs = {"Bucket": bucket, "Prefix": key_prefix, "MaxKeys": 1000}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token
            response = s3_client.list_objects_v2(**list_kwargs)
            contents = response.get('Contents', [])
            if not contents:
                break

            # Batch delete up to 1000
            to_delete = {"Objects": [{"Key": obj["Key"]} for obj in contents]}
            del_resp = s3_client.delete_objects(Bucket=bucket, Delete=to_delete)
            deleted = len(del_resp.get('Deleted', []))
            deleted_total += deleted
            if logger:
                logger.info(f"→ Deleted {deleted} objects under s3://{bucket}/{key_prefix} (running total {deleted_total})")

            # Pagination
            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                continuation_token = None
                # loop again to confirm empty

        if wait:
            # Confirm emptiness with retry to handle eventual consistency
            import time as _time
            deadline = _time.time() + max_wait_seconds
            while _time.time() < deadline:
                verify = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)
                if len(verify.get('Contents', [])) == 0:
                    if logger:
                        logger.info(f"✓ Prefix is now empty: s3://{bucket}/{key_prefix}")
                    break
                _time.sleep(1)

        return deleted_total
    except Exception as e:
        if logger:
            logger.error(f"✗ Error deleting prefix {prefix_path}: {str(e)}")
        raise

def s3_directory_exists_with_files(s3_path: str, file_pattern: str = "*.parquet", bucket_name: Optional[str] = None) -> bool:
    """Check if an S3 directory exists and contains files matching the pattern.
    
    This is specifically designed to handle wildcard paths like:
    s3://bucket/silver/pharmacy/**/*.parquet
    
    Args:
        s3_path: S3 path without wildcards (e.g., s3://bucket/silver/pharmacy/)
        file_pattern: File pattern to check for (default: *.parquet)
        bucket_name: Optional bucket name override
        
    Returns:
        True if directory exists and contains matching files, False otherwise
    """
    try:
        parsed_bucket, key_prefix = _parse_s3_path_components(s3_path)
        bucket = bucket_name if bucket_name else parsed_bucket
        
        # Ensure key_prefix ends with / for directory listing
        if not key_prefix.endswith('/'):
            key_prefix += '/'
            
        # List objects with the prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=key_prefix,
            MaxKeys=10  # We just need to know if files exist
        )
        
        # Check if any files match the pattern
        if 'Contents' in response:
            if file_pattern == "*.parquet":
                # Check for parquet files
                parquet_files = [obj for obj in response['Contents'] 
                               if obj['Key'].endswith('.parquet')]
                return len(parquet_files) > 0
            else:
                # For other patterns, just check if any files exist
                return len(response['Contents']) > 0
        
        return False
        
    except Exception as e:
        # If we can't check, assume it doesn't exist
        return False


def validate_input_dataset_paths(pharmacy_input: str, medical_input: str, logger) -> dict:
    """Validate pharmacy and medical input dataset paths with proper wildcard handling.
    
    Specifically designed to handle paths like:
    - s3://pgxdatalake/silver/pharmacy/**/*.parquet
    - s3://pgxdatalake/silver/medical/**/*.parquet
    
    Args:
        pharmacy_input: Pharmacy dataset path (with or without wildcards)
        medical_input: Medical dataset path (with or without wildcards)
        logger: Logger instance
        
    Returns:
        Dict with validation results: {"pharmacy": bool, "medical": bool, "details": dict}
    """
    results = {"pharmacy": False, "medical": False, "details": {}}
    
    try:
        logger.info("→ Validating input dataset paths...")
        
        # Validate pharmacy input
        pharmacy_base = pharmacy_input.replace("/**/*.parquet", "").replace("/*", "")
        if s3_directory_exists_with_files(pharmacy_base, "*.parquet"):
            results["pharmacy"] = True
            logger.info(f"✓ Pharmacy input path validated: {pharmacy_input}")
            results["details"]["pharmacy"] = "Directory exists with parquet files"
        else:
            # Try falling back to imputed/partitioned layout (global_imputation writes here)
            try:
                imputed_pharmacy = convert_raw_to_imputed_path(pharmacy_input, 'pharmacy')
                if s3_directory_exists_with_files(imputed_pharmacy, "*.parquet"):
                    results["pharmacy"] = True
                    logger.info(f"✓ Pharmacy imputed partition path validated: {imputed_pharmacy}")
                    results["details"]["pharmacy"] = f"Found imputed partitioned data at {imputed_pharmacy}"
                else:
                    logger.warning(f"Pharmacy input path may not exist or contain parquet files: {pharmacy_input}")
                    results["details"]["pharmacy"] = "Directory not found or no parquet files"
            except Exception:
                logger.warning(f"Pharmacy input path may not exist or contain parquet files: {pharmacy_input}")
                results["details"]["pharmacy"] = "Directory not found or no parquet files"
            
        # Validate medical input  
        medical_base = medical_input.replace("/**/*.parquet", "").replace("/*", "")
        if s3_directory_exists_with_files(medical_base, "*.parquet"):
            results["medical"] = True
            logger.info(f"✓ Medical input path validated: {medical_input}")
            results["details"]["medical"] = "Directory exists with parquet files"
        else:
            # Try falling back to imputed/partitioned layout (global_imputation writes here)
            try:
                imputed_medical = convert_raw_to_imputed_path(medical_input, 'medical')
                if s3_directory_exists_with_files(imputed_medical, "*.parquet"):
                    results["medical"] = True
                    logger.info(f"✓ Medical imputed partition path validated: {imputed_medical}")
                    results["details"]["medical"] = f"Found imputed partitioned data at {imputed_medical}"
                else:
                    logger.warning(f"Medical input path may not exist or contain parquet files: {medical_input}")
                    results["details"]["medical"] = "Directory not found or no parquet files"
            except Exception:
                logger.warning(f"Medical input path may not exist or contain parquet files: {medical_input}")
                results["details"]["medical"] = "Directory not found or no parquet files"
            
        # Summary
        if results["pharmacy"] and results["medical"]:
            logger.info("✓ All input dataset paths validated successfully")
        elif results["pharmacy"] or results["medical"]:
            logger.warning("⚠ Some input dataset paths could not be validated")
        else:
            logger.warning("⚠ No input dataset paths could be validated")
            
        return results
        
    except Exception as e:
        logger.warning(f"Could not validate input dataset paths: {e}")
        results["details"]["error"] = str(e)
        return results


def build_gold_globs(dataset: str, age_bands: Optional[List[str]] = None, years: Optional[List[str]] = None, all_partitions: bool = False) -> str:
    """Build a comma-separated list of S3 globs that target gold partitions first.

    This mirrors the helper previously located in the QA script but centralizes S3
    partition discovery so it can be reused across the codebase.

    Args:
        dataset: 'pharmacy' or 'medical'
        age_bands: optional list of age band strings (e.g., ['65-74'])
        years: optional list of years (as strings or ints)
        all_partitions: if True, return recursive wildcard for all partitions

    Returns:
        A single string suitable for passing into read_parquet('<glob1>,<glob2>')
        If no specific partitions are found, returns the original recursive wildcard.
    """
    # Build base gold directory for pharmacy or medical
    if dataset == 'pharmacy':
        base = f"s3://{S3_BUCKET}/gold/pharmacy"
    else:
        base = f"s3://{S3_BUCKET}/gold/medical"

    # Use recursive wildcard when no hints or user requested all partitions
    if all_partitions or (not age_bands and not years):
        return f"{base}/**/*.parquet"

    globs: List[str] = []
    if age_bands and years:
        for ab in age_bands:
            for y in years:
                globs.append(f"{base}/age_band={ab}/event_year={y}/*.parquet")
    elif age_bands:
        for ab in age_bands:
            globs.append(f"{base}/age_band={ab}/*.parquet")
    elif years:
        for y in years:
            globs.append(f"{base}/event_year={y}/*.parquet")

    # Verify at least one of the globs points to existing files; if not, fall back
    valid_globs: List[str] = []
    for g in globs:
        dir_prefix = g.rsplit('/', 1)[0]
        try:
            if s3_directory_exists_with_files(dir_prefix, file_pattern='*.parquet'):
                valid_globs.append(g)
        except Exception:
            continue

    if not valid_globs:
        logging.warning(f"No partitioned files found for requested partitions; falling back to full scan: {base}/**/*.parquet")
        return f"{base}/**/*.parquet"

    return ",".join(valid_globs)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_to_s3_parquet(df: pd.DataFrame, s3_path: str, logger: logging.Logger,
                    partition_cols: Optional[List[str]] = None) -> None:
    """Save DataFrame to S3 as parquet file without duplicate partition columns.

    Args:
        df: DataFrame to save
        s3_path: Full S3 path (s3://bucket/key)
        logger: Logger instance
        partition_cols: Optional list of partition columns to drop from DataFrame
    """

    import uuid

    temp_path = None
    try:
        if partition_cols:
            df = df.drop(columns=[col for col in partition_cols if col in df.columns])

        bucket, key = _parse_s3_path_components(s3_path)

        # Create unique temp file path to avoid race conditions
        temp_path = f"/tmp/temp_{uuid.uuid4().hex[:8]}.parquet"
        df.to_parquet(temp_path, index=False)

        # Upload to S3
        with open(temp_path, "rb") as f:
            s3_client.upload_fileobj(f, bucket, key)

        logger.info(f"Saved parquet file to {s3_path}")
    except Exception as e:
        logger.error(f"Error saving parquet file to {s3_path}: {str(e)}")
        raise
    finally:
        # Always clean up temp file, even if upload fails
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_e:
                logger.warning(f"Could not remove temp file {temp_path}: {str(cleanup_e)}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_to_s3_json(
    data: Union[str, dict, list],
    s3_path: str,
    logger: Optional[logging.Logger] = None,
    partition_cols: Optional[List[str]] = None
) -> None:
    """
    Save JSON data (dict, list, or JSON string) to S3, optionally excluding partition columns.

    Args:
        data: JSON string, dict, or list to save.
        s3_path: Full S3 path (s3://bucket/key)
        logger: Logger instance
        partition_cols: Optional list of partition columns to drop from dict(s)
    """
    try:
        if not isinstance(data, str):
            if isinstance(data, dict):
                if partition_cols:
                    data = {k: v for k, v in data.items() if k not in partition_cols}
                data = json.dumps(convert_json_serializable(data), indent=2)
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if partition_cols:
                    data = [
                        {k: v for k, v in item.items() if k not in partition_cols}
                        for item in data
                    ]
                data = json.dumps(convert_json_serializable(data), indent=2)
            else:
                raise ValueError("Data must be a JSON string, dict, or list of dicts")

        bucket, key = _parse_s3_path_components(s3_path)

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data.encode("utf-8"),
            ContentType="application/json"
        )

        if logger:
            logger.info(f"✓ Saved JSON file to {s3_path}")
    except Exception as e:
        msg = f"✗ Error saving JSON file to {s3_path}: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_from_s3_json(
    s3_path: str,
    logger: Optional[logging.Logger] = None
) -> Union[dict, list]:
    """
    Load JSON data (dict or list) from S3.

    Args:
        s3_path: Full S3 path (s3://bucket/key)
        logger: Logger instance

    Returns:
        Parsed JSON data as dict or list
    """
    try:
        bucket, key = _parse_s3_path_components(s3_path)
        
        if logger:
            logger.info(f"Loading JSON from S3: {s3_path}")
        
        # Download the object
        response = s3_client.get_object(Bucket=bucket, Key=key)
        json_content = response['Body'].read().decode('utf-8')
        
        # Parse JSON
        data = json.loads(json_content)
        
        if logger:
            logger.info(f"✓ Successfully loaded JSON from {s3_path}")
            
        return data
        
    except Exception as e:
        msg = f"✗ Error loading JSON from {s3_path}: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_to_s3_text(
    text: str,
    s3_path: str,
    logger: Optional[logging.Logger] = None,
    partition_cols: Optional[List[str]] = None
) -> None:
    """Save text data to S3, optionally excluding lines containing partition column keys."""
    try:
        # Remove partition columns from text (if applicable)
        if partition_cols:
            # Apply line filtering: remove any line that contains a partition column key
            lines = text.splitlines()
            filtered_lines = [
                line for line in lines
                if not any(col in line for col in partition_cols)
            ]
            text = "\n".join(filtered_lines)

        bucket, key = _parse_s3_path_components(s3_path)

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode('utf-8'),
            ContentType='text/plain'
        )

        if logger:
            logger.info(f"✓ Saved text file to {s3_path}")
        else:
            print(f"✓ Saved text file to {s3_path}")
    except Exception as e:
        msg = f"✗ Error saving text file to {s3_path}: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise


def save_to_s3_html(html_string, s3_path, logger=None):
    """
    Save an HTML string as an .html file to the specified S3 path.

    Parameters:
    - html_string (str): The full HTML content to be saved.
    - s3_path (str): S3 URI, e.g., "s3://my-bucket/path/to/file.html".
    - logger (Logger, optional): Optional logger for logging.
    """
    try:
        # Log input parameters
        if logger:
            logger.info("save_to_s3_html called with:")
            logger.info(f"  S3 path: {s3_path}")
            logger.info(f"  HTML content length: {len(html_string)} bytes")
            logger.info(f"  HTML content preview (first 500 chars): {html_string[:500]}")
        else:
            print("save_to_s3_html called with:")
            print(f"  S3 path: {s3_path}")
            print(f"  HTML content length: {len(html_string)} bytes")
            print(f"  HTML content preview (first 500 chars): {html_string[:500]}")

        bucket, key = _parse_s3_path_components(s3_path)

        # Log the attempt to save
        if logger:
            logger.info(f"Attempting to save HTML to s3://{bucket}/{key}")
            logger.info(f"HTML content length: {len(html_string)} bytes")
        else:
            print(f"Attempting to save HTML to s3://{bucket}/{key}")
            print(f"HTML content length: {len(html_string)} bytes")

        try:
            buffer = io.BytesIO(html_string.encode("utf-8"))
            s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType="text/html")
            if logger:
                logger.info("Successfully called s3_client.put_object")
        except Exception as put_error:
            if logger:
                logger.error(f"Failed to put object in S3: {str(put_error)}")
                logger.error(f"Error type: {type(put_error).__name__}")
            else:
                print(f"Failed to put object in S3: {str(put_error)}")
            raise

        # Verify the file was saved
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            if logger:
                logger.info(f"✓ Successfully saved and verified HTML to s3://{bucket}/{key}")
            else:
                print(f"✓ Successfully saved and verified HTML to s3://{bucket}/{key}")
        except Exception as verify_error:
            if logger:
                logger.error(f"⚠ File saved but verification failed: {str(verify_error)}")
                logger.error(f"Error type: {type(verify_error).__name__}")
            else:
                print(f"⚠ File saved but verification failed: {str(verify_error)}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to save HTML to {s3_path}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
        else:
            print(f"Failed to save HTML to {s3_path}: {str(e)}")
        raise


def save_pipeline_metrics(metrics, age_band, event_year, cohort_name, conn, logger):
    """Save metrics to S3 in JSON format."""
    logger.info(f"Saving metrics for {cohort_name} (age_band={age_band}, event_year={event_year})")

    # Generate output paths
    paths = get_output_paths(cohort_name, age_band, event_year)
    metrics_path = paths["pipeline_metrics_json"]

    try:
        json_metrics = json.dumps(metrics, default=convert_json_serializable)
        logger.debug("Successfully serialized metrics to JSON")
    except TypeError:
        logger.error("JSON serialization error. Dumping keys for inspection:")
        for key, val in metrics.items():
            try:
                json.dumps(val, default=convert_json_serializable)
            except Exception as inner_e:
                logger.error(f"Key: {key} → Error: {inner_e} → Value: {val}")
        raise

    try:
        # Write using parameterized SQL
        conn.sql("CREATE OR REPLACE TEMP TABLE metrics_table AS SELECT ? AS metrics", params=[json_metrics])
        conn.sql(f"COPY metrics_table TO '{metrics_path}' (FORMAT JSON)")
        logger.info(f"Successfully saved metrics to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {str(e)}")
        raise


# ===== S3 Folder Management =====
def get_output_paths(cohort_name, age_band, event_year, bucket_name="pgxdatalake"):
    """Return output paths organized for AWS Glue crawlers and QuickSight."""
    try:
        # Validate input parameters to prevent None values in S3 paths
        if cohort_name is None or cohort_name == "":
            raise ValueError("cohort_name cannot be None or empty")
        if age_band is None or age_band == "":
            raise ValueError("age_band cannot be None or empty")
        if event_year is None or event_year == "":
            raise ValueError("event_year cannot be None or empty")

        if isinstance(age_band, str) and age_band.startswith('age_band='):
            age_band = age_band.replace('age_band=', '')

        partitions = f"cohort_name={cohort_name}/event_year={event_year}/age_band={age_band}"
        network_plot_name = f"{cohort_name}_{age_band}_{event_year}_drug_network.html"

        # Final deliverables: write FP-Growth artifacts under GOLD fpgrowth/cohort
        cohort_base = f"s3://{bucket_name}/gold/fpgrowth/cohort/{partitions}"
        # Curated cohort parquet lives under GOLD cohorts (default)
        gold_cohorts_base = f"s3://{bucket_name}/gold/cohorts/{partitions}"

        paths = {
            "fpgrowth_features_parquet": f"{cohort_base}/fpgrowth_features.parquet",
            "itemsets_parquet": f"{cohort_base}/itemsets.parquet",
            "itemsets_json": f"{cohort_base}/itemsets.json",
            "features_manifest_json": f"{cohort_base}/feature_manifest.json",
            "rules_parquet": f"{cohort_base}/rules.parquet",
            "rules_json": f"{cohort_base}/rules.json",
            "network_statistics_json": f"{cohort_base}/network_stats.json",
            "combined_rules_json": f"{cohort_base}/combined_rules.json",
            "combined_itemsets_json": f"{cohort_base}/combined_itemsets.json",
            "combined_manifest_json": f"{cohort_base}/combined_manifest.json",
            "drug_encoding_json": f"{cohort_base}/drug_encoding.json",
            "drug_encoding_parquet": f"{cohort_base}/drug_encoding.parquet",
            "drug_manifest_json": f"{cohort_base}/drug_manifest.json",
            "drug_network_plot": f"{cohort_base}/{network_plot_name}",
            "malformed_rules_json": f"{cohort_base}/malformed_rules.json",
            "pattern_map_parquet": f"{cohort_base}/pattern_map.parquet",
            "model_metrics_json": f"{cohort_base}/model_metrics.json",
            "shap_values_parquet": f"{cohort_base}/shap_values.parquet",
            "shap_plots": f"{cohort_base}/shap_plots",
            "cattail_plots": f"{cohort_base}/cattail_plots",
            "causal_summary_json": f"{cohort_base}/causal_summary.json",
            "model_info_json": f"{cohort_base}/model_info.json",
            "calibration_plots": f"{cohort_base}/calibration_plots",
            "mirror_plots": f"{cohort_base}/mirror_plots",
            "axp_metrics_json": f"{cohort_base}/axp_metrics.json",
            "pipeline_metrics_json": f"{cohort_base}/pipeline_metrics.json",
            "cohort_parquet": f"{gold_cohorts_base}/cohort.parquet",
            "validation_results_json": f"{cohort_base}/validation_results.json"
        }
        return paths
    except Exception as e:
        # Add more detailed error information
        error_msg = f"Error generating output paths: cohort_name={cohort_name}, age_band={age_band}, event_year={event_year}, error={str(e)}"
        print(error_msg)  # Print to console since logger might not be available
        raise ValueError(error_msg) from e


def get_global_base_path(bucket_name: str = "pgxdatalake") -> str:
    """Return the GOLD base path for global FP-Growth outputs."""
    return f"s3://{bucket_name}/gold/fpgrowth/global"


# ===== Cohort name normalization and paths =====
COHORT_ALIASES = {
    "ed_non_opioid": "non_opioid_ed",
}

def normalize_cohort_name(name: str) -> str:
    """Normalize cohort name to canonical S3 partition slug."""
    return COHORT_ALIASES.get(str(name).lower(), str(name).lower())


def _sanitize_target_slug(raw: str) -> str:
    import re
    # Replace non alphanumeric with underscores, collapse repeats
    s = re.sub(r"[^A-Za-z0-9]+", "_", raw)
    s = s.strip("_")
    return s or "custom"

from helpers_1997_13.constants import (
    PGX_TARGET_NAME,
    PGX_TARGET_ICD_CODES,
    PGX_TARGET_CPT_CODES,
    PGX_TARGET_ICD_PREFIXES,
    PGX_TARGET_CPT_PREFIXES,
)


def _derive_target_slug_from_env() -> str | None:
    """Derive a target slug from env vars if provided.

    Priority order:
    1) PGX_TARGET_NAME
    2) First of PGX_TARGET_ICD_CODES, then PGX_TARGET_CPT_CODES
    3) First of PGX_TARGET_ICD_PREFIXES, PGX_TARGET_CPT_PREFIXES
    Returns sanitized slug or None if nothing set.
    """
    try:
        name = PGX_TARGET_NAME
        if name:
            return _sanitize_target_slug(name)

        icd_codes = [c.strip() for c in PGX_TARGET_ICD_CODES.split(',') if c.strip()]
        if icd_codes:
            return _sanitize_target_slug(icd_codes[0])

        cpt_codes = [c.strip() for c in PGX_TARGET_CPT_CODES.split(',') if c.strip()]
        if cpt_codes:
            return _sanitize_target_slug(cpt_codes[0])

        icd_pref = [p.strip() for p in PGX_TARGET_ICD_PREFIXES.split(',') if p.strip()]
        if icd_pref:
            return _sanitize_target_slug(icd_pref[0])

        cpt_pref = [p.strip() for p in PGX_TARGET_CPT_PREFIXES.split(',') if p.strip()]
        if cpt_pref:
            return _sanitize_target_slug(cpt_pref[0])
    except Exception:
        pass
    return None

def get_target_slug() -> str:
    """Public accessor for target slug; raises if not configured.

    Derives from PGX_TARGET_NAME or the first provided ICD/CPT code/prefix.
    """
    slug = _derive_target_slug_from_env()
    if not slug:
        raise ValueError(
            "Target not configured. Set PGX_TARGET_NAME or one of PGX_TARGET_ICD_CODES/PGX_TARGET_CPT_CODES/PGX_TARGET_ICD_PREFIXES/PGX_TARGET_CPT_PREFIXES"
        )
    return slug

def get_cohort_parquet_path(
    cohort_name: str,
    age_band: str,
    event_year: str | int,
    bucket_name: str = S3_BUCKET,
    target_slug: str | None = None,
) -> str:
    """Build S3 path to GOLD cohorts parquet for a cohort partition.
    
    Structure: cohorts_{target}/cohort_name={cohort}/event_year={year}/age_band={age_band}/cohort.parquet
    
    Organized by cohort name first, then by year and age-band partitions.

    If a target slug is available (either passed or from environment), put it into the directory name
    as cohorts_{slug} and write standard filename cohort.parquet. Otherwise, use default cohorts_clean directory.
    """
    cohort_slug = normalize_cohort_name(cohort_name)
    dir_slug = target_slug or _derive_target_slug_from_env()
    if not dir_slug:
        raise ValueError(
            "Target not configured. Set PGX_TARGET_NAME or one of PGX_TARGET_ICD_CODES/PGX_TARGET_CPT_CODES/PGX_TARGET_ICD_PREFIXES/PGX_TARGET_CPT_PREFIXES"
        )
    base_dir = f"s3://{bucket_name}/gold/cohorts_{dir_slug}/"
    return (
        f"{base_dir}"
        f"cohort_name={cohort_slug}/event_year={event_year}/age_band={age_band}/cohort.parquet"
    )


# ===== Lock Management =====
def acquire_lock(lock_key, bucket_name="pgxdatalake"):
    try:
        timestamp = datetime.now().isoformat()
        s3_client.put_object(Bucket=bucket_name, Key=lock_key, Body=timestamp.encode("utf-8"))
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
            print(f"Lock already exists at {lock_key}")
            return False
        raise


def release_lock(lock_key, bucket_name="pgxdatalake"):
    """Release a lock by deleting the lock file from S3."""
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=lock_key)
        return True
    except Exception as e:
        print(f"Error releasing lock: {e}")
        return False


# ===== Checkpoint Functions =====
def get_checkpoint_path(step_name: str, age_band: str, event_year: int, bucket_name: str = S3_BUCKET) -> str:
    """
    Generate checkpoint S3 path for a specific step.

    Args:
        step_name: Name of the step (e.g., 'step9_tagged_cohort_events', 'step11_control_cohort')
        age_band: Age band being processed
        event_year: Event year being processed
        bucket_name: S3 bucket name

    Returns:
        Full S3 path for the checkpoint
    """
    sanitized_age_band = _sanitize_age_band(age_band)
    checkpoint_key = f"checkpoints/cohorts/{step_name}_{sanitized_age_band}_{event_year}.json"
    return f"s3://{bucket_name}/{checkpoint_key}"


def checkpoint_exists(step_name: str, age_band: str, event_year: int, bucket_name: str = S3_BUCKET) -> bool:
    """
    Check if a checkpoint exists for a specific step.

    Args:
        step_name: Name of the step
        age_band: Age band being processed
        event_year: Event year being processed
        bucket_name: S3 bucket name

    Returns:
        True if checkpoint exists, False otherwise
    """
    checkpoint_path = get_checkpoint_path(step_name, age_band, event_year, bucket_name)
    return s3_exists(checkpoint_path, bucket_name)


def save_checkpoint(
    step_name: str,
    age_band: str,
    event_year: int,
    checkpoint_data: dict,
    logger: logging.Logger,
    bucket_name: str = S3_BUCKET
) -> bool:
    """
    Save checkpoint data for a specific step.

    Args:
        step_name: Name of the step
        age_band: Age band being processed
        event_year: Event year being processed
        checkpoint_data: Data to save in checkpoint
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        True if checkpoint saved successfully, False otherwise
    """
    try:
        checkpoint_path = get_checkpoint_path(step_name, age_band, event_year, bucket_name)

        # Add metadata to checkpoint data
        checkpoint_data.update({
            "checkpoint_metadata": {
                "step_name": step_name,
                "age_band": age_band,
                "event_year": event_year,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        })

        save_to_s3_json(checkpoint_data, checkpoint_path, logger)
        logger.info(f"✓ Checkpoint saved for {step_name}: {checkpoint_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Error saving checkpoint for {step_name}: {str(e)}")
        return False


def load_checkpoint(
    step_name: str,
    age_band: str,
    event_year: int,
    logger: logging.Logger,
    bucket_name: str = S3_BUCKET
) -> Optional[dict]:
    """
    Load checkpoint data for a specific step.

    Args:
        step_name: Name of the step
        age_band: Age band being processed
        event_year: Event year being processed
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        Checkpoint data if exists, None otherwise
    """
    try:
        checkpoint_path = get_checkpoint_path(step_name, age_band, event_year, bucket_name)

        if not s3_exists(checkpoint_path, bucket_name):
            logger.info(f"No checkpoint found for {step_name}: {checkpoint_path}")
            return None

        # Parse S3 path
        bucket, key = _parse_s3_path_components(checkpoint_path)

        # Download checkpoint data
        response = s3_client.get_object(Bucket=bucket, Key=key)
        checkpoint_data = json.loads(response['Body'].read().decode('utf-8'))

        logger.info(f"✓ Checkpoint loaded for {step_name}: {checkpoint_path}")
        logger.info(f"  Checkpoint timestamp: {checkpoint_data.get('checkpoint_metadata', {}).get('timestamp', 'unknown')}")

        return checkpoint_data

    except Exception as e:
        logger.error(f"✗ Error loading checkpoint for {step_name}: {str(e)}")
        return None


def delete_checkpoint(
    step_name: str,
    age_band: str,
    event_year: int,
    logger: logging.Logger,
    bucket_name: str = S3_BUCKET
) -> bool:
    """
    Delete checkpoint for a specific step.

    Args:
        step_name: Name of the step
        age_band: Age band being processed
        event_year: Event year being processed
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        True if checkpoint deleted successfully, False otherwise
    """
    try:
        checkpoint_path = get_checkpoint_path(step_name, age_band, event_year, bucket_name)
        bucket, key = _parse_s3_path_components(checkpoint_path)

        s3_client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"✓ Checkpoint deleted for {step_name}: {checkpoint_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Error deleting checkpoint for {step_name}: {str(e)}")
        return False


def list_checkpoints(
    age_band: str,
    event_year: int,
    logger: logging.Logger,
    bucket_name: str = S3_BUCKET
) -> List[str]:
    """
    List all checkpoints for a specific age_band and event_year.

    Args:
        age_band: Age band being processed
        event_year: Event year being processed
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        List of step names that have checkpoints
    """
    try:
        sanitized_age_band = _sanitize_age_band(age_band)
        prefix = "checkpoints/cohorts/"

        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )

        checkpoints = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Extract step name from key
                key = obj['Key']
                # Look for files matching the pattern: {step_name}_{age_band}_{event_year}.json
                if key.endswith('.json') and f"_{sanitized_age_band}_{event_year}.json" in key:
                    # Extract step name from the key
                    # Key format: checkpoints/cohorts/{step_name}_{age_band}_{event_year}.json
                    filename = key.split('/')[-1]  # Get the filename part
                    step_name = filename.replace(f"_{sanitized_age_band}_{event_year}.json", "")
                    checkpoints.append(step_name)

        logger.info(f"Found {len(checkpoints)} checkpoints for {age_band}/{event_year}: {checkpoints}")
        return checkpoints

    except Exception as e:
        logger.error(f"✗ Error listing checkpoints: {str(e)}")
        return []


def cleanup_checkpoints(
    age_band: str,
    event_year: int,
    logger: logging.Logger,
    bucket_name: str = S3_BUCKET
) -> bool:
    """
    Clean up all checkpoints for a specific age_band and event_year.

    Args:
        age_band: Age band being processed
        event_year: Event year being processed
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        checkpoints = list_checkpoints(age_band, event_year, logger, bucket_name)

        for step_name in checkpoints:
            delete_checkpoint(step_name, age_band, event_year, logger, bucket_name)

        logger.info(f"✓ Cleaned up {len(checkpoints)} checkpoints for {age_band}/{event_year}")
        return True

    except Exception as e:
        logger.error(f"✗ Error cleaning up checkpoints: {str(e)}")
        return False


def get_checkpoint_data_path(step_name: str, age_band: str, event_year: int, data_type: str, bucket_name: str = S3_BUCKET) -> str:
    """
    Generate checkpoint data file S3 path for a specific step and data type.

    Args:
        step_name: Name of the step (e.g., 'step9_tagged_cohort_events', 'step11_control_cohort')
        age_band: Age band being processed
        event_year: Event year being processed
        data_type: Type of data file (e.g., 'tagged_cohort_events', 'control_cohort_events')
        bucket_name: S3 bucket name

    Returns:
        Full S3 path for the checkpoint data file
    """
    sanitized_age_band = _sanitize_age_band(age_band)
    data_key = f"checkpoints/cohorts/data/{step_name}_{sanitized_age_band}_{event_year}/{data_type}.parquet"
    return f"s3://{bucket_name}/{data_key}"


def save_checkpoint_with_data(
    step_name: str,
    age_band: str,
    event_year: int,
    checkpoint_data: dict,
    data_files: dict,  # {view_name: DataFrame}
    conn,  # DuckDB connection
    logger: logging.Logger,
    bucket_name: str = "pgxdatalake"
) -> bool:
    """
    Save checkpoint data for a specific step, including actual data files.

    Args:
        step_name: Name of the step
        age_band: Age band being processed
        event_year: Event year being processed
        checkpoint_data: Metadata to save in checkpoint
        data_files: Dict mapping view names to their data (DataFrames or view names)
        conn: DuckDB connection for executing queries
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        True if checkpoint saved successfully, False otherwise
    """
    try:
        # Save metadata checkpoint
        checkpoint_path = get_checkpoint_path(step_name, age_band, event_year, bucket_name)

        # Add metadata to checkpoint data
        checkpoint_data.update({
            "checkpoint_metadata": {
                "step_name": step_name,
                "age_band": age_band,
                "event_year": event_year,
                "timestamp": datetime.now().isoformat(),
                "version": "2.0",  # New version with data files
                "has_data_files": True
            },
            "data_files": {}  # Will be populated with S3 paths
        })

        # Save each data file
        for view_name, data_source in data_files.items():
            try:
                data_path = get_checkpoint_data_path(step_name, age_band, event_year, view_name, bucket_name)

                if isinstance(data_source, str):
                    # data_source is a view name - query it
                    logger.info(f"→ Saving checkpoint data: {view_name} from view {data_source}")
                    df = conn.sql(f"SELECT * FROM {data_source}").df()
                else:
                    # data_source is already a DataFrame
                    logger.info(f"→ Saving checkpoint data: {view_name} from DataFrame")
                    df = data_source

                # Save as parquet
                save_to_s3_parquet(df, data_path, logger)

                # Add to checkpoint metadata
                checkpoint_data["data_files"][view_name] = data_path

                logger.info(f"→ ✓ Saved checkpoint data file: {data_path}")

            except Exception as data_e:
                logger.error(f"→ ✗ Error saving checkpoint data file {view_name}: {str(data_e)}")
                # Continue with other files, but mark this one as failed
                checkpoint_data["data_files"][view_name] = f"ERROR: {str(data_e)}"

        # Save the metadata checkpoint
        save_to_s3_json(checkpoint_data, checkpoint_path, logger)
        logger.info(f"✓ Checkpoint saved for {step_name}: {checkpoint_path}")
        logger.info(f"  Data files: {list(checkpoint_data['data_files'].keys())}")
        return True

    except Exception as e:
        logger.error(f"✗ Error saving checkpoint for {step_name}: {str(e)}")
        return False


def load_checkpoint_with_data(
    step_name: str,
    age_band: str,
    event_year: int,
    conn,  # DuckDB connection
    logger: logging.Logger,
    bucket_name: str = "pgxdatalake"
) -> Optional[dict]:
    """
    Load checkpoint data for a specific step, including actual data files.

    Args:
        step_name: Name of the step
        age_band: Age band being processed
        event_year: Event year being processed
        conn: DuckDB connection for creating views
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        Checkpoint data if exists, None otherwise
    """
    try:
        # Load metadata checkpoint
        checkpoint_data = load_checkpoint(step_name, age_band, event_year, logger, bucket_name)

        if not checkpoint_data:
            return None

        # Check if this checkpoint has data files
        if not checkpoint_data.get("checkpoint_metadata", {}).get("has_data_files", False):
            logger.info(f"→ Checkpoint {step_name} does not have data files (legacy checkpoint)")
            return checkpoint_data

        # Load data files and recreate views
        data_files = checkpoint_data.get("data_files", {})
        logger.info(f"→ Loading {len(data_files)} data files for checkpoint {step_name}")

        # Ensure AWS credentials are loaded for S3 reads
        try:
            conn.sql("CALL load_aws_credentials('');")
            logger.info("→ AWS credentials loaded for S3 reads")
        except Exception as cred_e:
            logger.warning(f"→ Could not load AWS credentials: {str(cred_e)}")
            logger.warning("→ S3 reads may fail if credentials are not available")

        for view_name, data_path in data_files.items():
            try:
                if isinstance(data_path, str) and data_path.startswith("ERROR:"):
                    logger.warning(f"→ Skipping {view_name} - had error during save: {data_path}")
                    continue

                logger.info(f"→ Loading checkpoint data: {view_name} from {data_path}")

                # Create view from parquet file - handle S3 reads without Glue partitions
                # Use explicit S3 configuration for reliable reads
                create_view_sql = f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet(
                    '{data_path}',
                    hive_partitioning=false,
                    filename=true
                )
                """

                logger.info(f"→ Executing SQL: {create_view_sql}")
                conn.sql(create_view_sql)

                # Verify view was created and has data
                try:
                    row_count = conn.sql(f"SELECT COUNT(*) as count FROM {view_name}").df()
                    logger.info(f"→ ✓ Loaded {view_name} with {row_count.iloc[0]['count']} rows")

                    # Additional verification - check if view has expected columns
                    try:
                        sample_data = conn.sql(f"SELECT * FROM {view_name} LIMIT 1").df()
                        logger.info(f"→ ✓ {view_name} has {len(sample_data.columns)} columns")
                    except Exception as sample_e:
                        logger.warning(f"→ Could not verify columns for {view_name}: {str(sample_e)}")

                except Exception as verify_e:
                    logger.error(f"→ ✗ Failed to verify {view_name} after creation: {str(verify_e)}")
                    # Try to get more information about the failure
                    try:
                        # Check if the file exists in S3
                        if data_path.startswith("s3://"):
                            bucket_key = data_path.replace("s3://", "")
                            parts = bucket_key.split("/", 1)
                            if len(parts) == 2:
                                bucket, key = parts
                                response = s3_client.head_object(Bucket=bucket, Key=key)
                                file_size = response.get('ContentLength', 0)
                                logger.info(f"→ S3 file exists: {data_path} ({file_size} bytes)")
                            else:
                                logger.error(f"→ Invalid S3 path format: {data_path}")
                    except Exception as s3_check_e:
                        logger.error(f"→ S3 file check failed: {str(s3_check_e)}")

                    # Continue with other files but mark this as problematic
                    continue

            except Exception as data_e:
                logger.error(f"→ ✗ Error loading checkpoint data file {view_name}: {str(data_e)}")
                logger.error(f"→ Error type: {type(data_e).__name__}")

                # Provide specific guidance for common S3 read issues
                if "Access Denied" in str(data_e) or "403" in str(data_e):
                    logger.error("→ This appears to be an S3 permissions issue")
                    logger.error("→ Check that the process has read access to {data_path}")
                elif "No such file" in str(data_e) or "404" in str(data_e):
                    logger.error("→ This appears to be a missing file issue")
                    logger.error("→ The checkpoint data file may have been deleted or moved")
                elif "Invalid parquet" in str(data_e):
                    logger.error("→ This appears to be a corrupted parquet file")
                    logger.error("→ The checkpoint data file may be incomplete")

                # Continue with other files
                continue

        logger.info(f"✓ Checkpoint loaded for {step_name} with data files")
        return checkpoint_data

    except Exception as e:
        logger.error(f"✗ Error loading checkpoint for {step_name}: {str(e)}")
        logger.error(f"✗ Error type: {type(e).__name__}")
        return None


def cleanup_checkpoints_with_data(
    age_band: str,
    event_year: int,
    logger: logging.Logger,
    bucket_name: str = "pgxdatalake"
) -> bool:
    """
    Clean up all checkpoints for a specific age_band and event_year, including data files.

    Args:
        age_band: Age band being processed
        event_year: Event year being processed
        logger: Logger instance
        bucket_name: S3 bucket name

    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        checkpoints = list_checkpoints(age_band, event_year, logger, bucket_name)

        for step_name in checkpoints:
            # Load checkpoint to get data file paths
            checkpoint_data = load_checkpoint(step_name, age_band, event_year, logger, bucket_name)

            if checkpoint_data and checkpoint_data.get("checkpoint_metadata", {}).get("has_data_files", False):
                # Delete data files
                data_files = checkpoint_data.get("data_files", {})
                for data_path in data_files.values():
                    if isinstance(data_path, str) and not data_path.startswith("ERROR:"):
                        try:
                            bucket, key = data_path.replace("s3://", "").split("/", 1)
                            s3_client.delete_object(Bucket=bucket, Key=key)
                            logger.info(f"→ Deleted checkpoint data file: {data_path}")
                        except Exception as data_e:
                            logger.warning(f"→ Could not delete checkpoint data file {data_path}: {str(data_e)}")

            # Delete metadata checkpoint
            delete_checkpoint(step_name, age_band, event_year, logger, bucket_name)

        logger.info(f"✓ Cleaned up {len(checkpoints)} checkpoints with data files for {age_band}/{event_year}")
        return True

    except Exception as e:
        logger.error(f"✗ Error cleaning up checkpoints: {str(e)}")
        return False


def verify_s3_data_availability(s3_path, logger, max_retries=3, retry_delay=5):
    """Verify that S3 data is available after saving, with retry logic for eventual consistency."""
    logger.info(f"→ Verifying S3 data availability: {s3_path}")

    for attempt in range(max_retries):
        try:
            # Extract bucket and key from S3 path
            if s3_path.startswith("s3://"):
                bucket_key = s3_path.replace("s3://", "")
                parts = bucket_key.split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts

                    # Check if object exists and has content
                    response = s3_client.head_object(Bucket=bucket, Key=key)
                    content_length = response.get('ContentLength', 0)

                    if content_length > 0:
                        logger.info(f"→ ✓ S3 data verified: {s3_path} ({content_length} bytes)")
                        return True
                    else:
                        logger.warning(f"→ S3 data exists but is empty: {s3_path}")
                        return False
                else:
                    logger.error(f"→ Invalid S3 path format: {s3_path}")
                    return False
            else:
                logger.error(f"→ S3 path does not start with s3://: {s3_path}")
                return False

        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                if attempt < max_retries - 1:
                    logger.warning(f"→ S3 data not yet available (attempt {attempt + 1}/{max_retries}): {s3_path}")
                    logger.info(f"→ Waiting {retry_delay} seconds for eventual consistency...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"→ S3 data not available after {max_retries} attempts: {s3_path}")
                    logger.error("→ This may indicate a save failure or eventual consistency issue")
                    return False
            else:
                logger.error(f"→ S3 error checking data availability: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"→ Unexpected error verifying S3 data: {str(e)}")
            return False

    return False


def save_checkpoint_with_verification(step_name, age_band, event_year, checkpoint_data, data_files, conn, logger):
    """Save checkpoint data with verification that files are actually available."""
    logger.info(f"→ Saving checkpoint with verification: {step_name}")

    try:
        # Save checkpoint data using the existing function
        success = save_checkpoint_with_data(
            step_name, age_band, event_year, checkpoint_data, data_files, conn, logger
        )

        if not success:
            logger.error(f"→ Failed to save checkpoint: {step_name}")
            return False

        # Verify that all data files are actually available
        logger.info(f"→ Verifying checkpoint data files for {step_name}...")
        data_files_metadata = checkpoint_data.get("data_files", {})

        verification_failures = []
        for view_name, data_path in data_files_metadata.items():
            if isinstance(data_path, str) and not data_path.startswith("ERROR:"):
                if not verify_s3_data_availability(data_path, logger):
                    verification_failures.append(view_name)
                    logger.error(f"→ Verification failed for {view_name}: {data_path}")

        if verification_failures:
            logger.error(f"→ Checkpoint verification failed for {len(verification_failures)} files: {verification_failures}")
            logger.error("→ This may cause issues during checkpoint restoration")
            logger.error("→ Consider re-running the step or checking S3 permissions")
            return False
        else:
            logger.info(f"→ ✓ All checkpoint data files verified for {step_name}")
            return True

    except Exception as e:
        logger.error(f"→ Error in save_checkpoint_with_verification: {str(e)}")
        return False


def load_checkpoint_with_verification(step_name, age_band, event_year, conn, logger):
    """Load checkpoint data with verification that files are accessible."""
    logger.info(f"→ Loading checkpoint with verification: {step_name}")

    try:
        # Load checkpoint metadata
        checkpoint_data = load_checkpoint_with_data(step_name, age_band, event_year, conn, logger)

        if not checkpoint_data:
            logger.info(f"→ No checkpoint found for {step_name}")
            return None

        # Verify that all data files are accessible before proceeding
        logger.info(f"→ Verifying checkpoint data file accessibility for {step_name}...")
        data_files = checkpoint_data.get("data_files", {})

        verification_failures = []
        for view_name, data_path in data_files.items():
            if isinstance(data_path, str) and not data_path.startswith("ERROR:"):
                if not verify_s3_data_availability(data_path, logger):
                    verification_failures.append(view_name)
                    logger.error(f"→ Data file not accessible: {view_name}: {data_path}")

        if verification_failures:
            logger.error(f"→ Checkpoint data files not accessible: {verification_failures}")
            logger.error(f"→ Cannot restore checkpoint {step_name} - data files missing")
            logger.error("→ This may be due to S3 eventual consistency or file deletion")
            return None
        else:
            logger.info(f"→ ✓ All checkpoint data files accessible for {step_name}")
            return checkpoint_data

    except Exception as e:
        logger.error(f"→ Error in load_checkpoint_with_verification: {str(e)}")
        return None


def load_s3_data_without_glue(conn, s3_path, view_name, logger, description="data"):
    """Load S3 data without relying on Glue partitions."""
    logger.info(f"→ Loading {description} from S3 without Glue partitions: {s3_path}")

    try:
        # Ensure AWS credentials are loaded
        try:
            conn.sql("CALL load_aws_credentials('');")
            logger.info("→ AWS credentials loaded for S3 reads")
        except Exception as cred_e:
            logger.warning(f"→ Could not load AWS credentials: {str(cred_e)}")
            logger.warning("→ S3 reads may fail if credentials are not available")

        # Create view with explicit S3 configuration
        create_view_sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT * FROM read_parquet(
            '{s3_path}',
            hive_partitioning=false,
            filename=true
        )
        """

        logger.info(f"→ Creating view {view_name} from S3 data...")
        conn.sql(create_view_sql)

        # Verify the view was created successfully
        try:
            row_count = conn.sql(f"SELECT COUNT(*) as count FROM {view_name}").df()
            logger.info(f"→ ✓ Successfully loaded {description}: {row_count.iloc[0]['count']} rows")

            # Check for any obvious data quality issues
            if row_count.iloc[0]['count'] == 0:
                logger.warning(f"→ WARNING: {description} view is empty!")
                logger.warning("→ This may indicate the S3 path is incorrect or the data is missing")

            return True

        except Exception as verify_e:
            logger.error(f"→ ✗ Failed to verify {view_name} after creation: {str(verify_e)}")

            # Try to get more information about the S3 path
            try:
                if s3_path.startswith("s3://"):
                    bucket_key = s3_path.replace("s3://", "")
                    parts = bucket_key.split("/", 1)
                    if len(parts) == 2:
                        bucket, key = parts
                        response = s3_client.head_object(Bucket=bucket, Key=key)
                        file_size = response.get('ContentLength', 0)
                        logger.info(f"→ S3 file exists: {s3_path} ({file_size} bytes)")
                    else:
                        logger.error(f"→ Invalid S3 path format: {s3_path}")
            except Exception as s3_check_e:
                logger.error(f"→ S3 file check failed: {str(s3_check_e)}")

            return False

    except Exception as e:
        logger.error(f"→ ✗ Error loading {description} from S3: {str(e)}")
        logger.error(f"→ Error type: {type(e).__name__}")

        # Provide specific guidance for common S3 read issues
        if "Access Denied" in str(e) or "403" in str(e):
            logger.error("→ This appears to be an S3 permissions issue")
            logger.error("→ Check that the process has read access to {s3_path}")
        elif "No such file" in str(e) or "404" in str(e):
            logger.error("→ This appears to be a missing file issue")
            logger.error("→ The S3 path may be incorrect or the data may not exist")
        elif "Invalid parquet" in str(e):
            logger.error("→ This appears to be a corrupted parquet file")
            logger.error("→ The S3 data may be incomplete or corrupted")
        elif "hive_partitioning" in str(e):
            logger.error("→ This appears to be a DuckDB version compatibility issue")
            logger.error("→ Try removing the hive_partitioning parameter")

        return False


def list_checkpoints_with_data(age_band, event_year, logger):
    """List all data checkpoints for a given age_band and event_year."""
    try:
        # List data checkpoints (those with .parquet extension in data subdirectory)
        data_checkpoints = []
        prefix = f"checkpoints/cohorts/data/"
        
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        # Filter for checkpoints matching this age_band and event_year
        sanitized_age_band = _sanitize_age_band(age_band)
        target_pattern = f"_{sanitized_age_band}_{event_year}/"
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            if target_pattern in key and key.endswith('.parquet'):
                # Extract step name from the key
                # Key format: checkpoints/cohorts/data/{step_name}_{age_band}_{event_year}/{data_type}.parquet
                parts = key.split('/')
                if len(parts) >= 5:
                    step_name = parts[3]  # Get the step_name part
                    data_checkpoints.append(step_name)
        
        # Remove duplicates (same step can have multiple data files)
        data_checkpoints = list(set(data_checkpoints))
        
        logger.info(f"→ Found {len(data_checkpoints)} data checkpoints")
        return data_checkpoints
        
    except Exception as e:
        logger.warning(f"→ Error listing data checkpoints: {str(e)}")
        return []


def cleanup_incorrect_checkpoints(age_band, event_year, logger):
    """Clean up checkpoints that were created with incorrect age bands or years."""
    try:
        logger.info(f"→ Cleaning up incorrect checkpoints for {age_band}/{event_year}")

        # List all checkpoints (both metadata and data checkpoints)
        metadata_checkpoints = list_checkpoints(age_band, event_year, logger)
        data_checkpoints = list_checkpoints_with_data(age_band, event_year, logger)

        all_checkpoints = metadata_checkpoints + data_checkpoints

        # Check for checkpoints with incorrect age bands
        incorrect_checkpoints = []
        for checkpoint in all_checkpoints:
            # Check if checkpoint name contains incorrect age band patterns
            if '18-64_2022' in checkpoint and age_band != '18-64':
                incorrect_checkpoints.append(checkpoint)
                logger.warning(f"→ Found incorrect checkpoint: {checkpoint}")
            elif '2022' in checkpoint and event_year != 2022:
                incorrect_checkpoints.append(checkpoint)
                logger.warning(f"→ Found incorrect checkpoint: {checkpoint}")

        # Clean up incorrect checkpoints
        for checkpoint in incorrect_checkpoints:
            try:
                delete_checkpoint(checkpoint, age_band, event_year, logger)
                logger.info(f"→ Deleted incorrect checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"→ Could not delete checkpoint {checkpoint}: {str(e)}")

        return len(incorrect_checkpoints)
    except Exception as e:
        logger.warning(f"→ Error during checkpoint cleanup: {str(e)}")
        return 0


# ===== Shared Connection Pool for Multiprocessing =====

# Global connection pool for multiprocessing
_connection_pool = {}
_pool_lock = threading.Lock()

def get_shared_s3_client(worker_id: Optional[str] = None) -> Any:
    """
    Get a shared S3 client for multiprocessing environments.
    
    Args:
        worker_id: Optional worker identifier for tracking
        
    Returns:
        boto3 S3 client with proper configuration
    """
    import boto3
    from botocore.config import Config
    
    # Create a unique key for this worker's connection
    worker_key = worker_id or f"worker_{threading.get_ident()}"
    
    with _pool_lock:
        if worker_key not in _connection_pool:
            # Create new S3 client with optimized configuration for multiprocessing
            config = Config(
                retries=dict(
                    max_attempts=3,
                    mode='adaptive'
                ),
                connect_timeout=10,
                read_timeout=30,
                max_pool_connections=10,  # Limit connection pool size
                region_name='us-east-1'
            )
            
            _connection_pool[worker_key] = boto3.client(
                "s3", 
                config=config, 
                verify=certifi.where()
            )
            
        return _connection_pool[worker_key]

def get_shared_duckdb_connection(worker_id: Optional[str] = None, logger=None):
    """
    Get a shared DuckDB connection with AWS credentials for multiprocessing.
    
    Args:
        worker_id: Optional worker identifier for tracking
        logger: Optional logger for tracking
        
    Returns:
        DuckDB connection with AWS credentials loaded
    """
    import duckdb
    
    # Create a unique key for this worker's connection
    worker_key = worker_id or f"worker_{threading.get_ident()}"
    
    with _pool_lock:
        if f"duckdb_{worker_key}" not in _connection_pool:
            try:
                # Create new DuckDB connection using standardized function
                from helpers_1997_13.duckdb_utils import get_shared_duckdb_connection
                conn = get_shared_duckdb_connection(worker_key, logger)
                
                _connection_pool[f"duckdb_{worker_key}"] = conn
                
                if logger:
                    logger.info(f"Created shared DuckDB connection for worker {worker_key}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Failed to create shared DuckDB connection: {str(e)}")
                raise
                
        return _connection_pool[f"duckdb_{worker_key}"]

@contextmanager
def shared_aws_connection(worker_id: Optional[str] = None, logger=None):
    """
    Context manager for shared AWS connections in multiprocessing.
    
    Args:
        worker_id: Optional worker identifier for tracking
        logger: Optional logger for tracking
        
    Yields:
        Tuple of (s3_client, duckdb_connection)
    """
    s3_client = None
    duckdb_conn = None
    
    try:
        # Get shared connections
        s3_client = get_shared_s3_client(worker_id)
        duckdb_conn = get_shared_duckdb_connection(worker_id, logger)
        
        yield s3_client, duckdb_conn
        
    except Exception as e:
        if logger:
            logger.error(f"Error in shared AWS connection: {str(e)}")
        raise
    finally:
        # Note: We don't close the connections here as they're shared
        # The connections will be cleaned up when the process ends
        pass

def cleanup_shared_connections(worker_id: Optional[str] = None, logger=None):
    """
    Clean up shared connections for a specific worker or all workers.
    
    Args:
        worker_id: Optional worker identifier. If None, cleans up all connections.
        logger: Optional logger for tracking
    """
    with _pool_lock:
        if worker_id:
            # Clean up specific worker connections
            keys_to_remove = [k for k in _connection_pool.keys() 
                             if worker_id in k or k == worker_id]
        else:
            # Clean up all connections
            keys_to_remove = list(_connection_pool.keys())
        
        for key in keys_to_remove:
            try:
                conn = _connection_pool[key]
                if hasattr(conn, 'close'):
                    conn.close()
                del _connection_pool[key]
                
                if logger:
                    logger.info(f"Cleaned up shared connection: {key}")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Error cleaning up connection {key}: {str(e)}")

def get_connection_pool_status() -> Dict[str, Any]:
    """
    Get status of the shared connection pool.
    
    Returns:
        Dictionary with pool status information
    """
    with _pool_lock:
        return {
            'total_connections': len(_connection_pool),
            'connection_keys': list(_connection_pool.keys()),
            's3_clients': len([k for k in _connection_pool.keys() if not k.startswith('duckdb_')]),
            'duckdb_connections': len([k for k in _connection_pool.keys() if k.startswith('duckdb_')])
        }


def s3_path_to_bucket_key(s3_path: str) -> tuple:
    """Parse S3 path into bucket and key components.
    
    Args:
        s3_path: S3 path like s3://bucket/path/to/file
        
    Returns:
        Tuple of (bucket_name, key_path)
    """
    from urllib.parse import urlparse
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def delete_s3_parquet_files(s3_partition_path: str, logger=None) -> int:
    """Delete all parquet files in an S3 partition.
    
    Args:
        s3_partition_path: S3 path to partition (e.g., s3://bucket/path/age_band=0-12/event_year=2020)
        logger: Optional logger instance
        
    Returns:
        Number of files deleted
    """
    if not s3_partition_path.startswith("s3://"):
        return 0
        
    try:
        s3 = boto3.client("s3")
        bucket, key_prefix = s3_path_to_bucket_key(s3_partition_path)
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        deleted_count = 0
        
        for obj in resp.get('Contents', []):
            if obj['Key'].endswith('.parquet'):
                s3.delete_object(Bucket=bucket, Key=obj['Key'])
                deleted_count += 1
        
        if deleted_count > 0 and logger:
            logger.info(f"→ Pre-deleted {deleted_count} Parquet files under {s3_partition_path}")
        
        return deleted_count
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Pre-delete encountered an issue (will still attempt write): {e}")
        return 0


def copy_rename_s3_parquet_file(silver_partition: str, gold_partition: str, 
                                 target_filename: str = "medical_data.parquet",
                                 logger=None) -> bool:
    """Copy and rename a single parquet file from silver to gold partition.
    
    Args:
        silver_partition: Source S3 partition path
        gold_partition: Destination S3 partition path
        target_filename: Target filename (default: medical_data.parquet)
        logger: Optional logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if not (silver_partition.startswith("s3://") and gold_partition.startswith("s3://")):
        return False
        
    try:
        s3 = boto3.client("s3")
        bucket_silver, key_silver_prefix = s3_path_to_bucket_key(silver_partition)
        bucket_gold, key_gold_prefix = s3_path_to_bucket_key(gold_partition)
        
        # Find parquet files in silver partition
        resp = s3.list_objects_v2(Bucket=bucket_silver, Prefix=key_silver_prefix)
        parquet_files = [obj['Key'] for obj in resp.get('Contents', []) if obj['Key'].endswith('.parquet')]
        
        if len(parquet_files) == 1:
            src_key = parquet_files[0]
            dst_key = f"{key_gold_prefix}/{target_filename}"
            
            # Copy file to gold with new name
            s3.copy_object(
                Bucket=bucket_gold,
                CopySource={'Bucket': bucket_silver, 'Key': src_key},
                Key=dst_key
            )
            if logger:
                logger.info(f"Copied {src_key} to {dst_key} in {bucket_gold}")
            
            # Delete the silver file for cleanup
            try:
                s3.delete_object(Bucket=bucket_silver, Key=src_key)
                if logger:
                    logger.info(f"Deleted silver file {src_key} from {bucket_silver}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not delete silver file {src_key}: {e}")
            
            return True
        else:
            if logger:
                logger.warning(f"Expected 1 parquet file in {silver_partition}, found {len(parquet_files)}")
            return False
            
    except Exception as e:
        if logger:
            logger.warning(f"S3 copy/rename failed: {e}")
        return False


# ===== Convenience Writers for "latest" frequency datasets =====
def write_parquet_and_csv_latest(
    df: pd.DataFrame,
    s3_parquet_path: str,
    s3_csv_path: Optional[str] = None,
) -> None:
    """Write a DataFrame to S3 as Parquet (and optional CSV) using DuckDB COPY.

    Uses OVERWRITE_OR_IGNORE TRUE for idempotent writes to stable "latest" keys.
    """
    con = duckdb.connect(database=':memory:')
    try:
        con.sql("INSTALL httpfs; LOAD httpfs;")
        con.sql("INSTALL aws; LOAD aws;")
        con.sql("CALL load_aws_credentials();")
        con.sql("SET s3_region='us-east-1'")
        con.sql("SET s3_url_style='path'")
        con.register('df', df)
        con.sql(f"COPY df TO '{s3_parquet_path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)")
        if s3_csv_path:
            con.sql(f"COPY df TO '{s3_csv_path}' (FORMAT CSV, HEADER TRUE, OVERWRITE_OR_IGNORE TRUE)")
    finally:
        con.close()


def write_drug_frequency_latest(df: pd.DataFrame) -> None:
    write_parquet_and_csv_latest(
        df,
        s3_parquet_path='s3://pgxdatalake/gold/drug_name/drug_frequency_latest.parquet',
        s3_csv_path='s3://pgxdatalake/gold/drug_name/drug_frequency_latest.csv',
    )


def write_drug_pairs_latest(pairs_df: pd.DataFrame) -> None:
    write_parquet_and_csv_latest(
        pairs_df,
        s3_parquet_path='s3://pgxdatalake/gold/drug_name/drug_pairs_latest.parquet',
        s3_csv_path=None,
    )


def write_target_code_latest(df: pd.DataFrame) -> None:
    write_parquet_and_csv_latest(
        df,
        s3_parquet_path='s3://pgxdatalake/gold/target_code/target_code_latest.parquet',
        s3_csv_path='s3://pgxdatalake/gold/target_code/target_code_latest.csv',
    )


# ============================================================================
# Feature Importance S3 Utilities
# ============================================================================

def check_feature_importance_results_exist(cohort_name: str, age_band: str, event_year: int) -> bool:
    """
    Check if feature importance results already exist in S3 (idempotency check)
    
    Args:
        cohort_name: Cohort name
        age_band: Age band
        event_year: Event year (test year)
        
    Returns:
        True if results exist, False otherwise
    """
    s3_base = "s3://pgxdatalake/gold/feature_importance"
    s3_key = f"cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}/{cohort_name}_{age_band}_{event_year}_feature_importance_aggregated.csv"
    
    try:
        bucket = S3_BUCKET
        
        # Extract key from s3:// path
        if s3_key.startswith('s3://'):
            s3_key = s3_key.replace(f's3://{bucket}/', '')
        
        # Check if object exists
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except Exception:
        # File doesn't exist or error occurred
        return False


def check_cohort_file_exists(cohort_name: str, age_band: str, event_year: int) -> bool:
    """
    Check if cohort parquet file exists locally
    
    Args:
        cohort_name: Cohort name
        age_band: Age band
        event_year: Event year
        
    Returns:
        True if file exists, False otherwise
    """
    # Try environment variable first
    local_data_path = os.getenv("LOCAL_DATA_PATH")
    
    # If not set, try common locations
    if not local_data_path:
        # Check Windows path first (for local development)
        windows_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cohorts_F1120")
        if os.path.exists(windows_path):
            local_data_path = windows_path
        else:
            # Fall back to EC2 path
            local_data_path = "/mnt/nvme/cohorts"
    
    parquet_file = os.path.join(
        local_data_path,
        f"cohort_name={cohort_name}",
        f"event_year={event_year}",
        f"age_band={age_band}",
        "cohort.parquet"
    )
    
    return os.path.exists(parquet_file)

