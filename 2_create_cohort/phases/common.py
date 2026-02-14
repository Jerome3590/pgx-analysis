"""
Common imports and utilities for all pipeline phases.
"""

import os
import sys
import platform
import json
import re
from datetime import datetime

# Windows emoji compatibility
IS_WINDOWS = platform.system() == 'Windows'
SYMBOLS = {
    'arrow': '->' if IS_WINDOWS else '→',
    'success': '[PASS]' if IS_WINDOWS else '✅',
    'fail': '[FAIL]' if IS_WINDOWS else '❌',
    'info': '[INFO]' if IS_WINDOWS else '📊',
    'check': '[CHECK]' if IS_WINDOWS else '🔍'
}

# Set root of project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if project_root not in sys.path:
    sys.path.append(project_root)

# Import constants (py_helpers)
from py_helpers.constants import OPIOID_ICD_CODES, get_opioid_icd_sql_condition, ALL_ICD_DIAGNOSIS_COLUMNS, S3_BUCKET
from py_helpers.env_utils import get_data_root, is_linux
from pathlib import Path
import subprocess
import shutil

# DuckDB temp file cleanup
def cleanup_duckdb_temp_files(logger):
    """
    Clean up DuckDB temporary files and old worker temp directories.
    Called at startup and after successful cohort completion.
    """
    try:
        from py_helpers.duckdb_utils import cleanup_old_duckdb_temp_dirs
        
        # Clean up old temp directories (older than 1 hour)
        # This handles leftover directories from previous runs or crashes
        cleaned_count = cleanup_old_duckdb_temp_dirs(max_age_hours=1)
        if cleaned_count > 0:
            logger.info(f"→ [CLEANUP] Cleaned up {cleaned_count} old DuckDB temp directories")
        else:
            logger.debug("→ [CLEANUP] No old DuckDB temp directories to clean")
    except Exception as e:
        logger.warning(f"→ [CLEANUP] Could not clean DuckDB temp files: {e}")

def enable_query_profiling(conn, logger, profile_format="json", output_path="/tmp/duckdb_profiling.json"):
    """
    Enable query profiling (currently a no-op shim).
    
    WARNING: Profiling is currently disabled. If you need query profiling,
    implement DuckDB profiling hooks or use EXPLAIN ANALYZE directly.
    """
    try:
        logger.warning(f"[PROFILING] enable_query_profiling() is currently a no-op. Profiling is disabled.")
        logger.debug(f"[shim] enable_query_profiling({profile_format}, {output_path}): no-op in simplified helpers")
    except Exception:
        pass

def disable_query_profiling(conn, logger):
    try:
        logger.debug("[shim] disable_query_profiling: no-op in simplified helpers")
    except Exception:
        pass

def force_checkpoint(conn, logger):
    try:
        logger.debug("[shim] force_checkpoint: no-op in simplified helpers")
    except Exception:
        pass

def monitor_disk_space(logger):
    try:
        logger.debug("[shim] monitor_disk_space: no-op in simplified helpers")
    except Exception:
        pass


_SCHEMA_CACHE = None

def _load_schemas():
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mappings_dir = os.path.join(base_dir, "table_mappings")
    medical_path = os.path.join(mappings_dir, "medical_schema.json")
    pharmacy_path = os.path.join(mappings_dir, "pharmacy_schema.json")
    medical_cols = set()
    pharmacy_cols = set()
    try:
        with open(medical_path, "r", encoding="utf-8") as f:
            for c in json.load(f):
                name = c.get("Name")
                if name:
                    medical_cols.add(name)
    except Exception:
        pass
    try:
        with open(pharmacy_path, "r", encoding="utf-8") as f:
            for c in json.load(f):
                name = c.get("Name")
                if name:
                    pharmacy_cols.add(name)
    except Exception:
        pass
    _SCHEMA_CACHE = {
        "medical": medical_cols,
        "pharmacy": pharmacy_cols,
        "paths": {
            "medical": medical_path,
            "pharmacy": pharmacy_path,
        }
    }
    return _SCHEMA_CACHE


def execute_sql_with_dev_validation(conn, logger, sql):
    """Execute SQL; on error, optionally emit schema hints if PGX_DEV_VALIDATION=1."""
    try:
        return conn.sql(sql)
    except Exception as e:
        import os
        if os.getenv("PGX_DEV_VALIDATION", "0") == "1":
            # Gate token scanning for large SQL strings (performance optimization)
            # Only scan if SQL is reasonably sized (< 50KB)
            if len(sql) < 50_000:
                schemas = _load_schemas()
                tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql))
                sql_keywords = {
                    "select","from","where","and","or","case","when","then","else","end","as","create","replace","view","union","all","left","join","on","order","by","limit","group","distinct","over","partition","null","not","is","between","count","row_number","coalesce","random"
                }
                allowed_extra = {
                    "age_imputed","gender_imputed","race_imputed","zip_imputed","county_imputed","payer_imputed",
                    "drug_name","therapeutic_class_1","primary_icd_diagnosis_code","event_date","mi_person_key",
                    "cohort_name","is_target_case","event_type","data_source","event_classification","event_sequence",
                    # Common table functions / tokens that are not schema fields
                    "read_parquet","read_json","httpfs","aws","s3","gold","silver","parquet","pgxdatalake"
                }
                medical_cols = schemas.get("medical", set())
                pharmacy_cols = schemas.get("pharmacy", set())
                allowed = {t.lower() for t in (medical_cols | pharmacy_cols | allowed_extra)}
                unknown = sorted({t for t in tokens if t.lower() not in sql_keywords and t.lower() not in allowed})
                if unknown:
                    logger.warning(f"[DEV VALIDATION] Unrecognized identifiers possibly not in schemas: {unknown[:20]}")
                logger.warning(f"[DEV VALIDATION] Refer to schemas for expected fields: medical={schemas['paths']['medical']}, pharmacy={schemas['paths']['pharmacy']}")
            else:
                logger.debug(f"[DEV VALIDATION] SQL string too large ({len(sql):,} chars), skipping token scan")
        raise


def _parquet_from_sql(paths):
    """Build SQL FROM clause: single path or UNION ALL of two (for 85-114 = 85-94 + 95-114)."""
    def esc(s):
        return s.replace("'", "''")
    if len(paths) == 1:
        return f"read_parquet('{esc(paths[0])}')"
    return f"(SELECT * FROM read_parquet('{esc(paths[0])}') UNION ALL SELECT * FROM read_parquet('{esc(paths[1])}'))"


def ensure_gold_views(conn, logger, age_band: str, event_year: int):
    """Ensure gold-backed views `medical` and `pharmacy` exist for this session.

    This allows later phases to run even if Phase 1 was skipped due to checkpoints.
    For age_band=85-114, uses 85-94 and 95-114 partitions as one when single 85-114 is not present.
    """
    # get_gold_data_paths is defined later in this module; resolve paths (1 or 2 for 85-114)
    medical_paths = get_gold_data_paths("medical", age_band, event_year)
    pharmacy_paths = get_gold_data_paths("pharmacy", age_band, event_year)
    if not medical_paths or not pharmacy_paths:
        raise FileNotFoundError(f"Gold data not found for age_band={age_band}, event_year={event_year}")
    if not medical_paths[0].startswith("s3://"):
        logger.info(f"[ensure_gold_views] Using local medical path(s): {medical_paths}")
    if not pharmacy_paths[0].startswith("s3://"):
        logger.info(f"[ensure_gold_views] Using local pharmacy path(s): {pharmacy_paths}")

    medical_from = _parquet_from_sql(medical_paths)
    pharmacy_from = _parquet_from_sql(pharmacy_paths)

    # Ensure `medical` view
    try:
        conn.sql("SELECT 1 FROM medical LIMIT 1").fetchone()
    except Exception:
        medical_sql = f"""
        CREATE OR REPLACE VIEW medical_base AS
        SELECT
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            member_age_dos AS age_imputed,
            member_gender AS gender_imputed,
            member_race AS race_imputed,
            member_zip_code_dos AS zip_imputed,
            member_county_dos AS county_imputed,
            payer_type AS payer_imputed,
            -- ALL ICD diagnosis codes (for ML feature discovery) - CAST to VARCHAR
            CAST(primary_icd_diagnosis_code AS VARCHAR) AS primary_icd_diagnosis_code,
            CAST(two_icd_diagnosis_code AS VARCHAR) AS two_icd_diagnosis_code,
            CAST(three_icd_diagnosis_code AS VARCHAR) AS three_icd_diagnosis_code,
            CAST(four_icd_diagnosis_code AS VARCHAR) AS four_icd_diagnosis_code,
            CAST(five_icd_diagnosis_code AS VARCHAR) AS five_icd_diagnosis_code,
            CAST(six_icd_diagnosis_code AS VARCHAR) AS six_icd_diagnosis_code,
            CAST(seven_icd_diagnosis_code AS VARCHAR) AS seven_icd_diagnosis_code,
            CAST(eight_icd_diagnosis_code AS VARCHAR) AS eight_icd_diagnosis_code,
            CAST(nine_icd_diagnosis_code AS VARCHAR) AS nine_icd_diagnosis_code,
            CAST(ten_icd_diagnosis_code AS VARCHAR) AS ten_icd_diagnosis_code,
            -- ALL ICD procedure codes (for ML feature discovery) - CAST to VARCHAR
            CAST(two_icd_procedure_code AS VARCHAR) AS two_icd_procedure_code,
            CAST(three_icd_procedure_code AS VARCHAR) AS three_icd_procedure_code,
            CAST(four_icd_procedure_code AS VARCHAR) AS four_icd_procedure_code,
            CAST(five_icd_procedure_code AS VARCHAR) AS five_icd_procedure_code,
            CAST(six_icd_procedure_code AS VARCHAR) AS six_icd_procedure_code,
            CAST(seven_icd_procedure_code AS VARCHAR) AS seven_icd_procedure_code,
            CAST(eight_icd_procedure_code AS VARCHAR) AS eight_icd_procedure_code,
            CAST(nine_icd_procedure_code AS VARCHAR) AS nine_icd_procedure_code,
            CAST(ten_icd_procedure_code AS VARCHAR) AS ten_icd_procedure_code,
            -- Include CPT/procedure fields for event features
            procedure_code,
            cpt_mod_1_code,
            cpt_mod_2_code,
            -- HCG fields for ED visit identification
            hcg_setting,
            hcg_line,
            hcg_detail,
            event_date,
            CAST(event_year AS INTEGER) AS event_year
        FROM {medical_from}
        WHERE mi_person_key IS NOT NULL
          AND CAST(mi_person_key AS VARCHAR) <> ''
          AND event_date IS NOT NULL;
        """
        execute_sql_with_dev_validation(conn, logger, medical_sql)

        medical_filtered = f"""
        CREATE OR REPLACE VIEW medical AS
        SELECT *
        FROM medical_base
        WHERE age_imputed IS NOT NULL
          AND age_imputed BETWEEN 1 AND 114
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31';
        """
        execute_sql_with_dev_validation(conn, logger, medical_filtered)
        logger.info("[ensure_gold_views] Created views: medical_base, medical")

    # Ensure `pharmacy` view
    try:
        conn.sql("SELECT 1 FROM pharmacy LIMIT 1").fetchone()
    except Exception:
        pharmacy_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_base AS
        SELECT 
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            NULL::INTEGER AS age_imputed,
            NULL::VARCHAR AS gender_imputed,
            NULL::VARCHAR AS race_imputed,
            NULL::VARCHAR AS zip_imputed,
            NULL::VARCHAR AS county_imputed,
            NULL::VARCHAR AS payer_imputed,
            drug_name,
            NULL::VARCHAR AS therapeutic_class_1,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') AS event_date,
            CAST(event_year AS INTEGER) AS event_year
        FROM {pharmacy_from}
        WHERE mi_person_key IS NOT NULL
          AND CAST(mi_person_key AS VARCHAR) <> ''
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL;
        """
        execute_sql_with_dev_validation(conn, logger, pharmacy_sql)

        pharmacy_filtered = f"""
        CREATE OR REPLACE VIEW pharmacy AS
        SELECT *
        FROM pharmacy_base
        WHERE event_date IS NOT NULL
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31'
          AND drug_name IS NOT NULL
          AND drug_name <> '';
        """
        execute_sql_with_dev_validation(conn, logger, pharmacy_filtered)
        logger.info("[ensure_gold_views] Created views: pharmacy_base, pharmacy")


def sync_gold_data_to_local(dataset: str, age_band: str, event_year: int, logger) -> bool:
    """
    Sync gold medical/pharmacy data from S3 to local /mnt/nvme if not already present.
    
    IMPORTANT: Only syncs medical and pharmacy data, NOT cohort datasets (cohorts are recreated).
    
    Args:
        dataset: 'medical' or 'pharmacy' (only these two are supported)
        age_band: Age band string (e.g., '13-24')
        event_year: Event year (e.g., 2016)
        logger: Logger instance
    
    Returns:
        True if sync succeeded or data already exists, False otherwise
    """
    # Only allow medical and pharmacy datasets
    if dataset not in ['medical', 'pharmacy']:
        logger.error(f"[SYNC] Invalid dataset '{dataset}'. Only 'medical' and 'pharmacy' are supported.")
        return False
    
    if not is_linux():
        # Only sync on Linux/EC2
        return True
    
    data_root = get_data_root()
    local_path = data_root / "gold" / dataset / f"age_band={age_band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
    
    # Check if file already exists locally and is not corrupted (size > 0)
    if local_path.exists():
        file_size = local_path.stat().st_size
        if file_size > 0:
            logger.info(f"[SYNC] Local {dataset} data already exists: {local_path} ({file_size:,} bytes)")
            return True
        else:
            logger.warning(f"[SYNC] Local {dataset} file exists but is empty (0 bytes), will re-sync: {local_path}")
            # Remove corrupted file
            try:
                local_path.unlink()
            except Exception as e:
                logger.warning(f"[SYNC] Could not remove corrupted file: {e}")
    
    # Check if AWS CLI is available (use full path on EC2)
    aws_cli = "/usr/local/bin/aws"
    if not Path(aws_cli).exists():
        # Fallback to PATH lookup if full path doesn't exist
        aws_cli = shutil.which("aws")
        if not aws_cli:
            logger.warning(f"[SYNC] AWS CLI not found, skipping sync for {dataset}")
            return False
    
    # S3 source path (only medical/pharmacy, NOT cohorts)
    s3_path = f"s3://{S3_BUCKET}/gold/{dataset}/age_band={age_band}/event_year={event_year}/"
    local_dir = local_path.parent
    
    # Create local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[SYNC] Syncing {dataset} data from S3 to local...")
    logger.info(f"  Source: {s3_path}")
    logger.info(f"  Dest: {local_dir}")
    logger.info(f"  Note: Only syncing {dataset} data (cohorts are recreated, not synced)")
    
    try:
        # Use aws s3 sync to sync the directory (will only download missing files)
        result = subprocess.run(
            [aws_cli, "s3", "sync", s3_path, str(local_dir), "--no-progress"],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=False
        )
        
        if result.returncode == 0:
            if local_path.exists():
                file_size = local_path.stat().st_size
                if file_size > 0:
                    logger.info(f"[SYNC] ✓ Successfully synced {dataset} data to: {local_path} ({file_size:,} bytes)")
                    return True
                else:
                    logger.warning(f"[SYNC] Synced file exists but is empty (0 bytes), may be corrupted: {local_path}")
                    return False
            else:
                logger.warning(f"[SYNC] Sync completed but file not found: {local_path}")
                return False
        else:
            logger.warning(f"[SYNC] Sync failed for {dataset}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"[SYNC] Sync timeout for {dataset} (exceeded 1 hour)")
        return False
    except Exception as e:
        logger.error(f"[SYNC] Error syncing {dataset}: {e}")
        return False


def resolve_gold_data_path(dataset: str, age_band: str, event_year: int) -> str:
    """
    Resolve path to gold medical/pharmacy data. Prefers filtered gold (from Step 1b --before-cohorts)
    when present, then local raw gold, then S3.
    
    Priority:
    1. Filtered (Step 1b before cohorts): gold/{dataset}_filtered/age_band=.../event_year=.../{dataset}_data.parquet
    2. Local raw: /mnt/nvme/gold/{dataset}/age_band=.../event_year=.../{dataset}_data.parquet
    3. S3 filtered: s3://.../gold/{dataset}_filtered/...
    4. S3 raw: s3://.../gold/{dataset}/...
    
    Args:
        dataset: 'medical' or 'pharmacy'
        age_band: Age band string (e.g., '13-24')
        event_year: Event year (e.g., 2016)
    
    Returns:
        Path string (local if exists, otherwise S3)
    """
    data_root = get_data_root()
    filtered_subdir = f"{dataset}_filtered"
    # 1) Local filtered (Step 1b run after Step 1a, before Step 2)
    if is_linux():
        local_filtered = data_root / "gold" / filtered_subdir / f"age_band={age_band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
        if local_filtered.exists():
            return str(local_filtered)
    project_data = Path(project_root) / "data" / "gold" / filtered_subdir / f"age_band={age_band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
    if project_data.exists():
        return str(project_data)
    # 2) Local raw
    if is_linux():
        local_path = data_root / "gold" / dataset / f"age_band={age_band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
        if local_path.exists():
            return str(local_path)
    local_alt = Path(project_root) / "data" / "gold" / dataset / f"age_band={age_band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
    if local_alt.exists():
        return str(local_alt)
    # 3) S3 filtered then S3 raw
    try:
        from py_helpers.common_imports import s3_client, S3_BUCKET
        key_filtered = f"gold/{filtered_subdir}/age_band={age_band}/event_year={event_year}/{dataset}_data.parquet"
        s3_client.head_object(Bucket=S3_BUCKET, Key=key_filtered)
        return f"s3://{S3_BUCKET}/{key_filtered}"
    except Exception:
        pass
    return f"s3://{S3_BUCKET}/gold/{dataset}/age_band={age_band}/event_year={event_year}/{dataset}_data.parquet"


def _check_gold_data_available_for_band(dataset: str, band: str, event_year: int) -> bool:
    """Return True if gold parquet exists for this single age_band (e.g. 85-94 or 95-114)."""
    data_root = get_data_root()
    filtered_subdir = f"{dataset}_filtered"
    if is_linux():
        p = data_root / "gold" / filtered_subdir / f"age_band={band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
        if p.exists() and p.stat().st_size > 0:
            return True
    p = Path(project_root) / "data" / "gold" / filtered_subdir / f"age_band={band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
    if p.exists() and p.stat().st_size > 0:
        return True
    if is_linux():
        p = data_root / "gold" / dataset / f"age_band={band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
        if p.exists() and p.stat().st_size > 0:
            return True
    p = Path(project_root) / "data" / "gold" / dataset / f"age_band={band}" / f"event_year={event_year}" / f"{dataset}_data.parquet"
    if p.exists() and p.stat().st_size > 0:
        return True
    try:
        from py_helpers.common_imports import s3_client, S3_BUCKET
        for key in [
            f"gold/{filtered_subdir}/age_band={band}/event_year={event_year}/{dataset}_data.parquet",
            f"gold/{dataset}/age_band={band}/event_year={event_year}/{dataset}_data.parquet",
        ]:
            s3_client.head_object(Bucket=S3_BUCKET, Key=key)
            return True
    except Exception:
        pass
    return False


def check_gold_data_available(dataset: str, age_band: str, event_year: int) -> bool:
    """
    Return True if the gold parquet file(s) exist. For age_band=85-114, returns True if either
    a single 85-114 partition exists or both 85-94 and 95-114 partitions exist (they are treated as one).
    """
    if age_band != "85-114":
        return _check_gold_data_available_for_band(dataset, age_band, event_year)
    # 85-114: accept single partition or both 85-94 and 95-114
    if _check_gold_data_available_for_band(dataset, "85-114", event_year):
        return True
    if _check_gold_data_available_for_band(dataset, "85-94", event_year) and _check_gold_data_available_for_band(dataset, "95-114", event_year):
        return True
    return False


def get_gold_data_paths(dataset: str, age_band: str, event_year: int):
    """
    Return a list of gold parquet paths (1 or 2) for the given dataset/age_band/year.
    For age_band=85-114, if no single 85-114 partition exists, returns [path_85_94, path_95_114]
    so the caller can UNION ALL the two partitions as one.
    """
    if age_band != "85-114":
        return [resolve_gold_data_path(dataset, age_band, event_year)]
    if _check_gold_data_available_for_band(dataset, "85-114", event_year):
        return [resolve_gold_data_path(dataset, "85-114", event_year)]
    if _check_gold_data_available_for_band(dataset, "85-94", event_year) and _check_gold_data_available_for_band(dataset, "95-114", event_year):
        return [
            resolve_gold_data_path(dataset, "85-94", event_year),
            resolve_gold_data_path(dataset, "95-114", event_year),
        ]
    return []


def get_dynamic_targeting_config():
    """
    Centralize environment variable parsing for dynamic targeting configuration.
    
    Returns:
        dict with keys: target_icd_codes, target_cpt_codes, target_icd_prefixes, target_cpt_prefixes
    """
    return {
        "target_icd_codes": [c.strip() for c in os.getenv("PGX_TARGET_ICD_CODES", "").split(',') if c.strip()],
        "target_cpt_codes": [c.strip() for c in os.getenv("PGX_TARGET_CPT_CODES", "").split(',') if c.strip()],
        "target_icd_prefixes": [p.strip() for p in os.getenv("PGX_TARGET_ICD_PREFIXES", "").split(',') if p.strip()],
        "target_cpt_prefixes": [p.strip() for p in os.getenv("PGX_TARGET_CPT_PREFIXES", "").split(',') if p.strip()]
    }


def ensure_unified_views(conn, logger):
    """Ensure unified views created by Phase 2 exist: unified_event_fact_table, unified_drug_exposure."""
    # unified_event_fact_table
    try:
        conn.sql("SELECT 1 FROM unified_event_fact_table LIMIT 1").fetchone()
    except Exception:
        # Build dynamic classification from env (mirror Phase 2)
        # Use centralized config helper to reduce drift across phases
        config = get_dynamic_targeting_config()
        target_icd_codes = config["target_icd_codes"]
        target_cpt_codes = config["target_cpt_codes"]
        target_icd_prefixes = config["target_icd_prefixes"]
        target_cpt_prefixes = config["target_cpt_prefixes"]

        icd_conditions = []
        if target_icd_codes:
            # Exact match (codes are normalized to F1120 format in gold tier)
            icd_conditions.append(f"primary_icd_diagnosis_code IN {tuple(target_icd_codes)}")
        for pref in target_icd_prefixes:
            # Normalize prefix and use LIKE with ESCAPE for wildcard safe match
            # CRITICAL: This normalization must match get_opioid_icd_sql_condition() logic
            # Both use: UPPER, remove '.', remove ' ' (spaces)
            # get_opioid_icd_sql_condition() checks codes already normalized in gold tier (F1120 format)
            # This prefix matching also normalizes to match gold tier format
            norm_pref = pref.upper().replace('.', '').replace(' ', '')
            like = norm_pref if ('%' in norm_pref or '_' in norm_pref) else (norm_pref + '%')
            icd_conditions.append(
                f"REPLACE(REPLACE(UPPER(primary_icd_diagnosis_code), '.', ''), ' ', '') LIKE '{like}'"
            )

        cpt_conditions = []
        if target_cpt_codes:
            tup = tuple(target_cpt_codes)
            cpt_conditions.append(f"procedure_code IN {tup} OR cpt_mod_1_code IN {tup} OR cpt_mod_2_code IN {tup}")
        for pref in target_cpt_prefixes:
            like = pref if ('%' in pref or '_' in pref) else (pref + '%')
            cpt_conditions.append(
                f"procedure_code LIKE '{like}' OR cpt_mod_1_code LIKE '{like}' OR cpt_mod_2_code LIKE '{like}'"
            )

        # HCG-based ED visit identification (for ED_NON_OPIOID cohort)
        # ED visits are identified by HCG line codes and details for precision
        # Use hcg_detail to distinguish actual ED visits from observation care
        # P51a = Observation Care (exclude), P51b = ED Visits (include)
        # O11 = Emergency Department (include)
        # P33 = Urgent Care Visits (include)
        ed_hcg_condition = """
            (hcg_line = 'P51 - ER Visits and Observation Care' AND hcg_detail = 'P51b - PHY ED Visits and Observation Care - ED Visits')
            OR hcg_line = 'O11 - Emergency Room'
            OR hcg_line = 'P33 - Urgent Care Visits'
        """
        
        # Default classification falls back to opioid_ed vs ed_non_opioid
        # Priority: 1) Opioid ICD codes (ANY position) → opioid_ed, 2) HCG ED visits → ed_non_opioid, 3) Other → ed_non_opioid
        # CRITICAL: Check ALL 10 ICD diagnosis columns for opioid codes
        opioid_icd_condition = get_opioid_icd_sql_condition()
        default_case = f"""
            CASE 
                WHEN {opioid_icd_condition} THEN 'opioid_ed'
                WHEN {ed_hcg_condition} THEN 'ed_non_opioid'
                ELSE 'ed_non_opioid'
            END
        """
        
        # If any env targets are provided, build a generic target/non_target classification
        # Priority: 1) Target ICD/CPT codes → target, 2) HCG ED visits → ed_non_opioid, 3) Other → non_target
        if icd_conditions or cpt_conditions:
            where_clause = " OR ".join(filter(None, icd_conditions + cpt_conditions)) or "1=0"
            classification_sql = f"""
                CASE 
                    WHEN ({where_clause}) THEN 'target'
                    WHEN {ed_hcg_condition} THEN 'ed_non_opioid'
                    ELSE 'non_target'
                END
            """
        else:
            classification_sql = default_case

        # CRITICAL FIX: Compute ROW_NUMBER() after UNION ALL to ensure global chronological ordering
        # Previously, ROW_NUMBER() was computed separately for medical and pharmacy, breaking global sequence
        event_fact_sql = f"""
        CREATE OR REPLACE VIEW unified_event_fact_table AS
        WITH unified_events AS (
            SELECT 
                mi_person_key,
                event_date,
                'medical' as event_type,
                'medical' as data_source,
                age_imputed,
                gender_imputed as member_gender,
                race_imputed as member_race,
                zip_imputed,
                county_imputed,
                payer_imputed,
                -- ALL ICD diagnosis codes (for ML feature discovery)
                primary_icd_diagnosis_code,
                two_icd_diagnosis_code,
                three_icd_diagnosis_code,
                four_icd_diagnosis_code,
                five_icd_diagnosis_code,
                six_icd_diagnosis_code,
                seven_icd_diagnosis_code,
                eight_icd_diagnosis_code,
                nine_icd_diagnosis_code,
                ten_icd_diagnosis_code,
                -- ALL ICD procedure codes (for ML feature discovery)
                two_icd_procedure_code,
                three_icd_procedure_code,
                four_icd_procedure_code,
                five_icd_procedure_code,
                six_icd_procedure_code,
                seven_icd_procedure_code,
                eight_icd_procedure_code,
                nine_icd_procedure_code,
                ten_icd_procedure_code,
                NULL as drug_name,
                NULL as therapeutic_class_1,
                -- CPT/procedure codes (medical)
                procedure_code,
                cpt_mod_1_code,
                cpt_mod_2_code,
                -- HCG fields for ED visit identification
                hcg_setting,
                hcg_line,
                hcg_detail,
                {classification_sql} as event_classification
            FROM medical
            WHERE primary_icd_diagnosis_code IS NOT NULL
            
            UNION ALL
            
            SELECT 
                mi_person_key,
                event_date,
                'pharmacy' as event_type,
                'pharmacy' as data_source,
                age_imputed,
                gender_imputed as member_gender,
                race_imputed as member_race,
                zip_imputed,
                county_imputed,
                payer_imputed,
                -- ICD diagnosis codes not present in pharmacy (set NULLs)
                NULL as primary_icd_diagnosis_code,
                NULL as two_icd_diagnosis_code,
                NULL as three_icd_diagnosis_code,
                NULL as four_icd_diagnosis_code,
                NULL as five_icd_diagnosis_code,
                NULL as six_icd_diagnosis_code,
                NULL as seven_icd_diagnosis_code,
                NULL as eight_icd_diagnosis_code,
                NULL as nine_icd_diagnosis_code,
                NULL as ten_icd_diagnosis_code,
                -- ICD procedure codes not present in pharmacy (set NULLs)
                NULL as two_icd_procedure_code,
                NULL as three_icd_procedure_code,
                NULL as four_icd_procedure_code,
                NULL as five_icd_procedure_code,
                NULL as six_icd_procedure_code,
                NULL as seven_icd_procedure_code,
                NULL as eight_icd_procedure_code,
                NULL as nine_icd_procedure_code,
                NULL as ten_icd_procedure_code,
                drug_name,
                therapeutic_class_1,
                -- CPT/procedure codes not present in pharmacy (set NULLs)
                NULL as procedure_code,
                NULL as cpt_mod_1_code,
                NULL as cpt_mod_2_code,
                -- HCG fields not present in pharmacy (set NULLs)
                NULL as hcg_setting,
                NULL as hcg_line,
                NULL as hcg_detail,
                {classification_sql} as event_classification
            FROM pharmacy
            WHERE drug_name IS NOT NULL
        )
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM unified_events;
        """
        execute_sql_with_dev_validation(conn, logger, event_fact_sql)
        logger.info("[ensure_unified_views] Created view: unified_event_fact_table")

    # unified_drug_exposure
    try:
        conn.sql("SELECT 1 FROM unified_drug_exposure LIMIT 1").fetchone()
    except Exception:
        drug_sql = """
        CREATE OR REPLACE VIEW unified_drug_exposure AS
        SELECT 
            mi_person_key,
            event_date,
            drug_name,
            therapeutic_class_1,
            age_imputed,
            gender_imputed as member_gender,
            race_imputed as member_race,
            zip_imputed,
            county_imputed,
            payer_imputed,
            NULL as days_to_target_event
        FROM pharmacy
        WHERE drug_name IS NOT NULL AND drug_name <> '';
        """
        execute_sql_with_dev_validation(conn, logger, drug_sql)
        logger.info("[ensure_unified_views] Created view: unified_drug_exposure")


def ensure_cohort_views(conn, logger):
    """Ensure Phase 3 cohort views exist: opioid_ed_cohort and ed_non_opioid_cohort.
    
    Uses dynamic classification labels matching Phase 3 logic (target/non_target vs opioid_ed/ed_non_opioid).
    
    WARNING: This is a FALLBACK implementation. If Phase 3 was skipped, these simplified cohorts
    do NOT match Phase 3 semantics:
    - No time windows for polypharmacy cohort
    - No drug lookbacks
    - No multi-window targets (7d, 14d, 21d, 30d, 45d)
    - No proper control ratio validation
    
    For production use, always run Phase 3 to get full cohort semantics.
    """
    import os
    logger.warning(
        "[ensure_cohort_views] Using fallback cohort logic (Phase 3 skipped). "
        "Cohort semantics differ from full pipeline. "
        "Missing: time windows, drug lookbacks, multi-window targets, proper control ratios."
    )
    # Determine classification labels based on dynamic targeting env (same logic as Phase 3)
    target_icd = os.getenv("PGX_TARGET_ICD_CODES", "").strip() or os.getenv("PGX_TARGET_ICD_PREFIXES", "").strip()
    target_cpt = os.getenv("PGX_TARGET_CPT_CODES", "").strip() or os.getenv("PGX_TARGET_CPT_PREFIXES", "").strip()
    dynamic_targeting = bool(target_icd or target_cpt)
    label_target = 'target' if dynamic_targeting else 'opioid_ed'
    # ED_NON_OPIOID always uses 'ed_non_opioid' because HCG ED visits are always classified as 'ed_non_opioid'
    # regardless of dynamic targeting (see Phase 2 classification logic)
    label_ed_non_opioid = 'ed_non_opioid'
    
    # opioid_ed_cohort
    try:
        conn.sql("SELECT 1 FROM opioid_ed_cohort LIMIT 1").fetchone()
    except Exception:
        opioid_ed_cohort_sql = f"""
        CREATE OR REPLACE VIEW opioid_ed_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification = '{label_target}'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != '{label_target}'
              AND NOT EXISTS (
                  SELECT 1
                  FROM target_cases tc
                  WHERE tc.mi_person_key = unified_event_fact_table.mi_person_key
              )
        ),
        sampled_controls AS (
            -- Hash-based deterministic sampling (replaces ORDER BY RANDOM() for performance)
            WITH target_count AS (
                SELECT COUNT(*) as target_cnt FROM target_cases
            ),
            needed_count AS (
                SELECT tc.target_cnt * 5 as needed FROM target_count tc
            ),
            available_controls AS (
                SELECT COUNT(*) as available FROM control_candidates
            ),
            sample_threshold AS (
                SELECT 
                    CAST(ROUND((SELECT needed FROM needed_count)::DOUBLE / GREATEST((SELECT available FROM available_controls), 1) * 10000) AS BIGINT) as threshold
            )
            SELECT 
                mi_person_key
            FROM control_candidates
            WHERE ABS(hash(mi_person_key)) % 10000 < (SELECT threshold FROM sample_threshold)
            LIMIT (
                SELECT LEAST(
                    (SELECT needed FROM needed_count),
                    (SELECT available FROM available_controls)
                )
            )
        )
        SELECT 
            uef.*,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as target,
            'OPIOID_ED' as cohort_name,
            CASE 
                WHEN tc.mi_person_key IS NOT NULL THEN 'OPIOID_ED'
                ELSE 'NON_ED'
            END as cohort,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case
        FROM unified_event_fact_table uef
        LEFT JOIN target_cases tc ON uef.mi_person_key = tc.mi_person_key
        LEFT JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
        WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
        """
        execute_sql_with_dev_validation(conn, logger, opioid_ed_cohort_sql)
        logger.info(f"[ensure_cohort_views] Created view: opioid_ed_cohort (using classification='{label_target}')")

    # ed_non_opioid_cohort
    try:
        conn.sql("SELECT 1 FROM ed_non_opioid_cohort LIMIT 1").fetchone()
    except Exception:
        # Exclude patients with opioid ICD codes from ED_NON_OPIOID target cases
        # CRITICAL: Check ALL 10 ICD diagnosis columns for opioid codes
        opioid_icd_condition = get_opioid_icd_sql_condition()
        ed_non_opioid_cohort_sql = f"""
        CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
        WITH opioid_patients AS (
            -- Patients with opioid ICD codes (F1120, etc.) in ANY diagnosis position - exclude from ED_NON_OPIOID targets
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE {opioid_icd_condition}
        ),
        target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification = '{label_ed_non_opioid}'
              AND NOT EXISTS (
                  SELECT 1
                  FROM opioid_patients op
                  WHERE op.mi_person_key = unified_event_fact_table.mi_person_key
              )
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != '{label_ed_non_opioid}'
              AND NOT EXISTS (
                  SELECT 1
                  FROM target_cases tc
                  WHERE tc.mi_person_key = unified_event_fact_table.mi_person_key
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM opioid_patients op
                  WHERE op.mi_person_key = unified_event_fact_table.mi_person_key
              )
              -- Exclude opioid patients from controls as well - complete separation
        ),
        sampled_controls AS (
            -- Hash-based deterministic sampling (replaces ORDER BY RANDOM() for performance)
            WITH target_count AS (
                SELECT COUNT(*) as target_cnt FROM target_cases
            ),
            needed_count AS (
                SELECT tc.target_cnt * 5 as needed FROM target_count tc
            ),
            available_controls AS (
                SELECT COUNT(*) as available FROM control_candidates
            ),
            sample_threshold AS (
                SELECT 
                    CAST(ROUND((SELECT needed FROM needed_count)::DOUBLE / GREATEST((SELECT available FROM available_controls), 1) * 10000) AS BIGINT) as threshold
            )
            SELECT 
                mi_person_key
            FROM control_candidates
            WHERE ABS(hash(mi_person_key)) % 10000 < (SELECT threshold FROM sample_threshold)
            LIMIT (
                SELECT LEAST(
                    (SELECT needed FROM needed_count),
                    (SELECT available FROM available_controls)
                )
            )
        )
        SELECT 
            uef.*,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as target,
            'ED_NON_OPIOID' as cohort_name,
            CASE 
                WHEN tc.mi_person_key IS NOT NULL THEN 'NON_OPIOID_ED'
                WHEN uef.event_type = 'medical' AND uef.hcg_line IS NULL THEN 'NON_ED'
                ELSE 'NON_ED'
            END as cohort,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case
        FROM unified_event_fact_table uef
        LEFT JOIN target_cases tc ON uef.mi_person_key = tc.mi_person_key
        LEFT JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
        WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
        """
        execute_sql_with_dev_validation(conn, logger, ed_non_opioid_cohort_sql)
        logger.info(f"[ensure_cohort_views] Created view: ed_non_opioid_cohort (using classification='{label_ed_non_opioid}')")
