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

# Import constants (helpers_1997_13)
from helpers_1997_13.constants import OPIOID_ICD_CODES, get_opioid_icd_sql_condition, ALL_ICD_DIAGNOSIS_COLUMNS

# Provide no-op shims for advanced duckdb utils to match simplified helpers
def cleanup_duckdb_temp_files(logger):
    try:
        logger.debug("[shim] cleanup_duckdb_temp_files: no-op in simplified helpers")
    except Exception:
        pass

def enable_query_profiling(conn, logger, profile_format="json", output_path="/tmp/duckdb_profiling.json"):
    try:
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
        raise


def ensure_gold_views(conn, logger, age_band: str, event_year: int):
    """Ensure gold-backed views `medical` and `pharmacy` exist for this session.

    This allows later phases to run even if Phase 1 was skipped due to checkpoints.
    """
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
            primary_icd_diagnosis_code,
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
        FROM read_parquet('s3://pgxdatalake/gold/medical/age_band={age_band}/event_year={event_year}/medical_data.parquet')
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
        FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band={age_band}/event_year={event_year}/pharmacy_data.parquet')
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


def ensure_unified_views(conn, logger):
    """Ensure unified views created by Phase 2 exist: unified_event_fact_table, unified_drug_exposure."""
    # unified_event_fact_table
    try:
        conn.sql("SELECT 1 FROM unified_event_fact_table LIMIT 1").fetchone()
    except Exception:
        # Build dynamic classification from env (mirror Phase 2)
        target_icd_codes = [c.strip() for c in os.getenv("PGX_TARGET_ICD_CODES", "").split(',') if c.strip()]
        target_cpt_codes = [c.strip() for c in os.getenv("PGX_TARGET_CPT_CODES", "").split(',') if c.strip()]
        target_icd_prefixes = [p.strip() for p in os.getenv("PGX_TARGET_ICD_PREFIXES", "").split(',') if p.strip()]
        target_cpt_prefixes = [p.strip() for p in os.getenv("PGX_TARGET_CPT_PREFIXES", "").split(',') if p.strip()]

        icd_conditions = []
        if target_icd_codes:
            # Exact match (codes are normalized to F1120 format in gold tier)
            icd_conditions.append(f"primary_icd_diagnosis_code IN {tuple(target_icd_codes)}")
        for pref in target_icd_prefixes:
            # Normalize prefix and use LIKE with ESCAPE for wildcard safe match
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
        # ED visits are identified by HCG line codes per README documentation
        ed_hcg_lines = [
            "P51 - ER Visits and Observation Care",
            "O11 - Emergency Room",
            "P33 - Urgent Care Visits"
        ]
        ed_hcg_condition = f"hcg_line IN {tuple(ed_hcg_lines)}"
        
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

        event_fact_sql = f"""
        CREATE OR REPLACE VIEW unified_event_fact_table AS
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
            primary_icd_diagnosis_code,
            NULL as drug_name,
            NULL as therapeutic_class_1,
            -- HCG fields for ED visit identification
            hcg_setting,
            hcg_line,
            hcg_detail,
            {classification_sql} as event_classification,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
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
            NULL as primary_icd_diagnosis_code,
            drug_name,
            therapeutic_class_1,
            -- HCG fields not present in pharmacy (set NULLs)
            NULL as hcg_setting,
            NULL as hcg_line,
            NULL as hcg_detail,
            {classification_sql} as event_classification,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM pharmacy
        WHERE drug_name IS NOT NULL;
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
    """
    import os
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
              AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
        ),
        sampled_controls AS (
            SELECT mi_person_key
            FROM control_candidates
            ORDER BY RANDOM()
            LIMIT (SELECT COUNT(*) * 5 FROM target_cases)
        )
        SELECT 
            uef.*,
            1 as target,
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
              AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != '{label_ed_non_opioid}'
              AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
              AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)
              -- Exclude opioid patients from controls as well - complete separation
        ),
        sampled_controls AS (
            SELECT mi_person_key
            FROM control_candidates
            ORDER BY RANDOM()
            LIMIT (SELECT COUNT(*) * 5 FROM target_cases)
        )
        SELECT 
            uef.*,
            1 as target,
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
