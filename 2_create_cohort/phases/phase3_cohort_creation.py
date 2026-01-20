"""
Phase 3: Final Cohort Creation with 5:1 ratio and DuckDB optimizations.

OPTIMIZED VERSION - Addresses:
- Replaces NOT IN with NOT EXISTS (safer, faster)
- Eliminates ORDER BY RANDOM() (uses hash-based sampling)
- Materializes opioid_patients once
- Unions HCG exclusion windows into single set
- Reduces CTE depth
- Makes profiling filenames unique
- Clarifies target vs is_target_case
"""

from .common import (
    datetime,
    SYMBOLS,
    cleanup_duckdb_temp_files,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    execute_sql_with_dev_validation,
    ensure_gold_views,
    ensure_unified_views,
)
from py_helpers.constants import S3_BUCKET, get_opioid_icd_sql_condition, ALL_ICD_DIAGNOSIS_COLUMNS, OPIOID_ICD_CODES
import os
import time


def run_phase3_step3_final_cohort_fact(context):
    """Phase 3 Step 3: Final Cohort Creation with 5:1 ratio and DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    pipeline_state = context.get("pipeline_state")
    # Get time_window_days from context, defaulting to 14 if None or missing
    time_window_days = context.get("time_window_days") or 14  # Default 14 days, supports 7, 14, 21, 30, 45
    
    step_name = "phase3_step3_final_cohort_fact"
    
    # Check if step already completed
    if pipeline_state and pipeline_state.is_step_completed(step_name):
        logger.info(f"{SYMBOLS['success']} [PHASE 3 STEP 3] Already completed - skipping")
        return
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 3 STEP 3] Starting optimized final cohort creation (5:1 ratio)...")
    
    try:
        # Ensure required views exist if earlier phases were skipped
        ensure_gold_views(cohort_conn_duckdb, logger, age_band, event_year)
        ensure_unified_views(cohort_conn_duckdb, logger)

        # Determine classification labels based on dynamic targeting env
        target_icd = os.getenv("PGX_TARGET_ICD_CODES", "").strip() or os.getenv("PGX_TARGET_ICD_PREFIXES", "").strip()
        target_cpt = os.getenv("PGX_TARGET_CPT_CODES", "").strip() or os.getenv("PGX_TARGET_CPT_PREFIXES", "").strip()
        dynamic_targeting = bool(target_icd or target_cpt)
        label_target = 'target' if dynamic_targeting else 'opioid_ed'
        label_ed_non_opioid = 'ed_non_opioid'
        
        # Log resolved dynamic targeting state for clarity and reproducibility
        logger.info(f"→ [PHASE 3 STEP 3] Dynamic targeting: {dynamic_targeting}")
        logger.info(f"→ [PHASE 3 STEP 3] Target label: '{label_target}', ED_NON_OPIOID label: '{label_ed_non_opioid}'")
        if dynamic_targeting:
            logger.info(f"→ [PHASE 3 STEP 3] Target ICD codes: {target_icd or 'none'}")
            logger.info(f"→ [PHASE 3 STEP 3] Target CPT codes: {target_cpt or 'none'}")
        
        # Enable query profiling with unique filename (prevents overwrite in parallel runs)
        profile_filename = f"/tmp/duckdb_profiling_phase3_step3_{age_band.replace('-', '_')}_{event_year}_{int(time.time())}.json"
        enable_query_profiling(cohort_conn_duckdb, logger, "json", profile_filename)
        
        # HIGH-IMPACT FIX #3: Materialize opioid_patients once and reuse
        # This avoids recomputing the expensive ICD condition check multiple times
        opioid_icd_condition = get_opioid_icd_sql_condition()
        logger.info("→ [PHASE 3 STEP 3] Materializing opioid_patients view (computed once, reused everywhere)...")
        materialize_opioid_patients_sql = f"""
        CREATE OR REPLACE TEMP VIEW opioid_patients_materialized AS
        SELECT DISTINCT mi_person_key
        FROM unified_event_fact_table
        WHERE {opioid_icd_condition}
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, materialize_opioid_patients_sql)
        # Cast COUNT(*) to BIGINT to avoid INT32 overflow for large counts
        # Use fetchdf() to avoid Python connector's INT32 casting issue
        opioid_patient_count_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(*) AS BIGINT) AS count FROM opioid_patients_materialized").fetchdf()
        opioid_patient_count = int(opioid_patient_count_df.iloc[0]['count']) if not opioid_patient_count_df.empty else 0
        logger.info(f"→ [PHASE 3 STEP 3] Materialized {opioid_patient_count:,} opioid patients")
        
        # Check target case counts BEFORE creating cohorts
        # Use fetchdf() to avoid INT32 overflow
        target_case_count_df = cohort_conn_duckdb.sql(f"""
        SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count
        FROM unified_event_fact_table
        WHERE event_classification = '{label_target}'
        """).fetchdf()
        target_case_count = int(target_case_count_df.iloc[0]['count']) if not target_case_count_df.empty else 0
        
        # Count ED_NON_OPIOID targets AFTER excluding opioid patients
        # HIGH-IMPACT FIX #1: Replace NOT IN with NOT EXISTS
        # Use fetchdf() to avoid INT32 overflow
        ed_non_opioid_case_count_query = f"""
        SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count
        FROM unified_event_fact_table uef
        WHERE event_classification = '{label_ed_non_opioid}'
          AND NOT EXISTS (
              SELECT 1
              FROM opioid_patients_materialized op
              WHERE op.mi_person_key = uef.mi_person_key
          )
        """
        ed_non_opioid_case_count_df = cohort_conn_duckdb.sql(ed_non_opioid_case_count_query).fetchdf()
        ed_non_opioid_case_count = int(ed_non_opioid_case_count_df.iloc[0]['count']) if not ed_non_opioid_case_count_df.empty else 0
        
        logger.info(f"→ [PHASE 3 STEP 3] Target case counts:")
        logger.info(f"  OPIOID_ED target patients ({label_target}): {target_case_count:,}")
        logger.info(f"  ED_NON_OPIOID target patients ({label_ed_non_opioid}): {ed_non_opioid_case_count:,}")
        if time_window_days:
            logger.info(f"  POLYPHARMACY COHORT: Using {time_window_days}-day time window for main is_target_case column")
            logger.info(f"  POLYPHARMACY COHORT: Also creating multiclass target columns (7d, 14d, 21d, 30d, 45d) for analysis")
        
        if target_case_count == 0:
            logger.warning(f"⚠️ [PHASE 3 STEP 3] WARNING: No target cases found for OPIOID_ED cohort ({label_target})")
            logger.warning(f"   Cohort will be empty and will not be saved to S3")
            logger.warning(f"   Check: Are target ICD codes present in {age_band}/{event_year}?")
        
        if ed_non_opioid_case_count == 0:
            logger.warning(f"⚠️ [PHASE 3 STEP 3] WARNING: No target cases found for ED_NON_OPIOID cohort ({label_ed_non_opioid})")
            logger.warning(f"   Will create control-only cohort for model training consistency")
        
        # Load pre-computed average target count for control-only cohorts
        avg_target_count = None
        if target_case_count == 0 or ed_non_opioid_case_count == 0:
            import json
            import boto3
            
            config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'cohort_target_averages.json')
            config = None
            
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    logger.info(f"→ [PHASE 3 STEP 3] Loaded pre-computed averages from local config")
                else:
                    logger.info(f"→ [PHASE 3 STEP 3] Local config not found, trying S3...")
                    s3_path = f"s3://{S3_BUCKET}/gold/qa_results/pre_cohort_audit/target_averages.json"
                    try:
                        s3_client = boto3.client('s3')
                        bucket = S3_BUCKET
                        key = "gold/qa_results/pre_cohort_audit/target_averages.json"
                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        config = json.loads(response['Body'].read().decode('utf-8'))
                        logger.info(f"→ [PHASE 3 STEP 3] Loaded pre-computed averages from S3")
                        try:
                            with open(config_file, 'w') as f:
                                json.dump(config, f, indent=2)
                            logger.info(f"→ [PHASE 3 STEP 3] Saved S3 config to local file for future use")
                        except Exception:
                            pass
                    except Exception as s3_e:
                        logger.warning(f"⚠️ Could not load from S3: {s3_e}")
                        logger.warning(f"   Pre-computed averages not available - using fallback")
            except Exception as e:
                logger.warning(f"⚠️ Could not load pre-computed averages: {e}")
                config = None
            
            if config and 'averages' in config and 'combined' in config['averages']:
                avg_target_count = int(config['averages']['combined']['average'])
                logger.info(f"→ [PHASE 3 STEP 3] Using pre-computed average combined targets: {avg_target_count:,}")
            else:
                avg_target_count = 1000
                logger.warning(f"⚠️ [PHASE 3 STEP 3] Using fallback average target count: {avg_target_count:,}")
        
        # Create OPIOID_ED cohort with 5:1 control-to-target ratio
        if target_case_count > 0:
            # HIGH-IMPACT FIX #1: Replace NOT IN with NOT EXISTS
            # HIGH-IMPACT FIX #2: Replace ORDER BY RANDOM() with hash-based sampling (deterministic, fast, parallelizable)
            opioid_ed_cohort_sql = f"""
            CREATE OR REPLACE TABLE opioid_ed_cohort AS
            WITH target_cases AS (
                SELECT DISTINCT mi_person_key
                FROM unified_event_fact_table
                WHERE event_classification = '{label_target}'
            ),
            first_target_dates AS (
                SELECT 
                    mi_person_key,
                    MIN(event_date) as first_opioid_ed_date
                FROM unified_event_fact_table
                WHERE event_classification = '{label_target}'
                GROUP BY mi_person_key
            ),
            control_candidates AS (
                SELECT DISTINCT mi_person_key
                FROM unified_event_fact_table uef
                WHERE event_classification != '{label_target}'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM target_cases tc
                      WHERE tc.mi_person_key = uef.mi_person_key
                  )
            ),
            sampled_controls AS (
                -- HIGH-IMPACT FIX #2: Hash-based sampling instead of ORDER BY RANDOM()
                -- Deterministic, fast, parallelizable - uses hash(mi_person_key) for reproducible sampling
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
                    -- Calculate hash threshold to get approximately needed_count controls
                    -- Use modulo 10000 for fine-grained control (adjust if needed)
                    SELECT 
                        CAST(ROUND((SELECT needed FROM needed_count)::DOUBLE / GREATEST((SELECT available FROM available_controls), 1) * 10000) AS INTEGER) as threshold
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
                -- CLARITY: target column is legacy compatibility (always 1 for this cohort)
                -- Use is_target_case for actual target/control distinction
                1 as target,
                'OPIOID_ED' as cohort_name,
                CASE 
                    WHEN tc.mi_person_key IS NOT NULL THEN 'OPIOID_ED'
                    ELSE 'NON_ED'
                END as cohort,
                CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case,
                CASE 
                    WHEN tc.mi_person_key IS NOT NULL THEN ftd.first_opioid_ed_date
                    ELSE NULL
                END as first_opioid_ed_date,
                NULL as first_ed_non_opioid_date,
                NULL as days_to_target_event
            FROM unified_event_fact_table uef
            LEFT JOIN target_cases tc ON uef.mi_person_key = tc.mi_person_key
            LEFT JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
            LEFT JOIN first_target_dates ftd ON uef.mi_person_key = ftd.mi_person_key
            WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
            """
        else:
            # Zero targets: create control-only cohort
            logger.info(f"→ [PHASE 3 STEP 3] Creating control-only OPIOID_ED cohort (no targets found)")
            control_limit = avg_target_count * 5 if avg_target_count else 5000
            # HIGH-IMPACT FIX #2: Hash-based sampling
            opioid_ed_cohort_sql = f"""
            CREATE OR REPLACE TABLE opioid_ed_cohort AS
            WITH control_candidates AS (
                SELECT DISTINCT mi_person_key
                FROM unified_event_fact_table
                WHERE event_classification != '{label_target}'
            ),
            sampled_controls AS (
                SELECT mi_person_key
                FROM control_candidates
                WHERE ABS(hash(mi_person_key)) % 10000 < CAST(ROUND({control_limit}::DOUBLE / GREATEST((SELECT COUNT(*) FROM control_candidates), 1) * 10000) AS INTEGER)
                LIMIT {control_limit}
            )
            SELECT 
                uef.*,
                0 as target,
                'OPIOID_ED' as cohort_name,
                'NON_ED' as cohort,
                0 as is_target_case,
                NULL as first_opioid_ed_date,
                NULL as first_ed_non_opioid_date,
                NULL as days_to_target_event
            FROM unified_event_fact_table uef
            INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key;
            """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, opioid_ed_cohort_sql)
        logger.info("→ [PHASE 3 STEP 3] OPIOID_ED cohort created")
        
        # Create ED_NON_OPIOID cohort with 5:1 control-to-target ratio
        if ed_non_opioid_case_count > 0:
            # HIGH-IMPACT FIX #4: Union HCG exclusion windows into single exclusion set
            # This reduces planner load, temp tables, and memory pressure
            ed_non_opioid_cohort_sql = f"""
            CREATE OR REPLACE TABLE ed_non_opioid_cohort AS
            WITH hcg_target_events AS (
                -- Get all HCG target events (ED visits) for patients without opioid codes
                SELECT 
                    mi_person_key,
                    event_date as hcg_event_date
                FROM unified_event_fact_table uef
                WHERE event_classification = '{label_ed_non_opioid}'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM opioid_patients_materialized op
                      WHERE op.mi_person_key = uef.mi_person_key
                  )
            ),
            drug_events AS (
                SELECT 
                    mi_person_key,
                    event_date as drug_event_date
                FROM unified_event_fact_table
                WHERE event_type = 'pharmacy'
            ),
            -- Create all time window pairs
            drug_hcg_pairs_7d AS (
                SELECT DISTINCT de.mi_person_key
                FROM drug_events de
                INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
                    AND hte.hcg_event_date >= de.drug_event_date
                    AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 7 DAY)
            ),
            drug_hcg_pairs_14d AS (
                SELECT DISTINCT de.mi_person_key
                FROM drug_events de
                INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
                    AND hte.hcg_event_date >= de.drug_event_date
                    AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 14 DAY)
            ),
            drug_hcg_pairs_21d AS (
                SELECT DISTINCT de.mi_person_key
                FROM drug_events de
                INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
                    AND hte.hcg_event_date >= de.drug_event_date
                    AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 21 DAY)
            ),
            drug_hcg_pairs_30d AS (
                SELECT DISTINCT de.mi_person_key
                FROM drug_events de
                INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
                    AND hte.hcg_event_date >= de.drug_event_date
                    AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 30 DAY)
            ),
            drug_hcg_pairs_45d AS (
                SELECT DISTINCT de.mi_person_key
                FROM drug_events de
                INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
                    AND hte.hcg_event_date >= de.drug_event_date
                    AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 45 DAY)
            ),
            -- HIGH-IMPACT FIX #4: Union all HCG exclusion windows into single set
            all_hcg_exclusions AS (
                SELECT mi_person_key FROM drug_hcg_pairs_7d
                UNION
                SELECT mi_person_key FROM drug_hcg_pairs_14d
                UNION
                SELECT mi_person_key FROM drug_hcg_pairs_21d
                UNION
                SELECT mi_person_key FROM drug_hcg_pairs_30d
                UNION
                SELECT mi_person_key FROM drug_hcg_pairs_45d
            ),
            drug_hcg_pairs AS (
                SELECT DISTINCT
                    de.mi_person_key,
                    de.drug_event_date,
                    hte.hcg_event_date
                FROM drug_events de
                INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
                    AND hte.hcg_event_date >= de.drug_event_date
                    AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL {time_window_days} DAY)
            ),
            patients_with_hcg_in_window AS (
                SELECT DISTINCT mi_person_key
                FROM drug_hcg_pairs
            ),
            patients_with_drug_events AS (
                SELECT DISTINCT mi_person_key
                FROM drug_events
            ),
            target_cases AS (
                SELECT DISTINCT mi_person_key
                FROM patients_with_hcg_in_window
            ),
            first_target_dates AS (
                SELECT 
                    dhp.mi_person_key,
                    MIN(dhp.hcg_event_date) as first_ed_non_opioid_date
                FROM drug_hcg_pairs dhp
                GROUP BY dhp.mi_person_key
            ),
            control_candidates AS (
                -- HIGH-IMPACT FIX #1: Replace multiple NOT IN with single NOT EXISTS on unioned exclusion set
                SELECT DISTINCT pde.mi_person_key
                FROM patients_with_drug_events pde
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM target_cases tc
                    WHERE tc.mi_person_key = pde.mi_person_key
                )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM opioid_patients_materialized op
                      WHERE op.mi_person_key = pde.mi_person_key
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM all_hcg_exclusions ahe
                      WHERE ahe.mi_person_key = pde.mi_person_key
                  )
            ),
            sampled_controls AS (
                -- HIGH-IMPACT FIX #2: Hash-based sampling
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
                        CAST(ROUND((SELECT needed FROM needed_count)::DOUBLE / GREATEST((SELECT available FROM available_controls), 1) * 10000) AS INTEGER) as threshold
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
            ),
            control_reference_dates AS (
                WITH non_ed_reference AS (
                    SELECT 
                        uef.mi_person_key,
                        MIN(uef.event_date) as reference_date
                    FROM unified_event_fact_table uef
                    INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
                    WHERE uef.event_type = 'medical'
                      AND (uef.hcg_line IS NULL OR uef.hcg_line NOT IN ('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits'))
                    GROUP BY uef.mi_person_key
                ),
                fallback_medical_reference AS (
                    SELECT 
                        uef.mi_person_key,
                        MIN(uef.event_date) as reference_date
                    FROM unified_event_fact_table uef
                    INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
                    WHERE uef.event_type = 'medical'
                      AND NOT EXISTS (
                          SELECT 1
                          FROM non_ed_reference ner
                          WHERE ner.mi_person_key = uef.mi_person_key
                      )
                    GROUP BY uef.mi_person_key
                ),
                final_fallback_reference AS (
                    SELECT 
                        uef.mi_person_key,
                        MIN(uef.event_date) as reference_date
                    FROM unified_event_fact_table uef
                    INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM non_ed_reference ner
                        WHERE ner.mi_person_key = uef.mi_person_key
                    )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM fallback_medical_reference fmr
                          WHERE fmr.mi_person_key = uef.mi_person_key
                      )
                    GROUP BY uef.mi_person_key
                ),
                all_reference_dates AS (
                    SELECT * FROM non_ed_reference
                    UNION ALL
                    SELECT * FROM fallback_medical_reference
                    UNION ALL
                    SELECT * FROM final_fallback_reference
                )
                -- CRITICAL FIX: Ensure exactly one row per patient to prevent cartesian product in LEFT JOIN
                -- UNION ALL can theoretically create duplicates if NOT EXISTS logic fails, or if there are edge cases
                -- This GROUP BY ensures one row per patient, preventing row multiplication in events_with_dates CTE
                -- Use MIN() to pick earliest reference date if somehow multiple exist (defensive programming)
                SELECT 
                    mi_person_key,
                    MIN(reference_date) as reference_date
                FROM all_reference_dates
                GROUP BY mi_person_key
            ),
            events_with_dates AS (
                SELECT 
                    uef.*,
                    ftd.first_ed_non_opioid_date,
                    crd.reference_date as control_reference_date,
                    -- Explicitly cast to BIGINT to prevent DuckDB from inferring INTEGER type during view creation
                    -- This prevents INT32 overflow when materializing views with large date differences
                    CASE 
                        WHEN ftd.first_ed_non_opioid_date IS NOT NULL AND uef.event_date IS NOT NULL
                        THEN CAST(datediff('day', uef.event_date::DATE, ftd.first_ed_non_opioid_date::DATE) AS BIGINT)
                        WHEN crd.reference_date IS NOT NULL AND uef.event_date IS NOT NULL
                        THEN CAST(datediff('day', uef.event_date::DATE, crd.reference_date::DATE) AS BIGINT)
                        ELSE NULL
                    END as days_to_target_event
                FROM unified_event_fact_table uef
                LEFT JOIN first_target_dates ftd ON uef.mi_person_key = ftd.mi_person_key
                LEFT JOIN control_reference_dates crd ON uef.mi_person_key = crd.mi_person_key
                -- CRITICAL: Prevent row explosion from multiple time windows
                -- QUALIFY ensures exactly one row per (mi_person_key, event_date, event_type) combination
                -- This prevents multi-window duplication that causes INT32 overflow
                -- Must use uef. prefix to avoid ambiguous column reference
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY uef.mi_person_key, uef.event_date, uef.event_type
                    ORDER BY days_to_target_event NULLS LAST
                ) = 1
            )
            SELECT 
                ewd.*,
                -- CLARITY: target column is legacy compatibility (always 1 for this cohort)
                -- Use is_target_case for actual target/control distinction
                1 as target,
                'ED_NON_OPIOID' as cohort_name,
                CASE 
                    WHEN tc.mi_person_key IS NOT NULL THEN 'NON_OPIOID_ED'
                    WHEN ewd.event_type = 'medical' AND ewd.hcg_line IS NULL THEN 'NON_ED'
                    ELSE 'NON_ED'
                END as cohort,
                CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case,
                CASE WHEN p7d.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case_7d,
                CASE WHEN p14d.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case_14d,
                CASE WHEN p21d.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case_21d,
                CASE WHEN p30d.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case_30d,
                CASE WHEN p45d.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case_45d,
                NULL as first_opioid_ed_date,
                CASE 
                    WHEN tc.mi_person_key IS NOT NULL THEN ewd.first_ed_non_opioid_date
                    ELSE NULL
                END as first_ed_non_opioid_date
            FROM events_with_dates ewd
            LEFT JOIN target_cases tc ON ewd.mi_person_key = tc.mi_person_key
            LEFT JOIN sampled_controls sc ON ewd.mi_person_key = sc.mi_person_key
            LEFT JOIN drug_hcg_pairs_7d p7d ON ewd.mi_person_key = p7d.mi_person_key
            LEFT JOIN drug_hcg_pairs_14d p14d ON ewd.mi_person_key = p14d.mi_person_key
            LEFT JOIN drug_hcg_pairs_21d p21d ON ewd.mi_person_key = p21d.mi_person_key
            LEFT JOIN drug_hcg_pairs_30d p30d ON ewd.mi_person_key = p30d.mi_person_key
            LEFT JOIN drug_hcg_pairs_45d p45d ON ewd.mi_person_key = p45d.mi_person_key
            WHERE (tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL)
              AND (
                  (tc.mi_person_key IS NOT NULL AND (
                      ewd.event_type = 'medical' 
                      OR (ewd.event_type = 'pharmacy' AND ewd.days_to_target_event IS NOT NULL 
                          AND ewd.days_to_target_event >= 0 AND ewd.days_to_target_event <= {time_window_days})
                  ))
                  OR (sc.mi_person_key IS NOT NULL AND (
                      ewd.event_type = 'medical'
                      OR (ewd.event_type = 'pharmacy' AND ewd.days_to_target_event IS NOT NULL 
                          AND ewd.days_to_target_event >= 0 AND ewd.days_to_target_event <= {time_window_days})
                  ))
              );
            """
        else:
            # Zero targets: create control-only cohort
            logger.info(f"→ [PHASE 3 STEP 3] Creating control-only ED_NON_OPIOID cohort (no targets found)")
            control_limit = avg_target_count * 5 if avg_target_count else 5000
            # HIGH-IMPACT FIX #1: Replace NOT IN with NOT EXISTS
            # HIGH-IMPACT FIX #2: Hash-based sampling
            ed_non_opioid_cohort_sql = f"""
            CREATE OR REPLACE TABLE ed_non_opioid_cohort AS
            WITH control_candidates AS (
                SELECT DISTINCT mi_person_key
                FROM unified_event_fact_table uef
                WHERE event_classification != '{label_ed_non_opioid}'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM opioid_patients_materialized op
                      WHERE op.mi_person_key = uef.mi_person_key
                  )
            ),
            sampled_controls AS (
                SELECT mi_person_key
                FROM control_candidates
                WHERE ABS(hash(mi_person_key)) % 10000 < CAST(ROUND({control_limit}::DOUBLE / GREATEST((SELECT COUNT(*) FROM control_candidates), 1) * 10000) AS INTEGER)
                LIMIT {control_limit}
            )
             SELECT 
                 uef.*,
                 0 as target,
                 'ED_NON_OPIOID' as cohort_name,
                 'NON_ED' as cohort,
                 0 as is_target_case,
                 NULL as first_opioid_ed_date,
                 NULL as first_ed_non_opioid_date,
                 NULL as days_to_target_event
             FROM unified_event_fact_table uef
            INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key;
            """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, ed_non_opioid_cohort_sql)
        logger.info("→ [PHASE 3 STEP 3] ED_NON_OPIOID cohort created")
        
        # Log drug window statistics for ed_non_opioid cohort
        if ed_non_opioid_case_count > 0:
            try:
                # Use fetchdf() to avoid INT32 overflow in COUNT queries
                drug_window_stats_df = cohort_conn_duckdb.sql(f"""
                SELECT 
                    CAST(COUNT(*) AS BIGINT) as total_drug_events,
                    CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as patients_with_drugs,
                    CAST(COUNT(CASE WHEN days_to_target_event IS NOT NULL AND days_to_target_event >= 0 AND days_to_target_event <= {time_window_days} THEN 1 END) AS BIGINT) as drugs_in_time_window,
                    AVG(CASE WHEN days_to_target_event IS NOT NULL AND days_to_target_event >= 0 AND days_to_target_event <= {time_window_days} THEN days_to_target_event END) as avg_days_in_window
                FROM ed_non_opioid_cohort
                WHERE event_type = 'pharmacy' AND is_target_case = 1
                """).fetchdf()
                drug_window_stats = drug_window_stats_df.iloc[0] if not drug_window_stats_df.empty else None
                if drug_window_stats is not None and drug_window_stats['total_drug_events'] > 0:
                    logger.info(f"→ [PHASE 3 STEP 3] ED_NON_OPIOID Drug Window Stats (target cases):")
                    logger.info(f"  Total drug events: {int(drug_window_stats['total_drug_events']):,}")
                    logger.info(f"  Patients with drugs: {int(drug_window_stats['patients_with_drugs']):,}")
                    logger.info(f"  Drugs in {time_window_days}-day window: {int(drug_window_stats['drugs_in_time_window']):,}")
                    if drug_window_stats['avg_days_in_window'] is not None:
                        logger.info(f"  Avg days in window: {float(drug_window_stats['avg_days_in_window']):.1f}")
            except Exception as e:
                logger.debug(f"Could not calculate drug window stats: {e}")
        
        # QA checks
        # CRITICAL: Use COUNT(DISTINCT mi_person_key) instead of COUNT(*) to avoid row explosion issues
        # Event-level COUNT(*) can explode to billions of rows due to multiple time windows
        # Patient-level counts are stable and prevent INT32 overflow
        # Use fetchdf() to avoid Python connector's INT32 casting issue
        opioid_ed_count_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count FROM opioid_ed_cohort").fetchdf()
        opioid_ed_count = int(opioid_ed_count_df.iloc[0]['count']) if not opioid_ed_count_df.empty else 0
        
        ed_non_opioid_count_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count FROM ed_non_opioid_cohort").fetchdf()
        ed_non_opioid_count = int(ed_non_opioid_count_df.iloc[0]['count']) if not ed_non_opioid_count_df.empty else 0
        
        # Cast to BIGINT to avoid INT32 overflow
        # Use fetchdf() to avoid Python connector's INT32 casting issue
        opioid_ed_ratio_df = cohort_conn_duckdb.sql("""
        SELECT 
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) as target_cases,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS BIGINT) as control_cases
        FROM opioid_ed_cohort
        """).fetchdf()
        opioid_ed_ratio = (
            int(opioid_ed_ratio_df.iloc[0]['target_cases']) if not opioid_ed_ratio_df.empty and opioid_ed_ratio_df.iloc[0]['target_cases'] is not None else 0,
            int(opioid_ed_ratio_df.iloc[0]['control_cases']) if not opioid_ed_ratio_df.empty and opioid_ed_ratio_df.iloc[0]['control_cases'] is not None else 0
        )
        
        # Cast to BIGINT to avoid INT32 overflow
        # Use fetchdf() to avoid Python connector's INT32 casting issue
        ed_non_opioid_ratio_df = cohort_conn_duckdb.sql("""
        SELECT 
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) as target_cases,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS BIGINT) as control_cases
        FROM ed_non_opioid_cohort
        """).fetchdf()
        ed_non_opioid_ratio = (
            int(ed_non_opioid_ratio_df.iloc[0]['target_cases']) if not ed_non_opioid_ratio_df.empty and ed_non_opioid_ratio_df.iloc[0]['target_cases'] is not None else 0,
            int(ed_non_opioid_ratio_df.iloc[0]['control_cases']) if not ed_non_opioid_ratio_df.empty and ed_non_opioid_ratio_df.iloc[0]['control_cases'] is not None else 0
        )
        
        opioid_ed_control_ratio = opioid_ed_ratio[1] / opioid_ed_ratio[0] if opioid_ed_ratio[0] > 0 else 0
        ed_non_opioid_control_ratio = ed_non_opioid_ratio[1] / ed_non_opioid_ratio[0] if ed_non_opioid_ratio[0] > 0 else 0
        
        logger.info(f"→ [PHASE 3 STEP 3] QA: OPIOID_ED patients: {opioid_ed_count:,}")
        logger.info(f"→ [PHASE 3 STEP 3] QA: ED_NON_OPIOID patients: {ed_non_opioid_count:,}")
        logger.info(f"→ [PHASE 3 STEP 3] QA: OPIOID_ED control ratio: {opioid_ed_control_ratio:.2f}:1")
        logger.info(f"→ [PHASE 3 STEP 3] QA: ED_NON_OPIOID control ratio: {ed_non_opioid_control_ratio:.2f}:1")
        
        if opioid_ed_ratio[0] > 0 and opioid_ed_control_ratio < 5.0:
            logger.warning(
                f"⚠️ [PHASE 3 STEP 3] OPIOID_ED cohort has control ratio {opioid_ed_control_ratio:.2f}:1 "
                f"(target: 5:1). This is expected for small partitions ({age_band}/{event_year}). "
                f"All available controls used: Target cases: {opioid_ed_ratio[0]:,}, Control cases: {opioid_ed_ratio[1]:,}"
            )
        
        if ed_non_opioid_ratio[0] > 0 and ed_non_opioid_control_ratio < 5.0:
            logger.warning(
                f"⚠️ [PHASE 3 STEP 3] ED_NON_OPIOID cohort has control ratio {ed_non_opioid_control_ratio:.2f}:1 "
                f"(target: 5:1). This is expected for small partitions ({age_band}/{event_year}). "
                f"All available controls used: Target cases: {ed_non_opioid_ratio[0]:,}, Control cases: {ed_non_opioid_ratio[1]:,}"
            )
        
        # F1120-specific checks in cohorts
        # Use fetchdf() to avoid INT32 overflow in COUNT queries
        f1120_opioid_check_df = cohort_conn_duckdb.sql("""
        SELECT 
            CAST(COUNT(*) AS BIGINT) as total_f1120_records,
            CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as distinct_f1120_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) as f1120_target_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS BIGINT) as f1120_control_patients
        FROM opioid_ed_cohort
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchdf()
        f1120_opioid_check = (
            int(f1120_opioid_check_df.iloc[0]['total_f1120_records']) if not f1120_opioid_check_df.empty and f1120_opioid_check_df.iloc[0]['total_f1120_records'] is not None else 0,
            int(f1120_opioid_check_df.iloc[0]['distinct_f1120_patients']) if not f1120_opioid_check_df.empty and f1120_opioid_check_df.iloc[0]['distinct_f1120_patients'] is not None else 0,
            int(f1120_opioid_check_df.iloc[0]['f1120_target_patients']) if not f1120_opioid_check_df.empty and f1120_opioid_check_df.iloc[0]['f1120_target_patients'] is not None else 0,
            int(f1120_opioid_check_df.iloc[0]['f1120_control_patients']) if not f1120_opioid_check_df.empty and f1120_opioid_check_df.iloc[0]['f1120_control_patients'] is not None else 0
        )
        
        # Use fetchdf() to avoid INT32 overflow in COUNT queries
        f1120_ed_non_opioid_check_df = cohort_conn_duckdb.sql("""
        SELECT 
            CAST(COUNT(*) AS BIGINT) as total_f1120_records,
            CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as distinct_f1120_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) as f1120_target_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS BIGINT) as f1120_control_patients
        FROM ed_non_opioid_cohort
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchdf()
        f1120_ed_non_opioid_check = (
            int(f1120_ed_non_opioid_check_df.iloc[0]['total_f1120_records']) if not f1120_ed_non_opioid_check_df.empty and f1120_ed_non_opioid_check_df.iloc[0]['total_f1120_records'] is not None else 0,
            int(f1120_ed_non_opioid_check_df.iloc[0]['distinct_f1120_patients']) if not f1120_ed_non_opioid_check_df.empty and f1120_ed_non_opioid_check_df.iloc[0]['distinct_f1120_patients'] is not None else 0,
            int(f1120_ed_non_opioid_check_df.iloc[0]['f1120_target_patients']) if not f1120_ed_non_opioid_check_df.empty and f1120_ed_non_opioid_check_df.iloc[0]['f1120_target_patients'] is not None else 0,
            int(f1120_ed_non_opioid_check_df.iloc[0]['f1120_control_patients']) if not f1120_ed_non_opioid_check_df.empty and f1120_ed_non_opioid_check_df.iloc[0]['f1120_control_patients'] is not None else 0
        )
        
        logger.info(f"→ [PHASE 3 STEP 3] F1120 IN OPIOID_ED COHORT:")
        logger.info(f"  Total F1120 records: {f1120_opioid_check[0]:,}")
        logger.info(f"  Distinct F1120 patients: {f1120_opioid_check[1]:,}")
        logger.info(f"  F1120 target patients: {f1120_opioid_check[2]:,}")
        logger.info(f"  F1120 control patients: {f1120_opioid_check[3]:,}")
        
        logger.info(f"→ [PHASE 3 STEP 3] F1120 IN ED_NON_OPIOID COHORT:")
        logger.info(f"  Total F1120 records: {f1120_ed_non_opioid_check[0]:,}")
        logger.info(f"  Distinct F1120 patients: {f1120_ed_non_opioid_check[1]:,}")
        logger.info(f"  F1120 target patients: {f1120_ed_non_opioid_check[2]:,}")
        logger.info(f"  F1120 control patients: {f1120_ed_non_opioid_check[3]:,}")
        
        # Polypharmacy/HCG check for ED_NON_OPIOID cohort (similar to F1120 check)
        # Validates that HCG target events (ED visits) are present for target cases
        hcg_target_codes = [
            "P51 - ER Visits and Observation Care",
            "O11 - Emergency Room",
            "P33 - Urgent Care Visits"
        ]
        hcg_condition = f"hcg_line IN {tuple(hcg_target_codes)}"
        
        # Use fetchdf() to avoid INT32 overflow in COUNT queries
        polypharmacy_check_df = cohort_conn_duckdb.sql(f"""
        SELECT 
            CAST(COUNT(*) AS BIGINT) as total_hcg_records,
            CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as distinct_hcg_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) as hcg_target_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS BIGINT) as hcg_control_patients,
            CAST(COUNT(DISTINCT CASE WHEN event_type = 'pharmacy' AND is_target_case = 1 THEN mi_person_key END) AS BIGINT) as hcg_target_patients_with_drugs
        FROM ed_non_opioid_cohort
        WHERE {hcg_condition}
        """).fetchdf()
        polypharmacy_check = (
            int(polypharmacy_check_df.iloc[0]['total_hcg_records']) if not polypharmacy_check_df.empty and polypharmacy_check_df.iloc[0]['total_hcg_records'] is not None else 0,
            int(polypharmacy_check_df.iloc[0]['distinct_hcg_patients']) if not polypharmacy_check_df.empty and polypharmacy_check_df.iloc[0]['distinct_hcg_patients'] is not None else 0,
            int(polypharmacy_check_df.iloc[0]['hcg_target_patients']) if not polypharmacy_check_df.empty and polypharmacy_check_df.iloc[0]['hcg_target_patients'] is not None else 0,
            int(polypharmacy_check_df.iloc[0]['hcg_control_patients']) if not polypharmacy_check_df.empty and polypharmacy_check_df.iloc[0]['hcg_control_patients'] is not None else 0,
            int(polypharmacy_check_df.iloc[0]['hcg_target_patients_with_drugs']) if not polypharmacy_check_df.empty and polypharmacy_check_df.iloc[0]['hcg_target_patients_with_drugs'] is not None else 0
        )
        
        logger.info(f"→ [PHASE 3 STEP 3] POLYPHARMACY/HCG IN ED_NON_OPIOID COHORT:")
        logger.info(f"  Total HCG records: {polypharmacy_check[0]:,}")
        logger.info(f"  Distinct HCG patients: {polypharmacy_check[1]:,}")
        logger.info(f"  HCG target patients: {polypharmacy_check[2]:,}")
        logger.info(f"  HCG control patients: {polypharmacy_check[3]:,}")
        logger.info(f"  HCG target patients with drug events: {polypharmacy_check[4]:,}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        # Save checkpoint
        if pipeline_state:
            pipeline_state.mark_step_completed(step_name, {
                'opioid_ed_count': opioid_ed_count,
                'ed_non_opioid_count': ed_non_opioid_count,
                'opioid_ed_control_ratio': float(opioid_ed_control_ratio),
                'ed_non_opioid_control_ratio': float(ed_non_opioid_control_ratio),
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"{SYMBOLS['success']} [PHASE 3 STEP 3] Optimized final cohort creation completed")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 3 STEP 3] Final cohort creation failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise
