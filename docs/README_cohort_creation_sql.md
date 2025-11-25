# Cohort Creation SQL Reference

This document provides a comprehensive reference for all SQL queries used in the Cohort Creation Pipeline. Each phase is documented with explanations, parameters, and example queries.

**Last Updated:** 2025-11-15  
**Version:** 4.4 (Statistical Independence + Balanced Temporal Windows + Column Matching + Comprehensive ICD Diagnosis Checking)

---

## Table of Contents

1. [Phase 1: Data Preparation](#phase-1-data-preparation)
2. [Phase 2: Event Processing](#phase-2-event-processing)
3. [Phase 3: Cohort Creation](#phase-3-cohort-creation)
4. [Phase 4: Finalization](#phase-4-finalization)
5. [Key Concepts](#key-concepts)

---

## Phase 1: Data Preparation

### Overview
Loads and filters medical and pharmacy data from the APCD gold tier, creating normalized views for downstream processing.

### Medical Data Loading

**View:** `medical_base`

```sql
CREATE OR REPLACE VIEW medical_base AS
SELECT
    CAST(mi_person_key AS VARCHAR) AS mi_person_key,
    -- Map gold medical fields to normalized names used downstream
    member_age_dos AS age_imputed,
    member_gender AS gender_imputed,
    member_race AS race_imputed,
    member_zip_code_dos AS zip_imputed,
    member_county_dos AS county_imputed,
    payer_type AS payer_imputed,
    primary_icd_diagnosis_code,
    -- Carry forward CPT/procedure fields for event features
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
```

**Parameters:**
- `{age_band}`: Age band partition (e.g., "45-54", "65-74")
- `{event_year}`: Event year partition (e.g., 2019, 2020)

**Purpose:** Loads raw medical data from S3 and normalizes column names.

---

### Medical Data Filtering

**View:** `medical`

```sql
CREATE OR REPLACE VIEW medical AS
SELECT *
FROM medical_base
WHERE age_imputed IS NOT NULL
  AND age_imputed BETWEEN 1 AND 114
  AND event_date >= '{event_year}-01-01'
  AND event_date <= '{event_year}-12-31';
```

**Purpose:** Applies data quality filters (valid age range, date range).

---

### Pharmacy Data Loading

**View:** `pharmacy_base`

```sql
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
    -- Build event_date from incurred_date for cohort processing
    TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') AS event_date,
    CAST(event_year AS INTEGER) AS event_year
FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band={age_band}/event_year={event_year}/pharmacy_data.parquet')
WHERE mi_person_key IS NOT NULL
  AND CAST(mi_person_key AS VARCHAR) <> ''
  AND incurred_date IS NOT NULL
  AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL;
```

**Purpose:** Loads pharmacy data and converts `incurred_date` (YYYYMMDD format) to `event_date`.

---

### Pharmacy Data Filtering

**View:** `pharmacy`

```sql
CREATE OR REPLACE VIEW pharmacy AS
SELECT *
FROM pharmacy_base
WHERE event_date IS NOT NULL
  AND event_date >= '{event_year}-01-01'
  AND event_date <= '{event_year}-12-31'
  AND drug_name IS NOT NULL
  AND drug_name <> '';
```

**Purpose:** Filters pharmacy data to valid date range and non-empty drug names.

---

## Phase 2: Event Processing

### Overview
Creates a unified event fact table that combines medical and pharmacy events with classification logic for target identification.

### Event Classification Logic

The classification logic uses a priority-based CASE statement:

**Priority Order:**
1. **Target ICD/CPT codes** → `'target'` (or `'opioid_ed'` if no dynamic targeting)
2. **HCG ED visits** → `'ed_non_opioid'`
3. **Other events** → `'non_target'` (or `'ed_non_opioid'` if default mode)

**Important:** ICD code checking now includes **ALL 10 ICD diagnosis columns** (primary through ten), not just `primary_icd_diagnosis_code`. This ensures no opioid-related events are missed regardless of which diagnosis position the code appears in.

**Dynamic Classification (when `PGX_TARGET_ICD_CODES` is set):**

```sql
CASE 
    WHEN (primary_icd_diagnosis_code IN ('F1120', ...) 
          OR two_icd_diagnosis_code IN ('F1120', ...)
          OR three_icd_diagnosis_code IN ('F1120', ...)
          -- ... through ten_icd_diagnosis_code
          OR procedure_code IN (...)) THEN 'target'
    WHEN hcg_line IN ('P51 - ER Visits and Observation Care', 
                      'O11 - Emergency Room', 
                      'P33 - Urgent Care Visits') THEN 'ed_non_opioid'
    ELSE 'non_target'
END
```

**Default Classification (opioid-specific):**

```sql
CASE 
    WHEN primary_icd_diagnosis_code IN ('F1120', 'F1121', ...)
         OR two_icd_diagnosis_code IN ('F1120', 'F1121', ...)
         OR three_icd_diagnosis_code IN ('F1120', 'F1121', ...)
         -- ... through ten_icd_diagnosis_code
         THEN 'opioid_ed'
    WHEN hcg_line IN ('P51 - ER Visits and Observation Care', 
                      'O11 - Emergency Room', 
                      'P33 - Urgent Care Visits') THEN 'ed_non_opioid'
    ELSE 'ed_non_opioid'
END
```

**Note:** The actual implementation uses a helper function `get_opioid_icd_sql_condition()` from `helpers_1997_13/constants.py` to generate the comprehensive SQL condition across all 10 ICD diagnosis columns.

---

### Unified Event Fact Table

**View:** `unified_event_fact_table`

```sql
CREATE OR REPLACE VIEW unified_event_fact_table AS
-- Medical events
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
    -- CPT/procedure codes (medical)
    procedure_code,
    cpt_mod_1_code,
    cpt_mod_2_code,
    -- HCG fields for ED visit identification
    hcg_setting,
    hcg_line,
    hcg_detail,
    -- Event classification (dynamic via env or default)
    {classification_sql} as event_classification,
    -- Event sequence number
    ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
FROM medical
WHERE primary_icd_diagnosis_code IS NOT NULL

UNION ALL

-- Pharmacy events
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
    -- CPT/procedure codes not present in pharmacy (set NULLs)
    NULL as procedure_code,
    NULL as cpt_mod_1_code,
    NULL as cpt_mod_2_code,
    -- HCG fields not present in pharmacy (set NULLs)
    NULL as hcg_setting,
    NULL as hcg_line,
    NULL as hcg_detail,
    -- Use same classification expression to preserve target logic across union
    {classification_sql} as event_classification,
    ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
FROM pharmacy
WHERE drug_name IS NOT NULL;
```

**Key Features:**
- Combines medical and pharmacy events into a single unified table
- Adds `event_classification` based on target codes and HCG ED visits
- Includes `event_sequence` to track chronological order per patient
- Preserves all demographic and clinical fields

---

## Phase 3: Cohort Creation

### Overview
Creates two cohorts (OPIOID_ED and ED_NON_OPIOID) with a target 5:1 control-to-target ratio. Patients with opioid ICD codes are completely excluded from ED_NON_OPIOID cohort.

**Key Features:**
- **Statistical Independence:** Controls are sampled without replacement (no reuse)
- **Temporal Fields:** Calculates `first_opioid_ed_date`, `first_ed_non_opioid_date`, and `days_to_target_event`
- **Balanced Windows:** ED_NON_OPIOID applies same 30-day lookback window to both targets and controls
- **Column Matching:** All columns match between target cases and controls
- **Ratio Warnings:** Logs warnings when ratio falls below 5:1

### OPIOID_ED Cohort (Normal Case - Has Targets)

**View:** `opioid_ed_cohort`

```sql
CREATE OR REPLACE VIEW opioid_ed_cohort AS
WITH target_cases AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification = 'target'  -- or 'opioid_ed' if no dynamic targeting
),
first_target_dates AS (
    -- Find first target event date per patient
    SELECT 
        mi_person_key,
        MIN(event_date) as first_opioid_ed_date
    FROM unified_event_fact_table
    WHERE event_classification = 'target'
    GROUP BY mi_person_key
),
control_candidates AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification != 'target'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
),
sampled_controls AS (
    -- Sample distinct controls only (no reuse to maintain statistical independence)
    -- Use all available controls if fewer than 5:1 ratio
    WITH target_count AS (
        SELECT COUNT(*) as target_cnt FROM target_cases
    ),
    needed_count AS (
        SELECT tc.target_cnt * 5 as needed FROM target_count tc
    ),
    available_controls AS (
        SELECT COUNT(*) as available FROM control_candidates
    )
    SELECT 
        mi_person_key
    FROM control_candidates
    ORDER BY RANDOM()
    LIMIT (
        SELECT LEAST(
            (SELECT needed FROM needed_count),
            (SELECT available FROM available_controls)
        )
    )
)
SELECT 
    uef.*,
    1 as target,
    'OPIOID_ED' as cohort_name,
    CASE 
        WHEN tc.mi_person_key IS NOT NULL THEN 'OPIOID_ED'
        ELSE 'NON_ED'
    END as cohort,
    CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case,
    -- Temporal fields: targets get first_opioid_ed_date, controls get NULL
    CASE 
        WHEN tc.mi_person_key IS NOT NULL THEN ftd.first_opioid_ed_date
        ELSE NULL
    END as first_opioid_ed_date,
    NULL as first_ed_non_opioid_date,
    NULL as days_to_target_event  -- Can be calculated from event_date and first_opioid_ed_date if needed
FROM unified_event_fact_table uef
LEFT JOIN target_cases tc ON uef.mi_person_key = tc.mi_person_key
LEFT JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
LEFT JOIN first_target_dates ftd ON uef.mi_person_key = ftd.mi_person_key
WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
```

**Logic:**
- **Target cases:** Patients with `event_classification = 'target'` (opioid ICD codes)
- **Controls:** Random sample of up to 5x target count from non-target patients (without replacement)
- **Statistical Independence:** No control reuse - each control patient appears only once
- **Temporal Fields:** 
  - `first_opioid_ed_date`: Populated for targets only (NULL for controls)
  - `days_to_target_event`: NULL (can be calculated manually if needed)
- **Cohort column:** `'OPIOID_ED'` for targets, `'NON_ED'` for controls
- **All Events Included:** Complete drug history for both targets and controls (no time restriction)

---

### OPIOID_ED Cohort (Control-Only Case - Zero Targets)

**View:** `opioid_ed_cohort` (when `target_case_count = 0`)

```sql
CREATE OR REPLACE VIEW opioid_ed_cohort AS
WITH control_candidates AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification != 'target'
),
sampled_controls AS (
    SELECT mi_person_key
    FROM control_candidates
    ORDER BY RANDOM()
    LIMIT {control_limit}  -- avg_target_count * 5 or default 5000
)
SELECT 
    uef.*,
    0 as target,  -- All controls, no targets
    'OPIOID_ED' as cohort_name,
    'NON_ED' as cohort,  -- All controls are non-ED
    0 as is_target_case  -- All are controls
FROM unified_event_fact_table uef
INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key;
```

**Purpose:** Creates a control-only cohort when no targets are found, using pre-computed average target count.

---

### ED_NON_OPIOID Cohort (Normal Case - Has Targets)

**View:** `ed_non_opioid_cohort`

```sql
CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
WITH opioid_patients AS (
    -- Patients with opioid ICD codes (F1120, etc.) - exclude from ED_NON_OPIOID entirely
    -- Checks ALL 10 ICD diagnosis columns to ensure complete opioid patient exclusion
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE primary_icd_diagnosis_code IN ('F1120', 'F1121', 'F1122', ...)
       OR two_icd_diagnosis_code IN ('F1120', 'F1121', 'F1122', ...)
       OR three_icd_diagnosis_code IN ('F1120', 'F1121', 'F1122', ...)
       -- ... through ten_icd_diagnosis_code IN (...)
),
target_cases AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification = 'ed_non_opioid'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)  -- Exclude opioid patients
),
first_target_dates AS (
    -- Find first ED_NON_OPIOID target event date per patient
    SELECT 
        mi_person_key,
        MIN(event_date) as first_ed_non_opioid_date
    FROM unified_event_fact_table
    WHERE event_classification = 'ed_non_opioid'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)
    GROUP BY mi_person_key
),
control_candidates AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification != 'ed_non_opioid'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
      AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)  -- Exclude opioid patients from controls
),
sampled_controls AS (
    -- Sample distinct controls only (no reuse to maintain statistical independence)
    -- Use all available controls if fewer than 5:1 ratio
    WITH target_count AS (
        SELECT COUNT(*) as target_cnt FROM target_cases
    ),
    needed_count AS (
        SELECT tc.target_cnt * 5 as needed FROM target_count tc
    ),
    available_controls AS (
        SELECT COUNT(*) as available FROM control_candidates
    )
    SELECT 
        mi_person_key
    FROM control_candidates
    ORDER BY RANDOM()
    LIMIT (
        SELECT LEAST(
            (SELECT needed FROM needed_count),
            (SELECT available FROM available_controls)
        )
    )
),
control_reference_dates AS (
    -- For controls, use first non-ED medical event as reference date (similar to target date for cases)
    -- This ensures balanced temporal windows between targets and controls
    -- Fallback to first medical event if no non-ED medical events exist
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
    fallback_reference AS (
        SELECT 
            uef.mi_person_key,
            MIN(uef.event_date) as reference_date
        FROM unified_event_fact_table uef
        INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
        WHERE uef.event_type = 'medical'
          AND uef.mi_person_key NOT IN (SELECT mi_person_key FROM non_ed_reference)
        GROUP BY uef.mi_person_key
    )
    SELECT * FROM non_ed_reference
    UNION ALL
    SELECT * FROM fallback_reference
),
events_with_dates AS (
    -- Calculate days_to_target_event for all events
    -- For targets: days to first ED_NON_OPIOID event
    -- For controls: days to reference date (first non-ED medical event) to balance temporal windows
    SELECT 
        uef.*,
        ftd.first_ed_non_opioid_date,
        crd.reference_date as control_reference_date,
        -- Calculate days_to_target_event
        CASE 
            WHEN ftd.first_ed_non_opioid_date IS NOT NULL AND uef.event_date IS NOT NULL
            THEN CAST(datediff(ftd.first_ed_non_opioid_date::DATE, uef.event_date::DATE) AS INTEGER)
            WHEN crd.reference_date IS NOT NULL AND uef.event_date IS NOT NULL
            THEN CAST(datediff(crd.reference_date::DATE, uef.event_date::DATE) AS INTEGER)
            ELSE NULL
        END as days_to_target_event
    FROM unified_event_fact_table uef
    LEFT JOIN first_target_dates ftd ON uef.mi_person_key = ftd.mi_person_key
    LEFT JOIN control_reference_dates crd ON uef.mi_person_key = crd.mi_person_key
)
SELECT 
    ewd.*,
    1 as target,
    'ED_NON_OPIOID' as cohort_name,
    CASE 
        WHEN tc.mi_person_key IS NOT NULL THEN 'NON_OPIOID_ED'
        WHEN ewd.event_type = 'medical' AND ewd.hcg_line IS NULL THEN 'NON_ED'
        ELSE 'NON_ED'
    END as cohort,
    CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case,
    -- Temporal fields: targets get first_ed_non_opioid_date, controls get NULL
    NULL as first_opioid_ed_date,
    CASE 
        WHEN tc.mi_person_key IS NOT NULL THEN ewd.first_ed_non_opioid_date
        ELSE NULL
    END as first_ed_non_opioid_date
FROM events_with_dates ewd
LEFT JOIN target_cases tc ON ewd.mi_person_key = tc.mi_person_key
LEFT JOIN sampled_controls sc ON ewd.mi_person_key = sc.mi_person_key
WHERE (tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL)
  -- Apply balanced 30-day lookback window to both targets and controls
  AND (
      -- Target cases: include medical events OR drug events within 30 days before target
      (tc.mi_person_key IS NOT NULL AND (
          ewd.event_type = 'medical' 
          OR (ewd.event_type = 'pharmacy' AND ewd.days_to_target_event IS NOT NULL 
              AND ewd.days_to_target_event >= 0 AND ewd.days_to_target_event <= 30)
      ))
      -- Controls: apply same temporal logic for balanced comparison
      OR (sc.mi_person_key IS NOT NULL AND (
          ewd.event_type = 'medical'
          OR (ewd.event_type = 'pharmacy' AND ewd.days_to_target_event IS NOT NULL 
              AND ewd.days_to_target_event >= 0 AND ewd.days_to_target_event <= 30)
      ))
  );
```

**Key Features:**
- **Target cases:** Patients with HCG ED visits (`event_classification = 'ed_non_opioid'`)
- **Exclusion:** Opioid patients are excluded from both targets AND controls
- **Complete separation for targets:** Ensures no opioid patients in ED_NON_OPIOID targets (controls can overlap)
- **Statistical Independence:** Controls sampled without replacement WITHIN cohort (can reuse across cohorts)
- **Balanced Temporal Windows:** Both targets and controls use 30-day lookback window
  - Targets: Reference date = first ED_NON_OPIOID event
  - Controls: Reference date = first non-ED medical event
- **Temporal Fields:**
  - `first_ed_non_opioid_date`: Populated for targets only (NULL for controls)
  - `days_to_target_event`: Calculated for both (days to reference date)
- **Cohort column:** `'NON_OPIOID_ED'` for targets, `'NON_ED'` for controls

---

### Phase 3 Summary: Key Improvements

**Statistical Soundness:**
- ✅ **No Control Reuse Within Cohorts:** Controls sampled without replacement WITHIN each cohort maintains statistical independence
- ✅ **Cross-Cohort Reuse Allowed:** Same controls CAN be reused between OPIOID_ED and ED_NON_OPIOID (independent studies)
- ✅ **Balanced Temporal Windows:** ED_NON_OPIOID applies same 30-day lookback to targets and controls
- ✅ **Column Matching:** All columns match between target cases and controls (NULL for cohort-specific fields)
- ✅ **Ratio Transparency:** Warnings logged when ratio falls below 5:1

**Temporal Field Differences:**

| Cohort | `first_opioid_ed_date` | `first_ed_non_opioid_date` | `days_to_target_event` | Temporal Window |
| :-- | :-- | :-- | :-- | :-- |
| **OPIOID_ED** | ✅ Targets only | ❌ NULL | ❌ NULL* | None (all events) |
| **ED_NON_OPIOID** | ❌ NULL | ✅ Targets only | ✅ Both (calculated) | 30-day (both) |

\* Can be calculated manually from `event_date` and `first_opioid_ed_date` if needed.

**Control Sampling Logic:**
- Uses `LEAST(needed_count, available_count)` to prevent over-sampling
- Should achieve 5:1 ratio unless partition (age_band + event_year) is genuinely small
- If fewer controls available than needed, uses all available (logs warning - expected only for small partitions)
- **Within-cohort:** No reuse ensures each control patient appears exactly once per cohort
- **Across-cohort:** Same controls can appear in both OPIOID_ED and ED_NON_OPIOID (independent studies)

---

### ED_NON_OPIOID Cohort (Control-Only Case - Zero Targets)

**View:** `ed_non_opioid_cohort` (when `ed_non_opioid_case_count = 0`)

```sql
CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
WITH opioid_patients AS (
    -- Patients with opioid ICD codes (F1120, etc.) - exclude from ED_NON_OPIOID entirely
    -- Checks ALL 10 ICD diagnosis columns to ensure complete opioid patient exclusion
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE primary_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR two_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR three_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       -- ... through ten_icd_diagnosis_code IN (...)
),
control_candidates AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification != 'ed_non_opioid'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)  -- Exclude opioid patients
),
sampled_controls AS (
    SELECT mi_person_key
    FROM control_candidates
    ORDER BY RANDOM()
    LIMIT {control_limit}  -- avg_target_count * 5 or default 5000
)
SELECT 
    uef.*,
    0 as target,  -- All controls, no targets
    'ED_NON_OPIOID' as cohort_name,
    'NON_ED' as cohort,  -- All controls are non-ED
    0 as is_target_case  -- All are controls
FROM unified_event_fact_table uef
INNER JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key;
```

**Purpose:** Creates a control-only cohort when no HCG ED targets are found, excluding opioid patients.

---

## Phase 4: Finalization

### Overview
Validates cohorts and saves them to S3 in Parquet format.

### Save OPIOID_ED Cohort

```sql
COPY opioid_ed_cohort TO 's3://pgxdatalake/gold/cohorts_{TARGET_NAME}/cohort_name=opioid_ed/event_year={event_year}/age_band={age_band}/cohort.parquet' 
(FORMAT PARQUET, COMPRESSION SNAPPY);
```

**Path Structure:** `cohorts_{TARGET_NAME}/cohort_name={cohort}/event_year={year}/age_band={age_band}/cohort.parquet`

**Example:** `s3://pgxdatalake/gold/cohorts_F1120/cohort_name=opioid_ed/event_year=2019/age_band=45-54/cohort.parquet`

---

### Save ED_NON_OPIOID Cohort

```sql
COPY ed_non_opioid_cohort TO 's3://pgxdatalake/gold/cohorts_{TARGET_NAME}/cohort_name=ed_non_opioid/event_year={event_year}/age_band={age_band}/cohort.parquet' 
(FORMAT PARQUET, COMPRESSION SNAPPY);
```

**Example:** `s3://pgxdatalake/gold/cohorts_F1120/cohort_name=ed_non_opioid/event_year=2019/age_band=45-54/cohort.parquet`

---

### QA Validation Queries

**Check cohort record counts:**

```sql
SELECT COUNT(*) FROM opioid_ed_cohort;
SELECT COUNT(*) FROM ed_non_opioid_cohort;
```

**Check control ratios:**

```sql
SELECT 
    COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
    COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases,
    CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS FLOAT) / 
         NULLIF(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END), 0) as control_ratio
FROM opioid_ed_cohort;

SELECT 
    COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
    COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases,
    CAST(COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) AS FLOAT) / 
         NULLIF(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END), 0) as control_ratio
FROM ed_non_opioid_cohort;
```

**Verify temporal fields:**

```sql
-- Check OPIOID_ED temporal fields
SELECT 
    COUNT(*) as total_records,
    COUNT(CASE WHEN first_opioid_ed_date IS NOT NULL THEN 1 END) as records_with_target_date,
    COUNT(CASE WHEN is_target_case = 1 AND first_opioid_ed_date IS NOT NULL THEN 1 END) as targets_with_date,
    COUNT(CASE WHEN is_target_case = 0 AND first_opioid_ed_date IS NULL THEN 1 END) as controls_with_null_date
FROM opioid_ed_cohort;

-- Check ED_NON_OPIOID temporal fields and balanced windows
SELECT 
    is_target_case,
    COUNT(*) as total_events,
    COUNT(CASE WHEN event_type = 'pharmacy' THEN 1 END) as drug_events,
    COUNT(CASE WHEN event_type = 'pharmacy' AND days_to_target_event IS NOT NULL 
               AND days_to_target_event >= 0 AND days_to_target_event <= 30 THEN 1 END) as drugs_in_window,
    AVG(CASE WHEN days_to_target_event IS NOT NULL AND days_to_target_event >= 0 
             AND days_to_target_event <= 30 THEN days_to_target_event END) as avg_days_in_window
FROM ed_non_opioid_cohort
GROUP BY is_target_case;
```

**Check F1120 presence (across ALL 10 ICD diagnosis columns):**

```sql
SELECT 
    COUNT(*) as total_f1120_records,
    COUNT(DISTINCT mi_person_key) as distinct_f1120_patients,
    COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as f1120_target_patients,
    COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as f1120_control_patients
FROM opioid_ed_cohort
WHERE primary_icd_diagnosis_code = 'F1120'
   OR two_icd_diagnosis_code = 'F1120'
   OR three_icd_diagnosis_code = 'F1120'
   OR four_icd_diagnosis_code = 'F1120'
   OR five_icd_diagnosis_code = 'F1120'
   OR six_icd_diagnosis_code = 'F1120'
   OR seven_icd_diagnosis_code = 'F1120'
   OR eight_icd_diagnosis_code = 'F1120'
   OR nine_icd_diagnosis_code = 'F1120'
   OR ten_icd_diagnosis_code = 'F1120';
```

**Verify cohort separation (no overlap - checks ALL 10 ICD diagnosis columns):**

```sql
-- Check if any opioid patients appear in ED_NON_OPIOID cohort
-- Checks all 10 ICD diagnosis columns to ensure complete separation
SELECT COUNT(DISTINCT mi_person_key) as opioid_patients_in_ed_non_opioid
FROM ed_non_opioid_cohort
WHERE mi_person_key IN (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE primary_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR two_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR three_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR four_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR five_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR six_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR seven_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR eight_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR nine_icd_diagnosis_code IN ('F1120', 'F1121', ...)
       OR ten_icd_diagnosis_code IN ('F1120', 'F1121', ...)
);
-- Should return 0
```

---

## Key Concepts

### Event Classification Priority

1. **Target ICD/CPT codes** → `'target'` (or `'opioid_ed'` if default mode)
2. **HCG ED visits** → `'ed_non_opioid'`
3. **Other events** → `'non_target'` (or `'ed_non_opioid'` if default mode)

### Cohort Separation

- **OPIOID_ED cohort:** Patients with opioid ICD codes (F1120, etc.) in **ANY of the 10 ICD diagnosis columns**
- **ED_NON_OPIOID cohort:** Patients with HCG ED visits, **excluding** all opioid patients (checked across all 10 ICD diagnosis columns)
- **Complete separation:** Opioid patients cannot appear in ED_NON_OPIOID as targets or controls
- **Comprehensive checking:** All 10 ICD diagnosis columns (`primary_icd_diagnosis_code` through `ten_icd_diagnosis_code`) are checked to ensure no opioid patients are missed or misclassified

### Control-Only Cohorts

When a partition has zero targets:
- Uses pre-computed average target count from `cohort_target_averages.json`
- Samples `avg_targets * 5` controls (maintains 5:1 structure)
- All records marked as `is_target_case = 0` and `target = 0`
- Ensures every partition has a cohort file for model training

### Cohort Column Values

The `cohort` column tracks three types:
- **`OPIOID_ED`:** Target cases in opioid_ed_cohort
- **`NON_OPIOID_ED`:** Target cases in ed_non_opioid_cohort
- **`NON_ED`:** Controls in both cohorts

### 5:1 Control Ratio

- For each target case, up to 5 controls are randomly sampled
- Controls are selected from patients who are NOT target cases
- Random sampling ensures unbiased control selection
- **Statistical Independence:** Controls are sampled **without replacement** (no reuse)
  - Each control patient appears only once
  - If fewer than 5:1 ratio is available, all available controls are used
  - Warnings are logged when ratio falls below 5:1
- Ratio is maintained even in control-only cohorts (using average target count)

### Temporal Fields and Drug Window Analysis

The pipeline calculates temporal relationships between events and target events, with different behavior for each cohort:

#### Temporal Fields Schema

| Field | Type | OPIOID_ED | ED_NON_OPIOID | Description |
| :-- | :-- | :-- | :-- | :-- |
| `first_opioid_ed_date` | STRING | ✅ Populated | ❌ NULL | Date of first opioid ED event per patient |
| `first_ed_non_opioid_date` | STRING | ❌ NULL | ✅ Populated | Date of first non-opioid ED event per patient |
| `days_to_target_event` | INTEGER | ❌ NULL | ✅ Calculated | Days from event to first target event |
| `event_date` | STRING | ✅ All | ✅ All | Date of the event |
| `event_sequence` | INTEGER | ✅ All | ✅ All | Sequential order of events per patient |

#### OPIOID_ED Cohort Temporal Behavior

- **Complete Drug History:** All drug events included (no time restriction)
- **No Filtering:** All pharmacy and medical events included regardless of timing
- **First Target Date:** Calculated as `MIN(event_date)` where `event_classification = 'opioid_ed'`
- **Days Calculation:** `days_to_target_event` is NULL; calculate manually if needed:
  ```sql
  SELECT 
    event_date,
    first_opioid_ed_date,
    datediff(first_opioid_ed_date::DATE, event_date::DATE) as days_to_target
  FROM opioid_ed_cohort
  WHERE first_opioid_ed_date IS NOT NULL
  ```

#### ED_NON_OPIOID Cohort Temporal Behavior

- **30-Day Lookback Window:** Applied to BOTH target cases AND controls for balanced comparison
- **Target Cases:**
  - Reference date: First ED_NON_OPIOID event
  - Includes: Medical events OR drug events within 30 days before target
- **Controls:**
  - Reference date: First non-ED medical event (fallback to first medical event)
  - Includes: Medical events OR drug events within 30 days before reference date
- **Drug Event Filtering:** Applied via SQL WHERE clause:
  ```sql
  WHERE (
    -- Target cases: medical events OR drugs within 30 days before target
    (is_target_case = 1 AND (
      event_type = 'medical' 
      OR (event_type = 'pharmacy' 
          AND days_to_target_event >= 0 
          AND days_to_target_event <= 30)
    ))
    -- Controls: same temporal logic for balanced comparison
    OR (is_target_case = 0 AND (
      event_type = 'medical'
      OR (event_type = 'pharmacy' 
          AND days_to_target_event >= 0 
          AND days_to_target_event <= 30)
    ))
  )
  ```
- **First Target Date:** Calculated as `MIN(event_date)` where `event_classification = 'ed_non_opioid'` (excluding opioid patients)
- **Control Reference Date:** Calculated as `MIN(event_date)` for first non-ED medical event per control
- **Days Calculation:** Pre-calculated as `datediff(reference_date::DATE, event_date::DATE)`
  - For targets: Days to first ED_NON_OPIOID event
  - For controls: Days to reference date (first non-ED medical event)
  - Positive values: Event before reference date (included in 30-day window)
  - Zero: Event on reference date
  - Negative values: Event after reference date (excluded for drug events)

#### SQL Implementation Example

```sql
-- First target dates calculation (OPIOID_ED)
WITH first_target_dates AS (
    SELECT 
        mi_person_key,
        MIN(event_date) as first_opioid_ed_date
    FROM unified_event_fact_table
    WHERE event_classification = 'opioid_ed'
    GROUP BY mi_person_key
)

-- Days calculation (ED_NON_OPIOID)
SELECT 
    uef.*,
    CASE 
        WHEN ftd.first_ed_non_opioid_date IS NOT NULL AND uef.event_date IS NOT NULL
        THEN CAST(datediff(ftd.first_ed_non_opioid_date::DATE, uef.event_date::DATE) AS INTEGER)
        ELSE NULL
    END as days_to_target_event
FROM unified_event_fact_table uef
LEFT JOIN first_target_dates ftd ON uef.mi_person_key = ftd.mi_person_key
```

---

## Environment Variables

The following environment variables control dynamic targeting:

| Variable | Description | Example |
| :-- | :-- | :-- |
| `PGX_TARGET_NAME` | Human-readable target name | `F1120` |
| `PGX_TARGET_ICD_CODES` | Comma-separated ICD codes | `F1120,F1121` |
| `PGX_TARGET_CPT_CODES` | Comma-separated CPT codes | `99281,99282` |
| `PGX_TARGET_ICD_PREFIXES` | Comma-separated ICD prefixes | `F11,F12` |
| `PGX_TARGET_CPT_PREFIXES` | Comma-separated CPT prefixes | `9928` |

When set, the pipeline uses generic `'target'`/`'non_target'` classification. When unset, it defaults to `'opioid_ed'`/`'ed_non_opioid'` classification.

---

## Related Documentation

- `docs/README_create_cohort.md` - Comprehensive pipeline guide
- `control_only_cohort_analysis.md` - Control-only cohort strategy analysis
- `precompute_target_averages.py` - Pre-computation script for target averages

---

**Note:** All SQL queries use DuckDB syntax and are executed via the Python pipeline. Parameters like `{age_band}`, `{event_year}`, and `{classification_sql}` are dynamically substituted at runtime.

