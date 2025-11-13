# Cohort Creation SQL Reference

This document provides a comprehensive reference for all SQL queries used in the Cohort Creation Pipeline. Each phase is documented with explanations, parameters, and example queries.

**Last Updated:** 2025-11-13  
**Version:** 4.2 (Dual-Target + Control-Only Cohorts + HCG Integration)

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

**Dynamic Classification (when `PGX_TARGET_ICD_CODES` is set):**

```sql
CASE 
    WHEN (primary_icd_diagnosis_code IN ('F1120', ...) 
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
    WHEN primary_icd_diagnosis_code IN ('F1120', 'F1121', ...) THEN 'opioid_ed'
    WHEN hcg_line IN ('P51 - ER Visits and Observation Care', 
                      'O11 - Emergency Room', 
                      'P33 - Urgent Care Visits') THEN 'ed_non_opioid'
    ELSE 'ed_non_opioid'
END
```

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
Creates two cohorts (OPIOID_ED and ED_NON_OPIOID) with a 5:1 control-to-target ratio. Patients with opioid ICD codes are completely excluded from ED_NON_OPIOID cohort.

### OPIOID_ED Cohort (Normal Case - Has Targets)

**View:** `opioid_ed_cohort`

```sql
CREATE OR REPLACE VIEW opioid_ed_cohort AS
WITH target_cases AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification = 'target'  -- or 'opioid_ed' if no dynamic targeting
),
control_candidates AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification != 'target'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
),
sampled_controls AS (
    SELECT mi_person_key
    FROM control_candidates
    ORDER BY RANDOM()
    LIMIT (SELECT COUNT(*) * 5 FROM target_cases)  -- 5:1 ratio
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
```

**Logic:**
- **Target cases:** Patients with `event_classification = 'target'` (opioid ICD codes)
- **Controls:** Random sample of 5x target count from non-target patients
- **Cohort column:** `'OPIOID_ED'` for targets, `'NON_ED'` for controls

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
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE primary_icd_diagnosis_code IN ('F1120', 'F1121', 'F1122', ...)  -- OPIOID_ICD_CODES
),
target_cases AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification = 'ed_non_opioid'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)  -- Exclude opioid patients
),
control_candidates AS (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE event_classification != 'ed_non_opioid'
      AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
      AND mi_person_key NOT IN (SELECT mi_person_key FROM opioid_patients)  -- Exclude opioid patients from controls
),
sampled_controls AS (
    SELECT mi_person_key
    FROM control_candidates
    ORDER BY RANDOM()
    LIMIT (SELECT COUNT(*) * 5 FROM target_cases)  -- 5:1 ratio
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
```

**Key Features:**
- **Target cases:** Patients with HCG ED visits (`event_classification = 'ed_non_opioid'`)
- **Exclusion:** Opioid patients are excluded from both targets AND controls
- **Complete separation:** Ensures no overlap with OPIOID_ED cohort
- **Cohort column:** `'NON_OPIOID_ED'` for targets, `'NON_ED'` for controls

---

### ED_NON_OPIOID Cohort (Control-Only Case - Zero Targets)

**View:** `ed_non_opioid_cohort` (when `ed_non_opioid_case_count = 0`)

```sql
CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
WITH opioid_patients AS (
    -- Patients with opioid ICD codes (F1120, etc.) - exclude from ED_NON_OPIOID entirely
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE primary_icd_diagnosis_code IN ('F1120', 'F1121', ...)  -- OPIOID_ICD_CODES
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
    COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
FROM opioid_ed_cohort;

SELECT 
    COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
    COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
FROM ed_non_opioid_cohort;
```

**Check F1120 presence:**

```sql
SELECT 
    COUNT(*) as total_f1120_records,
    COUNT(DISTINCT mi_person_key) as distinct_f1120_patients,
    COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as f1120_target_patients,
    COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as f1120_control_patients
FROM opioid_ed_cohort
WHERE primary_icd_diagnosis_code = 'F1120';
```

**Verify cohort separation (no overlap):**

```sql
-- Check if any opioid patients appear in ED_NON_OPIOID cohort
SELECT COUNT(DISTINCT mi_person_key) as opioid_patients_in_ed_non_opioid
FROM ed_non_opioid_cohort
WHERE mi_person_key IN (
    SELECT DISTINCT mi_person_key
    FROM unified_event_fact_table
    WHERE primary_icd_diagnosis_code IN ('F1120', 'F1121', ...)  -- OPIOID_ICD_CODES
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

- **OPIOID_ED cohort:** Patients with opioid ICD codes (F1120, etc.)
- **ED_NON_OPIOID cohort:** Patients with HCG ED visits, **excluding** all opioid patients
- **Complete separation:** Opioid patients cannot appear in ED_NON_OPIOID as targets or controls

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

- For each target case, 5 controls are randomly sampled
- Controls are selected from patients who are NOT target cases
- Random sampling ensures unbiased control selection
- Ratio is maintained even in control-only cohorts (using average target count)

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

- `Create_Cohort_README.md` - Comprehensive pipeline guide
- `control_only_cohort_analysis.md` - Control-only cohort strategy analysis
- `precompute_target_averages.py` - Pre-computation script for target averages

---

**Note:** All SQL queries use DuckDB syntax and are executed via the Python pipeline. Parameters like `{age_band}`, `{event_year}`, and `{classification_sql}` are dynamically substituted at runtime.

