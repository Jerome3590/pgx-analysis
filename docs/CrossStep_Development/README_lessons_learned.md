# Lessons Learned - Critical QA Checks and Debugging

This document captures critical lessons learned from production issues, debugging sessions, and QA discoveries that should inform future development.

## Table of Contents

1. [INT32 Overflow in COUNT Queries (January 2026)](#int32-overflow-in-count-queries-january-2026)
2. [Cartesian Product in CTEs with UNION ALL (January 2026)](#cartesian-product-in-ctes-with-union-all-january-2026)
3. [QA Check Methodology](#qa-check-methodology)

---

## INT32 Overflow in COUNT Queries (January 2026)

### Problem Discovery

**Error Message:**
```
Conversion Error: Type DOUBLE with value 22438400000.0 can't be cast because the value is out of range for the destination type INT32
```

**Location:** `2_create_cohort/phases/phase3_cohort_creation.py:605`

**Context:**
- Pipeline was creating cohorts with large row counts (22.4 billion rows)
- Error occurred during QA validation when counting `ed_non_opioid_cohort`
- Pipeline had successfully created cohorts but failed during validation step

### Root Cause Analysis

1. **DuckDB Behavior:**
   - `COUNT(*)` returns DOUBLE for very large counts (>2.1 billion)
   - Python DuckDB connector attempts to cast to INT32 by default
   - INT32 maximum: 2,147,483,647
   - Values exceeding this cause conversion errors

2. **Affected Queries:**
   - All `COUNT(*)` operations in QA checks
   - `COUNT(DISTINCT ...)` for patient counts
   - `COUNT(CASE WHEN ...)` in ratio calculations
   - Any aggregation that might exceed INT32 range

### Solution

**Pattern Applied:**
```python
# ❌ WRONG - Causes INT32 overflow
count = conn.sql("SELECT COUNT(*) FROM table").fetchone()[0]

# ✅ CORRECT - Use ::BIGINT and Python conversion
count_result = conn.sql("SELECT COUNT(*)::BIGINT FROM table").fetchone()[0]
count = int(count_result) if count_result is not None else 0
```

**Files Fixed:**
- `phase1_data_preparation.py` - Medical/pharmacy counts
- `phase2_event_processing.py` - Event and drug exposure counts
- `phase3_cohort_creation.py` - Cohort counts, ratios, F1120 checks
- `phase4_finalization.py` - Final cohort counts, target checks, HCG checks

### Prevention

**Required Pattern for All COUNT Queries:**
1. Always use `COUNT(*)::BIGINT` syntax (DuckDB shorthand)
2. Convert to int in Python after fetching
3. Handle None values gracefully
4. Log counts with thousands separator for readability

**Code Review Checklist:**
- [ ] All COUNT queries use `::BIGINT`
- [ ] Python conversion includes None check
- [ ] Counts are logged with context

---

## Cartesian Product in CTEs with UNION ALL (January 2026)

### Problem Discovery

**Symptom:**
- Expected row count: ~57 million (total events in source table)
- Actual row count: 22.4 billion (392x multiplication!)
- Pipeline completed but produced incorrect results

**Location:** `2_create_cohort/phases/phase3_cohort_creation.py:437-485`

**Context:**
- `control_reference_dates` CTE used `UNION ALL` to combine three reference date sources
- CTE was later LEFT JOINed in `events_with_dates`
- Each event was multiplied by number of reference date rows per patient

### Root Cause Analysis

**Problematic Code:**
```sql
control_reference_dates AS (
    SELECT * FROM non_ed_reference
    UNION ALL
    SELECT * FROM fallback_medical_reference
    UNION ALL
    SELECT * FROM final_fallback_reference
)
-- Later LEFT JOINed to events_with_dates → cartesian product!
```

**Why It Happened:**
1. `UNION ALL` preserves all rows, including potential duplicates
2. Even with `NOT EXISTS` clauses, edge cases could create multiple rows per patient
3. LEFT JOIN multiplies each event row by number of matching reference date rows
4. Result: 57M events × ~400 rows per patient = 22.4B rows

### Solution

**Fixed Code:**
```sql
control_reference_dates AS (
    WITH all_reference_dates AS (
        SELECT * FROM non_ed_reference
        UNION ALL
        SELECT * FROM fallback_medical_reference
        UNION ALL
        SELECT * FROM final_fallback_reference
    )
    -- CRITICAL: GROUP BY ensures exactly one row per patient
    SELECT 
        mi_person_key,
        MIN(reference_date) as reference_date
    FROM all_reference_dates
    GROUP BY mi_person_key
)
```

**Key Changes:**
1. Added intermediate CTE `all_reference_dates` for clarity
2. Added `GROUP BY mi_person_key` to ensure one row per patient
3. Used `MIN(reference_date)` to pick earliest if multiple exist (defensive)

### Prevention

**Required Pattern for CTEs That Will Be JOINed:**
1. Always GROUP BY on key column(s) if CTE will be JOINed
2. Use MIN()/MAX() to pick single value if multiple could exist
3. Validate row counts before using in JOINs
4. Prefer UNION over UNION ALL when you need distinct values

**Code Review Checklist:**
- [ ] CTEs with UNION ALL that are JOINed have GROUP BY
- [ ] Row counts are validated before JOINs
- [ ] Expected vs actual counts are compared and logged

**Red Flags:**
- CTEs with UNION ALL that are later JOINed
- Row counts 100x+ larger than expected
- Multiple LEFT JOINs to CTEs that might have duplicates

---

## QA Check Methodology

### Systematic Approach to Debugging Row Count Issues

**Step 1: Identify the Symptom**
- Error message (if any)
- Unexpected row counts
- Performance degradation
- Memory issues

**Step 2: Calculate Expected Values**
```python
# Calculate expected row count
total_events = 57_238_327
target_patients = 667_417
control_patients = target_patients * 5
total_patients = target_patients + control_patients
expected_max = total_events  # Should be less with filtering
```

**Step 3: Compare Expected vs Actual**
```python
actual_count = int(conn.sql("SELECT COUNT(*)::BIGINT FROM result").fetchone()[0])
if actual_count > expected_max * 2:
    logger.error(f"⚠️ Row count suspiciously high: {actual_count:,}")
    logger.error(f"   Expected max: {expected_max:,}")
    logger.error("   Possible cartesian product or row multiplication!")
```

**Step 4: Trace the Query**
- Identify all CTEs in the query
- Check which CTEs are JOINed
- Look for UNION ALL operations
- Verify GROUP BY on JOINed CTEs

**Step 5: Validate CTEs Individually**
```python
# Check for duplicates in CTEs
cte_count = int(conn.sql("SELECT COUNT(*)::BIGINT FROM cte_name").fetchone()[0])
distinct_count = int(conn.sql("SELECT COUNT(DISTINCT key_column)::BIGINT FROM cte_name").fetchone()[0])
if cte_count != distinct_count:
    logger.warning(f"⚠️ CTE has duplicates: {cte_count:,} rows, {distinct_count:,} distinct keys")
```

### Logging Requirements

**Required Logging Pattern:**
```python
# 1. Log all COUNT operations with context
count = int(conn.sql("SELECT COUNT(*)::BIGINT FROM table").fetchone()[0])
logger.info(f"→ [PHASE X] QA: Total records: {count:,}")

# 2. Log before/after for major operations
before = int(conn.sql("SELECT COUNT(*)::BIGINT FROM source").fetchone()[0])
logger.info(f"→ [OPERATION] Before: {before:,} rows")
# ... perform operation ...
after = int(conn.sql("SELECT COUNT(*)::BIGINT FROM result").fetchone()[0])
logger.info(f"→ [OPERATION] After: {after:,} rows")

# 3. Validate and warn on suspicious counts
if after > before * 10:
    logger.warning(f"⚠️ Significant row increase ({before:,} → {after:,})")
    logger.warning("   Possible cartesian product or row multiplication issue!")

# 4. Log CTE validation
cte_count = int(conn.sql("SELECT COUNT(*)::BIGINT FROM cte_name").fetchone()[0])
distinct_count = int(conn.sql("SELECT COUNT(DISTINCT key)::BIGINT FROM cte_name").fetchone()[0])
if cte_count != distinct_count:
    logger.warning(f"⚠️ CTE has {cte_count - distinct_count:,} duplicate rows")
```

### Key Metrics to Monitor

1. **Row Count Ratios:**
   - Source table count vs result table count
   - Expected max vs actual count
   - Before vs after operation counts

2. **Patient Count Ratios:**
   - Total rows vs distinct patients
   - Should be reasonable (e.g., 10-100 events per patient average)

3. **Memory Usage:**
   - Sudden spikes during JOIN operations
   - OOM errors during query execution

4. **Query Performance:**
   - Unexpectedly long execution times
   - Queries that should be fast but are slow

### Debugging Checklist

When investigating row count issues:

- [ ] Calculate expected row count from source data
- [ ] Compare expected vs actual with tolerance (e.g., 2x)
- [ ] Check all CTEs for potential duplicates
- [ ] Verify GROUP BY on all JOINed CTEs
- [ ] Look for UNION ALL operations that are JOINed
- [ ] Validate LEFT JOINs for cartesian products
- [ ] Check for missing DISTINCT where needed
- [ ] Review query execution plan if available
- [ ] Log intermediate counts at each CTE step
- [ ] Test with smaller dataset first

---

## Summary

**Critical Rules:**
1. **Always use `COUNT(*)::BIGINT`** for large counts
2. **Always GROUP BY** on CTEs that will be JOINed
3. **Always validate row counts** before and after operations
4. **Always log counts** with context and thousands separator
5. **Always compare expected vs actual** and warn on suspicious values

**Prevention is Better Than Debugging:**
- Follow patterns from the start
- Add validation checks proactively
- Log extensively for debugging
- Review queries for cartesian product risks

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Maintainers:** PGx Data Engineering & Analytics Team
