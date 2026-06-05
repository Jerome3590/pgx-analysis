# Lessons Learned - Critical QA Checks and Debugging

This document captures critical lessons learned from production issues, debugging sessions, and QA discoveries that should inform future development.

## Table of Contents

**Production bugs & QA:**
1. [INT32 Overflow in COUNT Queries (January 2026)](#int32-overflow-in-count-queries-january-2026)
2. [Cartesian Product in CTEs with UNION ALL (January 2026)](#cartesian-product-in-ctes-with-union-all-january-2026)
3. [Row Explosion from Multiple Time Windows (January 2026)](#row-explosion-from-multiple-time-windows-january-2026)
4. [Case/Control Event-Window Asymmetry Can Create Proxy Leakage (June 2026)](#casecontrol-event-window-asymmetry-can-create-proxy-leakage-june-2026)
5. [Cohort Pipeline Execution Strategy](#cohort-pipeline-execution-strategy)
6. [QA Check Methodology](#qa-check-methodology)

**Architecture & design decisions (final production workflow):**
7. [Feature Engineering Simplification](#feature-engineering-simplification)
8. [Model Selection Philosophy](#model-selection-philosophy) — PR-AUC primary, Ensemble eligible, per-bin models, SHAP/FFA fixed models
9. [Event Filter Placement](#event-filter-placement) — Step 1b before cohort creation
10. [Temporal Validation Strategy](#temporal-validation-strategy) — 2016-2018 train, 2019 holdout, 2020 excluded
11. [Drug Event Explosion Strategy](#drug-event-explosion-strategy)

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
- `phase2_event_processing.py` - Claims-event fact table and medication-event counts
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

## Row Explosion from Multiple Time Windows (January 2026)

### Problem Discovery

**Symptom:**
```
Conversion Error: Type DOUBLE with value 30149350000.0 can't be cast to INT32
```

**Location:** `2_create_cohort/phases/phase3_cohort_creation.py` - QA validation step

**Context:**
- Pipeline successfully created cohorts but failed during QA validation
- Error occurred when counting `ed_non_opioid_cohort`
- This was NOT just a COUNT issue - it was caused by row explosion

### Root Cause Analysis

**The Real Problem:**
Row explosion in `ed_non_opioid_cohort` from:
- `events_with_dates` LEFT JOINs with multiple time windows
- Multiple drug windows (7d, 14d, 21d, 30d, 45d) creating duplicates
- Joins against `sampled_controls` and reference dates
- A single patient could explode into millions of rows

**Why It Happened:**
1. Multiple time windows created duplicate event rows per patient
2. LEFT JOINs multiplied rows when multiple reference dates existed
3. DuckDB represented intermediate counts as DOUBLE (>2.1B)
4. Python connector attempted INT32 coercion
5. Pipeline crashed during QA COUNT, not during creation

### Solution

**Three Critical Fixes:**

1. **Replace COUNT(*) with COUNT(DISTINCT mi_person_key) for QA:**
   ```sql
   -- ❌ WRONG - Can explode to billions of rows
   SELECT COUNT(*) FROM ed_non_opioid_cohort;
   
   -- ✅ CORRECT - Patient-level count is stable
   SELECT COUNT(DISTINCT mi_person_key) FROM ed_non_opioid_cohort;
   ```

2. **Convert VIEW to TABLE:**
   ```sql
   -- ❌ WRONG - Re-executes during QA, unstable counts
   CREATE OR REPLACE VIEW ed_non_opioid_cohort AS ...
   
   -- ✅ CORRECT - Materializes once, stable counts
   CREATE OR REPLACE TABLE ed_non_opioid_cohort AS ...
   ```

3. **Add QUALIFY to prevent row explosion:**
   ```sql
   events_with_dates AS (
       SELECT ...
       FROM unified_event_fact_table uef
       LEFT JOIN ...
       -- CRITICAL: Prevent multi-window duplication
       QUALIFY ROW_NUMBER() OVER (
           PARTITION BY uef.mi_person_key, uef.event_date, uef.event_type
           ORDER BY days_to_target_event NULLS LAST
       ) = 1
   )
   ```

**Files Fixed:**
- `phase3_cohort_creation.py` - Changed VIEW to TABLE, added QUALIFY, patient-level counts
- `phase4_finalization.py` - Changed to patient-level counts

### Prevention

**Required Patterns:**
1. **Always use patient-level counts** for cohort QA: `COUNT(DISTINCT mi_person_key)`
2. **Always create TABLES, not VIEWS** for cohorts that will be counted
3. **Always add QUALIFY** when multiple time windows could create duplicates
4. **Always validate row counts** before QA operations

**Code Review Checklist:**
- [ ] Cohort QA uses `COUNT(DISTINCT mi_person_key)`, not `COUNT(*)`
- [ ] Cohorts are created as TABLES, not VIEWS
- [ ] QUALIFY clauses prevent row explosion from multiple windows
- [ ] Expected vs actual patient counts are validated

**Red Flags:**
- Event-level COUNT(*) on cohorts with time windows
- VIEWs that are repeatedly queried during QA
- Multiple time windows without QUALIFY deduplication
- Row counts that are 100x+ larger than patient counts

---

## Case/Control Event-Window Asymmetry Can Create Proxy Leakage (June 2026)

### Problem Discovery

**Symptom:**
- `non_opioid_ed / 75-84` showed extremely high model performance after temporal retraining.
- 2019 holdout recall was 1.0000 for XGBoost and Ensemble.
- Monte-Carlo CV metrics were also unusually high: AUC near 0.99, PR-AUC near 0.95, recall near 0.98.

**Location:** `4_model_data/create_model_data.py` and downstream Step 6 feature construction.

### Root Cause Analysis

The issue was not 2019 data being used in training. The temporal split was correct after Step 6 safeguards. The issue was **case/control event-window asymmetry** in the event-level modeling dataset:

- Cases were effectively represented by sparse pre-target/cohort rows.
- Controls retained broad gold medical/pharmacy event histories.
- Patient-level utilization features then encoded the construction artifact.

Diagnostic evidence for `non_opioid_ed / 75-84` before the fix:

| Group | Median `n_events` | Mean `n_events` | Median `pgx_num_drugs` |
|-------|-------------------|-----------------|-------------------------|
| Train controls | 256 | 411 | 7 |
| Train cases | 1 | 1.07 | 1 |
| 2019 controls | 409.5 | 573.97 | 12 |
| 2019 cases | 1 | 1.00 | 1 |

All training cases were in `n_event_bin_ordinal = 0`, while controls occupied all density bins. XGBoost feature importance was dominated by `n_event_bin_ordinal`, followed by PGx drug-count features.

### Solution

For `non_opioid_ed`, Step 4 now rebuilds `model_events.parquet` from gold medical/pharmacy events with symmetric pre-index windows:

```text
case events    = gold events in [first_ed_non_opioid_date - 365 days, first_ed_non_opioid_date)
control events = gold events in [control_index_date - 365 days, control_index_date)
```

Cases and controls therefore use the same source event tables and comparable observation windows before patient-level features are computed.

### Prevention

**Required QA before trusting final-model metrics:**

```sql
SELECT
  target,
  COUNT(DISTINCT mi_person_key) AS patients,
  AVG(n_events) AS mean_events,
  MEDIAN(n_events) AS median_events,
  MIN(n_events) AS min_events,
  MAX(n_events) AS max_events
FROM (
  SELECT mi_person_key, target, COUNT(*) AS n_events
  FROM read_parquet('model_events.parquet')
  GROUP BY mi_person_key, target
)
GROUP BY target
ORDER BY target;
```

**Code review checklist:**
- [ ] Cases and controls are sourced from comparable event tables.
- [ ] Cases and controls use the same pre-index lookback length when final-model features include event counts or drug counts.
- [ ] Cases and controls use symmetric row-inclusion filters in Step 4; do not apply Step 3b important-item inclusion only to cases while controls retain broad gold histories.
- [ ] `n_events`, `n_event_bin_ordinal`, `pgx_num_drugs`, and `pgx_num_cpic_drugs` distributions overlap by target class.
- [ ] Any recall/AUC/PR-AUC above 0.85 is audited for construction artifacts before being accepted.
- [ ] Step 4 logs the source-to-output target-date mapping (for example, `first_ed_non_opioid_date` → `first_o11_p_date`) and non-null counts before and after writing `model_events.parquet`.
- [ ] Step 4 logs patient-level survival through each stage: case index-date creation, gold-event join, item filtering, and pre-index lookback filtering.

**Red flags:**
- Cases have median `n_events` near 1 while controls have hundreds of events.
- All cases fall into one density bin.
- Top feature importance is a utilization-count proxy rather than a clinically interpretable item.
- A simple count threshold can recover nearly all cases.
- A forced rebuild moves local output but then downloads from S3 instead of rebuilding locally.
- A target-date output column exists but has zero non-null case rows after aliasing from the Step 2 cohort source column.

---

## Cohort Pipeline Execution Strategy

### Recommended Execution Model

**Two top-level jobs, one per cohort, sequential within cohort:**

- **Job A:** `opioid_ed` - Run all age bands + years sequentially
- **Job B:** `ed_non_opioid` - Run all age bands + years sequentially

**Why Sequential:**
- Avoids DuckDB memory fragmentation
- Prevents INT32/DOUBLE overflow amplification
- Reduces NVMe temp contention
- Prevents accidental fan-out of 30M–40M row joins

> **Critical:** Do **not** parallelize by age band for heavy cohorts (25–44, 65–74).

### CPU / Memory Mapping (32 cores, 1TB RAM)

| Level | Parallelism | Notes |
|-------|-------------|-------|
| Cohort | 2 max | opioid_ed + ed_non_opioid |
| Age band | sequential | esp. 25–44, 65–74 |
| DuckDB threads | 8–12 | beyond this gives no gain |
| concurrent_workers | **1** | always |

**Environment Variables:**
```bash
export PGX_THREADS_PER_WORKER=8
export DUCKDB_MEMORY_LIMIT=300GB
```

### Canonical Execution Commands

**Opioid ED (run first):**
```bash
python 2_create_cohort/0_create_cohort.py \
  --cohort opioid_ed \
  --concurrent-workers 1
```

**ED Non-Opioid (run second):**
```bash
python 2_create_cohort/0_create_cohort.py \
  --cohort ed_non_opioid \
  --time-window-days 14 \
  --concurrent-workers 1
```

> **Important:** Do **not** run these in the same shell concurrently for heavy bands.

### Debugging Queries

**Patient-level sanity checks:**
```sql
-- Check patient counts
SELECT COUNT(DISTINCT mi_person_key) FROM ed_non_opioid_cohort;

-- Check target/control distribution
SELECT is_target_case, COUNT(DISTINCT mi_person_key)
FROM ed_non_opioid_cohort
GROUP BY 1;
```

**Row explosion detector:**
```sql
-- Find patients with excessive rows (indicates fan-out bug)
SELECT mi_person_key, COUNT(*) AS rows
FROM ed_non_opioid_cohort
GROUP BY 1
ORDER BY rows DESC
LIMIT 20;
```

If any patient has > 500k rows → fan-out bug detected.

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
1. **Always use `COUNT(*)::BIGINT`** for large counts (or `COUNT(DISTINCT key)` for patient-level)
2. **Always use `fetchdf()`** instead of `fetchone()` for COUNT queries that might overflow
3. **Always GROUP BY** on CTEs that will be JOINed
4. **Always create TABLES, not VIEWS** for cohorts that will be counted
5. **Always add QUALIFY** when multiple time windows could create duplicates
6. **Always validate row counts** before and after operations
7. **Always log counts** with context and thousands separator
8. **Always compare expected vs actual** and warn on suspicious values

**Prevention is Better Than Debugging:**
- Follow patterns from the start
- Add validation checks proactively
- Log extensively for debugging
- Review queries for cartesian product risks

---

## Design Decisions and Architecture

### Feature Engineering Simplification

**Initial Approach:** Multiple feature engineering steps (BupaR, FP-Growth, DTW, PGx) with all features combined for final model.

**Final Approach:** Single feature engineering step (PGx only) with aggregated feature importances used directly.

**Key Lessons:**
- **FP-Growth Features:** Removed due to target leakage. Patterns mined from combined target+control data can encode target-specific information, creating artificial predictive power.
- **BupaR Features:** Moved to visualization-only. While valuable for exploration, process mining features add complexity without sufficient predictive benefit.
- **DTW:** Used for dashboard visualizations only. Doesn't contribute additional predictive signal beyond aggregated importances.
- **Aggregated Features:** Using MC-CV feature importances directly (without encoding) simplifies pipeline while maintaining predictive power.

**Result:** Streamlined workflow focused on core features with other methods reserved for exploration and visualization.

### Model Selection Philosophy

**Approach:** Train four candidates (XGBoost, XGBoost RF, CatBoost, Ensemble) and select by **PR-AUC mean** as the primary metric.

**Key Lessons:**
- **PR-AUC over Recall as primary:** Imbalanced healthcare datasets (5:1 control:case) make raw Recall misleading — a model predicting all cases as positive scores 1.0 Recall but is useless. PR-AUC captures the precision-recall tradeoff across all thresholds and is robust to class imbalance. **Recall is the secondary tiebreaker.**
- **Ensemble as eligible winner:** The probability-average Ensemble (XGB + CatBoost) is now eligible for selection. When selected, Lambda uses proportional composite-score weights across all three component models. When a single model wins, Lambda uses winner-take-all weights (1.0 for winner).
- **CatBoost FFA Limitation as Quality Control:** CatBoost's symmetric tree structure makes symbolic rule extraction unstable. FFA always uses the best XGBoost variant (`xgb` or `xgb_rf`) regardless of model selection, functioning as a quality filter — only features expressible as XGBoost rules reach the causal analysis tab.
- **SHAP fixed to both binaries:** SHAP always runs on XGBoost (`.ubj`) + CatBoost (`.cbm`) regardless of which model was selected. Cross-model consensus improves FFA rule filtering and causal analysis confidence.
- **Per-bin models:** `train_per_bin()` trains separate models for low/medium/high/extreme event density groups. Lambda inference is per-bin only — no full-cohort fallback. Missing bin models cause `FileNotFoundError` in production; always run `prepare_models.py` after any Step 6 re-run.

**Result:** PR-AUC–first selection avoids inflated Recall scores on imbalanced data; Ensemble eligibility captures cases where no single model dominates; fixed SHAP/FFA models ensure stable interpretability regardless of selection outcome.

### Event Filter Placement

**Problem:** ICD/administrative code filtering traditionally done inconsistently across pipeline, causing feature importance and model training to use different data.

**Solution:** **Step 1b: Event Filtering runs BEFORE cohort creation (Step 2)**

**Key Lessons:**
- Filtering at raw data level reduces downstream computation volume
- Feature importance (Step 3a) computed on same filtered event set as model training (Step 4+)
- Ensures true predictive features captured without confounding
- Validates data quality early in pipeline

**Result:** Improved efficiency and consistency. Feature importance and training guaranteed to use identical data.

### Temporal Validation Strategy

**Approach:** Strict temporal validation with separated time periods.

**Key Lessons:**
- **Train-Test Split:** 2016-2018 training, 2019 holdout, 2020 excluded (COVID-19)
- **Prevents Leakage:** Future data never seen during training ensures temporal isolation
- **COVID Exclusion:** Pandemic disruptions not representative of normal operations
- **Ensures Consistency:** Same train/test split across feature importance and modeling ensures features generalize temporally

**Result:** More reliable performance estimates and better generalization to future data.

### Drug Event Explosion Strategy

**Challenge:** Healthcare data is high-dimensional (hundreds of drugs, thousands of ICD codes per patient).

**Solution:** Patient-Level → Drug-Level Transformation enables sequence modeling while tracking temporal relationships.

**Key Lessons:**
- Context duplication (demographics per drug event) enables both cross-sectional and longitudinal analysis
- Enables sophisticated temporal analyses (BupaR, DTW, FpGrowth)
- Maintains temporal information critical for causal inference (`days_to_ade`, `days_to_outcome`)
- Larger data volume requires efficient formats (Parquet, DuckDB) but enables advanced pattern mining

**Result:** Natural representation for sequence methods while maintaining temporal information needed for symbolic reasoning and rule extraction.

---

## Related Documentation

- [README.md](../../README.md) - Main project documentation
- [README_data_pipeline_architecture.md](README_data_pipeline_architecture.md) - Pipeline architecture
- [README_data_pipeline_workflow.md](README_data_pipeline_workflow.md) - Workflow execution

**Version:** 2.2  
**Last Updated:** March 2026  
**Maintainers:** PGx Data Engineering & Analytics Team
