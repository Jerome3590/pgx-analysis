# Optimized Cohort Creation Pipeline – Recommended Execution & Fixes

This README consolidates **recommended structural updates**, **execution strategy**, and **notebook / CLI calls** for running the cohort pipeline reliably on a **32‑core / 1 TB RAM EC2 instance**, with special handling for *heavy cohorts* (25–44, 65–74).

---

## 1. High‑Level Execution Strategy (Authoritative)

### ✅ Recommended model (what you already converged on)

**Two top‑level jobs, one per cohort, sequential within cohort:**

- Job A: `opioid_ed`
- Job B: `ed_non_opioid`

Each job runs **all age bands + years sequentially**.

This avoids:
- DuckDB memory fragmentation
- INT32/DOUBLE overflow amplification
- NVMe temp contention
- Accidental fan‑out of 30M–40M row joins

> Do **not** parallelize by age band for 25–44 or 65–74.

---

## 2. Core Root Cause (Why Phase 3 Keeps Failing)

### ❌ Symptom

```
Conversion Error: Type DOUBLE with value 30149350000.0 can't be cast to INT32
```

### ✅ Root cause (confirmed)

This is **not a COUNT issue alone**.

It is caused by **row explosion** in `ed_non_opioid_cohort`:

- `events_with_dates` LEFT JOINs
- multiple drug windows (7d, 14d, 21d, 30d, 45d)
- joins against `sampled_controls`

➡ A *single patient* can explode into **millions of rows**.

Eventually:
- DuckDB represents intermediate counts as DOUBLE
- Python connector attempts INT32 coercion
- pipeline crashes during *QA COUNT*, not creation

---

## 3. Mandatory Fixes (Do These First)

### 3.1 Replace COUNT(*) QA on event‑level views

**Never run COUNT(*) directly on `ed_non_opioid_cohort`.**

#### ❌ Remove
```sql
SELECT COUNT(*) FROM ed_non_opioid_cohort;
```

#### ✅ Replace with patient‑level counts
```sql
SELECT COUNT(DISTINCT mi_person_key) FROM ed_non_opioid_cohort;
```

or (preferred):
```sql
SELECT COUNT(*) FROM (
  SELECT mi_person_key FROM ed_non_opioid_cohort GROUP BY mi_person_key
);
```

> This *alone* will stop the overflow.

---

### 3.2 Make Phase 3 Cohorts **TABLES**, not VIEWS (critical)

DuckDB repeatedly re‑executes views during QA.

#### ✅ Change
```sql
CREATE OR REPLACE VIEW ed_non_opioid_cohort AS ...
```

#### ✅ To
```sql
CREATE OR REPLACE TABLE ed_non_opioid_cohort AS ...
```

Do the same for `opioid_ed_cohort`.

This:
- materializes once
- prevents recomputation
- stabilizes row counts

---

### 3.3 Cap event fan‑out explicitly

Inside `events_with_dates`, add:

```sql
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY mi_person_key, event_date, event_type
  ORDER BY days_to_target_event
) = 1
```

This prevents multi‑window duplication.

---

## 4. CPU / Memory Mapping (32 cores)

### Recommended mapping

| Level | Parallelism | Notes |
|---|---|---|
| Cohort | 2 max | opioid_ed + ed_non_opioid |
| Age band | sequential | esp. 25–44, 65–74 |
| DuckDB threads | 8–12 | beyond this gives no gain |
| concurrent_workers | **1** | always |

Environment:
```bash
export PGX_THREADS_PER_WORKER=8
export DUCKDB_MEMORY_LIMIT=300GB
```

---

## 5. Canonical Execution Commands

### 5.1 Opioid ED (run first)

```bash
python 2_create_cohort/0_create_cohort.py \
  --cohort opioid_ed \
  --concurrent-workers 1
```

### 5.2 ED Non‑Opioid (run second)

```bash
python 2_create_cohort/0_create_cohort.py \
  --cohort ed_non_opioid \
  --time-window-days 14 \
  --concurrent-workers 1
```

> Do **not** run these in the same shell concurrently for heavy bands.

---

## 6. Notebook‑Level Debug Calls (Safe)

### 6.1 Sanity checks (patient‑level only)

```sql
SELECT COUNT(DISTINCT mi_person_key) FROM ed_non_opioid_cohort;
```

```sql
SELECT is_target_case, COUNT(DISTINCT mi_person_key)
FROM ed_non_opioid_cohort
GROUP BY 1;
```

### 6.2 Explosion detector

```sql
SELECT mi_person_key, COUNT(*) AS rows
FROM ed_non_opioid_cohort
GROUP BY 1
ORDER BY rows DESC
LIMIT 20;
```

If any patient > 500k rows → fan‑out bug.

---

## 7. Optional (Strongly Recommended)

### 7.1 Split ED_NON_OPIOID into two tables

- `ed_non_opioid_events`
- `ed_non_opioid_labels`

Join only for modeling.

### 7.2 Persist cohorts before Phase 4

```sql
COPY ed_non_opioid_cohort TO '.../ed_non_opioid.parquet';
```

Phase 4 should **never** recompute Phase 3.

---

## 8. TL;DR (What to Change Today)

1. Replace QA `COUNT(*)` with `COUNT(DISTINCT mi_person_key)`
2. Convert Phase 3 cohort views → tables
3. Enforce 1 worker per cohort
4. Sequential age bands for 25–44 and 65–74
5. Add row‑dedup QUALIFY in `events_with_dates`

Once those are in, the DOUBLE→INT32 error disappears permanently.

