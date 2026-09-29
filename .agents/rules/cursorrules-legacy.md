## Python Virtual Environment (venv) and WSL Notes

- **Python venv location**: Use `.venv/` in the project root for local development. Activate with `source .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (Linux/WSL).
- Always install Python dependencies into the active venv. Do not install globally.
- For WSL users:
  - Ensure `.venv/` is created and activated inside WSL, not Windows.
  - Use Linux paths and Python binaries inside WSL.
  - R in WSL: If you need R in WSL, install it via `sudo apt install r-base` and use the Linux R path (e.g., `/usr/bin/R`).
  - Windows R (e.g., `C:\Program Files\R\R-4.5.2\bin\R.exe`) is not directly accessible from WSL. Use the appropriate R for your environment.
  - For cross-environment workflows (Windows/WSL), document which scripts require which environment and paths.

## GitHub Authentication for Git Operations

- Use the Windows `GITHUB_PAT` environment variable for authenticated GitHub push/pull operations when available.
- Prefer non-interactive authentication; do not rely on username/password prompts for GitHub.
- If a Git operation is running from WSL and needs GitHub auth, run it through Windows PowerShell so `GITHUB_PAT` is available directly.
- If authentication fails, verify the token against the GitHub API before changing repository config or credentials.

# Cursor Development Rules for PGx Analysis Project

## R Installation and Rscript Usage (Windows)

- **R install path**: `C:\Program Files\R\R-4.5.2`
- When running or verifying R scripts (e.g. BupaR, Rscript), use: `"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"` if R is not on PATH.
- For parse checks: `& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" --vanilla -e "parse(file='path/to/script.R'); cat('Parse OK\n')"`

## Data Processing Preferences

### Performance and Efficiency
- **PREFER DuckDB and Parquet over pandas DataFrames and R dataframes whenever possible**
 - Use DuckDB SQL queries for data filtering, transformations, and aggregations
 - Read/write Parquet files directly with DuckDB instead of loading into memory
 - Only use pandas/R dataframes when:
 - DuckDB operations are not feasible (e.g., complex statistical operations)
 - Interfacing with libraries that require pandas/R dataframes
 - Small datasets that fit comfortably in memory
 - When pandas/R dataframes are necessary, minimize memory usage and prefer streaming/chunked operations

### Code Organization
- Python utilities should be in `py_helpers/`
- R utilities should be in `r_helpers/`
- Step-specific scripts should remain in their respective step directories
- Shared utilities used across multiple steps should be moved to helpers

### Naming Conventions
- Use explicit naming: "POLYPHARMACY COHORT" when referring to `non_opioid_ed` cohort
- Data partition names (e.g., `cohort_name="non_opioid_ed"`) must match S3/parquet partitions
- Human-readable names can differ from partition names for clarity
- **Claims events vs PK exposure** (manuscripts/docs): predictors are **claims events** (dated line occurred / counts in lookback, `item_*`); reserve **exposure** for dose, concentration, adherence, AUC. See `manuscript/TERMINOLOGY.md`. Legacy pipeline key `phase2_step2_drug_exposure` is unchanged in code.

### File Organization
- R scripts for BupaR analysis should be in `3b_feature_importance_eda/1_bupaR/`
- Shared R utilities should be in `r_helpers/`
- Shared Python utilities should be in `py_helpers/`

### EC2 NVMe file paths (Linux data root)
- **Data root on EC2**: `get_data_root()` → `/mnt/nvme` (or `PGX_DATA_ROOT` if set). Use `py_helpers.env_utils.get_data_root()` / `get_model_data_root()` in code; do not hardcode `/mnt/nvme` when a shared root is intended.
- **Project-specific generated cleanup root:** utility cleanup scripts must use `/mnt/nvme/pgx-analysis` (or `PGX_PROJECT_NVME_ROOT`) for generated project artifacts. Do not delete shared roots such as `/mnt/nvme/gold/cohorts`, `/mnt/nvme/cohorts_staging`, or `/mnt/nvme/4_model_data`.
- **Canonical NVMe paths (when on EC2):**
  - **Data root:** `/mnt/nvme`
  - **Model data (model_events per cohort/age_band):** `/mnt/nvme/4_model_data` (same as `get_model_data_root()` on Linux). S3 mirror: `s3://pgxdatalake/gold/cohorts_model_data/`.
  - **Gold inputs:** `/mnt/nvme/gold/cohorts`, `/mnt/nvme/gold/medical`, `/mnt/nvme/gold/pharmacy` (synced from S3; see `utility_scripts/sync_pgx_to_nvme.sh` and `cleanup_cohort_data.sh`).
  - **SHAP/FFA and feature importance:** `/mnt/nvme/gold/shap_analysis`, `/mnt/nvme/gold/ffa_analysis`, `/mnt/nvme/gold/feature_importance` (sync from S3 for dashboard visuals allowed codes source).
  - **DuckDB temp:** `/mnt/nvme/duckdb_tmp` (used by `py_helpers.duckdb_utils` for large queries).
  - **Legacy cohorts path (some scripts):** `/mnt/nvme/cohorts`.
- **Resolution order** for model_events (e.g. in `model_data_paths.py`, DTW, FP-Growth): try `3b` outputs, then `PGX_DATA_ROOT/4_model_data`, then `/mnt/nvme/4_model_data`, then `project_root/4_model_data` and `4a_model_data`. Document any new candidate roots in this list.

### Dashboard Visuals and Allowed Codes (Step 9 / Notebook 4)
- **Allowed codes are created on EC2** (SHAP/FFA pipeline). Local/dev **download from S3** via `python 9_dashboard_visuals/sync_visualization_data_from_s3.py --allowed-codes-only`. Never run BupaR/FP-Growth with "all codes" locally.
- **Allowed codes JSON (consumed by BupaR, DTW, FP-Growth):**
  - Local: `10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age_band_fname}.json`
  - S3 (download source): `s3://pgxdatalake/gold/bupar/allowed_codes/` (same filenames; sync writes into the local path above).
- **Combine step (notebook 3) output** (used to *build* allowed codes on EC2; not the JSON itself):
  - `10_risk_dashboard/outputs/{cohort}/{age_band_fname}/combined_importance.csv`
  - `write_shap_ffa_allowed_codes_for_bupar` (in `create_bupar_visuals`) prefers this file, then SHAP/FFA paths; writes the JSON to `10_risk_dashboard/visualizations/bupar/outputs/`.
- **Where model/SHAP/FFA Combine was performed:** The Combine step is run from **notebook 3** (`3_model_train_shap_ffa.ipynb`), cell "Combine: SHAP + FFA → dashboard outputs". There are no separate `.log` files for Combine; the run output (including "Saved combined importance to .../10_risk_dashboard/outputs/opioid_ed/85_114/combined_importance.csv" etc.) is in that cell’s **executed output** (stream text). On EC2 the paths are under the project root (e.g. `/home/pgx3874/pgx-analysis/10_risk_dashboard/outputs/{cohort}/{age_band_fname}/`). Step 7 (SHAP) and Step 8 (FFA) logs are also in notebook 3 cell outputs.
- **Visualization outputs** (BupaR, DTW, FP-Growth plots/HTML):
  - `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/plots/`
  - `10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band_fname}/plots/`
  - `10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band_fname}/plots/`
- When resolving "repo root" for these paths (e.g. in sync or scripts), use the directory that contains `10_risk_dashboard` (not a parent like `C:\Projects`).

### R Installation (Windows)
- **R install path**: `C:\Program Files\R\R-4.5.2`
- When running or verifying R scripts (e.g. BupaR, Rscript), use: `"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"` if R is not on PATH.
- For parse checks: `& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" --vanilla -e "parse(file='path/to/script.R'); cat('Parse OK\n')"`

### Markdown Filename Conventions
- **READMEs**: `README.md` or `README_<lowercase_with_underscores>.md` (e.g. `README_model_data_overview.md`). Use explicit, descriptive names; avoid numbered suffixes like `README2.md`.
- **Standalone / technical docs**: `UPPERCASE_WITH_UNDERSCORES.md` (e.g. `WORKFLOW_UPDATES.md`, `TIME_ESTIMATES.md`) for consistency and cross-platform case-sensitivity.
- **No spaces** in filenames; use underscores. Exceptions (e.g. `Presentations/`) may use Title_Case where needed.
- **References**: Use the exact filename case (e.g. `README_cross_ageband_analysis.md` not `README_CROSS_AGEBAND_ANALYSIS.md`) so links work on case-sensitive systems.

## DuckDB/SQL Development Rules

### COUNT Query Requirements
- **ALWAYS use `COUNT(*)::BIGINT`** for any COUNT operation that might return large values (>2.1B)
- **ALWAYS convert to int in Python** after fetching: `count = int(result) if result is not None else 0`
- **Apply to all COUNT variants**: `COUNT(*)`, `COUNT(DISTINCT ...)`, `COUNT(CASE WHEN ...)`
- **Rationale**: DuckDB returns DOUBLE for large counts, Python connector tries INT32 cast → overflow error

**Required Pattern:**
```python
# ✅ CORRECT
count_result = conn.sql("SELECT COUNT(*)::BIGINT FROM table").fetchone()[0]
count = int(count_result) if count_result is not None else 0

# ❌ WRONG - Will fail for counts > 2.1B
count = conn.sql("SELECT COUNT(*) FROM table").fetchone()[0]
```

### CTE and JOIN Safety Rules
- **ALWAYS GROUP BY** on CTEs that will be LEFT JOINed, even if NOT EXISTS should prevent duplicates
- **Use MIN()/MAX()** to pick single value if multiple rows could exist (defensive programming)
- **Validate row counts** before and after JOINs - log expected vs actual
- **Prefer UNION over UNION ALL** when you need distinct values, or add GROUP BY after UNION ALL

**Required Pattern:**
```sql
-- ✅ CORRECT - Prevents cartesian product
cte_name AS (
    WITH combined AS (
        SELECT * FROM source1
        UNION ALL
        SELECT * FROM source2
    )
    SELECT 
        key_column,
        MIN(value_column) as value_column
    FROM combined
    GROUP BY key_column  -- CRITICAL: Ensures one row per key
)

-- ❌ WRONG - Can create cartesian product in JOINs
cte_name AS (
    SELECT * FROM source1
    UNION ALL
    SELECT * FROM source2
)
```

### Logging Requirements for SQL Operations
- **Log all COUNT results** with thousands separator: `logger.info(f"Count: {count:,}")`
- **Log before/after row counts** for major operations to detect multiplication issues
- **Validate CTE row counts** before using in JOINs - check for duplicates
- **Compare expected vs actual** - if actual > expected × 2, log warning about possible cartesian product
- **Use descriptive context** in log messages: `logger.info(f"→ [PHASE X] QA: Total records: {count:,}")`

**Required Pattern:**
```python
# Log with context and validation
before_count = int(conn.sql("SELECT COUNT(*)::BIGINT FROM source").fetchone()[0])
logger.info(f"→ [OPERATION] Before: {before_count:,} rows")

# ... perform operation ...

after_count = int(conn.sql("SELECT COUNT(*)::BIGINT FROM result").fetchone()[0])
logger.info(f"→ [OPERATION] After: {after_count:,} rows")

if after_count > before_count * 10:
    logger.warning(f"⚠️ Significant row increase ({before_count:,} → {after_count:,})")
    logger.warning("   Possible cartesian product or row multiplication issue!")
```

### Arithmetic Operations and CAST Requirements
- **ALWAYS use `CAST(... AS BIGINT)` for arithmetic operations** that multiply large values, even if final result is small
- **Intermediate calculations can exceed INT32** even when final value is within range
- **Common patterns that overflow:**
  - Threshold calculations: `ROUND(needed * 10000 / available)` - intermediate `needed * 10000` can exceed INT32
  - Sampling ratios: `ROUND(count * ratio * multiplier)` - intermediate multiplication can overflow
  - Hash-based sampling: `ABS(hash(key)) % 10000 < threshold` - threshold calculation must be BIGINT

**Required Pattern:**
```sql
-- ✅ CORRECT - Use BIGINT for intermediate calculations
CAST(ROUND((SELECT needed FROM needed_count)::DOUBLE / GREATEST((SELECT available FROM available_controls), 1) * 10000) AS BIGINT) as threshold

-- ❌ WRONG - Intermediate calculation can overflow INT32
-- Even though final threshold is 0-10000, needed * 10000 can be > 2.1B
CAST(ROUND((SELECT needed FROM needed_count)::DOUBLE / GREATEST((SELECT available FROM available_controls), 1) * 10000) AS INTEGER) as threshold
```

**Rationale:**
- Example: `needed = 3,014,935` (target patients × 5)
- Calculation: `3,014,935 * 10,000 = 30,149,350,000` (exceeds INT32 max: 2,147,483,647)
- Even though final threshold after division is 0-10000, the intermediate multiplication overflows
- DuckDB represents large intermediate values as DOUBLE, Python connector tries INT32 cast → overflow error

**Safe INTEGER casts (small values only):**
- `CAST(event_year AS INTEGER)` - Safe, values are 2016-2019
- `CAST(small_count AS INTEGER)` - Safe if count < 2.1B
- **When in doubt, use BIGINT**

### Query Validation Checklist
Before committing SQL queries, verify:
- [ ] All COUNT queries use `::BIGINT` and Python int conversion
- [ ] All arithmetic operations with large multipliers use `CAST(... AS BIGINT)`
- [ ] Threshold calculations use BIGINT, not INTEGER
- [ ] CTEs that will be JOINed have GROUP BY to ensure one row per key
- [ ] Row counts are logged with context and thousands separator
- [ ] Expected vs actual counts are compared and warnings logged if suspicious
- [ ] UNION ALL in CTEs that are JOINed have defensive GROUP BY
- [ ] All LEFT JOINs are checked for potential cartesian products
