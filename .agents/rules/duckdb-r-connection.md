---
description: DuckDB connection pattern for R (BupaR scripts); avoid config on driver, set threads after connect
globs: 3b_feature_importance_eda/1_bupaR/*.R
alwaysApply: false
---

# DuckDB in R: connection and threads

When editing R code that uses DuckDB (e.g. BupaR scripts in `3b_feature_importance_eda/1_bupaR/`):

- **Do not** pass `config` to the driver: `dbConnect(duckdb::duckdb(config = list(threads = n)))` can cause *Expected string vector of length 1* on some R/DuckDB builds.
- **Do** connect with no config, then set threads after connect when `DUCKDB_THREADS` is set:

```r
con <- dbConnect(duckdb::duckdb())
n_threads <- Sys.getenv("DUCKDB_THREADS", "")
if (n_threads != "" && !is.na(suppressWarnings(as.integer(n_threads)))) {
  tryCatch(dbExecute(con, sprintf("SET threads = %s", n_threads)), error = function(e) {})
}
```

- Optional env var: `DUCKDB_THREADS` (e.g. `4` or `8`) to cap Parquet/UNPIVOT threads.

See: `docs/CrossStep_Development/README_duckdb_optimization.md` (§ R (BupaR) DuckDB connection).
