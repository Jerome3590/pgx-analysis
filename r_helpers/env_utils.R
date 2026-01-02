# ============================================================
# Environment and OS-aware configuration helpers for R pipelines
# ============================================================
# This module mirrors the behavior of the Python env_utils helper.
# It provides:
# - OS and resource detection
# - Sensible defaults for parallelism in feature engineering / modeling
# - OS-aware output roots (NVMe on Linux/EC2, user dir on Windows)
#
# Usage examples:
#   source("r_helpers/env_utils.R")
#   resources <- pgx_configure_environment_r()
#   data_root <- pgx_get_data_root_r()
#   out_dir <- pgx_ensure_output_dir_r("cohorts", cohort_name, age_band)
#
# All console output is ASCII-safe for Windows terminals.


# -----------------------------
# System resource detection
# -----------------------------

pgx_detect_system_resources_r <- function() {
  os_type <- Sys.info()[["sysname"]]
  if (is.null(os_type) || os_type == "") {
    os_type <- .Platform$OS.type
  }

  # CPU cores
  cores <- 1L
  if (requireNamespace("parallel", quietly = TRUE)) {
    cores <- parallel::detectCores(logical = TRUE)
  }
  if (is.na(cores) || cores < 1L) {
    cores <- 1L
  }

  # Total RAM in GB (best effort)
  ram_gb <- NA_integer_

  # Prefer psutil via reticulate if available (optional)
  if (requireNamespace("reticulate", quietly = TRUE)) {
    try({
      psutil <- reticulate::import("psutil", delay_load = TRUE)
      ram_bytes <- psutil$virtual_memory()$total
      ram_gb <- as.integer(ram_bytes / (1024^3))
    }, silent = TRUE)
  }

  # Fallback for Linux via /proc/meminfo
  if (is.na(ram_gb) && tolower(os_type) == "linux" && file.exists("/proc/meminfo")) {
    mem_line <- readLines("/proc/meminfo", warn = FALSE)
    mem_line <- mem_line[grepl("^MemTotal:", mem_line, ignore.case = TRUE)]
    if (length(mem_line) > 0L) {
      parts <- strsplit(mem_line[[1L]], "\\s+")[[1L]]
      # MemTotal is in kB
      mem_kb <- suppressWarnings(as.numeric(parts[2L]))
      if (!is.na(mem_kb) && mem_kb > 0) {
        ram_gb <- as.integer(max(1, mem_kb / 1024 / 1024))
      }
    }
  }

  # Fallback for Windows via memory.limit (approximate)
  if (is.na(ram_gb) && .Platform$OS.type == "windows") {
    lim_mb <- suppressWarnings(as.numeric(utils::memory.limit()))
    if (!is.na(lim_mb) && lim_mb > 0) {
      ram_gb <- as.integer(max(1, lim_mb / 1024))
    }
  }

  # Conservative default if all detection failed
  if (is.na(ram_gb)) {
    ram_gb <- 16L
  }

  list(
    os_type = os_type,
    cpu_cores = as.integer(cores),
    total_ram_gb = as.integer(ram_gb)
  )
}


pgx_recommend_parallelism_r <- function(resources) {
  ram <- as.integer(resources$total_ram_gb)

  # Medical-style workers (for heavy ETL or event processing)
  if (ram >= 512L) {
    workers_medical <- 28L
  } else if (ram >= 128L) {
    workers_medical <- 18L
  } else if (ram >= 64L) {
    workers_medical <- 12L
  } else {
    workers_medical <- 8L
  }

  # DuckDB-style per-worker memory (string with units)
  if (ram >= 256L) {
    duckdb_mem <- "3GB"
  } else if (ram >= 64L) {
    duckdb_mem <- "2GB"
  } else {
    duckdb_mem <- "1GB"
  }

  # Feature engineering / model training workers
  if (ram >= 256L) {
    fea_workers <- 8L
    model_workers <- 8L
  } else if (ram >= 64L) {
    fea_workers <- 4L
    model_workers <- 4L
  } else {
    fea_workers <- 2L
    model_workers <- 2L
  }

  list(
    PGX_WORKERS_MEDICAL = workers_medical,
    PGX_DUCKDB_MEMORY_LIMIT = duckdb_mem,
    PGX_THREADS_PER_WORKER = 1L,
    PGX_R_FEA_WORKERS = fea_workers,
    PGX_R_MODEL_WORKERS = model_workers
  )
}


pgx_configure_environment_r <- function(overwrite = FALSE, verbose = FALSE) {
  # Detect and set PGX_* environment variables for R pipelines.
  resources <- pgx_detect_system_resources_r()

  if (overwrite || Sys.getenv("PGX_OS_TYPE", "") == "") {
    Sys.setenv(PGX_OS_TYPE = resources$os_type)
  }
  if (overwrite || Sys.getenv("PGX_CPU_CORES", "") == "") {
    Sys.setenv(PGX_CPU_CORES = as.character(resources$cpu_cores))
  }
  if (overwrite || Sys.getenv("PGX_TOTAL_RAM_GB", "") == "") {
    Sys.setenv(PGX_TOTAL_RAM_GB = as.character(resources$total_ram_gb))
  }

  rec <- pgx_recommend_parallelism_r(resources)
  for (name in names(rec)) {
    if (overwrite || Sys.getenv(name, "") == "") {
      Sys.setenv(name = as.character(rec[[name]]))
    }
  }

  if (verbose) {
    msg <- paste0(
      "PGX R environment configured (OS=", resources$os_type,
      ", cores=", resources$cpu_cores,
      ", RAM_GB=", resources$total_ram_gb, ")."
    )
    cat(msg, "\n")
  }

  resources
}


# -----------------------------
# OS-aware output roots
# -----------------------------

pgx_is_windows_r <- function() {
  .Platform$OS.type == "windows"
}


pgx_is_linux_r <- function() {
  tolower(Sys.info()[["sysname"]] %||% "") == "linux"
}


`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0L || (is.character(a) && a[1L] == "")) {
    b
  } else {
    a
  }
}


pgx_get_project_root_r <- function() {
  # Try here::here if available; otherwise fall back to current working directory.
  if (requireNamespace("here", quietly = TRUE)) {
    return(here::here())
  }
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}


pgx_get_data_root_r <- function() {
  # Precedence:
  # 1. PGX_DATA_ROOT if set
  # 2. On Linux/EC2: /mnt/nvme
  # 3. On Windows: %USERPROFILE%/pgx_data
  # 4. Fallback: project_root/data
  env_root <- Sys.getenv("PGX_DATA_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, winslash = "/", mustWork = FALSE))
  }

  if (pgx_is_linux_r()) {
    return("/mnt/nvme")
  }

  if (pgx_is_windows_r()) {
    user_home <- path.expand("~")
    return(file.path(user_home, "pgx_data"))
  }

  project_root <- pgx_get_project_root_r()
  file.path(project_root, "data")
}


pgx_ensure_output_dir_r <- function(..., use_data_root = TRUE) {
  # Build and create an output directory, with OS-aware root selection.
  # Example:
  #   dir <- pgx_ensure_output_dir_r("cohorts", cohort_name, age_band)
  #   dir <- pgx_ensure_output_dir_r("model_data", cohort_name, age_band)
  if (use_data_root) {
    root <- pgx_get_data_root_r()
  } else {
    root <- pgx_get_project_root_r()
  }

  parts <- list(...)
  parts <- unlist(parts, use.names = FALSE)
  path <- do.call(file.path, c(list(root), as.list(parts)))

  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }

  normalizePath(path, winslash = "/", mustWork = FALSE)
}


pgx_get_r_fea_workers <- function(default = NULL) {
  val <- Sys.getenv("PGX_R_FEA_WORKERS", unset = "")
  if (nzchar(val)) {
    out <- suppressWarnings(as.integer(val))
    if (!is.na(out) && out > 0L) {
      return(out)
    }
  }
  resources <- pgx_configure_environment_r(overwrite = FALSE, verbose = FALSE)
  rec <- pgx_recommend_parallelism_r(resources)
  as.integer(rec$PGX_R_FEA_WORKERS)
}


pgx_get_r_model_workers <- function(default = NULL) {
  val <- Sys.getenv("PGX_R_MODEL_WORKERS", unset = "")
  if (nzchar(val)) {
    out <- suppressWarnings(as.integer(val))
    if (!is.na(out) && out > 0L) {
      return(out)
    }
  }
  resources <- pgx_configure_environment_r(overwrite = FALSE, verbose = FALSE)
  rec <- pgx_recommend_parallelism_r(resources)
  as.integer(rec$PGX_R_MODEL_WORKERS)
}

