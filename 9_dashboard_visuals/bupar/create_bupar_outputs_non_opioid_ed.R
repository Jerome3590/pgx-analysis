#!/usr/bin/env Rscript
#
# End-to-end bupaR analysis for Cohort 2 (POLYPHARMACY_ED, non_opioid_ed),
# configurable age band (65–74, 75–84, 85–94).
#
# - Builds target-only and combined event logs from model_data (allowed codes from SHAP/FFA only)
# - Runs pre-HCG sequence analyses (no post-target to avoid leakage)
# - Exports pre-HCG, time-to-HCG per-patient features, trace tables, and process matrices
#
# Target event: The first ED visit (identified by HCG Setting) within 21 days of a
# prescription drug event. Identified in model_events by hcg_line (P51/O11/P33) or
# first_ed_non_opioid_date. BupaR pre-sequences are events before that ED visit;
# post-sequences are not used to avoid target leakage.
#

suppressPackageStartupMessages({
  library(duckdb)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(readr)
  library(bupaR)
  library(bupaverse)
  library(processmapR)
  library(edeaR)
  library(ggplot2)
  library(lubridate)
  library(plotly)
  library(htmlwidgets)
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

project_root <- getwd()  # assume you launched from project root

cohort_name <- "non_opioid_ed"

# Optional command line argument to set age band; default is 65-74
args <- commandArgs(trailingOnly = TRUE)
age_band <- if (length(args) >= 1) args[[1]] else "65-74"

age_band_fname <- gsub("-", "_", age_band)
train_years    <- c(2016L, 2017L, 2018L)

cat("=== bupaR Analysis: Cohort 2 (POLYPHARMACY_ED, non_opioid_ed) ===\n")
cat("  Age band: ", age_band, " (control = within-cohort target=0, no HCG)\n\n", sep = "")

# Cohort-specific target: first ED visit (HCG Setting) within 21 days of a drug event.
# Identified by hcg_line or first_ed_non_opioid_date in model_events, NOT by an ICD code "HCG".
# target_date_map below uses those columns; target_icd_patterns is only for allowed_codes.
target_icd_patterns <- c("HCG")
# HCG line values that identify ED visits (match cohort creation / 4_model_data)
ed_hcg_lines <- c("P51 - ER Visits and Observation Care", "O11 - Emergency Room", "P33 - Urgent Care Visits")

# Resolve model_events path: Use Step 4 (4_model_data) which has full schema with all ICD/CPT columns.
# Step 3b model_events have different schema (missing ICD/CPT columns) and cause BupaR failures.
# Disabled Step 3b path to ensure consistent schema.
if (FALSE) {
  # DISABLED: Step 3b path has incomplete schema
  cohort_slug_3b <- if (cohort_name == "opioid_ed") "opioid" else "polypharmacy"
  path_3b <- file.path(
    project_root,
    "3b_feature_importance_eda", "outputs", "cohorts", "input_model_data",
    paste0("cohort_name=", cohort_slug_3b),
    paste0("age_band=", age_band),
    "model_events.parquet"
  )
  if (file.exists(path_3b)) {
    model_data_path   <- path_3b
    model_data_dir    <- dirname(path_3b)
    model_data_root   <- dirname(dirname(dirname(path_3b)))  # input_model_data dir (parent of cohort_name=...)
    cat("Using model_events from Step 2/3 (3b): ", path_3b, "\n", sep = "")
  }
}
if (TRUE) {
  # On EC2 model data is on NVMe; try /mnt/nvme first, then PGX_DATA_ROOT, then project. Prefer model_events_no_protocols if available.
  model_data_root <- NULL
  data_root <- Sys.getenv("PGX_DATA_ROOT")
  candidates <- c(
    "/mnt/nvme/4_model_data",
    if (nzchar(data_root)) file.path(data_root, "4_model_data") else character(0),
    file.path(project_root, "4_model_data"),
    file.path(project_root, "4a_model_data")
  )
  for (root_candidate in candidates) {
    if (nzchar(root_candidate) && dir.exists(root_candidate)) {
      model_data_root <- root_candidate
      break
    }
  }
  if (is.null(model_data_root)) {
    model_data_root <- file.path(project_root, "4_model_data")
  }
  # EC2 uses underscore in partition names (age_band=75_84). Try underscore first, then hyphen.
  model_data_dir_underscore <- file.path(
    model_data_root,
    paste0("cohort_name=", cohort_name),
    paste0("age_band=", age_band_fname)
  )
  model_data_dir_hyphen <- file.path(
    model_data_root,
    paste0("cohort_name=", cohort_name),
    paste0("age_band=", age_band)
  )
  if (file.exists(file.path(model_data_dir_underscore, "model_events_no_protocols.parquet")) ||
      file.exists(file.path(model_data_dir_underscore, "model_events.parquet"))) {
    model_data_dir <- model_data_dir_underscore
    model_data_no_protocols <- file.path(model_data_dir, "model_events_no_protocols.parquet")
    model_data_main         <- file.path(model_data_dir, "model_events.parquet")
    model_data_path <- if (file.exists(model_data_no_protocols)) model_data_no_protocols else model_data_main
    model_data_from_sql <- sprintf("read_parquet('%s')", model_data_path)
  } else if (file.exists(file.path(model_data_dir_hyphen, "model_events_no_protocols.parquet")) ||
             file.exists(file.path(model_data_dir_hyphen, "model_events.parquet"))) {
    model_data_dir <- model_data_dir_hyphen
    model_data_no_protocols <- file.path(model_data_dir, "model_events_no_protocols.parquet")
    model_data_main         <- file.path(model_data_dir, "model_events.parquet")
    model_data_path <- if (file.exists(model_data_no_protocols)) model_data_no_protocols else model_data_main
    model_data_from_sql <- sprintf("read_parquet('%s')", model_data_path)
  } else if (age_band == "85-114") {
    # 85-114: when single partition missing, union 85-94 and 95-114 (same as create_model_data / FP-Growth).
    band_94_dirs  <- list(
      file.path(model_data_root, paste0("cohort_name=", cohort_name), "age_band=85_94"),
      file.path(model_data_root, paste0("cohort_name=", cohort_name), "age_band=85-94")
    )
    band_114_dirs <- list(
      file.path(model_data_root, paste0("cohort_name=", cohort_name), "age_band=95_114"),
      file.path(model_data_root, paste0("cohort_name=", cohort_name), "age_band=95-114")
    )
    pick_file <- function(dirs) {
      for (d in dirs) {
        np <- file.path(d, "model_events_no_protocols.parquet")
        if (file.exists(np)) return(np)
        mp <- file.path(d, "model_events.parquet")
        if (file.exists(mp)) return(mp)
      }
      NULL
    }
    path_94  <- pick_file(band_94_dirs)
    path_114 <- pick_file(band_114_dirs)
    if (!is.null(path_94) && !is.null(path_114)) {
      model_data_path      <- path_94
      model_data_paths     <- c(path_94, path_114)
      model_data_from_sql  <- sprintf("(SELECT * FROM read_parquet('%s') UNION ALL SELECT * FROM read_parquet('%s'))", path_94, path_114)
      cat("Using model_events for 85-114 as union of 85-94 + 95-114\n")
    } else {
      stop("Model data not found for cohort ", cohort_name, " age_band ", age_band,
           " (tried single 85-114 and union 85-94 + 95-114)")
    }
  } else {
    stop("Model data not found for cohort ", cohort_name, " age_band ", age_band,
         " (tried age_band=", age_band_fname, " and age_band=", age_band, ")")
  }
}

cat("Project root:         ", project_root, "\n", sep = "")
cat("Model data path:      ", model_data_path, "\n", sep = "")
if (exists("model_data_paths", inherits = FALSE)) cat("(85-114 = union 85-94 + 95-114)\n", sep = "")
cat("\n", sep = "")

# -------------------------------------------------------------------
# Helper for saving CSVs locally + to S3, and central plots directory
# -------------------------------------------------------------------

bup_ar_output_root <- file.path(project_root, "10_risk_dashboard", "visualizations", "bupar", "outputs")

save_bupar_csv <- function(df, filename,
                           cohort = cohort_name,
                           age_fname = age_band_fname,
                           age_str = age_band) {
  out_dir <- file.path(bup_ar_output_root, cohort, age_fname, "features")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  local_path <- file.path(out_dir, filename)
  readr::write_csv(df, local_path)

  s3_key <- sprintf("gold/bupar/%s/%s/%s", cohort, age_str, filename)
  s3_uri <- paste0("s3://pgxdatalake/", s3_key)
  cmd <- sprintf("aws s3 cp \"%s\" \"%s\"", local_path, s3_uri)
  cat("Uploading to S3 with command:\n  ", cmd, "\n", sep = "")
  system(cmd)
  invisible(local_path)
}

# Central plots directory and PDF device for this cohort/age band.
plots_dir <- file.path(bup_ar_output_root, cohort_name, age_band_fname, "plots")
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
}

rplots_path <- file.path(
  plots_dir,
  sprintf("%s_%s_Rplots.pdf", cohort_name, age_band_fname)
)

pdf(file = rplots_path, width = 12, height = 9)

# -------------------------------------------------------------------
# Load model_data and build target-only subset
# -------------------------------------------------------------------

# For 85-114 union we have two paths; otherwise one path
if (exists("model_data_paths", inherits = FALSE)) {
  if (!file.exists(model_data_path) || !file.exists(model_data_paths[2])) {
    stop("model_data parquet(s) not found for 85-114 union.",
         "\nRun 4_model_data/create_model_data.py for 85-94 and 95-114 (or single 85-114) first.")
  }
} else if (!file.exists(model_data_path)) {
  stop("model_data parquet not found at: ", model_data_path,
       "\nRun 4_model_data/create_model_data.py for this cohort/age band first.")
}

con <- dbConnect(duckdb::duckdb())

# Detect if model_events has HCG target columns (needed for pre-HCG split)
schema_info <- dbGetQuery(con, paste0("DESCRIBE SELECT * FROM ", model_data_from_sql, " LIMIT 0"))
keys_received_schema <- schema_info$column_name
keys_expected_pre_hcg <- c("hcg_line", "first_ed_non_opioid_date", "event_date", "mi_person_key", "target")
has_hcg_line <- "hcg_line" %in% keys_received_schema
has_first_ed_date <- "first_ed_non_opioid_date" %in% keys_received_schema
cat("keys_expected (for pre-HCG): ", paste(keys_expected_pre_hcg, collapse = ", "), "\n", sep = "")
cat("keys_received (model_events schema): ", paste(head(keys_received_schema, 40), collapse = ", "), if (length(keys_received_schema) > 40) " ..." else "", "\n", sep = "")
if (has_hcg_line) cat("Model data has hcg_line column (first ED visit within 21d of drug event).\n")
if (has_first_ed_date) cat("Model data has first_ed_non_opioid_date column.\n")

query <- sprintf(
  paste0("SELECT * FROM ", model_data_from_sql, " WHERE event_year IN (%s)"),
  paste(train_years, collapse = ",")
)

pgx_df <- dbGetQuery(con, query)

cat("Loaded ", nrow(pgx_df), " events for ", cohort_name, " age_band=", age_band,
    " across years ", paste(train_years, collapse=","), "\n", sep = "")

pgx_df_target1 <- pgx_df %>%
  filter(target == 1L)

cat("Target=1 rows: ", nrow(pgx_df_target1), "\n", sep = "")

# -------------------------------------------------------------------
# Load allowed code set: SHAP/FFA causal feature importances only (no FP-Growth, no fallback).
# -------------------------------------------------------------------

allowed_codes_shap_ffa_path <- file.path(
  bup_ar_output_root,
  sprintf("allowed_codes_shap_ffa_%s_%s.json", cohort_name, age_band_fname)
)

allowed_codes <- character(0)

if (file.exists(allowed_codes_shap_ffa_path)) {
  allowed_codes <- fromJSON(allowed_codes_shap_ffa_path)
  if (!is.character(allowed_codes)) allowed_codes <- as.character(allowed_codes)
  cat("Loaded ", length(allowed_codes), " allowed codes from SHAP/FFA only (causal feature importances).\n", sep = "")
} else {
  cat("No SHAP/FFA allowed codes file; using all codes (event log = dataset filtered by dates only).\n", sep = "")
}

cat("Total unique allowed codes: ", length(allowed_codes), "\n", sep = "")
cat("Sample allowed_codes (first 10): ", paste(head(allowed_codes, 10), collapse = ", "), "\n\n", sep = "")

# -------------------------------------------------------------------
# Build DRUG/ICD/CPT activities and target_eventlog
# -------------------------------------------------------------------

keys_expected_model <- c("mi_person_key", "event_date", "target", "drug_name", "primary_icd_diagnosis_code",
  "two_icd_diagnosis_code", "three_icd_diagnosis_code", "four_icd_diagnosis_code", "five_icd_diagnosis_code",
  "six_icd_diagnosis_code", "seven_icd_diagnosis_code", "eight_icd_diagnosis_code", "nine_icd_diagnosis_code",
  "ten_icd_diagnosis_code", "procedure_code")
keys_received_model <- colnames(pgx_df_target1)
cat("keys_expected (model_events): ", paste(keys_expected_model, collapse = ", "), "\n", sep = "")
cat("keys_received (model_events): ", paste(keys_received_model, collapse = ", "), "\n", sep = "")
cat("Model events columns: ", paste(keys_received_model, collapse = ", "), "\n", sep = "")
cat("Sample drug_name values (first 5): ", paste(head(unique(pgx_df_target1$drug_name[!is.na(pgx_df_target1$drug_name)]), 5), collapse = ", "), "\n", sep = "")
cat("Sample primary_icd values (first 5): ", paste(head(unique(pgx_df_target1$primary_icd_diagnosis_code[!is.na(pgx_df_target1$primary_icd_diagnosis_code)]), 5), collapse = ", "), "\n\n", sep = "")

pgx_df_target1_long <- pgx_df_target1 %>%
  transmute(
    mi_person_key,
    event_date,
    drug_name,
    primary_icd_diagnosis_code,
    two_icd_diagnosis_code,
    three_icd_diagnosis_code,
    four_icd_diagnosis_code,
    five_icd_diagnosis_code,
    six_icd_diagnosis_code,
    seven_icd_diagnosis_code,
    eight_icd_diagnosis_code,
    nine_icd_diagnosis_code,
    ten_icd_diagnosis_code,
    procedure_code
  ) %>%
  mutate(across(
    c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    as.character
  )) %>%
  pivot_longer(
    cols = c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    names_to = "source",
    values_to = "code"
  ) %>%
  filter(!is.na(mi_person_key), !is.na(code), code != "", code != "NA")

# Diagnostics: counts before/after allowed_codes filter
n_long_before_allowed <- nrow(pgx_df_target1_long)
codes_in_data <- unique(pgx_df_target1_long$code)
n_codes_in_data <- length(codes_in_data)
cat("BupaR diagnostic: long rows before allowed_codes filter: ", n_long_before_allowed,
    " (distinct codes in data: ", n_codes_in_data, ").\n", sep = "")

# Event log = dataset filtered by causal (SHAP/FFA) codes when available, then by valid dates; never empty by design when data exist
pgx_df_target1_long <- pgx_df_target1_long %>%
  {
    if (length(allowed_codes) > 0) {
      dplyr::filter(., code %in% allowed_codes)
    } else {
      .  # no causal filter; use all codes so event log = dataset with dates
    }
  } %>%
  mutate(
    activity = dplyr::case_when(
      source == "drug_name" ~ paste0("DRUG:", code),
      grepl("icd_diagnosis_code", source) ~ paste0("ICD:", code),
      source == "procedure_code" ~ paste0("CPT:", code),
      TRUE ~ code
    )
  )

n_after_allowed <- nrow(pgx_df_target1_long)
cat("BupaR diagnostic: long rows after allowed_codes filter (before timestamp): ", n_after_allowed, ".\n", sep = "")

# Robust timestamp: parquet/DuckDB may give date as integer (days) or Date; as.POSIXct(integer) treats as seconds
to_ts <- function(x) {
  if (is.numeric(x)) {
    as.POSIXct(as.Date(x, origin = "1970-01-01"))
  } else {
    as.POSIXct(as.Date(x))
  }
}
pgx_df_target1_long <- pgx_df_target1_long %>%
  mutate(timestamp = suppressWarnings(to_ts(event_date)))

n_na_ts <- sum(is.na(pgx_df_target1_long$timestamp))
na_ts_event_date_sample <- if (n_na_ts > 0L) head(pgx_df_target1_long$event_date[is.na(pgx_df_target1_long$timestamp)], 10) else character(0)
if (n_na_ts > 0L) {
  cat("BupaR diagnostic: timestamp NA count: ", n_na_ts, " (event_date class: ", paste(class(pgx_df_target1_long$event_date), collapse = ", "), ").\n", sep = "")
}
pgx_df_target1_long <- pgx_df_target1_long %>%
  filter(!is.na(timestamp))

n_long_after <- nrow(pgx_df_target1_long)
cat("BupaR diagnostic: long rows after timestamp filter: ", n_long_after, ".\n", sep = "")
if (n_long_after == 0L && n_after_allowed > 0L) {
  cat("BupaR diagnostic: timestamp filter removed all rows. event_date sample (where timestamp was NA): ",
      paste(na_ts_event_date_sample, collapse = ", "), ".\n", sep = "")
}
if (n_long_after == 0L && length(allowed_codes) > 0L && n_long_before_allowed > 0L && n_after_allowed == 0L) {
  cat("BupaR diagnostic: no overlap between allowed_codes and data. Sample allowed_codes (max 20): ",
      paste(head(allowed_codes, 20), collapse = ", "), ".\n", sep = "")
  cat("BupaR diagnostic: sample codes in data (max 20): ",
      paste(head(codes_in_data, 20), collapse = ", "), ".\n", sep = "")
}

target_eventlog <- pgx_df_target1_long %>%
  transmute(
    case_id              = mi_person_key,
    activity             = activity,
    timestamp            = timestamp,
    activity_instance_id = dplyr::row_number(),
    lifecycle_id         = dplyr::case_when(
      grepl("^DRUG:", activity) ~ "Drug",
      grepl("^ICD:",  activity) ~ "ICD",
      grepl("^CPT:",  activity) ~ "CPT",
      TRUE ~ "Other"
    ),
    resource_id          = "Patient"
  ) %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id",
    timestamp            = "timestamp"
  )

keys_expected_eventlog <- c("case_id", "activity", "timestamp", "activity_instance_id", "lifecycle_id", "resource_id")
keys_received_eventlog <- names(target_eventlog)
cat("keys_expected (eventlog): ", paste(keys_expected_eventlog, collapse = ", "), "\n", sep = "")
cat("keys_received (eventlog): ", paste(keys_received_eventlog, collapse = ", "), "\n", sep = "")
cat("Target eventlog created.\n")
print(target_eventlog)

# -------------------------------------------------------------------
# Combined TARGET + CONTROL eventlog for Sankey
# Control = within-cohort non-target (target=0): no first ED (HCG) within 21d of drug for this cohort.
# -------------------------------------------------------------------

pgx_df_control <- pgx_df %>% filter(target == 0)
cat("Loaded ", nrow(pgx_df_control), " within-cohort control events (target=0, no HCG) for ", cohort_name,
    " age_band=", age_band, "\n", sep = "")

pgx_df_all <- bind_rows(
  pgx_df_target1 %>% mutate(group = "target"),
  pgx_df_control %>% mutate(group = "control")
)

pgx_df_all_long <- pgx_df_all %>%
  transmute(
    mi_person_key,
    event_date,
    group,
    drug_name,
    primary_icd_diagnosis_code,
    two_icd_diagnosis_code,
    three_icd_diagnosis_code,
    four_icd_diagnosis_code,
    five_icd_diagnosis_code,
    six_icd_diagnosis_code,
    seven_icd_diagnosis_code,
    eight_icd_diagnosis_code,
    nine_icd_diagnosis_code,
    ten_icd_diagnosis_code,
    procedure_code
  ) %>%
  mutate(across(
    c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    as.character
  )) %>%
  pivot_longer(
    cols = c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    names_to = "source",
    values_to = "code"
  ) %>%
  filter(!is.na(code), code != "", code != "NA") %>%
  {
    if (length(allowed_codes) > 0) {
      dplyr::filter(., code %in% allowed_codes)
    } else {
      .
    }
  } %>%
  mutate(
    activity = dplyr::case_when(
      source == "drug_name" ~ paste0("DRUG:", code),
      grepl("icd_diagnosis_code", source) ~ paste0("ICD:", code),
      source == "procedure_code" ~ paste0("CPT:", code),
      TRUE ~ code
    ),
    timestamp = suppressWarnings(to_ts(event_date))
  ) %>%
  filter(!is.na(timestamp))

sankey_eventlog <- pgx_df_all_long %>%
  transmute(
    case_id              = mi_person_key,
    activity             = activity,
    timestamp            = timestamp,
    group                = group,
    activity_instance_id = dplyr::row_number(),
    lifecycle_id         = dplyr::case_when(
      grepl("^DRUG:", activity) ~ "Drug",
      grepl("^ICD:",  activity) ~ "ICD",
      grepl("^CPT:",  activity) ~ "CPT",
      TRUE ~ "Other"
    ),
    resource_id          = "Patient"
  ) %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id",
    timestamp            = "timestamp"
  )

cat("Combined TARGET + CONTROL sankey_eventlog created.\n")
print(sankey_eventlog)

# -------------------------------------------------------------------
# Pre-HCG (before first ED visit within 21 days of drug event) sequences
# Target = first ED visit (HCG Setting) within 21 days of a prescription drug event;
# identified by hcg_line or first_ed_non_opioid_date in model_events.
# -------------------------------------------------------------------

cat("\n--- Pre-HCG (before first ED visit within 21d of drug event) analysis ---\n")

# Build target date per case from model_events (hcg_line or first_ed_non_opioid_date)
target_date_map <- NULL
if (has_hcg_line && "hcg_line" %in% names(pgx_df_target1)) {
  hcg_ed <- pgx_df_target1 %>%
    filter(!is.na(hcg_line), hcg_line %in% ed_hcg_lines, !is.na(event_date))
  hcg_ed$event_date_parsed <- suppressWarnings(as.Date(hcg_ed$event_date))
  target_date_map <- hcg_ed %>%
    filter(!is.na(event_date_parsed)) %>%
    group_by(mi_person_key) %>%
    summarise(target_date = min(event_date_parsed, na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(target_date), is.finite(as.numeric(target_date))) %>%
    rename(case_id = mi_person_key)
  cat("Target dates from hcg_line (ED visits): ", nrow(target_date_map), " cases.\n", sep = "")
} else if (has_first_ed_date && "first_ed_non_opioid_date" %in% names(pgx_df_target1)) {
  fed <- pgx_df_target1 %>%
    filter(!is.na(first_ed_non_opioid_date))
  fed$first_ed_parsed <- suppressWarnings(as.Date(fed$first_ed_non_opioid_date))
  target_date_map <- fed %>%
    filter(!is.na(first_ed_parsed)) %>%
    group_by(mi_person_key) %>%
    summarise(target_date = min(first_ed_parsed, na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(target_date), is.finite(as.numeric(target_date))) %>%
    rename(case_id = mi_person_key)
  cat("Target dates from first_ed_non_opioid_date: ", nrow(target_date_map), " cases.\n", sep = "")
}
if (is.null(target_date_map) || nrow(target_date_map) == 0L) {
  target_date_map <- data.frame(case_id = character(0), target_date = as.Date(integer(0)))
  if (!has_hcg_line && !has_first_ed_date) {
    cat("WARNING: model_events has no hcg_line or first_ed_non_opioid_date; pre-HCG events will be empty.\n")
    cat("  Ensure model_events includes HCG target columns (e.g. from 4_model_data or 3b with HCG). keys_received (schema): ", paste(head(keys_received_schema, 30), collapse = ", "), "\n", sep = "")
  } else {
    cat("WARNING: hcg_line/first_ed_non_opioid_date present but 0 target cases (no ED rows in ed_hcg_lines with valid event_date, or no first_ed_non_opioid_date). pre-HCG events will be empty.\n")
    cat("  keys_received (schema): ", paste(head(keys_received_schema, 30), collapse = ", "), "\n", sep = "")
  }
}

ev_all <- as.data.frame(target_eventlog) %>%
  left_join(target_date_map, by = "case_id") %>%
  arrange(case_id, timestamp) %>%
  group_by(case_id) %>%
  mutate(
    event_index = row_number(),
    is_target_event = !is.na(target_date) & as.Date(timestamp) >= target_date,
    has_target = any(!is.na(target_date)),
    first_target_index = ifelse(has_target, min(event_index[is_target_event], na.rm = TRUE), NA_integer_)
  ) %>%
  ungroup()

# Pre-target = events strictly before the first HCG (exclude the HCG event itself)
events_pre_target <- ev_all %>%
  filter(!is.na(first_target_index), event_index < first_target_index) %>%
  mutate(
    activity_instance_id = row_number(),
    lifecycle_id         = dplyr::case_when(
      grepl("^DRUG:", activity) ~ "Drug",
      grepl("^ICD:",  activity) ~ "ICD",
      grepl("^CPT:",  activity) ~ "CPT",
      TRUE ~ "Other"
    ),
    resource_id          = "Patient"
  )

pre_target_eventlog <- events_pre_target %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id",
    timestamp            = "timestamp"
  )

cat("Pre-HCG eventlog summary:\n")
print(pre_target_eventlog)

n_pre <- nrow(events_pre_target)
# 1) Trace explorer: save as PNG for dashboard
if (n_pre > 0L) {
  p_te_pre <- tryCatch(
    trace_explorer(pre_target_eventlog, n_traces = 30, label_size = 3.0, abbreviate = TRUE,
                   coverage_labels = c("relative", "absolute"), show_labels = TRUE),
    error = function(e) { cat(" [skip] trace_explorer(pre-HCG):", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(p_te_pre)) {
    ggsave(file.path(plots_dir, sprintf("%s_%s_trace_explorer_pre_hcg.png", cohort_name, age_band_fname)),
           plot = p_te_pre, width = 16, height = 12, dpi = 300)
    print(p_te_pre)
  }
} else {
  cat(" [skip] trace_explorer(pre-HCG): no pre-HCG events\n")
}

# 2) Drug-only sequences before HCG
pre_drug_sequences <- as.data.frame(pre_target_eventlog) %>%
  arrange(case_id, timestamp) %>%
  filter(grepl("^DRUG:", activity)) %>%
  group_by(case_id) %>%
  summarise(
    drug_sequence = list(activity),
    .groups = "drop"
  )

cat("Sample pre-HCG drug-only sequences:\n")
print(head(pre_drug_sequences))

# 3) Process map for pre-HCG trajectories (skip if empty to avoid errors)
if (n_pre > 0L) {
  tryCatch(
    process_map(pre_target_eventlog, type = "frequency"),
    error = function(e) cat(" [skip] process_map(pre-HCG):", conditionMessage(e), "\n")
  )
}

# 4) Per-patient pre-HCG features
pre_patient_features <- as.data.frame(pre_target_eventlog) %>%
  arrange(case_id, timestamp) %>%
  group_by(case_id) %>%
  summarise(
    pre_n_events            = n(),
    pre_n_drug_events       = sum(grepl("^DRUG:", activity)),
    pre_n_icd_events        = sum(grepl("^ICD:", activity)),
    pre_n_cpt_events        = sum(grepl("^CPT:", activity)),
    pre_n_unique_activities = n_distinct(activity),
    .groups = "drop"
  )

save_bupar_csv(
  pre_patient_features,
  sprintf("%s_%s_train_target_pre_hcg_patient_features_bupar.csv", cohort_name, age_band_fname)
)

# -------------------------------------------------------------------
# Time-to-HCG and time-window features (per patient)
# -------------------------------------------------------------------

# Use target_date_map (first ED within 21d of drug) for target_time; fallback to first event if no map
ev_df <- as.data.frame(target_eventlog)
n_ev <- nrow(ev_df)
if (n_ev == 0L) {
  target_times <- data.frame(
    case_id = character(0),
    first_time = as.POSIXct(character(0)),
    target_time = as.POSIXct(character(0))
  )
} else if (!is.null(target_date_map) && nrow(target_date_map) > 0L) {
  target_times <- ev_df %>%
    arrange(case_id, timestamp) %>%
    group_by(case_id) %>%
    summarise(first_time = min(timestamp, na.rm = TRUE), .groups = "drop") %>%
    filter(is.finite(as.numeric(first_time))) %>%
    inner_join(
      target_date_map %>% mutate(target_time = as.POSIXct(target_date)),
      by = "case_id"
    )
} else {
  target_times <- ev_df %>%
    arrange(case_id, timestamp) %>%
    group_by(case_id) %>%
    summarise(
      first_time  = min(timestamp, na.rm = TRUE),
      target_time = min(timestamp, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(is.finite(as.numeric(first_time)))
}

pre_events_with_t <- as.data.frame(pre_target_eventlog) %>%
  inner_join(target_times, by = "case_id") %>%
  mutate(
    dt_days = as.numeric(difftime(target_time, timestamp, units = "days"))
  )

if (nrow(pre_events_with_t) > 0L) {
  hcg_time_features <- pre_events_with_t %>%
    group_by(case_id, target_time, first_time) %>%
    summarise(
      time_to_HCG_days        = as.numeric(max(dt_days, na.rm = TRUE)),
      n_events_30d            = sum(dt_days <= 30, na.rm = TRUE),
      n_events_90d            = sum(dt_days <= 90, na.rm = TRUE),
      n_events_180d           = sum(dt_days <= 180, na.rm = TRUE),
      n_drug_events_30d       = sum(dt_days <= 30 & grepl("^DRUG:", activity), na.rm = TRUE),
      n_drug_events_90d       = sum(dt_days <= 90 & grepl("^DRUG:", activity), na.rm = TRUE),
      n_drug_events_180d      = sum(dt_days <= 180 & grepl("^DRUG:", activity), na.rm = TRUE),
      n_icd_events_30d        = sum(dt_days <= 30 & grepl("^ICD:", activity), na.rm = TRUE),
      n_icd_events_90d        = sum(dt_days <= 90 & grepl("^ICD:", activity), na.rm = TRUE),
      n_icd_events_180d      = sum(dt_days <= 180 & grepl("^ICD:", activity), na.rm = TRUE),
      n_cpt_events_30d       = sum(dt_days <= 30 & grepl("^CPT:", activity), na.rm = TRUE),
      n_cpt_events_90d       = sum(dt_days <= 90 & grepl("^CPT:", activity), na.rm = TRUE),
      n_cpt_events_180d      = sum(dt_days <= 180 & grepl("^CPT:", activity), na.rm = TRUE),
      .groups = "drop"
    )
} else {
  hcg_time_features <- data.frame(
    case_id = character(0),
    target_time = as.POSIXct(character(0)),
    first_time = as.POSIXct(character(0)),
    time_to_HCG_days = numeric(0),
    n_events_30d = integer(0), n_events_90d = integer(0), n_events_180d = integer(0),
    n_drug_events_30d = integer(0), n_drug_events_90d = integer(0), n_drug_events_180d = integer(0),
    n_icd_events_30d = integer(0), n_icd_events_90d = integer(0), n_icd_events_180d = integer(0),
    n_cpt_events_30d = integer(0), n_cpt_events_90d = integer(0), n_cpt_events_180d = integer(0)
  )
}

save_bupar_csv(
  hcg_time_features,
  sprintf("%s_%s_train_target_time_to_hcg_features_bupar.csv", cohort_name, age_band_fname)
)

# -------------------------------------------------------------------
# Target-only global process mining (traces + process matrix)
# -------------------------------------------------------------------

cat("\n--- Target-only global process mining ---\n")

# 1) Trace Explorer: save as PNG for dashboard
n_target <- nrow(as.data.frame(target_eventlog))
if (n_target > 0L) {
  p_te <- tryCatch(
    trace_explorer(target_eventlog, n_traces = 30, label_size = 3.0, abbreviate = TRUE,
                   coverage_labels = c("relative", "absolute"), show_labels = TRUE),
    error = function(e) { cat(" [skip] trace_explorer(target):", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(p_te)) {
    ggsave(file.path(plots_dir, sprintf("%s_%s_trace_explorer.png", cohort_name, age_band_fname)),
           plot = p_te, width = 16, height = 12, dpi = 300)
    print(p_te)
    
    # Create interactive Plotly version with year filtering
    tryCatch({
      # Extract year and compute trace frequencies by year
      trace_data_by_year <- target_eventlog %>%
        as.data.frame() %>%
        mutate(year = lubridate::year(timestamp)) %>%
        group_by(case_id, year) %>%
        arrange(timestamp) %>%
        summarise(trace = paste(activity, collapse = " -> "), .groups = "drop") %>%
        group_by(year, trace) %>%
        summarise(frequency = n(), .groups = "drop")
      
      # Get top 30 traces overall
      top_traces <- trace_data_by_year %>%
        group_by(trace) %>%
        summarise(total_freq = sum(frequency), .groups = "drop") %>%
        arrange(desc(total_freq)) %>%
        head(30) %>%
        pull(trace)
      
      # Filter to top traces and add "All Years" aggregation
      trace_filtered <- trace_data_by_year %>%
        filter(trace %in% top_traces)
      
      trace_all <- trace_filtered %>%
        group_by(trace) %>%
        summarise(frequency = sum(frequency), .groups = "drop") %>%
        mutate(year = 0)  # 0 = "All Years"
      
      trace_combined <- bind_rows(trace_all, trace_filtered) %>%
        arrange(desc(frequency))
      
      # Abbreviate traces for display (truncate long sequences)
      trace_combined <- trace_combined %>%
        mutate(trace_display = ifelse(nchar(trace) > 100, 
                                       paste0(substr(trace, 1, 97), "..."), 
                                       trace))
      
      # Create plotly figure with year buttons
      years <- c(0, 2016, 2017, 2018)
      year_labels <- c("All Years (2016-2018)", "2016", "2017", "2018")
      
      fig <- plot_ly()
      
      for (i in seq_along(years)) {
        yr <- years[i]
        data_year <- trace_combined %>%
          filter(year == yr) %>%
          arrange(desc(frequency)) %>%
          head(30)
        
        # Calculate relative coverage
        total_cases <- sum(data_year$frequency)
        data_year <- data_year %>%
          mutate(relative_pct = frequency / total_cases * 100,
                 cumulative_pct = cumsum(relative_pct))
        
        fig <- fig %>%
          add_trace(
            type = "bar",
            y = data_year$trace_display,
            x = data_year$frequency,
            name = "Trace Frequency",
            orientation = "h",
            visible = (i == 1),  # Show "All Years" by default
            marker = list(color = "#3b82f6"),
            text = sprintf("%.1f%% (cumulative: %.1f%%)", data_year$relative_pct, data_year$cumulative_pct),
            hovertemplate = paste0(
              "<b>Trace:</b> %{y}<br>",
              "<b>Frequency:</b> %{x}<br>",
              "<b>Coverage:</b> %{text}<br>",
              "<extra></extra>"
            )
          )
      }
      
      # Create year filter buttons
      updatemenus <- list(
        list(
          active = 0,
          type = "dropdown",
          x = 0.15,
          xanchor = "left",
          y = 1.08,
          yanchor = "top",
          buttons = lapply(seq_along(years), function(i) {
            visible_vec <- rep(FALSE, length(years))
            visible_vec[i] <- TRUE
            
            list(
              label = year_labels[i],
              method = "update",
              args = list(
                list(visible = visible_vec),
                list(title = paste("Top 30 Trace Patterns:", cohort_name, age_band, "-", year_labels[i]))
              )
            )
          })
        )
      )
      
      fig <- fig %>%
        layout(
          title = paste("Top 30 Trace Patterns:", cohort_name, age_band, "- All Years (2016-2018)"),
          xaxis = list(title = "Frequency (Number of Cases)"),
          yaxis = list(title = "", categoryorder = "total ascending"),
          updatemenus = updatemenus,
          margin = list(l = 300, r = 50, t = 100, b = 50),
          hovermode = "closest",
          height = 900
        )
      
      # Save interactive HTML as single self-contained file (no lib/ folder) for S3/dashboard
      saveWidget(
        fig,
        file.path(plots_dir, sprintf("%s_%s_trace_explorer_interactive.html", cohort_name, age_band_fname)),
        selfcontained = TRUE,
        libdir = NULL,
        title = paste("Trace Explorer:", cohort_name, age_band)
      )
      
      cat("Saved trace_explorer_interactive.html with year filtering\n")
    }, error = function(e) cat(" [skip] interactive trace explorer:", conditionMessage(e), "\n"))
  }
  # Performance spectrum (aggregated activity trace; requires psmineR)
  tryCatch({
    if (requireNamespace("psmineR", quietly = TRUE)) {
      p_ps <- target_eventlog %>% psmineR::ps_aggregated()
      ggsave(file.path(plots_dir, sprintf("%s_%s_performance_spectrum.png", cohort_name, age_band_fname)),
             plot = p_ps, width = 12, height = 8, dpi = 300)
      cat("Saved performance_spectrum.png\n")
    } else {
      cat(" [skip] performance_spectrum: psmineR not installed\n")
    }
  }, error = function(e) cat(" [skip] performance_spectrum:", conditionMessage(e), "\n"))
  # Frequency map (process_map with render = F then export_map to PNG; may be Plotly/HTML in some versions)
  freq_map_path <- file.path(plots_dir, sprintf("%s_%s_frequency_map.png", cohort_name, age_band_fname))
  tryCatch({
    pm_freq <- process_map(target_eventlog, type = "frequency", render = FALSE)
    if (exists("export_map", mode = "function")) {
      processmapR::export_map(pm_freq, file_name = freq_map_path, file_type = "png", width = 1200, height = 900)
      cat("Saved frequency_map.png\n")
    } else {
      cat(" [skip] frequency_map: export_map not found\n")
    }
  }, error = function(e) cat(" [skip] frequency_map:", conditionMessage(e), "\n"))
} else {
  cat(" [skip] trace_explorer(target): no events\n")
}

# Save trace summary as tabular output (bupaR::traces; edeaR::traces not exported in some versions)
traces_target <- tryCatch(
  bupaR::traces(target_eventlog),
  error = function(e) {
    cat(" [skip] traces(target_eventlog):", conditionMessage(e), "\n")
    data.frame(trace_id = character(0), trace = character(0), length = integer(0), first_activity = character(0), last_activity = character(0))
  }
)
save_bupar_csv(
  as.data.frame(traces_target),
  sprintf("%s_%s_train_target_traces_bupar.csv", cohort_name, age_band_fname)
)

# 2) Process Matrix and CSV export
# Use same event log as rest of script (activity = DRUG:/ICD:/CPT: from event log creation). Filter out NA in
# timestamp/activity/case_id so process_matrix does not hit "missing value where TRUE/FALSE needed".
target_eventlog_valid <- target_eventlog %>%
  filter(!is.na(timestamp), !is.na(activity), !is.na(case_id))
if (nrow(target_eventlog_valid) < nrow(target_eventlog)) {
  cat("BupaR: Dropped ", nrow(target_eventlog) - nrow(target_eventlog_valid),
      " rows with NA in timestamp/activity/case_id for process_matrix\n", sep = "")
}
if (n_target > 0L) {
  pm_target <- tryCatch(
    if (n_events(target_eventlog_valid) > 0L && n_cases(target_eventlog_valid) > 0L) {
      process_matrix(target_eventlog_valid, type = "frequency")
    } else {
      NULL
    },
    error = function(e) {
      cat("Note: process_matrix skipped due to error:", conditionMessage(e), "\n")
      cat("[ERROR_PARAMS] step=5_bupar step=process_matrix cohort_name=", cohort_name, " age_band=", age_band_fname, " error=", conditionMessage(e), "\n", sep = "")
      NULL
    })
  if (!is.null(pm_target)) {
    pm_target_df <- as.data.frame(pm_target)
    save_bupar_csv(
      pm_target_df,
      sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
    )
    
    # Generate process matrix heatmap visualization
    tryCatch({
      # Convert to long format for ggplot
      pm_long <- pm_target_df %>%
        tibble::rownames_to_column("from_activity") %>%
        tidyr::pivot_longer(cols = -from_activity, 
                           names_to = "to_activity", 
                           values_to = "frequency") %>%
        filter(frequency > 0)  # Remove zero-frequency cells
      
      # Filter to top activities (reduce clutter)
      top_activities <- target_eventlog %>%
        group_by(activity) %>%
        summarise(count = n(), .groups = "drop") %>%
        arrange(desc(count)) %>%
        head(25) %>%
        pull(activity)
      
      pm_long_filtered <- pm_long %>%
        filter(from_activity %in% top_activities,
               to_activity %in% top_activities)
      
      # Create heatmap
      p_matrix <- ggplot(pm_long_filtered, 
                        aes(x = to_activity, y = from_activity, fill = frequency)) +
        geom_tile(color = "white", size = 0.5) +
        geom_text(aes(label = ifelse(frequency > 0, frequency, "")), 
                 size = 2.5, color = "white") +
        scale_fill_viridis_c(option = "magma", 
                            trans = "log10",
                            breaks = c(1, 10, 100, 1000),
                            labels = scales::comma) +
        labs(title = paste("Process Matrix:", cohort_name, age_band),
             subtitle = "Frequency of directly-follows relationships (top 25 activities)",
             x = "To Activity →", 
             y = "← From Activity",
             fill = "Frequency\n(log scale)") +
        theme_minimal(base_size = 12) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
              axis.text.y = element_text(size = 9),
              panel.grid = element_blank(),
              legend.position = "right")
      
      ggsave(file.path(plots_dir, sprintf("%s_%s_process_matrix.png", cohort_name, age_band_fname)),
             plot = p_matrix, width = 16, height = 14, dpi = 300)
      
      cat("Saved process_matrix.png\n")
      
      # Create interactive Plotly version with year filtering
      tryCatch({
        # Extract year and compute process matrix by year
        pm_by_year <- target_eventlog %>%
          as.data.frame() %>%
          mutate(year = lubridate::year(timestamp)) %>%
          group_by(case_id, year) %>%
          arrange(timestamp) %>%
          mutate(next_activity = lead(activity)) %>%
          filter(!is.na(next_activity)) %>%
          ungroup() %>%
          group_by(year, activity, next_activity) %>%
          summarise(frequency = n(), .groups = "drop")
        
        # Get top 25 activities overall
        top_activities_pm <- target_eventlog %>%
          as.data.frame() %>%
          count(activity, sort = TRUE) %>%
          head(25) %>%
          pull(activity)
        
        # Filter and add "All Years"
        pm_filtered <- pm_by_year %>%
          filter(activity %in% top_activities_pm, next_activity %in% top_activities_pm)
        
        pm_all <- pm_filtered %>%
          group_by(activity, next_activity) %>%
          summarise(frequency = sum(frequency), .groups = "drop") %>%
          mutate(year = 0)
        
        pm_combined <- bind_rows(pm_all, pm_filtered)
        
        # Create plotly heatmap with year buttons
        years <- c(0, 2016, 2017, 2018)
        year_labels <- c("All Years (2016-2018)", "2016", "2017", "2018")
        
        fig <- plot_ly()
        
        for (i in seq_along(years)) {
          yr <- years[i]
          data_year <- pm_combined %>%
            filter(year == yr) %>%
            complete(activity = top_activities_pm, 
                     next_activity = top_activities_pm, 
                     fill = list(frequency = 0))
          
          # Create matrix (log scale)
          matrix_data <- data_year %>%
            mutate(log_freq = log10(frequency + 1)) %>%
            pivot_wider(id_cols = activity, names_from = next_activity, values_from = log_freq, values_fill = 0) %>%
            column_to_rownames("activity") %>%
            as.matrix()
          
          # Original frequency for hover
          freq_matrix <- data_year %>%
            pivot_wider(id_cols = activity, names_from = next_activity, values_from = frequency, values_fill = 0) %>%
            column_to_rownames("activity") %>%
            as.matrix()
          
          fig <- fig %>%
            add_trace(
              type = "heatmap",
              x = colnames(matrix_data),
              y = rownames(matrix_data),
              z = matrix_data,
              visible = (i == 1),  # Show "All Years" by default
              colorscale = "Magma",
              text = freq_matrix,
              hovertemplate = paste0(
                "<b>From:</b> %{y}<br>",
                "<b>To:</b> %{x}<br>",
                "<b>Frequency:</b> %{text}<br>",
                "<extra></extra>"
              ),
              colorbar = list(title = "log10(freq+1)")
            )
        }
        
        # Create year filter buttons
        updatemenus <- list(
          list(
            active = 0,
            type = "dropdown",
            x = 0.15,
            xanchor = "left",
            y = 1.05,
            yanchor = "top",
            buttons = lapply(seq_along(years), function(i) {
              visible_vec <- rep(FALSE, length(years))
              visible_vec[i] <- TRUE
              
              list(
                label = year_labels[i],
                method = "update",
                args = list(
                  list(visible = visible_vec),
                  list(title = paste("Process Matrix:", cohort_name, age_band, "-", year_labels[i]))
                )
              )
            })
          )
        )
        
        fig <- fig %>%
          layout(
            title = paste("Process Matrix:", cohort_name, age_band, "- All Years (2016-2018)"),
            xaxis = list(title = "To Activity →", tickangle = 45),
            yaxis = list(title = "← From Activity"),
            updatemenus = updatemenus,
            margin = list(l = 200, r = 50, t = 100, b = 150),
            height = 900,
            width = 1000
          )
        
        # Save interactive HTML as single self-contained file (no lib/ folder) for S3/dashboard
        saveWidget(
          fig,
          file.path(plots_dir, sprintf("%s_%s_process_matrix_interactive.html", cohort_name, age_band_fname)),
          selfcontained = TRUE,
          libdir = NULL,
          title = paste("Process Matrix:", cohort_name, age_band)
        )
        
        cat("Saved process_matrix_interactive.html with year filtering\n")
      }, error = function(e) cat(" [skip] interactive process matrix:", conditionMessage(e), "\n"))
    }, error = function(e) cat(" [skip] process_matrix heatmap:", conditionMessage(e), "\n"))
  }
  
  # Overall Activity Frequency plot with color coding
  tryCatch({
    target_activity_freq <- target_eventlog %>%
      mutate(activity_type = case_when(
        grepl("^DRUG:", activity) ~ "Drug",
        grepl("^ICD:", activity) ~ "Diagnosis",
        grepl("^CPT:", activity) ~ "Procedure",
        TRUE ~ "Other"
      )) %>%
      group_by(activity, activity_type) %>%
      summarise(count = n(), .groups = "drop") %>%
      arrange(desc(count)) %>%
      head(40)
    
    p_activity_freq <- ggplot(target_activity_freq, 
                aes(x = reorder(activity, count), y = count, fill = activity_type)) +
      geom_col() +
      coord_flip() +
      scale_fill_manual(values = c("Drug" = "#3b82f6", 
                                   "Diagnosis" = "#ef4444", 
                                   "Procedure" = "#10b981",
                                   "Other" = "#64748b"),
                       name = "Event Type") +
      labs(title = paste("Overall Activity Frequency:", cohort_name, age_band),
           subtitle = "Top 40 activities by frequency",
           x = NULL, y = "Frequency") +
      theme_minimal(base_size = 13) +
      theme(axis.text.y = element_text(size = 10),
            legend.position = "top",
            panel.grid.minor = element_blank())
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_overall_activity_frequency.png", cohort_name, age_band_fname)),
           plot = p_activity_freq, width = 14, height = 11, dpi = 300)
    
    cat("Saved overall_activity_frequency.png\n")
    
    # Create interactive Plotly version with year filtering
    tryCatch({
      # Extract year from eventlog and compute frequency by year
      activity_freq_by_year <- target_eventlog %>%
        as.data.frame() %>%
        mutate(year = lubridate::year(timestamp),
               activity_type = case_when(
                 grepl("^DRUG:", activity) ~ "Drug",
                 grepl("^ICD:", activity) ~ "Diagnosis",
                 grepl("^CPT:", activity) ~ "Procedure",
                 TRUE ~ "Other"
               )) %>%
        group_by(year, activity, activity_type) %>%
        summarise(count = n(), .groups = "drop")
      
      # Get top 40 activities overall (across all years)
      top_activities <- activity_freq_by_year %>%
        group_by(activity) %>%
        summarise(total_count = sum(count), .groups = "drop") %>%
        arrange(desc(total_count)) %>%
        head(40) %>%
        pull(activity)
      
      # Filter to top activities and add "All Years" aggregation
      activity_freq_filtered <- activity_freq_by_year %>%
        filter(activity %in% top_activities)
      
      activity_freq_all <- activity_freq_filtered %>%
        group_by(activity, activity_type) %>%
        summarise(count = sum(count), .groups = "drop") %>%
        mutate(year = 0)  # Use 0 to represent "All Years"
      
      activity_freq_combined <- bind_rows(activity_freq_all, activity_freq_filtered) %>%
        arrange(activity, year)
      
      # Create color mapping
      colors <- c("Drug" = "#3b82f6", "Diagnosis" = "#ef4444", "Procedure" = "#10b981", "Other" = "#64748b")
      
      # Create traces for each year
      years <- c(0, 2016, 2017, 2018)
      year_labels <- c("All Years (2016-2018)", "2016", "2017", "2018")
      
      # Build plotly figure with buttons
      fig <- plot_ly()
      
      for (i in seq_along(years)) {
        yr <- years[i]
        data_year <- activity_freq_combined %>%
          filter(year == yr) %>%
          arrange(desc(count)) %>%
          head(40)
        
        for (act_type in unique(data_year$activity_type)) {
          data_type <- data_year %>% filter(activity_type == act_type)
          
          fig <- fig %>%
            add_trace(
              type = "bar",
              y = data_type$activity,
              x = data_type$count,
              name = act_type,
              marker = list(color = colors[act_type]),
              orientation = "h",
              visible = (i == 1),  # Show "All Years" by default
              legendgroup = act_type,
              showlegend = (i == 1),
              hovertemplate = paste0(
                "<b>Activity:</b> %{y}<br>",
                "<b>Type:</b> ", act_type, "<br>",
                "<b>Count:</b> %{x}<br>",
                "<extra></extra>"
              )
            )
        }
      }
      
      # Create year filter buttons
      updatemenus <- list(
        list(
          active = 0,
          type = "dropdown",
          x = 0.15,
          xanchor = "left",
          y = 1.15,
          yanchor = "top",
          buttons = lapply(seq_along(years), function(i) {
            visible_vec <- rep(FALSE, length(years) * length(unique(activity_freq_combined$activity_type)))
            start_idx <- (i - 1) * length(unique(activity_freq_combined$activity_type)) + 1
            end_idx <- i * length(unique(activity_freq_combined$activity_type))
            visible_vec[start_idx:end_idx] <- TRUE
            
            list(
              label = year_labels[i],
              method = "update",
              args = list(
                list(visible = visible_vec),
                list(title = paste("Activity Frequency:", cohort_name, age_band, "-", year_labels[i]))
              )
            )
          })
        )
      )
      
      fig <- fig %>%
        layout(
          title = paste("Activity Frequency:", cohort_name, age_band, "- All Years (2016-2018)"),
          xaxis = list(title = "Frequency"),
          yaxis = list(title = "", categoryorder = "total ascending"),
          barmode = "stack",
          updatemenus = updatemenus,
          margin = list(l = 200, r = 50, t = 100, b = 50),
          legend = list(orientation = "h", y = 1.05, x = 0.5, xanchor = "center"),
          hovermode = "closest"
        )
      
      # Save interactive HTML as single self-contained file (no lib/ folder) for S3/dashboard
      saveWidget(
        fig,
        file.path(plots_dir, sprintf("%s_%s_activity_frequency_interactive.html", cohort_name, age_band_fname)),
        selfcontained = TRUE,
        libdir = NULL,
        title = paste("Activity Frequency:", cohort_name, age_band)
      )
      
      cat("Saved activity_frequency_interactive.html with year filtering\n")
    }, error = function(e) cat(" [skip] interactive activity frequency:", conditionMessage(e), "\n"))
  }, error = function(e) cat(" [skip] overall_activity_frequency:", conditionMessage(e), "\n"))
  
  tryCatch(
    process_map(target_eventlog, type = "frequency"),
    error = function(e) cat(" [skip] process_map(target):", conditionMessage(e), "\n")
  )
}

# Close the cohort-specific PDF device if it is still open so that any
# base graphics output is written under the correct cohort directory.
if (grDevices::dev.cur() > 1) {
  grDevices::dev.off()
}

cat("\n=== bupaR analysis for non_opioid_ed ", age_band, " completed. ===\n", sep = "")



