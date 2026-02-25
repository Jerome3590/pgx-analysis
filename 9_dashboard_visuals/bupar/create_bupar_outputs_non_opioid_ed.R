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

# Windows-only: add user library to .libPaths; on EC2/Linux use default
if (.Platform$OS.type == "windows") {
  user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.5")
  if (dir.exists(user_lib)) {
    .libPaths(c(user_lib, .libPaths()))
  }
}

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

# Ensure htmlwidgets (and plotly) available for interactive HTML output
if (!requireNamespace("htmlwidgets", quietly = TRUE)) {
  stop("Package 'htmlwidgets' is required for interactive HTML plots. Install with: install.packages(\"htmlwidgets\")")
}
if (!requireNamespace("plotly", quietly = TRUE)) {
  stop("Package 'plotly' is required for interactive HTML plots. Install with: install.packages(\"plotly\")")
}

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
# Identified by hcg_line or first_o11_p_date in model_events, NOT by an ICD code "HCG".
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
  } else {
    stop("Model data not found for cohort ", cohort_name, " age_band ", age_band,
         " (tried age_band=", age_band_fname, " and age_band=", age_band, ")")
  }
}

cat("Project root:         ", project_root, "\n", sep = "")
cat("Model data path:      ", model_data_path, "\n", sep = "")
cat("Model data SQL:       ", substr(model_data_from_sql, 1, 100), "...\n", sep = "")

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

# Abbreviate activity/trace for display: first 3 characters of Drug/ICD/CPT name; full name in hover/legend
first_three_activity <- function(act) {
  if (grepl("^DRUG:", act)) return(substr(sub("^DRUG:", "", act), 1, 3))
  if (grepl("^ICD:", act))  return(substr(sub("^ICD:", "", act), 1, 3))
  if (grepl("^CPT:", act))  return(substr(sub("^CPT:", "", act), 1, 3))
  substr(act, 1, 3)
}
first_three_trace <- function(tr) {
  parts <- strsplit(tr, " -> ", fixed = TRUE)[[1]]
  paste(vapply(parts, first_three_activity, character(1)), collapse = " -> ")
}

# Activity-frequency aggregated plot: one bar per unique activity, ordered by frequency (fix trace explorer "tiny tiles")
# Aligned to research N2/N6: "Which activities appear in pathways leading to target?" (aggregated activity frequency; see README_bupar_dashboard_visualizations.md)
trace_explorer_activity_frequency_plot <- function(eventlog_obj, title_prefix, top_n = 30) {
  df <- as.data.frame(eventlog_obj)
  if (is.null(df) || !"activity" %in% names(df) || nrow(df) == 0) return(NULL)
  type_from_activity <- function(a) {
    if (grepl("^DRUG:", a)) return("Drug")
    if (grepl("^ICD:", a))  return("Diagnosis")
    if (grepl("^CPT:", a))  return("Procedure")
    "Other"
  }
  agg <- df %>%
    count(activity, name = "freq") %>%
    arrange(desc(freq)) %>%
    mutate(type = vapply(activity, type_from_activity, character(1), USE.NAMES = FALSE))
  if (nrow(agg) == 0) return(NULL)
  if (nrow(agg) > top_n) {
    other_count <- agg %>% slice((top_n + 1):n()) %>% pull(freq) %>% sum()
    other_label <- paste0("Other (", nrow(agg) - top_n, " activities)")
    agg <- bind_rows(agg %>% slice(1:top_n), data.frame(activity = other_label, freq = other_count, type = "Other", stringsAsFactors = FALSE))
  }
  agg <- agg %>% mutate(activity = reorder(activity, freq))
  type_colors <- c("Drug" = "#3b82f6", "Diagnosis" = "#ef4444", "Procedure" = "#10b981", "Other" = "#64748b")
  ggplot(agg, aes(x = activity, y = freq, fill = type)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = type_colors, name = "Event Type") +
    labs(
      x = "Activity", y = "Frequency",
      title = paste0(title_prefix, " — Activity frequency (aggregated)"),
      subtitle = "Which activities appear in pathways? (ordered by frequency; supports research N2/N6)"
    ) +
    theme_bw() +
    theme(legend.position = "top", plot.subtitle = element_text(size = 9))
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

# Normalized target config: cohort uses target date column (not an ICD code) for pre-target split
target_code_label   <- "ED visit (HCG)"           # no single ICD; target = first ED within 21d of drug
# Step 4 writes first_o11_p_date; accept legacy first_ed_non_opioid_date for older parquets
target_date_col_canonical <- "first_o11_p_date"
target_date_columns <- c(target_date_col_canonical, "first_ed_non_opioid_date", "hcg_line")
schema_info <- dbGetQuery(con, paste0("DESCRIBE SELECT * FROM ", model_data_from_sql, " LIMIT 0"))
keys_received_schema <- schema_info$column_name
keys_expected_target <- c("event_date", "mi_person_key", "target", target_date_columns)
has_first_o11_p <- "first_o11_p_date" %in% keys_received_schema
has_first_ed_date <- "first_ed_non_opioid_date" %in% keys_received_schema
has_target_date_col <- has_first_o11_p || has_first_ed_date
has_hcg_line <- "hcg_line" %in% keys_received_schema
cat("cohort=", cohort_name, " target_code=", target_code_label, " target_date_columns_expected=", paste(target_date_columns, collapse = ", "), "\n", sep = "")
cat("keys_expected (for target date): ", paste(keys_expected_target, collapse = ", "), "\n", sep = "")
cat("keys_received (model_events schema): ", paste(keys_received_schema, collapse = ", "), "\n", sep = "")
if (!has_target_date_col && !has_hcg_line) cat("WARNING: no target date column (first_o11_p_date or first_ed_non_opioid_date) or hcg_line in schema.\n")
if (has_first_o11_p) cat("Model data has first_o11_p_date (use for target date after step 4 leakage removal).\n")
if (has_first_ed_date && !has_first_o11_p) cat("Model data has legacy first_ed_non_opioid_date (use for target date).\n")
if (has_hcg_line) cat("Model data has hcg_line column (fallback when target date column absent).\n")

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
# Load allowed code set: SHAP/FFA combined (required prerequisite; we never use all codes).
# -------------------------------------------------------------------

allowed_codes_shap_ffa_path <- file.path(
  bup_ar_output_root,
  sprintf("allowed_codes_shap_ffa_%s_%s.json", cohort_name, age_band_fname)
)

if (!file.exists(allowed_codes_shap_ffa_path)) {
  stop("SHAP/FFA allowed codes file is required (prerequisite). Not found: ", allowed_codes_shap_ffa_path,
       ". Generate the combined allowed_codes file before running BupaR.")
}
allowed_codes <- fromJSON(allowed_codes_shap_ffa_path)
if (!is.character(allowed_codes)) allowed_codes <- as.character(allowed_codes)
if (length(allowed_codes) == 0L) {
  stop("SHAP/FFA allowed codes file is empty: ", allowed_codes_shap_ffa_path, ". Cannot run BupaR without allowed codes.")
}
cat("Loaded ", length(allowed_codes), " allowed codes from SHAP/FFA (causal feature importances).\n", sep = "")
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

# Event log = dataset filtered by SHAP/FFA allowed codes, then by valid dates
pgx_df_target1_long <- pgx_df_target1_long %>%
  filter(code %in% allowed_codes) %>%
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
  filter(code %in% allowed_codes) %>%
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
# identified by hcg_line or first_o11_p_date (or legacy first_ed_non_opioid_date) in model_events.
# -------------------------------------------------------------------

cat("\n--- Pre-HCG (before first ED visit within 21d of drug event) analysis ---\n")

# Build target date per case from model_events.
# Step 4 (4_model_data) removes target leakage: only events before target date are kept;
# we use first_o11_p_date (or legacy first_ed_non_opioid_date) on every target row; hcg_line would yield 0 cases.
target_date_map <- NULL
target_date_col_used <- NULL
if (has_first_o11_p && "first_o11_p_date" %in% names(pgx_df_target1)) {
  target_date_col_used <- "first_o11_p_date"
} else if (has_first_ed_date && "first_ed_non_opioid_date" %in% names(pgx_df_target1)) {
  target_date_col_used <- "first_ed_non_opioid_date"
}
if (!is.null(target_date_col_used)) {
  fed <- pgx_df_target1 %>%
    filter(!is.na(.data[[target_date_col_used]]))
  fed$first_ed_parsed <- suppressWarnings(as.Date(fed[[target_date_col_used]]))
  fed_filtered <- fed %>% filter(!is.na(first_ed_parsed))
  if (nrow(fed_filtered) > 0) {
    target_date_map <- fed_filtered %>%
      group_by(mi_person_key) %>%
      summarise(target_date = min(first_ed_parsed, na.rm = TRUE), .groups = "drop") %>%
      filter(!is.na(target_date), is.finite(as.numeric(target_date))) %>%
      rename(case_id = mi_person_key)
  } else {
    target_date_map <- data.frame(case_id = character(0), target_date = as.Date(integer(0)))
  }
  cat("Target dates from ", target_date_col_used, ": ", nrow(target_date_map), " cases.\n", sep = "")
}
if ((is.null(target_date_map) || nrow(target_date_map) == 0L) && has_hcg_line && "hcg_line" %in% names(pgx_df_target1)) {
  # Fallback: hcg_line (e.g. when using 3b model_events that still contain ED rows)
  hcg_ed <- pgx_df_target1 %>%
    filter(!is.na(hcg_line), hcg_line %in% ed_hcg_lines, !is.na(event_date))
  hcg_ed$event_date_parsed <- suppressWarnings(as.Date(hcg_ed$event_date))
  hcg_filtered <- hcg_ed %>% filter(!is.na(event_date_parsed))
  if (nrow(hcg_filtered) > 0) {
    target_date_map <- hcg_filtered %>%
      group_by(mi_person_key) %>%
      summarise(target_date = min(event_date_parsed, na.rm = TRUE), .groups = "drop") %>%
      filter(!is.na(target_date), is.finite(as.numeric(target_date))) %>%
      rename(case_id = mi_person_key)
  } else {
    target_date_map <- data.frame(case_id = character(0), target_date = as.Date(integer(0)))
  }
  cat("Target dates from hcg_line (ED visits): ", nrow(target_date_map), " cases.\n", sep = "")
}
if (is.null(target_date_map) || nrow(target_date_map) == 0L) {
  target_date_map <- data.frame(case_id = character(0), target_date = as.Date(integer(0)))
  if (!has_hcg_line && !has_target_date_col) {
    cat("WARNING: model_events has no hcg_line or target date column (first_o11_p_date / first_ed_non_opioid_date); pre-HCG events will be empty.\n")
    cat("  Ensure model_events includes target date column (e.g. from 4_model_data). keys_received (schema): ", paste(head(keys_received_schema, 30), collapse = ", "), "\n", sep = "")
  } else {
    cat("WARNING: target date column/hcg_line present but 0 target cases. pre-HCG events will be empty.\n")
    cat("  For 4_model_data, target date comes from first_o11_p_date (ED row removed by leakage). keys_received (schema): ", paste(head(keys_received_schema, 30), collapse = ", "), "\n", sep = "")
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

# Post-HCG = events at or after first target (ED visit)
events_post_target <- ev_all %>%
  filter(!is.na(first_target_index), event_index >= first_target_index) %>%
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

post_target_eventlog <- events_post_target %>%
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
cat("Post-HCG eventlog summary:\n")
print(post_target_eventlog)

n_pre <- nrow(events_pre_target)
# 1) Trace explorer: frequency-aggregated activity plot (one bar per activity, ordered by frequency; N2/N6)
if (n_pre > 0L) {
  p_te_pre <- tryCatch(
    trace_explorer_activity_frequency_plot(pre_target_eventlog, "Pre-HCG", top_n = 30),
    error = function(e) { cat(" [skip] trace_explorer(pre-HCG):", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(p_te_pre)) {
    ggsave(file.path(plots_dir, sprintf("%s_%s_trace_explorer_pre_hcg.png", cohort_name, age_band_fname)),
           plot = p_te_pre, width = 14, height = 10, dpi = 300)
    cat("Saved trace_explorer_pre_hcg.png (activity frequency aggregated)\n")
  }
  # Export pre-target activity frequency as JSON (dashboard bar chart; no HTML)
  tryCatch({
    pre_freq_by_year <- pre_target_eventlog %>%
      as.data.frame() %>%
      mutate(year = lubridate::year(timestamp),
             activity_type = case_when(
               grepl("^DRUG:", activity) ~ "Drug",
               grepl("^ICD:", activity) ~ "Diagnosis",
               grepl("^CPT:", activity) ~ "Procedure",
               TRUE ~ "Other"
             )) %>%
      group_by(year, activity, activity_type) %>%
      summarise(count = n(), .groups = "drop") %>%
      mutate(activity_short = vapply(activity, first_three_activity, character(1)))
    pre_freq_all <- pre_freq_by_year %>%
      group_by(activity, activity_type, activity_short) %>%
      summarise(count = sum(count), .groups = "drop") %>%
      mutate(year = 0)
    pre_freq_combined <- bind_rows(pre_freq_all, pre_freq_by_year) %>%
      select(activity, activity_short, activity_type, count, year) %>%
      arrange(year, desc(count))
    year_labels_list <- list("0" = "All Years (2016-2018)", "2016" = "2016", "2017" = "2017", "2018" = "2018")
    pre_json_path <- file.path(plots_dir, sprintf("%s_%s_pre_target_activity_frequency.json", cohort_name, age_band_fname))
    jsonlite::write_json(list(year_labels = year_labels_list, data = pre_freq_combined), pre_json_path, dataframe = "rows", pretty = TRUE)
    cat("Saved pre_target_activity_frequency.json\n")
  }, error = function(e) cat(" [skip] pre_target_activity_frequency.json:", conditionMessage(e), "\n"))
  # Trace explorer interactive (pre-HCG only)
  tryCatch({
    trace_data_by_year <- pre_target_eventlog %>%
      as.data.frame() %>%
      mutate(year = lubridate::year(timestamp)) %>%
      group_by(case_id, year) %>%
      arrange(timestamp) %>%
      summarise(trace = paste(activity, collapse = " -> "), .groups = "drop") %>%
      group_by(year, trace) %>%
      summarise(frequency = n(), .groups = "drop")

    top_traces <- trace_data_by_year %>%
      group_by(trace) %>%
      summarise(total_freq = sum(frequency), .groups = "drop") %>%
      arrange(desc(total_freq)) %>%
      head(30) %>%
      pull(trace)

    trace_filtered <- trace_data_by_year %>% filter(trace %in% top_traces)
    trace_all <- trace_filtered %>%
      group_by(trace) %>%
      summarise(frequency = sum(frequency), .groups = "drop") %>%
      mutate(year = 0)

    trace_combined <- bind_rows(trace_all, trace_filtered) %>%
      arrange(desc(frequency)) %>%
      mutate(
        trace_display = ifelse(nchar(trace) > 100, paste0(substr(trace, 1, 97), "..."), trace),
        trace_display_short = vapply(trace, first_three_trace, character(1))
      )

    years <- c(0, 2016, 2017, 2018)
    year_labels <- c("All Years (2016-2018)", "2016", "2017", "2018")

    years_with_data <- integer(0)
    year_labels_with_data <- character(0)
    for (idx in seq_along(years)) {
      n <- nrow(trace_combined %>% filter(year == years[idx]) %>% head(30))
      if (n > 0L) {
        years_with_data <- c(years_with_data, years[idx])
        year_labels_with_data <- c(year_labels_with_data, year_labels[idx])
      }
    }

    if (length(years_with_data) > 0L) {
      fig <- plot_ly()
      traces_added <- 0L
      year_labels_added <- character(0)

      for (k in seq_along(years_with_data)) {
        yr <- years_with_data[k]
        data_year <- trace_combined %>%
          filter(year == yr) %>%
          arrange(desc(frequency)) %>%
          head(30)

        if (nrow(data_year) == 0L) next
        data_year <- data_year %>%
          filter(complete.cases(trace_display_short, trace_display, frequency)) %>%
          filter(nchar(as.character(trace_display_short)) >= 0L, frequency > 0)
        if (nrow(data_year) == 0L) next

        total_cases <- sum(data_year$frequency, na.rm = TRUE)
        total_cases <- if (total_cases <= 0) 1 else total_cases
        data_year <- data_year %>%
          mutate(relative_pct = frequency / total_cases * 100, cumulative_pct = cumsum(relative_pct))
        y_vals <- as.character(data_year$trace_display_short)
        x_vals <- as.numeric(data_year$frequency)
        if (length(y_vals) == 0L || length(x_vals) == 0L) next

        fig <- fig %>%
          add_trace(
            type = "bar",
            y = y_vals,
            x = x_vals,
            name = "Trace Frequency",
            customdata = as.character(data_year$trace_display),
            orientation = "h",
            visible = (traces_added == 0L),
            marker = list(color = "#3b82f6"),
            text = sprintf("%.1f%% (cumulative: %.1f%%)", data_year$relative_pct, data_year$cumulative_pct),
            hovertemplate = paste0(
              "<b>Trace:</b> %{customdata}<br><b>Frequency:</b> %{x}<br><b>Coverage:</b> %{text}<br><extra></extra>"
            )
          )

        year_labels_added <- c(year_labels_added, year_labels_with_data[k])
        traces_added <- traces_added + 1L
      }

      n_traces <- traces_added
      if (n_traces > 0L) {
        updatemenus <- list(
          list(
            active = 0,
            type = "dropdown",
            x = 0.15, xanchor = "left", y = 1.08, yanchor = "top",
            buttons = lapply(seq_len(n_traces), function(k) {
              visible_vec <- rep(FALSE, n_traces)
              visible_vec[k] <- TRUE
              list(
                label = year_labels_added[k],
                method = "update",
                args = list(
                  list(visible = visible_vec),
                  list(title = paste("Pre-HCG Trace Patterns:", cohort_name, age_band, "-", year_labels_added[k]))
                )
              )
            })
          )
        )

        fig <- fig %>%
          layout(
            title = list(text = paste("Pre-HCG Trace Patterns:", cohort_name, age_band, "-", year_labels_added[1L])),
            xaxis = list(title = list(text = "Frequency (Number of Cases)"), type = "linear", zeroline = TRUE),
            yaxis = list(title = list(text = ""), type = "category", categoryorder = "total ascending"),
            updatemenus = updatemenus,
            margin = list(l = 300, r = 50, t = 100, b = 50),
            hovermode = "closest",
            height = 900
          )

        trace_html_path <- file.path(plots_dir, sprintf("%s_%s_trace_explorer_interactive.html", cohort_name, age_band_fname))
        # Production HTML: selfcontained = TRUE (same pattern as fpgrowth). See 9_dashboard_visuals/HTML_PRODUCTION_PATTERN.md
        saveWidget(fig, trace_html_path, selfcontained = TRUE,
                  title = paste("Trace Explorer (Pre-HCG):", cohort_name, age_band))
        cat("Saved trace_explorer_interactive.html (pre-HCG); path=", trace_html_path, "\n", sep = "")
        # JSON for dashboard: frontend builds Plotly from this (same pattern as DTW)
        tryCatch({
          trace_json_path <- file.path(plots_dir, sprintf("%s_%s_trace_explorer_plot.json", cohort_name, age_band_fname))
          j <- plotly::plotly_json(fig, FALSE)
          write(j, trace_json_path)
          cat("Saved trace_explorer_plot.json\n")
        }, error = function(e) cat(" [skip] trace_explorer_plot.json:", conditionMessage(e), "\n"))
      } else {
        cat(" [skip] trace_explorer_interactive: no traces with data (empty pre-HCG)\n")
      }
    }
  }, error = function(e) cat(" [skip] interactive trace explorer (pre-HCG):", conditionMessage(e), "\n"))
} else {
  cat(" [skip] trace_explorer(pre-HCG): no pre-HCG events\n")
}

# -------------------------------------------------------------------
# Post-HCG (after first ED visit) plots: trace explorer + activity frequency
# -------------------------------------------------------------------
n_post <- nrow(events_post_target)
if (n_post > 0L) {
  cat("\n--- Post-HCG (after first ED visit) analysis ---\n")
  if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

  # 1) Trace explorer post-HCG PNG
  p_te_post <- tryCatch(
    trace_explorer_activity_frequency_plot(post_target_eventlog, "Post-HCG", top_n = 30),
    error = function(e) { cat(" [skip] trace_explorer(post-HCG):", conditionMessage(e), "\n"); NULL }
  )
  if (!is.null(p_te_post)) {
    ggsave(file.path(plots_dir, sprintf("%s_%s_trace_explorer_post_hcg.png", cohort_name, age_band_fname)),
           plot = p_te_post, width = 14, height = 10, dpi = 300)
    cat("Saved trace_explorer_post_hcg.png (activity frequency aggregated)\n")
  }

  # 2) Trace explorer interactive (post-HCG) HTML
  tryCatch({
    trace_data_by_year_post <- post_target_eventlog %>%
      as.data.frame() %>%
      mutate(year = lubridate::year(timestamp)) %>%
      group_by(case_id, year) %>%
      arrange(timestamp) %>%
      summarise(trace = paste(activity, collapse = " -> "), .groups = "drop") %>%
      group_by(year, trace) %>%
      summarise(frequency = n(), .groups = "drop")
    top_traces_post <- trace_data_by_year_post %>%
      group_by(trace) %>%
      summarise(total_freq = sum(frequency), .groups = "drop") %>%
      arrange(desc(total_freq)) %>%
      head(30) %>%
      pull(trace)
    trace_filtered_post <- trace_data_by_year_post %>% filter(trace %in% top_traces_post)
    trace_all_post <- trace_filtered_post %>%
      group_by(trace) %>%
      summarise(frequency = sum(frequency), .groups = "drop") %>%
      mutate(year = 0)
    trace_combined_post <- bind_rows(trace_all_post, trace_filtered_post) %>%
      arrange(desc(frequency)) %>%
      mutate(
        trace_display = ifelse(nchar(trace) > 100, paste0(substr(trace, 1, 97), "..."), trace),
        trace_display_short = vapply(trace, first_three_trace, character(1))
      )
    years_post <- c(0, 2016, 2017, 2018)
    year_labels_post <- c("All Years (2016-2018)", "2016", "2017", "2018")
    years_with_data_post <- integer(0)
    year_labels_with_data_post <- character(0)
    for (idx in seq_along(years_post)) {
      n <- nrow(trace_combined_post %>% filter(year == years_post[idx]) %>% head(30))
      if (n > 0L) {
        years_with_data_post <- c(years_with_data_post, years_post[idx])
        year_labels_with_data_post <- c(year_labels_with_data_post, year_labels_post[idx])
      }
    }
    if (length(years_with_data_post) > 0L) {
      fig_post <- plot_ly()
      traces_added_post <- 0L
      year_labels_added_post <- character(0)
      for (k in seq_along(years_with_data_post)) {
        yr <- years_with_data_post[k]
        data_year_post <- trace_combined_post %>%
          filter(year == yr) %>%
          arrange(desc(frequency)) %>%
          head(30)
        if (nrow(data_year_post) == 0L) next
        data_year_post <- data_year_post %>%
          filter(complete.cases(trace_display_short, trace_display, frequency)) %>%
          filter(nchar(as.character(trace_display_short)) >= 0L, frequency > 0)
        if (nrow(data_year_post) == 0L) next
        total_cases_post <- sum(data_year_post$frequency, na.rm = TRUE)
        total_cases_post <- if (total_cases_post <= 0) 1 else total_cases_post
        data_year_post <- data_year_post %>%
          mutate(relative_pct = frequency / total_cases_post * 100, cumulative_pct = cumsum(relative_pct))
        y_vals_post <- as.character(data_year_post$trace_display_short)
        x_vals_post <- as.numeric(data_year_post$frequency)
        if (length(y_vals_post) == 0L || length(x_vals_post) == 0L) next
        fig_post <- fig_post %>%
          add_trace(
            type = "bar",
            y = y_vals_post,
            x = x_vals_post,
            name = "Trace Frequency",
            customdata = as.character(data_year_post$trace_display),
            orientation = "h",
            visible = (traces_added_post == 0L),
            marker = list(color = "#dc2626"),
            text = sprintf("%.1f%% (cumulative: %.1f%%)", data_year_post$relative_pct, data_year_post$cumulative_pct),
            hovertemplate = paste0("<b>Trace:</b> %{customdata}<br><b>Frequency:</b> %{x}<br><b>Coverage:</b> %{text}<br><extra></extra>")
          )
        year_labels_added_post <- c(year_labels_added_post, year_labels_with_data_post[k])
        traces_added_post <- traces_added_post + 1L
      }
      n_traces_post <- traces_added_post
      if (n_traces_post > 0L) {
        updatemenus_post <- list(
          list(
            active = 0,
            type = "dropdown",
            x = 0.15, xanchor = "left", y = 1.08, yanchor = "top",
            buttons = lapply(seq_len(n_traces_post), function(k) {
              visible_vec <- rep(FALSE, n_traces_post)
              visible_vec[k] <- TRUE
              list(
                label = year_labels_added_post[k],
                method = "update",
                args = list(
                  list(visible = visible_vec),
                  list(title = paste("Post-HCG Trace Patterns:", cohort_name, age_band, "-", year_labels_added_post[k]))
                )
              )
            })
          )
        )
        fig_post <- fig_post %>%
          layout(
            title = list(text = paste("Post-HCG Trace Patterns:", cohort_name, age_band, "-", year_labels_added_post[1L])),
            xaxis = list(title = list(text = "Frequency (Number of Cases)"), type = "linear", zeroline = TRUE),
            yaxis = list(title = list(text = ""), type = "category", categoryorder = "total ascending"),
            updatemenus = updatemenus_post,
            margin = list(l = 300, r = 50, t = 100, b = 50),
            hovermode = "closest",
            height = 900
          )
        trace_html_post_path <- file.path(plots_dir, sprintf("%s_%s_trace_explorer_post_hcg_interactive.html", cohort_name, age_band_fname))
        # Production HTML: selfcontained = TRUE (same pattern as fpgrowth). See 9_dashboard_visuals/HTML_PRODUCTION_PATTERN.md
        saveWidget(fig_post, trace_html_post_path, selfcontained = TRUE, title = paste("Trace Explorer (Post-HCG):", cohort_name, age_band))
        cat("Saved trace_explorer_post_hcg_interactive.html; path=", trace_html_post_path, "\n", sep = "")
      } else {
        cat(" [skip] trace_explorer_interactive (post-HCG): no traces with data\n")
      }
    }
  }, error = function(e) cat(" [skip] interactive trace explorer (post-HCG):", conditionMessage(e), "\n"))

  # 3) Post-HCG activity frequency PNG (color-coded by event type)
  post_activity_freq <- post_target_eventlog %>%
    as.data.frame() %>%
    mutate(activity_type = case_when(
      grepl("^DRUG:", activity) ~ "Drug",
      grepl("^ICD:", activity) ~ "Diagnosis",
      grepl("^CPT:", activity) ~ "Procedure",
      TRUE ~ "Other"
    )) %>%
    group_by(activity, activity_type) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count)) %>%
    head(20)
  p2 <- ggplot(post_activity_freq, aes(x = reorder(activity, count), y = count, fill = activity_type)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = c("Drug" = "#3b82f6", "Diagnosis" = "#ef4444", "Procedure" = "#10b981", "Other" = "#64748b"), name = "Event Type") +
    labs(title = paste("Post-HCG Activity Frequency:", cohort_name, age_band),
         x = "Activity", y = "Frequency") +
    theme_minimal() +
    theme(legend.position = "top")
  ggsave(file.path(plots_dir, sprintf("%s_%s_post_hcg_activity_frequency.png", cohort_name, age_band_fname)),
         plot = p2, width = 10, height = 8, dpi = 300)
  cat("Saved post_hcg_activity_frequency.png\n")
  # Export post-target activity frequency as JSON (dashboard bar chart)
  tryCatch({
    post_freq_by_year <- post_target_eventlog %>%
      as.data.frame() %>%
      mutate(year = lubridate::year(timestamp),
             activity_type = case_when(
               grepl("^DRUG:", activity) ~ "Drug",
               grepl("^ICD:", activity) ~ "Diagnosis",
               grepl("^CPT:", activity) ~ "Procedure",
               TRUE ~ "Other"
             )) %>%
      group_by(year, activity, activity_type) %>%
      summarise(count = n(), .groups = "drop") %>%
      mutate(activity_short = vapply(activity, first_three_activity, character(1)))
    post_freq_all <- post_freq_by_year %>%
      group_by(activity, activity_type, activity_short) %>%
      summarise(count = sum(count), .groups = "drop") %>%
      mutate(year = 0)
    post_freq_combined <- bind_rows(post_freq_all, post_freq_by_year) %>%
      select(activity, activity_short, activity_type, count, year) %>%
      arrange(year, desc(count))
    year_labels_list <- list("0" = "All Years (2016-2018)", "2016" = "2016", "2017" = "2017", "2018" = "2018")
    post_json_path <- file.path(plots_dir, sprintf("%s_%s_post_target_activity_frequency.json", cohort_name, age_band_fname))
    jsonlite::write_json(list(year_labels = year_labels_list, data = post_freq_combined), post_json_path, dataframe = "rows", pretty = TRUE)
    cat("Saved post_target_activity_frequency.json\n")
  }, error = function(e) cat(" [skip] post_target_activity_frequency.json:", conditionMessage(e), "\n"))
} else {
  cat("\n--- Post-HCG: no events; skipping post-HCG plots ---\n")
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

# Trace explorer is pre-target only (see pre-HCG block above).
n_target <- nrow(as.data.frame(target_eventlog))
if (n_target > 0L) {
  # Process matrix (flows between activities; see https://bupaverse.github.io/docs/process_matrix.html)
  pm_df <- NULL
  tryCatch({
    pm_df <- target_eventlog %>%
      process_matrix(type = frequency("absolute"))
    p_pm <- plot(pm_df)
    if (!is.null(p_pm) && inherits(p_pm, "ggplot")) {
      ggsave(file.path(plots_dir, sprintf("%s_%s_process_matrix.png", cohort_name, age_band_fname)),
             plot = p_pm, width = 12, height = 10, dpi = 300)
      cat("Saved process_matrix.png\n")
    } else {
      pm_path <- file.path(plots_dir, sprintf("%s_%s_process_matrix.png", cohort_name, age_band_fname))
      png(pm_path, width = 12, height = 10, units = "in", res = 300)
      on.exit(dev.off(), add = TRUE)
      print(p_pm)
      cat("Saved process_matrix.png (via png device)\n")
    }
  }, error = function(e) cat(" [skip] process_matrix:", conditionMessage(e), "\n"))
  # Process matrix type-pair: Drug x Drug only (production pipeline for research questions).
  if (!is.null(pm_df) && nrow(pm_df) > 0L && "antecedent" %in% names(pm_df) && "consequent" %in% names(pm_df)) {
    pre_from <- "DRUG:"
    pre_to   <- "DRUG:"
    name     <- "drug_drug"
    tryCatch({
      pm_sub <- pm_df %>%
        filter(
          startsWith(as.character(antecedent), pre_from),
          startsWith(as.character(consequent), pre_to)
        )
      if (nrow(pm_sub) > 0L) {
        p_sub <- ggplot(pm_sub, aes(x = antecedent, y = consequent, fill = n)) +
          geom_tile() +
          scale_fill_viridis_c(option = "plasma", na.value = NA) +
          labs(title = "Process matrix: Drug x Drug", x = "Antecedent", y = "Consequent") +
          theme_minimal(base_size = 11) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text(size = 9))
        ggsave(file.path(plots_dir, sprintf("%s_%s_process_matrix_%s.png", cohort_name, age_band_fname, name)),
               plot = p_sub, width = 10, height = 8, dpi = 300)
        cat("Saved process_matrix_drug_drug.png\n")
        # JSON for dashboard: frontend builds Plotly heatmap from this
        tryCatch({
          pm_json_path <- file.path(plots_dir, sprintf("%s_%s_process_matrix_drug_drug.json", cohort_name, age_band_fname))
          write(jsonlite::toJSON(list(
            antecedent = as.character(pm_sub$antecedent),
            consequent = as.character(pm_sub$consequent),
            n = as.numeric(pm_sub$n)
          ), dataframe = "columns", auto_unbox = TRUE), pm_json_path)
          cat("Saved process_matrix_drug_drug.json\n")
        }, error = function(e) cat(" [skip] process_matrix_drug_drug.json:", conditionMessage(e), "\n"))
      }
    }, error = function(e) cat(" [skip] process_matrix_drug_drug: ", conditionMessage(e), "\n"))
  }
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
  # Overall Activity Frequency plot with color coding (only when we have target events)
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
        mutate(activity_short = vapply(activity, first_three_activity, character(1))) %>%
        arrange(activity, year)
      
      # Diagnostic logging for troubleshooting empty HTML
      cat("BupaR diagnostic [activity_frequency]: nrow(activity_freq_by_year)=", nrow(activity_freq_by_year),
          " nrow(activity_freq_combined)=", nrow(activity_freq_combined), "\n", sep = "")
      
      # Export overall activity frequency as JSON (dashboard bar chart; no HTML needed)
      af_json_path <- file.path(plots_dir, sprintf("%s_%s_activity_frequency.json", cohort_name, age_band_fname))
      export_df <- activity_freq_combined %>%
        select(activity, activity_short, activity_type, count, year) %>%
        arrange(year, desc(count))
      year_labels_list <- list("0" = "All Years (2016-2018)", "2016" = "2016", "2017" = "2017", "2018" = "2018")
      tryCatch({
        jsonlite::write_json(list(year_labels = year_labels_list, data = export_df), af_json_path, dataframe = "rows", pretty = TRUE)
        cat("Saved activity_frequency.json\n")
      }, error = function(e) cat(" [skip] activity_frequency.json:", conditionMessage(e), "\n"))
      
      # Create color mapping
      colors <- c("Drug" = "#3b82f6", "Diagnosis" = "#ef4444", "Procedure" = "#10b981", "Other" = "#64748b")
      
      years <- c(0, 2016, 2017, 2018)
      year_labels <- c("All Years (2016-2018)", "2016", "2017", "2018")
      # Only include years that have data so trace indices match visibility
      years_with_data <- integer(0)
      year_labels_with_data <- character(0)
      for (idx in seq_along(years)) {
        n <- nrow(activity_freq_combined %>% filter(year == years[idx]) %>% head(40))
        if (n > 0L) {
          years_with_data <- c(years_with_data, years[idx])
          year_labels_with_data <- c(year_labels_with_data, year_labels[idx])
        }
      }
      if (length(years_with_data) == 0L) {
        cat("BupaR diagnostic [activity_frequency]: no years with data; skipping interactive HTML\n")
      } else {
      # Build plotly figure: one batch of traces per year that has data; first batch visible by default
      fig <- plot_ly()
      trace_count <- 0L
      traces_per_year <- integer(length(years_with_data))
      for (k in seq_along(years_with_data)) {
        yr <- years_with_data[k]
        data_year <- activity_freq_combined %>%
          filter(year == yr) %>%
          arrange(desc(count)) %>%
          head(40)
        n_this <- 0L
        for (act_type in unique(data_year$activity_type)) {
          data_type <- data_year %>% filter(activity_type == act_type)
          if (nrow(data_type) == 0L) next
          n_this <- n_this + 1L
          trace_count <- trace_count + 1L
          fig <- fig %>%
            add_trace(
              type = "bar",
              y = data_type$activity_short,
              x = data_type$count,
              name = act_type,
              customdata = data_type$activity,
              marker = list(color = colors[act_type]),
              orientation = "h",
              visible = (k == 1L),  # First batch with data is visible by default
              legendgroup = act_type,
              showlegend = (k == 1L),
              hovertemplate = paste0(
                "<b>Activity:</b> %{customdata}<br>",
                "<b>Type:</b> ", act_type, "<br>",
                "<b>Count:</b> %{x}<br>",
                "<extra></extra>"
              )
            )
        }
        traces_per_year[k] <- n_this
      }
      n_traces_total <- trace_count
      # Build dropdown: visible_vec length must match actual number of traces
      visible_vec_len <- n_traces_total
      updatemenus <- list(
        list(
          active = 0,
          type = "dropdown",
          x = 0.15,
          xanchor = "left",
          y = 1.15,
          yanchor = "top",
          buttons = lapply(seq_along(years_with_data), function(k) {
            visible_vec <- rep(FALSE, visible_vec_len)
            start_idx <- 1L + sum(traces_per_year[seq_len(k - 1)])
            end_idx <- sum(traces_per_year[seq_len(k)])
            if (end_idx >= start_idx) visible_vec[start_idx:end_idx] <- TRUE
            list(
              label = year_labels_with_data[k],
              method = "update",
              args = list(
                list(visible = visible_vec),
                list(title = paste("Activity Frequency:", cohort_name, age_band, "-", year_labels_with_data[k]))
              )
            )
          })
        )
      )
      fig <- fig %>%
        layout(
          title = paste("Activity Frequency:", cohort_name, age_band, "-", year_labels_with_data[1L]),
          xaxis = list(title = "Frequency"),
          yaxis = list(title = "", categoryorder = "total ascending"),
          barmode = "stack",
          updatemenus = updatemenus,
          margin = list(l = 200, r = 50, t = 100, b = 50),
          legend = list(orientation = "h", y = 1.05, x = 0.5, xanchor = "center"),
          hovermode = "closest"
        )
      # Save interactive HTML only when we have at least one year with data
      af_html_path <- file.path(plots_dir, sprintf("%s_%s_activity_frequency_interactive.html", cohort_name, age_band_fname))
      # Production HTML: selfcontained = TRUE (same pattern as fpgrowth). See 9_dashboard_visuals/HTML_PRODUCTION_PATTERN.md
      saveWidget(
        fig,
        af_html_path,
        selfcontained = TRUE,
        title = paste("Activity Frequency:", cohort_name, age_band)
      )
      af_html_size <- file.info(af_html_path)$size
      cat("Saved activity_frequency_interactive.html with year filtering; path=", af_html_path, " size_bytes=", if (is.na(af_html_size)) "NA" else af_html_size, "\n", sep = "")
      }
    }, error = function(e) cat(" [skip] interactive activity frequency:", conditionMessage(e), "\n"))
  }, error = function(e) cat(" [skip] overall_activity_frequency:", conditionMessage(e), "\n"))
  
  tryCatch(
    process_map(target_eventlog, type = "frequency"),
    error = function(e) cat(" [skip] process_map(target):", conditionMessage(e), "\n")
  )
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

# Close the cohort-specific PDF device if it is still open so that any
# base graphics output is written under the correct cohort directory.
if (grDevices::dev.cur() > 1) {
  grDevices::dev.off()
}

cat("\n=== bupaR analysis for non_opioid_ed ", age_band, " completed. ===\n", sep = "")



