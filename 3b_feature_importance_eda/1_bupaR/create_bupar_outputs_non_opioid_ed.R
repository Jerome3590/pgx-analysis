#!/usr/bin/env Rscript
#
# End-to-end bupaR analysis for POLYPHARMACY COHORT
# (cohort_name="non_opioid_ed" in data partitions, but referred to as "polypharmacy cohort")
# Configurable age band (65–74, 75–84, 85–94) - cohorts 5, 6, 7
#
# - Builds target-only and combined event logs from model_data
# - Runs pre-HCG sequence analyses (no post-target to avoid leakage)
# - Exports pre-HCG, time-to-HCG per-patient features, trace tables, and process matrices
# Uses all events from model_events.parquet directly
#

suppressPackageStartupMessages({
  library(duckdb)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(bupaR)
  library(bupaverse)
  library(processmapR)
  library(edeaR)
  library(lubridate)
  library(ggplot2)
})

# -------------------------------------------------------------------
# Load Helper Functions
# -------------------------------------------------------------------

# Source logging utilities from r_helpers
project_root <- getwd()  # assume you launched from project root
logging_utils_path <- file.path(project_root, "r_helpers", "logging_utils.R")
if (file.exists(logging_utils_path)) {
  source(logging_utils_path)
} else {
  # Fallback: define log_msg locally if r_helpers not found
  log_msg <- function(msg, level = "INFO") {
    timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    cat(sprintf("[%s] [%s] %s\n", timestamp, level, msg))
    flush.console()
  }
}

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# POLYPHARMACY COHORT: cohort_name in data partitions is "non_opioid_ed"
# but we refer to this as "polypharmacy cohort" throughout
cohort_name    <- "non_opioid_ed"  # Data partition name (must match S3/parquet partitions)
polypharmacy_cohort_name <- "polypharmacy"  # Human-readable name for logging
control_cohort <- "non_opioid_non_ed"  # Control cohort name in data partitions

# Optional command line argument to set age band; default is 65-74
args <- commandArgs(trailingOnly = TRUE)
age_band <- if (length(args) >= 1) args[[1]] else "65-74"

age_band_fname <- gsub("-", "_", age_band)
train_years    <- c(2016L, 2017L, 2018L)

log_msg("=", level = "INFO")
log_msg("bupaR Analysis: POLYPHARMACY COHORT", level = "INFO")
log_msg(sprintf("  Data partition:  cohort_name=%s (non_opioid_ed)", cohort_name), level = "INFO")
log_msg(sprintf("  Age band:       %s", age_band), level = "INFO")
log_msg(sprintf("  Control cohort: %s (non_opioid_non_ed)", control_cohort), level = "INFO")
log_msg("  Target: First ED visit (HCG Setting) within 21 days of a prescription drug event", level = "INFO")
log_msg("  Note: Polypharmacy cohort (cohorts 5, 6, 7 with age > 64)", level = "INFO")
log_msg("=", level = "INFO")

# POLYPHARMACY COHORT: Target definition (consistent with 2_create_cohort and 4_model_data)
# - Target: First ED visit (identified by HCG Setting: P51/O11/P33) within 21 days of a prescription drug event
# - NOT F1120; applies to cohorts 5, 6, 7 with age band >= 65
# - Target events identified by hcg_line or first_o11_p_date (or legacy first_ed_non_opioid_date) in model_events

# Path cohort for model_events: this script is for non_opioid_ed only. Python writes to
# 3b.../outputs/cohort_name=non_opioid_ed/age_band=... and syncs to gold/cohorts_model_data/cohort_name=non_opioid_ed/...
path_cohort <- cohort_name  # "non_opioid_ed"

# OS-aware data root (EC2: /mnt/nvme, Windows: project root)
data_root <- Sys.getenv("PGX_DATA_ROOT", "")
if (data_root == "") {
  if (.Platform$OS.type == "unix") {
    data_root <- "/mnt/nvme"
  } else {
    data_root <- project_root
  }
}

# Aggregated feature importance is required—it defines the feature set and includes potential target leakage. Do not continue without it.
agg_fi_candidates_top <- c(
  file.path(project_root, "3a_feature_importance", "outputs", cohort_name, paste0(cohort_name, "_", age_band_fname, "_aggregated_feature_importance.csv")),
  file.path(project_root, "3a_feature_importance", "outputs", cohort_name, age_band, paste0(cohort_name, "_", age_band_fname, "_aggregated_feature_importance.csv")),
  file.path(project_root, "3a_feature_importance", "from_s3", "by_cohort", cohort_name, age_band, paste0(cohort_name, "_", age_band_fname, "_aggregated_feature_importance.csv"))
)
aggregated_fi_found <- FALSE
for (p in agg_fi_candidates_top) {
  if (file.exists(p)) {
    aggregated_fi_found <- TRUE
    break
  }
}
if (!aggregated_fi_found) {
  stop("Aggregated feature importance is required. Run Step 3a (2_feature_importance.ipynb) for cohort ", cohort_name, " age_band ", age_band, " first. Do not continue without it.")
}

# Step 3b: model_events written by create_bupar_input_from_cohort to
# 3b.../outputs/cohort_name=non_opioid_ed/age_band=... and synced to gold/cohorts_model_data/cohort_name=non_opioid_ed/...
model_data_candidates <- c(
  file.path(project_root, "3b_feature_importance_eda", "outputs", paste0("cohort_name=", path_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(data_root, "3b_feature_importance_eda", "outputs", paste0("cohort_name=", path_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(project_root, "3b_feature_importance_eda", "outputs", "cohorts", "input_model_data", paste0("cohort_name=", path_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(data_root, "3b_feature_importance_eda", "outputs", "cohorts", "input_model_data", paste0("cohort_name=", path_cohort), paste0("age_band=", age_band), "model_events.parquet")
)

model_data_path <- NULL
for (candidate in model_data_candidates) {
  if (file.exists(candidate)) {
    model_data_path <- candidate
    break
  }
}

# If not found, try downloading from S3 (gold/cohorts_model_data = where Python syncs)
if (is.null(model_data_path)) {
  model_data_path <- model_data_candidates[1]
  s3_path <- paste0("s3://pgxdatalake/gold/cohorts_model_data/cohort_name=", path_cohort, "/age_band=", age_band, "/model_events.parquet")
  cat("Model data not found locally. Checking S3: ", s3_path, "\n", sep = "")
  
  # Create directory if it doesn't exist
  dir.create(dirname(model_data_path), recursive = TRUE, showWarnings = FALSE)
  
  # Try AWS CLI sync
  aws_cli <- Sys.which("aws")
  if (aws_cli != "") {
    cat("Downloading from S3 using AWS CLI...\n")
    sync_cmd <- c("s3", "cp", s3_path, model_data_path)
    sync_result <- system2(aws_cli, sync_cmd, stdout = TRUE, stderr = TRUE)
    
    if (file.exists(model_data_path)) {
      cat("Successfully downloaded from S3: ", model_data_path, "\n", sep = "")
    } else {
      cat("Failed to download from S3. Error output:\n")
      cat(paste(sync_result, collapse = "\n"), "\n")
    }
  } else {
    cat("AWS CLI not found. Cannot download from S3.\n")
  }
}

cat("Project root:         ", project_root, "\n", sep = "")
cat("Data root:            ", data_root, "\n", sep = "")
cat("Model data path:      ", model_data_path, "\n", sep = "")
cat("Note: Using all codes from model_events.parquet\n\n", sep = "")

# -------------------------------------------------------------------
# Helper for saving CSVs locally + to S3
# -------------------------------------------------------------------

bup_ar_output_root <- file.path(project_root, "3b_feature_importance_eda", "outputs")

# Create plots directory and open PDF device to capture any base graphics
# This prevents Rplots.pdf from being created in the project root
plots_dir <- file.path(bup_ar_output_root, cohort_name, age_band_fname, "plots")
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
}

# Open a PDF device for base graphics (trace_explorer, process_map, etc.)
# This routes base graphics to the correct output directory instead of project root
rplots_path <- file.path(plots_dir, sprintf("%s_%s_Rplots.pdf", cohort_name, age_band_fname))
pdf(file = rplots_path, width = 12, height = 9)

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

# -------------------------------------------------------------------
# Load model_data and build target-only subset
# -------------------------------------------------------------------

if (!file.exists(model_data_path)) {
  stop("model_data parquet not found at: ", model_data_path,
       "\nRun 3b create_bupar_input_from_cohort.py (builds from cohort + 3a FI + target), or 4_model_data/create_model_data.py for this cohort/age band first.")
}

con <- dbConnect(duckdb::duckdb())
n_threads <- Sys.getenv("DUCKDB_THREADS", "")
if (n_threads != "" && !is.na(suppressWarnings(as.integer(n_threads)))) {
  tryCatch({
    dbExecute(con, sprintf("SET threads = %s", n_threads))
    log_msg(sprintf("DuckDB threads: %s", n_threads))
  }, error = function(e) {})
}

# Check if hcg_line or target date column (first_o11_p_date or legacy first_ed_non_opioid_date) exists
schema_query <- sprintf("DESCRIBE SELECT * FROM read_parquet('%s')", model_data_path)
schema_info <- dbGetQuery(con, schema_query)
has_hcg_line <- "hcg_line" %in% schema_info$column_name
has_first_o11_p <- "first_o11_p_date" %in% schema_info$column_name
has_first_ed_date <- "first_ed_non_opioid_date" %in% schema_info$column_name
has_target_date_col <- has_first_o11_p || has_first_ed_date

log_msg(sprintf("Loading target cohort from: %s (target=1, years: %s)", model_data_path, paste(train_years, collapse = ",")))

# Wide target load (needed for pgx_df_all bind with control for Sankey)
base_columns <- c(
  "mi_person_key", "event_date", "drug_name",
  "primary_icd_diagnosis_code", "two_icd_diagnosis_code", "three_icd_diagnosis_code", "four_icd_diagnosis_code",
  "five_icd_diagnosis_code", "six_icd_diagnosis_code", "seven_icd_diagnosis_code", "eight_icd_diagnosis_code",
  "nine_icd_diagnosis_code", "ten_icd_diagnosis_code", "procedure_code"
)
if (has_hcg_line) base_columns <- c(base_columns, "hcg_line")
if (has_first_o11_p) base_columns <- c(base_columns, "first_o11_p_date")
if (has_first_ed_date) base_columns <- c(base_columns, "first_ed_non_opioid_date")
query_target_wide <- sprintf(
  "SELECT %s FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1",
  paste(base_columns, collapse = ", "), model_data_path, paste(train_years, collapse = ",")
)
pgx_df_target1 <- dbGetQuery(con, query_target_wide)
log_msg(sprintf("  Loaded %d target=1 wide rows (for Sankey combined)", nrow(pgx_df_target1)))

# -------------------------------------------------------------------
# Using all codes from model_events.parquet
# This ensures we capture all pre-HCG events for analysis
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Build DRUG/ICD/CPT activities and target_eventlog
# -------------------------------------------------------------------

# Single DuckDB read: UNPIVOT wide→long in DuckDB (one Parquet scan; polypharmacy = drug_name only)
log_msg("Transforming target to long format using DuckDB UNPIVOT (single read; DRUG only)...")
query_long <- sprintf(
  "SELECT 
    mi_person_key,
    event_date,
    source,
    CAST(code AS VARCHAR) as code,
    CASE 
      WHEN source = 'drug_name' THEN 'DRUG:' || REPLACE(REPLACE(code, ' ', '_'), '/', '_')
      ELSE NULL
    END as activity
  FROM (
    SELECT 
      mi_person_key,
      event_date,
      CAST(drug_name AS VARCHAR) as drug_name,
      CAST(primary_icd_diagnosis_code AS VARCHAR) as primary_icd_diagnosis_code,
      CAST(two_icd_diagnosis_code AS VARCHAR) as two_icd_diagnosis_code,
      CAST(three_icd_diagnosis_code AS VARCHAR) as three_icd_diagnosis_code,
      CAST(four_icd_diagnosis_code AS VARCHAR) as four_icd_diagnosis_code,
      CAST(five_icd_diagnosis_code AS VARCHAR) as five_icd_diagnosis_code,
      CAST(six_icd_diagnosis_code AS VARCHAR) as six_icd_diagnosis_code,
      CAST(seven_icd_diagnosis_code AS VARCHAR) as seven_icd_diagnosis_code,
      CAST(eight_icd_diagnosis_code AS VARCHAR) as eight_icd_diagnosis_code,
      CAST(nine_icd_diagnosis_code AS VARCHAR) as nine_icd_diagnosis_code,
      CAST(ten_icd_diagnosis_code AS VARCHAR) as ten_icd_diagnosis_code,
      CAST(procedure_code AS VARCHAR) as procedure_code
    FROM read_parquet('%s') 
    WHERE event_year IN (%s) AND target = 1
  ) 
  UNPIVOT (
    code FOR source IN (
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
    )
  )
  WHERE code IS NOT NULL AND code != '' AND code != 'NA' AND source = 'drug_name'",
  model_data_path,
  paste(train_years, collapse = ",")
)

pgx_df_target1_long <- dbGetQuery(con, query_long) %>%
  filter(!is.na(activity)) %>%
  mutate(timestamp = as.POSIXct(event_date))

# First target date (HCG/first ED) per patient in DuckDB for pre/post split
if (has_first_o11_p) {
  query_first_target <- sprintf(
    "SELECT mi_person_key, min(first_o11_p_date) AS first_target_date FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1 AND first_o11_p_date IS NOT NULL GROUP BY 1",
    model_data_path, paste(train_years, collapse = ",")
  )
} else if (has_first_ed_date) {
  query_first_target <- sprintf(
    "SELECT mi_person_key, min(first_ed_non_opioid_date) AS first_target_date FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1 AND first_ed_non_opioid_date IS NOT NULL GROUP BY 1",
    model_data_path, paste(train_years, collapse = ",")
  )
} else if (has_hcg_line) {
  query_first_target <- sprintf(
    "SELECT mi_person_key, min(event_date) AS first_target_date FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1 AND hcg_line IN ('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits') GROUP BY 1",
    model_data_path, paste(train_years, collapse = ",")
  )
} else {
  query_first_target <- sprintf(
    "SELECT mi_person_key, min(event_date) AS first_target_date FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1 GROUP BY 1",
    model_data_path, paste(train_years, collapse = ",")
  )
}
first_target_df <- dbGetQuery(con, query_first_target)
pgx_df_target1_long <- pgx_df_target1_long %>% left_join(first_target_df, by = "mi_person_key")

log_msg(sprintf("✓ Loaded %d target=1 drug events (long + first_target_date) for %s age_band=%s",
                nrow(pgx_df_target1_long), cohort_name, age_band))

target_eventlog <- pgx_df_target1_long %>%
  select(-first_target_date) %>%
  transmute(
    case_id              = mi_person_key,
    activity             = activity,
    timestamp            = timestamp,
    activity_instance_id = dplyr::row_number(),
    lifecycle_id         = "complete",
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

cat("Target eventlog created.\n")
print(target_eventlog)

# -------------------------------------------------------------------
# Combined TARGET + CONTROL eventlog for Sankey
# -------------------------------------------------------------------

# Source utility functions for control cohort management
utils_path <- file.path(project_root, "r_helpers", "control_cohort_utils.R")
if (file.exists(utils_path)) {
  source(utils_path)
} else {
  stop("Control cohort utility functions not found. Expected at: ", utils_path)
}

# Control: same path layout as target (outputs/cohort_name=... and gold/cohorts_model_data); fallback to legacy cohorts/input_model_data
control_model_data_candidates <- c(
  file.path(project_root, "3b_feature_importance_eda", "outputs", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(data_root, "3b_feature_importance_eda", "outputs", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(data_root, "gold", "cohorts_model_data", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(project_root, "3b_feature_importance_eda", "outputs", "cohorts", "input_model_data", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(data_root, "3b_feature_importance_eda", "outputs", "cohorts", "input_model_data", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet")
)

control_model_data_path <- NULL
for (candidate in control_model_data_candidates) {
  if (file.exists(candidate)) {
    control_model_data_path <- candidate
    break
  }
}

# If not found, use 3b output path for creation (cohort_name= layout to match Python)
if (is.null(control_model_data_path)) {
  control_model_data_path <- file.path(project_root, "3b_feature_importance_eda", "outputs", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet")
}

# Resolve 3a aggregated FI path for control filtering (required; already verified at top of script)
aggregated_fi_path <- NULL
for (p in agg_fi_candidates_top) {
  if (file.exists(p)) {
    aggregated_fi_path <- p
    break
  }
}
if (is.null(aggregated_fi_path)) {
  stop("Aggregated feature importance is required. Run Step 3a (2_feature_importance.ipynb) for cohort ", cohort_name, " age_band ", age_band, " first.")
}

# Use utility function to validate and ensure control cohort with 5:1 ratio
control_result <- ensure_control_cohort_with_ratio(
  con = con,
  control_cohort = control_cohort,
  control_model_data_path = control_model_data_path,
  model_data_path = model_data_path,
  age_band = age_band,
  train_years = train_years,
  project_root = project_root,
  output_root_3b = file.path(project_root, "3b_feature_importance_eda", "outputs"),
  aggregated_fi_path = aggregated_fi_path,
  expected_ratio = 5.0,
  tolerance = 0.2
)

pgx_df_control <- control_result$pgx_df_control

# Assert that control PATIENTS are not duplicated (check distinct patients, not event-level data)
# Event-level data will have duplicate mi_person_key values (one row per event per patient)
if (nrow(pgx_df_control) > 0) {
  distinct_control_patients <- unique(pgx_df_control$mi_person_key)
  stopifnot(!anyDuplicated(distinct_control_patients))
  cat("Verified ", length(distinct_control_patients), " distinct control patients (", nrow(pgx_df_control), " total events)\n", sep = "")
}

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
  # For polypharmacy cohort, only analyze drug_name events (DRUG: activities)
  filter(source == "drug_name") %>%
  mutate(
    # Replace spaces and forward slashes with underscores in drug names
    code_cleaned = gsub("[ /]", "_", code),
    activity = paste0("DRUG:", code_cleaned),
    timestamp = as.POSIXct(event_date)
  ) %>%
  select(-code_cleaned)  # Remove temporary column

sankey_eventlog <- pgx_df_all_long %>%
  transmute(
    case_id              = mi_person_key,
    activity             = activity,
    timestamp            = timestamp,
    group                = group,
    activity_instance_id = dplyr::row_number(),
    lifecycle_id         = "complete",
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

log_msg("✓ Combined TARGET + CONTROL sankey_eventlog created")
cat("Combined eventlog summary:\n")
print(sankey_eventlog)

# -------------------------------------------------------------------
# Pre-HCG (before first ED visit within 21 days of drug event) sequences
# Target = first ED visit (HCG Setting) within 21 days of a prescription drug event.
# -------------------------------------------------------------------

log_msg("=", level = "INFO")
log_msg("Starting Pre-HCG (before first ED visit within 21d of drug event) analysis", level = "INFO")
log_msg("  Target: First ED visit (HCG Setting) within 21 days of a prescription drug event", level = "INFO")
log_msg("=", level = "INFO")

# Pre/post split using first_target_date from DuckDB (event_date < first_target_date = pre; > = post)
ed_hcg_lines <- c('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits')
target_date_map <- first_target_df %>%
  rename(case_id = mi_person_key, target_date = first_target_date)

log_msg("Splitting pre/post HCG using first_target_date from DuckDB...")
events_pre_target <- pgx_df_target1_long %>%
  filter(!is.na(first_target_date), event_date < first_target_date) %>%
  mutate(
    activity_instance_id = row_number(),
    case_id = mi_person_key,
    lifecycle_id = "complete",
    resource_id = "Patient"
  )

pre_target_eventlog <- events_pre_target %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    timestamp            = "timestamp",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id"
  )

log_msg("Pre-HCG eventlog summary:")
print(pre_target_eventlog)

# Check if pre-HCG eventlog is empty
if (n_events(pre_target_eventlog) == 0) {
  log_msg("⚠ No pre-HCG events found; skipping pre-HCG trace and feature analysis", level = "WARN")
  
  # Create empty data frames for consistency
  pre_drug_sequences <- data.frame(
    case_id = character(0),
    drug_sequence = I(list())
  )
  
  pre_patient_features <- data.frame(
    case_id = character(0),
    pre_n_events = integer(0),
    pre_n_drug_events = integer(0),
    pre_n_icd_events = integer(0),
    pre_n_cpt_events = integer(0),
    pre_n_unique_activities = integer(0)
  )
  
  # Save empty patient features (for consistency with workflow)
  save_bupar_csv(
    pre_patient_features,
    sprintf("%s_%s_train_target_pre_hcg_patient_features_bupar.csv", cohort_name, age_band_fname)
  )
  
} else {
# 1) Trace explorer (printed summary; visuals if running interactively)
  tryCatch({
trace_explorer(pre_target_eventlog, coverage = 0.8)
  }, error = function(e) {
    cat("Warning: trace_explorer failed:", conditionMessage(e), "\n")
  })

# 2) Drug-only sequences before HCG
  pre_drug_sequences <- pre_target_eventlog %>%
  arrange(case_id, timestamp) %>%
  filter(grepl("^DRUG:", activity)) %>%
  group_by(case_id) %>%
  summarise(
    drug_sequence = list(activity),
    .groups = "drop"
  )

log_msg("Sample pre-HCG drug-only sequences:")
print(head(pre_drug_sequences))

# 3) Process map for pre-HCG trajectories
  # Filter eventlog to ensure valid events before calling process_map
  tryCatch({
    valid_pre_eventlog <- pre_target_eventlog %>%
      filter(!is.na(activity), 
             activity != "", 
             activity != "NA",
             !is.na(timestamp))
    
    if (n_events(valid_pre_eventlog) > 0 && n_cases(valid_pre_eventlog) > 0) {
      process_map(valid_pre_eventlog, type = "frequency")
    } else {
      cat("Warning: Not enough valid events/cases for pre-HCG process_map (events: ", n_events(valid_pre_eventlog), ", cases: ", n_cases(valid_pre_eventlog), ")\n", sep = "")
    }
  }, error = function(e) {
    cat("Warning: process_map failed:", conditionMessage(e), "\n")
  })

# 4) Per-patient pre-HCG features
  pre_patient_features <- pre_target_eventlog %>%
  arrange(case_id, timestamp) %>%
  group_by(case_id) %>%
  summarise(
    pre_n_events            = n(),
      pre_n_drug_events       = sum(grepl("^DRUG:", activity)),
      # For polypharmacy cohort, only drug events exist (ICD/CPT filtered out)
      pre_n_icd_events        = 0L,
      pre_n_cpt_events        = 0L,
    pre_n_unique_activities = n_distinct(activity),
    .groups = "drop"
  )

save_bupar_csv(
  pre_patient_features,
  sprintf("%s_%s_train_target_pre_hcg_patient_features_bupar.csv", cohort_name, age_band_fname)
)
}

# -------------------------------------------------------------------
# Time-to-HCG and time-window features (per patient)
# For polypharmacy cohort: first ED visit (HCG Setting) within 21 days of drug event
# -------------------------------------------------------------------
log_msg("Calculating time-to-HCG and time-window features (per patient)...")

# Use the target_date_map we created earlier (in the pre-HCG section)
target_times <- target_date_map %>%
  mutate(
    target_time = as.POSIXct(target_date),
    first_time  = target_time  # For non_opioid_ed, first_time is same as target_time
  ) %>%
  select(case_id, target_time, first_time)

# Only create time-to-HCG features if we have pre-HCG events
if (n_events(pre_target_eventlog) > 0) {
  pre_events_with_t <- pre_target_eventlog %>%
  inner_join(target_times, by = "case_id") %>%
  mutate(
    dt_days = as.numeric(difftime(target_time, timestamp, units = "days"))
  )
} else {
  # Create empty data frame with same structure
  pre_events_with_t <- data.frame(
    case_id = character(0),
    activity = character(0),
    timestamp = as.POSIXct(character(0)),
    activity_instance_id = integer(0),
    lifecycle_id = character(0),
    resource_id = character(0),
    target_time = as.POSIXct(character(0)),
    first_time = as.POSIXct(character(0)),
    dt_days = numeric(0)
  )
}

hcg_time_features <- pre_events_with_t %>%
  group_by(case_id, target_time, first_time) %>%
  summarise(
    time_to_HCG_days        = ifelse(
      all(is.na(dt_days)), NA_real_,
      as.numeric(max(dt_days, na.rm = TRUE))
    ),
    n_events_30d            = sum(dt_days <= 30),
    n_events_90d            = sum(dt_days <= 90),
    n_events_180d           = sum(dt_days <= 180),
    n_drug_events_30d       = sum(dt_days <= 30 & grepl("^DRUG:", activity)),
    n_drug_events_90d       = sum(dt_days <= 90 & grepl("^DRUG:", activity)),
    n_drug_events_180d      = sum(dt_days <= 180 & grepl("^DRUG:", activity)),
    # For polypharmacy cohort, only drug events exist (ICD/CPT filtered out)
    n_icd_events_30d        = 0L,
    n_icd_events_90d        = 0L,
    n_icd_events_180d       = 0L,
    n_cpt_events_30d        = 0L,
    n_cpt_events_90d        = 0L,
    n_cpt_events_180d       = 0L,
    .groups = "drop"
  )

save_bupar_csv(
  hcg_time_features,
  sprintf("%s_%s_train_target_time_to_hcg_features_bupar.csv", cohort_name, age_band_fname)
)

# -------------------------------------------------------------------
# Target-only global process mining (traces + process matrix)
# -------------------------------------------------------------------

cat("\n--- Target-only global process mining ---\n")

# 1) Trace Explorer: most frequent target trajectories
trace_explorer(target_eventlog, coverage = 0.8)

# Save trace summary as tabular output
traces_target <- bupaR::traces(target_eventlog)
traces_target_df <- as.data.frame(traces_target) %>%
  arrange(desc(absolute_frequency))

save_bupar_csv(
  traces_target_df,
  sprintf("%s_%s_train_target_traces_bupar.csv", cohort_name, age_band_fname)
)

# Categorize traces into top sequences and rare sequences
total_cases <- n_cases(target_eventlog)
top_n_threshold <- max(20, ceiling(total_cases * 0.1))  # Top 20 sequences or top 10% of cases, whichever is larger
rare_threshold <- 1  # Sequences that appear only once

# Top sequences (most frequent)
top_sequences <- traces_target_df %>%
  filter(absolute_frequency >= top_n_threshold) %>%
  mutate(sequence_category = "top")

# Rare sequences (appear only once or very infrequently)
rare_sequences <- traces_target_df %>%
  filter(absolute_frequency <= rare_threshold) %>%
  mutate(sequence_category = "rare")

# Save top sequences
if (nrow(top_sequences) > 0) {
  save_bupar_csv(
    top_sequences,
    sprintf("%s_%s_train_target_traces_top_bupar.csv", cohort_name, age_band_fname)
  )
  cat(sprintf("Saved %d top sequences (frequency >= %d)\n", nrow(top_sequences), top_n_threshold))
} else {
  cat("No top sequences found (all sequences are rare)\n")
}

# Save rare sequences
if (nrow(rare_sequences) > 0) {
  save_bupar_csv(
    rare_sequences,
    sprintf("%s_%s_train_target_traces_rare_bupar.csv", cohort_name, age_band_fname)
  )
  cat(sprintf("Saved %d rare sequences (frequency <= %d)\n", nrow(rare_sequences), rare_threshold))
} else {
  cat("No rare sequences found\n")
}

# 2) Process Matrix and CSV export
# For polypharmacy cohort, only analyze drug_name events (DRUG: activities)
# Filter eventlog to ensure valid events before calling process_matrix
tryCatch({
  # Filter to only drug events (DRUG: activities) for polypharmacy analysis
  valid_df <- as.data.frame(target_eventlog) %>%
    filter(!is.na(activity), 
           activity != "", 
           activity != "NA",
           !is.na(timestamp),
           !is.na(case_id),
           grepl("^DRUG:", activity))  # Only drug events for polypharmacy
  
  if (nrow(valid_df) > 0) {
    # Recreate eventlog from filtered data frame
    valid_eventlog <- valid_df %>%
      mutate(activity_instance_id = row_number()) %>%  # Recreate activity_instance_id after filtering
      eventlog(
        case_id              = "case_id",
        activity_id          = "activity",
        activity_instance_id = "activity_instance_id",
        lifecycle_id         = "lifecycle_id",
        resource_id          = "resource_id",
        timestamp            = "timestamp"
      )
    
    # Check if we have enough events/cases for process_matrix
    if (n_events(valid_eventlog) > 0 && n_cases(valid_eventlog) > 0) {
      pm_target <- process_matrix(valid_eventlog, type = "frequency")
pm_target_df <- as.data.frame(pm_target)
save_bupar_csv(
  pm_target_df,
  sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
)
    } else {
      cat("Warning: Not enough valid drug events/cases for process_matrix (events: ", n_events(valid_eventlog), ", cases: ", n_cases(valid_eventlog), ")\n", sep = "")
      pm_target_df <- data.frame()
      save_bupar_csv(
        pm_target_df,
        sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
      )
    }
  } else {
    cat("Warning: No valid drug events after filtering for process_matrix\n")
    pm_target_df <- data.frame()
    save_bupar_csv(
      pm_target_df,
      sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
    )
  }
}, error = function(e) {
  cat("Warning: process_matrix failed:", conditionMessage(e), "\n")
  # Create empty data frame to avoid downstream errors
  pm_target_df <- data.frame()
  save_bupar_csv(
    pm_target_df,
    sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
  )
})

# 3) Process Map visualization
# For polypharmacy cohort, only analyze drug_name events (DRUG: activities)
tryCatch({
  # Filter to only drug events (DRUG: activities) for polypharmacy analysis
  valid_df <- as.data.frame(target_eventlog) %>%
    filter(!is.na(activity), 
           activity != "", 
           activity != "NA",
           !is.na(timestamp),
           !is.na(case_id),
           grepl("^DRUG:", activity))  # Only drug events for polypharmacy
  
  if (nrow(valid_df) > 0) {
    # Recreate eventlog from filtered data frame
    valid_eventlog <- valid_df %>%
      mutate(activity_instance_id = row_number()) %>%  # Recreate activity_instance_id after filtering
      eventlog(
        case_id              = "case_id",
        activity_id          = "activity",
        activity_instance_id = "activity_instance_id",
        lifecycle_id         = "lifecycle_id",
        resource_id          = "resource_id",
        timestamp            = "timestamp"
      )
    
    if (n_events(valid_eventlog) > 0 && n_cases(valid_eventlog) > 0) {
      process_map(valid_eventlog, type = "frequency")
    } else {
      cat("Warning: Not enough valid drug events/cases for process_map (events: ", n_events(valid_eventlog), ", cases: ", n_cases(valid_eventlog), ")\n", sep = "")
    }
  } else {
    cat("Warning: No valid drug events after filtering for process_map\n")
  }
}, error = function(e) {
  cat("Warning: process_map failed:", conditionMessage(e), "\n")
})

# 4) ggplot2 Visualizations (PNG files)
# For polypharmacy cohort, only drug events exist (ICD/CPT filtered out)
plots_dir <- file.path(bup_ar_output_root, cohort_name, age_band_fname, "plots")
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

# Activity frequency plot (overall)
target_activity_freq <- target_eventlog %>%
  group_by(activity) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(30)

if (nrow(target_activity_freq) > 0) {
  p3 <- ggplot(target_activity_freq, aes(x = reorder(activity, count), y = count)) +
    geom_bar(stat = "identity", fill = "darkgreen") +
    coord_flip() +
    labs(title = paste("Overall Activity Frequency:", cohort_name, age_band),
         x = "Activity", y = "Frequency") +
    theme_bw()
  
  ggsave(file.path(plots_dir, sprintf("%s_%s_overall_activity_frequency.png", cohort_name, age_band_fname)),
         plot = p3, width = 12, height = 10, dpi = 300)
  cat("Created overall activity frequency plot.\n")
}

# Gantt-style timeline (patient = job, activity = stage)
# For polypharmacy cohort, all activities are drugs (ICD/CPT filtered out)
# Each drug is a different event type for visualization
# Sample up to 30 cases for visualization
target_events_df <- as.data.frame(target_eventlog) %>%
  arrange(case_id, timestamp) %>%
  mutate(
    # Extract drug name from activity (format: "DRUG:drug_name")
    event_type = gsub("^DRUG:", "", activity)
  ) %>%
  # For point events, add small duration (1 day) to create visible bars
  mutate(start_time = timestamp,
         end_time = timestamp + lubridate::ddays(1))

sample_cases <- unique(target_events_df$case_id)[1:min(30, length(unique(target_events_df$case_id)))]
target_events_sample <- target_events_df %>%
  filter(case_id %in% sample_cases) %>%
  mutate(case_id_factor = factor(case_id, levels = rev(sample_cases)),
         entity_num = as.numeric(case_id_factor))

if (nrow(target_events_sample) > 0) {
  # Overall Gantt chart
  p4 <- ggplot(target_events_sample,
         aes(ymin = entity_num - 0.4,
             ymax = entity_num + 0.4,
             xmin = start_time,
             xmax = end_time,
             fill = event_type)) +
    geom_rect(alpha = 0.8) +
    scale_y_continuous(breaks = unique(target_events_sample$entity_num),
                       labels = levels(target_events_sample$case_id_factor)) +
    scale_x_datetime() +
    labs(title = paste("Activity Timeline (Gantt):", cohort_name, age_band),
         subtitle = "Each patient (row) shows activity codes as horizontal bars",
         x = "Event Time", y = "Patient ID", fill = "Event Type") +
    theme_bw() +
    theme(legend.position = "right",
          axis.text.y = element_text(size = 6))
  
  ggsave(file.path(plots_dir, sprintf("%s_%s_activity_milestones_gantt.png", cohort_name, age_band_fname)),
         plot = p4, width = 16, height = 12, dpi = 300)
  
  # Drug codes Gantt
  target_drug_events <- target_events_sample %>%
    filter(grepl("^DRUG:", activity)) %>%
    mutate(code_name = gsub("^DRUG:", "", activity))
  
  if (nrow(target_drug_events) > 0) {
    target_drug_entity_breaks <- sort(unique(target_drug_events$entity_num))
    target_drug_case_labels <- as.character(target_drug_events$case_id_factor[match(target_drug_entity_breaks, target_drug_events$entity_num)])
    
    p4_drug <- ggplot(target_drug_events,
         aes(ymin = entity_num - 0.4,
             ymax = entity_num + 0.4,
             xmin = start_time,
             xmax = end_time,
             fill = code_name)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = target_drug_entity_breaks,
                       labels = target_drug_case_labels) +
      scale_x_datetime() +
      labs(title = paste("Drug Codes Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows drug codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "Drug Code") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 6))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_gantt_drugs.png", cohort_name, age_band_fname)),
           plot = p4_drug, width = 18, height = 12, dpi = 300)
  }
  
  # ICD codes Gantt
  target_icd_events <- target_events_sample %>%
    filter(grepl("^ICD:", activity)) %>%
    mutate(code_name = gsub("^ICD:", "", activity))
  
  if (nrow(target_icd_events) > 0) {
    target_icd_entity_breaks <- sort(unique(target_icd_events$entity_num))
    target_icd_case_labels <- as.character(target_icd_events$case_id_factor[match(target_icd_entity_breaks, target_icd_events$entity_num)])
    
    p4_icd <- ggplot(target_icd_events,
         aes(ymin = entity_num - 0.4,
             ymax = entity_num + 0.4,
             xmin = start_time,
             xmax = end_time,
             fill = code_name)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = target_icd_entity_breaks,
                       labels = target_icd_case_labels) +
      scale_x_datetime() +
      labs(title = paste("ICD Codes Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows ICD codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "ICD Code") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 6))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_gantt_icd.png", cohort_name, age_band_fname)),
           plot = p4_icd, width = 18, height = 12, dpi = 300)
  }
  
  # CPT codes Gantt
  target_cpt_events <- target_events_sample %>%
    filter(grepl("^CPT:", activity)) %>%
    mutate(code_name = gsub("^CPT:", "", activity))
  
  if (nrow(target_cpt_events) > 0) {
    target_cpt_entity_breaks <- sort(unique(target_cpt_events$entity_num))
    target_cpt_case_labels <- as.character(target_cpt_events$case_id_factor[match(target_cpt_entity_breaks, target_cpt_events$entity_num)])
    
    p4_cpt <- ggplot(target_cpt_events,
         aes(ymin = entity_num - 0.4,
             ymax = entity_num + 0.4,
             xmin = start_time,
             xmax = end_time,
             fill = code_name)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = target_cpt_entity_breaks,
                       labels = target_cpt_case_labels) +
      scale_x_datetime() +
      labs(title = paste("CPT Codes Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows CPT codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "CPT Code") +
      theme_bw() +
      theme(legend.position = "right",
          axis.text.y = element_text(size = 6))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_gantt_cpt.png", cohort_name, age_band_fname)),
           plot = p4_cpt, width = 18, height = 12, dpi = 300)
  }
  
  # Activity sequence with top activities highlighted
  # For polypharmacy cohort, event_type = drug name (many levels); use color only to avoid
  # "Insufficient values in manual scale" when mapping shape to many event_types
  if (nrow(target_activity_freq) > 0) {
    top_activities <- target_activity_freq$activity[1:min(10, nrow(target_activity_freq))]
    target_events_top <- target_events_sample %>%
      mutate(activity_highlight = ifelse(activity %in% top_activities, activity, "Other"))
    
    p5 <- ggplot(target_events_top,
           aes(x = timestamp,
               y = case_id_factor,
               color = activity_highlight)) +
      geom_point(size = 2, alpha = 0.7, shape = 16) +
      scale_x_datetime() +
      labs(title = paste("Activity Sequence with Top Activities:", cohort_name, age_band),
           x = "Event Time", y = "Patient ID",
           color = "Activity (Top 10)") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 6))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_activity_sequence_top.png", cohort_name, age_band_fname)),
           plot = p5, width = 16, height = 12, dpi = 300)
  }
  
  log_msg("✓ Created overall activity frequency, Gantt timeline (overall + by code type), and activity sequence plots")
}

# Close PDF device (captures any base graphics from trace_explorer, process_map, etc.)
# This prevents Rplots.pdf from being created in the project root
dev.off()
cat("Closed PDF device. Base graphics saved to: ", rplots_path, "\n", sep = "")

log_msg("=", level = "INFO")
log_msg(sprintf("✓ bupaR analysis for POLYPHARMACY COHORT (first ED within 21d of drug) %s completed successfully", age_band), level = "INFO")
log_msg(sprintf("  Data partition: cohort_name=%s", cohort_name), level = "INFO")
log_msg("=", level = "INFO")



