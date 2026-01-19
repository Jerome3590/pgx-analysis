#!/usr/bin/env Rscript
#
# End-to-end bupaR analysis for Cohort 1 (OPIOID_ED), configurable age band
# - Builds target-only and combined event logs from model_data
# - Runs pre- and post-F1120 sequence analyses
# - Exports pre-/post-F1120 per-patient features, trace tables, and process matrices
# Uses all events from model_events.parquet directly
#

# Set up user library path for package loading (Windows compatibility)
# Use explicit version string to avoid evaluation issues
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.5")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

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
  library(ggplot2)
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

project_root <- getwd()  # assume you launched from project root

cohort_name    <- "opioid_ed"
control_cohort <- "non_opioid_non_ed"

# Optional command line argument to set age band; default is 0-12
args <- commandArgs(trailingOnly = TRUE)
age_band <- if (length(args) >= 1) args[[1]] else "0-12"

age_band_fname <- gsub("-", "_", age_band)
train_years    <- c(2016L, 2017L, 2018L)

cat("=== bupaR Analysis: Cohort 1 (OPIOID_ED) ===\n")
cat("  Age band:       ", age_band, "\n", sep = "")
cat("  Control cohort: ", control_cohort, "\n\n", sep = "")

# Cohort-specific target ICD definition
target_icd_patterns <- c("F1120")   # opioid ED
include_post_target <- TRUE        # use post-F1120 only for descriptive analysis

# OS-aware model data path resolution
# Try data root first (for EC2/Linux: /mnt/nvme/4a_model_data), then project root
data_root <- Sys.getenv("PGX_DATA_ROOT", "")
if (data_root == "") {
  # Auto-detect: Linux uses /mnt/nvme, Windows uses project root
  if (.Platform$OS.type == "unix") {
    data_root <- "/mnt/nvme"
  } else {
    data_root <- project_root
  }
}

# Try multiple locations for model_data
model_data_candidates <- c(
  file.path(data_root, "4a_model_data", paste0("cohort_name=", cohort_name), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(project_root, "4a_model_data", paste0("cohort_name=", cohort_name), paste0("age_band=", age_band), "model_events.parquet")
)

model_data_path <- NULL
for (candidate in model_data_candidates) {
  if (file.exists(candidate)) {
    model_data_path <- candidate
    break
  }
}

# If not found, try downloading from S3
if (is.null(model_data_path)) {
  model_data_path <- model_data_candidates[1]
  
  # Try to download from S3 if not found locally
  s3_path <- paste0("s3://pgxdatalake/gold/cohorts_model_data/cohort_name=", cohort_name, "/age_band=", age_band, "/model_events.parquet")
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
       "\nRun 3_feature_importance/create_model_data.py for this cohort/age band first.")
}

# -------------------------------------------------------------------
# Load Helper Functions
# -------------------------------------------------------------------

# Source logging utilities from r_helpers
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

log_msg("Connecting to DuckDB...")
con <- dbConnect(duckdb::duckdb())

log_msg(sprintf("Loading target cohort data from: %s", model_data_path))
log_msg(sprintf("Filtering for target=1 and years: %s", paste(train_years, collapse=",")))

# Optimized query: Filter target=1 in DuckDB and only select needed columns
# This avoids loading unnecessary data into R memory
query_target <- sprintf(
  "SELECT 
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
  FROM read_parquet('%s') 
  WHERE event_year IN (%s) AND target = 1",
  model_data_path,
  paste(train_years, collapse = ",")
)

log_msg("Executing DuckDB query for target cohort...")
pgx_df_target1 <- dbGetQuery(con, query_target)

log_msg(sprintf("✓ Loaded %d target=1 events for %s age_band=%s across years %s", 
                nrow(pgx_df_target1), cohort_name, age_band, paste(train_years, collapse=",")))

# -------------------------------------------------------------------
# Using all codes from model_events.parquet
# This ensures we capture all pre- and post-F1120 events for leakage analysis
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Build DRUG/ICD/CPT activities and target_eventlog
# -------------------------------------------------------------------

log_msg("Transforming target data from wide to long format using DuckDB UNPIVOT...")
# Optimized: Use DuckDB UNPIVOT to do wide-to-long transformation efficiently
# This processes the data in DuckDB's columnar format before loading into R
query_long <- sprintf(
  "SELECT 
    mi_person_key,
    event_date,
    source,
    CAST(code AS VARCHAR) as code,
    CASE 
      WHEN source = 'drug_name' THEN 'DRUG:' || code
      WHEN source LIKE '%%icd_diagnosis_code%%' THEN 'ICD:' || code
      WHEN source = 'procedure_code' THEN 'CPT:' || code
      ELSE code
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
  WHERE code IS NOT NULL AND code != '' AND code != 'NA'",
  model_data_path,
  paste(train_years, collapse = ",")
)

log_msg("Executing UNPIVOT query...")
pgx_df_target1_long <- dbGetQuery(con, query_long) %>%
  mutate(
    timestamp = as.POSIXct(event_date)
  )

log_msg(sprintf("✓ Transformed to long format: %d events", nrow(pgx_df_target1_long)))

log_msg("Creating BupaR eventlog object for target cohort...")
target_eventlog <- pgx_df_target1_long %>%
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

log_msg("✓ Target eventlog created")
cat("Target eventlog summary:\n")
print(target_eventlog)

# -------------------------------------------------------------------
# Combined TARGET + CONTROL eventlog for Sankey
# -------------------------------------------------------------------

# Source utility functions for control cohort management
utils_path <- file.path(project_root, "3b_feature_importance_eda", "1_bupaR", "control_cohort_utils.R")
if (file.exists(utils_path)) {
  source(utils_path)
} else {
  stop("Control cohort utility functions not found. Expected at: ", utils_path)
}

# Control model data path - use same OS-aware resolution
control_model_data_candidates <- c(
  file.path(data_root, "4a_model_data", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet"),
  file.path(project_root, "4a_model_data", paste0("cohort_name=", control_cohort), paste0("age_band=", age_band), "model_events.parquet")
)

control_model_data_path <- NULL
for (candidate in control_model_data_candidates) {
  if (file.exists(candidate)) {
    control_model_data_path <- candidate
    break
  }
}

# If not found, use first candidate
if (is.null(control_model_data_path)) {
  control_model_data_path <- control_model_data_candidates[1]
}

log_msg("Validating and ensuring control cohort with 5:1 ratio...")
# Use utility function to validate and ensure control cohort with 5:1 ratio
control_result <- ensure_control_cohort_with_ratio(
  con = con,
  control_cohort = control_cohort,
  control_model_data_path = control_model_data_path,
  model_data_path = model_data_path,
  age_band = age_band,
  train_years = train_years,
  project_root = project_root,
  expected_ratio = 5.0,
  tolerance = 0.2
)

pgx_df_control <- control_result$pgx_df_control
if (control_result$was_recreated) {
  log_msg("⚠ Control cohort was recreated to achieve 5:1 ratio", level = "WARN")
} else if (control_result$validation_passed) {
  log_msg("✓ Control cohort validation passed")
}

# Assert that control PATIENTS are not duplicated (check distinct patients, not event-level data)
# Event-level data will have duplicate mi_person_key values (one row per event per patient)
# Note: We only need mi_person_key for verification, so we can query just that column
if (nrow(pgx_df_control) > 0) {
  distinct_control_patients <- unique(pgx_df_control$mi_person_key)
  stopifnot(!anyDuplicated(distinct_control_patients))
  cat("Verified ", length(distinct_control_patients), " distinct control patients (", nrow(pgx_df_control), " total events)\n", sep = "")
}

log_msg("Creating combined target+control query with DuckDB UNION ALL and UNPIVOT...")
# Optimized: Use DuckDB UNION ALL and UNPIVOT for combined target+control transformation
# This processes both datasets in DuckDB before loading into R
query_combined_long <- sprintf(
  "SELECT 
    mi_person_key,
    event_date,
    'target' as group,
    source,
    CAST(code AS VARCHAR) as code,
    CASE 
      WHEN source = 'drug_name' THEN 'DRUG:' || code
      WHEN source LIKE '%%icd_diagnosis_code%%' THEN 'ICD:' || code
      WHEN source = 'procedure_code' THEN 'CPT:' || code
      ELSE code
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
  WHERE code IS NOT NULL AND code != '' AND code != 'NA'
  
  UNION ALL
  
  SELECT 
    mi_person_key,
    event_date,
    'control' as group,
    source,
    CAST(code AS VARCHAR) as code,
    CASE 
      WHEN source = 'drug_name' THEN 'DRUG:' || code
      WHEN source LIKE '%%icd_diagnosis_code%%' THEN 'ICD:' || code
      WHEN source = 'procedure_code' THEN 'CPT:' || code
      ELSE code
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
    WHERE event_year IN (%s)
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
  WHERE code IS NOT NULL AND code != '' AND code != 'NA'",
  model_data_path,
  paste(train_years, collapse = ","),
  control_model_data_path,
  paste(train_years, collapse = ",")
)

log_msg("Executing combined query for target+control data...")
pgx_df_all_long <- dbGetQuery(con, query_combined_long) %>%
  mutate(
    timestamp = as.POSIXct(event_date)
  )

log_msg(sprintf("✓ Loaded %d combined events (target + control)", nrow(pgx_df_all_long)))
log_msg("Creating combined BupaR eventlog for Sankey visualization...")
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
# Pre-F1120 (before first ICD:F1120) sequences
# -------------------------------------------------------------------

log_msg("=", level = "INFO")
log_msg("Starting Pre-F1120 (before first ICD:F1120) analysis", level = "INFO")
log_msg("=", level = "INFO")

log_msg("Identifying target events and calculating event indices...")
ev_all <- target_eventlog %>%
  arrange(case_id, timestamp) %>%
  group_by(case_id) %>%
  mutate(
    event_index = row_number(),
    is_target_icd = Reduce(`|`, lapply(target_icd_patterns, function(p) grepl(p, activity))),
    has_target   = any(is_target_icd),
    first_target_index = ifelse(has_target,
                                min(event_index[is_target_icd]),
                                NA_integer_)
  ) %>%
  ungroup()

events_pre_target <- ev_all %>%
  filter(!is.na(first_target_index),
         event_index < first_target_index) %>%  # Use < to EXCLUDE F1120 itself (for final model)
  mutate(activity_instance_id = row_number())

pre_target_eventlog <- events_pre_target %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    timestamp            = "timestamp",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id"
  )

log_msg("Pre-F1120 eventlog summary:")
print(pre_target_eventlog)

# Check if pre-F1120 eventlog is empty
if (n_events(pre_target_eventlog) == 0) {
  log_msg("⚠ No pre-F1120 events found; skipping pre-F1120 trace and feature analysis", level = "WARN")
  
  # Create empty data frames for consistency
  traces_pre_df <- data.frame(
    trace = character(0),
    absolute_frequency = integer(0),
    relative_frequency = numeric(0)
  )
  
  pre_total_cases <- 0
  pre_top_n_threshold <- 10
  pre_rare_threshold <- 1
  
  pre_top_sequences <- traces_pre_df
  pre_rare_sequences <- traces_pre_df
  
  # Create empty patient features
  pre_patient_features <- data.frame(
    case_id = character(0),
    pre_n_events = integer(0),
    pre_n_drug_events = integer(0),
    pre_n_icd_events = integer(0),
    pre_n_cpt_events = integer(0),
    pre_unique_drugs = integer(0),
    pre_unique_icds = integer(0),
    pre_unique_cpts = integer(0)
  )
  
  pre_drug_sequences <- data.frame(
    case_id = character(0),
    drug_sequence = I(list())
  )
  
} else {
  # 1) Trace explorer (printed summary; visuals if running interactively)
  tryCatch({
    trace_explorer(pre_target_eventlog, coverage = 0.8)
  }, error = function(e) {
    cat("Warning: trace_explorer failed:", conditionMessage(e), "\n")
  })
  
  # Save pre-F1120 traces and categorize into top/rare
  traces_pre <- bupaR::traces(pre_target_eventlog)
  traces_pre_df <- as.data.frame(traces_pre) %>%
    arrange(desc(absolute_frequency))
  
  pre_total_cases <- n_cases(pre_target_eventlog)
  pre_top_n_threshold <- max(10, ceiling(pre_total_cases * 0.1))
  pre_rare_threshold <- 1
  
  pre_top_sequences <- traces_pre_df %>%
    filter(absolute_frequency >= pre_top_n_threshold) %>%
    mutate(sequence_category = "top")
  
  pre_rare_sequences <- traces_pre_df %>%
    filter(absolute_frequency <= pre_rare_threshold) %>%
    mutate(sequence_category = "rare")

  # Save all pre-F1120 traces
  save_bupar_csv(
    traces_pre_df,
    sprintf("%s_%s_train_target_pre_f1120_traces_bupar.csv", cohort_name, age_band_fname)
  )
  
  # Save top pre-F1120 sequences
  if (nrow(pre_top_sequences) > 0) {
    save_bupar_csv(
      pre_top_sequences,
      sprintf("%s_%s_train_target_pre_f1120_traces_top_bupar.csv", cohort_name, age_band_fname)
    )
    cat(sprintf("Saved %d top pre-F1120 sequences (frequency >= %d)\n", nrow(pre_top_sequences), pre_top_n_threshold))
  }
  
  # Save rare pre-F1120 sequences
  if (nrow(pre_rare_sequences) > 0) {
    save_bupar_csv(
      pre_rare_sequences,
      sprintf("%s_%s_train_target_pre_f1120_traces_rare_bupar.csv", cohort_name, age_band_fname)
    )
    cat(sprintf("Saved %d rare pre-F1120 sequences (frequency <= %d)\n", nrow(pre_rare_sequences), pre_rare_threshold))
  }
  
  # 2) Drug-only sequences before F1120
  pre_drug_sequences <- pre_target_eventlog %>%
    arrange(case_id, timestamp) %>%
    filter(grepl("^DRUG:", activity)) %>%
    group_by(case_id) %>%
    summarise(
      drug_sequence = list(activity),
      .groups = "drop"
    )
  
  cat("Sample pre-F1120 drug-only sequences:\n")
  print(head(pre_drug_sequences))
  
  # 3) Process map for pre-F1120 trajectories
  # For small datasets, use ggplot2 visualizations instead of process_map
  plots_dir <- file.path(bup_ar_output_root, cohort_name, age_band_fname, "plots")
  if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)
  
  # Activity frequency plot
  pre_activity_freq <- pre_target_eventlog %>%
    group_by(activity) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count)) %>%
    head(20)
  
  if (nrow(pre_activity_freq) > 0) {
    p1 <- ggplot(pre_activity_freq, aes(x = reorder(activity, count), y = count)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(title = paste("Pre-F1120 Activity Frequency:", cohort_name, age_band),
           x = "Activity", y = "Frequency") +
      theme_bw()
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_pre_f1120_activity_frequency.png", cohort_name, age_band_fname)),
           plot = p1, width = 10, height = 8, dpi = 300)
  }
  
  # Pre-F1120 Gantt-style plot (patient = job, activity = stage)
  pre_events_df <- as.data.frame(pre_target_eventlog) %>%
    arrange(case_id, timestamp) %>%
    mutate(event_type = case_when(
      grepl("^DRUG:", activity) ~ "Drug",
      grepl("^ICD:", activity) ~ "Diagnosis",
      grepl("^CPT:", activity) ~ "Procedure",
      TRUE ~ "Other"
    )) %>%
    # For point events, add small duration (1 day) to create visible bars
    mutate(start_time = timestamp,
           end_time = timestamp + lubridate::ddays(1))
  
  pre_sample_cases <- unique(pre_events_df$case_id)[1:min(20, length(unique(pre_events_df$case_id)))]
  pre_events_sample <- pre_events_df %>%
    filter(case_id %in% pre_sample_cases) %>%
    mutate(case_id_factor = factor(case_id, levels = rev(pre_sample_cases)),
           entity_num = as.numeric(case_id_factor))
  
  if (nrow(pre_events_sample) > 0) {
    p1b <- ggplot(pre_events_sample,
           aes(ymin = entity_num - 0.4,
               ymax = entity_num + 0.4,
               xmin = start_time,
               xmax = end_time,
               fill = event_type)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = unique(pre_events_sample$entity_num),
                         labels = levels(pre_events_sample$case_id_factor)) +
      scale_x_datetime() +
      labs(title = paste("Pre-F1120 Activity Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows activity codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "Event Type") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 7))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_pre_f1120_gantt.png", cohort_name, age_band_fname)),
           plot = p1b, width = 14, height = 10, dpi = 300)
    
    # Pre-F1120 Gantt charts by code type (Drug, ICD, CPT)
    # Drug codes Gantt
    pre_drug_events <- pre_events_sample %>%
      filter(grepl("^DRUG:", activity)) %>%
      mutate(code_name = gsub("^DRUG:", "", activity))
    
    if (nrow(pre_drug_events) > 0) {
      # Get unique entity numbers and their corresponding case IDs
      drug_entity_breaks <- sort(unique(pre_drug_events$entity_num))
      drug_case_labels <- as.character(pre_drug_events$case_id_factor[match(drug_entity_breaks, pre_drug_events$entity_num)])
      
      p1c_drug <- ggplot(pre_drug_events,
             aes(ymin = entity_num - 0.4,
                 ymax = entity_num + 0.4,
                 xmin = start_time,
                 xmax = end_time,
                 fill = code_name)) +
        geom_rect(alpha = 0.8) +
        scale_y_continuous(breaks = drug_entity_breaks,
                           labels = drug_case_labels) +
        scale_x_datetime() +
        labs(title = paste("Pre-F1120 Drug Codes Timeline (Gantt):", cohort_name, age_band),
             subtitle = "Each patient (row) shows drug codes as horizontal bars",
             x = "Event Time", y = "Patient ID", fill = "Drug Code") +
        theme_bw() +
        theme(legend.position = "right",
              axis.text.y = element_text(size = 7))
      
      ggsave(file.path(plots_dir, sprintf("%s_%s_pre_f1120_gantt_drugs.png", cohort_name, age_band_fname)),
             plot = p1c_drug, width = 16, height = 10, dpi = 300)
    }
    
    # ICD codes Gantt
    pre_icd_events <- pre_events_sample %>%
      filter(grepl("^ICD:", activity)) %>%
      mutate(code_name = gsub("^ICD:", "", activity))
    
    if (nrow(pre_icd_events) > 0) {
      icd_entity_breaks <- sort(unique(pre_icd_events$entity_num))
      icd_case_labels <- as.character(pre_icd_events$case_id_factor[match(icd_entity_breaks, pre_icd_events$entity_num)])
      
      p1c_icd <- ggplot(pre_icd_events,
             aes(ymin = entity_num - 0.4,
                 ymax = entity_num + 0.4,
                 xmin = start_time,
                 xmax = end_time,
                 fill = code_name)) +
        geom_rect(alpha = 0.8) +
        scale_y_continuous(breaks = icd_entity_breaks,
                           labels = icd_case_labels) +
        scale_x_datetime() +
        labs(title = paste("Pre-F1120 ICD Codes Timeline (Gantt):", cohort_name, age_band),
             subtitle = "Each patient (row) shows ICD codes as horizontal bars",
             x = "Event Time", y = "Patient ID", fill = "ICD Code") +
        theme_bw() +
        theme(legend.position = "right",
              axis.text.y = element_text(size = 7))
      
      ggsave(file.path(plots_dir, sprintf("%s_%s_pre_f1120_gantt_icd.png", cohort_name, age_band_fname)),
             plot = p1c_icd, width = 16, height = 10, dpi = 300)
    }
    
    # CPT codes Gantt
    pre_cpt_events <- pre_events_sample %>%
      filter(grepl("^CPT:", activity)) %>%
      mutate(code_name = gsub("^CPT:", "", activity))
    
    if (nrow(pre_cpt_events) > 0) {
      cpt_entity_breaks <- sort(unique(pre_cpt_events$entity_num))
      cpt_case_labels <- as.character(pre_cpt_events$case_id_factor[match(cpt_entity_breaks, pre_cpt_events$entity_num)])
      
      p1c_cpt <- ggplot(pre_cpt_events,
             aes(ymin = entity_num - 0.4,
                 ymax = entity_num + 0.4,
                 xmin = start_time,
                 xmax = end_time,
                 fill = code_name)) +
        geom_rect(alpha = 0.8) +
        scale_y_continuous(breaks = cpt_entity_breaks,
                           labels = cpt_case_labels) +
        scale_x_datetime() +
        labs(title = paste("Pre-F1120 CPT Codes Timeline (Gantt):", cohort_name, age_band),
             subtitle = "Each patient (row) shows CPT codes as horizontal bars",
             x = "Event Time", y = "Patient ID", fill = "CPT Code") +
        theme_bw() +
        theme(legend.position = "right",
              axis.text.y = element_text(size = 7))
      
      ggsave(file.path(plots_dir, sprintf("%s_%s_pre_f1120_gantt_cpt.png", cohort_name, age_band_fname)),
             plot = p1c_cpt, width = 16, height = 10, dpi = 300)
    }
  }
  
  cat("Created pre-F1120 activity frequency and Gantt timeline plots (overall + by code type).\n")
  
  # 4) Per-patient pre-F1120 features
  pre_patient_features <- pre_target_eventlog %>%
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
    sprintf("%s_%s_train_target_pre_f1120_patient_features_bupar.csv", cohort_name, age_band_fname)
  )
  
  # -------------------------------------------------------------------
  # Time-to-F1120 and time-window features (per patient)
  # -------------------------------------------------------------------
  
  library(lubridate)
  
  target_times <- target_eventlog %>%
    arrange(case_id, timestamp) %>%
    group_by(case_id) %>%
    mutate(
      is_target_icd = Reduce(`|`, lapply(target_icd_patterns, function(p) grepl(p, activity))),
      has_target    = any(is_target_icd)
    ) %>%
    filter(has_target) %>%
    summarise(
      target_time = min(timestamp[is_target_icd]),
      first_time  = min(timestamp),
      .groups = "drop"
    )
  
  pre_events_with_t <- pre_target_eventlog %>%
    inner_join(target_times, by = "case_id") %>%
    mutate(
      dt_days = as.numeric(difftime(target_time, timestamp, units = "days"))
    )
  
  time_to_event_features <- pre_events_with_t %>%
    group_by(case_id, target_time, first_time) %>%
    summarise(
      time_to_F1120_days        = ifelse(
        all(is.na(dt_days)), NA_real_,
        as.numeric(max(dt_days, na.rm = TRUE))
      ),
      n_events_30d              = sum(dt_days <= 30),
      n_events_90d              = sum(dt_days <= 90),
      n_events_180d             = sum(dt_days <= 180),
      n_drug_events_30d         = sum(dt_days <= 30 & grepl("^DRUG:", activity)),
      n_drug_events_90d         = sum(dt_days <= 90 & grepl("^DRUG:", activity)),
      n_drug_events_180d        = sum(dt_days <= 180 & grepl("^DRUG:", activity)),
      n_icd_events_30d          = sum(dt_days <= 30 & grepl("^ICD:", activity)),
      n_icd_events_90d          = sum(dt_days <= 90 & grepl("^ICD:", activity)),
      n_icd_events_180d         = sum(dt_days <= 180 & grepl("^ICD:", activity)),
      n_cpt_events_30d          = sum(dt_days <= 30 & grepl("^CPT:", activity)),
      n_cpt_events_90d          = sum(dt_days <= 90 & grepl("^CPT:", activity)),
      n_cpt_events_180d         = sum(dt_days <= 180 & grepl("^CPT:", activity)),
      .groups = "drop"
    )
  
  save_bupar_csv(
    time_to_event_features,
    sprintf("%s_%s_train_target_time_to_f1120_features_bupar.csv", cohort_name, age_band_fname)
  )
}  # End of else block for non-empty pre-F1120 eventlog

# -------------------------------------------------------------------
# Post-F1120 (after first ICD:F1120) sequences – descriptive only
# -------------------------------------------------------------------

if (include_post_target) {
  cat("\n--- Post-F1120 (after first ICD:F1120) analysis ---\n")

  events_post_target <- ev_all %>%
    filter(!is.na(first_target_index),
           event_index > first_target_index) %>%
    mutate(activity_instance_id = row_number())

  post_target_eventlog <- events_post_target %>%
    eventlog(
      case_id              = "case_id",
      activity_id          = "activity",
      activity_instance_id = "activity_instance_id",
      timestamp            = "timestamp",
      lifecycle_id         = "lifecycle_id",
      resource_id          = "resource_id"
    )

  cat("Post-F1120 eventlog summary:\n")
  print(post_target_eventlog)

  # 1) Trace explorer: post-F1120 trajectories (descriptive)
  trace_explorer(post_target_eventlog, coverage = 0.8)
  
  # Save post-F1120 traces and categorize into top/rare
  traces_post <- bupaR::traces(post_target_eventlog)
  traces_post_df <- as.data.frame(traces_post) %>%
    arrange(desc(absolute_frequency))
  
  post_total_cases <- n_cases(post_target_eventlog)
  post_top_n_threshold <- max(10, ceiling(post_total_cases * 0.1))
  post_rare_threshold <- 1
  
  post_top_sequences <- traces_post_df %>%
    filter(absolute_frequency >= post_top_n_threshold) %>%
    mutate(sequence_category = "top")
  
  post_rare_sequences <- traces_post_df %>%
    filter(absolute_frequency <= post_rare_threshold) %>%
    mutate(sequence_category = "rare")
  
  # Save all post-F1120 traces
  save_bupar_csv(
    traces_post_df,
    sprintf("%s_%s_train_target_post_f1120_traces_bupar.csv", cohort_name, age_band_fname)
  )
  
  # Save top post-F1120 sequences
  if (nrow(post_top_sequences) > 0) {
    save_bupar_csv(
      post_top_sequences,
      sprintf("%s_%s_train_target_post_f1120_traces_top_bupar.csv", cohort_name, age_band_fname)
    )
    cat(sprintf("Saved %d top post-F1120 sequences (frequency >= %d)\n", nrow(post_top_sequences), post_top_n_threshold))
  }
  
  # Save rare post-F1120 sequences
  if (nrow(post_rare_sequences) > 0) {
    save_bupar_csv(
      post_rare_sequences,
      sprintf("%s_%s_train_target_post_f1120_traces_rare_bupar.csv", cohort_name, age_band_fname)
    )
    cat(sprintf("Saved %d rare post-F1120 sequences (frequency <= %d)\n", nrow(post_rare_sequences), post_rare_threshold))
  }

  # 2) Process map for post-F1120 trajectories
  # For small datasets, use ggplot2 visualizations instead of process_map
  plots_dir <- file.path(bup_ar_output_root, cohort_name, age_band_fname, "plots")
  if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)
  
  # Activity frequency plot
  post_activity_freq <- post_target_eventlog %>%
    group_by(activity) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count)) %>%
    head(20)
  
  p2 <- ggplot(post_activity_freq, aes(x = reorder(activity, count), y = count)) +
    geom_bar(stat = "identity", fill = "darkred") +
    coord_flip() +
    labs(title = paste("Post-F1120 Activity Frequency:", cohort_name, age_band),
         x = "Activity", y = "Frequency") +
    theme_minimal()
  
  ggsave(file.path(plots_dir, sprintf("%s_%s_post_f1120_activity_frequency.png", cohort_name, age_band_fname)),
         plot = p2, width = 10, height = 8, dpi = 300)
  
  # Post-F1120 Gantt-style plot (patient = job, activity = stage)
  post_events_df <- as.data.frame(post_target_eventlog) %>%
    arrange(case_id, timestamp) %>%
    mutate(event_type = case_when(
      grepl("^DRUG:", activity) ~ "Drug",
      grepl("^ICD:", activity) ~ "Diagnosis",
      grepl("^CPT:", activity) ~ "Procedure",
      TRUE ~ "Other"
    )) %>%
    # For point events, add small duration (1 day) to create visible bars
    mutate(start_time = timestamp,
           end_time = timestamp + lubridate::ddays(1))
  
  post_sample_cases <- unique(post_events_df$case_id)[1:min(20, length(unique(post_events_df$case_id)))]
  post_events_sample <- post_events_df %>%
    filter(case_id %in% post_sample_cases) %>%
    mutate(case_id_factor = factor(case_id, levels = rev(post_sample_cases)),
           entity_num = as.numeric(case_id_factor))
  
  p2b <- ggplot(post_events_sample,
         aes(ymin = entity_num - 0.4,
             ymax = entity_num + 0.4,
             xmin = start_time,
             xmax = end_time,
             fill = event_type)) +
    geom_rect(alpha = 0.8) +
    scale_y_continuous(breaks = unique(post_events_sample$entity_num),
                       labels = levels(post_events_sample$case_id_factor)) +
    scale_x_datetime() +
    labs(title = paste("Post-F1120 Activity Timeline (Gantt):", cohort_name, age_band),
         subtitle = "Each patient (row) shows activity codes as horizontal bars",
         x = "Event Time", y = "Patient ID", fill = "Event Type") +
    theme_bw() +
    theme(legend.position = "right",
          axis.text.y = element_text(size = 7))
  
  ggsave(file.path(plots_dir, sprintf("%s_%s_post_f1120_gantt.png", cohort_name, age_band_fname)),
         plot = p2b, width = 14, height = 10, dpi = 300)
  
  # Post-F1120 Gantt charts by code type (Drug, ICD, CPT)
  # Drug codes Gantt
  post_drug_events <- post_events_sample %>%
    filter(grepl("^DRUG:", activity)) %>%
    mutate(code_name = gsub("^DRUG:", "", activity))
  
  if (nrow(post_drug_events) > 0) {
    post_drug_entity_breaks <- sort(unique(post_drug_events$entity_num))
    post_drug_case_labels <- as.character(post_drug_events$case_id_factor[match(post_drug_entity_breaks, post_drug_events$entity_num)])
    
    p2c_drug <- ggplot(post_drug_events,
           aes(ymin = entity_num - 0.4,
               ymax = entity_num + 0.4,
               xmin = start_time,
               xmax = end_time,
               fill = code_name)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = post_drug_entity_breaks,
                         labels = post_drug_case_labels) +
      scale_x_datetime() +
      labs(title = paste("Post-F1120 Drug Codes Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows drug codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "Drug Code") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 7))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_post_f1120_gantt_drugs.png", cohort_name, age_band_fname)),
           plot = p2c_drug, width = 16, height = 10, dpi = 300)
  }
  
  # ICD codes Gantt
  post_icd_events <- post_events_sample %>%
    filter(grepl("^ICD:", activity)) %>%
    mutate(code_name = gsub("^ICD:", "", activity))
  
  if (nrow(post_icd_events) > 0) {
    post_icd_entity_breaks <- sort(unique(post_icd_events$entity_num))
    post_icd_case_labels <- as.character(post_icd_events$case_id_factor[match(post_icd_entity_breaks, post_icd_events$entity_num)])
    
    p2c_icd <- ggplot(post_icd_events,
           aes(ymin = entity_num - 0.4,
               ymax = entity_num + 0.4,
               xmin = start_time,
               xmax = end_time,
               fill = code_name)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = post_icd_entity_breaks,
                         labels = post_icd_case_labels) +
      scale_x_datetime() +
      labs(title = paste("Post-F1120 ICD Codes Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows ICD codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "ICD Code") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 7))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_post_f1120_gantt_icd.png", cohort_name, age_band_fname)),
           plot = p2c_icd, width = 16, height = 10, dpi = 300)
  }
  
  # CPT codes Gantt
  post_cpt_events <- post_events_sample %>%
    filter(grepl("^CPT:", activity)) %>%
    mutate(code_name = gsub("^CPT:", "", activity))
  
  if (nrow(post_cpt_events) > 0) {
    post_cpt_entity_breaks <- sort(unique(post_cpt_events$entity_num))
    post_cpt_case_labels <- as.character(post_cpt_events$case_id_factor[match(post_cpt_entity_breaks, post_cpt_events$entity_num)])
    
    p2c_cpt <- ggplot(post_cpt_events,
           aes(ymin = entity_num - 0.4,
               ymax = entity_num + 0.4,
               xmin = start_time,
               xmax = end_time,
               fill = code_name)) +
      geom_rect(alpha = 0.8) +
      scale_y_continuous(breaks = post_cpt_entity_breaks,
                         labels = post_cpt_case_labels) +
      scale_x_datetime() +
      labs(title = paste("Post-F1120 CPT Codes Timeline (Gantt):", cohort_name, age_band),
           subtitle = "Each patient (row) shows CPT codes as horizontal bars",
           x = "Event Time", y = "Patient ID", fill = "CPT Code") +
      theme_bw() +
      theme(legend.position = "right",
            axis.text.y = element_text(size = 7))
    
    ggsave(file.path(plots_dir, sprintf("%s_%s_post_f1120_gantt_cpt.png", cohort_name, age_band_fname)),
           plot = p2c_cpt, width = 16, height = 10, dpi = 300)
  }
  
  cat("Created post-F1120 activity frequency and Gantt timeline plots (overall + by code type).\n")

  # 3) Per-patient post-F1120 features (for descriptive analysis only)
  post_patient_features <- post_target_eventlog %>%
    arrange(case_id, timestamp) %>%
    group_by(case_id) %>%
    summarise(
      post_n_events            = n(),
      post_n_drug_events       = sum(grepl("^DRUG:", activity)),
      post_n_icd_events        = sum(grepl("^ICD:", activity)),
      post_n_cpt_events        = sum(grepl("^CPT:", activity)),
      post_n_unique_activities = n_distinct(activity),
      .groups = "drop"
    )

  save_bupar_csv(
    post_patient_features,
    sprintf("%s_%s_train_target_post_f1120_patient_features_bupar.csv", cohort_name, age_band_fname)
  )
}

# -------------------------------------------------------------------
# Target-only global process mining (traces + process matrix)
# -------------------------------------------------------------------

cat("\n--- Target-only global process mining ---\n")

# 1) Trace Explorer: most frequent target trajectories
trace_explorer(target_eventlog, coverage = 0.8)

# Save trace summary as tabular output
traces_target <- bupaR::traces(target_eventlog)
save_bupar_csv(
  traces_target,
  sprintf("%s_%s_train_target_traces_bupar.csv", cohort_name, age_band_fname)
)

# Categorize traces into top sequences and rare sequences
# Top sequences: most frequent traces (e.g., top 20% by frequency or top N by absolute frequency)
# Rare sequences: traces that appear only once or very infrequently
traces_target_df <- as.data.frame(traces_target) %>%
  arrange(desc(absolute_frequency))

# Define thresholds
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
# Note: process_matrix may fail with small datasets - wrap in tryCatch
pm_target <- tryCatch({
  process_matrix(target_eventlog, type = "frequency")
}, error = function(e) {
  cat("Note: process_matrix skipped due to error:", conditionMessage(e), "\n")
  NULL
})
pm_target_df <- as.data.frame(pm_target)
save_bupar_csv(
  pm_target_df,
  sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
)

# 3) Process Map visualization
# For small datasets, use ggplot2 visualizations instead of process_map
plots_dir <- file.path(bup_ar_output_root, cohort_name, age_band_fname, "plots")
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

# Activity frequency plot (overall)
target_activity_freq <- target_eventlog %>%
  group_by(activity) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(30)

p3 <- ggplot(target_activity_freq, aes(x = reorder(activity, count), y = count)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = paste("Overall Activity Frequency:", cohort_name, age_band),
       x = "Activity", y = "Frequency") +
  theme_bw()

ggsave(file.path(plots_dir, sprintf("%s_%s_overall_activity_frequency.png", cohort_name, age_band_fname)),
       plot = p3, width = 12, height = 10, dpi = 300)

# Gantt-style timeline (patient = job, activity = stage)
# Sample up to 30 cases for visualization
target_events_df <- as.data.frame(target_eventlog) %>%
  arrange(case_id, timestamp) %>%
  mutate(event_type = case_when(
    grepl("^DRUG:", activity) ~ "Drug",
    grepl("^ICD:", activity) ~ "Diagnosis",
    grepl("^CPT:", activity) ~ "Procedure",
    TRUE ~ "Other"
  )) %>%
  # For point events, add small duration (1 day) to create visible bars
  mutate(start_time = timestamp,
         end_time = timestamp + lubridate::ddays(1))

sample_cases <- unique(target_events_df$case_id)[1:min(30, length(unique(target_events_df$case_id)))]
target_events_sample <- target_events_df %>%
  filter(case_id %in% sample_cases) %>%
  mutate(case_id_factor = factor(case_id, levels = rev(sample_cases)),
         entity_num = as.numeric(case_id_factor))

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

# Overall Gantt charts by code type (Drug, ICD, CPT)
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
top_activities <- target_activity_freq$activity[1:min(10, nrow(target_activity_freq))]
target_events_top <- target_events_sample %>%
  mutate(activity_highlight = ifelse(activity %in% top_activities, activity, "Other"))

p5 <- ggplot(target_events_top,
       aes(x = timestamp,
           y = case_id_factor,
           color = activity_highlight,
           shape = event_type)) +
  geom_point(size = 2, alpha = 0.7) +
  scale_x_datetime() +
  scale_shape_manual(values = c("Drug" = 16, "Diagnosis" = 17, "Procedure" = 18, "Other" = 1)) +
  labs(title = paste("Activity Sequence with Top Activities:", cohort_name, age_band),
       x = "Event Time", y = "Patient ID",
       color = "Activity (Top 10)", shape = "Event Type") +
  theme_bw() +
  theme(legend.position = "right",
        axis.text.y = element_text(size = 6))

ggsave(file.path(plots_dir, sprintf("%s_%s_activity_sequence_top.png", cohort_name, age_band_fname)),
       plot = p5, width = 16, height = 12, dpi = 300)

cat("Created overall activity frequency, Gantt timeline (overall + by code type), and activity sequence plots.\n")

# Close PDF device (captures any base graphics from trace_explorer, process_map, etc.)
# This prevents Rplots.pdf from being created in the project root
dev.off()
cat("Closed PDF device. Base graphics saved to: ", rplots_path, "\n", sep = "")

cat("\n=== bupaR analysis for opioid_ed ", age_band, " completed. ===\n", sep = "")


