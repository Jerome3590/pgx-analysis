#!/usr/bin/env Rscript
#
# End-to-end bupaR analysis for Cohort 2 (POLYPHARMACY_ED, non_opioid_ed),
# configurable age band (65–74, 75–84, 85–94).
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
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

project_root <- getwd()  # assume you launched from project root

cohort_name    <- "non_opioid_ed"
control_cohort <- "opioid_ed"

# Optional command line argument to set age band; default is 65-74
args <- commandArgs(trailingOnly = TRUE)
age_band <- if (length(args) >= 1) args[[1]] else "65-74"

age_band_fname <- gsub("-", "_", age_band)
train_years    <- c(2016L, 2017L, 2018L)

cat("=== bupaR Analysis: Cohort 2 (POLYPHARMACY_ED, non_opioid_ed) ===\n")
cat("  Age band:       ", age_band, "\n", sep = "")
cat("  Control cohort: ", control_cohort, "\n\n", sep = "")

# Cohort-specific target ICD definition (HCG* codes)
target_icd_patterns <- c("HCG")

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

# If not found, use first candidate (will error if file doesn't exist)
if (is.null(model_data_path)) {
  model_data_path <- model_data_candidates[1]
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
       "\nRun 3_feature_importance/create_model_data.py for this cohort/age band first.")
}

con <- dbConnect(duckdb::duckdb())

query <- sprintf(
  "SELECT * FROM read_parquet('%s') WHERE event_year IN (%s)",
  model_data_path,
  paste(train_years, collapse = ",")
)

pgx_df <- dbGetQuery(con, query)

cat("Loaded ", nrow(pgx_df), " events for ", cohort_name, " age_band=", age_band,
    " across years ", paste(train_years, collapse=","), "\n", sep = "")

pgx_df_target1 <- pgx_df %>%
  filter(target == 1L)

cat("Target=1 rows: ", nrow(pgx_df_target1), "\n", sep = "")

# -------------------------------------------------------------------
# Using all codes from model_events.parquet
# This ensures we capture all pre-HCG events for analysis
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Build DRUG/ICD/CPT activities and target_eventlog
# -------------------------------------------------------------------

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
  filter(!is.na(code), code != "", code != "NA") %>%
  mutate(
    activity = dplyr::case_when(
      source == "drug_name" ~ paste0("DRUG:", code),
      grepl("icd_diagnosis_code", source) ~ paste0("ICD:", code),
      source == "procedure_code" ~ paste0("CPT:", code),
      TRUE ~ code
    ),
    timestamp = as.POSIXct(event_date)
  )

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

cat("Target eventlog created.\n")
print(target_eventlog)

# -------------------------------------------------------------------
# Combined TARGET + CONTROL eventlog for Sankey
# -------------------------------------------------------------------

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

if (file.exists(control_model_data_path)) {
  query_control <- sprintf(
    "SELECT * FROM read_parquet('%s') WHERE event_year IN (%s)",
    control_model_data_path,
    paste(train_years, collapse = ",")
  )
  pgx_df_control <- dbGetQuery(con, query_control)
  cat("Loaded ", nrow(pgx_df_control), " control events for ", control_cohort,
      " age_band=", age_band, " across years ", paste(train_years, collapse=","), "\n", sep = "")
} else {
  warning("Control model_data parquet not found: ", control_model_data_path)
  pgx_df_control <- pgx_df[0, ]
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
  mutate(
    activity = dplyr::case_when(
      source == "drug_name" ~ paste0("DRUG:", code),
      grepl("icd_diagnosis_code", source) ~ paste0("ICD:", code),
      source == "procedure_code" ~ paste0("CPT:", code),
      TRUE ~ code
    ),
    timestamp = as.POSIXct(event_date)
  )

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

cat("Combined TARGET + CONTROL sankey_eventlog created.\n")
print(sankey_eventlog)

# -------------------------------------------------------------------
# Pre-HCG (before first HCG ICD) sequences
# -------------------------------------------------------------------

cat("\n--- Pre-HCG (before first HCG ICD) analysis ---\n")

ev_all <- events(target_eventlog) %>%
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
         event_index <= first_target_index)

pre_target_eventlog <- events_pre_target %>%
  eventlog(
    case_id     = "case_id",
    activity_id = "activity",
    timestamp   = "timestamp"
  )

cat("Pre-HCG eventlog summary:\n")
print(pre_target_eventlog)

# 1) Trace explorer (printed summary; visuals if running interactively)
trace_explorer(pre_target_eventlog, coverage = 0.8)

# 2) Drug-only sequences before HCG
pre_drug_sequences <- events(pre_target_eventlog) %>%
  arrange(case_id, timestamp) %>%
  filter(grepl("^DRUG:", activity)) %>%
  group_by(case_id) %>%
  summarise(
    drug_sequence = list(activity),
    .groups = "drop"
  )

cat("Sample pre-HCG drug-only sequences:\n")
print(head(pre_drug_sequences))

# 3) Process map for pre-HCG trajectories
process_map(pre_target_eventlog, type = "frequency")

# 4) Per-patient pre-HCG features
pre_patient_features <- events(pre_target_eventlog) %>%
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

target_times <- events(target_eventlog) %>%
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

pre_events_with_t <- events(pre_target_eventlog) %>%
  inner_join(target_times, by = "case_id") %>%
  mutate(
    dt_days = as.numeric(difftime(target_time, timestamp, units = "days"))
  )

hcg_time_features <- pre_events_with_t %>%
  group_by(case_id, target_time, first_time) %>%
  summarise(
    time_to_HCG_days        = as.numeric(max(dt_days, na.rm = TRUE)),
    n_events_30d            = sum(dt_days <= 30),
    n_events_90d            = sum(dt_days <= 90),
    n_events_180d           = sum(dt_days <= 180),
    n_drug_events_30d       = sum(dt_days <= 30 & grepl("^DRUG:", activity)),
    n_drug_events_90d       = sum(dt_days <= 90 & grepl("^DRUG:", activity)),
    n_drug_events_180d      = sum(dt_days <= 180 & grepl("^DRUG:", activity)),
    n_icd_events_30d        = sum(dt_days <= 30 & grepl("^ICD:", activity)),
    n_icd_events_90d        = sum(dt_days <= 90 & grepl("^ICD:", activity)),
    n_icd_events_180d       = sum(dt_days <= 180 & grepl("^ICD:", activity)),
    n_cpt_events_30d        = sum(dt_days <= 30 & grepl("^CPT:", activity)),
    n_cpt_events_90d        = sum(dt_days <= 90 & grepl("^CPT:", activity)),
    n_cpt_events_180d       = sum(dt_days <= 180 & grepl("^CPT:", activity)),
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
traces_target <- edeaR::traces(target_eventlog)
save_bupar_csv(
  traces_target,
  sprintf("%s_%s_train_target_traces_bupar.csv", cohort_name, age_band_fname)
)

# 2) Process Matrix and CSV export
pm_target <- process_matrix(target_eventlog, type = "frequency")
pm_target_df <- as.data.frame(pm_target)
save_bupar_csv(
  pm_target_df,
  sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name, age_band_fname)
)

# 3) Process Map visualization
process_map(target_eventlog, type = "frequency")

# Close PDF device (captures any base graphics from trace_explorer, process_map, etc.)
# This prevents Rplots.pdf from being created in the project root
dev.off()
cat("Closed PDF device. Base graphics saved to: ", rplots_path, "\n", sep = "")

cat("\n=== bupaR analysis for non_opioid_ed ", age_band, " completed. ===\n", sep = "")



