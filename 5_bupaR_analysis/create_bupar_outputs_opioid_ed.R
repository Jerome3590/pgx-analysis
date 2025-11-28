#!/usr/bin/env Rscript
#
# End-to-end bupaR analysis for Cohort 1 (OPIOID_ED), configurable age band
# - Builds target-only and combined event logs from model_data + FP-Growth TRAIN outputs
# - Runs pre- and post-F1120 sequence analyses
# - Exports pre-/post-F1120 per-patient features, trace tables, and process matrices
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
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

project_root <- getwd()  # assume you launched from project root

cohort_name    <- "opioid_ed"
control_cohort <- "non_opioid_ed"

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

model_data_path <- file.path(
  project_root,
  "model_data",
  paste0("cohort_name=", cohort_name),
  paste0("age_band=", age_band),
  "model_events.parquet"
)

fpgrowth_root <- file.path(
  project_root,
  "4_fpgrowth_analysis",
  "outputs",
  cohort_name
)

target_dir_train <- file.path(fpgrowth_root, "target", age_band_fname, "train")

itemsets_drug_target_path    <- file.path(target_dir_train, "drug_name_itemsets_target_only.json")
itemsets_icd_target_path     <- file.path(target_dir_train, "icd_code_itemsets_target_only.json")
itemsets_medical_target_path <- file.path(target_dir_train, "medical_code_itemsets_target_only.json")

cat("Project root:         ", project_root, "\n", sep = "")
cat("Model data path:      ", model_data_path, "\n", sep = "")
cat("FP-Growth target dir: ", target_dir_train, "\n\n", sep = "")

# -------------------------------------------------------------------
# Helper for saving CSVs locally + to S3
# -------------------------------------------------------------------

bup_ar_output_root <- file.path(project_root, "5_bupaR_analysis", "outputs")

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
# Load FP-Growth target-only itemsets and build allowed code set
# -------------------------------------------------------------------

allowed_codes <- character(0)

if (file.exists(itemsets_drug_target_path)) {
  drug_itemsets_target <- fromJSON(itemsets_drug_target_path, simplifyDataFrame = TRUE)
  drug_codes <- unique(unlist(drug_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, drug_codes)
  cat("Loaded ", length(drug_codes), " unique drug codes from target-only itemsets.\n", sep = "")
} else {
  warning("Drug target-only itemsets not found at ", itemsets_drug_target_path)
}

if (file.exists(itemsets_icd_target_path)) {
  icd_itemsets_target <- fromJSON(itemsets_icd_target_path, simplifyDataFrame = TRUE)
  icd_codes <- unique(unlist(icd_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, icd_codes)
  cat("Loaded ", length(icd_codes), " unique ICD codes from target-only itemsets.\n", sep = "")
} else {
  warning("ICD target-only itemsets not found at ", itemsets_icd_target_path)
}

if (file.exists(itemsets_medical_target_path)) {
  medical_itemsets_target <- fromJSON(itemsets_medical_target_path, simplifyDataFrame = TRUE)
  medical_codes <- unique(unlist(medical_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, medical_codes)
  cat("Loaded ", length(medical_codes), " unique medical (ICD+CPT) codes from target-only itemsets.\n", sep = "")
} else {
  warning("Medical target-only itemsets not found at ", itemsets_medical_target_path)
}

# Always ensure F1120 is included in the activity alphabet
allowed_codes <- union(allowed_codes, "F1120")

cat("Total unique allowed codes from FP-Growth itemsets (incl. F1120): ",
    length(allowed_codes), "\n\n", sep = "")

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

control_model_data_path <- file.path(
  project_root,
  "model_data",
  paste0("cohort_name=", control_cohort),
  paste0("age_band=", age_band),
  "model_events.parquet"
)

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
# Pre-F1120 (before first ICD:F1120) sequences
# -------------------------------------------------------------------

cat("\n--- Pre-F1120 (before first ICD:F1120) analysis ---\n")

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

cat("Pre-F1120 eventlog summary:\n")
print(pre_target_eventlog)

# 1) Trace explorer (printed summary; visuals if running interactively)
trace_explorer(pre_target_eventlog, coverage = 0.8)

# 2) Drug-only sequences before F1120
pre_drug_sequences <- events(pre_target_eventlog) %>%
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
process_map(pre_target_eventlog, type = "frequency")

# 4) Per-patient pre-F1120 features
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
  sprintf("%s_%s_train_target_pre_f1120_patient_features_bupar.csv", cohort_name, age_band_fname)
)

# -------------------------------------------------------------------
# Time-to-F1120 and time-window features (per patient)
# -------------------------------------------------------------------

library(lubridate)

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

time_to_event_features <- pre_events_with_t %>%
  group_by(case_id, target_time, first_time) %>%
  summarise(
    time_to_F1120_days        = as.numeric(max(dt_days, na.rm = TRUE)),
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

# -------------------------------------------------------------------
# Post-F1120 (after first ICD:F1120) sequences – descriptive only
# -------------------------------------------------------------------

if (include_post_target) {
  cat("\n--- Post-F1120 (after first ICD:F1120) analysis ---\n")

  events_post_target <- ev_all %>%
    filter(!is.na(first_target_index),
           event_index > first_target_index)

  post_target_eventlog <- events_post_target %>%
    eventlog(
      case_id     = "case_id",
      activity_id = "activity",
      timestamp   = "timestamp"
    )

  cat("Post-F1120 eventlog summary:\n")
  print(post_target_eventlog)

  # 1) Trace explorer: post-F1120 trajectories (descriptive)
  trace_explorer(post_target_eventlog, coverage = 0.8)

  # 2) Process map for post-F1120 trajectories
  process_map(post_target_eventlog, type = "frequency")

  # 3) Per-patient post-F1120 features (for descriptive analysis only)
  post_patient_features <- events(post_target_eventlog) %>%
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

cat("\n=== bupaR analysis for opioid_ed ", age_band, " completed. ===\n", sep = "")


