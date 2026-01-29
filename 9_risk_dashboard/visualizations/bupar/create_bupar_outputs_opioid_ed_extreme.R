#!/usr/bin/env Rscript
#
# bupaR analysis for extreme-density subset of Cohort 1 (OPIOID_ED)
# - Uses 4a_model_data/cohort_name=opioid_ed_extreme_density as the event source
# - Reuses FP-Growth TARGET-only itemsets from the base opioid_ed cohort
# - Produces the same style of process-mining plots and per-patient features
#   as create_bupar_outputs_opioid_ed.R, but scoped to the extreme-density patients.
#

# Set up user library path for package loading (Windows compatibility)
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.5")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
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
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

project_root <- getwd()  # assume you launched from project root

# Extreme-density cohort uses its own model_data root
cohort_name_extreme <- "opioid_ed_extreme_density"

# Base cohort used for FP-Growth itemsets / activity alphabet
base_fpgrowth_cohort <- "opioid_ed"

control_cohort <- "non_opioid_ed"

# Optional command line argument to set age band; default is 25-44
args <- commandArgs(trailingOnly = TRUE)
age_band <- if (length(args) >= 1) args[[1]] else "25-44"

age_band_fname <- gsub("-", "_", age_band)
train_years    <- c(2016L, 2017L, 2018L)

cat("=== bupaR Analysis: Extreme-Density Subset (OPIOID_ED) ===\n")
cat("  Age band:          ", age_band, "\n", sep = "")
cat("  Extreme cohort:    ", cohort_name_extreme, "\n", sep = "")
cat("  Base FP-Growth:    ", base_fpgrowth_cohort, "\n", sep = "")
cat("  Control cohort:    ", control_cohort, "\n\n", sep = "")

# Cohort-specific target ICD definition
target_icd_patterns <- c("F1120")   # opioid ED
include_post_target <- TRUE        # use post-F1120 only for descriptive analysis

# Prefer DTW protocol-filtered model_events_no_protocols.parquet if available.
model_data_dir <- file.path(
  project_root,
  "4a_model_data",
  paste0("cohort_name=", cohort_name_extreme),
  paste0("age_band=", age_band)
)
model_data_no_protocols <- file.path(model_data_dir, "model_events_no_protocols.parquet")
model_data_main         <- file.path(model_data_dir, "model_events.parquet")

if (file.exists(model_data_no_protocols)) {
  model_data_path <- model_data_no_protocols
} else {
  model_data_path <- model_data_main
}

fpgrowth_root <- file.path(
  project_root,
  "4_fpgrowth_analysis",
  "outputs",
  base_fpgrowth_cohort
)

target_dir_train <- file.path(fpgrowth_root, "target", age_band_fname, "train")

itemsets_drug_target_path    <- file.path(target_dir_train, "drug_name_itemsets_target_only.json")
itemsets_icd_target_path     <- file.path(target_dir_train, "icd_code_itemsets_target_only.json")
itemsets_medical_target_path <- file.path(target_dir_train, "medical_code_itemsets_target_only.json")

cat("Project root:          ", project_root, "\n", sep = "")
cat("Extreme model data:    ", model_data_path, "\n", sep = "")
cat("FP-Growth target dir:  ", target_dir_train, "\n\n", sep = "")

# -------------------------------------------------------------------
# Helper for saving CSVs locally + to S3, and central plots directory
# -------------------------------------------------------------------

bup_ar_output_root <- file.path(project_root, "10c_bupaR_dashboard_visual", "outputs")

save_bupar_csv <- function(df, filename,
                           cohort = cohort_name_extreme,
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

plots_dir <- file.path(bup_ar_output_root, cohort_name_extreme, age_band_fname, "plots")
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
}

rplots_path <- file.path(
  plots_dir,
  sprintf("%s_%s_Rplots.pdf", cohort_name_extreme, age_band_fname)
)

pdf(file = rplots_path, width = 12, height = 9)

# -------------------------------------------------------------------
# Load extreme model_data and build target-only subset
# -------------------------------------------------------------------

if (!file.exists(model_data_path)) {
  stop("Extreme model_data parquet not found at: ", model_data_path,
       "\nRun extract_extreme_density_cohort.py for this cohort/age band first.")
}

con <- dbConnect(duckdb::duckdb())

query <- sprintf(
  "SELECT * FROM read_parquet('%s') WHERE event_year IN (%s)",
  model_data_path,
  paste(train_years, collapse = ",")
)

pgx_df <- dbGetQuery(con, query)

cat("Loaded ", nrow(pgx_df), " events for ", cohort_name_extreme, " age_band=", age_band,
    " across years ", paste(train_years, collapse=","), "\n", sep = "")

pgx_df_target1 <- pgx_df %>%
  filter(target == 1L)

cat("Target=1 rows (extreme): ", nrow(pgx_df_target1), "\n", sep = "")

# -------------------------------------------------------------------
# Load FP-Growth target-only itemsets from BASE cohort and build allowed code set
# -------------------------------------------------------------------

allowed_codes <- character(0)

if (file.exists(itemsets_drug_target_path)) {
  drug_itemsets_target <- fromJSON(itemsets_drug_target_path, simplifyDataFrame = TRUE)
  drug_codes <- unique(unlist(drug_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, drug_codes)
  cat("Loaded ", length(drug_codes), " unique drug codes from base target-only itemsets.\n", sep = "")
} else {
  warning("Drug target-only itemsets not found at ", itemsets_drug_target_path)
}

if (file.exists(itemsets_icd_target_path)) {
  icd_itemsets_target <- fromJSON(itemsets_icd_target_path, simplifyDataFrame = TRUE)
  icd_codes <- unique(unlist(icd_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, icd_codes)
  cat("Loaded ", length(icd_codes), " unique ICD codes from base target-only itemsets.\n", sep = "")
} else {
  warning("ICD target-only itemsets not found at ", itemsets_icd_target_path)
}

if (file.exists(itemsets_medical_target_path)) {
  medical_itemsets_target <- fromJSON(itemsets_medical_target_path, simplifyDataFrame = TRUE)
  medical_codes <- unique(unlist(medical_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, medical_codes)
  cat("Loaded ", length(medical_codes), " unique medical (ICD+CPT) codes from base target-only itemsets.\n", sep = "")
} else {
  warning("Medical target-only itemsets not found at ", itemsets_medical_target_path)
}

# Always ensure F1120 is included in the activity alphabet
allowed_codes <- union(allowed_codes, "F1120")

cat("Total unique allowed codes from base FP-Growth itemsets (incl. F1120): ",
    length(allowed_codes), "\n\n", sep = "")

# -------------------------------------------------------------------
# Build DRUG/ICD/CPT activities and target_eventlog for EXTREME cohort
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

cat("Extreme target eventlog created.\n")
print(target_eventlog)

# -------------------------------------------------------------------
# Reuse the same analysis blocks as in create_bupar_outputs_opioid_ed.R,
# but scoped only to the extreme target_eventlog (no CONTROL cohort).
# -------------------------------------------------------------------

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

cat("\n--- Pre-F1120 (before first ICD:F1120) analysis: EXTREME cohort ---\n")

events_pre_target <- ev_all %>%
  filter(!is.na(first_target_index),
         event_index < first_target_index) %>%
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

cat("Pre-F1120 eventlog summary (extreme):\n")
print(pre_target_eventlog)

plots_dir <- file.path(bup_ar_output_root, cohort_name_extreme, age_band_fname, "plots")
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

if (nrow(as.data.frame(pre_target_eventlog)) > 0) {
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

  save_bupar_csv(
    traces_pre_df,
    sprintf("%s_%s_train_target_pre_f1120_traces_bupar.csv", cohort_name_extreme, age_band_fname)
  )

  if (nrow(pre_top_sequences) > 0) {
    save_bupar_csv(
      pre_top_sequences,
      sprintf("%s_%s_train_target_pre_f1120_traces_top_bupar.csv", cohort_name_extreme, age_band_fname)
    )
  }

  if (nrow(pre_rare_sequences) > 0) {
    save_bupar_csv(
      pre_rare_sequences,
      sprintf("%s_%s_train_target_pre_f1120_traces_rare_bupar.csv", cohort_name_extreme, age_band_fname)
    )
  }

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
    sprintf("%s_%s_train_target_pre_f1120_patient_features_bupar.csv", cohort_name_extreme, age_band_fname)
  )
} else {
  cat("No pre-F1120 events for extreme cohort; skipping pre-F1120 trace/features.\n")
}

# Time-to-F1120 features for extreme cohort
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

if (nrow(pre_events_with_t) > 0) {
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
    sprintf("%s_%s_train_target_time_to_f1120_features_bupar.csv", cohort_name_extreme, age_band_fname)
  )
}

# Post-F1120 descriptive analysis for extreme cohort
if (include_post_target) {
  cat("\n--- Post-F1120 (after first ICD:F1120) analysis: EXTREME cohort ---\n")

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

  cat("Post-F1120 eventlog summary (extreme):\n")
  print(post_target_eventlog)

  if (nrow(as.data.frame(post_target_eventlog)) > 0) {
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

    save_bupar_csv(
      traces_post_df,
      sprintf("%s_%s_train_target_post_f1120_traces_bupar.csv", cohort_name_extreme, age_band_fname)
    )

    if (nrow(post_top_sequences) > 0) {
      save_bupar_csv(
        post_top_sequences,
        sprintf("%s_%s_train_target_post_f1120_traces_top_bupar.csv", cohort_name_extreme, age_band_fname)
      )
    }

    if (nrow(post_rare_sequences) > 0) {
      save_bupar_csv(
        post_rare_sequences,
        sprintf("%s_%s_train_target_post_f1120_traces_rare_bupar.csv", cohort_name_extreme, age_band_fname)
      )
    }

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
      sprintf("%s_%s_train_target_post_f1120_patient_features_bupar.csv", cohort_name_extreme, age_band_fname)
    )
  } else {
    cat("No post-F1120 events for extreme cohort; skipping post-F1120 trace/features.\n")
  }
}

# Target-only global process mining for extreme cohort
cat("\n--- Target-only global process mining: EXTREME cohort ---\n")

traces_target <- bupaR::traces(target_eventlog)
save_bupar_csv(
  traces_target,
  sprintf("%s_%s_train_target_traces_bupar.csv", cohort_name_extreme, age_band_fname)
)

traces_target_df <- as.data.frame(traces_target) %>%
  arrange(desc(absolute_frequency))

total_cases <- n_cases(target_eventlog)
top_n_threshold <- max(20, ceiling(total_cases * 0.1))
rare_threshold <- 1

top_sequences <- traces_target_df %>%
  filter(absolute_frequency >= top_n_threshold) %>%
  mutate(sequence_category = "top")

rare_sequences <- traces_target_df %>%
  filter(absolute_frequency <= rare_threshold) %>%
  mutate(sequence_category = "rare")

if (nrow(top_sequences) > 0) {
  save_bupar_csv(
    top_sequences,
    sprintf("%s_%s_train_target_traces_top_bupar.csv", cohort_name_extreme, age_band_fname)
  )
}

if (nrow(rare_sequences) > 0) {
  save_bupar_csv(
    rare_sequences,
    sprintf("%s_%s_train_target_traces_rare_bupar.csv", cohort_name_extreme, age_band_fname)
  )
}

pm_target <- tryCatch({
  process_matrix(target_eventlog, type = "frequency")
}, error = function(e) {
  cat("Note: process_matrix skipped due to error:", conditionMessage(e), "\n")
  NULL
})

pm_target_df <- as.data.frame(pm_target)
save_bupar_csv(
  pm_target_df,
  sprintf("%s_%s_train_target_process_matrix_bupar.csv", cohort_name_extreme, age_band_fname)
)

# Basic activity frequency and Gantt-style plots for extreme cohort
plots_dir <- file.path(bup_ar_output_root, cohort_name_extreme, age_band_fname, "plots")
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

target_activity_freq <- target_eventlog %>%
  group_by(activity) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(30)

p3 <- ggplot(target_activity_freq, aes(x = reorder(activity, count), y = count)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = paste("Overall Activity Frequency (EXTREME):", cohort_name_extreme, age_band),
       x = "Activity", y = "Frequency") +
  theme_bw()

ggsave(file.path(plots_dir, sprintf("%s_%s_overall_activity_frequency.png", cohort_name_extreme, age_band_fname)),
       plot = p3, width = 12, height = 10, dpi = 300)

target_events_df <- as.data.frame(target_eventlog) %>%
  arrange(case_id, timestamp) %>%
  mutate(event_type = case_when(
    grepl("^DRUG:", activity) ~ "Drug",
    grepl("^ICD:", activity) ~ "Diagnosis",
    grepl("^CPT:", activity) ~ "Procedure",
    TRUE ~ "Other"
  )) %>%
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
  labs(title = paste("Activity Timeline (Gantt, EXTREME):", cohort_name_extreme, age_band),
       subtitle = "Each patient (row) shows activity codes as horizontal bars",
       x = "Event Time", y = "Patient ID", fill = "Event Type") +
  theme_bw() +
  theme(legend.position = "right",
        axis.text.y = element_text(size = 6))

ggsave(file.path(plots_dir, sprintf("%s_%s_activity_milestones_gantt.png", cohort_name_extreme, age_band_fname)),
       plot = p4, width = 16, height = 12, dpi = 300)

# Mirror plots to central 5_feature_engineering/feature_engineering_outputs directory
fe_plots_dir <- file.path(
  project_root,
  "5_feature_engineering",
  "feature_engineering_outputs",
  "5_bupar",
  cohort_name_extreme,
  age_band,
  "plots"
)
if (dir.exists(plots_dir)) {
  dir.create(fe_plots_dir, recursive = TRUE, showWarnings = FALSE)
  plot_files <- list.files(plots_dir, full.names = TRUE)
  if (length(plot_files) > 0) {
    cat("[INFO] Copying extreme BupaR plots to", fe_plots_dir, "\n")
    file.copy(plot_files, fe_plots_dir, overwrite = TRUE)
  }
}

if (grDevices::dev.cur() > 1) {
  grDevices::dev.off()
}

cat("\n=== bupaR analysis for EXTREME opioid_ed subset ", age_band, " completed. ===\n", sep = "")

