#!/usr/bin/env Rscript
#
# Merge all per-patient BupaR features into a final tabular dataset.
#
# This script combines:
# 1. Pre-F1120 features (from R script)
# 2. Post-F1120 features (from R script)
# 3. Time-to-F1120 features (from R script)
# 4. Sequence features (from create_sequence_features.R, if available)
#
# Output:
# - Saves final merged features to: outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv
# - This is the final file ready for joining with model_data in the final model step.
#

# Set up user library path for package loading (Windows compatibility)
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.5")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(duckdb)
  library(dplyr)
  library(readr)
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
project_root <- "."
cohort_name <- "opioid_ed"
age_band <- "0-12"
train_label <- "train"

i <- 1
while (i <= length(args)) {
  if (args[i] == "--project-root") {
    project_root <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--cohort-name" || args[i] == "--cohort") {
    cohort_name <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--age-band") {
    age_band <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--train-label") {
    train_label <- args[i + 1]
    i <- i + 2
  } else {
    i <- i + 1
  }
}

project_root <- normalizePath(project_root)
age_band_fname <- gsub("-", "_", age_band)

cat("=== Merging BupaR Features ===\n")
cat("  Cohort:      ", cohort_name, "\n", sep = "")
cat("  Age band:    ", age_band, "\n", sep = "")
cat("  Train label: ", train_label, "\n\n", sep = "")

# -------------------------------------------------------------------
# Paths: resolve model_events same as create_bupar_outputs (3b first, then 4_model_data)
# -------------------------------------------------------------------

cohort_slug_3b <- if (cohort_name == "opioid_ed") "opioid" else "polypharmacy"
path_3b <- file.path(project_root,
                     "3b_feature_importance_eda", "outputs", "cohorts", "input_model_data",
                     paste0("cohort_name=", cohort_slug_3b),
                     paste0("age_band=", age_band),
                     "model_events.parquet")
if (file.exists(path_3b)) {
  model_data_path <- path_3b
} else {
  model_data_root <- NULL
  data_root <- Sys.getenv("PGX_DATA_ROOT")
  candidates <- c(
    if (nzchar(data_root)) file.path(data_root, "4_model_data") else character(0),
    "/mnt/nvme/4_model_data",
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
  model_data_dir <- file.path(model_data_root,
                              paste0("cohort_name=", cohort_name),
                              paste0("age_band=", age_band))
  model_data_no_protocols <- file.path(model_data_dir, "model_events_no_protocols.parquet")
  model_data_main         <- file.path(model_data_dir, "model_events.parquet")
  model_data_path <- if (file.exists(model_data_no_protocols)) model_data_no_protocols else model_data_main
}

bupar_output_dir <- file.path(project_root, "10c_bupaR_dashboard_visual", "outputs",
                               cohort_name, age_band_fname, "features")

pre_features_csv <- file.path(bupar_output_dir,
                              paste0(cohort_name, "_", age_band_fname, "_", train_label,
                                     "_target_pre_f1120_patient_features_bupar.csv"))

post_features_csv <- file.path(bupar_output_dir,
                               paste0(cohort_name, "_", age_band_fname, "_", train_label,
                                      "_target_post_f1120_patient_features_bupar.csv"))

time_to_features_csv <- file.path(bupar_output_dir,
                                  paste0(cohort_name, "_", age_band_fname, "_", train_label,
                                         "_target_time_to_f1120_features_bupar.csv"))

sequence_features_csv <- file.path(project_root, "10c_bupaR_dashboard_visual", "outputs",
                                   "feature_engineering",
                                   paste0("sequence_features_", cohort_name, "_", age_band_fname, ".csv"))

# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

if (!file.exists(model_data_path)) {
  stop(paste("model_data parquet not found:", model_data_path))
}

# For some cohorts/age bands (e.g., where F1120 is always the first event),
# there may be no pre-/post-/time-to-F1120 sequences. In that case we treat
# the corresponding CSVs as optional and fall back to empty feature frames.

if (!file.exists(pre_features_csv)) {
  cat("[INFO] Pre-F1120 BupaR features not found at", pre_features_csv, "- using empty feature frame\n")
}

if (!file.exists(post_features_csv)) {
  cat("[INFO] Post-F1120 BupaR features not found at", post_features_csv, "- using empty feature frame\n")
}

if (!file.exists(time_to_features_csv)) {
  cat("[INFO] Time-to-F1120 BupaR features not found at", time_to_features_csv, "- using empty feature frame\n")
}

# -------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------

cat("[INFO] Reading model_data from", model_data_path, "\n")
con <- dbConnect(duckdb())
base_df <- dbGetQuery(con, paste0("
  SELECT DISTINCT mi_person_key
  FROM read_parquet('", model_data_path, "')
  WHERE target = 1
"))
dbDisconnect(con)

cat("[INFO] Loaded", nrow(base_df), "unique target patients from model_data\n")

if (file.exists(pre_features_csv)) {
cat("[INFO] Reading pre-F1120 features from", pre_features_csv, "\n")
pre_df <- read_csv(pre_features_csv, show_col_types = FALSE)
} else {
  pre_df <- tibble(mi_person_key = character(0))
}

if (file.exists(post_features_csv)) {
cat("[INFO] Reading post-F1120 features from", post_features_csv, "\n")
post_df <- read_csv(post_features_csv, show_col_types = FALSE)
} else {
  post_df <- tibble(mi_person_key = character(0))
}

if (file.exists(time_to_features_csv)) {
cat("[INFO] Reading time-to-F1120 features from", time_to_features_csv, "\n")
time_to_df <- read_csv(time_to_features_csv, show_col_types = FALSE)
} else {
  time_to_df <- tibble(mi_person_key = character(0))
}

# Expect case_id column from BupaR outputs; rename to mi_person_key for consistency
if ("case_id" %in% names(pre_df)) {
  pre_df <- rename(pre_df, mi_person_key = case_id)
}
if ("case_id" %in% names(post_df)) {
  post_df <- rename(post_df, mi_person_key = case_id)
}
if ("case_id" %in% names(time_to_df)) {
  time_to_df <- rename(time_to_df, mi_person_key = case_id)
}

# Ensure mi_person_key is character type for consistent merging
base_df$mi_person_key <- as.character(base_df$mi_person_key)
pre_df$mi_person_key <- as.character(pre_df$mi_person_key)
post_df$mi_person_key <- as.character(post_df$mi_person_key)
time_to_df$mi_person_key <- as.character(time_to_df$mi_person_key)

# Load sequence features if available
sequence_df <- NULL
if (file.exists(sequence_features_csv)) {
  cat("[INFO] Reading sequence features from", sequence_features_csv, "\n")
  sequence_df <- read_csv(sequence_features_csv, show_col_types = FALSE)
  if ("case_id" %in% names(sequence_df)) {
    sequence_df <- rename(sequence_df, mi_person_key = case_id)
  }
  if ("mi_person_key" %in% names(sequence_df)) {
    sequence_df$mi_person_key <- as.character(sequence_df$mi_person_key)
  }
} else {
  cat("[INFO] Sequence features not found at", sequence_features_csv, ", skipping\n")
}

# -------------------------------------------------------------------
# Merge Features
# -------------------------------------------------------------------

merged <- base_df %>%
  left_join(pre_df, by = "mi_person_key") %>%
  left_join(post_df, by = "mi_person_key") %>%
  left_join(time_to_df, by = "mi_person_key")

# Add sequence features if available
if (!is.null(sequence_df)) {
  merged <- merged %>%
    left_join(sequence_df, by = "mi_person_key")
  cat("[INFO] Added", ncol(sequence_df) - 1, "sequence features\n")
}

# -------------------------------------------------------------------
# Save Output
# -------------------------------------------------------------------

out_dir <- file.path(project_root, "10c_bupaR_dashboard_visual", "outputs", "feature_engineering")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_path <- file.path(out_dir, paste0("bupaR_added_features_", cohort_name, "_", age_band_fname, ".csv"))
cat("[INFO] Writing merged BupaR features to", out_path, "(", nrow(merged), "rows)\n")
write_csv(merged, out_path)

# Mirror BupaR features (and optional sequence features) to central 5_feature_engineering/feature_engineering_outputs directory
fe_root <- file.path(
  project_root,
  "5_feature_engineering",
  "feature_engineering_outputs",
  "5_bupar",
  cohort_name,
  age_band
)
dir.create(fe_root, recursive = TRUE, showWarnings = FALSE)

fe_target <- file.path(fe_root, basename(out_path))
cat("[INFO] Copying merged BupaR features to", fe_target, "\n")
file.copy(out_path, fe_target, overwrite = TRUE)

if (file.exists(sequence_features_csv)) {
  seq_target <- file.path(fe_root, basename(sequence_features_csv))
  cat("[INFO] Copying sequence features to", seq_target, "\n")
  file.copy(sequence_features_csv, seq_target, overwrite = TRUE)
}

# -------------------------------------------------------------------
# Upload to S3
# -------------------------------------------------------------------

s3_path <- paste0("s3://pgxdatalake/gold/feature_engineering/5_bupar/", cohort_name, "/", age_band,
                  "/bupaR_added_features_", cohort_name, "_", age_band_fname, ".csv")

aws_cli <- Sys.which("aws")
if (aws_cli != "") {
  tryCatch({
    cat("[INFO] Uploading to S3:", s3_path, "\n")
    system2(aws_cli, c("s3", "cp", out_path, s3_path), stdout = TRUE, stderr = TRUE)
    cat("[INFO] S3 upload successful\n")
  }, error = function(e) {
    warning(paste("S3 upload failed:", e$message))
    cat("[WARNING] S3 upload failed:", e$message, "\n")
  })
} else {
  cat("[INFO] AWS CLI not found, skipping S3 upload\n")
}

cat("[INFO] Done.\n")

