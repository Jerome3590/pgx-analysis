#!/usr/bin/env Rscript
#
# Create sequence features from top and rare trace sequences.
#
# This script extracts features from bupaR trace analysis:
# - Top sequences: Most frequent patient trajectories
# - Rare sequences: Unique or infrequent patient trajectories
#
# Features created:
# - Binary indicators for whether patient follows top/rare sequences
# - Sequence frequency features
# - Sequence category features (top vs rare)
#
# Output:
# - Saves to: outputs/feature_engineering/sequence_features_{cohort}_{age_band}.csv
# - This intermediate file is then merged with other bupaR features by add_bupar_features_to_model_data.R
#

# Set up user library path for package loading (Windows compatibility)
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.5")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

suppressPackageStartupMessages({
  library(duckdb)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Parse command line arguments (supporting --cohort-name and --age-band format)
args <- commandArgs(trailingOnly = TRUE)
cohort_name <- NULL
age_band <- NULL
train_label <- "train"

i <- 1
while (i <= length(args)) {
  if (args[i] == "--cohort-name" || args[i] == "--cohort") {
    cohort_name <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--age-band") {
    age_band <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--train-label") {
    train_label <- args[i + 1]
    i <- i + 2
  } else {
    # Legacy format: positional arguments
    if (is.null(cohort_name)) {
      cohort_name <- args[i]
    } else if (is.null(age_band)) {
      age_band <- args[i]
    } else {
      train_label <- args[i]
    }
    i <- i + 1
  }
}

if (is.null(cohort_name) || is.null(age_band)) {
  stop("Usage: Rscript create_sequence_features.R --cohort-name <cohort> --age-band <age_band> [--train-label <label>]")
}

project_root <- getwd()
age_band_fname <- gsub("-", "_", age_band)

cat("=== Creating Sequence Features ===\n")
cat("  Cohort:      ", cohort_name, "\n", sep = "")
cat("  Age band:    ", age_band, "\n", sep = "")
cat("  Train label: ", train_label, "\n\n", sep = "")

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

load_traces <- function(traces_path) {
  # Load trace sequences from CSV file
  if (!file.exists(traces_path)) {
    warning(paste("Traces file not found:", traces_path))
    return(data.frame())
  }
  
  traces_df <- read_csv(traces_path, show_col_types = FALSE)
  cat("Loaded", nrow(traces_df), "traces from", basename(traces_path), "\n")
  return(traces_df)
}

extract_patient_traces_from_model_data <- function(model_data_path, cohort_name, target_filter = NULL) {
  # Extract patient traces from model_data parquet file
  # Creates a trace (sequence of activities) for each patient
  if (!file.exists(model_data_path)) {
    warning(paste("Model data file not found:", model_data_path))
    return(data.frame())
  }
  
  con <- dbConnect(duckdb())
  
  # Build target filter clause
  target_clause <- ""
  if (!is.null(target_filter)) {
    target_clause <- paste("AND target =", target_filter)
  }
  
  # Extract patient traces
  query <- paste0("
    WITH patient_events AS (
        SELECT 
            mi_person_key as case_id,
            event_date,
            CASE 
                WHEN drug_name IS NOT NULL THEN 'DRUG:' || drug_name
                WHEN primary_icd_diagnosis_code IS NOT NULL THEN 'ICD:' || primary_icd_diagnosis_code
                WHEN procedure_code IS NOT NULL THEN 'CPT:' || procedure_code
                ELSE NULL
            END as activity
        FROM read_parquet('", model_data_path, "')
        WHERE (drug_name IS NOT NULL 
               OR primary_icd_diagnosis_code IS NOT NULL 
               OR procedure_code IS NOT NULL)
          ", target_clause, "
    )
    SELECT 
        case_id,
        LIST(activity ORDER BY event_date) as trace_list
    FROM patient_events
    WHERE activity IS NOT NULL
    GROUP BY case_id
  ")
  
  traces_df <- dbGetQuery(con, query)
  dbDisconnect(con)
  
  # Convert list to comma-separated string
  if (nrow(traces_df) > 0) {
    traces_df$trace <- sapply(traces_df$trace_list, function(x) {
      if (is.list(x)) {
        paste(x, collapse = ",")
      } else {
        ""
      }
    })
    traces_df <- traces_df[, c("case_id", "trace")]
  }
  
  filter_desc <- if (is.null(target_filter)) {
    "all"
  } else if (target_filter == 1) {
    "target"
  } else {
    "control"
  }
  cat("Extracted traces for", nrow(traces_df), filter_desc, "patients from model_data\n")
  
  return(traces_df)
}

create_sequence_features <- function(patient_traces, top_sequences, rare_sequences, 
                                     sequence_type = "overall", match_method = "exact") {
  # Create sequence features from patient traces and top/rare sequences
  cat("Creating", sequence_type, "sequence features...\n")
  
  if (nrow(patient_traces) == 0) {
    return(data.frame())
  }
  
  # Create lookup sets and dictionaries
  top_traces_set <- if (nrow(top_sequences) > 0) {
    unique(as.character(top_sequences$trace))
  } else {
    character(0)
  }
  
  rare_traces_set <- if (nrow(rare_sequences) > 0) {
    unique(as.character(rare_sequences$trace))
  } else {
    character(0)
  }
  
  # Create frequency lookup
  top_freq_dict <- if (nrow(top_sequences) > 0) {
    setNames(top_sequences$absolute_frequency, as.character(top_sequences$trace))
  } else {
    numeric(0)
  }
  
  rare_freq_dict <- if (nrow(rare_sequences) > 0) {
    setNames(rare_sequences$absolute_frequency, as.character(rare_sequences$trace))
  } else {
    numeric(0)
  }
  
  # Initialize feature dataframe
  features_df <- patient_traces[, "case_id", drop = FALSE]
  
  trace_str <- as.character(patient_traces$trace)
  
  # Feature 1: Binary indicator for top sequence match
  features_df[[paste0(sequence_type, "_is_top_sequence")]] <- as.integer(trace_str %in% top_traces_set)
  
  # Feature 2: Binary indicator for rare sequence match
  features_df[[paste0(sequence_type, "_is_rare_sequence")]] <- as.integer(trace_str %in% rare_traces_set)
  
  # Feature 3: Top sequence frequency (if matched)
  features_df[[paste0(sequence_type, "_top_sequence_frequency")]] <- 
    ifelse(trace_str %in% names(top_freq_dict), 
           unname(top_freq_dict[trace_str]), 
           0)
  
  # Feature 4: Rare sequence frequency (if matched)
  features_df[[paste0(sequence_type, "_rare_sequence_frequency")]] <- 
    ifelse(trace_str %in% names(rare_freq_dict), 
           unname(rare_freq_dict[trace_str]), 
           0)
  
  # Feature 5: Sequence category (top, rare, or other)
  features_df[[paste0(sequence_type, "_sequence_category")]] <- 
    ifelse(trace_str %in% top_traces_set, "top",
           ifelse(trace_str %in% rare_traces_set, "rare", "other"))
  
  # Feature 6: Sequence match count (how many top sequences match)
  count_top_matches <- function(trace_str_val) {
    if (length(top_traces_set) == 0) return(0)
    count <- 0
    for (top_seq in top_traces_set) {
      if (match_method == "exact") {
        if (trace_str_val == top_seq) count <- count + 1
      } else {
        if (grepl(top_seq, trace_str_val, fixed = TRUE) || grepl(trace_str_val, top_seq, fixed = TRUE)) {
          count <- count + 1
        }
      }
    }
    return(count)
  }
  
  features_df[[paste0(sequence_type, "_top_sequence_match_count")]] <- 
    sapply(trace_str, count_top_matches)
  
  # Feature 7: Sequence match count for rare sequences
  count_rare_matches <- function(trace_str_val) {
    if (length(rare_traces_set) == 0) return(0)
    count <- 0
    for (rare_seq in rare_traces_set) {
      if (match_method == "exact") {
        if (trace_str_val == rare_seq) count <- count + 1
      } else {
        if (grepl(rare_seq, trace_str_val, fixed = TRUE) || grepl(trace_str_val, rare_seq, fixed = TRUE)) {
          count <- count + 1
        }
      }
    }
    return(count)
  }
  
  features_df[[paste0(sequence_type, "_rare_sequence_match_count")]] <- 
    sapply(trace_str, count_rare_matches)
  
  # Feature 8: Maximum top sequence frequency
  max_top_frequency <- function(trace_str_val) {
    if (length(top_freq_dict) == 0) return(0)
    max_freq <- 0
    for (top_seq in names(top_freq_dict)) {
      freq <- top_freq_dict[[top_seq]]
      if (match_method == "exact") {
        if (trace_str_val == top_seq) max_freq <- max(max_freq, freq)
      } else {
        if (grepl(top_seq, trace_str_val, fixed = TRUE) || grepl(trace_str_val, top_seq, fixed = TRUE)) {
          max_freq <- max(max_freq, freq)
        }
      }
    }
    return(max_freq)
  }
  
  features_df[[paste0(sequence_type, "_max_top_sequence_frequency")]] <- 
    sapply(trace_str, max_top_frequency)
  
  # Feature 9: Maximum rare sequence frequency
  max_rare_frequency <- function(trace_str_val) {
    if (length(rare_freq_dict) == 0) return(0)
    max_freq <- 0
    for (rare_seq in names(rare_freq_dict)) {
      freq <- rare_freq_dict[[rare_seq]]
      if (match_method == "exact") {
        if (trace_str_val == rare_seq) max_freq <- max(max_freq, freq)
      } else {
        if (grepl(rare_seq, trace_str_val, fixed = TRUE) || grepl(trace_str_val, rare_seq, fixed = TRUE)) {
          max_freq <- max(max_freq, freq)
        }
      }
    }
    return(max_freq)
  }
  
  features_df[[paste0(sequence_type, "_max_rare_sequence_frequency")]] <- 
    sapply(trace_str, max_rare_frequency)
  
  cat("Created", ncol(features_df) - 1, "sequence features for", nrow(features_df), "patients\n")
  
  return(features_df)
}

extract_pre_f1120_traces <- function(model_data_path, cohort_name) {
  # Extract patient traces up to first F1120 event (for targets) or all events (for controls)
  con <- dbConnect(duckdb())
  
  query <- paste0("
    WITH patient_events AS (
        SELECT 
            mi_person_key as case_id,
            event_date,
            primary_icd_diagnosis_code,
            target,
            CASE 
                WHEN drug_name IS NOT NULL THEN 'DRUG:' || drug_name
                WHEN primary_icd_diagnosis_code IS NOT NULL THEN 'ICD:' || primary_icd_diagnosis_code
                WHEN procedure_code IS NOT NULL THEN 'CPT:' || procedure_code
                ELSE NULL
            END as activity,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_seq
        FROM read_parquet('", model_data_path, "')
        WHERE (drug_name IS NOT NULL 
               OR primary_icd_diagnosis_code IS NOT NULL 
               OR procedure_code IS NOT NULL)
    ),
    f1120_first AS (
        SELECT 
            case_id,
            MIN(event_seq) as f1120_seq
        FROM patient_events
        WHERE primary_icd_diagnosis_code LIKE '%F1120%'
          AND target = 1
        GROUP BY case_id
    ),
    pre_f1120_events AS (
        SELECT pe.case_id, pe.activity, pe.event_date
        FROM patient_events pe
        LEFT JOIN f1120_first f ON pe.case_id = f.case_id
        WHERE (
            (pe.target = 1 AND (f.f1120_seq IS NULL OR pe.event_seq < f.f1120_seq))
            OR
            (pe.target = 0)
        )
          AND pe.activity IS NOT NULL
    )
    SELECT 
        case_id,
        LIST(activity ORDER BY event_date) as trace_list
    FROM pre_f1120_events
    GROUP BY case_id
  ")
  
  traces_df <- dbGetQuery(con, query)
  dbDisconnect(con)
  
  if (nrow(traces_df) > 0) {
    traces_df$trace <- sapply(traces_df$trace_list, function(x) {
      if (is.list(x)) {
        paste(x, collapse = ",")
      } else {
        ""
      }
    })
    traces_df <- traces_df[, c("case_id", "trace")]
    cat("Extracted pre-F1120 traces for", nrow(traces_df), "patients (target + control)\n")
  }
  
  return(traces_df)
}

extract_post_f1120_traces <- function(model_data_path, cohort_name) {
  # Extract patient traces after first F1120 event
  con <- dbConnect(duckdb())
  
  query <- paste0("
    WITH patient_events AS (
        SELECT 
            mi_person_key as case_id,
            event_date,
            primary_icd_diagnosis_code,
            CASE 
                WHEN drug_name IS NOT NULL THEN 'DRUG:' || drug_name
                WHEN primary_icd_diagnosis_code IS NOT NULL THEN 'ICD:' || primary_icd_diagnosis_code
                WHEN procedure_code IS NOT NULL THEN 'CPT:' || procedure_code
                ELSE NULL
            END as activity,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_seq
        FROM read_parquet('", model_data_path, "')
        WHERE target = 1
            AND (drug_name IS NOT NULL 
                 OR primary_icd_diagnosis_code IS NOT NULL 
                 OR procedure_code IS NOT NULL)
    ),
    f1120_first AS (
        SELECT 
            case_id,
            MIN(event_seq) as f1120_seq
        FROM patient_events
        WHERE primary_icd_diagnosis_code LIKE '%F1120%'
        GROUP BY case_id
    ),
    post_f1120_events AS (
        SELECT pe.case_id, pe.activity, pe.event_date
        FROM patient_events pe
        INNER JOIN f1120_first f ON pe.case_id = f.case_id
        WHERE pe.event_seq > f.f1120_seq
            AND pe.activity IS NOT NULL
    )
    SELECT 
        case_id,
        LIST(activity ORDER BY event_date) as trace_list
    FROM post_f1120_events
    GROUP BY case_id
  ")
  
  traces_df <- dbGetQuery(con, query)
  dbDisconnect(con)
  
  if (nrow(traces_df) > 0) {
    traces_df$trace <- sapply(traces_df$trace_list, function(x) {
      if (is.list(x)) {
        paste(x, collapse = ",")
      } else {
        ""
      }
    })
    traces_df <- traces_df[, c("case_id", "trace")]
    cat("Extracted post-F1120 traces for", nrow(traces_df), "patients\n")
  }
  
  return(traces_df)
}

create_all_sequence_features <- function(project_root, cohort_name, age_band, train_label = "train") {
  # Create all sequence features (overall, pre-F1120, post-F1120)
  age_band_fname <- gsub("-", "_", age_band)
  bupar_output_dir <- file.path(project_root, "5_bupaR_analysis", "outputs", cohort_name, age_band_fname, "features")
  
  # Load all sequence files
  cat("Loading sequence files...\n")
  
  # Overall sequences
  overall_traces_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_traces_bupar.csv"))
  overall_top_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_traces_top_bupar.csv"))
  overall_rare_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_traces_rare_bupar.csv"))
  
  # Pre-F1120 sequences
  pre_traces_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_pre_f1120_traces_bupar.csv"))
  pre_top_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_pre_f1120_traces_top_bupar.csv"))
  pre_rare_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_pre_f1120_traces_rare_bupar.csv"))
  
  # Post-F1120 sequences
  post_traces_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_post_f1120_traces_bupar.csv"))
  post_top_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_post_f1120_traces_top_bupar.csv"))
  post_rare_path <- file.path(bupar_output_dir, paste0(cohort_name, "_", age_band_fname, "_", train_label, "_target_post_f1120_traces_rare_bupar.csv"))
  
  # Load traces
  overall_traces <- load_traces(overall_traces_path)
  pre_traces <- load_traces(pre_traces_path)
  post_traces <- load_traces(post_traces_path)
  
  # Load top sequences
  overall_top <- load_traces(overall_top_path)
  pre_top <- load_traces(pre_top_path)
  post_top <- load_traces(post_top_path)
  
  # Load rare sequences
  overall_rare <- load_traces(overall_rare_path)
  pre_rare <- load_traces(pre_rare_path)
  post_rare <- load_traces(post_rare_path)
  
  # Get patient traces from model_data
  model_data_path <- file.path(project_root, "model_data", 
                                paste0("cohort_name=", cohort_name),
                                paste0("age_band=", age_band),
                                "model_events.parquet")
  
  if (!file.exists(model_data_path)) {
    stop(paste("Model data not found:", model_data_path))
  }
  
  # Extract patient traces from model_data (BOTH target and control)
  cat("Extracting patient traces from model_data (target + control)...\n")
  overall_patient_traces <- extract_patient_traces_from_model_data(
    model_data_path = model_data_path,
    cohort_name = cohort_name,
    target_filter = NULL
  )
  
  if (nrow(overall_patient_traces) == 0) {
    stop("Could not extract patient traces from model_data")
  }
  
  cat("Extracted traces for", nrow(overall_patient_traces), "patients (target + control)\n")
  
  # Create features for each sequence type
  all_features <- list(overall_patient_traces[, "case_id", drop = FALSE])
  
  # Overall sequence features
  if (nrow(overall_traces) > 0 && nrow(overall_top) > 0 && nrow(overall_rare) > 0) {
    cat("Creating overall sequence features...\n")
    overall_features <- create_sequence_features(
      patient_traces = overall_patient_traces,
      top_sequences = overall_top,
      rare_sequences = overall_rare,
      sequence_type = "overall",
      match_method = "exact"
    )
    overall_features <- overall_features[, !names(overall_features) %in% "case_id", drop = FALSE]
    all_features <- c(all_features, list(overall_features))
  }
  
  # Pre-F1120 sequence features
  if (nrow(pre_traces) > 0 && nrow(pre_top) > 0 && nrow(pre_rare) > 0) {
    cat("Extracting pre-F1120 patient traces...\n")
    pre_patient_traces <- extract_pre_f1120_traces(model_data_path, cohort_name)
    
    if (nrow(pre_patient_traces) > 0) {
      cat("Creating pre-F1120 sequence features...\n")
      pre_features <- create_sequence_features(
        patient_traces = pre_patient_traces,
        top_sequences = pre_top,
        rare_sequences = pre_rare,
        sequence_type = "pre_f1120",
        match_method = "exact"
      )
      pre_features <- pre_features[, !names(pre_features) %in% "case_id", drop = FALSE]
      all_features <- c(all_features, list(pre_features))
    }
  }
  
  # Post-F1120 sequence features
  if (nrow(post_traces) > 0 && nrow(post_top) > 0 && nrow(post_rare) > 0) {
    cat("Extracting post-F1120 patient traces...\n")
    post_patient_traces <- extract_post_f1120_traces(model_data_path, cohort_name)
    
    if (nrow(post_patient_traces) > 0) {
      cat("Creating post-F1120 sequence features...\n")
      post_features <- create_sequence_features(
        patient_traces = post_patient_traces,
        top_sequences = post_top,
        rare_sequences = post_rare,
        sequence_type = "post_f1120",
        match_method = "exact"
      )
      post_features <- post_features[, !names(post_features) %in% "case_id", drop = FALSE]
      all_features <- c(all_features, list(post_features))
    }
  }
  
  # Combine all features by case_id
  combined_features <- all_features[[1]]
  for (i in 2:length(all_features)) {
    feature_df <- all_features[[i]]
    if ("case_id" %in% names(feature_df)) {
      combined_features <- left_join(combined_features, feature_df, by = "case_id")
    } else if (nrow(combined_features) == nrow(feature_df)) {
      # Same number of rows, bind columns
      combined_features <- bind_cols(combined_features, feature_df)
    } else {
      warning(paste("Skipping feature_df", i, "- row count mismatch"))
    }
  }
  
  # Fill NaN values with 0 for numeric columns, 'other' for category columns
  for (col in names(combined_features)) {
    if (col != "case_id") {
      if (grepl("category", col)) {
        combined_features[[col]] <- ifelse(is.na(combined_features[[col]]), "other", combined_features[[col]])
      } else if (grepl("frequency|count|is_", col)) {
        combined_features[[col]] <- ifelse(is.na(combined_features[[col]]), 0, combined_features[[col]])
      } else {
        combined_features[[col]] <- ifelse(is.na(combined_features[[col]]), 0, combined_features[[col]])
      }
    }
  }
  
  cat("Created", ncol(combined_features) - 1, "sequence features for", nrow(combined_features), "patients\n")
  
  return(combined_features)
}

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

# Create sequence features
sequence_features <- create_all_sequence_features(
  project_root = project_root,
  cohort_name = cohort_name,
  age_band = age_band,
  train_label = train_label
)

# Rename case_id to mi_person_key for joining with model_data
if ("case_id" %in% names(sequence_features)) {
  sequence_features <- rename(sequence_features, mi_person_key = case_id)
}

# Set output path
feature_eng_dir <- file.path(project_root, "5_bupaR_analysis", "outputs", "feature_engineering")
dir.create(feature_eng_dir, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(feature_eng_dir, paste0("sequence_features_", cohort_name, "_", age_band_fname, ".csv"))

# Save features
write_csv(sequence_features, output_path)

cat("\nCreated", ncol(sequence_features) - 1, "sequence features for", nrow(sequence_features), "patients\n")
cat("Output format: Ready for joining with model_data (uses mi_person_key)\n")
cat("Saved to:", output_path, "\n")

# Upload to S3 gold location (intermediate file)
s3_path <- paste0("s3://pgxdatalake/gold/feature_engineering/5_bupar/", cohort_name, "/", age_band, 
                  "/sequence_features_", cohort_name, "_", age_band_fname, ".csv")

# Check for AWS CLI
aws_cli <- Sys.which("aws")
if (aws_cli != "") {
  tryCatch({
    cat("\nUploading to S3:", s3_path, "\n")
    system2(aws_cli, c("s3", "cp", output_path, s3_path), stdout = TRUE, stderr = TRUE)
    cat("S3 upload successful:", s3_path, "\n")
  }, error = function(e) {
    warning(paste("S3 upload failed:", e$message))
    cat("Warning: Could not upload to S3:", e$message, "\n")
  })
} else {
  cat("Note: AWS CLI not found, skipping S3 upload\n")
}

cat("\nFeature columns:\n")
for (col in names(sequence_features)) {
  if (col != "mi_person_key") {
    cat("  -", col, "\n")
  }
}

