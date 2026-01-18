#!/usr/bin/env Rscript
#
# Utility functions for control cohort validation and management
# Used by both opioid_ed and non_opioid_ed BupaR analysis scripts
#

#' Validate and ensure control cohort exists with correct 5:1 ratio
#'
#' @param con DuckDB connection
#' @param control_cohort Name of control cohort (e.g., "non_opioid_non_ed")
#' @param control_model_data_path Path to control cohort model_events.parquet
#' @param model_data_path Path to target cohort model_events.parquet
#' @param age_band Age band (e.g., "13-24")
#' @param train_years Vector of training years (e.g., c(2016L, 2017L, 2018L))
#' @param project_root Project root directory
#' @param expected_ratio Expected control:case ratio (default: 5.0)
#' @param tolerance Tolerance for ratio validation (default: 0.2, i.e., 20%)
#'
#' @return List with:
#'   - pgx_df_control: Data frame with control events (empty if not found/created)
#'   - was_recreated: Logical indicating if control cohort was recreated
#'   - validation_passed: Logical indicating if ratio validation passed
#'
ensure_control_cohort_with_ratio <- function(
  con,
  control_cohort,
  control_model_data_path,
  model_data_path,
  age_band,
  train_years,
  project_root,
  expected_ratio = 5.0,
  tolerance = 0.2
) {
  # Initialize return values
  pgx_df_control <- data.frame()
  was_recreated <- FALSE
  validation_passed <- FALSE
  
  # Step 1: Try to download from S3 if not found locally
  if (!file.exists(control_model_data_path)) {
    control_s3_path <- paste0("s3://pgxdatalake/gold/cohorts_model_data/cohort_name=", control_cohort, "/age_band=", age_band, "/model_events.parquet")
    cat("Control model data not found locally. Checking S3: ", control_s3_path, "\n", sep = "")
    
    # Create directory if it doesn't exist
    dir.create(dirname(control_model_data_path), recursive = TRUE, showWarnings = FALSE)
    
    # Try AWS CLI sync
    aws_cli <- Sys.which("aws")
    if (aws_cli != "") {
      cat("Downloading control cohort from S3 using AWS CLI...\n")
      sync_cmd <- c("s3", "cp", control_s3_path, control_model_data_path)
      sync_result <- system2(aws_cli, sync_cmd, stdout = TRUE, stderr = TRUE)
      
      if (file.exists(control_model_data_path)) {
        cat("Successfully downloaded control cohort from S3: ", control_model_data_path, "\n", sep = "")
      } else {
        cat("Failed to download control cohort from S3. Error output:\n")
        cat(paste(sync_result, collapse = "\n"), "\n")
      }
    } else {
      cat("AWS CLI not found. Cannot download control cohort from S3.\n")
    }
  }
  
  # Step 2: Validate 5:1 ratio if control cohort exists
  needs_recreation <- FALSE
  n_cases <- 0
  n_controls <- 0
  
  if (file.exists(control_model_data_path)) {
    # Check ratio: should be approximately 5:1 (controls:cases)
    query_control_count <- sprintf(
      "SELECT COUNT(DISTINCT mi_person_key) as n_controls FROM read_parquet('%s') WHERE event_year IN (%s)",
      control_model_data_path,
      paste(train_years, collapse = ",")
    )
    n_controls <- dbGetQuery(con, query_control_count)$n_controls[1]
    
    # Get number of cases from target cohort
    query_case_count <- sprintf(
      "SELECT COUNT(DISTINCT mi_person_key) as n_cases FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1",
      model_data_path,
      paste(train_years, collapse = ",")
    )
    n_cases <- dbGetQuery(con, query_case_count)$n_cases[1]
    
    # Calculate actual ratio
    actual_ratio <- ifelse(n_cases > 0, n_controls / n_cases, 0)
    min_ratio <- expected_ratio * (1 - tolerance)
    max_ratio <- expected_ratio * (1 + tolerance)
    
    if (actual_ratio < min_ratio || actual_ratio > max_ratio) {
      cat("\n⚠️  Control cohort ratio validation failed:\n", sep = "")
      cat("   Actual ratio: ", sprintf("%.2f", actual_ratio), ":1 (", n_controls, " controls, ", n_cases, " cases)\n", sep = "")
      cat("   Expected ratio: ", sprintf("%.2f", expected_ratio), ":1 (tolerance: ", sprintf("%.2f", min_ratio), "-", sprintf("%.2f", max_ratio), ":1)\n", sep = "")
      cat("   Will recreate control cohort to achieve ", sprintf("%.2f", expected_ratio), ":1 ratio...\n\n", sep = "")
      needs_recreation <- TRUE
    } else {
      cat("✅ Control cohort ratio validation passed: ", sprintf("%.2f", actual_ratio), ":1 (", n_controls, " controls, ", n_cases, " cases)\n", sep = "")
      validation_passed <- TRUE
    }
  }
  
  # Step 3: Recreate control cohort if needed or if missing
  if (needs_recreation || !file.exists(control_model_data_path)) {
    if (needs_recreation) {
      # Remove existing file
      if (file.exists(control_model_data_path)) {
        file.remove(control_model_data_path)
        cat("[INFO] Removed existing control cohort file for recreation\n")
      }
    }
    
    # Calculate required sample size for target ratio
    if (n_cases == 0) {
      query_case_count <- sprintf(
        "SELECT COUNT(DISTINCT mi_person_key) as n_cases FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1",
        model_data_path,
        paste(train_years, collapse = ",")
      )
      n_cases <- dbGetQuery(con, query_case_count)$n_cases[1]
    }
    
    required_controls <- max(ceiling(n_cases * expected_ratio), 1000)  # At least 1000 controls, or expected_ratio x cases
    
    cat("[INFO] Creating control cohort with ", required_controls, " controls (target: ", sprintf("%.2f", expected_ratio), ":1 ratio with ", n_cases, " cases)\n", sep = "")
    
    # Call Python script to create control cohort
    python_script <- file.path(project_root, "4a_model_data", "create_control_cohort_model_data.py")
    python_cmd <- Sys.which("python3")
    if (python_cmd == "") {
      python_cmd <- Sys.which("python")
    }
    
    if (python_cmd != "" && file.exists(python_script)) {
      recreate_cmd <- c(
        python_script,
        "--age-band", age_band,
        "--sample-size", as.character(required_controls)
      )
      
      cat("[INFO] Running: ", python_cmd, " ", paste(recreate_cmd, collapse = " "), "\n", sep = "")
      recreate_result <- system2(python_cmd, recreate_cmd, stdout = TRUE, stderr = TRUE)
      
      if (file.exists(control_model_data_path)) {
        cat("[OK] Control cohort recreated successfully\n")
        was_recreated <- TRUE
      } else {
        cat("[WARN] Control cohort recreation may have failed. Check output above.\n")
      }
    } else {
      cat("[ERROR] Cannot recreate control cohort: Python or script not found\n")
      cat("   Python: ", python_cmd, "\n", sep = "")
      cat("   Script: ", python_script, "\n", sep = "")
      cat("   Please run manually:\n")
      cat("   python 4a_model_data/create_control_cohort_model_data.py --age-band ", age_band, " --sample-size ", required_controls, "\n\n", sep = "")
    }
  }
  
  # Step 4: Load control cohort (after recreation if needed)
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
    cat("\n⚠️  Control cohort '", control_cohort, "' model_events.parquet not found.\n", sep = "")
    cat("   To create it, run:\n")
    cat("   python 4a_model_data/create_control_cohort_model_data.py --age-band ", age_band, "\n\n", sep = "")
    # Return empty data frame with same structure as target
    # This will be handled by the calling script
  }
  
  return(list(
    pgx_df_control = pgx_df_control,
    was_recreated = was_recreated,
    validation_passed = validation_passed
  ))
}
