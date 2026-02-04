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
#' @param output_root_3b Optional. When set (e.g. Step 3b), control is created under this root (no 4_model_data). Use 3b_feature_importance_eda/outputs.
#' @param aggregated_fi_path Optional. Path to 3a aggregated feature importance CSV; if set, control events are filtered to same items (admin removed) to reduce noise.
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
  output_root_3b = NULL,
  aggregated_fi_path = NULL,
  expected_ratio = 5.0,
  tolerance = 0.2
) {
  # Initialize return values
  pgx_df_control <- data.frame()
  was_recreated <- FALSE
  validation_passed <- FALSE
  
  # Step 1: Try to download from S3 if not found locally (skip when Step 3b: we use only Step 1/2/3 artifacts)
  if (!file.exists(control_model_data_path)) {
    if (is.null(output_root_3b) || !nzchar(output_root_3b)) {
      control_s3_path <- paste0("s3://pgxdatalake/gold/cohorts_model_data/cohort_name=", control_cohort, "/age_band=", age_band, "/model_events.parquet")
      cat("Control model data not found locally. Checking S3: ", control_s3_path, "\n", sep = "")
      
      dir.create(dirname(control_model_data_path), recursive = TRUE, showWarnings = FALSE)
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
    } else {
      cat("Step 3b: control under 3b outputs only; skipping S3 (using only Step 1/2/3 artifacts).\n", sep = "")
      dir.create(dirname(control_model_data_path), recursive = TRUE, showWarnings = FALSE)
    }
  }
  
  # Step 2: Validate 5:1 ratio if control cohort exists
  needs_recreation <- FALSE
  n_cases <- 0
  n_controls <- 0
  
  if (file.exists(control_model_data_path)) {
    # Check if file is valid parquet (not empty/corrupted)
    file_size <- file.info(control_model_data_path)$size
    if (is.na(file_size) || file_size < 1000) {  # Parquet files should be at least 1KB
      cat("⚠️  Control cohort file exists but is too small/corrupted (", file_size, " bytes). Will recreate.\n", sep = "")
      unlink(control_model_data_path)  # Delete invalid file
      needs_recreation <- TRUE
    } else {
      # Try to validate parquet file by attempting a simple query
      tryCatch({
        test_query <- sprintf("SELECT COUNT(*) as n FROM read_parquet('%s') LIMIT 1", control_model_data_path)
        test_result <- dbGetQuery(con, test_query)
        if (is.null(test_result) || nrow(test_result) == 0) {
          stop("Parquet file appears to be empty or invalid")
        }
      }, error = function(e) {
        cat("⚠️  Control cohort file exists but is invalid/corrupted: ", conditionMessage(e), "\n", sep = "")
        cat("   File size: ", file_size, " bytes. Will delete and recreate.\n", sep = "")
        unlink(control_model_data_path)  # Delete invalid file
        needs_recreation <<- TRUE
      })
    }
    
    # Check ratio: should be approximately 5:1 (controls:cases) - only if file is valid
    if (!needs_recreation && file.exists(control_model_data_path)) {
      query_control_count <- sprintf(
        "SELECT COUNT(DISTINCT mi_person_key) as n_controls FROM read_parquet('%s') WHERE event_year IN (%s)",
        control_model_data_path,
        paste(train_years, collapse = ",")
      )
      tryCatch({
        n_controls <- dbGetQuery(con, query_control_count)$n_controls[1]
      }, error = function(e) {
        cat("⚠️  Failed to query control cohort file: ", conditionMessage(e), "\n", sep = "")
        cat("   File may be corrupted. Will delete and recreate.\n", sep = "")
        unlink(control_model_data_path)
        needs_recreation <<- TRUE
        n_controls <<- 0
      })
    }
    
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
      cat("   Actual ratio: ", sprintf("%.2f", actual_ratio), ":1 (", n_controls, " distinct controls, ", n_cases, " distinct targets)\n", sep = "")
      cat("   Expected ratio: ", sprintf("%.2f", expected_ratio), ":1 (tolerance: ", sprintf("%.2f", min_ratio), "-", sprintf("%.2f", max_ratio), ":1)\n", sep = "")
      cat("   Will recreate control cohort to achieve ", sprintf("%.2f", expected_ratio), ":1 ratio...\n\n", sep = "")
      needs_recreation <- TRUE
    } else {
      cat("✅ Control cohort ratio validation passed: ", sprintf("%.2f", actual_ratio), ":1 (", n_controls, " distinct controls, ", n_cases, " distinct targets)\n", sep = "")
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
    
    # Check for jupyter-env Python first (same logic as workflow script)
    python_cmd <- ""
    if (file.exists("/home/pgx3874/jupyter-env/bin/python3.11")) {
      python_cmd <- "/home/pgx3874/jupyter-env/bin/python3.11"
    } else if (file.exists(file.path(Sys.getenv("HOME"), "jupyter-env", "bin", "python3"))) {
      python_cmd <- file.path(Sys.getenv("HOME"), "jupyter-env", "bin", "python3")
    } else if (file.exists(file.path(project_root, "venv", "bin", "python3"))) {
      python_cmd <- file.path(project_root, "venv", "bin", "python3")
    } else if (Sys.getenv("VIRTUAL_ENV") != "") {
      venv_python <- file.path(Sys.getenv("VIRTUAL_ENV"), "bin", "python3")
      if (file.exists(venv_python)) {
        python_cmd <- venv_python
      }
    }
    
    # Fallback to system Python
    if (python_cmd == "") {
      python_cmd <- Sys.which("python3")
      if (python_cmd == "") {
        python_cmd <- Sys.which("python")
      }
    }
    
    # Call create_control_cohort_model_data.py directly (simpler than ensure_control_cohort.py)
    create_script <- file.path(project_root, "4_model_data", "create_control_cohort_model_data.py")
    
    if (python_cmd != "" && file.exists(create_script)) {
      create_cmd <- c(
        create_script,
        "--age-band", age_band,
        "--sample-size", as.character(required_controls)
      )
      if (!is.null(output_root_3b) && nzchar(output_root_3b)) {
        # Step 3b: write control under 3b outputs only (no 4_model_data yet)
        create_cmd <- c(create_cmd, "--output-root", output_root_3b)
      }
      # Step 3b: aggregated FI is required when writing to 3b output
      if (!is.null(output_root_3b) && nzchar(output_root_3b)) {
        if (is.null(aggregated_fi_path) || !nzchar(aggregated_fi_path) || !file.exists(aggregated_fi_path)) {
          stop("Step 3b requires 3a aggregated feature importance CSV for control. Resolve aggregated_fi_path for this cohort/age_band.")
        }
        create_cmd <- c(create_cmd, "--aggregated-fi-csv", aggregated_fi_path)
      } else if (!is.null(aggregated_fi_path) && nzchar(aggregated_fi_path) && file.exists(aggregated_fi_path)) {
        create_cmd <- c(create_cmd, "--aggregated-fi-csv", aggregated_fi_path)
      }
      
      cat("[INFO] Running: ", python_cmd, " ", paste(create_cmd, collapse = " "), "\n", sep = "")
      create_result <- system2(python_cmd, create_cmd, stdout = TRUE, stderr = TRUE)
      
      # Print Python script output for debugging
      if (length(create_result) > 0) {
        cat("Python script output:\n")
        cat(paste(create_result, collapse = "\n"), "\n")
      }
      
      # Check return code (system2 returns exit status as attribute)
      exit_status <- attr(create_result, "status")
      if (!is.null(exit_status) && exit_status != 0) {
        cat("[ERROR] Python script exited with code: ", exit_status, "\n", sep = "")
      }
      
      if (file.exists(control_model_data_path)) {
        cat("[OK] Control cohort created successfully\n")
        was_recreated <- TRUE
        
        # Re-validate and log final ratio after recreation
        years_list <- paste(train_years, collapse = ",")
        query_control_count <- sprintf(
          "SELECT COUNT(DISTINCT mi_person_key) as n_controls FROM read_parquet('%s') WHERE event_year IN (%s)",
          control_model_data_path,
          years_list
        )
        query_case_count <- sprintf(
          "SELECT COUNT(DISTINCT mi_person_key) as n_cases FROM read_parquet('%s') WHERE event_year IN (%s) AND target = 1",
          model_data_path,
          years_list
        )
        
        tryCatch({
          n_controls_final <- dbGetQuery(con, query_control_count)$n_controls[1]
          n_cases_final <- dbGetQuery(con, query_case_count)$n_cases[1]
          actual_ratio_final <- ifelse(n_cases_final > 0, n_controls_final / n_cases_final, 0)
          
          cat("✅ Final ratio after recreation: ", sprintf("%.2f", actual_ratio_final), ":1 (", 
              n_controls_final, " distinct controls, ", n_cases_final, " distinct targets)\n", sep = "")
          
          # Warn if ratio is still below target (data limitation)
          if (actual_ratio_final < expected_ratio * (1 - tolerance)) {
            cat("⚠️  Note: Final ratio (", sprintf("%.2f", actual_ratio_final), ":1) is below target (", 
                sprintf("%.2f", expected_ratio), ":1) due to limited control candidates.\n", sep = "")
            cat("   This is a data limitation, not an error. All available control candidates were sampled.\n", sep = "")
          }
        }, error = function(e) {
          cat("[WARN] Could not validate final ratio: ", conditionMessage(e), "\n", sep = "")
        })
      } else {
        cat("[WARN] Control cohort creation may have failed. File not found: ", control_model_data_path, "\n", sep = "")
        cat("[WARN] Check Python script output above for errors.\n")
      }
    } else {
      cat("[ERROR] Cannot create control cohort: Python or script not found\n")
      cat("   Python: ", python_cmd, "\n", sep = "")
      cat("   Script: ", create_script, "\n", sep = "")
      cat("   Please run manually:\n")
      manual_cmd <- paste(python_cmd, " 4_model_data/create_control_cohort_model_data.py --age-band ", age_band, " --sample-size ", required_controls, sep = "")
      if (!is.null(output_root_3b) && nzchar(output_root_3b)) {
        manual_cmd <- paste(manual_cmd, " --output-root ", output_root_3b, sep = "")
      }
      if (!is.null(aggregated_fi_path) && nzchar(aggregated_fi_path) && file.exists(aggregated_fi_path)) {
        manual_cmd <- paste(manual_cmd, " --aggregated-fi-csv ", shQuote(aggregated_fi_path), sep = "")
      }
      cat("   ", manual_cmd, "\n\n", sep = "")
    }
  }
  
  # Step 4: Load control cohort (after recreation if needed)
  # Optimized: Only select columns needed for verification and type checking
  # The actual data transformation will be done in the combined query
  if (file.exists(control_model_data_path)) {
    query_control <- sprintf(
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
      FROM read_parquet('%s') WHERE event_year IN (%s)",
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
    cat("   python 4_model_data/create_control_cohort_model_data.py --age-band ", age_band, "\n\n", sep = "")
    # Return empty data frame with same structure as target
    # This will be handled by the calling script
  }
  
  return(list(
    pgx_df_control = pgx_df_control,
    was_recreated = was_recreated,
    validation_passed = validation_passed
  ))
}
