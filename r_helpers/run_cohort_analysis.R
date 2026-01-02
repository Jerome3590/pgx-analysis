# ============================================================
# COHORT ANALYSIS WRAPPER
# ============================================================
# Wrapper function to run complete feature importance analysis for a single cohort
# This function wraps the entire pipeline: data loading, feature engineering, MC-CV, aggregation

# Run complete analysis for a single cohort
# Args:
#   cohort_name: Cohort name (e.g., "opioid_ed" or "non_opioid_ed")
#   age_band: Age band (e.g., "25-44")
#   event_year: Event year (e.g., 2016)
#   n_splits: Number of MC-CV splits (default: from global N_SPLITS if available)
#   train_prop: Training proportion (default: from global TRAIN_PROP if available)
#   n_workers: Number of workers for MC-CV (default: from global N_WORKERS if available)
#   scaling_metric: Scaling metric for feature importance (default: from global SCALING_METRIC if available)
#   model_params: Model parameters (default: from global MODEL_PARAMS if available)
#   debug_mode: Debug mode flag (default: from global DEBUG_MODE if available)
# Returns:
#   List with cohort name, status, aggregated results, and output file path
run_cohort_analysis <- function(cohort_name, age_band, event_year,
                                 n_splits = NULL, train_prop = NULL, n_workers = NULL,
                                 scaling_metric = NULL, model_params = NULL, debug_mode = NULL) {
  # Initialize logging early to capture parameter issues
  log_setup <- setup_r_logging(cohort_name, age_band, event_year)
  logger <- log_setup$logger
  log_file_path <- log_setup$log_file_path
  
  # Get parameters from arguments or globals (for backward compatibility)
  # Log parameter values for debugging
  logger$info("Received parameters: n_splits=%s, train_prop=%s, n_workers=%s",
               if (is.null(n_splits)) "NULL" else as.character(n_splits),
               if (is.null(train_prop)) "NULL" else as.character(train_prop),
               if (is.null(n_workers)) "NULL" else as.character(n_workers))
  
  if (is.null(n_splits) || is.na(n_splits) || length(n_splits) == 0) {
    logger$info("n_splits is NULL/NA, checking global N_SPLITS...")
    if (!exists("N_SPLITS") || is.null(N_SPLITS) || is.na(N_SPLITS) || length(N_SPLITS) == 0) {
      error_msg <- sprintf("N_SPLITS must be provided as argument or available in global environment. Checked: exists=%s, value=%s",
                          exists("N_SPLITS"),
                          if (exists("N_SPLITS")) as.character(N_SPLITS) else "N/A")
      logger$error(error_msg)
      stop(error_msg)
    }
    n_splits <- N_SPLITS
    logger$info("Using global N_SPLITS: %s", as.character(n_splits))
  }
  
  # Validate n_splits is a valid positive integer
  if (!is.numeric(n_splits) || length(n_splits) != 1 || n_splits <= 0 || is.na(n_splits)) {
    error_msg <- sprintf("n_splits must be a positive integer. Got: %s (type: %s, length: %d)",
                        paste(n_splits, collapse=", "), typeof(n_splits), length(n_splits))
    logger$error(error_msg)
    stop(error_msg)
  }
  
  logger$info("Validated n_splits: %d", as.integer(n_splits))
  if (is.null(train_prop)) {
    if (!exists("TRAIN_PROP") || is.null(TRAIN_PROP)) {
      stop("TRAIN_PROP must be provided as argument or available in global environment")
    }
    train_prop <- TRAIN_PROP
  }
  if (is.null(n_workers)) {
    if (!exists("N_WORKERS") || is.null(N_WORKERS)) {
      n_workers <- max(1, parallel::detectCores() - 2)
    } else {
      n_workers <- N_WORKERS
    }
  }
  
  # Validate and cap worker count based on available cores
  # Check available cores (respects mc.cores and other limits)
  available_cores <- tryCatch({
    parallelly::availableCores()
  }, error = function(e) {
    # Fallback to detectCores if parallelly not available
    parallel::detectCores()
  })
  
  # Cap workers to available cores (with safety margin)
  # Use max 80% of available cores to avoid overloading
  max_workers <- max(1, floor(available_cores * 0.8))
  if (n_workers > max_workers) {
    warning_msg <- sprintf("Requested %d workers exceeds available cores (%d). Capping to %d workers.",
                          n_workers, available_cores, max_workers)
    # Log warning if logger is available (it will be set up later)
    if (exists("logger")) {
      logger$warning(warning_msg)
    } else {
      cat(sprintf("WARNING: %s\n", warning_msg))
    }
    n_workers <- max_workers
  }
  if (is.null(scaling_metric)) {
    if (!exists("SCALING_METRIC")) {
      scaling_metric <- "recall"
    } else {
      scaling_metric <- SCALING_METRIC
    }
  }
  if (is.null(model_params)) {
    if (!exists("MODEL_PARAMS")) {
      stop("MODEL_PARAMS must be provided as argument or available in global environment")
    }
    model_params <- MODEL_PARAMS
  }
  if (is.null(debug_mode)) {
    if (!exists("DEBUG_MODE")) {
      debug_mode <- FALSE
    } else {
      debug_mode <- DEBUG_MODE
    }
  }
  
  # Store in local variables for use throughout function
  # Validate and convert to proper types
  N_SPLITS <- as.integer(n_splits)
  if (is.na(N_SPLITS) || length(N_SPLITS) != 1 || N_SPLITS <= 0) {
    error_msg <- sprintf("N_SPLITS conversion failed. Input: %s (type: %s), Output: %s (type: %s)", 
                        paste(n_splits, collapse=", "), typeof(n_splits),
                        paste(N_SPLITS, collapse=", "), typeof(N_SPLITS))
    logger$error(error_msg)
    stop(error_msg)
  }
  logger$info("Converted N_SPLITS to integer: %d", N_SPLITS)
  
  TRAIN_PROP <- as.numeric(train_prop)
  if (is.na(TRAIN_PROP) || TRAIN_PROP <= 0 || TRAIN_PROP >= 1) {
    stop(sprintf("TRAIN_PROP must be between 0 and 1. Got: %s", as.character(TRAIN_PROP)))
  }
  
  N_WORKERS <- as.integer(n_workers)
  SCALING_METRIC <- scaling_metric
  MODEL_PARAMS <- model_params
  DEBUG_MODE <- debug_mode
  cat(sprintf("\n%s\n", paste(rep("=", 80), collapse="")))
  cat(sprintf("Starting analysis for cohort: %s\n", cohort_name))
  cat(sprintf("%s\n", paste(rep("=", 80), collapse="")))
  
  # Initialize logging for this cohort
  log_setup <- setup_r_logging(cohort_name, age_band, event_year)
  logger <- log_setup$logger
  log_file_path <- log_setup$log_file_path
  
  # Log header
  logger$info("==================================================================================")
  logger$info("🚀 FEATURE IMPORTANCE ANALYSIS - MONTE CARLO CROSS-VALIDATION")
  logger$info("==================================================================================")
  logger$info("📊 Cohort: %s", cohort_name)
  logger$info("📊 Age Band: %s", age_band)
  logger$info("📊 Event Year: %d", event_year)
  logger$info("📊 MC-CV Splits: %d", N_SPLITS)
  logger$info("📊 Scaling Metric: %s", SCALING_METRIC)
  logger$info("📊 Debug Mode: %s", if (DEBUG_MODE) "Enabled" else "Disabled")
  logger$info("==================================================================================")
  logger$info("🔧 CONFIGURATION:")
  logger$info("   - Monte Carlo Cross-Validation with stratified sampling")
  logger$info("   - CatBoost and Random Forest models")
  logger$info("   - Feature importance scaled by MC-CV performance")
  logger$info("   - Parallel processing with furrr/future (%d workers)", N_WORKERS)
  logger$info("   - Comprehensive logging to console, file, and S3")
  logger$info("==================================================================================")
  logger$info("Log file: %s", log_file_path)
  
  check_memory_usage_r(logger, "After Setup and Configuration")
  
  tryCatch({
    # ============================================================
    # DATA LOADING
    # ============================================================
    logger$info("Loading cohort data...")
    check_memory_usage_r(logger, "Before DuckDB Connection")
    
    # Determine local data path
    local_data_path <- Sys.getenv("LOCAL_DATA_PATH", "/mnt/nvme/cohorts")
    if (!dir.exists(local_data_path)) {
      local_data_path <- Sys.getenv("LOCAL_DATA_PATH", "C:/Projects/pgx-analysis/data/gold/cohorts_F1120")
    }
    
    parquet_file <- file.path(local_data_path, 
                              paste0("cohort_name=", cohort_name),
                              paste0("event_year=", event_year),
                              paste0("age_band=", age_band),
                              "cohort.parquet")
    
    if (!file.exists(parquet_file)) {
      error_msg <- sprintf("Cohort file not found: %s\nPlease check:\n  1. LOCAL_DATA_PATH environment variable\n  2. Cohort data exists for this age-band\n  3. File path structure matches expected format", parquet_file)
      logger$error(error_msg)
      logger$close()
      save_logs_to_s3_r(log_file_path, cohort_name, age_band, event_year, logger)
      # Return error status instead of stopping - allows other tasks to continue
      return(list(
        cohort = cohort_name,
        age_band = age_band,
        event_year = event_year,
        status = "error",
        error = sprintf("Cohort file not found: %s", parquet_file)
      ))
    }
    
    logger$info("Loading from: %s", parquet_file)
    
    # Load using DuckDB
    con <- DBI::dbConnect(duckdb::duckdb(), dbdir = ":memory:")
    check_memory_usage_r(logger, "After DuckDB Connection")
    
    query <- sprintf("
      SELECT 
        mi_person_key,
        is_target_case as target,
        drug_name,
        primary_icd_diagnosis_code,
        two_icd_diagnosis_code,
        three_icd_diagnosis_code,
        four_icd_diagnosis_code,
        five_icd_diagnosis_code,
        procedure_code,
        event_type
      FROM read_parquet('%s')
    ", parquet_file)
    
    check_memory_usage_r(logger, "Before SQL Query Execution")
    cohort_data <- DBI::dbGetQuery(con, query)
    check_memory_usage_r(logger, "After SQL Query Execution")
    DBI::dbDisconnect(con)
    check_memory_usage_r(logger, "After DuckDB Disconnection")
    
    logger$info("Loaded %d event-level records, %d unique patients", 
                nrow(cohort_data), length(unique(cohort_data$mi_person_key)))
    check_memory_usage_r(logger, "After Data Loading")
    
    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================
    logger$info("Creating patient-level features...")
    
    # Extract patient-item pairs
    patient_items <- cohort_data %>%
      dplyr::filter(!is.na(drug_name) & drug_name != "" & event_type == "pharmacy") %>%
      dplyr::select(mi_person_key, item = drug_name) %>%
      dplyr::bind_rows(
        cohort_data %>%
          dplyr::filter(!is.na(primary_icd_diagnosis_code) & primary_icd_diagnosis_code != "" & event_type == "medical") %>%
          dplyr::select(mi_person_key, item = primary_icd_diagnosis_code)
      ) %>%
      dplyr::bind_rows(
        cohort_data %>%
          dplyr::filter(!is.na(two_icd_diagnosis_code) & two_icd_diagnosis_code != "" & event_type == "medical") %>%
          dplyr::select(mi_person_key, item = two_icd_diagnosis_code)
      ) %>%
      dplyr::bind_rows(
        cohort_data %>%
          dplyr::filter(!is.na(three_icd_diagnosis_code) & three_icd_diagnosis_code != "" & event_type == "medical") %>%
          dplyr::select(mi_person_key, item = three_icd_diagnosis_code)
      ) %>%
      dplyr::bind_rows(
        cohort_data %>%
          dplyr::filter(!is.na(four_icd_diagnosis_code) & four_icd_diagnosis_code != "" & event_type == "medical") %>%
          dplyr::select(mi_person_key, item = four_icd_diagnosis_code)
      ) %>%
      dplyr::bind_rows(
        cohort_data %>%
          dplyr::filter(!is.na(five_icd_diagnosis_code) & five_icd_diagnosis_code != "" & event_type == "medical") %>%
          dplyr::select(mi_person_key, item = five_icd_diagnosis_code)
      ) %>%
      dplyr::bind_rows(
        cohort_data %>%
          dplyr::filter(!is.na(procedure_code) & procedure_code != "" & event_type == "medical") %>%
          dplyr::select(mi_person_key, item = procedure_code)
      ) %>%
      dplyr::distinct() %>%
      dplyr::filter(!is.na(item) & item != "")
    
    # Get target per patient
    patient_targets <- cohort_data %>%
      dplyr::select(mi_person_key, target) %>%
      dplyr::distinct() %>%
      dplyr::group_by(mi_person_key) %>%
      dplyr::summarise(target = dplyr::first(target), .groups = 'drop')
    
    # Create feature matrices
    all_unique_items <- sort(unique(patient_items$item))
    n_items <- length(all_unique_items)
    n_patients <- length(unique(patient_items$mi_person_key))
    
    logger$info("Creating feature matrices: %d unique items, %d patients", n_items, n_patients)
    
    # Check for potential integer overflow (R's max integer is ~2.1 billion)
    # pivot_wider creates a matrix of size n_patients x n_items
    max_matrix_size <- as.numeric(n_patients) * as.numeric(n_items)
    if (max_matrix_size > 2e9) {
      logger$warning("Large feature matrix detected (%.0e elements). This may cause memory issues.", max_matrix_size)
    }
    
    # CatBoost format
    # Use pivot_wider with explicit type handling to avoid integer overflow
    feature_matrix_catboost <- patient_items %>%
      tidyr::pivot_wider(
        id_cols = mi_person_key,
        names_from = item,
        values_from = item,
        values_fill = NA_character_,
        names_prefix = "item_"
      ) %>%
      dplyr::left_join(patient_targets, by = "mi_person_key")
    
    data_catboost <- feature_matrix_catboost %>%
      dplyr::select(-mi_person_key) %>%
      dplyr::mutate(dplyr::across(-target, ~ as.factor(.x)))
    
    data <- data_catboost
    
    # Random Forest format
    feature_matrix_rf <- patient_items %>%
      dplyr::mutate(value = 1L) %>%  # Use integer instead of numeric to save memory
      tidyr::pivot_wider(
        id_cols = mi_person_key,
        names_from = item,
        values_from = value,
        values_fill = 0L,  # Use integer instead of numeric
        names_prefix = "item_"
      ) %>%
      dplyr::left_join(patient_targets, by = "mi_person_key")
    
    data_rf <- feature_matrix_rf %>%
      dplyr::select(-mi_person_key)
    
    # Clean data
    data <- data %>% dplyr::filter(!is.na(target))
    data_rf <- data_rf %>% dplyr::filter(!is.na(target))
    
    logger$info("Feature engineering complete: %d patients, %d features", 
                nrow(data), ncol(data) - 1)
    check_memory_usage_r(logger, "After Feature Engineering")
    
    # ============================================================
    # MC-CV SPLIT CREATION
    # ============================================================
    logger$info("Creating MC-CV splits...")
    check_memory_usage_r(logger, "Before MC-CV Split Creation")
    
    # Ensure N_SPLITS and TRAIN_PROP are available (should be in globals)
    # Explicitly check and provide helpful error if missing
    if (!exists("N_SPLITS") || is.null(N_SPLITS) || is.na(N_SPLITS)) {
      error_msg <- sprintf("N_SPLITS is not defined or is NULL/NA. Current value: %s. Ensure it's set in the main notebook and included in globals.",
                           if (exists("N_SPLITS")) as.character(N_SPLITS) else "not found")
      logger$error(error_msg)
      stop(error_msg)
    }
    if (!exists("TRAIN_PROP") || is.null(TRAIN_PROP) || is.na(TRAIN_PROP)) {
      error_msg <- sprintf("TRAIN_PROP is not defined or is NULL/NA. Current value: %s. Ensure it's set in the main notebook and included in globals.",
                           if (exists("TRAIN_PROP")) as.character(TRAIN_PROP) else "not found")
      logger$error(error_msg)
      stop(error_msg)
    }
    
    # Ensure N_SPLITS is numeric and positive
    if (!is.numeric(N_SPLITS) || N_SPLITS <= 0) {
      error_msg <- sprintf("N_SPLITS must be a positive numeric value. Current value: %s (type: %s)",
                           as.character(N_SPLITS), typeof(N_SPLITS))
      logger$error(error_msg)
      stop(error_msg)
    }
    
    logger$info("MC-CV parameters: N_SPLITS=%d, TRAIN_PROP=%.2f", N_SPLITS, TRAIN_PROP)
    
    # Final validation - use the local N_SPLITS variable directly
    # Double-check that N_SPLITS is valid (it should be set earlier in the function)
    if (is.null(N_SPLITS) || is.na(N_SPLITS) || length(N_SPLITS) != 1 || !is.numeric(N_SPLITS) || N_SPLITS <= 0) {
      error_msg <- sprintf("N_SPLITS is invalid before mc_cv call. Value: %s (type: %s, length: %d). This indicates a bug in parameter handling.",
                          paste(N_SPLITS, collapse=", "),
                          typeof(N_SPLITS),
                          length(N_SPLITS))
      logger$error(error_msg)
      logger$error("Debug info: n_splits parameter was: %s", if (exists("n_splits")) as.character(n_splits) else "not found")
      stop(error_msg)
    }
    
    # Convert to integer explicitly - use the local N_SPLITS variable
    n_splits_value <- as.integer(N_SPLITS[1])  # Ensure single value
    train_prop_value <- as.numeric(TRAIN_PROP[1])  # Ensure single value
    
    # Final validation on converted values
    if (is.na(n_splits_value) || length(n_splits_value) != 1 || n_splits_value <= 0) {
      error_msg <- sprintf("n_splits_value conversion failed. N_SPLITS=%s, n_splits_value=%s (type: %s, length: %d)",
                          paste(N_SPLITS, collapse=", "), 
                          paste(n_splits_value, collapse=", "),
                          typeof(n_splits_value),
                          length(n_splits_value))
      logger$error(error_msg)
      stop(error_msg)
    }
    
    if (is.na(train_prop_value) || length(train_prop_value) != 1 || train_prop_value <= 0 || train_prop_value >= 1) {
      error_msg <- sprintf("train_prop_value is invalid: %s (TRAIN_PROP=%s, type: %s, length: %d)",
                          as.character(train_prop_value),
                          paste(TRAIN_PROP, collapse=", "),
                          typeof(train_prop_value),
                          length(train_prop_value))
      logger$error(error_msg)
      stop(error_msg)
    }
    
    logger$info("Calling rsample::mc_cv with times=%d (integer), prop=%.2f (numeric)", n_splits_value, train_prop_value)
    logger$info("Data dimensions: %d rows, %d columns", nrow(data), ncol(data))
    logger$info("Target distribution: %d positives, %d negatives", 
                sum(data$target == 1, na.rm = TRUE), 
                sum(data$target == 0, na.rm = TRUE))
    
    # Call mc_cv with validated values - use explicit integer/numeric values
    # rsample::mc_cv uses non-standard evaluation, so we pass values directly
    tryCatch({
      mc_splits <- rsample::mc_cv(
        data = data,
        prop = train_prop_value,
        times = n_splits_value,
        strata = target
      )
    }, error = function(e) {
      logger$error("rsample::mc_cv failed with error: %s", e$message)
      logger$error("Parameters passed: times=%s (type: %s), prop=%s (type: %s)", 
                   as.character(n_splits_value), typeof(n_splits_value),
                   as.character(train_prop_value), typeof(train_prop_value))
      logger$error("N_SPLITS local variable: %s (type: %s)", 
                   paste(N_SPLITS, collapse=", "), typeof(N_SPLITS))
      stop(e)
    })
    
    # Extract indices
    logger$info("Extracting indices from %d MC-CV splits...", N_SPLITS)
    split_indices <- lapply(1:N_SPLITS, function(i) {
      split <- mc_splits$splits[[i]]
      if (is.null(split)) {
        stop(sprintf("mc_splits$splits[[%d]] is NULL", i))
      }
      train_idx <- split$in_id
      test_idx <- setdiff(seq_len(nrow(data)), train_idx)
      
      # Validate indices
      if (is.null(train_idx) || length(train_idx) == 0) {
        stop(sprintf("train_idx is NULL or empty for split %d", i))
      }
      if (is.null(test_idx) || length(test_idx) == 0) {
        stop(sprintf("test_idx is NULL or empty for split %d", i))
      }
      if (max(train_idx, na.rm = TRUE) > nrow(data) || max(test_idx, na.rm = TRUE) > nrow(data)) {
        stop(sprintf("Index out of bounds for split %d: max(train_idx)=%d, max(test_idx)=%d, nrow(data)=%d",
                    i, max(train_idx, na.rm = TRUE), max(test_idx, na.rm = TRUE), nrow(data)))
      }
      
      list(train_idx = train_idx, test_idx = test_idx)
    })
    
    logger$info("Extracted indices for %d splits. Validating structure...", length(split_indices))
    
    # Validate split_indices structure
    if (length(split_indices) != N_SPLITS) {
      stop(sprintf("split_indices length (%d) does not match N_SPLITS (%d)", 
                  length(split_indices), N_SPLITS))
    }
    
    # Validate first split as sample
    if (is.null(split_indices[[1]]) || !is.list(split_indices[[1]])) {
      stop("split_indices[[1]] is NULL or not a list")
    }
    if (!"train_idx" %in% names(split_indices[[1]]) || !"test_idx" %in% names(split_indices[[1]])) {
      stop(sprintf("split_indices[[1]] missing required fields. Got: %s", 
                  paste(names(split_indices[[1]]), collapse=", ")))
    }
    
    logger$info("Split indices validated. First split: %d train, %d test samples",
                length(split_indices[[1]]$train_idx), length(split_indices[[1]]$test_idx))
    
    check_memory_usage_r(logger, "After MC-CV Split Creation")
    check_memory_usage_r(logger, "After Index Extraction")
    
    # ============================================================
    # RUN MC-CV ANALYSIS
    # ============================================================
    logger$info("Running MC-CV analysis...")
    
    # Set up nested parallel plan for MC-CV within this cohort
    # Save current plan (cohort-level plan with 2 workers)
    cohort_plan <- future::plan("list")
    
    # Validate worker count again (in case we're in a nested worker environment)
    # Check available cores (respects mc.cores and other limits)
    available_cores <- tryCatch({
      parallelly::availableCores()
    }, error = function(e) {
      # Fallback to detectCores if parallelly not available
      parallel::detectCores()
    })
    
    # Cap workers to available cores (with safety margin)
    max_workers <- max(1, floor(available_cores * 0.8))
    if (N_WORKERS > max_workers) {
      logger$warning("Requested %d workers exceeds available cores (%d). Capping to %d workers.",
                     N_WORKERS, available_cores, max_workers)
      N_WORKERS <- max_workers
    }
    
    logger$info("Setting up MC-CV parallel plan with %d workers (available cores: %d)", 
                N_WORKERS, available_cores)
    
    # Set up MC-CV plan - use multisession directly with workers parameter
    # In worker environments, use plan() with explicit strategy and workers
    future::plan(strategy = future::multisession, workers = N_WORKERS)
    
    methods <- c("catboost", "random_forest")
    all_results <- list()
    
    check_memory_usage_r(logger, "Before MC-CV Execution")
    
    for (method in methods) {
      logger$info("Running MC-CV for %s...", method)
      check_memory_usage_r(logger, sprintf("Before MC-CV: %s", method))
      
      # Validate data before calling run_mc_cv_method
      if (is.null(data) || nrow(data) == 0) {
        stop(sprintf("data is NULL or empty for method %s", method))
      }
      if (is.null(split_indices) || length(split_indices) == 0) {
        stop(sprintf("split_indices is NULL or empty for method %s", method))
      }
      if (method == "random_forest") {
        if (is.null(data_rf) || nrow(data_rf) == 0) {
          stop("data_rf is NULL or empty for Random Forest method")
        }
        logger$info("Calling run_mc_cv_method for Random Forest: data=%d rows, data_rf=%d rows, splits=%d",
                   nrow(data), nrow(data_rf), length(split_indices))
        result <- run_mc_cv_method(data, method, split_indices, data_rf = data_rf)
      } else {
        logger$info("Calling run_mc_cv_method for CatBoost: data=%d rows, splits=%d",
                   nrow(data), length(split_indices))
        result <- run_mc_cv_method(data, method, split_indices)
      }
      all_results[[method]] <- result
      
      check_memory_usage_r(logger, sprintf("After MC-CV: %s", method))
      
      # Save individual results
      output_file <- file.path(output_dir, sprintf("%s_%s_%s_%s_feature_importance.csv",
                                                    cohort_name, age_band, event_year, method))
      readr::write_csv(result, output_file)
      logger$info("Saved: %s", output_file)
    }
    
    check_memory_usage_r(logger, "After MC-CV Execution")
    
    # Restore cohort-level plan after MC-CV is complete
    if (length(cohort_plan) > 0) {
      future::plan(cohort_plan)
      logger$info("Restored cohort-level parallel plan")
    }
    
    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    logger$info("Aggregating results...")
    check_memory_usage_r(logger, "Before Results Aggregation")
    
    metric_mean_col <- sprintf("mc_cv_%s_mean", if (SCALING_METRIC == "recall") "recall" else "logloss_inverted")
    metric_std_col <- sprintf("mc_cv_%s_std", if (SCALING_METRIC == "recall") "recall" else "logloss_inverted")
    
    top50_per_model <- list()
    for (method in names(all_results)) {
      model_recall <- mean(all_results[[method]][[metric_mean_col]], na.rm = TRUE)
      top50 <- all_results[[method]] %>%
        dplyr::arrange(dplyr::desc(importance_normalized)) %>%
        utils::head(50) %>%
        dplyr::mutate(
          importance_scaled = importance_normalized * .data[[metric_mean_col]],
          model = method,
          model_recall = model_recall
        ) %>%
        dplyr::select(feature, importance_normalized, importance_scaled, model, model_recall, 
               dplyr::all_of(c(metric_mean_col, metric_std_col)))
      top50_per_model[[method]] <- top50
    }
    
    all_top_features <- dplyr::bind_rows(top50_per_model)
    aggregated <- all_top_features %>%
      dplyr::group_by(feature) %>%
      dplyr::summarise(
        importance_normalized = sum(importance_normalized),
        importance_scaled = sum(importance_scaled),
        n_models = dplyr::n(),
        models = paste(model, collapse = ", "),
        recall_mean = mean(.data[[metric_mean_col]]),
        recall_std = mean(.data[[metric_std_col]]),
        .groups = 'drop'
      ) %>%
      dplyr::arrange(dplyr::desc(importance_scaled)) %>%
      dplyr::mutate(rank = dplyr::row_number())
    
    names(aggregated)[names(aggregated) == "recall_mean"] <- metric_mean_col
    names(aggregated)[names(aggregated) == "recall_std"] <- metric_std_col
    
    # Save aggregated results
    output_file <- file.path(output_dir, sprintf("%s_%s_%d_feature_importance_aggregated.csv",
                                                  cohort_name, age_band, event_year))
    readr::write_csv(aggregated, output_file)
    logger$info("Saved aggregated results: %s", output_file)
    
    check_memory_usage_r(logger, "After Results Aggregation")
    
    # ============================================================
    # SAVE LOGS TO S3
    # ============================================================
    logger$info("Saving logs to S3...")
    logger$close()
    save_logs_to_s3_r(log_file_path, cohort_name, age_band, event_year, logger)
    
    return(list(
      cohort = cohort_name,
      age_band = age_band,
      event_year = event_year,
      status = "success",
      aggregated = aggregated,
      output_file = output_file
    ))
    
  }, error = function(e) {
    # Capture error message and traceback before closing logger
    error_msg <- e$message
    tb_text <- ""
    tryCatch({
      tb <- capture.output(traceback())
      tb_text <- paste(tb, collapse = "\n")
    }, error = function(tb_err) {
      tb_text <- sprintf("Could not capture traceback: %s", tb_err$message)
    })
    
    # Log error (logger may have connection issues in parallel, so use tryCatch)
    tryCatch({
      logger$error("Analysis failed: %s", error_msg)
      logger$error("Traceback: %s", tb_text)
      logger$close()
      save_logs_to_s3_r(log_file_path, cohort_name, age_band, event_year, logger)
    }, error = function(log_err) {
      # If logging fails, at least print to console
      cat(sprintf("ERROR: Analysis failed for cohort %s: %s\n", cohort_name, error_msg))
      cat(sprintf("Traceback: %s\n", tb_text))
    })
    
    # Return error result (without logger object to avoid serialization issues)
    return(list(
      cohort = cohort_name,
      age_band = age_band,
      event_year = event_year,
      status = "error",
      error = error_msg
    ))
  })
}

