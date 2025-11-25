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
# Returns:
#   List with cohort name, status, aggregated results, and output file path
run_cohort_analysis <- function(cohort_name, age_band, event_year) {
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
      stop(sprintf("Cohort file not found: %s", parquet_file))
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
    
    # CatBoost format
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
      dplyr::mutate(value = 1) %>%
      tidyr::pivot_wider(
        id_cols = mi_person_key,
        names_from = item,
        values_from = value,
        values_fill = 0,
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
    
    mc_splits <- rsample::mc_cv(
      data = data,
      prop = TRAIN_PROP,
      times = N_SPLITS,
      strata = target
    )
    
    # Extract indices
    split_indices <- lapply(1:N_SPLITS, function(i) {
      split <- mc_splits$splits[[i]]
      train_idx <- split$in_id
      test_idx <- setdiff(seq_len(nrow(data)), train_idx)
      list(train_idx = train_idx, test_idx = test_idx)
    })
    
    check_memory_usage_r(logger, "After MC-CV Split Creation")
    check_memory_usage_r(logger, "After Index Extraction")
    
    # ============================================================
    # RUN MC-CV ANALYSIS
    # ============================================================
    logger$info("Running MC-CV analysis...")
    methods <- c("catboost", "random_forest")
    all_results <- list()
    
    check_memory_usage_r(logger, "Before MC-CV Execution")
    
    for (method in methods) {
      logger$info("Running MC-CV for %s...", method)
      check_memory_usage_r(logger, sprintf("Before MC-CV: %s", method))
      
      if (method == "random_forest") {
        result <- run_mc_cv_method(data, method, split_indices, data_rf = data_rf)
      } else {
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
      status = "success",
      aggregated = aggregated,
      output_file = output_file
    ))
    
  }, error = function(e) {
    logger$error("Analysis failed: %s", e$message)
    logger$error("Traceback: %s", paste(utils::capture.output(utils::traceback()), collapse = "\n"))
    logger$close()
    save_logs_to_s3_r(log_file_path, cohort_name, age_band, event_year, logger)
    return(list(
      cohort = cohort_name,
      status = "error",
      error = e$message
    ))
  })
}

