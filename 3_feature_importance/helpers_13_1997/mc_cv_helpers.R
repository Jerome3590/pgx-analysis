# ============================================================
# MONTE CARLO CROSS-VALIDATION HELPERS
# ============================================================
# Functions for running MC-CV analysis with CatBoost and Random Forest

# Run MC-CV for a single method (CatBoost or Random Forest)
# Args:
#   data: Data frame with target column (CatBoost format - factors)
#   method: "catboost" or "random_forest"
#   split_indices: List of train/test indices (lightweight, extracted from mc_splits)
#   data_rf: Optional data frame for Random Forest (numeric binary format)
# Returns:
#   Data frame with feature importance results
run_mc_cv_method <- function(data, method, split_indices, data_rf = NULL) {
  cat(sprintf("\n--- Running MC-CV for %s ---\n", method))

  # Select features and target based on model type ----------------------------
  if (method == "catboost") {
    X_all <- data %>% dplyr::select(-target)
    y_all <- data$target
    data_full <- data  # Keep full data for CatBoost (needs factor structure)
  } else if (method == "random_forest") {
    if (is.null(data_rf)) stop("Random Forest requires data_rf")
    X_all <- data_rf %>% dplyr::select(-target)
    y_all <- data_rf$target
    data_full <- NULL  # Not needed for RF
  } else {
    stop(sprintf("Unknown method: %s", method))
  }

  feature_names <- colnames(X_all)
  n_obs <- length(y_all)

  # Progress bar --------------------------------------------------------------
  p <- progressr::progressor(steps = N_SPLITS)

  # Core MC-CV loop -----------------------------------------------------------
  # Use split_indices instead of mc_splits to avoid passing large object
  # Memory optimization: split_indices is ~few MB vs mc_splits ~11GB
  results <- furrr::future_map(
    1:N_SPLITS,
    function(i) {
      p()

      # Get indices for this split (lightweight - just integers)
      indices <- split_indices[[i]]
      train_idx <- indices$train_idx
      test_idx <- indices$test_idx

      # Slice train/test based on method
      if (method == "catboost") {
        # For CatBoost, reconstruct data frame with factors from original data
        train_data <- data_full[train_idx, , drop = FALSE]
        test_data <- data_full[test_idx, , drop = FALSE]
        
        X_train <- train_data %>% dplyr::select(-target)
        y_train <- train_data$target
        
        X_test <- test_data %>% dplyr::select(-target)
        y_test <- test_data$target
      } else {
        # Random Forest: use indices directly on X_all/y_all
        X_train <- X_all[train_idx, , drop = FALSE]
        X_test <- X_all[test_idx, , drop = FALSE]
        y_train <- y_all[train_idx]
        y_test <- y_all[test_idx]
      }

      # Train --------------------------------------------------------------
      if (method == "catboost") {
        model <- train_catboost_r(X_train, y_train, MODEL_PARAMS$catboost)
      } else {
        model <- train_random_forest_r(X_train, y_train, MODEL_PARAMS$random_forest)
      }

      # Predict ------------------------------------------------------------
      if (method == "catboost") {
        y_pred       <- predict_catboost_r(model, X_test)
        y_pred_proba <- predict_proba_catboost_r(model, X_test)
      } else {
        y_pred       <- predict_random_forest_r(model, X_test)
        y_pred_proba <- predict_proba_random_forest_r(model, X_test)
      }

      # Metric -------------------------------------------------------------
      if (SCALING_METRIC == "recall") {
        metric_value <- calculate_recall(y_test, y_pred)
      } else {
        logloss <- calculate_logloss(y_test, y_pred_proba)
        metric_value <- if (is.infinite(logloss) || logloss == 0) 0 else 1 / logloss
      }

      # Feature importance -------------------------------------------------
      if (method == "catboost") {
        imp <- get_importance_catboost_r(model, X_test)
      } else {
        imp <- get_importance_random_forest_r(model)
      }

      # Fix importance length if needed
      if (length(imp) != length(feature_names)) {
        imp <- rep(0, length(feature_names))
      }
      names(imp) <- feature_names

      list(metric = metric_value, imp = imp)
    },
    .options = furrr::furrr_options(
      seed = 42,
      # Explicitly control globals to avoid copying unnecessary objects
      # Only pass what's needed: split_indices, X_all, y_all, data_full, method, feature_names
      # and helper functions/constants
      globals = c("split_indices", "X_all", "y_all", "data_full", "method", "feature_names",
                  "MODEL_PARAMS", "SCALING_METRIC",
                  "train_catboost_r", "train_random_forest_r",
                  "predict_catboost_r", "predict_random_forest_r",
                  "predict_proba_catboost_r", "predict_proba_random_forest_r",
                  "get_importance_catboost_r", "get_importance_random_forest_r",
                  "calculate_recall", "calculate_logloss")
    )
  )

  # Aggregate ---------------------------------------------------------------
  metric_values <- purrr::map_dbl(results, "metric")
  imp_matrix    <- do.call(rbind, purrr::map(results, "imp"))

  avg_imp <- colMeans(imp_matrix)

  # Normalize 0–1
  if (max(avg_imp) > min(avg_imp)) {
    norm_imp <- (avg_imp - min(avg_imp)) / (max(avg_imp) - min(avg_imp))
  } else {
    norm_imp <- rep(1 / length(avg_imp), length(avg_imp))
  }

  scaled_imp <- norm_imp * mean(metric_values)

  results_df <- tibble::tibble(
    feature               = feature_names,
    importance_raw        = avg_imp,
    importance_normalized = norm_imp,
    importance_scaled     = scaled_imp,
    model_type            = method,
    mc_cv_mean            = mean(metric_values),
    mc_cv_sd              = sd(metric_values)
  ) %>%
    dplyr::arrange(dplyr::desc(importance_scaled)) %>%
    dplyr::mutate(rank = dplyr::row_number())

  cat(sprintf("  Mean metric: %.4f ± %.4f\n",
              mean(metric_values), sd(metric_values)))
  cat("  Top features:\n")
  print(head(results_df, 50))

  results_df
}

