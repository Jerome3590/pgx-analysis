# Replicate 20-Feature Selection with Monte Carlo Cross-Validation
#
# This script implements the feature selection methodology from the original study
# with PROPER VALIDATION using Monte Carlo Cross-Validation (MC-CV):
# - Uses 500 random 75/25 train/test splits with stratification
# - Parallel processing with furrr/future
# - Evaluates models on unseen test data
# - Aggregates results across all splits
#
# Methods: RSF, CatBoost, AORSF
# Periods: Original (2010-2019), Full (2010-2024), Full No COVID
#
# Output: Feature importance + C-index metrics with confidence intervals

# ============================================================================
# SETUP
# ============================================================================

library(here)
library(dplyr)
library(readr)
library(survival)
library(ranger)
library(recipes)
library(tidyr)
library(tibble)
library(purrr)
library(janitor)
library(haven)
library(riskRegression)
library(prodlim)
library(aorsf)
library(catboost)
library(rsample)  # For MC-CV
library(furrr)    # For parallel processing
library(future)   # For parallel backend
library(progressr) # For progress bars

cat("=== Monte Carlo Cross-Validation Feature Selection ===\n")
cat("This script uses MC-CV with 500 train/test splits\n")
cat("Models are trained on 75% and evaluated on unseen 25%\n\n")

# Source required functions
cat("Sourcing required functions...\n")
if (file.exists(here("graft-loss-parallel-processing", "scripts", "R", "clean_phts.R"))) {
  source(here("graft-loss-parallel-processing", "scripts", "R", "clean_phts.R"))
  source(here("graft-loss-parallel-processing", "scripts", "R", "make_final_features.R"))
  source(here("graft-loss-parallel-processing", "scripts", "R", "select_rsf.R"))
  source(here("graft-loss-parallel-processing", "scripts", "R", "make_recipe.R"))
  source(here("graft-loss-parallel-processing", "scripts", "R", "make_labels.R"))
} else if (file.exists(here("graft-loss", "R", "clean_phts.R"))) {
  source(here("graft-loss", "R", "clean_phts.R"))
  source(here("graft-loss", "R", "select_rsf.R"))
  source(here("graft-loss", "R", "make_final_features.R"))
  source(here("graft-loss", "R", "make_recipe.R"))
  if (file.exists(here("graft-loss", "R", "make_labels.R"))) {
    source(here("graft-loss", "R", "make_labels.R"))
  }
} else {
  stop("Cannot find required R scripts")
}

# ============================================================================
# DEBUG/TEST MODE - Quick testing before full 1000-split run
# ============================================================================
# Set DEBUG_MODE = TRUE for quick testing (5 splits, ~2-5 min)
# Set DEBUG_MODE = FALSE for full analysis (1000 splits, ~30-45 min on EC2)

DEBUG_MODE <- as.logical(Sys.getenv("DEBUG_MODE", "FALSE"))  # Can set via environment variable

if (DEBUG_MODE) {
  cat("\n")
  cat("╔════════════════════════════════════════════════════════════════╗\n")
  cat("║                    🔍 DEBUG MODE ENABLED                       ║\n")
  cat("╚════════════════════════════════════════════════════════════════╝\n")
  cat("\n")
  cat("Quick test configuration:\n")
  cat("  • MC-CV Splits: 5 (instead of 1000)\n")
  cat("  • Period: Original only (2010-2019)\n")
  cat("  • Trees: Reduced (RSF: 100, AORSF: 50)\n")
  cat("  • Expected time: 2-5 minutes\n")
  cat("  • Purpose: Verify everything works before full run\n")
  cat("\n")
  cat("To run full analysis, set DEBUG_MODE=FALSE or remove environment variable\n")
  cat("\n")
}

# Configuration
n_predictors <- 20                                        # Top 20 features
n_trees_rsf <- if (DEBUG_MODE) 100 else 500              # RSF trees (reduced in debug)
n_trees_aorsf <- if (DEBUG_MODE) 50 else 100             # AORSF trees (reduced in debug)
horizon <- 1                                              # 1-year prediction
n_mc_splits <- if (DEBUG_MODE) 5 else 1000               # MC-CV splits (5 for debug, 1000 for full)
train_prop <- 0.75                                        # 75% training, 25% testing

# Set up parallel processing
# EC2 optimization: Use 30 out of 32 cores (leave 2 for system)
n_workers <- as.integer(Sys.getenv("N_WORKERS", "0"))
if (n_workers < 1) {
  # Auto-detect: use all cores minus 2 for system
  total_cores <- parallel::detectCores()
  n_workers <- max(1, total_cores - 2)
  cat(sprintf("Auto-detected %d cores, using %d workers\n", total_cores, n_workers))
}
cat(sprintf("Setting up parallel processing with %d workers...\n", n_workers))

# Increase future.globals.maxSize for large MC-CV splits object
# With 1TB RAM on EC2, we can handle large transfers
options(future.globals.maxSize = 20 * 1024^3)  # 20 GB limit (plenty for 1000 splits)
cat("Set future.globals.maxSize to 20 GB\n")

plan(multisession, workers = n_workers)

# Define Wisotzkey variables
wisotzkey_variables <- c(
  "prim_dx", "tx_mcsd", "chd_sv", "hxsurg", "txsa_r", "txbun_r",
  "txecmo", "txpl_year", "weight_txpl", "txalt", "bmi_txpl",
  "pra_listing", "egfr_tx", "hxmed", "listing_year"
)

# Create output directory
output_dir <- here("feature_importance", "outputs")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("Output directory:", output_dir, "\n\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Prepare data for modeling
prepare_modeling_data <- function(data) {
  # Find time and status columns
  time_col <- intersect(c("time", "outcome_int_graft_loss", "int_graft_loss"), names(data))[1]
  status_col <- intersect(c("status", "outcome_graft_loss", "graft_loss"), names(data))[1]
  
  if (is.na(time_col) || is.na(status_col)) {
    stop("Cannot find time/status columns")
  }
  
  # Rename to standard names
  if (time_col != "time") data <- data %>% rename(time = !!time_col)
  if (status_col != "status") data <- data %>% rename(status = !!status_col)
  
  # Exclude leakage variables and identifier columns
  exclude_exact <- c(
    "ID", "ptid_e", "int_dead", "int_death", "graft_loss", "txgloss", "death", "event",
    "dpricaus", "deathspc", "concod", "age_death", "dlist", "txpl_year",
    "rrace_b", "rrace_a", "rrace_ai", "rrace_pi", "rrace_o", "rrace_un", "race",
    "patsupp", "pmorexam", "papooth", "pacuref", "pishltgr",
    "pathero", "pcadrec", "pcadrem", "pdiffib", "cpathneg",
    "dcardiac", "dneuro", "dreject", "dsecaccs", "dpriaccs",
    "dconmbld", "dconmal", "dconcard", "dconneur", "dconrej",
    "dmajbld", "dmalcanc"
  )
  
  exclude_prefixes <- c("dtx_", "cc_", "dcon", "dpri", "dsec", "dmaj", "sd")
  
  exclude_by_prefix <- character(0)
  for (prefix in exclude_prefixes) {
    exclude_by_prefix <- c(exclude_by_prefix, 
                           names(data)[startsWith(names(data), prefix)])
  }
  
  exclude_all <- unique(c(exclude_exact, exclude_by_prefix))
  
  # Remove excluded variables
  data <- data %>% select(-any_of(exclude_all))
  
  # Median imputation for numeric variables
  numeric_vars <- names(data)[sapply(data, is.numeric) & names(data) != "time" & names(data) != "status"]
  for (var in numeric_vars) {
    if (any(is.na(data[[var]]))) {
      median_val <- median(data[[var]], na.rm = TRUE)
      data[[var]][is.na(data[[var]])] <- median_val
    }
  }
  
  # Mode imputation for categorical variables
  categorical_vars <- names(data)[sapply(data, function(x) is.factor(x) | is.character(x))]
  for (var in categorical_vars) {
    if (any(is.na(data[[var]]))) {
      mode_val <- names(sort(table(data[[var]]), decreasing = TRUE))[1]
      data[[var]][is.na(data[[var]])] <- mode_val
    }
  }
  
  # Remove constant columns
  constant_cols <- names(data)[sapply(data, function(x) length(unique(na.omit(x))) <= 1)]
  if (length(constant_cols) > 0) {
    data <- data %>% select(-any_of(constant_cols))
  }
  
  # Convert character to factor
  data <- data %>%
    mutate(across(where(is.character), as.factor))
  
  return(data)
}

# C-index calculation
calculate_cindex <- function(time, status, risk_scores, horizon = NULL) {
  valid_idx <- !is.na(time) & !is.na(status) & !is.na(risk_scores) &
               is.finite(time) & is.finite(risk_scores) & time > 0
  
  time   <- as.numeric(time[valid_idx])
  status <- as.numeric(status[valid_idx])
  risk   <- as.numeric(risk_scores[valid_idx])
  
  n <- length(time)
  events <- sum(status == 1)
  
  if (n < 10 || events < 1 || length(unique(risk)) == 1) {
    return(list(cindex_td = NA_real_, cindex_ti = NA_real_))
  }
  
  # Time-independent C-index (Harrell's)
  num_conc_ti <- 0
  num_disc_ti <- 0
  num_ties_ti <- 0
  
  for (i in seq_len(n)) {
    if (status[i] != 1) next
    for (j in seq_len(n)) {
      if (i == j) next
      if (time[i] < time[j]) {
        if (risk[i] > risk[j]) {
          num_conc_ti <- num_conc_ti + 1
        } else if (risk[i] < risk[j]) {
          num_disc_ti <- num_disc_ti + 1
        } else {
          num_ties_ti <- num_ties_ti + 1
        }
      }
    }
  }
  
  denom_ti <- num_conc_ti + num_disc_ti + num_ties_ti
  cindex_ti <- if (denom_ti > 0) (num_conc_ti + 0.5 * num_ties_ti) / denom_ti else NA_real_
  
  # Time-dependent C-index (using riskRegression::Score if available)
  cindex_td <- tryCatch({
    score_data <- data.frame(time = time, status = status)
    pred_matrix <- matrix(risk, ncol = 1)
    
    evaluation <- riskRegression::Score(
      object = list(Model = pred_matrix),
      formula = Surv(time, status) ~ 1,
      data = score_data,
      times = if (!is.null(horizon)) horizon else median(time[status == 1]),
      summary = "risks",
      metrics = "auc",
      se.fit = FALSE
    )
    
    as.numeric(evaluation$AUC$score$AUC[1])
  }, error = function(e) {
    cindex_ti  # Fall back to time-independent
  })
  
  return(list(cindex_td = cindex_td, cindex_ti = cindex_ti))
}

# ranger_predictrisk function (needed for RSF)
ranger_predictrisk <- function(object, newdata, times) {
  preds <- predict(object, data = newdata, type = "response")
  if (is.null(preds$survival)) {
    stop("ranger prediction did not return survival probabilities")
  }
  
  surv_matrix <- preds$survival
  time_points <- preds$unique.death.times
  
  # Find closest time point
  closest_idx <- which.min(abs(time_points - times))
  risk_scores <- 1 - surv_matrix[, closest_idx]
  
  return(as.numeric(risk_scores))
}

# ============================================================================
# MONTE CARLO CROSS-VALIDATION FUNCTIONS
# ============================================================================

# Run MC-CV for a single method and time period
run_mc_cv_method <- function(data, method, period_name, mc_splits) {
  
  cat(sprintf("\n=== Running MC-CV for %s (%s) ===\n", method, period_name))
  cat(sprintf("Splits: %d | Train: %.0f%% | Test: %.0f%%\n", 
              n_mc_splits, train_prop * 100, (1 - train_prop) * 100))
  
  # Extract split information
  split_ids <- seq_len(n_mc_splits)
  
  # Run splits in parallel with progress bar
  with_progress({
    p <- progressor(steps = n_mc_splits)
    
    results <- future_map(split_ids, function(split_id) {
      p()  # Update progress
      
      # Get train/test data from split
      split <- mc_splits$splits[[split_id]]
      train_data <- rsample::analysis(split)
      test_data <- rsample::assessment(split)
      
      # Train model on training set
      model <- NULL
      predictions <- NULL
      feature_importance <- NULL
      
      tryCatch({
        if (method == "RSF") {
          # RSF model
          model <- ranger(
            Surv(time, status) ~ .,
            data = train_data,
            num.trees = n_trees_rsf,
            importance = "permutation",
            min.node.size = 20,
            splitrule = "extratrees",
            num.random.splits = 10
          )
          
          # Predict on TEST data
          predictions <- ranger_predictrisk(
            object = model,
            newdata = test_data,
            times = horizon
          )
          
          # Get feature importance
          feature_importance <- model$variable.importance
          
        } else if (method == "AORSF") {
          # Remove constant columns from training data (can occur after train/test split)
          constant_cols <- names(train_data)[sapply(train_data, function(x) {
            length(unique(na.omit(x))) <= 1
          })]
          constant_cols <- setdiff(constant_cols, c("time", "status"))
          if (length(constant_cols) > 0) {
            train_data <- train_data %>% select(-any_of(constant_cols))
            test_data <- test_data %>% select(-any_of(constant_cols))
          }
          
          # AORSF model
          model <- aorsf::orsf(
            data = train_data,
            formula = Surv(time, status) ~ .,
            n_tree = n_trees_aorsf,
            na_action = 'impute_meanmode'
          )
          
          # Predict on TEST data
          pred_obj <- predict(model, new_data = test_data, 
                              pred_type = 'risk', pred_horizon = horizon)
          predictions <- if (is.matrix(pred_obj)) as.numeric(pred_obj[, 1]) else as.numeric(pred_obj)
          
          # Get feature importance
          feature_importance <- aorsf::orsf_vi_permute(model)
          
        } else if (method == "CatBoost") {
          # Remove constant columns from training data (can occur after train/test split)
          constant_cols <- names(train_data)[sapply(train_data, function(x) {
            length(unique(na.omit(x))) <= 1
          })]
          constant_cols <- setdiff(constant_cols, c("time", "status"))
          if (length(constant_cols) > 0) {
            train_data <- train_data %>% select(-any_of(constant_cols))
            test_data <- test_data %>% select(-any_of(constant_cols))
          }
          
          # CatBoost model
          # Prepare data
          train_pool <- catboost.load_pool(
            data = train_data %>% select(-time, -status),
            label = train_data$time
          )
          
          test_pool <- catboost.load_pool(
            data = test_data %>% select(-time, -status)
          )
          
          # Train model
          params <- list(
            loss_function = 'Cox',
            iterations = 100,
            learning_rate = 0.1,
            depth = 6,
            thread_count = 1,
            logging_level = 'Silent',
            verbose = 0L  # Integer 0 for silent mode (not boolean FALSE)
          )
          
          model <- catboost.train(train_pool, params = params)
          
          # Predict on TEST data
          predictions <- catboost.predict(model, test_pool)
          
          # Get feature importance - CatBoost returns a matrix with rownames as feature names
          # IMPORTANT: catboost.get_feature_importance() returns a matrix (not a named vector)
          # - Values are in the first column: importance_matrix[, 1]
          # - Feature names are in rownames: rownames(importance_matrix)
          # Convert to named vector for consistency with RSF and AORSF (which return named vectors directly)
          importance_matrix <- catboost.get_feature_importance(model)
          feature_importance <- as.numeric(importance_matrix[, 1])
          names(feature_importance) <- rownames(importance_matrix)
        }
        
        # Calculate C-index on TEST data
        cindex_result <- calculate_cindex(
          time = test_data$time,
          status = test_data$status,
          risk_scores = predictions,
          horizon = horizon
        )
        
        return(list(
          split_id = split_id,
          cindex_td = cindex_result$cindex_td,
          cindex_ti = cindex_result$cindex_ti,
          feature_importance = feature_importance,
          n_train = nrow(train_data),
          n_test = nrow(test_data),
          success = TRUE
        ))
        
      }, error = function(e) {
        return(list(
          split_id = split_id,
          cindex_td = NA_real_,
          cindex_ti = NA_real_,
          feature_importance = NULL,
          n_train = nrow(train_data),
          n_test = nrow(test_data),
          success = FALSE,
          error = e$message
        ))
      })
    }, .options = furrr_options(seed = TRUE))
  })
  
  # Aggregate results
  successful_splits <- Filter(function(x) x$success, results)
  n_successful <- length(successful_splits)
  
  cat(sprintf("Successful splits: %d / %d\n", n_successful, n_mc_splits))
  
  if (n_successful == 0) {
    stop(sprintf("All MC-CV splits failed for %s (%s)", method, period_name))
  }
  
  # Extract C-indexes
  cindex_td_values <- sapply(successful_splits, function(x) x$cindex_td)
  cindex_ti_values <- sapply(successful_splits, function(x) x$cindex_ti)
  
  # Remove NAs
  cindex_td_values <- cindex_td_values[!is.na(cindex_td_values)]
  cindex_ti_values <- cindex_ti_values[!is.na(cindex_ti_values)]
  
  # Aggregate feature importance across splits
  # All methods (RSF, AORSF, CatBoost) return named numeric vectors
  all_feature_names <- unique(unlist(lapply(successful_splits, function(x) {
    if (is.null(x$feature_importance)) return(NULL)
    names(x$feature_importance)
  })))
  
  aggregated_importance <- sapply(all_feature_names, function(feature) {
    importances <- sapply(successful_splits, function(x) {
      if (is.null(x$feature_importance)) return(NA_real_)
      if (feature %in% names(x$feature_importance)) {
        return(as.numeric(x$feature_importance[feature]))
      }
      return(NA_real_)
    })
    mean(importances, na.rm = TRUE)
  })
  
  # Ensure aggregated_importance is a numeric vector
  aggregated_importance <- as.numeric(aggregated_importance)
  names(aggregated_importance) <- all_feature_names
  
  # Sort by importance and get top 20
  top_features <- sort(aggregated_importance, decreasing = TRUE)[1:min(n_predictors, length(aggregated_importance))]
  
  # Calculate statistics
  results_summary <- list(
    method = method,
    period = period_name,
    n_splits = n_mc_splits,
    n_successful = n_successful,
    # C-index statistics
    cindex_td_mean = mean(cindex_td_values, na.rm = TRUE),
    cindex_td_sd = sd(cindex_td_values, na.rm = TRUE),
    cindex_td_ci_lower = quantile(cindex_td_values, 0.025, na.rm = TRUE),
    cindex_td_ci_upper = quantile(cindex_td_values, 0.975, na.rm = TRUE),
    cindex_ti_mean = mean(cindex_ti_values, na.rm = TRUE),
    cindex_ti_sd = sd(cindex_ti_values, na.rm = TRUE),
    cindex_ti_ci_lower = quantile(cindex_ti_values, 0.025, na.rm = TRUE),
    cindex_ti_ci_upper = quantile(cindex_ti_values, 0.975, na.rm = TRUE),
    # Top features
    top_features = top_features
  )
  
  # Print summary
  cat(sprintf("\n--- Results for %s (%s) ---\n", method, period_name))
  cat(sprintf("Time-Dependent C-Index: %.4f ± %.4f (95%% CI: %.4f - %.4f)\n",
              results_summary$cindex_td_mean,
              results_summary$cindex_td_sd,
              results_summary$cindex_td_ci_lower,
              results_summary$cindex_td_ci_upper))
  cat(sprintf("Time-Independent C-Index: %.4f ± %.4f (95%% CI: %.4f - %.4f)\n",
              results_summary$cindex_ti_mean,
              results_summary$cindex_ti_sd,
              results_summary$cindex_ti_ci_lower,
              results_summary$cindex_ti_ci_upper))
  # Display top 10 features sorted alphabetically for easier comparison
  top10_features <- names(top_features)[1:min(10, length(top_features))]
  top10_features_sorted <- sort(top10_features)
  cat(sprintf("Top 10 features (alphabetical): %s\n", paste(top10_features_sorted, collapse = ", ")))
  
  return(results_summary)
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

cat("Loading base data...\n")

# Load data
sas_path_local <- here("data", "phts_txpl_ml.sas7bdat")
sas_path_external <- here("graft-loss-parallel-processing", "data", "phts_txpl_ml.sas7bdat")
sas_path <- if (file.exists(sas_path_local)) sas_path_local else sas_path_external

if (!file.exists(sas_path)) {
  # Try graft-loss/data
  sas_path <- here("graft-loss", "data", "phts_txpl_ml.sas7bdat")
  if (!file.exists(sas_path)) {
    stop("Cannot find phts_txpl_ml.sas7bdat")
  }
}

phts_base <- haven::read_sas(sas_path) %>%
  filter(TXPL_YEAR >= 2010) %>%
  janitor::clean_names() %>%
  rename(
    outcome_int_graft_loss = int_graft_loss,
    outcome_graft_loss = graft_loss
  ) %>%
  mutate(
    ID = 1:n(),
    across(.cols = where(is.character), ~ ifelse(.x %in% c("", "unknown", "missing"), NA_character_, .x)),
    across(.cols = where(is.character), as.factor),
    tx_mcsd = if ('txnomcsd' %in% names(.)) {
      if_else(txnomcsd == 'yes', 0, 1)
    } else if ('txmcsd' %in% names(.)) {
      txmcsd
    } else {
      NA_real_
    }
  )

cat(sprintf("Loaded data: %d rows, %d columns\n", nrow(phts_base), ncol(phts_base)))

# Define time periods
periods <- list()
periods$original <- phts_base %>% filter(txpl_year >= 2010 & txpl_year <= 2019)
periods$full <- phts_base %>% filter(txpl_year >= 2010)
periods$full_no_covid <- phts_base %>% filter(txpl_year >= 2010 & !(txpl_year >= 2020 & txpl_year <= 2023))

# Select periods and methods to run
# In DEBUG_MODE, only run original period for speed
period_names <- if (DEBUG_MODE) {
  c("original")  # Debug: just one period (~2-5 min)
} else {
  c("original", "full", "full_no_covid")  # Full: all periods (~30-45 min)
}

method_names <- c("RSF", "CatBoost", "AORSF")  # All methods

# Run analysis for each period and method
all_results <- list()

for (period_name in period_names) {
  cat(sprintf("\n========================================\n"))
  cat(sprintf("Processing Period: %s\n", period_name))
  cat(sprintf("========================================\n"))
  
  # Prepare data
  period_data <- prepare_modeling_data(periods[[period_name]])
  
  cat(sprintf("Period data: %d rows, %d columns\n", nrow(period_data), ncol(period_data)))
  cat(sprintf("Events: %d (%.2f%%)\n", sum(period_data$status), 
              100 * sum(period_data$status) / nrow(period_data)))
  
  # Create MC-CV splits (stratified by outcome)
  cat(sprintf("Creating %d MC-CV splits (stratified)...\n", n_mc_splits))
  mc_splits <- mc_cv(
    data = period_data,
    prop = train_prop,
    times = n_mc_splits,
    strata = status
  )
  
  # Run each method
  period_results <- list()
  
  for (method in method_names) {
    result <- run_mc_cv_method(period_data, method, period_name, mc_splits)
    period_results[[method]] <- result
    
    # Save top features (sorted alphabetically for easier comparison)
    top_features_df <- tibble(
      feature = names(result$top_features),
      importance = as.numeric(result$top_features),
      cindex_td = result$cindex_td_mean,
      cindex_ti = result$cindex_ti_mean
    ) %>%
      arrange(feature)  # Sort alphabetically for easier visual comparison
    
    output_file <- file.path(output_dir, sprintf("%s_%s_top20.csv", 
                                                  period_name, tolower(method)))
    write_csv(top_features_df, output_file)
    cat(sprintf("Saved: %s\n", output_file))
  }
  
  all_results[[period_name]] <- period_results
}

# ============================================================================
# SAVE SUMMARY RESULTS
# ============================================================================

cat("\n========================================\n")
cat("Saving Summary Results\n")
cat("========================================\n")

# Create C-index comparison table
cindex_comparison <- map_df(period_names, function(period) {
  map_df(method_names, function(method) {
    result <- all_results[[period]][[method]]
    tibble(
      period = period,
      method = method,
      cindex_td_mean = result$cindex_td_mean,
      cindex_td_sd = result$cindex_td_sd,
      cindex_td_ci_lower = result$cindex_td_ci_lower,
      cindex_td_ci_upper = result$cindex_td_ci_upper,
      cindex_ti_mean = result$cindex_ti_mean,
      cindex_ti_sd = result$cindex_ti_sd,
      cindex_ti_ci_lower = result$cindex_ti_ci_lower,
      cindex_ti_ci_upper = result$cindex_ti_ci_upper,
      n_splits = result$n_successful
    )
  })
})

write_csv(cindex_comparison, file.path(output_dir, "cindex_comparison_mc_cv.csv"))
cat("Saved: cindex_comparison_mc_cv.csv\n")

# Create summary statistics
summary_stats <- map_df(period_names, function(period) {
  period_data <- periods[[period]]
  tibble(
    period = period,
    n_patients = nrow(period_data),
    n_events = sum(period_data$outcome_graft_loss, na.rm = TRUE),
    event_rate = 100 * sum(period_data$outcome_graft_loss, na.rm = TRUE) / nrow(period_data)
  )
})

write_csv(summary_stats, file.path(output_dir, "summary_statistics_mc_cv.csv"))
cat("Saved: summary_statistics_mc_cv.csv\n")

# Close parallel processing
plan(sequential)

cat("\n========================================\n")
cat("Analysis Complete!\n")
cat("========================================\n")
cat(sprintf("Output directory: %s\n", output_dir))
cat(sprintf("MC-CV splits: %d\n", n_mc_splits))
cat(sprintf("Train/Test ratio: %.0f/%.0f\n", train_prop * 100, (1 - train_prop) * 100))
cat("\nResults show C-indexes with 95% confidence intervals\n")
cat("based on", n_mc_splits, "independent train/test splits.\n")

