#!/usr/bin/env Rscript
# Feature Importance Analysis - Multi-Model with MC-CV
# Based on: https://github.com/Jerome3590/phts/blob/main/graft-loss/feature_importance/replicate_20_features_MC_CV.R
# Adapted for binary classification (Recall instead of C-index)

# ============================================================================
# SETUP
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(tibble)
  library(purrr)
  library(catboost)
  library(randomForest)
  library(rsample)  # For MC-CV
  library(furrr)    # For parallel processing
  library(future)   # For parallel backend
  library(progressr) # For progress bars
  library(duckdb)   # For loading parquet files
  library(DBI)       # Database interface for DuckDB
})

cat("=== Feature Importance with MC-CV (R Implementation) ===\n")
cat("Models: CatBoost (R), Random Forest (R)\n")
cat("Metric: Recall (binary classification)\n\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript feature_importance_mc_cv.R <cohort_name> <age_band> <event_year> [n_splits] [test_size] [n_workers]")
}

COHORT_NAME <- args[1]
AGE_BAND <- args[2]
EVENT_YEAR <- as.integer(args[3])
N_SPLITS <- if (length(args) >= 4) as.integer(args[4]) else 10
TEST_SIZE <- if (length(args) >= 5) as.numeric(args[5]) else 0.2
N_WORKERS <- if (length(args) >= 6) as.integer(args[6]) else 0

# Debug mode
DEBUG_MODE <- as.logical(Sys.getenv("DEBUG_MODE", "FALSE"))
if (DEBUG_MODE) {
  N_SPLITS <- 5
  cat("🔍 DEBUG MODE: Using 5 splits\n\n")
}

# Model parameters
MODEL_PARAMS <- list(
  catboost = list(
    iterations = 100,
    learning_rate = 0.1,
    depth = 6,
    verbose = FALSE,
    random_seed = 42
  ),
  random_forest = list(
    ntree = 100,
    mtry = NULL,  # Will be set to sqrt(n_features)
    nodesize = 1,
    maxnodes = NULL
  )
)

# Output directory
OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("Configuration:\n"))
cat(sprintf("  Cohort: %s\n", COHORT_NAME))
cat(sprintf("  Age Band: %s\n", AGE_BAND))
cat(sprintf("  Event Year: %d\n", EVENT_YEAR))
cat(sprintf("  MC-CV Splits: %d\n", N_SPLITS))
cat(sprintf("  Test Size: %.1f%%\n", TEST_SIZE * 100))
cat(sprintf("  Output Dir: %s\n\n", OUTPUT_DIR))

# ============================================================================
# PARALLEL PROCESSING SETUP
# ============================================================================

if (N_WORKERS < 1) {
  total_cores <- parallel::detectCores()
  N_WORKERS <- max(1, total_cores - 2)
  cat(sprintf("Auto-detected %d cores, using %d workers\n", total_cores, N_WORKERS))
} else {
  cat(sprintf("Using %d workers\n", N_WORKERS))
}

# Increase future.globals.maxSize for large MC-CV splits
options(future.globals.maxSize = 20 * 1024^3)  # 20 GB
plan(multisession, workers = N_WORKERS)

# ============================================================================
# DATA LOADING
# ============================================================================

cat("\nLoading data...\n")

# Load data using DuckDB (same logic as FP-Growth notebook)
# Path structure: local_data_path/cohort_name={cohort}/event_year={year}/age_band={band}/cohort.parquet

# Determine local data path (same as FP-Growth)
LOCAL_DATA_PATH <- Sys.getenv("LOCAL_DATA_PATH", "/mnt/nvme/cohorts")
if (!dir.exists(LOCAL_DATA_PATH)) {
  # Try Windows path
  LOCAL_DATA_PATH <- Sys.getenv("LOCAL_DATA_PATH", "C:/Projects/pgx-analysis/data/gold/cohorts_F1120")
}

parquet_file <- file.path(LOCAL_DATA_PATH, 
                          paste0("cohort_name=", COHORT_NAME),
                          paste0("event_year=", EVENT_YEAR),
                          paste0("age_band=", AGE_BAND),
                          "cohort.parquet")

if (!file.exists(parquet_file)) {
  stop(sprintf("Cohort file not found: %s\nPlease check LOCAL_DATA_PATH and file structure.", parquet_file))
}

cat(sprintf("Loading from: %s\n", parquet_file))

# Load using DuckDB (via R's duckdb package)
if (!require(duckdb, quietly = TRUE)) {
  stop("duckdb package not installed. Install with: install.packages('duckdb')")
}

con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")

# Load cohort data (same columns as FP-Growth notebook)
query <- sprintf("
  SELECT 
    mi_person_key,
    target,
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

cohort_data <- dbGetQuery(con, query)
dbDisconnect(con)

cat(sprintf("Loaded %d event-level records\n", nrow(cohort_data)))
cat(sprintf("Unique patients: %d\n", length(unique(cohort_data$mi_person_key))))

# Create patient-level features (aggregate events per patient)
# Extract all unique items (drugs, ICD codes, CPT codes) per patient
cat("\nCreating patient-level features...\n")

# Get unique items per patient (same logic as FP-Growth)
patient_items <- cohort_data %>%
  # Drug names (pharmacy events)
  filter(!is.na(drug_name) & drug_name != "" & event_type == "pharmacy") %>%
  select(mi_person_key, item = drug_name) %>%
  # ICD codes (medical events) - all 5 columns
  bind_rows(
    cohort_data %>%
      filter(event_type == "medical") %>%
      select(mi_person_key, item = primary_icd_diagnosis_code) %>%
      filter(!is.na(item) & item != ""),
    cohort_data %>%
      filter(event_type == "medical") %>%
      select(mi_person_key, item = two_icd_diagnosis_code) %>%
      filter(!is.na(item) & item != ""),
    cohort_data %>%
      filter(event_type == "medical") %>%
      select(mi_person_key, item = three_icd_diagnosis_code) %>%
      filter(!is.na(item) & item != ""),
    cohort_data %>%
      filter(event_type == "medical") %>%
      select(mi_person_key, item = four_icd_diagnosis_code) %>%
      filter(!is.na(item) & item != ""),
    cohort_data %>%
      filter(event_type == "medical") %>%
      select(mi_person_key, item = five_icd_diagnosis_code) %>%
      filter(!is.na(item) & item != "")
  ) %>%
  # CPT codes (medical events)
  bind_rows(
    cohort_data %>%
      filter(!is.na(procedure_code) & procedure_code != "" & event_type == "medical") %>%
      select(mi_person_key, item = procedure_code)
  ) %>%
  distinct() %>%
  filter(!is.na(item) & item != "")

cat(sprintf("Extracted %d unique patient-item pairs\n", nrow(patient_items)))
cat(sprintf("Unique items: %d\n", length(unique(patient_items$item))))

# Create binary feature matrix (one-hot encoding)
# Each item becomes a binary feature (1 if patient has it, 0 otherwise)
feature_matrix <- patient_items %>%
  mutate(value = 1) %>%
  pivot_wider(
    id_cols = mi_person_key,
    names_from = item,
    values_from = value,
    values_fill = 0,
    names_prefix = "item_"
  )

# Get target per patient (should be same for all events of a patient)
patient_targets <- cohort_data %>%
  select(mi_person_key, target) %>%
  distinct() %>%
  group_by(mi_person_key) %>%
  summarise(target = first(target), .groups = 'drop')

# Merge features and target
data <- feature_matrix %>%
  left_join(patient_targets, by = "mi_person_key") %>%
  select(-mi_person_key)  # Remove ID column

# Extract target
y <- data$target
X <- data %>% select(-target)

# Convert to matrix for modeling
X <- as.matrix(X)

cat(sprintf("\nFinal dataset:\n"))
cat(sprintf("  Patients: %d\n", nrow(X)))
cat(sprintf("  Features: %d\n", ncol(X)))
cat(sprintf("  Target distribution: %d (%.1f%%) positive, %d (%.1f%%) negative\n",
            sum(y == 1), 100 * mean(y == 1), 
            sum(y == 0), 100 * mean(y == 0)))

# Convert back to data.frame for MC-CV
data <- as.data.frame(X)
data$target <- y

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Train CatBoost model (R)
train_catboost_r <- function(X_train, y_train, params) {
  # Prepare data for CatBoost
  train_pool <- catboost.load_pool(data = X_train, label = y_train)
  
  # Set parameters
  catboost_params <- list(
    iterations = params$iterations,
    learning_rate = params$learning_rate,
    depth = params$depth,
    loss_function = 'Logloss',
    eval_metric = 'Recall',
    verbose = params$verbose,
    random_seed = params$random_seed
  )
  
  # Train model
  model <- catboost.train(train_pool, NULL, catboost_params)
  return(model)
}

# Train Random Forest model (R)
train_random_forest_r <- function(X_train, y_train, params) {
  # Set mtry if not provided
  if (is.null(params$mtry)) {
    params$mtry <- floor(sqrt(ncol(X_train)))
  }
  
  # Convert to factor for classification
  y_train_factor <- as.factor(y_train)
  
  # Train model
  model <- randomForest(
    x = X_train,
    y = y_train_factor,
    ntree = params$ntree,
    mtry = params$mtry,
    nodesize = params$nodesize,
    maxnodes = params$maxnodes,
    importance = TRUE  # Calculate importance
  )
  
  return(model)
}

# Get feature importance from CatBoost (R)
get_importance_catboost_r <- function(model, X_test) {
  test_pool <- catboost.load_pool(data = X_test)
  importance <- catboost.get_feature_importance(model, pool = test_pool, type = 'PredictionValuesChange')
  return(importance)
}

# Get feature importance from Random Forest (R)
get_importance_random_forest_r <- function(model) {
  # Use MeanDecreaseGini (default)
  importance <- importance(model)[, "MeanDecreaseGini"]
  return(importance)
}

# Calculate Recall
calculate_recall <- function(y_true, y_pred) {
  tp <- sum((y_true == 1) & (y_pred == 1))
  fn <- sum((y_true == 1) & (y_pred == 0))
  if (tp + fn == 0) return(0)
  return(tp / (tp + fn))
}

# Predict with CatBoost (R)
predict_catboost_r <- function(model, X_test) {
  test_pool <- catboost.load_pool(data = X_test)
  pred_proba <- catboost.predict(model, test_pool, prediction_type = 'Probability')
  pred <- ifelse(pred_proba > 0.5, 1, 0)
  return(pred)
}

# Predict with Random Forest (R)
predict_random_forest_r <- function(model, X_test) {
  pred <- predict(model, X_test, type = 'response')
  pred <- as.integer(pred) - 1  # Convert factor to 0/1
  return(pred)
}

# Run MC-CV for a single model type
run_mc_cv_method <- function(data, method, mc_splits) {
  cat(sprintf("\n--- Running MC-CV for %s ---\n", method))
  
  # Extract features and target
  X <- data %>% select(-target)
  y <- data$target
  feature_names <- colnames(X)
  
  # Create progress bar
  p <- progressor(steps = N_SPLITS)
  
  # Run MC-CV in parallel
  results <- future_map(1:N_SPLITS, function(i) {
    p()
    
    # Get train/test split
    split <- mc_splits$splits[[i]]
    train_idx <- split$in_id
    test_idx <- split$out_id
    
    X_train <- X[train_idx, , drop = FALSE]
    X_test <- X[test_idx, , drop = FALSE]
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    # Train model
    model <- NULL
    if (method == "catboost") {
      model <- train_catboost_r(X_train, y_train, MODEL_PARAMS$catboost)
    } else if (method == "random_forest") {
      model <- train_random_forest_r(X_train, y_train, MODEL_PARAMS$random_forest)
    } else {
      stop(sprintf("Unknown method: %s", method))
    }
    
    # Get predictions
    if (method == "catboost") {
      y_pred <- predict_catboost_r(model, X_test)
    } else if (method == "random_forest") {
      y_pred <- predict_random_forest_r(model, X_test)
    }
    
    # Calculate Recall
    recall <- calculate_recall(y_test, y_pred)
    
    # Get feature importance
    if (method == "catboost") {
      importance <- get_importance_catboost_r(model, X_test)
    } else if (method == "random_forest") {
      importance <- get_importance_random_forest_r(model)
    }
    
    # Return results
    list(
      model = model,
      recall = recall,
      importance = importance
    )
  }, .options = furrr_options(seed = 42))
  
  # Extract results
  recalls <- map_dbl(results, ~ .x$recall)
  importance_matrix <- do.call(rbind, map(results, ~ .x$importance))
  
  # Average importance across splits
  avg_importance <- colMeans(importance_matrix)
  names(avg_importance) <- feature_names
  
  # Normalize importance (0-1 scale)
  min_imp <- min(avg_importance)
  max_imp <- max(avg_importance)
  if (max_imp > min_imp) {
    normalized_importance <- (avg_importance - min_imp) / (max_imp - min_imp)
  } else {
    normalized_importance <- rep(1 / length(avg_importance), length(avg_importance))
  }
  
  # Scale by mean MC-CV Recall
  mean_recall <- mean(recalls)
  scaled_importance <- normalized_importance * mean_recall
  
  # Create results DataFrame
  results_df <- tibble(
    feature = feature_names,
    importance_raw = avg_importance,
    importance_normalized = normalized_importance,
    importance_scaled = scaled_importance,
    model_type = method,
    mc_cv_recall_mean = mean_recall,
    mc_cv_recall_std = sd(recalls)
  ) %>%
    arrange(desc(importance_scaled)) %>%
    mutate(rank = row_number())
  
  cat(sprintf("  Mean Recall: %.4f ± %.4f\n", mean_recall, sd(recalls)))
  cat(sprintf("  Top 5 features:\n"))
  top5 <- head(results_df, 5)
  for (i in 1:nrow(top5)) {
    cat(sprintf("    %d. %s (scaled=%.6f)\n", 
                top5$rank[i], top5$feature[i], top5$importance_scaled[i]))
  }
  
  return(results_df)
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

cat("\n========================================\n")
cat("Creating MC-CV Splits\n")
cat("========================================\n")

# Create MC-CV splits (stratified by target)
mc_splits <- mc_cv(
  data = data,
  prop = 1 - TEST_SIZE,  # Training proportion
  times = N_SPLITS,
  strata = target  # Stratified by target
)

cat(sprintf("Created %d MC-CV splits (stratified)\n", N_SPLITS))

# Run each method
methods <- c("catboost", "random_forest")
all_results <- list()

for (method in methods) {
  result <- run_mc_cv_method(data, method, mc_splits)
  all_results[[method]] <- result
  
  # Save individual results
  output_file <- file.path(OUTPUT_DIR, sprintf("%s_%s_%s_%d_feature_importance.csv",
                                                COHORT_NAME, AGE_BAND, EVENT_YEAR, method))
  write_csv(result, output_file)
  cat(sprintf("Saved: %s\n", output_file))
}

# Aggregate across models
cat("\n========================================\n")
cat("Aggregating Results Across Models\n")
cat("========================================\n")

combined_df <- bind_rows(all_results)

# Aggregate by feature (average scaled importance)
aggregated <- combined_df %>%
  group_by(feature) %>%
  summarise(
    importance_raw = mean(importance_raw),
    importance_normalized = mean(importance_normalized),
    importance_scaled = mean(importance_scaled),
    mc_cv_recall_mean = mean(mc_cv_recall_mean),
    mc_cv_recall_std = mean(mc_cv_recall_std),
    .groups = 'drop'
  ) %>%
  arrange(desc(importance_scaled)) %>%
  mutate(rank = row_number())

# Save aggregated results
output_file <- file.path(OUTPUT_DIR, sprintf("%s_%s_%d_feature_importance_aggregated.csv",
                                              COHORT_NAME, AGE_BAND, EVENT_YEAR))
write_csv(aggregated, output_file)
cat(sprintf("Saved: %s\n", output_file))

# Print summary
cat("\n========================================\n")
cat("Summary\n")
cat("========================================\n")
cat(sprintf("Total features: %d\n", nrow(aggregated)))
cat(sprintf("Models used: %s\n", paste(methods, collapse = ", ")))
cat(sprintf("Mean MC-CV Recall: %.4f\n", mean(aggregated$mc_cv_recall_mean)))
cat("\nTop 10 features:\n")
top10 <- head(aggregated, 10)
for (i in 1:nrow(top10)) {
  cat(sprintf("  %2d. %-40s | scaled=%.6f | recall=%.4f\n",
              top10$rank[i], top10$feature[i], top10$importance_scaled[i], top10$mc_cv_recall_mean[i]))
}

# Close parallel processing
plan(sequential)

cat("\n========================================\n")
cat("Analysis Complete!\n")
cat("========================================\n")

