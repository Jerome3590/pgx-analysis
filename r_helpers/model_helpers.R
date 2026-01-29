# ============================================================
# MODEL HELPERS
# ============================================================
# Functions for training, prediction, and feature importance extraction
# Supports CatBoost and Random Forest models

# Train CatBoost model (R) - uses categorical factor features
# Args:
#   X_train: Training features (all columns must be factors for CatBoost)
#   y_train: Training labels (binary 0/1)
#   params: List of CatBoost parameters
# Returns:
#   Trained CatBoost model
train_catboost_r <- function(X_train, y_train, params) {
  # X_train should have ALL columns as factors (categorical features)
  # R's CatBoost automatically detects factor columns as categorical - no need to specify
  # Verify all columns are factors
  factor_cols <- names(X_train)[sapply(X_train, is.factor)]
  if (length(factor_cols) != ncol(X_train)) {
    warning(sprintf("Not all columns are factors! Factors: %d, Total: %d", 
                    length(factor_cols), ncol(X_train)))
  }
  
  # Create pool - R CatBoost automatically handles factor columns as categorical
  train_pool <- catboost::catboost.load_pool(
    data = X_train, 
    label = y_train
  )
  
  catboost_params <- list(
    iterations = params$iterations,
    learning_rate = params$learning_rate,
    depth = params$depth,
    loss_function = 'Logloss',
    eval_metric = 'Recall',
    verbose = params$verbose,
    logging_level = 'Silent',  # Suppress CatBoost logging to avoid thread safety warnings
    random_seed = params$random_seed
  )
  
  model <- catboost::catboost.train(train_pool, NULL, catboost_params)
  return(model)
}

# Train Random Forest model (R)
# Args:
#   X_train: Training features (numeric)
#   y_train: Training labels (binary 0/1)
#   params: List of Random Forest parameters
# Returns:
#   Trained Random Forest model
train_random_forest_r <- function(X_train, y_train, params) {
  if (is.null(params$mtry)) {
    params$mtry <- floor(sqrt(ncol(X_train)))
  }
  
  y_train_factor <- as.factor(y_train)
  
  model <- randomForest::randomForest(
    x = X_train,
    y = y_train_factor,
    ntree = params$ntree,
    mtry = params$mtry,
    nodesize = params$nodesize,
    maxnodes = params$maxnodes,
    importance = TRUE
  )
  
  return(model)
}

# Predict with CatBoost (R) - returns binary predictions
# Args:
#   model: Trained CatBoost model
#   X_test: Test features (all columns must be factors)
# Returns:
#   Binary predictions (0/1)
predict_catboost_r <- function(model, X_test) {
  # R's CatBoost automatically detects factor columns as categorical - no need to specify
  # Create test pool - R CatBoost automatically handles factor columns as categorical
  test_pool <- catboost::catboost.load_pool(data = X_test)
  pred_proba <- catboost::catboost.predict(model, test_pool, prediction_type = 'Probability')
  
  # Handle NA values: if pred_proba is NA, default to 0 (negative class)
  pred <- ifelse(is.na(pred_proba), 0, ifelse(pred_proba > 0.5, 1, 0))
  
  # Ensure no NA values remain
  if (any(is.na(pred))) {
    warning("NA values in CatBoost predictions, replacing with 0")
    pred[is.na(pred)] <- 0
  }
  
  return(pred)
}

# Predict probabilities with CatBoost (R) - returns probability values
# Args:
#   model: Trained CatBoost model
#   X_test: Test features (all columns must be factors)
# Returns:
#   Probability predictions (0-1)
predict_proba_catboost_r <- function(model, X_test) {
  test_pool <- catboost::catboost.load_pool(data = X_test)
  pred_proba <- catboost::catboost.predict(model, test_pool, prediction_type = 'Probability')
  
  # Handle NA values
  if (any(is.na(pred_proba))) {
    warning("NA values in CatBoost probability predictions, replacing with 0.5")
    pred_proba[is.na(pred_proba)] <- 0.5
  }
  
  return(pred_proba)
}

# Predict with Random Forest (R) - returns binary predictions
# Args:
#   model: Trained Random Forest model
#   X_test: Test features (numeric)
# Returns:
#   Binary predictions (0/1)
predict_random_forest_r <- function(model, X_test) {
  pred <- predict(model, X_test, type = 'response')
  pred <- as.integer(pred) - 1  # Convert factor to 0/1
  
  # Handle NA values: if prediction is NA, default to 0 (negative class)
  if (any(is.na(pred))) {
    warning("NA values in Random Forest predictions, replacing with 0")
    pred[is.na(pred)] <- 0
  }
  
  return(pred)
}

# Predict probabilities with Random Forest (R) - returns probability values
# Args:
#   model: Trained Random Forest model
#   X_test: Test features (numeric)
# Returns:
#   Probability predictions (0-1)
predict_proba_random_forest_r <- function(model, X_test) {
  pred_proba <- predict(model, X_test, type = 'prob')[, 2]  # Get probability of class 1
  
  # Handle NA values
  if (any(is.na(pred_proba))) {
    warning("NA values in Random Forest probability predictions, replacing with 0.5")
    pred_proba[is.na(pred_proba)] <- 0.5
  }
  
  return(pred_proba)
}

# Get feature importance from CatBoost (R)
# Args:
#   model: Trained CatBoost model
#   X_test: Test features (all columns must be factors)
# Returns:
#   Named vector of feature importances
get_importance_catboost_r <- function(model, X_test) {
  # R's CatBoost automatically detects factor columns as categorical - no need to specify
  # Verify all columns are factors (should match training)
  factor_cols <- names(X_test)[sapply(X_test, is.factor)]
  if (length(factor_cols) != ncol(X_test)) {
    warning(sprintf("Test data: Not all columns are factors! Factors: %d, Total: %d", 
                    length(factor_cols), ncol(X_test)))
  }
  
  # Create test pool - R CatBoost automatically handles factor columns as categorical
  test_pool <- catboost::catboost.load_pool(data = X_test)
  
  # Get feature importance - ensure it returns a vector
  importance <- catboost::catboost.get_feature_importance(model, pool = test_pool, type = 'PredictionValuesChange')
  
  # Ensure importance is a named vector with correct length
  if (length(importance) == 1 && ncol(X_test) > 1) {
    warning(sprintf("Feature importance returned single value instead of vector (expected %d features)", ncol(X_test)))
    # Fallback: return zeros with feature names
    importance <- setNames(rep(0, ncol(X_test)), names(X_test))
  } else if (length(importance) != ncol(X_test)) {
    warning(sprintf("Feature importance length mismatch: got %d, expected %d", length(importance), ncol(X_test)))
    # Try to pad or truncate to match
    if (length(importance) < ncol(X_test)) {
      importance <- c(importance, rep(0, ncol(X_test) - length(importance)))
    } else {
      importance <- importance[1:ncol(X_test)]
    }
    names(importance) <- names(X_test)
  } else if (is.null(names(importance))) {
    # Ensure names are set
    names(importance) <- names(X_test)
  }
  
  return(importance)
}

# Get feature importance from Random Forest (R)
# Args:
#   model: Trained Random Forest model
# Returns:
#   Named vector of feature importances (MeanDecreaseGini)
get_importance_random_forest_r <- function(model) {
  importance <- randomForest::importance(model)[, "MeanDecreaseGini"]
  return(importance)
}

