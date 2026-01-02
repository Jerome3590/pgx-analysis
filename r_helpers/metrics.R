# ============================================================
# METRICS CALCULATION
# ============================================================
# Functions for calculating evaluation metrics (Recall, LogLoss)
# All functions handle NA values appropriately

# Calculate Recall
# Args:
#   y_true: True binary labels (0/1)
#   y_pred: Predicted binary labels (0/1)
# Returns:
#   Recall score (0-1), or 0 if calculation fails
calculate_recall <- function(y_true, y_pred) {
  # Check inputs
  if (length(y_true) != length(y_pred)) {
    warning(sprintf("Length mismatch: y_true=%d, y_pred=%d", length(y_true), length(y_pred)))
    return(0)
  }
  
  # Remove NA values from both vectors
  valid_idx <- !is.na(y_true) & !is.na(y_pred)
  
  if (sum(valid_idx) == 0) {
    warning(sprintf("No valid predictions for recall calculation (y_true: %d NAs/%d, y_pred: %d NAs/%d)", 
                    sum(is.na(y_true)), length(y_true), sum(is.na(y_pred)), length(y_pred)))
    return(0)
  }
  
  y_true_clean <- y_true[valid_idx]
  y_pred_clean <- y_pred[valid_idx]
  
  # Check if we have any valid data
  if (length(y_true_clean) == 0) {
    warning("No valid data after filtering NAs")
    return(0)
  }
  
  tp <- sum((y_true_clean == 1) & (y_pred_clean == 1))
  fn <- sum((y_true_clean == 1) & (y_pred_clean == 0))
  
  # Handle edge case: no positive cases in true labels
  if (tp + fn == 0) {
    warning(sprintf("No positive cases in y_true for recall calculation (total valid: %d, positives: %d)", 
                    length(y_true_clean), sum(y_true_clean == 1)))
    return(0)
  }
  
  return(tp / (tp + fn))
}

# Calculate LogLoss (logarithmic loss)
# Lower is better, so we'll invert it for scaling (1/logloss or use negative)
# Args:
#   y_true: True binary labels (0/1)
#   y_pred_proba: Predicted probabilities (0-1)
# Returns:
#   LogLoss score, or Inf if calculation fails
calculate_logloss <- function(y_true, y_pred_proba) {
  # Remove NA values
  valid_idx <- !is.na(y_true) & !is.na(y_pred_proba)
  if (sum(valid_idx) == 0) {
    warning("No valid predictions for logloss calculation")
    return(Inf)  # Return Inf so 1/logloss = 0
  }
  
  y_true_clean <- y_true[valid_idx]
  y_pred_proba_clean <- y_pred_proba[valid_idx]
  
  # Clip probabilities to avoid log(0) or log(1)
  y_pred_proba_clean <- pmax(pmin(y_pred_proba_clean, 1 - 1e-15), 1e-15)
  
  # Calculate logloss
  logloss <- -mean(y_true_clean * log(y_pred_proba_clean) + (1 - y_true_clean) * log(1 - y_pred_proba_clean))
  
  return(logloss)
}

