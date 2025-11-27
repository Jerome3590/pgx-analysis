"""
Model evaluation and metrics utilities.
"""
import sys
import os

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.common_imports import *


def analyze_drug_importance(model: Any, feature_name: str) -> Dict[str, float]:
    
    try:
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Get feature index
        feature_idx = model.feature_names_.index(feature_name)
        
        # Calculate metrics
        metrics = {
            'importance': importance[feature_idx],
            'rank': np.argsort(importance)[::-1].tolist().index(feature_idx) + 1,
            'percentile': np.percentile(importance, importance[feature_idx])
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing drug importance: {str(e)}")
        raise


def save_model_artifacts(model, X_train, X_test, y_train, y_test, shap_values, cattail_plots, 
                        model_metrics, causal_summary, paths, logger):
    """Save model artifacts to S3."""
    try:
        # Save model metrics
        if model_metrics:
            save_to_s3_json(
                json.dumps(model_metrics, indent=2),
                paths['model_metrics'],
                logger
            )
            logger.info(f"✓ Saved model metrics to {paths['model_metrics']}")

        # Save SHAP values
        if shap_values is not None:
            save_to_s3_parquet(
                pd.DataFrame(shap_values),
                paths['shap_values'],
                logger
            )
            logger.info(f"✓ Saved SHAP values to {paths['shap_values']}")

        # Save Cattail plots
        if cattail_plots:
            for plot_name, plot_data in cattail_plots.items():
                plot_path = f"{paths['cattail_plots']}/{plot_name}.html"
                save_to_s3_html(plot_data, plot_path, logger)
                logger.info(f"✓ Saved Cattail plot {plot_name} to {plot_path}")

        # Save causal summary
        if causal_summary:
            save_to_s3_json(
                json.dumps(causal_summary, indent=2),
                paths['causal_summary'],
                logger
            )
            logger.info(f"✓ Saved causal summary to {paths['causal_summary']}")

        # Create and save model info
        model_info = create_model_summary_json(
            model, X_train, X_test, y_train, y_test,
            shap_values, cattail_plots, model_metrics,
            causal_summary, paths, None, logger
        )
        save_to_s3_json(
            json.dumps(model_info, indent=2),
            paths['model_info'],
            logger
        )
        logger.info(f"✓ Saved model info to {paths['model_info']}")

    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise


def create_model_summary_json(model, X_train, X_test, y_train, y_test, shap_values, 
                            cattail_plots, model_metrics, causal_summary, paths, 
                            output_dir=None, logger=None):
    """Create a summary JSON for the model."""
    try:
        # Basic model information
        summary = {
            'model_type': type(model).__name__,
            'training_date': datetime.now().isoformat(),
            'feature_count': X_train.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': model_metrics if model_metrics else {},
            'causal_summary': causal_summary if causal_summary else {},
            'paths': {
                'shap_values': paths['shap_values'],
                'cattail_plots': paths['cattail_plots'],
                'model_metrics': paths['model_metrics'],
                'causal_summary': paths['causal_summary']
            }
        }

        # Add SHAP summary if available
        if shap_values is not None:
            summary['shap_summary'] = {
                'mean_abs_values': np.mean(np.abs(shap_values), axis=0).tolist(),
                'feature_importance': pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': np.mean(np.abs(shap_values), axis=0)
                }).sort_values('importance', ascending=False).to_dict('records')
            }

        # Add Cattail plot information
        if cattail_plots:
            summary['cattail_plots'] = {
                'plot_count': len(cattail_plots),
                'plot_names': list(cattail_plots.keys())
            }

        return summary

    except Exception as e:
        if logger:
            logger.error(f"Error creating model summary: {str(e)}")
        return {}


def consolidate_model_metrics(cohorts, age_bands, years, logger=None):
    """Consolidate model metrics across multiple cohorts."""
    try:
        all_metrics = []
        
        for cohort in cohorts:
            for band in age_bands:
                for year in years:
                    try:
                        # Load metrics for this cohort
                        metrics_path = f"s3://pgxdatalake/model_metrics/cohort_name={cohort}/age_band={band}/event_year={year}/model_metrics.json"
                        if s3_exists(metrics_path):
                            metrics = json.loads(s3_client.get_object(
                                Bucket="pgxdatalake",
                                Key=metrics_path.split('/', 3)[3]
                            )['Body'].read().decode('utf-8'))
                            
                            # Add cohort information
                            metrics['cohort'] = cohort
                            metrics['age_band'] = band
                            metrics['event_year'] = year
                            
                            all_metrics.append(metrics)
                    except Exception as e:
                        if logger:
                            logger.warning(f"Error loading metrics for {cohort}/{band}/{year}: {str(e)}")
                        continue
        
        # Convert to DataFrame
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            return df
        return pd.DataFrame()

    except Exception as e:
        if logger:
            logger.error(f"Error consolidating model metrics: {str(e)}")
        return pd.DataFrame()


def has_valid_metrics(metrics, cohort):
    """Check if metrics are valid for a given cohort."""
    required_fields = ['accuracy', 'precision', 'recall', 'f1_score']
    return all(field in metrics for field in required_fields)


def collect_metrics_as_dict(query, key_fields, value_fields, logger):
    """Collect metrics from a query into a dictionary."""
    try:
        results = {}
        for row in query:
            key = tuple(row[field] for field in key_fields)
            values = {field: row[field] for field in value_fields}
            results[key] = values
        return results
    except Exception as e:
        if logger:
            logger.error(f"Error collecting metrics: {str(e)}")
        return {}


def save_model_metrics(metrics, age_band, event_year, cohort):
    """Save metrics to S3."""
    try:
        metrics_path = f"s3://pgxdatalake/model_metrics/cohort_name={cohort}/age_band={age_band}/event_year={event_year}/model_metrics.json"
        save_to_s3_json(
            json.dumps(metrics, indent=2),
            metrics_path
        )
        return True
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")
        return False 


# ============================================================================
# Feature Importance Metrics
# ============================================================================

def calculate_recall(y_true, y_pred):
    """
    Calculate Recall (Sensitivity)
    
    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        
    Returns:
        Recall score (0-1), or 0 if calculation fails
    """
    from sklearn.metrics import recall_score
    
    # Check inputs
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        print(f"Warning: Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        return 0.0
    
    # Remove NA values from both arrays
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if np.sum(valid_idx) == 0:
        print(f"Warning: No valid predictions for recall calculation")
        return 0.0
    
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]
    
    # Check if we have any valid data
    if len(y_true_clean) == 0:
        print("Warning: No valid data after filtering NAs")
        return 0.0
    
    # Use sklearn for robust calculation
    try:
        return recall_score(y_true_clean, y_pred_clean, zero_division=0.0)
    except Exception as e:
        print(f"Warning: Error calculating recall: {e}")
        return 0.0


def calculate_logloss(y_true, y_pred_proba):
    """
    Calculate LogLoss (logarithmic loss)
    Lower is better, so we'll invert it for scaling (1/logloss)
    
    Args:
        y_true: True binary labels (0/1)
        y_pred_proba: Predicted probabilities (0-1)
        
    Returns:
        LogLoss score, or Inf if calculation fails
    """
    from sklearn.metrics import log_loss
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Remove NA values
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    if np.sum(valid_idx) == 0:
        print("Warning: No valid predictions for logloss calculation")
        return np.inf  # Return Inf so 1/logloss = 0
    
    y_true_clean = y_true[valid_idx]
    y_pred_proba_clean = y_pred_proba[valid_idx]
    
    # Clip probabilities to avoid log(0) or log(1)
    y_pred_proba_clean = np.clip(y_pred_proba_clean, 1e-15, 1 - 1e-15)
    
    # Calculate logloss using sklearn
    try:
        logloss = log_loss(y_true_clean, y_pred_proba_clean)
    except Exception as e:
        print(f"Warning: Error calculating logloss: {e}")
        return np.inf
    
    return logloss 