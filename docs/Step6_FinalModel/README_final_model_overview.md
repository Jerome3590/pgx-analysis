# Final Model Overview

## Hyperparameter Optimization with Optuna

This project integrates [Optuna](https://optuna.org/) for automated hyperparameter optimization of the final models (CatBoost, XGBoost, XGBoost RF). Optuna is used to search for the best hyperparameters using cross-validation and composite scoring (PR-AUC and logloss). The best parameters found by Optuna are used in the final model training pipeline. Example usage and results can be found in the training scripts and model summary outputs.

**Key steps:**
- Define an Optuna objective function for each model type.
- Run Optuna studies to optimize hyperparameters using MC-CV splits.
- Store and report the best hyperparameters and scores in the model summary.

See `train_final_model.py` for integration details.
# Step 8: Final Model Development

This folder contains documentation for final model training, evaluation, and CatBoost details.

## Documentation

- **[README_final_model.md](README_final_model.md)** - Final model training and evaluation
- **[README_catboost.md](README_catboost.md)** - CatBoost model details

## Related Documentation

- **Step 3**: See [`../Step3_FeatureImportance/`](../Step3_FeatureImportance/) for feature importance analysis
- **Step 10**: See [`../Step9_RiskDashboard/`](../Step9_RiskDashboard/) for dashboard deployment
- **Main Index**: See [`../README.md`](../README.md) for complete documentation index

