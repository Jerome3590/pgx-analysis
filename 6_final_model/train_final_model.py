#!/usr/bin/env python3
"""
Train final model for a given cohort and age band.

This script:
1. Loads the final feature table
2. Adds target column (all patients are target=1)
3. Runs MC-CV with CatBoost, XGBoost, XGBoost RF
4. Selects best model by logloss and PR-AUC
5. Trains final model on full dataset
6. Saves model artifacts

Usage:
    python train_final_model.py --cohort-name opioid_ed --age-band 0-12 --n-splits 200
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, log_loss, average_precision_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from py_helpers.feature_importance_model_utils import (  # noqa: E402
    train_catboost,
    train_xgboost,
    train_xgboost_rf,
    predict_catboost,
    predict_xgboost,
    predict_proba_catboost,
    predict_proba_xgboost,
)
from py_helpers.env_utils import get_sklearn_n_jobs, get_mc_cv_n_splits  # noqa: E402

# Default model parameters (matching feature importance analysis)
MODEL_PARAMS = {
    "catboost": {
        "iterations": 500,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "thread_count": get_sklearn_n_jobs(),
        "random_seed": 1997,
        "verbose": False,
        "task_type": "CPU",  # Disable GPU to avoid CUDA errors
    },
    "xgboost": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "n_jobs": get_sklearn_n_jobs(),
        "random_state": 1997,
    },
    "xgboost_rf": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "n_jobs": get_sklearn_n_jobs(),
        "random_state": 1997,
        "tree_method": "hist",
    },
}


def run_mc_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: list,
    n_splits: int = 200,
    train_prop: float = 0.8,
    random_state: int = 1997,
) -> pd.DataFrame:
    """Run Monte Carlo Cross-Validation for multiple models."""
    results = []

    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=1 - train_prop, random_state=random_state
    )

    for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name in models:
            try:
                # Train model
                if model_name == "catboost":
                    # CatBoost handles categorical features natively - use full X
                    model = train_catboost(
                        X_train, y_train, MODEL_PARAMS["catboost"]
                    )
                    y_pred = predict_catboost(model, X_test)
                    y_proba = predict_proba_catboost(model, X_test)
                elif model_name == "xgboost":
                    # XGBoost now handles categorical features natively
                    model = train_xgboost(X_train, y_train, MODEL_PARAMS["xgboost"])
                    y_pred = predict_xgboost(model, X_test)
                    y_proba = predict_proba_xgboost(model, X_test)
                elif model_name == "xgboost_rf":
                    # XGBoost RF now handles categorical features natively
                    model = train_xgboost_rf(
                        X_train, y_train, MODEL_PARAMS["xgboost_rf"]
                    )
                    y_pred = predict_xgboost(model, X_test)
                    y_proba = predict_proba_xgboost(model, X_test)
                else:
                    continue

                # Calculate metrics
                recall = recall_score(y_test, y_pred, zero_division=0)
                precision = precision_score(y_test, y_pred, zero_division=0)
                logloss = log_loss(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)

                results.append(
                    {
                        "split": split_idx,
                        "model": model_name,
                        "recall": recall,
                        "precision": precision,
                        "logloss": logloss,
                        "pr_auc": pr_auc,
                    }
                )

                if (split_idx + 1) % 50 == 0:
                    print(
                        f"Completed {split_idx + 1}/{n_splits} splits for {model_name}"
                    )

            except Exception as e:
                print(f"Error in split {split_idx} for {model_name}: {e}")
                continue

    return pd.DataFrame(results)


def train_final_model(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    n_splits: int | None = None,
) -> None:
    """Train final model for a cohort and age band."""

    age_band_fname = age_band.replace("-", "_")

    # Load final feature table
    # Use no-leakage version if available, otherwise use regular
    no_leakage_path = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
    )

    if no_leakage_path.exists():
        feature_table_path = no_leakage_path
        print(f"[INFO] Using no-leakage feature table: {feature_table_path}")
    else:
        feature_table_path = (
            project_root
            / "8_final_model"
            / "outputs"
            / cohort_name
            / age_band_fname
            / f"{cohort_name}_{age_band_fname}_train_final_features.csv"
        )
        print(f"[INFO] Using regular feature table: {feature_table_path}")

    if not feature_table_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")

    print(f"[INFO] Loading feature table from {feature_table_path}")
    df = pd.read_csv(feature_table_path)

    # Add target column (all patients are target=1)
    if "target" not in df.columns:
        df["target"] = 1
        print("[INFO] Added target column (all patients are target cases)")

    # Prepare X and y
    X = df.drop(columns=["mi_person_key", "target"], errors="ignore")
    y = df["target"].astype(int)

    # Drop datetime columns (target_time, first_time) - these are not features for modeling
    datetime_cols = ["target_time", "first_time"]
    cols_to_drop = [c for c in datetime_cols if c in X.columns]
    if cols_to_drop:
        print(f"[INFO] Dropping datetime columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop, errors='ignore')

    # Drop any remaining object columns that look like datetimes or IDs
    remaining_object_cols = X.select_dtypes(include=['object']).columns.tolist()
    cols_to_drop_obj = []
    for col in remaining_object_cols:
        sample_vals = X[col].dropna()
        if len(sample_vals) > 0:
            sample_str = str(sample_vals.iloc[0])
            # Check for datetime patterns
            if 'T' in sample_str and ('-' in sample_str[:10] or len(sample_str) > 15):
                cols_to_drop_obj.append(col)
            # Drop if all values are unique (likely IDs)
            elif X[col].nunique() == len(X):
                cols_to_drop_obj.append(col)

    if cols_to_drop_obj:
        print(f"[INFO] Dropping additional object columns: {cols_to_drop_obj}")
        X = X.drop(columns=cols_to_drop_obj, errors='ignore')

    print(f"[INFO] Dataset: {len(X)} patients, {len(X.columns)} features")
    print(f"[INFO] Target distribution: {y.value_counts().to_dict()}")

    # Check if we have both classes
    unique_classes = y.unique()
    if len(unique_classes) == 1:
        print(f"\n[WARNING] Only one class present ({unique_classes[0]}). Binary classification requires both classes.")
        print("[WARNING] This test workflow will skip model training.")
        print("[INFO] For full workflow, ensure control patients are included in the feature table.")
        print("[INFO] Feature table prepared successfully. Ready for full workflow with control patients.")
        return

    # Identify categorical features (keep them for CatBoost)
    cat_cols = [c for c in X.columns if X[c].dtype in ["object", "category"]]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"[INFO] Categorical features: {len(cat_cols)}")
    if cat_cols:
        print(f"[INFO] Categorical columns: {cat_cols}")
    print(f"[INFO] Numeric features: {len(num_cols)}")

    # For XGBoost, we'll encode categorical features or skip them
    # CatBoost will use them natively

    # Handle infinite values (replace with NaN, then fill with 0)
    inf_count = (X == float('inf')).sum().sum() + (X == float('-inf')).sum().sum()
    if inf_count > 0:
        print(f"[INFO] Found {inf_count} infinite values, replacing with 0")
        X = X.replace([float('inf'), float('-inf')], 0)

    # Fill any remaining NaN values with 0
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"[INFO] Filling {nan_count} NaN values with 0")
        X = X.fillna(0)

    # Run MC-CV
    if n_splits is None:
        n_splits = get_mc_cv_n_splits()
    print(f"\n[INFO] Running MC-CV with {n_splits} splits...")
    models = ["catboost", "xgboost", "xgboost_rf"]
    mc_cv_results = run_mc_cv(X, y, models, n_splits=n_splits)

    # Summarize results
    if len(mc_cv_results) == 0:
        print("\n[ERROR] No successful MC-CV runs. Check errors above.")
        return

    print("\n[INFO] MC-CV Results Summary:")
    summary = (
        mc_cv_results.groupby("model")
        .agg({
            "logloss": ["mean", "std"],
            "pr_auc": ["mean", "std"]
        })
        .round(4)
    )
    print(summary)

    # Select best model using logloss and PR-AUC
    # Lower logloss is better, higher PR-AUC is better
    # Composite score combines both: weighted average normalized to [0, 1]
    model_scores = []
    for model_name in mc_cv_results["model"].unique():
        model_data = mc_cv_results[mc_cv_results["model"] == model_name]
        mean_logloss = model_data["logloss"].mean()
        mean_pr_auc = model_data["pr_auc"].mean()

        # Normalize logloss: lower is better, so use inverse
        # Normalize to [0, 1] range: 1 / (1 + logloss)
        # This gives higher score for lower logloss
        normalized_logloss_score = 1 / (1 + mean_logloss)

        # PR-AUC is already in [0, 1], higher is better
        normalized_pr_auc_score = mean_pr_auc

        # Composite score: weighted average of normalized PR-AUC and logloss
        # Equal weight to both metrics (can adjust weights if needed)
        # Higher score is better
        composite_score = 0.5 * normalized_pr_auc_score + 0.5 * normalized_logloss_score

        model_scores.append({
            "model": model_name,
            "mean_logloss": mean_logloss,
            "mean_pr_auc": mean_pr_auc,
            "normalized_logloss": normalized_logloss_score,
            "normalized_pr_auc": normalized_pr_auc_score,
            "composite_score": composite_score
        })

    model_scores_df = pd.DataFrame(model_scores)
    model_scores_df = model_scores_df.sort_values("composite_score", ascending=False)

    print("\n[INFO] Model Ranking (by composite score: 0.5 * PR-AUC + 0.5 * (1/(1+logloss))):")
    print(model_scores_df[["model", "mean_logloss", "mean_pr_auc", "composite_score"]].to_string(index=False))

    best_model_name = model_scores_df.iloc[0]["model"]
    best_stats = model_scores_df.iloc[0]
    print(f"\n[INFO] Best model: {best_model_name}")
    print(f"  - LogLoss: {best_stats['mean_logloss']:.4f}")
    print(f"  - PR-AUC: {best_stats['mean_pr_auc']:.4f}")
    print(f"  - Normalized LogLoss: {best_stats['normalized_logloss']:.4f}")
    print(f"  - Normalized PR-AUC: {best_stats['normalized_pr_auc']:.4f}")
    print(f"  - Composite Score: {best_stats['composite_score']:.4f}")

    # Train final model on full dataset
    print(f"\n[INFO] Training final {best_model_name} model on full dataset...")

    # Identify categorical columns for final model training
    cat_cols = [c for c in X.columns if X[c].dtype in ["object", "category"]]

    # Train all three model types for reference
    all_models = {}
    
    print(f"\n[INFO] Training all three model types for reference...")
    
    # Train CatBoost
    print(f"[INFO] Training CatBoost model...")
    catboost_model = train_catboost(X, y, MODEL_PARAMS["catboost"])
    all_models["catboost"] = catboost_model
    
    # Train XGBoost
    print(f"[INFO] Training XGBoost model...")
    xgboost_model = train_xgboost(X, y, MODEL_PARAMS["xgboost"])
    all_models["xgboost"] = xgboost_model
    
    # Train XGBoost RF
    print(f"[INFO] Training XGBoost RF model...")
    xgboost_rf_model = train_xgboost_rf(X, y, MODEL_PARAMS["xgboost_rf"])
    all_models["xgboost_rf"] = xgboost_rf_model
    
    # Select the best model as the final model
    final_model = all_models[best_model_name]

    # Save model artifacts
    output_dir = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create final_model_json folder
    json_output_dir = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "final_model_json"
    )
    json_output_dir.mkdir(parents=True, exist_ok=True)

    # Save best model (joblib format)
    model_path = output_dir / f"{cohort_name}_{age_band_fname}_final_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"[INFO] Saved best model ({best_model_name}) to {model_path}")

    # Export all models to JSON for reference
    print(f"\n[INFO] Exporting all model types to JSON for reference...")
    feature_names = list(X.columns)
    
    for model_name, model in all_models.items():
        json_model_path = json_output_dir / f"{cohort_name}_{age_band_fname}_final_model_{model_name}.json"
        try:
            if model_name == "catboost":
                # CatBoost supports native JSON export
                model.save_model(str(json_model_path), format="json")
                print(f"[INFO] Exported CatBoost model to JSON: {json_model_path}")
            elif model_name in ["xgboost", "xgboost_rf"]:
                # XGBoost: get_booster() returns the booster, or model is already a Booster
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                else:
                    # Model is already a Booster object (XGBoost RF)
                    booster = model
                
                # Export model configuration and trees
                model_json = {
                    "model_type": model_name,
                    "feature_names": feature_names,
                    "n_features": len(feature_names),
                    "n_classes": 2,  # Binary classification
                    "objective": MODEL_PARAMS[model_name].get("objective", "binary:logistic"),
                    "trees": []
                }

                # Get tree dumps (as JSON-serializable format)
                tree_dumps = booster.get_dump(with_stats=True)
                for i, tree_dump in enumerate(tree_dumps):
                    model_json["trees"].append({
                        "tree_index": i,
                        "tree_dump": tree_dump
                    })

                # Save to JSON file
                with open(json_model_path, 'w') as f:
                    json.dump(model_json, f, indent=2)
                print(f"[INFO] Exported {model_name} model to JSON: {json_model_path}")
            else:
                print(f"[WARNING] JSON export not implemented for model type: {model_name}")
        except Exception as e:
            print(f"[WARNING] Failed to export {model_name} model to JSON: {e}")
    
    # Also save the best model with the default name for backward compatibility
    best_json_path = json_output_dir / f"{cohort_name}_{age_band_fname}_final_model.json"
    best_json_path_specific = json_output_dir / f"{cohort_name}_{age_band_fname}_final_model_{best_model_name}.json"
    if best_json_path_specific.exists() and best_json_path != best_json_path_specific:
        import shutil
        shutil.copy2(best_json_path_specific, best_json_path)
        print(f"[INFO] Copied best model JSON to default path: {best_json_path}")

    # Save MC-CV results
    results_path = output_dir / f"{cohort_name}_{age_band_fname}_mc_cv_results.csv"
    mc_cv_results.to_csv(results_path, index=False)
    print(f"[INFO] Saved MC-CV results to {results_path}")

    # Save summary
    summary_path = output_dir / f"{cohort_name}_{age_band_fname}_model_summary.txt"
    best_stats = model_scores_df.iloc[0]
    with open(summary_path, "w") as f:
        f.write("Final Model Training Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Cohort: {cohort_name}\n")
        f.write(f"Age Band: {age_band}\n")
        f.write(f"MC-CV Splits: {n_splits}\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write("Selection Method: Composite score (0.5 * PR-AUC + 0.5 * (1/(1+logloss)))\n\n")
        f.write("Best Model Metrics:\n")
        f.write(f"  - LogLoss: {best_stats['mean_logloss']:.4f}\n")
        f.write(f"  - PR-AUC: {best_stats['mean_pr_auc']:.4f}\n")
        f.write(f"  - Normalized LogLoss: {best_stats['normalized_logloss']:.4f}\n")
        f.write(f"  - Normalized PR-AUC: {best_stats['normalized_pr_auc']:.4f}\n")
        f.write(f"  - Composite Score: {best_stats['composite_score']:.4f}\n\n")
        f.write("MC-CV Results:\n")
        f.write(str(summary))
    print(f"[INFO] Saved summary to {summary_path}")

    print("\n[INFO] Final model training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train final model for a cohort and age band"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of MC-CV splits (default: 5 for test workflows)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    train_final_model(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()

