#!/usr/bin/env python3
"""
Run SHAP analysis for final models for a given (cohort, age_band).

Outputs:
  8_shap_analysis/outputs/{cohort}/{age_band_fname}/
    - {cohort}_{age_band_fname}_shap_global_importance_xgboost.csv
    - {cohort}_{age_band_fname}_shap_global_importance_catboost.csv
    - {cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet
    - {cohort}_{age_band_fname}_shap_sample_values_catboost.parquet
    - summary bar / beeswarm plots (PNG) per model
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # type: ignore


def _load_final_features(cohort: str, age_band: str) -> Tuple[pd.DataFrame, pd.Series]:
    age_band_fname = age_band_to_fname(age_band)
    features_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Final features file not found: {features_path}")

    df = pd.read_csv(features_path)
    if "target" not in df.columns:
        raise ValueError(f"'target' column not found in {features_path}")

    y = df["target"].astype(int)
    X = df.drop(columns=["mi_person_key", "target"], errors="ignore")

    # Keep numeric columns only (model is trained on numeric features)
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols].copy()
    return X, y


from py_helpers.env_utils import get_xgb_cpu_nthread  # noqa: E402


def _load_best_models(cohort: str, age_band: str):
    """
    Load the best models selected by the final model training step.
    
    Returns:
        - best_catboost_model: CatBoost model loaded from .cbm binary
        - model_selection_metadata: Dict with selection information
    """
    age_band_fname = age_band_to_fname(age_band)
    
    # Load model selection metadata
    metadata_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
    )
    
    import json
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            model_selection_metadata = json.load(f)
    else:
        print(f"Warning: Model selection metadata not found at {metadata_path}")
        model_selection_metadata = {}
    
    # Try loading CatBoost binary model from models directory first (preferred, consistent with XGBoost)
    cb_binary_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "models"
        / "catboost_model.cbm"
    )
    
    # Fallback to model_outputs location
    if not cb_binary_path.exists():
        cb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / "models"
            / "catboost_model.cbm"
        )
    
    # Fallback to final_model_json location (legacy)
    if not cb_binary_path.exists():
        cb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "outputs"
            / cohort
            / age_band_fname
            / "final_model_json"
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
    
    # Final fallback to model_outputs root (legacy)
    if not cb_binary_path.exists():
        cb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
    
    if not cb_binary_path.exists():
        raise FileNotFoundError(
            f"Best CatBoost model binary not found. Checked:\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / 'models' / 'catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'final_model_json' / f'{cohort}_{age_band_fname}_best_catboost_model.cbm'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / f'{cohort}_{age_band_fname}_best_catboost_model.cbm'}\n"
            f"Please run 6_final_model_selection/run_final_model.py first."
        )
    
    from catboost import CatBoostClassifier  # type: ignore
    cb_model = CatBoostClassifier()
    cb_model.load_model(str(cb_binary_path))
    print(f"Loaded best CatBoost model from {cb_binary_path}")
    
    return cb_model, model_selection_metadata


def _load_best_xgboost_model(cohort: str, age_band: str):
    """
    Load the best XGBoost model saved by the final model training step.

    Prefers native XGBoost booster binary model (UBJ format, most reliable for SHAP).
    Falls back to joblib if binary not available.

    Returns:
        - best_xgboost_model: XGBoost model (loaded from binary or joblib)
    """
    import xgboost as xgb  # type: ignore
    
    age_band_fname = age_band_to_fname(age_band)

    # Try loading native XGBoost booster binary model first (preferred for SHAP)
    xgb_binary_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "models"
        / "xgboost_model.ubj"
    )

    # Fallback to model_outputs location
    if not xgb_binary_path.exists():
        xgb_binary_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / "models"
            / "xgboost_model.ubj"
        )

    if xgb_binary_path.exists():
        # Load from native binary model (most reliable for SHAP, avoids base_score issues)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_binary_path))
        print(f"Loaded best XGBoost model from native binary: {xgb_binary_path}")
        return xgb_model

    # Fallback to joblib if JSON not available
    xgb_joblib_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / cohort
        / age_band_fname
        / "models"
        / "xgboost.joblib"
    )

    if not xgb_joblib_path.exists():
        xgb_joblib_path = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
            / "models"
            / "xgboost.joblib"
        )

    if not xgb_joblib_path.exists():
        raise FileNotFoundError(
            f"Best XGBoost model not found. Checked:\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'xgboost_model.ubj'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / 'models' / 'xgboost_model.ubj'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'models' / 'xgboost.joblib'}\n"
            f"  - {PROJECT_ROOT / '6_final_model' / 'model_outputs' / cohort / age_band_fname / 'models' / 'xgboost.joblib'}\n"
            f"Please run 6_final_model_selection/run_final_model.py first."
        )

    # Load from joblib and convert to booster for SHAP
    xgb_model = joblib.load(str(xgb_joblib_path))
    print(f"Loaded best XGBoost model from joblib: {xgb_joblib_path}")
    
    # Convert to booster and fix base_score issue for SHAP compatibility
    if hasattr(xgb_model, 'get_booster'):
        import tempfile
        import os
        import json
        import ast
        
        booster = xgb_model.get_booster()
        
        # Fix base_score in booster config if it's in string array format
        config = json.loads(booster.save_config())
        learner_model_param = config.get('learner', {}).get('learner_train_param', {})
        base_score_str = learner_model_param.get('base_score', '0.5')
        
        # Check if base_score is in problematic format like '[1.6610055E-1]'
        if isinstance(base_score_str, str) and base_score_str.startswith('[') and base_score_str.endswith(']'):
            try:
                # Parse the array string and extract the float value
                base_score_value = ast.literal_eval(base_score_str)
                if isinstance(base_score_value, list) and len(base_score_value) > 0:
                    base_score_value = float(base_score_value[0])
                else:
                    base_score_value = float(base_score_value)
                
                # Update the config with the fixed base_score
                learner_model_param['base_score'] = str(base_score_value)
                config['learner']['learner_train_param'] = learner_model_param
                
                # Save to temp file and reload to apply the fix
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
                    json.dump(config, tmp_json, indent=2)
                    tmp_json_path = tmp_json.name
                
                # Load the fixed config into a new booster
                booster.load_config(tmp_json_path)
                try:
                    os.unlink(tmp_json_path)
                except:
                    pass
                
                print(f"Fixed base_score from '{base_score_str}' to '{base_score_value}'")
            except Exception as e:
                print(f"[WARNING] Could not fix base_score: {e}")
        
        # Save booster to temp binary (UBJ) and reload into new model
        with tempfile.NamedTemporaryFile(suffix='.ubj', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        booster.save_model(tmp_path)
        xgb_model_for_shap = xgb.XGBClassifier()
        xgb_model_for_shap.load_model(tmp_path)
        try:
            os.unlink(tmp_path)
        except:
            pass
        print("Converted joblib model to booster format for SHAP compatibility")
        return xgb_model_for_shap
    
    return xgb_model


def _fit_models_for_shap(X: pd.DataFrame, y: pd.Series, cohort: str, age_band: str, random_seed: int = 42):
    """
    Load best CatBoost and XGBoost models for SHAP analysis.

    Uses the best models selected by the final model training step.
    """
    # Load best CatBoost model
    cb_model, model_selection_metadata = _load_best_models(cohort, age_band)

    # Load best XGBoost model (instead of retraining)
    try:
        xgb_model = _load_best_xgboost_model(cohort, age_band)
    except FileNotFoundError:
        # Fallback: if model not found, retrain (shouldn't happen in normal workflow)
        print("Warning: Best XGBoost model not found. Retraining from scratch...")
        import xgboost as xgb  # type: ignore
        nthread = get_xgb_cpu_nthread()
        from py_helpers.env_utils import is_linux
        device = "cpu" if is_linux() else "cuda"
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=random_seed,
        )
        try:
            xgb_model.fit(X, y)
        except Exception:
            xgb_model.set_params(tree_method="hist")
            if "device" in xgb_model.get_params():
                xgb_model.set_params(device="cpu")
            xgb_model.fit(X, y)

    return xgb_model, cb_model


def run_shap_analysis(
    cohort: str,
    age_band: str,
    n_background: int = 1000,
    n_eval: int = 2000,
) -> bool:
    """
    Run SHAP analysis for XGBoost and CatBoost models.
    
    Returns:
        bool: True if at least one model was successfully analyzed, False otherwise
    """
    import matplotlib.pyplot as plt

    try:
        import shap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'shap' library is required for SHAP analysis. "
            "Install with: pip install shap"
        ) from e

    age_band_fname = age_band_to_fname(age_band)
    out_dir = (
        PROJECT_ROOT / "8_shap_analysis" / "outputs" / cohort / age_band_fname
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading final features for {cohort}, {age_band}...")
    X, y = _load_final_features(cohort, age_band)
    print(f"Final feature matrix: {X.shape[0]} rows, {X.shape[1]} features.")

    # Sample background and evaluation sets for SHAP efficiency
    rng = np.random.default_rng(42)
    idx_all = np.arange(X.shape[0])
    rng.shuffle(idx_all)

    bg_idx = idx_all[: min(n_background, len(idx_all))]
    eval_idx = idx_all[: min(n_eval, len(idx_all))]
    X_bg = X.iloc[bg_idx]
    X_eval = X.iloc[eval_idx]

    print("Loading best models for SHAP...")
    xgb_clf, cb_clf = _fit_models_for_shap(X, y, cohort, age_band)

    feature_names = list(X.columns)
    s3_outputs = []  # Track S3 uploads for checkpointing
    
    # Track whether at least one model was successfully analyzed
    models_analyzed = []
    shap_xgb = None  # Initialize for scope

    # ------------------- XGBoost SHAP -------------------
    print("Computing SHAP values for XGBoost...")
    
    # Use booster directly for SHAP (model was loaded from native JSON or converted from joblib)
    # Native JSON models avoid base_score parsing issues entirely
    try:
        import xgboost as xgb  # type: ignore
        
        if hasattr(xgb_clf, 'get_booster'):
            booster = xgb_clf.get_booster()
            # Try TreeExplainer with booster directly (most reliable, no base_score parsing issues)
            try:
                expl_xgb = shap.TreeExplainer(
                    booster, data=X_bg.values, feature_perturbation="interventional", model_output="probability"
                )
                shap_xgb = expl_xgb.shap_values(X_eval.values)
                print("✅ Successfully computed SHAP values using TreeExplainer with booster")
            except (ValueError, TypeError) as e:
                # If base_score issue persists, use PermutationExplainer as fallback
                print(f"[WARNING] TreeExplainer failed ({e}), falling back to PermutationExplainer...")
                expl_xgb = shap.PermutationExplainer(
                    xgb_clf.predict_proba, X_bg.values, max_evals=100
                )
                shap_xgb = expl_xgb.shap_values(X_eval.values)
                # PermutationExplainer returns (n_samples, n_classes, n_features) for binary classification
                # Extract SHAP values for class 1 (positive class)
                if shap_xgb.ndim == 3:
                    shap_xgb = shap_xgb[:, 1, :]  # Extract class 1 SHAP values
                print("✅ Successfully computed SHAP values using PermutationExplainer")
        else:
            # Fallback: use model directly if no booster available
            try:
                expl_xgb = shap.TreeExplainer(
                    xgb_clf, data=X_bg, feature_perturbation="interventional", model_output="probability"
                )
                shap_xgb = expl_xgb.shap_values(X_eval)
                print("✅ Successfully computed SHAP values using TreeExplainer with model")
            except (ValueError, TypeError) as e:
                # If TreeExplainer fails, use PermutationExplainer
                print(f"[WARNING] TreeExplainer failed ({e}), falling back to PermutationExplainer...")
                expl_xgb = shap.PermutationExplainer(
                    xgb_clf.predict_proba, X_bg, max_evals=100
                )
                shap_xgb = expl_xgb.shap_values(X_eval)
                if shap_xgb.ndim == 3:
                    shap_xgb = shap_xgb[:, 1, :]  # Extract class 1 SHAP values
                print("✅ Successfully computed SHAP values using PermutationExplainer")
        
        models_analyzed.append("xgboost")
    except Exception as e:
        print(f"[ERROR] XGBoost SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        shap_xgb = None

    # Global importance (only if XGBoost was analyzed)
    if "xgboost" in models_analyzed and shap_xgb is not None:
        mean_abs_xgb = np.abs(shap_xgb).mean(axis=0)
        mean_xgb = shap_xgb.mean(axis=0)  # Mean SHAP value (captures direction)
        xgb_imp_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_xgb,
            "mean_shap": mean_xgb,  # Direction: positive = increases risk, negative = decreases risk
        })
        # Filter to features with mean_abs_shap > 0 and sort by importance
        xgb_imp_df = xgb_imp_df[xgb_imp_df['mean_abs_shap'] > 0].sort_values("mean_abs_shap", ascending=False)
        xgb_imp_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
        )
        xgb_imp_df.to_csv(xgb_imp_path, index=False)
        print(f"Saved XGBoost SHAP global importance to {xgb_imp_path}")

        # Sample SHAP values
        xgb_shap_sample_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
        )
        shap_sample_df = pd.DataFrame(
            shap_xgb, columns=feature_names, index=X_eval.index
        )
        shap_sample_df.to_parquet(xgb_shap_sample_path, index=True)
        print(f"Saved XGBoost SHAP sample values to {xgb_shap_sample_path}")

        # Summary plots
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_xgb,
            X_eval,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
        )
        bar_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_summary_bar_xgboost.png"
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_xgb,
            X_eval,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        beeswarm_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_shap_summary_beeswarm_xgboost.png"
        )
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=300)
        plt.close()

        print(f"Saved XGBoost SHAP summary plots to {out_dir}")
        
        # Upload XGBoost SHAP outputs
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
            if xgb_imp_path.exists():
                s3_xgb_imp = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
                if upload_file_to_s3(xgb_imp_path, s3_xgb_imp):
                    s3_outputs.append(s3_xgb_imp)
            if xgb_shap_sample_path.exists():
                s3_xgb_sample = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
                if upload_file_to_s3(xgb_shap_sample_path, s3_xgb_sample):
                    s3_outputs.append(s3_xgb_sample)
        except ImportError:
            pass

    # ------------------- CatBoost SHAP -------------------
    if cb_clf is not None:
        try:
            print("Computing SHAP values for CatBoost...")
            from catboost import Pool  # type: ignore

            # Identify categorical features (item_* features that were marked as categorical during training)
            # CatBoost requires us to specify categorical features when creating Pool
            feature_names_list = list(X_eval.columns)
            cat_feature_indices = [
                i for i, name in enumerate(feature_names_list)
                if name.startswith('item_')
            ]
            
            if cat_feature_indices:
                print(f"Marking {len(cat_feature_indices)} item_* features as categorical for CatBoost SHAP")
                pool_eval = Pool(
                    X_eval,
                    y.iloc[eval_idx],
                    cat_features=cat_feature_indices
                )
            else:
                pool_eval = Pool(X_eval, y.iloc[eval_idx])
            
            shap_cb = cb_clf.get_feature_importance(
                type="ShapValues", data=pool_eval
            )
            shap_cb = np.array(shap_cb)

            # CatBoost returns:
            # - binary/Regression: (n_samples, n_features + 1) [last col = expected value]
            # - multiclass: (n_samples, n_classes, n_features + 1)
            # We want per-feature SHAP values with the expected value stripped.
            if shap_cb.ndim == 2:
                # (n_samples, n_features + 1)
                shap_cb_feat = shap_cb[:, :-1]  # drop expected value column
            elif shap_cb.ndim == 3:
                # (n_samples, n_classes, n_features + 1) → collapse classes
                shap_cb_feat = shap_cb[:, :, :-1].mean(axis=1)
            else:
                raise ValueError(
                    f"Unexpected CatBoost SHAP array shape: {shap_cb.shape}"
                )

            shap_cb_mean_abs = np.abs(shap_cb_feat).mean(axis=0).ravel()
            shap_cb_mean = shap_cb_feat.mean(axis=0).ravel()  # Mean SHAP value (captures direction)

            cb_imp_df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": shap_cb_mean_abs,
                "mean_shap": shap_cb_mean,  # Direction: positive = increases risk, negative = decreases risk
            })
            # Filter to features with mean_abs_shap > 0 and sort by importance
            cb_imp_df = cb_imp_df[cb_imp_df['mean_abs_shap'] > 0].sort_values("mean_abs_shap", ascending=False)
            cb_imp_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
            )
            cb_imp_df.to_csv(cb_imp_path, index=False)
            print(f"Saved CatBoost SHAP global importance to {cb_imp_path}")

            cb_shap_sample_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet"
            )
            shap_cb_sample_df = pd.DataFrame(
                shap_cb_feat,
                index=X_eval.index,
                columns=feature_names,
            )
            shap_cb_sample_df.to_parquet(cb_shap_sample_path, index=True)
            print(f"Saved CatBoost SHAP sample values to {cb_shap_sample_path}")

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_cb_feat,
                X_eval,
                feature_names=feature_names,
                show=False,
                plot_type="bar",
            )
            cb_bar_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_summary_bar_catboost.png"
            )
            plt.tight_layout()
            plt.savefig(cb_bar_path, dpi=300)
            plt.close()

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_cb_feat,
                X_eval,
                feature_names=feature_names,
                show=False,
                plot_type="dot",
            )
            cb_beeswarm_path = (
                out_dir
                / f"{cohort}_{age_band_fname}_shap_summary_beeswarm_catboost.png"
            )
            plt.tight_layout()
            plt.savefig(cb_beeswarm_path, dpi=300)
            plt.close()

            print(f"Saved CatBoost SHAP summary plots to {out_dir}")
            
            # Mark CatBoost as successfully analyzed
            models_analyzed.append("catboost")

            # Upload CatBoost SHAP outputs if they exist
            try:
                from py_helpers.checkpoint_utils import upload_file_to_s3

                if cb_imp_path.exists():
                    s3_cb_imp = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
                    if upload_file_to_s3(cb_imp_path, s3_cb_imp):
                        s3_outputs.append(s3_cb_imp)
            except ImportError:
                pass
        except Exception as e:
            print(f"[ERROR] CatBoost SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Save checkpoint after all SHAP analysis completes (only if at least one model was analyzed)
    if models_analyzed:
        try:
            from py_helpers.checkpoint_utils import save_step_checkpoint

            save_step_checkpoint(
                step_name="8_shap_analysis",
                cohort=cohort,
                age_band=age_band,
                metadata={"n_background": n_background, "n_eval": n_eval, "models_analyzed": models_analyzed},
                output_paths=s3_outputs,
            )
        except ImportError:
            pass  # Checkpoint saving is optional
    
    # Return True if at least one model was analyzed
    return len(models_analyzed) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for final models for a given cohort/age_band."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    parser.add_argument(
        "--n_background",
        type=int,
        default=1000,
        help="Number of background samples for SHAP (default: 1000).",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=2000,
        help="Number of evaluation samples for SHAP (default: 2000).",
    )
    args = parser.parse_args()

    age_band_fname = args.age_band.replace("-", "_")
    out_dir = (
        PROJECT_ROOT / "8_shap_analysis" / "outputs" / args.cohort / age_band_fname
    )

    # Check for existing local outputs (idempotency - check local first)
    # SHAP generates outputs for both XGBoost and CatBoost (if available)
    expected_outputs = [
        f"{args.cohort}_{age_band_fname}_shap_global_importance_xgboost.csv",
        f"{args.cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet",
        f"{args.cohort}_{age_band_fname}_shap_summary_bar_xgboost.png",
        f"{args.cohort}_{age_band_fname}_shap_summary_beeswarm_xgboost.png",
    ]
    
    # CatBoost outputs are optional (model might not be available)
    optional_outputs = [
        f"{args.cohort}_{age_band_fname}_shap_global_importance_catboost.csv",
        f"{args.cohort}_{age_band_fname}_shap_sample_values_catboost.parquet",
        f"{args.cohort}_{age_band_fname}_shap_summary_bar_catboost.png",
        f"{args.cohort}_{age_band_fname}_shap_summary_beeswarm_catboost.png",
    ]

    all_required_exist = all((out_dir / fname).exists() for fname in expected_outputs)
    
    if all_required_exist:
        print(f"[SKIP] Step 8 outputs already exist locally for {args.cohort}/{args.age_band}")
        
        # Still try to upload to S3 if not already there (idempotent upload)
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint
            
            s3_outputs = []
            for fname in expected_outputs + optional_outputs:
                local_path = out_dir / fname
                if local_path.exists():
                    if fname.endswith('.csv'):
                        s3_path = f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    elif fname.endswith('.parquet'):
                        s3_path = f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    else:
                        continue  # Skip PNG files for S3 upload (they're large and optional)
                    
                    if upload_file_to_s3(local_path, s3_path):
                        s3_outputs.append(s3_path)
            
            # Save checkpoint if outputs uploaded
            if s3_outputs:
                save_step_checkpoint(
                    step_name="8_shap_analysis",
                    cohort=args.cohort,
                    age_band=args.age_band,
                    metadata={"n_background": args.n_background, "n_eval": args.n_eval, "models_analyzed": ["xgboost"]},
                    output_paths=s3_outputs,
                )
        except ImportError:
            pass  # S3 upload is optional
        
        return

    # Check S3 for existing outputs (idempotency - fallback if local doesn't exist)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists

        s3_output_paths = [
            f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_shap_global_importance_xgboost.csv",
            f"s3://pgxdatalake/gold/shap_analysis/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet",
        ]

        if check_step_outputs_exist(s3_output_paths) or check_step_checkpoint_exists("8_shap_analysis", args.cohort, args.age_band):
            print(f"[SKIP] Step 8 outputs already exist in S3 for {args.cohort}/{args.age_band}; downloading to local.")
            
            # Download from S3 to local
            try:
                import boto3
                s3_client = boto3.client("s3")
                S3_BUCKET = "pgxdatalake"
                
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Download XGBoost outputs
                for fname in expected_outputs:
                    s3_key = f"gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    local_path = out_dir / fname
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                        print(f"Downloaded {local_path} from S3")
                    except Exception as e:
                        print(f"Warning: Could not download {s3_key}: {e}")
                
                # Try to download CatBoost outputs (optional)
                for fname in optional_outputs:
                    s3_key = f"gold/shap_analysis/{args.cohort}/{args.age_band}/{fname}"
                    local_path = out_dir / fname
                    try:
                        s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                        print(f"Downloaded {local_path} from S3")
                    except Exception:
                        pass  # CatBoost outputs are optional
                
                print(f"[SKIP] Step 8 outputs downloaded from S3 for {args.cohort}/{args.age_band}")
                return
            except Exception as e:
                print(f"Warning: Could not download from S3: {e}. Will regenerate outputs.")
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    success = run_shap_analysis(
        cohort=args.cohort,
        age_band=args.age_band,
        n_background=args.n_background,
        n_eval=args.n_eval,
    )
    
    if not success:
        print("\n[ERROR] No models were successfully analyzed.")
        print("This step cannot complete without at least one model being analyzed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

