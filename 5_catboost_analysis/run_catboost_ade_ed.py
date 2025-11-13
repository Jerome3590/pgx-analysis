#!/usr/bin/env python3
"""
ADE CatBoost Model Script
Runs ADE (Adverse Drug Event) CatBoost models with temporal filtering, Optuna optimization, and proper artifact saving.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import duckdb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
import optuna
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import existing utilities
from helpers_1997_13.common_imports import (
    s3_client, 
    S3_BUCKET, 
    get_logger, 
    ClientError
)

from helpers_1997_13.constants import (
    S3_BUCKET,
    METRICS_BUCKET,
    NOTIFICATION_EMAIL
)

from helpers_1997_13.aws_utils import (
    notify_error,
    send_email
)

from helpers_1997_13.s3_utils import (
    upload_file_to_s3,
    download_file_from_s3,
    get_output_paths,
    save_to_s3_json,
    save_to_s3_parquet
)

from helpers_1997_13.model_utils import (
    save_model_artifacts,
    load_model_artifacts
)


def setup_logging(age_band, event_year):
    """Setup logging for the model run"""
    return get_logger(f"ade_catboost_{age_band}_{event_year}", "all", "all")


def load_cohort_data(age_band, event_year, logger):
    """Load cohort data from S3"""
    logger.info(f"Loading ADE cohort data for {age_band}/{event_year}")
    
    # Enable S3/httpfs support in DuckDB
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CALL load_aws_credentials();")
    
    # S3 path for input cohort
    cohort_s3_path = f"s3://pgxdatalake/catboost/input_datasets/{age_band}/"
    
    # Read the parquet files
    query = f"SELECT * FROM read_parquet('{cohort_s3_path}/*.parquet')"
    df = con.execute(query).df()
    
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def apply_temporal_filtering(df, logger):
    """Apply temporal filtering to drop events after first target event per person"""
    logger.info("Applying temporal filtering to prevent data leakage")
    
    # Sort by person_id and event_date
    df = df.sort_values(['person_id', 'event_date'])
    
    # Find first target event per person
    df['first_target_event_idx'] = df.groupby('person_id')['target'].transform('idxmax')
    
    # Filter to keep only events up to and including first target event
    df_filtered = df[df.index <= df['first_target_event_idx']].copy()
    
    # Drop the helper column
    df_filtered = df_filtered.drop(columns=['first_target_event_idx'])
    
    logger.info(f"Temporal filtering: {df.shape[0]} -> {df_filtered.shape[0]} rows")
    return df_filtered


def prepare_features(df, logger):
    """Prepare features for modeling"""
    logger.info("Preparing features for ADE modeling")
    
    # Drop high-cardinality and not-needed columns
    not_needed_high_card = [
        'billing_provider_name', 'billing_provider_npi_med', 'service_provider_name_med',
        'billing_provider_tin_med', 'service_provider_npi_med', 'billing_provider_npi_pharm',
        'service_provider_tin_med', 'service_provider_name_pharm', 'service_provider_npi_pharm',
        'billing_provider_zip_med', 'row_number'
    ]
    
    lagging_indicators = [
        'primary_icd_diagnosis_code_group', 'two_icd_diagnosis_code_group',                      
        'three_icd_diagnosis_code_group', 'four_icd_diagnosis_code_group', 'five_icd_diagnosis_code_group',
        'six_icd_diagnosis_code_group', 'seven_icd_diagnosis_code_group', 'eight_icd_diagnosis_code_group',
        'nine_icd_diagnosis_code_group', 'ten_icd_diagnosis_code_group', 'place_of_service'
    ]
    
    not_needed = [
        'ade_window', 'Event_med', 'age_band', 'hispanic_indicator_pharm', 'member_state_enroll_med',
        'billing_provider_taxonomy_pharm', 'service_provider_taxonomy_pharm', 
        'billing_provider_taxonomy_med', 'service_provider_taxonomy_med', 'service_provider_msa_pharm', 
        'service_provider_msa_med', 'total_utilization_med', 'total_paid', 'total_allowed_pharm', 
        'total_utilization_pharm', 'total_rx_paid', 'total_rx_days_supply', 'member_age_dos_med', 
        'gpi', 'strength', 'dosage_form', 'bill_type_id', 'bill_type_class', 
        'service_provider_specialty_med', 'hispanic_indicator_med', 'cchg_label', 
        'member_age_band_dos_med', 'cchg_grouping', 'hcg_setting_med', 'hcg_line_med',
        'hcg_detail_med', 'bill_type_description', 'member_age_dos_pharm', 'member_age_band_dos_pharm',
        'member_gender_pharm', 'member_race_pharm', 'hcg_setting_pharm', 'hcg_line_pharm',
        'hcg_detail_pharm', 'payer_type_pharm', 'revenue_code', 'billing_provider_specialty_med',
        'billing_provider_county_med', 'billing_provider_state_med', 'billing_provider_msa_med',
        'service_provider_county_med', 'service_provider_state_med', 'billing_provider_specialty_pharm',
        'billing_provider_zip_pharm', 'billing_provider_county_pharm', 'billing_provider_state_pharm',
        'billing_provider_msa_pharm', 'billing_provider_tin_pharm', 'service_provider_specialty_pharm',
        'service_provider_county_pharm', 'service_provider_state_pharm', 'service_provider_tin_pharm',
        'payer_type_med', 'member_county_dos_med', 'member_zip_code_dos_pharm', 'member_county_dos_pharm',
        'member_state_enroll_pharm', 'total_allowed_med', 'admit_type', 'DayOfWeekString', 'Weekend',
        'WeekOfMonth', 'NearUSHoliday', 'ndc'
    ]
    
    drop_cols = set(not_needed_high_card + not_needed + lagging_indicators)
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Separate target and features
    target_col = 'target'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Get categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from feature columns
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    logger.info(f"Features prepared: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")
    return df, categorical_cols, numerical_cols, target_col


def create_optuna_objective(X_train, y_train, X_val, y_val, categorical_cols):
    """Create Optuna objective function for hyperparameter optimization"""
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'cat_features': categorical_cols
        }
        
        # Train model
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        return auc_score
    
    return objective


def train_model(X_train, y_train, X_val, y_val, categorical_cols, logger):
    """Train CatBoost model with Optuna optimization"""
    logger.info("Starting Optuna hyperparameter optimization")
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    objective = create_optuna_objective(X_train, y_train, X_val, y_val, categorical_cols)
    
    # Optimize
    study.optimize(objective, n_trials=50, timeout=3600)  # 1 hour timeout
    
    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    best_params = study.best_trial.params
    best_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'cat_features': categorical_cols
    })
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    return final_model, study.best_trial.params


def evaluate_model(model, X_test, y_test, logger):
    """Evaluate model performance"""
    logger.info("Evaluating model performance")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_pred_proba)
    }
    
    logger.info(f"Test Metrics: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
    return metrics


def generate_shap_explanations(model, X_test, categorical_cols, logger):
    """Generate SHAP explanations"""
    logger.info("Generating SHAP explanations")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Generate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    return shap_values, feature_importance, explainer


def save_model_artifacts(model, metrics, feature_importance, best_params, 
                        age_band, event_year, logger):
    """Save model artifacts locally and to S3 using standardized paths"""
    logger.info("Saving model artifacts")
    
    # Create local output directory
    output_dir = f"catboost_models/ed_non_opioid/{age_band.replace('-', '_')}_{event_year}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model files locally
    model_path = os.path.join(output_dir, "model.cbm")
    model.save_model(model_path)
    
    # Save model info locally
    model_info = {
        'model_type': 'ade_ed',
        'age_band': age_band,
        'event_year': event_year,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'best_params': best_params,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    model_info_path = os.path.join(output_dir, "model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save feature importance locally
    feature_importance_path = os.path.join(output_dir, "feature_importance.csv")
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Get standardized S3 paths using get_output_paths
    paths = get_output_paths(
        cohort_name="ed_non_opioid",
        age_band=age_band,
        event_year=event_year
    )
    
    try:
        # Save model artifacts to S3 using standardized paths
        logger.info("Uploading model artifacts to S3...")
        
        # Save model info JSON
        save_to_s3_json(
            data=model_info,
            s3_path=paths["model_info_json"],
            logger=logger
        )
        
        # Save model metrics JSON
        save_to_s3_json(
            data=metrics,
            s3_path=paths["model_metrics_json"],
            logger=logger
        )
        
        # Save feature importance as parquet
        save_to_s3_parquet(
            df=feature_importance,
            s3_path=paths["shap_values_parquet"],
            logger=logger
        )
        
        # Upload the actual model file (not in get_output_paths, so use custom path)
        s3_model_key = f"catboost_models/ed_non_opioid/{age_band.replace('-', '_')}_{event_year}/model.cbm"
        s3_model_path = f"s3://{S3_BUCKET}/{s3_model_key}"
        
        upload_file_to_s3(model_path, S3_BUCKET, s3_model_key)
        
        logger.info(f"✓ Model artifacts uploaded to S3:")
        logger.info(f"  Model info: {paths['model_info_json']}")
        logger.info(f"  Model metrics: {paths['model_metrics_json']}")
        logger.info(f"  Feature importance: {paths['shap_values_parquet']}")
        logger.info(f"  Model file: {s3_model_path}")
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        # Continue with local saving even if S3 fails
    
    return output_dir


def run_catboost_ade_ed(age_band, event_year):
    """Main function to run ADE CatBoost model"""
    try:
        # Setup logging
        logger = setup_logging(age_band, event_year)
        logger.info("="*80)
        logger.info(f"ADE CATBOOST MODEL: {age_band}/{event_year}")
        logger.info("="*80)
        
        # Load data
        df = load_cohort_data(age_band, event_year, logger)
        
        # Apply temporal filtering
        df = apply_temporal_filtering(df, logger)
        
        # Prepare features
        df, categorical_cols, numerical_cols, target_col = prepare_features(df, logger)
        
        # Split data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        
        # Train model
        model, best_params = train_model(X_train, y_train, X_val, y_val, categorical_cols, logger)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, logger)
        
        # Generate SHAP explanations
        shap_values, feature_importance, explainer = generate_shap_explanations(
            model, X_test, categorical_cols, logger
        )
        
        # Save artifacts
        output_dir = save_model_artifacts(
            model, metrics, feature_importance, best_params, age_band, event_year, logger
        )
        
        logger.info("="*80)
        logger.info("ADE CATBOOST MODEL COMPLETE")
        logger.info("="*80)
        
        return {
            'status': 'success',
            'output_dir': output_dir,
            'metrics': metrics,
            'model': model
        }
        
    except Exception as e:
        error_msg = f"ADE CatBoost model failed for {age_band}/{event_year}: {str(e)}"
        logger.error(error_msg)
        notify_error("ade_catboost_model", error_msg, logger)
        raise


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Run ADE CatBoost model')
    parser.add_argument('--age-band', required=True, help='Age band to process')
    parser.add_argument('--event-year', type=int, required=True, help='Event year to process')
    
    args = parser.parse_args()
    
    result = run_catboost_ade_ed(args.age_band, args.event_year)
    print(f"ADE model completed: {result['status']}")


if __name__ == "__main__":
    main() 