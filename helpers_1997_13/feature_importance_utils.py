"""
Feature Importance Analysis Utilities
Main function to run feature importance analysis for a single cohort/age-band combination
"""

import os
import sys
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from helpers_1997_13.common_imports import *
from helpers_1997_13.constants import AGE_BANDS, COHORT_NAMES, EVENT_YEARS, S3_BUCKET
from helpers_1997_13.logging_utils import setup_r_logging, save_logs_to_s3_r, check_memory_usage_r
from helpers_1997_13.model_utils import calculate_recall, calculate_logloss
from helpers_1997_13.mc_cv_utils import run_mc_cv_method
from helpers_1997_13.s3_utils import check_feature_importance_results_exist, check_cohort_file_exists


def run_cohort_analysis(
    cohort_name: str,
    age_band: str,
    train_years: List[int] = [2016, 2017, 2018],
    test_year: int = 2019,
    n_splits: int = 100,
    train_prop: float = 0.8,
    n_workers: int = 1,
    scaling_metric: str = 'recall',
    model_params: Optional[Dict] = None,
    debug_mode: bool = False,
    output_dir: str = 'outputs'
) -> Dict:
    """
    Run complete feature importance analysis for a single cohort
    
    Args:
        cohort_name: Cohort name (e.g., "opioid_ed" or "non_opioid_ed")
        age_band: Age band (e.g., "25-44")
        train_years: List of years to use for training (default: [2016, 2017, 2018])
        test_year: Year to use for testing (default: 2019)
        n_splits: Number of MC-CV splits
        train_prop: Training proportion for sampling from train_years (default 0.8)
        n_workers: Number of parallel workers for MC-CV
        scaling_metric: Metric for scaling importance ('recall' or 'logloss')
        model_params: Model parameters dictionary
        debug_mode: Debug mode flag
        output_dir: Output directory for results
        
    Returns:
        Dictionary with results and status
    """
    # Setup logging (use test_year for log file naming)
    log_setup = setup_r_logging(cohort_name, age_band, test_year)
    logger = log_setup['logger']
    log_file_path = log_setup['log_file_path']
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'catboost': {
                'iterations': 100 if debug_mode else 500,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                'random_seed': 42
            },
            'random_forest': {
                'ntree': 100 if debug_mode else 500,
                'mtry': None,  # Will be set to sqrt(n_features)
                'nodesize': 1,
                'maxnodes': None,
                'random_seed': 42
            },
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100 if debug_mode else 500,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_seed': 42
            },
            'xgboost_rf': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100 if debug_mode else 500,
                'subsample': 0.8,
                'max_features': None,  # Will be set to sqrt(n_features)
                'random_seed': 42
            },
            'lightgbm': {
                'n_estimators': 100 if debug_mode else 500,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 1.0,
                'bagging_fraction': 1.0,
                'bagging_freq': 0,
                'random_seed': 42
            },
            'extratrees': {
                'n_estimators': 100 if debug_mode else 500,
                'max_features': None,  # Will be set to sqrt(n_features)
                'min_samples_leaf': 1,
                'max_depth': None,
                'random_seed': 42
            },
            'logistic_regression': {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_seed': 42
            },
            'linearsvc': {
                'penalty': 'l2',
                'C': 1.0,
                'loss': 'squared_hinge',
                'max_iter': 1000,
                'dual': True,
                'random_seed': 42
            },
            'elasticnet': {
                'C': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'random_seed': 42
            },
            'lasso': {
                'C': 1.0,
                'max_iter': 1000,
                'random_seed': 42
            }
        }
    
    logger.info("="*80)
    logger.info("🚀 FEATURE IMPORTANCE ANALYSIS - MONTE CARLO CROSS-VALIDATION")
    logger.info("="*80)
    logger.info("📊 Cohort: %s", cohort_name)
    logger.info("📊 Age Band: %s", age_band)
    logger.info("📊 Train Years: %s", ', '.join(map(str, train_years)))
    logger.info("📊 Test Year: %d", test_year)
    logger.info("📊 MC-CV Splits: %d", n_splits)
    logger.info("📊 Scaling Metric: %s", scaling_metric)
    logger.info("📊 Debug Mode: %s", "Enabled" if debug_mode else "Disabled")
    logger.info("="*80)
    
    try:
        # Load data
        logger.info("Loading cohort data...")
        check_memory_usage_r(logger, "Before Data Loading")
        
        local_data_path = os.getenv("LOCAL_DATA_PATH", "/mnt/nvme/cohorts")
        if not os.path.exists(local_data_path):
            local_data_path = os.getenv("LOCAL_DATA_PATH", "C:/Projects/pgx-analysis/data/gold/cohorts_F1120")
        
        # Load training data from multiple years (2016-2018)
        logger.info("Loading training data from years: %s", ', '.join(map(str, train_years)))
        train_data_list = []
        
        for year in train_years:
            parquet_file = os.path.join(
                local_data_path,
                f"cohort_name={cohort_name}",
                f"event_year={year}",
                f"age_band={age_band}",
                "cohort.parquet"
            )
            
            if not os.path.exists(parquet_file):
                logger.warning("Training file not found for year %d: %s", year, parquet_file)
                continue
            
            con = duckdb.connect()
            query = f"""
                SELECT
                    mi_person_key,
                    is_target_case as target,
                    drug_name,
                    primary_icd_diagnosis_code,
                    two_icd_diagnosis_code,
                    three_icd_diagnosis_code,
                    four_icd_diagnosis_code,
                    five_icd_diagnosis_code,
                    procedure_code,
                    event_type
                FROM read_parquet('{parquet_file}')
            """
            
            year_data = con.execute(query).df()
            con.close()
            train_data_list.append(year_data)
            logger.info("Loaded %d records from year %d", len(year_data), year)
        
        if len(train_data_list) == 0:
            raise FileNotFoundError(f"No training data found for years {train_years}")
        
        # Combine training data from all years
        train_cohort_data = pd.concat(train_data_list, ignore_index=True)
        logger.info("Combined training data: %d event-level records, %d unique patients",
                   len(train_cohort_data), train_cohort_data['mi_person_key'].nunique())
        
        # Load test data from 2019
        logger.info("Loading test data from year: %d", test_year)
        test_parquet_file = os.path.join(
            local_data_path,
            f"cohort_name={cohort_name}",
            f"event_year={test_year}",
            f"age_band={age_band}",
            "cohort.parquet"
        )
        
        if not os.path.exists(test_parquet_file):
            raise FileNotFoundError(f"Test file not found: {test_parquet_file}")
        
        con = duckdb.connect()
        test_query = f"""
            SELECT
                mi_person_key,
                is_target_case as target,
                drug_name,
                primary_icd_diagnosis_code,
                two_icd_diagnosis_code,
                three_icd_diagnosis_code,
                four_icd_diagnosis_code,
                five_icd_diagnosis_code,
                procedure_code,
                event_type
            FROM read_parquet('{test_parquet_file}')
        """
        
        test_cohort_data = con.execute(test_query).df()
        con.close()
        
        logger.info("Test data: %d event-level records, %d unique patients",
                   len(test_cohort_data), test_cohort_data['mi_person_key'].nunique())
        check_memory_usage_r(logger, "After Data Loading")
        
        # Feature engineering for training data
        logger.info("Creating patient-level features for training data...")
        
        train_patient_items = train_cohort_data.melt(
            id_vars=['mi_person_key'],
            value_vars=['drug_name', 'primary_icd_diagnosis_code', 'two_icd_diagnosis_code',
                       'three_icd_diagnosis_code', 'four_icd_diagnosis_code',
                       'five_icd_diagnosis_code', 'procedure_code', 'event_type'],
            var_name='feature_type',
            value_name='item'
        ).dropna(subset=['item'])[['mi_person_key', 'item']].drop_duplicates()
        
        train_patient_targets = train_cohort_data[['mi_person_key', 'target']].drop_duplicates()
        
        # Feature engineering for test data
        logger.info("Creating patient-level features for test data...")
        
        test_patient_items = test_cohort_data.melt(
            id_vars=['mi_person_key'],
            value_vars=['drug_name', 'primary_icd_diagnosis_code', 'two_icd_diagnosis_code',
                       'three_icd_diagnosis_code', 'four_icd_diagnosis_code',
                       'five_icd_diagnosis_code', 'procedure_code', 'event_type'],
            var_name='feature_type',
            value_name='item'
        ).dropna(subset=['item'])[['mi_person_key', 'item']].drop_duplicates()
        
        test_patient_targets = test_cohort_data[['mi_person_key', 'target']].drop_duplicates()
        
        # Get all unique items from both train and test to ensure consistent feature space
        all_items = pd.concat([train_patient_items[['item']], test_patient_items[['item']]]).drop_duplicates()
        all_item_list = all_items['item'].tolist()
        
        logger.info("Total unique items across train and test: %d", len(all_item_list))
        
        # Helper function to create feature matrices
        def create_feature_matrix(patient_items, patient_targets, all_items, is_catboost=False):
            """Create feature matrix with consistent feature space"""
            # Create pivot table with all items to ensure consistent columns
            if is_catboost:
                # CatBoost format: categorical features
                feature_matrix = patient_items.pivot_table(
                    index='mi_person_key',
                    columns='item',
                    values='item',
                    aggfunc='first',
                    fill_value=None
                ).reset_index()
                
                # Add missing items as None columns
                for item in all_item_list:
                    if item not in feature_matrix.columns:
                        feature_matrix[item] = None
                
                # Add prefix
                feature_cols = [col for col in feature_matrix.columns if col != 'mi_person_key']
                feature_matrix = feature_matrix.rename(
                    columns={col: f'item_{col}' for col in feature_cols}
                )
                
                # Join with targets
                data = feature_matrix.merge(patient_targets, on='mi_person_key', how='left')
                
                # Convert to categorical for CatBoost
                for col in data.columns:
                    if col.startswith('item_') and col != 'target':
                        data[col] = data[col].astype('category')
            else:
                # Random Forest/XGBoost format: binary features
                patient_items_with_value = patient_items.copy()
                patient_items_with_value['value'] = 1
                
                feature_matrix = patient_items_with_value.pivot_table(
                    index='mi_person_key',
                    columns='item',
                    values='value',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                
                # Add missing items as 0 columns
                for item in all_item_list:
                    if item not in feature_matrix.columns:
                        feature_matrix[item] = 0
                
                # Add prefix
                feature_cols = [col for col in feature_matrix.columns if col != 'mi_person_key']
                feature_matrix = feature_matrix.rename(
                    columns={col: f'item_{col}' for col in feature_cols}
                )
                
                # Join with targets
                data = feature_matrix.merge(patient_targets, on='mi_person_key', how='left')
            
            # Clean data
            data = data.dropna(subset=['target'])
            return data
        
        # Create feature matrices for train and test
        train_data_catboost = create_feature_matrix(train_patient_items, train_patient_targets, all_item_list, is_catboost=True)
        train_data_rf = create_feature_matrix(train_patient_items, train_patient_targets, all_item_list, is_catboost=False)
        
        test_data_catboost = create_feature_matrix(test_patient_items, test_patient_targets, all_item_list, is_catboost=True)
        test_data_rf = create_feature_matrix(test_patient_items, test_patient_targets, all_item_list, is_catboost=False)
        
        logger.info("Feature engineering complete:")
        logger.info("  Training: %d patients, %d features", len(train_data_catboost), len([c for c in train_data_catboost.columns if c.startswith('item_')]))
        logger.info("  Test: %d patients, %d features", len(test_data_catboost), len([c for c in test_data_catboost.columns if c.startswith('item_')]))
        check_memory_usage_r(logger, "After Feature Engineering")
        
        # Create MC-CV splits
        # Each split samples from training data (2016-2018) and tests on 2019
        logger.info("Creating MC-CV splits (sampling from train years, testing on %d)...", test_year)
        check_memory_usage_r(logger, "Before MC-CV Split Creation")
        
        # Use StratifiedShuffleSplit to sample from training data
        # Each split uses a different random subset of training data
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1-train_prop, random_state=42)
        split_indices = []
        
        # Get test indices (all test data)
        n_test = len(test_data_catboost)
        test_indices = np.arange(n_test)
        
        # Create splits: each split samples from training data
        for train_subset_idx, _ in sss.split(train_data_catboost.drop(columns=['target']), train_data_catboost['target']):
            split_indices.append({
                'train_idx': train_subset_idx,  # Subset of training data
                'test_idx': test_indices  # All test data (2019)
            })
        
        logger.info("Created %d MC-CV splits (train: sampled from %s, test: %d)", 
                   len(split_indices), ', '.join(map(str, train_years)), test_year)
        check_memory_usage_r(logger, "After MC-CV Split Creation")
        
        # Run MC-CV for each method
        logger.info("Running MC-CV analysis...")
        methods = [
            'catboost', 'random_forest', 'xgboost', 'xgboost_rf',
            'lightgbm', 'extratrees', 'logistic_regression',
            'linearsvc', 'elasticnet', 'lasso'
        ]
        all_results = {}
        
        check_memory_usage_r(logger, "Before MC-CV Execution")
        
        for method in methods:
            logger.info("Running MC-CV for %s...", method)
            check_memory_usage_r(logger, f"Before MC-CV: {method}")
            
            if method == 'catboost':
                result = run_mc_cv_method(
                    train_data_catboost,
                    method,
                    split_indices,
                    model_params,
                    scaling_metric,
                    n_jobs=n_workers,
                    data_catboost=train_data_catboost,
                    test_data=test_data_catboost,
                    test_data_catboost=test_data_catboost
                )
            else:
                result = run_mc_cv_method(
                    train_data_rf,
                    method,
                    split_indices,
                    model_params,
                    scaling_metric,
                    n_jobs=n_workers,
                    test_data=test_data_rf
                )
            
            all_results[method] = result
            
            # Save individual results
            os.makedirs(output_dir, exist_ok=True)
            train_years_str = '_'.join(map(str, train_years))
            output_file = os.path.join(
                output_dir,
                f"{cohort_name}_{age_band}_train{train_years_str}_test{test_year}_{method}_feature_importance.csv"
            )
            result.to_csv(output_file, index=False)
            logger.info("Saved: %s", output_file)
            
            check_memory_usage_r(logger, f"After MC-CV: {method}")
        
        check_memory_usage_r(logger, "After MC-CV Execution")
        
        # Aggregate results across models
        logger.info("Aggregating results...")
        aggregated = aggregate_feature_importance(all_results, scaling_metric)
        
        # Save aggregated results
        train_years_str = '_'.join(map(str, train_years))
        output_file = os.path.join(
            output_dir,
            f"{cohort_name}_{age_band}_train{train_years_str}_test{test_year}_feature_importance_aggregated.csv"
        )
        aggregated.to_csv(output_file, index=False)
        logger.info("Saved aggregated results: %s", output_file)
        
        # Save logs to S3
        logger.info("Saving logs to S3...")
        # Close file handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        save_logs_to_s3_r(log_file_path, cohort_name, age_band, test_year, logger)
        
        return {
            'cohort': cohort_name,
            'age_band': age_band,
            'train_years': train_years,
            'test_year': test_year,
            'status': 'success',
            'aggregated': aggregated,
            'output_file': output_file
        }
        
    except Exception as e:
        logger.error("Analysis failed: %s", str(e))
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        # Close file handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        save_logs_to_s3_r(log_file_path, cohort_name, age_band, test_year, logger)
        
        return {
            'cohort': cohort_name,
            'age_band': age_band,
            'train_years': train_years,
            'test_year': test_year,
            'status': 'error',
            'error': str(e)
        }


def aggregate_feature_importance(all_results: Dict[str, pd.DataFrame], scaling_metric: str) -> pd.DataFrame:
    """
    Aggregate feature importance across models
    
    Args:
        all_results: Dictionary of results from each model
        scaling_metric: Metric used for scaling
        
    Returns:
        Aggregated DataFrame with combined feature importance
    """
    # Combine all feature importances
    combined = []
    
    for method, result_df in all_results.items():
        result_df['method'] = method
        combined.append(result_df[['feature', 'scaled_importance_mean', 'method']])
    
    combined_df = pd.concat(combined, ignore_index=True)
    
    # Aggregate by feature (sum across models)
    aggregated = combined_df.groupby('feature').agg({
        'scaled_importance_mean': 'sum'
    }).reset_index()
    
    aggregated.columns = ['feature', 'aggregated_importance']
    aggregated = aggregated.sort_values('aggregated_importance', ascending=False)
    
    return aggregated

