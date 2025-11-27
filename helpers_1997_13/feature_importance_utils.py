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

# S3 client for uploading results
try:
    from helpers_1997_13.common_imports import s3_client
except ImportError:
    import boto3
    s3_client = boto3.client('s3')


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
    logger.info("FEATURE IMPORTANCE ANALYSIS - MONTE CARLO CROSS-VALIDATION")
    logger.info("="*80)
    logger.info("Cohort: %s", cohort_name)
    logger.info("Age Band: %s", age_band)
    logger.info("Train Years: %s", ', '.join(map(str, train_years)))
    logger.info("Test Year: %d", test_year)
    logger.info("MC-CV Splits: %d", n_splits)
    logger.info("Scaling Metric: %s", scaling_metric)
    logger.info("Debug Mode: %s", "Enabled" if debug_mode else "Disabled")
    logger.info("="*80)
    
    try:
        # Load data
        logger.info("Loading cohort data...")
        check_memory_usage_r(logger, "Before Data Loading")
        
        # Try environment variable first
        local_data_path = os.getenv("LOCAL_DATA_PATH")
        
        # If not set, try common locations
        if not local_data_path:
            # Check project-relative path first (works for both Windows and Linux)
            project_root = Path(__file__).parent.parent
            project_data_path = project_root / "data" / "cohorts_F1120"
            if project_data_path.exists():
                local_data_path = str(project_data_path)
            else:
                # Fall back to EC2 path
                local_data_path = "/mnt/nvme/cohorts"
        
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
        
        # Filter out non-meaningful items and target codes
        # - event_type contains metadata values like "pharmacy" and "medical" which are not features
        # - F1120 is the target ICD code (opioid use disorder) - including it would cause data leakage
        excluded_items = {'pharmacy', 'medical', 'F1120'}  # Filter out event_type metadata and target code
        train_patient_items = train_patient_items[~train_patient_items['item'].isin(excluded_items)]
        
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
        
        # Filter out non-meaningful items and target codes (same exclusions as training)
        excluded_items = {'pharmacy', 'medical', 'F1120'}  # Filter out event_type metadata and target code
        test_patient_items = test_patient_items[~test_patient_items['item'].isin(excluded_items)]
        
        test_patient_targets = test_cohort_data[['mi_person_key', 'target']].drop_duplicates()
        
        # Get all unique items from both train and test to ensure consistent feature space
        all_items = pd.concat([train_patient_items[['item']], test_patient_items[['item']]]).drop_duplicates()
        all_item_list = all_items['item'].tolist()
        
        logger.info("Total unique items across train and test: %d", len(all_item_list))
        
        # Helper function to create feature matrices
        def create_feature_matrix(patient_items, patient_targets, all_items, is_catboost=False):
            """Create feature matrix with consistent feature space"""
            if is_catboost:
                # CatBoost format: categorical features with item names as values
                # Create a value column with the item name for pivot
                patient_items_pivot = patient_items.copy()
                patient_items_pivot['value'] = patient_items_pivot['item']
                
                feature_matrix = patient_items_pivot.pivot_table(
                    index='mi_person_key',
                    columns='item',
                    values='value',
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
                
                # Drop mi_person_key - it's not a feature, only used for joining
                if 'mi_person_key' in data.columns:
                    data = data.drop(columns=['mi_person_key'])
                
                # Convert to categorical for CatBoost
                # Replace NaN/None with empty string (CatBoost accepts empty strings as a category level)
                for col in data.columns:
                    if col.startswith('item_') and col != 'target':
                        data[col] = data[col].fillna('').astype(str)
                        # Keep as string - CatBoost will handle it as categorical when we specify cat_features
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
                
                # Drop mi_person_key - it's not a feature, only used for joining
                if 'mi_person_key' in data.columns:
                    data = data.drop(columns=['mi_person_key'])
            
            # Clean data
            data = data.dropna(subset=['target'])
            
            return data
        
        # Create feature matrices for train and test
        train_data_catboost = create_feature_matrix(train_patient_items, train_patient_targets, all_item_list, is_catboost=True)
        train_data_rf = create_feature_matrix(train_patient_items, train_patient_targets, all_item_list, is_catboost=False)
        
        test_data_catboost = create_feature_matrix(test_patient_items, test_patient_targets, all_item_list, is_catboost=True)
        test_data_rf = create_feature_matrix(test_patient_items, test_patient_targets, all_item_list, is_catboost=False)
        
        # Filter constant features globally (before creating splits)
        # Only check features that actually appear in training data
        # Features that only appear in test will be constant in train (all empty strings)
        logger.info("Filtering constant features...")
        
        # Get items that actually appear in training data
        train_items_set = set(train_patient_items['item'].unique())
        logger.info("Items in training data: %d", len(train_items_set))
        logger.info("Items in test data: %d", len(test_patient_items['item'].unique()))
        logger.info("Total items (train + test): %d", len(all_item_list))
        
        train_feature_cols = [col for col in train_data_catboost.columns if col not in ['target', 'mi_person_key']]
        constant_features = []
        
        # Debug: Check a sample of items that appear in training
        sample_train_items = list(train_items_set)[:5]
        logger.info("Sample training items: %s", sample_train_items)
        
        for col in train_feature_cols:
            # Extract item name from column (remove 'item_' prefix)
            item_name = col.replace('item_', '', 1)
            
            # Only check for constants if this item appears in training data
            # Items that only appear in test will be constant in train (all empty strings)
            if item_name in train_items_set:
                # For CatBoost categorical features, check unique values
                # A feature is constant if ALL patients have the same value
                # If some patients have the item name and some have empty strings, it's NOT constant
                nunique = train_data_catboost[col].nunique()
                
                # Debug first few training items
                if item_name in sample_train_items:
                    logger.info("Item '%s': nunique=%d, sample values: %s", 
                               item_name, nunique, train_data_catboost[col].value_counts().head(3).to_dict())
                
                # Feature is constant if it has <= 1 unique value (all same)
                if nunique <= 1:
                    constant_features.append(col)
            else:
                # Item only appears in test, will be constant in train - mark for removal
                constant_features.append(col)
        
        if constant_features:
            logger.info("Removing %d constant features (out of %d total)", len(constant_features), len(train_feature_cols))
            
            # Save constant features list for inspection
            constant_features_df = pd.DataFrame({
                'feature': constant_features,
                'item_name': [col.replace('item_', '', 1) for col in constant_features]
            })
            
            # Save locally
            os.makedirs(output_dir, exist_ok=True)
            constant_features_file = os.path.join(
                output_dir,
                f"{cohort_name}_{age_band}_constant_features.csv"
            )
            constant_features_df.to_csv(constant_features_file, index=False)
            logger.info("Saved constant features list: %s", constant_features_file)
            
            # Upload to S3 (model_metadata folder)
            s3_key_const = f"gold/feature_importance/model_metadata/{cohort_name}/{age_band}_constant_features.csv"
            if upload_csv_to_s3(constant_features_file, s3_key_const):
                logger.info("Uploaded constant features to S3: s3://pgxdatalake/%s", s3_key_const)
            else:
                logger.warning("Failed to upload constant features to S3: s3://pgxdatalake/%s", s3_key_const)
            
            # Remove constant features from datasets
            train_data_catboost = train_data_catboost.drop(columns=constant_features)
            train_data_rf = train_data_rf.drop(columns=constant_features)
            test_data_catboost = test_data_catboost.drop(columns=constant_features)
            test_data_rf = test_data_rf.drop(columns=constant_features)
        
        # Check if we have any features left
        remaining_features = [c for c in train_data_catboost.columns if c.startswith('item_')]
        if len(remaining_features) == 0:
            logger.warning("No non-constant features found. This cohort/age-band combination has insufficient signal for feature importance analysis.")
            logger.warning("Skipping analysis for %s/%s (train: %s, test: %d)", cohort_name, age_band, ', '.join(map(str, train_years)), test_year)
            return {
                'cohort': cohort_name,
                'age_band': age_band,
                'train_years': train_years,
                'test_year': test_year,
                'status': 'skipped',
                'reason': 'No non-constant features found'
            }
        
        logger.info("Feature engineering complete:")
        logger.info("  Training: %d patients, %d features", len(train_data_catboost), len(remaining_features))
        logger.info("  Test: %d patients, %d features", len(test_data_catboost), len(remaining_features))
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
            # Check if this model's results already exist (idempotency)
            # Check local file first, then S3
            local_file = os.path.join(output_dir, f"{cohort_name}_{age_band}_{method}_feature_importance.csv")
            s3_key_method = f"gold/feature_importance/{cohort_name}/{age_band}/{method}_feature_importance.csv"
            
            # Check local file first
            if os.path.exists(local_file):
                logger.info("Skipping %s: results already exist locally (%s)", method, local_file)
                try:
                    existing_result = pd.read_csv(local_file)
                    all_results[method] = existing_result
                    logger.info("Loaded existing %s results from local file: %d features", method, len(existing_result))
                    # Optionally upload to S3 if not already there
                    try:
                        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_method)
                        logger.debug("Results also exist in S3, skipping upload")
                    except Exception:
                        # Upload to S3 if not present
                        if upload_csv_to_s3(local_file, s3_key_method):
                            logger.info("Uploaded existing local results to S3: s3://%s/%s", S3_BUCKET, s3_key_method)
                        else:
                            logger.warning("Failed to upload existing local results to S3")
                    continue
                except Exception as e:
                    logger.warning("Error loading local file %s: %s. Will check S3 or re-run.", local_file, str(e))
            
            # Check S3 if local file doesn't exist
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_method)
                logger.info("Skipping %s: results already exist in S3 (s3://%s/%s)", method, S3_BUCKET, s3_key_method)
                # Load existing results from S3
                import io
                obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_method)
                existing_result = pd.read_csv(io.BytesIO(obj['Body'].read()))
                all_results[method] = existing_result
                logger.info("Loaded existing %s results from S3: %d features", method, len(existing_result))
                # Save locally for future use
                os.makedirs(output_dir, exist_ok=True)
                existing_result.to_csv(local_file, index=False)
                logger.info("Saved S3 results locally: %s", local_file)
                continue
            except Exception:
                # Results don't exist locally or in S3, proceed with running the model
                pass
            
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
            
            # Save individual results locally
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"{cohort_name}_{age_band}_{method}_feature_importance.csv"
            )
            result.to_csv(output_file, index=False)
            logger.info("Saved locally: %s", output_file)
            
            # Upload to S3
            s3_key = f"gold/feature_importance/{cohort_name}/{age_band}/{method}_feature_importance.csv"
            if upload_csv_to_s3(output_file, s3_key):
                logger.info("Uploaded to S3: s3://pgxdatalake/%s", s3_key)
            else:
                logger.warning("Failed to upload to S3: s3://pgxdatalake/%s", s3_key)
            
            check_memory_usage_r(logger, f"After MC-CV: {method}")
        
        check_memory_usage_r(logger, "After MC-CV Execution")
        
        # Aggregate results across models
        logger.info("Aggregating results...")
        aggregated = aggregate_feature_importance(all_results, scaling_metric, logger=logger)
        
        # Save aggregated results locally
        output_file = os.path.join(
            output_dir,
            f"{cohort_name}_{age_band}_aggregated_feature_importance.csv"
        )
        aggregated.to_csv(output_file, index=False)
        logger.info("Saved aggregated results locally: %s", output_file)
        
        # Upload aggregated results to S3
        s3_key_agg = f"gold/feature_importance/{cohort_name}/{age_band}/aggregated_feature_importance.csv"
        if upload_csv_to_s3(output_file, s3_key_agg):
            logger.info("Uploaded aggregated results to S3: s3://pgxdatalake/%s", s3_key_agg)
        else:
            logger.warning("Failed to upload aggregated results to S3: s3://pgxdatalake/%s", s3_key_agg)
        
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


def upload_csv_to_s3(local_file_path: str, s3_key: str, bucket: str = None) -> bool:
    """
    Upload a CSV file to S3
    
    Args:
        local_file_path: Path to local CSV file
        s3_key: S3 key (path within bucket)
        bucket: S3 bucket name (default: S3_BUCKET from constants)
        
    Returns:
        True if successful, False otherwise
    """
    if bucket is None:
        from helpers_1997_13.constants import S3_BUCKET
        bucket = S3_BUCKET
    try:
        s3_client.upload_file(local_file_path, bucket, s3_key)
        return True
    except Exception as e:
        print(f"Error uploading {local_file_path} to s3://{bucket}/{s3_key}: {e}")
        return False


def aggregate_feature_importance(all_results: Dict[str, pd.DataFrame], scaling_metric: str, logger=None) -> pd.DataFrame:
    """
    Aggregate feature importance across models with normalization and scaling by best model performance
    
    Args:
        all_results: Dictionary of results from each model (DataFrames with columns: 
                     feature, scaled_importance_mean, importance_mean, recall_mean, logloss_mean)
        scaling_metric: Metric used for scaling ('recall' or 'logloss')
        logger: Optional logger instance
        
    Returns:
        Aggregated DataFrame with normalized and scaled feature importance
        Columns: feature, importance_normalized, importance_scaled
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Find best model performance for scaling
    best_performance = 0
    best_method = None
    
    for method, result_df in all_results.items():
        if scaling_metric == 'recall':
            # Get mean recall for this model
            if 'recall_mean' in result_df.columns and len(result_df) > 0:
                mean_recall = result_df['recall_mean'].iloc[0]
                if mean_recall > best_performance:
                    best_performance = mean_recall
                    best_method = method
        elif scaling_metric == 'logloss':
            # Get mean logloss for this model (lower is better)
            if 'logloss_mean' in result_df.columns and len(result_df) > 0:
                mean_logloss = result_df['logloss_mean'].iloc[0]
                if best_performance == 0 or mean_logloss < best_performance:
                    best_performance = mean_logloss
                    best_method = method
    
    logger.info(f"Best model: {best_method} with {scaling_metric}={best_performance:.4f}")
    
    # Combine all feature importances
    combined = []
    
    for method, result_df in all_results.items():
        result_df = result_df.copy()
        result_df['method'] = method
        
        # Get model performance
        if scaling_metric == 'recall':
            model_performance = result_df['recall_mean'].iloc[0] if 'recall_mean' in result_df.columns and len(result_df) > 0 else 0
        elif scaling_metric == 'logloss':
            model_performance = result_df['logloss_mean'].iloc[0] if 'logloss_mean' in result_df.columns and len(result_df) > 0 else float('inf')
        else:
            model_performance = 1.0
        
        result_df['model_performance'] = model_performance
        # Ensure we have the required columns
        required_cols = ['feature', 'importance_mean', 'method', 'model_performance']
        if all(col in result_df.columns for col in required_cols):
            combined.append(result_df[required_cols])
    
    if not combined:
        logger.warning("No valid results to aggregate")
        return pd.DataFrame(columns=['feature', 'importance_normalized', 'importance_scaled'])
    
    combined_df = pd.concat(combined, ignore_index=True)
    
    # Normalize importance values to [0, 1] per model
    normalized_combined = []
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method].copy()
        if len(method_df) > 0:
            importance_max = method_df['importance_mean'].max()
            importance_min = method_df['importance_mean'].min()
            if importance_max > importance_min:
                method_df['importance_normalized'] = (method_df['importance_mean'] - importance_min) / (importance_max - importance_min)
            else:
                method_df['importance_normalized'] = 0.0
            normalized_combined.append(method_df)
    
    if not normalized_combined:
        logger.warning("No normalized results")
        return pd.DataFrame(columns=['feature', 'importance_normalized', 'importance_scaled'])
    
    normalized_df = pd.concat(normalized_combined, ignore_index=True)
    
    # Scale normalized importance by model performance
    if scaling_metric == 'recall':
        # Scale by recall (higher is better)
        normalized_df['importance_scaled_by_model'] = normalized_df['importance_normalized'] * normalized_df['model_performance']
    elif scaling_metric == 'logloss':
        # Scale by inverse logloss (lower is better, so invert)
        normalized_df['importance_scaled_by_model'] = normalized_df['importance_normalized'] * (1.0 / normalized_df['model_performance'].replace(0, float('inf')))
    else:
        normalized_df['importance_scaled_by_model'] = normalized_df['importance_normalized']
    
    # Aggregate by feature (sum across models)
    aggregated = normalized_df.groupby('feature').agg({
        'importance_scaled_by_model': 'sum',
        'importance_normalized': 'sum'
    }).reset_index()
    
    # Renormalize final aggregated importance to [0, 1]
    scaled_max = aggregated['importance_scaled_by_model'].max()
    scaled_min = aggregated['importance_scaled_by_model'].min()
    if scaled_max > scaled_min:
        aggregated['importance_normalized'] = (aggregated['importance_scaled_by_model'] - scaled_min) / (scaled_max - scaled_min)
    else:
        aggregated['importance_normalized'] = 0.0
    
    # Scale by best model performance
    if best_performance > 0:
        aggregated['importance_scaled'] = aggregated['importance_normalized'] * best_performance
    else:
        aggregated['importance_scaled'] = aggregated['importance_normalized']
    
    # Select final columns and sort
    aggregated = aggregated[['feature', 'importance_normalized', 'importance_scaled']]
    aggregated = aggregated.sort_values('importance_scaled', ascending=False)
    
    return aggregated

