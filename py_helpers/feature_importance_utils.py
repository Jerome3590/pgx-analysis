"""
Feature Importance Analysis Utilities
Main function to run feature importance analysis for a single cohort/age-band combination
"""

import os
import sys
import platform
import gc
import shutil
import pandas as pd
import numpy as np
import duckdb
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_helpers.common_imports import *
from py_helpers.constants import AGE_BANDS, COHORT_NAMES, EVENT_YEARS, S3_BUCKET
from py_helpers.logging_utils import setup_r_logging, save_logs_to_s3_r, check_memory_usage_r
from py_helpers.model_utils import calculate_recall, calculate_logloss
from py_helpers.mc_cv_utils import run_mc_cv_method
from py_helpers.feature_importance_model_utils import (
    train_xgboost,
    get_importance_xgboost,
    get_permutation_importance,
    predict_xgboost,
    predict_proba_xgboost,
    train_catboost,
    predict_catboost,
    predict_proba_catboost,
)
from py_helpers.s3_utils import check_feature_importance_results_exist, check_cohort_file_exists
from py_helpers.aws_utils import send_status_email_ses

# S3 client for uploading results
try:
    from py_helpers.common_imports import s3_client
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
        # Primary model ensemble for feature importance and final evaluation:
        # 1) CatBoost (categorical boosting)
        # 2) XGBoost (gradient-boosted trees)
        # 3) XGBoost RF mode (random-forest style trees)
        model_params = {
            'catboost': {
                'iterations': 100 if debug_mode else 500,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                'random_seed': 42,
            },
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100 if debug_mode else 500,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_seed': 42,
            },
            'xgboost_rf': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100 if debug_mode else 500,
                'subsample': 0.8,
                'max_features': None,  # Will be set to sqrt(n_features)
                'random_seed': 42,
            },
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
        # Normalize age-band for local filenames (convert hyphens to underscores)
        age_band_fname = age_band.replace('-', '_') if isinstance(age_band, str) else str(age_band)
        
        # Optional override: allow forcing a full re-run even if aggregated
        # results already exist locally or in S3.
        force_rerun = os.getenv("PGX_FORCE_RERUN", "0") == "1"
        # Optional mode: recompute rare-variant scan and aggregation while
        # reusing existing per-model MC-CV results when available.
        rare_only = os.getenv("PGX_RARE_ONLY", "0") == "1"
        if force_rerun:
            logger.info(
                "PGX_FORCE_RERUN=1 detected; ignoring existing aggregated results "
                "locally and in S3 and recomputing from scratch."
            )

        # ------------------------------------------------------------------
        # Fast idempotency check: if aggregated results already exist,
        # skip ALL heavy work (data loading, feature matrix build, MC-CV),
        # unless PGX_FORCE_RERUN is set.
        #
        # In rare-only mode (PGX_RARE_ONLY=1), we still want to be able to
        # skip entirely when *both* rare-variant outputs and the aggregated
        # CSV already exist, so we add an explicit rare-only shortcut below.
        # ------------------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        aggregated_local = os.path.join(
            output_dir,
            f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv",
        )
        s3_key_agg = (
            f"gold/feature_importance/{cohort_name}/{age_band}/"
            f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
        )

        # Rare-only shortcut: if the main aggregated CSV and both rare-variant
        # per-model CSVs already exist locally, we can skip the entire run.
        if rare_only and not force_rerun and os.path.exists(aggregated_local):
            rare_xgb_local = os.path.join(
                output_dir,
                f"{cohort_name}_{age_band_fname}_xgboost_rare_feature_importance.csv",
            )
            rare_cat_local = os.path.join(
                output_dir,
                f"{cohort_name}_{age_band_fname}_catboost_rare_feature_importance.csv",
            )
            if os.path.exists(rare_xgb_local) and os.path.exists(rare_cat_local):
                logger.info(
                    "PGX_RARE_ONLY=1 and rare-variant outputs already exist locally "
                    "(%s, %s) along with aggregated CSV (%s); skipping rare-only rerun.",
                    rare_xgb_local,
                    rare_cat_local,
                    aggregated_local,
                )
                try:
                    agg_df = pd.read_csv(aggregated_local)
                    n_features = int(agg_df.get("feature", agg_df.iloc[:, 0]).nunique())
                except Exception:
                    n_features = None

                return {
                    "cohort": cohort_name,
                    "age_band": age_band,
                    "train_years": train_years,
                    "test_year": test_year,
                    "status": "skipped",
                    "reason": "Rare-only mode: rare-variant and aggregated results already exist locally",
                    "output_file": aggregated_local,
                    "n_features": n_features,
                }

        if os.path.exists(aggregated_local) and not force_rerun and not rare_only:
            logger.info(
                "Aggregated feature-importance already exists locally (%s); "
                "skipping full feature engineering and MC-CV.",
                aggregated_local,
            )
            try:
                agg_df = pd.read_csv(aggregated_local)
                n_features = int(agg_df["feature"].nunique()) if "feature" in agg_df.columns else len(agg_df)
            except Exception:
                n_features = None

            return {
                "cohort": cohort_name,
                "age_band": age_band,
                "train_years": train_years,
                "test_year": test_year,
                "status": "skipped",
                "reason": "Aggregated feature-importance already exists locally",
                "output_file": aggregated_local,
                "n_features": n_features,
            }

        # If not present locally, and not forcing, check S3 and download if available.
        if not force_rerun and not rare_only:
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_agg)
                logger.info(
                    "Aggregated feature-importance exists in S3 (s3://%s/%s); "
                    "downloading instead of recomputing.",
                    S3_BUCKET,
                    s3_key_agg,
                )
                import io

                obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_agg)
                agg_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
                agg_df.to_csv(aggregated_local, index=False)
                logger.info("Saved aggregated feature-importance locally: %s", aggregated_local)
                n_features = int(agg_df["feature"].nunique()) if "feature" in agg_df.columns else len(agg_df)

                return {
                    "cohort": cohort_name,
                    "age_band": age_band,
                    "train_years": train_years,
                    "test_year": test_year,
                    "status": "skipped",
                    "reason": "Aggregated feature-importance already exists in S3",
                    "output_file": aggregated_local,
                    "n_features": n_features,
                }
            except Exception:
                # No aggregated results found locally or in S3 → proceed as normal.
                pass

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
                    six_icd_diagnosis_code,
                    seven_icd_diagnosis_code,
                    eight_icd_diagnosis_code,
                    nine_icd_diagnosis_code,
                    ten_icd_diagnosis_code,
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
                six_icd_diagnosis_code,
                seven_icd_diagnosis_code,
                eight_icd_diagnosis_code,
                nine_icd_diagnosis_code,
                ten_icd_diagnosis_code,
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

        # Base feature palette for opioid ED and general feature-importance runs:
        # medication claims events + diagnosis codes + procedures + event_type.
        feature_cols = [
            'drug_name',
            'primary_icd_diagnosis_code', 'two_icd_diagnosis_code',
            'three_icd_diagnosis_code', 'four_icd_diagnosis_code',
            'five_icd_diagnosis_code', 'six_icd_diagnosis_code',
            'seven_icd_diagnosis_code', 'eight_icd_diagnosis_code',
            'nine_icd_diagnosis_code', 'ten_icd_diagnosis_code',
            'procedure_code',
            'event_type',
        ]

        # For the polypharmacy cohort (`non_opioid_ed`) in older adults (65+),
        # we intentionally restrict the feature space to medication claims events only.
        # This focuses the importance rankings on:
        #   - individual drugs,
        #   - combinations of drugs at the patient level,
        # while keeping ICD/CPT context out of this specific screen.
        try:
            age_band_str = str(age_band)
            low_age = int(age_band_str.split("-")[0])
        except Exception:
            low_age = None

        if cohort_name == "non_opioid_ed" and low_age is not None and low_age >= 65:
            feature_cols = ['drug_name']

        def _build_patient_items(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            """
            Memory-aware melt into (mi_person_key, item) by processing feature
            columns in batches, instead of all at once.
            """
            melt_batch_size = int(os.getenv("PGX_MELT_COLS_PER_BATCH", "4"))
            chunks: List[pd.DataFrame] = []

            for i in range(0, len(cols), melt_batch_size):
                sub_cols = cols[i : i + melt_batch_size]
                logger.info(
                    "Melt batch %d-%d (of %d columns) for %s",
                    i,
                    min(i + melt_batch_size, len(cols)),
                    len(cols),
                    "train" if df is train_cohort_data else "test",
                )
                melted = (
                    df.melt(
                        id_vars=["mi_person_key"],
                        value_vars=sub_cols,
                        var_name="feature_type",
                        value_name="item",
                    )
                    .dropna(subset=["item"])[["mi_person_key", "item"]]
                    .drop_duplicates()
                )
                chunks.append(melted)

            if not chunks:
                return df[["mi_person_key"]].head(0).assign(
                    item=pd.Series(dtype=df[cols[0]].dtype)
                )

            combined = pd.concat(chunks, ignore_index=True)
            combined = combined.drop_duplicates()
            return combined

        train_patient_items = _build_patient_items(train_cohort_data, feature_cols)
        
        # Filter out non-meaningful items and target codes
        # - event_type contains metadata values like "pharmacy" and "medical" which are not features
        # - F1120 is the target ICD code (opioid use disorder) - including it would cause data leakage
        excluded_items = {'pharmacy', 'medical', 'F1120'}  # Filter out event_type metadata and target code
        train_patient_items = train_patient_items[~train_patient_items['item'].isin(excluded_items)]
        
        train_patient_targets = train_cohort_data[['mi_person_key', 'target']].drop_duplicates()
        
        # Feature engineering for test data
        logger.info("Creating patient-level features for test data...")

        test_patient_items = _build_patient_items(test_cohort_data, feature_cols)
        
        # Filter out non-meaningful items and target codes (same exclusions as training)
        excluded_items = {'pharmacy', 'medical', 'F1120'}  # Filter out event_type metadata and target code
        test_patient_items = test_patient_items[~test_patient_items['item'].isin(excluded_items)]
        
        test_patient_targets = test_cohort_data[['mi_person_key', 'target']].drop_duplicates()
        
        # Get all unique items from both train and test to ensure consistent feature space
        all_items = pd.concat([train_patient_items[['item']], test_patient_items[['item']]]).drop_duplicates()

        # ------------------------------------------------------------------
        # Dimensionality reduction: prune extremely rare items *before*
        # building any feature matrices. This keeps memory usage in check
        # while preserving the frequent signals we care about for screening.
        # ------------------------------------------------------------------
        # Count how many unique patients have each item in the TRAIN data.
        item_patient_counts = (
            train_patient_items
            .groupby('item')['mi_person_key']
            .nunique()
        )
        # Optional debug summary so we can see how heavy-tailed the item distribution is.
        try:
            n_items_ge_1 = int((item_patient_counts >= 1).sum())
            n_items_ge_5 = int((item_patient_counts >= 5).sum())
            n_items_ge_10 = int((item_patient_counts >= 10).sum())
            n_items_ge_25 = int((item_patient_counts >= 25).sum())
            logger.info(
                "Item frequency summary (train only): "
                ">=1 patients: %d, >=5: %d, >=10: %d, >=25: %d",
                n_items_ge_1, n_items_ge_5, n_items_ge_10, n_items_ge_25,
            )
        except Exception:
            # Best-effort debug; don't fail pipeline if this summary breaks.
            pass
        # Tunable threshold: minimum number of patients required for an item
        # (drug / ICD / CPT / event token) to be kept as a feature. For pure
        # screening, we can safely drop ultra-rare items.
        MIN_PATIENTS_PER_ITEM = 25
        frequent_items = item_patient_counts[item_patient_counts >= MIN_PATIENTS_PER_ITEM].index

        n_items_before = len(all_items)
        all_items = all_items[all_items['item'].isin(frequent_items)]
        all_item_list = all_items['item'].tolist()

        logger.info(
            "Total unique items across train and test: %d (after pruning rare items; min patients/item=%d, was %d)",
            len(all_item_list),
            MIN_PATIENTS_PER_ITEM,
            n_items_before,
        )
        
        # Simple helper to explicitly free large intermediates and trigger GC
        def _cleanup_memory(label: str, *objs) -> None:
            for obj in objs:
                try:
                    del obj
                except Exception:
                    # Best-effort cleanup; ignore if already deleted/out of scope
                    pass
            gc.collect()
            try:
                check_memory_usage_r(logger, f"After memory cleanup: {label}")
            except Exception:
                # Don't let memory logging failures break the pipeline
                pass

        # Helper function to create feature matrices
        def create_feature_matrix(patient_items, patient_targets, all_items, is_catboost=False):
            """Create feature matrix with consistent feature space"""
            if is_catboost:
                # CatBoost format: categorical features with item names as values
                # Optimized approach: use groupby instead of pivot_table to reduce memory
                # First, get unique patients
                unique_patients = patient_targets['mi_person_key'].unique()
                n_patients = len(unique_patients)
                n_features = len(all_item_list)

                logger.info("Building CatBoost feature matrix: %d patients × %d features", n_patients, n_features)

                # Create base DataFrame with all patients and empty string defaults
                # Use more memory-efficient approach: build column by column
                data_dict = {'mi_person_key': unique_patients}

                # Group by patient and item, take first value (in case of duplicates)
                # Use drop_duplicates instead to avoid the reset_index issue
                patient_item_map = patient_items[['mi_person_key', 'item']].drop_duplicates()

                # Build columns more efficiently using groupby
                # Parallelize column creation for better performance on multi-core systems
                def build_feature_columns_batch(items_batch):
                    """Build multiple feature columns in a batch to reduce overhead"""
                    batch_results = []
                    for item in items_batch:
                        # Get patient keys that have this item
                        item_patients = patient_item_map[patient_item_map['item'] == item]['mi_person_key'].values

                        # Create full series with all patients, filling missing with empty string
                        full_series = pd.Series('', index=unique_patients, dtype='string')
                        if len(item_patients) > 0:
                            # Set item name for patients that have this item
                            full_series.loc[item_patients] = item
                        batch_results.append((f'item_{item}', full_series.values))
                    return batch_results

                # Use parallel processing with batching to reduce overhead.
                # OS-specific defaults:
                #   - On Windows: default to 1 worker and threading backend to avoid
                #     large pickled payloads (MemoryError in joblib/loky).
                #   - On Linux/EC2: use process-based parallelism with up to 16 workers.
                is_windows = platform.system().lower().startswith("win")

                env_workers = os.getenv("PGX_CATBOOST_FEATURE_WORKERS")
                if env_workers and env_workers.isdigit():
                    n_workers_matrix = max(1, int(env_workers))
                else:
                    if is_windows:
                        n_workers_matrix = 1
                    else:
                        n_workers_matrix = min(
                            16, max(1, multiprocessing.cpu_count() - 2)
                        )

                batch_size = max(
                    1, len(all_item_list) // (n_workers_matrix * 4) or 1
                )  # ~4 batches per worker
                batches = [
                    all_item_list[i : i + batch_size]
                    for i in range(0, len(all_item_list), batch_size)
                ]

                logger.info(
                    "Feature matrix parallel workers: %d (CPU count: %d), batch size: %d, batches: %d",
                    n_workers_matrix,
                    multiprocessing.cpu_count(),
                    batch_size,
                    len(batches),
                )

                # Set environment variables to limit threading in joblib workers
                # This prevents each worker from spawning multiple threads (pandas/numpy)
                original_env = {}
                threading_vars = [
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ]
                for var in threading_vars:
                    original_env[var] = os.environ.get(var)
                    os.environ[var] = "1"  # Force single-threaded in workers

                try:
                    backend = "threading" if is_windows else "loky"
                    batch_results = Parallel(
                        n_jobs=n_workers_matrix,
                        verbose=10,
                        backend=backend,
                    )(
                        delayed(build_feature_columns_batch)(batch)
                        for batch in batches
                    )
                finally:
                    # Restore original environment
                    for var, value in original_env.items():
                        if value is None:
                            os.environ.pop(var, None)
                        else:
                            os.environ[var] = value
                
                # Flatten batch results
                column_results = []
                for batch_result in batch_results:
                    column_results.extend(batch_result)

                # Add columns to dictionary
                for col_name, col_values in column_results:
                    data_dict[col_name] = col_values
                
                # Create DataFrame from dict (more memory efficient than pivot_table)
                feature_matrix = pd.DataFrame(data_dict)
                
                # Join with targets
                data = feature_matrix.merge(patient_targets, on='mi_person_key', how='left')
                
                # Drop mi_person_key - it's not a feature, only used for joining
                if 'mi_person_key' in data.columns:
                    data = data.drop(columns=['mi_person_key'])
                
                # All columns are already strings with empty string defaults, no need to fillna
                logger.info("CatBoost feature matrix created: %d rows × %d features", len(data), len([c for c in data.columns if c.startswith('item_')]))
            else:
                # Random Forest / XGBoost / ExtraTrees format: binary features
                # To keep the matrix compact and avoid huge pivot tables,
                # restrict to the pruned feature vocabulary (all_item_list)
                # before pivoting.
                patient_items_filtered = patient_items[
                    patient_items["item"].isin(all_item_list)
                ].copy()
                if patient_items_filtered.empty:
                    # No items survive pruning → create an all-zero matrix
                    feature_matrix = patient_targets[["mi_person_key"]].drop_duplicates()
                else:
                    patient_items_filtered["value"] = 1

                    # Pivot only for patients that have at least one (kept) item
                    feature_matrix = patient_items_filtered.pivot_table(
                        index="mi_person_key",
                        columns="item",
                        values="value",
                        aggfunc="max",  # presence/absence; keeps dtype small
                        fill_value=0,
                    ).reset_index()

                # Ensure we have one row for EVERY patient in patient_targets,
                # even those with zero items (they’ll get all zeros)
                all_patients = patient_targets[['mi_person_key']].drop_duplicates()
                feature_matrix = all_patients.merge(feature_matrix, on='mi_person_key', how='left')

                # Make sure all item columns exist
                for item in all_item_list:
                    if item not in feature_matrix.columns:
                        feature_matrix[item] = 0

                # Fill any remaining NaNs in feature columns with 0
                feature_cols = [col for col in feature_matrix.columns if col != 'mi_person_key']
                feature_matrix[feature_cols] = feature_matrix[feature_cols].fillna(0)

                # Add 'item_' prefix
                feature_matrix = feature_matrix.rename(
                    columns={col: f'item_{col}' for col in feature_cols}
                )

                # Join with targets (now one row per patient, matching CatBoost)
                data = feature_matrix.merge(patient_targets, on='mi_person_key', how='left')

                # Drop mi_person_key - it's not a feature
                if 'mi_person_key' in data.columns:
                    data = data.drop(columns=['mi_person_key'])
            
            # Clean data
            data = data.dropna(subset=['target'])
            
            return data
        
        # Helper: derive a compact binary RF/XGBoost matrix from the CatBoost matrix
        def catboost_to_rf(df: pd.DataFrame) -> pd.DataFrame:
            """
            Convert CatBoost-style item columns (string item names or '')
            into binary 0/1 indicators for RF/XGBoost.
            """
            rf_df = df.copy()
            feature_cols = [c for c in rf_df.columns if c.startswith("item_")]
            for col in feature_cols:
                rf_df[col] = (rf_df[col] != "").astype("uint8")
            return rf_df

        # Create feature matrices for train and test (BUILT ONCE, reused for all MC-CV splits)
        logger.info("Building feature matrices (one-time operation, will be reused for all %d MC-CV splits)...", n_splits)
        train_data_catboost = create_feature_matrix(train_patient_items, train_patient_targets, all_item_list, is_catboost=True)
        test_data_catboost = create_feature_matrix(test_patient_items, test_patient_targets, all_item_list, is_catboost=True)

        # Derive RF/XGBoost matrices from CatBoost matrices to avoid a second
        # wide pivot_table and keep memory usage bounded.
        train_data_rf = catboost_to_rf(train_data_catboost)
        test_data_rf = catboost_to_rf(test_data_catboost)
        logger.info("Feature matrices complete. These will be indexed (not rebuilt) for each MC-CV split.")
        
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
        constant_features_file = os.path.join(
            output_dir,
            f"{cohort_name}_{age_band_fname}_constant_features.csv"
        )
        
        # Idempotent handling: if constant-features file already exists (locally or in S3),
        # load and reuse it instead of recomputing over all columns.
        if os.path.exists(constant_features_file):
            logger.info(
                "Constant-features file already exists locally (%s); "
                "loading instead of recomputing.", constant_features_file
            )
            constant_features_df = pd.read_csv(constant_features_file)
            constant_features = constant_features_df['feature'].tolist()
        else:
            # Try S3 before recomputing
            s3_key_const = f"gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_constant_features.csv"
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_const)
                logger.info(
                    "Constant-features file exists in S3 (s3://%s/%s); "
                    "downloading instead of recomputing.",
                    S3_BUCKET, s3_key_const
                )
                import io
                obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_const)
                constant_features_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                constant_features = constant_features_df['feature'].tolist()
                os.makedirs(output_dir, exist_ok=True)
                constant_features_df.to_csv(constant_features_file, index=False)
                logger.info("Saved constant-features list locally: %s", constant_features_file)
            except Exception:
                # Not available locally or in S3 → compute from scratch
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
                f"{cohort_name}_{age_band_fname}_constant_features.csv"
            )
            constant_features_df.to_csv(constant_features_file, index=False)
            logger.info("Saved constant features list: %s", constant_features_file)
            
            # Upload to S3 alongside other feature-importance artifacts
            # Folder pattern: gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_constant_features.csv
            s3_key_const = f"gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_constant_features.csv"
            if upload_csv_to_s3(constant_features_file, s3_key_const):
                logger.info("Uploaded constant features to S3: s3://pgxdatalake/%s", s3_key_const)
            else:
                logger.warning("Failed to upload constant features to S3: s3://pgxdatalake/%s", s3_key_const)
            
            # Remove constant features from datasets. In some legacy runs,
            # the constant-features list may include columns that are no
            # longer present after vocabulary/pruning changes (especially
            # for small cohorts like 0-12). To keep things robust and
            # idempotent, intersect with the actual columns before dropping.
            cols_cat_train = set(train_data_catboost.columns)
            cols_rf_train = set(train_data_rf.columns)
            cols_cat_test = set(test_data_catboost.columns)
            cols_rf_test = set(test_data_rf.columns)

            cf_set = set(constant_features)
            drop_cat_train = list(cf_set & cols_cat_train)
            drop_rf_train = list(cf_set & cols_rf_train)
            drop_cat_test = list(cf_set & cols_cat_test)
            drop_rf_test = list(cf_set & cols_rf_test)

            missing_for_train = cf_set - cols_cat_train
            if missing_for_train:
                logger.info(
                    "Constant-features list includes %d columns not present in "
                    "current train_data_catboost; ignoring those entries.",
                    len(missing_for_train),
                )

            train_data_catboost = train_data_catboost.drop(columns=drop_cat_train)
            train_data_rf = train_data_rf.drop(columns=drop_rf_train)
            test_data_catboost = test_data_catboost.drop(columns=drop_cat_test)
            test_data_rf = test_data_rf.drop(columns=drop_rf_test)
        
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
        
        # After constant feature filtering, we no longer need the raw
        # patient-level item tables or item-frequency helpers. Free them
        # proactively to keep memory headroom for MC-CV + permutation
        # importance (which is memory intensive with XGBoost on GPU).
        _cleanup_memory(
            "post-feature-engineering (patient-level items)",
            train_patient_items,
            test_patient_items,
            all_items,
            item_patient_counts,
        )

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

        # Free any transient stratification arrays that are no longer needed
        # now that we have materialized split_indices.
        _cleanup_memory("post MC-CV split creation")
        
        # Run MC-CV for each method in the primary model ensemble
        logger.info("Running MC-CV analysis...")
        # Restricted to three core models for both MC feature importance and final evaluation
        methods = ['catboost', 'xgboost', 'xgboost_rf']
        all_results = {}
        
        check_memory_usage_r(logger, "Before MC-CV Execution")
        
        for method in methods:
            # Check if this model's results already exist (idempotency)
            # Check local file first, then S3
            local_file = os.path.join(output_dir, f"{cohort_name}_{age_band_fname}_{method}_feature_importance.csv")
            # S3 key uses age_band in the folder name and age_band_fname (hyphens -> underscores) in the filename
            s3_key_method = f"gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_{method}_feature_importance.csv"
            
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

            # Configure per-method checkpoint directory so we can resume
            # long-running MC-CV jobs without losing completed splits.
            checkpoints_root = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoints_root, exist_ok=True)
            checkpoint_dir = os.path.join(
                checkpoints_root,
                f"{cohort_name}_{age_band_fname}_{method}",
            )
            
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
                    test_data_catboost=test_data_catboost,
                    checkpoint_dir=checkpoint_dir,
                    force_rerun_checkpoints=force_rerun,
                )
            else:
                # For XGBoost / XGBoost RF we now keep the dense DataFrame path
                # on all OSes. The sparse conversion introduced subtle feature
                # shape mismatches between train/test for some splits, while
                # the dense path has been stable both locally and on EC2.
                result = run_mc_cv_method(
                    train_data_rf,
                    method,
                    split_indices,
                    model_params,
                    scaling_metric,
                    n_jobs=n_workers,
                    test_data=test_data_rf,
                    checkpoint_dir=checkpoint_dir,
                    force_rerun_checkpoints=force_rerun,
                )
            
            all_results[method] = result
            
            # Save individual results locally
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"{cohort_name}_{age_band_fname}_{method}_feature_importance.csv"
            )
            result.to_csv(output_file, index=False)
            logger.info("Saved locally: %s", output_file)
            
            # Upload to S3
            # Folder pattern: gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_{method}_feature_importance.csv
            s3_key = f"gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_{method}_feature_importance.csv"
            if upload_csv_to_s3(output_file, s3_key):
                logger.info("Uploaded to S3: s3://pgxdatalake/%s", s3_key)
            else:
                logger.warning("Failed to upload to S3: s3://pgxdatalake/%s", s3_key)
            
            check_memory_usage_r(logger, f"After MC-CV: {method}")
        
        check_memory_usage_r(logger, "After MC-CV Execution")

        # ------------------------------------------------------------------
        # SECOND PASS: Rare-variant scans on the target cohort (holdout year)
        # ------------------------------------------------------------------
        try:
            logger.info("Starting rare-variant scans on target cohort (holdout only)...")
            # Identify rare features on the 2019 holdout using the RF/XGB
            # feature matrix (binary 0/1). This pass is restricted to the
            # temporal holdout and to items that appear in a small number of
            # patients, to focus on rare but potentially important signals.
            rare_min = int(os.getenv("PGX_RARE_MIN_PATIENTS", "5"))
            rare_max = int(os.getenv("PGX_RARE_MAX_PATIENTS", "25"))
            logger.info(
                "Rare-variant thresholds (holdout patients per feature): min=%d, max=%d",
                rare_min,
                rare_max,
            )

            # Restrict to item_* feature columns (event/drug/etc. tokens).
            item_cols = [
                c for c in test_data_rf.columns
                if c.startswith("item_")
            ]
            if not item_cols:
                logger.warning("No item_* columns found in test_data_rf; skipping rare-variant scans.")
            else:
                # Count how many holdout patients have each feature.
                counts = test_data_rf[item_cols].sum(axis=0)
                rare_cols = [
                    col for col in item_cols
                    if rare_min <= counts[col] < rare_max
                ]

                if not rare_cols:
                    logger.info(
                        "No rare features found on target cohort with %d <= patients < %d; skipping rare-variant scan.",
                        rare_min,
                        rare_max,
                    )
                else:
                    logger.info(
                        "Rare-variant scan: %d candidate features on holdout (from %d item_* columns).",
                        len(rare_cols),
                        len(item_cols),
                    )

                    # Build a slim matrix for the holdout year only (RF/XGB view).
                    X_rare = test_data_rf[rare_cols].copy()
                    y_rare = test_data_rf["target"].values

                    # Simple train/eval split within the holdout to avoid
                    # re-using the same rows for both fitting and evaluation.
                    sss_rare = StratifiedShuffleSplit(
                        n_splits=1, test_size=0.5, random_state=42
                    )
                    train_idx_rare, eval_idx_rare = next(sss_rare.split(X_rare, y_rare))

                    X_train_rare = X_rare.iloc[train_idx_rare]
                    y_train_rare = y_rare[train_idx_rare]
                    X_eval_rare = X_rare.iloc[eval_idx_rare]
                    y_eval_rare = y_rare[eval_idx_rare]

                    logger.info(
                        "Rare-variant scan: train=%d, eval=%d patients; %d rare features.",
                        len(X_train_rare),
                        len(X_eval_rare),
                        len(rare_cols),
                    )

                    from py_helpers.model_utils import calculate_recall, calculate_logloss

                    # ------------------------------------------------------
                    # XGBoost rare-variant scan
                    # ------------------------------------------------------
                    try:
                        xgb_params = (model_params or {}).get("xgboost", {})
                        xgb_rare_model = train_xgboost(X_train_rare, y_train_rare, xgb_params)

                        rare_feat_names = list(X_eval_rare.columns)
                        rare_importance_xgb = get_permutation_importance(
                            xgb_rare_model,
                            X_eval_rare,
                            y_eval_rare,
                            rare_feat_names,
                            scoring=scaling_metric,
                        )

                        y_pred_rare_xgb = predict_xgboost(xgb_rare_model, X_eval_rare)
                        y_pred_proba_rare_xgb = predict_proba_xgboost(xgb_rare_model, X_eval_rare)

                        recall_rare_xgb = calculate_recall(y_eval_rare, y_pred_rare_xgb)
                        logloss_rare_xgb = calculate_logloss(y_eval_rare, y_pred_proba_rare_xgb)

                        if scaling_metric == "recall":
                            scale_factor_rare_xgb = recall_rare_xgb if recall_rare_xgb > 0 else 0.001
                        elif scaling_metric == "logloss":
                            scale_factor_rare_xgb = (
                                1.0 / logloss_rare_xgb if logloss_rare_xgb > 0 else 0.001
                            )
                        else:
                            scale_factor_rare_xgb = 1.0

                        rare_importance_xgb["scaled_importance"] = (
                            rare_importance_xgb["importance"] * scale_factor_rare_xgb
                        )
                        rare_importance_xgb["recall"] = recall_rare_xgb
                        rare_importance_xgb["logloss"] = logloss_rare_xgb

                        rare_result_for_agg_xgb = pd.DataFrame({
                            "feature": rare_importance_xgb["feature"],
                            "scaled_importance_mean": rare_importance_xgb["scaled_importance"],
                            "importance_mean": rare_importance_xgb["importance"],
                            "recall_mean": recall_rare_xgb,
                            "logloss_mean": logloss_rare_xgb,
                        })
                        all_results["xgboost_rare"] = rare_result_for_agg_xgb

                        rare_output_file_xgb = os.path.join(
                            output_dir,
                            f"{cohort_name}_{age_band_fname}_xgboost_rare_feature_importance.csv",
                        )
                        rare_importance_xgb.to_csv(rare_output_file_xgb, index=False)
                        logger.info(
                            "Saved rare-variant XGBoost importance locally: %s",
                            rare_output_file_xgb,
                        )

                        rare_s3_key_xgb = (
                            f"gold/feature_importance/{cohort_name}/{age_band}/"
                            f"{cohort_name}_{age_band_fname}_xgboost_rare_feature_importance.csv"
                        )
                        if upload_csv_to_s3(rare_output_file_xgb, rare_s3_key_xgb):
                            logger.info(
                                "Uploaded rare-variant XGBoost importance to S3: s3://pgxdatalake/%s",
                                rare_s3_key_xgb,
                            )
                        else:
                            logger.warning(
                                "Failed to upload rare-variant XGBoost importance to S3: s3://pgxdatalake/%s",
                                rare_s3_key_xgb,
                            )
                    except Exception:
                        logger.warning(
                            "Rare-variant XGBoost scan failed; continuing without rare-variant XGBoost results.",
                            exc_info=True,
                        )

                    # ------------------------------------------------------
                    # CatBoost rare-variant scan
                    # ------------------------------------------------------
                    try:
                        # Reuse the same train/eval split indices on the CatBoost
                        # feature matrix to keep evaluation comparable.
                        X_rare_cat = test_data_catboost[rare_cols].copy()
                        y_rare_cat = test_data_catboost["target"].values

                        X_train_rare_cat = X_rare_cat.iloc[train_idx_rare]
                        y_train_rare_cat = y_rare_cat[train_idx_rare]
                        X_eval_rare_cat = X_rare_cat.iloc[eval_idx_rare]
                        y_eval_rare_cat = y_rare_cat[eval_idx_rare]

                        cat_params = (model_params or {}).get("catboost", {})
                        cat_rare_model = train_catboost(X_train_rare_cat, y_train_rare_cat, cat_params)

                        rare_feat_names_cat = list(X_eval_rare_cat.columns)
                        rare_importance_cat = get_permutation_importance(
                            cat_rare_model,
                            X_eval_rare_cat,
                            y_eval_rare_cat,
                            rare_feat_names_cat,
                            scoring=scaling_metric,
                        )

                        y_pred_rare_cat = predict_catboost(cat_rare_model, X_eval_rare_cat)
                        y_pred_proba_rare_cat = predict_proba_catboost(cat_rare_model, X_eval_rare_cat)

                        recall_rare_cat = calculate_recall(y_eval_rare_cat, y_pred_rare_cat)
                        logloss_rare_cat = calculate_logloss(y_eval_rare_cat, y_pred_proba_rare_cat)

                        if scaling_metric == "recall":
                            scale_factor_rare_cat = recall_rare_cat if recall_rare_cat > 0 else 0.001
                        elif scaling_metric == "logloss":
                            scale_factor_rare_cat = (
                                1.0 / logloss_rare_cat if logloss_rare_cat > 0 else 0.001
                            )
                        else:
                            scale_factor_rare_cat = 1.0

                        rare_importance_cat["scaled_importance"] = (
                            rare_importance_cat["importance"] * scale_factor_rare_cat
                        )
                        rare_importance_cat["recall"] = recall_rare_cat
                        rare_importance_cat["logloss"] = logloss_rare_cat

                        rare_result_for_agg_cat = pd.DataFrame({
                            "feature": rare_importance_cat["feature"],
                            "scaled_importance_mean": rare_importance_cat["scaled_importance"],
                            "importance_mean": rare_importance_cat["importance"],
                            "recall_mean": recall_rare_cat,
                            "logloss_mean": logloss_rare_cat,
                        })
                        all_results["catboost_rare"] = rare_result_for_agg_cat

                        rare_output_file_cat = os.path.join(
                            output_dir,
                            f"{cohort_name}_{age_band_fname}_catboost_rare_feature_importance.csv",
                        )
                        rare_importance_cat.to_csv(rare_output_file_cat, index=False)
                        logger.info(
                            "Saved rare-variant CatBoost importance locally: %s",
                            rare_output_file_cat,
                        )

                        rare_s3_key_cat = (
                            f"gold/feature_importance/{cohort_name}/{age_band}/"
                            f"{cohort_name}_{age_band_fname}_catboost_rare_feature_importance.csv"
                        )
                        if upload_csv_to_s3(rare_output_file_cat, rare_s3_key_cat):
                            logger.info(
                                "Uploaded rare-variant CatBoost importance to S3: s3://pgxdatalake/%s",
                                rare_s3_key_cat,
                            )
                        else:
                            logger.warning(
                                "Failed to upload rare-variant CatBoost importance to S3: s3://pgxdatalake/%s",
                                rare_s3_key_cat,
                            )
                    except Exception:
                        logger.warning(
                            "Rare-variant CatBoost scan failed; continuing without rare-variant CatBoost results.",
                            exc_info=True,
                        )
        except Exception:
            # Rare-variant analysis is best-effort and should never break
            # the main cohort analysis pipeline.
            logger.warning(
                "Rare-variant scans failed; continuing without rare-variant results.",
                exc_info=True,
            )
        
        # Aggregate results across models (including rare-variant XGBoost
        # when available).
        logger.info("Aggregating results (including rare-variant model if available)...")
        aggregated = aggregate_feature_importance(all_results, scaling_metric, logger=logger)
        
        # Save aggregated results locally
        output_file = os.path.join(
            output_dir,
            f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
        )
        aggregated.to_csv(output_file, index=False)
        logger.info("Saved aggregated results locally: %s", output_file)

        # ------------------------------------------------------------------
        # Best-effort cleanup of per-split checkpoint files now that the
        # run has completed successfully and results are materialized.
        # ------------------------------------------------------------------
        try:
            checkpoints_root = os.path.join(output_dir, "checkpoints")
            for method in methods:
                ckpt_dir = os.path.join(
                    checkpoints_root,
                    f"{cohort_name}_{age_band_fname}_{method}",
                )
                if os.path.isdir(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                    logger.info("Removed checkpoint directory after successful run: %s", ckpt_dir)
        except Exception as e:
            logger.warning("Failed to remove checkpoint directories: %s", e)
        
        # Upload aggregated results to S3
        # Folder pattern: gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv
        s3_key_agg = f"gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
        upload_ok = upload_csv_to_s3(output_file, s3_key_agg)
        if upload_ok:
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

        # Send SES notification email about S3 upload status
        try:
            subject = f"[PGX Feature Importance] Upload status for {cohort_name} {age_band}"
            status_text = "SUCCESS" if upload_ok else "FAILED"
            body_lines = [
                f"Cohort: {cohort_name}",
                f"Age band: {age_band}",
                f"Train years: {', '.join(map(str, train_years))}",
                f"Test year: {test_year}",
                f"S3 key (aggregated): {s3_key_agg}",
                f"Upload status: {status_text}",
                "",
                f"Local output file: {output_file}",
            ]
            send_status_email_ses(subject, "\n".join(body_lines))
        except Exception:
            # Email failures should not break the analysis pipeline
            pass
        
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
        from py_helpers.constants import S3_BUCKET
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
    
    # Aggregate by feature across models:
    # - sum importance_scaled_by_model,
    # - count how many models contributed (n_models),
    # - capture which models contributed (models),
    # - compute the mean contribution per model, then renormalize.
    agg = normalized_df.groupby('feature').agg(
        importance_scaled_by_model_sum=('importance_scaled_by_model', 'sum'),
        importance_normalized_sum=('importance_normalized', 'sum'),
        n_models=('method', 'nunique'),
        models=('method', lambda s: ",".join(sorted(s.unique()))),
    ).reset_index()
    
    # Avoid division by zero in pathological cases
    agg['n_models'] = agg['n_models'].replace(0, 1)
    
    # Mean scaled contribution per model (scales by number of models used)
    agg['importance_scaled_mean'] = (
        agg['importance_scaled_by_model_sum'] / agg['n_models']
    )
    
    # Renormalize final aggregated importance to [0, 1]
    scaled_max = agg['importance_scaled_mean'].max()
    scaled_min = agg['importance_scaled_mean'].min()
    if scaled_max > scaled_min:
        agg['importance_normalized'] = (
            agg['importance_scaled_mean'] - scaled_min
        ) / (scaled_max - scaled_min)
    else:
        agg['importance_normalized'] = 0.0
    
    # Scale by best model performance
    if best_performance > 0:
        if scaling_metric == 'logloss':
            # For logloss, lower is better, so invert to turn it into a
            # "larger is better" scaling factor.
            scale_factor = 1.0 / best_performance
        else:
            # For recall (or other metrics where higher is better),
            # use best_performance directly.
            scale_factor = best_performance
        agg['importance_scaled'] = agg['importance_normalized'] * scale_factor
    else:
        agg['importance_scaled'] = agg['importance_normalized']
    
    # Select final columns and sort
    agg = agg[['feature', 'importance_normalized', 'importance_scaled', 'n_models', 'models']]
    agg = agg.sort_values('importance_scaled', ascending=False)
    
    return agg
