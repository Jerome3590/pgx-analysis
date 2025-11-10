import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
import optuna


class ADECatBoostTarget:
    """Runs ADE CatBoost models by age band"""
    
    def __init__(self, event_years=None, max_workers=4):
        self.event_years = event_years or [2016, 2017, 2018, 2019, 2020]
        self.max_workers = max_workers
        
        # Script path
        self.ade_script = os.path.join(project_root, "catboost_analysis", "run_catboost_ade_ed.py")
        
        # Results storage
        self.results_dir = os.path.join(project_root, "catboost_analysis", "ade_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("ade_catboost", "all", "all")
        
        # Validate script exists
        if not os.path.exists(self.ade_script):
            raise FileNotFoundError(f"ADE script not found: {self.ade_script}")
    
    def run_age_band_models(self, age_band):
        """Run ADE models for a specific age band across all years"""
        self.logger.info(f"Starting ADE models for age band: {age_band}")
        
        results = []
        for event_year in self.event_years:
            try:
                cmd = [
                    sys.executable, self.ade_script,
                    "--age-band", age_band,
                    "--event-year", str(event_year)
                ]
                
                self.logger.info(f"Running ADE model for {age_band}/{event_year}...")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=3600  # 1 hour timeout
                )
                
                self.logger.info(f"✓ ADE {age_band}/{event_year} completed successfully")
                
                results.append({
                    "age_band": age_band,
                    "event_year": event_year,
                    "status": "success",
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode
                })
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"✗ ADE {age_band}/{event_year} timed out")
                results.append({
                    "age_band": age_band,
                    "event_year": event_year,
                    "status": "timeout",
                    "output": "",
                    "error": "Process timed out after 1 hour",
                    "return_code": -1
                })
            except subprocess.CalledProcessError as e:
                self.logger.error(f"✗ ADE {age_band}/{event_year} failed: {e}")
                results.append({
                    "age_band": age_band,
                    "event_year": event_year,
                    "status": "error",
                    "output": e.stdout,
                    "error": e.stderr,
                    "return_code": e.returncode
                })
            except Exception as e:
                self.logger.error(f"✗ ADE {age_band}/{event_year} unexpected error: {e}")
                results.append({
                    "age_band": age_band,
                    "event_year": event_year,
                    "status": "error",
                    "output": "",
                    "error": str(e),
                    "return_code": -1
                })
        
        return results
    
    def run_parallel_age_bands(self, age_bands):
        """Run ADE models for multiple age bands in parallel"""
        self.logger.info("Starting parallel ADE CatBoost pipeline...")
        self.logger.info(f"Processing {len(age_bands)} age bands × {len(self.event_years)} years = {len(age_bands) * len(self.event_years)} total jobs")
        
        all_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_age_band = {
                executor.submit(self.run_age_band_models, age_band): age_band
                for age_band in age_bands
            }
            
            for future in concurrent.futures.as_completed(future_to_age_band):
                age_band = future_to_age_band[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    
                    # Log summary for this age band
                    successful = len([r for r in results if r["status"] == "success"])
                    self.logger.info(f"Age band {age_band}: {successful}/{len(results)} models completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error processing age band {age_band}: {e}")
        
        return all_results
    
    def analyze_results(self, results):
        """Analyze results from ADE models"""
        self.logger.info("Analyzing ADE model results...")
        
        # Success rates
        total_jobs = len(results)
        successful_jobs = len([r for r in results if r["status"] == "success"])
        
        # Group by age band
        age_band_results = {}
        for result in results:
            age_band = result["age_band"]
            if age_band not in age_band_results:
                age_band_results[age_band] = []
            age_band_results[age_band].append(result)
        
        # Calculate success rates by age band
        age_band_success_rates = {}
        for age_band, band_results in age_band_results.items():
            successful = len([r for r in band_results if r["status"] == "success"])
            age_band_success_rates[age_band] = successful / len(band_results)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "ade_ed",
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "overall_success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
            "age_band_success_rates": age_band_success_rates,
            "event_years_processed": self.event_years,
            "results": results
        }
        
        # Log summary
        self.logger.info(f"ADE Pipeline Summary:")
        self.logger.info(f"  Total jobs: {analysis['total_jobs']}")
        self.logger.info(f"  Successful jobs: {analysis['successful_jobs']}")
        self.logger.info(f"  Overall success rate: {analysis['overall_success_rate']:.1%}")
        self.logger.info(f"  Success rates by age band:")
        for age_band, rate in age_band_success_rates.items():
            self.logger.info(f"    {age_band}: {rate:.1%}")
        
        return analysis
    
    def save_results(self, analysis):
        """Save results to local file and S3"""
        try:
            # Save analysis summary
            analysis_file = os.path.join(self.results_dir, "ade_pipeline_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Upload to S3
            s3_key = f"catboost_analysis/ade_pipeline/{datetime.now().strftime('%Y%m%d_%H%M%S')}/pipeline_analysis.json"
            s3_client.upload_file(analysis_file, S3_BUCKET, s3_key)
            self.logger.info(f"ADE analysis results uploaded to s3://{S3_BUCKET}/{s3_key}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def run_pipeline(self, age_bands):
        """Run the complete ADE CatBoost pipeline"""
        try:
            self.logger.info("="*80)
            self.logger.info("ADE CATBOOST PIPELINE STARTING")
            self.logger.info("="*80)
            
            # Run models
            results = self.run_parallel_age_bands(age_bands)
            
            # Analyze results
            analysis = self.analyze_results(results)
            
            # Save results
            self.save_results(analysis)
            
            # Send completion notification
            if analysis["successful_jobs"] == analysis["total_jobs"]:
                self.logger.info("🎉 All ADE models completed successfully!")
            else:
                self.logger.warning(f"⚠️ {analysis['total_jobs'] - analysis['successful_jobs']} ADE models failed")
            
            self.logger.info("="*80)
            self.logger.info("ADE CATBOOST PIPELINE COMPLETE")
            self.logger.info("="*80)
            
            return analysis
            
        except Exception as e:
            error_msg = f"ADE CatBoost pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            notify_error("ade_catboost_pipeline", error_msg, self.logger)
            raise


# Set up environment variables for age band and event year if needed
AGE_BAND = os.environ.get('AGE_BAND', 'all')
EVENT_YEAR = os.environ.get('EVENT_YEAR', 'all')

# Load your ADE (ED_NON_OPIOID) cohort data here
# Example: df = pd.read_parquet('path_to_ade_cohort_data.parquet')
df = pd.read_parquet(f"s3://pgxdatalake/cohorts/cohort_name=ed_non_opioid/age_band={AGE_BAND}/event_year={EVENT_YEAR}/cohort.parquet")

# Drop high-cardinality and not-needed columns
not_needed_high_card = [
    'billing_provider_name', 'billing_provider_npi_med', 'service_provider_name_med',
    'billing_provider_tin_med', 'service_provider_npi_med', 'billing_provider_npi_pharm',
    'service_provider_tin_med', 'service_provider_name_pharm', 'service_provider_npi_pharm',
    'billing_provider_zip_med','row_number'
]
lagging_indicators = [
    'primary_icd_diagnosis_code_group', 'two_icd_diagnosis_code_group',                      
    'three_icd_diagnosis_code_group', 'four_icd_diagnosis_code_group', 'five_icd_diagnosis_code_group',
    'six_icd_diagnosis_code_group', 'seven_icd_diagnosis_code_group', 'eight_icd_diagnosis_code_group',
    'nine_icd_diagnosis_code_group', 'ten_icd_diagnosis_code_group','place_of_service'
]
not_needed = [
    'ade_window', 'Event_med', 'age_band', 'hispanic_indicator_pharm', 'member_state_enroll_med',
    'billing_provider_taxonomy_pharm', 'service_provider_taxonomy_pharm', 
    'billing_provider_taxonomy_med', 'service_provider_taxonomy_med', 'service_provider_msa_pharm', 'service_provider_msa_med',
    'total_utilization_med', 'total_paid', 'total_allowed_pharm', 'total_utilization_pharm',
    'total_rx_paid', 'total_rx_days_supply', 'member_age_dos_med', 'gpi', 'strength',
    'dosage_form', 'bill_type_id', 'bill_type_class', 'service_provider_specialty_med',
    'hispanic_indicator_med','hispanic_indicator_pharm', 'cchg_label', 'member_age_band_dos_med',
    'cchg_grouping', 'hcg_setting_med', 'hcg_line_med',
    'hcg_detail_med', 'bill_type_description', 'member_age_dos_pharm', 'member_age_band_dos_pharm',
    'member_gender_pharm', 'member_race_pharm', 'hcg_setting_pharm', 'hcg_line_pharm',
    'hcg_detail_pharm', 'payer_type_pharm', 'revenue_code', 'billing_provider_specialty_med',
    'billing_provider_county_med', 'billing_provider_state_med', 'billing_provider_msa_med',
    'service_provider_county_med', 'service_provider_state_med', 'billing_provider_specialty_pharm',
    'billing_provider_zip_pharm', 'billing_provider_county_pharm', 'billing_provider_state_pharm',
    'billing_provider_msa_pharm', 'billing_provider_tin_pharm', 'service_provider_specialty_pharm',
    'service_provider_county_pharm', 'service_provider_state_pharm', 'service_provider_tin_pharm',
    'payer_type_med','member_county_dos_med','member_zip_code_dos_pharm','member_county_dos_pharm',
    'member_state_enroll_pharm','total_allowed_med','admit_type','DayOfWeekString','Weekend',
    'WeekOfMonth','NearUSHoliday','ndc'
]
drop_cols = set(not_needed_high_card + not_needed + lagging_indicators)
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Identify categorical and numerical columns
def get_categorical_columns(df):
    return df.select_dtypes(include=['object','category','string']).columns.tolist()

def get_numerical_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

categorical_cols = [col for col in get_categorical_columns(df) if col not in ['mi_person_key', 'event_date', 'Year']]
numerical_cols = [col for col in get_numerical_columns(df) if col != 'Target']

# Fill NA in categoricals with 'None'
for col in categorical_cols:
    df[col] = df[col].fillna('None').astype(str)

# ⭐ TEMPORAL FILTERING: Drop events after first ADE (ED_NON_OPIOID) event to prevent data leakage
print("Applying temporal filtering to prevent data leakage for ADE events...")
print(f"Before filtering: {len(df)} events")

df['event_date'] = pd.to_datetime(df['event_date'])

# For each person, find the first ADE event date (Target == 1)
first_target_dates = df[df['Target'] == 1].groupby('mi_person_key')['event_date'].min().reset_index()
first_target_dates.columns = ['mi_person_key', 'first_target_date']

df = df.merge(first_target_dates, on='mi_person_key', how='left')

df_filtered = df[
    (df['first_target_date'].isna()) |
    (df['event_date'] < df['first_target_date'])
].copy()
df_filtered = df_filtered.drop('first_target_date', axis=1)

print(f"After filtering: {len(df_filtered)} events")
print(f"Removed {len(df) - len(df_filtered)} events that occurred after first ADE event")

df = df_filtered

df = df.sort_values(by='event_date')
df['Year'] = df['Year'].astype(int)
df = df.drop_duplicates(keep='first')

target_label = 'Target'
train_df = df[df['Year'].isin([2016, 2017, 2018, 2019])]
test_df = df[df['Year'] == 2020]

X_train = train_df.drop([target_label, 'mi_person_key', 'event_date', 'Year'], axis=1)
y_train = train_df[target_label]
X_test = test_df.drop([target_label, 'mi_person_key', 'event_date', 'Year'], axis=1)
y_test = test_df[target_label]

cat_features_model = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]

def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 3, 10),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 12),
        "boosting_type": "Ordered",
        "bootstrap_type": "MVS",
        "early_stopping_rounds": 50,
        "eval_metric": "AUC",
        "random_seed": 1997,
        "verbose": 0
    }
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, y_train, cat_features=cat_features_model)
    test_pool = Pool(X_test, y_test, cat_features=cat_features_model)
    model.fit(train_pool, eval_set=test_pool, verbose=0)
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

print("Running Optuna hyperparameter search for ADE events...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, timeout=1800)
print("Best params:", study.best_params)

final_params = study.best_params
final_params.update({
    "iterations": 2000,
    "boosting_type": "Ordered",
    "bootstrap_type": "MVS",
    "early_stopping_rounds": 100,
    "eval_metric": "AUC",
    "random_seed": 1997,
    "verbose": 100
})
final_model = CatBoostClassifier(**final_params)
train_pool = Pool(X_train, y_train, cat_features=cat_features_model)
test_pool = Pool(X_test, y_test, cat_features=cat_features_model)
final_model.fit(train_pool, eval_set=test_pool, verbose=100)

output_dir = f"catboost_results_ade_{AGE_BAND}_{EVENT_YEAR}"
os.makedirs(output_dir, exist_ok=True)

model_cbm_path = os.path.join(output_dir, "catboost_model.cbm")
final_model.save_model(model_cbm_path)
print(f"Saved CatBoost model to: {model_cbm_path}")

model_json_path = os.path.join(output_dir, "catboost_model.json")
final_model.save_model(model_json_path, format="json")
print(f"Saved CatBoost model in JSON format to: {model_json_path}")

feature_names = {i: name for i, name in enumerate(X_train.columns)}
feature_names_path = os.path.join(output_dir, "feature_names.json")
with open(feature_names_path, 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"Saved feature names mapping to: {feature_names_path}")

y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
metrics = {
    "AUC": roc_auc_score(y_test, y_pred_proba),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "LogLoss": log_loss(y_test, y_pred_proba)
}
print("Test metrics:", json.dumps(metrics, indent=2))

importances = final_model.get_feature_importance(prettified=True)
importances.to_csv(os.path.join(output_dir, "catboost_feature_importances.csv"), index=False)
print("Top features:\n", importances.head(20))

print("Calculating SHAP values for ADE events...")
explainer = shap.Explainer(final_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "catboost_shap_summary.png"), dpi=300)
plt.close()

np.save(os.path.join(output_dir, "catboost_shap_values.npy"), shap_values.values)
X_test.to_csv(os.path.join(output_dir, "catboost_X_test.csv"), index=False)
with open(os.path.join(output_dir, "catboost_test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Pipeline complete. Artifacts saved in {output_dir}/") 