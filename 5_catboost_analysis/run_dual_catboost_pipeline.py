#!/usr/bin/env python3
"""
Dual CatBoost Pipeline: Opioid ED and ADE Models
Runs both Opioid ED and ADE (ED_NON_OPIOID) CatBoost models in parallel
and provides comprehensive comparison and analysis.
"""

import os
import sys
import json
import argparse
import subprocess
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set root of project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import existing utilities
from helpers.common_imports import (
    s3_client, 
    S3_BUCKET, 
    get_logger, 
    ClientError
)

from helpers.duckdb_utils import (
    get_duckdb_connection,
    execute_duckdb_query,
    setup_duckdb_environment
)

from helpers.s3_utils import (
    save_to_s3_parquet,
    save_to_s3_json,
    parse_s3_path,
    get_output_paths
)

from helpers.constants import (
    S3_BUCKET,
    METRICS_BUCKET,
    NOTIFICATION_EMAIL
)

from helpers.aws_utils import (
    notify_error,
    send_email
)


class DualCatBoostPipeline:
    """Orchestrates running both Opioid ED and ADE CatBoost models"""
    
    def __init__(self, age_bands=None, event_years=None, max_workers=4):
        self.age_bands = age_bands or ["25-44", "45-54", "55-64", "65-74", "75-84"]
        self.event_years = event_years or [2016, 2017, 2018, 2019, 2020]
        self.max_workers = max_workers
        
        # Script paths
        self.opioid_script = os.path.join(project_root, "catboost_analysis", "run_catboost_opioid_ed.py")
        self.ade_script = os.path.join(project_root, "catboost_analysis", "run_catboost_ade_ed.py")
        
        # Results storage
        self.results_dir = os.path.join(project_root, "catboost_analysis", "dual_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.logger = get_logger("dual_catboost", "all", "all")
        
        # Validate scripts exist
        if not os.path.exists(self.opioid_script):
            raise FileNotFoundError(f"Opioid script not found: {self.opioid_script}")
        if not os.path.exists(self.ade_script):
            raise FileNotFoundError(f"ADE script not found: {self.ade_script}")
    
    def run_single_model(self, script_path, age_band, event_year, model_type):
        """Run a single CatBoost model"""
        try:
            cmd = [
                sys.executable, script_path,
                "--age-band", age_band,
                "--event-year", str(event_year)
            ]
            
            self.logger.info(f"Running {model_type} model for {age_band}/{event_year}...")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=3600  # 1 hour timeout
            )
            
            self.logger.info(f"✓ {model_type} {age_band}/{event_year} completed successfully")
            
            return {
                "model_type": model_type,
                "age_band": age_band,
                "event_year": event_year,
                "status": "success",
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ {model_type} {age_band}/{event_year} timed out")
            return {
                "model_type": model_type,
                "age_band": age_band,
                "event_year": event_year,
                "status": "timeout",
                "output": "",
                "error": "Process timed out after 1 hour",
                "return_code": -1
            }
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ {model_type} {age_band}/{event_year} failed: {e}")
            return {
                "model_type": model_type,
                "age_band": age_band,
                "event_year": event_year,
                "status": "error",
                "output": e.stdout,
                "error": e.stderr,
                "return_code": e.returncode
            }
        except Exception as e:
            self.logger.error(f"✗ {model_type} {age_band}/{event_year} unexpected error: {e}")
            return {
                "model_type": model_type,
                "age_band": age_band,
                "event_year": event_year,
                "status": "error",
                "output": "",
                "error": str(e),
                "return_code": -1
            }
    
    def run_parallel_models(self):
        """Run both Opioid ED and ADE models in parallel"""
        self.logger.info("Starting dual CatBoost pipeline...")
        self.logger.info(f"Processing {len(self.age_bands)} age bands × {len(self.event_years)} years = {len(self.age_bands) * len(self.event_years)} combinations per model")
        
        # Create all jobs
        jobs = []
        for age_band in self.age_bands:
            for event_year in self.event_years:
                # Opioid ED job
                jobs.append((self.opioid_script, age_band, event_year, "opioid_ed"))
                # ADE job
                jobs.append((self.ade_script, age_band, event_year, "ade"))
        
        # Run jobs in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self.run_single_model, script, age_band, event_year, model_type): 
                (script, age_band, event_year, model_type)
                for script, age_band, event_year, model_type in jobs
            }
            
            for future in concurrent.futures.as_completed(future_to_job):
                result = future.result()
                results.append(result)
                
                # Log progress
                if result["status"] == "success":
                    self.logger.info(f"✓ {result['model_type']} {result['age_band']}/{result['event_year']} completed")
                else:
                    self.logger.error(f"✗ {result['model_type']} {result['age_band']}/{result['event_year']} failed: {result['error']}")
        
        return results
    
    def analyze_results(self, results):
        """Analyze and compare results from both models"""
        self.logger.info("Analyzing model results...")
        
        # Separate results by model type
        opioid_results = [r for r in results if r["model_type"] == "opioid_ed"]
        ade_results = [r for r in results if r["model_type"] == "ade"]
        
        # Success rates
        opioid_success = len([r for r in opioid_results if r["status"] == "success"])
        ade_success = len([r for r in ade_results if r["status"] == "success"])
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_jobs": len(results),
            "successful_jobs": len([r for r in results if r["status"] == "success"]),
            "opioid_results": opioid_results,
            "ade_results": ade_results,
            "opioid_success_rate": opioid_success/len(opioid_results) if opioid_results else 0,
            "ade_success_rate": ade_success/len(ade_results) if ade_results else 0,
            "age_bands_processed": self.age_bands,
            "event_years_processed": self.event_years
        }
        
        # Log summary
        self.logger.info(f"Pipeline Summary:")
        self.logger.info(f"  Total jobs: {analysis['total_jobs']}")
        self.logger.info(f"  Successful jobs: {analysis['successful_jobs']}")
        self.logger.info(f"  Opioid ED success rate: {analysis['opioid_success_rate']:.1%}")
        self.logger.info(f"  ADE success rate: {analysis['ade_success_rate']:.1%}")
        
        return analysis
    
    def load_model_metrics(self, age_band, event_year, model_type):
        """Load metrics from a completed model"""
        try:
            if model_type == "opioid_ed":
                metrics_path = f"catboost_results_{age_band}_{event_year}/catboost_test_metrics.json"
            else:  # ade
                metrics_path = f"catboost_results_ade_{age_band}_{event_year}/catboost_test_metrics.json"
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.warning(f"Could not load metrics for {model_type} {age_band}/{event_year}: {e}")
            return None
    
    def compare_model_performance(self, results):
        """Compare performance metrics between Opioid ED and ADE models"""
        self.logger.info("Comparing model performance...")
        
        comparison_data = []
        
        for age_band in self.age_bands:
            for event_year in self.event_years:
                # Load metrics for both models
                opioid_metrics = self.load_model_metrics(age_band, event_year, "opioid_ed")
                ade_metrics = self.load_model_metrics(age_band, event_year, "ade")
                
                if opioid_metrics and ade_metrics:
                    comparison_data.append({
                        "age_band": age_band,
                        "event_year": event_year,
                        "opioid_auc": opioid_metrics.get("AUC", 0),
                        "ade_auc": ade_metrics.get("AUC", 0),
                        "opioid_f1": opioid_metrics.get("F1", 0),
                        "ade_f1": ade_metrics.get("F1", 0),
                        "opioid_precision": opioid_metrics.get("Precision", 0),
                        "ade_precision": ade_metrics.get("Precision", 0),
                        "opioid_recall": opioid_metrics.get("Recall", 0),
                        "ade_recall": ade_metrics.get("Recall", 0),
                        "opioid_accuracy": opioid_metrics.get("Accuracy", 0),
                        "ade_accuracy": ade_metrics.get("Accuracy", 0)
                    })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Calculate average metrics
            avg_metrics = {
                "Opioid ED": {
                    "AUC": df_comparison["opioid_auc"].mean(),
                    "F1": df_comparison["opioid_f1"].mean(),
                    "Precision": df_comparison["opioid_precision"].mean(),
                    "Recall": df_comparison["opioid_recall"].mean(),
                    "Accuracy": df_comparison["opioid_accuracy"].mean()
                },
                "ADE": {
                    "AUC": df_comparison["ade_auc"].mean(),
                    "F1": df_comparison["ade_f1"].mean(),
                    "Precision": df_comparison["ade_precision"].mean(),
                    "Recall": df_comparison["ade_recall"].mean(),
                    "Accuracy": df_comparison["ade_accuracy"].mean()
                }
            }
            
            # Log comparison
            self.logger.info(f"\nAverage Performance Metrics:")
            self.logger.info(f"{'Metric':<12} {'Opioid ED':<12} {'ADE':<12} {'Difference':<12}")
            self.logger.info("-" * 50)
            for metric in ["AUC", "F1", "Precision", "Recall", "Accuracy"]:
                opioid_val = avg_metrics["Opioid ED"][metric]
                ade_val = avg_metrics["ADE"][metric]
                diff = opioid_val - ade_val
                self.logger.info(f"{metric:<12} {opioid_val:<12.4f} {ade_val:<12.4f} {diff:<12.4f}")
            
            # Save comparison results
            comparison_file = os.path.join(self.results_dir, "model_comparison.csv")
            df_comparison.to_csv(comparison_file, index=False)
            self.logger.info(f"Comparison results saved to {comparison_file}")
            
            # Create visualization
            self.create_performance_plots(df_comparison)
            
            return df_comparison, avg_metrics
        else:
            self.logger.warning("No comparison data available - models may not have completed successfully")
            return None, None
    
    def create_performance_plots(self, df_comparison):
        """Create performance comparison visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle("Opioid ED vs ADE Model Performance Comparison", fontsize=16)
            
            metrics = ["AUC", "F1", "Precision", "Recall", "Accuracy"]
            for i, metric in enumerate(metrics):
                ax = axes[i//3, i%3]
                opioid_col = f"opioid_{metric.lower()}"
                ade_col = f"ade_{metric.lower()}"
                
                ax.scatter(df_comparison[opioid_col], df_comparison[ade_col], alpha=0.6)
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
                ax.set_xlabel(f"Opioid ED {metric}")
                ax.set_ylabel(f"ADE {metric}")
                ax.set_title(f"{metric} Comparison")
                ax.grid(True, alpha=0.3)
            
            # Add summary statistics in the last subplot
            ax = axes[1, 2]
            ax.axis('off')
            
            # Calculate summary stats
            opioid_avg = df_comparison[[col for col in df_comparison.columns if col.startswith('opioid_')]].mean()
            ade_avg = df_comparison[[col for col in df_comparison.columns if col.startswith('ade_')]].mean()
            
            summary_text = "Average Performance:\n\n"
            for metric in ["auc", "f1", "precision", "recall", "accuracy"]:
                opioid_val = opioid_avg[f"opioid_{metric}"]
                ade_val = ade_avg[f"ade_{metric}"]
                diff = opioid_val - ade_val
                summary_text += f"{metric.upper()}: {opioid_val:.3f} vs {ade_val:.3f} (Δ={diff:+.3f})\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.results_dir, "performance_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance comparison plot saved to {plot_file}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating performance plots: {e}")
    
    def save_results_to_s3(self, analysis, comparison_df=None):
        """Save results to S3"""
        try:
            # Save analysis summary
            analysis_file = os.path.join(self.results_dir, "pipeline_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Upload to S3
            s3_key = f"catboost_analysis/dual_pipeline/{datetime.now().strftime('%Y%m%d_%H%M%S')}/pipeline_analysis.json"
            s3_client.upload_file(analysis_file, S3_BUCKET, s3_key)
            self.logger.info(f"Analysis results uploaded to s3://{S3_BUCKET}/{s3_key}")
            
            # Upload comparison data if available
            if comparison_df is not None:
                comparison_file = os.path.join(self.results_dir, "model_comparison.csv")
                s3_key = f"catboost_analysis/dual_pipeline/{datetime.now().strftime('%Y%m%d_%H%M%S')}/model_comparison.csv"
                s3_client.upload_file(comparison_file, S3_BUCKET, s3_key)
                self.logger.info(f"Comparison data uploaded to s3://{S3_BUCKET}/{s3_key}")
            
        except Exception as e:
            self.logger.error(f"Error saving results to S3: {e}")
    
    def run_pipeline(self):
        """Run the complete dual CatBoost pipeline"""
        try:
            self.logger.info("="*80)
            self.logger.info("DUAL CATBOOST PIPELINE STARTING")
            self.logger.info("="*80)
            
            # Step 1: Run both models
            results = self.run_parallel_models()
            
            # Step 2: Analyze results
            analysis = self.analyze_results(results)
            
            # Step 3: Compare performance
            comparison_df, avg_metrics = self.compare_model_performance(results)
            
            # Step 4: Save results
            self.save_results_to_s3(analysis, comparison_df)
            
            # Step 5: Send completion notification
            if analysis["successful_jobs"] == analysis["total_jobs"]:
                self.logger.info("🎉 All models completed successfully!")
            else:
                self.logger.warning(f"⚠️ {analysis['total_jobs'] - analysis['successful_jobs']} models failed")
            
            self.logger.info("="*80)
            self.logger.info("DUAL CATBOOST PIPELINE COMPLETE")
            self.logger.info("="*80)
            
            return analysis, comparison_df
            
        except Exception as e:
            error_msg = f"Dual CatBoost pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            notify_error("dual_catboost_pipeline", error_msg, self.logger)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run dual CatBoost pipeline for Opioid ED and ADE models')
    parser.add_argument('--age-bands', nargs='+', 
                       default=["25-44", "45-54", "55-64", "65-74", "75-84"],
                       help='Age bands to process')
    parser.add_argument('--event-years', nargs='+', type=int,
                       default=[2016, 2017, 2018, 2019, 2020],
                       help='Event years to process')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - Would process:")
        print(f"  Age bands: {args.age_bands}")
        print(f"  Event years: {args.event_years}")
        print(f"  Total jobs: {len(args.age_bands) * len(args.event_years) * 2}")
        print(f"  Max workers: {args.max_workers}")
        return
    
    # Run pipeline
    pipeline = DualCatBoostPipeline(
        age_bands=args.age_bands,
        event_years=args.event_years,
        max_workers=args.max_workers
    )
    
    analysis, comparison_df = pipeline.run_pipeline()
    
    print(f"\nPipeline complete!")
    print(f"Results saved to: {pipeline.results_dir}")
    print(f"Successful models: {analysis['successful_jobs']}/{analysis['total_jobs']}")


if __name__ == "__main__":
    main() 