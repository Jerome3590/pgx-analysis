#!/usr/bin/env python3
"""
ADE CatBoost Target Script
Runs ADE CatBoost models for all age bands in parallel, using existing helper files and DuckDB utilities.
"""

import os
import sys
import json
import argparse
import subprocess
import concurrent.futures
import pandas as pd
from datetime import datetime
from pathlib import Path

# Set root of project
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run ADE CatBoost models by age band')
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
        print(f"  Total jobs: {len(args.age_bands) * len(args.event_years)}")
        print(f"  Max workers: {args.max_workers}")
        return
    
    # Run pipeline
    pipeline = ADECatBoostTarget(
        event_years=args.event_years,
        max_workers=args.max_workers
    )
    
    analysis = pipeline.run_pipeline(args.age_bands)
    
    print(f"\nADE pipeline complete!")
    print(f"Results saved to: {pipeline.results_dir}")
    print(f"Successful models: {analysis['successful_jobs']}/{analysis['total_jobs']}")


if __name__ == "__main__":
    main() 