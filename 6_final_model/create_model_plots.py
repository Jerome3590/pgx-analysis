#!/usr/bin/env python3
"""
Create visualizations for final model feature importance analysis.

This script creates the same 4 plots used in feature importance analysis:
1. Top 50 Features Bar Chart (Scaled Importance)
2. Top 50 Features with Recall Confidence Intervals
3. Normalized vs Scaled Importance Comparison
4. Feature Categories Distribution

Cross-platform compatible: Works on both Linux EC2 and Windows.
"""

import os
import sys
import platform
from pathlib import Path

# Add project root to path
if '__file__' in globals():
    project_root = Path(__file__).parent.parent
else:
    # Running from notebook or interactive mode
    project_root = Path(os.getcwd())
    if project_root.name == "8_final_model":
        project_root = project_root.parent
    elif "pgx-analysis" in str(project_root):
        # Find pgx-analysis in path
        for parent in project_root.parents:
            if parent.name == "pgx-analysis":
                project_root = parent
                break

sys.path.insert(0, str(project_root))

from py_helpers.create_feature_importance_visualizations import create_feature_importance_plots

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create final model feature importance visualizations"
    )
    parser.add_argument(
        '--cohort-name',
        type=str,
        required=True,
        help='Cohort name (e.g., opioid_ed)'
    )
    parser.add_argument(
        '--age-band',
        type=str,
        required=True,
        help='Age band (e.g., 0-12)'
    )
    parser.add_argument(
        '--feature-file',
        type=str,
        default=None,
        help='Path to feature importance CSV (default: auto-detect from outputs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: outputs/{cohort}/{age_band}/plots)'
    )
    parser.add_argument(
        '--event-year',
        type=int,
        default=2019,
        help='Event year for titles (default: 2019)'
    )
    parser.add_argument(
        '--no-s3-upload',
        action='store_true',
        help='Skip S3 upload'
    )
    
    args = parser.parse_args()
    
    # Auto-detect feature file if not provided
    if args.feature_file is None:
        age_band_fname = args.age_band.replace("-", "_")
        feature_file = (
            project_root
            / "8_final_model"
            / "outputs"
            / args.cohort_name
            / age_band_fname
            / f"{args.cohort_name}_{age_band_fname}_final_feature_importance_aggregated_scaled.csv"
        )
        
        # Fallback to top_50 if aggregated_scaled doesn't exist
        if not feature_file.exists():
            feature_file = (
                project_root
                / "8_final_model"
                / "outputs"
                / args.cohort_name
                / age_band_fname
                / f"{args.cohort_name}_{age_band_fname}_final_feature_importance_top_50.csv"
            )
        
        if not feature_file.exists():
            print(f"[ERROR] Feature importance file not found. Expected:")
            print(f"  - {project_root / '8_final_model' / 'outputs' / args.cohort_name / age_band_fname / f'{args.cohort_name}_{age_band_fname}_final_feature_importance_aggregated_scaled.csv'}")
            print(f"  - {project_root / '8_final_model' / 'outputs' / args.cohort_name / age_band_fname / f'{args.cohort_name}_{age_band_fname}_final_feature_importance_top_50.csv'}")
            print(f"\nPlease run extract_final_feature_importance.py first, or specify --feature-file")
            sys.exit(1)
    else:
        feature_file = Path(args.feature_file)
        if not feature_file.exists():
            print(f"[ERROR] Feature file not found: {feature_file}")
            sys.exit(1)
    
    # Auto-detect output directory if not provided
    if args.output_dir is None:
        age_band_fname = args.age_band.replace("-", "_")
        output_dir = (
            project_root
            / "8_final_model"
            / "outputs"
            / args.cohort_name
            / age_band_fname
        )
    else:
        output_dir = Path(args.output_dir)
    
    print(f"[INFO] Creating final model visualizations...")
    print(f"[INFO] Feature file: {feature_file}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Cohort: {args.cohort_name}, Age Band: {args.age_band}")
    
    # Create plots using the same function as feature importance
    plot_files = create_feature_importance_plots(
        aggregated_file=str(feature_file),
        output_dir=str(output_dir),
        s3_upload=not args.no_s3_upload,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        event_year=args.event_year,
    )
    
    # Update S3 path for final model (different location than feature importance)
    if not args.no_s3_upload:
        import subprocess
        import shutil
        
        aws_cmd = shutil.which("aws")
        if aws_cmd:
            age_band_fname = args.age_band.replace("-", "_")
            s3_base = (
                f"s3://pgxdatalake/gold/final_model/"
                f"{args.cohort_name}/{args.age_band}/plots"
            )
            
            plot_dir = output_dir / "plots"
            print(f"\n[INFO] Uploading plots to S3: {s3_base}")
            
            upload_count = 0
            for plot_name, local_file in plot_files.items():
                s3_path = f"{s3_base}/{Path(local_file).name}"
                
                try:
                    result = subprocess.run(
                        [aws_cmd, 's3', 'cp', local_file, s3_path],
                        capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        print(f"[INFO] Uploaded: {Path(local_file).name}")
                        upload_count += 1
                    else:
                        print(f"[WARNING] Failed to upload: {Path(local_file).name}")
                        if result.stderr:
                            print(f"  Error: {result.stderr}")
                except Exception as e:
                    print(f"[WARNING] Error uploading {Path(local_file).name}: {e}")
            
            if upload_count == len(plot_files):
                print(f"\n[INFO] All {upload_count} plots uploaded to S3")
                print(f"[INFO] S3 Location: {s3_base}")
    
    print(f"\n[INFO] Visualization complete!")
    print(f"[INFO] Plots saved to: {output_dir / 'plots'}")

