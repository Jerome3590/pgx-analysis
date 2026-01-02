#!/usr/bin/env python3
"""
Wrapper script to create visualizations for feature importance analysis.
This Python version replaces the R script for consistency across the workflow.

Cross-platform compatible: Works on both Linux EC2 and Windows.
"""

import os
import sys
import platform
from pathlib import Path

# Add project root to path
# Handle both script execution and notebook execution
if '__file__' in globals():
    project_root = Path(__file__).parent.parent
else:
    # Running from notebook or interactive mode
    project_root = Path(os.getcwd())
    if project_root.name == "3_feature_importance":
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
        description="Create feature importance visualizations"
    )
    parser.add_argument(
        'aggregated_file',
        type=str,
        help='Path to aggregated feature importance CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: same as aggregated_file directory)'
    )
    parser.add_argument(
        '--cohort-name',
        type=str,
        default=None,
        help='Cohort name (default: extracted from filename)'
    )
    parser.add_argument(
        '--age-band',
        type=str,
        default=None,
        help='Age band (default: extracted from filename)'
    )
    parser.add_argument(
        '--event-year',
        type=int,
        default=2019,
        help='Event year (default: 2019)'
    )
    parser.add_argument(
        '--no-s3-upload',
        action='store_true',
        help='Skip S3 upload'
    )
    
    args = parser.parse_args()
    
    create_feature_importance_plots(
        aggregated_file=args.aggregated_file,
        output_dir=args.output_dir,
        s3_upload=not args.no_s3_upload,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        event_year=args.event_year,
    )

