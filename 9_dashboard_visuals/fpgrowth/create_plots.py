#!/usr/bin/env python3
"""
Wrapper script to create visualizations for FP-Growth analysis.
Template: 4_fpgrowth_analysis/create_plots.py (same create_all_fpgrowth_plots, same HTML pattern).
Cross-platform compatible: Works on both Linux EC2 and Windows.
Outputs: 10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band}/plots/
HTML: Production single-file Plotly via py_helpers.create_fpgrowth_visualizations (include_plotlyjs=True).
"""

import os
import sys

from pathlib import Path

# Script lives in 9_dashboard_visuals/fpgrowth; outputs in 10_risk_dashboard/visualizations/fpgrowth
if '__file__' in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = Path(os.getcwd())
    if "pgx-analysis" in str(REPO_ROOT):
        for parent in REPO_ROOT.parents:
            if parent.name == "pgx-analysis":
                REPO_ROOT = parent
                break
project_root = REPO_ROOT  # for py_helpers
sys.path.insert(0, str(REPO_ROOT))

from py_helpers.create_fpgrowth_visualizations import create_all_fpgrowth_plots

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create FP-Growth visualizations"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='10_risk_dashboard/visualizations/fpgrowth/outputs',
        help='Base directory containing FP-Growth outputs'
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
        '--event-year',
        type=str,
        default='train',
        help='Event year (default: train)'
    )
    parser.add_argument(
        '--split-type',
        type=str,
        default='combined',
        choices=['combined', 'target'],
        help='Split type (default: combined)'
    )
    parser.add_argument(
        '--item-types',
        type=str,
        nargs='+',
        default=None,
        help='Item types to process (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: {base_dir}/plots)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=30,
        help='Number of top itemsets to display (default: 30)'
    )
    parser.add_argument(
        '--no-s3-upload',
        action='store_true',
        help='Skip S3 upload'
    )
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default=None,
        help='S3 bucket for uploads (default: from env S3_DASHBOARD_BUCKET or jerome-dixon.io)'
    )
    parser.add_argument(
        '--s3-prefix',
        type=str,
        default=None,
        help='S3 key prefix (default: {S3_DASHBOARD_PREFIX}/visualizations/fpgrowth)'
    )
    parser.add_argument(
        '--code-mapping',
        type=str,
        default=None,
        help='Path to CSV with code,description for viewable labels (default: 9_dashboard_visuals/fpgrowth/code_mappings/fpgrowth_code_descriptions.csv)'
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(args.base_dir) / "plots"
    
    # Use dashboard bucket/prefix so FP-Growth assets are where the dashboard is rendered from
    s3_bucket = args.s3_bucket or os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    s3_prefix = args.s3_prefix
    if s3_prefix is None:
        dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
        use_builds = (os.environ.get("S3_VISUALIZATIONS_BUILDS", "") or "").strip().lower() in ("1", "true", "yes")
        builds_suffix = "/builds" if use_builds else ""
        s3_prefix = f"{dashboard_prefix.rstrip('/')}/visualizations/fpgrowth{builds_suffix}"
    
    # Create visualizations for all item types; upload to S3 when not --no-s3-upload
    result = create_all_fpgrowth_plots(
        base_dir=args.base_dir,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        event_year=args.event_year,
        split_type=args.split_type,
        item_types=args.item_types,
        output_dir=str(args.output_dir),
        s3_upload=not args.no_s3_upload,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        top_n=args.top_n,
        code_mapping_path=args.code_mapping,
    )
    all_plots = result.get("plots", result) if isinstance(result, dict) else result
    total_plots = sum(len(plots) for plots in all_plots.values())
    
    print(f"\n{'=' * 70}")
    print("Visualization Summary")
    print(f"{'=' * 70}")
    print(f"Total plots created: {total_plots}")
    for item_type, plots in all_plots.items():
        print(f"\n{item_type}: {len(plots)} plots")
        for plot_name in plots.keys():
            print(f"  - {plot_name}")
    if result.get("s3_urls"):
        print(f"\nUploaded to dashboard bucket: s3://{s3_bucket}/{s3_prefix}/<cohort>/<age_band>/plots/")

