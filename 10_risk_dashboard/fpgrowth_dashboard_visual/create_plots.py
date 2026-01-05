#!/usr/bin/env python3
"""
Wrapper script to create visualizations for FP-Growth analysis.
Cross-platform compatible: Works on both Linux EC2 and Windows.
"""

import os
import sys

from pathlib import Path

# Add project root to path
if '__file__' in globals():
    project_root = Path(__file__).parent.parent
else:
    # Running from notebook or interactive mode
    project_root = Path(os.getcwd())
    if project_root.name == "10b_fpgrowth_dashboard_visual":
        project_root = project_root.parent
    elif "pgx-analysis" in str(project_root):
        # Find pgx-analysis in path
        for parent in project_root.parents:
            if parent.name == "pgx-analysis":
                project_root = parent
                break

sys.path.insert(0, str(project_root))

from py_helpers.create_fpgrowth_visualizations import create_all_fpgrowth_plots

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create FP-Growth visualizations"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='10b_fpgrowth_dashboard_visual/outputs',
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
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = Path(args.base_dir) / "plots"
    
    # Create visualizations for all item types
    all_plots = create_all_fpgrowth_plots(
        base_dir=args.base_dir,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        event_year=args.event_year,
        split_type=args.split_type,
        item_types=args.item_types,
        output_dir=str(args.output_dir),
        s3_upload=not args.no_s3_upload,
        top_n=args.top_n,
    )
    
    print(f"\n{'=' * 70}")
    print("Visualization Summary")
    print(f"{'=' * 70}")
    total_plots = sum(len(plots) for plots in all_plots.values())
    print(f"Total plots created: {total_plots}")
    for item_type, plots in all_plots.items():
        print(f"\n{item_type}: {len(plots)} plots")
        for plot_name in plots.keys():
            print(f"  - {plot_name}")

