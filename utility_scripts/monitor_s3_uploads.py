#!/usr/bin/env python3
"""
Local S3 Upload Monitor - Track EC2 visualization uploads to S3

Monitors S3 buckets for dashboard visualization uploads (BupaR, DTW, FP-Growth)
and provides real-time status tracking, comparison reports, and missing file detection.

Usage:
    # Check current S3 status for all visualizations
    python monitor_s3_uploads.py --check-all
    
    # Monitor specific cohort/age_band
    python monitor_s3_uploads.py --cohort opioid_ed --age-band 1-0-12
    
    # Watch mode - continuously monitor for new uploads
    python monitor_s3_uploads.py --watch --interval 30
    
    # Generate detailed report
    python monitor_s3_uploads.py --report --output status/s3_upload_report.json
    
    # Check only specific visualization type
    python monitor_s3_uploads.py --viz-type bupar --check-all
    
    # Find missing uploads
    python monitor_s3_uploads.py --find-missing

Run from repo root. Requires AWS credentials configured.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import boto3
except ImportError:
    print("Error: boto3 required. Install: pip install boto3")
    sys.exit(1)

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from py_helpers.constants import REQUIRED_COHORTS
except ImportError:
    REQUIRED_COHORTS = {
        "opioid_ed": ['0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114'],
        "non_opioid_ed": ['0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114']
    }

# S3 Configuration
S3_DASHBOARD_BUCKET = "jerome-dixon.io"  # Update to your dashboard bucket
S3_DASHBOARD_PREFIX = "vcu/pgx-risk-calculator"

# Visualization types and expected files (age_band in filenames uses underscore, e.g. 25_44)
VIZ_TYPES = {
    "bupar": {
        "prefix": "bupar",
        "files": [
            "{cohort}_{age_band}_overall_activity_frequency.png",
            "{cohort}_{age_band}_activity_frequency_interactive.html",
            "{cohort}_{age_band}_activity_frequency.json",
            "{cohort}_{age_band}_pre_target_activity_frequency.json",
            "{cohort}_{age_band}_post_target_activity_frequency.json",
            "{cohort}_{age_band}_process_matrix.png",
            "{cohort}_{age_band}_trace_explorer_interactive.html",
            "{cohort}_{age_band}_trace_explorer_pre_f1120.png",
            "{cohort}_{age_band}_trace_explorer_pre_hcg.png",
            "{cohort}_{age_band}_frequency_map.png",
        ]
    },
    "dtw": {
        "prefix": "dtw",
        "files": [
            "dtw_trajectory_cluster_interactive_{cohort}_{age_band}.html",
            "dtw_trajectory_analysis_{cohort}_{age_band}.png",
            "dtw_sample_trajectories_{cohort}_{age_band}.png",
            "chart_data.json"
        ]
    },
    "fpgrowth": {
        "prefix": "fpgrowth",
        "files": [
            "{cohort}_{age_band}_drug_name_itemsets_interactive.html",
            "{cohort}_{age_band}_drug_name_network_interactive.html",
            "{cohort}_{age_band}_icd_code_itemsets_interactive.html",
            "{cohort}_{age_band}_icd_code_network_interactive.html",
            "{cohort}_{age_band}_cpt_code_itemsets_interactive.html",
            "{cohort}_{age_band}_cpt_code_network_interactive.html",
            "{cohort}_{age_band}_medical_code_itemsets_interactive.html",
            "{cohort}_{age_band}_medical_code_network_interactive.html"
        ]
    }
}


class S3UploadMonitor:
    """Monitor S3 uploads for dashboard visualizations"""
    
    def __init__(self, bucket: str, prefix: str, profile: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.s3 = session.client("s3")
        self.cache: Dict[str, Dict] = {}
        self.last_check = None
        
    def list_objects(self, prefix: str, max_keys: int = 1000) -> List[Dict]:
        """List objects in S3 with pagination"""
        objects = []
        paginator = self.s3.get_paginator("list_objects_v2")
        
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects.append({
                        "Key": obj["Key"],
                        "Size": obj["Size"],
                        "LastModified": obj["LastModified"],
                        "ETag": obj.get("ETag", "").strip('"')
                    })
                    if len(objects) >= max_keys:
                        return objects
        except Exception as e:
            print(f"Error listing S3 objects: {e}")
            return []
        
        return objects
    
    def check_viz_type(self, viz_type: str, cohort: str, age_band: str) -> Dict:
        """Check if all expected files exist for a visualization type"""
        if viz_type not in VIZ_TYPES:
            return {"error": f"Unknown viz type: {viz_type}"}
        
        config = VIZ_TYPES[viz_type]
        age_band_fname = age_band.replace("-", "_")
        
        # Build expected file list
        expected_files = []
        for filename_template in config["files"]:
            filename = filename_template.format(cohort=cohort, age_band=age_band_fname)
            expected_files.append(filename)
        
        # Check S3
        s3_prefix = f"{self.prefix}/{config['prefix']}/{cohort}/{age_band}/plots/"
        s3_objects = self.list_objects(s3_prefix)
        
        found_files = {}
        for obj in s3_objects:
            filename = obj["Key"].split("/")[-1]
            if filename in expected_files:
                found_files[filename] = {
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "url": f"https://{self.bucket}/{obj['Key']}"
                }
        
        missing_files = [f for f in expected_files if f not in found_files]
        
        return {
            "viz_type": viz_type,
            "cohort": cohort,
            "age_band": age_band,
            "s3_prefix": s3_prefix,
            "expected_count": len(expected_files),
            "found_count": len(found_files),
            "missing_count": len(missing_files),
            "found_files": found_files,
            "missing_files": missing_files,
            "complete": len(missing_files) == 0
        }
    
    def check_all_cohorts(self, viz_types: Optional[List[str]] = None) -> Dict:
        """Check all cohorts and age bands"""
        if viz_types is None:
            viz_types = list(VIZ_TYPES.keys())
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "summary": {
                "total_combinations": 0,
                "complete": 0,
                "incomplete": 0,
                "total_files_expected": 0,
                "total_files_found": 0,
                "total_files_missing": 0
            },
            "by_viz_type": {},
            "details": []
        }
        
        for viz_type in viz_types:
            results["by_viz_type"][viz_type] = {
                "complete": 0,
                "incomplete": 0,
                "files_expected": 0,
                "files_found": 0,
                "files_missing": 0
            }
        
        for cohort, age_bands in REQUIRED_COHORTS.items():
            for age_band in age_bands:
                results["summary"]["total_combinations"] += 1
                
                combo_result = {
                    "cohort": cohort,
                    "age_band": age_band,
                    "viz_types": {}
                }
                
                all_complete = True
                for viz_type in viz_types:
                    check_result = self.check_viz_type(viz_type, cohort, age_band)
                    combo_result["viz_types"][viz_type] = check_result
                    
                    # Update stats
                    vt_stats = results["by_viz_type"][viz_type]
                    if check_result["complete"]:
                        vt_stats["complete"] += 1
                    else:
                        vt_stats["incomplete"] += 1
                        all_complete = False
                    
                    vt_stats["files_expected"] += check_result["expected_count"]
                    vt_stats["files_found"] += check_result["found_count"]
                    vt_stats["files_missing"] += check_result["missing_count"]
                    
                    results["summary"]["total_files_expected"] += check_result["expected_count"]
                    results["summary"]["total_files_found"] += check_result["found_count"]
                    results["summary"]["total_files_missing"] += check_result["missing_count"]
                
                if all_complete:
                    results["summary"]["complete"] += 1
                else:
                    results["summary"]["incomplete"] += 1
                
                combo_result["complete"] = all_complete
                results["details"].append(combo_result)
        
        self.last_check = results
        return results
    
    def print_summary(self, results: Dict) -> None:
        """Print human-readable summary"""
        print("\n" + "=" * 80)
        print(f"S3 Upload Status - {results['timestamp']}")
        print("=" * 80)
        print(f"Bucket: s3://{results['bucket']}/{results['prefix']}")
        print()
        
        summary = results["summary"]
        print("Overall Summary:")
        print(f"  Total cohort/age_band combinations: {summary['total_combinations']}")
        print(f"  Complete: {summary['complete']} ({summary['complete']/summary['total_combinations']*100:.1f}%)")
        print(f"  Incomplete: {summary['incomplete']} ({summary['incomplete']/summary['total_combinations']*100:.1f}%)")
        print(f"  Files: {summary['total_files_found']}/{summary['total_files_expected']} found")
        print(f"  Missing: {summary['total_files_missing']} files")
        print()
        
        print("By Visualization Type:")
        for viz_type, stats in results["by_viz_type"].items():
            total = stats["complete"] + stats["incomplete"]
            pct = stats["complete"] / total * 100 if total > 0 else 0
            print(f"  {viz_type.upper()}:")
            print(f"    Complete: {stats['complete']}/{total} ({pct:.1f}%)")
            print(f"    Files: {stats['files_found']}/{stats['files_expected']} found, {stats['files_missing']} missing")
        print()
        
        # Show incomplete combinations
        incomplete = [d for d in results["details"] if not d["complete"]]
        if incomplete:
            print(f"Incomplete Combinations ({len(incomplete)}):")
            for detail in incomplete[:10]:  # Show first 10
                missing_types = [vt for vt, info in detail["viz_types"].items() if not info["complete"]]
                print(f"  {detail['cohort']} / {detail['age_band']}: {', '.join(missing_types)}")
                for vt in missing_types:
                    info = detail["viz_types"][vt]
                    if info["missing_files"]:
                        print(f"    Missing {vt}: {info['missing_count']} files")
            if len(incomplete) > 10:
                print(f"  ... and {len(incomplete) - 10} more")
        else:
            print("✅ All combinations complete!")
        print()
    
    def find_missing(self) -> Dict:
        """Find all missing uploads across all cohorts"""
        results = self.check_all_cohorts()
        
        missing_by_type = defaultdict(list)
        for detail in results["details"]:
            for viz_type, info in detail["viz_types"].items():
                if not info["complete"]:
                    missing_by_type[viz_type].append({
                        "cohort": detail["cohort"],
                        "age_band": detail["age_band"],
                        "missing_files": info["missing_files"],
                        "missing_count": info["missing_count"]
                    })
        
        return {
            "timestamp": results["timestamp"],
            "total_incomplete": results["summary"]["incomplete"],
            "total_missing_files": results["summary"]["total_files_missing"],
            "by_viz_type": dict(missing_by_type)
        }
    
    def watch(self, interval: int = 30, viz_types: Optional[List[str]] = None) -> None:
        """Continuously monitor S3 for changes"""
        print(f"Starting watch mode (checking every {interval} seconds)...")
        print("Press Ctrl+C to stop")
        print()
        
        previous_state = None
        
        try:
            while True:
                current_state = self.check_all_cohorts(viz_types)
                
                # Detect changes
                if previous_state:
                    prev_found = previous_state["summary"]["total_files_found"]
                    curr_found = current_state["summary"]["total_files_found"]
                    
                    if curr_found > prev_found:
                        new_files = curr_found - prev_found
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🆕 {new_files} new file(s) uploaded!")
                        
                        # Find which combinations changed
                        for i, detail in enumerate(current_state["details"]):
                            prev_detail = previous_state["details"][i]
                            for vt, info in detail["viz_types"].items():
                                prev_info = prev_detail["viz_types"][vt]
                                if info["found_count"] > prev_info["found_count"]:
                                    diff = info["found_count"] - prev_info["found_count"]
                                    print(f"  {detail['cohort']} / {detail['age_band']} - {vt}: +{diff} files")
                    elif curr_found < prev_found:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Warning: File count decreased!")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] No changes ({curr_found}/{current_state['summary']['total_files_expected']} files)")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial check: {current_state['summary']['total_files_found']}/{current_state['summary']['total_files_expected']} files found")
                
                previous_state = current_state
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor S3 uploads for dashboard visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Actions
    parser.add_argument("--check-all", action="store_true",
                       help="Check status for all cohorts/age_bands")
    parser.add_argument("--find-missing", action="store_true",
                       help="Find all missing uploads")
    parser.add_argument("--watch", action="store_true",
                       help="Continuously monitor for changes")
    
    # Filters
    parser.add_argument("--cohort", choices=["opioid_ed", "non_opioid_ed"],
                       help="Check specific cohort")
    parser.add_argument("--age-band",
                       help="Check specific age band (e.g., 1-0-12)")
    parser.add_argument("--viz-type", choices=list(VIZ_TYPES.keys()),
                       help="Check specific visualization type")
    
    # Options
    parser.add_argument("--interval", type=int, default=30,
                       help="Watch mode check interval in seconds (default: 30)")
    parser.add_argument("--output", type=Path,
                       help="Save report to JSON file")
    parser.add_argument("--profile",
                       help="AWS CLI profile name")
    parser.add_argument("--bucket", default=S3_DASHBOARD_BUCKET,
                       help=f"S3 bucket (default: {S3_DASHBOARD_BUCKET})")
    parser.add_argument("--prefix", default=S3_DASHBOARD_PREFIX,
                       help=f"S3 prefix (default: {S3_DASHBOARD_PREFIX})")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = S3UploadMonitor(args.bucket, args.prefix, args.profile)
    
    # Determine viz types to check
    viz_types = [args.viz_type] if args.viz_type else None
    
    # Execute actions
    if args.watch:
        monitor.watch(args.interval, viz_types)
    
    elif args.find_missing:
        missing = monitor.find_missing()
        print("\n" + "=" * 80)
        print(f"Missing Uploads Report - {missing['timestamp']}")
        print("=" * 80)
        print(f"Total incomplete combinations: {missing['total_incomplete']}")
        print(f"Total missing files: {missing['total_missing_files']}")
        print()
        
        for viz_type, items in missing["by_viz_type"].items():
            print(f"{viz_type.upper()} - {len(items)} incomplete combination(s):")
            for item in items:
                print(f"  {item['cohort']} / {item['age_band']}: {item['missing_count']} missing")
                for filename in item["missing_files"][:3]:
                    print(f"    - {filename}")
                if len(item["missing_files"]) > 3:
                    print(f"    ... and {len(item['missing_files']) - 3} more")
            print()
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(missing, f, indent=2, default=str)
            print(f"Report saved to: {args.output}")
    
    elif args.check_all:
        results = monitor.check_all_cohorts(viz_types)
        monitor.print_summary(results)
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Full report saved to: {args.output}")
    
    elif args.cohort and args.age_band:
        viz_types_to_check = [args.viz_type] if args.viz_type else list(VIZ_TYPES.keys())
        
        print(f"\nChecking {args.cohort} / {args.age_band}")
        print("=" * 60)
        
        for viz_type in viz_types_to_check:
            result = monitor.check_viz_type(viz_type, args.cohort, args.age_band)
            status = "✅ Complete" if result["complete"] else f"❌ Incomplete ({result['missing_count']} missing)"
            print(f"\n{viz_type.upper()}: {status}")
            print(f"  Found: {result['found_count']}/{result['expected_count']} files")
            print(f"  S3 prefix: s3://{monitor.bucket}/{result['s3_prefix']}")
            
            if result["missing_files"]:
                print(f"  Missing files:")
                for filename in result["missing_files"]:
                    print(f"    - {filename}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python monitor_s3_uploads.py --check-all")
        print("  python monitor_s3_uploads.py --watch --interval 30")
        print("  python monitor_s3_uploads.py --cohort opioid_ed --age-band 1-0-12")
        print("  python monitor_s3_uploads.py --find-missing --output status/missing_uploads.json")


if __name__ == "__main__":
    main()
