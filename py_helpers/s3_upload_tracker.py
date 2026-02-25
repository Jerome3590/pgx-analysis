"""
Local tracking system for S3 uploads from notebooks.

This module provides utilities to:
1. Track all uploads to S3 in a local JSON log
2. Query upload status by cohort, age_band, visualization type
3. Generate upload status reports
4. Identify missing uploads
5. Monitor EC2 uploads in real-time
6. Compare local state with S3 state
7. Generate Progress reports during batch uploads
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3UploadTracker:
    """Track S3 uploads locally for monitoring and validation."""
    
    def __init__(self, tracker_file: str = "status/s3_upload_tracker.json"):
        """
        Initialize tracker.
        
        Args:
            tracker_file: Path to JSON tracking file (default: status/s3_upload_tracker.json)
        """
        self.tracker_file = Path(tracker_file)
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        self.uploads: Dict[str, Any] = self._load_tracker()
    
    def _load_tracker(self) -> Dict[str, Any]:
        """Load tracking data from JSON file."""
        if not self.tracker_file.exists():
            return {
                "uploads": [],
                "last_updated": None,
                "summary": {}
            }
        
        try:
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load tracker file: {e}, starting fresh")
            return {
                "uploads": [],
                "last_updated": None,
                "summary": {}
            }
    
    def _save_tracker(self) -> None:
        """Save tracking data to JSON file."""
        self.uploads["last_updated"] = datetime.utcnow().isoformat()
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.uploads, f, indent=2)
    
    def log_upload(
        self,
        local_path: str,
        s3_path: str,
        visualization_type: str,
        cohort: str,
        age_band: str,
        item_type: Optional[str] = None,
        file_size_mb: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an S3 upload attempt.
        
        Args:
            local_path: Local file path
            s3_path: S3 destination path
            visualization_type: Type (bupar, dtw, fpgrowth, feature_importance)
            cohort: Cohort name (opioid_ed, non_opioid_ed)
            age_band: Age band (e.g., "1-0-12")
            item_type: Item type for FP-Growth (drug_name, icd_code, etc.)
            file_size_mb: File size in MB
            success: Whether upload succeeded
            error: Error message if failed
            metadata: Additional metadata
        """
        upload_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "local_path": str(local_path),
            "s3_path": s3_path,
            "visualization_type": visualization_type,
            "cohort": cohort,
            "age_band": age_band,
            "item_type": item_type,
            "file_size_mb": file_size_mb,
            "success": success,
            "error": error,
            "metadata": metadata or {}
        }
        
        self.uploads["uploads"].append(upload_record)
        self._update_summary()
        self._save_tracker()
    
    def _update_summary(self) -> None:
        """Update summary statistics."""
        uploads = self.uploads["uploads"]
        
        summary = {
            "total_uploads": len(uploads),
            "successful_uploads": sum(1 for u in uploads if u["success"]),
            "failed_uploads": sum(1 for u in uploads if not u["success"]),
            "by_visualization_type": {},
            "by_cohort": {},
            "by_age_band": {},
            "total_size_mb": sum(u.get("file_size_mb", 0) or 0 for u in uploads),
        }
        
        # Group by visualization type
        for upload in uploads:
            viz_type = upload["visualization_type"]
            if viz_type not in summary["by_visualization_type"]:
                summary["by_visualization_type"][viz_type] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                }
            summary["by_visualization_type"][viz_type]["total"] += 1
            if upload["success"]:
                summary["by_visualization_type"][viz_type]["successful"] += 1
            else:
                summary["by_visualization_type"][viz_type]["failed"] += 1
        
        # Group by cohort
        for upload in uploads:
            cohort = upload["cohort"]
            if cohort not in summary["by_cohort"]:
                summary["by_cohort"][cohort] = {"total": 0, "successful": 0}
            summary["by_cohort"][cohort]["total"] += 1
            if upload["success"]:
                summary["by_cohort"][cohort]["successful"] += 1
        
        # Group by age band
        for upload in uploads:
            age_band = upload["age_band"]
            if age_band not in summary["by_age_band"]:
                summary["by_age_band"][age_band] = {"total": 0, "successful": 0}
            summary["by_age_band"][age_band]["total"] += 1
            if upload["success"]:
                summary["by_age_band"][age_band]["successful"] += 1
        
        self.uploads["summary"] = summary
    
    def get_uploads(
        self,
        visualization_type: Optional[str] = None,
        cohort: Optional[str] = None,
        age_band: Optional[str] = None,
        item_type: Optional[str] = None,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query uploads with filters.
        
        Args:
            visualization_type: Filter by visualization type
            cohort: Filter by cohort
            age_band: Filter by age band
            item_type: Filter by item type
            success_only: Only return successful uploads
        
        Returns:
            List of matching upload records
        """
        uploads = self.uploads["uploads"]
        
        if visualization_type:
            uploads = [u for u in uploads if u["visualization_type"] == visualization_type]
        if cohort:
            uploads = [u for u in uploads if u["cohort"] == cohort]
        if age_band:
            uploads = [u for u in uploads if u["age_band"] == age_band]
        if item_type:
            uploads = [u for u in uploads if u.get("item_type") == item_type]
        if success_only:
            uploads = [u for u in uploads if u["success"]]
        
        return uploads
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.uploads.get("summary", {})
    
    def print_summary(self) -> None:
        """Print a formatted summary report."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("S3 UPLOAD TRACKER SUMMARY")
        print("="*80)
        print(f"\nLast Updated: {self.uploads.get('last_updated', 'Never')}")
        print(f"\nTotal Uploads: {summary.get('total_uploads', 0)}")
        print(f"  ✓ Successful: {summary.get('successful_uploads', 0)}")
        print(f"  ✗ Failed: {summary.get('failed_uploads', 0)}")
        print(f"\nTotal Size: {summary.get('total_size_mb', 0):.2f} MB")
        
        # By visualization type
        if summary.get("by_visualization_type"):
            print("\n" + "-"*80)
            print("BY VISUALIZATION TYPE:")
            print("-"*80)
            for viz_type, stats in sorted(summary["by_visualization_type"].items()):
                print(f"\n  {viz_type}:")
                print(f"    Total: {stats['total']}")
                print(f"    ✓ Successful: {stats['successful']}")
                print(f"    ✗ Failed: {stats['failed']}")
        
        # By cohort
        if summary.get("by_cohort"):
            print("\n" + "-"*80)
            print("BY COHORT:")
            print("-"*80)
            for cohort, stats in sorted(summary["by_cohort"].items()):
                print(f"\n  {cohort}:")
                print(f"    Total: {stats['total']}")
                print(f"    ✓ Successful: {stats['successful']}")
        
        # By age band
        if summary.get("by_age_band"):
            print("\n" + "-"*80)
            print("BY AGE BAND:")
            print("-"*80)
            for age_band, stats in sorted(summary["by_age_band"].items()):
                print(f"  {age_band}: {stats['successful']}/{stats['total']} successful")
        
        print("\n" + "="*80 + "\n")
    
    def get_missing_uploads(
        self,
        expected_cohorts: List[str],
        expected_age_bands: List[str],
        expected_viz_types: List[str],
        expected_item_types: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Identify missing uploads based on expected combinations.
        
        Args:
            expected_cohorts: List of expected cohorts
            expected_age_bands: List of expected age bands
            expected_viz_types: List of expected visualization types
            expected_item_types: List of expected item types (for FP-Growth)
        
        Returns:
            Dictionary of missing uploads by visualization type
        """
        missing = defaultdict(list)
        
        # Get successful uploads
        successful = self.get_uploads(success_only=True)
        
        for viz_type in expected_viz_types:
            if viz_type == "fpgrowth" and expected_item_types:
                # FP-Growth requires item_type dimension
                for cohort in expected_cohorts:
                    for age_band in expected_age_bands:
                        for item_type in expected_item_types:
                            found = any(
                                u["visualization_type"] == viz_type
                                and u["cohort"] == cohort
                                and u["age_band"] == age_band
                                and u.get("item_type") == item_type
                                for u in successful
                            )
                            if not found:
                                missing[viz_type].append(f"{cohort}/{age_band}/{item_type}")
            else:
                # Other viz types just need cohort + age_band
                for cohort in expected_cohorts:
                    for age_band in expected_age_bands:
                        found = any(
                            u["visualization_type"] == viz_type
                            and u["cohort"] == cohort
                            and u["age_band"] == age_band
                            for u in successful
                        )
                        if not found:
                            missing[viz_type].append(f"{cohort}/{age_band}")
        
        return dict(missing)
    
    def print_missing_uploads(
        self,
        expected_cohorts: List[str],
        expected_age_bands: List[str],
        expected_viz_types: List[str],
        expected_item_types: Optional[List[str]] = None
    ) -> None:
        """Print a report of missing uploads."""
        missing = self.get_missing_uploads(
            expected_cohorts,
            expected_age_bands,
            expected_viz_types,
            expected_item_types
        )
        
        print("\n" + "="*80)
        print("MISSING UPLOADS REPORT")
        print("="*80)
        
        total_missing = sum(len(v) for v in missing.values())
        
        if total_missing == 0:
            print("\n✓ All expected uploads are present!")
        else:
            print(f"\n✗ Total Missing: {total_missing}")
            for viz_type, items in sorted(missing.items()):
                print(f"\n  {viz_type}: {len(items)} missing")
                for item in items[:10]:  # Show first 10
                    print(f"    - {item}")
                if len(items) > 10:
                    print(f"    ... and {len(items) - 10} more")
        
        print("\n" + "="*80 + "\n")
    
    def clear_tracker(self) -> None:
        """Clear all tracking data (use with caution)."""
        self.uploads = {
            "uploads": [],
            "last_updated": None,
            "summary": {}
        }
        self._save_tracker()
        print("✓ Tracker cleared")
    
    def check_s3_state(
        self,
        s3_bucket: str,
        s3_prefix: str,
        cohorts: Optional[List[str]] = None,
        age_bands: Optional[List[str]] = None,
        visualization_types: Optional[List[str]] = None,
        aws_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare local tracking data with actual S3 state.
        
        Args:
            s3_bucket: S3 bucket name
            s3_prefix: S3 key prefix (e.g., "vcu/pgx-risk-calculator")
            cohorts: Filter by cohorts (default: all tracked)
            age_bands: Filter by age bands (default: all tracked)
            visualization_types: Filter by viz types (default: all tracked)
            aws_profile: AWS CLI profile name
        
        Returns:
            Dictionary with comparison results
        """
        if not BOTO3_AVAILABLE:
            return {"error": "boto3 not available, install with: pip install boto3"}
        
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        s3 = session.client("s3")
        
        # Get tracked uploads
        tracked_uploads = self.get_uploads(success_only=True)
        
        # Filter by criteria
        if cohorts:
            tracked_uploads = [u for u in tracked_uploads if u["cohort"] in cohorts]
        if age_bands:
            tracked_uploads = [u for u in tracked_uploads if u["age_band"] in age_bands]
        if visualization_types:
            tracked_uploads = [u for u in tracked_uploads if u["visualization_type"] in visualization_types]
        
        # Build expected S3 keys from tracked uploads
        tracked_keys = set()
        for upload in tracked_uploads:
            s3_path = upload["s3_path"]
            # Extract key from s3://bucket/key format
            if s3_path.startswith("s3://"):
                key = "/".join(s3_path.split("/")[3:])
                tracked_keys.add(key)
        
        # List objects in S3
        s3_keys = set()
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    s3_keys.add(obj["Key"])
        except Exception as e:
            return {"error": f"Failed to list S3 objects: {e}"}
        
        # Compare
        in_tracker_not_s3 = tracked_keys - s3_keys
        in_s3_not_tracker = s3_keys - tracked_keys
        in_both = tracked_keys & s3_keys
        
        return {
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "total_tracked": len(tracked_keys),
            "total_in_s3": len(s3_keys),
            "matched": len(in_both),
            "tracked_but_missing_from_s3": list(in_tracker_not_s3),
            "in_s3_but_not_tracked": list(in_s3_not_tracker),
            "match_percentage": (len(in_both) / len(tracked_keys) * 100) if tracked_keys else 0
        }
    
    def print_s3_comparison(
        self,
        s3_bucket: str,
        s3_prefix: str,
        cohorts: Optional[List[str]] = None,
        age_bands: Optional[List[str]] = None,
        visualization_types: Optional[List[str]] = None,
        aws_profile: Optional[str] = None
    ) -> None:
        """Print formatted S3 state comparison report."""
        result = self.check_s3_state(
            s3_bucket, s3_prefix, cohorts, age_bands, visualization_types, aws_profile
        )
        
        if "error" in result:
            print(f"\n✗ Error: {result['error']}\n")
            return
        
        print("\n" + "="*80)
        print("S3 STATE COMPARISON REPORT")
        print("="*80)
        print(f"\nS3 Bucket: {result['s3_bucket']}")
        print(f"S3 Prefix: {result['s3_prefix']}")
        print(f"\nTotal Tracked Uploads: {result['total_tracked']}")
        print(f"Total Files in S3: {result['total_in_s3']}")
        print(f"Matched: {result['matched']} ({result['match_percentage']:.1f}%)")
        
        missing_from_s3 = result["tracked_but_missing_from_s3"]
        if missing_from_s3:
            print(f"\n⚠ Tracked but MISSING from S3: {len(missing_from_s3)}")
            for key in missing_from_s3[:10]:
                print(f"  - {key}")
            if len(missing_from_s3) > 10:
                print(f"  ... and {len(missing_from_s3) - 10} more")
        else:
            print("\n✓ All tracked uploads are present in S3")
        
        not_tracked = result["in_s3_but_not_tracked"]
        if not_tracked:
            print(f"\nℹ In S3 but NOT tracked: {len(not_tracked)}")
            for key in not_tracked[:10]:
                print(f"  - {key}")
            if len(not_tracked) > 10:
                print(f"  ... and {len(not_tracked) - 10} more")
        
        print("\n" + "="*80 + "\n")
    
    def get_progress_stats(
        self,
        expected_total: int,
        visualization_type: Optional[str] = None,
        cohort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get progress statistics for batch uploads.
        
        Args:
            expected_total: Total number of expected uploads
            visualization_type: Filter by visualization type
            cohort: Filter by cohort
        
        Returns:
            Dictionary with progress stats
        """
        uploads = self.get_uploads(
            visualization_type=visualization_type,
            cohort=cohort,
            success_only=True
        )
        
        completed = len(uploads)
        percentage = (completed / expected_total * 100) if expected_total > 0 else 0
        
        # Calculate recent rate (uploads per minute from last 10)
        recent_uploads = sorted(uploads, key=lambda x: x["timestamp"], reverse=True)[:10]
        if len(recent_uploads) >= 2:
            first_time = datetime.fromisoformat(recent_uploads[-1]["timestamp"])
            last_time = datetime.fromisoformat(recent_uploads[0]["timestamp"])
            time_diff = (last_time - first_time).total_seconds() / 60  # minutes
            if time_diff > 0:
                rate = len(recent_uploads) / time_diff
            else:
                rate = 0
        else:
            rate = 0
        
        # Estimate time remaining
        remaining = expected_total - completed
        if rate > 0:
            eta_minutes = remaining / rate
        else:
            eta_minutes = None
        
        return {
            "completed": completed,
            "expected_total": expected_total,
            "remaining": remaining,
            "percentage": percentage,
            "uploads_per_minute": rate,
            "eta_minutes": eta_minutes
        }
    
    def print_progress(
        self,
        expected_total: int,
        visualization_type: Optional[str] = None,
        cohort: Optional[str] = None,
        compact: bool = False
    ) -> None:
        """
        Print progress report for batch uploads.
        
        Args:
            expected_total: Total number of expected uploads
            visualization_type: Filter by visualization type
            cohort: Filter by cohort
            compact: If True, print single-line compact format
        """
        stats = self.get_progress_stats(expected_total, visualization_type, cohort)
        
        if compact:
            # Single line format for frequent updates
            bar_length = 40
            filled = int(bar_length * stats["percentage"] / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            eta_str = f"{stats['eta_minutes']:.1f}m" if stats['eta_minutes'] else "?"
            rate_str = f"{stats['uploads_per_minute']:.1f}/min" if stats['uploads_per_minute'] > 0 else "—"
            
            print(
                f"\r[{bar}] {stats['completed']}/{stats['expected_total']} "
                f"({stats['percentage']:.1f}%) | Rate: {rate_str} | ETA: {eta_str}  ",
                end="",
                flush=True
            )
        else:
            # Multi-line detailed format
            print("\n" + "-"*80)
            print("UPLOAD PROGRESS")
            print("-"*80)
            print(f"Completed: {stats['completed']}/{stats['expected_total']} ({stats['percentage']:.1f}%)")
            print(f"Remaining: {stats['remaining']}")
            
            if stats['uploads_per_minute'] > 0:
                print(f"Rate: {stats['uploads_per_minute']:.2f} uploads/minute")
            
            if stats['eta_minutes']:
                hours = int(stats['eta_minutes'] // 60)
                minutes = int(stats['eta_minutes'] % 60)
                if hours > 0:
                    print(f"ETA: {hours}h {minutes}m")
                else:
                    print(f"ETA: {minutes}m")
            
            # Progress bar
            bar_length = 50
            filled = int(bar_length * stats["percentage"] / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\n[{bar}] {stats['percentage']:.1f}%")
            print("-"*80 + "\n")


class BatchUploadMonitor:
    """Monitor batch upload progress in real-time."""
    
    def __init__(
        self,
        tracker: S3UploadTracker,
        expected_total: int,
        visualization_type: Optional[str] = None,
        cohort: Optional[str] = None,
        update_interval: int = 5
    ):
        """
        Initialize batch upload monitor.
        
        Args:
            tracker: S3UploadTracker instance
            expected_total: Total number of expected uploads
            visualization_type: Filter by visualization type
            cohort: Filter by cohort
            update_interval: Number of uploads between progress updates
        """
        self.tracker = tracker
        self.expected_total = expected_total
        self.visualization_type = visualization_type
        self.cohort = cohort
        self.update_interval = update_interval
        self.uploads_since_update = 0
        self.start_time = datetime.utcnow()
    
    def on_upload(self, **kwargs) -> None:
        """
        Callback for each upload. Pass this to upload functions.
        
        Args:
            **kwargs: Upload parameters (passed to tracker.log_upload)
        """
        # Log the upload
        self.tracker.log_upload(**kwargs)
        
        # Update progress display
        self.uploads_since_update += 1
        if self.uploads_since_update >= self.update_interval:
            self.tracker.print_progress(
                expected_total=self.expected_total,
                visualization_type=self.visualization_type,
                cohort=self.cohort,
                compact=True
            )
            self.uploads_since_update = 0
    
    def finish(self) -> None:
        """Call when batch upload completes."""
        print()  # New line after compact progress
        elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 60
        self.tracker.print_progress(
            expected_total=self.expected_total,
            visualization_type=self.visualization_type,
            cohort=self.cohort,
            compact=False
        )
        print(f"✓ Batch upload completed in {elapsed:.1f} minutes\n")


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)


# Convenience function for quick status check
def print_upload_status(tracker_file: str = "status/s3_upload_tracker.json") -> None:
    """Print upload status summary."""
    tracker = S3UploadTracker(tracker_file)
    tracker.print_summary()
    
    # Check for missing uploads (standard configuration)
    tracker.print_missing_uploads(
        expected_cohorts=["opioid_ed", "non_opioid_ed"],
        expected_age_bands=["1-0-12", "1-13-24", "1-25-44", "1-45-54", "1-55-64", "1-65-74", "1-75-84", "1-85-114"],
        expected_viz_types=["bupar", "dtw", "fpgrowth"],
        expected_item_types=["drug_name"]  # Research focus: drugs only for FP-Growth
    )


def check_s3_sync_status(
    tracker_file: str = "status/s3_upload_tracker.json",
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    aws_profile: Optional[str] = None
) -> None:
    """
    Check S3 synchronization status (compare tracked uploads with actual S3 state).
    
    Args:
        tracker_file: Path to tracker JSON file
        s3_bucket: S3 bucket name (default: from env S3_DASHBOARD_BUCKET)
        s3_prefix: S3 prefix (default: from env S3_DASHBOARD_PREFIX or "vcu/pgx-risk-calculator")
        aws_profile: AWS CLI profile name
    """
    if not s3_bucket:
        s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    
    if not s3_prefix:
        s3_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    
    tracker = S3UploadTracker(tracker_file)
    
    # Print summary first
    tracker.print_summary()
    
    # Then print S3 comparison
    tracker.print_s3_comparison(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        aws_profile=aws_profile
    )


if __name__ == "__main__":
    import sys
    
    # Command-line interface
    if len(sys.argv) > 1 and sys.argv[1] == "check-s3":
        # python s3_upload_tracker.py check-s3 [--profile PROFILE]
        profile = None
        if "--profile" in sys.argv:
            idx = sys.argv.index("--profile")
            if idx + 1 < len(sys.argv):
                profile = sys.argv[idx + 1]
        check_s3_sync_status(aws_profile=profile)
    else:
        # Default: just print status
        print_upload_status()
