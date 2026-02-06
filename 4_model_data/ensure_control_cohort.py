#!/usr/bin/env python3
"""
Utility function to ensure control cohort exists with correct 5:1 ratio.

**Performance Optimizations:**
- Uses DuckDB with Parquet for all operations (no pandas memory overhead)
- Single combined query for validation (counts both control and case patients in one pass)
- Direct Parquet-to-Parquet operations via DuckDB COPY ... TO ... (FORMAT PARQUET)
- Efficient columnar reads using read_parquet() with predicate pushdown

This function:
1. Checks if control cohort model_events.parquet exists locally
2. Downloads from S3 if not found locally
3. Validates the control:case ratio (should be ~5:1) using efficient DuckDB Parquet queries
4. Recreates the control cohort if ratio is invalid or file is missing (using DuckDB)
5. Returns the path to the validated control cohort file

Can be called from Python scripts or from R scripts via subprocess.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import duckdb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import DEFAULT_SAMPLE_RATIO, get_cohort_slug
from py_helpers.env_utils import get_model_data_root

try:
    from py_helpers.checkpoint_utils import check_s3_output_exists
except ImportError:
    def check_s3_output_exists(s3_path: str) -> bool:
        """Dummy function if checkpoint_utils not available."""
        return False


def get_control_cohort_path(control_cohort: str, age_band: str) -> Path:
    """Get the path to control cohort model_events.parquet (OS-aware).
    
    New structure: cohorts/input_model_data/cohort_name={slug}/age_band={age_band}/model_events.parquet
    where slug is "opioid" or "polypharmacy" based on age band.
    """
    model_data_root = get_model_data_root()
    # Get cohort slug based on age band: "opioid" for < 65, "polypharmacy" for >= 65
    cohort_slug = get_cohort_slug(age_band)
    # New structure: cohorts/input_model_data/cohort_name={slug}/age_band={age_band}/
    return (
        model_data_root 
        / "cohorts" 
        / "input_model_data"
        / f"cohort_name={cohort_slug}" 
        / f"age_band={age_band}" 
        / "model_events.parquet"
    )


def download_control_cohort_from_s3(control_cohort: str, age_band: str, local_path: Path) -> bool:
    """Download control cohort from S3 if it exists there."""
    import subprocess
    import shutil
    
    # Get cohort slug based on age band: "opioid" for < 65, "polypharmacy" for >= 65
    cohort_slug = get_cohort_slug(age_band)
    # New format: s3://pgxdatalake/gold/cohorts/input_model_data/cohort_name={slug}/
    s3_path = f"s3://pgxdatalake/gold/cohorts/input_model_data/cohort_name={cohort_slug}/age_band={age_band}/model_events.parquet"
    
    # Check if exists in S3
    if not check_s3_output_exists(s3_path):
        return False
    
    print(f"[INFO] Control cohort found in S3. Downloading: {s3_path}")
    
    # Create directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try AWS CLI
    aws_cli = shutil.which("aws")
    if not aws_cli:
        print("[WARN] AWS CLI not found. Cannot download from S3.")
        return False
    
    try:
        result = subprocess.run(
            [aws_cli, "s3", "cp", s3_path, str(local_path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        
        if result.returncode == 0 and local_path.exists():
            print(f"[OK] Successfully downloaded control cohort from S3: {local_path}")
            return True
        else:
            print(f"[WARN] Failed to download from S3: {result.stderr if result.stderr else 'Unknown error'}")
            return False
    except Exception as e:
        print(f"[WARN] Error downloading from S3: {e}")
        return False


def validate_control_cohort_ratio(
    control_cohort_path: Path,
    target_cohort_path: Path,
    expected_ratio: float = DEFAULT_SAMPLE_RATIO,
    tolerance: float = 0.2,
    train_years: list = [2016, 2017, 2018],
) -> Tuple[bool, float, int, int]:
    """
    Validate that control cohort has correct ratio with target cohort.
    
    Uses DuckDB with Parquet for efficient single-query validation.
    
    Returns:
        (is_valid, actual_ratio, n_controls, n_cases)
    """
    if not control_cohort_path.exists() or not target_cohort_path.exists():
        return (False, 0.0, 0, 0)
    
    con = duckdb.connect()
    try:
        # Single efficient query to count both control and case patients using DuckDB Parquet
        years_list = ','.join(map(str, train_years))
        query = f"""
            WITH control_counts AS (
                SELECT COUNT(DISTINCT mi_person_key) as n_controls
                FROM read_parquet('{control_cohort_path}')
                WHERE event_year IN ({years_list})
            ),
            case_counts AS (
                SELECT COUNT(DISTINCT mi_person_key) as n_cases
                FROM read_parquet('{target_cohort_path}')
                WHERE event_year IN ({years_list}) AND target = 1
            )
            SELECT 
                c.n_controls,
                ca.n_cases
            FROM control_counts c
            CROSS JOIN case_counts ca
        """
        result = con.execute(query).fetchone()
        n_controls = result[0] if result else 0
        n_cases = result[1] if result else 0
        
        if n_cases == 0:
            return (False, 0.0, n_controls, n_cases)
        
        actual_ratio = n_controls / n_cases
        min_ratio = expected_ratio * (1 - tolerance)
        max_ratio = expected_ratio * (1 + tolerance)
        
        is_valid = min_ratio <= actual_ratio <= max_ratio
        return (is_valid, actual_ratio, n_controls, n_cases)
    finally:
        con.close()


def ensure_control_cohort_with_ratio(
    control_cohort: str,
    age_band: str,
    target_cohort_path: Path,
    expected_ratio: float = DEFAULT_SAMPLE_RATIO,
    tolerance: float = 0.2,
    train_years: list = [2016, 2017, 2018],
    force_recreate: bool = False,
) -> Tuple[Path, bool]:
    """
    Ensure control cohort exists with correct ratio. Recreates if needed.
    
    Args:
        control_cohort: Name of control cohort (e.g., "non_opioid_non_ed")
        age_band: Age band (e.g., "13-24")
        target_cohort_path: Path to target cohort model_events.parquet
        expected_ratio: Expected control:case ratio (default: 5.0)
        tolerance: Tolerance for ratio validation (default: 0.2 = 20%)
        train_years: List of training years
        force_recreate: Force recreation even if file exists and ratio is valid
    
    Returns:
        (control_cohort_path, was_recreated)
    """
    control_cohort_path = get_control_cohort_path(control_cohort, age_band)
    
    # Step 1: Check if exists locally
    if not control_cohort_path.exists():
        # Step 2: Try downloading from S3
        if not download_control_cohort_from_s3(control_cohort, age_band, control_cohort_path):
            print(f"[INFO] Control cohort not found locally or in S3. Will create new one.")
    
    # Step 3: Validate ratio if file exists
    needs_recreation = force_recreate
    if control_cohort_path.exists() and not force_recreate:
        is_valid, actual_ratio, n_controls, n_cases = validate_control_cohort_ratio(
            control_cohort_path, target_cohort_path, expected_ratio, tolerance, train_years
        )
        
        if is_valid:
            print(
                f"[OK] Control cohort ratio validation passed: {actual_ratio:.2f}:1 "
                f"({n_controls:,} distinct controls, {n_cases:,} distinct targets)"
            )
            return (control_cohort_path, False)
        else:
            print(
                f"[WARN] Control cohort ratio validation failed: {actual_ratio:.2f}:1 "
                f"({n_controls:,} distinct controls, {n_cases:,} distinct targets)"
            )
            print(f"[INFO] Expected ratio: {expected_ratio:.2f}:1 (tolerance: {expected_ratio * (1 - tolerance):.2f}-{expected_ratio * (1 + tolerance):.2f}:1)")
            print(f"[INFO] Will recreate control cohort to achieve {expected_ratio:.2f}:1 ratio...")
            needs_recreation = True
    
    # Step 4: Recreate if needed
    if needs_recreation or not control_cohort_path.exists():
        # Get case count to calculate required controls (using DuckDB Parquet for efficiency)
        con = duckdb.connect()
        try:
            years_list = ','.join(map(str, train_years))
            query_cases = f"""
                SELECT COUNT(DISTINCT mi_person_key) as n_cases
                FROM read_parquet('{target_cohort_path}')
                WHERE event_year IN ({years_list}) AND target = 1
            """
            n_cases = con.execute(query_cases).fetchone()[0]
        finally:
            con.close()
        
        if n_cases == 0:
            print(f"[ERROR] No cases found in target cohort. Cannot create control cohort.")
            return (control_cohort_path, False)
        
        import math
        required_controls = max(math.ceil(n_cases * expected_ratio), 1000)  # At least 1000 controls
        
        print(f"[INFO] Creating control cohort with {required_controls} controls (target: {expected_ratio:.2f}:1 ratio with {n_cases} cases)")
        
        # Remove existing file if recreating
        if control_cohort_path.exists():
            control_cohort_path.unlink()
            print(f"[INFO] Removed existing control cohort file for recreation")
        
        # Import and call the creation function
        from create_control_cohort_model_data import create_control_cohort_model_data
        
        # Use the same model_data_root to ensure consistency
        model_data_root = get_model_data_root()
        
        create_control_cohort_model_data(
            age_band=age_band,
            years=train_years,
            sample_size=required_controls,
            output_root=model_data_root,  # Explicitly pass to ensure correct path
            target_cohort_path=target_cohort_path,  # Pass target path for ratio logging
            time_window_days=14,  # Default 14-day window (supports 7, 14, 21, 30, 45)
        )
        
        if control_cohort_path.exists():
            # Validate and log the actual ratio after recreation
            is_valid, actual_ratio, n_controls, n_cases = validate_control_cohort_ratio(
                control_cohort_path, target_cohort_path, expected_ratio, tolerance, train_years
            )
            print(f"[OK] Control cohort recreated successfully: {control_cohort_path}")
            print(
                f"[OK] Final ratio: {actual_ratio:.2f}:1 "
                f"({n_controls:,} distinct controls, {n_cases:,} distinct targets)"
            )
            return (control_cohort_path, True)
        else:
            print(f"[ERROR] Control cohort recreation failed. File not found: {control_cohort_path}")
            return (control_cohort_path, False)
    
    return (control_cohort_path, False)


def main():
    """CLI entry point for ensuring control cohort with ratio."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ensure control cohort exists with correct 5:1 ratio"
    )
    parser.add_argument(
        "--control-cohort",
        type=str,
        required=True,
        help="Control cohort name (e.g., non_opioid_non_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 13-24)",
    )
    parser.add_argument(
        "--target-cohort-path",
        type=str,
        required=True,
        help="Path to target cohort model_events.parquet",
    )
    parser.add_argument(
        "--expected-ratio",
        type=float,
        default=DEFAULT_SAMPLE_RATIO,
        help=f"Expected control:case ratio (default: {DEFAULT_SAMPLE_RATIO})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Tolerance for ratio validation (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation even if file exists and ratio is valid",
    )
    
    args = parser.parse_args()
    
    control_path, was_recreated = ensure_control_cohort_with_ratio(
        control_cohort=args.control_cohort,
        age_band=args.age_band,
        target_cohort_path=Path(args.target_cohort_path),
        expected_ratio=args.expected_ratio,
        tolerance=args.tolerance,
        force_recreate=args.force_recreate,
    )
    
    if control_path.exists():
        print(f"[OK] Control cohort ready: {control_path}")
        sys.exit(0)
    else:
        print(f"[ERROR] Control cohort not available: {control_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
