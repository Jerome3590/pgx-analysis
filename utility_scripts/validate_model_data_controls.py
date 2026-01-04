#!/usr/bin/env python3
"""
Validate that model_events.parquet files contain both cases (target=1) and controls (target=0).

This script can check:
- Local files (in 4a_model_data/ or /mnt/nvme/4a_model_data/)
- S3 files (downloads temporarily to check)

Usage:
    python utility_scripts/validate_model_data_controls.py [--cohort <cohort>] [--age-band <age_band>] [--check-s3]
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

# Import path resolution utilities
from py_helpers.env_utils import get_data_root


def validate_file(parquet_path: Path) -> dict:
    """Validate a single model_events.parquet file."""
    if not parquet_path.exists():
        return {
            "exists": False,
            "has_controls": False,
            "n_cases": 0,
            "n_controls": 0,
            "error": "File not found",
        }
    
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                COUNT(*) FILTER (WHERE target = 0) AS n_controls,
                COUNT(*) AS n_total
            FROM read_parquet('{parquet_path}')
            """
        ).fetchone()
        
        n_cases = result[0] if result else 0
        n_controls = result[1] if result else 0
        n_total = result[2] if result else 0
        
        return {
            "exists": True,
            "has_controls": n_controls > 0,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_total": n_total,
            "error": None,
        }
    except Exception as e:
        return {
            "exists": True,
            "has_controls": False,
            "n_cases": 0,
            "n_controls": 0,
            "error": str(e),
        }
    finally:
        con.close()


def find_local_model_data_files(cohort: str = None, age_band: str = None) -> list:
    """Find all local model_events.parquet files."""
    files = []
    
    # Check multiple locations
    data_root = get_data_root()
    candidates = [
        data_root / "4a_model_data",  # Linux/EC2: /mnt/nvme/4a_model_data
        PROJECT_ROOT / "4a_model_data",  # Windows/local dev
    ]
    
    for base_dir in candidates:
        if not base_dir.exists():
            continue
        
        pattern = "**/model_events.parquet"
        for file_path in base_dir.rglob(pattern):
            # Extract cohort and age_band from path
            path_str = str(file_path)
            path_cohort = None
            path_age_band = None
            
            if "cohort_name=" in path_str:
                parts = path_str.split("cohort_name=")
                if len(parts) > 1:
                    path_cohort = parts[1].split("/")[0]
            
            if "age_band=" in path_str:
                parts = path_str.split("age_band=")
                if len(parts) > 1:
                    path_age_band = parts[1].split("/")[0]
            
            # Filter by cohort/age_band if specified
            if cohort and path_cohort != cohort:
                continue
            if age_band and path_age_band != age_band:
                continue
            
            files.append((file_path, path_cohort, path_age_band))
    
    return files


def check_s3_file(s3_path: str) -> dict:
    """Validate a single S3 file by querying it directly with DuckDB (no download)."""
    import duckdb
    
    con = duckdb.connect()
    try:
        # Query S3 file directly using DuckDB's S3 support
        # This avoids downloading large files
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                COUNT(*) FILTER (WHERE target = 0) AS n_controls,
                COUNT(*) AS n_total
            FROM read_parquet('{s3_path}')
            """
        ).fetchone()
        
        if result is None:
            return {
                "exists": False,
                "has_controls": False,
                "error": "Could not read file from S3",
            }
        
        n_cases = result[0] if result else 0
        n_controls = result[1] if result else 0
        n_total = result[2] if result else 0
        
        return {
            "exists": True,
            "has_controls": n_controls > 0,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_total": n_total,
            "s3_path": s3_path,
            "error": None,
        }
    except Exception as e:
        return {
            "exists": False,
            "has_controls": False,
            "error": f"Error reading S3 file: {str(e)}",
        }
    finally:
        con.close()


def main():
    # Reconfigure stdout for UTF-8 on Windows to handle warning symbols
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(
        description="Validate model_events.parquet files for presence of controls"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        help="Filter by cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Filter by age band (e.g., 13-24)",
    )
    parser.add_argument(
        "--check-s3",
        action="store_true",
        help="Also check files in S3",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Model Data Controls Validation")
    print("=" * 80)
    print()
    
    # Check local files
    print("Checking local files...")
    local_files = find_local_model_data_files(
        cohort=args.cohort,
        age_band=args.age_band,
    )
    
    if not local_files:
        print("  No local model_events.parquet files found.")
    else:
        print(f"  Found {len(local_files)} file(s)")
        print()
        
        all_valid = True
        for file_path, cohort_name, age_band in local_files:
            validation = validate_file(file_path)
            
            status = "✓" if validation["has_controls"] else "✗"
            print(f"{status} {cohort_name}/{age_band}: {file_path}")
            
            if validation["error"]:
                print(f"    Error: {validation['error']}")
                all_valid = False
            elif validation["exists"]:
                print(
                    f"    Cases: {validation['n_cases']:,}, "
                    f"Controls: {validation['n_controls']:,}, "
                    f"Total: {validation['n_total']:,}"
                )
                if not validation["has_controls"]:
                    all_valid = False
                    print("    ⚠ WARNING: Missing controls!")
            else:
                all_valid = False
            print()
        
        if all_valid:
            print("✓ All local files have controls")
        else:
            print("✗ Some local files are missing controls")
    
    # Check S3 if requested
    if args.check_s3:
        print()
        print("Checking S3 files...")
        import subprocess
        import shutil
        
        aws_cli = shutil.which("aws")
        if not aws_cli:
            print("  AWS CLI not found, skipping S3 check")
        else:
            s3_bucket = "pgxdatalake"
            s3_prefix = "gold/cohorts_model_data"
            
            # List files in S3
            result = subprocess.run(
                [aws_cli, "s3", "ls", f"s3://{s3_bucket}/{s3_prefix}/", "--recursive"],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                print(f"  Error listing S3 files: {result.stderr}")
            else:
                s3_files = [
                    line.split()[-1]
                    for line in result.stdout.splitlines()
                    if "model_events.parquet" in line
                ]
                
                if not s3_files:
                    print("  No model_events.parquet files found in S3")
                else:
                    print(f"  Found {len(s3_files)} file(s) in S3")
                    print()
                    
                    for s3_key in s3_files:
                        s3_path = f"s3://{s3_bucket}/{s3_key}"
                        validation = check_s3_file(s3_path)
                        
                        status = "✓" if validation.get("has_controls") else "✗"
                        print(f"{status} {s3_path}")
                        
                        if validation.get("error"):
                            print(f"    Error: {validation['error']}")
                        elif validation.get("exists"):
                            print(
                                f"    Cases: {validation['n_cases']:,}, "
                                f"Controls: {validation['n_controls']:,}, "
                                f"Total: {validation['n_total']:,}"
                            )
                            if not validation.get("has_controls"):
                                print("    ⚠ WARNING: Missing controls!")
                        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()

