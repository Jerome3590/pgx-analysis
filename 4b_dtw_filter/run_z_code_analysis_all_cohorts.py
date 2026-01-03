#!/usr/bin/env python3
"""
Run Z code analysis for all cohorts used in the risk model.

This script runs analyze_z_codes_in_cohorts.py for:
- opioid_ed (all age bands)
- non_opioid_ed (all age bands)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import COHORT_NAMES, AGE_BANDS

SCRIPT_PATH = PROJECT_ROOT / "4b_dtw_filter" / "analyze_z_codes_in_cohorts.py"

def main():
    """Run Z code analysis for all cohorts and age bands."""
    print("=" * 80)
    print("Running Z Code Analysis for All Risk Model Cohorts")
    print("=" * 80)
    print()
    print(f"Cohorts: {', '.join(COHORT_NAMES)}")
    print(f"Age bands: {', '.join(AGE_BANDS)}")
    print()
    
    total_runs = len(COHORT_NAMES) * len(AGE_BANDS)
    current_run = 0
    
    for cohort_name in COHORT_NAMES:
        print("=" * 80)
        print(f"Cohort: {cohort_name}")
        print("=" * 80)
        print()
        
        for age_band in AGE_BANDS:
            current_run += 1
            print(f"[{current_run}/{total_runs}] {cohort_name} / {age_band}")
            print("-" * 80)
            
            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--cohort", cohort_name,
                "--age-band", age_band
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=False,  # Show output in real-time
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"[OK] Completed: {cohort_name} / {age_band}")
                else:
                    print(f"[WARN] Exit code {result.returncode}: {cohort_name} / {age_band}")
                
            except Exception as e:
                print(f"[ERROR] Failed: {cohort_name} / {age_band}")
                print(f"  Error: {e}")
            
            print()
    
    print("=" * 80)
    print("All analyses complete!")
    print("=" * 80)
    print()
    print(f"Results saved to: {PROJECT_ROOT / '4b_dtw_filter' / 'outputs' / 'z_code_analysis'}")

if __name__ == "__main__":
    main()
