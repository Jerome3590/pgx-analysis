#!/usr/bin/env python3
"""
Clear all step checkpoints for all cohorts/age bands.
Works on both Windows and Linux.
"""

import json
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
TIME_LOG_DIR = PROJECT_ROOT / "logs" / "time_tracking"

def clear_all_checkpoints():
    """Clear all step checkpoints."""
    if not TIME_LOG_DIR.exists():
        print(f"No checkpoint directory found: {TIME_LOG_DIR}")
        print("Nothing to clear.")
        return 0
    
    print("Clearing all step checkpoints...")
    print(f"Checkpoint directory: {TIME_LOG_DIR}")
    print()
    
    # Find all JSON checkpoint files
    checkpoint_files = list(TIME_LOG_DIR.glob("*.json"))
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        return 0
    
    cleared_count = 0
    for checkpoint_file in checkpoint_files:
        cohort_age = checkpoint_file.stem
        print(f"Clearing checkpoints for: {cohort_age}")
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            if 'step_times' in data:
                # Clear all step completion flags
                steps_cleared = 0
                for step_key in data['step_times']:
                    if data['step_times'][step_key].get('completed', False):
                        data['step_times'][step_key]['completed'] = False
                        steps_cleared += 1
                
                if steps_cleared > 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"  Cleared {steps_cleared} step checkpoint(s)")
                    cleared_count += 1
                else:
                    print(f"  No completed steps to clear")
            else:
                print(f"  No step checkpoints found")
                
        except Exception as e:
            print(f"  Error processing {checkpoint_file.name}: {e}")
            continue
    
    print()
    if cleared_count > 0:
        print(f"Cleared checkpoints for {cleared_count} cohort/age_band combination(s)")
        print("All steps will rerun on next workflow execution.")
    else:
        print("No checkpoints were cleared (may already be cleared or no completed steps found).")
    
    return 0

if __name__ == "__main__":
    sys.exit(clear_all_checkpoints())
