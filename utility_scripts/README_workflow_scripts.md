# Workflow Scripts

Bash scripts to run the complete PGx analysis pipeline for each cohort.

## Scripts

### `run_cohort_workflow.sh`
Runs the complete workflow for a single cohort/age band combination.

**Usage:**
```bash
./run_cohort_workflow.sh <cohort_name> <age_band> [--skip-steps STEP1,STEP2]
```

**Examples:**
```bash
# Run full workflow for opioid_ed / 13-24
./run_cohort_workflow.sh opioid_ed 13-24

# Run workflow but skip DTW step (5d)
./run_cohort_workflow.sh opioid_ed 13-24 --skip-steps 5d

# Run workflow but skip multiple steps
./run_cohort_workflow.sh non_opioid_ed 65-74 --skip-steps 5d,9
```

**Steps:**
- **3**: Feature Importance (Monte Carlo CV)
- **4a**: Model Data Extraction
- **4b**: DTW Protocol Filtering
- **5a**: BupaR Process Mining
- **5b**: FP-Growth Analysis
- **5c**: PGx Feature Engineering
- **5d**: DTW Trajectory Analysis (optional)
- **6**: Final Model Training
- **7**: FFA Analysis
- **8**: SHAP Analysis
- **9**: Combined SHAP + FFA

### `run_opioid_ed_workflow.sh`
Runs the complete workflow for all opioid_ed age bands:
- 13-24
- 25-44
- 45-54
- 55-64

**Usage:**
```bash
./run_opioid_ed_workflow.sh [--skip-steps STEP1,STEP2]
```

**Example:**
```bash
# Run all opioid_ed cohorts
./run_opioid_ed_workflow.sh

# Skip DTW for all
./run_opioid_ed_workflow.sh --skip-steps 5d
```

### `run_non_opioid_ed_workflow.sh`
Runs the complete workflow for all non_opioid_ed age bands:
- 65-74
- 75-84
- 85-94

**Usage:**
```bash
./run_non_opioid_ed_workflow.sh [--skip-steps STEP1,STEP2]
```

**Example:**
```bash
# Run all non_opioid_ed cohorts
./run_non_opioid_ed_workflow.sh
```

### `run_all_cohorts_workflow.sh`
Runs the complete workflow for **all** cohorts and age bands.

**Usage:**
```bash
./run_all_cohorts_workflow.sh [--skip-steps STEP1,STEP2]
```

**Example:**
```bash
# Run everything (all cohorts, all age bands)
./run_all_cohorts_workflow.sh
```

## Setup

1. **Make scripts executable:**
```bash
cd utility_scripts
chmod +x run_*.sh
```

2. **Fix line endings (if on Windows):**
```bash
sed -i 's/\r$//' run_*.sh
```

3. **Ensure you're in the project root or scripts directory:**
```bash
# From project root
bash utility_scripts/run_cohort_workflow.sh opioid_ed 13-24

# Or from utility_scripts directory
cd utility_scripts
./run_cohort_workflow.sh opioid_ed 13-24
```

## Prerequisites

- Python 3.x with all project dependencies installed
- R (for BupaR analysis)
- AWS credentials configured (for S3 access)
- Project data synced (or S3 access available)

## Notes

- Scripts will stop on first error (use `set -euo pipefail`)
- Some steps may be optional (e.g., 5d DTW) and will warn but continue
- Check individual script documentation for specific requirements
- Logs are written to console; check individual step outputs for details

## Troubleshooting

**Script not found:**
```bash
# Make sure you're in the right directory
pwd
ls -la utility_scripts/run_*.sh
```

**Permission denied:**
```bash
chmod +x utility_scripts/run_*.sh
```

**Line ending errors:**
```bash
sed -i 's/\r$//' utility_scripts/run_*.sh
```

**Step fails:**
- Check if prerequisites are met (previous steps completed)
- Verify command-line arguments match script expectations
- Check individual script logs for detailed error messages

