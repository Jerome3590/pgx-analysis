# Running Cohorts in Parallel (Step 8)

This guide explains how to clear all Step 8 outputs and then run each cohort in separate terminals for parallelization.

## Step 1: Clear All Step 8 Outputs (Run Once)

```bash
# Preview what will be deleted
./utility_scripts/clear_all_step8_outputs.sh --dry-run

# Actually clear all Step 8 outputs
./utility_scripts/clear_all_step8_outputs.sh
```

This will:
- Clear all local Step 8 output files (`8_ffa_analysis/outputs/`)
- Clear all S3 checkpoints (`s3://pgx-repository/pipeline_checkpoints/8_ffa_analysis/`)
- Clear all S3 outputs (`s3://pgxdatalake/gold/ffa_analysis/`)
- Clear Step 8 completion flags from time logs

## Step 2: Run Each Cohort in Separate Terminals

Open **7 separate terminal windows/tabs** and run one cohort in each:

### Terminal 1:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh opioid_ed 13-24
```

### Terminal 2:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh opioid_ed 25-44
```

### Terminal 3:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh opioid_ed 45-54
```

### Terminal 4:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh opioid_ed 55-64
```

### Terminal 5:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh non_opioid_ed 65-74
```

### Terminal 6:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh non_opioid_ed 75-84
```

### Terminal 7:
```bash
cd /home/pgx3874/pgx-analysis
./utility_scripts/run_cohort_workflow.sh non_opioid_ed 85-94
```

## Using tmux/screen for Multiple Sessions

If you prefer using `tmux` or `screen`:

### Using tmux:
```bash
# Create a new tmux session
tmux new-session -d -s pgx_workflow

# Create 7 windows (one for each cohort)
tmux new-window -t pgx_workflow:1 -n 'opioid_13-24' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 13-24'
tmux new-window -t pgx_workflow:2 -n 'opioid_25-44' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 25-44'
tmux new-window -t pgx_workflow:3 -n 'opioid_45-54' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 45-54'
tmux new-window -t pgx_workflow:4 -n 'opioid_55-64' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 55-64'
tmux new-window -t pgx_workflow:5 -n 'non_opioid_65-74' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 65-74'
tmux new-window -t pgx_workflow:6 -n 'non_opioid_75-84' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 75-84'
tmux new-window -t pgx_workflow:7 -n 'non_opioid_85-94' 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 85-94'

# Attach to the session
tmux attach-session -t pgx_workflow

# Navigate between windows: Ctrl+b then window number (0-6)
# Detach: Ctrl+b then d
```

### Using screen:
```bash
# Create 7 screen sessions
screen -dmS pgx_opioid_13-24 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 13-24'
screen -dmS pgx_opioid_25-44 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 25-44'
screen -dmS pgx_opioid_45-54 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 45-54'
screen -dmS pgx_opioid_55-64 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh opioid_ed 55-64'
screen -dmS pgx_non_opioid_65-74 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 65-74'
screen -dmS pgx_non_opioid_75-84 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 75-84'
screen -dmS pgx_non_opioid_85-94 bash -c 'cd /home/pgx3874/pgx-analysis && ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 85-94'

# List all sessions
screen -ls

# Attach to a session
screen -r pgx_opioid_13-24

# Detach: Ctrl+a then d
```

## Monitoring Progress

You can monitor progress by checking:

1. **Time logs**: `logs/time_tracking/{cohort}_{age_band}.json`
2. **Output files**: `8_ffa_analysis/outputs/{cohort}/{age_band}/xgboost/`
3. **S3 checkpoints**: `s3://pgx-repository/pipeline_checkpoints/8_ffa_analysis/`
4. **S3 outputs**: `s3://pgxdatalake/gold/ffa_analysis/`

## Quick Status Check

```bash
# Check which cohorts have completed Step 8
python utility_scripts/check_step8_outputs.py --all-cohorts
```

## Notes

- Each workflow will automatically skip Steps 1-7 if they're already complete
- Step 8 will run with the binary feature fix
- Workflows are independent - if one fails, others continue
- Each workflow saves checkpoints, so you can resume if interrupted
