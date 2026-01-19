# Running Multiple Notebooks Interactively on EC2

Yes, you can run multiple Jupyter notebooks interactively at the same time! Each notebook runs in its own kernel, so they operate independently.

## Quick Start

1. **Start Jupyter** (if not already running):
   ```bash
   cd /home/pgx3874/pgx-analysis
   jupyter notebook --no-browser --port=8888
   ```

2. **Open multiple notebooks** in separate browser tabs:
   - `step3b_interactive_analysis_cohort5.ipynb` (non_opioid_ed / 65-74)
   - `step3b_interactive_analysis_cohort6.ipynb` (non_opioid_ed / 75-84)
   - `step3b_interactive_analysis_cohort7.ipynb` (non_opioid_ed / 85-94)

3. **Run cells independently** - Each notebook has its own kernel and can run cells independently.

## Important Considerations

### 1. **Resource Usage**
- Each notebook uses memory and CPU
- Running 3 notebooks simultaneously will use ~3x the resources
- Monitor with `htop` or `nvidia-smi` (if using GPU)

### 2. **File Conflicts**
- **Output files**: Each notebook writes to cohort-specific directories:
  - `outputs/non_opioid_ed/65_74/`
  - `outputs/non_opioid_ed/75_84/`
  - `outputs/non_opioid_ed/85_94/`
- **No conflicts**: Different output directories prevent file conflicts
- **S3 uploads**: May happen simultaneously, but AWS handles this

### 3. **R Script Execution**
- Each notebook calls R scripts via `subprocess`
- Multiple R processes can run simultaneously
- R scripts write to cohort-specific output directories (no conflicts)

### 4. **Control Cohort Creation**
- If multiple notebooks need to create the same control cohort (`non_opioid_non_ed`), they may conflict
- **Solution**: Run one notebook first to create the control cohort, or use the script-based approach for control cohort creation

## Best Practices

### Option 1: Sequential Execution (Recommended for First Run)
1. Run cohort 5 first (creates control cohort if needed)
2. Once complete, run cohort 6
3. Then run cohort 7

This ensures:
- Control cohort is created before other notebooks need it
- Easier to monitor progress
- Less resource contention

### Option 2: Parallel Execution (After Control Cohort Exists)
1. First, ensure control cohort exists:
   ```bash
   python 4a_model_data/ensure_control_cohort.py \
     --cohort non_opioid_non_ed \
     --age-band 65-74 \
     --target-cohort-path /mnt/nvme/4a_model_data/cohort_name=non_opioid_ed/age_band=65-74/model_events.parquet
   ```
   (Repeat for 75-84 and 85-94)

2. Then open all 3 notebooks and run them in parallel

### Option 3: Use tmux/screen for Multiple Sessions
```bash
# Start tmux session
tmux new -s cohort5
# In tmux, start Jupyter
jupyter notebook --no-browser --port=8888

# Create new tmux window for cohort 6
tmux new-window -t cohort5:1
jupyter notebook --no-browser --port=8889

# Create new tmux window for cohort 7
tmux new-window -t cohort5:2
jupyter notebook --no-browser --port=8890
```

Then access:
- Cohort 5: `http://your-ec2-ip:8888`
- Cohort 6: `http://your-ec2-ip:8889`
- Cohort 7: `http://your-ec2-ip:8890`

## Monitoring

### Check Running Notebooks
```bash
# List all Jupyter processes
ps aux | grep jupyter

# Check notebook kernels
jupyter kernelspec list
```

### Monitor Resources
```bash
# CPU and memory
htop

# Disk I/O
iostat -x 1

# Check output directories
ls -lh /home/pgx3874/pgx-analysis/3b_feature_importance_eda/outputs/non_opioid_ed/
```

## Troubleshooting

### Issue: "Control cohort not found" errors
**Solution**: Run one notebook first to create the control cohort, or create it manually:
```bash
python 4a_model_data/create_control_cohort_model_data.py \
  --age-band 65-74 \
  --sample-size 100000
```

### Issue: Out of memory
**Solution**: 
- Run notebooks sequentially instead of parallel
- Close other applications
- Consider using the script-based approach (`run_step_3b.py`) which is more memory-efficient

### Issue: R script conflicts
**Solution**: R scripts write to cohort-specific directories, so no conflicts. If you see errors, check:
- Each notebook is using the correct cohort/age_band
- Output directories are separate

## Alternative: Script-Based Approach

If you want to avoid interactive notebooks, use the script-based approach:

```bash
# Run all three cohorts sequentially
python 3b_feature_importance_eda/run_step_3b.py --cohort non_opioid_ed --age-band 65-74
python 3b_feature_importance_eda/run_step_3b.py --cohort non_opioid_ed --age-band 75-84
python 3b_feature_importance_eda/run_step_3b.py --cohort non_opioid_ed --age-band 85-94

# Or use the parallel script
bash 3b_feature_importance_eda/run_multiple_cohorts.sh
```

This is more resource-efficient and easier to monitor via logs.
