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
- ✅ **No conflicts**: Control cohorts are age-band specific
  - Cohort 5 (65-74) uses: `cohort_name=non_opioid_non_ed/age_band=65-74/model_events.parquet`
  - Cohort 6 (75-84) uses: `cohort_name=non_opioid_non_ed/age_band=75-84/model_events.parquet`
  - Cohort 7 (85-94) uses: `cohort_name=non_opioid_non_ed/age_band=85-94/model_events.parquet`
- Each notebook creates/uses its own age-band-specific control cohort file
- **Safe to run in parallel**: No file conflicts between different age bands

## Best Practices

### Option 1: Parallel Execution (Recommended)
✅ **Safe to run all 3 notebooks in parallel immediately** - Each uses its own age-band-specific control cohort:
- Cohort 5 (65-74) → `cohort_name=non_opioid_non_ed/age_band=65-74/model_events.parquet`
- Cohort 6 (75-84) → `cohort_name=non_opioid_non_ed/age_band=75-84/model_events.parquet`
- Cohort 7 (85-94) → `cohort_name=non_opioid_non_ed/age_band=85-94/model_events.parquet`

**No conflicts**: Each notebook creates/uses a separate control cohort file.

### Option 2: Sequential Execution (If Resource Constrained)
If you have limited resources (memory/CPU), run sequentially:
1. Run cohort 5 first
2. Once complete, run cohort 6
3. Then run cohort 7

This ensures:
- Easier to monitor progress
- Less resource contention
- But not required for avoiding conflicts (each age band has its own control cohort)

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
**Solution**: Each notebook will automatically create its age-band-specific control cohort if it doesn't exist. If you want to pre-create them:
```bash
# Create control cohorts for all three age bands
python 4a_model_data/create_control_cohort_model_data.py --age-band 65-74 --sample-size 100000
python 4a_model_data/create_control_cohort_model_data.py --age-band 75-84 --sample-size 100000
python 4a_model_data/create_control_cohort_model_data.py --age-band 85-94 --sample-size 100000
```

**Note**: Each age band creates its own separate control cohort file, so you can run these in parallel too!

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
