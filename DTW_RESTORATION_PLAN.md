# DTW Feature Creation: Restoration & Optimization Action Plan

**Date:** February 17, 2026  
**Priority:** CRITICAL  
**Estimated Time:** 4-8 hours implementation + testing  

---

## Problem Statement

The DTW feature creation step (`create_dtw_features.py`) is **missing from the active codebase**, causing the DTW visualization pipeline to fail. The archived version exists at:
```
archived/dashboard_feature_engineering/dtw/create_dtw_features.py
```

This must be restored to `9_dashboard_visuals/dtw/create_dtw_features.py` and integrated into the workflow.

---

## Current Architecture Analysis

### Archived Implementation Review

**Strengths:**
- ✅ **Parallelization**: Uses `multiprocessing.Pool` for DTW distance computation
- ✅ **SHAP/FFA Filtering**: Integrates with `get_shap_ffa_allowed_codes_combined()`
- ✅ **Target-aligned trajectories**: Proper leakage prevention with cutoff dates
- ✅ **Prototype selection**: Median-length approach from combined target+control
- ✅ **Admin ICD integration**: Computes `admin_icd_event_count` for routine analysis
- ✅ **DuckDB SQL filtering**: Efficient pre-filtering before Python processing
- ✅ **Research mode**: Optional comprehensive trajectory capture

**Performance Characteristics:**
```python
# From archived code (lines 789-812):
if n_workers is None:
    n_workers = max(1, cpu_count() - 1)  # Leave one CPU free

if n_workers > 1 and len(compute_args) > 100:  # Only parallelize for large datasets
    logger.info(f"Computing DTW distances in parallel using {n_workers} workers")
    with Pool(n_workers) as pool:
        features_list = pool.map(_compute_dtw_for_patient, compute_args)
else:
    logger.info("Computing DTW distances sequentially")
    # Sequential processing for small cohorts
```

**Key Performance Parameters:**
- `n_prototypes`: Default 5 (configurable)
- `n_workers`: Defaults to `cpu_count() - 1` (e.g., 31 on 32-core EC2)
- `max_lookback_months`: Default 24 (restricts trajectory window)
- Parallelization threshold: 100 patients minimum

**Estimated Runtime (Based on Code Analysis):**
- Small cohort (<1000 patients): 1-2 minutes
- Medium cohort (1000-5000 patients): 3-7 minutes
- Large cohort (10,000+ patients): 10-20 minutes
- **32-core EC2 with 31 workers:** ~5-10 minutes per cohort/age_band
- **Full pipeline (16 combinations):** 1.5-3 hours total

---

## Restoration Plan

### Phase 1: File Recovery & Path Updates (30 minutes)

#### Step 1.1: Copy Script to Active Location
```bash
# From project root
cp archived/dashboard_feature_engineering/dtw/create_dtw_features.py 9_dashboard_visuals/dtw/create_dtw_features.py
```

#### Step 1.2: Update Path References
The archived script has incorrect path references that need updating:

**File:** `9_dashboard_visuals/dtw/create_dtw_features.py`

Find and replace (lines 31-34):
```python
# OLD (archived):
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

# NEW (correct for 9_dashboard_visuals/dtw/):
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Up to repo root
REPO_ROOT = PROJECT_ROOT
```

#### Step 1.3: Verify Import Paths
The script uses several py_helpers modules. Ensure these imports work:
```python
# Should be present (verify):
from py_helpers.shap_ffa_fpgrowth_utils import get_shap_ffa_allowed_codes_combined, _parse_feature_name
from py_helpers.model_data_paths import resolve_model_events_path
# Optional (may need fallback):
from py_helpers.checkpoint_utils import save_step_checkpoint
```

---

### Phase 2: Pipeline Integration (1-2 hours)

#### Step 2.1: Update `4_dashboard_visuals.ipynb`

**Current cell (line ~185):**
```python
def run_dtw_one(cohort_name, age_band):
    # Visuals only; we do not create DTW features in this pipeline 
    # (create_dtw_visuals loads existing CSV if present)
    r = subprocess.run(
        [sys.executable, str(DTW_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return (cohort_name, age_band, r.returncode, r.stdout, r.stderr)
```

**Update to TWO-STEP process:**
```python
def run_dtw_one(cohort_name, age_band):
    """Run DTW feature creation then visualization publishing (two-step process)."""
    # Step 1: Create DTW features (distance computation)
    r_features = subprocess.run(
        [sys.executable, str(DTW_FEATURES_SCRIPT), 
         "--cohort", cohort_name, 
         "--age-band", age_band,
         "--n-prototypes", "5"] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    
    # Check if feature creation succeeded
    if r_features.returncode != 0:
        return (cohort_name, age_band, r_features.returncode, r_features.stdout, r_features.stderr, None, None)
    
    # Step 2: Create and publish visualizations
    r_visuals = subprocess.run(
        [sys.executable, str(DTW_VISUALS_SCRIPT), 
         "--cohort-name", cohort_name, 
         "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    
    return (cohort_name, age_band, r_features.returncode, r_features.stdout, r_features.stderr, 
            r_visuals.returncode, r_visuals.stdout, r_visuals.stderr)
```

**Add script path at top of notebook (with BUPAR_VISUALS_SCRIPT, etc.):**
```python
DTW_FEATURES_SCRIPT = STEP9_ROOT / "dtw" / "create_dtw_features.py"
DTW_VISUALS_SCRIPT = STEP9_ROOT / "dtw" / "create_dtw_visuals.py"
```

**Update result handling:**
```python
with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futures = {ex.submit(run_dtw_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, feat_code, feat_out, feat_err, vis_code, vis_out, vis_err = fut.result()
        print(f"  [DTW] {cohort_name} / {age_band}")
        print(f"    Features: exit {feat_code}")
        if vis_code is not None:
            print(f"    Visuals: exit {vis_code}")
        
        if feat_code != 0:
            print(f"    Feature creation failed (exit {feat_code})")
            if feat_err:
                print("    stderr:", (feat_err[:1500] + "..." if len(feat_err) > 1500 else feat_err))
            if FAIL_FAST:
                raise RuntimeError(f"DTW feature creation failed: {cohort_name} / {age_band}")
        
        if vis_code is not None and vis_code != 0:
            print(f"    Visualization publishing failed (exit {vis_code})")
            if vis_err:
                print("    stderr:", (vis_err[:1500] + "..." if len(vis_err) > 1500 else vis_err))
            if FAIL_FAST:
                raise RuntimeError(f"DTW create_dtw_visuals failed: {cohort_name} / {age_band}")

print("DTW done (features + visuals).")
```

#### Step 2.2: Update `run_dashboard_visuals.py`

**Add to main() function (around line 100):**
```python
# Add to script paths section (around line 98):
dtw_features_script = step9_root / "dtw" / "create_dtw_features.py"
dtw_visuals_script = step9_root / "dtw" / "create_dtw_visuals.py"

# Add to parallel execution section (after BupaR, before FP-Growth):
def run_dtw_features(cohort: str, age_band: str) -> tuple:
    """Run DTW feature creation."""
    cmd = [sys.executable, str(dtw_features_script), 
           "--cohort", cohort, "--age-band", age_band, "--n-prototypes", "5"]
    if args.force:
        cmd.append("--force")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return (cohort, age_band, "features", r.returncode)

def run_dtw_visuals(cohort: str, age_band: str) -> tuple:
    """Run DTW visualization publishing."""
    cmd = [sys.executable, str(dtw_visuals_script),
           "--cohort-name", cohort, "--age-band", age_band, "--project-root", str(REPO_ROOT)]
    if args.force:
        cmd.append("--force")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return (cohort, age_band, "visuals", r.returncode)

# Run DTW (two-step: features then visuals)
print("\n" + "=" * 60)
print("DTW Trajectory Analysis (Features + Visuals)")
print("=" * 60)

# Step 1: Create features (can parallelize)
print("\nStep 1: Creating DTW features...")
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(run_dtw_features, c, ab) for c, ab in combinations]
    for fut in as_completed(futures):
        c, ab, step, code = fut.result()
        print(f"  [DTW Features] {c} / {ab} -> exit {code}")
        if code != 0 and args.fail_fast:
            sys.exit(1)

# Step 2: Create visuals (after all features exist)
print("\nStep 2: Creating DTW visualizations...")
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(run_dtw_visuals, c, ab) for c, ab in combinations]
    for fut in as_completed(futures):
        c, ab, step, code = fut.result()
        print(f"  [DTW Visuals] {c} / {ab} -> exit {code}")
        if code != 0 and args.fail_fast:
            sys.exit(1)

print("DTW done.")
```

---

### Phase 3: Testing & Validation (1-2 hours)

#### Test Scenario 1: Single Cohort/Age Band
```bash
# From repo root
python 9_dashboard_visuals/dtw/create_dtw_features.py \
    --cohort opioid_ed \
    --age-band 25-44 \
    --n-prototypes 5 \
    --force

# Expected output:
# - dtw_features_opioid_ed_25_44.csv in 10_risk_dashboard/visualizations/dtw/outputs/feature_engineering/
# - S3 upload to s3://pgxdatalake/gold/feature_engineering/6_dtw/opioid_ed/25-44/
```

**Validation checks:**
1. CSV exists with expected columns:
   ```python
   import pandas as pd
   df = pd.read_csv("10_risk_dashboard/visualizations/dtw/outputs/feature_engineering/dtw_features_opioid_ed_25_44.csv")
   
   expected_cols = [
       'mi_person_key', 'target', 'seq_pattern_str', 'admin_icd_event_count',
       'dtw_distance_to_prototype_0', 'dtw_distance_to_prototype_1', 
       'dtw_distance_to_prototype_2', 'dtw_distance_to_prototype_3', 
       'dtw_distance_to_prototype_4',
       'dtw_min_distance', 'dtw_max_distance', 'dtw_mean_distance', 'dtw_std_distance',
       'trajectory_length', 'trajectory_diversity'
   ]
   
   assert all(col in df.columns for col in expected_cols), f"Missing columns: {set(expected_cols) - set(df.columns)}"
   print(f"✅ All expected columns present ({len(df.columns)} total)")
   print(f"✅ {len(df)} patients with DTW features")
   ```

2. No NaN/inf in critical columns:
   ```python
   critical_cols = ['dtw_min_distance', 'trajectory_length', 'trajectory_diversity']
   for col in critical_cols:
       assert df[col].notna().all(), f"{col} has NaN values"
       assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf values"
   print("✅ No NaN/inf in critical columns")
   ```

3. Target distribution (should match model_data):
   ```python
   target_dist = df['target'].value_counts()
   print(f"Target distribution: {target_dist.to_dict()}")
   assert len(target_dist) == 2, "Should have both target=0 and target=1"
   ```

#### Test Scenario 2: Visualization Publishing
```bash
python 9_dashboard_visuals/dtw/create_dtw_visuals.py \
    --cohort-name opioid_ed \
    --age-band 25-44 \
    --project-root . \
    --force

# Expected output:
# - Plots in 10_risk_dashboard/visualizations/dtw/outputs/opioid_ed/25_44/plots/
# - chart_data.json with routine_comparison and high_risk_trajectories
# - S3 upload to dashboard bucket
```

#### Test Scenario 3: Full Pipeline Integration
```python
# In Jupyter notebook or Python script
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
combinations = [("opioid_ed", "25-44"), ("non_opioid_ed", "25-44")]

for cohort, age_band in combinations:
    print(f"\nTesting {cohort} / {age_band}")
    
    # Features
    r1 = subprocess.run([
        sys.executable, "9_dashboard_visuals/dtw/create_dtw_features.py",
        "--cohort", cohort, "--age-band", age_band, "--force"
    ], cwd=REPO_ROOT)
    assert r1.returncode == 0, f"Features failed for {cohort}/{age_band}"
    
    # Visuals
    r2 = subprocess.run([
        sys.executable, "9_dashboard_visuals/dtw/create_dtw_visuals.py",
        "--cohort-name", cohort, "--age-band", age_band, 
        "--project-root", ".", "--force"
    ], cwd=REPO_ROOT)
    assert r2.returncode == 0, f"Visuals failed for {cohort}/{age_band}"
    
    print(f"✅ {cohort} / {age_band} complete")
```

---

## Performance Optimization Strategies

### Current Performance Profile (From Archived Code)

**Bottlenecks:**
1. **DTW distance computation:** O(N*M) per patient-prototype pair
   - 20,000 patients × 5 prototypes = 100,000 distance computations
   - Each computation: ~100-500 events per sequence
2. **Trajectory extraction:** DuckDB SQL filtering (efficient)
3. **Encoding:** Linear time (not a bottleneck)

**Parallelization Strategy:**
- Uses `multiprocessing.Pool` with `n_workers = cpu_count() - 1`
- Threshold: Only parallelize if >100 patients (avoids overhead)
- Each worker computes distances for 1 patient to all prototypes

### Optimization 1: Sakoe-Chiba Window (Already Implemented)

The `dtaidistance.dtw.distance()` function supports window constraints:
```python
# Add to _compute_dtw_for_patient function (around line 704):
from dtaidistance import dtw
distance = dtw.distance(encoded_traj, proto_traj, window=6)  # ±6 steps warping
```

**Rationale:** Limits warping to ±6 time steps (~6 months with monthly bucketing). Prevents unrealistic alignments and improves performance.

**Expected speedup:** 20-30% reduction in computation time

### Optimization 2: Fast Distance Matrix (For Prototype Selection)

If computing full distance matrix for prototype selection:
```python
from dtaidistance.dtw import distance_matrix_fast

# Faster approach for prototype selection from sample
sample_trajectories = {pid: traj for pid, traj in list(patient_trajectories.items())[:2000]}
sample_encoded = {pid: [global_encoding[item] for item in traj] 
                  for pid, traj in sample_trajectories.items()}

# Use C-accelerated distance matrix (if needed)
# dm = distance_matrix_fast(list(sample_encoded.values()), window=6, parallel=True, use_c=True)
```

**Note:** Current median-length approach avoids full matrix computation (good!)

### Optimization 3: Trajectory Length Capping

Add max length parameter to prevent extremely long sequences:
```python
# In extract_patient_trajectories or similar (around line 400-500)
MAX_TRAJECTORY_LENGTH = 100  # Cap at 100 events

for pid, events in patient_events.items():
    trajectory = sorted(events, key=lambda e: e['event_date'])[:MAX_TRAJECTORY_LENGTH]
    trajectories[pid] = trajectory
```

**Rationale:** Very long trajectories (>100 events) are rare and expensive to compute. Capping reduces worst-case runtime without losing signal.

### Optimization 4: Batch Processing (For Very Large Cohorts)

If a cohort has >50,000 patients:
```python
BATCH_SIZE = 10000

def compute_dtw_in_batches(patient_trajectories, n_prototypes=5, n_workers=None):
    """Process large cohorts in batches to manage memory."""
    all_features = []
    patient_ids = list(patient_trajectories.keys())
    
    for i in range(0, len(patient_ids), BATCH_SIZE):
        batch_ids = patient_ids[i:i + BATCH_SIZE]
        batch_trajs = {pid: patient_trajectories[pid] for pid in batch_ids}
        
        batch_features = compute_dtw_distances_to_prototypes(
            batch_trajs, n_prototypes, n_workers
        )
        all_features.append(batch_features)
        
        # Clear memory
        del batch_trajs
        gc.collect()
    
    return pd.concat(all_features, ignore_index=True)
```

### Optimization 5: Checkpointing for Long Runs

Add intermediate checkpoints to resume on failure:
```python
# In create_all_dtw_features (around line 850)
checkpoint_dir = _dtw_output_root(project_root) / "outputs" / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_file = checkpoint_dir / f"dtw_checkpoint_{cohort_name}_{age_band}.pkl"

# Load checkpoint if exists
if checkpoint_file.exists() and not force:
    import pickle
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    patient_trajectories = checkpoint_data['trajectories']
    logger.info(f"Loaded {len(patient_trajectories)} trajectories from checkpoint")
else:
    # Extract trajectories (slow step)
    patient_trajectories = extract_patient_trajectories(...)
    
    # Save checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'trajectories': patient_trajectories}, f)

# Continue with DTW computation...
```

---

## Testing Checklist

### Pre-deployment Validation

- [ ] Script exists at `9_dashboard_visuals/dtw/create_dtw_features.py`
- [ ] Path references updated (PROJECT_ROOT, REPO_ROOT)
- [ ] All imports resolve (py_helpers modules)
- [ ] `dtaidistance` package installed (`pip install dtaidistance`)
- [ ] Test run on one cohort/age_band succeeds
- [ ] Output CSV has all expected columns
- [ ] No NaN/inf in critical columns
- [ ] Target distribution matches input data
- [ ] S3 upload succeeds (if AWS configured)
- [ ] Visualization publishing works with generated CSV
- [ ] Plots uploaded to dashboard bucket
- [ ] chart_data.json generated correctly
- [ ] Pipeline checkpoint created (`9_dashboard_visuals`)
- [ ] Full pipeline run (2 cohorts × 2 age bands) succeeds
- [ ] Runtime reasonable (<15 min per cohort/age_band on 32-core)

### Performance Validation

- [ ] Parallelization activates for large cohorts (>100 patients)
- [ ] Worker count = cpu_count() - 1 (e.g., 31 on 32-core EC2)
- [ ] No memory overflow on large cohorts
- [ ] Log messages confirm parallel execution
- [ ] Total runtime for 16 combinations <4 hours

---

## Rollback Plan

If issues arise, temporary workaround:

1. **Keep existing behavior:** Pipeline skips DTW if CSV missing
2. **Manual DTW generation:** Run `create_dtw_features.py` separately before dashboard workflow
3. **Phased rollout:** Enable DTW for one cohort/age_band first

---

## Documentation Updates

After successful restoration:

1. Update `9_dashboard_visuals/README.md`:
   - Add note that DTW uses two-step process (features → visuals)
   - Document `create_dtw_features.py` CLI arguments

2. Update `9_dashboard_visuals/dtw/DTW_VISUALIZATION_STATUS.md`:
   - Change status from "missing" to "active"
   - Add performance notes

3. Update `4_dashboard_visuals.ipynb` markdown cell:
   - Remove "we do not create DTW features in this pipeline" comment
   - Add explanation of two-step DTW process

4. Update `DASHBOARD_VISUALS_REVIEW.md`:
   - Mark DTW restoration as complete
   - Add actual runtime measurements

---

## Success Criteria

✅ **Feature Creation Restored:**
- `create_dtw_features.py` successfully generates CSVs for all cohort/age_band combinations
- Runtime: <15 minutes per combination on 32-core EC2

✅ **Pipeline Integration:**
- `4_dashboard_visuals.ipynb` runs both feature creation and visualization publishing
- No manual intervention required

✅ **Visualization Quality:**
- Plots show meaningful clusters
- chart_data.json has valid routine vs. no routine comparison
- Dashboard DTW tab displays correctly

✅ **Performance:**
- Full pipeline (16 combinations): <4 hours total
- No memory issues on 32-core EC2 with 1TB RAM

✅ **Idempotency:**
- Re-running pipeline skips completed steps (unless --force)
- S3 checkpoints prevent duplicate work

---

## Next Steps After Restoration

1. **Baseline Performance Measurement:**
   - Run full pipeline and record actual runtimes
   - Identify any remaining bottlenecks

2. **Enhanced Visualizations:**
   - Implement year filtering (framework exists in create_dtw_plots.py)
   - Add barycenter/archetype plots

3. **Interactive Features:**
   - Convert static plots to Plotly HTML (like FP-Growth networks)
   - Enable drill-down into trajectory clusters

4. **Research Applications:**
   - Use `--research-mode` flag to capture comprehensive trajectory data
   - Analyze temporal patterns and time windows

---

## Contact & Support

For issues during restoration:
- Check logs in `9_dashboard_visuals/logs/dtw/`
- Review S3 status: `python 9_dashboard_visuals/dtw/check_dtw_s3_status.py`
- Test notebook: `9_dashboard_visuals/test_dashboard_visuals.ipynb`
