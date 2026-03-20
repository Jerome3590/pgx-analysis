# -*- coding: utf-8 -*-
# Auto-generated from 2_feature_importance.ipynb (VS Code Python notebook script format)

# %% [markdown]
# # Feature Importance Workflow (Steps 3a–3c)
#
# Run this **after** cohorts exist (Step 2; see `1_cohort_workflow.ipynb`). One cell per cohort × age_band for Step 3a and Step 3b (full grid from `py_helpers.constants.REQUIRED_COHORTS`); run Configuration and Sync once, then the cell(s) for the combination you need.
#
# ## Order of operations
#
# 1. **Configuration** — Project root, data root, cohort/age-band list (run once).
# 2. **Sync inputs** — Sync `gold/cohorts` from S3 to local/NVMe (idempotent).
# 3. **Step 3a** — **[1] Updating baseline feature importances.** MC-CV feature importance: one runnable cell per cohort × age_band. Produces `aggregated_feature_importance.csv`; checkpoint skip per cohort/age_band.
# 4. **Update cohort data before Step 3b** — Sync `gold/cohorts`, `gold/medical`, `gold/pharmacy` from S3 (run once before any Step 3b cell) so 3b uses the latest data.
# 5. **Step 3b** — **[2] Identifying target leakage** (BupaR post-target analysis → `*_bupar_post_target_analysis.csv`) and **[3] Removing target leakage from model data training set** (filter_and_refine → `cohort_feature_importance.csv`; Step 4 uses only this list to build model data). One runnable cell per cohort × age_band; for interactive EDA use `step3b_interactive_analysis_cohortN.ipynb`.
# 6. **Step 3c** — **Final update to features passed into Step 4.** Strip any remaining BupaR-identified leakage from each `cohort_feature_importance.csv`. Step 4 uses only these CSVs to build model data; run Step 3c after all Step 3b cells (required).
#
# ## Cohorts
#
# Both cohorts use the **full set of age bands** (from `py_helpers.constants.REQUIRED_COHORTS`): 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114 (8 bands; last band 85-114).
#
# | Cohort | Age bands |
# |--------|-----------|
# | **OPIOID_ED** | Full set (0-12 through 85-114) |
# | **POLYPHARMACY** (non_opioid_ed) | Full set (same) |
#
# ## Workflow: baseline FI, target leakage, model data
#
# 1. **Updating baseline feature importances** — **Step 3a** (MC-CV feature importance) produces `aggregated_feature_importance.csv` per cohort × age_band. These are the baseline feature lists that Step 3b consumes.
# 2. **Identifying target leakage** — **Step 3b** runs BupaR post-target analysis and `create_bupar_post_target_analysis.py`, producing `*_bupar_post_target_analysis.csv` with `is_post_target_leakage` per feature.
# 3. **Removing target leakage from model data training set** — **Step 3b** runs `filter_and_refine_features.py`, which removes post-target leakage (and other filters) from the feature list and writes `cohort_feature_importance.csv`. **Step 3c** is the final update: strip any remaining BupaR-identified leakage from those CSVs so the feature list passed to Step 4 is clean. **Step 4** (4_model_data) uses only `cohort_feature_importance.csv` to build model data.
#
# ## Reference
#
# - Step 3a: `3a_feature_importance/run_mc_feature_importance.py`
# - Step 3b: `3b_feature_importance_eda/feature_importance_eda_workflow.py`; interactive EDA in `3b_feature_importance_eda/`
# - Step 3c: this notebook (final update to features passed into Step 4; required)
# - Step 4: `4_model_data/create_model_data.py` — builds model data from `cohort_feature_importance.csv`
# - Step 6: `6_final_model/` — model training and selection.

# %% [markdown]
# ## Configuration

# %%
import sys
import os
from pathlib import Path
import subprocess
import logging

try:
    from IPython.display import Image, display  # type: ignore
except Exception:  # pragma: no cover
    Image = None

    def display(*args, **kwargs):
        return None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.env_utils import get_data_root
from py_helpers.workflow_sync_checkpoint import (
    sync_s3_to_local,
    check_step_checkpoint_exists,
    save_step_checkpoint,
)
from py_helpers.feature_importance_heatmap import create_aggregated_fi_heatmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PYTHON_BIN = Path(sys.executable)
S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
DATA_ROOT = get_data_root()
AWS_PROFILE = os.environ.get("AWS_PROFILE")

try:
    from py_helpers.constants import REQUIRED_COHORTS
    COHORTS = REQUIRED_COHORTS
except ImportError:
    _all_bands = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
    COHORTS = {"opioid_ed": _all_bands, "non_opioid_ed": _all_bands}

# Set True to overwrite/force Step 3a feature importance (rerun even when results exist)
FORCE_FEATURE_IMPORTANCE = False

print(f"Project root: {PROJECT_ROOT}")
print(f"Data root (NVMe/local): {DATA_ROOT}")
print(f"Python: {PYTHON_BIN}")
print(f"Force feature importance (overwrite): {FORCE_FEATURE_IMPORTANCE}")

# Step 3a outputs base (same as run_mc_feature_importance.py)
OUTPUTS_BASE_3A = Path(os.environ.get("PGX_FEATURE_IMPORTANCE_OUTPUTS", str(PROJECT_ROOT / "3a_feature_importance" / "outputs")))

# %% [markdown]
# ## Sync required inputs from S3 to NVMe (idempotent)
#
# Sync **gold/cohorts** from S3 so Step 3a can read cohort parquet from local/NVMe. **Idempotent:** `aws s3 sync` only updates changed or missing files.

# %%
# Sync gold/cohorts from S3 to local/NVMe (required for 3a feature importance)
s3_cohorts = f"s3://{S3_BUCKET}/gold/cohorts/"
local_cohorts = DATA_ROOT / "gold" / "cohorts"
ok = sync_s3_to_local(s3_cohorts, local_cohorts, profile=AWS_PROFILE)
print(f"  gold/cohorts: {'OK' if ok else 'FAILED or skipped (no AWS CLI)'}")

# %% [markdown]
# ## pgx-repository baseline (aggregated feature importance)
#
# Current aggregated feature importance files in **pgx-repository** (used by Step 3a second pass as baseline). Run this cell to confirm row counts and feature counts before running Step 3a.

# %%
# Load and display pgx-repository baseline summary (all cohort/age_band)
import sys
sys.path.insert(0, str(PROJECT_ROOT / "3a_feature_importance"))
from load_pgx_repo_fi import get_baseline_summary_df

baseline_df = get_baseline_summary_df()
display(baseline_df)

# %% [markdown]
# ## Step 3a: MC-CV feature importance
#
# Monte Carlo CV feature importance (CatBoost, XGBoost, XGBoost RF). **One runnable cell per cohort × age_band** so you can run and troubleshoot a single combination. Skipped when checkpoint exists for that cohort/age_band. Set **FORCE_FEATURE_IMPORTANCE = True** in Configuration to overwrite/force rerun (passes **--force** to the script). After running the age_band cells for a cohort, run that cohort’s heatmap cell to build the aggregated feature importance heatmap (feature × age band).

# %% [markdown]
# ### (Optional) Run Step 3a for all cohorts × age bands
#
# One cell to run Step 3a for **both cohorts** and **all age bands**. Respects checkpoint skip and **FORCE_FEATURE_IMPORTANCE** from Configuration.

# %%
# Step 3a: run for all cohorts and all age bands (loop)
# Uses same checkpoint/skip and FORCE_FEATURE_IMPORTANCE as the per–age_band cells below.
step_name_3a = "3a_feature_importance"
script_3a = PROJECT_ROOT / "3a_feature_importance" / "run_mc_feature_importance.py"
for cohort, age_bands in COHORTS.items():
    for age_band in age_bands:
        if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
            print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
        else:
            print(f"Running Step 3a for {cohort}/{age_band}...")
            cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
            if FORCE_FEATURE_IMPORTANCE:
                cmd.append("--force")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode == 0:
                save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
            print(f"  {cohort}/{age_band} exit code: {result.returncode}")
print("Step 3a loop done.")

# %% [markdown]
# ### Cohort 1: OPIOID_ED — one cell per age_band

# %%
# Step 3a: opioid_ed / 0-12 only
step_name_3a = "3a_feature_importance"
script_3a = PROJECT_ROOT / "3a_feature_importance" / "run_mc_feature_importance.py"
cohort, age_band = "opioid_ed", "0-12"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 13-24 only
step_name_3a = "3a_feature_importance"
script_3a = PROJECT_ROOT / "3a_feature_importance" / "run_mc_feature_importance.py"
cohort, age_band = "opioid_ed", "13-24"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 25-44 only
cohort, age_band = "opioid_ed", "25-44"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 45-54 only
cohort, age_band = "opioid_ed", "45-54"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 55-64 only
cohort, age_band = "opioid_ed", "55-64"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 65-74 only
cohort, age_band = "opioid_ed", "65-74"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 75-84 only
cohort, age_band = "opioid_ed", "75-84"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: opioid_ed / 85-114 only
cohort, age_band = "opioid_ed", "85-114"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Aggregated feature importance heatmap for OPIOID_ED (run after age_band cells above)
cohort = "opioid_ed"
heatmap_path = create_aggregated_fi_heatmap(cohort, COHORTS[cohort], OUTPUTS_BASE_3A, top_n=50)
if heatmap_path and heatmap_path.exists():
    print(f"Heatmap saved: {heatmap_path}")
    display(Image(filename=str(heatmap_path)))
else:
    print("Heatmap skipped (need at least 2 age bands with aggregated CSVs).")

# %% [markdown]
# ### Cohort 2: POLYPHARMACY (non_opioid_ed) — one cell per age_band

# %%
# Step 3a: non_opioid_ed / 0-12 only
cohort, age_band = "non_opioid_ed", "0-12"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 13-24 only
cohort, age_band = "non_opioid_ed", "13-24"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 25-44 only
cohort, age_band = "non_opioid_ed", "25-44"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 45-54 only
cohort, age_band = "non_opioid_ed", "45-54"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 55-64 only
cohort, age_band = "non_opioid_ed", "55-64"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 65-74 only
cohort, age_band = "non_opioid_ed", "65-74"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 75-84 only
cohort, age_band = "non_opioid_ed", "75-84"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Step 3a: non_opioid_ed / 85-114 only
cohort, age_band = "non_opioid_ed", "85-114"
if not FORCE_FEATURE_IMPORTANCE and check_step_checkpoint_exists(step_name_3a, cohort, age_band, logger):
    print(f"Step 3a already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3a for {cohort}/{age_band}...")
    cmd = [str(PYTHON_BIN), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    if FORCE_FEATURE_IMPORTANCE:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        save_step_checkpoint(step_name_3a, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %%
# Aggregated feature importance heatmap for POLYPHARMACY (run after age_band cells above)
cohort = "non_opioid_ed"
heatmap_path = create_aggregated_fi_heatmap(cohort, COHORTS[cohort], OUTPUTS_BASE_3A, top_n=50)
if heatmap_path and heatmap_path.exists():
    print(f"Heatmap saved: {heatmap_path}")
    display(Image(filename=str(heatmap_path)))
else:
    print("Heatmap skipped (need at least 2 age bands with aggregated CSVs).")

# %% [markdown]
# ## Final combined feature importance heatmap (per cohort, all age bands)
#
# One heatmap **per cohort**: feature × age band for that cohort. Rows = top features (union across age bands), columns = age bands for that cohort. Run after the Step 3a age_band cells for each cohort.

# %%
# Combined heatmap per cohort: feature × age band for each cohort (all age bands for that cohort)
for cohort in COHORTS:
    heatmap_path = create_aggregated_fi_heatmap(cohort, COHORTS[cohort], OUTPUTS_BASE_3A, top_n=80)
    if heatmap_path and heatmap_path.exists():
        print(f"Combined heatmap for {cohort} saved: {heatmap_path}")
        display(Image(filename=str(heatmap_path)))
    else:
        print(f"Combined heatmap for {cohort} skipped (need at least 2 age bands with aggregated CSVs).")

# %% [markdown]
# ## Step 3b: Remove Target Leakage
#
# BupaR post-target analysis and refined `cohort_feature_importance.csv` in `3b_feature_importance_eda/outputs/`. **One runnable cell per cohort × age_band** (same as Step 3a): run the cell for the combination you want; skipped when checkpoint exists. For interactive EDA and plots use `3b_feature_importance_eda/step3b_interactive_analysis_cohort1.ipynb` … `cohort7.ipynb`.

# %% [markdown]
# ### Update cohort data before Step 3b
#
# Sync **gold/cohorts**, **gold/medical**, and **gold/pharmacy** from S3 so Step 3b has the latest data when building model_events (gold cohort filtered by 3a FI + admin removed). Run this cell **once before running any Step 3b cell** to keep the pipeline seamless between 3a and 3b.

# %%
# Sync cohort and gold medical/pharmacy before Step 3b (idempotent)
S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
data_root = get_data_root()
syncs = [
    (f"s3://{S3_BUCKET}/gold/cohorts/", data_root / "gold" / "cohorts"),
    (f"s3://{S3_BUCKET}/gold/medical/", data_root / "gold" / "medical"),
    (f"s3://{S3_BUCKET}/gold/pharmacy/", data_root / "gold" / "pharmacy"),
]
for s3_prefix, local_dir in syncs:
    ok = sync_s3_to_local(s3_prefix, local_dir, profile=AWS_PROFILE)
    print(f"  {local_dir.name}: {'OK' if ok else 'FAILED or skipped'}")
print("Cohort data updated. Ready for Step 3b.")

# %% [markdown]
# ### Step 3b: opioid_ed / 0-12

# %%
# Step 3b: opioid_ed / 0-12 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "0-12"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 13-24

# %%
# Step 3b: opioid_ed / 13-24 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "13-24"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 25-44

# %%
# Step 3b: opioid_ed / 25-44 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "25-44"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 45-54

# %%
# Step 3b: opioid_ed / 45-54 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "45-54"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 55-64

# %%
# Step 3b: opioid_ed / 55-64 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "55-64"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 65-74

# %%
# Step 3b: opioid_ed / 65-74 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "65-74"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 75-84

# %%
# Step 3b: opioid_ed / 75-84 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "75-84"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: opioid_ed / 85-114

# %%
# Step 3b: opioid_ed / 85-114 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "opioid_ed", "85-114"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 0-12

# %%
# Step 3b: non_opioid_ed / 0-12 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "0-12"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 13-24

# %%
# Step 3b: non_opioid_ed / 13-24 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "13-24"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 25-44

# %%
# Step 3b: non_opioid_ed / 25-44 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "25-44"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 45-54

# %%
# Step 3b: non_opioid_ed / 45-54 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "45-54"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 55-64

# %%
# Step 3b: non_opioid_ed / 55-64 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "55-64"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 65-74

# %%
# Step 3b: non_opioid_ed / 65-74 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "65-74"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 75-84

# %%
# Step 3b: non_opioid_ed / 75-84 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "75-84"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ### Step 3b: non_opioid_ed / 85-114

# %%
# Step 3b: non_opioid_ed / 85-114 only (one cell per cohort × age_band)
STEP3B_DIR = PROJECT_ROOT / "3b_feature_importance_eda"
script_3b = STEP3B_DIR / "feature_importance_eda_workflow.py"
step_name_3b = "3b_feature_importance_eda"
cohort, age_band = "non_opioid_ed", "85-114"
if check_step_checkpoint_exists(step_name_3b, cohort, age_band, logger):
    print(f"Step 3b already completed for {cohort}/{age_band}. Skipping.")
else:
    print(f"Running Step 3b for {cohort}/{age_band}...")
    result = subprocess.run(
        [str(PYTHON_BIN), str(script_3b), "--cohort", cohort, "--age-band", age_band],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        save_step_checkpoint(step_name_3b, cohort, age_band, logger=logger)
    print(f"  {cohort}/{age_band} exit code: {result.returncode}")

# %% [markdown]
# ## Step 3c: Final update to features passed into Step 4
#
# **Required.** Final update to the feature list passed into Step 4: strip any remaining BupaR-identified post-target leakage from each `cohort_feature_importance.csv`. Step 4 uses only these CSVs to build model data. Run after all Step 3b cohort cells.

# %%
# Update final model features: remove BupaR-identified target leakage from cohort_feature_importance
import pandas as pd
from py_helpers.feature_importance_eda_utils import resolve_cohort_fi_path

OUTPUTS_3B = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs"
updated_count = 0
for cohort in COHORTS:
    for age_band in COHORTS[cohort]:
        age_fname = age_band.replace("-", "_")
        # Resolve cohort_feature_importance from 3b outputs, DATA_ROOT/gold, or S3 (same as Step 4/6)
        refined_path = resolve_cohort_fi_path(cohort, age_band, PROJECT_ROOT)
        bupar_path = OUTPUTS_3B / cohort / age_fname / f"{cohort}_{age_fname}_bupar_post_target_analysis.csv"
        if refined_path is None or not refined_path.exists():
            print(f"  Skip {cohort}/{age_band}: no cohort_feature_importance.csv (checked 3b/outputs, DATA_ROOT/gold/feature_importance, S3)")
            continue
        if not bupar_path.exists():
            print(f"  Skip {cohort}/{age_band}: no BupaR post-target analysis CSV")
            continue
        refined = pd.read_csv(refined_path)
        bupar = pd.read_csv(bupar_path)
        if "feature" not in refined.columns:
            print(f"  Skip {cohort}/{age_band}: no 'feature' column in refined CSV")
            continue
        leakage = set()
        if "is_post_target_leakage" in bupar.columns and "feature" in bupar.columns:
            leakage = set(bupar.loc[bupar["is_post_target_leakage"] == 1, "feature"].dropna().astype(str).tolist())
        if not leakage:
            msg = f"  {cohort}/{age_band}: no leakage features to remove"
            if cohort == "non_opioid_ed":
                msg += " (expected for polypharmacy: events only up to windowed HCG/ED)"
            print(msg)
            continue
        n_before = len(refined)
        refined_clean = refined[~refined["feature"].astype(str).isin(leakage)].copy()
        n_after = len(refined_clean)
        removed = n_before - n_after
        if removed > 0:
            refined_clean.to_csv(refined_path, index=False)
            print(f"  {cohort}/{age_band}: removed {removed} leakage features; {n_after} features saved to {refined_path.name}")
            updated_count += 1
        else:
            print(f"  {cohort}/{age_band}: leakage set had no overlap with refined features (already clean)")
if updated_count > 0:
    print(f"\nUpdated {updated_count} cohort_feature_importance file(s). Ready for Step 4.")
else:
    print("\nNo files updated (already clean or missing BupaR/refined outputs).")

# %%
