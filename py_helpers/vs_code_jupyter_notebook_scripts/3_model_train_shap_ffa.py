# -*- coding: utf-8 -*-
# Auto-generated from 3_model_train_shap_ffa.ipynb (VS Code Python notebook script format)

# %% [markdown]
# # 3. Model train + SHAP/FFA analysis
#
# **Purpose:** Run pipeline (model data → PGx → final model), then SHAP and FFA analysis and combine results for the Causal tab. No build or deploy here.
#
# **Flow:** Run this notebook first. Then [4_dashboard_visuals.ipynb](4_dashboard_visuals.ipynb) (BupaR, DTW, FP-Growth). Then [5_build_and_deploy.ipynb](5_build_and_deploy.ipynb).
#
# **Steps:** Sync inputs → Verify → Pipeline Phase 4 (model data) → Phase 5 (PGx) → Phase 6 (final model) → Step 1a (metadata) → Step 7 (SHAP) → Step 8 (FFA) → Combine (SHAP+FFA to dashboard) → optional inspection.
#
# **Memory:** Pipeline scripts use **DuckDB and Parquet** where possible for efficient memory use (Step 4 model data, Step 6/7 SHAP/FFA data prep, combine); pandas is used only where required (e.g. model/SHAP APIs). See project `.cursorrules` for data-processing preferences.\n\nPrerequisites: Cohorts (Step 2), feature importance (Step 3/3b). Run from repo root.

# %%
# Setup: paths and project root
import sys
import os
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
from py_helpers.env_utils import get_data_root
from py_helpers.workflow_sync_checkpoint import sync_s3_to_local, check_step_checkpoint_exists, save_step_checkpoint

DASHBOARD_DIR = PROJECT_ROOT / "10_risk_dashboard"
DATA_PREP_DIR = DASHBOARD_DIR / "data_preparation"
DEPLOY_DIR = DASHBOARD_DIR / "deployment"
S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
DATA_ROOT = get_data_root()
AWS_PROFILE = os.environ.get("AWS_PROFILE")

print("PGx Risk Calculator Workflow")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Dashboard dir: {DASHBOARD_DIR}")
print(f"Data prep: {DATA_PREP_DIR}")
print(f"Data root (NVMe/local): {DATA_ROOT}")
print("=" * 60)

# %%
# Both cohorts use full age band set
from py_helpers.constants import REQUIRED_COHORTS

# Input dirs (required for pipeline Step 4–6)
# Cohorts: Step 2 cohort.parquet files (create_model_data reads case/control and target dates from here).
COHORTS_ROOT = DATA_ROOT / "gold" / "cohorts"
# Feature importance: Step 3/3b outputs — cohort_feature_importance.csv and feature_filtering_summary.json per cohort/age_band.
FI_ROOT = DATA_ROOT / "gold" / "feature_importance"
STEP3_OUTPUTS = STEP3B_OUTPUTS = FI_ROOT
# Model data: single canonical location (Step 4 output, Step 5/6 input).
from py_helpers.env_utils import get_model_data_root
MODEL_DATA_ROOT = get_model_data_root()

# Output dirs (Step 6 final model outputs; data prep and Lambda read from these)
FINAL_MODEL_OUTPUTS = PROJECT_ROOT / "6_final_model" / "outputs"
FINAL_MODEL_OUTPUTS_ALT = DATA_ROOT / "6_final_model" / "outputs"
FINAL_MODEL_GOLD = DATA_ROOT / "gold" / "final_model"  # S3 layout: cohort/13-24/*.joblib

print("Cohorts and age bands:")
for cohort, bands in REQUIRED_COHORTS.items():
    print(f"  {cohort}: {bands}")
print("\nInput dirs (for Step 4–6):")
print(f"  Cohorts (2):              {COHORTS_ROOT}")
print(f"  Feature importance (3/3b): {FI_ROOT}  (CSVs + feature_filtering_summary.json)")
print(f"  Model data (4; in/out):   {MODEL_DATA_ROOT}")
print("\nOutput dirs (Step 6):")
print(f"  Project:   {FINAL_MODEL_OUTPUTS}")
print(f"  NVMe:      {FINAL_MODEL_OUTPUTS_ALT}")
print(f"  gold/NVMe: {FINAL_MODEL_GOLD}")

# Run scopes for this notebook.
# Step 4 builds model_events.parquet. Downstream controls Step 5 and later.
# Set age bands to None for all configured age bands, or a list like ["75-84"].
STEP4_COHORTS = ["opioid_ed", "non_opioid_ed"]
STEP4_AGE_BANDS = None
DOWNSTREAM_COHORTS = ["non_opioid_ed"]
DOWNSTREAM_AGE_BANDS = None

def iter_scope(cohorts=None, age_bands=None):
    for cohort, bands in REQUIRED_COHORTS.items():
        if cohorts is not None and cohort not in cohorts:
            continue
        for age_band in bands:
            if age_bands is not None and age_band not in age_bands:
                continue
            yield cohort, age_band

def iter_step4_cohorts():
    yield from iter_scope(STEP4_COHORTS, STEP4_AGE_BANDS)

def iter_downstream_cohorts():
    yield from iter_scope(DOWNSTREAM_COHORTS, DOWNSTREAM_AGE_BANDS)

print("\nStep 4 model-data build scope:")
for cohort, age_band in iter_step4_cohorts():
    print(f"  {cohort} / {age_band}")
print("\nDownstream Step 5+ scope:")
for cohort, age_band in iter_downstream_cohorts():
    print(f"  {cohort} / {age_band}")

# %% [markdown]
# ## Clear all checkpoints and pipeline outputs (optional — for a fresh run)
#
# Run this cell **once** when you want to rebuild the full pipeline from Step 4 through SHAP/FFA from scratch. It (1) clears S3 **checkpoints** (pgx-repository: 4_model_data, 6_final_model, 9_dashboard_metadata), (2) deletes S3 **pipeline outputs** (pgxdatalake: `gold/cohorts_model_data/`, `gold/final_model/`) so Step 4 and Step 6 re-run instead of re-downloading, (3) removes local output directories. After this, run the Sync cell and then Steps 4 → 5 → 6 → 1a → Step 7 → Step 8 → Combine.

# %%
# Clear S3 checkpoints, S3 pipeline outputs, and local outputs for a fresh model + SHAP/FFA run.
import shutil
import subprocess
from py_helpers.workflow_sync_checkpoint import clear_step_checkpoints, delete_step_checkpoint

# 1) S3 checkpoint metadata (pgx-repository) so steps don't think they're done
for step in ("4_model_data", "6_final_model"):
    for cohort, bands in REQUIRED_COHORTS.items():
        n = clear_step_checkpoints(step, cohort, bands, logger=None)
        print(f"Cleared {n} checkpoint(s) for {step} / {cohort}")
delete_step_checkpoint("9_dashboard_metadata", "all", "all", logger=None)
print("Cleared checkpoint 9_dashboard_metadata (all/all)")

# 2) S3 pipeline outputs (pgxdatalake) so Step 4 and Step 6 re-run instead of re-downloading
_aws = shutil.which("aws") or "aws"
_profile = ["--profile", AWS_PROFILE] if AWS_PROFILE else []
for prefix in ("gold/cohorts_model_data/", "gold/final_model/"):
    uri = f"s3://{S3_BUCKET}/{prefix}"
    r = subprocess.run([_aws, "s3", "rm", uri, "--recursive"] + _profile, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"Cleared S3 {uri}")
    else:
        print(f"S3 rm {uri}: exit {r.returncode} (check credentials); {r.stderr or r.stdout or ''}")

# 3) Local output directories
dirs_to_clear = [
    MODEL_DATA_ROOT,
    FINAL_MODEL_OUTPUTS,
    FINAL_MODEL_OUTPUTS_ALT,
    PROJECT_ROOT / "7_shap_analysis" / "outputs",
    PROJECT_ROOT / "8_ffa_analysis" / "outputs",
    PROJECT_ROOT / "10_risk_dashboard" / "outputs",
]
for d in dirs_to_clear:
    d = Path(d)
    if d.exists():
        shutil.rmtree(d)
        print(f"Removed {d}")
    else:
        print(f"(skip, not present) {d}")
print("Done. Re-run Sync and then Steps 4 → 5 → 6 → 1a → Step 7 → Step 8 → Combine for a fresh model and SHAP/FFA outputs.")

# %% [markdown]
# ## Sync required inputs from S3 to NVMe (idempotent)
#
# Sync **cohorts** (Step 2), **feature importance** (Step 3/3b), and **Step 6** final model outputs from S3 so pipeline and data preparation can read from local/NVMe. **Idempotent:** `aws s3 sync` only updates changed or missing files.

# %%
# Sync cohorts (Step 2), Step 3a/3b feature importance, and Step 6 final models from S3 to NVMe (DATA_ROOT).
# Cohorts -> COHORTS_ROOT (gold/cohorts); Feature importance -> gold/feature_importance; Step 6 -> gold/final_model.
COHORTS_ROOT.mkdir(parents=True, exist_ok=True)
FI_SYNC_TARGET = DATA_ROOT / "gold" / "feature_importance"
FI_SYNC_TARGET.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_GOLD.mkdir(parents=True, exist_ok=True)

sync_s3_to_local(f"s3://{S3_BUCKET}/gold/cohorts/", COHORTS_ROOT, profile=AWS_PROFILE)
sync_s3_to_local(f"s3://{S3_BUCKET}/gold/feature_importance/", FI_SYNC_TARGET, profile=AWS_PROFILE)
sync_s3_to_local(f"s3://{S3_BUCKET}/gold/final_model/", FINAL_MODEL_GOLD, profile=AWS_PROFILE)
print("Sync complete. Run Step 0 verification below.")

# %% [markdown]
# ## Step 0: Verify inputs (FI required; 4_model_data and Step 6 informational)
#
# **Required:** **Feature importance** (Step 3/3b) — must exist for each cohort/age_band so Pipeline Step 4 can run.
#
# **Informational:** **ModelData** checks `DATA_ROOT/4_model_data` and `PROJECT_ROOT/4_model_data` (same location `create_model_data.py` writes to). **Model** = Step 6 outputs. Both are produced by Pipeline Step 4–6 cells below; if already present, you can skip those cells.

# %%
def check_feature_importance(cohort: str, age_band: str) -> bool:
    """Check if feature importance exists using FileResolver pattern."""
    from py_helpers.file_resolver import FileResolver
    # Check Step 3b refined cohort feature importance first
    resolver_3b = FileResolver(
        file_type="cohort_feature_importance",
        project_root=PROJECT_ROOT,
        cohort=cohort,
        age_band=age_band,
        auto_download=False
    )
    if resolver_3b.exists():
        return True
    # Fallback to Step 3a aggregated feature importance
    resolver_3a = FileResolver(
        file_type="aggregated_feature_importance",
        project_root=PROJECT_ROOT,
        cohort=cohort,
        age_band=age_band,
        auto_download=False
    )
    return resolver_3a.exists()

def check_cohorts(cohort: str, age_band: str) -> bool:
    """Check Step 2 cohort.parquet exists for at least one year (2016–2019). Layout: COHORTS_ROOT/cohort_name=X/event_year=Y/age_band=Z/cohort.parquet."""
    for year in (2016, 2017, 2018, 2019):
        p = COHORTS_ROOT / f"cohort_name={cohort}" / f"event_year={year}" / f"age_band={age_band}" / "cohort.parquet"
        if p.exists():
            return True
    return False

def check_model_data(cohort: str, age_band: str) -> bool:
    """Check model_events.parquet at canonical MODEL_DATA_ROOT (same location create_model_data.py writes to)."""
    p = MODEL_DATA_ROOT / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet"
    return p.exists()

def check_final_model(cohort: str, age_band: str) -> bool:
    ab = age_band.replace("-", "_")
    # 1) Project or DATA_ROOT/6_final_model/outputs: cohort/13_24/models/*.joblib
    for base in (FINAL_MODEL_OUTPUTS, FINAL_MODEL_OUTPUTS_ALT):
        model_dir = base / cohort / ab
        if not model_dir.exists():
            continue
        models_sub = model_dir / "models"
        if models_sub.exists() and any(models_sub.glob("*.joblib")):
            return True
        if (model_dir / "feature_schema.json").exists():
            return True
    # 2) DATA_ROOT/gold/final_model (S3-synced): cohort/13-24/*.joblib (hyphen in age_band)
    gold_dir = FINAL_MODEL_GOLD / cohort / age_band
    if gold_dir.exists() and any(gold_dir.glob("*.joblib")):
        return True
    return False

print("Step 0: Verify feature importance (required); cohorts and 4_model_data (Step 4 inputs); Step 6 (informational)")
print("  Locations: Cohorts=COHORTS_ROOT, FI=Step 3/3b, ModelData=MODEL_DATA_ROOT, Model=Step 6 outputs")
fi_ok_all = True
for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        cohorts_ok = check_cohorts(cohort, age_band)
        fi_ok = check_feature_importance(cohort, age_band)
        model_data_ok = check_model_data(cohort, age_band)
        model_ok = check_final_model(cohort, age_band)
        if not fi_ok:
            fi_ok_all = False
        status = "ready" if fi_ok else "missing FI"
        print(f"  {cohort} / {age_band}:  Cohorts={cohorts_ok}, FI={fi_ok}, ModelData={model_data_ok}, Model={model_ok}  -> {status}")
if fi_ok_all:
    print("\nAll prerequisites are available to build model data. Run Pipeline Step 4–6 cells below.")
    print("  (If Step 6 is already built elsewhere, you can sync from S3 or skip those cells.)")
else:
    print("\nMissing feature importance for some cohort/age_band. Sync from S3 or run Step 3/3b first, then re-run this cell.")
if fi_ok_all:
    cohorts_missing = [(c, ab) for c, bands in REQUIRED_COHORTS.items() for ab in bands if not check_cohorts(c, ab)]
    if cohorts_missing:
        print("\nCohorts=False for some cohort/age_band. Sync gold/cohorts from S3 (run Sync cell) or run Step 2. Expected layout: COHORTS_ROOT/cohort_name=X/event_year=Y/age_band=Z/cohort.parquet (Y in 2016–2019).")

# %% [markdown]
# # Pipeline Phase 4: Model data
#
# Build `model_events.parquet` for each cohort/age_band from Step 2 cohort data and Step 3b feature importance. Outputs go to `MODEL_DATA_ROOT/cohort_name={cohort}/age_band={age_band}/model_events.parquet`. Run the cell below for all PGx cohorts/age_bands defined in this notebook.

# %%
# Pipeline Step 4: BUILD model_events.parquet by running create_model_data.py, then QA.
# The script READS: COHORTS_ROOT (cohort.parquet), gold/medical, gold/pharmacy, and feature importance.
# It WRITES: MODEL_DATA_ROOT/cohort_name={cohort}/age_band={age_band}/model_events.parquet
import duckdb

FORCE_STEP4_NON_OPIOID = True
FORCE_STEP4_ALL = False
STEP4_BACKUP_ROOT = DATA_ROOT / "backups" / "step4_model_events_pre_rebuild"
REBUILT_STEP4 = set()

def _model_data_candidates(cohort: str, age_band: str):
    """Canonical location for model_events.parquet (Step 4 writes to MODEL_DATA_ROOT)."""
    return [MODEL_DATA_ROOT]

def _model_data_path(cohort: str, age_band: str) -> Path:
    """Resolve model_events.parquet path (Step 4 writes to get_model_data_root() = DATA_ROOT or PROJECT on Linux)."""
    for base in _model_data_candidates(cohort, age_band):
        p = base / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet"
        if p.exists():
            return p
    return None

def _backup_existing_model_data_if_forced(cohort: str, age_band: str) -> None:
    force = FORCE_STEP4_ALL or (FORCE_STEP4_NON_OPIOID and cohort == "non_opioid_ed")
    if not force:
        return
    path = _model_data_path(cohort, age_band)
    if not path:
        return
    backup_dir = STEP4_BACKUP_ROOT / f"cohort_name={cohort}" / f"age_band={age_band}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "model_events.parquet"
    if backup_path.exists():
        backup_path.unlink()
    shutil.move(str(path), str(backup_path))
    print(f"  [FORCE] Moved existing model_events.parquet to {backup_path}")

def _log_model_data_qa(cohort: str, age_band: str) -> None:
    """Log location, target distribution, and control:case ratio for model_events.parquet."""
    path = _model_data_path(cohort, age_band)
    if not path:
        print(f"  [WARN] model_events.parquet not found for {cohort}/{age_band}")
        for base in _model_data_candidates(cohort, age_band):
            p = base / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet"
            print(f"    Checked: {p}  (exists: {p.exists()})")
        print(f"    Build did not write output. Check script stdout above: [INFO] data roots and example cohort path (exists=?). Layout must be {COHORTS_ROOT}/cohort_name=X/event_year=Y/age_band=Z/cohort.parquet (Y in 2016–2019). Sync cohorts to COHORTS_ROOT if needed, then re-run this cell.")
        return
    print(f"  Location: {path}")
    con = duckdb.connect()
    try:
        dist = con.execute("SELECT target, COUNT(*)::BIGINT AS n FROM read_parquet(?) GROUP BY target ORDER BY target", [str(path)]).fetchall()
        total = sum(row[1] for row in dist)
        by_target = {int(row[0]): int(row[1]) for row in dist}
        n_controls = by_target.get(0, 0)
        n_cases = by_target.get(1, 0)
        ratio = (n_controls / n_cases) if n_cases else 0
        print(f"  Target distribution: {by_target} (total rows: {total:,})")
        print(f"  Control:case ratio: {n_controls:,}:{n_cases:,} = {ratio:.2f}:1")
        event_counts = con.execute(
            """
            SELECT
              target,
              COUNT(DISTINCT mi_person_key)::BIGINT AS patients,
              AVG(n_events) AS mean_events,
              MEDIAN(n_events) AS median_events,
              MIN(n_events) AS min_events,
              MAX(n_events) AS max_events
            FROM (
              SELECT mi_person_key, target, COUNT(*) AS n_events
              FROM read_parquet(?)
              GROUP BY mi_person_key, target
            )
            GROUP BY target
            ORDER BY target
            """,
            [str(path)],
        ).df()
        print("  Patient-level event-count QA:")
        print(event_counts.to_string(index=False))
    finally:
        con.close()

for cohort, age_band in iter_step4_cohorts():
    print(f"→ Step 4: {cohort} / {age_band} (building model_events.parquet)")
    force_step4 = FORCE_STEP4_ALL or (FORCE_STEP4_NON_OPIOID and cohort == "non_opioid_ed")
    _backup_existing_model_data_if_forced(cohort, age_band)
    cmd = [sys.executable, "create_model_data.py", "--cohort", cohort, "--age_band", age_band]
    if force_step4:
        cmd.append("--force-rebuild")
    r = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT / "4_model_data",
        capture_output=False,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    REBUILT_STEP4.add((cohort, age_band))
    _log_model_data_qa(cohort, age_band)
print("Step 4 complete.")

# %% [markdown]
# # Pipeline Phase 5: PGx analysis
#
# Add PGx features (e.g. CPIC drug counts) to model data. Reads from Step 4 outputs and writes updated model data used by Step 6. Run for each cohort/age_band.

# %%
# Pipeline Step 5: run_analysis.py for each REQUIRED_COHORTS (cohort, age_band)
# Set FORCE_STEP5 = True to re-run even when S3 outputs or checkpoints exist
FORCE_STEP5 = True
for cohort, age_band in iter_downstream_cohorts():
    print(f"→ Step 5: {cohort} / {age_band}")
    cmd = [sys.executable, "run_analysis.py", "--cohort-name", cohort, "--age-band", age_band]
    if FORCE_STEP5:
        cmd.append("--force")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT / "5_pgx_analysis")
    if r.returncode != 0:
        raise SystemExit(r.returncode)
print("Step 5 complete.")

# %% [markdown]
# # Pipeline Phase 6: Final model deployment outputs
#
# Train final models per cohort/age_band. **Default** is **per-bin** training (`--train-mode per_bin`): separate models under `outputs/.../bin_models/{low|medium|high|extreme}/`, then mirror one bin (prefer `medium`) to the cohort-level `outputs/.../{age_band}/` tree for prepare_models / Lambda. Use `--train-mode aggregate` for a single cohort-wide model only, or `both` for cohort-wide plus per-bin. Reads Step 4 model data and Step 5 PGx features; writes models and feature CSVs to `6_final_model/outputs` (or DATA_ROOT).

# %%
# Pipeline Step 6: run_final_model.py for each REQUIRED_COHORTS (cohort, age_band)
# Note: script uses --age_band (underscore). Default --train-mode is per_bin (omit flag).
FORCE_STEP6 = False
FORCE_STEP6_REBUILT_ONLY = True
STEP6_TRAIN_MODE = None
for cohort, age_band in iter_downstream_cohorts():
    print(f"→ Step 6: {cohort} / {age_band}")
    cmd = [sys.executable, "run_final_model.py", "--cohort", cohort, "--age_band", age_band]
    if STEP6_TRAIN_MODE:
        cmd.extend(["--train-mode", STEP6_TRAIN_MODE])
    force_step6 = FORCE_STEP6 or (FORCE_STEP6_REBUILT_ONLY and (cohort, age_band) in REBUILT_STEP4)
    if force_step6:
        cmd.append("--force-retrain")
    r = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT / "6_final_model",
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)
print("Step 6 complete.")

# %% [markdown]
# ### Model performance per density bin and Top 20 XGBoost importance (cohort-level snapshot)
#
# Metrics are read from `bin_models/{bin}/` (default Step 6). The cohort-level `.../{age_band}/` XGBoost FI CSV is the **mirrored deploy snapshot** (preferred bin, usually `medium`) when using `--train-mode per_bin`.

# %%
# Per-bin model metrics + cohort-level FI plot (mirrored bin for deploy)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from py_helpers.event_density_utils import DENSITY_BINS as _DENSITY_BINS

# Resolve outputs base (project or NVMe)
def _outputs_base():
    for base in (FINAL_MODEL_OUTPUTS, FINAL_MODEL_OUTPUTS_ALT):
        if base and base.exists():
            return base
    return FINAL_MODEL_OUTPUTS

base = _outputs_base()
print("Final model performance — per density bin (selected model per bin)")
print("=" * 80)
all_metrics = []
for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        ab_f = age_band.replace("-", "_")
        for bin_name in _DENSITY_BINS:
            path = (
                base / cohort / ab_f / "bin_models" / bin_name
                / f"{cohort}_{ab_f}_model_metrics_summary.csv"
            )
            if not path.exists():
                continue
            df = pd.read_csv(path)
            selected = df.loc[df["selected"] == True]
            if selected.empty:
                selected = df.head(1)
            for _, row in selected.iterrows():
                all_metrics.append({
                    "cohort": cohort,
                    "age_band": age_band,
                    "bin": bin_name,
                    "model": row["model"],
                    "recall_mean": row["recall_mean"],
                    "pr_auc_mean": row["pr_auc_mean"],
                    "auc_mean": row.get("auc_mean", None),
                    "logloss_mean": row.get("logloss_mean", None),
                })
if all_metrics:
    summary = pd.DataFrame(all_metrics)
    print(summary.to_string(index=False))
else:
    print("  No per-bin metrics CSVs under", base / "<cohort>" / "<age_band>" / "bin_models")
print()
print("Top 20 feature importance (XGBoost) — cohort-level CSV (deploy mirror when per-bin mode)")
print("=" * 80)
for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        ab_f = age_band.replace("-", "_")
        fi_path = base / cohort / ab_f / f"{cohort}_{ab_f}_xgboost_feature_importance.csv"
        if not fi_path.exists():
            print(f"  [skip] {cohort} / {age_band}: no cohort-level feature importance CSV")
            continue
        fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(20)
        if fi.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(fi)), fi["importance"].values, align="center")
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Top 20 features — {cohort} / {age_band} (cohort-level / deploy snapshot)")
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### Step 1a: Generate Model Metadata
#
# Extract valid codes (drugs, ICD, CPT) from feature importance for dashboard dropdowns. Uses Step 3b `cohort_feature_importance` when available, else Step 3 aggregated. **Checkpoint:** step is skipped if S3 checkpoint exists. Set `FORCE_STEP1A = True` in the cell below to re-run and rebuild `outputs/metadata/` (e.g. for Prepare Lambda directory). Run this before Step 7 (SHAP) so metadata is ready for deployment.

# %%
import logging
logger = logging.getLogger(__name__)
FORCE_STEP1A = True  # Set True to re-run and build outputs/metadata even when checkpoint exists
if not FORCE_STEP1A and check_step_checkpoint_exists("9_dashboard_metadata", "all", "all", logger):
    print("Step 1 (generate metadata) already completed (checkpoint exists). Skipping.")
else:
    r = subprocess.run([sys.executable, "generate_metadata.py", "--all"], cwd=DATA_PREP_DIR)
    if r.returncode == 0:
        save_step_checkpoint("9_dashboard_metadata", "all", "all", logger=logger)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

# %% [markdown]
# ### Step 7: SHAP values
#
# **7_shap_analysis/run_shap_analysis.py** — per-bin SHAP only for bins that Step 6 actually trained (`list_trained_density_bins`); if there are no per-bin models but cohort-level (aggregate) Step 6 outputs exist, runs once **without** `--bin`. Skips cohort/age bands with no Step 6 models. Run this **before** Step 8 (FFA) and Combine.

# %%
# Step 7: SHAP — per trained bin, else cohort-level aggregate if present.
from py_helpers.event_density_utils import (
    cohort_aggregate_final_model_has_artifacts,
    list_trained_density_bins,
)

SHAP_SCRIPT = PROJECT_ROOT / "7_shap_analysis" / "run_shap_analysis.py"
for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        trained = list_trained_density_bins(PROJECT_ROOT, cohort, age_band)
        if trained:
            for bin_name in trained:
                print(f"→ Step 7 (SHAP bin={bin_name}): {cohort} / {age_band}")
                r = subprocess.run(
                    [
                        sys.executable,
                        str(SHAP_SCRIPT),
                        "--cohort",
                        cohort,
                        "--age_band",
                        age_band,
                        "--bin",
                        bin_name,
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=False,
                )
                if r.returncode != 0:
                    raise SystemExit(r.returncode)
        elif cohort_aggregate_final_model_has_artifacts(PROJECT_ROOT, cohort, age_band):
            print(f"→ Step 7 (SHAP cohort-level): {cohort} / {age_band}")
            r = subprocess.run(
                [sys.executable, str(SHAP_SCRIPT), "--cohort", cohort, "--age_band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=False,
            )
            if r.returncode != 0:
                raise SystemExit(r.returncode)
        else:
            print(f"[skip] Step 7: no Step 6 models for {cohort} / {age_band}")
print("Step 7 (SHAP) complete.")

# %% [markdown]
# ### Step 8: FFA rules
#
# **run_shap_ffa_workflow.py** with **--skip-shap --skip-combine** — same bin resolution as Step 7 (trained bins only, else cohort-level if aggregate models exist). Run this **after** Step 7 and **before** Combine.

# %%
# Step 8: FFA — per trained bin, else cohort-level if aggregate Step 6 + Step 7 exist.
from py_helpers.event_density_utils import (
    cohort_aggregate_final_model_has_artifacts,
    list_trained_density_bins,
)

for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        trained = list_trained_density_bins(PROJECT_ROOT, cohort, age_band)
        if trained:
            for bin_name in trained:
                print(f"→ Step 8 (FFA bin={bin_name}): {cohort} / {age_band}")
                r = subprocess.run(
                    [
                        sys.executable,
                        "run_shap_ffa_workflow.py",
                        "--cohort",
                        cohort,
                        "--age-band",
                        age_band,
                        "--bin",
                        bin_name,
                        "--skip-shap",
                        "--skip-combine",
                    ],
                    cwd=DATA_PREP_DIR,
                    capture_output=False,
                )
                if r.returncode != 0:
                    raise SystemExit(r.returncode)
        elif cohort_aggregate_final_model_has_artifacts(PROJECT_ROOT, cohort, age_band):
            print(f"→ Step 8 (FFA cohort-level): {cohort} / {age_band}")
            r = subprocess.run(
                [
                    sys.executable,
                    "run_shap_ffa_workflow.py",
                    "--cohort",
                    cohort,
                    "--age-band",
                    age_band,
                    "--skip-shap",
                    "--skip-combine",
                ],
                cwd=DATA_PREP_DIR,
                capture_output=False,
            )
            if r.returncode != 0:
                raise SystemExit(r.returncode)
        else:
            print(f"[skip] Step 8: no Step 6 models for {cohort} / {age_band}")
print("Step 8 (FFA) complete.")

# %% [markdown]
# ### Combine: SHAP + FFA → dashboard outputs
#
# **combine_shap_ffa_results.py** — per trained bin (under `.../causal/{cohort}/{age_band}/{bin}/`), or cohort-level output under `.../causal/{cohort}/{age_band}/` when Step 7/8 used aggregate paths. Use `--workers 0` for auto worker count or `--workers 1` for sequential.

# %%
# Combine: Merge SHAP + FFA per trained bin, else cohort-level.
from py_helpers.event_density_utils import (
    cohort_aggregate_final_model_has_artifacts,
    list_trained_density_bins,
)

CAUSAL_VISUALS = PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "causal"
COMBINE_SCRIPT = DATA_PREP_DIR / "combine_shap_ffa_results.py"
for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        trained = list_trained_density_bins(PROJECT_ROOT, cohort, age_band)
        if trained:
            for bin_name in trained:
                print(f"→ Combine (bin={bin_name}): {cohort} / {age_band}")
                r = subprocess.run(
                    [
                        sys.executable,
                        str(COMBINE_SCRIPT),
                        "--cohort",
                        cohort,
                        "--age-band",
                        age_band,
                        "--bin",
                        bin_name,
                        "--output-dir",
                        str(CAUSAL_VISUALS),
                        "--workers",
                        "0",
                    ],
                    cwd=DATA_PREP_DIR,
                    capture_output=False,
                )
                if r.returncode != 0:
                    raise SystemExit(r.returncode)
        elif cohort_aggregate_final_model_has_artifacts(PROJECT_ROOT, cohort, age_band):
            print(f"→ Combine (cohort-level): {cohort} / {age_band}")
            r = subprocess.run(
                [
                    sys.executable,
                    str(COMBINE_SCRIPT),
                    "--cohort",
                    cohort,
                    "--age-band",
                    age_band,
                    "--output-dir",
                    str(CAUSAL_VISUALS),
                    "--workers",
                    "0",
                ],
                cwd=DATA_PREP_DIR,
                capture_output=False,
            )
            if r.returncode != 0:
                raise SystemExit(r.returncode)
        else:
            print(f"[skip] Combine: no Step 6 / trained bins for {cohort} / {age_band}")
print("Combine complete.")

# %% [markdown]
# ### Review combined SHAP/FFA and metadata
#
# **Code set by design:**
# - **opioid_ed**: Drug + ICD + CPT (all three used for Causal tab dropdowns).
# - **non_opioid_ed**: Drug only (no ICD/CPT).
#
# **How this aligns to output:** The pipeline enforces this before model and dashboard outputs.
# - **Step 3a** (`py_helpers/feature_importance_utils.py`): For `non_opioid_ed` and age band ≥65, the feature palette is restricted to `drug_name` only, so aggregated feature importance (and thus Step 3b refined FI) contains only drug features.
# - **Step 4**: Uses Step 3b feature list to build `model_events.parquet`; for non_opioid_ed that list is drug-only.
# - **Step 6** (`6_final_model/run_final_model.py`): For non_opioid_ed, any `item_icd_*` and `item_cpt_*` columns are explicitly removed from the final feature matrix (polypharmacy = drugs only).
# - **Step 1a** (generate_metadata): Reads the same feature importance CSVs; opioid_ed gets drugs + ICDs + CPTs, non_opioid_ed gets drugs only (and 0 CPT / few ICD in metadata is expected).
#
# Run the cell below after **Combine** to verify metadata code counts and that combined importance has the expected columns. Dashboard/Lambda use **metadata** (Step 1a) for Drug/ICD/CPT lists; the combine script outputs `feature` names and scores only.

# %%
import json
import pandas as pd
from pathlib import Path

DASHBOARD_OUT = PROJECT_ROOT / "10_risk_dashboard" / "outputs"
CAUSAL_VISUALS = PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "causal"
META_DIR = DASHBOARD_OUT / "metadata"

DRUG_PREFIX = "item_drug_"
ICD_PREFIX  = "item_icd_"
CPT_PREFIX  = "item_cpt_"

def check_cohort_expectations(
    cohort: str,
    n_drugs: int, n_icds: int, n_cpts: int,
    n_drug_f: int, n_icd_f: int, n_cpt_f: int
) -> tuple[bool, str]:
    """Return (matches_expectation, reason)."""
    if cohort == "opioid_ed":
        if n_drugs <= 0 or n_icds <= 0 or n_cpts <= 0:
            return False, (
                f"metadata expected Drug+ICD+CPT (all >0), got drugs={n_drugs} icds={n_icds} cpts={n_cpts}"
            )
        if n_icd_f <= 0 and n_cpt_f <= 0:
            return False, (
                f"combined_importance expected ICD/CPT features, got drug={n_drug_f} icd={n_icd_f} cpt={n_cpt_f}"
            )
        return True, "Drug+ICD+CPT"

    if cohort == "non_opioid_ed":
        if n_drugs <= 0:
            return False, f"metadata expected Drug only (drugs>0), got drugs={n_drugs}"
        if n_icds != 0 or n_cpts != 0:
            return False, (
                f"metadata expected Drug only (icds=0,cpts=0), got icds={n_icds} cpts={n_cpts}"
            )
        if n_icd_f > 0 or n_cpt_f > 0:
            return False, (
                f"combined_importance expected drug-only, got drug={n_drug_f} icd={n_icd_f} cpt={n_cpt_f}"
            )
        return True, "Drug only"

    return True, "no rule (cohort not in expectations)"

def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

print("Combined SHAP/FFA and metadata review (by cohort)")
print("Expected: opioid_ed = Drug + ICD + CPT; non_opioid_ed = Drug only")
print("=" * 90)

all_ok = True

for cohort, bands in REQUIRED_COHORTS.items():
    meta_path = META_DIR / f"metadata_{cohort}.json"
    meta_exists = meta_path.exists()
    meta = read_json(meta_path) if meta_exists else {}

    for age_band in bands:
        ab = age_band.replace("-", "_")

        codes = meta.get("codes", {}).get(age_band, {"drugs": [], "icds": [], "cpts": []})
        n_drugs = len(codes.get("drugs", []) or [])
        n_icds  = len(codes.get("icds", []) or [])
        n_cpts  = len(codes.get("cpts", []) or [])

        for bin_name in _DENSITY_BINS:
            combined_path = CAUSAL_VISUALS / cohort / ab / bin_name / "combined_importance.csv"
            combined_exists = combined_path.exists()

            n_features = 0
            cols = []
            n_drug_f = n_icd_f = n_cpt_f = 0

            if combined_exists:
                df = pd.read_csv(combined_path)
                cols = list(df.columns)
                n_features = len(df)

                if "feature" in df.columns:
                    feats = df["feature"].astype(str)
                    n_drug_f = int(feats.str.startswith(DRUG_PREFIX, na=False).sum())
                    n_icd_f  = int(feats.str.startswith(ICD_PREFIX,  na=False).sum())
                    n_cpt_f  = int(feats.str.startswith(CPT_PREFIX,  na=False).sum())
                else:
                    n_drug_f = n_icd_f = n_cpt_f = 0

            ok, reason = check_cohort_expectations(
                cohort=cohort,
                n_drugs=n_drugs, n_icds=n_icds, n_cpts=n_cpts,
                n_drug_f=n_drug_f, n_icd_f=n_icd_f, n_cpt_f=n_cpt_f
            )

            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"

            print(
                f"{cohort:14s}  age_band={age_band:7s}  bin={bin_name:7s}  "
                f"meta(drug/icd/cpt)={n_drugs:4d}/{n_icds:4d}/{n_cpts:4d}  "
                f"combined_exists={str(combined_exists):5s}  "
                f"combined(drug/icd/cpt)={n_drug_f:4d}/{n_icd_f:4d}/{n_cpt_f:4d}  "
                f"status={status:4s}  {reason}"
            )

            if not meta_exists:
                print(f"  NOTE: metadata file missing: {meta_path}")
            if not combined_exists:
                print(f"  NOTE: combined_importance missing: {combined_path}")

print("=" * 90)
print("Overall:", "OK" if all_ok else "FAIL")

# %% [markdown]
# # Shutdown EC2

# %%
SHUTDOWN_EC2 = True  # Set to False to disable auto-shutdown

print("=" * 80)
print("Final Step: EC2 Instance Shutdown (Optional)")
print("=" * 80)

if SHUTDOWN_EC2:
    print("\nShutting down EC2 instance...")
    print("-" * 80)

    import subprocess
    import shutil
    import os

    try:
        # Retrieve EC2 instance ID from metadata service
        result = subprocess.run(
            ["curl", "-s", "http://169.254.169.254/latest/meta-data/instance-id"],
            capture_output=True,
            text=True,
            timeout=5
        )

        instance_id = result.stdout.strip()

        if instance_id:
            print(f"Instance ID: {instance_id}")

            # Locate AWS CLI
            aws_cmd = shutil.which("aws")
            if not aws_cmd:
                for path in [
                    "/usr/local/bin/aws",
                    "/usr/bin/aws",
                    "/home/ec2-user/.local/bin/aws"
                ]:
                    if os.path.exists(path):
                        aws_cmd = path
                        break

            if not aws_cmd:
                print("\nWarning: AWS CLI not found. Cannot stop instance.")
                print("Install AWS CLI or ensure it is in your PATH.")
                logger.warning("AWS CLI not found; cannot stop EC2 instance")
            else:
                shutdown_cmd = [
                    aws_cmd,
                    "ec2",
                    "stop-instances",
                    "--instance-ids",
                    instance_id
                ]

                print(f"Running: {' '.join(shutdown_cmd)}")
                result = subprocess.run(
                    shutdown_cmd,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    print("\nEC2 stop command sent successfully.")
                    print("Instance will stop shortly.")
                    print("Note: This is a STOP (not terminate).")
                    logger.info(
                        f"EC2 instance {instance_id} stop command issued"
                    )
                else:
                    print(
                        f"\nWarning: EC2 stop command failed "
                        f"(exit code {result.returncode})"
                    )
                    if result.stderr:
                        print(f"Error: {result.stderr.strip()}")
                    logger.warning(
                        f"EC2 stop command failed: {result.stderr}"
                    )
        else:
            print("\nWarning: Instance ID not found. Skipping shutdown.")
            print("Manual shutdown command:")
            print("  aws ec2 stop-instances --instance-ids <instance-id>")
            logger.warning("EC2 instance ID could not be determined")

    except subprocess.TimeoutExpired:
        print("\nWarning: Timeout contacting EC2 metadata service.")
        logger.warning("Timeout retrieving EC2 instance ID")

    except Exception as e:
        print(f"\nWarning: Error during EC2 shutdown: {e}")
        logger.warning(f"EC2 shutdown exception: {e}")

else:
    print("\nEC2 Auto-Shutdown: DISABLED")
    print("Set SHUTDOWN_EC2 = True to enable it.")

print("\n" + "=" * 80)
print("Workflow Complete!")
print("=" * 80)

# %%
