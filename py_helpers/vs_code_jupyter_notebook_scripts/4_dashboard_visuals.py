# -*- coding: utf-8 -*-
# Auto-generated from 4_dashboard_visuals.ipynb (VS Code Python notebook script format)

# %% [markdown]
# # 4. Dashboard visuals (pipeline step 9)
#
# This notebook runs **pipeline step 9** (`9_dashboard_visuals`). It **prebuilds all dashboard visualization artifacts** (**Scenario Analysis** (SHAP+FFA combined), BupaR, DTW, FP-Growth, **Cohort PGx** for the PGx Cohort tab, and **PGx Card** for the patient card tab) on **EC2** and **saves them to the final local destination** that notebook 5 Step 6 syncs to S3 (`10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth,cohort_pgx,scenario}/`). The dashboard loads these prebuilt assets **static-first** from S3 using the **manifest** (`10_risk_dashboard/visualizations/dashboard_visual_objects.json`) as the single source of truth for tab → S3 paths and static files; **JSON-first** for visuals (e.g. DTW: chart_data, sequence_heatmap, trajectory_overview_plot from static, then PNG/HTML fallbacks; API only when static fails). The API returns URLs to prebuilt S3 assets; no computation at request time. Visuals are **SHAP/FFA-driven**: model data and feature lists come from Step 3b / 7 / 8 so process mining and itemset mining use only important features.
#
# **Flow:** Run after [3_model_train_shap_ffa.ipynb](3_model_train_shap_ffa.ipynb). Then run [5_build_and_deploy.ipynb](5_build_and_deploy.ipynb) once to build and deploy.
#
# **Prerequisites (all contained in this notebook):**
# - **Primary**: Step 3b feature importance (`3b_feature_importance_eda/outputs/{cohort}/{age_band}/` or `3a_feature_importance/...`) or combined importance (`10_risk_dashboard/outputs/{cohort}/{age_band}/combined_importance.csv`).
# - **No need to re-run an earlier notebook**: If neither exists, this notebook runs the **combine SHAP+FFA** step (Steps 7 and 8) to generate `combined_importance.csv` in `10_risk_dashboard/outputs/`, then proceeds with allowed_codes, BupaR, DTW, FP-Growth, and Cohort PGx. You only need 7_shap_analysis and 8_ffa_analysis outputs to exist.
#
# ## Steps
#
# 1. **Setup** – Resolve paths (scripts in `9_dashboard_visuals/`; outputs under `10_risk_dashboard/visualizations/{scenario,bupar,dtw,fpgrowth,cohort_pgx}/`).
# 2. **Feature importance heatmaps** – Aggregated and combined heatmaps for the dashboard **Feature Importance** tab; saved to `3a_feature_importance/{cohort}/plots/{cohort}_aggregated_fi_heatmap.png` and `3a_feature_importance/plots/combined_cohorts_feature_importance_heatmap.png`. Notebook 5 (deploy) expects these paths and syncs them to S3.
# 3. **BupaR** – Process mining sequences and plots (SHAP/FFA allowed codes when available); **saved locally** to `10_risk_dashboard/visualizations/bupar/{cohort}/{age_band_fname}/plots/` (notebook 5 Step 6 syncs to S3 `visualizations/bupar/{cohort}/{age_band}/plots/`).
# 4. **DTW** – Trajectory features and plots **based on SHAP/FFA important codes** (same as BupaR/FP-Growth); **drug-only** for both cohorts. **Saved locally** to `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/`: `chart_data.json`, `sequence_heatmap.json`, `plots/trajectory_overview_plot.json`, `plots/dtw_trajectory_analysis_{base}.png`, `plots/dtw_sample_trajectories_{base}.png`, `plots/dtw_trajectory_cluster_1d/3d_{base}.html`. Notebook 5 Step 6 syncs to S3 `visualizations/dtw/{cohort}/{age_band}/` (age_band with **hyphen** on S3). **No empty artifacts:** when a plot doesn’t produce data, the pipeline writes JSON with `message`, `empty: true`, and `metrics` (why). The **manifest** (`dashboard_visual_objects.json`) lists all DTW objects and S3 paths; the dashboard uses **JSON-first** loading (chart_data, sequence_heatmap, trajectory_overview_plot from static first, then PNG/HTML fallbacks, API only when static fails). DTW tab includes **Routine vs No Routine (Outcomes)** (admin ICD codes identify routine appointments) and **High-Risk vs Low-Risk Trajectories**. Full pipeline (2016–2019). **Extreme-density** (optional): same routine vs no routine for high-utilizer subgroups.
#
# 5. **FP-Growth** – Itemsets, rules, **Plotly network HTML**, and PNGs; **saved locally** to `10_risk_dashboard/visualizations/fpgrowth/{cohort}/{age_band_fname}/plots/` and `.../data/` (notebook 5 Step 6 syncs to S3 `visualizations/fpgrowth/{cohort}/{age_band}/`). The dashboard loads the **network plot by cohort** from these URLs.
#
# **Scenario Analysis tab** – Combined SHAP+FFA produces `dashboard_data.json` per cohort/age_band (via `combine_shap_ffa_results.py`); saved under `10_risk_dashboard/visualizations/scenario/{cohort}/{age_band_fname}/`. Run **Upload Scenario dashboard JSON to S3** (cell below) to push to the dashboard bucket as `visualizations/scenario/{cohort}/{age_band}/scenario_data.json`. The Scenario Analysis tab loads from S3; notebook 5 Step 6 also runs this during deploy.
#
# Idempotent. Run from repo root. Prerequisites: `4_model_data`, `7_shap_analysis`, `8_ffa_analysis`; R and bupaR for BupaR. Then run notebook 5 to deploy (syncs all visuals including Causal, Cohort PGx, and others to S3).
#
#
# 6. **Cohort PGx (PGx Cohort tab)** – Fetch PharmGKB VIP reports (Step 1), query NCBI PubMed for supporting literature (Step 1.5 – `lit_review` search_pubmed_all pattern, last 5 years, pharmacogenomics + cohort-context queries), and build interactive network topology per cohort/age_band (Step 2); **saved locally** to `10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/`. Notebook 5 Step 6 syncs to S3 `visualizations/cohort_pgx/networks/{cohort}/{age_band}/`. The **PGx Cohort** dashboard tab calls GET /visualizations/cohort-pgx and displays the network iframe.
#
# 7. **PGx Card (optional)** – Prepares CPIC gene–drug data, PharmGKB VIP JSON, and QR codes for the **PGx Card** dashboard tab. Lambda `POST /pgx/card` uses `pgx-patient-card/data/` (e.g. `cpic_gene-drug_pairs.xlsx`, `pharmgkb_vip_genes.json`, `pgx_database`); notebook 5 packages these into the Lambda image.
#
# 8. **Model performance metrics and cohort metadata** – Prebuilt via `generate_metrics.py` and `generate_metadata.py` (no recomputation). Deploy (5_build_and_deploy) uploads to the dashboard bucket: `metadata/model_performance_metrics.json` (Documentation tab) and `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json` (dropdowns). Frontend loads these same-origin; Lambda GET /metrics and GET /metadata are fallbacks.
#
# 9. **API** – Returns URLs to prebuilt S3 assets only (no server-side computation for visuals). Lambda GET /visualizations/scenario serves the Scenario Analysis tab; GET /visualizations/cohort-pgx serves the PGx Cohort tab; Lambda POST /pgx/card serves the PGx Card tab.

# %%
# Setup: paths (outputs under 10_risk_dashboard/visualizations/)
# Notebook 4 builds visuals to LOCAL only. Notebook 5 Step 6 is the single place that syncs to S3 (idempotent).
import sys
import os
import subprocess
from pathlib import Path

os.environ["SKIP_DASHBOARD_S3_UPLOAD"] = "1"  # Write local only; notebook 5 Step 6 syncs to S3

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from py_helpers.env_utils import get_workflow_python_bin
PYTHON_BIN = get_workflow_python_bin()  # EC2: jupyter-env; local: sys.executable

from py_helpers.env_utils import get_data_root, get_model_data_root

# Data root (with fallback to local nvme or Windows pgx_data)
DATA_ROOT = get_data_root()
MODEL_DATA_ROOT = get_model_data_root()
S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")

# Creation code (step 9); outputs go to 10_risk_dashboard/visualizations
STEP9_ROOT = REPO_ROOT / "9_dashboard_visuals"
VISUAL_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations"
BUPAR_VISUALS_SCRIPT = STEP9_ROOT / "bupar" / "create_bupar_visuals.py"
DTW_TRAJECTORIES_SCRIPT = STEP9_ROOT / "dtw" / "create_dtw_trajectories.py"
DTW_FEATURES_SCRIPT = STEP9_ROOT / "dtw" / "create_dtw_features.py"
DTW_VISUALS_SCRIPT = STEP9_ROOT / "dtw" / "create_dtw_visuals.py"
FPGROWTH_VISUALS_SCRIPT = STEP9_ROOT / "fpgrowth" / "create_fpgrowth_visuals.py"

print(f"Repo root: {REPO_ROOT}")
print(f"Data root (NVMe/local): {DATA_ROOT}")
print(f"Model data root: {MODEL_DATA_ROOT}")
print(f"S3 bucket: {S3_BUCKET}")
print(f"Step 9 (scripts): {STEP9_ROOT}")
print(f"Outputs: {VISUAL_ROOT}")
print(f"Plots (PNG/HTML) appear under: {VISUAL_ROOT}/bupar/<cohort>/<age_band>/plots/ (and dtw/, fpgrowth/, cohort_pgx/networks/, scenario/) — run the cells below to generate them.")

# %% [markdown]
# ## Config: cohorts and age bands
#
# Defaults match **run_dashboard_visuals.py**: all cohorts and all age bands (from REQUIRED_COHORTS), one worker per (cohort, age_band) combo (capped by CPU), no dry run. Leave `COHORTS_TO_RUN` and `AGE_BANDS_TO_RUN` empty for full pipeline; set either to limit scope.

# %% [markdown]
# ### Upload Scenario dashboard JSON to S3 (Scenario Analysis tab)
#
# Run `upload_scenario_outputs_to_s3.py` to upload **scenario visualizations** from `10_risk_dashboard/visualizations/scenario/{cohort}/{age_band_fname}/dashboard_data.json` (produced by `combine_shap_ffa_results.py`) to the dashboard bucket as `visualizations/scenario/{cohort}/{age_band}/scenario_data.json`. The Scenario Analysis tab loads from S3; notebook 5 Step 6 also runs this during deploy. Running here lets the tab have data after building visuals in this notebook.

# %%
# Upload scenario dashboard JSON to S3 (Scenario Analysis tab)
# Script reads 10_risk_dashboard/visualizations/scenario/{cohort}/{age_band_fname}/dashboard_data.json
upload_scenario_script = REPO_ROOT / "10_risk_dashboard" / "data_preparation" / "upload_scenario_outputs_to_s3.py"
if upload_scenario_script.exists():
    r = subprocess.run(
        [str(PYTHON_BIN), str(upload_scenario_script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(r.stdout if r.stdout else "✓ Scenario dashboard JSON uploaded to S3 (visualizations/scenario/{cohort}/{age_band}/scenario_data.json)")
    else:
        print("⚠ Upload scenario to S3:", r.stderr or r.stdout or "non-zero exit")
else:
    print("  upload_scenario_outputs_to_s3.py not found; Step 6 in notebook 5 will upload scenario outputs during deploy.")

# %%
from py_helpers.constants import COHORT_NAMES, AGE_BANDS

try:
    from py_helpers.constants import REQUIRED_COHORTS
except ImportError:
    _all_bands = ['0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114']
    REQUIRED_COHORTS = {"opioid_ed": _all_bands, "non_opioid_ed": _all_bands}

COHORTS_TO_RUN = []
AGE_BANDS_TO_RUN = []

# Default: pipeline (cohort, age_band) for dashboard visuals; exclude 0-12 (Risk Assessment min age 13)
try:
    from py_helpers.constants import DASHBOARD_VISUAL_AGE_BANDS
except ImportError:
    DASHBOARD_VISUAL_AGE_BANDS = [b for b in AGE_BANDS if b != '0-12']
if not COHORTS_TO_RUN and not AGE_BANDS_TO_RUN:
    combinations = [(c, ab) for c, bands in REQUIRED_COHORTS.items() for ab in bands if ab in DASHBOARD_VISUAL_AGE_BANDS]
    print("Using pipeline-supported cohort/age_band (REQUIRED_COHORTS, excluding 0-12 for dashboard visuals)")
else:
    if not COHORTS_TO_RUN:
        COHORTS_TO_RUN = COHORT_NAMES.copy()
    if not AGE_BANDS_TO_RUN:
        AGE_BANDS_TO_RUN = AGE_BANDS.copy()
    combinations = [(c, ab) for c in COHORTS_TO_RUN for ab in AGE_BANDS_TO_RUN]

print(f"Cohorts: {COHORTS_TO_RUN if COHORTS_TO_RUN else list(REQUIRED_COHORTS.keys())}")
print(f"Age bands: {AGE_BANDS_TO_RUN if AGE_BANDS_TO_RUN else 'per-cohort (REQUIRED_COHORTS)'}")
print(f"Total: {len(combinations)} combinations")

# Idempotent: skip when output exists. Set FORCE_RERUN=True to pass --force and re-run all (BupaR, FP-Growth, DTW).
# FP-Growth: --force re-creates itemsets and plots. DTW: --force ignores pipeline checkpoint and plots.
FORCE_RERUN = True
# Parallel workers: one per (cohort, age_band) combo, capped by CPU (matches run_dashboard_visuals.py default).
_ncpu = getattr(os, "cpu_count", lambda: 4)() or 4
PARALLEL_WORKERS = min(_ncpu, len(combinations))
FPGROWTH_WORKERS = min(_ncpu, len(combinations))

# Feature importance: use Step 3b, 3a, or combined_importance. If none exist, generate combined_importance from Steps 7+8 (no need to re-run an earlier notebook).
from py_helpers.event_density_utils import DENSITY_BINS as _SCENARIO_DENSITY_BINS
from py_helpers.shap_ffa_fpgrowth_utils import write_shap_ffa_allowed_codes_for_bupar, _allowed_codes_needs_regen
BUPAR_OUTPUTS = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar"
BUPAR_OUTPUTS.mkdir(parents=True, exist_ok=True)
DASHBOARD_OUTPUTS = REPO_ROOT / "10_risk_dashboard" / "outputs"
SCENARIO_VISUALS = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "causal"
COMBINE_SCRIPT = REPO_ROOT / "10_risk_dashboard" / "data_preparation" / "combine_shap_ffa_results.py"

def _has_fi_source(cohort_name, age_band):
    age_band_fname = age_band.replace("-", "_")
    step3b_dir = REPO_ROOT / "3b_feature_importance_eda" / "outputs" / cohort_name / age_band_fname
    step3b_any = any(step3b_dir.glob("*cohort_feature_importance*.csv")) if step3b_dir.exists() else False
    step3a_path = REPO_ROOT / "3a_feature_importance" / "outputs" / cohort_name / age_band_fname / "cohort_feature_importance.csv"
    scenario_base = SCENARIO_VISUALS / cohort_name / age_band_fname
    combined_scenario = scenario_base / "combined_importance.csv"
    combined_per_bin = any(
        (scenario_base / b / "combined_importance.csv").exists() for b in _SCENARIO_DENSITY_BINS
    )
    combined_out = DASHBOARD_OUTPUTS / cohort_name / age_band_fname / "combined_importance.csv"
    return (
        step3b_any
        or step3a_path.exists()
        or combined_scenario.exists()
        or combined_out.exists()
        or combined_per_bin
    )

# Ensure combined_importance.csv exists when no other FI source: run combine SHAP+FFA from Steps 7/8
import subprocess
if COMBINE_SCRIPT.exists():
    combined_generated = 0
    for cohort_name, age_band in combinations:
        if _has_fi_source(cohort_name, age_band):
            continue
        age_band_fname = age_band.replace("-", "_")
        scenario_base = SCENARIO_VISUALS / cohort_name / age_band_fname
        if (scenario_base / "combined_importance.csv").exists():
            continue
        if any((scenario_base / b / "combined_importance.csv").exists() for b in _SCENARIO_DENSITY_BINS):
            continue
        # Default scenario path is per-bin; combine requires --bin (use medium for this fallback).
        r = subprocess.run(
            [
                str(PYTHON_BIN), str(COMBINE_SCRIPT),
                "--cohort", cohort_name, "--age-band", age_band,
                "--bin", "medium",
                "--output-dir", str(SCENARIO_VISUALS), "--workers", "1",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            combined_generated += 1
            print(f"  [Combine SHAP+FFA] {cohort_name}/{age_band} -> generated combined_importance (bin=medium)")
        else:
            print(f"  [Combine SHAP+FFA] {cohort_name}/{age_band} -> skipped (missing 7/8 outputs or error)")
    if combined_generated:
        print(f"Generated {combined_generated} combined_importance.csv from Steps 7 and 8.\n")

print("\n" + "="*80)
print("Generating allowed_codes JSON files from feature importance data...")
print("="*80)
print(f"Repo root: {REPO_ROOT}")
print(f"Data root: {DATA_ROOT}")
print(f"Looking for:")
print(f"  - Step 3b: 3b_feature_importance_eda/outputs/{{cohort}}/{{age_band}}/*cohort_feature_importance*.csv")
print(f"  - Step 3a: 3a_feature_importance/{{cohort}}/{{age_band}}/cohort_feature_importance.csv")
print(f"  - Combined: 10_risk_dashboard/visualizations/scenario/{{cohort}}/{{age_band}}/ or outputs/ (combined_importance.csv)")
print()

generated = 0
skipped = 0
failed = 0
for cohort_name, age_band in combinations:
    age_band_fname = age_band.replace("-", "_")
    allowed_path = BUPAR_OUTPUTS / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
    if not _allowed_codes_needs_regen(allowed_path):
        skipped += 1
        continue
    if allowed_path.exists():
        allowed_path.unlink(missing_ok=True)  # remove stale file before regen
    
    step3b_dir = REPO_ROOT / "3b_feature_importance_eda" / "outputs" / cohort_name / age_band_fname
    step3b_any = any(step3b_dir.glob("*cohort_feature_importance*.csv")) if step3b_dir.exists() else False
    scenario_base = SCENARIO_VISUALS / cohort_name / age_band_fname
    combined_scenario = scenario_base / "combined_importance.csv"
    combined_out = DASHBOARD_OUTPUTS / cohort_name / age_band_fname / "combined_importance.csv"
    combined_path = None
    if combined_scenario.exists():
        combined_path = combined_scenario
    elif combined_out.exists():
        combined_path = combined_out
    else:
        for b in _SCENARIO_DENSITY_BINS:
            p = scenario_base / b / "combined_importance.csv"
            if p.exists():
                combined_path = p
                break
    if combined_path is None:
        combined_path = combined_out
    step3a_path = REPO_ROOT / "3a_feature_importance" / "outputs" / cohort_name / age_band_fname / "cohort_feature_importance.csv"
    
    print(f"{cohort_name}/{age_band}:")
    print(f"  Step 3b CSV: {'✓' if step3b_any else '✗'} (3b_feature_importance_eda/outputs/...)")
    print(f"  Step 3a CSV: {'✓' if step3a_path.exists() else '✗'} {step3a_path}")
    print(f"  Combined CSV: {'✓' if combined_path.exists() else '✗'} {combined_path}")
    
    # Generate from Step 3b or notebook 3 combined_importance.csv
    try:
        if write_shap_ffa_allowed_codes_for_bupar(
            cohort_name, age_band, allowed_path, top_n=500, 
            project_root=REPO_ROOT, data_root=DATA_ROOT
        ):
            generated += 1
            print(f"  → ✓ Generated {allowed_path.name}")
        else:
            failed += 1
            print(f"  → ✗ No codes extracted (both sources missing or empty)")
    except Exception as e:
        failed += 1
        print(f"  → ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()

print("="*80)
print(f"Summary: Generated={generated}, Skipped={skipped}, Failed={failed}, Total={len(combinations)}")
if failed > 0:
    print("⚠️  Some files failed to generate. Check paths above.")
print("="*80)
print()

# Prerequisite: SHAP/FFA combined allowed codes (required by BupaR, DTW, Cohort PGx; we never use all codes).
# Built from Step 3b, Step 3a, or combined_importance.csv (generated in this notebook from Steps 7+8 when missing).
import json
missing = []
empty = []
for cohort_name, age_band in combinations:
    age_band_fname = age_band.replace("-", "_")
    path = BUPAR_OUTPUTS / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
    if not path.exists():
        missing.append(f"{cohort_name}/{age_band} ({path.name})")
    else:
        try:
            with open(path, encoding="utf-8") as f:
                codes = json.load(f)
            if not codes or (isinstance(codes, list) and len(codes) == 0):
                empty.append(f"{cohort_name}/{age_band} ({path.name})")
        except Exception as e:
            empty.append(f"{cohort_name}/{age_band} ({path.name}): {e}")
if missing or empty:
    msg = "SHAP/FFA combined allowed codes are required for BupaR and DTW (prerequisite).\n"
    if missing:
        msg += f"  Missing: {', '.join(missing)}\n"
    if empty:
        msg += f"  Empty or invalid: {', '.join(empty)}\n"
    msg += "  Sources (checked in order):\n"
    msg += "    1. Step 3b feature importance: 3a_feature_importance/{cohort}/{age_band}/cohort_feature_importance.csv\n"
    msg += "    2. Notebook 3 combined importance: 10_risk_dashboard/outputs/{cohort}/{age_band}/combined_importance.csv\n"
    msg += "  Generate allowed codes by running create_bupar_visuals.py with --write-allowed-codes, or sync from S3 gold/bupar/.\n"
    msg += "  If both sources above are missing, re-run feature importance (notebook 2) or SHAP/FFA combine (notebook 3)."
    raise RuntimeError(msg)
print(f"Prerequisite check passed: all {len(combinations)} SHAP/FFA combined allowed codes files present.")
print("  Sources: Step 3b/3a feature importance or combined_importance.csv (generated from Steps 7+8 in this notebook when missing)")

# %%
# Prerequisite: n_event_bin_thresholds.json (written by Step 6 run_final_model.py in notebook 3).
# DTW, BupaR, FP-Growth, and risk inference load these thresholds to apply the same
# low/medium/high/extreme bin cuts as the trained model. If EC2 local files are missing,
# pull the deployed copies from S3 before failing.
import json

FINAL_MODEL_OUTPUTS = REPO_ROOT / "6_final_model" / "outputs"
MODEL_THRESHOLDS_S3_PREFIX = "gold/dashboard/models"

def _valid_threshold_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
        return all(k in data for k in ("p25", "p50", "p95"))
    except Exception:
        return False

try:
    import boto3
    _threshold_s3 = boto3.client("s3", region_name="us-east-1")
except Exception as exc:
    _threshold_s3 = None
    print(f"[WARN] boto3 unavailable; cannot pull missing n_event_bin_thresholds.json from S3: {exc}")

_thresh_missing = []
_thresh_downloaded = 0
for cohort_name, age_band in combinations:
    age_band_fname = age_band.replace("-", "_")
    thresh_path = FINAL_MODEL_OUTPUTS / cohort_name / age_band_fname / "n_event_bin_thresholds.json"
    if thresh_path.exists() and _valid_threshold_json(thresh_path):
        continue

    # Some Step 6/per-bin runs write the same thresholds only under bin_models/{bin}/.
    # Promote the first valid per-bin copy to the canonical cohort/age root path.
    for bin_name in ("low", "medium", "high", "extreme"):
        bin_thresh_path = FINAL_MODEL_OUTPUTS / cohort_name / age_band_fname / "bin_models" / bin_name / "n_event_bin_thresholds.json"
        if bin_thresh_path.exists() and _valid_threshold_json(bin_thresh_path):
            thresh_path.parent.mkdir(parents=True, exist_ok=True)
            thresh_path.write_text(bin_thresh_path.read_text())
            _thresh_downloaded += 1
            break
    if thresh_path.exists() and _valid_threshold_json(thresh_path):
        continue

    s3_key = f"{MODEL_THRESHOLDS_S3_PREFIX}/{cohort_name}/{age_band_fname}/n_event_bin_thresholds.json"
    if _threshold_s3 is not None:
        try:
            thresh_path.parent.mkdir(parents=True, exist_ok=True)
            obj = _threshold_s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            if all(k in data for k in ("p25", "p50", "p95")):
                thresh_path.write_text(json.dumps(data, indent=2))
                _thresh_downloaded += 1
                continue
            _thresh_missing.append(f"{cohort_name}/{age_band} invalid S3 JSON: s3://{S3_BUCKET}/{s3_key}")
            continue
        except Exception as exc:
            _thresh_missing.append(f"{cohort_name}/{age_band} ({thresh_path}; S3 s3://{S3_BUCKET}/{s3_key}: {exc})")
            continue

    _thresh_missing.append(f"{cohort_name}/{age_band} ({thresh_path}; S3 not checked)")

if _thresh_missing:
    msg = (
        "n_event_bin_thresholds.json not found for some cohorts/age-bands locally or in S3.\n"
        "These files are written by Step 6 (run_final_model.py) and deployed with prepare_models.py.\n"
        "Run notebook 3, or sync/deploy model artifacts to s3://{bucket}/{prefix}/.\n"
        "Missing:\n" + "\n".join(f"  {m}" for m in _thresh_missing)
    ).format(bucket=S3_BUCKET, prefix=MODEL_THRESHOLDS_S3_PREFIX)
    raise RuntimeError(msg)
print(
    f"Prerequisite check passed: n_event_bin_thresholds.json present for all {len(combinations)} "
    f"cohort/age-band combinations ({_thresh_downloaded} downloaded from S3)."
)

# %%
# (Prerequisite checks run in the Config cell above.)

# %% [markdown]
# ## Feature importance heatmaps (dashboard Feature Importance tab)
#
# Build aggregated and combined feature importance heatmaps from Step 3a outputs. **Saved locations** (used by notebook 5 and deploy sync):
#
# - Per cohort: `3a_feature_importance/{cohort}/plots/{cohort}_aggregated_fi_heatmap.png`
# - Combined: `3a_feature_importance/plots/combined_cohorts_feature_importance_heatmap.png`
#
# Prerequisite: Step 3a aggregated CSVs at `3a_feature_importance/{cohort}/{cohort}_{age_band}_aggregated_feature_importance.csv`.

# %% [markdown]
# ### Convert FI CSVs to JSON (cohort / model / age_band filters)
#
# Run after the heatmaps cell. Scans `3a_feature_importance/outputs` for aggregated and per-model CSVs, writes:
# - `feature_importance_index.json` — available cohorts, age_bands, models for dashboard dropdowns
# - `{cohort}/plots/{cohort}_{model}_fi_heatmap.json` — heatmap when 2+ age bands (model = aggregated, catboost, xgboost, xgboost_rf)
# - `{cohort}/plots/{cohort}_{model}_{age_band}_fi.json` — single-age feature list for bar chart
#
# Deploy (notebook 5) uploads these to S3 so the Feature Importance tab can filter by model and age_band.

# %%
# Build FI heatmaps for dashboard (saved where notebook 5 / deploy expect them)
from py_helpers.feature_importance_heatmap import create_aggregated_fi_heatmap, create_combined_cohorts_fi_heatmap

FI_OUTPUTS_BASE = REPO_ROOT / "3a_feature_importance" / "outputs"
# REQUIRED_COHORTS from Config cell: {cohort: [age_bands]}
paths_per_cohort = []
for cohort, age_bands in REQUIRED_COHORTS.items():
    p = create_aggregated_fi_heatmap(cohort, age_bands, FI_OUTPUTS_BASE, top_n=50)
    if p:
        paths_per_cohort.append(p)
        print(f"  ✓ {cohort}: {p}")
    else:
        print(f"  ✗ {cohort}: no aggregated CSVs found under {FI_OUTPUTS_BASE / cohort}")

combined_path = create_combined_cohorts_fi_heatmap(FI_OUTPUTS_BASE, REQUIRED_COHORTS, top_n=80)
if combined_path:
    print(f"  ✓ Combined: {combined_path}")
else:
    print("  ✗ Combined: no data (ensure at least one cohort has aggregated CSVs)")

print()
print("Save locations (must match notebook 5 requirement check and deploy sync):")
print(f"  Per cohort: {FI_OUTPUTS_BASE}/<cohort>/plots/<cohort>_aggregated_fi_heatmap.png")
print(f"  Combined:  {FI_OUTPUTS_BASE}/plots/combined_cohorts_feature_importance_heatmap.png")

# Copy FI heatmaps to 10_risk_dashboard/visualizations/feature_importance/ (same location as other dashboard visuals)
import shutil
FI_VIZ = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "feature_importance"
FI_VIZ.mkdir(parents=True, exist_ok=True)
copied = 0
for cohort in REQUIRED_COHORTS:
    for base in (FI_OUTPUTS_BASE, REPO_ROOT / "3a_feature_importance"):
        src_plots = base / cohort / "plots"
        if not src_plots.exists():
            continue
        dest_dir = FI_VIZ / cohort
        dest_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "json"):
            src = src_plots / f"{cohort}_aggregated_fi_heatmap.{ext}"
            if src.exists():
                shutil.copy2(src, dest_dir / f"aggregated_fi_heatmap.{ext}")
                copied += 1
        for model in ("catboost", "xgboost", "xgboost_rf"):
            src_m = src_plots / f"{cohort}_{model}_fi_heatmap.json"
            if src_m.exists():
                shutil.copy2(src_m, dest_dir / f"{model}_fi_heatmap.json")
                copied += 1
        break
for base in (FI_OUTPUTS_BASE, REPO_ROOT / "3a_feature_importance"):
    if (base / "plots" / "combined_cohorts_feature_importance_heatmap.png").exists():
        shutil.copy2(base / "plots" / "combined_cohorts_feature_importance_heatmap.png", FI_VIZ / "combined_cohorts_feature_importance_heatmap.png")
        copied += 1
    combined_json = base / "combined" / "aggregated_fi_heatmap.json"
    if combined_json.exists():
        (FI_VIZ / "combined").mkdir(parents=True, exist_ok=True)
        shutil.copy2(combined_json, FI_VIZ / "combined" / "aggregated_fi_heatmap.json")
        copied += 1
    break
if copied:
    print(f"\n  Copied {copied} FI file(s) to {FI_VIZ} (notebook 5 / deploy sync from here).")

# %% [markdown]
# ## Run BupaR process mining
#
# Plots are **saved locally** to `10_risk_dashboard/visualizations/bupar/{cohort}/{age_band_fname}/plots/` (the final destination; notebook 5 Step 6 syncs to S3 `visualizations/bupar/{cohort}/{age_band}/plots/`). **BupaR features are not used for feature engineering** (same as DTW and FP-Growth); they are computed for dashboard visualization and analysis.

# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

FAIL_FAST = True
FORCE_RERUN = True
force_flag = ["--force"] if FORCE_RERUN else []

def run_bupar_one(cohort_name, age_band):
    r = subprocess.run(
        [str(PYTHON_BIN), str(BUPAR_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return (cohort_name, age_band, r.returncode, r.stdout, r.stderr)

with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futures = {ex.submit(run_bupar_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, code, stdout, stderr = fut.result()
        print(f"  [BupaR] {cohort_name} / {age_band} -> exit {code}")
        if code != 0:
            ab_f = age_band.replace("-", "_")
            print(f"    create_bupar_visuals failed (exit {code}). Check 9_dashboard_visuals/logs/bupaR/bupar_{cohort_name}_{ab_f}.log if available")
            if stderr:
                print("    stderr:", (stderr[:1500] + "..." if len(stderr) > 1500 else stderr))
            if stdout:
                print("    stdout:", (stdout[:800] + "..." if len(stdout) > 800 else stdout))
            if FAIL_FAST:
                raise RuntimeError(f"BupaR failed: {cohort_name} / {age_band}")
print("BupaR done.")

# %% [markdown]
# ## Run DTW trajectory analysis (visualization and analysis)
#
# **DTW trajectories and alignment are for visualization and analysis** (not feature engineering). **DTW is drug-only for both cohorts** (opioid_ed and non_opioid_ed): only prescription (drug) events in the SHAP/FFA allowed set are included. **Routine vs no routine** uses administrative ICD codes (e.g. well visits, screenings) to identify routine appointments; `admin_icd_event_count` drives those charts.
#
# This step:
# 1. **Extracts trajectories** from model_data filtered by SHAP/FFA important **drug** codes (same allowed-codes flow as BupaR/FP-Growth)
# 2. **Sequence alignment** – DTW distance computation to prototype trajectories to identify common patterns
# 3. **Creates visualizations** (trajectory cluster plots, routine vs no routine charts, sequence heatmaps)
# 4. **Saves locally** to `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/` (EC2 paths use **underscore** in age_band). Notebook 5 Step 6 syncs to S3 `visualizations/dtw/{cohort}/{age_band}/` (S3 uses **hyphen**).
#
# **No empty artifacts.** When a plot doesn’t produce data, the pipeline **always** writes a JSON artifact with `message`, `empty: true`, `cohort`, `age_band`, and `metrics` (e.g. `reason`, `dtw_rows`) so the dashboard can show why. Never a missing file or plain `{}`.
#
# **What's created (per cohort/age_band):**
# - **Feature engineering** (in `.../dtw/feature_engineering/`): `dtw_features_{cohort}_{age_band_fname}.csv`, `common_sequences_*.json`
# - **Dashboard artifacts** (in `.../dtw/{cohort}/{age_band_fname}/`):
#   - `chart_data.json` – routine_comparison, high_risk_trajectories, target_pathway_patterns, times_between_sequences, routine_comparison_counts, event_density_bins (full or empty-state JSON)
#   - `sequence_heatmap.json` – code×position counts by drug/icd/cpt (full or empty-state JSON)
#   - `plots/trajectory_overview_plot.json` – Plotly overview (full or empty-state JSON)
#   - `plots/dtw_trajectory_analysis_{base}.png`, `plots/dtw_sample_trajectories_{base}.png` – PNG fallbacks (when kaleido available)
#   - `plots/dtw_trajectory_cluster_1d_{base}.html`, `plots/dtw_trajectory_cluster_3d_{base}.html` – interactive overview
#
# The **manifest** (`dashboard_visual_objects.json`) lists all DTW objects and S3 paths. The dashboard uses **JSON-first** loading: fetches `chart_data.json`, `sequence_heatmap.json`, and `plots/trajectory_overview_plot.json` from static (manifest URLs) first, then PNG/HTML fallbacks from manifest, API only when static fails.
#
# **Runtime:** ~10–30 minutes per cohort/age_band (includes DTW distance matrix computation).
#
# **Research question:** Routine vs no routine appointments → outcomes; sequence-level patterns (drug pathways preceding adverse events). **Date scope:** full pipeline (2016–2019).

# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    FAIL_FAST
except NameError:
    FAIL_FAST = True  # Stop on first failure; set False to continue (also in config/BupaR cell)
force_flag = ["--force"] if FORCE_RERUN else []

def run_dtw_one(cohort_name, age_band):
    """Run DTW trajectory extraction, alignment, and visualization (three-step process)."""
    # Step 1: Extract trajectories from model_data (filtered by SHAP/FFA)
    r_traj = subprocess.run(
        [str(PYTHON_BIN), str(DTW_TRAJECTORIES_SCRIPT), 
         "--cohort", cohort_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r_traj.returncode != 0:
        return (cohort_name, age_band, "trajectories", r_traj.returncode, r_traj.stdout, r_traj.stderr)
    
    # Step 2: DTW alignment (compute distances to prototypes, identify common sequences)
    r_align = subprocess.run(
        [str(PYTHON_BIN), str(DTW_FEATURES_SCRIPT), 
         "--cohort", cohort_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r_align.returncode != 0:
        return (cohort_name, age_band, "alignment", r_align.returncode, r_align.stdout, r_align.stderr)
    
    # Step 3: Create and publish visualizations
    r_vis = subprocess.run(
        [str(PYTHON_BIN), str(DTW_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r_vis.returncode != 0:
        return (cohort_name, age_band, "visuals", r_vis.returncode, r_vis.stdout, r_vis.stderr)
    
    return (cohort_name, age_band, "success", 0, "", "")

with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futures = {ex.submit(run_dtw_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, step, code, stdout, stderr = fut.result()
        if code == 0:
            print(f"  [DTW] {cohort_name} / {age_band} -> SUCCESS")
        else:
            print(f"  [DTW] {cohort_name} / {age_band} -> FAILED at {step} (exit {code})")
            if stderr:
                print(f"    stderr: {stderr[:1500]}{'...' if len(stderr) > 1500 else ''}")
            if FAIL_FAST:
                raise RuntimeError(f"DTW {step} failed: {cohort_name} / {age_band}")
print("DTW done (trajectories + alignment + visuals).")
print("Per-bin DTW charts auto-built: create_dtw_features splits by event_density_bin; create_dtw_visuals writes density/{bin}/chart_data.json + sequence_heatmap.json per bin.")

# %%
# Bin transition analysis: track patients moving between event-density bins across years (2016-2019).
# Output: density/transitions/bin_transitions.json with transition matrix, Sankey data, escalation rates.
# Clinically: patients escalating from low→extreme density represent disease progression or worsening polypharmacy.
BIN_TRANSITIONS_SCRIPT = STEP9_ROOT / "dtw" / "create_bin_transitions.py"

def run_bin_transitions_one(cohort_name, age_band):
    r = subprocess.run(
        [str(PYTHON_BIN), str(BIN_TRANSITIONS_SCRIPT),
         "--cohort", cohort_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return (cohort_name, age_band, r.returncode, r.stdout, r.stderr)

if BIN_TRANSITIONS_SCRIPT.exists():
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = {ex.submit(run_bin_transitions_one, c, ab): (c, ab) for c, ab in combinations}
        for fut in as_completed(futures):
            cohort_name, age_band, code, stdout, stderr = fut.result()
            if code == 0:
                print(f"  [Bin Transitions] {cohort_name} / {age_band} -> SUCCESS")
            else:
                print(f"  [Bin Transitions] {cohort_name} / {age_band} -> SKIPPED/WARN (exit {code})")
                if stderr:
                    print(f"    {stderr[:400]}{'...' if len(stderr) > 400 else ''}")
    print("Bin transition analysis done.")
    print("  Output: density/transitions/bin_transitions.json per cohort/age_band")
    print("  Contains: transition_matrix, sankey_nodes/links, escalation_rate, de_escalation_rate, per_year_distribution")
else:
    print(f"  create_bin_transitions.py not found at {BIN_TRANSITIONS_SCRIPT}; skipping.")

# %%
# Combined per-bin heatmaps: activity frequency (BupaR), itemset support (FP-Growth), and
# sequence-code frequency (DTW) across all density bins in a single matrix view.
# Format: rows=activities/items/codes, cols=bins (low/medium/high/extreme), values=rate_per_patient or support.
# Same row_labels/column_labels/matrix shape as feature importance heatmaps — renderable with same Plotly component.
COMBINED_HEATMAP_SCRIPT = STEP9_ROOT / "create_combined_bin_heatmap.py"

def run_combined_heatmap_one(cohort_name, age_band):
    r = subprocess.run(
        [str(PYTHON_BIN), str(COMBINED_HEATMAP_SCRIPT),
         "--cohort", cohort_name, "--age-band", age_band,
         "--top-n", "30",
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return (cohort_name, age_band, r.returncode, r.stdout, r.stderr)

if COMBINED_HEATMAP_SCRIPT.exists():
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = {ex.submit(run_combined_heatmap_one, c, ab): (c, ab) for c, ab in combinations}
        for fut in as_completed(futures):
            cohort_name, age_band, code, stdout, stderr = fut.result()
            if code == 0:
                print(f"  [Combined Heatmap] {cohort_name} / {age_band} -> SUCCESS")
            else:
                print(f"  [Combined Heatmap] {cohort_name} / {age_band} -> SKIPPED/WARN (exit {code})")
                if stderr:
                    print(f"    {stderr[:400]}{'...' if len(stderr) > 400 else ''}")
    print("Combined bin heatmaps done.")
    print("  Outputs per cohort/age_band:")
    print("    bupar/density/combined/bupar_activity_heatmap.json+.png")
    print("    fpgrowth/density/combined/fpgrowth_itemset_heatmap.json+.png")
    print("    dtw/density/combined/dtw_sequence_heatmap.json+.png")
    print("    dtw/density/combined/bin_summary.json  (n_patients + escalation rates)")
else:
    print(f"  create_combined_bin_heatmap.py not found at {COMBINED_HEATMAP_SCRIPT}; skipping.")

# %%
# Convert FI CSVs to JSON for dashboard filters (model, cohort, age_band, all age bands)
from py_helpers.feature_importance_heatmap import build_fi_dashboard_jsons

FI_OUTPUTS_BASE = REPO_ROOT / "3a_feature_importance" / "outputs"
result = build_fi_dashboard_jsons(FI_OUTPUTS_BASE, top_n=50, single_band_top_n=100)
print(f"Wrote {len(result['written'])} JSONs (index + heatmaps + single-age).")
for p in result["written"][:15]:
    print(f"  {p}")
if len(result["written"]) > 15:
    print(f"  ... and {len(result['written']) - 15} more")

# %% [markdown]
# ### Appointments vs no appointments and extreme-density cohorts
#
# **Research question (N1):** Is there a difference in outcomes for patients without routine appointments vs those with routine care? The DTW tab answers this via **Routine vs No Routine (Outcomes)** and **High-Risk vs Low-Risk Trajectories** (see above). These are shown over the **full pipeline (2016–2019)**, not a single year.
#
# **Extreme-density cohorts** are high-utilizer patients (top ~5% by medical_code transaction density) split out so they do not dominate main models (see `docs/Step4_ModelData/README_model_data_and_extreme_split.md`). For **each cohort and age band**, running extract + DTW (and optionally BupaR) for the extreme-density subgroup lets you compare **routine vs no routine** (outcomes and trajectories) in the high-utilizer subgroup and how **extreme densities** and **extreme-density trajectories** differ across age bands and cohorts. By default the cell below uses the **same (cohort, age_band) combinations** as the main pipeline. Set `EXTREME_COMBINATIONS = []` to skip. Requires **Step 4** (model data) first.

# %%
# Default: same (cohort, age_band) as main pipeline so we get routine vs no routine and
# extreme-density trajectories for every cohort and age band. Set to [] to skip.
# Full DTW pipeline for extreme: extract -> trajectories -> features -> visuals
# (create_dtw_visuals needs dtw_features_* CSV; trajectories uses base cohort allowed_codes when cohort ends with _extreme_density)
EXTREME_COMBINATIONS = combinations  # from config cell above

EXTREME_EXTRACT_SCRIPT = STEP9_ROOT / "dtw" / "extract_extreme_density_cohort.py"
extreme_force_flag = ["--force"] if FORCE_RERUN else []

def run_extreme_one(cohort_name, age_band):
    r0 = subprocess.run(
        [str(PYTHON_BIN), str(EXTREME_EXTRACT_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r0.returncode != 0:
        return (cohort_name, age_band, r0.returncode, None, None, None, r0.stdout, r0.stderr, None, None, None)
    extreme_name = f"{cohort_name}_extreme_density"
    r_traj = subprocess.run(
        [str(PYTHON_BIN), str(DTW_TRAJECTORIES_SCRIPT), "--cohort-name", extreme_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + extreme_force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r_traj.returncode != 0:
        return (cohort_name, age_band, r0.returncode, r_traj.returncode, None, None, r0.stdout, r0.stderr, r_traj.stdout, r_traj.stderr, None)
    r_feat = subprocess.run(
        [str(PYTHON_BIN), str(DTW_FEATURES_SCRIPT), "--cohort", extreme_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + extreme_force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r_feat.returncode != 0:
        return (cohort_name, age_band, r0.returncode, r_traj.returncode, r_feat.returncode, None, r0.stdout, r0.stderr, r_traj.stderr, r_feat.stderr, None)
    r_vis = subprocess.run(
        [str(PYTHON_BIN), str(DTW_VISUALS_SCRIPT), "--cohort-name", extreme_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + extreme_force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return (cohort_name, age_band, r0.returncode, r_traj.returncode, r_feat.returncode, r_vis.returncode, r0.stdout, r0.stderr, r_traj.stderr, r_feat.stderr, r_vis.stderr)

if not EXTREME_COMBINATIONS:
    print("EXTREME_COMBINATIONS is empty; skipping extreme-density cohort extraction and DTW.")
else:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = {ex.submit(run_extreme_one, c, ab): (c, ab) for c, ab in EXTREME_COMBINATIONS}
        for fut in as_completed(futures):
            result = fut.result()
            cohort_name, age_band = result[0], result[1]
            c0, c_traj, c_feat, c_vis = result[2], result[3], result[4], result[5]
            print(f"  [Extreme] {cohort_name} / {age_band} -> extract={c0}, traj={c_traj}, feat={c_feat}, vis={c_vis}")
            if c0 != 0:
                print(f"    extract_extreme_density_cohort failed (exit {c0})")
                if len(result) > 7 and result[7]:
                    print("    stderr:", (result[7][:1500] + "..." if len(result[7]) > 1500 else result[7]))
                if FAIL_FAST:
                    raise RuntimeError(f"Extract extreme cohort failed: {cohort_name} / {age_band}")
            if c_traj is not None and c_traj != 0:
                print(f"    create_dtw_trajectories failed (exit {c_traj})")
                if FAIL_FAST:
                    raise RuntimeError(f"DTW create_dtw_trajectories failed: {cohort_name}_extreme_density / {age_band}")
            if c_feat is not None and c_feat != 0:
                print(f"    create_dtw_features failed (exit {c_feat})")
                if FAIL_FAST:
                    raise RuntimeError(f"DTW create_dtw_features failed: {cohort_name}_extreme_density / {age_band}")
            if c_vis is not None and c_vis != 0:
                print(f"    create_dtw_visuals failed (exit {c_vis})")
                if len(result) > 10 and result[10]:
                    print("    stderr:", (result[10][:800] + "..." if len(result[10]) > 800 else result[10]))
                if FAIL_FAST:
                    raise RuntimeError(f"DTW create_dtw_visuals failed: {cohort_name}_extreme_density / {age_band}")
    print(f"Done: extreme-density extract + DTW (trajectories + features + visuals) for {len(EXTREME_COMBINATIONS)} combinations (parallel).")

# %% [markdown]
# ## Run FP-Growth (itemsets, Plotly network HTML, S3 upload)
#
# FP-Growth uses **SHAP/FFA-refined** model data: inputs come from `4_model_data` (built from Step 3b `cohort_feature_importance.csv`). **Item types included: drugs (`drug_name`), ICD diagnosis codes (`icd_code`), and CPT procedure codes (`cpt_code`)** (plus combined `medical_code`). For each cohort/age band this step: (1) ensures itemsets exist, (2) creates PNGs and **Plotly interactive network HTML**, (3) **saves locally** to `10_risk_dashboard/visualizations/fpgrowth/{cohort}/{age_band_fname}/plots/` and `.../data/` (notebook 5 Step 6 syncs to S3 `visualizations/fpgrowth/{cohort}/{age_band}/`). The dashboard then shows the **network plot for the user-selected cohort** via the `/visualizations/fpgrowth` API. **FP-Growth itemsets and rules are not used for feature engineering** (same as DTW and BupaR); they are computed for dashboard visualization and analysis.

# %% [markdown]
# Run the cell below in parallel (FPGROWTH_WORKERS at a time; builds itemsets and Plotly HTML, saves to final local destination). **Exit 0** = itemsets/plots produced for that cohort/age_band; **exit 1** = no outputs (e.g. model_data missing or too few transactions). Logs: `logs/9_fpgrowth/` (EC2) and S3 `s3://pgx-repository/9_fpgrowth_log/{cohort}/{age_band}/` (mirrored on log_summary()).

# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    FAIL_FAST
except NameError:
    FAIL_FAST = True
try:
    FPGROWTH_WORKERS
except NameError:
    # Maximize CPU utilization: use all cores up to number of combinations
    import os
    FPGROWTH_WORKERS = min(os.cpu_count() or 4, len(combinations))
force_flag = ["--force"] if FORCE_RERUN else []

def run_fpgrowth_one(cohort_name, age_band):
    r = subprocess.run(
        [str(PYTHON_BIN), str(FPGROWTH_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return (cohort_name, age_band, r.returncode, r.stdout, r.stderr)

with ThreadPoolExecutor(max_workers=FPGROWTH_WORKERS) as ex:
    futures = {ex.submit(run_fpgrowth_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, code, stdout, stderr = fut.result()
        print(f"  [FP-Growth] {cohort_name} / {age_band} -> exit {code}")
        if code != 0:
            ab_f = age_band.replace("-", "_")
            print(f"    No itemsets produced (exit 1). Check logs/9_fpgrowth/create_fpgrowth_visuals_{cohort_name}_{ab_f}_*.log or s3://pgx-repository/9_fpgrowth_log/{cohort_name}/{age_band}/")
            if stderr:
                print("    stderr:", (stderr[:1500] + "..." if len(stderr) > 1500 else stderr))
            if stdout:
                print("    stdout:", (stdout[:800] + "..." if len(stdout) > 800 else stdout))
            if FAIL_FAST:
                raise RuntimeError(f"FP-Growth failed: {cohort_name} / {age_band}")
print("FP-Growth done.")

# %% [markdown]
# ## View visualization outputs
#
# **Outputs are not in the repo** — they are created when you run the FI heatmaps, BupaR, DTW, FP-Growth, Scenario, and Cohort PGx cells above. Paths: **`10_risk_dashboard/visualizations/`** with **`bupar/`**, **`dtw/`**, **`fpgrowth/`**, **`feature_importance/`**, **`scenario/`**, **`cohort_pgx/`** (each has cohort/age_band or cohort-only structure; plots in `plots/` or at root). Run the cell below **after** those steps to list paths and preview sample **JSON** (truncated), **PNG** (image), and **HTML** (link). If you see "No outputs yet", run the pipeline cells above first.

# %%
# Where outputs live and sample preview (run after FI heatmaps, BupaR, DTW, FP-Growth, Scenario, Cohort PGx)
import html
import json
from pathlib import Path
try:
    from IPython.display import display, Image, HTML, IFrame  # type: ignore
except Exception:  # pragma: no cover
    display = lambda *args, **kwargs: None
    Image = HTML = IFrame = lambda *args, **kwargs: None

def _first_plots_dir(base: Path, subdir: str, use_outputs: bool = False):
    """Find first cohort/age_band/plots dir. Try direct (bupar/dtw) then outputs/ (fpgrowth)."""
    roots = [base / subdir / "outputs", base / subdir] if use_outputs else [base / subdir, base / subdir / "outputs"]
    for root in roots:
        if not root.exists():
            continue
        for cohort in sorted(root.iterdir()):
            if not cohort.is_dir() or cohort.name.startswith("."):
                continue
            for age in sorted(cohort.iterdir()):
                if not age.is_dir():
                    continue
                plots = age / "plots"
                if plots.exists():
                    return plots
    return None

def _preview_json(path: Path, max_lines: int = 25):
    """Show truncated JSON (first max_lines of pretty-printed), HTML-escaped for safe display."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = json.dumps(data, indent=2)
        lines = raw.split("\n")[:max_lines]
        text = "\n".join(lines) + ("\n..." if len(raw.split("\n")) > max_lines else "")
        return html.escape(text)
    except Exception:
        return "(could not parse JSON)"

base = REPO_ROOT / "10_risk_dashboard" / "visualizations"
print("Output root:", base)
print()

# --- BupaR, DTW, FP-Growth (plots: PNG, HTML, JSON) ---
for name, subdir in [("BupaR", "bupar"), ("DTW", "dtw"), ("FP-Growth", "fpgrowth")]:
    plots_dir = _first_plots_dir(base, subdir, use_outputs=(subdir == "fpgrowth"))
    print(f"--- {name} ---")
    if not plots_dir:
        print(f"  No outputs yet. Run the \"Run {name}\" cell above, then re-run this cell.")
        print(f"  Paths: {base / subdir}/<cohort>/<age_band>/plots/ or .../outputs/...")
        continue
    print(f"  Sample dir: {plots_dir}")
    pngs = sorted(plots_dir.glob("*.png"))
    htmls = sorted(plots_dir.glob("*.html"))
    jsons = sorted(plots_dir.glob("*.json"))
    print(f"  PNGs: {len(pngs)}, HTMLs: {len(htmls)}, JSONs: {len(jsons)}")
    if pngs:
        display(HTML(f"<b>{name} (sample PNG)</b>"))
        display(Image(filename=str(pngs[0]), width=600))
    if jsons:
        display(HTML(f"<b>{name} (sample JSON preview)</b>"))
        display(HTML(f"<pre>{_preview_json(jsons[0])}</pre>"))
    if htmls:
        display(HTML(f"<b>{name} (sample HTML)</b>"))
        display(HTML(f'<a href="file:///{htmls[0].resolve().as_posix()}" target="_blank">Open in browser</a>'))
    print()

# --- Feature importance (PNG + JSON under feature_importance/ and feature_importance/{cohort}/, combined/) ---
fi_base = base / "feature_importance"
print("--- Feature importance ---")
if fi_base.exists():
    fi_pngs = list(fi_base.glob("*.png")) + list(fi_base.rglob("*.png"))
    fi_jsons = list(fi_base.glob("*.json")) + list(fi_base.rglob("*.json"))
    print(f"  Path: {fi_base}")
    print(f"  PNGs: {len(fi_pngs)}, JSONs: {len(fi_jsons)}")
    if fi_pngs:
        display(HTML("<b>Feature importance (sample PNG)</b>"))
        display(Image(filename=str(fi_pngs[0]), width=600))
    if fi_jsons:
        display(HTML("<b>Feature importance (sample JSON preview)</b>"))
        display(HTML(f"<pre>{_preview_json(fi_jsons[0])}</pre>"))
else:
    print("  No outputs yet. Run the FI heatmaps cell and copy step above.")
print()

# --- Scenario (dashboard_data.json per cohort/age_band) ---
scenario_base = base / "scenario"
print("--- Scenario ---")
scenario_jsons = list(scenario_base.rglob("dashboard_data.json")) if scenario_base.exists() else []
if scenario_jsons:
    print(f"  Path: {scenario_base}/<cohort>/<age_band_fname>/")
    print(f"  JSONs: {len(scenario_jsons)}")
    display(HTML("<b>Scenario (sample JSON preview)</b>"))
    display(HTML(f"<pre>{_preview_json(scenario_jsons[0])}</pre>"))
else:
    print("  No outputs yet. Run the combine SHAP+FFA / Scenario step above.")
print()

# --- Cohort PGx (network HTML + JSON under cohort_pgx/networks/) ---
pgx_base = base / "cohort_pgx" / "networks"
print("--- Cohort PGx ---")
if pgx_base.exists():
    pgx_htmls = list(pgx_base.rglob("*.html"))
    pgx_jsons = list(pgx_base.rglob("*.json"))
    print(f"  Path: {pgx_base}/<cohort>/<age_band_fname>/")
    print(f"  HTMLs: {len(pgx_htmls)}, JSONs: {len(pgx_jsons)}")
    if pgx_htmls:
        display(HTML("<b>Cohort PGx (sample HTML)</b>"))
        display(HTML(f'<a href="file:///{pgx_htmls[0].resolve().as_posix()}" target="_blank">Open in browser</a>'))
    if pgx_jsons:
        display(HTML("<b>Cohort PGx (sample JSON preview)</b>"))
        display(HTML(f"<pre>{_preview_json(pgx_jsons[0])}</pre>"))
else:
    print("  No outputs yet. Run the Cohort PGx step above.")
print()
print("All outputs: 10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth,feature_importance,scenario,cohort_pgx/networks}/...")

# %% [markdown]
# ## PGx Patient Card Setup (Optional)
#
# The dashboard includes a **PGx Patient Card** feature (Tab 2) that generates personalized pharmacogenomic cards from genetic variants. The Lambda function (`POST /pgx/card`) uses **CPIC gene-drug pairs** data to match variants to drugs requiring dosing modifications.
#
# **Optional enhancement**: Add **PharmGKB VIP URLs** and **QR codes** to patient cards for richer gene information. Run the cells below to:
# 1. Fetch PharmGKB VIP gene data (uses current API)
# 2. Generate QR codes pointing to ClinPGx VIP pages
# 3. Build unified PGx database for Lambda integration
#
# **Note**: The Lambda already works with CPIC data alone. This step adds PharmGKB integration for enhanced cards.
#
# **Python migration (2026)**: Old R scripts (`PGx.Rmd`, `Build_PGx_Database.Rmd`) are deprecated due to PharmGKB API changes. See `pgx-patient-card/README_PYTHON.md` for details.

# %%
# PGx Patient Card Setup - Paths
PGX_CARD_DIR = REPO_ROOT / "pgx-patient-card"
PGX_DATA_DIR = PGX_CARD_DIR / "data"
PGX_QR_DIR = PGX_CARD_DIR / "qr_codes"

# Scripts
DOWNLOAD_CPIC_SCRIPT = PGX_CARD_DIR / "download_cpic_excel.py"
FETCH_PHARMGKB_SCRIPT = PGX_CARD_DIR / "fetch_pharmgkb_data.py"
GENERATE_QR_SCRIPT = PGX_CARD_DIR / "generate_pgx_qr_codes.py"
BUILD_DATABASE_SCRIPT = PGX_CARD_DIR / "build_pgx_database.py"

# Data files
CPIC_EXCEL = PGX_DATA_DIR / "cpic_gene-drug_pairs.xlsx"
VIP_JSON = PGX_DATA_DIR / "pharmgkb_vip_genes.json"
QR_MAPPINGS_JSON = PGX_DATA_DIR / "qr_code_mappings.json"
PGX_DATABASE_DIR = PGX_DATA_DIR / "pgx_database"

print(f"PGx card directory: {PGX_CARD_DIR}")
print(f"Data directory: {PGX_DATA_DIR}")
print(f"QR codes directory: {PGX_QR_DIR}")
print()
print("Files:")
print(f"  CPIC Excel: {CPIC_EXCEL} {'✓' if CPIC_EXCEL.exists() else '✗'}")
print(f"  VIP JSON: {VIP_JSON} {'✓' if VIP_JSON.exists() else '✗'}")
print(f"  QR mappings: {QR_MAPPINGS_JSON} {'✓' if QR_MAPPINGS_JSON.exists() else '✗'}")
print(f"  Database: {PGX_DATABASE_DIR} {'✓' if PGX_DATABASE_DIR.exists() else '✗'}")

# %% [markdown]
# ### Step 1: Download CPIC Data
#
# Download the latest CPIC gene-drug pairs Excel file. This is the **primary data source** for the PGx card feature.
#
# **URL**: `https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx`
#
# **Verified**: February 2026 (31KB, last modified Feb 5, 2026)

# %%
# Download CPIC gene-drug pairs Excel file (uses download_cpic_excel.py with SSL fallback)
if CPIC_EXCEL.exists() and CPIC_EXCEL.stat().st_size > 0:
    print(f"✓ CPIC Excel already exists: {CPIC_EXCEL}")
    print(f"  Size: {CPIC_EXCEL.stat().st_size / 1024:.1f} KB")
else:
    result = subprocess.run(
        [str(PYTHON_BIN), str(DOWNLOAD_CPIC_SCRIPT)],
        cwd=str(PGX_CARD_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout)
    else:
        print(result.stdout or "")
        print(result.stderr or "")
        raise RuntimeError(f"CPIC download failed (exit {result.returncode})")

# %% [markdown]
# ### Step 2: Fetch PharmGKB VIP Data
#
# Fetch VIP (Very Important Pharmacogene) data from PharmGKB API. This adds **ClinPGx VIP URLs** for genes.
#
# **API**: `https://api.pharmgkb.org/v1/data/gene?symbol=...` (documented in Postman)
#
# **Output**: `data/pharmgkb_vip_genes.json` with VIP URLs for 20 genes

# %%
# Fetch PharmGKB VIP gene data
if VIP_JSON.exists():
    print(f"✓ VIP data already exists: {VIP_JSON}")
    import json
    with open(VIP_JSON) as f:
        vip_data = json.load(f)
    print(f"  Contains {len(vip_data)} VIP genes")
else:
    print("Fetching PharmGKB VIP gene data...")
    print(f"  Running: {FETCH_PHARMGKB_SCRIPT}")
    result = subprocess.run(
        [str(PYTHON_BIN), str(FETCH_PHARMGKB_SCRIPT)],
        cwd=str(PGX_CARD_DIR),
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ VIP data fetched successfully")
        print(result.stdout)
    else:
        print(f"✗ Failed (exit {result.returncode})")
        print(result.stderr)

# %% [markdown]
# ### Step 3: Generate QR Codes
#
# Generate QR codes for ClinPGx VIP pages. Each gene gets a QR code pointing to `https://www.clinpgx.org/vip/{PA_ID}/overview`.
#
# **Prerequisites**: `pip install qrcode[pil]`
#
# **Output**: `qr_codes/{GENE}.png` (20 QR code images, 200x200 px)

# %%
# Generate QR codes for VIP pages
if QR_MAPPINGS_JSON.exists() and PGX_QR_DIR.exists() and len(list(PGX_QR_DIR.glob("*.png"))) > 0:
    print(f"✓ QR codes already exist: {PGX_QR_DIR}")
    print(f"  Contains {len(list(PGX_QR_DIR.glob('*.png')))} QR code images")
else:
    print("Generating QR codes for VIP pages...")
    print(f"  Running: {GENERATE_QR_SCRIPT}")
    result = subprocess.run(
        [str(PYTHON_BIN), str(GENERATE_QR_SCRIPT)],
        cwd=str(PGX_CARD_DIR),
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ QR codes generated successfully")
        print(result.stdout)
    else:
        print(f"✗ Failed (exit {result.returncode})")
        print(result.stderr)
        if "ModuleNotFoundError" in result.stderr and "qrcode" in result.stderr:
            print("\nInstall missing dependency:")
            print("  pip install qrcode[pil]")

# %% [markdown]
# ### Step 4: Build Unified PGx Database
#
# Merge CPIC, PharmGKB VIP, and QR code data into a unified database for patient card generation.
#
# **Output**:
# - `data/pgx_database/pgx_database.csv` - Merged data (CSV)
# - `data/pgx_database/pgx_database.json` - Merged data (JSON)
# - `data/pgx_database/pgx_database.xlsx` - Merged data (Excel)
# - `data/pgx_database/pgx_database_summary.json` - Summary statistics

# %%
# Build unified PGx database
if PGX_DATABASE_DIR.exists() and (PGX_DATABASE_DIR / "pgx_database.csv").exists():
    print(f"✓ PGx database already exists: {PGX_DATABASE_DIR}")
    import json
    summary_path = PGX_DATABASE_DIR / "pgx_database_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("\nDatabase summary:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
else:
    print("Building unified PGx database...")
    print(f"  Running: {BUILD_DATABASE_SCRIPT}")
    result = subprocess.run(
        [str(PYTHON_BIN), str(BUILD_DATABASE_SCRIPT)],
        cwd=str(PGX_CARD_DIR),
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ PGx database built successfully")
        print(result.stdout)
    else:
        print(f"✗ Failed (exit {result.returncode})")
        print(result.stderr)

# %% [markdown]
# ### Integration with Lambda
#
# To integrate VIP URLs with the Lambda function (`POST /pgx/card`):
#
# 1. **Copy VIP data to Lambda container** (in Dockerfile or build script):
#    ```dockerfile
#    COPY pgx-patient-card/data/pharmgkb_vip_genes.json ${LAMBDA_TASK_ROOT}/data/
#    ```
#
# 2. **Load at Lambda startup** (in `lambda_function.py`):
#    ```python
#    VIP_URL_CACHE = {}
#
#    def load_vip_urls():
#        vip_path = '/var/task/data/pharmgkb_vip_genes.json'
#        if os.path.exists(vip_path):
#            with open(vip_path) as f:
#                vip_data = json.load(f)
#            VIP_URL_CACHE = {item['gene'].upper(): item['vip_url'] for item in vip_data}
#
#    load_vip_urls()  # Call at module level
#    ```
#
# 3. **Add VIP URLs to card response** (in `generate_pgx_card()`):
#    ```python
#    for gene_entry in genes_processed:
#        if gene_entry['gene'] in VIP_URL_CACHE:
#            gene_entry['vip_url'] = VIP_URL_CACHE[gene_entry['gene']]
#    ```
#
# See `pgx-patient-card/README_PYTHON.md` for full Lambda integration details.

# %% [markdown]
# ## Cohort PGx Network Topology
#
# The dashboard includes a **Cohort PGx** tab that combines PharmGKB VIP reports for all important genes in a cohort with network topology analysis. This feature:
#
# 1. **Extracts PGx genes** from SHAP/FFA feature importance (top N genes from Step 3b or notebook 3)
# 2. **Fetches PharmGKB VIP reports** with clinical annotations and drug interactions
# 3. **Builds network topology** using:
#    - **pytextrank**: Key phrase extraction and entity recognition from VIP text
#    - **AWS Comprehend** (optional): Medical entity recognition and key phrase extraction
#    - **Network analysis**: Gene-drug-phenotype relationships visualized as Plotly interactive graph
#
# **Output**: Interactive network visualization showing how cohort-specific genes relate to drugs and phenotypes, plus exportable node/edge data for further analysis.
#
# **Research value**: Identifies gene-drug interaction patterns specific to each cohort/age band, revealing pharmacogenomic mechanisms underlying adverse event risk.

# %%
# Cohort PGx Network Topology - Paths
COHORT_PGX_DIR = STEP9_ROOT / "cohort_pgx"
COHORT_PGX_REPORTS_DIR = VISUAL_ROOT / "cohort_pgx" / "reports"
COHORT_PGX_NETWORKS_DIR = VISUAL_ROOT / "cohort_pgx" / "networks"

# Scripts
FETCH_VIP_REPORTS_SCRIPT     = COHORT_PGX_DIR / "fetch_vip_reports.py"
FETCH_PUBMED_CITATIONS_SCRIPT = COHORT_PGX_DIR / "fetch_pubmed_citations.py"
BUILD_NETWORK_TOPOLOGY_SCRIPT = COHORT_PGX_DIR / "build_network_topology.py"

print(f"Cohort PGx directory: {COHORT_PGX_DIR}")
print(f"Scripts:")
print(f"  Fetch VIP reports:     {FETCH_VIP_REPORTS_SCRIPT} {'✓' if FETCH_VIP_REPORTS_SCRIPT.exists() else '✗'}")
print(f"  Fetch PubMed cites:    {FETCH_PUBMED_CITATIONS_SCRIPT} {'✓' if FETCH_PUBMED_CITATIONS_SCRIPT.exists() else '✗'}")
print(f"  Build network:         {BUILD_NETWORK_TOPOLOGY_SCRIPT} {'✓' if BUILD_NETWORK_TOPOLOGY_SCRIPT.exists() else '✗'}")
print()
print(f"Outputs:")
print(f"  Reports: {COHORT_PGX_REPORTS_DIR}")
print(f"  Networks: {COHORT_PGX_NETWORKS_DIR}")

# Config
COHORT_PGX_TOP_N = 50  # Top N genes from feature importance
COHORT_PGX_COHORTS = combinations  # Use same cohort/age_band combinations as main pipeline
USE_COMPREHEND = True  # Set False to skip AWS Comprehend (pytextrank only)

print()
print(f"Config:")
print(f"  Top N genes per cohort: {COHORT_PGX_TOP_N}")
print(f"  Cohort combinations: {len(COHORT_PGX_COHORTS)}")
print(f"  AWS Comprehend: {'Enabled' if USE_COMPREHEND else 'Disabled (pytextrank only)'}")

# %% [markdown]
# ### Step 1: Fetch PharmGKB VIP Reports
#
# For each cohort/age band, extract top N genes from feature importance and fetch VIP reports from PharmGKB API.
#
# **Prerequisites**:
# - Feature importance data (Step 3b or notebook 3 combined_importance.csv)
# - `pip install beautifulsoup4` for VIP page text extraction
#
# **Output**: `{cohort}_{age_band}_vip_reports.json` with gene metadata, clinical annotations, and VIP page text
#
# **Runtime**: ~1-2 minutes per cohort/age band (API rate limited to 0.5s between requests)

# %%
# Fetch VIP reports for all cohort/age_band combinations
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_fetch_vip_reports(cohort_name, age_band, bin_name=None):
    """Fetch PharmGKB VIP reports for one cohort/age band (optionally per density bin)."""
    args = [
        str(PYTHON_BIN), str(FETCH_VIP_REPORTS_SCRIPT),
        "--cohort", cohort_name,
        "--age-band", age_band,
        "--top-n", str(COHORT_PGX_TOP_N),
        "--project-root", str(REPO_ROOT),
        "--output-dir", str(COHORT_PGX_REPORTS_DIR)
    ]
    if bin_name:
        args += ["--bin", bin_name]
    # Skip VIP page fetching if needed (faster but less text data)
    # args.append("--no-vip-pages")
    result = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return (cohort_name, age_band, bin_name, result.returncode, result.stdout, result.stderr)

# Build task list: full-cohort + per-bin for each combination
_pgx_tasks = [
    (c, ab, None) for c, ab in COHORT_PGX_COHORTS
] + [
    (c, ab, b) for c, ab in COHORT_PGX_COHORTS for b in _SCENARIO_DENSITY_BINS
]
print(f"Fetching VIP reports for {len(COHORT_PGX_COHORTS)} cohort/age_band combinations + {len(_SCENARIO_DENSITY_BINS)} bins each...")
print(f"  Top {COHORT_PGX_TOP_N} genes per cohort; total tasks: {len(_pgx_tasks)}")
print()

# Run in parallel (max 2 at a time to respect API rate limits)
MAX_WORKERS_PGX = 2  # Conservative to avoid API throttling

with ThreadPoolExecutor(max_workers=MAX_WORKERS_PGX) as ex:
    futures = {ex.submit(run_fetch_vip_reports, c, ab, b): (c, ab, b) for c, ab, b in _pgx_tasks}
    for fut in as_completed(futures):
        cohort_name, age_band, bin_name, code, stdout, stderr = fut.result()
        label = f"{cohort_name} / {age_band}" + (f" [{bin_name}]" if bin_name else "")
        if code == 0:
            print(f"  [VIP Reports] {label} -> SUCCESS")
        else:
            print(f"  [VIP Reports] {label} -> FAILED (exit {code})")
            if stderr:
                print(f"    stderr: {stderr[:800]}{'...' if len(stderr) > 800 else ''}")
            if FAIL_FAST:
                raise RuntimeError(f"Fetch VIP reports failed: {label}")

print("\n✓ VIP reports fetched for all cohorts (full-cohort + per-bin)")
print(f"  Output: {COHORT_PGX_REPORTS_DIR}")

# %% [markdown]
# ### Step 1.5: Fetch PubMed Citations (Literature QA)
#
# For each cohort/age band (and per density bin), query NCBI PubMed E-utilities to retrieve
# supporting literature for the PGx genes found in the VIP reports.  Follows the same
# search methodology as `lit_review/lit_review.qmd` (search_pubmed_all pattern):
#   - Date range: last 5 years  (`{year-5}:{year}[PDAT]`)
#   - Query 1: gene + pharmacogenomics MeSH  (general clinical evidence)
#   - Query 2: gene + cohort-context keyword  (opioid / emergency department)
#   - XML efetch to capture PMC IDs + BioC JSON full-text URL
#
# **Prerequisites**: VIP reports from Step 1 must exist.
#
# **Output per cohort/age band**: `pubmed_citations.json` in the networks output directory
# (same location as `network_topology.html`), synced to S3 by `sync_cohort_pgx_to_s3.py`.
#
# **Rate limiting**: 3 req/s without API key; pass `--ncbi-api-key` env var to speed up.
# **Runtime**: ~1-3 min per cohort/age band (NCBI polite limit).

# %%
# Fetch PubMed citations for all cohort/age_band combinations (full-cohort + per-bin)
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure Cohort PGx paths are defined (run "Cohort PGx Network Topology - Paths" cell first)
try:
    FETCH_PUBMED_CITATIONS_SCRIPT
except NameError:
    from pathlib import Path
    _r = globals().get("REPO_ROOT", Path.cwd())
    _step9 = _r / "10_risk_dashboard"
    _visual = _step9 / "visualizations"
    COHORT_PGX_DIR = _step9 / "cohort_pgx"
    COHORT_PGX_REPORTS_DIR = _visual / "cohort_pgx" / "reports"
    COHORT_PGX_NETWORKS_DIR = _visual / "cohort_pgx" / "networks"
    FETCH_PUBMED_CITATIONS_SCRIPT = COHORT_PGX_DIR / "fetch_pubmed_citations.py"

try:
    FAIL_FAST
except NameError:
    FAIL_FAST = True  # default; set in Config/BupaR cell when running full pipeline

# Optional NCBI API key (set env var NCBI_API_KEY to use; raises rate limit to 10 req/s)
import os as _os
_NCBI_API_KEY = _os.environ.get("NCBI_API_KEY") or None

def run_fetch_pubmed_citations(cohort_name, age_band, bin_name=None):
    """Fetch PubMed citations for one cohort/age band (optionally per density bin)."""
    age_band_fname = age_band.replace("-", "_")
    bin_suffix = f"_{bin_name}" if bin_name else ""
    reports_file = COHORT_PGX_REPORTS_DIR / f"{cohort_name}_{age_band_fname}{bin_suffix}_vip_reports.json"
    output_dir = (
        COHORT_PGX_NETWORKS_DIR / cohort_name / age_band_fname / "density" / bin_name
        if bin_name
        else COHORT_PGX_NETWORKS_DIR / cohort_name / age_band_fname
    )

    # When VIP reports missing (no drugs resolved to genes), script writes minimal output; no short-circuit.
    args = [
        str(PYTHON_BIN), str(FETCH_PUBMED_CITATIONS_SCRIPT),
        "--cohort", cohort_name,
        "--age-band", age_band,
        "--reports", str(reports_file),
        "--output-dir", str(output_dir),
        "--project-root", str(REPO_ROOT),
    ]
    if bin_name:
        args += ["--bin", bin_name]
    if _NCBI_API_KEY:
        args += ["--ncbi-api-key", _NCBI_API_KEY]

    result = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return (cohort_name, age_band, bin_name, result.returncode, result.stdout, result.stderr)

# Task list mirrors VIP reports: full-cohort + per-bin for each combination
_pubmed_tasks = [
    (c, ab, None) for c, ab in COHORT_PGX_COHORTS
] + [
    (c, ab, b) for c, ab in COHORT_PGX_COHORTS for b in _SCENARIO_DENSITY_BINS
]
print(f"Fetching PubMed citations for {len(COHORT_PGX_COHORTS)} cohort/age_band combinations + {len(_SCENARIO_DENSITY_BINS)} bins each...")
print(f"  NCBI API key: {'set (10 req/s)' if _NCBI_API_KEY else 'not set (3 req/s polite limit)'}; total tasks: {len(_pubmed_tasks)}")
print()

# Max 2 concurrent tasks to respect NCBI rate limits (same as VIP reports step)
MAX_WORKERS_PUBMED = 2

with ThreadPoolExecutor(max_workers=MAX_WORKERS_PUBMED) as ex:
    futures = {ex.submit(run_fetch_pubmed_citations, c, ab, b): (c, ab, b) for c, ab, b in _pubmed_tasks}
    for fut in as_completed(futures):
        cohort_name, age_band, bin_name, code, stdout, stderr = fut.result()
        label = f"{cohort_name} / {age_band}" + (f" [{bin_name}]" if bin_name else "")
        if code == 0:
            print(f"  [PubMed Citations] {label} -> SUCCESS")
        else:
            print(f"  [PubMed Citations] {label} -> FAILED (exit {code})")
            if stderr:
                print(f"    stderr: {stderr[:800]}{'...' if len(stderr) > 800 else ''}")
            if FAIL_FAST:
                raise RuntimeError(f"Fetch PubMed citations failed: {label}")

print("\n✓ PubMed citations fetched for all cohorts (full-cohort + per-bin)")
print(f"  Output: {COHORT_PGX_NETWORKS_DIR} (pubmed_citations.json alongside network_topology.html)")

# %% [markdown]
# ### Step 2: Build Network Topology
#
# Build interactive network topology from VIP reports using pytextrank and AWS Comprehend.
#
# **Prerequisites**:
# - `pip install spacy pytextrank networkx plotly`
# - `python -m spacy download en_core_web_sm` (spaCy model)
# - `pip install boto3` (optional, for AWS Comprehend)
#
# **Output per cohort/age band**:
# - `network_topology.html` - Interactive Plotly network visualization
# - `network_nodes.csv` - Node data (genes, drugs, phenotypes)
# - `network_edges.csv` - Edge data (relationships and weights)
# - `key_phrases.json` - Extracted key phrases per gene
# - `network_stats.json` - Network statistics (density, degree, etc.)
#
# **Runtime**: ~2-5 minutes per cohort/age band (depends on text volume and Comprehend usage)

# %%
# Build network topology for all cohort/age_band combinations
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_build_network(cohort_name, age_band, bin_name=None):
    """Build network topology for one cohort/age band (optionally per density bin)."""
    age_band_fname = age_band.replace("-", "_")
    bin_suffix = f"_{bin_name}" if bin_name else ""
    reports_file = COHORT_PGX_REPORTS_DIR / f"{cohort_name}_{age_band_fname}{bin_suffix}_vip_reports.json"
    output_dir = (
        COHORT_PGX_NETWORKS_DIR / cohort_name / age_band_fname / "density" / bin_name
        if bin_name
        else COHORT_PGX_NETWORKS_DIR / cohort_name / age_band_fname
    )

    if not reports_file.exists():
        return (cohort_name, age_band, bin_name, -1, f"Reports file not found: {reports_file}", "")

    args = [
        str(PYTHON_BIN), str(BUILD_NETWORK_TOPOLOGY_SCRIPT),
        "--reports", str(reports_file),
        "--output-dir", str(output_dir),
        "--cohort", cohort_name,
        "--age-band", age_band,
    ]
    if bin_name:
        args += ["--bin", bin_name]

    if not USE_COMPREHEND:
        args.append("--no-comprehend")

    result = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return (cohort_name, age_band, bin_name, result.returncode, result.stdout, result.stderr)

# Build task list: full-cohort + per-bin for each combination
_network_tasks = [
    (c, ab, None) for c, ab in COHORT_PGX_COHORTS
] + [
    (c, ab, b) for c, ab in COHORT_PGX_COHORTS for b in _SCENARIO_DENSITY_BINS
]
print(f"Building network topology for {len(COHORT_PGX_COHORTS)} cohort/age_band combinations + {len(_SCENARIO_DENSITY_BINS)} bins each...")
print(f"  AWS Comprehend: {'Enabled' if USE_COMPREHEND else 'Disabled'}; total tasks: {len(_network_tasks)}")
print()

# Run in parallel (max 4 at a time)
MAX_WORKERS_NETWORK = 4

with ThreadPoolExecutor(max_workers=MAX_WORKERS_NETWORK) as ex:
    futures = {ex.submit(run_build_network, c, ab, b): (c, ab, b) for c, ab, b in _network_tasks}
    for fut in as_completed(futures):
        cohort_name, age_band, bin_name, code, stdout, stderr = fut.result()
        label = f"{cohort_name} / {age_band}" + (f" [{bin_name}]" if bin_name else "")
        if code == 0:
            print(f"  [Network Topology] {label} -> SUCCESS")
        else:
            print(f"  [Network Topology] {label} -> FAILED (exit {code})")
            if stderr:
                print(f"    stderr: {stderr[:800]}{'...' if len(stderr) > 800 else ''}")
            if stdout and "not found" in stdout.lower():
                print(f"    {stdout[:400]}{'...' if len(stdout) > 400 else ''}")
            if FAIL_FAST:
                raise RuntimeError(f"Build network topology failed: {label}")

print("\n✓ Network topology built for all cohorts (full-cohort + per-bin)")
print(f"  Output: {COHORT_PGX_NETWORKS_DIR}")

# %% [markdown]
# ### View Cohort PGx Network Outputs
#
# View network topology visualizations and statistics for all cohorts.

# %%
# View Cohort PGx network outputs
from pathlib import Path
try:
    from IPython.display import display, HTML, IFrame  # type: ignore
except Exception:  # pragma: no cover
    display = lambda *args, **kwargs: None
    HTML = IFrame = lambda *args, **kwargs: None
import json

print(f"Cohort PGx Network Outputs: {COHORT_PGX_NETWORKS_DIR}")
print()

if not COHORT_PGX_NETWORKS_DIR.exists():
    print("✗ No outputs yet. Run the cells above to fetch VIP reports and build networks.")
else:
    # Find all network topology HTML files
    html_files = sorted(COHORT_PGX_NETWORKS_DIR.glob("*/*/network_topology.html"))
    
    if not html_files:
        print("✗ No network topology files found. Check if build step completed successfully.")
    else:
        print(f"Found {len(html_files)} network topology visualizations\n")
        
        # Show first one as example
        first_html = html_files[0]
        cohort_name = first_html.parent.parent.name
        age_band = first_html.parent.name.replace("_", "-")
        
        print(f"Example: {cohort_name} / {age_band}")
        print(f"  HTML: {first_html}")
        
        # Show statistics
        stats_file = first_html.parent / "network_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            print(f"\nNetwork Statistics:")
            for key, value in stats.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nInteractive visualization:")
        display(HTML(f'<a href="file:///{first_html.resolve().as_posix()}" target="_blank">Open {cohort_name}/{age_band} network in browser</a>'))
        
        # List all outputs
        print(f"\nAll network outputs:")
        for html_file in html_files:
            cohort = html_file.parent.parent.name
            age = html_file.parent.name.replace("_", "-")
            print(f"  - {cohort} / {age}: {html_file.parent}")

print(f"\nDashboard integration: Build step uploads to S3 automatically (same as BupaR/DTW/FP-Growth).")
print(f"  S3 path: {{S3_DASHBOARD_PREFIX}}/cohort_pgx/networks/{{cohort}}/{{age_band}}/")

# %% [markdown]
# ### Dashboard Integration
#
# To integrate Cohort PGx network topology into the dashboard:
#
# 1. **Upload to S3** (in notebook 5 or deploy script):
#    ```bash
#    aws s3 sync 10_risk_dashboard/visualizations/cohort_pgx/ \
#      s3://{{DASHBOARD_BUCKET}}/{{S3_PREFIX}}/cohort_pgx/ \
#      --exclude "*.csv" --exclude "*.json" --include "*.html"
#    ```
#
# 2. **Add API endpoint** (in Lambda `lambda_function.py`):
#    ```python
#    @app.get("/visualizations/cohort-pgx")
#    def get_cohort_pgx_viz(cohort: str, age_band: str):
#        age_band_fname = age_band.replace("-", "_")
#        base_url = f"https://{DASHBOARD_BUCKET}/{S3_PREFIX}/cohort_pgx/{cohort}/{age_band_fname}"
#        return {
#            "network_topology": f"{base_url}/network_topology.html",
#            "network_nodes": f"{base_url}/network_nodes.csv",
#            "network_edges": f"{base_url}/network_edges.csv",
#            "network_stats": f"{base_url}/network_stats.json"
#        }
#    ```
#
# 3. **Add dashboard tab** (in frontend `index.html`):
#    - Create new tab "Cohort PGx" after "PGx Card"
#    - Load network topology HTML via iframe from API endpoint
#    - Show network statistics in sidebar
#
# See `10_risk_dashboard/docs/` for full integration guide.

# %% [markdown]
# ## API (reference)
#
# Lambda receives **user input** (cohort, age_band, model/feature selections) and **filters** only—it does not process or generate visualization data. All BupaR, DTW, and FP-Growth visuals are **prebuilt on EC2** and **saved to S3**; the API returns **URLs** to those prebuilt assets (filtered by cohort/age_band). Endpoints: `GET /visualizations/scenario`, `/visualizations/bupar`, `/visualizations/dtw`, `/visualizations/fpgrowth`. See `10_risk_dashboard/backend/README.md`.

# %%
print("Dashboard endpoints: 10_risk_dashboard/backend/README.md")
print("API Gateway deploy: utility_scripts/create_api_gateway_pgx_risk_calculator.sh")

# %% [markdown]
# ## Next: Build and deploy
#
# Build and deploy run **only** in [5_build_and_deploy.ipynb](5_build_and_deploy.ipynb). Run that notebook after this one.
#
# **Run the cell below** to print all dashboard visual objects (file paths, S3 path, visual name, dashboard tab) and write `10_risk_dashboard/visualizations/dashboard_visual_objects.json` for notebook 5 to consume.

# %%
# Print all dashboard visual objects (file paths + visual name / dashboard tab) for notebook 5 to consume
import json
import os
from pathlib import Path

# Paths (same as used in this notebook; define if not set by earlier cells)
_root = REPO_ROOT
_vis = _root / "10_risk_dashboard" / "visualizations"
_out = _root / "10_risk_dashboard" / "outputs"
_fi_base = _root / "3a_feature_importance" / "outputs"
_s3_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator").strip("/") if 'os' in dir() else "vcu/pgx-risk-calculator"

dashboard_visual_objects = [
    # Feature Importance tab
    {
        "visual_name": "Feature importance heatmap (per cohort)",
        "dashboard_tab": "Feature Importance",
        "path": "10_risk_dashboard/visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png",
        "path_type": "file_pattern",
        "s3_path": _s3_prefix + "/visualizations/feature_importance/{cohort}/aggregated_fi_heatmap.png",
        "notes": "Per cohort; notebook 4 copies from 3a; notebook 5 uploads from visualizations/feature_importance.",
        "static_files": ["aggregated_fi_heatmap.json", "aggregated_fi_heatmap.png"],
        "cohort_scope": "per_cohort",
    },
    {
        "visual_name": "Feature importance heatmap (combined)",
        "dashboard_tab": "Feature Importance",
        "path": "10_risk_dashboard/visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png",
        "path_type": "file",
        "s3_path": _s3_prefix + "/visualizations/feature_importance/combined_cohorts_feature_importance_heatmap.png",
        "notes": "Combined cohorts; notebook 4 copies from 3a.",
        "static_files": ["combined/aggregated_fi_heatmap.json", "combined_cohorts_feature_importance_heatmap.png"],
        "cohort_scope": "combined",
    },
    # BupaR tab
    {
        "visual_name": "BupaR process matrix and activity frequency",
        "dashboard_tab": "BupaR Process Mining",
        "path": "10_risk_dashboard/visualizations/bupar",
        "path_type": "directory",
        "s3_path": _s3_prefix + "/visualizations/bupar/{cohort}/{age_band}/plots/",
        "notes": "{base}={cohort}_{age_band_fname}. JSON + Plotly first for Trace Explorer, Trace Explorer Pre-Target, Process Matrix, Sequences to Target, activity frequency; PNG fallback. Uploaded by 4_dashboard_visuals.",
        "static_files": [
            "{base}_activity_frequency.json",
            "{base}_pre_target_activity_frequency.json",
            "{base}_post_target_activity_frequency.json",
            "{base}_trace_explorer_plot.json",
            "{base}_process_matrix_drug_drug.json",
            "{base}_activity_sequence_top.json",
            "{base}_overall_activity_frequency.png",
            "{base}_activity_sequence_top.png",
            "{base}_process_matrix_drug_drug.png",
            "{base}_pre_f1120_activity_frequency.png",
            "{base}_pre_hcg_activity_frequency.png",
            "{base}_trace_explorer_pre_f1120.png",
            "{base}_trace_explorer_pre_hcg.png",
            "{base}_trace_explorer_post_f1120.png",
            "{base}_trace_explorer_post_hcg.png",
        ],
    },
    # DTW tab
    {
        "visual_name": "DTW chart_data and sequence_heatmap and plots",
        "dashboard_tab": "DTW Trajectories",
        "path": "10_risk_dashboard/visualizations/dtw",
        "path_type": "directory",
        "s3_path": _s3_prefix + "/visualizations/dtw/{cohort}/{age_band}/",
        "notes": "{base}={cohort}_{age_band_fname}. JSON-first: chart_data.json (0), sequence_heatmap.json (1), plots/trajectory_overview_plot.json (2); PNG fallbacks (3)(4), interactive HTML (5)(6). Uploaded by 4_dashboard_visuals.",
        "static_files": [
            "chart_data.json",
            "sequence_heatmap.json",
            "plots/trajectory_overview_plot.json",
            "plots/dtw_trajectory_analysis_{base}.png",
            "plots/dtw_sample_trajectories_{base}.png",
            "plots/dtw_trajectory_cluster_1d_{base}.html",
            "plots/dtw_trajectory_cluster_3d_{base}.html",
        ],
    },
    # FP-Growth tab — drug_name_itemsets.json lives under data/ (lambda fetches from data/)
    {
        "visual_name": "FP-Growth itemsets and drug network",
        "dashboard_tab": "FP-Growth Patterns",
        "path": "10_risk_dashboard/visualizations/fpgrowth",
        "path_type": "directory",
        "s3_path": _s3_prefix + "/visualizations/fpgrowth/{cohort}/{age_band}/",
        "notes": "{base}={cohort}_{age_band_fname}. JSON + Plotly first for Top Itemsets and Itemset Support Distribution (drug_name_itemsets.json); PNG fallback. plots: network HTML, itemsets PNG. plots/empty_state.json when pipeline finds no rules. Step 6 sync.",
        "static_files": [
            "data/drug_name_itemsets.json",
            "plots/{base}_combined_rules_network.html",
            "plots/{base}_drug_name_combined_top_itemsets.png",
            "plots/empty_state.json",
        ],
    },
    # PGx Cohort tab
    {
        "visual_name": "Cohort PGx network topology",
        "dashboard_tab": "PGx Cohort",
        "path": "10_risk_dashboard/visualizations/cohort_pgx/networks",
        "path_type": "directory",
        "s3_path": _s3_prefix + "/visualizations/cohort_pgx/networks/{cohort}/{age_band}/",
        "notes": "EC2 age_band_fname; Step 6 sync_visuals_to_s3 → S3 hyphen.",
        "static_files": ["network_topology.html"],
    },
    # Scenario Analysis tab — rename-on-upload (dashboard_data.json → scenario_data.json) handled by upload_scenario_outputs_to_s3.py
    {
        "visual_name": "Scenario dashboard JSON",
        "dashboard_tab": "Scenario Analysis",
        "path": "10_risk_dashboard/visualizations/scenario/{cohort}/{age_band_fname}/dashboard_data.json",
        "path_type": "file_pattern",
        "s3_path": _s3_prefix + "/visualizations/scenario/{cohort}/{age_band}/scenario_data.json",
        "notes": "upload_scenario_outputs_to_s3 → visualizations/scenario/ (S3 hyphen). One file per cohort/age_band.",
        "static_files": ["scenario_data.json"],
    },
]

# Write JSON for notebook 5
_vis.mkdir(parents=True, exist_ok=True)
manifest_path = _vis / "dashboard_visual_objects.json"
_manifest_doc = {
    "repo_root": "",
    "s3_prefix": _s3_prefix,
    "manifest_version": "1.0",
    "description": "Single source of truth for dashboard tab → S3 paths and static files. Frontend loads this first and builds static URLs from s3_path + static_files. API fallback when objects are missing.",
    "metadata_files": [
        "metadata/model_performance_metrics.json",
        "metadata/opioid_ed.json",
        "metadata/non_opioid_ed.json",
    ],
    "metadata_notes": "Required for Documentation tab (metrics) and cohort/drug/ICD/CPT dropdowns. Step 6 uploads these under prefix.",
    "visual_objects": dashboard_visual_objects,
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(_manifest_doc, f, indent=2)
print(f"Wrote manifest for notebook 5: {manifest_path}")
print()

# Print table: visual name | dashboard tab | path
print("Dashboard visual objects (for notebook 5):")
print("=" * 100)
for i, obj in enumerate(dashboard_visual_objects, 1):
    print(f"{i}. {obj['visual_name']}")
    print(f"   Tab:     {obj['dashboard_tab']}")
    print(f"   Path:    {obj['path']}")
    if obj.get('s3_path'):
        print(f"   S3 path: {obj['s3_path']}")
    print(f"   Type:    {obj['path_type']}")
    if obj.get("notes"):
        print(f"   Notes:   {obj['notes']}")
    print()
print("=" * 100)
print(f"Manifest: {manifest_path}")
print("Notebook 5 can load: json.load(open('10_risk_dashboard/visualizations/dashboard_visual_objects.json'))['visual_objects']")

# %%
# Generate and save time-between-events histogram JSON for each cohort/age band
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Set paths and cohorts/age bands as in notebook setup
FEATURE_ENGINEERING_DIR = Path('10_risk_dashboard/visualizations/dtw/feature_engineering')
HISTOGRAM_OUTPUT_DIR = Path('10_risk_dashboard/visualizations/dtw/outputs')

def save_time_between_events_histogram(cohort, age_band, bins=30):
    age_band_fname = age_band.replace('-', '_')
    parquet_path = FEATURE_ENGINEERING_DIR / f'dtw_features_{cohort}_{age_band_fname}.parquet'
    output_dir = HISTOGRAM_OUTPUT_DIR / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'time_between_events_histogram.json'

    if not parquet_path.exists():
        print(f'File not found: {parquet_path}')
        return

    df = pd.read_parquet(parquet_path)
    values = df['mean_days_between_events'].dropna().values
    if len(values) == 0:
        print(f'No valid mean_days_between_events for {cohort} {age_band}')
        return

    counts, bin_edges = np.histogram(values, bins=bins)
    histogram_json = {
        "type": "histogram",
        "x": values.tolist(),
        "nbinsx": bins,
        "name": "Time Between Events",
        "x_label": "Mean Days Between Events",
        "y_label": "Patient Count",
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist()
    }
    with open(json_path, 'w') as f:
        json.dump(histogram_json, f, indent=2)
    print(f'Saved histogram JSON: {json_path}')

# Example usage for one cohort/age band
save_time_between_events_histogram('opioid_ed', '65-74')
