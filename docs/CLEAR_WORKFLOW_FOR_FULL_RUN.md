# Clearing All Checkpoints, S3 Artifacts, and EC2 Artifacts for Full Workflow Run

Use this when you want to run the workflow **all the way through** from a clean state: no checkpoints, no S3 outputs, and no EC2/local outputs so every step runs from scratch.

## Quick: Run the Cleanup Script

```bash
cd ~/pgx-analysis   # or your project root
chmod +x 2_create_cohort/cleanup_cohort_data.sh
./2_create_cohort/cleanup_cohort_data.sh
# Confirm when prompted, or use --yes to skip confirmation
```

**Options:**
- `--skip-checkpoints` — Keep S3 checkpoints (steps will still skip if outputs exist in S3).
- `--skip-s3` — Only delete local/EC2 files.
- `--skip-local` — Only delete S3 (and keep EC2/local).
- `--yes` — Skip confirmation prompt.

**Preserved (never deleted):** The script only deletes under **gold/cohorts** and other step outputs. It **never** deletes **gold/medical** or **gold/pharmacy** (`/mnt/nvme/gold/medical`, `/mnt/nvme/gold/pharmacy`). Project-local cohort copy uses **data/gold/cohorts** (not data/gold_cohorts). Baseline feature importances on S3: if you want to keep them, use `--skip-s3` for `gold/feature_importance/` or manually exclude `_baseline/`.

---

## What Gets Cleared

### 1. Checkpoints (S3: pgx-repository)

| Location | Purpose |
|----------|---------|
| `s3://pgx-repository/pipeline_checkpoints/` | Step checkpoints used by `py_helpers.checkpoint_utils` (1b, 4_model_data, 6, etc.). Steps skip if checkpoint exists. |
| `s3://pgx-repository/pgx-pipeline-status/` | Legacy/alternate pipeline status (create_cohort, feature_importance_eda, model_data, final_model). |

Clearing these forces steps to re-run (unless they also check for output files in S3).

### 2. S3 Artifacts (pgxdatalake)

| Prefix | Step | Contents |
|--------|------|----------|
| `gold/cohorts/` | 2 | Cohort parquet (cohort_name=non_opioid_ed, cohort_name=opioid_ed). |
| `gold/cohorts_model_data/` | 4 | Model data (model_events.parquet) — current path. |
| `gold/model_data/` | 4 | Model data — alternate path. |
| `gold/event_filter/` | 1b | model_events_no_protocols.parquet, protocol summaries. |
| `gold/feature_importance/` | 3a | Aggregated FI CSVs (and `_baseline/` subfolder). |
| `gold/bupar/` | 3b | BupaR / feature importance EDA outputs. |
| `gold/pgx_features/` | 5 | PGx feature engineering outputs. |
| `gold/final_model/` | 6 | Trained model binaries and metadata. |
| `gold/shap_analysis/` | 7 | SHAP outputs. |
| `gold/ffa_analysis/` | 8 | FFA (AXP) outputs. |
| `gold/combined_analysis/` | 9 | Combined risk dashboard inputs. |
| `gold/models/` | 6 (legacy) | Legacy trained models path. |
| `gold/4a_model_data/` | 4 (legacy) | Legacy model data path. |

### 3. EC2 / Local Artifacts

**Data root** (default on Linux: `PGX_DATA_ROOT` or `/mnt/nvme`):

| Path | Step | Contents |
|------|------|----------|
| `$PGX_DATA_ROOT/gold/cohorts/` | 2 | Synced cohort parquet. |
| `$PGX_DATA_ROOT/4_model_data/` or `4a_model_data/` | 4 | Model data by cohort/age_band. |

**Project root** (e.g. `~/pgx-analysis`):

| Path | Step | Contents |
|------|------|----------|
| `data/gold/cohorts/` | 2 | Project-local copy of cohort parquet (same layout as S3 gold/cohorts). |
| `2_create_cohort/` (cohort_metrics, etc.) | 2 | Local cohort metrics. |
| `3b_feature_importance_eda/outputs/` | 3b | Feature importance EDA. |
| `3_feature_importance/outputs/` | 3a | Feature importance (legacy naming). |
| `3a_feature_importance/outputs/` | 3a | Feature importance MC CV + aggregated FI. |
| `1b_apcd_event_filter/outputs/` | 1b | Event filter outputs, for_review. |
| `4_model_data/` (local outputs under project) | 4 | model_events*.parquet if written to project. |
| `5_pgx_analysis/` outputs | 5 | PGx feature files. |
| `6_final_model/models/` | 6 | Trained model files. |
| `7_shap_analysis/` outputs | 7 | SHAP outputs. |
| `8_ffa_analysis/` outputs | 8 | FFA outputs. |
| `9_risk_dashboard/` outputs | 9 | Dashboard outputs. |

**Optional env (EC2):** If you set `PGX_FEATURE_IMPORTANCE_OUTPUTS` (e.g. `/mnt/nvme/feature_importance/outputs`), that directory is used for 3a outputs; the cleanup script does not delete it unless you add it. You can clear it manually: `rm -rf $PGX_FEATURE_IMPORTANCE_OUTPUTS/*`.

---

## After Clearing

1. Run the workflow in order (see `WORKFLOW_EXECUTION_TODO.md`).
2. Baseline feature importance is expected on S3 (`gold/feature_importance/.../_baseline/`). If you did **not** clear `gold/feature_importance/` (or kept _baseline), Step 1b will use it. If you cleared everything, run Step 3a with `--baseline` first for each cohort/age_band, then run 1b, then 3a without `--baseline`.
