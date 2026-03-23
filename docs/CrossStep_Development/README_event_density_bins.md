# Event Density Bins (n_event_bin) Architecture

**Shared utility:** `py_helpers/event_density_utils.py`

Event density binning classifies each patient into one of four groups based on total event count in their `model_events.parquet` record. The same thresholds and bin labels flow consistently through model training, SHAP/FFA, dashboard visualizations, and live Lambda inference.

---

## Bin definition

| Bin | Rule | Default cutoff |
|-----|------|---------------|
| `low` | n_events ≤ P25 | ≤ 5 |
| `medium` | P25 < n_events ≤ P50 | 6–15 |
| `high` | P50 < n_events ≤ P95 | 16–50 |
| `extreme` | n_events > P95 | > 50 |

`n_events` = `COUNT(*)` per `mi_person_key` in `model_events.parquet`. Cut-points are population percentiles (P25, P50, P95) computed from the training cohort, unique per `(cohort, age_band)`. Default fallbacks (p25=5, p50=15, p95=50) apply only when training data is unavailable.

---

## Threshold lifecycle

```
Step 4  →  model_events.parquet created
Step 6  →  first call to load_or_compute_thresholds()
           computes P25/P50/P95 from model_events.parquet
           saves: 6_final_model/outputs/{cohort}/{ab}/n_event_bin_thresholds.json
Step 6  →  train_per_bin() uses thresholds to split training data
Step 7  →  SHAP per-bin: --bin filters CSV rows by n_event_bin column
Step 8  →  FFA per-bin: same --bin pattern
Step 9  →  DTW/FP-Growth/BupaR: load thresholds JSON for labeling
Lambda  →  load thresholds JSON at startup; assign bin from submitted code count
```

**Step 6 is the canonical threshold producer.** All subsequent steps call `load_or_compute_thresholds(cache_path=<Step6 path>)`. If the JSON exists they use it; if not, they compute and write it themselves. This guarantees every step uses the same population-derived cut-points.

### Threshold JSON format

```json
{
  "p25": 8.0,
  "p50": 22.0,
  "p95": 87.0,
  "n": 45231,
  "min": 1.0,
  "max": 412.0,
  "cohort": "opioid_ed",
  "age_band": "25-44"
}
```

**Canonical path:** `6_final_model/outputs/{cohort}/{age_band_fname}/n_event_bin_thresholds.json`

---

## Step-by-step usage

### Step 6 — Per-bin model training (`run_final_model.py`)

`train_per_bin()` trains four separate models, one per density bin:

```python
for bin_name in DENSITY_BINS:   # ("low", "medium", "high", "extreme")
    subset = df[df["n_event_bin"] == bin_name]
    train_and_evaluate(subset, ...)
```

Each per-bin model writes to `outputs/{cohort}/{ab}/bin_models/{bin_name}/`:

```
bin_models/{bin_name}/
  {model_type}.joblib
  models/xgboost_model.ubj
  models/catboost_model.cbm
  models/calibration_{model_type}.joblib
  {cohort}_{ab}_{model_type}_feature_importance.csv
  {cohort}_{ab}_model_metrics_summary.csv
  {cohort}_{ab}_model_selection_metadata.json
```

Optuna hyperparameter tuning runs automatically inside `train_and_evaluate()` when `n_runs` is passed (same path as full-cohort training).

### Step 7 — SHAP per-bin

```bash
python 7_shap_analysis/run_shap_analysis.py \
    --cohort opioid_ed --age-band 25-44 --bin low
```

Filters the features CSV to `n_event_bin = 'low'` rows, loads `bin_models/low/models/xgboost_model.ubj`, writes output to `7_shap_analysis/outputs/{cohort}/{ab}/bin_models/low/`.

### Step 8 — FFA per-bin

Same `--bin` argument pattern. Loads the best XGBoost variant from `bin_models/{bin_name}/` for symbolic rule extraction.

### Step 9 — Dashboard visuals

**DTW** (`create_dtw_trajectories.py`): computes `event_density_bin` for each patient using the saved thresholds JSON. `chart_data.json` includes density-stratified series; the dashboard "Event density" dropdown filter uses these bins.

**FP-Growth** and **BupaR**: use `event_density_bin` labels for transaction density grouping within their pipelines.

### Deployment — `prepare_models.py`

Copies all four `bin_models/{bin_name}/` subdirectories (models + calibration + FI CSVs) into the Lambda output directory:

```
10_risk_dashboard/outputs/models/{cohort}/{ab}/
  bin_models/low/
  bin_models/medium/
  bin_models/high/
  bin_models/extreme/
```

### Lambda inference (`lambda_function.py`)

**Per-bin only — no full-cohort fallback.** If a bin model is missing, Lambda raises `FileNotFoundError`.

```python
def handle_risk(event, context):
    body = json.loads(event["body"])
    n_codes = len(body.get("drugs", []) + body.get("icds", []) + body.get("cpts", []))
    bin_name = assign_n_event_bin(n_codes, thresholds)   # e.g. "medium"
    model     = load_model(cohort, age_band, bin_name=bin_name)
    calibration = load_calibration_model(cohort, age_band, bin_name=bin_name)
    causal_fi   = load_causal_importance(cohort, age_band, bin_name=bin_name)
```

The submitted code count (number of drug + ICD + CPT codes in the request body) is used as the event-density proxy at inference time.

---

## API reference (`py_helpers/event_density_utils.py`)

| Symbol | Description |
|--------|-------------|
| `DENSITY_BINS` | `("low", "medium", "high", "extreme")` — ordered tuple |
| `compute_bin_thresholds(series)` | Returns P25/P50/P95 dict from a numeric Series |
| `assign_n_event_bin(value, thresholds)` | Classify a single value → bin string |
| `assign_n_event_bins(series, thresholds=None)` | Vectorized: Series → Series of bin strings |
| `save_thresholds(thresholds, path)` | Write dict to JSON |
| `load_thresholds(path)` | Read JSON; returns None if missing/invalid |
| `load_or_compute_thresholds(model_events_path, cache_path, cohort, age_band)` | Priority: cache → compute from parquet → default |
| `default_threshold_cache_path(project_root, cohort, age_band)` | Returns canonical `6_final_model/outputs/...` path |

---

## Critical: re-deploy required after re-training

```
train_per_bin() (Notebook 3)
       ↓
prepare_models.py   ← MUST re-run after any bin model change
       ↓
Docker build + ECR push + Lambda update (Notebook 5)
```

If `train_per_bin()` is re-run but `prepare_models.py` is not, Lambda still uses the old models and **inference will fail with FileNotFoundError** for any bin whose model path changed.

---

## Related documentation

- `py_helpers/event_density_utils.py` — full API source
- `6_final_model/README.md` — per-bin training and selection details
- `docs/Step7_SHAP/README_shap_analysis.md` — per-bin SHAP
- `10_risk_dashboard/README.md` — Lambda model loading and weight logic
