# %% [markdown]
# # CH4 utilization-free model sensitivity (CTS-2026-0235R2)
#
# **AE ask:** Refit the CH4 (`non_opioid_ed`) model **without utilization-derived features**
# and compare vs the primary model on (1) top SHAP drug ranks, (2) interaction
# (pair/triplet) overlap, (3) holdout AUPRC — reporting degradation honestly.
#
# ## How to run (PowerShell)
#
# ```powershell
# cd C:\Projects\pgx-analysis
# .\.venv\Scripts\Activate.ps1
# # Option A — VS Code / Cursor: open this .py as a notebook and Run All
# # Option B — script:
# python 6_final_model\notebooks\dev\ch04_utilization_free_sensitivity.py
# # Option C — Jupyter:
# jupyter nbconvert --to notebook --execute `
#   6_final_model\notebooks\dev\ch04_utilization_free_sensitivity.ipynb `
#   --output-dir 6_final_model\notebooks\dev
# ```
#
# Requires AWS credentials with read access to `s3://pgxdatalake` (models, FI, PGx,
# `model_events_no_protocols`, scenario SHAP/interaction CSVs).
#
# ## Scope / design choice (read before interpreting)
#
# | Choice | Decision |
# |--------|----------|
# | Cohort | **CH4 only** — `non_opioid_ed` (polypharmacy / non-opioid ED) |
# | Band | **65–74** first (primary geriatric band in METRICS / Table 2 framing) |
# | Density | **`low` bin** (matches manuscript low-density holdout reporting) |
# | Learner | **CatBoost** for AUPRC (selected model for this band in `METRICS.md`); **XGBoost TreeSHAP** for drug ranks (aligned with FFA / XGB explain path) |
# | Refit | **Actual refit** of CatBoost + XGBoost on 2016–2018 train, **without** utilization features; skip full MC-CV and use production-like hyperparameters for runtime |
# | Interactions | Primary pairs/triplets from published FFA synergy tables + scenario `top_interaction_factors`; util-free pairs from TreeSHAP interaction strengths on top drugs (full FFA AXP re-run is multi-hour and out of scope here) |
#
# Expand later: set `AGE_BANDS = ["65-74","75-84","85-114"]` and/or `DENSITY_BINS_RUN = ["low","medium",...]`.
#
# ## Outputs
#
# - Local run dir: `6_final_model/outputs/notebook_artifacts/ch04_utilization_free_sensitivity/<run>/`
# - Manuscript-ready copies: `manuscript/data/supplementary/ch04_utilization_free_sensitivity/`
# - Key files: `auprc_comparison.csv`, `shap_drug_rank_overlap.csv`, `interaction_overlap.csv`,
#   `utilization_features_removed.json`, `summary.json`

# %%
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "py_helpers").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # noqa: E402
from py_helpers.env_utils import get_data_root, get_model_data_root  # noqa: E402
from py_helpers.feature_importance_model_utils import (  # noqa: E402
    train_catboost,
    train_xgboost,
    predict_proba_catboost,
    predict_proba_xgboost,
)
from py_helpers.notebook_artifacts import (  # noqa: E402
    github_artifact_path,
    local_artifact_path,
    setup_notebook_artifacts,
)

NB_CONTEXT = setup_notebook_artifacts(
    notebook_file=__file__ if "__file__" in globals() else "ch04_utilization_free_sensitivity.ipynb",
    step_name="6_final_model",
    run_label="cts_0235r2_ch04_util_free",
)
OUT = Path(NB_CONTEXT.local_output_dir)
OUT.mkdir(parents=True, exist_ok=True)
SUPP = PROJECT_ROOT / "manuscript" / "data" / "supplementary" / "ch04_utilization_free_sensitivity"
SUPP.mkdir(parents=True, exist_ok=True)

print("GitHub artifact dir:", NB_CONTEXT.github_dir)
print("Local output dir:", OUT)
print("Supplementary mirror:", SUPP)

# %% [markdown]
# ## Config

# %%
COHORT = "non_opioid_ed"
AGE_BAND = "65-74"
AGE_FNAME = age_band_to_fname(AGE_BAND)
DENSITY_BIN = "low"
TOP_K_DRUGS = 20
TOP_K_INTERACTIONS = 50
S3_BUCKET = "pgxdatalake"
RANDOM_SEED = 1997

# Production-like hyperparams (train_final_model / Step 6 defaults); MC-CV skipped for runtime.
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "task_type": "CPU",
}
XGBOOST_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
}

# Exact utilization columns emitted by `run_final_model._compute_temporal_event_dynamics`
# plus density ordinal / raw count (R1 framing: n_events / utilization density + temporal
# utilization-control layer).
UTILIZATION_EXACT = {
    "n_events",
    "n_event_bin",
    "n_event_bin_ordinal",
    "event_span_days",
    "event_rate_per30",
    "mean_inter_event_days",
    "median_inter_event_days",
    "std_inter_event_days",
    "event_burstiness",
    "early_event_rate_per30",
    "late_event_rate_per30",
    "event_rate_delta_per30",
    "event_rate_ratio_late_vs_early",
    "recent30_event_count",
    "recent90_event_count",
    "recent30_event_fraction",
    "recent90_event_fraction",
}
UTILIZATION_PREFIXES = ("event_rate", "recent30_", "recent90_", "mean_inter_event", "median_inter_event", "std_inter_event")

META_COLS = {"mi_person_key", "target", "patient_year", "event_year", "n_event_bin"}


def is_utilization_feature(name: str) -> bool:
    n = str(name)
    if n in UTILIZATION_EXACT:
        return True
    if n.startswith("n_event"):
        return True
    return any(n.startswith(p) or p in n for p in UTILIZATION_PREFIXES)


def mirror_to_supplementary(path: Path) -> Path:
    dest = SUPP / path.name
    dest.write_bytes(path.read_bytes())
    return dest


def s3_download(key: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] exists {dest.name}")
        return True
    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client("s3", region_name="us-east-1")
        client.download_file(S3_BUCKET, key, str(dest))
        print(f"[ok] s3://{S3_BUCKET}/{key} -> {dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] download failed {key}: {exc}")
        return False


# %% [markdown]
# ## Sync inputs (S3 → local)

# %%
DATA_ROOT = get_data_root()
MODEL_DATA_ROOT = get_model_data_root()
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Primary models (dashboard deployment bundle — live after gold/final_model cleanup)
primary_model_dir = CACHE / "primary_models" / COHORT / AGE_FNAME / "bin_models" / DENSITY_BIN
s3_download(
    f"gold/dashboard/models/{COHORT}/{AGE_FNAME}/bin_models/{DENSITY_BIN}/catboost.joblib",
    primary_model_dir / "catboost.joblib",
)
s3_download(
    f"gold/dashboard/models/{COHORT}/{AGE_FNAME}/bin_models/{DENSITY_BIN}/xgboost.joblib",
    primary_model_dir / "xgboost.joblib",
)
s3_download(
    f"gold/dashboard/models/{COHORT}/{AGE_FNAME}/feature_schema.json",
    CACHE / "feature_schema.json",
)
s3_download(
    f"gold/dashboard/models/{COHORT}/{AGE_FNAME}/n_event_bin_thresholds.json",
    CACHE / "n_event_bin_thresholds.json",
)

# Scenario SHAP + interaction tables (primary reference)
scenario_dir = CACHE / "scenario" / DENSITY_BIN
s3_download(
    f"gold/dashboard/visualizations/scenario/{COHORT}/{AGE_FNAME}/{DENSITY_BIN}/combined_shap_importance.csv",
    scenario_dir / "combined_shap_importance.csv",
)
s3_download(
    f"gold/dashboard/visualizations/scenario/{COHORT}/{AGE_FNAME}/{DENSITY_BIN}/top_interaction_factors.csv",
    scenario_dir / "top_interaction_factors.csv",
)
# Manuscript gold SHAP (optional secondary)
s3_download(
    f"gold/manuscript_checkpoints/shap/{COHORT}/{AGE_BAND}/{DENSITY_BIN}/shap_manuscript_summary.json",
    CACHE / "shap_manuscript_summary.json",
)

# Step 3b FI + PGx (paths expected by run_final_model.build_final_features)
fi_dest = (
    PROJECT_ROOT
    / "3b_feature_importance_eda"
    / "outputs"
    / COHORT
    / AGE_FNAME
    / f"{COHORT}_{AGE_FNAME}_cohort_feature_importance.csv"
)
s3_download(
    f"gold/feature_importance/{COHORT}/{AGE_BAND}/{COHORT}_{AGE_FNAME}_cohort_feature_importance.csv",
    fi_dest,
)
# Also mirror under DATA_ROOT gold for file_resolver consumers
s3_download(
    f"gold/feature_importance/{COHORT}/{AGE_BAND}/{COHORT}_{AGE_FNAME}_cohort_feature_importance.csv",
    DATA_ROOT
    / "gold"
    / "feature_importance"
    / COHORT
    / AGE_BAND
    / f"{COHORT}_{AGE_FNAME}_cohort_feature_importance.csv",
)
s3_download(
    f"gold/feature_importance/{COHORT}/{AGE_BAND}/{COHORT}_{AGE_FNAME}_aggregated_feature_importance.csv",
    DATA_ROOT
    / "gold"
    / "feature_importance"
    / COHORT
    / AGE_BAND
    / f"{COHORT}_{AGE_FNAME}_aggregated_feature_importance.csv",
)
pgx_dest = (
    PROJECT_ROOT
    / "5_pgx_analysis"
    / "outputs"
    / "feature_engineering"
    / f"pgx_added_features_{COHORT}_{AGE_FNAME}.csv"
)
s3_download(
    f"gold/feature_engineering/7_pgx/{COHORT}/{AGE_BAND}/pgx_added_features_{COHORT}_{AGE_FNAME}.csv",
    pgx_dest,
)

# model_events: prefer DTW-filtered no-protocols parquet currently on S3
events_dest = (
    MODEL_DATA_ROOT
    / f"cohort_name={COHORT}"
    / f"age_band={AGE_BAND}"
    / "model_events.parquet"
)
ok_events = s3_download(
    f"gold/dtw_filter/{COHORT}/{AGE_BAND}/model_events_no_protocols.parquet",
    events_dest,
)
if not ok_events:
    print(
        "[blocker] model_events missing. Re-run Step 4 or restore "
        f"s3://{S3_BUCKET}/gold/dtw_filter/{COHORT}/{AGE_BAND}/model_events_no_protocols.parquet"
    )

# %% [markdown]
# ## Identify utilization features + build / load matrix

# %%
schema_path = CACHE / "feature_schema.json"
schema_features: list[str] = []
if schema_path.exists():
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_features = list(schema.get("features") or schema.get("feature_names") or [])

util_from_schema = sorted(f for f in schema_features if is_utilization_feature(f))
util_path = OUT / "utilization_features_removed.json"
util_payload = {
    "cohort": COHORT,
    "age_band": AGE_BAND,
    "density_bin": DENSITY_BIN,
    "definition": (
        "Utilization-derived = density ordinal / raw counts + pre-index temporal "
        "event-rate / gap / burstiness / recency features from Step 6 "
        "(_compute_temporal_event_dynamics). PGx burden counts are RETAINED "
        "(medication content, not utilization volume)."
    ),
    "exact_catalog": sorted(UTILIZATION_EXACT),
    "present_in_primary_schema": util_from_schema,
}
util_path.write_text(json.dumps(util_payload, indent=2), encoding="utf-8")
mirror_to_supplementary(util_path)
print("Utilization features in primary schema:", util_from_schema or "(none listed / schema empty)")


def load_or_build_feature_frame() -> pd.DataFrame:
    """Prefer rebuilt Step-6 features; fall back to any local no-leakage CSV."""
    candidates = [
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / COHORT
        / AGE_FNAME
        / f"{COHORT}_{AGE_FNAME}_train_final_features_no_leakage.csv",
        DATA_ROOT
        / "6_final_model"
        / "outputs"
        / COHORT
        / AGE_FNAME
        / f"{COHORT}_{AGE_FNAME}_train_final_features_no_leakage.csv",
        OUT / f"{COHORT}_{AGE_FNAME}_train_final_features_no_leakage.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            print(f"Loading existing features: {p}")
            return pd.read_csv(p)

    if not events_dest.exists():
        raise FileNotFoundError(
            f"Cannot build features; missing model_events at {events_dest}"
        )

    print("Building feature matrix via run_final_model.build_final_features ...")
    import importlib.util

    # Import by path — top-level package name is numeric-prefixed.
    spec = importlib.util.spec_from_file_location(
        "run_final_model",
        PROJECT_ROOT / "6_final_model" / "run_final_model.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_final_model"] = mod
    spec.loader.exec_module(mod)
    df = mod.build_final_features(COHORT, AGE_BAND)
    out_csv = OUT / f"{COHORT}_{AGE_FNAME}_train_final_features_no_leakage.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote rebuilt features: {out_csv} shape={df.shape}")
    return df


FEATURES_OK = True
try:
    df_all = load_or_build_feature_frame()
except Exception as exc:  # noqa: BLE001
    FEATURES_OK = False
    df_all = pd.DataFrame()
    print(f"[blocker] feature matrix unavailable: {exc}")

# %% [markdown]
# ## Temporal + density split; drop utilization columns for sensitivity matrix

# %%
BLOCKERS: list[str] = []
if not FEATURES_OK:
    BLOCKERS.append("feature_matrix")

train_df = hold_df = pd.DataFrame()
util_removed: list[str] = []

if FEATURES_OK:
    # Attach patient_year from events if needed for temporal split
    if "patient_year" not in df_all.columns and "event_year" not in df_all.columns:
        if events_dest.exists():
            import duckdb

            years = duckdb.query(
                f"""
                SELECT CAST(mi_person_key AS VARCHAR) AS mi_person_key,
                       CAST(MAX(event_year) AS INTEGER) AS patient_year
                FROM read_parquet('{str(events_dest).replace(chr(92), "/")}')
                GROUP BY 1
                """
            ).df()
            df_all = df_all.merge(years, on="mi_person_key", how="left")
        else:
            BLOCKERS.append("patient_year")
            print("[blocker] no patient_year and no events to derive it")

    year_col = "patient_year" if "patient_year" in df_all.columns else "event_year"
    if year_col in df_all.columns:
        df_all[year_col] = pd.to_numeric(df_all[year_col], errors="coerce").fillna(2018).astype(int)
        train_df = df_all[df_all[year_col] <= 2018].copy()
        hold_df = df_all[df_all[year_col] == 2019].copy()
    else:
        BLOCKERS.append("temporal_split")
        train_df = df_all.copy()
        hold_df = pd.DataFrame()

    if "n_event_bin" in train_df.columns:
        train_df = train_df[train_df["n_event_bin"] == DENSITY_BIN].copy()
        if not hold_df.empty:
            hold_df = hold_df[hold_df["n_event_bin"] == DENSITY_BIN].copy()
    else:
        print("[warn] n_event_bin missing — sensitivity uses full age-band (not low-bin only)")

    util_removed = sorted(c for c in train_df.columns if is_utilization_feature(c))
    util_payload["removed_from_matrix"] = util_removed
    util_path.write_text(json.dumps(util_payload, indent=2), encoding="utf-8")
    mirror_to_supplementary(util_path)

    print(
        f"Train n={len(train_df)} cases={int(train_df['target'].sum()) if 'target' in train_df else '?'} | "
        f"Holdout n={len(hold_df)} | util cols removed={len(util_removed)}"
    )
    print("Removed:", util_removed)

# %% [markdown]
# ## Fit util-free models + evaluate AUPRC vs primary

# %%
auprc_rows: list[dict] = []


def feature_matrix(df: pd.DataFrame, drop_util: bool) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target"].astype(int)
    cols = [
        c
        for c in df.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and not (drop_util and is_utilization_feature(c))
    ]
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def predict_joblib(model_path: Path, X: pd.DataFrame) -> np.ndarray | None:
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    names = None
    for attr in ("feature_names_", "feature_names"):
        names = getattr(model, attr, None)
        if names is not None:
            break
    if names is None and hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
    if names is None and hasattr(model, "feature_names_"):
        names = model.feature_names_
    X_aligned = X.reindex(columns=list(names), fill_value=0) if names is not None else X
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X_aligned)[:, 1], dtype=float)
        except Exception:
            pass
    try:
        return np.asarray(predict_proba_catboost(model, X_aligned), dtype=float)
    except Exception:
        pass
    try:
        return np.asarray(predict_proba_xgboost(model, X_aligned), dtype=float)
    except Exception:
        return None


sens_cb = sens_xgb = None
X_train_full = X_hold_full = X_train_sens = X_hold_sens = None
y_train = y_hold = None

if FEATURES_OK and not hold_df.empty and "temporal_split" not in BLOCKERS:
    X_train_full, y_train = feature_matrix(train_df, drop_util=False)
    X_hold_full, y_hold = feature_matrix(hold_df, drop_util=False)
    X_train_sens, _ = feature_matrix(train_df, drop_util=True)
    X_hold_sens, _ = feature_matrix(hold_df, drop_util=True)

    # Primary CatBoost AUPRC (selected learner)
    primary_cb = primary_model_dir / "catboost.joblib"
    p_primary = predict_joblib(primary_cb, X_hold_full)
    if p_primary is not None and y_hold.nunique() > 1:
        auprc_rows.append(
            {
                "model": "primary_catboost_dashboard",
                "features": "with_utilization",
                "auprc": float(average_precision_score(y_hold, p_primary)),
                "auroc": float(roc_auc_score(y_hold, p_primary)),
                "n_holdout": int(len(y_hold)),
                "n_features": int(X_hold_full.shape[1]),
                "prevalence": float(y_hold.mean()),
            }
        )
    else:
        BLOCKERS.append("primary_catboost_predict")
        print("[warn] could not score primary CatBoost on holdout")

    print("Refitting util-free CatBoost + XGBoost (single final fit, no MC-CV)...")
    sens_cb = train_catboost(X_train_sens, y_train, CATBOOST_PARAMS)
    sens_xgb = train_xgboost(X_train_sens, y_train, XGBOOST_PARAMS)
    joblib.dump(sens_cb, OUT / "util_free_catboost.joblib")
    joblib.dump(sens_xgb, OUT / "util_free_xgboost.joblib")

    p_sens_cb = predict_proba_catboost(sens_cb, X_hold_sens)
    p_sens_xgb = predict_proba_xgboost(sens_xgb, X_hold_sens)
    for label, proba, n_feat in (
        ("util_free_catboost", p_sens_cb, X_hold_sens.shape[1]),
        ("util_free_xgboost", p_sens_xgb, X_hold_sens.shape[1]),
    ):
        auprc_rows.append(
            {
                "model": label,
                "features": "without_utilization",
                "auprc": float(average_precision_score(y_hold, proba)),
                "auroc": float(roc_auc_score(y_hold, proba)),
                "n_holdout": int(len(y_hold)),
                "n_features": int(n_feat),
                "prevalence": float(y_hold.mean()),
            }
        )

    auprc_df = pd.DataFrame(auprc_rows)
    if len(auprc_df) >= 2 and "primary_catboost_dashboard" in set(auprc_df["model"]):
        base = float(auprc_df.loc[auprc_df["model"] == "primary_catboost_dashboard", "auprc"].iloc[0])
        sens = float(auprc_df.loc[auprc_df["model"] == "util_free_catboost", "auprc"].iloc[0])
        auprc_df["auprc_delta_vs_primary_catboost"] = auprc_df["auprc"] - base
        print(f"AUPRC primary={base:.4f}  util-free CatBoost={sens:.4f}  delta={sens - base:+.4f}")
    auprc_path = OUT / "auprc_comparison.csv"
    auprc_df.to_csv(auprc_path, index=False)
    mirror_to_supplementary(auprc_path)
    print(auprc_df.to_string(index=False))
else:
    print("[skip] AUPRC — features/holdout not ready:", BLOCKERS)
    pd.DataFrame(auprc_rows).to_csv(OUT / "auprc_comparison.csv", index=False)

# %% [markdown]
# ## Top SHAP drug features — primary vs util-free

# %%


def mean_abs_shap_xgb(model, X: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    Xs = X
    if len(Xs) > max_rows:
        Xs = Xs.sample(n=max_rows, random_state=RANDOM_SEED)
    names = list(booster.feature_names or Xs.columns)
    Xs = Xs.reindex(columns=names, fill_value=0).astype("float32")
    dmat = xgb.DMatrix(Xs, feature_names=names)
    contribs = booster.predict(dmat, pred_contribs=True)
    # last column is bias
    shap_vals = contribs[:, :-1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def drug_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["feature"].astype(str).str.startswith("item_drug_")].copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


shap_overlap_path = OUT / "shap_drug_rank_overlap.csv"
primary_shap = pd.DataFrame()
sens_shap = pd.DataFrame()

# Primary drug ranks from scenario combined SHAP (authoritative for manuscript display)
scenario_shap_path = scenario_dir / "combined_shap_importance.csv"
if scenario_shap_path.exists():
    raw = pd.read_csv(scenario_shap_path)
    feat_col = "feature" if "feature" in raw.columns else raw.columns[0]
    score_col = next(
        (c for c in ("mean_abs_shap", "combined_importance", "shap_importance", "importance") if c in raw.columns),
        raw.columns[1],
    )
    primary_shap = raw.rename(columns={feat_col: "feature", score_col: "mean_abs_shap"})[
        ["feature", "mean_abs_shap"]
    ]
    primary_shap = drug_only(primary_shap.sort_values("mean_abs_shap", ascending=False))
else:
    print("[warn] scenario combined_shap_importance missing")

if sens_xgb is not None and X_hold_sens is not None:
    sens_shap = drug_only(mean_abs_shap_xgb(sens_xgb, X_hold_sens))
    sens_shap.to_csv(OUT / "util_free_shap_global_drugs.csv", index=False)
    mirror_to_supplementary(OUT / "util_free_shap_global_drugs.csv")

if not primary_shap.empty and not sens_shap.empty:
    top_p = primary_shap.head(TOP_K_DRUGS).copy()
    top_s = sens_shap.head(TOP_K_DRUGS).copy()
    rank_s = {r.feature: int(r.rank) for r in top_s.itertuples(index=False)}
    # also allow lookup outside top-k
    rank_s_all = {r.feature: int(r.rank) for r in sens_shap.itertuples(index=False)}
    rows = []
    for r in top_p.itertuples(index=False):
        rows.append(
            {
                "feature": r.feature,
                "primary_rank": int(r.rank),
                "primary_mean_abs_shap": float(r.mean_abs_shap),
                "util_free_rank": rank_s_all.get(r.feature),
                "rank_shift": (rank_s_all[r.feature] - int(r.rank)) if r.feature in rank_s_all else None,
                "in_util_free_top_k": r.feature in rank_s,
            }
        )
    overlap = pd.DataFrame(rows)
    set_p = set(top_p["feature"])
    set_s = set(top_s["feature"])
    jaccard = len(set_p & set_s) / max(len(set_p | set_s), 1)
    summary_row = {
        "feature": "__SUMMARY__",
        "primary_rank": None,
        "primary_mean_abs_shap": None,
        "util_free_rank": None,
        "rank_shift": None,
        "in_util_free_top_k": None,
        "top_k": TOP_K_DRUGS,
        "n_overlap": len(set_p & set_s),
        "jaccard": jaccard,
    }
    overlap = pd.concat([overlap, pd.DataFrame([summary_row])], ignore_index=True)
    overlap.to_csv(shap_overlap_path, index=False)
    mirror_to_supplementary(shap_overlap_path)
    print(f"Top-{TOP_K_DRUGS} drug SHAP overlap={len(set_p & set_s)}  Jaccard={jaccard:.3f}")
    print(overlap.head(12).to_string(index=False))
else:
    print("[skip] SHAP drug overlap — missing primary or util-free ranks")
    pd.DataFrame().to_csv(shap_overlap_path, index=False)

# %% [markdown]
# ## Interaction overlap (pairs / triplets)
#
# Primary synergistic pairs come from manuscript `ffa_synergy_pairs.json` (FFA IE > 1)
# filtered to 65–74, plus scenario `top_interaction_factors.csv` when present.
# Util-free interactions are TreeSHAP pairwise interaction strengths among top drugs
# (practical substitute for full multi-hour FFA AXP re-extraction on the refit).

# %%


def normalize_drug_token(x: str) -> str:
    s = str(x).strip()
    if s.startswith("item_drug_"):
        return s
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.upper()).strip("_")
    return f"item_drug_{s}"


def pair_key(a: str, b: str) -> frozenset:
    return frozenset({normalize_drug_token(a), normalize_drug_token(b)})


def load_primary_pairs() -> set[frozenset]:
    pairs: set[frozenset] = set()
    manu = PROJECT_ROOT / "manuscript" / "data" / "ffa_synergy_pairs.json"
    if manu.exists():
        data = json.loads(manu.read_text(encoding="utf-8"))
        for row in data:
            if str(row.get("age_band")) != AGE_BAND:
                continue
            pairs.add(pair_key(row["drug_a"], row["drug_b"]))
    ix_path = scenario_dir / "top_interaction_factors.csv"
    if ix_path.exists():
        ix = pd.read_csv(ix_path)
        # Flexible column names seen across exports
        for _, row in ix.iterrows():
            cols = {c.lower(): c for c in ix.columns}
            if "feature_combination" in cols:
                parts = str(row[cols["feature_combination"]]).split("|")
                drugs = [p for p in parts if str(p).startswith("item_drug_")]
                for a, b in combinations(sorted(set(drugs)), 2):
                    pairs.add(pair_key(a, b))
            elif "drug_a" in cols and "drug_b" in cols:
                pairs.add(pair_key(row[cols["drug_a"]], row[cols["drug_b"]]))
            elif "feature_a" in cols and "feature_b" in cols:
                pairs.add(pair_key(row[cols["feature_a"]], row[cols["feature_b"]]))
    return pairs


def shap_interaction_pairs(model, X: pd.DataFrame, top_features: list[str], max_rows: int = 400) -> pd.DataFrame:
    """Pairwise mean |SHAP interaction| on a feature subset (runtime-safe)."""
    import shap

    Xs = X[top_features].copy()
    if len(Xs) > max_rows:
        Xs = Xs.sample(n=max_rows, random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    # shap_interaction_values can be heavy; subset features keeps it tractable
    vals = explainer.shap_interaction_values(Xs)
    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    vals = np.asarray(vals)
    # vals shape: (n, f, f)
    mean_abs = np.abs(vals).mean(axis=0)
    rows = []
    for i, j in combinations(range(len(top_features)), 2):
        rows.append(
            {
                "feature_a": top_features[i],
                "feature_b": top_features[j],
                "mean_abs_interaction": float(mean_abs[i, j]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)


interaction_path = OUT / "interaction_overlap.csv"
primary_pairs = load_primary_pairs()
print(f"Primary synergistic/scenario pairs (65-74): {len(primary_pairs)}")

if sens_xgb is not None and X_hold_sens is not None and not sens_shap.empty:
    top_drugs = sens_shap.head(min(15, len(sens_shap)))["feature"].tolist()
    # Restrict to columns present
    top_drugs = [f for f in top_drugs if f in X_hold_sens.columns]
    try:
        sens_ix = shap_interaction_pairs(sens_xgb, X_hold_sens, top_drugs)
        sens_ix.to_csv(OUT / "util_free_shap_interaction_pairs.csv", index=False)
        mirror_to_supplementary(OUT / "util_free_shap_interaction_pairs.csv")
        sens_pairs = {
            pair_key(r.feature_a, r.feature_b)
            for r in sens_ix.head(TOP_K_INTERACTIONS).itertuples(index=False)
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] SHAP interactions failed ({exc}); falling back to co-ranked drug pairs")
        sens_pairs = {
            pair_key(a, b) for a, b in combinations(top_drugs[:10], 2)
        }
        sens_ix = pd.DataFrame()

    inter = len(primary_pairs & sens_pairs)
    union = len(primary_pairs | sens_pairs) or 1
    ix_summary = pd.DataFrame(
        [
            {
                "metric": "n_primary_pairs",
                "value": len(primary_pairs),
            },
            {
                "metric": "n_util_free_top_pairs",
                "value": len(sens_pairs),
            },
            {
                "metric": "n_overlap",
                "value": inter,
            },
            {
                "metric": "jaccard",
                "value": inter / union,
            },
            {
                "metric": "note",
                "value": (
                    "Primary=FFA synergy pairs (IE>1) + scenario top_interaction_factors; "
                    "util-free=TreeSHAP interaction strengths on top drug features after util-free XGB refit. "
                    "Not a full FFA AXP re-extraction."
                ),
            },
        ]
    )
    # Detail overlap list
    detail = []
    for p in sorted(primary_pairs & sens_pairs, key=lambda s: sorted(s)):
        a, b = sorted(p)
        detail.append({"pair": f"{a}|{b}", "in_both": True})
    for p in sorted(primary_pairs - sens_pairs, key=lambda s: sorted(s))[:30]:
        a, b = sorted(p)
        detail.append({"pair": f"{a}|{b}", "in_both": False, "only": "primary"})
    detail_df = pd.DataFrame(detail)
    ix_summary.to_csv(interaction_path, index=False)
    detail_df.to_csv(OUT / "interaction_overlap_detail.csv", index=False)
    mirror_to_supplementary(interaction_path)
    mirror_to_supplementary(OUT / "interaction_overlap_detail.csv")
    print(ix_summary.to_string(index=False))
else:
    print("[skip] interaction overlap — models/SHAP not ready")
    pd.DataFrame([{"metric": "status", "value": "skipped"}]).to_csv(interaction_path, index=False)

# %% [markdown]
# ## Summary artifact for response letter

# %%
summary = {
    "revision": "CTS-2026-0235R2",
    "cohort": COHORT,
    "age_band": AGE_BAND,
    "density_bin": DENSITY_BIN,
    "utilization_features_removed": util_removed,
    "blockers": BLOCKERS,
    "outputs": {
        "local": str(OUT),
        "supplementary": str(SUPP),
        "auprc": str(SUPP / "auprc_comparison.csv"),
        "shap_drugs": str(SUPP / "shap_drug_rank_overlap.csv"),
        "interactions": str(SUPP / "interaction_overlap.csv"),
        "util_list": str(SUPP / "utilization_features_removed.json"),
    },
    "interpretation_guide": {
        "auprc": (
            "Report util-free CatBoost AUPRC next to primary dashboard CatBoost. "
            "A drop is expected if utilization helped discrimination; persistence of "
            "drug SHAP ranks / pair overlap still supports medication-signal robustness."
        ),
        "shap_drugs": (
            "Jaccard and rank_shift on top-20 item_drug_* features. High overlap / "
            "small rank shifts = key drug findings persist without utilization features."
        ),
        "interactions": (
            "Jaccard between manuscript FFA synergistic pairs and util-free TreeSHAP "
            "interaction pairs. Full FFA re-run would be the strictest match; this "
            "answers the AE ask for overlap evidence at revision-feasible runtime."
        ),
        "expand": (
            "Re-run with AGE_BAND in {75-84, 85-114} and/or additional density bins; "
            "optionally enable MC-CV in CATBOOST_PARAMS loop for production parity."
        ),
    },
}
summary_path = OUT / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
mirror_to_supplementary(summary_path)
gh = github_artifact_path(NB_CONTEXT, "summary.json")
gh.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nDone. Supplementary mirror:", SUPP)
