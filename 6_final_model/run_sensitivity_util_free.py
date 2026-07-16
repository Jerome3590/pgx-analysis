"""CH4 utilization-free sensitivity — all modeled age bands (CTS-2026-0235R2).

Run via root 3_model_sensitivity.ipynb (EC2) or:
  python 6_final_model/run_sensitivity_util_free.py
  python 6_final_model/run_sensitivity_util_free.py --age-bands 65-74,75-84
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import AGE_BANDS, age_band_to_fname  # noqa: E402
from py_helpers.notebook_artifacts import (  # noqa: E402
    github_artifact_path,
    local_artifact_path,
    setup_notebook_artifacts,
)

COHORT = "non_opioid_ed"
S3_BUCKET = "pgxdatalake"
# Public GitHub SSOT keyed by journal manuscript ID (manuscript/ is private).
MS_DIR = PROJECT_ROOT / "reports" / "CTS-2026-0235R2"

# Utilization-derived columns to drop for the AE sensitivity refit.
UTIL_EXACT = {
    "n_events",
    "n_event_bin",
    "n_event_bin_ordinal",
    "event_span_days",
    "event_burstiness",
    "event_rate_per30",
    "event_rate_per_30d",
    "mean_inter_event_days",
    "median_inter_event_days",
    "std_inter_event_days",
    "early_event_rate_per30",
    "late_event_rate_per30",
    "event_rate_delta_per30",
    "event_rate_ratio_late_vs_early",
    "early_event_rate_per_30d",
    "late_event_rate_per_30d",
    "event_rate_delta_per_30d",
    "event_rate_ratio_late_vs_early_legacy",
    "recent30_event_count",
    "recent90_event_count",
    "recent30_event_fraction",
    "recent90_event_fraction",
}
UTIL_SUBSTR = (
    "inter_event",
    "burstiness",
    "event_rate",
    "recent30_",
    "recent90_",
    "event_span",
)

PUBLISHED_PAIRS = [
    ("Acetaminophen", "Levofloxacin"),
    ("Levofloxacin", "Lorazepam"),
    ("Carvedilol", "Levofloxacin"),
    ("Gabapentin", "Levofloxacin"),
    ("Digoxin", "Simvastatin"),
]
PUBLISHED_TRIPLETS = [
    ("Furosemide", "Hydrochlorothiazide", "Lisinopril"),
    ("Digoxin", "Furosemide", "Amiodarone"),
]

# Optional manuscript holdout AUPRC anchors when Step-6 metrics JSON missing.
MANUSCRIPT_AUPRC_BY_BAND = {
    "65-74": 0.301,
}

TOP_K_DRUG = 25
RANDOM_STATE = 1997
MAX_SHAP_ROWS = 2000


def is_util_feature(col: str) -> bool:
    if col in UTIL_EXACT:
        return True
    c = col.lower()
    if c.startswith("n_event"):
        return True
    return any(s in c for s in UTIL_SUBSTR)


def drugish(col: str) -> bool:
    c = col.lower()
    return c.startswith("item_") or "drug" in c or c.startswith("ndc_")


def normalize_drug_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def feature_matches_drug(feat: str, drug: str) -> bool:
    f = feat.lower()
    tok = normalize_drug_token(drug)
    return tok in f or tok.replace("_", "") in f.replace("_", "")


def paths_for(age_band: str) -> dict[str, Path]:
    age_fname = age_band_to_fname(age_band)
    primary = PROJECT_ROOT / "6_final_model" / "outputs" / COHORT / age_fname
    sens = PROJECT_ROOT / "6_final_model" / "outputs" / COHORT / f"{age_fname}_util_free"
    comp = (
        PROJECT_ROOT
        / "8_ffa_analysis"
        / "outputs"
        / COHORT
        / f"{age_fname}_util_free_sensitivity"
    )
    ms_band = MS_DIR / age_fname
    return {
        "age_fname": Path(age_fname),  # marker; use str when needed
        "primary": primary,
        "sens": sens,
        "comp": comp,
        "ms_band": ms_band,
    }


def ensure_dirs(p: dict[str, Path]) -> None:
    for key in ("sens", "comp", "ms_band"):
        p[key].mkdir(parents=True, exist_ok=True)
    (p["sens"] / "models").mkdir(parents=True, exist_ok=True)
    MS_DIR.mkdir(parents=True, exist_ok=True)


def sync_primary_artifacts(age_band: str, primary_dir: Path) -> None:
    age_fname = age_band_to_fname(age_band)
    prefixes = [
        f"gold/final_model/{COHORT}/{age_band}",
        f"gold/manuscript/final_model/{COHORT}/{age_band}",
    ]
    rels = [
        f"{COHORT}_{age_fname}_holdout_2019_metrics.json",
        f"{COHORT}_{age_fname}_train_final_features_no_leakage.csv",
        "inputs/model_train/final_features.parquet",
        "inputs/model_test/final_features.parquet",
        "models/xgboost.joblib",
        f"final_model_json/{COHORT}_{age_fname}_best_xgboost_model.json",
    ]
    try:
        import boto3

        s3 = boto3.client("s3")
        for prefix in prefixes:
            for rel in rels:
                key = f"{prefix}/{rel}"
                dest = primary_dir / Path(rel)
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                try:
                    s3.head_object(Bucket=S3_BUCKET, Key=key)
                except Exception:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"[GET] s3://{S3_BUCKET}/{key} -> {dest}")
                s3.download_file(S3_BUCKET, key, str(dest))
    except Exception as exc:
        print(f"[WARN] S3 sync failed for {age_band} ({exc}). Continuing with local.")


def load_frame(prefer_parquet: Path, prefer_csv: Path) -> pd.DataFrame:
    if prefer_parquet.exists():
        print(f"[LOAD] {prefer_parquet}")
        return pd.read_parquet(prefer_parquet)
    if prefer_csv.exists():
        print(f"[LOAD] {prefer_csv}")
        return pd.read_csv(prefer_csv)
    raise FileNotFoundError(
        f"Missing features. Expected {prefer_parquet} or {prefer_csv}."
    )


def build_features_from_events(
    age_band: str, primary_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import duckdb

    sys.path.insert(0, str(PROJECT_ROOT / "6_final_model"))
    from run_final_model import (  # type: ignore
        _resolve_model_events_path,
        build_final_features,
    )

    age_fname = age_band_to_fname(age_band)
    print(f"[BUILD] build_final_features({COHORT}, {age_band}) — may take several minutes")
    df = build_final_features(COHORT, age_band)
    if df.empty:
        raise RuntimeError(f"build_final_features returned empty frame for {age_band}")

    events_path = _resolve_model_events_path(COHORT, age_band)
    ep = str(events_path).replace("\\", "/")
    con = duckdb.connect()
    year_df = con.execute(
        f"SELECT CAST(mi_person_key AS VARCHAR) AS mi_person_key, "
        f"MAX(event_year) AS patient_year "
        f"FROM read_parquet('{ep}') GROUP BY mi_person_key"
    ).df()
    con.close()
    df["mi_person_key"] = df["mi_person_key"].astype(str)
    df = df.merge(year_df, on="mi_person_key", how="left")
    df["patient_year"] = df["patient_year"].fillna(2018).astype(int)
    df_test = df[df["patient_year"] == 2019].drop(columns=["patient_year"]).copy()
    df_train = df[df["patient_year"] <= 2018].drop(columns=["patient_year"]).copy()
    print(
        f"[BUILD] {age_band}: train={len(df_train)} holdout2019={len(df_test)} "
        f"features={df_train.shape[1]}"
    )

    train_out = primary_dir / "inputs" / "model_train"
    test_out = primary_dir / "inputs" / "model_test"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(train_out / "final_features.parquet", index=False)
    df_test.to_parquet(test_out / "final_features.parquet", index=False)
    df_train.to_csv(
        primary_dir / f"{COHORT}_{age_fname}_train_final_features_no_leakage.csv",
        index=False,
    )
    return df_train, df_test


def load_or_build_frames(
    age_band: str, primary_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    age_fname = age_band_to_fname(age_band)
    train_parquet = primary_dir / "inputs" / "model_train" / "final_features.parquet"
    test_parquet = primary_dir / "inputs" / "model_test" / "final_features.parquet"
    train_csv = primary_dir / f"{COHORT}_{age_fname}_train_final_features_no_leakage.csv"
    if train_parquet.exists() and test_parquet.exists():
        return load_frame(train_parquet, train_csv), load_frame(test_parquet, train_csv)
    if train_csv.exists() and "event_year" in pd.read_csv(train_csv, nrows=2).columns:
        df = pd.read_csv(train_csv)
        return (
            df[df["event_year"].astype(str) != "2019"].copy(),
            df[df["event_year"].astype(str) == "2019"].copy(),
        )
    return build_features_from_events(age_band, primary_dir)


def split_xy(
    df: pd.DataFrame,
    drop_util: bool,
    drop_broken_pgx: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    exclude = {"mi_person_key", "target", "n_event_bin", "n_events", "event_year", "year"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if drop_util and is_util_feature(c):
            continue
        if drop_broken_pgx and c.startswith("pgx_"):
            continue
        if not (pd.api.types.is_numeric_dtype(df[c]) or str(df[c].dtype) == "bool"):
            continue
        cols.append(c)
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["target"].astype(int)
    return X, y, cols


def fit_xgb(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBClassifier

    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    spw = (neg / pos) if pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=10,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X, y)
    return model


def holdout_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    if len(y) == 0 or y.nunique() < 2:
        return {
            "n_holdout": int(len(y)),
            "n_cases": int(y.sum()) if len(y) else 0,
            "prevalence": None,
            "auroc": None,
            "auprc": None,
            "pr_lift": None,
        }
    proba = model.predict_proba(X)[:, 1]
    prev = float(y.mean())
    auprc = float(average_precision_score(y, proba))
    auroc = float(roc_auc_score(y, proba))
    return {
        "n_holdout": int(len(y)),
        "n_cases": int(y.sum()),
        "prevalence": round(prev, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "pr_lift": round(auprc / prev, 4) if prev > 0 else None,
    }


def global_shap_drugs(model, X: pd.DataFrame, feature_names: list[str], max_rows: int) -> pd.DataFrame:
    from xgboost import DMatrix

    n = min(max_rows, len(X))
    Xs = X.iloc[:n]
    booster = model.get_booster()
    dmat = DMatrix(Xs)
    contribs = booster.predict(dmat, pred_contribs=True)
    mean_abs = np.abs(np.asarray(contribs)[:, :-1]).mean(axis=0)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["is_drug"] = out["feature"].map(drugish)
    return out


def resolve_feature_for_drug(feature_names: list[str], drug: str) -> str | None:
    hits = [f for f in feature_names if feature_matches_drug(f, drug)]
    if not hits:
        return None
    hits = sorted(hits, key=lambda f: (0 if f.startswith("item_") else 1, len(f)))
    return hits[0]


def pair_ie(model, X: pd.DataFrame, f_a: str, f_b: str, n_boot: int = 100) -> dict:
    if f_a not in X.columns or f_b not in X.columns:
        return {"ie": None, "ci_low": None, "ci_high": None}

    def _mean_prob(Xa, Xb):
        Z = X.copy()
        Z[f_a] = Xa
        Z[f_b] = Xb
        return float(model.predict_proba(Z)[:, 1].mean())

    ones = np.ones(len(X))
    zeros = np.zeros(len(X))
    ie = (
        _mean_prob(ones, ones)
        - _mean_prob(ones, zeros)
        - _mean_prob(zeros, ones)
        + _mean_prob(zeros, zeros)
    )

    rng = np.random.default_rng(RANDOM_STATE)
    boots = []
    idx = np.arange(len(X))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        Xs = X.iloc[sample]

        def mp(a_val, b_val, Z=Xs):
            Z2 = Z.copy()
            Z2[f_a] = a_val
            Z2[f_b] = b_val
            return float(model.predict_proba(Z2)[:, 1].mean())

        boots.append(mp(1, 1) - mp(1, 0) - mp(0, 1) + mp(0, 0))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "ie": round(float(ie), 6),
        "ci_low": round(float(lo), 6),
        "ci_high": round(float(hi), 6),
    }


def run_age_band(age_band: str) -> dict:
    age_fname = age_band_to_fname(age_band)
    p = paths_for(age_band)
    primary_dir, sens_dir, comp_dir, ms_band = (
        p["primary"],
        p["sens"],
        p["comp"],
        p["ms_band"],
    )
    ensure_dirs(p)
    print("\n" + "=" * 70)
    print(f"AGE BAND {age_band} ({age_fname})")
    print("=" * 70)

    sync_primary_artifacts(age_band, primary_dir)
    df_train, df_test = load_or_build_frames(age_band, primary_dir)
    print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
    if len(df_train) == 0 or len(df_test) == 0:
        raise RuntimeError(f"Empty train/test for {age_band}")

    util_in_train = sorted(c for c in df_train.columns if is_util_feature(c))
    pd.Series(util_in_train, name="util_feature").to_csv(
        comp_dir / "util_features_dropped.csv", index=False
    )
    print(f"Dropping {len(util_in_train)} utilization features")

    X_tr_u, y_tr, feat_util_free = split_xy(df_train, drop_util=True)
    X_te_u, y_te, _ = split_xy(df_test, drop_util=True)
    X_te_u = X_te_u.reindex(columns=feat_util_free, fill_value=0)

    print(f"Util-free features: {len(feat_util_free)}")
    model_u = fit_xgb(X_tr_u, y_tr)
    joblib.dump(model_u, sens_dir / "models" / "xgboost.joblib")
    (sens_dir / "feature_names.json").write_text(json.dumps(feat_util_free, indent=2))

    metrics_u = holdout_metrics(model_u, X_te_u, y_te)
    (sens_dir / f"{COHORT}_{age_fname}_util_free_holdout_2019_metrics.json").write_text(
        json.dumps(
            {"xgboost": metrics_u, "note": "aggregate util-free refit; n_runs=1"},
            indent=2,
        )
    )
    print("Util-free holdout:", metrics_u)

    X_tr_p, _, feat_primaryish = split_xy(df_train, drop_util=False)
    X_te_p, _, _ = split_xy(df_test, drop_util=False)
    X_te_p = X_te_p.reindex(columns=feat_primaryish, fill_value=0)
    model_p_local = fit_xgb(X_tr_p, y_tr)
    metrics_p_local = holdout_metrics(model_p_local, X_te_p, y_te)
    print("With-util (local retrain) holdout:", metrics_p_local)

    primary_metrics_path = primary_dir / f"{COHORT}_{age_fname}_holdout_2019_metrics.json"
    manuscript_auprc = MANUSCRIPT_AUPRC_BY_BAND.get(age_band)
    baseline = {"source": "local_with_util_retrain", "xgboost": metrics_p_local}
    if primary_metrics_path.exists():
        loaded = json.loads(primary_metrics_path.read_text(encoding="utf-8"))
        if isinstance(loaded.get("xgboost"), dict) and (
            loaded["xgboost"].get("pr_auc") is not None
            or loaded["xgboost"].get("auprc") is not None
        ):
            baseline = {"source": str(primary_metrics_path), "xgboost": loaded["xgboost"]}
            print("Primary holdout JSON:", baseline["xgboost"])
    else:
        xgb_path = primary_dir / "models" / "xgboost.joblib"
        meta_path = (
            primary_dir / "final_model_json" / f"{COHORT}_{age_fname}_best_xgboost_model.json"
        )
        if xgb_path.exists() and meta_path.exists():
            primary = joblib.load(xgb_path)
            feats = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_names", [])
            X_hold = (
                df_test.reindex(columns=feats, fill_value=0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            baseline = {
                "source": "evaluated_primary_model",
                "xgboost": holdout_metrics(primary, X_hold, y_te),
            }
            print("Primary holdout (evaluated):", baseline["xgboost"])
        elif manuscript_auprc is not None:
            baseline["manuscript_table_auprc"] = manuscript_auprc
            print("[INFO] manuscript table AUPRC anchor =", manuscript_auprc)

    shap_u = global_shap_drugs(model_u, X_te_u, feat_util_free, MAX_SHAP_ROWS)
    shap_u.to_csv(comp_dir / "shap_global_util_free.csv", index=False)
    drugs_u = shap_u[shap_u["is_drug"]].head(TOP_K_DRUG).copy()
    drugs_u.to_csv(comp_dir / "top_drug_shap_util_free.csv", index=False)

    primary_shap_candidates = list(
        (PROJECT_ROOT / "7_shap_analysis" / "outputs" / COHORT / age_fname).glob(
            "*shap_global_importance*.csv"
        )
    )
    drugs_p = None
    if primary_shap_candidates:
        shap_p = pd.read_csv(primary_shap_candidates[0])
        feat_col = "feature" if "feature" in shap_p.columns else shap_p.columns[0]
        val_col = (
            "mean_abs_shap"
            if "mean_abs_shap" in shap_p.columns
            else [c for c in shap_p.columns if "shap" in c.lower() or "importance" in c.lower()][0]
        )
        shap_p = shap_p.rename(columns={feat_col: "feature", val_col: "mean_abs_shap"})
        shap_p = shap_p.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_p["rank"] = np.arange(1, len(shap_p) + 1)
        shap_p["is_drug"] = shap_p["feature"].map(drugish)
        drugs_p = shap_p[shap_p["is_drug"]].head(TOP_K_DRUG).copy()
        drugs_p.to_csv(comp_dir / "top_drug_shap_primary.csv", index=False)
    else:
        print("[INFO] Computing SHAP for local with-util retrain as primary comparator.")
        shap_p = global_shap_drugs(model_p_local, X_te_p, feat_primaryish, MAX_SHAP_ROWS)
        shap_p.to_csv(comp_dir / "shap_global_with_util_local.csv", index=False)
        drugs_p = shap_p[shap_p["is_drug"]].head(TOP_K_DRUG).copy()
        drugs_p.to_csv(comp_dir / "top_drug_shap_primary.csv", index=False)

    if drugs_p is not None:
        set_u = set(drugs_u["feature"])
        set_p = set(drugs_p["feature"])
        inter = set_u & set_p
        union = set_u | set_p
        jaccard = len(inter) / len(union) if union else 0.0
        rank_p = dict(zip(drugs_p["feature"], drugs_p["rank"]))
        rank_u = dict(zip(drugs_u["feature"], drugs_u["rank"]))
        overlap_rows = []
        for f in sorted(union):
            overlap_rows.append(
                {
                    "feature": f,
                    "in_primary_top": f in set_p,
                    "in_util_free_top": f in set_u,
                    "rank_primary": rank_p.get(f),
                    "rank_util_free": rank_u.get(f),
                    "rank_shift": (rank_u.get(f) - rank_p.get(f))
                    if f in rank_p and f in rank_u
                    else None,
                }
            )
        pd.DataFrame(overlap_rows).to_csv(comp_dir / "top_drug_overlap.csv", index=False)
        summary_overlap = {
            "top_k": TOP_K_DRUG,
            "jaccard": round(jaccard, 4),
            "n_intersection": len(inter),
            "intersection": sorted(inter),
        }
    else:
        summary_overlap = {"top_k": TOP_K_DRUG, "jaccard": None, "note": "primary SHAP missing"}

    print("Drug SHAP overlap:", summary_overlap)

    pair_rows = []
    for a, b in PUBLISHED_PAIRS:
        fa = resolve_feature_for_drug(feat_util_free, a)
        fb = resolve_feature_for_drug(feat_util_free, b)
        rec = {
            "drug_a": a,
            "drug_b": b,
            "feature_a": fa,
            "feature_b": fb,
            "features_resolved": bool(fa and fb),
        }
        if fa and fb:
            rec.update(pair_ie(model_u, X_te_u, fa, fb, n_boot=100))
            rec["synergy_positive"] = bool(rec["ie"] is not None and rec["ie"] > 0)
        else:
            rec.update(
                {"ie": None, "ci_low": None, "ci_high": None, "synergy_positive": False}
            )
        pair_rows.append(rec)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(comp_dir / "published_pair_ie_util_free.csv", index=False)
    n_persist = int(pair_df["synergy_positive"].fillna(False).sum()) if len(pair_df) else 0
    print(f"Published pairs with positive util-free IE: {n_persist}/{len(PUBLISHED_PAIRS)}")

    trip_rows = []
    for drugs in PUBLISHED_TRIPLETS:
        feats = [resolve_feature_for_drug(feat_util_free, d) for d in drugs]
        ok = all(feats)
        row = {"drugs": " + ".join(drugs), "features": feats, "features_resolved": ok}
        if ok:
            mask = np.ones(len(X_te_u), dtype=bool)
            for f in feats:
                mask &= X_te_u[f].to_numpy() > 0
            n_all = int(mask.sum())
            case_rate = float(y_te[mask].mean()) if n_all else None
            base = float(y_te.mean())
            row.update(
                {
                    "n_all_three": n_all,
                    "case_rate_all_three": None
                    if case_rate is None
                    else round(case_rate, 4),
                    "prevalence": round(base, 4),
                    "lift_vs_prevalence": None
                    if not case_rate or base <= 0
                    else round(case_rate / base, 4),
                }
            )
        trip_rows.append(row)
    pd.DataFrame(trip_rows).to_csv(
        comp_dir / "published_triplet_persistence_util_free.csv", index=False
    )

    local_with_util_auprc = metrics_p_local.get("auprc")
    util_free_auprc = metrics_u.get("auprc")
    auprc_delta_ms = (
        round(float(util_free_auprc) - float(manuscript_auprc), 4)
        if util_free_auprc is not None and manuscript_auprc is not None
        else None
    )
    auprc_delta_local = (
        round(float(util_free_auprc) - float(local_with_util_auprc), 4)
        if util_free_auprc is not None and local_with_util_auprc is not None
        else None
    )

    summary = {
        "cohort": COHORT,
        "age_band": age_band,
        "train_mode": "aggregate_util_free_refit",
        "n_util_features_dropped": len(util_in_train),
        "util_features_dropped": util_in_train,
        "n_features_util_free": len(feat_util_free),
        "pgx_features_excluded": True,
        "pgx_exclusion_reason": (
            "Local PGx join leaves controls near-zero on pgx_has_any_drug while cases "
            "are 100%; excluded so utilization ablation is not confounded by broken "
            "PGx coverage."
        ),
        "manuscript_primary_auprc": manuscript_auprc,
        "local_with_util_auprc": local_with_util_auprc,
        "util_free_auprc": util_free_auprc,
        "util_free_auroc": metrics_u.get("auroc"),
        "util_free_pr_lift": metrics_u.get("pr_lift"),
        "auprc_delta_vs_manuscript": auprc_delta_ms,
        "auprc_delta_vs_local_with_util": auprc_delta_local,
        "primary_metrics_source": baseline.get("source"),
        "drug_shap_overlap": summary_overlap,
        "published_pairs_positive_ie": f"{n_persist}/{len(PUBLISHED_PAIRS)}",
        "outputs": {
            "sens_dir": str(sens_dir),
            "comp_dir": str(comp_dir),
            "ms_band_dir": str(ms_band),
        },
    }
    summary_path = comp_dir / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ms_band / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for name in [
        "util_features_dropped.csv",
        "top_drug_shap_util_free.csv",
        "top_drug_overlap.csv",
        "published_pair_ie_util_free.csv",
        "published_triplet_persistence_util_free.csv",
    ]:
        src = comp_dir / name
        if src.exists():
            (ms_band / name).write_bytes(src.read_bytes())

    print(f"[OK] {age_band}: util_free_auprc={util_free_auprc} pairs={n_persist}/{len(PUBLISHED_PAIRS)}")
    return summary


def parse_age_bands(arg: str | None) -> list[str]:
    if not arg or arg.strip().lower() in {"all", "*"}:
        return list(AGE_BANDS)
    bands = [b.strip() for b in arg.split(",") if b.strip()]
    unknown = [b for b in bands if b not in AGE_BANDS]
    if unknown:
        raise SystemExit(f"Unknown age bands: {unknown}; expected subset of {AGE_BANDS}")
    return bands


def main(age_bands: list[str] | None = None) -> dict:
    bands = list(age_bands) if age_bands is not None else list(AGE_BANDS)
    nb_context = setup_notebook_artifacts(
        notebook_file="3_model_sensitivity.ipynb",
        step_name="6_final_model",
        run_label="ch04_util_free_sensitivity",
    )
    print("GitHub artifact dir:", nb_context.github_dir)
    print("Local output dir:", nb_context.local_output_dir)
    print(f"Cohort={COHORT}; age bands ({len(bands)}): {bands}")

    per_band: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for age_band in bands:
        try:
            per_band[age_band] = run_age_band(age_band)
        except Exception as exc:
            failures[age_band] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {age_band}: {failures[age_band]}")
            traceback.print_exc()

    rollup = {
        "cohort": COHORT,
        "age_bands_requested": bands,
        "age_bands_succeeded": sorted(per_band.keys()),
        "age_bands_failed": failures,
        "n_requested": len(bands),
        "n_succeeded": len(per_band),
        "per_band": {
            k: {
                "util_free_auprc": v.get("util_free_auprc"),
                "util_free_pr_lift": v.get("util_free_pr_lift"),
                "manuscript_primary_auprc": v.get("manuscript_primary_auprc"),
                "auprc_delta_vs_manuscript": v.get("auprc_delta_vs_manuscript"),
                "drug_shap_jaccard": (v.get("drug_shap_overlap") or {}).get("jaccard"),
                "published_pairs_positive_ie": v.get("published_pairs_positive_ie"),
            }
            for k, v in per_band.items()
        },
        "scope_note": (
            "All modeled age bands for non_opioid_ed (consistent with Step-6 "
            "partition-first modeling). AE asked for a single supplemental analysis; "
            "multi-band coverage is an internal consistency choice."
        ),
    }
    # Keep prior filename for 65-74 convenience + rollup filenames.
    MS_DIR.mkdir(parents=True, exist_ok=True)
    (MS_DIR / "sensitivity_summary_all_bands.json").write_text(
        json.dumps(rollup, indent=2), encoding="utf-8"
    )
    if "65-74" in per_band:
        (MS_DIR / "sensitivity_summary.json").write_text(
            json.dumps(per_band["65-74"], indent=2), encoding="utf-8"
        )

    rows = []
    for band, s in per_band.items():
        rows.append(
            {
                "age_band": band,
                "util_free_auprc": s.get("util_free_auprc"),
                "util_free_pr_lift": s.get("util_free_pr_lift"),
                "manuscript_primary_auprc": s.get("manuscript_primary_auprc"),
                "auprc_delta_vs_manuscript": s.get("auprc_delta_vs_manuscript"),
                "drug_shap_jaccard": (s.get("drug_shap_overlap") or {}).get("jaccard"),
                "published_pairs_positive_ie": s.get("published_pairs_positive_ie"),
            }
        )
    if rows:
        pd.DataFrame(rows).sort_values("age_band").to_csv(
            MS_DIR / "sensitivity_auprc_by_age_band.csv", index=False
        )

    gh = github_artifact_path(nb_context, "sensitivity_summary_all_bands.json")
    gh.parent.mkdir(parents=True, exist_ok=True)
    gh.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    local_artifact_path(nb_context, "sensitivity_summary_all_bands.json").write_text(
        json.dumps(rollup, indent=2), encoding="utf-8"
    )

    print("\n=== ROLLUP ===")
    print(json.dumps(rollup, indent=2))
    print("Journal SSOT (CTS-2026-0235R2):", MS_DIR)
    return rollup


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CH4 util-free sensitivity (all age bands)")
    ap.add_argument(
        "--age-bands",
        default="all",
        help="Comma-separated age bands or 'all' (default: all modeled bands)",
    )
    # parse_known_args: tolerate Jupyter / ipykernel launcher argv when run via runpy
    args, _unknown = ap.parse_known_args()
    main(parse_age_bands(args.age_bands))
