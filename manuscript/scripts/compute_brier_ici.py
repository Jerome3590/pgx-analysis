"""
Compute Brier score and ICI for all manuscript cohort/age_band combinations.
Uses: gold/final_model/{cohort}/{age_band}/inputs/model_test/final_features.parquet
      gold/final_model/{cohort}/{age_band}/bin_models/low/{model}.joblib
"""
import boto3, io, joblib, json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BASE   = "gold/final_model"


def get_object(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def load_parquet(key):
    return pd.read_parquet(io.BytesIO(get_object(key)))


def load_model(key):
    return joblib.load(io.BytesIO(get_object(key)))


def ici(y_true, y_prob, n_bins=10):
    """Integrated Calibration Index via calibration curve (trapezoid integral)."""
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return float(np.mean(np.abs(frac_pos - mean_pred)))
    except Exception:
        return float("nan")


configs = [
    ("opioid_ed",     ["0_12", "13_24", "25_44", "45_54", "55_64", "65_74", "75_84", "85_114"], "CatBoost"),
    ("non_opioid_ed", ["0_12", "13_24", "25_44", "45_54", "55_64", "65_74", "75_84", "85_114"], "CatBoost"),
]

print(f"{'Cohort':<15s} {'Band':>7s} {'Model':>10s}  {'Brier':>7s}  {'ICI':>7s}  {'N_test':>7s}")
print("-" * 62)

all_results = {}
for cohort, bands, default_model in configs:
    all_results[cohort] = {}
    for ab in bands:
        band = ab.replace("_", "-")
        ab_s = ab.replace("_", "_")

        # Load test features (ALL bins — n_event_bin column used to route)
        test_key = f"{BASE}/{cohort}/{band}/inputs/model_test/final_features.parquet"
        try:
            test_df = load_parquet(test_key)
        except Exception as e:
            print(f"  SKIP {cohort}/{band}: test features missing ({e})")
            continue

        META_COLS = {"target", "mi_person_key", "age_band", "cohort_name"}

        BINS = ("low", "medium", "high", "extreme")
        all_y_true, all_y_prob = [], []

        import tempfile, os
        from catboost import CatBoostClassifier

        # Pre-load feature names from low-bin .cbm (same feature set across all bins)
        known_feats = None
        for cbm_name in (f"{cohort}_{ab_s}_best_catboost_model.cbm", "catboost_model.cbm"):
            try:
                cbm_data = get_object(f"{BASE}/{cohort}/{band}/bin_models/low/{cbm_name}")
                with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                    tmp.write(cbm_data); tmp_path = tmp.name
                _ref = CatBoostClassifier(); _ref.load_model(tmp_path); os.unlink(tmp_path)
                known_feats = _ref.feature_names_
                break
            except Exception:
                pass

        for bin_name in BINS:
            # Filter test rows to this bin
            if "n_event_bin" in test_df.columns:
                bin_df = test_df[test_df["n_event_bin"] == bin_name].copy()
            else:
                bin_df = test_df.copy() if bin_name == "low" else pd.DataFrame()

            if len(bin_df) == 0:
                continue

            y_bin = bin_df["target"].values

            # Try .cbm first, fall back to .joblib using known_feats for alignment
            model = None
            model_feats = known_feats
            for cbm_name in (f"{cohort}_{ab_s}_best_catboost_model.cbm", "catboost_model.cbm"):
                try:
                    cbm_data = get_object(f"{BASE}/{cohort}/{band}/bin_models/{bin_name}/{cbm_name}")
                    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                        tmp.write(cbm_data); tmp_path = tmp.name
                    model = CatBoostClassifier(); model.load_model(tmp_path)
                    model_feats = model.feature_names_
                    os.unlink(tmp_path)
                    break
                except Exception:
                    pass

            if model is None and known_feats:
                # Fall back to .joblib with feature alignment
                for jname in ("catboost.joblib", "xgboost.joblib"):
                    try:
                        model = load_model(f"{BASE}/{cohort}/{band}/bin_models/{bin_name}/{jname}")
                        break
                    except Exception:
                        pass

            if model is None:
                continue

            # Align features
            X = bin_df.drop(columns=[c for c in (META_COLS | {"n_event_bin"}) if c in bin_df.columns])
            if model_feats:
                for col in model_feats:
                    if col not in X.columns:
                        X[col] = 0
                X = X[model_feats]

            try:
                y_prob_bin = model.predict_proba(X)[:, 1]
                all_y_true.extend(y_bin.tolist())
                all_y_prob.extend(y_prob_bin.tolist())
            except Exception:
                pass

        if not all_y_true:
            print(f"  SKIP {cohort}/{band}: no predictions generated")
            continue

        y_true_all = np.array(all_y_true)
        y_prob_all = np.array(all_y_prob)
        brier = brier_score_loss(y_true_all, y_prob_all)
        ici_val = ici(y_true_all, y_prob_all)
        n = len(y_true_all)
        mname = "catboost_per_bin"

        print(f"{cohort:<15s} {band:>7s} {mname:>16s}  {brier:7.4f}  {ici_val:7.4f}  {n:7d}")
        all_results[cohort][band] = {"brier": round(brier, 4), "ici": round(ici_val, 4),
                                     "n_test": n, "model": mname}

print("\n=== Summary for manuscript ===")
print("opioid_ed (CH_2, CH_3) Brier range:",
      min(v["brier"] for v in all_results.get("opioid_ed", {}).values()),
      "–",
      max(v["brier"] for v in all_results.get("opioid_ed", {}).values()))
print("opioid_ed ICI range:",
      min(v["ici"] for v in all_results.get("opioid_ed", {}).values()),
      "–",
      max(v["ici"] for v in all_results.get("opioid_ed", {}).values()))
print("non_opioid_ed (CH_4) Brier range:",
      min(v["brier"] for v in all_results.get("non_opioid_ed", {}).values()),
      "–",
      max(v["brier"] for v in all_results.get("non_opioid_ed", {}).values()))
print("non_opioid_ed ICI range:",
      min(v["ici"] for v in all_results.get("non_opioid_ed", {}).values()),
      "–",
      max(v["ici"] for v in all_results.get("non_opioid_ed", {}).values()))

# Save
import json as _json
with open("brier_ici_results.json", "w") as f:
    _json.dump(all_results, f, indent=2)
print("\nSaved brier_ici_results.json")
