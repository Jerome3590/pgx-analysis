"""
Fix all four JSON data files to cover ALL age bands for BOTH cohorts.
Generates:
  cohort_counts.json          — unique patients (cases + controls) per band from model_events
  cohort_counts_train.json    — matched train cohort (2016-2018) case/control counts
  brier_ici_results.json      — Brier score + ICI per band from model_test predictions
  ffa_manuscript_data.json    — FFA causal factors summary per band (low bin)

Cohorts:  opioid_ed     → 8 bands: 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114
          non_opioid_ed → 8 bands: same
"""
import boto3, io, json, tempfile, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

COHORTS = ["opioid_ed", "non_opioid_ed"]
BANDS   = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
BINS    = ["low", "medium", "high", "extreme"]


def s3_read_parquet(key, columns=None):
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(data), columns=columns)
    except Exception:
        return None


def s3_read_csv(key):
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


def s3_head(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


# ── ICI helper ────────────────────────────────────────────────────────────────
def ici(y_true, y_prob, n_bins=10):
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return float(np.mean(np.abs(frac_pos - mean_pred)))
    except Exception:
        return float("nan")


# =============================================================================
# 1. cohort_counts.json — unique patients from model_events.parquet
# =============================================================================
print("\n" + "=" * 70)
print("1. cohort_counts.json  (unique patients from cohorts_model_data)")
print("=" * 70)

cohort_counts = {}
for cohort in COHORTS:
    cohort_counts[cohort] = {}
    for band in BANDS:
        key = (f"gold/cohorts_model_data/cohort_name={cohort}/"
               f"age_band={band}/model_events.parquet")
        df = s3_read_parquet(key, columns=["mi_person_key", "target"])
        if df is None:
            print(f"  SKIP {cohort}/{band}: not found")
            continue
        pts   = df.drop_duplicates("mi_person_key")
        cases = int((pts["target"] == 1).sum())
        ctrl  = int((pts["target"] == 0).sum())
        cohort_counts[cohort][band] = {"cases": cases, "controls": ctrl}
        print(f"  {cohort:15s} | {band:7s} | cases={cases:7,} | controls={ctrl:8,}")

with open("cohort_counts.json", "w") as f:
    json.dump(cohort_counts, f, indent=2)
print("  Saved cohort_counts.json")


# =============================================================================
# 2. cohort_counts_train.json — from final_features train parquet (all bands)
# =============================================================================
print("\n" + "=" * 70)
print("2. cohort_counts_train.json  (model train parquet)")
print("=" * 70)

cohort_counts_train = {}
for cohort in COHORTS:
    cohort_counts_train[cohort] = {}
    for band in BANDS:
        ab = band.replace("-", "_")
        # Try parquet first, then CSV
        key_pq  = f"gold/final_model/{cohort}/{band}/inputs/model_train/final_features.parquet"
        key_csv = f"gold/final_model/{cohort}/{band}/{cohort}_{ab}_train_final_features_no_leakage.csv"
        df = s3_read_parquet(key_pq, columns=["mi_person_key", "target"])
        if df is None:
            df = s3_read_csv(key_csv)
            if df is not None:
                df = df[["target"]].copy() if "target" in df.columns else None
        if df is None:
            print(f"  SKIP {cohort}/{band}: no train data")
            continue
        cases = int((df["target"] == 1).sum())
        ctrl  = int((df["target"] == 0).sum())
        cohort_counts_train[cohort][band] = {
            "total": cases + ctrl, "cases": cases, "controls": ctrl}
        print(f"  {cohort:15s} | {band:7s} | cases={cases:7,} | controls={ctrl:8,}")

with open("cohort_counts_train.json", "w") as f:
    json.dump(cohort_counts_train, f, indent=2)
print("  Saved cohort_counts_train.json")


# =============================================================================
# 3. cohort_counts_test.json — from final_features test parquet (all bands)
# =============================================================================
print("\n" + "=" * 70)
print("3. cohort_counts_test.json  (model test parquet)")
print("=" * 70)

cohort_counts_test = {}
for cohort in COHORTS:
    cohort_counts_test[cohort] = {}
    for band in BANDS:
        key = f"gold/final_model/{cohort}/{band}/inputs/model_test/final_features.parquet"
        df  = s3_read_parquet(key, columns=["mi_person_key", "target"])
        if df is None:
            print(f"  SKIP {cohort}/{band}: not found")
            continue
        cases = int((df["target"] == 1).sum())
        ctrl  = int((df["target"] == 0).sum())
        cohort_counts_test[cohort][band] = {
            "total": cases + ctrl, "cases": cases, "controls": ctrl}
        print(f"  {cohort:15s} | {band:7s} | cases={cases:7,} | controls={ctrl:8,}")

with open("cohort_counts_test.json", "w") as f:
    json.dump(cohort_counts_test, f, indent=2)
print("  Saved cohort_counts_test.json")


# =============================================================================
# 4. brier_ici_results.json — Brier + ICI from model_test predictions
# =============================================================================
print("\n" + "=" * 70)
print("4. brier_ici_results.json  (Brier + ICI, all bands)")
print("=" * 70)

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("  WARNING: catboost not available; Brier/ICI skipped")

META = {"target", "mi_person_key", "age_band", "cohort_name",
        "n_event_bin", "n_event_bin_ordinal", "is_target_case"}

brier_ici = {}
for cohort in COHORTS:
    brier_ici[cohort] = {}
    for band in BANDS:
        ab = band.replace("-", "_")
        test_key = f"gold/final_model/{cohort}/{band}/inputs/model_test/final_features.parquet"
        test_df  = s3_read_parquet(test_key)
        if test_df is None or not HAS_CATBOOST:
            if not HAS_CATBOOST and test_df is not None:
                # Still record cohort sizes even without model
                pass
            print(f"  SKIP {cohort}/{band}")
            continue

        all_y_true, all_y_prob = [], []

        # Load reference feature names from first available low-bin model
        known_feats = None
        for cbm_name in (f"{cohort}_{ab}_best_catboost_model.cbm", "catboost_model.cbm"):
            cbm_key = f"gold/final_model/{cohort}/{band}/bin_models/low/{cbm_name}"
            if not s3_head(cbm_key):
                continue
            try:
                cbm_data = s3.get_object(Bucket=BUCKET, Key=cbm_key)["Body"].read()
                with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                    tmp.write(cbm_data); tp = tmp.name
                _ref = CatBoostClassifier(); _ref.load_model(tp); os.unlink(tp)
                known_feats = _ref.feature_names_
                break
            except Exception:
                pass

        for bin_name in BINS:
            if "n_event_bin" in test_df.columns:
                bin_df = test_df[test_df["n_event_bin"] == bin_name].copy()
            else:
                bin_df = test_df.copy() if bin_name == "low" else pd.DataFrame()
            if len(bin_df) == 0:
                continue

            y_bin = bin_df["target"].values
            model = None
            mfeats = known_feats

            for cbm_name in (f"{cohort}_{ab}_best_catboost_model.cbm", "catboost_model.cbm"):
                cbm_key = f"gold/final_model/{cohort}/{band}/bin_models/{bin_name}/{cbm_name}"
                if not s3_head(cbm_key):
                    continue
                try:
                    cbm_data = s3.get_object(Bucket=BUCKET, Key=cbm_key)["Body"].read()
                    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
                        tmp.write(cbm_data); tp = tmp.name
                    model = CatBoostClassifier(); model.load_model(tp)
                    mfeats = model.feature_names_
                    os.unlink(tp)
                    break
                except Exception:
                    pass

            if model is None:
                continue

            X = bin_df.drop(columns=[c for c in META if c in bin_df.columns])
            if mfeats:
                for col in mfeats:
                    if col not in X.columns:
                        X[col] = 0
                X = X[mfeats]

            try:
                y_prob = model.predict_proba(X)[:, 1]
                all_y_true.extend(y_bin.tolist())
                all_y_prob.extend(y_prob.tolist())
            except Exception:
                pass

        if not all_y_true:
            print(f"  SKIP {cohort}/{band}: no predictions")
            continue

        yt = np.array(all_y_true)
        yp = np.array(all_y_prob)
        b  = round(brier_score_loss(yt, yp), 4)
        i  = round(ici(yt, yp), 4)
        n  = len(yt)
        brier_ici[cohort][band] = {"brier": b, "ici": i, "n_test": n,
                                    "model": "catboost_per_bin"}
        print(f"  {cohort:15s} | {band:7s} | brier={b:.4f} | ici={i:.4f} | n={n:,}")

with open("brier_ici_results.json", "w") as f:
    json.dump(brier_ici, f, indent=2)
print("  Saved brier_ici_results.json")


# =============================================================================
# 5. ffa_manuscript_data.json — FFA causal factors (all bands, low bin)
# =============================================================================
print("\n" + "=" * 70)
print("5. ffa_manuscript_data.json  (FFA causal factors, all bands)")
print("=" * 70)

ffa_data = {}
for cohort in COHORTS:
    ffa_data[cohort] = {}
    for band in BANDS:
        ab = band.replace("-", "_")
        key = f"gold/ffa_analysis/{cohort}/{ab}/bin_models/low/ffa_causal_factors.csv"
        df  = s3_read_csv(key)
        if df is None:
            print(f"  SKIP {cohort}/{band}: ffa_causal_factors.csv not found")
            continue

        total_rules = int(df["total_rules"].iloc[0]) if "total_rules" in df.columns else 0
        n_features  = len(df)

        meta_feats = {"n_events", "pgx_num_drugs", "pgx_num_cpic_drugs"}
        feat_col = next((c for c in ("feature", "feature_name") if c in df.columns),
                        df.columns[0])
        score_col = next((c for c in ("causal_responsibility", "causal_score", "cr")
                          if c in df.columns), None)
        if score_col is None:
            print(f"  SKIP {cohort}/{band}: no causal_responsibility column")
            continue

        drugs = df[~df[feat_col].isin(meta_feats)].copy()
        top   = drugs.nlargest(5, score_col)

        ffa_data[cohort][band] = {
            "n_features":   n_features,
            "total_rules":  total_rules,
            "top_drugs":    top[feat_col].str.replace("item_drug_", "", regex=False).tolist()[:5],
            "top_cr":       top[score_col].tolist()[:5],
            "top_rule_freq": (top["rule_frequency"].tolist()[:5]
                              if "rule_frequency" in top.columns else []),
        }
        print(f"  {cohort:15s} | {band:7s} | n_feat={n_features:3d} | "
              f"rules={total_rules:,} | top={top[feat_col].iloc[0] if len(top) else 'n/a'}")

with open("ffa_manuscript_data.json", "w") as f:
    json.dump(ffa_data, f, indent=2, default=str)
print("  Saved ffa_manuscript_data.json")

print("\n=== ALL JSON FILES UPDATED ===")
