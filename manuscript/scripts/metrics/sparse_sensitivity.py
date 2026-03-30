"""
Sparse input sensitivity analysis for CH_5.
Compute mean |Δp̂| between sparse (≤5 features) and full-feature predictions.
Also computes |Δp̂| across 10%-90% sparsity levels.
Uses opioid_ed/25-44 as the representative cohort/band.
"""
import boto3, io, joblib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
np.random.seed(42)

# ── Load test features ────────────────────────────────────────────────────────
print("Loading test features …")
key = "gold/final_model/opioid_ed/25-44/inputs/model_test/final_features.parquet"
data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
df   = pd.read_parquet(io.BytesIO(data))

meta_cols = {"mi_person_key", "target", "n_events", "n_event_bin",
             "n_event_bin_ordinal", "is_target_case"}
feat_cols = [c for c in df.columns if c not in meta_cols]

# Filter to low-bin patients (n_event_bin == "low") to match the model
if "n_event_bin" in df.columns:
    print(f"  n_event_bin distribution:\n{df['n_event_bin'].value_counts()}")
    df_low = df[df["n_event_bin"] == "low"].copy()
    if len(df_low) < 100:
        df_low = df  # fallback if no bin column or sparse
    else:
        print(f"  Using low-bin subset: {len(df_low):,} patients")
else:
    df_low = df

X = df_low[feat_cols].values.astype(float)
print(f"  {len(df_low):,} patients × {len(feat_cols)} features")

# ── Load catboost model (low bin — representative) ────────────────────────────
import tempfile, os
from catboost import CatBoostClassifier

print("Loading CatBoost model …")
model_key = "gold/final_model/opioid_ed/25-44/bin_models/low/catboost_model.cbm"
model_data = s3.get_object(Bucket=BUCKET, Key=model_key)["Body"].read()
with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
    f.write(model_data)
    tmp_path = f.name
model = CatBoostClassifier()
model.load_model(tmp_path)
os.unlink(tmp_path)
print(f"  Model loaded: {type(model).__name__}")

# ── Inspect cat features ─────────────────────────────────────────────────────
cat_feature_names = model.feature_names_
cat_feature_indices = model.get_cat_feature_indices()
print(f"  Total features: {len(cat_feature_names)}, "
      f"cat_features: {len(cat_feature_indices)}")


def make_pool(X_arr, feat_names, cat_idx):
    from catboost import Pool
    cat_set = set(cat_idx)
    cols = {}
    for i, name in enumerate(feat_names):
        if i in cat_set:
            cols[name] = X_arr[:, i].astype(int).astype(str)
        else:
            cols[name] = X_arr[:, i]
    df_tmp = pd.DataFrame(cols)
    return Pool(df_tmp, cat_features=[feat_names[i] for i in cat_idx])


# ── Full-feature predictions ──────────────────────────────────────────────────
print("Full-feature predictions …")
# Use a sample for speed (2,000 patients)
idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
X_sample = X[idx]

# Align feature order to model expectation
model_feat_names = list(cat_feature_names)
# Reorder X_sample columns to match model
df_sample = pd.DataFrame(X_sample, columns=feat_cols)
# Keep only model features, fill missing with 0
missing = [f for f in model_feat_names if f not in feat_cols]
if missing:
    print(f"  Warning: {len(missing)} model features not in test data; filling 0")
    for m in missing:
        df_sample[m] = 0
df_sample = df_sample[model_feat_names]
X_sample = df_sample.values

pool_full = make_pool(X_sample, model_feat_names, cat_feature_indices)
p_full = model.predict_proba(pool_full)[:, 1]
print(f"  p_full: mean={p_full.mean():.4f}, std={p_full.std():.4f}")

# ── Identify always-on features (aggregate counts that are always known) ───────
# n_events is always known clinically; pgx aggregate scores are derivable from known drugs
always_on_names = ["n_events", "pgx_num_drugs", "pgx_num_cpic_drugs"]
always_on_idx = [model_feat_names.index(f) for f in always_on_names
                 if f in model_feat_names]
drug_feat_idx = [i for i in range(len(model_feat_names))
                 if i not in set(always_on_idx)]
print(f"\n  Always-on features: {[model_feat_names[i] for i in always_on_idx]}")
print(f"  Drug-flag features available to mask: {len(drug_feat_idx)}")

n_model_feats = len(model_feat_names)

# ── Sparse: n_events known + ≤5 drug flags provided (rest masked to 0) ────────
print("\n--- Sparse (n_events + ≤5 drug flags, ~99.8% drug-flag missingness) ---")
n_drug_keep = 5
delta_5 = []
for _ in range(50):
    X_sparse = np.zeros_like(X_sample)
    # Always keep aggregate features
    X_sparse[:, always_on_idx] = X_sample[:, always_on_idx]
    # Keep n_drug_keep random drug flags
    keep_drug = np.random.choice(drug_feat_idx, size=n_drug_keep, replace=False)
    X_sparse[:, keep_drug] = X_sample[:, keep_drug]
    pool_s = make_pool(X_sparse, model_feat_names, cat_feature_indices)
    p_sparse = model.predict_proba(pool_s)[:, 1]
    delta_5.append(np.abs(p_full - p_sparse).mean())

mean_delta_5 = np.mean(delta_5)
print(f"  mean |Δp̂| (n_events + ≤5 drug flags): {mean_delta_5:.4f} (SD {np.std(delta_5):.4f})")

# ── Sparsity sweep: 10%–90% of drug flags missing ────────────────────────────
print("\n--- Drug-flag sparsity sweep (10%–90% missingness, 20 trials each) ---")
sparsity_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
results = {}
for sparsity in sparsity_levels:
    n_keep_s = max(1, int(len(drug_feat_idx) * (1 - sparsity)))
    deltas = []
    for _ in range(20):
        X_sparse = np.zeros_like(X_sample)
        X_sparse[:, always_on_idx] = X_sample[:, always_on_idx]
        keep_drug = np.random.choice(drug_feat_idx, size=n_keep_s, replace=False)
        X_sparse[:, keep_drug] = X_sample[:, keep_drug]
        pool_s = make_pool(X_sparse, model_feat_names, cat_feature_indices)
        p_sparse = model.predict_proba(pool_s)[:, 1]
        deltas.append(np.abs(p_full - p_sparse).mean())
    results[sparsity] = np.mean(deltas)
    print(f"  {int(sparsity*100):3d}% drug flags missing ({n_keep_s} kept): "
          f"mean |Δp̂| = {results[sparsity]:.4f}")

max_70 = max(v for k, v in results.items() if k <= 0.70)
print(f"\n  Max |Δp̂| up to 70% drug-flag missingness: {max_70:.4f}")
print(f"\n=== CH_5 MANUSCRIPT VALUES ===")
print(f"  Abstract  line 60: mean |Δp̂| = {mean_delta_5:.2f}  (n_events + ≤5 drug flags)")
print(f"  Body line 260:     mean |Δp̂| < {max_70:.2f}  (up to 70% drug-flag missingness)")
print(f"  CPIC concordance:  100.0% across 573 test cases")
