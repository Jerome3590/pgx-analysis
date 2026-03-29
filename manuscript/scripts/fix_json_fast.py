"""
Fast fix for all four JSON data files — covers ALL bands for BOTH cohorts.
Uses S3 Select for train CSV counts (fast, no full download).
Uses model_test parquet column-projection for test counts.
Merges with existing brier_ici_results.json for Brier/ICI.
"""
import boto3, io, json
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

COHORTS = ["opioid_ed", "non_opioid_ed"]
BANDS   = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]


# ── helpers ───────────────────────────────────────────────────────────────────
def head(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def read_parquet_cols(key, cols):
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(data), columns=cols)
    except Exception:
        return None


def s3_select_counts(key):
    """Count target==1 and target==0 via S3 Select (no full download)."""
    results = {}
    for tgt in ("1", "0"):
        try:
            resp = s3.select_object_content(
                Bucket=BUCKET, Key=key,
                ExpressionType="SQL",
                Expression=f"SELECT COUNT(*) FROM S3Object WHERE s.target = '{tgt}'",
                InputSerialization={"CSV": {"FileHeaderInfo": "USE"},
                                    "CompressionType": "NONE"},
                OutputSerialization={"JSON": {}},
            )
            out = "".join(
                e["Records"]["Payload"].decode()
                for e in resp["Payload"] if "Records" in e
            )
            results[tgt] = int(json.loads(out.strip())["_1"])
        except Exception as e:
            print(f"    S3Select error for target={tgt}: {e}")
            results[tgt] = 0
    return results.get("1", 0), results.get("0", 0)


def read_csv_s3(key):
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


# =============================================================================
# 1. cohort_counts.json  — combined train+test unique patients per band
# =============================================================================
print("=" * 70)
print("1. cohort_counts.json  (train+test from final_model parquets)")
print("=" * 70)

cohort_counts = {}
for cohort in COHORTS:
    cohort_counts[cohort] = {}
    for band in BANDS:
        # Combine train + test parquets (both confirmed to exist for study bands)
        cases_total = 0
        ctrl_total  = 0
        pkeys_seen  = set()
        found_any   = False

        for split in ("model_train", "model_test"):
            key = f"gold/final_model/{cohort}/{band}/inputs/{split}/final_features.parquet"
            df  = read_parquet_cols(key, ["mi_person_key", "target"])
            if df is None:
                continue
            found_any = True
            # Deduplicate mi_person_key across splits
            df = df[~df["mi_person_key"].isin(pkeys_seen)]
            pkeys_seen.update(df["mi_person_key"].tolist())
            cases_total += int((df["target"] == 1).sum())
            ctrl_total  += int((df["target"] == 0).sum())

        if not found_any:
            print(f"  SKIP {cohort}/{band}: no parquet found")
            continue

        cohort_counts[cohort][band] = {"cases": cases_total, "controls": ctrl_total}
        print(f"  {cohort:15s} | {band:7s} | cases={cases_total:7,} | controls={ctrl_total:8,}")

with open("cohort_counts.json", "w") as f:
    json.dump(cohort_counts, f, indent=2)
print("Saved cohort_counts.json\n")


# =============================================================================
# 2. cohort_counts_train.json  — train-only via S3 Select on CSV
# =============================================================================
print("=" * 70)
print("2. cohort_counts_train.json  (train CSV via S3 Select)")
print("=" * 70)

cohort_counts_train = {}
for cohort in COHORTS:
    cohort_counts_train[cohort] = {}
    for band in BANDS:
        ab      = band.replace("-", "_")
        key_csv = (f"gold/final_model/{cohort}/{band}/"
                   f"{cohort}_{ab}_train_final_features_no_leakage.csv")
        key_pq  = (f"gold/final_model/{cohort}/{band}/inputs/"
                   f"model_train/final_features.parquet")

        # Try CSV via S3 Select first (fastest)
        if head(key_csv):
            cases, ctrl = s3_select_counts(key_csv)
            if cases + ctrl > 0:
                cohort_counts_train[cohort][band] = {
                    "total": cases + ctrl, "cases": cases, "controls": ctrl}
                print(f"  {cohort:15s} | {band:7s} | cases={cases:7,} | controls={ctrl:8,} [CSV]")
                continue

        # Fallback: train parquet
        df = read_parquet_cols(key_pq, ["target"])
        if df is not None:
            cases = int((df["target"] == 1).sum())
            ctrl  = int((df["target"] == 0).sum())
            cohort_counts_train[cohort][band] = {
                "total": cases + ctrl, "cases": cases, "controls": ctrl}
            print(f"  {cohort:15s} | {band:7s} | cases={cases:7,} | controls={ctrl:8,} [PQ]")
        else:
            print(f"  SKIP {cohort}/{band}: no train data")

with open("cohort_counts_train.json", "w") as f:
    json.dump(cohort_counts_train, f, indent=2)
print("Saved cohort_counts_train.json\n")


# =============================================================================
# 3. cohort_counts_test.json  — 2019 holdout counts
# =============================================================================
print("=" * 70)
print("3. cohort_counts_test.json  (model_test parquet)")
print("=" * 70)

cohort_counts_test = {}
for cohort in COHORTS:
    cohort_counts_test[cohort] = {}
    for band in BANDS:
        key = f"gold/final_model/{cohort}/{band}/inputs/model_test/final_features.parquet"
        df  = read_parquet_cols(key, ["target"])
        if df is None:
            print(f"  SKIP {cohort}/{band}")
            continue
        cases = int((df["target"] == 1).sum())
        ctrl  = int((df["target"] == 0).sum())
        cohort_counts_test[cohort][band] = {
            "total": cases + ctrl, "cases": cases, "controls": ctrl}
        print(f"  {cohort:15s} | {band:7s} | cases={cases:7,} | controls={ctrl:8,}")

with open("cohort_counts_test.json", "w") as f:
    json.dump(cohort_counts_test, f, indent=2)
print("Saved cohort_counts_test.json\n")


# =============================================================================
# 4. brier_ici_results.json  — preserve existing; add any missing bands
#    (full recomputation is too slow here; see compute_brier_ici.py separately)
# =============================================================================
print("=" * 70)
print("4. brier_ici_results.json  (extend existing with missing band entries)")
print("=" * 70)

try:
    with open("brier_ici_results.json") as f:
        brier_ici = json.load(f)
except Exception:
    brier_ici = {}

# Add skeleton entries for any bands that have test data but no Brier/ICI yet
for cohort in COHORTS:
    if cohort not in brier_ici:
        brier_ici[cohort] = {}
    for band in BANDS:
        if band in brier_ici[cohort]:
            continue  # already computed
        # Check if test parquet exists; if so mark as needs_compute
        key = f"gold/final_model/{cohort}/{band}/inputs/model_test/final_features.parquet"
        df  = read_parquet_cols(key, ["target"])
        if df is None:
            continue
        n = len(df)
        # Mark as placeholder — rerun compute_brier_ici.py to fill
        brier_ici[cohort][band] = {
            "brier": None, "ici": None, "n_test": n, "model": "needs_compute"}
        print(f"  Added placeholder {cohort}/{band}  (n_test={n:,})")

# Print current status
print("\nCurrent brier_ici_results.json coverage:")
for cohort in COHORTS:
    bands_with_data = [b for b, v in brier_ici.get(cohort, {}).items()
                       if v.get("brier") is not None]
    bands_placeholder = [b for b, v in brier_ici.get(cohort, {}).items()
                         if v.get("brier") is None]
    print(f"  {cohort}: computed={bands_with_data}, placeholder={bands_placeholder}")

with open("brier_ici_results.json", "w") as f:
    json.dump(brier_ici, f, indent=2)
print("Saved brier_ici_results.json\n")


# =============================================================================
# 5. ffa_manuscript_data.json  — extend to all bands/cohorts
# =============================================================================
print("=" * 70)
print("5. ffa_manuscript_data.json  (FFA causal factors, all bands)")
print("=" * 70)

try:
    with open("ffa_manuscript_data.json") as f:
        ffa_data = json.load(f)
except Exception:
    ffa_data = {}

for cohort in COHORTS:
    if cohort not in ffa_data:
        ffa_data[cohort] = {}
    for band in BANDS:
        if band in ffa_data[cohort]:
            continue  # already populated
        ab  = band.replace("-", "_")
        key = (f"gold/ffa_analysis/{cohort}/{ab}/bin_models/low/"
               f"ffa_causal_factors.csv")
        df  = read_csv_s3(key)
        if df is None:
            print(f"  SKIP {cohort}/{band}: ffa_causal_factors.csv not found")
            continue

        total_rules = (int(df["total_rules"].iloc[0])
                       if "total_rules" in df.columns else 0)
        n_features  = len(df)
        meta_feats  = {"n_events", "pgx_num_drugs", "pgx_num_cpic_drugs"}
        feat_col    = next((c for c in ("feature", "feature_name")
                            if c in df.columns), df.columns[0])
        score_col   = next((c for c in ("causal_responsibility",
                                        "causal_score", "cr")
                            if c in df.columns), None)
        if score_col is None:
            continue

        drugs = df[~df[feat_col].isin(meta_feats)].copy()
        top   = drugs.nlargest(5, score_col)

        ffa_data[cohort][band] = {
            "n_features":    n_features,
            "total_rules":   total_rules,
            "top_drugs":     top[feat_col].str.replace(
                "item_drug_", "", regex=False).tolist()[:5],
            "top_cr":        top[score_col].tolist()[:5],
            "top_rule_freq": (top["rule_frequency"].tolist()[:5]
                              if "rule_frequency" in top.columns else []),
        }
        print(f"  Added {cohort}/{band}: n_feat={n_features}, "
              f"rules={total_rules:,}")

with open("ffa_manuscript_data.json", "w") as f:
    json.dump(ffa_data, f, indent=2, default=str)
print("Saved ffa_manuscript_data.json\n")

print("=" * 70)
print("ALL JSON FILES UPDATED")
print("=" * 70)
print("\nSummary:")
for name, data in [("cohort_counts",       cohort_counts),
                   ("cohort_counts_train",  cohort_counts_train),
                   ("cohort_counts_test",   cohort_counts_test),
                   ("brier_ici",            brier_ici)]:
    for cohort in COHORTS:
        bands = list(data.get(cohort, {}).keys())
        print(f"  {name}[{cohort}]: {len(bands)} bands → {bands}")
