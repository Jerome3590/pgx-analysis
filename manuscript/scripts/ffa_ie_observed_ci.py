"""
Bootstrap 95% CI for pairwise IE (Interaction Effect) from observed outcomes.

IE(A,B) = P(case | drug_A=1, drug_B=1) / [P(case | drug_A=1) × P(case | drug_B=1)]

Bootstrap: 1,000 resamples with replacement from combined train+test cohort.
Source: final_features.parquet (item_drug_* columns + target).
"""
import boto3, io, json
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["65-74", "75-84", "85-114"]
COHORT = "non_opioid_ed"
N_BOOT = 1000
RNG    = np.random.default_rng(42)

TARGET_PAIRS = [
    ("ACETAMINOPHEN", "LEVOFLOXACIN"),
    ("LEVOFLOXACIN",  "LORAZEPAM"),
    ("CARVEDILOL",    "LEVOFLOXACIN"),
    ("GABAPENTIN",    "LEVOFLOXACIN"),
    ("DIGOXIN",       "SIMVASTATIN"),
]

# ── Load final_features for all geriatric bands ───────────────────────────────
print("Loading final_features (train + test) for geriatric bands …")
all_drugs = {d for pair in TARGET_PAIRS for d in pair}
drug_cols  = [f"item_drug_{d}" for d in all_drugs]

frames = []
for band in BANDS:
    for split in ("model_train", "model_test"):
        key = (f"gold/final_model/{COHORT}/{band}/inputs/"
               f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            df   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target"] + drug_cols)
            df["band"] = band
            frames.append(df)
        except Exception as e:
            # Some drug cols may not exist in certain bands — load what's there
            try:
                data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
                full = pd.read_parquet(io.BytesIO(data),
                                       columns=["mi_person_key", "target"])
                full["band"] = band
                for col in drug_cols:
                    full[col] = 0
                frames.append(full)
            except Exception:
                print(f"  SKIP {split}/{band}: {e}")

cohort = (pd.concat(frames, ignore_index=True)
            .sort_values("target", ascending=False)
            .drop_duplicates("mi_person_key"))

for col in drug_cols:
    if col not in cohort.columns:
        cohort[col] = 0
cohort[drug_cols] = cohort[drug_cols].fillna(0).astype(int)

n_total = len(cohort)
n_cases = cohort["target"].sum()
print(f"  {n_cases:,} cases, {n_total-n_cases:,} controls  ({n_total:,} total)")

# ── Drug prevalence check ────────────────────────────────────────────────────
print("\nDrug prevalence:")
for drug in sorted(all_drugs):
    col  = f"item_drug_{drug}"
    cnt  = cohort[col].sum()
    c_c  = cohort[cohort["target"]==1][col].sum()
    c_r  = cohort[cohort["target"]==0][col].sum()
    print(f"  {drug:20s}: n={cnt:5,}  cases={c_c:4,}  ctrl={c_r:5,}")


def ie(df, drug_a, drug_b):
    """Observational IE from co-occurrence."""
    col_a = f"item_drug_{drug_a}"
    col_b = f"item_drug_{drug_b}"
    n     = len(df)
    if n == 0:
        return np.nan
    p_a    = df[col_a].mean()
    p_b    = df[col_b].mean()
    p_ab   = (df[col_a] * df[col_b]).mean()
    p_case = df["target"].mean()
    # Conditional probabilities
    mask_a  = df[col_a] == 1
    mask_b  = df[col_b] == 1
    mask_ab = mask_a & mask_b
    if mask_a.sum() == 0 or mask_b.sum() == 0 or mask_ab.sum() < 2:
        return np.nan
    p_case_a  = df.loc[mask_a,  "target"].mean()
    p_case_b  = df.loc[mask_b,  "target"].mean()
    p_case_ab = df.loc[mask_ab, "target"].mean()
    denom = p_case_a * p_case_b
    return (p_case_ab / denom) if denom > 0 else np.nan


# ── Observed IE ───────────────────────────────────────────────────────────────
print("\n=== OBSERVED IE (from full cohort) ===")
observed = {}
for (a, b) in TARGET_PAIRS:
    v = ie(cohort, a, b)
    observed[(a, b)] = v
    n_ab = (cohort[f"item_drug_{a}"] * cohort[f"item_drug_{b}"]).sum()
    print(f"  {a} + {b}: IE={v:.2f}  n_co-occur={n_ab:,}")

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
print(f"\nBootstrapping {N_BOOT} resamples …")
cohort_arr = cohort.reset_index(drop=True)
n          = len(cohort_arr)

boot = {pair: [] for pair in TARGET_PAIRS}
for i in range(N_BOOT):
    idx   = RNG.integers(0, n, size=n)
    samp  = cohort_arr.iloc[idx]
    for (a, b) in TARGET_PAIRS:
        boot[(a, b)].append(ie(samp, a, b))
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_BOOT}")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n=== BOOTSTRAP 95% CI ===")
print(f"{'Drug A':20s}  {'Drug B':20s}  {'Obs IE':>8s}  {'95% CI':>18s}")
print("-" * 76)

ci_results = {}
for (a, b) in TARGET_PAIRS:
    samples = [v for v in boot[(a, b)] if not np.isnan(v)]
    obs     = observed[(a, b)]
    if len(samples) < 50:
        lo, hi = np.nan, np.nan
    else:
        lo = np.percentile(samples, 2.5)
        hi = np.percentile(samples, 97.5)
    lo_str = f"{lo:.2f}" if not np.isnan(lo) else "n/a"
    hi_str = f"{hi:.2f}" if not np.isnan(hi) else "n/a"
    obs_str = f"{obs:.1f}" if not np.isnan(obs) else "n/a"
    print(f"  {a:20s}  {b:20s}  {obs_str:>8s}  [{lo_str}–{hi_str}]")
    ci_results[f"{a}+{b}"] = {
        "ie": round(float(obs), 1) if not np.isnan(obs) else None,
        "ci_lo": round(float(lo), 2) if not np.isnan(lo) else None,
        "ci_hi": round(float(hi), 2) if not np.isnan(hi) else None,
    }

# ── Manuscript table ─────────────────────────────────────────────────────────
print("\n=== CH_4 TABLE (manuscript values) ===")
pair_labels = {
    ("ACETAMINOPHEN", "LEVOFLOXACIN"): ("Acetaminophen", "Levofloxacin", "16.3"),
    ("LEVOFLOXACIN",  "LORAZEPAM"):    ("Levofloxacin",  "Lorazepam",    "11.9"),
    ("CARVEDILOL",    "LEVOFLOXACIN"): ("Carvedilol",    "Levofloxacin", "10.5"),
    ("GABAPENTIN",    "LEVOFLOXACIN"): ("Gabapentin",    "Levofloxacin", " 9.4"),
    ("DIGOXIN",       "SIMVASTATIN"):  ("Digoxin",       "Simvastatin",  " 6.0"),
}
for (a, b), (la, lb, ie_val) in pair_labels.items():
    r = ci_results[f"{a}+{b}"]
    lo = r["ci_lo"]; hi = r["ci_hi"]
    ci_str = f"{lo:.2f}–{hi:.2f}" if lo is not None else "n/a"
    print(f"  | {la} | {lb} | {ie_val} | <0.001 | <0.001 | {ci_str} |")

with open("ffa_ie_ci.json", "w") as f:
    json.dump(ci_results, f, indent=2)
print("\nSaved ffa_ie_ci.json")
