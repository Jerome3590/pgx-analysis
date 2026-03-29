"""
Bootstrap 95% CI for FFA pairwise IE from AXP explanations (predicted_class=1 rules).
IE(A,B) = P(A∩B in AXP | class=1) / [P(A | class=1) × P(B | class=1)]
Aggregates across all 4 bins for non_opioid_ed/85_114.
"""
import boto3, io, re, json
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BINS   = ["low", "medium", "high", "extreme"]
N_BOOT = 1000
RNG    = np.random.default_rng(42)

TARGET_PAIRS = [
    ("item_drug_ACETAMINOPHEN", "item_drug_LEVOFLOXACIN"),
    ("item_drug_LEVOFLOXACIN",  "item_drug_LORAZEPAM"),
    ("item_drug_CARVEDILOL",    "item_drug_LEVOFLOXACIN"),
    ("item_drug_GABAPENTIN",    "item_drug_LEVOFLOXACIN"),
    ("item_drug_DIGOXIN",       "item_drug_SIMVASTATIN"),
]

PAIR_LABELS = {
    ("item_drug_ACETAMINOPHEN", "item_drug_LEVOFLOXACIN"): ("Acetaminophen", "Levofloxacin", 16.3),
    ("item_drug_LEVOFLOXACIN",  "item_drug_LORAZEPAM"):    ("Levofloxacin",  "Lorazepam",    11.9),
    ("item_drug_CARVEDILOL",    "item_drug_LEVOFLOXACIN"): ("Carvedilol",    "Levofloxacin", 10.5),
    ("item_drug_GABAPENTIN",    "item_drug_LEVOFLOXACIN"): ("Gabapentin",    "Levofloxacin",  9.4),
    ("item_drug_DIGOXIN",       "item_drug_SIMVASTATIN"):  ("Digoxin",       "Simvastatin",   6.0),
}


def parse_drugs(axp_val):
    terms = axp_val if isinstance(axp_val, (list, np.ndarray)) else str(axp_val).split("'")
    feats = set()
    for t in terms:
        t = str(t).strip()
        m = re.match(r"^(item_drug_\S+)\s*(>|>=|<|<=|==|!=)", t)
        if m:
            feats.add(m.group(1))
    return feats


def compute_ie(drug_sets, drug_a, drug_b):
    n = len(drug_sets)
    if n == 0:
        return np.nan
    p_a  = sum(drug_a  in ds for ds in drug_sets) / n
    p_b  = sum(drug_b  in ds for ds in drug_sets) / n
    p_ab = sum(drug_a in ds and drug_b in ds for ds in drug_sets) / n
    denom = p_a * p_b
    return (p_ab / denom) if denom > 1e-12 else np.nan


# ── Load all bins ─────────────────────────────────────────────────────────────
print("Loading AXP explanations for non_opioid_ed/85_114 …")
all_pos = []   # class-1 drug sets
for binn in BINS:
    key = ("gold/ffa_analysis/non_opioid_ed/85_114/bin_models/"
           f"{binn}/xgboost/axp_explanations.parquet")
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        df   = pd.read_parquet(io.BytesIO(data))
        df["drug_set"] = df["axp"].apply(parse_drugs)
        pos  = df[df["predicted_class"] == 1]["drug_set"].tolist()
        all_pos.extend(pos)
        print(f"  {binn}: {len(pos)} class-1 patients (total={len(df)})")
    except Exception as e:
        print(f"  {binn}: {e}")

n_pos = len(all_pos)
print(f"\nTotal class-1 drug sets: {n_pos:,}")

# Drug prevalence in class-1
sup = Counter()
for ds in all_pos:
    sup.update(ds)
print("\nTarget drug prevalence (class-1 AXPs):")
for d in sorted({d for p in TARGET_PAIRS for d in p}):
    print(f"  {d}: {sup[d]} / {n_pos} ({sup[d]/n_pos*100:.1f}%)")

# ── Observed IE ───────────────────────────────────────────────────────────────
print("\n=== OBSERVED IE (aggregated across all bins) ===")
observed = {}
for (a, b) in TARGET_PAIRS:
    v = compute_ie(all_pos, a, b)
    observed[(a, b)] = v
    n_co = sum(a in ds and b in ds for ds in all_pos)
    print(f"  {a.replace('item_drug_','')} + {b.replace('item_drug_','')}: "
          f"IE={v:.2f}  n_co={n_co}")

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
print(f"\nBootstrapping {N_BOOT} resamples of {n_pos} class-1 patients …")
all_pos_arr = np.array(all_pos, dtype=object)
boot = {pair: [] for pair in TARGET_PAIRS}

for i in range(N_BOOT):
    idx  = RNG.integers(0, n_pos, size=n_pos)
    samp = all_pos_arr[idx].tolist()
    for (a, b) in TARGET_PAIRS:
        boot[(a, b)].append(compute_ie(samp, a, b))
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_BOOT}")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n=== BOOTSTRAP 95% CI ===")
ci_results = {}
for (a, b) in TARGET_PAIRS:
    la, lb, ie_ms = PAIR_LABELS[(a, b)]
    samples = [v for v in boot[(a, b)] if not np.isnan(v)]
    obs     = observed[(a, b)]
    lo = np.percentile(samples, 2.5)  if len(samples) > 50 else np.nan
    hi = np.percentile(samples, 97.5) if len(samples) > 50 else np.nan
    lo_s  = f"{lo:.2f}"  if not np.isnan(lo) else "n/a"
    hi_s  = f"{hi:.2f}"  if not np.isnan(hi) else "n/a"
    obs_s = f"{obs:.2f}" if not np.isnan(obs) else "n/a"
    print(f"  {la:14s} + {lb:14s}: "
          f"obs={obs_s}  manuscript={ie_ms}  95%CI [{lo_s}–{hi_s}]")
    ci_results[f"{la}+{lb}"] = {
        "drug_a": la, "drug_b": lb,
        "ie_manuscript": ie_ms,
        "ie_axp":  round(float(obs), 2) if not np.isnan(obs) else None,
        "ci_lo":   round(float(lo),  2) if not np.isnan(lo)  else None,
        "ci_hi":   round(float(hi),  2) if not np.isnan(hi)  else None,
    }

# ── Manuscript table ──────────────────────────────────────────────────────────
print("\n=== CH_4 TABLE (fill-in values) ===")
for (a, b) in TARGET_PAIRS:
    la, lb, ie_ms = PAIR_LABELS[(a, b)]
    r  = ci_results[f"{la}+{lb}"]
    lo = r["ci_lo"]; hi = r["ci_hi"]
    ci = f"{lo:.2f}–{hi:.2f}" if lo is not None else "pending"
    print(f"  | {la} | {lb} | {ie_ms} | <0.001 | <0.001 | {ci} |")

with open("ffa_ie_ci.json", "w") as f:
    json.dump(ci_results, f, indent=2)
print("\nSaved ffa_ie_ci.json")
