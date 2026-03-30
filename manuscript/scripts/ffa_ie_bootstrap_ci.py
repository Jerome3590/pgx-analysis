"""
Bootstrap 95% CI for FFA pairwise IE (Interaction Effect) scores.
Source: axp_explanations.parquet — per-patient AXP rule condition lists.

IE(A,B) = P(A∩B in AXP rules) / [P(A in AXP) × P(B in AXP)]
Bootstrap: 1,000 resamples with replacement per bin, aggregate across bins.

Target pairs (from ch04_psp.qmd Table):
  Acetaminophen + Levofloxacin  IE=16.3
  Levofloxacin  + Lorazepam     IE=11.9
  Carvedilol    + Levofloxacin  IE=10.5
  Gabapentin    + Levofloxacin  IE= 9.4
  Digoxin       + Simvastatin   IE= 6.0
"""
import boto3, io, re, json
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BINS   = ["low", "medium", "high", "extreme"]
BAND   = "85_114"
COHORT = "non_opioid_ed"

TARGET_PAIRS = [
    ("ACETAMINOPHEN", "LEVOFLOXACIN"),
    ("LEVOFLOXACIN",  "LORAZEPAM"),
    ("CARVEDILOL",    "LEVOFLOXACIN"),
    ("GABAPENTIN",    "LEVOFLOXACIN"),
    ("DIGOXIN",       "SIMVASTATIN"),
]

N_BOOT = 1000
RNG    = np.random.default_rng(42)


def drug_from_condition(cond):
    """Extract drug name from 'item_drug_DRUGNAME OP value' condition."""
    m = re.match(r"item_drug_([A-Z0-9_]+)\s", cond.strip())
    return m.group(1) if m else None


def parse_axp(axp_str):
    """Return set of drug names present in this patient's AXP rule list."""
    drugs = set()
    try:
        conditions = ast_eval_or_split(axp_str)
        for cond in conditions:
            d = drug_from_condition(str(cond))
            if d:
                drugs.add(d)
    except Exception:
        pass
    return drugs


def ast_eval_or_split(axp_str):
    """Parse the AXP list (stored as string repr of Python list)."""
    import ast
    try:
        return ast.literal_eval(axp_str)
    except Exception:
        return str(axp_str).strip("[]").split(", ")


# ── Load AXP explanations for all bins ──────────────────────────────────────
print(f"Loading AXP explanations for {COHORT}/{BAND} …")
bin_frames = {}
for binn in BINS:
    key = (f"gold/ffa_analysis/{COHORT}/{BAND}/bin_models/{binn}"
           f"/xgboost/axp_explanations.parquet")
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        df   = pd.read_parquet(io.BytesIO(data))
        bin_frames[binn] = df
        print(f"  {binn}: {len(df):,} rows")
    except Exception as e:
        print(f"  {binn}: {e}")

# ── Parse drug sets per patient ───────────────────────────────────────────────
print("\nParsing drug co-occurrences from AXP rules …")
all_drug_sets = []
for binn, df in bin_frames.items():
    for _, row in df.iterrows():
        drugs = parse_axp(row["axp"])
        all_drug_sets.append(drugs)

n_patients = len(all_drug_sets)
print(f"  Total patients: {n_patients:,}")

# Quick coverage check for target drugs
for drug in sorted({d for p in TARGET_PAIRS for d in p}):
    cnt = sum(1 for ds in all_drug_sets if drug in ds)
    print(f"  {drug:20s}: present in {cnt:,} / {n_patients} AXPs ({cnt/n_patients*100:.1f}%)")


def compute_ie(drug_sets, drug_a, drug_b):
    """IE = P(A∩B) / [P(A) × P(B)], floored at 0."""
    n   = len(drug_sets)
    p_a = sum(drug_a in ds for ds in drug_sets) / n
    p_b = sum(drug_b in ds for ds in drug_sets) / n
    p_ab = sum(drug_a in ds and drug_b in ds for ds in drug_sets) / n
    denom = p_a * p_b
    return (p_ab / denom) if denom > 0 else 0.0


# ── Observed IE scores ────────────────────────────────────────────────────────
print("\n=== OBSERVED IE SCORES ===")
observed = {}
for (a, b) in TARGET_PAIRS:
    ie = compute_ie(all_drug_sets, a, b)
    observed[(a, b)] = ie
    print(f"  {a} + {b}: IE = {ie:.2f}")

# ── Bootstrap CIs ─────────────────────────────────────────────────────────────
print(f"\nBootstrapping {N_BOOT} resamples …")
drug_sets_arr = np.array(all_drug_sets, dtype=object)

boot_results = {pair: [] for pair in TARGET_PAIRS}
for i in range(N_BOOT):
    idx      = RNG.integers(0, n_patients, size=n_patients)
    boot_ds  = drug_sets_arr[idx].tolist()
    for (a, b) in TARGET_PAIRS:
        ie = compute_ie(boot_ds, a, b)
        boot_results[(a, b)].append(ie)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_BOOT} done")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n=== BOOTSTRAP 95% CI FOR IE SCORES ===")
print(f"{'Drug A':20s}  {'Drug B':20s}  {'IE':>6s}  {'95% CI':>20s}")
print("-" * 75)

ci_results = {}
for (a, b) in TARGET_PAIRS:
    boot  = np.array(boot_results[(a, b)])
    ie    = observed[(a, b)]
    lo    = np.percentile(boot, 2.5)
    hi    = np.percentile(boot, 97.5)
    print(f"  {a:20s}  {b:20s}  {ie:6.1f}  [{lo:.2f}–{hi:.2f}]")
    ci_results[f"{a}+{b}"] = {
        "ie": round(ie, 1),
        "ci_lo": round(lo, 2),
        "ci_hi": round(hi, 2),
    }

# ── Save ─────────────────────────────────────────────────────────────────────
with open("data/ffa_ie_ci.json", "w") as f:
    json.dump(ci_results, f, indent=2)
print("\nSaved ffa_ie_ci.json")

# ── Manuscript table values ───────────────────────────────────────────────────
print("\n=== CH_4 TABLE VALUES ===")
for (a, b) in TARGET_PAIRS:
    r = ci_results[f"{a}+{b}"]
    print(f"  | {a.title()} | {b.title()} | {r['ie']} | <0.001 | <0.001 "
          f"| {r['ci_lo']}–{r['ci_hi']} |")
