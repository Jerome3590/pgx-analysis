"""
Compute lift-based drug pair synergy from axp_explanations.parquet.
Lift = P(A,B|class=1) / P(A,B|class=0).
Pairs with lift > 1 are "synergistic" (over-represented in positive AXP rules).
IE score approximation = P(A,B|1) / [P(A|1) × P(B|1)]  (inner lift = FFA interaction effect).
IR_A, IR_B = causal_responsibility from ffa_causal_factors.csv.
"""
import boto3, io, re
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
PREFIX = "gold/ffa_analysis"


def read_parquet(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        return None


def read_csv_s3(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        return None


def parse_drugs(axp_val):
    """Return set of drug feature names present in one AXP rule."""
    if axp_val is None:
        return set()
    terms = axp_val if isinstance(axp_val, (list, np.ndarray)) else str(axp_val).split("'")
    feats = set()
    for t in terms:
        t = str(t).strip()
        m = re.match(r"^(item_drug_\S+)\s*(>|>=|<|<=|==|!=)", t)
        if m:
            feats.add(m.group(1))
    return feats


def clean(f):
    return f.replace("item_drug_", "").title()


def analyze(cohort, ab, top_n_pairs=10, top_n_trips=5, min_lift=1.0):
    band = ab.replace("_", "-")
    label = f"{cohort} / {band}"

    cf = read_csv_s3(f"{PREFIX}/{cohort}/{ab}/bin_models/low/ffa_causal_factors.csv")
    ir_map = dict(zip(cf["feature"], cf["causal_responsibility"])) if cf is not None else {}

    df = read_parquet(f"{PREFIX}/{cohort}/{ab}/bin_models/low/xgboost/axp_explanations.parquet")
    if df is None:
        print(f"  {label}: no data"); return

    df["drug_set"] = df["axp"].apply(parse_drugs)
    pos = df[df["predicted_class"] == 1]
    neg = df[df["predicted_class"] == 0]
    n1, n0 = len(pos), len(neg)

    # Per-drug support in class-1 and class-0
    sup1 = Counter()
    sup0 = Counter()
    for ds in pos["drug_set"]: sup1.update(ds)
    for ds in neg["drug_set"]: sup0.update(ds)

    # Pair support
    psup1 = Counter()
    psup0 = Counter()
    for ds in pos["drug_set"]:
        dl = sorted(ds)
        for pair in combinations(dl, 2):
            psup1[pair] += 1
    for ds in neg["drug_set"]:
        dl = sorted(ds)
        for pair in combinations(dl, 2):
            psup0[pair] += 1

    # Triplet support (class-1 only)
    tsup1 = Counter()
    for ds in pos["drug_set"]:
        dl = sorted(ds)
        for trip in combinations(dl, 3):
            tsup1[trip] += 1

    # Compute lift and IE for each pair
    rows = []
    for pair, cnt1 in psup1.items():
        cnt0   = psup0.get(pair, 0)
        p_pair_1 = cnt1 / n1
        p_pair_0 = cnt0 / n0 if n0 > 0 else 1e-9
        lift   = p_pair_1 / max(p_pair_0, 1e-9)
        # IE = pair support relative to product of individual supports (independence baseline)
        p_a1 = sup1.get(pair[0], 0) / n1
        p_b1 = sup1.get(pair[1], 0) / n1
        ie    = p_pair_1 / max(p_a1 * p_b1, 1e-9)  # >1 = synergistic
        rows.append({
            "drug_a": pair[0], "drug_b": pair[1],
            "support_pos": cnt1, "support_neg": cnt0,
            "p_pair_1": p_pair_1, "p_pair_0": p_pair_0,
            "lift": lift, "ie": ie,
            "ir_a": ir_map.get(pair[0], 0),
            "ir_b": ir_map.get(pair[1], 0),
        })

    pairs_df = pd.DataFrame(rows)
    synergistic = pairs_df[pairs_df["ie"] > min_lift].sort_values("ie", ascending=False)
    n_syn_pairs = len(synergistic)

    # Triplets > 1 appearance
    n_syn_trips = sum(1 for v in tsup1.values() if v > 1)

    print(f"\n{'='*72}")
    print(f"  {label}  (class-1={n1}, class-0={n0})")
    print(f"  Unique drug pairs identified:       {len(psup1):,}")
    print(f"  Synergistic pairs (IE > {min_lift:.1f}):    {n_syn_pairs:,}")
    print(f"  Unique drug triplets:               {len(tsup1):,}")
    print(f"  High-risk triplets (freq > 1):      {n_syn_trips:,}")
    print(f"{'='*72}")

    print(f"\n  Top {top_n_pairs} synergistic drug pairs (by IE score):")
    print(f"  {'Drug A':<28s} {'Drug B':<28s} {'IE':>6s} {'IR_A':>7s} {'IR_B':>7s} {'Sup1':>5s}")
    for _, r in synergistic.head(top_n_pairs).iterrows():
        print(f"  {clean(r['drug_a']):<28s} {clean(r['drug_b']):<28s} "
              f"{r['ie']:6.2f} {r['ir_a']:7.4f} {r['ir_b']:7.4f} {int(r['support_pos']):5d}")

    top_trips = sorted(tsup1.items(), key=lambda x: -x[1])[:top_n_trips]
    if top_trips:
        print(f"\n  Top {top_n_trips} drug triplets (by co-occurrence):")
        for trip, cnt in top_trips:
            names = " + ".join(clean(d) for d in trip)
            print(f"    {names}  (n={cnt})")

    return {
        "n_class1": n1, "n_class0": n0,
        "n_unique_pairs": len(psup1),
        "n_synergistic_pairs": n_syn_pairs,
        "n_unique_triplets": len(tsup1),
        "n_high_risk_triplets": n_syn_trips,
        "top_pairs": synergistic.head(10).to_dict("records"),
    }


# ── Run for manuscript cohorts ──────────────────────────────────────────────
print("FFA Drug Pair Synergy Analysis (IE > 1.0)\n")
results = {}

# CH_4: non_opioid_ed (primary)
for ab in ["65_74", "75_84", "85_114"]:
    r = analyze("non_opioid_ed", ab, min_lift=1.0)
    if r:
        results[f"non_opioid_ed/{ab.replace('_','-')}"] = r

# CH_3: opioid_ed (for reference)
for ab in ["25_44"]:
    r = analyze("opioid_ed", ab, min_lift=1.0)
    if r:
        results[f"opioid_ed/{ab.replace('_','-')}"] = r
