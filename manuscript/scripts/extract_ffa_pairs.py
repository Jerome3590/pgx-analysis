"""
Extract drug pair co-occurrences from axp_explanations.parquet.
Pairs = drug features appearing together in the same AXP minimal explanation.
Also computes per-drug IR from ffa_causal_factors.csv for the DDI table.
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
        print(f"  parquet error {key}: {e}")
        return None


def read_csv_s3(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        return None


def parse_axp_rule(axp_val):
    """Extract feature names from an AXP rule (string list or numpy array)."""
    if axp_val is None:
        return []
    if isinstance(axp_val, (list, np.ndarray)):
        terms = [str(t) for t in axp_val]
    else:
        terms = str(axp_val).split("'")
        terms = [t.strip() for t in terms if t.strip() and t.strip() not in (",", "[", "]")]

    features = []
    for term in terms:
        # Extract feature name before operator (>, <=, ==, !=)
        m = re.match(r"^(\S+)\s*(>|>=|<|<=|==|!=)", term.strip())
        if m:
            features.append(m.group(1))
    return features


def is_drug(feat):
    return feat.startswith("item_drug_")


def clean_drug(feat):
    return feat.replace("item_drug_", "").title()


configs = [
    ("non_opioid_ed", ["65_74", "75_84", "85_114"]),
    ("opioid_ed",     ["25_44", "45_54", "55_64"]),
]

for cohort, bands in configs:
    for ab in bands:
        band = ab.replace("_", "-")
        print(f"\n{'='*70}")
        print(f"  {cohort} / {band}")
        print(f"{'='*70}")

        # Load IR scores from causal_factors
        cf_key = f"{PREFIX}/{cohort}/{ab}/bin_models/low/ffa_causal_factors.csv"
        cf = read_csv_s3(cf_key)
        ir_map = {}
        if cf is not None:
            ir_map = dict(zip(cf["feature"], cf["causal_responsibility"]))

        # Load AXP explanations
        axp_key = f"{PREFIX}/{cohort}/{ab}/bin_models/low/xgboost/axp_explanations.parquet"
        df = read_parquet(axp_key)
        if df is None:
            print("  No axp_explanations found")
            continue

        # Focus on class-1 predictions (ADE/OUD cases)
        df_pos = df[df["predicted_class"] == 1] if "predicted_class" in df.columns else df

        # Parse AXP rules; collect drug-only terms
        pair_counter = Counter()
        triplet_counter = Counter()
        drug_freq = Counter()
        n_rules_with_drugs = 0
        n_rules_with_pairs = 0
        n_rules_with_triplets = 0

        for axp_val in df_pos["axp"]:
            feats = parse_axp_rule(axp_val)
            drugs = [f for f in feats if is_drug(f)]
            if not drugs:
                continue
            n_rules_with_drugs += 1
            drug_freq.update(drugs)

            if len(drugs) >= 2:
                n_rules_with_pairs += 1
                for pair in combinations(sorted(drugs), 2):
                    pair_counter[pair] += 1

            if len(drugs) >= 3:
                n_rules_with_triplets += 1
                for trip in combinations(sorted(drugs), 3):
                    triplet_counter[trip] += 1

        n_total = len(df_pos)
        print(f"  Class-1 explanations: {n_total}")
        print(f"  Rules with drug features:  {n_rules_with_drugs}")
        print(f"  Rules with drug pairs:     {n_rules_with_pairs}")
        print(f"  Rules with drug triplets:  {n_rules_with_triplets}")
        print(f"  Unique drug pairs:         {len(pair_counter)}")
        print(f"  Unique drug triplets:      {len(triplet_counter)}")

        print(f"\n  Top 10 drug pairs (by co-occurrence frequency in class-1 rules):")
        for (a, b), cnt in pair_counter.most_common(10):
            ir_a = ir_map.get(a, 0)
            ir_b = ir_map.get(b, 0)
            pct = cnt / n_total * 100
            print(f"    {clean_drug(a):<25s} + {clean_drug(b):<25s}  freq={cnt:4d} ({pct:.1f}%)  "
                  f"IR_A={ir_a:.4f}  IR_B={ir_b:.4f}")

        if triplet_counter:
            print(f"\n  Top 5 drug triplets:")
            for trip, cnt in triplet_counter.most_common(5):
                names = " + ".join(clean_drug(d) for d in trip)
                pct = cnt / n_total * 100
                print(f"    {names}  freq={cnt} ({pct:.1f}%)")
