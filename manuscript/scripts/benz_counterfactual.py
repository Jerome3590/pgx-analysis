"""
Compute FFA counterfactual risk reduction for benzodiazepine removal (CH5 placeholder).
Uses causal_responsibility from ffa_causal_factors + mean case predicted probability.
"""
import boto3, io
import pandas as pd
import numpy as np

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

# Load causal factors for 85-114 low bin
key = "gold/ffa_analysis/non_opioid_ed/85_114/bin_models/low/ffa_causal_factors.csv"
data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
cf = pd.read_csv(io.BytesIO(data))

print("Top 15 features by causal_responsibility:")
top = cf.nlargest(15, "causal_responsibility")
for _, r in top.iterrows():
    print(f"  {r['feature']:40s}  CR={r['causal_responsibility']:.4f}")

benz_patterns = ["LORAZEPAM", "DIAZEPAM", "ALPRAZOLAM", "CLONAZEPAM", "TEMAZEPAM"]
benz = cf[cf["feature"].str.contains("|".join(benz_patterns), case=False, na=False)]
print("\nBenzodiazepine features:")
print(benz[["feature", "causal_responsibility", "rule_frequency"]].to_string())

# The IR (Intervention Rate) is CR * mean_case_probability
# For 85-114, n_cases=168, mean_pred ~ case_rate ~ 168/(168+2355) ~ 0.067
case_rate_85_114 = 168 / (168 + 2355)
print(f"\nApprox case rate (85-114): {case_rate_85_114:.4f}")

# Lorazepam counterfactual Δp̂
if len(benz) > 0:
    lorazepam = cf[cf["feature"].str.contains("LORAZEPAM", case=False, na=False)]
    if len(lorazepam) > 0:
        cr = lorazepam.iloc[0]["causal_responsibility"]
        delta_p = cr * case_rate_85_114
        pct = delta_p * 100
        print(f"\nLorazepam: CR={cr:.4f}  |Dp_hat|~{delta_p:.4f}  ~{pct:.0f}%")

# Also compute from rule frequency-weighted approach
# IR = (rule_frequency / total_rules) * mean causal contribution
total_rules = cf["total_rules"].iloc[0] if "total_rules" in cf.columns else None
if total_rules:
    for bname in benz_patterns:
        row = cf[cf["feature"].str.contains(bname, case=False, na=False)]
        if len(row) > 0:
            r = row.iloc[0]
            rf = r["rule_frequency"] / total_rules
            print(f"{bname}: rule_freq_prop={rf:.4f}  CR={r['causal_responsibility']:.5f}")
