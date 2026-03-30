"""
Compute Spearman rank correlation of IR (causal_responsibility) scores
across the three non_opioid_ed geriatric age bands.
Source: gold/ffa_analysis/non_opioid_ed/{band}/{bin}/ffa_causal_factors.csv
"""
import boto3, io
import pandas as pd
from scipy.stats import spearmanr

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["65-74", "75-84", "85-114"]
BINS   = ["low", "medium", "high", "extreme"]


def load_ffa(band, bin_name):
    band_key = band.replace("-", "_")
    key = f"gold/ffa_analysis/non_opioid_ed/{band_key}/bin_models/{bin_name}/ffa_causal_factors.csv"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        df = pd.read_csv(io.BytesIO(data))
        return df
    except Exception:
        return None


# Collect IR scores per band (aggregate across bins by mean causal_responsibility)
band_ir = {}
for band in BANDS:
    frames = []
    for bin_name in BINS:
        df = load_ffa(band, bin_name)
        if df is not None and "causal_responsibility" in df.columns:
            feat_col = next((c for c in ("feature", "feature_name", "drug", "drug_name")
                             if c in df.columns), df.columns[0])
            sub = df[[feat_col, "causal_responsibility"]].rename(
                columns={feat_col: "feature", "causal_responsibility": f"ir_{bin_name}"})
            frames.append(sub)
            print(f"  {band}/{bin_name}: {len(df)} features")
    if frames:
        # Merge bins on feature name, then average IR across bins
        from functools import reduce
        merged = reduce(lambda a, b: a.merge(b, on="feature", how="outer"), frames)
        ir_cols = [c for c in merged.columns if c.startswith("ir_")]
        merged["ir_mean"] = merged[ir_cols].mean(axis=1)
        band_ir[band] = merged.set_index("feature")["ir_mean"]
        print(f"  {band}: {len(merged)} total features, IR range [{merged['ir_mean'].min():.6f}, {merged['ir_mean'].max():.6f}]")
    else:
        print(f"  {band}: no FFA data found")

print()
if len(band_ir) < 2:
    print("Not enough bands loaded for correlation.")
else:
    # Align all bands on common features
    combined = pd.DataFrame(band_ir).dropna()
    print(f"Common features across all bands: {len(combined)}")
    print(f"\nTop 10 features by mean IR (65-74):")
    if "65-74" in combined.columns:
        print(combined["65-74"].sort_values(ascending=False).head(10))

    # Pairwise Spearman ρ
    print(f"\n=== Spearman ρ of IR Rankings ===")
    pairs = [
        ("65-74", "75-84"),
        ("65-74", "85-114"),
        ("75-84", "85-114"),
    ]
    rhos = []
    for b1, b2 in pairs:
        if b1 in combined.columns and b2 in combined.columns:
            rho, pval = spearmanr(combined[b1], combined[b2])
            rhos.append(rho)
            print(f"  {b1} vs {b2}: ρ = {rho:.3f}  (p = {pval:.4f})")

    if rhos:
        print(f"\n  Range: ρ = {min(rhos):.2f}–{max(rhos):.2f}")
        print(f"  Formatted for manuscript: ρ = {min(rhos):.2f}–{max(rhos):.2f}")
