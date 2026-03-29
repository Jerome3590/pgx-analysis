"""
Approximate DTW cluster sizes for CH_3.
Cluster 1 (Rapid-Onset):  first-opioid→OUD-ED span < 6 months
Cluster 2 (Chronic-Escalation): span ≥ 6 months
Uses event_intervals parquet (target=1 cases) + model_events for case IDs.
"""
import boto3, io
import pandas as pd
import numpy as np

s3     = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"
BANDS  = ["13-24", "25-44", "45-54", "55-64"]
THRESHOLD_MONTHS = 6  # Rapid-Onset if span < 6 months

results = []

for band in BANDS:
    ab = band.replace("-", "_")
    print(f"\n=== {band} ===")

    # ── Load case patient IDs from test + train final_features ────────────
    cases = set()
    for split in ["model_test", "model_train"]:
        fp_key = (f"gold/final_model/opioid_ed/{band}/inputs/"
                  f"{split}/final_features.parquet")
        try:
            data = s3.get_object(Bucket=BUCKET, Key=fp_key)["Body"].read()
            fp   = pd.read_parquet(io.BytesIO(data),
                                   columns=["mi_person_key", "target"])
            ids  = set(fp.loc[fp["target"] == 1, "mi_person_key"].unique())
            cases |= ids
            print(f"  {split}: {len(ids):,} cases")
        except Exception as e:
            print(f"  {split} error: {e}")
    if not cases:
        continue
    print(f"  Total cases: {len(cases):,}")

    # ── Load event_intervals and compute span for each case ────────────────
    ei_key = f"gold/dtw_filter/opioid_ed/{band}/event_intervals_opioid_ed_{ab}.parquet"
    try:
        data = s3.get_object(Bucket=BUCKET, Key=ei_key)["Body"].read()
        ei   = pd.read_parquet(io.BytesIO(data),
                               columns=["mi_person_key", "event_date", "target"])
    except Exception as e:
        print(f"  event_intervals error: {e}")
        continue

    # Filter to cases only
    ei_cases = ei[ei["mi_person_key"].isin(cases)].copy()
    ei_cases["event_date"] = pd.to_datetime(ei_cases["event_date"])
    print(f"  Event rows for cases: {len(ei_cases):,}")

    # Compute span per patient (days from first to last event)
    span = (ei_cases.groupby("mi_person_key")["event_date"]
            .agg(first="min", last="max")
            .assign(span_days=lambda d: (d["last"] - d["first"]).dt.days)
            .reset_index())

    span["span_months"] = span["span_days"] / 30.4
    threshold_days = THRESHOLD_MONTHS * 30.4

    rapid    = span[span["span_days"] < threshold_days]
    chronic  = span[span["span_days"] >= threshold_days]

    n_rapid   = len(rapid)
    n_chronic = len(chronic)
    n_total   = len(span)
    pct_rapid   = n_rapid  / n_total * 100
    pct_chronic = n_chronic / n_total * 100
    mean_rapid   = rapid["span_months"].mean()
    mean_chronic = chronic["span_months"].mean()

    print(f"  Rapid-Onset    (<{THRESHOLD_MONTHS} mo): n={n_rapid:,}  "
          f"({pct_rapid:.1f}%)  mean_span={mean_rapid:.1f} mo")
    print(f"  Chronic-Escal  (≥{THRESHOLD_MONTHS} mo): n={n_chronic:,}  "
          f"({pct_chronic:.1f}%)  mean_span={mean_chronic:.1f} mo")

    results.append({
        "band": band,
        "n_rapid": n_rapid, "pct_rapid": pct_rapid,
        "n_chronic": n_chronic, "pct_chronic": pct_chronic,
        "mean_span_rapid": mean_rapid, "mean_span_chronic": mean_chronic,
    })

# ── Aggregate across all bands ─────────────────────────────────────────────
if results:
    df = pd.DataFrame(results)
    total_rapid   = df["n_rapid"].sum()
    total_chronic = df["n_chronic"].sum()
    grand_total   = total_rapid + total_chronic
    pct_r = total_rapid  / grand_total * 100
    pct_c = total_chronic / grand_total * 100

    print(f"\n{'='*60}")
    print(f"OVERALL (opioid_ed, all bands):")
    print(f"  Rapid-Onset:         {total_rapid:,}  ({pct_r:.1f}%)")
    print(f"  Chronic-Escalation:  {total_chronic:,}  ({pct_c:.1f}%)")
    print(f"  Grand total:         {grand_total:,}")
    print(f"\n=== CH_3 MANUSCRIPT VALUES ===")
    print(f"  Cluster 1 (Rapid-Onset):       n={total_rapid:,}  ({pct_r:.0f}% of cases)")
    print(f"  Cluster 2 (Chronic-Escal):     n={total_chronic:,}  ({pct_c:.0f}% of cases)")
