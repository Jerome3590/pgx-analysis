#!/usr/bin/env python3
"""
Extract visual-pipeline manuscript data from S3 manuscript checkpoints.

Reads structured JSON checkpoints written by notebook 4 (4_dashboard_visuals.ipynb)
from s3://pgxdatalake/gold/manuscript_checkpoints/ and writes:

  manuscript/visual_manuscript_data.json   — FP-Growth, DTW, SHAP, PGx per cohort/age_band
  manuscript/pgx_coverage.json             — PGx feature coverage % (CH_5 placeholder)
  manuscript/shap_top_features.json        — SHAP top-10 per cohort/age_band (CH_3, CH_4, CH_5)

Run after notebooks 3 and 4 complete on EC2.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

BUCKET       = "pgxdatalake"
CHKPT_PREFIX = "gold/manuscript_checkpoints"
SCRIPT_DIR   = Path(__file__).parent
MANUSCRIPT   = SCRIPT_DIR.parent

COHORTS    = ["opioid_ed", "non_opioid_ed"]
AGE_BANDS  = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
DENSITY_BINS = ["low", "medium", "high", "extreme"]

s3 = boto3.client("s3", region_name="us-east-1")


def _get_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            return None
        raise


def main() -> None:
    visual_data: Dict[str, Any] = {}
    pgx_coverage: Dict[str, Any] = {}
    shap_features: Dict[str, Any] = {}
    found = missing = 0

    for cohort in COHORTS:
        visual_data[cohort]   = {}
        pgx_coverage[cohort]  = {}
        shap_features[cohort] = {}

        for ab in AGE_BANDS:
            print(f"\n── {cohort}/{ab} ──")
            visual_data[cohort][ab]   = {"age_band": ab, "bins": {}}
            shap_features[cohort][ab] = {}

            # ── PGx coverage — cohort/age_band level (not per-bin) ────────────
            pgx = _get_json(f"{CHKPT_PREFIX}/pgx/{cohort}/{ab}/pgx_manuscript_summary.json")
            if pgx and "pct_coverage" in pgx:
                pgx_coverage[cohort][ab] = {
                    "n_patients":     pgx["n_patients"],
                    "n_with_pgx":     pgx["n_with_pgx"],
                    "pct_coverage":   pgx["pct_coverage"],
                    "mean_pgx_drugs": pgx.get("mean_pgx_drugs"),
                    "mean_cpic_drugs": pgx.get("mean_cpic_drugs"),
                }
                print(f"  [PGx] {pgx['pct_coverage']}% coverage "
                      f"({pgx['n_with_pgx']}/{pgx['n_patients']})")
                found += 1
            else:
                pgx_coverage[cohort][ab] = None
                print(f"  [PGx] not found")
                missing += 1

            # ── Per-bin: FP-Growth / DTW / SHAP / FFA ────────────────────
            for bin_name in DENSITY_BINS:
                bin_entry: Dict[str, Any] = {"bin": bin_name}

                # FP-Growth
                fpg = _get_json(f"{CHKPT_PREFIX}/fpgrowth/{cohort}/{ab}/{bin_name}/fpgrowth_manuscript_summary.json")
                if fpg:
                    bin_entry["fpgrowth"] = {
                        "total_rules": fpg.get("total_rules", 0),
                        "top_rules":   fpg.get("top_rules", [])[:5],
                    }
                    conf = fpg["top_rules"][0]["confidence"] if fpg.get("top_rules") else "n/a"
                    print(f"  [FPG/{bin_name}] rules={fpg.get('total_rules',0)}  top_conf={conf}")
                    found += 1
                else:
                    bin_entry["fpgrowth"] = None
                    print(f"  [FPG/{bin_name}] not found")
                    missing += 1

                # DTW
                dtw = _get_json(f"{CHKPT_PREFIX}/dtw/{cohort}/{ab}/{bin_name}/dtw_manuscript_summary.json")
                if dtw:
                    bin_entry["dtw"] = {
                        "total_trajectories": dtw.get("total_trajectories", 0),
                        "trajectory_length":  dtw.get("trajectory_length", {}),
                        "target_counts":      dtw.get("target_counts", {}),
                        "n_clusters":         dtw.get("n_clusters", 0),
                    }
                    print(f"  [DTW/{bin_name}] trajectories={dtw.get('total_trajectories',0)}  "
                          f"clusters={dtw.get('n_clusters',0)}")
                    found += 1
                else:
                    bin_entry["dtw"] = None
                    print(f"  [DTW/{bin_name}] not found")
                    missing += 1

                # SHAP
                shap = _get_json(f"{CHKPT_PREFIX}/shap/{cohort}/{ab}/{bin_name}/shap_manuscript_summary.json")
                if shap and shap.get("top_features"):
                    bin_entry["shap_top10"] = shap["top_features"]
                    shap_features[cohort][ab][bin_name] = shap["top_features"]
                    top1 = shap["top_features"][0]["feature"]
                    print(f"  [SHAP/{bin_name}] top={top1}")
                    found += 1
                else:
                    bin_entry["shap_top10"] = []
                    shap_features[cohort][ab][bin_name] = []
                    print(f"  [SHAP/{bin_name}] not found or empty")
                    missing += 1

                # FFA
                ffa = _get_json(f"{CHKPT_PREFIX}/ffa/{cohort}/{ab}/{bin_name}/ffa_manuscript_summary.json")
                if ffa:
                    bin_entry["ffa"] = {
                        "n_causal_features": ffa.get("n_causal_features", 0),
                        "top_features":      ffa.get("top_features", [])[:10],
                    }
                    print(f"  [FFA/{bin_name}] features={ffa.get('n_causal_features',0)}")
                    found += 1
                else:
                    bin_entry["ffa"] = None
                    print(f"  [FFA/{bin_name}] not found")
                    missing += 1

                visual_data[cohort][ab]["bins"][bin_name] = bin_entry

    # ── Write output files ───────────────────────────────────────────
    out_visual = MANUSCRIPT / "visual_manuscript_data.json"
    out_pgx    = MANUSCRIPT / "pgx_coverage.json"
    out_shap   = MANUSCRIPT / "shap_top_features.json"

    out_visual.write_text(json.dumps(visual_data,   indent=2), encoding="utf-8")
    out_pgx.write_text(   json.dumps(pgx_coverage,  indent=2), encoding="utf-8")
    out_shap.write_text(  json.dumps(shap_features, indent=2), encoding="utf-8")

    print()
    print(f"Written: {out_visual}")
    print(f"Written: {out_pgx}")
    print(f"Written: {out_shap}")
    print(f"\nCheckpoints: {found} found, {missing} missing (run after notebooks 3+4)")

    # ── Console summaries ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("PGx Coverage (for CH_5 placeholder)")
    print("=" * 70)
    for cohort in COHORTS:
        coverages = [
            v["pct_coverage"]
            for v in pgx_coverage[cohort].values()
            if v and "pct_coverage" in v
        ]
        if coverages:
            print(f"  {cohort}: {min(coverages):.1f}%–{max(coverages):.1f}% "
                  f"(mean {sum(coverages)/len(coverages):.1f}%)")

    print()
    print("=" * 70)
    print("SHAP Top Feature per cohort/age_band/bin (CH_3, CH_4, CH_5)")
    print("=" * 70)
    for cohort in COHORTS:
        for ab in AGE_BANDS:
            for bin_name in DENSITY_BINS:
                feats = shap_features.get(cohort, {}).get(ab, {}).get(bin_name, [])
                if feats:
                    print(f"  {cohort}/{ab}/{bin_name}: {feats[0]['feature']} "
                          f"(|SHAP|={feats[0]['mean_abs_shap']:.4f})")

    print()
    print("=" * 70)
    print("FP-Growth Top Rule per cohort/age_band/bin (CH_3, CH_4)")
    print("=" * 70)
    for cohort in COHORTS:
        for ab in AGE_BANDS:
            for bin_name in DENSITY_BINS:
                fpg = visual_data.get(cohort,{}).get(ab,{}).get("bins",{}).get(bin_name,{}).get("fpgrowth") or {}
                rules = fpg.get("top_rules", [])
                if rules:
                    r = rules[0]
                    print(f"  {cohort}/{ab}/{bin_name}: {r['antecedents']} → {r['consequents']} "
                          f"(conf={r['confidence']}, lift={r['lift']})")


if __name__ == "__main__":
    main()
