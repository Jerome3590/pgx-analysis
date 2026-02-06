#!/usr/bin/env python3
"""
Run SHAP + FFA workflow for a single cohort/age_band (PHTS-style).

Uses PHTS risk calculator pattern:
1. Ensure Step 7 (SHAP) artifacts exist; run 7_shap_analysis if missing.
2. Load XGBoost JSON and SHAP from Step 6/7; run FFA (rule extraction + SHAP filtering).
3. Write FFA outputs to 8_ffa_analysis/outputs for combine.
4. Run combine_shap_ffa_results to produce dashboard_data.json etc.

Usage:
    python run_shap_ffa_workflow.py --cohort opioid_ed --age-band 13-24
"""

import argparse
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 8_ffa_analysis for FFA explainer and ffa_utils
FFA_DIR = PROJECT_ROOT / "8_ffa_analysis"
if str(FFA_DIR) not in sys.path:
    sys.path.insert(0, str(FFA_DIR))


def _age_band_fname(age_band: str) -> str:
    return age_band.replace("-", "_")


def _ensure_shap_artifacts(cohort: str, age_band: str) -> None:
    """Run Step 7 (SHAP) if outputs are missing."""
    age_band_fname = _age_band_fname(age_band)
    out_dir = PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    required = out_dir / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    if required.exists():
        logger.info(f"SHAP artifacts already exist for {cohort}/{age_band}; skipping Step 7.")
        return
    logger.info(f"Running Step 7 (SHAP) for {cohort}/{age_band}...")
    script = PROJECT_ROOT / "7_shap_analysis" / "run_shap_analysis.py"
    r = subprocess.run(
        [sys.executable, str(script), "--cohort", cohort, "--age_band", age_band],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _load_shap_for_ffa(cohort: str, age_band: str) -> Tuple[dict, Optional[pd.DataFrame]]:
    """Load SHAP global importance (map) and sample values (DataFrame) from Step 7."""
    age_band_fname = _age_band_fname(age_band)
    base = PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    csv_path = base / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    parquet_path = base / f"{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
    if not csv_path.exists():
        raise FileNotFoundError(f"SHAP global importance not found: {csv_path}")
    df_global = pd.read_csv(csv_path)
    if "feature" not in df_global.columns or "mean_abs_shap" not in df_global.columns:
        raise ValueError(f"Expected columns feature, mean_abs_shap in {csv_path}")
    shap_map = dict(zip(df_global["feature"], df_global["mean_abs_shap"].astype(float), strict=False))
    max_shap = max(shap_map.values()) if shap_map else 1.0
    if max_shap > 0:
        shap_map = {k: v / max_shap for k, v in shap_map.items()}
    shap_values_df = None
    if parquet_path.exists():
        shap_values_df = pd.read_parquet(parquet_path)
        # Drop row id / bias if present so columns = feature SHAP values
        for drop in ("row_id", "bias", "mi_person_key"):
            if drop in shap_values_df.columns:
                shap_values_df = shap_values_df.drop(columns=[drop])
        logger.info(f"Loaded SHAP sample values: {shap_values_df.shape}")
    else:
        logger.warning("SHAP sample parquet not found; FFA will use global SHAP only (no instance-level).")
    return shap_map, shap_values_df


def _find_xgboost_json(cohort: str, age_band: str) -> Path:
    age_band_fname = _age_band_fname(age_band)
    candidates = [
        PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname / "final_model_json"
        / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
        PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname / "models" / "xgboost_model.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"XGBoost JSON not found; tried: {candidates}")


def _load_test_data(cohort: str, age_band: str) -> Optional[pd.DataFrame]:
    """Load training (or test) features for rule application; optional."""
    age_band_fname = _age_band_fname(age_band)
    csv_path = (
        PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
        / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, nrows=2000)
    drop = ["mi_person_key", "target"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[numeric].copy()


def _run_ffa_with_shap(
    cohort: str,
    age_band: str,
    xgboost_json: Path,
    shap_map: dict,
    shap_values_df: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Run FFA: load JSON, fit explainer, extract rule-feature counts, build causal_df."""
    from ffa_utils import load_model_json, extract_feature_mappings
    from xgboost_axp_explainer import XGBoostSymbolicExplainer, PathConfig

    model_json = load_model_json(xgboost_json)
    extract_feature_mappings(model_json)

    if shap_values_df is None or len(shap_values_df) == 0:
        # Build a minimal per-instance SHAP df from global map so explainer accepts it
        features = list(shap_map.keys())
        shap_values_df = pd.DataFrame(
            np.tile(np.array([list(shap_map.values())]), (min(100, len(shap_map)), 1)),
            columns=features,
        )
        logger.warning("Using global SHAP repeated as proxy for instance SHAP (no sample parquet).")

    path_config = PathConfig(
        model_path=str(xgboost_json),
        data_dir=str(output_dir),
        output_dir=str(output_dir),
        tree_rules_path=None,
        age_band=age_band,
    )
    explainer = XGBoostSymbolicExplainer(
        path_config=path_config,
        shap_importance_map=shap_map,
        shap_values_df=shap_values_df,
    )
    if "feature_names" in model_json and model_json["feature_names"]:
        explainer.feature_names = {i: n for i, n in enumerate(model_json["feature_names"])}
    explainer.model_json = model_json
    explainer.fit_from_model_json(model_json)
    logger.info(f"Explainer fitted: {len(explainer.rule_clauses)} rules")

    rule_feature_counts = defaultdict(int)
    for clause in explainer.rule_clauses:
        if not clause:
            continue
        for lit in clause:
            if lit in getattr(explainer, "id_condition_map", {}):
                feat_idx, _, _ = explainer.id_condition_map[lit]
                feat_name = (explainer.feature_names or {}).get(feat_idx, f"feature_{feat_idx}")
                rule_feature_counts[feat_name] += 1

    total_rule_firings = sum(rule_feature_counts.values()) or 1
    causal_results = []
    for feature, rule_count in rule_feature_counts.items():
        shap_importance = shap_map.get(feature, 0.0)
        causal_responsibility = (rule_count / total_rule_firings) * shap_importance
        causal_results.append({
            "feature": feature,
            "causal_responsibility": causal_responsibility,
            "shap_importance": shap_importance,
            "rule_frequency": rule_count,
            "total_rules": len(explainer.rule_clauses),
        })
    causal_df = pd.DataFrame(causal_results).sort_values("causal_responsibility", ascending=False)
    return causal_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP + FFA workflow (PHTS-style) for one cohort/age_band.")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g. opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g. 13-24)")
    parser.add_argument("--skip-shap", action="store_true", help="Do not run Step 7 if SHAP missing (fail instead)")
    parser.add_argument("--skip-combine", action="store_true", help="Do not run combine step after FFA")
    args = parser.parse_args()
    age_band_fname = _age_band_fname(args.age_band)

    if not args.skip_shap:
        _ensure_shap_artifacts(args.cohort, args.age_band)

    shap_map, shap_values_df = _load_shap_for_ffa(args.cohort, args.age_band)
    xgboost_json = _find_xgboost_json(args.cohort, args.age_band)
    X_test = _load_test_data(args.cohort, args.age_band)

    ffa_out_base = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / args.cohort / age_band_fname / "xgboost"
    ffa_out_base.mkdir(parents=True, exist_ok=True)

    causal_df = _run_ffa_with_shap(
        args.cohort,
        args.age_band,
        xgboost_json,
        shap_map,
        shap_values_df,
        X_test,
        ffa_out_base,
    )

    # Write FFA outputs so combine_shap_ffa_results finds them
    importance_df = causal_df.copy()
    importance_df["importance"] = importance_df["causal_responsibility"]
    importance_path = ffa_out_base / "feature_importance_axp.parquet"
    importance_df.to_parquet(importance_path, index=False)
    logger.info(f"Wrote {importance_path}")
    causal_csv = ffa_out_base.parent / "ffa_causal_factors.csv"
    causal_df.to_csv(causal_csv, index=False)
    logger.info(f"Wrote {causal_csv}")

    if not args.skip_combine:
        combine_script = Path(__file__).parent / "combine_shap_ffa_results.py"
        dashboard_out = PROJECT_ROOT / "9_risk_dashboard" / "outputs"
        r = subprocess.run(
            [
                sys.executable,
                str(combine_script),
                "--cohort",
                args.cohort,
                "--age-band",
                args.age_band,
                "--output-dir",
                str(dashboard_out),
            ],
            cwd=Path(__file__).parent,
        )
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    logger.info("SHAP + FFA workflow done.")


if __name__ == "__main__":
    main()
