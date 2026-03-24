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

from py_helpers.env_utils import get_workflow_python_bin
from py_helpers.event_density_utils import (
    DENSITY_BINS,
    cohort_aggregate_final_model_has_artifacts,
    final_model_bin_has_trained_artifacts,
)


def _age_band_fname(age_band: str) -> str:
    return age_band.replace("-", "_")


def _ensure_shap_artifacts(
    cohort: str,
    age_band: str,
    bin_name: str | None = None,
    *,
    skip_missing_bin: bool = False,
) -> None:
    """Run Step 7 (SHAP) if outputs are missing."""
    age_band_fname = _age_band_fname(age_band)
    if bin_name:
        out_dir = PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname / "bin_models" / bin_name
    else:
        out_dir = PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    required = out_dir / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    if required.exists():
        logger.info(f"SHAP artifacts already exist for {cohort}/{age_band}{f' bin={bin_name}' if bin_name else ''}; skipping Step 7.")
        return
    logger.info(f"Running Step 7 (SHAP) for {cohort}/{age_band}{f' bin={bin_name}' if bin_name else ''}...")
    script = PROJECT_ROOT / "7_shap_analysis" / "run_shap_analysis.py"
    cmd = [str(get_workflow_python_bin()), str(script), "--cohort", cohort, "--age_band", age_band]
    if bin_name:
        cmd.extend(["--bin", bin_name])
    if skip_missing_bin:
        cmd.append("--skip-missing-bin")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _load_shap_for_ffa(
    cohort: str, age_band: str, max_shap_rows: int = 5000, bin_name: str | None = None
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """Load SHAP global importance (map) and sample values from Step 7. Uses DuckDB to limit parquet rows on EC2."""
    import duckdb
    age_band_fname = _age_band_fname(age_band)
    if bin_name:
        base = PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname / "bin_models" / bin_name
    else:
        base = PROJECT_ROOT / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    csv_path = base / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    parquet_path = base / f"{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
    if not csv_path.exists():
        raise FileNotFoundError(f"SHAP global importance not found: {csv_path}")
    con = duckdb.connect()
    try:
        df_global = con.execute(f"SELECT feature, mean_abs_shap FROM read_csv_auto('{str(csv_path)}')").df()
    finally:
        con.close()
    if "feature" not in df_global.columns or "mean_abs_shap" not in df_global.columns:
        raise ValueError(f"Expected columns feature, mean_abs_shap in {csv_path}")
    shap_map = dict(zip(df_global["feature"], df_global["mean_abs_shap"].astype(float), strict=False))
    max_shap = max(shap_map.values()) if shap_map else 1.0
    if max_shap > 0:
        shap_map = {k: v / max_shap for k, v in shap_map.items()}
    shap_values_df = None
    if parquet_path.exists():
        con = duckdb.connect()
        try:
            # Limit rows for EC2 memory; exclude row_id/bias in SQL if possible
            shap_values_df = con.execute(
                f"SELECT * FROM read_parquet('{str(parquet_path)}') LIMIT {int(max_shap_rows)}"
            ).df()
        finally:
            con.close()
        for drop in ("row_id", "bias", "mi_person_key"):
            if drop in shap_values_df.columns:
                shap_values_df = shap_values_df.drop(columns=[drop])
        logger.info(f"Loaded SHAP sample values: {shap_values_df.shape}")
    else:
        logger.warning("SHAP sample parquet not found; FFA will use global SHAP only (no instance-level).")
    return shap_map, shap_values_df


def _find_xgboost_json(cohort: str, age_band: str, bin_name: str | None = None) -> Path:
    age_band_fname = _age_band_fname(age_band)
    _base = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
    _bin_base = _base / "bin_models" / bin_name if bin_name else _base
    candidates = [
        _bin_base / "final_model_json" / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
        _bin_base / "models" / "xgboost_model.json",
        _base / "final_model_json" / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
        _base / "models" / "xgboost_model.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"XGBoost JSON not found; tried: {candidates}")


def _find_xgboost_binary(cohort: str, age_band: str, bin_name: str | None = None) -> Optional[Path]:
    """Path to native binary (e.g. .ubj) for Booster.load_model; avoids JSON parse errors."""
    age_band_fname = _age_band_fname(age_band)
    _base = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
    _bin_base = _base / "bin_models" / bin_name if bin_name else _base
    candidates = [
        _bin_base / "models" / "xgboost_model.ubj",
        _bin_base / "models" / "xgboost_model.model",
        _base / "models" / "xgboost_model.ubj",
        PROJECT_ROOT / "6_final_model" / "model_outputs" / cohort / age_band_fname / "models" / "xgboost_model.ubj",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_test_data(cohort: str, age_band: str, max_rows: int = 2000, bin_name: str | None = None) -> Optional[pd.DataFrame]:
    """Load a sample of training features via DuckDB (avoids loading full CSV on EC2)."""
    import duckdb
    age_band_fname = _age_band_fname(age_band)
    csv_path = (
        PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
        / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    if not csv_path.exists():
        return None
    con = duckdb.connect()
    try:
        _bin_filter = f" WHERE n_event_bin = '{bin_name}'" if bin_name else ""
        df = con.execute(
            f"SELECT * FROM read_csv_auto('{str(csv_path)}'){_bin_filter} LIMIT {int(max_rows)}"
        ).df()
    finally:
        con.close()
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
    bin_name: str | None = None,
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

    # Generate and write per-instance AXP explanations when test data is available (required by combine)
    if X_test is not None and len(X_test) > 0:
        import xgboost as xgb
        feature_names = model_json.get("feature_names") or []
        if not feature_names:
            logger.warning("Model has no feature_names; cannot align X_test for AXP explanations.")
        else:
            # Align X_test to model feature order; fill missing columns with 0
            X_aligned = X_test.reindex(columns=feature_names, fill_value=0.0)
            missing_f = [c for c in feature_names if c not in X_test.columns]
            if missing_f:
                logger.warning("X_test missing %d model features (e.g. %s); filled with 0", len(missing_f), missing_f[:5])
            X_mat = np.asarray(X_aligned, dtype=np.float32)
            # Prefer native binary (.ubj) for Booster; JSON can fail with "Invalid cast, from Null to Object"
            booster = xgb.Booster()
            xgb_binary = _find_xgboost_binary(cohort, age_band, bin_name=bin_name)
            if xgb_binary is not None:
                booster.load_model(str(xgb_binary))
                logger.debug("Loaded XGBoost from binary: %s", xgb_binary)
            else:
                try:
                    booster.load_model(str(xgboost_json))
                except Exception as e:
                    logger.error(
                        "XGBoost failed to load from JSON (%s). Use native binary: "
                        "6_final_model/outputs/%s/%s/models/xgboost_model.ubj",
                        e, cohort, _age_band_fname(age_band),
                    )
                    raise
            dmat = xgb.DMatrix(X_mat, feature_names=feature_names)
            y_pred = (booster.predict(dmat) > 0.5).astype(int)
            df_axps = explainer.explain_dataset(X_mat, predictions=y_pred, show_progress=True)
            out_path = output_dir / "axp_explanations.parquet"
            df_axps.to_parquet(out_path, index=False)
            logger.info("Wrote %s (%d rows)", out_path, len(df_axps))

    return causal_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP + FFA workflow (PHTS-style) for one cohort/age_band.")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g. opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g. 13-24)")
    parser.add_argument(
        "--bin",
        default=None,
        metavar="BIN",
        help="Optional density bin (low|medium|high|extreme). Per-bin FFA under bin_models/{bin}/. "
        "Omit for cohort-level (aggregate) models only.",
    )
    parser.add_argument(
        "--skip-missing-bin",
        action="store_true",
        help="If --bin is set but Step 6 did not train that bin, exit 0 (or skip Step 7) instead of failing.",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Do not run Step 7 if SHAP missing (fail instead)")
    parser.add_argument("--skip-combine", action="store_true", help="Do not run combine step after FFA")
    parser.add_argument(
        "--max-shap-rows",
        type=int,
        default=5000,
        help="Max SHAP sample rows to load from parquet (DuckDB LIMIT). Default 5000 to reduce memory.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=2000,
        help="Max train feature rows to load for FFA (DuckDB LIMIT). Default 2000 to reduce memory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Combine step: parallel workers for patient explanations (0=auto from CPU count, 1=sequential). Always passed to combine.",
    )
    parser.add_argument(
        "--upload-to-dashboard",
        action="store_true",
        help="After combine, upload causal_data.json to S3 dashboard bucket (for GET /visualizations/causal).",
    )
    args = parser.parse_args()
    age_band_fname = _age_band_fname(args.age_band)
    bin_name: str | None = getattr(args, "bin", None)

    if bin_name is not None and bin_name not in DENSITY_BINS:
        parser.error(f"--bin must be one of {list(DENSITY_BINS)}, got {bin_name!r}")

    if bin_name:
        if not final_model_bin_has_trained_artifacts(PROJECT_ROOT, args.cohort, args.age_band, bin_name):
            if args.skip_missing_bin:
                logger.info(
                    "Skipping workflow: no Step 6 per-bin model for bin=%s (%s / %s).",
                    bin_name,
                    args.cohort,
                    args.age_band,
                )
                sys.exit(0)
            logger.error(
                "No Step 6 per-bin model for bin=%s (%s / %s). Run 6_final_model or use --skip-missing-bin.",
                bin_name,
                args.cohort,
                args.age_band,
            )
            sys.exit(1)
    elif not cohort_aggregate_final_model_has_artifacts(PROJECT_ROOT, args.cohort, args.age_band):
        logger.error(
            "No cohort-level Step 6 models for %s / %s. Use --bin <density_bin> for per-bin FFA or train aggregate Step 6.",
            args.cohort,
            args.age_band,
        )
        sys.exit(1)

    if not args.skip_shap:
        _ensure_shap_artifacts(
            args.cohort,
            args.age_band,
            bin_name=bin_name,
            skip_missing_bin=args.skip_missing_bin,
        )

    shap_map, shap_values_df = _load_shap_for_ffa(
        args.cohort, args.age_band, max_shap_rows=args.max_shap_rows, bin_name=bin_name
    )
    xgboost_json = _find_xgboost_json(args.cohort, args.age_band, bin_name=bin_name)
    X_test = _load_test_data(args.cohort, args.age_band, max_rows=args.max_test_rows, bin_name=bin_name)
    if X_test is None or len(X_test) == 0:
        logger.error(
            "Required test data missing: cannot generate axp_explanations.parquet. "
            "Expected 6_final_model/outputs/%s/%s/%s_%s_train_final_features_no_leakage.csv%s",
            args.cohort, age_band_fname, args.cohort, age_band_fname,
            f" (bin={bin_name})" if bin_name else "",
        )
        sys.exit(1)

    if bin_name:
        ffa_out_base = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / args.cohort / age_band_fname / "bin_models" / bin_name / "xgboost"
    else:
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
        bin_name=bin_name,
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
        causal_visuals_dir = PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "causal"
        combine_cmd = [
            str(get_workflow_python_bin()),
            str(combine_script),
            "--cohort",
            args.cohort,
            "--age-band",
            args.age_band,
            "--output-dir",
            str(causal_visuals_dir),
            "--workers",
            str(args.workers),
        ]
        if bin_name:
            combine_cmd.extend(["--bin", bin_name])
        if getattr(args, "upload_to_dashboard", False):
            combine_cmd.append("--upload-to-dashboard")
        r = subprocess.run(combine_cmd, cwd=Path(__file__).parent)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    logger.info("SHAP + FFA workflow done.")


if __name__ == "__main__":
    main()
