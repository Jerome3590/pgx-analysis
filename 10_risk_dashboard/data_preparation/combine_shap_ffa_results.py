#!/usr/bin/env python3
"""
Combine SHAP and FFA Results for Final Reporting

This script aggregates and combines SHAP and FFA analysis results from Steps 7 and 8
to create comprehensive patient-level explanations. Note: Consensus is already reflected
in FFA's causal importance scores, which use SHAP-prioritized rules.

Usage:
    python 10_risk_dashboard/data_preparation/combine_shap_ffa_results.py \\
        --cohort opioid_ed --age-band 25-44
    # Writes to 10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json (EC2 path).
    # Upload to S3: visualizations/causal/{cohort}/{age_band}/causal_data.json (use --upload-to-dashboard or upload_causal_outputs_to_s3.py).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import json
import ast
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Add project root to path
# This script is in 10_risk_dashboard/data_preparation/
# Project root is 3 levels up
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_shap_results(cohort: str, age_band: str, project_root: Path, bin_name: str | None = None) -> Optional[Path]:
    """Find SHAP results from Step 7 (7_shap_analysis). Prefer global importance CSV for combine."""
    age_band_fname = age_band.replace("-", "_")
    base = f"{cohort}_{age_band_fname}"
    shap_base = project_root / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    if bin_name:
        shap_dir = shap_base / "bin_models" / bin_name
    else:
        shap_dir = shap_base
    possible_paths = [
        shap_dir / f"{base}_shap_global_importance_xgboost.csv",
        shap_dir / f"{base}_shap_global_importance_catboost.csv",
        # Legacy / other layouts
        project_root / "8_final_model" / "outputs" / cohort / age_band_fname / "shap_values.npy",
        project_root / "8_final_model" / "outputs" / cohort / age_band_fname / "shap_feature_importance.csv",
    ]
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found SHAP results: {path}")
            return path
    logger.warning("SHAP results not found - will skip SHAP analysis")
    return None


def find_shap_sample_parquet(cohort: str, age_band: str, project_root: Path, bin_name: str | None = None) -> Optional[Path]:
    """Find SHAP sample values parquet from Step 7 (same layout as global importance). Used for row-level SHAP in patient explanations."""
    age_band_fname = age_band.replace("-", "_")
    base = f"{cohort}_{age_band_fname}"
    shap_base = project_root / "7_shap_analysis" / "outputs" / cohort / age_band_fname
    shap_dir = shap_base / "bin_models" / bin_name if bin_name else shap_base
    for name in (f"{base}_shap_sample_values_xgboost.parquet", f"{base}_shap_sample_values_catboost.parquet"):
        p = shap_dir / name
        if p.exists():
            return p
    return None


def find_ffa_results(cohort: str, age_band: str, project_root: Path, bin_name: str | None = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Find FFA results from Step 8 (8_ffa_analysis). PGx uses parquet under xgboost/ or catboost/.
    Tries both age_band dir names: underscore (65_74) and hyphen (65-74) for cohort naming consistency."""
    age_band_fname = age_band.replace("-", "_")
    # Try underscore first (what run_shap_ffa_workflow uses), then hyphen (some S3/other scripts use it)
    candidates = [age_band_fname, age_band]
    if age_band_fname == age_band:
        candidates = [age_band_fname]
    explanations_path = None
    importance_path = None
    for age_dir in candidates:
        _age_base = project_root / "8_ffa_analysis" / "outputs" / cohort / age_dir
        ffa_base = _age_base / "bin_models" / bin_name if bin_name else _age_base
        for model in ("xgboost", "catboost"):
            model_dir = ffa_base / model
            exp_p = model_dir / "axp_explanations.parquet"
            if not exp_p.exists():
                exp_p = model_dir / "axp_explanations.csv"
            imp_p = model_dir / "feature_importance_axp.parquet"
            if not imp_p.exists():
                imp_p = model_dir / "feature_importance_axp.csv"
            if exp_p.exists():
                explanations_path = exp_p
                logger.info(f"Found FFA explanations: {explanations_path}")
                break
            if imp_p.exists() and importance_path is None:
                importance_path = imp_p
        if explanations_path is not None:
            break
    if explanations_path is None:
        logger.warning(
            "FFA explanations not found (looked under 8_ffa_analysis/outputs/%s/{%s|%s}/xgboost|catboost/axp_explanations.*). "
            "Patient explanations will be skipped. Run Step 1b (SHAP+FFA workflow) or full 8_ffa_analysis for this cohort/age_band to generate them.",
            cohort, age_band_fname, age_band,
        )
    if importance_path is None:
        for age_dir in candidates:
            _age_base = project_root / "8_ffa_analysis" / "outputs" / cohort / age_dir
            ffa_base = _age_base / "bin_models" / bin_name if bin_name else _age_base
            for model in ("xgboost", "catboost"):
                imp_p = ffa_base / model / "feature_importance_axp.parquet"
                if not imp_p.exists():
                    imp_p = ffa_base / model / "feature_importance_axp.csv"
                if imp_p.exists():
                    importance_path = imp_p
                    logger.info(f"Found FFA importance: {importance_path}")
                    break
            if importance_path is not None:
                break
    if importance_path is None:
        logger.warning("FFA importance not found")
    return explanations_path, importance_path


def load_shap_data(shap_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """Load SHAP values and/or importance. Step 7 global importance CSV has feature, mean_abs_shap. Uses DuckDB for CSV."""
    if shap_path.suffix == '.npy':
        shap_values = np.load(shap_path)
        return shap_values, None
    if shap_path.suffix == '.csv':
        import duckdb
        con = duckdb.connect()
        try:
            df = con.execute(f"SELECT * FROM read_csv_auto('{str(shap_path)}')").df()
        finally:
            con.close()
        # Importance table (feature + importance column) — use as importance, no row-level SHAP
        if any(c in df.columns for c in ('feature', 'shap_value', 'importance', 'mean_abs_shap')):
            if 'feature' not in df.columns and 'mean_abs_shap' in df.columns:
                df = df.rename(columns={df.columns[0]: 'feature'})
            return None, df
        shap_values = df.values
        return shap_values, None
    raise ValueError(f"Unsupported SHAP file format: {shap_path.suffix}")


def load_shap_sample_parquet(parquet_path: Path, feature_names: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """Load row-level SHAP values from Step 7 sample parquet. Drops row_id, bias, mi_person_key. Returns (n_samples, n_features) array."""
    import duckdb
    con = duckdb.connect()
    try:
        path_esc = str(parquet_path).replace("'", "''")
        df = con.execute(f"SELECT * FROM read_parquet('{path_esc}')").df()
    finally:
        con.close()
    drop = ["row_id", "bias", "mi_person_key"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    if feature_names:
        cols = [c for c in feature_names if c in df.columns]
        if cols:
            df = df[cols]
    # Ensure numeric and consistent column order
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return None
    return df[numeric].values


def extract_features_from_ffa_rules(rules: List) -> Set[str]:
    """Extract feature names from FFA rule strings."""
    features = set()
    
    if isinstance(rules, str):
        try:
            rules = ast.literal_eval(rules)
        except:
            # Try regex parsing
            import re
            feature_matches = re.findall(r'(\w+)\s*[><=]', rules)
            features.update(feature_matches)
            return features
    
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, str):
                import re
                feature_matches = re.findall(r'(\w+)\s*[><=]', rule)
                features.update(feature_matches)
    
    return features


def _process_patient_chunk(
    args: Tuple[int, pd.DataFrame, Optional[np.ndarray], List[str], Set[str]]
) -> List[Dict]:
    """Process a chunk of patients for parallel explanation generation (top-level for pickling)."""
    start_offset, ffa_chunk, shap_values, feature_names, consensus_set = args
    results = []
    for local_i, (idx, row) in enumerate(ffa_chunk.iterrows()):
        row_pos = start_offset + local_i
        patient_explanation = {
            "patient_index": idx,
            "patient_id": row.get("instance_id", idx),
        }
        if "axp" in row:
            matched_rules = row["axp"]
            ffa_features = extract_features_from_ffa_rules(matched_rules)
            patient_explanation["ffa_matched_rules"] = str(matched_rules)
            patient_explanation["ffa_features"] = list(ffa_features)
            patient_explanation["ffa_rule_count"] = (
                len(matched_rules) if isinstance(matched_rules, list) else 1
            )
        else:
            patient_explanation["ffa_features"] = []
            patient_explanation["ffa_rule_count"] = 0
        if shap_values is not None and row_pos < len(shap_values):
            patient_shap = shap_values[row_pos]
            shap_df = pd.DataFrame(
                {
                    "feature": feature_names[: len(patient_shap)],
                    "shap_value": patient_shap,
                }
            ).sort_values("shap_value", ascending=False)
            patient_explanation["shap_top_positive"] = shap_df.head(5)["feature"].tolist()
            patient_explanation["shap_top_negative"] = shap_df.tail(5)["feature"].tolist()
            patient_explanation["shap_total"] = float(patient_shap.sum())
            shap_top_set = set(shap_df.head(10)["feature"].values)
            ffa_features_set = set(patient_explanation.get("ffa_features", []))
            patient_feature_set = shap_top_set | ffa_features_set
            patient_explanation["consensus_features"] = sorted(
                consensus_set & patient_feature_set
            )
        else:
            patient_explanation["shap_top_positive"] = []
            patient_explanation["shap_top_negative"] = []
            patient_explanation["shap_total"] = None
            ffa_features_set = set(patient_explanation.get("ffa_features", []))
            patient_explanation["consensus_features"] = sorted(
                consensus_set & ffa_features_set
            )
        results.append(patient_explanation)
    return results


def calculate_consensus_features(
    shap_importance: pd.DataFrame,
    ffa_importance: pd.DataFrame,
    top_k: int = 20
) -> Dict[str, any]:
    """Calculate consensus features between SHAP and FFA."""
    if shap_importance is None or ffa_importance is None:
        return {
            'consensus_features': [],
            'shap_only': [],
            'ffa_only': [],
            'consensus_count': 0
        }
    
    # Get top K features from each
    shap_col = 'importance' if 'importance' in shap_importance.columns else shap_importance.columns[1]
    ffa_col = 'importance' if 'importance' in ffa_importance.columns else ffa_importance.columns[1]
    
    shap_top = set(shap_importance.head(top_k)['feature'].values)
    ffa_top = set(ffa_importance.head(top_k)['feature'].values)
    
    consensus = shap_top.intersection(ffa_top)
    shap_only = shap_top - ffa_top
    ffa_only = ffa_top - shap_top
    
    return {
        'consensus_features': sorted(list(consensus)),
        'shap_only': sorted(list(shap_only)),
        'ffa_only': sorted(list(ffa_only)),
        'consensus_count': len(consensus),
        'shap_count': len(shap_top),
        'ffa_count': len(ffa_top),
        'consensus_rate': len(consensus) / top_k if top_k > 0 else 0
    }


def combine_importance_scores(
    shap_importance: Optional[pd.DataFrame],
    ffa_importance: Optional[pd.DataFrame],
    weight_shap: float = 0.5,
    weight_ffa: float = 0.5
) -> pd.DataFrame:
    """Combine SHAP and FFA importance scores."""
    if shap_importance is None and ffa_importance is None:
        return pd.DataFrame()

    # Single source: still produce combined_importance, shap_norm, ffa_norm for report
    if shap_importance is None:
        df = ffa_importance.copy()
        col = 'importance' if 'importance' in df.columns else df.columns[1]
        norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        df['shap_norm'] = 0.0
        df['ffa_norm'] = norm.values
        df['combined_importance'] = weight_ffa * df['ffa_norm']
        return df.sort_values('combined_importance', ascending=False)
    if ffa_importance is None:
        df = shap_importance.copy()
        col = 'importance' if 'importance' in df.columns else df.columns[1]
        norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        df['shap_norm'] = norm.values
        df['ffa_norm'] = 0.0
        df['combined_importance'] = weight_shap * df['shap_norm']
        return df.sort_values('combined_importance', ascending=False)

    # Normalize both to [0, 1]
    shap_col = 'importance' if 'importance' in shap_importance.columns else shap_importance.columns[1]
    ffa_col = 'importance' if 'importance' in ffa_importance.columns else ffa_importance.columns[1]
    
    shap_norm = (shap_importance[shap_col] - shap_importance[shap_col].min()) / \
                (shap_importance[shap_col].max() - shap_importance[shap_col].min() + 1e-10)
    
    ffa_norm = (ffa_importance[ffa_col] - ffa_importance[ffa_col].min()) / \
               (ffa_importance[ffa_col].max() - ffa_importance[ffa_col].min() + 1e-10)
    
    # Merge
    combined = shap_importance[['feature']].merge(
        ffa_importance[['feature', ffa_col]],
        on='feature',
        how='outer',
        suffixes=('_shap', '_ffa')
    )
    
    # Add normalized scores
    shap_dict = dict(zip(shap_importance['feature'], shap_norm))
    ffa_dict = dict(zip(ffa_importance['feature'], ffa_norm))
    
    combined['shap_norm'] = combined['feature'].map(shap_dict).fillna(0)
    combined['ffa_norm'] = combined['feature'].map(ffa_dict).fillna(0)
    
    # Weighted combination
    combined['combined_importance'] = (
        weight_shap * combined['shap_norm'] + 
        weight_ffa * combined['ffa_norm']
    )
    
    return combined.sort_values('combined_importance', ascending=False)


def generate_patient_explanations(
    shap_values: Optional[np.ndarray],
    ffa_explanations: pd.DataFrame,
    feature_names: List[str],
    n_samples: int = 0,
    global_consensus_features: Optional[List[str]] = None,
    n_workers: int = 0,
) -> pd.DataFrame:
    """Generate comprehensive patient-level explanations combining SHAP and FFA.

    For each patient, consensus_features is the set of global consensus features
    that appear in this patient's explanation (SHAP top-10 or FFA rules), so the
    report metric 'Patients with consensus features' counts patients who have
    at least one high-confidence (global consensus) feature.

    n_workers: 0 = auto (CPU count - 1, min 1), 1 = sequential, >1 = parallel chunks.
    """
    consensus_set = set(global_consensus_features or [])
    sample_size = len(ffa_explanations) if n_samples <= 0 else min(n_samples, len(ffa_explanations))
    ffa_sample = ffa_explanations.head(sample_size)

    workers = n_workers if n_workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    use_parallel = workers > 1 and sample_size > 0

    if not use_parallel:
        chunks_args = [(0, ffa_sample, shap_values, feature_names, consensus_set)]
        results = _process_patient_chunk(chunks_args[0])
    else:
        chunk_size = max(1, (sample_size + workers - 1) // workers)
        chunks_args = []
        for start in range(0, sample_size, chunk_size):
            end = min(start + chunk_size, sample_size)
            chunks_args.append(
                (start, ffa_sample.iloc[start:end].copy(), shap_values, feature_names, consensus_set)
            )
        logger.info(f"Patient explanations: {sample_size} patients, {len(chunks_args)} chunks, {workers} workers")
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk_results in executor.map(_process_patient_chunk, chunks_args):
                results.extend(chunk_results)

    return pd.DataFrame(results)


def _feature_type_counts(combined_importance: pd.DataFrame) -> Dict[str, int]:
    """Count features by prefix: item_drug_, item_icd_, item_cpt_, other."""
    counts = {"drug": 0, "icd": 0, "cpt": 0, "other": 0}
    if combined_importance.empty or "feature" not in combined_importance.columns:
        return counts
    for f in combined_importance["feature"].astype(str):
        if f.startswith("item_drug_"):
            counts["drug"] += 1
        elif f.startswith("item_icd_"):
            counts["icd"] += 1
        elif f.startswith("item_cpt_"):
            counts["cpt"] += 1
        else:
            counts["other"] += 1
    return counts


def generate_summary_report(
    consensus_data: Dict,
    combined_importance: pd.DataFrame,
    patient_explanations: pd.DataFrame,
    cohort: Optional[str] = None,
    age_band: Optional[str] = None,
) -> str:
    """Generate a human-readable summary report. If cohort/age_band given, include feature-type check by design."""
    report = []
    report.append("="*80)
    report.append("SHAP + FFA COMBINED ANALYSIS SUMMARY")
    if cohort and age_band:
        report.append(f"  Cohort: {cohort} / {age_band}")
    report.append("="*80)
    report.append("")
    
    # Feature types by cohort (design: opioid_ed = Drug+ICD+CPT, non_opioid_ed = Drug only)
    if not combined_importance.empty and cohort:
        counts = _feature_type_counts(combined_importance)
        report.append("FEATURE TYPES (combined importance):")
        report.append(f"  drug: {counts['drug']}, icd: {counts['icd']}, cpt: {counts['cpt']}, other: {counts['other']}")
        if cohort == "opioid_ed":
            expect = "Drug + ICD + CPT"
            ok = counts["drug"] > 0 and counts["icd"] > 0 and counts["cpt"] > 0
        elif cohort == "non_opioid_ed":
            expect = "Drug only"
            ok = counts["drug"] > 0 and counts["icd"] == 0 and counts["cpt"] == 0
        else:
            expect = ""
            ok = True
        if expect:
            status = "OK" if ok else "CHECK (expected " + expect + ")"
            report.append(f"  Expected: {expect}  [{status}]")
        total_feats = counts["drug"] + counts["icd"] + counts["cpt"] + counts["other"]
        if total_feats <= 5 or (cohort == "non_opioid_ed" and counts["drug"] == 0):
            report.append("")
            report.append(
                "  *** WARNING: Very few or no item features (model may have only n_events + PGx). "
                "Re-run Step 3b and Step 6 for this cohort/age_band; Step 6 will fall back to "
                "distinct drugs from model_events if Step 3b yields no drug codes."
            )
        report.append("")

    # Consensus summary
    report.append("CONSENSUS FEATURES:")
    report.append(f"  - Consensus features: {consensus_data['consensus_count']}")
    report.append(f"  - SHAP-only features: {len(consensus_data['shap_only'])}")
    report.append(f"  - FFA-only features: {len(consensus_data['ffa_only'])}")
    report.append(f"  - Consensus rate: {consensus_data['consensus_rate']:.1%}")
    report.append("")
    
    if consensus_data['consensus_features']:
        report.append("  High-confidence features (consensus):")
        for feat in consensus_data['consensus_features'][:10]:
            report.append(f"    - {feat}")
    report.append("")
    
    # Combined importance summary
    if not combined_importance.empty:
        report.append("COMBINED FEATURE IMPORTANCE (Top 10):")
        top_features = combined_importance.head(10)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feat = row.get('feature', '')
            comb = row.get('combined_importance', 0.0)
            sn = row.get('shap_norm', 0.0)
            fn = row.get('ffa_norm', 0.0)
            report.append(f"  {i}. {feat}: {comb:.4f} (SHAP: {sn:.3f}, FFA: {fn:.3f})")
        # For opioid_ed, show top ICD and CPT so output clearly shows Drug+ICD+CPT
        if cohort == "opioid_ed" and "feature" in combined_importance.columns:
            icd_rows = combined_importance[combined_importance["feature"].astype(str).str.startswith("item_icd_")].head(3)
            cpt_rows = combined_importance[combined_importance["feature"].astype(str).str.startswith("item_cpt_")].head(3)
            if not icd_rows.empty or not cpt_rows.empty:
                report.append("  Top ICD/CPT in combined importance:")
                for _, row in icd_rows.iterrows():
                    report.append(f"    - {row.get('feature', '')}: {row.get('combined_importance', 0):.4f}")
                for _, row in cpt_rows.iterrows():
                    report.append(f"    - {row.get('feature', '')}: {row.get('combined_importance', 0):.4f}")
        report.append("")
    
    # Patient explanation summary (patient_explanations is None when FFA explanations are missing)
    if patient_explanations is not None:
        if not patient_explanations.empty:
            report.append("PATIENT EXPLANATIONS:")
            report.append(f"  - Total patients analyzed: {len(patient_explanations)}")
            if 'consensus_features' in patient_explanations.columns:
                patients_with_consensus = patient_explanations[
                    patient_explanations['consensus_features'].apply(lambda x: len(x) > 0)
                ]
                report.append(f"  - Patients with consensus features: {len(patients_with_consensus)} "
                             f"({len(patients_with_consensus)/len(patient_explanations):.1%})")
            report.append("")
        # else: empty DataFrame, skip section
    # else: None (FFA explanations not produced), skip section
    
    report.append("="*80)
    
    return "\n".join(report)


def generate_dashboard_outputs_phts_style(
    combined_importance: pd.DataFrame,
    output_dir: Path,
    cohort: str,
    age_band: str,
    top_k: int = 20,
) -> Dict:
    """
    Generate PHTS-style dashboard_data.json and top_causal_factors.csv
    so dashboard and Lambda can consume the same structure as PHTS.
    """
    if combined_importance.empty:
        logger.warning("No combined importance; skipping PHTS-style dashboard outputs")
        return {}
    # Normalize combined_importance for summary stats (like PHTS combined_importance_norm)
    col = "combined_importance" if "combined_importance" in combined_importance.columns else combined_importance.columns[1]
    vals = combined_importance[col].fillna(0)
    mn, mx = vals.min(), vals.max()
    combined_importance = combined_importance.copy()
    combined_importance["combined_importance_norm"] = (vals - mn) / (mx - mn + 1e-10)
    # Include all features with importance > 0 in JSON (dashboard filter: Top 10 / Top 20 / All)
    combined_importance = combined_importance[vals > 0].sort_values(col, ascending=False)
    all_causal = combined_importance.copy()
    all_causal = all_causal.rename(columns={"combined_importance_norm": "causal_responsibility"})
    if "causal_responsibility" not in all_causal.columns:
        all_causal["causal_responsibility"] = all_causal.get("combined_importance", all_causal.iloc[:, 1])
    all_causal["shap_importance"] = all_causal.get("shap_norm", all_causal["causal_responsibility"])
    all_causal["rule_frequency"] = 0
    all_causal["total_rules"] = 0
    # Top K for CSV / summary (backward compat)
    top_causal = all_causal.head(top_k).copy()
    # Summary (PHTS format)
    combined_filtered = combined_importance
    summary = {
        "total_features": len(combined_filtered),
        "top_k": top_k,
        "mean_importance": float(combined_filtered["combined_importance_norm"].mean()),
        "max_importance": float(combined_filtered["combined_importance_norm"].max()),
        "top_feature": all_causal.iloc[0]["feature"] if len(all_causal) > 0 else None,
        "top_feature_importance": float(all_causal.iloc[0]["causal_responsibility"]) if len(all_causal) > 0 else None,
    }
    dashboard_data = {
        "cohort": cohort,
        "age_band": age_band,
        "timestamp": datetime.now().isoformat(),
        "ffa_method": "shap_ffa_combined",
        "top_causal_factors": all_causal.to_dict("records"),
        "summary": summary,
        "feature_importance": combined_filtered.head(50).to_dict("records"),
        "notes": {
            "source": "combine_shap_ffa_results (PGx)",
            "shap_source": "7_shap_analysis",
            "ffa_source": "8_ffa_analysis",
        },
    }
    json_path = output_dir / "dashboard_data.json"
    with open(json_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    logger.info(f"Saved dashboard_data.json (PHTS-style) to {json_path}")
    # Caller may upload this to dashboard S3 via --upload-to-dashboard (same shape as causal_data.json)
    csv_path = output_dir / "top_causal_factors.csv"
    top_causal.to_csv(csv_path, index=False)
    logger.info(f"Saved top_causal_factors.csv to {csv_path}")
    combined_shap_path = output_dir / "combined_shap_importance.csv"
    combined_importance.to_csv(combined_shap_path, index=False)
    logger.info(f"Saved combined_shap_importance.csv to {combined_shap_path}")
    return dashboard_data


def upload_causal_data_to_dashboard(json_path: Path, cohort: str, age_band: str, bin_name: str | None = None) -> bool:
    """Upload dashboard_data.json to S3 dashboard bucket as causal_data.json for GET /visualizations/causal.
    S3 paths use age_band with hyphen (e.g. 25-44); EC2 paths use underscore (25_44)."""
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not available; skipping upload to dashboard S3")
        return False
    bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    prefix = (os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator") or "").strip("/")
    _bin_seg = f"/{bin_name}" if bin_name else ""
    key = f"{prefix}/visualizations/causal/{cohort}/{age_band}{_bin_seg}/causal_data.json"
    try:
        s3 = boto3.client("s3")
        s3.upload_file(
            str(json_path),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/json"},
        )
        logger.info("Uploaded causal_data.json to s3://%s/%s", bucket, key)
        return True
    except Exception as e:
        logger.warning("Failed to upload causal_data.json to S3: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Combine SHAP and FFA results for final reporting")
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument(
        "--bin",
        required=True,
        choices=["low", "medium", "high", "extreme"],
        help="Density bin. Reads SHAP/FFA from per-bin subdirs and writes output to {output_dir}/{cohort}/{age_band}/{bin}/.",
    )
    parser.add_argument("--output-dir", default="10_risk_dashboard/visualizations/causal", help="Output directory (EC2: .../visualizations/causal/{cohort}/{age_band_fname}/)")
    parser.add_argument("--top-k", type=int, default=20, help="Top K features for consensus")
    parser.add_argument("--weight-shap", type=float, default=0.5, help="Weight for SHAP (0-1)")
    parser.add_argument("--weight-ffa", type=float, default=0.5, help="Weight for FFA (0-1)")
    parser.add_argument("--n-patients", type=int, default=0, help="Number of patients to analyze (0 = all)")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for patient explanations (0=auto from CPU count, 1=sequential)",
    )
    parser.add_argument("--all-cohorts", action="store_true", help="Process all cohorts")
    parser.add_argument(
        "--upload-to-dashboard",
        action="store_true",
        help="Upload dashboard_data.json to S3 dashboard bucket as visualizations/causal/{cohort}/{age_band}/causal_data.json (set S3_DASHBOARD_BUCKET, S3_DASHBOARD_PREFIX)",
    )
    args = parser.parse_args()
    
    # This script is in 10_risk_dashboard/data_preparation/; project root is 3 levels up
    project_root = Path(__file__).parent.parent.parent
    
    if args.all_cohorts:
        # Process all cohorts (implement as needed)
        logger.info("Processing all cohorts...")
        # TODO: Implement batch processing
        return
    
    _bin_name: str | None = getattr(args, "bin", None)
    # EC2 path: 10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}[/{bin}]/ (README_dashboard_validation.md)
    _out_base = Path(args.output_dir) / args.cohort / args.age_band.replace("-", "_")
    output_dir = _out_base / _bin_name if _bin_name else _out_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Combining SHAP and FFA results for {args.cohort} / {args.age_band}{f' (bin={_bin_name})' if _bin_name else ''}")
    
    # Find results (same path layout for opioid_ed and non_opioid_ed: 7_shap_analysis/outputs/{cohort}/{age_band_fname}/)
    shap_path = find_shap_results(args.cohort, args.age_band, project_root, bin_name=_bin_name)
    shap_sample_path = find_shap_sample_parquet(args.cohort, args.age_band, project_root, bin_name=_bin_name)
    ffa_explanations_path, ffa_importance_path = find_ffa_results(args.cohort, args.age_band, project_root, bin_name=_bin_name)
    
    # All inputs are required; log errors and exit if any missing
    age_band_fname = args.age_band.replace("-", "_")
    missing = []
    if not shap_path:
        missing.append("SHAP importance")
        logger.error("Required input missing: SHAP importance. Expected under 7_shap_analysis/outputs/%s/%s/", args.cohort, age_band_fname)
    if not shap_sample_path:
        missing.append("SHAP sample values (parquet)")
        logger.error("Required input missing: SHAP sample values. Expected 7_shap_analysis/outputs/%s/%s/*_shap_sample_values_xgboost.parquet", args.cohort, age_band_fname)
    if not ffa_explanations_path:
        missing.append("FFA explanations (axp_explanations.parquet)")
        logger.error("Required input missing: FFA explanations. Expected 8_ffa_analysis/outputs/%s/%s/xgboost/axp_explanations.parquet (run Step 1b to generate)", args.cohort, age_band_fname)
    if not ffa_importance_path:
        missing.append("FFA importance")
        logger.error("Required input missing: FFA importance. Expected 8_ffa_analysis/outputs/%s/%s/xgboost/feature_importance_axp.parquet", args.cohort, age_band_fname)
    if missing:
        logger.error("Cannot combine: missing required inputs: %s. Fix the above and re-run.", ", ".join(missing))
        sys.exit(1)

    logger.info("All required inputs found: SHAP importance, SHAP sample, FFA explanations, FFA importance")

    # Load data
    shap_values = None
    shap_importance = None
    if shap_path:
        shap_values, shap_importance = load_shap_data(shap_path)
    
    ffa_explanations = None
    ffa_importance = None
    import duckdb
    if ffa_explanations_path:
        if str(ffa_explanations_path).endswith(".parquet"):
            con = duckdb.connect()
            try:
                # Limit rows only when n_patients > 0 (for memory); 0 = load all
                path_esc = str(ffa_explanations_path).replace("'", "''")
                if args.n_patients > 0:
                    limit = max(args.n_patients * 2, 5000)
                    ffa_explanations = con.execute(
                        f"SELECT * FROM read_parquet('{path_esc}') LIMIT {int(limit)}"
                    ).df()
                else:
                    ffa_explanations = con.execute(
                        f"SELECT * FROM read_parquet('{path_esc}')"
                    ).df()
            finally:
                con.close()
        else:
            con = duckdb.connect()
            try:
                path_esc = str(ffa_explanations_path).replace("'", "''")
                if args.n_patients > 0:
                    limit = max(args.n_patients * 2, 5000)
                    ffa_explanations = con.execute(
                        f"SELECT * FROM read_csv_auto('{path_esc}') LIMIT {int(limit)}"
                    ).df()
                else:
                    ffa_explanations = con.execute(
                        f"SELECT * FROM read_csv_auto('{path_esc}')"
                    ).df()
            finally:
                con.close()
    if ffa_importance_path:
        con_imp = duckdb.connect()
        try:
            if str(ffa_importance_path).endswith(".parquet"):
                ffa_importance = con_imp.execute(
                    f"SELECT * FROM read_parquet('{str(ffa_importance_path)}')"
                ).df()
            else:
                ffa_importance = con_imp.execute(
                    f"SELECT * FROM read_csv_auto('{str(ffa_importance_path)}')"
                ).df()
        finally:
            con_imp.close()
        # Normalize to 'feature' + 'importance' for downstream (consensus/combine expect 'feature' + importance col)
        if ffa_importance is not None and 'feature' in ffa_importance.columns:
            cand = ['importance', 'normalized_importance', 'mean_abs_shap', 'causal_importance', 'raw_count']
            imp_col = next((c for c in cand if c in ffa_importance.columns), None)
            if imp_col is None and len(ffa_importance.columns) > 1:
                # First non-feature numeric column
                for c in ffa_importance.columns:
                    if c != 'feature' and pd.api.types.is_numeric_dtype(ffa_importance[c]):
                        imp_col = c
                        break
            if imp_col and imp_col != 'importance':
                ffa_importance = ffa_importance.rename(columns={imp_col: 'importance'})
    
    # Get feature names
    if shap_importance is not None:
        feature_names = shap_importance['feature'].tolist()
    elif ffa_importance is not None:
        feature_names = ffa_importance['feature'].tolist()
    elif shap_values is not None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    else:
        logger.error("Cannot determine feature names - need SHAP or FFA importance")
        return
    
    # Optionally load SHAP sample parquet for row-level values (patient explanations)
    if shap_values is None and shap_sample_path is not None and feature_names:
        shap_values = load_shap_sample_parquet(shap_sample_path, feature_names)
        if shap_values is not None:
            logger.info("Loaded SHAP sample values for patient explanations: shape %s", shap_values.shape)
    
    # Calculate consensus
    consensus_data = calculate_consensus_features(shap_importance, ffa_importance, args.top_k)
    
    # Combine importance scores
    combined_importance = combine_importance_scores(
        shap_importance, ffa_importance, args.weight_shap, args.weight_ffa
    )
    
    # Generate patient explanations (use global consensus set for "consensus_features" per patient)
    patient_explanations = None
    if ffa_explanations is not None:
        patient_explanations = generate_patient_explanations(
            shap_values,
            ffa_explanations,
            feature_names,
            args.n_patients,
            global_consensus_features=consensus_data.get("consensus_features"),
            n_workers=args.workers,
        )
    
    # Save results
    consensus_path = output_dir / 'consensus_features.json'
    with open(consensus_path, 'w') as f:
        json.dump(consensus_data, f, indent=2)
    logger.info(f"Saved consensus features to {consensus_path}")
    
    if not combined_importance.empty:
        combined_path = output_dir / 'combined_importance.csv'
        combined_importance.to_csv(combined_path, index=False)
        logger.info(f"Saved combined importance to {combined_path}")
        # PHTS-style outputs for dashboard/Lambda compatibility
        generate_dashboard_outputs_phts_style(
            combined_importance, output_dir, args.cohort, args.age_band, args.top_k
        )
        if getattr(args, "upload_to_dashboard", False):
            upload_causal_data_to_dashboard(
                output_dir / "dashboard_data.json", args.cohort, args.age_band,
                bin_name=getattr(args, "bin", None),
            )
    
    if patient_explanations is not None:
        explanations_path = output_dir / 'patient_explanations.csv'
        patient_explanations.to_csv(explanations_path, index=False)
        logger.info(f"Saved patient explanations to {explanations_path}")
    
    # Generate summary report (include cohort/age_band so report shows feature-type check by design)
    summary = generate_summary_report(
        consensus_data, combined_importance, patient_explanations,
        cohort=args.cohort, age_band=args.age_band,
    )
    summary_path = output_dir / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Saved summary report to {summary_path}")
    
    # Print summary
    print("\n" + summary)


if __name__ == "__main__":
    main()

