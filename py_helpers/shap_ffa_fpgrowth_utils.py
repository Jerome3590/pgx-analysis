"""
SHAP/FFA-driven FP-Growth: load model-important features for filtering.

Used by FP-Growth visualization pipeline to run on the original dataset
restricted to items identified as important by SHAP and/or FFA analysis.
"""

import json
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd


def _parse_feature_name(feature: str) -> Tuple[str, str]:
    """
    Parse feature name to (code_type, code). Handles "item_<code>" format.
    Returns: (code_type, code); code_type is 'drug', 'icd', 'cpt', or 'other'.
    """
    if feature is None or (isinstance(feature, float) and pd.isna(feature)):
        return ("other", "")
    feature = str(feature).strip()
    if not feature:
        return ("other", "")
    if feature.startswith("item_"):
        code = feature[5:].strip()
    else:
        code = feature
    if not code:
        return ("other", feature)
    if code.isdigit():
        return ("cpt", code)
    if code[0].isalpha() and len(code) >= 2:
        rest = code[1:].replace(".", "").replace("-", "")
        if rest.isdigit():
            return ("icd", code)
        if len(code) <= 5 and code.isalnum():
            return ("icd", code)
        return ("drug", code)
    if code.replace(".", "").isdigit():
        return ("cpt", code)
    return ("drug", code)


def _load_shap_importance(
    cohort: str,
    age_band: str,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load SHAP global importance. Returns DataFrame with columns: feature, importance."""
    age_band_fname = age_band.replace("-", "_")
    base = f"{cohort}_{age_band_fname}"
    filename = f"{base}_shap_global_importance_xgboost.csv"
    candidates = []
    if project_root:
        candidates.append(project_root / "7_shap_analysis" / "outputs" / cohort / age_band_fname / filename)
    if data_root:
        candidates.append(data_root / "gold" / "shap_analysis" / cohort / age_band / filename)
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            if "feature" not in df.columns and len(df.columns) >= 1:
                df = df.rename(columns={df.columns[0]: "feature"})
            imp_col = next(
                (c for c in df.columns if "shap" in c.lower() or "importance" in c.lower()),
                df.columns[1] if len(df.columns) > 1 else None,
            )
            if imp_col is None:
                return pd.DataFrame()
            df = df[["feature", imp_col]].copy()
            df.columns = ["feature", "importance"]
            return df
    return pd.DataFrame()


def _load_ffa_importance(
    cohort: str,
    age_band: str,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load FFA causal importance. Returns DataFrame with columns: feature, importance."""
    age_band_fname = age_band.replace("-", "_")
    candidates = []
    if project_root:
        candidates.append(project_root / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / "xgboost" / "causal_importance.parquet")
    if data_root:
        candidates.append(data_root / "gold" / "ffa_analysis" / cohort / age_band / "xgboost" / "causal_importance.parquet")
    for path in candidates:
        if path and path.exists():
            df = pd.read_parquet(path)
            if "feature" not in df.columns:
                return pd.DataFrame()
            imp_col = next(
                (c for c in df.columns if "causal" in c.lower() or "importance" in c.lower()),
                df.columns[1] if len(df.columns) > 1 else None,
            )
            if imp_col is None:
                return pd.DataFrame()
            df = df[["feature", imp_col]].copy()
            df.columns = ["feature", "importance"]
            return df
    return pd.DataFrame()


def get_shap_ffa_important_codes(
    cohort: str,
    age_band: str,
    item_type: str,
    top_n: int = 500,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
    use_shap: bool = True,
    use_ffa: bool = True,
) -> Set[str]:
    """
    Return the set of item codes (drug/ICD/CPT) to use for FP-Growth, from SHAP and/or FFA.

    item_type: 'drug_name', 'icd_code', 'cpt_code', or 'medical_code'.
    For medical_code, returns union of drug, icd, and cpt codes.
    top_n: max features to consider from combined SHAP+FFA (by importance).
    """
    combined = []
    if use_shap:
        shap_df = _load_shap_importance(cohort, age_band, project_root, data_root)
        if not shap_df.empty:
            combined.append(shap_df)
    if use_ffa:
        ffa_df = _load_ffa_importance(cohort, age_band, project_root, data_root)
        if not ffa_df.empty:
            combined.append(ffa_df)
    if not combined:
        return set()
    merged = pd.concat(combined, ignore_index=True)
    # Dedupe by feature, keep max importance
    merged = merged.groupby("feature", as_index=False)["importance"].max()
    merged = merged.sort_values("importance", ascending=False).head(top_n)
    # Map to code_type and code
    code_sets = {"drug": set(), "icd": set(), "cpt": set()}
    for feat in merged["feature"].astype(str):
        code_type, code = _parse_feature_name(feat)
        if code and code_type in code_sets:
            code_sets[code_type].add(code)
    if item_type == "drug_name":
        return code_sets["drug"]
    if item_type == "icd_code":
        return code_sets["icd"]
    if item_type == "cpt_code":
        return code_sets["cpt"]
    if item_type == "medical_code":
        return code_sets["drug"] | code_sets["icd"] | code_sets["cpt"]
    return set()


def get_shap_ffa_allowed_codes_combined(
    cohort: str,
    age_band: str,
    top_n: int = 500,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
    use_shap: bool = True,
    use_ffa: bool = True,
) -> Set[str]:
    """
    Return the union of all SHAP/FFA important codes (drug + ICD + CPT) for BupaR/DTW.
    """
    drug = get_shap_ffa_important_codes(
        cohort, age_band, "drug_name", top_n, project_root, data_root, use_shap, use_ffa
    )
    icd = get_shap_ffa_important_codes(
        cohort, age_band, "icd_code", top_n, project_root, data_root, use_shap, use_ffa
    )
    cpt = get_shap_ffa_important_codes(
        cohort, age_band, "cpt_code", top_n, project_root, data_root, use_shap, use_ffa
    )
    return drug | icd | cpt


def write_shap_ffa_allowed_codes_for_bupar(
    cohort: str,
    age_band: str,
    output_path: Path,
    top_n: int = 500,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
    use_shap: bool = True,
    use_ffa: bool = True,
) -> bool:
    """
    Write a JSON array of allowed codes (for BupaR) from SHAP/FFA.
    Returns True if the file was written (at least one code), False otherwise.
    """
    codes = get_shap_ffa_allowed_codes_combined(
        cohort, age_band, top_n, project_root, data_root, use_shap, use_ffa
    )
    if not codes:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted(codes), f, indent=0)
    return True
