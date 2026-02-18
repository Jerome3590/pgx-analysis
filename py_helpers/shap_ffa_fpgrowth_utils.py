"""
Feature importance utilities for dashboard visuals.

- BupaR and DTW: use SHAP/FFA combined (get_shap_ffa_allowed_codes_combined, write_shap_ffa_allowed_codes_for_bupar).
- FP-Growth: use final feature importances (get_final_feature_importance_codes) from cohort_feature_importance.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd


def _parse_feature_name(feature: str) -> Tuple[str, str]:
    """
    Parse feature name to (code_type, code). Handles "item_<code>", "drug_<code>", "icd_<code>", "cpt_<code>".
    Returns the **code** that matches model data (e.g. LORAZEPAM not drug_LORAZEPAM) for BupaR/DTW filtering.
    Returns: (code_type, code); code_type is 'drug', 'icd', 'cpt', or 'other'.
    """
    if feature is None or (isinstance(feature, float) and pd.isna(feature)):
        return ("other", "")
    feature = str(feature).strip()
    if not feature:
        return ("other", "")
    # Strip known prefixes so we get the code that appears in model data (drug_name, icd columns, procedure_code)
    if feature.startswith("drug_"):
        code = feature[5:].strip()
        return ("drug", code) if code else ("other", "")
    if feature.startswith("icd_"):
        code = feature[4:].strip()
        return ("icd", code) if code else ("other", "")
    if feature.startswith("cpt_"):
        code = feature[4:].strip()
        return ("cpt", code) if code else ("other", "")
    if feature.startswith("item_"):
        code = feature[5:].strip()
        # Handle second-level prefixes (item_drug_X → drug_X → X)
        if code.startswith("drug_"):
            code = code[5:].strip()
            return ("drug", code) if code else ("other", "")
        if code.startswith("icd_"):
            code = code[4:].strip()
            return ("icd", code) if code else ("other", "")
        if code.startswith("cpt_"):
            code = code[4:].strip()
            return ("cpt", code) if code else ("other", "")
    else:
        code = feature
    if not code:
        return ("other", "")
    # Non-code features (e.g. n_events, pgx_num_drugs) should not be added to allowed codes
    if "_" in code and not code.replace(".", "").replace("-", "").replace("_", "").isalnum():
        return ("other", "")
    if code.isdigit():
        return ("cpt", code)
    if code[0].isalpha() and len(code) >= 2:
        rest = code[1:].replace(".", "").replace("-", "")
        if rest.isdigit():
            return ("icd", code)
        if len(code) <= 5 and code.isalnum():
            return ("icd", code)
        # Skip numeric/aggregate-like names (e.g. n_events, pgx_num_cpic_drugs)
        if "num_" in code or code.startswith("n_") or "pgx_" in code.lower():
            return ("other", "")
        return ("drug", code)
    if code.replace(".", "").isdigit():
        return ("cpt", code)
    return ("other", "")


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
    """
    Load FFA importance. Returns DataFrame with columns: feature, importance.
    Tries causal_importance.parquet first, then feature_importance_axp.parquet
    (same file Combine step uses) so BupaR finds FFA when SHAP/FFA pipeline completed.
    """
    age_band_fname = age_band.replace("-", "_")
    candidates = []
    if project_root:
        base = project_root / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / "xgboost"
        candidates.append(base / "causal_importance.parquet")
        candidates.append(base / "feature_importance_axp.parquet")
    if data_root:
        base = data_root / "gold" / "ffa_analysis" / cohort / age_band / "xgboost"
        candidates.append(base / "causal_importance.parquet")
        candidates.append(base / "feature_importance_axp.parquet")
    for path in candidates:
        if path and path.exists():
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if "feature" not in df.columns or df.empty:
                continue
            imp_col = next(
                (c for c in df.columns if c != "feature" and ("causal" in c.lower() or "importance" in c.lower())),
                None,
            )
            if imp_col is None and len(df.columns) > 1:
                for c in df.columns:
                    if c != "feature" and pd.api.types.is_numeric_dtype(df[c]):
                        imp_col = c
                        break
            if imp_col is None:
                continue
            df = df[["feature", imp_col]].copy()
            df.columns = ["feature", "importance"]
            return df
    return pd.DataFrame()


def _load_combined_importance_from_dashboard(
    cohort: str,
    age_band: str,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load combined SHAP+FFA importance from the Combine step output.
    Used as fallback when 7_shap_analysis / 8_ffa_analysis paths are missing.
    Location: 10_risk_dashboard/outputs/{cohort}/{age_band_fname}/combined_importance.csv
    Returns DataFrame with columns: feature, importance.
    """
    if not project_root:
        return pd.DataFrame()
    age_band_fname = age_band.replace("-", "_")
    path = (
        project_root
        / "10_risk_dashboard"
        / "outputs"
        / cohort
        / age_band_fname
        / "combined_importance.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "feature" not in df.columns or df.empty:
            return pd.DataFrame()
        imp_col = next(
            (
                c
                for c in df.columns
                if c != "feature"
                and ("combined" in c.lower() or "importance" in c.lower())
            ),
            df.columns[1] if len(df.columns) > 1 else None,
        )
        if imp_col is None:
            imp_col = df.columns[1] if len(df.columns) > 1 else None
        if imp_col is None:
            return pd.DataFrame()
        df = df[["feature", imp_col]].copy()
        df.columns = ["feature", "importance"]
        return df
    except Exception:
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
        # Fallback: use Combine step output (10_risk_dashboard/outputs/.../combined_importance.csv)
        dashboard_df = _load_combined_importance_from_dashboard(
            cohort, age_band, project_root
        )
        if not dashboard_df.empty:
            combined.append(dashboard_df)
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
    Write a JSON array of allowed codes (for BupaR) from SHAP/FFA combined.
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


def _load_final_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load final (cohort) feature importance CSV. Returns DataFrame with columns: feature, importance."""
    project_root = project_root or Path.cwd()
    try:
        from py_helpers.file_resolver import FileResolver
    except ImportError:
        return pd.DataFrame()
    resolver = FileResolver(
        file_type="cohort_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band,
    )
    path = resolver.resolve()
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "feature" not in df.columns and len(df.columns) >= 1:
        df = df.rename(columns={df.columns[0]: "feature"})
    imp_col = next(
        (c for c in df.columns if c != "feature" and ("importance" in c.lower() or "mean" in c.lower())),
        df.columns[1] if len(df.columns) > 1 else None,
    )
    if imp_col is None and "feature" in df.columns:
        df = df[["feature"]].copy()
        df["importance"] = 1.0
    elif imp_col is not None:
        df = df[["feature", imp_col]].copy()
        df.columns = ["feature", "importance"]
    return df


def get_final_feature_importance_codes(
    cohort: str,
    age_band: str,
    item_type: str,
    top_n: int = 500,
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
) -> Set[str]:
    """
    Return the set of item codes from final (cohort) feature importance for FP-Growth.

    item_type: 'drug_name', 'icd_code', 'cpt_code', or 'medical_code'.
    Used by FP-Growth only; BupaR and DTW use get_shap_ffa_allowed_codes_combined.
    """
    df = _load_final_feature_importance(cohort, age_band, project_root, data_root)
    if df.empty:
        return set()
    if "importance" not in df.columns:
        df["importance"] = 1.0
    df = df.sort_values("importance", ascending=False).head(top_n)
    code_sets: Dict[str, Set[str]] = {"drug": set(), "icd": set(), "cpt": set()}
    for feat in df["feature"].astype(str):
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
