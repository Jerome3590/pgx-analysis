"""
Final feature importance filters: exclude administrative/Z codes and target-leakage features.
Aligns with 3a_feature_importance/run_mc_feature_importance.py so dashboard JSONs use the same
final FI as downstream steps (Step 4 / Step 6).
"""

from pathlib import Path
from typing import Optional, Set

import pandas as pd

# Lazy import to avoid circular deps
def _load_administrative_codes_to_exclude(project_root: Path) -> Set[str]:
    """
    Build set of administrative/Z codes to exclude (same logic as run_mc_feature_importance).
    Uses 1b_apcd_event_filter/administrative_codes_lookup.json.
    """
    try:
        from py_helpers.file_resolver import load_administrative_codes
    except ImportError:
        return set()
    codes = load_administrative_codes(project_root)
    if not codes:
        return set()
    exclude: Set[str] = set()
    for code in codes.get("icd", []) + codes.get("cpt", []) + codes.get("drug", []):
        c = str(code).strip()
        if not c:
            continue
        exclude.add(c)
        if c[0].isalpha() and any(x.isdigit() for x in c):
            exclude.add(c.replace(".", ""))
            if "." not in c and len(c) >= 4:
                exclude.add(f"{c[:3]}.{c[3:]}")
    return exclude


def _normalize_feature_for_admin_check(feature: str) -> str:
    """Strip item_ prefix for comparison against administrative code list."""
    return str(feature).strip().removeprefix("item_")


def is_leakage_feature(feature_name: str) -> bool:
    """
    True if this feature name indicates target leakage (same patterns as
    run_mc_feature_importance._remove_target_leakage_features).
    """
    s = str(feature_name).strip()
    if not s:
        return False
    if s.startswith("post_"):
        return True
    if "time_to" in s.lower() or "time_to_" in s.lower():
        return True
    if any(x in s for x in ["_30d", "_90d", "_180d"]) and "interval" not in s.lower():
        return True
    if s in ("target_time", "first_time"):
        return True
    if "dtw" in s.lower():
        return True
    if "F1120" in s.upper():
        return True
    return False


def filter_fi_df_final(
    df: pd.DataFrame,
    project_root: Path,
    cohort: Optional[str] = None,
    feature_col: str = "feature",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Keep only final feature importance rows: drop administrative/Z codes and leakage features.
    For non_opioid_ed, also keep only drug-name features (same as pipeline).
    """
    if df is None or df.empty or feature_col not in df.columns:
        return df
    out = df.copy()

    # Admin/Z codes
    exclude = _load_administrative_codes_to_exclude(project_root)
    if exclude:
        normalized = out[feature_col].astype(str).map(_normalize_feature_for_admin_check)
        mask = ~normalized.isin(exclude)
        n_admin = (~mask).sum()
        out = out.loc[mask].copy()
        if verbose and n_admin:
            print(f"[FI filter] Removed {n_admin} administrative/Z row(s)")

    # Leakage
    leakage_mask = out[feature_col].astype(str).map(is_leakage_feature)
    n_leak = leakage_mask.sum()
    out = out.loc[~leakage_mask].copy()
    if verbose and n_leak:
        print(f"[FI filter] Removed {n_leak} leakage feature row(s)")

    # non_opioid_ed: drug-only (same as pipeline)
    if cohort == "non_opioid_ed":
        try:
            from py_helpers.feature_utils import filter_fi_to_drug_only
            out = filter_fi_to_drug_only(out, feature_col=feature_col)
        except ImportError:
            pass

    return out.reset_index(drop=True)
