#!/usr/bin/env python3
"""
Monte-Carlo feature-importance runner for the final, leakage-filtered feature set.

Flow:
  1. **Historical aggregated FI (baseline)** — in **pgx-repository** (read-only). Used to filter
     cohort features after cohorts are built (1b event filter).
  2. **Second pass** (default): start from baseline aggregated FI (not original full set). Load
     historical FI from pgx-repository → minus admin/Z codes → use that list as features (~11K) →
     build patient-level feature matrix from cohort.parquet → run MC CV.
  3. **New cohorts (no baseline in pgx-repository):** When historical is missing, we run a
     **baseline** pass first (baseline=True). The baseline now builds a full feature matrix from
     **cohort-derived ICD/CPT/drug codes** (minus admin/Z), not just n_events, so the resulting
     aggregated FI has many features. That baseline is then used for the second pass.
  4. **Second-pass FI are always saved to pgxdatalake** (gold/feature_importance/{cohort}/{age_band}/).
  5. **Final model training** uses these second-pass feature importances from pgxdatalake for
     train features (Step 6 build_final_cohort_model_features / run_final_model).

Usage (example):

    python 3a_feature_importance/run_mc_feature_importance.py \
        --cohort opioid_ed \
        --age_band 13-24 \
        --n_runs 25
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, recall_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import duckdb
import matplotlib.pyplot as plt

# Ensure project root on path so we can import final_model utilities
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # noqa: E402
from py_helpers.env_utils import get_xgb_cpu_nthread, get_data_root, is_linux  # noqa: E402
from py_helpers.s3_utils import normalize_cohort_name, get_cohort_parquet_path  # noqa: E402
from py_helpers.feature_importance_utils import aggregate_feature_importance  # noqa: E402

# Event years to combine when loading cohort data (matches 4_model_data default)
DEFAULT_EVENT_YEARS = [2016, 2017, 2018, 2019]

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET  # noqa: E402
except ImportError:
    import boto3  # noqa: E402
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

# Historical baseline aggregated FI (read-only): used to filter cohort features after cohorts built; second pass = this minus admin/Z
PGX_REPO_BUCKET = "pgx-repository"
PGX_REPO_FI_PREFIX = "pgx-analysis/3_feature_importance/outputs"

def _load_feature_table(path: Path, required: bool = True) -> pd.DataFrame:
    """Simplified loader mirroring 6_final_model.run_final_model._load_feature_table."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required feature file not found: {path}")
        print(f"Feature file not found (skipping): {path}")
        return pd.DataFrame()
    print(f"Loading features from {path}")
    return pd.read_csv(path)


def _remove_target_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious target-leakage features based on naming conventions,
    mirroring the logic in the legacy remove_target_leakage.py script and
    6_final_model.run_final_model.remove_target_leakage_features.

    This drops:
      - Columns starting with 'post_'
      - Columns containing 'time_to' / 'time_to_'
      - Time-window features with suffixes like '_30d', '_90d', '_180d'
        (except those with 'interval' in the name)
      - Datetime helper columns: 'target_time', 'first_time'
      - DTW-derived features (any column with 'dtw' in its name)
      - Any feature whose name contains 'F1120'
    """
    cols = list(df.columns)
    leakage: set[str] = set()

    post_features = [c for c in cols if c.startswith("post_")]
    leakage.update(post_features)

    time_to_features = [
        c for c in cols if "time_to" in c.lower() or "time_to_" in c.lower()
    ]
    leakage.update(time_to_features)

    time_window_features = [
        c
        for c in cols
        if any(x in c for x in ["_30d", "_90d", "_180d"])
        and "interval" not in c.lower()
    ]
    leakage.update(time_window_features)

    datetime_features = [c for c in ("target_time", "first_time") if c in cols]
    leakage.update(datetime_features)

    dtw_features = [c for c in cols if "dtw" in c.lower()]
    leakage.update(dtw_features)

    f1120_features = [c for c in cols if "F1120" in c.upper()]
    leakage.update(f1120_features)

    if leakage:
        kept = [c for c in cols if c not in leakage]
        print(
            "Removing potential target-leakage features:\n  "
            + ", ".join(sorted(leakage))
        )
        return df[kept].copy()

    return df


def _load_administrative_codes_to_exclude() -> Set[str]:
    """
    Load administrative and Z codes to exclude from aggregated feature importance output.
    Uses 1b_apcd_event_filter/administrative_codes_lookup.json (ICD, CPT, drug).
    Returns a set of normalized code strings; for ICD-style codes both dot and no-dot
    variants are included so we can match feature names with or without dots.
    """
    admin_path = PROJECT_ROOT / "1b_apcd_event_filter" / "administrative_codes_lookup.json"
    if not admin_path.exists():
        return set()
    with open(admin_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    codes = data.get("administrative_codes", {})
    exclude: Set[str] = set()
    for code in codes.get("icd", []) + codes.get("cpt", []) + codes.get("drug", []):
        c = str(code).strip()
        if not c:
            continue
        exclude.add(c)
        # ICD-style: add no-dot variant for matching
        if c[0].isalpha() and any(x.isdigit() for x in c):
            exclude.add(c.replace(".", ""))
            if "." not in c and len(c) >= 4:
                exclude.add(f"{c[:3]}.{c[3:]}")
    return exclude


def _normalize_feature_for_admin_check(feature: str) -> str:
    """Strip item_ prefix for comparison against administrative code list."""
    return str(feature).strip().removeprefix("item_")


def _filter_aggregated_fi_admin_codes(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from aggregated FI where feature is an administrative or Z code.
    Keeps the second pass aligned with 'aggregated feature importances minus admin Z codes'.
    """
    exclude = _load_administrative_codes_to_exclude()
    if not exclude:
        return agg_df
    if "feature" not in agg_df.columns:
        return agg_df
    before = len(agg_df)
    normalized = agg_df["feature"].astype(str).map(_normalize_feature_for_admin_check)
    mask = ~normalized.isin(exclude)
    out = agg_df.loc[mask].copy()
    n_removed = before - len(out)
    if n_removed > 0:
        print(
            f"[INFO] Filtered out {n_removed} administrative/Z code(s) from aggregated FI: "
            f"{sorted(normalized[~mask].unique().tolist())}"
        )
    return out


def _load_historical_aggregated_fi_from_pgx_repo(
    cohort: str, age_band_fname: str
) -> Optional[pd.DataFrame]:
    """
    Load historical (baseline) aggregated feature importance from pgx-repository.
    Used for second pass: feature set = historical minus admin/Z codes.
    Returns None if not found.
    """
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    s3_key = f"{PGX_REPO_FI_PREFIX}/{filename}"
    try:
        import io
        obj = s3_client.get_object(Bucket=PGX_REPO_BUCKET, Key=s3_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        if "feature" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _get_feature_list_minus_admin_z(agg_df: pd.DataFrame) -> List[str]:
    """Get list of feature names from aggregated FI CSV, excluding administrative/Z codes (no item_ prefix)."""
    exclude = _load_administrative_codes_to_exclude()
    features: List[str] = []
    for raw in agg_df["feature"].astype(str).unique():
        name = _normalize_feature_for_admin_check(raw)
        if not name or name == "nan":
            continue
        if name in exclude:
            continue
        features.append(name)
    return list(dict.fromkeys(features))  # preserve order, dedupe


# Minimum features in historical FI to use it as baseline; below this we use cohort-derived feature list
MIN_BASELINE_FEATURES = 100


def _get_cohort_feature_list_minus_admin_z(cohort: str, age_band: str) -> List[str]:
    """
    Get all distinct codes from cohort.parquet event-level code columns, minus admin/Z.
    Used when historical aggregated FI from pgx-repository has too few features (e.g. only n_events).
    """
    cohort_paths = _resolve_cohort_parquet_paths(cohort, age_band)
    if not cohort_paths:
        raise FileNotFoundError(
            f"Cohort data not found for cohort={cohort}, age_band={age_band}. "
            "Run Step 2 (2_create_cohort) first."
        )
    code_cols = [
        "primary_icd_diagnosis_code", "two_icd_diagnosis_code", "three_icd_diagnosis_code",
        "four_icd_diagnosis_code", "five_icd_diagnosis_code", "six_icd_diagnosis_code",
        "seven_icd_diagnosis_code", "eight_icd_diagnosis_code", "nine_icd_diagnosis_code",
        "ten_icd_diagnosis_code", "procedure_code", "cpt_mod_1_code", "cpt_mod_2_code",
        "drug_name",
    ]
    con = duckdb.connect()
    try:
        desc = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({repr(cohort_paths[0])})"
        ).fetchall()
        existing = {r[0] for r in desc}
    except Exception:
        existing = set()
    code_cols_present = [c for c in code_cols if c in existing]
    con.close()
    if not code_cols_present:
        return []
    paths_sql = ", ".join(repr(p) for p in cohort_paths)
    selects = [
        f"SELECT DISTINCT TRIM(CAST({c} AS VARCHAR)) AS code FROM read_parquet([{paths_sql}]) "
        f"WHERE {c} IS NOT NULL AND TRIM(CAST({c} AS VARCHAR)) <> ''"
        for c in code_cols_present
    ]
    union_sql = " UNION ".join(selects)
    con = duckdb.connect()
    codes_df = con.execute(
        f"SELECT code FROM ({union_sql}) WHERE code <> ''"
    ).df()
    con.close()
    raw_codes = codes_df["code"].astype(str).unique().tolist()
    exclude = _load_administrative_codes_to_exclude()
    features: List[str] = []
    for raw in raw_codes:
        name = _normalize_feature_for_admin_check(raw)
        if not name or name == "nan":
            continue
        if name in exclude:
            continue
        features.append(name)
    return list(dict.fromkeys(features))


def _build_patient_features_from_cohort_and_fi_list(
    cohort: str, age_band: str, allowed_features: List[str]
) -> pd.DataFrame:
    """
    Build patient-level feature matrix from cohort.parquet using only allowed features
    (historical aggregated FI minus admin/Z). Cohort is event-level; we aggregate to
    one row per patient with columns = [mi_person_key, target, <feature_count>...].
    """
    cohort_paths = _resolve_cohort_parquet_paths(cohort, age_band)
    if not cohort_paths:
        raise FileNotFoundError(
            f"Cohort data not found for cohort={cohort}, age_band={age_band}. "
            "Run Step 2 (2_create_cohort) first."
        )
    # Normalize allowed features: strip item_ for matching; build code->feature_name map (with dot/no-dot)
    code_to_feature: Dict[str, str] = {}
    for f in allowed_features:
        name = _normalize_feature_for_admin_check(f)
        code_to_feature[name] = f
        if name and name[0].isalpha() and any(c.isdigit() for c in name):
            no_dot = name.replace(".", "")
            code_to_feature[no_dot] = f
            if "." not in name and len(name) >= 4:
                code_to_feature[f"{name[:3]}.{name[3:]}"] = f
    if not code_to_feature:
        raise ValueError("No allowed features after normalization.")

    paths_sql = ", ".join(repr(p) for p in cohort_paths)
    # Code columns that may appear in cohort.parquet (from unified_event_fact_table)
    code_cols = [
        "primary_icd_diagnosis_code", "two_icd_diagnosis_code", "three_icd_diagnosis_code",
        "four_icd_diagnosis_code", "five_icd_diagnosis_code", "six_icd_diagnosis_code",
        "seven_icd_diagnosis_code", "eight_icd_diagnosis_code", "nine_icd_diagnosis_code",
        "ten_icd_diagnosis_code", "procedure_code", "cpt_mod_1_code", "cpt_mod_2_code",
        "drug_name",
    ]
    # Build UNPIVOT-like query: one row per (mi_person_key, is_target_case, code)
    selects = []
    for c in code_cols:
        selects.append(f"SELECT mi_person_key, is_target_case, CAST({c} AS VARCHAR) AS code FROM t WHERE {c} IS NOT NULL AND CAST({c} AS VARCHAR) <> ''")
    union_sql = " UNION ALL ".join(selects)
    con = duckdb.connect()
    # Get schema of first file to see which code columns exist
    try:
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet({repr(cohort_paths[0])})").fetchall()
        existing = {r[0] for r in desc}
    except Exception:
        existing = set()
    con.close()
    code_cols_present = [c for c in code_cols if c in existing]
    if not code_cols_present:
        raise ValueError(
            f"Cohort parquet has no code columns among {code_cols}. "
            "Cannot build feature matrix from historical FI list."
        )
    selects = [
        f"SELECT mi_person_key, is_target_case, CAST({c} AS VARCHAR) AS code FROM read_parquet([{paths_sql}]) WHERE {c} IS NOT NULL AND TRIM(CAST({c} AS VARCHAR)) <> ''"
        for c in code_cols_present
    ]
    union_sql = " UNION ALL ".join(selects)
    con = duckdb.connect()
    events_sql = f"""
    WITH unpivoted AS (
        {union_sql}
    )
    SELECT mi_person_key, is_target_case, TRIM(code) AS code
    FROM unpivoted
    WHERE TRIM(code) <> ''
    """
    events_df = con.execute(events_sql).df()
    # All patients and target from cohort (so we include patients with zero matching events)
    patients_sql = f"""
    SELECT
        CAST(mi_person_key AS VARCHAR) AS mi_person_key,
        CAST(MAX(is_target_case) AS INTEGER) AS target
    FROM read_parquet([{paths_sql}])
    GROUP BY mi_person_key
    """
    patients_df = con.execute(patients_sql).df()
    con.close()
    patients_df["target"] = patients_df["target"].astype(int).clip(lower=0, upper=1)
    # Filter events to allowed codes (match via code_to_feature)
    events_df = events_df[events_df["code"].isin(code_to_feature.keys())]
    events_df["feature_name"] = events_df["code"].map(code_to_feature)
    pivot_df = events_df.groupby(["mi_person_key", "feature_name"], as_index=False).size()
    pivot_wide = pivot_df.pivot(index="mi_person_key", columns="feature_name", values="size")
    pivot_wide = pivot_wide.reindex(columns=allowed_features, fill_value=0)
    final = patients_df.merge(pivot_wide, on="mi_person_key", how="left")
    for col in allowed_features:
        if col not in final.columns:
            final[col] = 0
    final = final[["mi_person_key", "target"] + [c for c in allowed_features if c in final.columns]]
    final = final.fillna(0)
    final = _remove_target_leakage_features(final)
    return final


def _validate_cohort_file_has_controls(path_or_s3: str) -> dict:
    """
    Validate that a cohort.parquet file contains both cases (is_target_case=1) and controls (is_target_case=0).
    path_or_s3: local path or s3:// URL.
    Returns:
        dict with keys: has_controls (bool), n_cases (int), n_controls (int), error (str or None)
    """
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE is_target_case = 1) AS n_cases,
                COUNT(*) FILTER (WHERE is_target_case = 0) AS n_controls
            FROM read_parquet('{path_or_s3}')
            """
        ).fetchone()
        n_cases = int(result[0]) if result else 0
        n_controls = int(result[1]) if result else 0
        return {
            "has_controls": n_controls > 0,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "error": None,
        }
    except Exception as e:
        return {"has_controls": False, "n_cases": 0, "n_controls": 0, "error": str(e)}
    finally:
        con.close()


def _cohort_local_root() -> Path:
    """Local root for syncing cohort.parquet from S3 (NVMe on EC2). DuckDB uses only local paths."""
    return get_data_root() / "gold" / "cohorts"


def _resolve_cohort_parquet_paths(cohort: str, age_band: str) -> List[str]:
    """
    Resolve paths to cohort.parquet for (cohort, age_band) across DEFAULT_EVENT_YEARS.
    Returns only **local** paths: if a file exists only on S3, it is synced to local (NVMe/data) first,
    so DuckDB never sees mixed local/S3 paths.
    """
    cohort_slug = normalize_cohort_name(cohort)
    local_root = _cohort_local_root()
    # Candidate local roots to check before syncing (same layout as 4_model_data)
    data_root = get_data_root()
    check_roots: List[Path] = [
        data_root / "gold" / "cohorts",
        data_root / "data" / "gold_cohorts",
        PROJECT_ROOT / "data" / "gold_cohorts",
    ]
    if os.environ.get("LOCAL_DATA_PATH"):
        check_roots.insert(0, Path(os.environ["LOCAL_DATA_PATH"]))

    found: List[str] = []
    for year in DEFAULT_EVENT_YEARS:
        rel = f"cohort_name={cohort_slug}/event_year={year}/age_band={age_band}/cohort.parquet"
        local_path = local_root / rel
        # Prefer existing local file from any check root
        for root in check_roots:
            p = root / rel
            if p.exists():
                found.append(str(p))
                break
        else:
            # Not found locally: try S3 and sync to local_root (NVMe) then use that path
            s3_path = get_cohort_parquet_path(cohort_slug, age_band, year)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(s3_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                s3_client.head_object(Bucket=bucket, Key=key)
            except Exception:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                with open(local_path, "wb") as f:
                    f.write(obj["Body"].read())
                print(f"Synced cohort.parquet from S3 to local: {local_path}")
                found.append(str(local_path))
            except Exception as e:
                print(f"Warning: could not sync {s3_path} to {local_path}: {e}")
    return found


def build_final_features_for_mc(cohort: str, age_band: str, prefer_filtered: bool = True) -> pd.DataFrame:
    """
    Build feature matrix for Step 3 (Feature Importance) from **cohort data** (Step 2 cohort.parquet).

    Does not use Step 4 (model_events.parquet). Loads cohort.parquet for (cohort, age_band)
    across DEFAULT_EVENT_YEARS (2016–2019); files missing locally are synced from S3 to
    local (NVMe) first so DuckDB sees only local paths. Aggregates to patient-level:
    mi_person_key, target = MAX(is_target_case), n_events = COUNT(*).

    prefer_filtered is kept for API compatibility but has no effect when using cohort data.
    """
    cohort_paths = _resolve_cohort_parquet_paths(cohort, age_band)
    if not cohort_paths:
        raise FileNotFoundError(
            f"Cohort data not found for cohort={cohort}, age_band={age_band}. "
            f"Checked local gold/cohorts and S3 gold/cohorts/ for event years {DEFAULT_EVENT_YEARS}. "
            "Run Step 2 (2_create_cohort) first to produce cohort.parquet files."
        )

    # Validate at least one file has controls
    for path in cohort_paths[:1]:
        v = _validate_cohort_file_has_controls(path)
        if v.get("error"):
            print(f"Warning: Could not validate cohort file: {path} - {v['error']}")
        elif not v.get("has_controls", False):
            print(
                f"Warning: Cohort file has no controls: {path} "
                f"(cases={v.get('n_cases', 0)}, controls={v.get('n_controls', 0)})"
            )

    print(f"Loading cohort data (cases + controls) from {len(cohort_paths)} file(s)")
    paths_sql = ", ".join(repr(p) for p in cohort_paths)
    con = duckdb.connect()
    grouped = con.execute(
        f"""
        SELECT
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            CAST(MAX(is_target_case) AS INTEGER) AS target,
            COUNT(*)::BIGINT AS n_events
        FROM read_parquet([{paths_sql}])
        GROUP BY mi_person_key
        """
    ).df()
    con.close()

    grouped["target"] = grouped["target"].astype(int).clip(lower=0, upper=1)

    final = grouped.copy()
    final = final.dropna(subset=["target"])
    final = _remove_target_leakage_features(final)
    # Log aggregated table shape and columns (for verifying dataset build)
    print(f"[DATASET] Aggregated patient-level: {len(final):,} rows, columns: {list(final.columns)}")
    return final


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare numeric feature matrix X and label y from the assembled DataFrame."""
    feature_cols = [c for c in df.columns if c not in ("mi_person_key", "target")]
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns available for MC feature importance.")

    X = df[numeric_feature_cols].replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(0)
    y = df["target"].astype(int)

    return X, y, numeric_feature_cols


def _non_constant_mask_and_slices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """Drop columns that are constant in X_train so models (e.g. CatBoost) do not see all-constant features.

    Returns:
        non_constant: boolean array of shape (n_features,) True where column has variance > 0 in X_train
        X_train_active: X_train with only non-constant columns
        X_test_active: X_test with same columns
        active_feature_names: list of feature names for non-constant columns
    """
    # Column order matches feature_names (X was df[numeric_feature_cols])
    X_tr = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    var_per_col = np.var(X_tr, axis=0)
    non_constant = np.asarray(var_per_col, dtype=float) > 0
    if not np.any(non_constant):
        raise ValueError(
            "In this train split all features are constant (zero variance). "
            "Cannot fit CatBoost/XGBoost. Try a different random seed or larger cohort."
        )
    idx = np.where(non_constant)[0]
    active_names = [feature_names[i] for i in idx]
    X_train_active = X_train.iloc[:, idx]
    X_test_active = X_test.iloc[:, idx]
    return non_constant, X_train_active, X_test_active, active_names


def run_mc_feature_importance(
    cohort: str,
    age_band: str,
    n_runs: int = 25,
    test_size: float = 0.3,
    random_seed: int = 42,
    force: bool = False,
    baseline: bool = False,
    run_baseline_if_missing: bool = True,
) -> pd.DataFrame:
    """Run Monte-Carlo CV for multiple models and aggregate feature importances.

    Uses cohort data (Step 2 cohort.parquet) to build the feature matrix. Does not
    depend on Step 4 (model_events.parquet).

    Default (baseline=False, second pass): load historical aggregated FI from pgx-repository,
    minus admin/Z codes; build feature matrix from cohort with those features; write to outputs/{cohort}/ and pgxdatalake.
    If historical baseline is missing and run_baseline_if_missing=True (default), runs a baseline pass first
    (permutation feature importance on cohort-derived features), then uses that result for the second pass.

    When baseline=True: write to outputs/{cohort}/_baseline/. Use only when generating
    a local baseline; normal pipeline uses historical baseline in pgx-repository.

    Models:
      - XGBoost (gradient boosted trees, CPU on Linux, GPU on Windows if available)
      - XGBoost RF (XGBRFClassifier, CPU on Linux, GPU on Windows if available)
      - CatBoost (if installed; CPU only)

    This function is idempotent - it will skip if results already exist locally or in S3,
    unless force=True is specified.
    """
    age_band_fname = age_band_to_fname(age_band)
    # Optional: write to NVMe on EC2 (set PGX_FEATURE_IMPORTANCE_OUTPUTS e.g. /mnt/nvme/feature_importance/outputs)
    _outputs_base = os.environ.get("PGX_FEATURE_IMPORTANCE_OUTPUTS")
    if _outputs_base:
        out_dir = Path(_outputs_base) / cohort
        print(f"[INFO] Writing Step 3 outputs to NVMe: {out_dir}")
    else:
        out_dir = PROJECT_ROOT / "3a_feature_importance" / "outputs" / cohort
    if baseline:
        out_dir = out_dir / "_baseline"
        print("[INFO] Baseline run: writing to _baseline subfolder (original aggregated FI for 1b event filter)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing aggregated results (idempotency)
    agg_path = out_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"

    if not force and agg_path.exists():
        print(f"✓ Aggregated feature importance already exists locally: {agg_path}")
        print("  Skipping Monte-Carlo feature importance computation.")
        print("  Use --force to rerun.")
        return pd.read_csv(agg_path)

    # Check S3 if not found locally (use _baseline in S3 key when baseline=True)
    if not force:
        s3_suffix = "_baseline/" if baseline else ""
        s3_key_agg = (
            f"gold/feature_importance/{cohort}/{age_band}/{s3_suffix}"
            f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_agg)
            print(f"✓ Aggregated feature importance exists in S3: s3://{S3_BUCKET}/{s3_key_agg}")
            print("  Downloading instead of recomputing...")
            import io
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_agg)
            agg_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            agg_df.to_csv(agg_path, index=False)
            print(f"  Saved locally: {agg_path}")
            return agg_df
        except Exception:
            # File doesn't exist in S3, proceed with computation
            pass

    # Assemble final feature matrix
    # Second pass (baseline=False): use historical aggregated FI from pgx-repository minus admin/Z codes
    # Baseline (baseline=True) or if historical not found: use cohort-only n_events
    df: Optional[pd.DataFrame] = None
    if not baseline:
        hist_df = _load_historical_aggregated_fi_from_pgx_repo(cohort, age_band_fname)
        # If baseline missing and run_baseline_if_missing: run permutation FI baseline first, then use it
        if (hist_df is None or hist_df.empty) and run_baseline_if_missing:
            print(
                "[INFO] Baseline missing in pgx-repository; running baseline (permutation feature importance) first."
            )
            run_mc_feature_importance(
                cohort=cohort,
                age_band=age_band,
                n_runs=n_runs,
                test_size=test_size,
                random_seed=random_seed,
                force=force,
                baseline=True,
                run_baseline_if_missing=False,
            )
            baseline_path = out_dir / "_baseline" / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
            if baseline_path.exists():
                hist_df = pd.read_csv(baseline_path)
                print(f"[INFO] Loaded baseline from {baseline_path} for second pass")
            else:
                hist_df = None
        if hist_df is not None and not hist_df.empty:
            feature_list = _get_feature_list_minus_admin_z(hist_df)
            if feature_list:
                print(
                    f"[INFO] Second pass: using aggregated FI (minus admin/Z): {len(feature_list)} features"
                )
                df = _build_patient_features_from_cohort_and_fi_list(cohort, age_band, feature_list)
            else:
                raise ValueError(
                    "Baseline aggregated FI has no features after removing admin/Z codes. "
                    "Refusing to run with n_events only; we must have a proper feature set."
                )
        else:
            if not run_baseline_if_missing or not (out_dir / "_baseline" / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv").exists():
                raise FileNotFoundError(
                    "Historical aggregated FI not found in pgx-repository and no local baseline. "
                    "Run with default (run_baseline_if_missing=True) to create baseline from cohort, "
                    f"or provide s3://pgx-repository/{PGX_REPO_FI_PREFIX}/{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
                )
    # Baseline run (new cohorts): build feature matrix from cohort-derived ICD/CPT/drug list only (never n_events only)
    if baseline and (df is None or df.empty):
        feature_list = _get_cohort_feature_list_minus_admin_z(cohort, age_band)
        if not feature_list:
            raise ValueError(
                f"No cohort-derived features (ICD/CPT/drug minus admin/Z) for {cohort}/{age_band}. "
                "Ensure Step 2 cohort.parquet has code columns and is not empty."
            )
        print(
            f"[INFO] Baseline run: using cohort-derived feature list (minus admin/Z): {len(feature_list)} features"
        )
        df = _build_patient_features_from_cohort_and_fi_list(cohort, age_band, feature_list)
    if df is None or df.empty:
        raise ValueError(
            f"No feature matrix assembled for cohort={cohort}, age_band={age_band}. "
            "We require baseline aggregated importances or historical FI (minus admin/Z); never n_events only."
        )
    # Refuse to run with a single feature (n_events) so we always produce proper aggregated FI
    feature_cols = [c for c in df.columns if c not in ("mi_person_key", "target")]
    if len(feature_cols) < 2:
        raise ValueError(
            f"Feature matrix has only {len(feature_cols)} feature(s) (need many for aggregated importances). "
            "Use baseline or historical aggregated FI; never n_events only."
        )

    X, y, feature_names = _prepare_xy(df)

    # Dataset build verification (for logs: verify features and target are correct)
    n_patients = len(df)
    n_cases = int((y == 1).sum())
    n_controls = int((y == 0).sum())
    print(f"[DATASET] Built patient-level table: {n_patients:,} rows")
    print(f"[DATASET] Target: {n_cases:,} cases (1), {n_controls:,} controls (0)")
    print(f"[DATASET] Features ({len(feature_names)}): {feature_names}")
    print(f"[DATASET] X shape: {X.shape}, y shape: {y.shape}")
    if n_cases == 0 or n_controls == 0:
        raise ValueError(
            f"Dataset must have both cases and controls. Got cases={n_cases}, controls={n_controls}. "
            "Check cohort.parquet has is_target_case 0 and 1."
        )
    print("[DATASET] OK: both classes present, ready for MC-CV")

    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise ImportError(
            "XGBoost is required for Monte-Carlo feature importance. "
            "Install with: pip install xgboost"
        ) from exc

    try:
        from catboost import CatBoostClassifier  # type: ignore

        have_catboost = True
    except Exception:
        have_catboost = False
        print(
            "CatBoost not available; skipping CatBoost feature importance. "
            "Install with: pip install catboost"
        )

    rng = np.random.default_rng(random_seed)

    model_keys = ["xgb", "xgb_rf"]
    if have_catboost:
        model_keys.append("catboost")
        print(f"[INFO] CatBoost is available - will run for all {n_runs} MC CV splits")
    else:
        print(f"[INFO] CatBoost not available - only running XGBoost models for {n_runs} MC CV splits")

    # Storage for per-run metrics and importances, per model
    per_feature_importances: Dict[str, Dict[str, List[float]]] = {
        m: {f: [] for f in feature_names} for m in model_keys
    }
    per_feature_scaled: Dict[str, Dict[str, List[float]]] = {
        m: {f: [] for f in feature_names} for m in model_keys
    }
    aucs: Dict[str, List[float]] = {m: [] for m in model_keys}
    pr_aucs: Dict[str, List[float]] = {m: [] for m in model_keys}
    recalls: Dict[str, List[float]] = {m: [] for m in model_keys}
    loglosses: Dict[str, List[float]] = {m: [] for m in model_keys}

    nthread = get_xgb_cpu_nthread()
    
    # Determine device: CPU on Linux, CUDA on Windows (if available)
    device = "cpu" if is_linux() else "cuda"

    print(f"\n[INFO] Starting Monte-Carlo CV with {n_runs} splits")
    print(f"[INFO] Models to run: {', '.join(model_keys)}")
    
    for run_idx in range(n_runs):
        rs = int(rng.integers(0, np.iinfo(np.int32).max))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )
        # Drop columns that are constant in this train split (avoids CatBoost/XGBoost "all constant" error)
        non_constant, X_train_active, X_test_active, _ = _non_constant_mask_and_slices(
            X_train, X_test, feature_names
        )
        n_active = int(non_constant.sum())
        if run_idx == 0 and n_active < len(feature_names):
            print(f"[INFO] Split 0: using {n_active:,} non-constant features (dropped {len(feature_names) - n_active:,} constant in train)")

        # --------------------
        # XGBoost (boosting)
        # --------------------
        xgb_clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=rs,
        )
        try:
            xgb_clf.fit(X_train_active, y_train)
        except Exception:
            # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
            xgb_clf.set_params(tree_method="hist")
            if "device" in xgb_clf.get_params():
                xgb_clf.set_params(device="cpu")
            xgb_clf.fit(X_train_active, y_train)

        for model_name, clf in [("xgb", xgb_clf)]:
            y_proba = clf.predict_proba(X_test_active)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            recalls[model_name].append(recall_score(y_test, y_pred))
            loglosses[model_name].append(log_loss(y_test, y_proba))
            aucs[model_name].append(roc_auc_score(y_test, y_proba))
            pr_aucs[model_name].append(average_precision_score(y_test, y_proba))

            importances_active = np.asarray(clf.feature_importances_, dtype=float)
            full_importances = np.zeros(len(feature_names), dtype=float)
            full_importances[non_constant] = importances_active
            mean_imp = float(full_importances.mean()) if full_importances.size > 0 else 0.0
            if mean_imp > 0:
                full_scaled = full_importances / mean_imp
            else:
                full_scaled = np.zeros_like(full_importances)
            for fname, imp, imp_scaled in zip(feature_names, full_importances, full_scaled, strict=True):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        # --------------------
        # XGBoost RF (XGBRFClassifier)
        # --------------------
        xgbrf_clf = xgb.XGBRFClassifier(
            n_estimators=500,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=rs + 1,
        )
        try:
            xgbrf_clf.fit(X_train_active, y_train)
        except Exception:
            # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
            xgbrf_clf.set_params(tree_method="hist")
            if "device" in xgbrf_clf.get_params():
                xgbrf_clf.set_params(device="cpu")
            xgbrf_clf.fit(X_train_active, y_train)

        for model_name, clf in [("xgb_rf", xgbrf_clf)]:
            y_proba = clf.predict_proba(X_test_active)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            recalls[model_name].append(recall_score(y_test, y_pred))
            loglosses[model_name].append(log_loss(y_test, y_proba))
            aucs[model_name].append(roc_auc_score(y_test, y_proba))
            pr_aucs[model_name].append(average_precision_score(y_test, y_proba))

            importances_active = np.asarray(clf.feature_importances_, dtype=float)
            full_importances = np.zeros(len(feature_names), dtype=float)
            full_importances[non_constant] = importances_active
            mean_imp = float(full_importances.mean()) if full_importances.size > 0 else 0.0
            if mean_imp > 0:
                full_scaled = full_importances / mean_imp
            else:
                full_scaled = np.zeros_like(full_importances)
            for fname, imp, imp_scaled in zip(feature_names, full_importances, full_scaled, strict=True):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        # --------------------
        # CatBoost (optional)
        # --------------------
        if have_catboost:
            cb_clf = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="Logloss",
                grow_policy="SymmetricTree",
                random_seed=rs + 2,
                verbose=False,
            )
            try:
                cb_clf.fit(X_train_active, y_train)
            except Exception:
                # Fallback is still CPU since CatBoost manages devices internally
                cb_clf = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    grow_policy="SymmetricTree",
                    random_seed=rs + 2,
                    verbose=False,
                )
                cb_clf.fit(X_train_active, y_train)

            model_name = "catboost"
            y_proba = cb_clf.predict_proba(X_test_active)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            recalls[model_name].append(recall_score(y_test, y_pred))
            loglosses[model_name].append(log_loss(y_test, y_proba))
            aucs[model_name].append(roc_auc_score(y_test, y_proba))
            pr_aucs[model_name].append(average_precision_score(y_test, y_proba))

            importances_active = np.asarray(cb_clf.get_feature_importance(), dtype=float)
            full_importances = np.zeros(len(feature_names), dtype=float)
            full_importances[non_constant] = importances_active
            mean_imp = float(full_importances.mean()) if full_importances.size > 0 else 0.0
            if mean_imp > 0:
                full_scaled = full_importances / mean_imp
            else:
                full_scaled = np.zeros_like(full_importances)
            for fname, imp, imp_scaled in zip(feature_names, full_importances, full_scaled, strict=True):
                per_feature_importances[model_name][fname].append(float(imp))
                per_feature_scaled[model_name][fname].append(float(imp_scaled))

        # Log progress for all models
        log_msg = f"[MC] Run {run_idx + 1}/{n_runs} "
        log_msg += f"XGB_recall={recalls['xgb'][-1]:.4f} XGB_logloss={loglosses['xgb'][-1]:.4f}"
        if "xgb_rf" in recalls and recalls["xgb_rf"]:
            log_msg += f" XGB_RF_recall={recalls['xgb_rf'][-1]:.4f}"
        if "catboost" in recalls and recalls["catboost"]:
            log_msg += f" CatBoost_recall={recalls['catboost'][-1]:.4f}"
        print(log_msg)

    # Aggregate across runs per model
    # (out_dir already created above during idempotency check)

    model_label_map = {
        "xgb": "xgboost",
        "xgb_rf": "xgboost_rf",
        "catboost": "catboost",
    }

    results = {}

    for model_name in model_keys:
        records = []
        recall_mean = float(np.mean(recalls[model_name])) if recalls[model_name] else float("nan")
        logloss_mean = (
            float(np.mean(loglosses[model_name])) if loglosses[model_name] else float("nan")
        )
        auc_mean = float(np.mean(aucs[model_name])) if aucs[model_name] else float("nan")
        pr_auc_mean = float(np.mean(pr_aucs[model_name])) if pr_aucs[model_name] else float("nan")

        for fname in feature_names:
            imp_values = np.array(
                per_feature_importances[model_name][fname], dtype=float
            )
            scaled_values = np.array(
                per_feature_scaled[model_name][fname], dtype=float
            )

            records.append(
                {
                    "feature": fname,
                    "scaled_importance_mean": float(scaled_values.mean())
                    if scaled_values.size
                    else 0.0,
                    "scaled_importance_std": float(scaled_values.std(ddof=0))
                    if scaled_values.size
                    else 0.0,
                    "scaled_importance_count": int(
                        np.count_nonzero(scaled_values > 0.0)
                    ),
                    "importance_mean": float(imp_values.mean())
                    if imp_values.size
                    else 0.0,
                    "importance_std": float(imp_values.std(ddof=0))
                    if imp_values.size
                    else 0.0,
                    "recall_mean": recall_mean,
                    "logloss_mean": logloss_mean,
                    "auc_mean": auc_mean,
                    "pr_auc_mean": pr_auc_mean,
                }
            )

        fi_df = pd.DataFrame.from_records(records)
        fi_df = fi_df.sort_values("scaled_importance_mean", ascending=False)
        label = model_label_map[model_name]

        out_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_{label}_feature_importance_mc{n_runs}.csv"
        )
        fi_df.to_csv(out_path, index=False)
        print(
            f"\nSaved Monte-Carlo {label} feature importances to {out_path} "
            f"(top 10 features shown below)."
        )
        print(fi_df.head(10).to_string(index=False))
        results[model_name] = fi_df

        # Basic visuals: top 50 barplot and raw vs scaled scatter
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        top_n = 50
        top_df = fi_df.head(top_n)

        plt.figure(figsize=(10, max(6, top_n * 0.18)))
        plt.barh(top_df["feature"][::-1], top_df["scaled_importance_mean"][::-1])
        plt.xlabel("Scaled importance (mean)")
        plt.title(
            f"{label} top {top_n} features\n"
            f"recall_mean={recall_mean:.3f}, "
            f"logloss_mean={logloss_mean:.3f}, "
            f"AUC_mean={auc_mean:.3f}, PR-AUC_mean={pr_auc_mean:.3f}"
        )
        plt.tight_layout()
        bar_path = (
            plots_dir
            / f"{cohort}_{age_band_fname}_{label}_top{top_n}_features_mc{n_runs}.png"
        )
        plt.savefig(bar_path, dpi=150)
        plt.close()

        # Scatter: raw vs scaled importance
        plt.figure(figsize=(6, 5))
        plt.scatter(
            fi_df["importance_mean"],
            fi_df["scaled_importance_mean"],
            alpha=0.6,
            s=10,
        )
        plt.xlabel("Raw importance (mean)")
        plt.ylabel("Scaled importance (mean)")
        plt.title(f"{label} importance: raw vs scaled")
        plt.tight_layout()
        scatter_path = (
            plots_dir
            / f"{cohort}_{age_band_fname}_{label}_normalized_vs_scaled_mc{n_runs}.png"
        )
        plt.savefig(scatter_path, dpi=150)
        plt.close()

    # Aggregated file: normalize across all model types with weighting for best model (recall)
    if "xgb" in results:
        agg_path = (
            out_dir
            / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )

        # Build all_results for aggregate_feature_importance (expects feature, importance_mean, recall_mean, logloss_mean)
        all_results = {}
        for model_name in model_keys:
            if model_name not in results:
                continue
            df = results[model_name]
            if "feature" not in df.columns or "importance_mean" not in df.columns:
                continue
            label = model_label_map[model_name]
            all_results[label] = df

        if len(all_results) >= 1:
            # Use cross-model aggregation with normalization and best-model weighting when we have multiple models
            is_multi_model_aggregated = len(all_results) >= 2
            if is_multi_model_aggregated:
                agg_combined = aggregate_feature_importance(
                    all_results, scaling_metric="recall", logger=None
                )
                # Map to schema expected by downstream: feature, scaled_importance_mean, importance_mean, recall_mean, logloss_mean
                best_model_key = max(
                    results.keys(),
                    key=lambda k: results[k]["recall_mean"].iloc[0] if len(results[k]) > 0 else 0,
                )
                best_recall = results[best_model_key]["recall_mean"].iloc[0]
                best_logloss = results[best_model_key]["logloss_mean"].iloc[0]
                agg_df = agg_combined[["feature", "importance_normalized", "importance_scaled", "n_models", "models"]].copy()
                agg_df.rename(columns={"importance_scaled": "scaled_importance_mean", "importance_normalized": "importance_mean"}, inplace=True)
                agg_df["scaled_importance_std"] = 0.0
                agg_df["scaled_importance_count"] = n_runs
                agg_df["importance_std"] = 0.0
                agg_df["recall_mean"] = best_recall
                agg_df["logloss_mean"] = best_logloss
                if "auc_mean" in list(results.values())[0].columns:
                    agg_df["auc_mean"] = list(results.values())[0]["auc_mean"].iloc[0]
                if "pr_auc_mean" in list(results.values())[0].columns:
                    agg_df["pr_auc_mean"] = list(results.values())[0]["pr_auc_mean"].iloc[0]
                print("[INFO] Aggregated feature importance: normalized across all model types with best-model (recall) weighting")
            else:
                # Single model: keep existing behavior (one model's output)
                agg_df = list(all_results.values())[0].copy()

            # Filter out features with zero or negative importance (no signal)
            initial_count = len(agg_df)
            if "scaled_importance_mean" in agg_df.columns:
                agg_df = agg_df[agg_df["scaled_importance_mean"] > 1e-10].copy()
            elif "importance_mean" in agg_df.columns:
                agg_df = agg_df[agg_df["importance_mean"] > 1e-10].copy()
            filtered_count = len(agg_df)
            if filtered_count < initial_count:
                print(f"[INFO] Filtered out {initial_count - filtered_count} features with zero/negative importance")
                print(f"[INFO] Keeping {filtered_count} features with importance > 0")

            # Remove duplicate features (keep first occurrence)
            initial_count = len(agg_df)
            agg_df = agg_df.drop_duplicates(subset=["feature"], keep="first")
            if len(agg_df) < initial_count:
                print(f"[INFO] Removed {initial_count - len(agg_df)} duplicate features")

            # Ensure sorted by importance (descending)
            if "scaled_importance_mean" in agg_df.columns:
                agg_df = agg_df.sort_values("scaled_importance_mean", ascending=False)
            elif "importance_mean" in agg_df.columns:
                agg_df = agg_df.sort_values("importance_mean", ascending=False)

            # Remove administrative/Z codes so second pass = aggregated FI minus admin Z
            agg_df = _filter_aggregated_fi_admin_codes(agg_df)

            agg_df.to_csv(agg_path, index=False)
            print(f"Saved aggregated feature importance to {agg_path}")
            print(f"[INFO] Final aggregated CSV contains {len(agg_df)} unique features with signal")
            # Print top 20 aggregated feature importances after final MC CV run
            imp_col = "scaled_importance_mean" if "scaled_importance_mean" in agg_df.columns else "importance_mean"
            top20 = agg_df.head(20)[["feature", imp_col]]
            if is_multi_model_aggregated:
                print(f"\nTop 20 aggregated feature importances (all models, normalized by best-model recall weight, MC n_runs={n_runs}):")
            else:
                single_label = list(all_results.keys())[0] if all_results else "single model"
                print(f"\nTop 20 feature importances (single model: {single_label}, MC n_runs={n_runs}):")
            print(top20.to_string(index=False))

            # Second-pass aggregated FI is always saved to pgxdatalake and used for final model train features (Step 4 / Step 6).
            # Do not write to pgx-repository so historical baseline is never overwritten.
            import io
            obj_bytes = agg_df.to_csv(index=False).encode('utf-8')
            s3_suffix = "_baseline/" if baseline else ""
            filename_agg = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
            s3_key_agg = f"gold/feature_importance/{cohort}/{age_band}/{s3_suffix}{filename_agg}"
            try:
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_key_agg,
                    Body=io.BytesIO(obj_bytes),
                    ContentType='text/csv'
                )
                print(f"✓ Uploaded aggregated feature importance to pgxdatalake: s3://{S3_BUCKET}/{s3_key_agg}")
                if not baseline:
                    print("[INFO] Second-pass FI in pgxdatalake is the source for final model train features (Step 4 / Step 6).")
            except Exception as e:
                print(f"[WARN] Failed to upload to pgxdatalake: {e}")
                print(f"  File saved locally at: {agg_path}")
        else:
            print("[WARN] No valid per-model results to aggregate; skipping aggregated CSV.")

    # Return the XGBoost boosting table by default
    return results.get("xgb", pd.DataFrame())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Monte-Carlo XGBoost feature importance on final features. "
        "This script is idempotent - it will skip if results already exist unless --force is used."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=25,
        help="Number of Monte-Carlo CV runs (default: 25)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if results already exist",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="First-pass run: write to outputs/{cohort}/_baseline/. "
        "Default is no baseline: write to outputs/{cohort}/. "
        "Use --baseline only when generating baseline FI for the first time; baseline is usually already on S3.",
    )
    parser.add_argument(
        "--no-run-baseline-if-missing",
        action="store_false",
        dest="run_baseline_if_missing",
        default=True,
        help="Do not run baseline when missing in pgx-repository; second pass will fail if no historical FI. Default is to run baseline when missing.",
    )
    args = parser.parse_args()

    run_mc_feature_importance(
        cohort=args.cohort,
        age_band=args.age_band,
        n_runs=args.n_runs,
        force=args.force,
        baseline=args.baseline,
        run_baseline_if_missing=args.run_baseline_if_missing,
    )


if __name__ == "__main__":
    main()

