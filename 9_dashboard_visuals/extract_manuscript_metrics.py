#!/usr/bin/env python3
"""
Extract manuscript-ready metrics from pipeline visualization outputs.

Harvests data missing from PIPELINE_RESULTS.md:
  1. FP-Growth top rules (by lift) per cohort/age_band/item_type
  2. DTW trajectory summary (total, target_1/0, trajectory_length stats)
  3. SHAP top-10 features per cohort/age_band (aggregate + per-bin)
  4. PGx feature coverage % (patients with >=1 PGx feature vs total)

Sources (local first, S3 fallback):
  FP-Growth : 10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{ab_fname}/{item_type}_rules.json
  DTW       : 10_risk_dashboard/visualizations/dtw/{cohort}/{ab_fname}/chart_data.json
  SHAP FI   : s3://pgxdatalake/gold/final_model/{cohort}/{ab}/{cohort}_{ab_}_xgboost_feature_importance.csv
              s3://pgxdatalake/gold/final_model/{cohort}/{ab}/bin_models/{bin}/{cohort}_{ab_}_{model}_feature_importance.csv
  PGx CSV   : s3://pgx-repository/5_pgx_analysis_checkpoint/{cohort}/{ab}/pgx_added_features_{cohort}_{ab_}.csv

Output:
  manuscript/PIPELINE_RESULTS_AUTO.json   (machine-readable, all cohorts/bands)
  manuscript/PIPELINE_RESULTS_AUTO.md     (human-readable tables for copy-paste into manuscript)

Usage:
  python 9_dashboard_visuals/extract_manuscript_metrics.py
  python 9_dashboard_visuals/extract_manuscript_metrics.py --no-s3          # skip S3 fallback
  python 9_dashboard_visuals/extract_manuscript_metrics.py --top-n-rules 10 # default 5
  python 9_dashboard_visuals/extract_manuscript_metrics.py --top-n-shap 20  # default 10
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
FPGROWTH_LOCAL = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth" / "outputs"
DTW_LOCAL = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "dtw"

COHORTS = ["opioid_ed", "non_opioid_ed"]
AGE_BANDS = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
BINS = ["low", "medium", "high", "extreme"]
MODELS = ["xgboost", "catboost"]

S3_GOLD = "s3://pgxdatalake/gold/final_model"
S3_PGX_CHECKPOINT = "s3://pgx-repository/5_pgx_analysis_checkpoint"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_manuscript_metrics")

# Minimum expected rule counts for primary age bands (13-24 through 75-84)
# Cohorts with fewer rules than this threshold trigger a WARNING
_MIN_RULES_PRIMARY = 5
_MIN_RULES_EXTREME = 1  # extreme/oldest bands may have fewer
_PRIMARY_BANDS = {"13-24", "25-44", "45-54", "55-64", "65-74", "75-84"}
_SPARSE_BANDS = {"0-12", "85-114"}
# non_opioid_ed small cohort — lower thresholds
_MIN_RULES_NON_OPIOID = 1


def _issue(
    issues: List[Dict],
    section: str,
    cohort: str,
    age_band: str,
    severity: str,
    reason: str,
    action: str,
    **extra: Any,
) -> None:
    """Append a structured diagnostic issue and emit a log line."""
    entry = {
        "section": section,
        "cohort": cohort,
        "age_band": age_band,
        "severity": severity,
        "reason": reason,
        "action": action,
        **extra,
    }
    issues.append(entry)
    prefix = "[WARN]" if severity == "warning" else "[ERR] "
    log.warning("%s %s  %s/%s — %s  →  %s", prefix, section.upper(), cohort, age_band, reason, action)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_read_bytes(s3_uri: str) -> Optional[bytes]:
    """Download S3 object bytes; return None on any error."""
    try:
        import boto3
        from botocore.exceptions import ClientError
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3")
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        return buf.getvalue()
    except Exception as e:
        log.debug("S3 read failed %s: %s", s3_uri, e)
        return None


def _s3_read_csv(s3_uri: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return (DataFrame, None) on success or (None, error_reason) on failure."""
    data = _s3_read_bytes(s3_uri)
    if data is None:
        return None, "s3_object_not_found"
    try:
        df = pd.read_csv(io.BytesIO(data))
        return df, None
    except Exception as e:
        log.debug("CSV parse failed %s: %s", s3_uri, e)
        return None, f"csv_parse_error: {e}"


# ---------------------------------------------------------------------------
# 1. FP-Growth top rules
# ---------------------------------------------------------------------------

def extract_fpgrowth_rules(
    top_n: int = 5,
    use_s3: bool = True,
    issues: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    For each (cohort, age_band, item_type): load rules JSON, return top-N by lift.
    Falls back to S3 gold/fpgrowth path if local file absent.
    Logs diagnostic issues into the shared `issues` list.
    """
    if issues is None:
        issues = []
    results: Dict[str, Any] = {}

    for cohort in COHORTS:
        results[cohort] = {}
        for ab in AGE_BANDS:
            ab_fname = ab.replace("-", "_")
            results[cohort][ab] = {}

            # item types to try (drug + icd for opioid; drug for polypharmacy)
            item_types = ["drug", "icd", "cpt"] if cohort == "opioid_ed" else ["drug"]

            for item_type in item_types:
                rules_df = None
                source = None

                # Local first
                local_path = FPGROWTH_LOCAL / cohort / ab_fname / f"{item_type}_rules.json"
                if local_path.exists():
                    try:
                        rules_df = pd.read_json(local_path)
                        source = "local"
                        log.debug("FPG local %s/%s/%s: %d rules", cohort, ab, item_type, len(rules_df))
                    except Exception as e:
                        _issue(issues, "fpgrowth", cohort, ab, "error",
                               f"{item_type}: local file parse failed: {e}",
                               f"Delete {local_path.name} and rerun FP-Growth",
                               item_type=item_type, local_path=str(local_path))

                # S3 fallback
                if rules_df is None and use_s3:
                    s3_uri = (
                        f"s3://pgxdatalake/gold/fpgrowth/{item_type}"
                        f"/cohort_name={cohort}/age_band={ab}/event_year=train/rules.json"
                    )
                    data = _s3_read_bytes(s3_uri)
                    if data:
                        try:
                            rules_df = pd.read_json(io.BytesIO(data))
                            source = "s3"
                            log.debug("FPG S3 %s/%s/%s: %d rules", cohort, ab, item_type, len(rules_df))
                        except Exception as e:
                            _issue(issues, "fpgrowth", cohort, ab, "error",
                                   f"{item_type}: S3 file parse failed: {e}",
                                   "Rerun FP-Growth step for this cohort/age_band",
                                   item_type=item_type, s3_uri=s3_uri)
                    else:
                        # Missing from both local and S3 — log conditionally
                        if not local_path.exists():
                            sev = "error" if ab in _PRIMARY_BANDS else "warning"
                            _issue(issues, "fpgrowth", cohort, ab, sev,
                                   f"{item_type}: rules.json not found locally or on S3",
                                   "Rerun 9_dashboard_visuals/run_dashboard_visuals.py --force",
                                   item_type=item_type,
                                   local_checked=str(local_path),
                                   s3_checked=s3_uri)

                if rules_df is None or rules_df.empty:
                    if rules_df is not None:  # file found but empty
                        _issue(issues, "fpgrowth", cohort, ab, "warning",
                               f"{item_type}: rules JSON loaded but contains 0 rows",
                               "Check FP-Growth min_support/min_confidence thresholds; cohort may be too sparse",
                               item_type=item_type, source=source)
                    results[cohort][ab][item_type] = {"n_rules": 0, "top_rules": [], "source": source}
                    continue

                # Normalise antecedents/consequents (may be lists or strings)
                def _join(val):
                    if isinstance(val, list):
                        return " + ".join(str(v) for v in val)
                    return str(val)

                for col in ("antecedents", "consequents"):
                    if col in rules_df.columns:
                        rules_df[col] = rules_df[col].apply(_join)

                # Sort by lift descending, take top_n
                sort_col = "lift" if "lift" in rules_df.columns else (
                    "confidence" if "confidence" in rules_df.columns else None
                )
                if sort_col:
                    rules_df = rules_df.sort_values(sort_col, ascending=False)
                else:
                    _issue(issues, "fpgrowth", cohort, ab, "warning",
                           f"{item_type}: neither 'lift' nor 'confidence' column found — cannot rank rules",
                           "Verify cohort_fpgrowth.py produces association_rules with lift/confidence columns",
                           item_type=item_type, columns=list(rules_df.columns))

                n_rules = len(rules_df)

                # Threshold check: warn if fewer rules than expected for primary bands
                min_expected = (
                    _MIN_RULES_NON_OPIOID if cohort == "non_opioid_ed"
                    else (_MIN_RULES_PRIMARY if ab in _PRIMARY_BANDS else _MIN_RULES_EXTREME)
                )
                if n_rules < min_expected:
                    _issue(issues, "fpgrowth", cohort, ab, "warning",
                           f"{item_type}: only {n_rules} rules (expected >= {min_expected} for {ab})",
                           "Check min_support/min_confidence settings or cohort case count",
                           item_type=item_type, n_rules=n_rules, min_expected=min_expected)

                top = rules_df.head(top_n)
                keep_cols = [c for c in ("antecedents", "consequents", "support", "confidence", "lift") if c in top.columns]
                top_records = top[keep_cols].round(4).to_dict(orient="records")

                results[cohort][ab][item_type] = {
                    "n_rules": n_rules,
                    "top_rules": top_records,
                    "source": source,
                }
                log.info("FPG %s/%s/%s (%s): %d total rules, top %d extracted",
                         cohort, ab, item_type, source, n_rules, len(top_records))

    return results


# ---------------------------------------------------------------------------
# 2. DTW trajectory summary
# ---------------------------------------------------------------------------

def extract_dtw_summary(
    issues: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    For each (cohort, age_band): read chart_data.json → extract summary block.
    Logs diagnostic issues into the shared `issues` list.
    """
    if issues is None:
        issues = []
    results: Dict[str, Any] = {}

    for cohort in COHORTS:
        results[cohort] = {}
        for ab in AGE_BANDS:
            ab_fname = ab.replace("-", "_")
            chart_path = DTW_LOCAL / cohort / ab_fname / "chart_data.json"

            if not chart_path.exists():
                sev = "error" if ab in _PRIMARY_BANDS else "warning"
                _issue(issues, "dtw", cohort, ab, sev,
                       "chart_data.json not found locally",
                       "Rerun create_dtw_visuals.py for this cohort/age_band",
                       path_checked=str(chart_path))
                results[cohort][ab] = {"status": "not_found"}
                continue

            try:
                with open(chart_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                _issue(issues, "dtw", cohort, ab, "error",
                       f"chart_data.json parse failed: {e}",
                       "Delete chart_data.json and rerun create_dtw_visuals.py",
                       path=str(chart_path))
                results[cohort][ab] = {"status": "parse_error", "error": str(e)}
                continue

            # Pipeline wrote an empty-state artifact
            if data.get("empty"):
                reason_detail = data.get("metrics", {}).get("reason", "unknown")
                sql_diag = data.get("metrics", {}).get("sql_diagnostics")
                _issue(issues, "dtw", cohort, ab, "warning",
                       f"chart_data.json is an empty-state artifact (reason={reason_detail})",
                       "Check create_dtw_trajectories.py output; ensure parquet data exists for this band",
                       empty_reason=reason_detail, sql_diagnostics=sql_diag)
                results[cohort][ab] = {
                    "status": "empty",
                    "empty_reason": reason_detail,
                    "charts_not_built": data.get("metrics", {}).get("charts_not_built", {}),
                }
                continue

            summary = data.get("summary", {})
            metrics = data.get("metrics", {})
            charts_built = metrics.get("charts_built", [])
            charts_not_built = metrics.get("charts_not_built", {})
            total = summary.get("total_trajectories", 0) or 0
            t1 = summary.get("target_counts", {}).get("target_1", 0) or 0

            log.info(
                "DTW %s/%s: total=%d target_1=%d charts_built=%s",
                cohort, ab, total, t1, charts_built,
            )

            # Diagnostic conditions
            if total == 0:
                _issue(issues, "dtw", cohort, ab, "error",
                       "total_trajectories=0 in summary",
                       "create_dtw_trajectories.py produced no rows; check parquet inputs",
                       charts_not_built=charts_not_built)
            elif t1 == 0 and ab not in _SPARSE_BANDS:
                _issue(issues, "dtw", cohort, ab, "warning",
                       f"target_1=0 in primary age band (total={total})",
                       "Verify target column is set correctly in create_dtw_trajectories.py",
                       total_trajectories=total)

            if charts_not_built:
                for chart_name, reason in charts_not_built.items():
                    _issue(issues, "dtw", cohort, ab, "warning",
                           f"chart '{chart_name}' not built: {reason}",
                           "Check create_dtw_visuals.py logic for this chart type",
                           chart=chart_name, chart_reason=reason)

            results[cohort][ab] = {
                "status": "ok",
                "total_trajectories": total,
                "target_1": t1,
                "target_0": summary.get("target_counts", {}).get("target_0"),
                "trajectory_length": summary.get("trajectory_length", {}),
                "trajectories_with_time_between": summary.get("trajectories_with_time_between"),
                "charts_built": charts_built,
                "charts_not_built": charts_not_built,
            }

    return results


# ---------------------------------------------------------------------------
# 3. SHAP top-N features
# ---------------------------------------------------------------------------

def _extract_top_features(
    df: pd.DataFrame,
    top_n: int,
    label: str,
    issues: List[Dict],
    cohort: str,
    ab: str,
    **extra: Any,
) -> Optional[List[Dict]]:
    """Extract top-N rows from a feature importance DataFrame; log issues if columns are unexpected."""
    imp_col = next(
        (c for c in ("importance", "gain", "weight", "cover", "total_gain") if c in df.columns),
        None,
    )
    feat_col = next(
        (c for c in ("feature", "Feature", "name") if c in df.columns),
        df.columns[0] if len(df.columns) >= 1 else None,
    )
    if imp_col is None:
        _issue(issues, "shap", cohort, ab, "error",
               f"{label}: no importance column found — columns={list(df.columns)}",
               "Verify run_final_model.py saves XGBoost FI with 'importance'/'gain' column",
               **extra)
        return None
    if feat_col is None:
        _issue(issues, "shap", cohort, ab, "error",
               f"{label}: no feature name column found — columns={list(df.columns)}",
               "Verify FI CSV has 'feature' or 'Feature' column",
               **extra)
        return None
    if len(df) < top_n:
        _issue(issues, "shap", cohort, ab, "warning",
               f"{label}: only {len(df)} features available (requested top_{top_n})",
               "Model may have been trained on fewer features than expected",
               n_features=len(df), **extra)
    top = df.nlargest(top_n, imp_col)[[feat_col, imp_col]]
    return top.rename(columns={feat_col: "feature", imp_col: "importance"}).round(4).to_dict(orient="records")


def extract_shap_features(
    top_n: int = 10,
    use_s3: bool = True,
    issues: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    For each (cohort, age_band): read aggregate XGBoost FI CSV → top-N features by importance.
    Also reads per-bin FI CSVs.
    Logs diagnostic issues into the shared `issues` list.
    """
    if issues is None:
        issues = []
    results: Dict[str, Any] = {}

    for cohort in COHORTS:
        results[cohort] = {}
        for ab in AGE_BANDS:
            ab_fname = ab.replace("-", "_")
            results[cohort][ab] = {"aggregate": None, "per_bin": {}}

            if not use_s3:
                results[cohort][ab]["aggregate"] = None
                continue

            # Aggregate FI
            agg_uri = f"{S3_GOLD}/{cohort}/{ab}/{cohort}_{ab_fname}_xgboost_feature_importance.csv"
            fi_df, err = _s3_read_csv(agg_uri)
            if fi_df is None:
                sev = "error" if ab in _PRIMARY_BANDS else "warning"
                _issue(issues, "shap", cohort, ab, sev,
                       f"aggregate FI CSV not found on S3 ({err})",
                       "Check gold/final_model path; rerun 6_final_model step if missing",
                       s3_uri=agg_uri)
            elif fi_df.empty:
                _issue(issues, "shap", cohort, ab, "error",
                       "aggregate FI CSV is empty (0 rows)",
                       "Model training may have failed for this cohort/age_band",
                       s3_uri=agg_uri)
            else:
                top_feats = _extract_top_features(
                    fi_df, top_n, "aggregate", issues, cohort, ab, s3_uri=agg_uri
                )
                results[cohort][ab]["aggregate"] = top_feats
                if top_feats:
                    log.info("SHAP agg %s/%s: top %d features extracted", cohort, ab, len(top_feats))

            # Per-bin FI (XGBoost) — expected bins depend on cohort
            expected_bins = ["low", "medium", "high"] if cohort == "opioid_ed" else ["low"]
            for bin_name in BINS:
                bin_uri = (
                    f"{S3_GOLD}/{cohort}/{ab}/bin_models/{bin_name}"
                    f"/{cohort}_{ab_fname}_xgboost_feature_importance.csv"
                )
                fi_bin, bin_err = _s3_read_csv(bin_uri)
                if fi_bin is None:
                    if bin_name in expected_bins and ab in _PRIMARY_BANDS:
                        _issue(issues, "shap", cohort, ab, "warning",
                               f"per-bin FI CSV missing for expected bin '{bin_name}' ({bin_err})",
                               "Rerun train_per_bin() in notebook 3 for this cohort/age_band",
                               bin_name=bin_name, s3_uri=bin_uri)
                    log.debug("SHAP bin not found (ok for sparse): %s", bin_uri)
                    continue
                if fi_bin.empty:
                    _issue(issues, "shap", cohort, ab, "warning",
                           f"per-bin FI CSV empty for bin '{bin_name}'",
                           "Check train_per_bin() output for this bin",
                           bin_name=bin_name)
                    continue
                top_bin = _extract_top_features(
                    fi_bin, top_n, f"bin_{bin_name}", issues, cohort, ab, bin_name=bin_name
                )
                if top_bin:
                    results[cohort][ab]["per_bin"][bin_name] = top_bin
                    log.info("SHAP bin %s/%s/%s: top %d features", cohort, ab, bin_name, len(top_bin))

    return results


# ---------------------------------------------------------------------------
# 4. PGx feature coverage
# ---------------------------------------------------------------------------

def extract_pgx_coverage(
    use_s3: bool = True,
    issues: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    For each (cohort, age_band): read pgx_added_features CSV from S3 checkpoint.
    Computes:
      - n_total: rows in CSV
      - n_pgx: rows where any pgx column is non-zero / non-null
      - pct_pgx: n_pgx / n_total * 100
    Logs diagnostic issues into the shared `issues` list.
    """
    if issues is None:
        issues = []
    results: Dict[str, Any] = {}

    for cohort in COHORTS:
        results[cohort] = {}
        for ab in AGE_BANDS:
            ab_fname = ab.replace("-", "_")
            s3_uri = (
                f"{S3_PGX_CHECKPOINT}/{cohort}/{ab}"
                f"/pgx_added_features_{cohort}_{ab_fname}.csv"
            )

            if not use_s3:
                results[cohort][ab] = {"status": "skipped_no_s3"}
                continue

            df, err = _s3_read_csv(s3_uri)
            if df is None:
                _issue(issues, "pgx", cohort, ab, "error",
                       f"PGx checkpoint CSV not found on S3 ({err})",
                       "Rerun step 5 (5_pgx_analysis) for this cohort/age_band",
                       s3_uri=s3_uri)
                results[cohort][ab] = {"status": "not_found", "s3_uri": s3_uri}
                continue

            if df.empty:
                _issue(issues, "pgx", cohort, ab, "error",
                       "PGx checkpoint CSV is empty (0 rows)",
                       "PGx feature attachment step may have failed; check 5_pgx_analysis_log",
                       s3_uri=s3_uri)
                results[cohort][ab] = {"status": "empty"}
                continue

            n_total = len(df)
            id_cols = {c for c in df.columns if c.lower() in (
                "mi_person_key", "person_key", "target", "age_band", "cohort_name",
                "event_year", "index",
            )}
            pgx_cols = [c for c in df.columns if c not in id_cols]

            if not pgx_cols:
                _issue(issues, "pgx", cohort, ab, "error",
                       f"CSV has {n_total} rows but 0 PGx feature columns (only id cols found: {list(id_cols)})",
                       "5_pgx_analysis may have written identity-only CSV; check PGx gene column generation",
                       s3_uri=s3_uri, columns=list(df.columns))
                results[cohort][ab] = {
                    "status": "no_pgx_columns",
                    "n_total": n_total,
                    "columns_found": list(df.columns),
                }
                continue

            has_pgx = (df[pgx_cols].fillna(0).abs() > 0).any(axis=1)
            n_pgx = int(has_pgx.sum())
            pct = round(n_pgx / n_total * 100, 2) if n_total > 0 else 0.0

            # Suspicious coverage conditions
            if pct == 0.0:
                _issue(issues, "pgx", cohort, ab, "error",
                       f"0% PGx coverage ({n_total} patients, 0 with any PGx value)",
                       "All PGx columns are zero/null; check PharmGKB VIP report fetch and join logic",
                       s3_uri=s3_uri, n_total=n_total, pgx_cols_sample=pgx_cols[:5])
            elif pct < 1.0 and ab in _PRIMARY_BANDS:
                _issue(issues, "pgx", cohort, ab, "warning",
                       f"Very low PGx coverage ({pct:.2f}%) for primary age band",
                       "Verify PharmGKB VIP gene lists cover this age band's top drugs",
                       pct_pgx=pct, n_pgx=n_pgx, n_total=n_total)
            elif pct > 99.5:
                _issue(issues, "pgx", cohort, ab, "warning",
                       f"Suspiciously high PGx coverage ({pct:.2f}%) — possible all-ones column",
                       "Inspect pgx_cols for constant-value indicator columns that inflate coverage",
                       pct_pgx=pct, pgx_cols_sample=pgx_cols[:5])

            results[cohort][ab] = {
                "status": "ok",
                "n_total": n_total,
                "n_pgx": n_pgx,
                "pct_pgx": pct,
                "pgx_columns": len(pgx_cols),
                "sample_pgx_columns": pgx_cols[:10],
            }
            log.info(
                "PGx %s/%s: %d/%d patients with PGx data (%.1f%%), %d PGx cols",
                cohort, ab, n_pgx, n_total, pct, len(pgx_cols),
            )

    return results


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _md_fpgrowth(fpg: Dict) -> str:
    lines = ["## FP-Growth Top Rules\n"]
    for cohort in COHORTS:
        lines.append(f"### {cohort}\n")
        for ab in AGE_BANDS:
            ab_data = fpg.get(cohort, {}).get(ab, {})
            for item_type, d in ab_data.items():
                n = d.get("n_rules", 0)
                top = d.get("top_rules", [])
                if not top:
                    continue
                lines.append(f"**{ab} / {item_type}** — {n:,} total rules\n")
                lines.append("| Antecedents | Consequents | Support | Confidence | Lift |")
                lines.append("|:------------|:-----------|--------:|-----------:|-----:|")
                for r in top:
                    lines.append(
                        f"| {r.get('antecedents','')} | {r.get('consequents','')} "
                        f"| {r.get('support',''):.4f} | {r.get('confidence',''):.4f} "
                        f"| {r.get('lift',''):.4f} |"
                    )
                lines.append("")
    return "\n".join(lines)


def _md_dtw(dtw: Dict) -> str:
    lines = ["## DTW Trajectory Summary\n"]
    for cohort in COHORTS:
        lines.append(f"### {cohort}\n")
        lines.append("| Age Band | Total Traj | Target=1 | Target=0 | Traj Length (mean) |")
        lines.append("|:---------|----------:|---------:|---------:|-------------------:|")
        for ab in AGE_BANDS:
            d = dtw.get(cohort, {}).get(ab, {})
            if d.get("status") != "ok":
                lines.append(f"| {ab} | — | — | — | — |")
                continue
            tl = d.get("trajectory_length", {})
            mean_len = tl.get("mean", "—") if tl else "—"
            lines.append(
                f"| {ab} | {d.get('total_trajectories','—'):,} "
                f"| {d.get('target_1','—'):,} | {d.get('target_0','—'):,} "
                f"| {mean_len} |"
            )
        lines.append("")
    return "\n".join(lines)


def _md_shap(shap: Dict, top_n: int) -> str:
    lines = [f"## SHAP Top-{top_n} Features (XGBoost, Aggregate)\n"]
    for cohort in COHORTS:
        lines.append(f"### {cohort}\n")
        for ab in AGE_BANDS:
            agg = shap.get(cohort, {}).get(ab, {}).get("aggregate")
            if not agg:
                continue
            lines.append(f"**{ab}**\n")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|-----:|:--------|----------:|")
            for i, row in enumerate(agg, 1):
                lines.append(f"| {i} | {row['feature']} | {row['importance']:.4f} |")
            lines.append("")
    return "\n".join(lines)


def _md_issues(issues: List[Dict]) -> str:
    if not issues:
        return "## Diagnostic Issues\n\n_No issues detected._\n"
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    lines = [
        "## Diagnostic Issues\n",
        f"**{len(errors)} error(s)** · **{len(warnings)} warning(s)**\n",
    ]
    if errors:
        lines.append("### Errors (data missing or unparseable — rerun required)\n")
        lines.append("| Section | Cohort | Age Band | Reason | Action |")
        lines.append("|:--------|:-------|:---------|:-------|:-------|")
        for i in errors:
            lines.append(
                f"| {i['section']} | {i['cohort']} | {i['age_band']} "
                f"| {i['reason']} | {i['action']} |"
            )
        lines.append("")
    if warnings:
        lines.append("### Warnings (data present but suspicious or sparse)\n")
        lines.append("| Section | Cohort | Age Band | Reason | Action |")
        lines.append("|:--------|:-------|:---------|:-------|:-------|")
        for i in warnings:
            lines.append(
                f"| {i['section']} | {i['cohort']} | {i['age_band']} "
                f"| {i['reason']} | {i['action']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _md_pgx(pgx: Dict) -> str:
    lines = ["## PGx Feature Coverage\n"]
    for cohort in COHORTS:
        lines.append(f"### {cohort}\n")
        lines.append("| Age Band | Total N | PGx N | PGx % | PGx Cols |")
        lines.append("|:---------|--------:|------:|------:|---------:|")
        for ab in AGE_BANDS:
            d = pgx.get(cohort, {}).get(ab, {})
            if d.get("status") != "ok":
                lines.append(f"| {ab} | — | — | — | — |")
                continue
            lines.append(
                f"| {ab} | {d['n_total']:,} | {d['n_pgx']:,} "
                f"| {d['pct_pgx']:.1f}% | {d['pgx_columns']} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract manuscript metrics from pipeline outputs.")
    ap.add_argument("--no-s3", action="store_true", help="Skip S3 fallback (local files only)")
    ap.add_argument("--top-n-rules", type=int, default=5, help="Top N FP-Growth rules per cohort/band/item_type (default 5)")
    ap.add_argument("--top-n-shap", type=int, default=10, help="Top N SHAP features (default 10)")
    ap.add_argument("--skip-fpgrowth", action="store_true")
    ap.add_argument("--skip-dtw", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-pgx", action="store_true")
    args = ap.parse_args()

    use_s3 = not args.no_s3

    log.info("=" * 60)
    log.info("extract_manuscript_metrics  (use_s3=%s)", use_s3)
    log.info("=" * 60)

    issues: List[Dict] = []  # shared accumulator threaded through all sections

    output: Dict[str, Any] = {
        "generated_by": "extract_manuscript_metrics.py",
        "top_n_rules": args.top_n_rules,
        "top_n_shap": args.top_n_shap,
    }

    # 1. FP-Growth
    if not args.skip_fpgrowth:
        log.info("--- FP-Growth rules ---")
        output["fpgrowth"] = extract_fpgrowth_rules(
            top_n=args.top_n_rules, use_s3=use_s3, issues=issues
        )
    else:
        log.info("--- FP-Growth: skipped ---")

    # 2. DTW
    if not args.skip_dtw:
        log.info("--- DTW trajectory summary ---")
        output["dtw"] = extract_dtw_summary(issues=issues)
    else:
        log.info("--- DTW: skipped ---")

    # 3. SHAP
    if not args.skip_shap:
        log.info("--- SHAP feature importance ---")
        output["shap"] = extract_shap_features(
            top_n=args.top_n_shap, use_s3=use_s3, issues=issues
        )
    else:
        log.info("--- SHAP: skipped ---")

    # 4. PGx coverage
    if not args.skip_pgx:
        log.info("--- PGx feature coverage ---")
        output["pgx_coverage"] = extract_pgx_coverage(use_s3=use_s3, issues=issues)
    else:
        log.info("--- PGx: skipped ---")

    # Attach issues to output JSON
    output["issues"] = issues
    output["issue_summary"] = {
        "total": len(issues),
        "errors": sum(1 for i in issues if i["severity"] == "error"),
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
        "by_section": {
            sec: sum(1 for i in issues if i["section"] == sec)
            for sec in ("fpgrowth", "dtw", "shap", "pgx")
        },
    }

    # Emit final issues summary to log
    log.info("=" * 60)
    log.info("ISSUES SUMMARY: %d total (%d errors, %d warnings)",
             output["issue_summary"]["total"],
             output["issue_summary"]["errors"],
             output["issue_summary"]["warnings"])
    for sec, count in output["issue_summary"]["by_section"].items():
        if count:
            log.info("  %-10s %d issue(s)", sec, count)
    if output["issue_summary"]["errors"] > 0:
        log.warning("ACTION REQUIRED: %d section(s) have errors — rerun pipeline for affected cohorts.",
                    output["issue_summary"]["errors"])
    log.info("=" * 60)

    # Write JSON
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    json_out = MANUSCRIPT_DIR / "PIPELINE_RESULTS_AUTO.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Wrote %s", json_out)

    # Write Markdown
    md_sections = [
        "# Pipeline Results — Auto-Extracted Manuscript Metrics\n",
        "_Generated by `9_dashboard_visuals/extract_manuscript_metrics.py`_\n",
        "---\n",
        _md_issues(issues),
    ]
    if "fpgrowth" in output:
        md_sections.append(_md_fpgrowth(output["fpgrowth"]))
    if "dtw" in output:
        md_sections.append(_md_dtw(output["dtw"]))
    if "shap" in output:
        md_sections.append(_md_shap(output["shap"], args.top_n_shap))
    if "pgx_coverage" in output:
        md_sections.append(_md_pgx(output["pgx_coverage"]))

    md_out = MANUSCRIPT_DIR / "PIPELINE_RESULTS_AUTO.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections))
    log.info("Wrote %s", md_out)

    log.info("Done.")


if __name__ == "__main__":
    main()
