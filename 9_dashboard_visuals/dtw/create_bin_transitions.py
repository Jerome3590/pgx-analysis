#!/usr/bin/env python3
"""
Bin transition analysis: track patients moving between event-density bins across years.

For each cohort/age_band, computes per-patient n_event_bin for each year in the training
window (2016-2019), then identifies transitions (e.g. low→medium, medium→extreme) and
patients whose clinical utilization escalated or de-escalated over time.

Output: density/transitions/bin_transitions.json with:
  - transition_matrix: {from_bin: {to_bin: count}} across all consecutive year pairs
  - sankey_nodes / sankey_links: for dashboard Sankey diagram
  - escalation_rate: fraction of patients whose bin increased year-over-year
  - de_escalation_rate: fraction whose bin decreased
  - stable_rate: fraction whose bin stayed the same
  - top_escalation_drugs: most common drug codes among escalating patients
  - top_de_escalation_drugs: most common drug codes among de-escalating patients
  - per_year_distribution: {year: {bin: count}} for trend visualization

Usage:
    python create_bin_transitions.py --cohort opioid_ed --age-band 13-24
    python create_bin_transitions.py --cohort opioid_ed --age-band 13-24 --project-root /path/to/repo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.pipeline_logger import setup_pipeline_logger  # noqa: E402
from py_helpers.model_data_paths import resolve_model_events_paths  # noqa: E402
from py_helpers.event_density_utils import (  # noqa: E402
    DENSITY_BINS,
    assign_n_event_bins,
    load_thresholds,
)

TRAIN_YEARS = [2016, 2017, 2018, 2019]
BIN_ORDER = {b: i for i, b in enumerate(DENSITY_BINS)}  # low=0, medium=1, high=2, extreme=3
THRESHOLDS_S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
THRESHOLDS_S3_PREFIX = "gold/dashboard/models"
EVENTS_S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
DASHBOARD_AGE_BANDS = ("13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114")
DASHBOARD_COHORTS = ("opioid_ed", "non_opioid_ed")
# Legacy folder names used by older DTW-filter extracts
AGE_BAND_ALIASES = {
    "85-114": ("85-114", "85-94"),
    "85-94": ("85-94", "85-114"),
}


def _age_variants(age_band: str) -> List[str]:
    hyphen = age_band.replace("_", "-")
    under = age_band.replace("-", "_")
    aliases = AGE_BAND_ALIASES.get(hyphen, (hyphen,))
    out: List[str] = []
    for token in (hyphen, under, *aliases, *(a.replace("-", "_") for a in aliases)):
        if token not in out:
            out.append(token)
    return out


def _dtw_output_root(project_root: Path) -> Path:
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


def _load_thresholds_any(project_root: Path, cohort_name: str, age_band: str, logger=None):
    """Local Step 6 JSON first, then gold/dashboard/models on S3."""
    age_band_fname = age_band.replace("-", "_")
    local_path = (
        project_root / "6_final_model" / "outputs" / cohort_name / age_band_fname
        / "n_event_bin_thresholds.json"
    )
    if local_path.exists():
        return load_thresholds(local_path)
    try:
        import boto3
        s3 = boto3.client("s3")
        last_exc = None
        for token in _age_variants(age_band):
            key = f"{THRESHOLDS_S3_PREFIX}/{cohort_name}/{token.replace('-', '_')}/n_event_bin_thresholds.json"
            try:
                body = s3.get_object(Bucket=THRESHOLDS_S3_BUCKET, Key=key)["Body"].read()
                tmp = local_path
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(body)
                if logger:
                    logger.info("Downloaded thresholds from s3://%s/%s", THRESHOLDS_S3_BUCKET, key)
                return load_thresholds(tmp)
            except Exception as exc:
                last_exc = exc
                continue
        if logger:
            logger.warning("Could not load n_event_bin_thresholds.json: %s", last_exc)
        return None
    except Exception as exc:
        if logger:
            logger.warning("Could not load n_event_bin_thresholds.json: %s", exc)
        return None


def _s3_download(key: str, dest: Path, logger=None) -> bool:
    """HEAD + download. Avoids DuckDB httpfs 400s on hive-style '=' keys."""
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.head_object(Bucket=EVENTS_S3_BUCKET, Key=key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(EVENTS_S3_BUCKET, key, str(dest))
        return True
    except Exception as exc:
        if logger:
            logger.info("S3 miss %s (%s)", key, exc)
        return False


def _person_year_from_s3(cohort_name: str, age_band: str, logger=None) -> Optional[pd.DataFrame]:
    """
    Count events per patient-year from, in order:
      1. gold/cohorts_model_data model_events.parquet
      2. gold/dtw_filter event_intervals (including 85-94 alias)
      3. gold/cohorts year partitions
    Returns columns: mi_person_key, event_year, n_events
    """
    import duckdb

    year_list = ", ".join(str(y) for y in TRAIN_YEARS)
    variants = _age_variants(age_band)
    cache = Path(os.environ.get("TEMP", "/tmp")) / "pgx_bin_transitions_cache" / cohort_name / age_band.replace("-", "_")
    cache.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    try:
        def _sql_path(p: Path) -> str:
            return str(p).replace("\\", "/")

        def _group_dates(from_sql: str) -> Optional[pd.DataFrame]:
            return con.execute(
                f"""
                SELECT mi_person_key,
                       YEAR(CAST(event_date AS DATE))::INTEGER AS event_year,
                       COUNT(*)::BIGINT AS n_events
                FROM ({from_sql}) src
                WHERE mi_person_key IS NOT NULL
                  AND YEAR(CAST(event_date AS DATE)) IN ({year_list})
                GROUP BY 1, 2
                """
            ).df()

        for token in variants:
            key = f"gold/cohorts_model_data/cohort_name={cohort_name}/age_band={token}/model_events.parquet"
            dest = cache / f"model_events_{token}.parquet"
            if _s3_download(key, dest, logger):
                df = con.execute(
                    f"""
                    SELECT mi_person_key,
                           CAST(event_year AS INTEGER) AS event_year,
                           COUNT(*)::BIGINT AS n_events
                    FROM read_parquet('{_sql_path(dest)}')
                    WHERE mi_person_key IS NOT NULL
                      AND CAST(event_year AS INTEGER) IN ({year_list})
                    GROUP BY 1, 2
                    """
                ).df()
                if df is not None and not df.empty:
                    if logger:
                        logger.info("Loaded %s person-years from s3://%s/%s", f"{len(df):,}", EVENTS_S3_BUCKET, key)
                    return df

        for token in variants:
            hyphen = token.replace("_", "-")
            under = token.replace("-", "_")
            key = f"gold/dtw_filter/{cohort_name}/{hyphen}/event_intervals_{cohort_name}_{under}.parquet"
            dest = cache / f"event_intervals_{under}.parquet"
            if _s3_download(key, dest, logger):
                df = _group_dates(f"SELECT mi_person_key, event_date FROM read_parquet('{_sql_path(dest)}')")
                if df is not None and not df.empty:
                    if logger:
                        logger.info("Loaded %s person-years from s3://%s/%s", f"{len(df):,}", EVENTS_S3_BUCKET, key)
                    return df

        for token in variants:
            local_years: List[Path] = []
            hyphen = token.replace("_", "-")
            for year in TRAIN_YEARS:
                key = (
                    f"gold/cohorts/cohort_name={cohort_name}/"
                    f"event_year={year}/age_band={hyphen}/cohort.parquet"
                )
                dest = cache / f"cohort_{hyphen}_{year}.parquet"
                if _s3_download(key, dest, logger):
                    local_years.append(dest)
            if not local_years:
                continue
            union_sql = " UNION ALL ".join(
                f"SELECT mi_person_key, event_date FROM read_parquet('{_sql_path(p)}')"
                for p in local_years
            )
            df = _group_dates(union_sql)
            if df is not None and not df.empty:
                if logger:
                    logger.info("Loaded %s person-years from gold/cohorts %s/%s (%d years)", f"{len(df):,}", cohort_name, hyphen, len(local_years))
                return df
        return None
    finally:
        con.close()


def compute_bin_transitions(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    force: bool = False,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """Compute per-patient bin transitions across years and write output JSON."""
    def _log(level: str, msg: str, *args: Any) -> None:
        if logger is not None:
            getattr(logger, level)(msg, *args)
        else:
            print(f"[{level.upper()}] " + (msg % args if args else msg))

    age_band_fname = age_band.replace("-", "_")
    out_dir = _dtw_output_root(project_root) / cohort_name / age_band_fname / "density" / "transitions"
    out_path = out_dir / "bin_transitions.json"

    if not force and out_path.exists():
        _log("info", "Bin transitions already exist at %s; skipping (use --force)", out_path)
        return json.loads(out_path.read_text(encoding="utf-8"))

    thresholds = _load_thresholds_any(project_root, cohort_name, age_band, logger)
    if not thresholds:
        _log("warning", "n_event_bin_thresholds.json not found locally or on S3; cannot compute bin transitions")
        return None

    events_per_patient_year = None
    paths = resolve_model_events_paths(project_root, cohort_name, age_band)
    if paths:
        try:
            import duckdb
            con = duckdb.connect(":memory:")
            if len(paths) == 1:
                from_clause = f"read_parquet('{str(paths[0]).replace(chr(92), '/')}')"
            else:
                from_clause = (
                    f"(SELECT * FROM read_parquet('{str(paths[0]).replace(chr(92), '/')}') "
                    f"UNION ALL SELECT * FROM read_parquet('{str(paths[1]).replace(chr(92), '/')}'))"
                )
            year_list = ", ".join(str(y) for y in TRAIN_YEARS)
            df = con.execute(
                f"SELECT mi_person_key, event_year FROM {from_clause} "
                f"WHERE event_year IN ({year_list}) AND mi_person_key IS NOT NULL"
            ).df()
            con.close()
            if df is not None and not df.empty:
                df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
                df = df.dropna(subset=["event_year"]).copy()
                df["event_year"] = df["event_year"].astype(int)
                df = df[df["event_year"].isin(TRAIN_YEARS)]
                events_per_patient_year = (
                    df.groupby(["mi_person_key", "event_year"]).size().rename("n_events").reset_index()
                )
        except Exception as e:
            _log("warning", "Local model_events unreadable (%s); trying S3 sources", e)

    if events_per_patient_year is None or events_per_patient_year.empty:
        events_per_patient_year = _person_year_from_s3(cohort_name, age_band, logger)

    if events_per_patient_year is None or events_per_patient_year.empty:
        _log("warning", "No person-year event counts for %s/%s", cohort_name, age_band)
        return None

    events_per_patient_year["event_year"] = pd.to_numeric(
        events_per_patient_year["event_year"], errors="coerce"
    )
    events_per_patient_year = events_per_patient_year.dropna(subset=["event_year"]).copy()
    events_per_patient_year["event_year"] = events_per_patient_year["event_year"].astype(int)
    events_per_patient_year = events_per_patient_year[
        events_per_patient_year["event_year"].isin(TRAIN_YEARS)
    ]
    events_per_patient_year["n_event_bin"] = assign_n_event_bins(
        events_per_patient_year.set_index("mi_person_key")["n_events"],
        thresholds,
    ).values
    _log("info", "Assigned bins for %d patient-year rows", len(events_per_patient_year))

    # Per-year distribution
    per_year_dist: Dict[str, Dict[str, int]] = {}
    for year in TRAIN_YEARS:
        yr_df = events_per_patient_year[events_per_patient_year["event_year"] == year]
        per_year_dist[str(year)] = {b: int((yr_df["n_event_bin"] == b).sum()) for b in DENSITY_BINS}

    # Build transition matrix from consecutive year pairs
    transition_matrix: Dict[str, Dict[str, int]] = {b: {b2: 0 for b2 in DENSITY_BINS} for b in DENSITY_BINS}
    pair_matrices: List[tuple] = []

    pivot = events_per_patient_year.pivot(index="mi_person_key", columns="event_year", values="n_event_bin")
    years_present = sorted([y for y in TRAIN_YEARS if y in pivot.columns])

    for i in range(len(years_present) - 1):
        yr_from, yr_to = years_present[i], years_present[i + 1]
        pair = pivot[[yr_from, yr_to]].dropna()
        mat = {b: {b2: 0 for b2 in DENSITY_BINS} for b in DENSITY_BINS}
        counts = pair.value_counts()
        for (b_from, b_to), n in counts.items():
            b_from, b_to = str(b_from), str(b_to)
            n = int(n)
            if b_from in mat and b_to in mat[b_from]:
                mat[b_from][b_to] += n
                transition_matrix[b_from][b_to] += n
        pair_matrices.append((yr_from, yr_to, mat))

    # Escalation / de-escalation / stable rates
    total_transitions = n_escalate = n_deescalate = n_stable = 0
    for _yr_from, _yr_to, mat in pair_matrices:
        for b_from in DENSITY_BINS:
            for b_to in DENSITY_BINS:
                n = mat[b_from][b_to]
                total_transitions += n
                cmp = BIN_ORDER.get(b_to, 0) - BIN_ORDER.get(b_from, 0)
                if cmp > 0:
                    n_escalate += n
                elif cmp < 0:
                    n_deescalate += n
                else:
                    n_stable += n

    escalation_rate = round(n_escalate / total_transitions, 4) if total_transitions > 0 else 0.0
    de_escalation_rate = round(n_deescalate / total_transitions, 4) if total_transitions > 0 else 0.0
    stable_rate = round(n_stable / total_transitions, 4) if total_transitions > 0 else 0.0

    # Sankey nodes and links
    sankey_nodes = [{"id": f"{b}_{yr}", "label": f"{b} ({yr})", "bin": b, "year": yr}
                    for yr in years_present for b in DENSITY_BINS
                    if per_year_dist.get(str(yr), {}).get(b, 0) > 0]
    node_id_map = {n["id"]: idx for idx, n in enumerate(sankey_nodes)}

    sankey_links: List[Dict] = []
    for yr_from, yr_to, mat in pair_matrices:
        for b_from in DENSITY_BINS:
            for b_to in DENSITY_BINS:
                count = mat[b_from].get(b_to, 0)
                if count == 0:
                    continue
                src_id = f"{b_from}_{yr_from}"
                tgt_id = f"{b_to}_{yr_to}"
                if src_id in node_id_map and tgt_id in node_id_map:
                    sankey_links.append({
                        "source": node_id_map[src_id],
                        "target": node_id_map[tgt_id],
                        "value": count,
                        "from_bin": b_from,
                        "to_bin": b_to,
                    })

    result = {
        "cohort": cohort_name,
        "age_band": age_band.replace("_", "-"),
        "transition_matrix": transition_matrix,
        "sankey_nodes": sankey_nodes,
        "sankey_links": sankey_links,
        "escalation_rate": escalation_rate,
        "de_escalation_rate": de_escalation_rate,
        "stable_rate": stable_rate,
        "total_transitions": total_transitions,
        "n_patients_tracked": int(pivot.shape[0]),
        "years_present": years_present,
        "per_year_distribution": per_year_dist,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log("info", "Bin transitions written: %s (patients=%d, transitions=%d, escalation=%.1f%%)",
         out_path, result["n_patients_tracked"], total_transitions, escalation_rate * 100)

    # S3 upload
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() not in ("1", "true", "yes"):
        try:
            import boto3 as _boto3
            s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
            dash_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "pgx")
            age_hyphen = age_band.replace("_", "-")
            s3_key = f"{dash_prefix.rstrip('/')}/visualizations/dtw/{cohort_name}/{age_hyphen}/density/transitions/bin_transitions.json"
            _boto3.client("s3").put_object(
                Bucket=s3_bucket, Key=s3_key,
                Body=out_path.read_bytes(), ContentType="application/json",
            )
            _log("info", "Bin transitions uploaded: s3://%s/%s", s3_bucket, s3_key)
        except Exception as e:
            _log("warning", "Bin transitions S3 upload failed: %s", e)

    return result


def _run_one(project_root: Path, cohort: str, age_band: str, force: bool) -> Optional[Dict[str, Any]]:
    logger = setup_pipeline_logger(
        step_name="9_dtw",
        cohort=cohort,
        age_band=age_band,
        script_name="create_bin_transitions",
    )
    result = compute_bin_transitions(
        project_root=project_root,
        cohort_name=cohort,
        age_band=age_band,
        force=force,
        logger=logger.logger,
    )
    if result is None:
        logger.warning("Bin transitions not produced for %s/%s", cohort, age_band)
        logger.log_summary()
        return None
    logger.info(
        "Done: %d patients tracked, %d transitions, escalation=%.1f%%, de-escalation=%.1f%%, stable=%.1f%%",
        result["n_patients_tracked"],
        result["total_transitions"],
        result["escalation_rate"] * 100,
        result["de_escalation_rate"] * 100,
        result["stable_rate"] * 100,
    )
    logger.log_summary()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute event-density bin transitions across years")
    parser.add_argument("--cohort", "--cohort-name", dest="cohort")
    parser.add_argument("--age-band")
    parser.add_argument("--all", action="store_true", help="Build every dashboard cohort × age band")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if not args.all and (not args.cohort or not args.age_band):
        parser.error("Provide --cohort and --age-band, or --all")

    project_root = Path(args.project_root).resolve()
    combos = (
        [(c, ab) for c in DASHBOARD_COHORTS for ab in DASHBOARD_AGE_BANDS]
        if args.all
        else [(args.cohort, args.age_band)]
    )
    inventory: List[Dict[str, Any]] = []
    failed = 0
    for cohort, age_band in combos:
        result = _run_one(project_root, cohort, age_band, args.force)
        row = {"cohort": cohort, "age_band": age_band, "ok": bool(result)}
        if result:
            row.update({
                "n_patients_tracked": result["n_patients_tracked"],
                "total_transitions": result["total_transitions"],
                "escalation_rate": result["escalation_rate"],
                "dashboard_s3": (
                    f"s3://{os.environ.get('S3_DASHBOARD_BUCKET', 'jerome-dixon.io')}/"
                    f"{os.environ.get('S3_DASHBOARD_PREFIX', 'pgx').rstrip('/')}/"
                    f"visualizations/dtw/{cohort}/{age_band.replace('_', '-')}/density/transitions/bin_transitions.json"
                ),
            })
        else:
            failed += 1
        inventory.append(row)

    inv_path = _dtw_output_root(project_root) / "bin_transition_rebuild_inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps({
        "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "n_ok": len(inventory) - failed,
        "n_failed": failed,
        "bands": inventory,
        "rebuild_notes": {
            "thresholds": "s3://pgxdatalake/gold/dashboard/models/{cohort}/{age_under}/n_event_bin_thresholds.json",
            "events_preferred": "s3://pgxdatalake/gold/cohorts_model_data/cohort_name={cohort}/age_band={age}/model_events.parquet",
            "events_dtw_filter": "s3://pgxdatalake/gold/dtw_filter/{cohort}/{age}/event_intervals_{cohort}_{age_under}.parquet",
            "events_gold_cohorts": "s3://pgxdatalake/gold/cohorts/cohort_name={cohort}/event_year={year}/age_band={age}/cohort.parquet",
            "command": "python 9_dashboard_visuals/dtw/create_bin_transitions.py --all --force",
        },
    }, indent=2), encoding="utf-8")
    try:
        import boto3
        boto3.client("s3").put_object(
            Bucket=os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io"),
            Key=f"{os.environ.get('S3_DASHBOARD_PREFIX', 'pgx').rstrip('/')}/visualizations/dtw/bin_transition_rebuild_inventory.json",
            Body=inv_path.read_bytes(),
            ContentType="application/json",
        )
    except Exception:
        pass
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
