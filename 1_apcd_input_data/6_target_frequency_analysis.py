#!/usr/bin/env python3
"""
Target Frequency Analysis (ICD Diagnostic Codes and CPT Codes)

Modeled after 4_drug_frequency_analysis.py, but extended to multiple columns.

Outputs CSVs with frequency counts per year, both per-column and aggregated
across like columns (all ICD positions combined; both CPT mod fields combined).
"""

import duckdb
import pandas as pd
from datetime import datetime
import warnings
import os
import sys
import re
import json
import argparse
from typing import Optional, List, Tuple
# Visualization helpers (plots saved to S3)
from helpers_1997_13.visualization_utils import (
    plot_stacked_by_year,
    plot_top_bars,
    plot_heatmap_from_pairs,
    save_and_display_chart,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
warnings.filterwarnings('ignore')


# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


GOLD_MEDICAL_PATH = 's3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet'


    


def create_duckdb_conn(threads: int = 1):
    """Create a DuckDB connection with amortized extension load and sensible defaults."""
    conn = duckdb.connect(database=':memory:')
    conn.sql("LOAD httpfs; LOAD aws;")
    conn.sql("CALL load_aws_credentials();")
    conn.sql("SET s3_region='us-east-1'; SET s3_url_style='path';")
    try:
        t = max(1, int(threads))
        conn.sql(f"PRAGMA threads={t}")
        s3_conn_env = os.getenv('PGX_S3_MAX_CONNECTIONS')
        if s3_conn_env and s3_conn_env.isdigit() and int(s3_conn_env) > 0:
            conn.sql(f"SET s3_max_connections={int(s3_conn_env)}")
    except Exception:
        pass
    return conn


def _log_cpu_context(prefix: str, threads: int):
    try:
        import psutil  # type: ignore
        p = psutil.Process()
        try:
            affinity = p.cpu_affinity()
        except Exception:
            affinity = None
        print(f"{prefix} CPU context: pid={p.pid}, logical_cpus={os.cpu_count()}, affinity={affinity}, duckdb_threads={threads}")
    except Exception:
        try:
            cpus = None
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(list(os.sched_getaffinity(0)))  # type: ignore
            print(f"{prefix} CPU context: pid={os.getpid()}, logical_cpus={os.cpu_count()}, affinity={cpus}, duckdb_threads={threads}")
        except Exception:
            print(f"{prefix} CPU context: pid={os.getpid()}, logical_cpus={os.cpu_count()}, duckdb_threads={threads}")


def _list_medical_partitions(glob_path: str) -> pd.DataFrame:
    conn = create_duckdb_conn(threads=1)
    try:
        return conn.sql(
            """
            WITH f AS (
              SELECT file AS filename FROM glob(?)
            ), p AS (
              SELECT
                regexp_extract(filename, 'age_band=([^/]+)', 1) AS age_band,
                CAST(regexp_extract(filename, 'event_year=([0-9]{4})', 1) AS INTEGER) AS event_year,
                filename
              FROM f
            )
            SELECT * FROM p
            """,
            params=[glob_path],
        ).df()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _analyze_partition(filename: str, age_band: str, event_year: int, log_cpu: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Threads per worker from env (default 1)
    t_env = os.getenv('PGX_THREADS_PER_WORKER')
    threads = int(t_env) if t_env and t_env.isdigit() and int(t_env) > 0 else 1
    conn = create_duckdb_conn(threads=threads)
    try:
        if log_cpu:
            _log_cpu_context(prefix="[tfa-worker]", threads=threads)
        # ICD by position
        icd_by_position = conn.sql(
            f"""
            WITH icd AS (
              SELECT '{event_year}'::INT AS event_year,
                     CASE ord WHEN 1 THEN 'primary' WHEN 2 THEN 'second' WHEN 3 THEN 'third'
                              WHEN 4 THEN 'fourth' WHEN 5 THEN 'fifth' WHEN 6 THEN 'sixth'
                              WHEN 7 THEN 'seventh' WHEN 8 THEN 'eighth' WHEN 9 THEN 'ninth'
                              WHEN 10 THEN 'tenth' END AS icd_position,
                     code AS target_code
                            FROM read_parquet('{filename}')
                            CROSS JOIN UNNEST(LIST_VALUE(
                                CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                CAST(ten_icd_diagnosis_code AS VARCHAR)
                            )) WITH ORDINALITY AS t(code, ord)
              WHERE code IS NOT NULL AND code <> ''
            )
            SELECT event_year, icd_position, target_code, COUNT(*) AS frequency
            FROM icd
            GROUP BY ALL
            """
        ).df()

        # ICD aggregated
        icd_agg = conn.sql(
            f"""
            WITH icd AS (
              SELECT '{event_year}'::INT AS event_year,
                     code AS target_code
                            FROM read_parquet('{filename}')
                            CROSS JOIN UNNEST(LIST_VALUE(
                                CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                CAST(ten_icd_diagnosis_code AS VARCHAR)
                            )) AS t(code)
              WHERE code IS NOT NULL AND code <> ''
            )
            SELECT event_year, target_code, COUNT(*) AS frequency
            FROM icd GROUP BY ALL
            """
        ).df()

        # ICD by age band (use path-derived age_band)
        icd_by_age = conn.sql(
            f"""
            WITH icd AS (
              SELECT '{event_year}'::INT AS event_year,
                     '{age_band}' AS age_band,
                     code AS target_code
                            FROM read_parquet('{filename}')
                            CROSS JOIN UNNEST(LIST_VALUE(
                                CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                CAST(ten_icd_diagnosis_code AS VARCHAR)
                            )) AS t(code)
              WHERE code IS NOT NULL AND code <> ''
            )
            SELECT event_year, target_code, age_band, COUNT(*) AS frequency
            FROM icd GROUP BY ALL
            """
        ).df()

        # CPT by field
        cpt_by_field = conn.sql(
            f"""
            WITH cpt AS (
              SELECT '{event_year}'::INT AS event_year,
                     u.unnest.field AS cpt_field,
                     u.unnest.code AS target_code
                            FROM read_parquet('{filename}')
                            CROSS JOIN UNNEST([
                                STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                                STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
                            ]) AS u(unnest)
              WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
            )
            SELECT event_year, cpt_field, target_code, COUNT(*) AS frequency
            FROM cpt GROUP BY ALL
            """
        ).df()

        # CPT aggregated
        cpt_agg = conn.sql(
            f"""
            WITH cpt AS (
              SELECT '{event_year}'::INT AS event_year,
                     u.unnest.code AS target_code
                            FROM read_parquet('{filename}')
                            CROSS JOIN UNNEST([
                                STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                                STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
                            ]) AS u(unnest)
              WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
            )
            SELECT event_year, target_code, COUNT(*) AS frequency
            FROM cpt GROUP BY ALL
            """
        ).df()

        # CPT by age band (use path-derived age_band)
        cpt_by_age = conn.sql(
            f"""
            WITH cpt AS (
              SELECT '{event_year}'::INT AS event_year,
                     '{age_band}' AS age_band,
                     u.unnest.code AS target_code
                            FROM read_parquet('{filename}')
                            CROSS JOIN UNNEST([
                                STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                                STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
                            ]) AS u(unnest)
              WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
            )
            SELECT event_year, target_code, age_band, COUNT(*) AS frequency
            FROM cpt GROUP BY ALL
            """
        ).df()

        return icd_by_position, icd_agg, icd_by_age, cpt_by_field, cpt_agg, cpt_by_age
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_analysis(years=(2016, 2020), out_dir="/home/pgx3874/pgx-analysis/1_apcd_input_data"):
    conn = create_duckdb_conn()
    y0, y1 = years
    path = GOLD_MEDICAL_PATH

    # Build base/icd/cpt views once (Hive partition awareness)
    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW base AS
        SELECT CAST(event_year AS INT) AS event_year,
               CAST(claim_id AS VARCHAR) AS claim_id,
               member_age_band_dos AS age_band,
               primary_icd_diagnosis_code, two_icd_diagnosis_code, three_icd_diagnosis_code,
               four_icd_diagnosis_code, five_icd_diagnosis_code, six_icd_diagnosis_code,
               seven_icd_diagnosis_code, eight_icd_diagnosis_code, nine_icd_diagnosis_code,
               ten_icd_diagnosis_code,
               cpt_mod_1_code, cpt_mod_2_code
        FROM read_parquet(?, HIVE_PARTITIONING=1)
        WHERE event_year BETWEEN ? AND ?;
        """,
        [path, y0, y1],
    )

    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW icd AS
        SELECT event_year,
               claim_id,
               CASE ord WHEN 1 THEN 'primary' WHEN 2 THEN 'second' WHEN 3 THEN 'third'
                        WHEN 4 THEN 'fourth' WHEN 5 THEN 'fifth' WHEN 6 THEN 'sixth'
                        WHEN 7 THEN 'seventh' WHEN 8 THEN 'eighth' WHEN 9 THEN 'ninth'
                        WHEN 10 THEN 'tenth' END AS icd_position,
               code AS target_code, age_band
        FROM base
                CROSS JOIN UNNEST(LIST_VALUE(
                    CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                    CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                    CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                    CAST(ten_icd_diagnosis_code AS VARCHAR)
                )) WITH ORDINALITY AS t(code, ord)
        WHERE code IS NOT NULL AND code <> '';
        """
    )

    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW cpt AS
        SELECT event_year, claim_id, u.unnest.field AS cpt_field, u.unnest.code AS target_code, age_band
        FROM base
                CROSS JOIN UNNEST([
                    STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                    STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
                ]) AS u(unnest)
        WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> '';
        """
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Example COPY outputs (fast, no pandas involved)
    conn.execute(
        f"""
        COPY (
          SELECT event_year, icd_position, target_code, COUNT(*) AS frequency
          FROM icd GROUP BY ALL ORDER BY event_year, icd_position, frequency DESC
        ) TO '{out_dir}/icd_frequency_by_position_{ts}.csv' (HEADER, DELIMITER ',');
        """
    )

    conn.execute(
        f"""
        COPY (
          SELECT event_year, target_code, COUNT(*) AS frequency
          FROM cpt GROUP BY ALL
        ) TO 's3://pgxdatalake/gold/target_code/cpt_frequency_aggregated_{ts}.parquet' (FORMAT PARQUET);
        """
    )

    # Minimal pandas frames (for downstream unified latest + mapping suggestions)
    icd_agg_df = conn.sql(
        """
        SELECT event_year, target_code, COUNT(*) AS frequency
        FROM icd GROUP BY ALL
        """
    ).df()

    cpt_agg_df = conn.sql(
        """
        SELECT event_year, target_code, COUNT(*) AS frequency
        FROM cpt GROUP BY ALL
        """
    ).df()

    # Fast co-occurrence using unnested icd/cpt views; prefer claim_id join if present
    top_n_icd, top_n_cpt = 25, 25
    try:
        pairs_df = conn.sql(
            f"""
            WITH icd_top AS (
              SELECT target_code AS target_icd
              FROM icd GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {top_n_icd}
            ),
            cpt_top AS (
              SELECT target_code AS target_cpt
              FROM cpt GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {top_n_cpt}
            )
            SELECT i.event_year,
                   i.target_code AS target_icd,
                   c.target_code AS target_cpt,
                   COUNT(*) AS frequency
            FROM icd i
            JOIN cpt c USING (claim_id, event_year)
            WHERE i.target_code IN (SELECT target_icd FROM icd_top)
              AND c.target_code IN (SELECT target_cpt FROM cpt_top)
            GROUP BY ALL
            """
        ).df()
    except Exception:
        # Fallback without claim_id (may overcount across events)
        pairs_df = conn.sql(
            f"""
            WITH icd_top AS (
              SELECT target_code AS target_icd
              FROM icd GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {top_n_icd}
            ),
            cpt_top AS (
              SELECT target_code AS target_cpt
              FROM cpt GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT {top_n_cpt}
            )
            SELECT i.event_year,
                   i.target_code AS target_icd,
                   c.target_code AS target_cpt,
                   COUNT(*) AS frequency
            FROM icd i
            JOIN cpt c USING (event_year)
            WHERE i.target_code IN (SELECT target_icd FROM icd_top)
              AND c.target_code IN (SELECT target_cpt FROM cpt_top)
            GROUP BY ALL
            """
        ).df()

    # Build unified all_targets DataFrame defensively so this function
    # works when only one system (ICD or CPT) is present or when one
    # of the frames is empty.
    parts = []
    if icd_agg_df is not None and not icd_agg_df.empty:
        parts.append(icd_agg_df.assign(target_system='icd'))
    if cpt_agg_df is not None and not cpt_agg_df.empty:
        parts.append(cpt_agg_df.assign(target_system='cpt'))

    if parts:
        all_targets_df = pd.concat(parts, ignore_index=True)
    else:
        # Provide an empty DataFrame with the expected schema so callers
        # can safely operate on `all_targets` regardless of input size.
        all_targets_df = pd.DataFrame(columns=['event_year', 'target_code', 'frequency', 'target_system'])

    # Save unified latest to S3
    try:
        from helpers_1997_13.visualization_utils import write_target_code_latest
        write_target_code_latest(all_targets_df)
        print("📤 Unified latest written to S3 (target_code)")
    except Exception as e:
        print(f"⚠️ Failed writing unified latest: {e}")

    return {
        'icd_aggregated': icd_agg_df,
        'cpt_aggregated': cpt_agg_df,
        'all_targets': all_targets_df,
        'icd_cpt_pairs': pairs_df,
    }


# Note: This script does not normalize or apply mappings; it only computes frequencies


def get_icd_frequency_data(conn, path, y0, y1):
    """Compute ICD diagnostic code frequencies by year.

    Produces two DataFrames:
    - icd_by_position_df: frequency by (event_year, icd_position, icd_code)
    - icd_agg_df: frequency by (event_year, icd_code) across all positions
    """
    print("📊 Querying ICD diagnostic code frequencies by year...")

    # One read using Hive partition awareness; create a temp base table
    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE base_once AS
        SELECT CAST(event_year AS INT) AS event_year,
               member_age_band_dos AS age_band,
               primary_icd_diagnosis_code, two_icd_diagnosis_code, three_icd_diagnosis_code,
               four_icd_diagnosis_code, five_icd_diagnosis_code, six_icd_diagnosis_code,
               seven_icd_diagnosis_code, eight_icd_diagnosis_code, nine_icd_diagnosis_code,
               ten_icd_diagnosis_code
        FROM read_parquet(?, HIVE_PARTITIONING=1)
        WHERE event_year BETWEEN ? AND ?
        """,
        [path, y0, y1],
    )

    icd_by_position_df = conn.sql(
        """
        WITH icd AS (
          SELECT event_year,
                 CASE ord
                   WHEN 1 THEN 'primary'
                   WHEN 2 THEN 'second'
                   WHEN 3 THEN 'third'
                   WHEN 4 THEN 'fourth'
                   WHEN 5 THEN 'fifth'
                   WHEN 6 THEN 'sixth'
                   WHEN 7 THEN 'seventh'
                   WHEN 8 THEN 'eighth'
                   WHEN 9 THEN 'ninth'
                   WHEN 10 THEN 'tenth'
                 END AS icd_position,
                 code AS target_code,
                 age_band
          FROM base_once
          CROSS JOIN UNNEST(LIST_VALUE(
                                    CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                    CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                    CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                    CAST(ten_icd_diagnosis_code AS VARCHAR)
          )) WITH ORDINALITY AS t(code, ord)
          WHERE code IS NOT NULL AND code <> ''
        )
        SELECT event_year, icd_position, target_code, COUNT(*) AS frequency
        FROM icd
        GROUP BY ALL
        ORDER BY event_year, icd_position, frequency DESC
        """
    ).df()

    icd_agg_df = conn.sql(
        """
        WITH icd AS (
          SELECT event_year,
                 code AS target_code
          FROM base_once
                    CROSS JOIN UNNEST(LIST_VALUE(
                        CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                        CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                        CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                        CAST(ten_icd_diagnosis_code AS VARCHAR)
                    )) AS t(code)
          WHERE code IS NOT NULL AND code <> ''
        )
        SELECT event_year, target_code, COUNT(*) AS frequency
        FROM icd
        GROUP BY ALL
        ORDER BY event_year, frequency DESC
        """
    ).df()
    icd_agg_raw = icd_agg_df.copy()

    # No normalization: use raw codes as-is

    # By-age aggregation for ICD (no extra join)
    icd_age_df = conn.sql(
        """
        WITH icd AS (
          SELECT event_year,
                 code AS target_code,
                 age_band
          FROM base_once
                    CROSS JOIN UNNEST(LIST_VALUE(
                        CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                        CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                        CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                        CAST(ten_icd_diagnosis_code AS VARCHAR)
                    )) AS t(code)
          WHERE code IS NOT NULL AND code <> '' AND age_band IS NOT NULL AND age_band <> ''
        )
        SELECT event_year, target_code, age_band, COUNT(*) AS frequency
        FROM icd
        GROUP BY ALL
        """
    ).df()
    # No normalization on age-band output

    print(f"✅ Retrieved {len(icd_agg_df):,} ICD-year combinations (aggregated across positions)")
    return icd_by_position_df, icd_agg_df, icd_age_df, icd_agg_raw


def get_cpt_frequency_data(conn, path, y0, y1):
    """Compute CPT code frequencies by year from medical gold data.

    We use both cpt_mod_1_code and cpt_mod_2_code, and also aggregate across them.
    Returns two DataFrames similar to ICD.
    """
    print("📊 Querying CPT code frequencies by year...")

    # One read using Hive partition awareness; create a temp base table
    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE base_once AS
        SELECT CAST(event_year AS INT) AS event_year,
               member_age_band_dos AS age_band,
               cpt_mod_1_code, cpt_mod_2_code
        FROM read_parquet(?, HIVE_PARTITIONING=1)
        WHERE event_year BETWEEN ? AND ?
        """,
        [path, y0, y1],
    )

    cpt_by_field_df = conn.sql(
        """
        WITH cpt AS (
          SELECT event_year,
                 CASE ord WHEN 1 THEN 'cpt_mod_1_code' WHEN 2 THEN 'cpt_mod_2_code' END AS cpt_field,
                 code AS target_code,
                 age_band
          FROM base_once
                    CROSS JOIN UNNEST(LIST_VALUE(CAST(cpt_mod_1_code AS VARCHAR), CAST(cpt_mod_2_code AS VARCHAR))) WITH ORDINALITY AS t(code, ord)
          WHERE code IS NOT NULL AND code <> ''
        )
        SELECT event_year, cpt_field, target_code, COUNT(*) AS frequency
        FROM cpt
        GROUP BY ALL
        ORDER BY event_year, cpt_field, frequency DESC
        """
    ).df()

    cpt_agg_df = conn.sql(
        """
        WITH cpt AS (
          SELECT event_year,
                 code AS target_code
          FROM base_once
                    CROSS JOIN UNNEST(LIST_VALUE(CAST(cpt_mod_1_code AS VARCHAR), CAST(cpt_mod_2_code AS VARCHAR))) AS t(code)
          WHERE code IS NOT NULL AND code <> ''
        )
        SELECT event_year, target_code, COUNT(*) AS frequency
        FROM cpt
        GROUP BY ALL
        ORDER BY event_year, frequency DESC
        """
    ).df()
    cpt_agg_raw = cpt_agg_df.copy()

    print(f"✅ Retrieved {len(cpt_agg_df):,} CPT-year combinations (aggregated across fields)")
    # By-age aggregation for CPT (no extra join)
    cpt_age_df = conn.sql(
        """
        WITH cpt AS (
          SELECT event_year,
                 code AS target_code,
                 age_band
          FROM base_once
                    CROSS JOIN UNNEST(LIST_VALUE(CAST(cpt_mod_1_code AS VARCHAR), CAST(cpt_mod_2_code AS VARCHAR))) AS t(code)
          WHERE code IS NOT NULL AND code <> '' AND age_band IS NOT NULL AND age_band <> ''
        )
        SELECT event_year, target_code, age_band, COUNT(*) AS frequency
        FROM cpt
        GROUP BY ALL
        """
    ).df()

    return cpt_by_field_df, cpt_agg_df, cpt_age_df, cpt_agg_raw


# ===== Suggested mapping creation from aggregated counts =====
def _suggest_mapping_from_agg(agg_df: pd.DataFrame, system: str, only_groups=None) -> dict:
    if agg_df is None or agg_df.empty:
        return {}
    df = agg_df.copy()
    df['variant'] = df['target_code'].astype(str)
    # Total frequency per variant
    totals = df.groupby('variant', as_index=False)['frequency'].sum()
    # Build grouping key
    if system == 'icd':
        def group_key(x: str) -> str:
            s = str(x).upper().replace(' ', '')
            m = re.search(r'([A-Z]\d{2}(?:\.?[A-Z0-9]{1,4})?)', s)
            token = m.group(1) if m else s
            return token.replace('.', '')
        totals['g'] = totals['variant'].map(group_key)
        if only_groups:
            norm = set(re.sub(r'[^A-Z0-9]', '', str(g).upper()) for g in only_groups)
            totals = totals[totals['g'].isin(norm)]
        # choose canonical: undotted ICD token (e.g., F1120)
        canon = {}
        for g, sub in totals.groupby('g'):
            c = g  # undotted canonical code
            for v in sub['variant']:
                if v != c:
                    canon[v] = c
        return canon
    else:  # cpt (modifiers or codes)
        def group_key(x: str) -> str:
            return str(x).upper().replace(' ', '').replace('.', '')
        totals['g'] = totals['variant'].map(group_key)
        if only_groups:
            norm = set(re.sub(r'[^A-Z0-9]', '', str(g).upper()) for g in only_groups)
            totals = totals[totals['g'].isin(norm)]
        canon = {}
        for g, sub in totals.groupby('g'):
            sub = sub.sort_values('frequency', ascending=False)
            candidates = list(sub['variant'])
            # prefer 5-digit numeric if present
            preferred = [v for v in candidates if re.fullmatch(r'\d{5}', str(v))]
            c = preferred[0] if preferred else sub.iloc[0]['variant']
            for v in candidates:
                if v != c:
                    canon[v] = c
        return canon


def get_icd_cpt_cooccurrence(top_n_icd: int = 25, top_n_cpt: int = 25, years=(2016, 2020)):
    """Compute co-occurrence counts between ICD diagnostic codes and CPT codes per event.

    Returns:
      pairs_df: DataFrame[event_year, target_icd, target_cpt, frequency]
      heatmap_pivot: pivoted DataFrame[target_icd x target_cpt] of frequencies (overall)
    """
    print("📊 Computing ICD x CPT co-occurrence heatmap data...")
    conn = create_duckdb_conn()
    y0, y1 = years
    query = f"""
    WITH base AS (
        SELECT claim_id, event_year,
               primary_icd_diagnosis_code, two_icd_diagnosis_code, three_icd_diagnosis_code,
               four_icd_diagnosis_code, five_icd_diagnosis_code, six_icd_diagnosis_code,
               seven_icd_diagnosis_code, eight_icd_diagnosis_code, nine_icd_diagnosis_code,
               ten_icd_diagnosis_code,
               cpt_mod_1_code, cpt_mod_2_code
        FROM read_parquet('{GOLD_MEDICAL_PATH}', HIVE_PARTITIONING=1)
        WHERE event_year BETWEEN {y0} AND {y1}
    ),
    icd_raw AS (
        SELECT claim_id, event_year, primary_icd_diagnosis_code AS target_icd FROM base WHERE primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, two_icd_diagnosis_code FROM base WHERE two_icd_diagnosis_code IS NOT NULL AND two_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, three_icd_diagnosis_code FROM base WHERE three_icd_diagnosis_code IS NOT NULL AND three_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, four_icd_diagnosis_code FROM base WHERE four_icd_diagnosis_code IS NOT NULL AND four_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, five_icd_diagnosis_code FROM base WHERE five_icd_diagnosis_code IS NOT NULL AND five_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, six_icd_diagnosis_code FROM base WHERE six_icd_diagnosis_code IS NOT NULL AND six_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, seven_icd_diagnosis_code FROM base WHERE seven_icd_diagnosis_code IS NOT NULL AND seven_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, eight_icd_diagnosis_code FROM base WHERE eight_icd_diagnosis_code IS NOT NULL AND eight_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, nine_icd_diagnosis_code FROM base WHERE nine_icd_diagnosis_code IS NOT NULL AND nine_icd_diagnosis_code <> ''
        UNION ALL SELECT claim_id, event_year, ten_icd_diagnosis_code FROM base WHERE ten_icd_diagnosis_code IS NOT NULL AND ten_icd_diagnosis_code <> ''
    ),
    cpt_raw AS (
        SELECT claim_id, event_year, cpt_mod_1_code AS target_cpt FROM base WHERE cpt_mod_1_code IS NOT NULL AND cpt_mod_1_code <> ''
        UNION ALL
        SELECT claim_id, event_year, cpt_mod_2_code AS target_cpt FROM base WHERE cpt_mod_2_code IS NOT NULL AND cpt_mod_2_code <> ''
    ),
    icd_tot AS (
        SELECT target_icd, COUNT(*) AS freq FROM icd_raw GROUP BY target_icd ORDER BY freq DESC LIMIT {top_n_icd}
    ),
    cpt_tot AS (
        SELECT target_cpt, COUNT(*) AS freq FROM cpt_raw GROUP BY target_cpt ORDER BY freq DESC LIMIT {top_n_cpt}
    ),
    pairs AS (
        SELECT i.event_year, i.target_icd, c.target_cpt, COUNT(*) AS frequency
        FROM icd_raw i
        JOIN cpt_raw c
          ON i.claim_id = c.claim_id AND i.event_year = c.event_year
        WHERE i.target_icd IN (SELECT target_icd FROM icd_tot)
          AND c.target_cpt IN (SELECT target_cpt FROM cpt_tot)
        GROUP BY i.event_year, i.target_icd, c.target_cpt
    )
    SELECT * FROM pairs
    """
    pairs_df = conn.sql(query).df()
    conn.close()

    # Overall pivot across all years for a compact heatmap
    heatmap_pivot = pairs_df.groupby(['target_icd', 'target_cpt'])['frequency'].sum().reset_index().pivot(
        index='target_icd', columns='target_cpt', values='frequency'
    ).fillna(0)

    return pairs_df, heatmap_pivot


def _aggregate_and_write_outputs(icd_by_position_df: pd.DataFrame,
                                 icd_agg_df: pd.DataFrame,
                                 icd_age_df: pd.DataFrame,
                                 cpt_by_field_df: pd.DataFrame,
                                 cpt_agg_df: pd.DataFrame,
                                 cpt_age_df: pd.DataFrame,
                                 codes_of_interest: Optional[List[str]] = None,
                                 log_s3: bool = False) -> dict:
    # Write outputs to project-mounted data folder; create an `outputs/` subfolder.
    data_dir = os.path.join(project_root, '1_apcd_input_data')
    outputs_dir = os.path.join(data_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Write CSVs locally
    try:
        if icd_by_position_df is not None and not icd_by_position_df.empty:
            icd_by_position_df.to_csv(os.path.join(outputs_dir, f"icd_frequency_by_position_{ts}.csv"), index=False)
        if cpt_by_field_df is not None and not cpt_by_field_df.empty:
            cpt_by_field_df.to_csv(os.path.join(outputs_dir, f"cpt_frequency_by_field_{ts}.csv"), index=False)
        if icd_agg_df is not None and not icd_agg_df.empty:
            icd_agg_df.to_csv(os.path.join(outputs_dir, f"icd_frequency_aggregated_{ts}.csv"), index=False)
    except Exception as e:
        print(f"⚠️ Local CSV writes failed: {e}")

    # Write Parquet to S3 via DuckDB from DataFrames
    try:
        conn = create_duckdb_conn(threads=1)
        if cpt_agg_df is not None and not cpt_agg_df.empty:
            conn.register('cpt_agg_df', cpt_agg_df)
            cpt_key = f"gold/target_code/cpt_frequency_aggregated_{ts}.parquet"
            t0 = time.time()
            conn.sql(
                f"COPY cpt_agg_df TO 's3://pgxdatalake/{cpt_key}' (FORMAT PARQUET)"
            )
            t1 = time.time()
            if log_s3:
                try:
                    import boto3  # type: ignore
                    s3 = boto3.client('s3')
                    post = s3.head_object(Bucket='pgxdatalake', Key=cpt_key)
                    size_b = post.get('ContentLength') or 0
                    elapsed = max(1e-6, t1 - t0)
                    mbps = (size_b / (1024 * 1024)) / elapsed
                    print(f"[tfa-parent] S3 write: key={cpt_key}, size_bytes={size_b}, elapsed_sec={elapsed:.3f}, approx_MBps={mbps:.2f}, etag={post.get('ETag')}")
                except Exception as _e:
                    print(f"[tfa-parent] S3 throughput logging failed (cpt_agg): {_e}")

        # Unified latest
        all_targets_df = pd.concat([
            icd_agg_df.assign(target_system='icd') if icd_agg_df is not None else pd.DataFrame(columns=['event_year','target_code','frequency','target_system']),
            cpt_agg_df.assign(target_system='cpt') if cpt_agg_df is not None else pd.DataFrame(columns=['event_year','target_code','frequency','target_system'])
        ], ignore_index=True)
        if not all_targets_df.empty:
            conn.register('all_targets_df', all_targets_df)
            key_parquet = 'gold/target_code/target_code_latest.parquet'
            key_csv = 'gold/target_code/target_code_latest.csv'
            t0 = time.time()
            conn.sql(
                f"COPY all_targets_df TO 's3://pgxdatalake/{key_parquet}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)"
            )
            t1 = time.time()
            if log_s3:
                try:
                    import boto3  # type: ignore
                    s3 = boto3.client('s3')
                    post = s3.head_object(Bucket='pgxdatalake', Key=key_parquet)
                    size_b = post.get('ContentLength') or 0
                    elapsed = max(1e-6, t1 - t0)
                    mbps = (size_b / (1024 * 1024)) / elapsed
                    print(f"[tfa-parent] S3 write: key={key_parquet}, size_bytes={size_b}, elapsed_sec={elapsed:.3f}, approx_MBps={mbps:.2f}, etag={post.get('ETag')}")
                except Exception as _e:
                    print(f"[tfa-parent] S3 throughput logging failed (latest parquet): {_e}")
            # CSV
            t2 = time.time()
            conn.sql(
                f"COPY all_targets_df TO 's3://pgxdatalake/{key_csv}' (FORMAT CSV, HEADER TRUE, OVERWRITE_OR_IGNORE)"
            )
            t3 = time.time()
            if log_s3:
                try:
                    import boto3  # type: ignore
                    s3 = boto3.client('s3')
                    post = s3.head_object(Bucket='pgxdatalake', Key=key_csv)
                    size_b = post.get('ContentLength') or 0
                    elapsed = max(1e-6, t3 - t2)
                    mbps = (size_b / (1024 * 1024)) / elapsed
                    print(f"[tfa-parent] S3 write: key={key_csv}, size_bytes={size_b}, elapsed_sec={elapsed:.3f}, approx_MBps={mbps:.2f}, etag={post.get('ETag')}")
                except Exception as _e:
                    print(f"[tfa-parent] S3 throughput logging failed (latest csv): {_e}")
        print("📤 Unified latest written to S3 via COPY")
        # Small visualizations for inspection: top codes and stacked-by-year plots
        try:
            import matplotlib.pyplot as plt

            # ICD visualizations
            if icd_agg_df is not None and not icd_agg_df.empty:
                try:
                    tot_icd = icd_agg_df.groupby('target_code', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)
                    top_n = min(20, len(tot_icd))
                    top_icd = tot_icd.head(top_n)

                    # Top ICD bars
                    plot_top_bars(top_icd, target_col='target_code', value_col='frequency', top_n=top_n, title='Top ICD codes (total frequency)')
                    s3_icd_top = save_and_display_chart(plt.gcf(), 'icd_top_codes', 'target_code', display=False, close_fig=True)
                    print(f"[tfa-parent] ICD top codes chart saved to: {s3_icd_top}")

                    # Stacked by year for top 10 ICDs
                    top10 = list(top_icd.head(10)['target_code'])
                    plot_stacked_by_year(icd_agg_df[icd_agg_df['target_code'].isin(top10)], target_col='target_code', year_col='event_year', freq_col='frequency', ordered_targets=top10, title_suffix='Top 10 ICD codes')
                    s3_icd_stack = save_and_display_chart(plt.gcf(), 'icd_top10_stacked', 'target_code', display=False, close_fig=True)
                    print(f"[tfa-parent] ICD stacked-by-year chart saved to: {s3_icd_stack}")
                except Exception as _e:
                    print(f"[tfa-parent] ICD visualization failed: {_e}")

            # CPT visualizations
            if cpt_agg_df is not None and not cpt_agg_df.empty:
                try:
                    tot_cpt = cpt_agg_df.groupby('target_code', as_index=False)['frequency'].sum().sort_values('frequency', ascending=False)
                    top_n_c = min(20, len(tot_cpt))
                    top_cpt = tot_cpt.head(top_n_c)

                    plot_top_bars(top_cpt, target_col='target_code', value_col='frequency', top_n=top_n_c, title='Top CPT codes (total frequency)')
                    s3_cpt_top = save_and_display_chart(plt.gcf(), 'cpt_top_codes', 'target_code', display=False, close_fig=True)
                    print(f"[tfa-parent] CPT top codes chart saved to: {s3_cpt_top}")

                    top10c = list(top_cpt.head(10)['target_code'])
                    plot_stacked_by_year(cpt_agg_df[cpt_agg_df['target_code'].isin(top10c)], target_col='target_code', year_col='event_year', freq_col='frequency', ordered_targets=top10c, title_suffix='Top 10 CPT codes')
                    s3_cpt_stack = save_and_display_chart(plt.gcf(), 'cpt_top10_stacked', 'target_code', display=False, close_fig=True)
                    print(f"[tfa-parent] CPT stacked-by-year chart saved to: {s3_cpt_stack}")
                except Exception as _e:
                    print(f"[tfa-parent] CPT visualization failed: {_e}")
        except Exception:
            # Visualization is optional; continue if plotting libs or S3 access missing
            pass
    except Exception as e:
        print(f"❌ Failed to write outputs via COPY: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Suggested mappings
    try:
        interest_list = codes_of_interest
        if not interest_list:
            env_interest = os.getenv('PGX_CODES_OF_INTEREST', '')
            if env_interest:
                interest_list = [c.strip() for c in env_interest.split(',') if c.strip()]
        if interest_list:
            icd_map_suggest = _suggest_mapping_from_agg(icd_agg_df, 'icd', only_groups=interest_list)
            cpt_map_suggest = _suggest_mapping_from_agg(cpt_agg_df, 'cpt', only_groups=interest_list)
        else:
            icd_map_suggest = _suggest_mapping_from_agg(icd_agg_df, 'icd')
            cpt_map_suggest = _suggest_mapping_from_agg(cpt_agg_df, 'cpt')
        suggest_dir = os.path.join(project_root, '1_apcd_input_data', 'claim_mappings')
        os.makedirs(suggest_dir, exist_ok=True)
        icd_path = os.path.join(suggest_dir, 'icd_mappings_suggested.json')
        cpt_path = os.path.join(suggest_dir, 'cpt_mappings_suggested.json')
        with open(icd_path, 'w', encoding='utf-8') as f:
            json.dump(icd_map_suggest, f, indent=2, ensure_ascii=False)
        with open(cpt_path, 'w', encoding='utf-8') as f:
            json.dump(cpt_map_suggest, f, indent=2, ensure_ascii=False)
        print("📝 Suggested mapping files written:")
        print(f"  • {icd_path} (ICD variants → canonical)")
        print(f"  • {cpt_path} (CPT variants → canonical)")
    except Exception as e:
        print(f"⚠️ Could not write suggested mapping files: {e}")

    return {
        'icd_by_position': icd_by_position_df,
        'icd_aggregated': icd_agg_df,
        'icd_by_age': icd_age_df,
        'cpt_by_field': cpt_by_field_df,
        'cpt_aggregated': cpt_agg_df,
        'cpt_by_age': cpt_age_df,
    }


def main(codes_of_interest: Optional[List[str]] = None, years: Tuple[int, int] = (2016, 2020), path: Optional[str] = None, workers: int = 1, log_cpu: bool = False, log_s3: bool = False):
    print("🎯 TARGET FREQUENCY ANALYSIS (ICD + CPT)")
    print("="*60)

    y0, y1 = years
    if path is None:
        path = GOLD_MEDICAL_PATH
    # Resolve default workers from env if not provided
    if not workers or workers <= 1:
        w_env = os.getenv('PGX_WORKERS_MEDICAL')
        workers = int(w_env) if w_env and w_env.isdigit() and int(w_env) > 0 else 1

    if workers and workers > 1:
        # Parallel path: enumerate partitions and aggregate
        parts = _list_medical_partitions(GOLD_MEDICAL_PATH)
        if parts is None or parts.empty:
            print("No medical gold partitions found.")
            return {}
        y0, y1 = years
        parts = parts[(parts['event_year'] >= y0) & (parts['event_year'] <= y1)]
        results = []
        print(f"🚀 Parallel target frequency analysis with {workers} workers across {len(parts)} partitions...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_analyze_partition, row['filename'], row['age_band'], int(row['event_year']), log_cpu)
                for _, row in parts.iterrows()
            ]
            for f in as_completed(futures):
                results.append(f.result())

        # Reduce partials
        def _concat_and_sum(items, cols):
            df = pd.concat(items, ignore_index=True) if items else pd.DataFrame(columns=cols)
            if df.empty:
                return df
            group_cols = [c for c in df.columns if c != 'frequency']
            return df.groupby(group_cols, as_index=False)['frequency'].sum()

        icd_by_position_df = _concat_and_sum([r[0] for r in results], ['event_year','icd_position','target_code','frequency'])
        icd_agg_df        = _concat_and_sum([r[1] for r in results], ['event_year','target_code','frequency'])
        icd_age_df        = _concat_and_sum([r[2] for r in results], ['event_year','target_code','age_band','frequency'])
        cpt_by_field_df   = _concat_and_sum([r[3] for r in results], ['event_year','cpt_field','target_code','frequency'])
        cpt_agg_df        = _concat_and_sum([r[4] for r in results], ['event_year','target_code','frequency'])
        cpt_age_df        = _concat_and_sum([r[5] for r in results], ['event_year','target_code','age_band','frequency'])

        return _aggregate_and_write_outputs(icd_by_position_df, icd_agg_df, icd_age_df, cpt_by_field_df, cpt_agg_df, cpt_age_df, codes_of_interest=codes_of_interest, log_s3=log_s3)

    conn = create_duckdb_conn() # Fallback: single-connection path

    icd_by_position_df, icd_agg_df, icd_age_df, icd_agg_raw = get_icd_frequency_data(conn, path, y0, y1)
    cpt_by_field_df, cpt_agg_df, cpt_age_df, cpt_agg_raw = get_cpt_frequency_data(conn, path, y0, y1)

    # COPY-based outputs (avoid large pandas transfers)
    data_dir = os.path.join(project_root, '1_apcd_input_data')
    outputs_dir = os.path.join(data_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # COPY via DuckDB (single-read approach)
    try:
        # Reuse the primary connection and the selected years/path
        conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE base_once AS
            SELECT CAST(event_year AS INT) AS event_year,
                   member_age_band_dos AS age_band,
                   primary_icd_diagnosis_code, two_icd_diagnosis_code, three_icd_diagnosis_code,
                   four_icd_diagnosis_code, five_icd_diagnosis_code, six_icd_diagnosis_code,
                   seven_icd_diagnosis_code, eight_icd_diagnosis_code, nine_icd_diagnosis_code,
                   ten_icd_diagnosis_code,
                   cpt_mod_1_code, cpt_mod_2_code
            FROM read_parquet(?, HIVE_PARTITIONING=1)
            WHERE event_year BETWEEN ? AND ?
            """,
            [path, y0, y1],
        )

        # Local CSV: ICD by position
        conn.execute(
            f"""
            COPY (
              WITH icd AS (
                SELECT event_year,
                       CASE ord
                         WHEN 1 THEN 'primary'
                         WHEN 2 THEN 'second'
                         WHEN 3 THEN 'third'
                         WHEN 4 THEN 'fourth'
                         WHEN 5 THEN 'fifth'
                         WHEN 6 THEN 'sixth'
                         WHEN 7 THEN 'seventh'
                         WHEN 8 THEN 'eighth'
                         WHEN 9 THEN 'ninth'
                         WHEN 10 THEN 'tenth'
                       END AS icd_position,
                       code AS target_code,
                       age_band
                FROM base_once
                CROSS JOIN UNNEST(LIST_VALUE(
                                    CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                    CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                    CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                    CAST(ten_icd_diagnosis_code AS VARCHAR)
                                )) WITH ORDINALITY AS t(code, ord)
                WHERE code IS NOT NULL AND code <> ''
              )
              SELECT event_year, icd_position, target_code, COUNT(*) AS frequency
              FROM icd GROUP BY ALL
            ) TO '{outputs_dir}/icd_frequency_by_position_{ts}.csv' (HEADER, DELIMITER ',')
            """
        )

        # Local CSV: ICD aggregated
        conn.execute(
            f"""
            COPY (
              WITH icd AS (
                SELECT event_year, code AS target_code
                FROM base_once
                CROSS JOIN UNNEST(LIST_VALUE(
                                    CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                    CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                    CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                    CAST(ten_icd_diagnosis_code AS VARCHAR)
                                )) AS t(code)
                WHERE code IS NOT NULL AND code <> ''
              )
              SELECT event_year, target_code, COUNT(*) AS frequency
              FROM icd GROUP BY ALL
            ) TO '{outputs_dir}/icd_frequency_aggregated_{ts}.csv' (HEADER, DELIMITER ',')
            """
        )

    # Local CSV: CPT by field
        conn.execute(
            f"""
            COPY (
          WITH cpt AS (
            SELECT event_year, u.unnest.field AS cpt_field, u.unnest.code AS target_code
            FROM base_once
            CROSS JOIN UNNEST([
                STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
            ]) AS u(unnest)
            WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
          )
          SELECT event_year, cpt_field, target_code, COUNT(*) AS frequency
              FROM cpt GROUP BY ALL ORDER BY event_year, cpt_field, frequency DESC
            ) TO '{outputs_dir}/cpt_frequency_by_field_{ts}.csv' (HEADER, DELIMITER ',')
            """
        )

        # S3 Parquet: CPT aggregated
        conn.execute(
            f"""
            COPY (
          WITH cpt AS (
            SELECT event_year, u.unnest.code AS target_code
            FROM base_once
            CROSS JOIN UNNEST([
                STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
            ]) AS u(unnest)
            WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
          )
              SELECT event_year, target_code, COUNT(*) AS frequency
              FROM cpt GROUP BY ALL
            ) TO 's3://pgxdatalake/gold/target_code/cpt_frequency_aggregated_{ts}.parquet' (FORMAT PARQUET)
            """
        )
        print("📤 COPY outputs written (ICD CSV local, CPT Parquet to S3)")
    except Exception as e:
        print(f"⚠️ COPY outputs failed: {e}")

    # Build unified dataframe of all target codes (ICD + CPT)
    all_targets_df = pd.concat(
        [icd_agg_df.assign(target_system='icd'),
         cpt_agg_df.assign(target_system='cpt')],
        ignore_index=True
    )
    # Unified latest via DuckDB COPY (avoid large pandas to S3)
    try:
        conn.execute(
            """
            COPY (
              WITH icd AS (
                SELECT event_year, REPLACE(code, '.', '') AS target_code
                                FROM base_once
                                CROSS JOIN UNNEST(LIST_VALUE(
                                    CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                    CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                    CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                    CAST(ten_icd_diagnosis_code AS VARCHAR)
                                )) AS t(code)
                WHERE code IS NOT NULL AND code <> ''
              ),
              cpt AS (
                                SELECT event_year, u.unnest.code AS target_code
                                FROM base_once
                                CROSS JOIN UNNEST([
                                    STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                                    STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
                                ]) AS u(unnest)
                WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
              )
              SELECT event_year, target_code, COUNT(*) AS frequency, 'icd' AS target_system
              FROM icd GROUP BY ALL
              UNION ALL
              SELECT event_year, target_code, COUNT(*) AS frequency, 'cpt' AS target_system
              FROM cpt GROUP BY ALL
            ) TO 's3://pgxdatalake/gold/target_code/target_code_latest.parquet' (FORMAT PARQUET, OVERWRITE_OR_IGNORE);
            """
        )
        conn.execute(
            """
            COPY (
              WITH icd AS (
                SELECT event_year, REPLACE(code, '.', '') AS target_code
                FROM base_once
                                CROSS JOIN UNNEST(LIST_VALUE(
                                    CAST(primary_icd_diagnosis_code AS VARCHAR), CAST(two_icd_diagnosis_code AS VARCHAR), CAST(three_icd_diagnosis_code AS VARCHAR),
                                    CAST(four_icd_diagnosis_code AS VARCHAR), CAST(five_icd_diagnosis_code AS VARCHAR), CAST(six_icd_diagnosis_code AS VARCHAR),
                                    CAST(seven_icd_diagnosis_code AS VARCHAR), CAST(eight_icd_diagnosis_code AS VARCHAR), CAST(nine_icd_diagnosis_code AS VARCHAR),
                                    CAST(ten_icd_diagnosis_code AS VARCHAR)
                                )) AS t(code)
                WHERE code IS NOT NULL AND code <> ''
              ),
              cpt AS (
                SELECT event_year, u.unnest.code AS target_code
                FROM base_once
                                CROSS JOIN UNNEST([
                                    STRUCT_PACK(field := 'cpt_mod_1_code', code := CAST(cpt_mod_1_code AS VARCHAR)),
                                    STRUCT_PACK(field := 'cpt_mod_2_code', code := CAST(cpt_mod_2_code AS VARCHAR))
                                ]) AS u(unnest)
                WHERE u.unnest.code IS NOT NULL AND u.unnest.code <> ''
              )
              SELECT event_year, target_code, COUNT(*) AS frequency, 'icd' AS target_system
              FROM icd GROUP BY ALL
              UNION ALL
              SELECT event_year, target_code, COUNT(*) AS frequency, 'cpt' AS target_system
              FROM cpt GROUP BY ALL
            ) TO 's3://pgxdatalake/gold/target_code/target_code_latest.csv' (FORMAT CSV, HEADER TRUE, OVERWRITE_OR_IGNORE);
            """
        )
        print("📤 Unified latest written to S3 via COPY")
    except Exception as e:
        print(f"❌ Failed to write unified latest via COPY: {e}")

    # Suggest JSON mapping files for QA (variants -> canonical)
    try:
        # Determine codes of interest from CLI or env
        interest_list = codes_of_interest
        if not interest_list:
            env_interest = os.getenv('PGX_CODES_OF_INTEREST', '')
            if env_interest:
                interest_list = [c.strip() for c in env_interest.split(',') if c.strip()]

        if interest_list:
            icd_map_suggest = _suggest_mapping_from_agg(icd_agg_raw, 'icd', only_groups=interest_list)
            cpt_map_suggest = _suggest_mapping_from_agg(cpt_agg_raw, 'cpt', only_groups=interest_list)
        else:
            icd_map_suggest = _suggest_mapping_from_agg(icd_agg_raw, 'icd')
            cpt_map_suggest = _suggest_mapping_from_agg(cpt_agg_raw, 'cpt')
        suggest_dir = os.path.join(project_root, '1_apcd_input_data', 'claim_mappings')
        try:
            os.makedirs(suggest_dir, exist_ok=True)
        except Exception:
            pass
        icd_path = os.path.join(suggest_dir, 'icd_mappings_suggested.json')
        cpt_path = os.path.join(suggest_dir, 'cpt_mappings_suggested.json')
        import json as _json
        with open(icd_path, 'w', encoding='utf-8') as f:
            _json.dump(icd_map_suggest, f, indent=2, ensure_ascii=False)
        with open(cpt_path, 'w', encoding='utf-8') as f:
            _json.dump(cpt_map_suggest, f, indent=2, ensure_ascii=False)
        print("📝 Suggested mapping files written:")
        print(f"  • {icd_path} (ICD variants → canonical)")
        print(f"  • {cpt_path} (CPT variants → canonical)")
    except Exception as e:
        print(f"⚠️ Could not write suggested mapping files: {e}")

    # QA check: list any remaining variants that normalize to F1120
    try:
        tmp = all_targets_df.copy()
        tmp['code_flat'] = tmp['target_code'].astype(str).str.upper().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)
        qa = (
            tmp[(tmp['target_system'] == 'icd') & (tmp['code_flat'].str.contains('F1120', na=False))]
            .groupby('target_code', as_index=False)
            .agg(rows=('event_year', 'count'), total_frequency=('frequency', 'sum'))
            .sort_values(['total_frequency', 'rows', 'target_code'], ascending=[False, False, True])
        )
        if not qa.empty:
            print("\n🔎 QA: Variants matching F1120 after normalization:")
            for _, r in qa.iterrows():
                tf = int(r['total_frequency']) if pd.notnull(r['total_frequency']) else 'NULL'
                print(f"  {r['target_code']}\trows={int(r['rows'])}\ttotal_frequency={tf}")
        else:
            print("\n🔎 QA: No variants matching F1120 found after normalization.")
    except Exception as e:
        print(f"⚠️ QA check failed: {e}")

    # Skipping ICD x CPT heatmap for now (too many codes). Keep helper available for future cohort-level filtering.
    pairs_df = None
    heatmap_pivot = None

    # Return dataframes for notebooks/visuals
    result = {
        'icd_by_position': icd_by_position_df,
        'icd_aggregated': icd_agg_df,
        'icd_by_age': icd_age_df,
        'cpt_by_field': cpt_by_field_df,
        'cpt_aggregated': cpt_agg_df,
        'cpt_by_age': cpt_age_df,
        'all_targets': all_targets_df,
        'icd_cpt_pairs': pairs_df if 'pairs_df' in locals() else None,
        'icd_cpt_heatmap': heatmap_pivot if 'heatmap_pivot' in locals() else None,
    }
    try:
        conn.close()
    except Exception:
        pass
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Target code frequency analysis (ICD + CPT)')
    parser.add_argument('--codes-of-interest', help='Comma-separated ICD/CPT codes to focus mapping suggestions on (e.g., F11.20,99213)', default='')
    # Default from env PGX_WORKERS_MEDICAL or 1
    default_workers = int(os.getenv('PGX_WORKERS_MEDICAL')) if os.getenv('PGX_WORKERS_MEDICAL', '').isdigit() else 1
    parser.add_argument('--workers', type=int, default=default_workers, help='Number of parallel workers (processes). Default from PGX_WORKERS_MEDICAL or 1')
    parser.add_argument('--min-year', type=int, default=2016)
    parser.add_argument('--max-year', type=int, default=2020)
    parser.add_argument('--log-cpu', action='store_true', help='Log per-worker CPU affinity/context')
    parser.add_argument('--log-s3', action='store_true', help='Log S3 throughput for final COPY outputs')
    args = parser.parse_args()
    coi = [c.strip() for c in args.codes_of_interest.split(',') if c.strip()] if args.codes_of_interest else None

    data = main(codes_of_interest=coi, years=(args.min_year, args.max_year), workers=args.workers, log_cpu=args.log_cpu, log_s3=args.log_s3)

    # Apply target ICD mappings at source as a best-effort normalization step
    # so saved artifacts contain canonical target codes (e.g., AF1120 -> F1120).
    try:
        map_paths = [
            os.path.join(project_root, '1_apcd_input_data', 'target_mapping', 'target_icd_mapping.json'),
            os.path.join(project_root, '1_apcd_input_data', 'claim_mappings', 'target_icd_mapping.json')
        ]
        map_dict = {}
        for mp in map_paths:
            if os.path.exists(mp):
                try:
                    with open(mp, 'r', encoding='utf-8') as fh:
                        j = json.load(fh)
                    if isinstance(j, dict):
                        map_dict.update(j)
                except Exception as _:
                    print(f"⚠️ Could not load mapping file '{mp}' - skipping")
        if map_dict:
            # Apply replacements to any DataFrame-like entries that contain target_code
            for k, v in list(data.items()):
                try:
                    if hasattr(v, 'columns') and 'target_code' in v.columns:
                        data[k]['target_code'] = data[k]['target_code'].astype(str).replace(map_dict)
                except Exception:
                    # Non-fatal: mapping is best-effort
                    pass
    except Exception:
        pass

    # Save data for notebook use (mirrors 4_drug_frequency_analysis.py)
    # Make this idempotent: if an existing pickle is present, preserve it
    # under a timestamped 'orig' filename, then write the new pickle and
    # also write a timestamped 'updated' copy so callers can diff/compare.
    import pickle
    import shutil
    

    # Persist pickles under the project's 1_apcd_input_data/outputs directory on EC2
    data_dir = os.path.join(project_root, '1_apcd_input_data')
    outputs_dir = os.path.join(data_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    # Canonical, stable filenames (idempotent): overwrite canonical and keep stable updated/ orig copies
    pickle_path = os.path.join(outputs_dir, 'target_analysis_data.pkl')
    # Stable backup names (no timestamps); keep previous run as .orig.pkl and provide an _updated stable copy
    orig_copy = os.path.join(outputs_dir, 'target_analysis_data.orig.pkl')
    updated_copy = os.path.join(outputs_dir, 'target_analysis_data_updated.pkl')

    try:
        # If an existing canonical pickle exists, move/copy it to the stable orig path
        if os.path.exists(pickle_path):
            try:
                shutil.copy2(pickle_path, orig_copy)
                print(f"💾 Existing pickle moved/copied to '{orig_copy}'")
            except Exception as e:
                print(f"⚠️ Could not preserve existing pickle to '{orig_copy}': {e}")

        # Write the current data to the canonical path
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n💾 Data saved to '{pickle_path}' for notebook visualization")

        # Also write/overwrite a stable 'updated' copy (no timestamp)
        try:
            shutil.copy2(pickle_path, updated_copy)
            print(f"💾 Updated copy written to '{updated_copy}'")
        except Exception as e:
            print(f"⚠️ Failed to write updated copy '{updated_copy}': {e}")

        # No legacy back-compat writes: consumers must use files under outputs_dir

    except Exception as e:
        print(f"❌ Failed to save pickle data: {e}")


