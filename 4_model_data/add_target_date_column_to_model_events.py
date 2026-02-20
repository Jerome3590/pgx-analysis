#!/usr/bin/env python3
"""
One-off: Add first_ed_non_opioid_date to existing non_opioid_ed model_events.parquet
when the column is missing. Populates with a date per target=1 patient (max(event_date)+1)
so BupaR pre-HCG logic sees events; target=0 stays NULL. Use for local testing only.
"""
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_ROOT = PROJECT_ROOT / "4_model_data"
COHORT = "non_opioid_ed"
AGE_BAND = "85-114"
TARGET_DATE_COL = "first_ed_non_opioid_date"


def main() -> int:
    path = MODEL_DATA_ROOT / f"cohort_name={COHORT}" / f"age_band={AGE_BAND}" / "model_events.parquet"
    if not path.exists():
        print(f"[ERROR] Not found: {path}")
        return 1

    path_str = str(path).replace("'", "''")
    con = duckdb.connect()
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_str}')").fetchall()
    col_names = [row[0] for row in schema]
    if TARGET_DATE_COL in col_names:
        print(f"[INFO] Column '{TARGET_DATE_COL}' already present; nothing to do.")
        con.close()
        return 0

    # Build SELECT list: all existing columns (double-quote for DuckDB identifiers)
    existing = ", ".join(f'"{c}"' for c in col_names)
    out_tmp = path.parent / "model_events.parquet.tmp"
    out_str = str(out_tmp).replace("'", "''")
    # Add column: for target=1 set first_ed_non_opioid_date = max(event_date)+1 day per patient
    con.execute(f"""
        COPY (
            WITH with_max_date AS (
                SELECT
                    *,
                    MAX(CAST(event_date AS DATE)) OVER (PARTITION BY mi_person_key, target) AS max_event_date
                FROM read_parquet('{path_str}')
            )
            SELECT
                {existing},
                CASE
                    WHEN target = 1 AND max_event_date IS NOT NULL
                    THEN CAST(DATE_ADD(CAST(max_event_date AS TIMESTAMP), INTERVAL 1 DAY) AS VARCHAR)
                    ELSE NULL
                END AS {TARGET_DATE_COL}
            FROM with_max_date
        ) TO '{out_str}'
        (FORMAT PARQUET)
    """)
    con.close()
    path.unlink()
    out_tmp.replace(path)
    print(f"[INFO] Added '{TARGET_DATE_COL}' to {path} (target=1: max(event_date)+1 day; target=0: NULL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
