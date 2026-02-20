#!/usr/bin/env python3
"""Check a sample parquet for HCG columns and hcg_setting vs hcg_line.

O11_P is the canonical name for the polypharmacy ED target (first_o11_p_date) and includes all qualifying ED codes (P51b, O11, P33). See 2_create_cohort README § HCG-Based ED Visit Targets.
"""
import sys
from pathlib import Path

import duckdb

# Default: use downloaded non_opioid_ed 55-64 model_events; or pass path as first arg
PROJECT_ROOT = Path(__file__).resolve().parents[1]
default_path = PROJECT_ROOT / "4_model_data" / "cohort_name=non_opioid_ed" / "age_band=55-64" / "model_events.parquet"
parquet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
if not parquet_path.exists():
    print(f"Not found: {parquet_path}")
    sys.exit(1)

path_str = str(parquet_path.resolve()).replace("\\", "/")
con = duckdb.connect()

print("=== Schema ===")
schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_str}')").fetchall()
cols = [r[0] for r in schema]
print("Columns:", ", ".join(cols))
hcg_cols = [c for c in cols if "hcg" in c.lower()]
print("HCG-related:", hcg_cols or "(none)")

if "hcg_line" not in cols:
    print("\nNo hcg_line in this parquet (e.g. model_events may not carry HCG). Try a cohort or gold medical parquet.")
    con.close()
    sys.exit(0)

print("\n=== Distinct hcg_setting, hcg_line (where hcg_line not null) ===")
q = f"""
SELECT hcg_setting, hcg_line, COUNT(*)::BIGINT as cnt
FROM read_parquet('{path_str}')
WHERE hcg_line IS NOT NULL AND hcg_line <> ''
GROUP BY 1, 2 ORDER BY 3 DESC
LIMIT 30
"""
r = con.execute(q).fetchdf()
print(r.to_string())

if "hcg_detail" in cols:
    print("\n=== Sample: hcg_setting | hcg_line | hcg_detail (ED-related) ===")
    q2 = f"""
    SELECT DISTINCT hcg_setting, hcg_line, hcg_detail
    FROM read_parquet('{path_str}')
    WHERE hcg_line IS NOT NULL
      AND (hcg_line LIKE '%O11%' OR hcg_line LIKE '%P51%' OR hcg_line LIKE '%P33%'
           OR hcg_line LIKE '%Emergency%' OR hcg_line LIKE '%ER%')
    LIMIT 20
    """
    r2 = con.execute(q2).fetchdf()
    print(r2.to_string())

con.close()
print("\nDone.")
