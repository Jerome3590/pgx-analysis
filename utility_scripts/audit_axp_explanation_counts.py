#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def audit_axp_counts(project_root: Path) -> pd.DataFrame:
    import duckdb

    ffa_root = project_root / "8_ffa_analysis" / "outputs"
    rows: list[dict] = []
    con = duckdb.connect()
    try:
        for path in sorted(ffa_root.rglob("bin_models/*/xgboost/axp_explanations.parquet")):
            rel = path.relative_to(ffa_root).parts
            cohort, age, bin_name = rel[0], rel[1], rel[3]
            cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{str(path)}')").df()["column_name"].tolist()
            row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{str(path)}')").fetchone()[0]
            axp_col = next((c for c in ("axp", "explanation", "conditions", "features") if c in cols), None)
            nonempty = None
            if axp_col:
                nonempty = con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{str(path)}') "
                    f"WHERE {axp_col} IS NOT NULL AND CAST({axp_col} AS VARCHAR) NOT IN ('[]', '')"
                ).fetchone()[0]
            rows.append(
                {
                    "cohort": cohort,
                    "age": age,
                    "bin": bin_name,
                    "rows": row_count,
                    "axp_col": axp_col,
                    "nonempty": nonempty,
                    "cols": ",".join(cols[:8]),
                }
            )
    finally:
        con.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit existing per-bin AXP explanation row counts and non-empty explanations.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("audit_axp_explanations_existing_counts.csv"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    df = audit_axp_counts(project_root)
    output = args.output if args.output.is_absolute() else project_root / args.output
    df.to_csv(output, index=False)
    print(f"Saved {output}")
    if not df.empty:
        print(f"rows_total={int(df['rows'].sum())}")
        print(f"nonempty_total={int(df['nonempty'].fillna(0).sum())}")


if __name__ == "__main__":
    main()
