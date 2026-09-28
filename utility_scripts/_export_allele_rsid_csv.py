"""One-off: export official CPIC allele-rsid parquet to CSV for Lambda (no pyarrow)."""
import tempfile
from pathlib import Path

import boto3

BUCKET = "pgxdatalake"
SRC = "gold/reference/cpic/cpic_allele_rsid.parquet"
DESTS = (
    "gold/reference/cpic/cpic_allele_rsid.csv",
    "gold/dashboard/data/cpic/cpic_allele_rsid.csv",
)


def main() -> None:
    s3 = boto3.client("s3")
    tmp = Path(tempfile.gettempdir()) / "cpic_allele_rsid.parquet"
    csv_path = Path(tempfile.gettempdir()) / "cpic_allele_rsid.csv"
    s3.download_file(BUCKET, SRC, str(tmp))
    try:
        import duckdb

        duckdb.execute(
            f"COPY (SELECT * FROM read_parquet('{tmp.as_posix()}')) "
            f"TO '{csv_path.as_posix()}' (HEADER, DELIMITER ',')"
        )
        print(f"duckdb wrote {csv_path} ({csv_path.stat().st_size} bytes)")
    except Exception as exc:
        print(f"duckdb failed ({exc}); trying pandas")
        import pandas as pd

        df = pd.read_parquet(tmp)
        df.to_csv(csv_path, index=False)
        print(f"pandas wrote {csv_path} rows={len(df)}")
    header = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    print("header:", header)
    for dest in DESTS:
        s3.upload_file(str(csv_path), BUCKET, dest)
        print(f"uploaded s3://{BUCKET}/{dest}")


if __name__ == "__main__":
    main()
