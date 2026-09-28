"""One-off: write official diplotype→phenotype CSV next to the parquet (no invented rows)."""
import os
from pathlib import Path

src = Path(os.environ.get("TEMP", ".")) / "cpic_diplotype_phenotype.parquet"
dst = Path(os.environ.get("TEMP", ".")) / "cpic_diplotype_phenotype.csv"
import pandas as pd

df = pd.read_parquet(src)
print("rows", len(df), "cols", list(df.columns))
df.to_csv(dst, index=False)
print("wrote", dst, "bytes", dst.stat().st_size)
