import csv, shutil
from pathlib import Path
from datetime import datetime

src  = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review\rep-24-0183.pdf")
dest = Path("data/scholar_pdfs/HSH2d4fb8ea.pdf")
LOG  = Path("scripts/vcu_download_log.csv")

shutil.copy2(src, dest)

with open(LOG, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([
        "HSH2d4fb8ea",
        "IMPACT OF REAL-LIFE ENVIRONMENTAL EXPOSURES ON REPRODUCTION",
        "10.1530/rep-24-0183",
        "manual",
        "manual",
        dest.stat().st_size,
        datetime.utcnow().isoformat(),
    ])

on_disk = len(list(Path("data/scholar_pdfs").glob("*.pdf")))
print(f"Imported: {dest.name}  ({dest.stat().st_size // 1024} KB)")
print(f"On disk : {on_disk}/117")
