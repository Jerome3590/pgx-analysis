"""Read row counts from parquet metadata only (no full file download)."""
import boto3, io, json
import pyarrow.parquet as pq

s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

cohorts = {
    "opioid_ed":     ["13-24","25-44","45-54","55-64"],
    "non_opioid_ed": ["65-74","75-84","85-114"],
}
years = ["2016","2017","2018"]

def parquet_row_count(key):
    """Return (cases, controls) by reading only parquet footer metadata."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    pf  = pq.ParquetFile(buf)
    meta = pf.metadata
    cases = 0; ctrl = 0
    for rg in range(meta.num_row_groups):
        # read only 'target' column from each row group
        table = pf.read_row_group(rg, columns=["target"])
        col = table["target"].to_pylist()
        cases += sum(1 for v in col if v == 1)
        ctrl  += sum(1 for v in col if v == 0)
    return cases, ctrl

totals = {}
for c, bands in cohorts.items():
    totals[c] = {}
    for b in bands:
        tc, tr = 0, 0
        for y in years:
            key = f"gold/cohorts/cohort_name={c}/event_year={y}/age_band={b}/cohort.parquet"
            try:
                cases, ctrl = parquet_row_count(key)
                tc += cases; tr += ctrl
            except Exception as e:
                print(f"  SKIP {c}/{y}/{b}: {e}")
        totals[c][b] = {"cases": tc, "controls": tr}
        print(f"{c:20s} | {b:8s} | cases={tc:8,} | controls={tr:8,}")

print("\n=== GRAND TOTALS ===")
for c, bands in totals.items():
    gc = sum(v["cases"]    for v in bands.values())
    gr = sum(v["controls"] for v in bands.values())
    print(f"{c}: cases={gc:,}  controls={gr:,}")

with open("data/cohort_counts.json","w") as f:
    json.dump(totals, f, indent=2)
print("Saved cohort_counts.json")
