import boto3, json
s3 = boto3.client("s3", region_name="us-east-1")
BUCKET = "pgxdatalake"

def get_meta(cohort):
    obj = s3.get_object(Bucket=BUCKET, Key=f"gold/dashboard/metadata/metadata_{cohort}.json")
    return json.loads(obj["Body"].read())

# ── non_opioid_ed ──────────────────────────────────────────────────
print("\n=== non_opioid_ed top features (dashboard metadata) ===")
d = get_meta("non_opioid_ed")
codes = d["codes"]
for band in ["65-74", "75-84", "85-114"]:
    bc = codes.get(band, {})
    drugs = sorted(bc.get("drugs", []), key=lambda r: -r.get("importance", 0))
    icds  = sorted(bc.get("icds",  []), key=lambda r: -r.get("importance", 0))
    cpts  = sorted(bc.get("cpts",  []), key=lambda r: -r.get("importance", 0))
    print(f"\n  non_opioid_ed / {band}  (drugs={len(drugs)} icds={len(icds)} cpts={len(cpts)})")
    for i, x in enumerate(drugs[:10]):
        disp = x.get("display", x.get("code", "?"))
        print(f"    drug {i+1:2d}. {disp:<45s} imp={x.get('importance', 0):.3f}")
    for i, x in enumerate(icds[:5]):
        disp = x.get("display", x.get("code", "?"))
        print(f"    icd  {i+1:2d}. {disp:<45s} imp={x.get('importance', 0):.3f}")

# ── opioid_ed ──────────────────────────────────────────────────────
print("\n=== opioid_ed top features (dashboard metadata) ===")
d = get_meta("opioid_ed")
codes = d["codes"]
for band in ["13-24", "25-44", "45-54", "55-64"]:
    bc = codes.get(band, {})
    drugs = sorted(bc.get("drugs", []), key=lambda r: -r.get("importance", 0))
    icds  = sorted(bc.get("icds",  []), key=lambda r: -r.get("importance", 0))
    cpts  = sorted(bc.get("cpts",  []), key=lambda r: -r.get("importance", 0))
    print(f"\n  opioid_ed / {band}  (drugs={len(drugs)} icds={len(icds)} cpts={len(cpts)})")
    for i, x in enumerate(drugs[:5]):
        disp = x.get("display", x.get("code", "?"))
        print(f"    drug {i+1:2d}. {disp:<45s} imp={x.get('importance', 0):.3f}")
    for i, x in enumerate(icds[:5]):
        disp = x.get("display", x.get("code", "?"))
        print(f"    icd  {i+1:2d}. {disp:<45s} imp={x.get('importance', 0):.3f}")

# ── check for any FFA interaction keys ────────────────────────────
print("\n=== checking for FFA interaction fields in metadata ===")
for cohort in ["opioid_ed", "non_opioid_ed"]:
    d2 = get_meta(cohort)
    for k, v in d2.items():
        if k not in ("cohort", "age_bands", "codes"):
            print(f"  {cohort}: extra key '{k}' => {str(v)[:100]}")
