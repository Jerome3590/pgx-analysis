"""
Refined Lambda timing: percentile distribution + ECR image pull estimate.
"""
import boto3, time
from datetime import datetime, timedelta, timezone

logs = boto3.client("logs", region_name="us-east-1")
cw   = boto3.client("cloudwatch", region_name="us-east-1")
ecr  = boto3.client("ecr",  region_name="us-east-1")
amp  = boto3.client("amplify", region_name="us-east-1")

end   = datetime.now(timezone.utc)
start = end - timedelta(days=90)
lg    = "/aws/lambda/pgx-risk-calculator"


def query(qs, wait=12):
    q = logs.start_query(
        logGroupName=lg,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=qs,
    )
    time.sleep(wait)
    return logs.get_query_results(queryId=q["queryId"])


# ── 1. Warm-only percentile distribution ─────────────────────────────────────
print("=== Warm (non-cold) REPORT durations ===")
res = query(
    "fields @message "
    "| filter @message like /REPORT/ "
    "| filter @message not like /Init Duration/ "
    "| parse @message 'Duration: * ms' as dur_ms "
    "| stats "
    "    count()        as n, "
    "    avg(dur_ms)    as avg, "
    "    stddev(dur_ms) as sd, "
    "    pct(dur_ms,50) as p50, "
    "    pct(dur_ms,95) as p95, "
    "    pct(dur_ms,99) as p99",
    wait=14
)
for row in res.get("results", []):
    d = {r["field"]: r["value"] for r in row}
    print(f"  n={d.get('n')}  avg={d.get('avg')}ms  SD={d.get('sd')}ms  "
          f"p50={d.get('p50')}ms  p95={d.get('p95')}ms  p99={d.get('p99')}ms")

# ── 2. All REPORT lines — parse billed duration ───────────────────────────────
print("\n=== Billed duration (warm only) ===")
res = query(
    "fields @message "
    "| filter @message like /REPORT/ "
    "| filter @message not like /Init Duration/ "
    "| parse @message 'Billed Duration: * ms' as billed_ms "
    "| stats avg(billed_ms) as avg_billed, stddev(billed_ms) as sd_billed, "
    "        pct(billed_ms,50) as p50, count() as n",
    wait=12
)
for row in res.get("results", []):
    d = {r["field"]: r["value"] for r in row}
    print(f"  n={d.get('n')}  avg={d.get('avg_billed')}ms  "
          f"SD={d.get('sd_billed')}ms  p50={d.get('p50')}ms")

# ── 3. ECR image size (proxy for pull time) ───────────────────────────────────
print("\n=== ECR image sizes ===")
try:
    repos = ecr.describe_repositories()["repositories"]
    for repo in repos:
        name = repo["repositoryName"]
        try:
            imgs = ecr.describe_images(
                repositoryName=name,
                filter={"tagStatus": "TAGGED"},
            )["imageDetails"]
            imgs_sorted = sorted(imgs, key=lambda x: x.get("imagePushedAt", 0), reverse=True)
            latest = imgs_sorted[0] if imgs_sorted else None
            if latest:
                size_mb = latest.get("imageSizeInBytes", 0) / 1e6
                pushed  = latest.get("imagePushedAt", "")
                print(f"  {name}: {size_mb:.0f} MB  (pushed {pushed})")
        except Exception:
            pass
except Exception as e:
    print(f"  ECR error: {e}")

# ── 4. CloudFront / Amplify for frontend timing ───────────────────────────────
print("\n=== Amplify apps ===")
try:
    apps = amp.list_apps()["apps"]
    for app in apps:
        print(f"  {app['name']}: {app.get('defaultDomain','')}  "
              f"lastModified={app.get('updateTime','')}")
except Exception as e:
    print(f"  Amplify error: {e}")

# ── 5. CloudFront distributions ───────────────────────────────────────────────
print("\n=== CloudFront distributions ===")
try:
    cf = boto3.client("cloudfront", region_name="us-east-1")
    dists = cf.list_distributions()
    items = dists.get("DistributionList", {}).get("Items", [])
    for d in items:
        aliases = d.get("Aliases", {}).get("Items", [])
        origins = [o["DomainName"] for o in d.get("Origins", {}).get("Items", [])]
        print(f"  {d['Id']}: aliases={aliases}  origins={origins[:2]}")
except Exception as e:
    print(f"  CloudFront error: {e}")
