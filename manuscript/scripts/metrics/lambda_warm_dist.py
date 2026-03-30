"""Get warm Lambda duration percentile distribution."""
import boto3, time, numpy as np
from datetime import datetime, timedelta, timezone

logs  = boto3.client("logs", region_name="us-east-1")
end   = datetime.now(timezone.utc)
start = end - timedelta(days=90)

q = logs.start_query(
    logGroupName="/aws/lambda/pgx-risk-calculator",
    startTime=int(start.timestamp()),
    endTime=int(end.timestamp()),
    queryString=(
        "fields @message "
        "| filter @message like /REPORT/ "
        "| filter @message not like /Init Duration/ "
        "| parse @message 'Duration: * ms' as dur_ms "
        "| sort dur_ms asc "
        "| limit 200"
    ),
)
time.sleep(14)
res  = logs.get_query_results(queryId=q["queryId"])
durs = []
for row in res.get("results", []):
    for r in row:
        if r["field"] == "dur_ms":
            try:
                durs.append(float(r["value"]))
            except Exception:
                pass

a = np.array(durs)
print(f"n={len(a)}")
print(f"mean={np.mean(a):.1f}  sd={np.std(a):.1f}")
print(f"p10={np.percentile(a,10):.1f}  p25={np.percentile(a,25):.1f}  "
      f"p50={np.percentile(a,50):.1f}  p75={np.percentile(a,75):.1f}  "
      f"p90={np.percentile(a,90):.1f}  p95={np.percentile(a,95):.1f}  "
      f"p99={np.percentile(a,99):.1f}")
print("\nAll values:", sorted(a.tolist()))
