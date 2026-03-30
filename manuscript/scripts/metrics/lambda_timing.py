"""
Query CloudWatch for Lambda Duration metrics for CH_5 benchmark table.
"""
import boto3
from datetime import datetime, timedelta, timezone

logs = boto3.client("logs", region_name="us-east-1")
cw   = boto3.client("cloudwatch", region_name="us-east-1")

end   = datetime.now(timezone.utc)
start = end - timedelta(days=180)

functions = ["pgx-risk-calculator", "pgx-demo", "pgx-images"]

print("=== CloudWatch Duration stats (last 180 days) ===")
for fn in functions:
    resp = cw.get_metric_statistics(
        Namespace="AWS/Lambda",
        MetricName="Duration",
        Dimensions=[{"Name": "FunctionName", "Value": fn}],
        StartTime=start, EndTime=end,
        Period=86400 * 180,
        Statistics=["Average", "Maximum", "SampleCount"],
    )
    pts = resp["Datapoints"]
    if pts:
        p = pts[0]
        avg = p["Average"]
        mx  = p["Maximum"]
        n   = int(p["SampleCount"])
        print(f"  {fn}: avg={avg:.1f}ms  max={mx:.1f}ms  n={n}")
    else:
        print(f"  {fn}: no datapoints")

# Also pull Init Duration (cold start) from log insights
print("\n=== Init Duration (cold-start) from REPORT logs ===")
for fn in ["pgx-risk-calculator", "pgx-demo"]:
    log_group = f"/aws/lambda/{fn}"
    try:
        q = logs.start_query(
            logGroupName=log_group,
            startTime=int((end - timedelta(days=90)).timestamp()),
            endTime=int(end.timestamp()),
            queryString=(
                "fields @message "
                "| filter @message like /Init Duration/ "
                "| parse @message 'Init Duration: * ms' as init_ms "
                "| stats avg(init_ms) as avg_init, "
                "        stddev(init_ms) as sd_init, "
                "        count() as n"
            ),
        )
        import time; time.sleep(5)
        res = logs.get_query_results(queryId=q["queryId"])
        for row in res.get("results", []):
            vals = {r["field"]: r["value"] for r in row}
            print(f"  {fn}: avg_init={vals.get('avg_init','?')}ms  n={vals.get('n','?')}")
    except Exception as e:
        print(f"  {fn}: {e}")

# PGx card endpoint timing from application logs
print("\n=== /pgx-card endpoint duration from logs ===")
for fn in ["pgx-risk-calculator", "pgx-demo"]:
    log_group = f"/aws/lambda/{fn}"
    try:
        q = logs.start_query(
            logGroupName=log_group,
            startTime=int((end - timedelta(days=90)).timestamp()),
            endTime=int(end.timestamp()),
            queryString=(
                "fields @message "
                "| filter @message like /pgx.card/ or @message like /cpic/ "
                "| filter @message like /duration/ or @message like /ms/ "
                "| limit 20"
            ),
        )
        import time; time.sleep(5)
        res = logs.get_query_results(queryId=q["queryId"])
        if res.get("results"):
            for row in res["results"][:5]:
                print(f"  {fn}:", {r["field"]: r["value"] for r in row}.get("@message", "")[:120])
        else:
            print(f"  {fn}: no matching log entries")
    except Exception as e:
        print(f"  {fn}: {e}")
