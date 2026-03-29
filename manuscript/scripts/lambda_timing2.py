"""
Detailed Lambda timing from CloudWatch Logs Insights.
Separates warm vs cold, and tries to find per-endpoint timing.
"""
import boto3, time
from datetime import datetime, timedelta, timezone

logs = boto3.client("logs", region_name="us-east-1")
cw   = boto3.client("cloudwatch", region_name="us-east-1")

end   = datetime.now(timezone.utc)
start = end - timedelta(days=90)
lg    = "/aws/lambda/pgx-risk-calculator"


def run_query(query_str, wait=10):
    q = logs.start_query(
        logGroupName=lg,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=query_str,
    )
    time.sleep(wait)
    return logs.get_query_results(queryId=q["queryId"])


# ── 1. Warm vs cold REPORT duration ──────────────────────────────────────────
print("=== REPORT duration: warm vs cold ===")
res = run_query(
    "fields @message "
    "| filter @message like /REPORT/ "
    "| parse @message 'Duration: * ms' as dur_ms "
    "| parse @message 'Init Duration: * ms' as init_ms "
    "| stats avg(dur_ms) as avg_dur, stddev(dur_ms) as sd_dur, "
    "        count() as n, "
    "        avg(init_ms) as avg_init "
    "by ispresent(init_ms) as is_cold",
    wait=12
)
for row in res.get("results", []):
    d = {r["field"]: r["value"] for r in row}
    cold = "COLD" if d.get("is_cold") == "1" else "WARM"
    print(f"  {cold}: avg_dur={d.get('avg_dur','?')}ms  sd={d.get('sd_dur','?')}ms  n={d.get('n','?')}")

# ── 2. Sample raw messages to find endpoint routing ──────────────────────────
print("\n=== Sample app log messages (non-REPORT) ===")
res = run_query(
    "fields @message "
    "| filter @message not like /REPORT/ "
    "| filter @message not like /START/ "
    "| filter @message not like /END/ "
    "| filter @message not like /INIT_START/ "
    "| limit 30",
    wait=10
)
for row in res.get("results", [])[:20]:
    d = {r["field"]: r["value"] for r in row}
    msg = d.get("@message", "")[:160]
    if msg.strip():
        print(" ", msg)

# ── 3. Per-endpoint duration if path is logged ───────────────────────────────
print("\n=== Per-path timing (if path logged) ===")
for path in ["/risk", "/cpic", "/pgx", "/card", "/images", "/visualize"]:
    res = run_query(
        f"fields @message "
        f"| filter @message like '{path}' "
        f"| limit 5",
        wait=8
    )
    rows = res.get("results", [])
    if rows:
        print(f"  {path}: {len(rows)} log entries found")
        for row in rows[:2]:
            msg = {r['field']: r['value'] for r in row}.get('@message', '')[:120]
            print(f"    {msg}")
    else:
        print(f"  {path}: no entries")

# ── 4. pgx-images function stats ─────────────────────────────────────────────
print("\n=== pgx-images Lambda stats ===")
lg2 = "/aws/lambda/pgx-images"
try:
    res = run_query.__func__ if False else None
    q = logs.start_query(
        logGroupName=lg2,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=(
            "fields @message "
            "| filter @message like /REPORT/ "
            "| parse @message 'Duration: * ms' as dur_ms "
            "| stats avg(dur_ms) as avg_dur, stddev(dur_ms) as sd_dur, count() as n"
        ),
    )
    time.sleep(10)
    res = logs.get_query_results(queryId=q["queryId"])
    for row in res.get("results", []):
        d = {r["field"]: r["value"] for r in row}
        print(f"  pgx-images: avg={d.get('avg_dur','?')}ms  sd={d.get('sd_dur','?')}ms  n={d.get('n','?')}")
except Exception as e:
    print(f"  pgx-images: {e}")
