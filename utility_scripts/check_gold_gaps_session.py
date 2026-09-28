"""SSM status for the gold-cohort gap session."""
from pathlib import Path

import boto3

IID = "i-08d39967023dcc7a1"


def main() -> None:
    s = boto3.Session(profile_name="mushin", region_name="us-east-1")
    ssm = s.client("ssm")
    cmd = ssm.send_command(
        InstanceIds=[IID],
        DocumentName="AWS-RunShellScript",
        TimeoutSeconds=45,
        Parameters={
            "commands": [
                "echo ===PS===",
                "pgrep -af '0_create_cohort|run_gold|run_ec2_analysis|cancel_pgx' || true",
                "echo ===WRAP===",
                "grep -E 'KEEP_ALIVE|Not on EC2|Cancel|Terminate|SESSION OK|SES |DONE instance' /mnt/nvme/pgx-analysis/logs/session_*.log /home/ec2-user/pgx-analysis/logs/session_*.log 2>/dev/null | tail -n 30 || true",
                "echo ===NVME===",
                "df -h /mnt/nvme 2>/dev/null || df -h /",
                "ls -ld /mnt/nvme 2>/dev/null || echo no-nvme",
                "echo ===LOCAL_GOLD===",
                "find /mnt/nvme/gold/cohorts -name 'cohort.parquet' 2>/dev/null | sort || true",
                "echo ===CREATE_MARKERS===",
                "grep -E '==== CREATE COHORT|==== BIN TRANSITIONS|==== JOB DONE|Saved|Uploaded|Writing cohort|error|Error|ERROR' /mnt/nvme/pgx-analysis/logs/session_*.log 2>/dev/null | tail -n 40 || true",
                "echo ===TAIL===",
                "LOG=$(ls -t /mnt/nvme/pgx-analysis/logs/session_*.log /home/ec2-user/pgx-analysis/logs/session_*.log 2>/dev/null | head -1)",
                "echo LOGFILE=$LOG",
                'if [ -n "$LOG" ]; then tail -n 20 "$LOG"; fi',
            ]
        },
    )
    cid = cmd["Command"]["CommandId"]
    ssm.get_waiter("command_executed").wait(
        CommandId=cid, InstanceId=IID, WaiterConfig={"Delay": 3, "MaxAttempts": 15}
    )
    inv = ssm.get_command_invocation(CommandId=cid, InstanceId=IID)
    text = f"{inv.get('Status')}\n{inv.get('StandardOutputContent') or ''}\n{inv.get('StandardErrorContent') or ''}"
    Path(__file__).with_name("check_gold_gaps_session.out.txt").write_text(text, encoding="utf-8")
    print(inv.get("Status"), "wrote check_gold_gaps_session.out.txt")


if __name__ == "__main__":
    main()
