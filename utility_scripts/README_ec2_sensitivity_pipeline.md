# EC2 CH4 util-free sensitivity pipeline — plan & status

**Last updated:** 2026-07-15 ~08:08 EDT (12:08 UTC)  
**Owner intent:** CTS-2026-0235R2 CH4 (`non_opioid_ed`) utilization-free sensitivity for **all** modeled age bands, with SES emails per step and EC2 auto-stop on success.

Update the **Status** section whenever you check the instance so a new chat can resume without prior context.

---

## Status (live run)

| Field | Value |
|:------|:------|
| Overall | **IN PROGRESS** — cohort create for missing age bands |
| Instance | `pgx-analysis-1a` / `i-0e7d1bd469620c0bb` / **running** |
| Public IP (may change after stop/start) | `52.200.121.30` |
| SSH user / key | `ec2-user` + `C:\Projects\mushin_pgx.pem` (repo under `/home/pgx3874/pgx-analysis`; do **not** SSH as `pgx3874` — key refuses) |
| Repo commit on EC2 | Prefer `main` ≥ `45afa88d` (orchestrator). Earlier start used `96afe677`; script was scp’d independently. |
| Orchestrator PID | `21720` (`bash utility_scripts/ec2_run_sensitivity_pipeline.sh`) |
| Log | `/mnt/nvme/pgx-analysis/logs/sensitivity_pipeline_20260715_114655.log` |
| Nohup mirror | `/mnt/nvme/pgx-analysis/logs/nohup_sensitivity.out` |
| Current work (as of last check) | `0_create_cohort.py --age-band 13-24 --event-year 2018 --cohort ed_non_opioid` |
| SES recipient | `dixonrj@vcu.edu` (from `jerome@mushinsolutions.com`) |

### Step checklist

| Step | State | Notes |
|:-----|:------|:------|
| 1. Sync code | **DONE** | SES sent: `[pgx-analysis-1a] STEP1 COMPLETE: code synced` |
| 2. Check model data | **DONE** | Incomplete on S3; SES step2 + step2b after gold sync |
| 2b. Sync gold medical/pharmacy (+ FI) for missing bands | **DONE** | → `/mnt/nvme/gold/` |
| 3. Create cohorts (missing bands × 2016–2019) | **IN PROGRESS** | Finished `0-12` all years; working through `13-24` (on 2018). Still queued: rest of `13-24`, then `25-44`, `45-54`, `75-84` |
| 4. Create model data for missing bands | PENDING | `create_model_data.py --cohort non_opioid_ed --age-band …` |
| 5. Run sensitivity (all 8 bands) | PENDING | `python 6_final_model/run_sensitivity_util_free.py` |
| 6. Stop EC2 + final SES | PENDING | Script calls `aws ec2 stop-instances` then SES FINAL |

### `model_events` on NVMe

| Age band | Present? | Source |
|:---------|:---------|:-------|
| 55-64 | Yes | SCP from local Windows `4_model_data/` |
| 65-74 | Yes | SCP from local |
| 85-114 | Yes | SCP from local |
| 85-94 | Yes (extra) | SCP from local; not in `AGE_BANDS` |
| 0-12, 13-24, 25-44, 45-54, 75-84 | **No** (at last check) | Being rebuilt via cohorts → Step 4 |

---

## Plan (recovery SSOT)

### Why this path

- S3 `gold/cohorts_model_data/` and `gold/final_model/` were **empty** (cleanup earlier).
- Local Windows only had partial `model_events` (55-64, 65-74, 85-114).
- Gold medical (~86G) + pharmacy (~5G) remain on S3; NVMe (`nvme1n1`) was formatted/mounted at `/mnt/nvme` on this boot (ephemeral — **lost if instance stop/terminate without re-sync**).

### Orchestrator

- Script: [`utility_scripts/ec2_run_sensitivity_pipeline.sh`](ec2_run_sensitivity_pipeline.sh) (committed on `main` as `45afa88d`)
- Runner / science: [`6_final_model/run_sensitivity_util_free.py`](../6_final_model/run_sensitivity_util_free.py)
- EC2 driver notebook (optional): repo-root `3_model_sensitivity.ipynb`
- Manuscript numbers SSOT: `manuscript/data/supplementary/ch04_util_free_sensitivity/` (+ `sensitivity_summary_all_bands.json`)

### Intended sequence

1. SSH as `ec2-user` → work as / under `pgx3874` home / sudo-chown NVMe as needed.
2. Ensure repo at `/home/pgx3874/pgx-analysis` (`git pull` or clone).
3. Mount `/mnt/nvme` if needed; set `PGX_DATA_ROOT=/mnt/nvme`.
4. If all age bands have `model_events` under `/mnt/nvme/4_model_data/cohort_name=non_opioid_ed/` → skip to sensitivity.
5. Else: sync gold for **missing** physical age bands only → create `ed_non_opioid` cohorts (2016–2019) → `create_model_data` for missing → sensitivity all bands.
6. SES after each major step; on success stop instance and send FINAL email.

### Age bands

`0-12`, `13-24`, `25-44`, `45-54`, `55-64`, `65-74`, `75-84`, `85-114`  
(`85-114` uses physical gold `85-94` + `95-114` when rebuilding.)

---

## How to monitor (new session)

```powershell
# Instance state + IP
aws ec2 describe-instances --instance-ids i-0e7d1bd469620c0bb `
  --query "Reservations[0].Instances[0].{State:State.Name,IP:PublicIpAddress}" `
  --output json --profile mushin

# SSH (refresh IP first)
$key = "C:\Projects\mushin_pgx.pem"
$ip = "<PublicIpAddress>"
ssh -i $key -o BatchMode=yes -o IdentitiesOnly=yes ec2-user@$ip

# On box:
pgrep -af 'ec2_run_sensitivity|0_create_cohort|create_model_data|run_sensitivity'
tail -f /mnt/nvme/pgx-analysis/logs/sensitivity_pipeline_20260715_114655.log
ls -lh /mnt/nvme/4_model_data/cohort_name=non_opioid_ed/*/model_events.parquet
```

**SES subjects to watch for:**  
`STEP1` … `STEP5` … `FINAL: sensitivity OK + EC2 shutdown initiated`  
Errors: `ERROR: cohort create failed` / `model data failed` / `sensitivity analysis failed`

---

## Resume / restart if job dies

1. Confirm instance still running; remount NVMe if `/mnt/nvme` empty after stop/start (**requires re-format or remount + re-sync** — this volume was created empty this session).
2. Re-pull repo as `pgx3874` if needed.
3. Re-check which `model_events` exist; only rebuild missing bands.
4. Re-launch:

```bash
sudo -u pgx3874 -H bash -lc '
cd /home/pgx3874/pgx-analysis
nohup bash utility_scripts/ec2_run_sensitivity_pipeline.sh \
  >> /mnt/nvme/pgx-analysis/logs/nohup_sensitivity.out 2>&1 &
'
```

5. If cohorts already exist on S3 (`s3://pgxdatalake/gold/cohorts/cohort_name=non_opioid_ed/...`) but NVMe gold was wiped, sync cohorts + medical/pharmacy for needed bands before Step 4.
6. Sensitivity alone (all local model_events present):

```bash
cd /home/pgx3874/pgx-analysis
export PYTHONPATH=$PWD PGX_DATA_ROOT=/mnt/nvme
/home/pgx3874/jupyter-env/bin/python3.11 6_final_model/run_sensitivity_util_free.py
```

7. **Do not** stop the instance until sensitivity succeeds unless intentionally aborting (user asked for shutdown only after final success).

---

## Known constraints

- Login: **`ec2-user`**, not `pgx3874` (Permission denied with `mushin_pgx.pem`).
- Ephemeral NVMe: data under `/mnt/nvme` does not survive instance stop unless copied to EBS/S3; orchestrator uploads cohorts to S3 during create.
- Full missing-band rebuild = 5 bands × 4 years cohort jobs + 5× `create_model_data` + multi-band XGB sensitivity — **hours**.
- Python: `/home/pgx3874/jupyter-env/bin/python3.11` (xgboost etc. verified).

---

## Update log

| When (local) | Note |
|:-------------|:-----|
| 2026-07-15 ~07:30–08:00 EDT | Started instance, mounted/formatted NVMe, cloned repo, SES step1, SCP’d partial model_events, launched orchestrator |
| 2026-07-15 ~08:08 EDT | Still on cohort `13-24` / `2018`; `0-12` years done; this README added for session recovery |
