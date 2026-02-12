#!/usr/bin/env python3
"""
Log current processes, memory utilization, and CPU utilization.

Useful when running long pipelines (e.g. dashboard visuals, DTW, BupaR) to see
what is running and whether the machine is under load. Run from repo root or anywhere.

Usage:
  python utility_scripts/log_system_status.py
  python utility_scripts/log_system_status.py --pgx-only   # only Python/R processes that look like pgx/dtw/bupar
  python utility_scripts/log_system_status.py --top 20     # show top 20 by CPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psutil
except ImportError:
    print("psutil not installed. pip install psutil")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Log current processes, memory and CPU utilization.")
    ap.add_argument("--pgx-only", action="store_true", help="Only show processes that look like pgx/dtw/bupar/R")
    ap.add_argument("--top", type=int, default=30, help="Max number of processes to list (default 30)")
    args = ap.parse_args()

    # System memory
    vm = psutil.virtual_memory()
    print("Memory")
    print("-" * 50)
    print(f"  Total:     {vm.total / (1024**3):.1f} GB")
    print(f"  Available: {vm.available / (1024**3):.1f} GB")
    print(f"  Used:      {vm.used / (1024**3):.1f} GB  ({vm.percent}%)")
    print()

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    print("CPU")
    print("-" * 50)
    print(f"  Overall:   {cpu_percent}%")
    print(f"  Per-CPU:   {len(per_cpu)} cores, avg={sum(per_cpu)/len(per_cpu):.1f}%  (sample: {per_cpu[:8]})")
    print()

    # Processes (one quick pass to prime CPU %, then collect)
    keywords = ("python", "Rscript", "R.exe", "create_dtw", "bupar", "fpgrowth", "pgx", "jupyter")
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pinfo = p.info
            name = (pinfo.get("name") or "").lower()
            cmdline = pinfo.get("cmdline") or []
            cmd = " ".join(cmdline) if isinstance(cmdline, list) else str(cmdline)
            if args.pgx_only and not any(k in name or k in cmd for k in keywords):
                continue
            try:
                cpu = p.cpu_percent()
                mem = p.memory_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu = mem = 0
            procs.append({
                "pid": pinfo.get("pid"),
                "name": pinfo.get("name") or "?",
                "cpu": cpu,
                "mem": mem,
                "cmd": f"{cmd[:120]}..." if len(cmd) > 120 else cmd,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda x: (x["cpu"], x["mem"]), reverse=True)
    procs = procs[: args.top]

    print("Processes" + (" (pgx-related only)" if args.pgx_only else "") + f" (top {len(procs)})")
    print("-" * 50)
    print(f"  {'PID':<8} {'CPU%':<7} {'MEM%':<7} Name / command")
    for p in procs:
        print(f"  {p['pid']:<8} {p['cpu']:<7.1f} {p['mem']:<7.1f} {p['name']}  {p['cmd'][:80]}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
