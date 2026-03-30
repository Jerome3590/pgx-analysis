#!/usr/bin/env python3
"""
Build CH2 fig_attrition.pdf from TikZ using data/attrition_ch02.json.

  - Intermediate counts: edit data/attrition_ch02.json (or paste from cohort logs).
  - Cohort logic / exclusions / targets: see _meta in data/attrition_ch02.json
    (authoring notes; not for manuscript body text).
  - APCD universe (first box): optional Athena refresh via --athena using
    data/attrition_athena_apcd.sql (adjust SQL to match Glue catalog).
  - Final box N: use --sync-final to set from cohort_counts.json (S3-derived
    train+test deduped counts from scripts/get_cohort_counts.py).

Outputs:
  manuscript/figures/ch02/fig_attrition.tikz  (fragment)
  manuscript/figures/ch02/fig_attrition_standalone.tex
  manuscript/figures/ch02/fig_attrition.pdf     (if xelatex/pdflatex available)

Usage:
  python scripts/generate_fig_attrition_ch02.py
  python scripts/generate_fig_attrition_ch02.py --sync-final --athena
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT = SCRIPT_DIR.parent
DATA_DIR = MANUSCRIPT / "data"
FIG_DIR = MANUSCRIPT / "figures" / "ch02"
ATTRITION_JSON = DATA_DIR / "attrition_ch02.json"
COHORT_COUNTS_JSON = MANUSCRIPT / "data/cohort_counts.json"
ATHENA_SQL = DATA_DIR / "attrition_athena_apcd.sql"


def _latex_escape_text(s: str) -> str:
    """Escape plain text for LaTeX (column titles; no math)."""
    out = []
    for ch in s:
        if ch in ("\\", "&", "%", "#", "_", "{", "}"):
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _fmt_n_tex(n: int) -> str:
    """Format integer with thousands separators for TikZ text mode."""
    return f"{n:,}"


def _sum_cohort_totals(cohort: str, data: Dict[str, Any]) -> int:
    bands = data.get(cohort) or {}
    total = 0
    for _band, row in bands.items():
        total += int(row.get("cases", 0)) + int(row.get("controls", 0))
    return total


def _athena_count(sql: str, database: str, workgroup: str, output_uri: str) -> Optional[int]:
    try:
        import boto3
    except ImportError:
        print("boto3 not installed; skip Athena.", file=sys.stderr)
        return None

    athena = boto3.client("athena", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    r = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_uri},
        WorkGroup=workgroup,
    )
    qid = r["QueryExecutionId"]
    for _ in range(60):
        time.sleep(1)
        st = athena.get_query_execution(QueryExecutionId=qid)
        state = st["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
    if state != "SUCCEEDED":
        reason = st["QueryExecution"]["Status"].get("StateChangeReason", "")
        print(f"Athena failed: {state} {reason}", file=sys.stderr)
        return None
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    if len(rows) < 2:
        return None
    val = rows[1]["Data"][0].get("VarCharValue", "")
    try:
        return int(val.replace(",", "").strip())
    except ValueError:
        return None


def _strip_sql_comments(sql: str) -> str:
    lines = []
    for line in sql.splitlines():
        s = line.strip()
        if s.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _build_tikz_fragment(
    prefix: str,
    title: str,
    steps: List[Dict[str, Any]],
    x: float,
) -> str:
    """Single column: column title + CONSORT-style boxes at fixed x."""
    lines: List[str] = []
    dy = 2.25
    y_title = 0.35
    y0 = -1.05

    lines.append(
        rf"\node[font=\bfseries\footnotesize\sffamily, anchor=south] (hdr-{prefix}) "
        rf"at ({x:.2f},{y_title:.2f}) {{{_latex_escape_text(title)}}};"
    )

    for i, st in enumerate(steps):
        n = st["n"]
        label = st["label"]
        if n is None:
            raise ValueError(f"Step {i} in {prefix} has null n after sync")
        n_str = _fmt_n_tex(int(n))
        y = y0 - i * dy
        lines.append(
            rf"\node[box] (box-{prefix}-{i}) at ({x:.2f},{y:.2f}) "
            rf"{{{label} \\[0.35em] "
            rf"\textbf{{N = {n_str}}}}};"
        )

    for i in range(1, len(steps)):
        prev_n = steps[i - 1]["n"]
        n = steps[i]["n"]
        excl = int(prev_n) - int(n)
        if excl > 0:
            lines.append(
                rf"\path (box-{prefix}-{i-1}.south) -- (box-{prefix}-{i}.north) "
                rf"coordinate[pos=0.52] (mid-{prefix}-{i});"
            )
            lines.append(
                rf"\node[excl, anchor=west] at ([xshift=3mm]mid-{prefix}-{i}) "
                rf"{{excluded {_fmt_n_tex(excl)}}};"
            )
    for i in range(len(steps) - 1):
        lines.append(
            rf"\draw[arr] (box-{prefix}-{i}.south) -- (box-{prefix}-{i+1}.north);"
        )

    return "\n".join(lines)


def _build_standalone_tex(left: str, right: str) -> str:
    return rf"""
\documentclass[tikz,border=3mm]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{positioning,arrows.meta,calc}}
\begin{{document}}
\begin{{tikzpicture}}[
  font=\footnotesize\sffamily,
  box/.style={{draw, rounded corners=2pt, align=center, text width=3.55cm,
    minimum height=1.05cm, inner sep=5pt, fill=white, draw=gray!75}},
  arr/.style={{-{{Stealth[length=2.2mm]}}, thick, draw=gray!60}},
  excl/.style={{font=\scriptsize\sffamily, text=red!65!black}},
  every node/.style={{align=center}}
]
{left}
{right}
\end{{tikzpicture}}
\end{{document}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate CH2 attrition TikZ + PDF.")
    ap.add_argument(
        "--sync-final",
        action="store_true",
        help="Set last step N from cohort_counts.json (cases+controls summed over bands).",
    )
    ap.add_argument(
        "--athena",
        action="store_true",
        help="Refresh first-step APCD N via Athena (needs data/attrition_athena_apcd.sql).",
    )
    ap.add_argument(
        "--athena-database",
        default=os.environ.get("ATHENA_ATTRITION_DB", "silver_medical"),
        help="Glue database for APCD count query.",
    )
    ap.add_argument(
        "--athena-workgroup",
        default=os.environ.get("ATHENA_WORKGROUP", "APCD"),
    )
    ap.add_argument(
        "--athena-output",
        default=os.environ.get("ATHENA_OUTPUT", "s3://pgxdatalake/athena-query-results/"),
    )
    ap.add_argument("--no-compile", action="store_true", help="Write .tex only; skip PDF.")
    args = ap.parse_args()

    with open(ATTRITION_JSON, encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    if args.sync_final and COHORT_COUNTS_JSON.exists():
        with open(COHORT_COUNTS_JSON, encoding="utf-8") as f:
            cc = json.load(f)
        for cohort_key in ("opioid_ed", "non_opioid_ed"):
            col = cfg.get(cohort_key)
            if not col:
                continue
            steps: List[Dict[str, Any]] = col["steps"]
            if not steps or steps[-1].get("n") is not None:
                continue
            steps[-1]["n"] = _sum_cohort_totals(cohort_key, cc)
    else:
        for cohort_key in ("opioid_ed", "non_opioid_ed"):
            col = cfg.get(cohort_key)
            if not col:
                continue
            steps = col["steps"]
            if steps and steps[-1].get("n") is None:
                if COHORT_COUNTS_JSON.exists():
                    with open(COHORT_COUNTS_JSON, encoding="utf-8") as f:
                        cc = json.load(f)
                    steps[-1]["n"] = _sum_cohort_totals(cohort_key, cc)
                else:
                    print(
                        "Warning: last step n is null and cohort_counts.json missing; "
                        "use --sync-final after get_cohort_counts.py",
                        file=sys.stderr,
                    )

    if args.athena and ATHENA_SQL.exists():
        raw_sql = ATHENA_SQL.read_text(encoding="utf-8")
        sql = _strip_sql_comments(raw_sql)
        if sql:
            n0 = _athena_count(sql, args.athena_database, args.athena_workgroup, args.athena_output)
            if n0 is not None:
                for cohort_key in ("opioid_ed", "non_opioid_ed"):
                    col = cfg.get(cohort_key)
                    if col and col["steps"]:
                        col["steps"][0]["n"] = n0
                print(f"Athena: APCD universe N = {n0:,}")
            else:
                print("Athena: query returned no count; keeping JSON.", file=sys.stderr)
        else:
            print("attrition_athena_apcd.sql has no executable SQL after stripping comments.", file=sys.stderr)

    # Validate steps
    for cohort_key in ("opioid_ed", "non_opioid_ed"):
        col = cfg.get(cohort_key)
        if not col:
            continue
        prev: Optional[int] = None
        for st in col["steps"]:
            n = st.get("n")
            if n is None:
                raise SystemExit(f"Invalid {cohort_key}: null n remains. Use --sync-final.")
            n = int(n)
            if prev is not None and n > prev:
                print(
                    f"Warning: {cohort_key} step N={n:,} increases from previous {prev:,} "
                    "(attrition should be non-increasing).",
                    file=sys.stderr,
                )
            prev = n

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    left = _build_tikz_fragment(
        "opioid",
        cfg["opioid_ed"]["title"],
        cfg["opioid_ed"]["steps"],
        x=0.0,
    )

    right = _build_tikz_fragment(
        "polypharmacy",
        cfg["non_opioid_ed"]["title"],
        cfg["non_opioid_ed"]["steps"],
        x=9.0,
    )

    frag = "% Auto-generated by generate_fig_attrition_ch02.py\n" + left + "\n\n" + right
    (FIG_DIR / "fig_attrition.tikz").write_text(frag, encoding="utf-8")

    standalone = _build_standalone_tex(left, right)
    tex_path = FIG_DIR / "fig_attrition_standalone.tex"
    tex_path.write_text(standalone, encoding="utf-8")

    if args.no_compile:
        print(f"Wrote {tex_path} (skipped PDF).")
        return 0

    for engine in ("xelatex", "pdflatex"):
        try:
            r = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", str(tex_path.name)],
                cwd=FIG_DIR,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0:
                pdf_out = FIG_DIR / "fig_attrition.pdf"
                # standalone produces fig_attrition_standalone.pdf
                alt = FIG_DIR / "fig_attrition_standalone.pdf"
                if alt.exists():
                    alt.replace(pdf_out)
                print(f"OK: {pdf_out} ({engine})")
                return 0
            print(r.stdout[-2000:] if r.stdout else "", file=sys.stderr)
            print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            print(f"{engine} timed out", file=sys.stderr)

    print(
        "LaTeX engine not found or compile failed; wrote .tex — run xelatex in figures/ch02/",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
