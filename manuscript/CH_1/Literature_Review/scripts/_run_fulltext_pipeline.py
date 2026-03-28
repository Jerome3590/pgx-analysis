"""
_run_fulltext_pipeline.py — Master idempotent full-text pipeline
────────────────────────────────────────────────────────────────
Runs all full-text acquisition and scoring steps in order.
Every step checks what already exists before doing work.
Safe to run multiple times — re-running completes only the remaining gaps.

Steps (all automated except 3e which requires Duo 2FA):
  1.  Extract PDFs in data/scholar_pdfs/              → data/scholar_json/
  2.  Parse local PMC BioC JSONs                      → data/scholar_json/
  3.  PMC Open-Access API fetch (real PMC IDs)        → data/scholar_json/
  3c. DOI lookup: NCBI+CrossRef for VCU queue         → scripts/vcu_queue_with_dois.csv
  3d. Free OA scan: EuropePMC/CORE/SemanticScholar    → data/scholar_json/ (direct)
  3e. VCU proxy download (Puppeteer, needs Duo auth)  → data/scholar_pdfs/
  3f. Extract new PDFs from step 3e                   → data/scholar_json/
  3b. Import manually-placed PDFs from vcu_downloads/ → data/scholar_json/
  4.  Re-score with pytextrank (full text)            → articles_screened.csv
  4b. NIH AI + operational performance tagging        → articles_screened.csv (+nih_ai_tags, +op_perf_tags)
  5.  Rebuild checklist                               → manual_review/article_review_checklist.csv
  5b. Enrich scholar_json with classification metadata → data/scholar_json/*.json
  5c. Generate PRISMA 2020 flowchart                  → figures/fig_prisma_flowchart.pdf
  5d. Regenerate wordclouds (always last)             → data/wordclouds/*.png/.pdf
  6.  [MANUAL] Google Sheets review → _apply_checklist_decisions.py

Usage:
  python scripts/_run_fulltext_pipeline.py              # full run from step 1
  python scripts/_run_fulltext_pipeline.py --step 3c    # DOI lookup + OA scan + VCU
  python scripts/_run_fulltext_pipeline.py --step 3d    # OA scan only
  python scripts/_run_fulltext_pipeline.py --step 3e    # VCU proxy (needs credentials)
  python scripts/_run_fulltext_pipeline.py --step 4     # rescore only
  python scripts/_run_fulltext_pipeline.py --step 4b    # NIH AI + op_perf tagging only
  python scripts/_run_fulltext_pipeline.py --step 5b    # enrich JSONs only
  python scripts/_run_fulltext_pipeline.py --step 5c    # PRISMA figure only
  python scripts/_run_fulltext_pipeline.py --step 5d    # wordclouds only
  python scripts/_run_fulltext_pipeline.py --dry-run    # show status only
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path

# ── Quick coverage stats (no heavy imports) ────────────────────────────────────
def coverage_stats() -> dict:
    screened    = Path("data/ontology/articles_screened.csv")
    scholar_json = Path("data/scholar_json")
    pdf_dir     = Path("data/scholar_pdfs")

    json_index = {p.stem for p in scholar_json.glob("*.json")}
    rows = list(csv.DictReader(open(screened, encoding="utf-8-sig")))

    have_json = sum(
        1 for r in rows
        if r.get("pmc_id","").strip() in json_index
        or f"article_{r.get('article_id','')}".replace("article_", "") in json_index
        or r.get("pmc_id","").strip() in json_index
    )
    # Recount accurately
    have_json = 0
    for r in rows:
        pmc = r.get("pmc_id","").strip()
        aid = r.get("article_id","").strip()
        if pmc in json_index or f"article_{aid}" in json_index:
            have_json += 1

    include_rows = [r for r in rows if r.get("human_decision") == "include"]
    exclude_rows = [r for r in rows if r.get("human_decision") == "exclude"]

    return {
        "total":        len(rows),
        "have_json":    have_json,
        "missing_json": len(rows) - have_json,
        "include":      len(include_rows),
        "exclude":      len(exclude_rows),
        "pdfs":         len(list(pdf_dir.glob("*.pdf"))),
        "scholar_json": len(json_index),
    }


def print_status():
    s = coverage_stats()
    pct = s["have_json"] / s["total"] * 100
    print(f"\n{'─'*55}")
    print(f"Full-text coverage  : {s['have_json']:,} / {s['total']:,}  ({pct:.1f}%)")
    print(f"  scholar_json/ files : {s['scholar_json']:,}")
    print(f"  PDFs on disk        : {s['pdfs']:,}")
    print(f"  Missing full text   : {s['missing_json']:,}")
    print(f"human_decision: include={s['include']:,}  exclude={s['exclude']:,}")
    print(f"{'─'*55}\n")


def run_step(cmd: list[str], label: str, dry_run: bool):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    if dry_run:
        print(f"  [dry-run] Would run: {' '.join(cmd)}")
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n⚠️  Step exited with code {result.returncode}. Continuing...")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step",      default="1",
                        help="Start from this step (1/2/3/3c/3d/3e/3f/3b/4/4b/5, default 1)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max articles to fetch in step 3 (for batching)")
    parser.add_argument("--api-key",   default=None,
                        help="NCBI API key for step 3 (10 req/s vs 3 req/s)")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="pytextrank threshold for step 4 (default 0.20)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--priority-only", action="store_true",
                        help="Step 3: only fetch score >= 0.10 articles")
    parser.add_argument("--skip-vcu", action="store_true",
                        help="Skip step 3e (VCU Puppeteer proxy) — run manually when Duo 2FA is available")
    args = parser.parse_args()

    py = sys.executable
    print("\n📋 Full-text pipeline — idempotent run")
    print_status()

    step = str(args.step)
    STEP_ORDER = ["1", "2", "3", "3c", "3d", "3e", "3f", "3b", "4", "4b", "5", "5b", "5c", "5d"]
    start_idx  = STEP_ORDER.index(step) if step in STEP_ORDER else 0
    active     = set(STEP_ORDER[start_idx:])

    # ── Step 1: PDF → scholar_json ─────────────────────────────────────────────
    if "1" in active:
        run_step(
            [py, "scripts/_build_full_json.py", "--pdfs", "--skip-existing"],
            "Step 1 — Extract text from PDFs → scholar_json/",
            args.dry_run,
        )

    # ── Step 2: PMC BioC source → scholar_json ──────────────────────────────────
    if "2" in active:
        run_step(
            [py, "scripts/_build_full_json.py", "--pmc", "--skip-existing"],
            "Step 2 — Parse local PMC BioC JSONs → scholar_json/",
            args.dry_run,
        )

    # ── Step 3: Fetch missing from PMC Open-Access API ────────────────────────
    if "3" in active:
        cmd = [py, "scripts/_fetch_missing_fulltext.py"]
        if args.priority_only:
            cmd.append("--priority-only")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        run_step(cmd,
                 "Step 3 — Fetch missing full text from PMC Open-Access API",
                 args.dry_run)

    # ── Step 3c: DOI lookup for VCU queue ─────────────────────────────────────
    if "3c" in active:
        cmd = [py, "scripts/_build_vcu_doi_map.py"]
        if args.api_key:
            cmd += ["--api-key", args.api_key]
        run_step(cmd,
                 "Step 3c — DOI lookup: NCBI ESummary + CrossRef → vcu_queue_with_dois.csv",
                 args.dry_run)

    # ── Step 3d: Free OA scan ──────────────────────────────────────────────────
    if "3d" in active:
        run_step(
            [py, "scripts/scholar_lookup.py", "--vcu-queue", "--source", "epmc"],
            "Step 3d — Free OA scan: EuropePMC → scholar_json/ (no auth needed)",
            args.dry_run,
        )
        run_step(
            [py, "scripts/scholar_lookup.py", "--vcu-queue", "--source", "core"],
            "Step 3d — Free OA scan: CORE.ac.uk → scholar_json/",
            args.dry_run,
        )
        run_step(
            [py, "scripts/scholar_lookup.py", "--vcu-queue", "--source", "ss"],
            "Step 3d — Free OA scan: Semantic Scholar → scholar_json/",
            args.dry_run,
        )

    # ── Step 3e: VCU proxy download (Puppeteer, requires Duo 2FA) ─────────────
    if "3e" in active:
        if args.skip_vcu:
            print("\nStep 3e — Skipped (--skip-vcu). Run manually when Duo 2FA is available:")
            print("  node scripts/vcu_download.js --input scripts/vcu_queue_with_dois.csv")
        elif not (doi_map := Path("scripts/vcu_queue_with_dois.csv")).exists():
            print("\nStep 3e — Skipped: run step 3c first to build DOI map")
        else:
            print("\n" + "="*55)
            print("  Step 3e — VCU proxy download (Puppeteer)")
            print("  Requires: node, VCU credentials in secrets/secrets.txt, Duo 2FA")
            print("="*55)
            if args.dry_run:
                print("  [dry-run] Would run: node scripts/vcu_download.js "
                      "--input scripts/vcu_queue_with_dois.csv")
            else:
                result = subprocess.run(
                    ["node", "scripts/vcu_download.js",
                     "--input", "scripts/vcu_queue_with_dois.csv"],
                    check=False,
                )
                if result.returncode != 0:
                    print(f"\n⚠️  vcu_download.js exited with code {result.returncode}")

    # ── Step 3f: Extract PDFs added by step 3e ────────────────────────────────
    if "3f" in active:
        run_step(
            [py, "scripts/_build_full_json.py", "--pdfs", "--skip-existing"],
            "Step 3f — Extract new PDFs from VCU downloads → scholar_json/",
            args.dry_run,
        )

    # ── Step 3b: Import manually-placed PDFs from data/vcu_downloads/ ────────
    if "3b" in active:
        vcu_manual = list(Path("data/vcu_downloads").glob("*.pdf")) \
                     if Path("data/vcu_downloads").exists() else []
        if vcu_manual:
            run_step(
                [py, "scripts/_import_vcu_pdfs.py"],
                f"Step 3b — Import {len(vcu_manual)} manual VCU PDFs from data/vcu_downloads/",
                args.dry_run,
            )
        else:
            print("\nStep 3b — No manual PDFs in data/vcu_downloads/ (skipping)")

    # ── Step 4: Re-score with pytextrank ─────────────────────────────────────
    if "4" in active:
        run_step(
            [py, "scripts/_phase7_review.py",
             "--threshold", str(args.threshold), "--write"],
            f"Step 4 — Re-score all articles with pytextrank (threshold={args.threshold})",
            args.dry_run,
        )

    # ── Step 4b: NIH AI + operational performance tagging ──────────────────────
    if "4b" in active:
        run_step(
            [py, "scripts/_classify_nih_ai_checklist.py"],
            "Step 4b — Tag NIH AI domains + operational performance dims → articles_screened.csv",
            args.dry_run,
        )

    # ── Step 5: Rebuild checklist ─────────────────────────────────────────────
    if "5" in active:
        run_step(
            [py, "scripts/_build_review_checklist.py"],
            "Step 5 — Rebuild Google Sheets checklist",
            args.dry_run,
        )

    # ── Step 5b: Enrich scholar_json with all classification metadata ──────────
    if "5b" in active:
        run_step(
            [py, "scripts/_enrich_scholar_json.py"],
            "Step 5b — Embed OODA/CRISP-DM/NIH-AI/OpPerf tags into scholar_json/ docs",
            args.dry_run,
        )

    # ── Step 5c: PRISMA 2020 flowchart ────────────────────────────────────────
    if "5c" in active:
        run_step(
            [py, "scripts/_generate_prisma.py"],
            "Step 5c — Generate PRISMA 2020 flowchart → figures/fig_prisma_flowchart.pdf",
            args.dry_run,
        )

    # ── Step 5d: Regenerate wordclouds ────────────────────────────────────────────
    if "5d" in active:
        run_step(
            [py, "scripts/generate_wordclouds.py"],
            "Step 5d — Regenerate wordclouds → data/wordclouds/",
            args.dry_run,
        )

    print("\n✅ Pipeline complete")
    if not args.dry_run:
        print_status()


if __name__ == "__main__":
    main()
