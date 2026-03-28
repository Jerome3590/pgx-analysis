"""
Apply manual review decisions from article_review_checklist.csv back to
data/ontology/articles_screened.csv (human_decision column).

Reads the Google-Sheets-exported checklist from infrastructure_setup/manual_review/
and updates human_decision for every article_id that has selected=Y or selected=N.

Usage:
  python scripts/_apply_checklist_decisions.py
  python scripts/_apply_checklist_decisions.py --dry-run
  python scripts/_apply_checklist_decisions.py --checklist path/to/custom.csv
"""
import argparse, csv
from pathlib import Path
from collections import Counter

CHECKLIST  = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review\article_review_checklist.csv")
SCREENED   = Path("data/ontology/articles_screened.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--checklist", default=str(CHECKLIST),
                        help="Path to reviewed checklist CSV")
    args = parser.parse_args()

    checklist_path = Path(args.checklist)
    if not checklist_path.exists():
        raise FileNotFoundError(f"Checklist not found: {checklist_path}")

    # ── Load checklist decisions ───────────────────────────────────────────────
    decisions: dict[str, str] = {}   # article_id → include | exclude
    blank_count = 0
    invalid = []

    for row in csv.DictReader(open(checklist_path, encoding="utf-8-sig")):
        article_id = row.get("article_id", "").strip()
        selected   = row.get("selected", "").strip().upper()
        if not article_id:
            continue
        if selected == "Y":
            decisions[article_id] = "include"
        elif selected == "N":
            decisions[article_id] = "exclude"
        elif selected == "":
            blank_count += 1
        else:
            invalid.append((article_id, selected))

    print(f"Checklist: {checklist_path.name}")
    print(f"  selected=Y (include) : {sum(1 for v in decisions.values() if v=='include')}")
    print(f"  selected=N (exclude) : {sum(1 for v in decisions.values() if v=='exclude')}")
    print(f"  selected=blank       : {blank_count}  ← not yet reviewed")
    if invalid:
        print(f"  invalid values       : {len(invalid)}  (e.g. {invalid[:3]})")
    print()

    if blank_count > 0:
        print(f"⚠️  {blank_count} rows still blank in checklist.")
        print("   These will keep their current human_decision value in articles_screened.csv.")
        print()

    # ── Load screened CSV ──────────────────────────────────────────────────────
    rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())

    changed    = Counter()
    unchanged  = 0
    not_found  = 0

    for row in rows:
        aid = row.get("article_id", "").strip()
        if aid not in decisions:
            unchanged += 1
            continue
        new_val = decisions[aid]
        old_val = row.get("human_decision", "").strip()
        if new_val != old_val:
            changed[f"{old_val or '(empty)'} → {new_val}"] += 1
            row["human_decision"] = new_val
        else:
            unchanged += 1

    # Articles in checklist not found in screened CSV
    screened_ids = {r["article_id"] for r in rows}
    not_found = len(set(decisions) - screened_ids)

    print(f"articles_screened.csv: {len(rows)} rows")
    print(f"  Changes applied:")
    for transition, n in sorted(changed.items()):
        print(f"    {transition:<30} {n:>5}")
    total_changed = sum(changed.values())
    print(f"  Total changed  : {total_changed}")
    print(f"  Unchanged      : {unchanged}")
    if not_found:
        print(f"  Not in screened: {not_found}  (checklist has extra article_ids)")
    print()

    # Final human_decision distribution
    final = Counter(r.get("human_decision","") for r in rows)
    print(f"Final human_decision distribution:")
    for k, v in sorted(final.items()):
        print(f"  {k or '(empty)':<12} : {v:>5}")
    print()

    if args.dry_run:
        print("[dry-run] No changes written.")
        return

    if total_changed == 0:
        print("Nothing to update — checklist matches current decisions.")
        return

    # ── Write updated screened CSV ─────────────────────────────────────────────
    with open(SCREENED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"✓ Wrote {total_changed} updated decisions to {SCREENED}")

    # Rebuild article_review_checklist.csv to mark blank rows as applied
    # (Update 'selected' for any blanks that now have a decision from a prior run)
    checklist_rows = list(csv.DictReader(open(checklist_path, encoding="utf-8-sig")))
    cl_fields = list(checklist_rows[0].keys())
    screened_map = {r["article_id"]: r["human_decision"] for r in rows}
    updated_cl = 0
    for row in checklist_rows:
        aid = row.get("article_id","").strip()
        if row.get("selected","").strip() == "" and aid in screened_map:
            hd = screened_map[aid]
            if hd == "include":
                row["selected"] = "Y"
                updated_cl += 1
            elif hd == "exclude":
                row["selected"] = "N"
                updated_cl += 1

    if updated_cl:
        with open(checklist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cl_fields)
            w.writeheader()
            w.writerows(checklist_rows)
        print(f"✓ Back-filled {updated_cl} blank rows in checklist from existing decisions")


if __name__ == "__main__":
    main()
