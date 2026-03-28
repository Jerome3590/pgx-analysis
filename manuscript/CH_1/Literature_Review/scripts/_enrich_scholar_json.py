"""
_enrich_scholar_json.py
────────────────────────────────────────────────────────────────────────
Embed all classification metadata from articles_screened.csv into each
scholar_json document under a top-level "classifications" key.

Fields written to JSON["classifications"]:
  composite_score    pytextrank_score   combined_score
  ooda_phase         crisp_dm_phase     ooda_crisp_label
  nih_ai_score       nih_ai_pct         nih_ai_tags        (list)
  op_perf_score      op_perf_tags       (list)
  human_decision

Matching logic:
  1. scholar_json filename stem  →  article_id  (e.g. HSH0b7386ff)
  2. json["pmc_id"]              →  pmc_id      (e.g. PMC12345678)
  3. json["id"]                  →  article_id  (fallback)

Idempotent — safe to re-run; overwrites "classifications" block only.

Usage:
  python scripts/_enrich_scholar_json.py
  python scripts/_enrich_scholar_json.py --dry-run
"""
import argparse, csv, json
from pathlib import Path

SCREENED     = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON = Path("data/scholar_json")

_NUM_FIELDS  = {"composite_score","pytextrank_score","combined_score",
                "nih_ai_score","nih_ai_pct","op_perf_score"}
_LIST_FIELDS = {"nih_ai_tags","op_perf_tags"}
_STR_FIELDS  = {"ooda_phase_primary","crisp_dm_phase","ooda_crisp_label","human_decision"}

_RENAME = {"ooda_phase_primary": "ooda_phase"}


def _parse_screened() -> tuple[dict, dict, dict]:
    """Return (by_article_id, by_pmc_full, by_pmc_bare) lookup dicts.

    articles_screened uses PMC-prefixed IDs (e.g. 'PMC10002439').
    scholar_json files store bare numeric IDs (e.g. '10002439').
    We build both so either form matches.
    """
    by_aid      = {}
    by_pmc_full = {}
    by_pmc_bare = {}
    rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))
    for row in rows:
        aid = row.get("article_id", "").strip()
        pid = row.get("pmc_id", "").strip()
        rec = {}
        for f in _STR_FIELDS:
            key = _RENAME.get(f, f)
            rec[key] = row.get(f, "") or ""
        for f in _NUM_FIELDS:
            raw = row.get(f, "") or ""
            try:
                rec[f] = float(raw) if raw else None
            except ValueError:
                rec[f] = None
        for f in _LIST_FIELDS:
            raw = row.get(f, "") or ""
            rec[f] = [t.strip() for t in raw.split("|") if t.strip()]
        if aid:
            by_aid[aid] = rec
        if pid:
            by_pmc_full[pid] = rec
            bare = pid.lstrip("PMCpmc").lstrip("0") or pid  # "PMC10002439" → "10002439"
            bare_raw = pid[3:] if pid.upper().startswith("PMC") else pid
            by_pmc_bare[bare_raw] = rec
    return by_aid, by_pmc_full, by_pmc_bare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    by_aid, by_pmc_full, by_pmc_bare = _parse_screened()
    files = sorted(SCHOLAR_JSON.glob("*.json"))
    print(f"scholar_json/ files  : {len(files):,}")
    print(f"articles_screened rows: {len(by_aid):,}  ({len(by_pmc_full):,} with PMC ID)\n")

    updated = skipped = no_match = 0

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP (read error) {path.name}: {e}")
            skipped += 1
            continue

        stem    = path.stem                          # e.g. "PMC10002439" or "HSH0b7386ff"
        pmc_id  = (doc.get("pmc_id") or "").strip() # e.g. "10002439" (bare) or ""
        doc_id  = (doc.get("id") or "").strip()      # same as pmc_id for PMC files

        rec = (by_aid.get(stem)          # HSH articles: stem == article_id
               or by_pmc_full.get(stem)  # if stem is "PMC10002439" form
               or by_pmc_bare.get(stem.lstrip("PMCpmc"))   # strip prefix from stem
               or by_pmc_bare.get(pmc_id)                  # bare numeric from JSON field
               or by_pmc_bare.get(doc_id)                  # bare numeric id field
               or by_aid.get(doc_id))

        if rec is None:
            no_match += 1
            continue

        # Skip if classifications block already matches — no rework
        existing = doc.get("classifications", {})
        if (existing.get("human_decision") == rec.get("human_decision") and
                existing.get("pytextrank_score") == rec.get("pytextrank_score") and
                existing.get("composite_score") == rec.get("composite_score")):
            skipped += 1
            continue

        doc["classifications"] = rec

        if not args.dry_run:
            path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        updated += 1

    print(f"Updated  : {updated:,}")
    print(f"Unchanged: {skipped:,}  (classifications already current — skipped)")
    print(f"No match : {no_match:,}  (JSON not in articles_screened)")
    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        print(f"\n✓ {updated:,} scholar_json files enriched with 'classifications' block.")


if __name__ == "__main__":
    main()
