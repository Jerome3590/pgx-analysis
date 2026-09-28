"""Resolve Drive view URLs for PGx UC01–UC09 (README + screenshots only).

Does not invent file IDs. Sources, in order:
1. Google Drive for desktop metadata (DriveFS) for the jerome.dixon90 / G: tree
2. notebookLM-automation Drive API (account jerome_dixon90) if OAuth files exist
3. Otherwise exit and print the one-time gdrive list command

Never prints OAuth tokens.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO_ROOT / "10_risk_dashboard" / "docs" / "use_case_training" / "notebooklm_drive_urls.json"
DRIVE_PACK = Path(r"G:\My Drive\PGx_Dashboard_Use_Cases")
DRIVEFS_ROOT = Path.home() / "AppData" / "Local" / "Google" / "DriveFS"
NLM_ACCOUNTS = Path(r"C:\Projects\notebookLM-automation\config\gdrive_accounts.yaml")
PACK_FOLDER = "PGx_Dashboard_Use_Cases"

# Personal Gmail DriveFS account that contains the G: pack (small tree, 2k items).
# VCU/CANA copies exist under other DriveFS accounts; do not use those as primary.
GMAIL_DRIVEFS_ACCOUNT = "110518936715501675447"

SKIP_NAMES = {
    "NOTEBOOKLM_DRIVE_IMPORT.md",
    "NOTEBOOKLM_NEXT_CLICK.md",
    "NOTEBOOKLM_PROMPTS.md",
    "NOTEBOOKLM_URL_PASTE.md",
    "CHROME_DEBUG.md",
}

UCS = {
    "UC01_cohort_risk": "PGx UC01 — Cohort risk",
    "UC02_scenario_analysis": "PGx UC02 — Scenario analysis",
    "UC03_density_bin_exploration": "PGx UC03 — Density bin exploration",
    "UC04_feature_importance": "PGx UC04 — Feature importance",
    "UC05_pattern_process": "PGx UC05 — Pattern and process",
    "UC06_claims_pgx_card": "PGx UC06 — Claims PGx card",
    "UC07_personalized_pgx_card": "PGx UC07 — Personalized PGx card",
    "UC08_cohort_vs_card": "PGx UC08 — Cohort vs card",
    "UC09_documentation": "PGx UC09 — Documentation",
}


def view_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def open_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)


def children(con: sqlite3.Connection, parent_stable_id: int) -> list[tuple]:
    return con.execute(
        """
        SELECT i.id, i.stable_id, i.local_title, i.mime_type, i.is_folder, i.trashed
        FROM stable_parents sp
        JOIN items i ON i.stable_id = sp.item_stable_id
        WHERE sp.parent_stable_id = ?
        ORDER BY i.local_title
        """,
        (parent_stable_id,),
    ).fetchall()


def wanted_file(rel_parts: list[str], name: str, is_folder: bool) -> bool:
    if is_folder or name in SKIP_NAMES:
        return False
    if "notebooklm" in rel_parts:
        return False
    if name == "README.md":
        return True
    if len(rel_parts) >= 2 and rel_parts[-1] == "screenshots" and name.lower().endswith(".png"):
        return True
    return False


def resolve_drivefs(account: str) -> dict:
    db = DRIVEFS_ROOT / account / "metadata_sqlite_db"
    if not db.exists():
        raise FileNotFoundError(f"DriveFS metadata missing: {db}")
    con = open_ro(db)
    try:
        roots = con.execute(
            """
            SELECT id, stable_id, local_title
            FROM items
            WHERE local_title = ? AND is_folder = 1 AND trashed = 0
            """,
            (PACK_FOLDER,),
        ).fetchall()
        if not roots:
            raise RuntimeError(f"Folder {PACK_FOLDER!r} not in DriveFS account {account}")
        if len(roots) > 1:
            print(f"WARN multiple {PACK_FOLDER} folders; using first id={roots[0][0]}", file=sys.stderr)
        root_id, root_stable, _ = roots[0]
        files: list[dict] = []
        stack: list[tuple[int, list[str]]] = [(root_stable, [])]
        while stack:
            parent_stable, rel = stack.pop()
            for file_id, stable_id, title, mime, is_folder, trashed in children(con, parent_stable):
                if trashed or not title:
                    continue
                if str(file_id).startswith("local-"):
                    continue
                child_rel = rel + [title]
                if is_folder:
                    if title == "notebooklm":
                        continue
                    stack.append((stable_id, child_rel))
                    continue
                if wanted_file(rel, title, False):
                    files.append(
                        {
                            "rel": "/".join(child_rel),
                            "name": title,
                            "id": file_id,
                            "url": view_url(file_id),
                            "mime": mime,
                        }
                    )
        return {
            "source": "drivefs",
            "drivefs_account": account,
            "root_folder_id": root_id,
            "files": files,
        }
    finally:
        con.close()


def resolve_drive_api() -> dict:
    sys.path.insert(0, str(Path(r"C:\Projects\notebookLM-automation\src")))
    from notebooklm_automation.gdrive import (  # type: ignore
        FOLDER_MIME_TYPE,
        build_drive_service,
        get_account,
        list_folder_files,
    )

    account = get_account(NLM_ACCOUNTS, "jerome_dixon90")
    if not account.token.exists() or not account.client_secrets.exists():
        raise FileNotFoundError(
            "notebookLM-automation OAuth files for jerome_dixon90 are missing "
            f"(token={account.token.exists()} secrets={account.client_secrets.exists()})"
        )
    service = build_drive_service(account)
    resp = (
        service.files()
        .list(
            q=(
                f"name = '{PACK_FOLDER}' and mimeType = '{FOLDER_MIME_TYPE}' "
                "and trashed = false"
            ),
            fields="files(id, name)",
            pageSize=10,
        )
        .execute()
    )
    folders = resp.get("files") or []
    if not folders:
        raise RuntimeError(f"Drive API did not find folder {PACK_FOLDER!r}")
    root_id = folders[0]["id"]
    files: list[dict] = []

    def walk(folder_id: str, rel: list[str]) -> None:
        for item in list_folder_files(service, folder_id):
            name = item["name"]
            mime = item["mimeType"]
            if mime == FOLDER_MIME_TYPE:
                if name == "notebooklm":
                    continue
                walk(item["id"], rel + [name])
                continue
            if wanted_file(rel, name, False):
                files.append(
                    {
                        "rel": "/".join(rel + [name]),
                        "name": name,
                        "id": item["id"],
                        "url": view_url(item["id"]),
                        "mime": mime,
                    }
                )

    walk(root_id, [])
    return {
        "source": "drive_api",
        "account": "jerome_dixon90",
        "root_folder_id": root_id,
        "files": files,
    }


def group_ucs(files: list[dict]) -> list[dict]:
    by_uc: dict[str, list[dict]] = {uc: [] for uc in UCS}
    extras: list[dict] = []
    for item in files:
        parts = item["rel"].split("/")
        uc = parts[0] if parts else ""
        if uc in by_uc:
            by_uc[uc].append(item)
        else:
            extras.append(item)
    notebooks = []
    for uc, title in UCS.items():
        items = sorted(by_uc[uc], key=lambda x: x["rel"])
        notebooks.append(
            {
                "uc": uc,
                "title": title,
                "file_count": len(items),
                "files": items,
            }
        )
    return notebooks, extras


def expected_from_g() -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    if not DRIVE_PACK.exists():
        return expected
    for uc in UCS:
        paths: list[str] = []
        readme = DRIVE_PACK / uc / "README.md"
        if readme.exists():
            paths.append(f"{uc}/README.md")
        shots = DRIVE_PACK / uc / "screenshots"
        if shots.is_dir():
            for png in sorted(shots.glob("*.png")):
                paths.append(f"{uc}/screenshots/{png.name}")
        expected[uc] = paths
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Write notebooklm_drive_urls.json from real Drive IDs.")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--source",
        choices=("auto", "drivefs", "api"),
        default="auto",
        help="auto tries DriveFS then Drive API",
    )
    parser.add_argument("--drivefs-account", default=GMAIL_DRIVEFS_ACCOUNT)
    args = parser.parse_args()

    resolved = None
    errors: list[str] = []
    order = ("drivefs", "api") if args.source == "auto" else (args.source,)
    for source in order:
        try:
            if source == "drivefs":
                resolved = resolve_drivefs(args.drivefs_account)
            else:
                resolved = resolve_drive_api()
            break
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")

    if resolved is None:
        print("Could not resolve Drive IDs. Tried:")
        for err in errors:
            print(" -", err)
        print()
        print("One-time API list (after OAuth files exist for jerome_dixon90):")
        print("  cd C:\\Projects\\notebookLM-automation")
        print("  .\\.venv\\Scripts\\python.exe -c \"from notebooklm_automation.gdrive import get_account, get_credentials; get_credentials(get_account('config/gdrive_accounts.yaml','jerome_dixon90'))\"")
        print("  cd C:\\Projects\\pgx-analysis")
        print("  .\\.venv\\Scripts\\python.exe utility_scripts\\resolve_pgx_uc_drive_urls.py --source api")
        print()
        print("Do not invent file IDs. Re-run this resolver after DriveFS sync or OAuth.")
        return 2

    notebooks, extras = group_ucs(resolved["files"])
    expected = expected_from_g()
    missing: list[str] = []
    extra_rels = {item["rel"] for item in resolved["files"]}
    for uc, paths in expected.items():
        for rel in paths:
            if rel not in extra_rels:
                missing.append(rel)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account_email": "jerome.dixon90@gmail.com",
        "pack": PACK_FOLDER,
        "url_kind": "drive_file_view",
        "skip": sorted(SKIP_NAMES | {"notebooklm/"}),
        **{k: v for k, v in resolved.items() if k != "files"},
        "notebooks": notebooks,
        "unassigned": extras,
        "missing_vs_g_pack": missing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"source={payload.get('source')} root={payload.get('root_folder_id')}")
    for nb in notebooks:
        print(f"  {nb['uc']}: {nb['file_count']} files")
    if missing:
        print(f"MISSING vs G: pack ({len(missing)}):")
        for rel in missing:
            print(" -", rel)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
