---
description: Notebook output access and indexing rules for pgx-analysis
alwaysApply: true
---

# Notebook Output Access

## Project Notebook Metadata

- Project slug: `pgx-analysis`
- Default notebook metadata bucket: `mushin-solutions-project-metadata`
- Default notebook metadata prefix: `notebooks`
- Full notebook metadata root: `s3://mushin-solutions-project-metadata/notebooks/pgx-analysis/`
- Use `python cursor_setup.py push-outputs <notebook>` and `python cursor_setup.py fetch-outputs <notebook>` without repeating bucket or slug flags unless intentionally overriding these defaults.

This repository contains many large workflow notebooks. Do not rely on raw notebook JSON as primary context. Follow `docs/NotebookDevelopmentWorkflow.md` for the **final production** notebook workflow (script-first, artifacts on disk/S3). Standing rule: document final production paths and remove intermediates — see `docs/CrossStep_Development/README_lessons_learned.md`.

- Prefer paired `# %%` Python workflow scripts and step `run_*.py` over ballooning exploratory notebooks for production / manuscript work.
- Do not rerun completed analyses just to reorganize notebooks.
- When asked about notebook results, look for a sidecar `*.outputs.json` pointer next to the notebook.
- Fetch synced outputs with `python cursor_setup.py fetch-outputs <notebook_path>`.
- Use `python cursor_setup.py status` to inventory notebooks and pointer files.
- Do not clear or strip outputs unless the user explicitly asks or a commit workflow requires it.
- Do not rely on inline notebook output as a durable cache; write expensive intermediates to Parquet, pickle, Feather, or S3-backed artifacts and load them explicitly on later runs.
- For any notebook / percent-script that will be executed, include the setup cell from `docs/NotebookDevelopmentWorkflow.md` and use `py_helpers.notebook_artifacts` so GitHub-trackable artifacts, local outputs, and S3 artifact prefixes are deterministic.

**Cursor blank/frozen notebook tabs — confirmed root causes** (canonical docs in shared repo utils; **not** a `notebooks/dev` vs `published` layout problem):

1. Missing `[jupyter]` workspace settings (disable Cursor Tab/CPP/format-on-save on notebooks) — see `.vscode/settings.json`
2. CRLF + `.gitattributes` / broken Windows `nbstripout` filter corrupting `.ipynb` JSON
3. Conflicting Python/Jupyter extensions (duplicate env managers; prefer `ms-python.python:venv` + Pylance)

Full write-up: `C:\Projects\project_utility_scripts\CURSOR_DEV_RULES.md` → Confirmed Cursor notebook crash causes. Short: that repo’s `DEV_RULES.md` → Notebook Defaults. Project lessons: `docs/CrossStep_Development/README_lessons_learned.md` → Cursor Notebook Stability.

## Final Notebook Editing Workflow

When editing `*.ipynb` files in this repo:

- Prefer `nbformat` structural edits for targeted cell insert/update work; avoid broad JSON/text rewrites.
- Use a temporary Python script from PowerShell here-strings for complex notebook edits. Do not use Unix heredoc syntax (`python - <<'PY'`) in PowerShell.
- Keep edits targeted to known cell indices/headers. After modifying, run `nbformat.validate(nb)` before writing or before commit.
- Normalize legacy notebook outputs before validation when needed:
  - stream outputs require `name` (usually `stdout`);
  - display/execute/error outputs require `metadata`.
- Preserve this repo's older notebook schema when present: strip generated cell `id` fields if validation reports `id` as an unexpected property.
- Do not execute or clear long-running workflow notebooks just to edit markdown/code guidance unless explicitly requested.
- Avoid printing large dataframes, large JSON payloads, or full artifact contents directly into cells; save tables and plots to disk/S3 instead.
- If Cursor notebook tabs become blank, frozen, or unstable, triage settings / CRLF / `nbstripout` / extensions first; then clear outputs, save, close/reopen, restart the kernel, and rerun only the minimal cells needed.
- For production-style or long-running work, prefer paired `# %%` Python scripts or step workflow scripts.

## Publishing and Artifact Pattern

- Keep committed notebooks lean; ship numbers via CSVs/Parquet/JSON under step `outputs/` or `reports/notebook_artifacts/`.
- Existing numbered workflow notebooks do not require mass migration.
- Treat HTML reports, `.nbcache/`, `.jupyter_cache/`, and `.ipynb_checkpoints/` as generated artifacts unless explicitly designated as source.
- When a rendered artifact is needed, execute from a clean environment and publish outputs separately rather than using notebook JSON as the artifact store.
- For repeated expensive computations: compute once, write the artifact to S3 or a deterministic local output path, check for that artifact first on later runs, and recompute only if missing or intentionally invalidated.
- After a protocol ships, remove or clearly mark abandoned intermediate runners — do not leave dual “canonical” paths.

## Deployment Lessons Learned

- Dashboard notebook S3 links should use path-style URLs under `https://s3.us-east-1.amazonaws.com/{bucket}/{prefix}/...` and should be verified with public `HEAD`/`GET`.
- For dashboard visual updates, sync frontend files and `visualizations/dashboard_visual_objects.json`, then invalidate CloudFront.
- Lambda S3 code override may fail if the image entrypoint cannot write to `/var/task`; for backend handler changes, rebuild and push a Lambda-compatible ECR image.
- For Lambda ECR images built with Docker BuildKit/buildx, use Lambda-compatible image output (`--provenance=false --sbom=false`, `linux/amd64`) before `update-function-code`.
