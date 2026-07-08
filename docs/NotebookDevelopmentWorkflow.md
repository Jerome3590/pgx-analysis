# Notebook Development Workflow

This project uses lightweight development notebooks to keep Cursor stable while preserving reproducible analysis artifacts. The goal is to avoid rerunning expensive analyses solely to reorganize notebooks.

## Current migration rule

Do not rerun completed analyses just to create lighter notebooks. Existing output-rich notebooks can remain as historical or published artifacts. When a notebook needs active editing again, create or move a lightweight development copy and keep the output-heavy version as an artifact.

## Recommended layout

For new notebook work inside a pipeline step directory, use:

```text
<step_dir>/notebooks/dev/
<step_dir>/notebooks/published/
<step_dir>/reports/
```

Use `notebooks/dev/` for notebooks actively edited in Cursor. Use `notebooks/published/` or `reports/` for output-rich notebooks, HTML exports, and shareable renderings.

Examples:

```text
6_final_model/notebooks/dev/
6_final_model/notebooks/published/
6_final_model/reports/

10_risk_dashboard/notebooks/dev/
10_risk_dashboard/notebooks/published/
10_risk_dashboard/reports/
```

## Existing notebooks

Existing numbered workflow notebooks do not need a mass move. Apply this rule opportunistically:

1. If the notebook is not being edited, leave it where it is.
2. If it needs active Cursor editing, create a lightweight copy under `<step_dir>/notebooks/dev/`.
3. Clear outputs in the development copy.
4. Keep the original output-rich notebook, rendered HTML, or published notebook as the reproducibility artifact.
5. Store expensive intermediate outputs as deterministic files or S3 artifacts rather than embedded notebook outputs.

## Development notebook rules

Development notebooks should:

- stay small enough to open reliably in Cursor;
- avoid large embedded tables, plots, JSON payloads, and full dataframe prints;
- write large outputs to deterministic local paths or S3;
- use `*.outputs.json` sidecars when notebook outputs are synced with `cursor_setup.py`;
- be validated structurally after scripted edits with `nbformat.validate(nb)`;
- be cleared before commit when output size causes instability or unnecessary diffs.

## Published artifact rules

Published notebooks and reports may contain outputs, but they should be treated as generated artifacts unless intentionally tracked for a manuscript, supplement, or dashboard audit.

Prefer these artifact locations:

```text
<step_dir>/notebooks/published/
<step_dir>/reports/
<step_dir>/outputs/
```

Do not open output-heavy published notebooks in Cursor during normal development.

## Useful commands

Inspect notebook/output pointer status:

```bash
python cursor_setup.py status
```

Push notebook outputs to S3 and write a sidecar pointer:

```bash
python cursor_setup.py push-outputs <notebook_path>
```

Fetch synced notebook outputs:

```bash
python cursor_setup.py fetch-outputs <notebook_path>
```

Clear notebook outputs without running analysis:

```bash
jupyter nbconvert --clear-output --inplace <notebook_path>
```

## Cursor recovery pattern

If Cursor starts hanging, showing blank notebook tabs, or repeatedly reloading:

1. Close output-heavy notebooks.
2. Work from a lightweight `notebooks/dev/` copy.
3. Clear outputs in the development copy without rerunning analysis.
4. Restart the kernel only if needed.
5. Use paired Python scripts or `# %%` files for long-running production-style workflow edits.
