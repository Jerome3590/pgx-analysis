Promotion plan for sandbox notebook calls and scripts

Summary:
- The `1_apcd_input_data/sandbox/notebook_calls.md` contains runnable notebook cells used during development. We archived a snapshot under `1_apcd_input_data/sandbox_archived_20251110_000000`.
- This file (`docs/promote_sandbox_notes.md`) records the proposed actions to promote sandbox content to canonical docs and the checklist for doing so safely.

Checklist:
1. Review sandbox cells and identify canonical cells to be included in `docs/notebook_calls.md`.
2. Remove any environment-specific absolute paths (e.g., `/home/pgx3874/`) and replace with relative or templated variables.
3. Ensure commands that require AWS credentials or NVMe are annotated with a note about required environment setup.
4. Update any references to local staging paths to use variables like `$PGX_LOCAL_STAGING_DIR`.
5. Create a small smoke test section at the end with minimal commands that can run in a dev environment without S3 access (use local sample data).
6. Run linter/markdown checks and commit.

Notes:
- I can proceed and apply these changes (copy sanitized cells into `docs/notebook_calls.md`) if you want me to. This file is a short plan and checklist only.
