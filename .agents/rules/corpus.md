---
description: Corpus-first lookup and local utility reuse for pgx-analysis
alwaysApply: true
---

# Corpus-First Lookup

Before generating code or answers in this repository:

1. Search local code, `CLAUDE.md`, `py_helpers/`, and `utility_scripts/` for existing patterns.
2. Prefer established S3, path, cohort, age-band, and dashboard conventions over new abstractions.
3. Do not duplicate utilities that already exist in `py_helpers/`, step directories, or `utility_scripts/`.
4. Treat external APIs as a fallback only when local project context is insufficient.
