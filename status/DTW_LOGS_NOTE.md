# DTW logs in `status/logs/`

## Expected warnings (drug-name-only cohorts)

For **non_opioid_ed 65-74** and **non_opioid_ed 75-84**, DTW logs may show:

- `CSV empty or missing seq_pattern_str; skipping alignment.`
- `DTW alignment skipped (empty or invalid trajectories); exiting 0 so pipeline continues.`

This is **expected** when the cohort/age_band has only drug names (prescription claims) and no ICD/CPT trajectories in the trajectory CSV. The pipeline exits 0 so other steps and cohorts are unaffected. See `9_dashboard_visuals/dtw/README_DTW_COHORT_ANALYSIS.md` (§ Empty alignment).

## Errors

If a log shows **Errors: 1** (or non-zero) in the PIPELINE STEP SUMMARY, treat that as a real failure and investigate.
