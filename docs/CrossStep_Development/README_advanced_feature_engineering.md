# Advanced Feature Engineering Notes

This document records feature-engineering lessons from the regenerated density-bin SHAP/FFA scenario outputs and CH3 manuscript trace.

## Why explicit absence features matter

The current model matrix primarily represents clinical-code presence:

- `item_cpt_97110`
- `item_cpt_97530`
- `item_drug_GABAPENTIN`
- `item_icd_M545`

These features can tell the model that an event/code occurred. They do not directly encode clinically meaningful non-occurrence such as:

- no physical therapy
- no non-opioid analgesic alternative
- no follow-up within a time window
- delayed physical therapy initiation

A rule such as:

```text
IF oxycodone count >= 2
AND ICD-10 M54.5 low back pain
AND NOT CPT 97110/97530 physical therapy >= 1
THEN high opioid_ed risk
```

requires an explicit negated or absence-derived representation. If only positive CPT indicators exist, SHAP/FFA may identify physical-therapy-related utilization, but the pipeline cannot robustly claim that absence of physical therapy is a leading driver.

## CH3 physical-therapy rule trace

A previous CH3 manuscript draft included a strong physical-therapy absence claim:

- Physical therapy absence was described as protective/risk-related.
- The example rule used `NOT CPT 97110/97530 physical therapy >= 1`.
- The 25-44 opioid cohort was described as driven partly by oxycodone/gabapentin, low back pain, and physical therapy absence.

The regenerated density-bin scenario outputs do not support that strong claim as a top consensus finding.

Current regenerated scenario evidence:

- 63 per-bin combined SHAP/FFA outputs were analyzed.
- 124 PT-like CPT rows appeared across all ranks.
- 16 bins contained any PT-like CPT row.
- Only 3 PT-like rows appeared in top-20 combined SHAP/FFA features.
- PT-like rows were restricted to older `opioid_ed` strata: `55_64`, `65_74`, `75_84`, and `85_114`.
- The strongest PT-like rows were presence features such as `item_cpt_97110`, `item_cpt_97112`, `item_cpt_97162`, `item_cpt_97140`, and `item_cpt_97530`.

Therefore, current defensible interpretation is:

> Physical-therapy-related CPT codes are detectable as secondary `opioid_ed` signals in older age strata, but they are not consistently top-ranked SHAP/FFA consensus features. The current feature representation captures PT code presence rather than explicit PT absence, so prior `NOT CPT 97110/97530` rules should be treated as exploratory/historical unless explicit absence-of-PT features are regenerated and validated.

## Recommended future PT-derived features

For a future rerun, add explicit physical-therapy features before Step 6 final model training:

- `has_physical_therapy`
- `no_physical_therapy`
- `pt_visit_count`
- `pt_started_within_30d`
- `pt_started_within_90d`
- `days_to_first_pt`
- `pt_visits_per30`
- `recent_pt_count`

These should be derived from pre-target CPT events only and should respect the same temporal leakage rules as other model features.

## Recommended non-opioid-care pathway features

The same design applies to other hypothesized care-pathway gaps:

- `has_non_opioid_analgesic`
- `no_non_opioid_analgesic`
- `days_to_first_non_opioid_analgesic`
- `has_behavioral_health_followup`
- `no_behavioral_health_followup`
- `days_to_first_behavioral_health_followup`
- `followup_visit_count_30d`
- `followup_visit_count_90d`

These features are better suited to testing care-pathway addition hypotheses than relying on the absence of individual positive code indicators.

## Interaction with density bins

The regenerated pipeline trains and explains models within event-density strata:

- `low`
- `medium`
- `high`
- `extreme`

A feature that was globally important in an earlier full-cohort or hybrid analysis may become diluted or redistributed across bins. When interpreting dropped rules, check whether the signal:

1. disappeared entirely,
2. remains present below top-k thresholds,
3. shifted to specific age bands or density bins,
4. is SHAP-visible but weak in FFA, or
5. requires an explicit absence/delay/count feature.

## Interaction with SHAP/FFA consensus

The current scenario combine path requires XGBoost-aligned inputs:

- XGBoost SHAP global importance
- XGBoost SHAP sample values
- XGBoost FFA explanations
- XGBoost FFA importance

This makes scenario outputs stricter and more internally consistent. However, it can drop exploratory findings that were previously driven by:

- CatBoost-only SHAP signals,
- hybrid/intermediate outputs,
- global rather than density-bin models,
- positive-code proxies for absence rules, or
- features that are SHAP-important but not strongly represented in XGBoost symbolic rules.

## Audit artifacts

The current scenario audit is stored in:

```text
reports/scenario_audit/
```

Key files:

- `README.md`
- `CH3_PT_RULE_TRACE.md`
- `all_combined_features_ranked.csv`
- `top20_combined_features.csv`
- `top20_feature_recurrence.csv`
- `top20_feature_family_by_cohort.csv`
- `pt_like_features_all_ranks.csv`
- `consensus_summary_by_bin.csv`

Use these artifacts when updating manuscript claims or deciding whether a historical rule remains supported by regenerated outputs.
