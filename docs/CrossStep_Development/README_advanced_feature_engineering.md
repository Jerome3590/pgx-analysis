# Advanced Feature Engineering Notes

This document records feature-engineering lessons from the regenerated density-bin SHAP/FFA scenario outputs and CH3 manuscript trace.

## What we added and why

The recent pipeline update expanded the model feature space beyond raw clinical-code indicators. The goal was to separate specific clinical signals from general utilization intensity, temporal care-seeking patterns, and pharmacogenomic medication burden.

Added or emphasized feature groups:

- `n_event_bin_ordinal`: event-density stratum encoded as an ordinal utilization-intensity signal.
- `pgx_num_drugs`: total PGx-relevant medication burden.
- `pgx_num_cpic_drugs`: CPIC-actionable medication burden.
- `pgx_cpic_fraction`: proportion of PGx-relevant drugs that are CPIC-actionable.
- `pgx_non_cpic_drugs`: PGx-relevant medication burden outside CPIC-actionable subset.
- `event_span_days`: longitudinal span of pre-target events.
- `event_rate_per30`: average event rate per 30 days.
- `early_event_rate_per30`: early-window utilization rate.
- `late_event_rate_per30`: late-window utilization rate.
- `event_rate_delta_per30`: acceleration or deceleration in utilization rate.
- `event_rate_ratio_late_vs_early`: late-vs-early utilization ratio.
- `event_burstiness`: clustering/irregularity of event timing.
- `mean_inter_event_days`: average gap between events.
- `median_inter_event_days`: median gap between events.
- `std_inter_event_days`: variability in event gaps.
- `recent30_event_count`: event count near the end of the pre-target window.
- `recent90_event_count`: 90-day recent event count.
- `recent30_event_fraction`: share of events occurring in the recent 30-day window.
- `recent90_event_fraction`: share of events occurring in the recent 90-day window.

Why these were added:

1. To reduce overinterpretation of high-frequency CPT/ICD/drug codes as purely clinical effects when they may partly reflect high healthcare utilization.
2. To let the model distinguish stable chronic utilization from rapidly escalating or burst-like care patterns.
3. To expose PGx medication burden as an interpretable model signal rather than relying only on individual drug names.
4. To make downstream SHAP/FFA scenario outputs more clinically contextual and more defensible for manuscript interpretation.
5. To support density-bin modeling where low-, medium-, high-, and extreme-utilization patients can have different predictors and explanation profiles.

## How this improved the analysis

The regenerated scenario audit showed that the added contextual features improved interpretation in several ways:

- `pgx_num_drugs` appeared in the top-20 combined SHAP/FFA features in 60 of 63 bins.
- `pgx_num_cpic_drugs` appeared in the top-20 combined SHAP/FFA features in 58 of 63 bins.
- `non_opioid_ed` outputs were cleanly drug/PGx-driven, matching the intended drug-only design.
- `opioid_ed` outputs retained drug, ICD, CPT, PGx, and utilization/temporal context, matching the broader opioid feature design.
- Consensus rates were high after the XGBoost-aligned combine hardening: mean top-20 consensus was approximately 96.7% for `non_opioid_ed` and 91.6% for `opioid_ed`.
- CPT/procedure-heavy findings can now be interpreted with better utilization context, because event density and temporal dynamics are explicit model features.
- Historical rules can now be audited more rigorously: if a prior rule disappears, we can determine whether it disappeared entirely, shifted to certain bins, became SHAP-only, became weak in FFA, or requires explicit absence/delay/count features.

The main interpretive improvement is that top scenario features are no longer just a list of clinical codes. They are a combined view of medication burden, PGx burden, chronic pain/encounter context, and temporal utilization dynamics.

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

## Multi-drug profile regeneration

The current regenerated scenario outputs do not include the older full `interaction_analysis.parquet` artifacts used by CH4 to claim explicit pair/triplet interaction effects. To support manuscript and dashboard review, the model-training notebook now includes a lightweight regeneration step:

```text
8_ffa_analysis/regenerate_multidrug_interactions_from_scenario.py
```

This script reads the regenerated scenario folders and creates explicit multi-drug profile tables from:

- top combined SHAP/FFA drug features per cohort/age/bin,
- patient-level FFA rule profiles,
- strict rule support where multiple drug conditions appear in the same explanation text.

Outputs are written to:

```text
reports/scenario_audit/multidrug_interactions/
```

Key files:

- `multidrug_scenario_profiles.csv`
- `recurrent_multidrug_profiles.csv`
- `recurrent_multidrug_profiles_opioid_ed.csv`
- `recurrent_multidrug_profiles_non_opioid_ed.csv`
- `top_drug_features_by_bin.csv`
- `summary.json`

These tables show recurring pair and triplet medication profiles, including combinations such as gabapentin plus benzodiazepines in `opioid_ed` and antibiotic/corticosteroid profiles in `non_opioid_ed`.

Important limitation:

> These regenerated tables identify co-occurring top SHAP/FFA drug profiles and rule-supported medication sets. They are not a substitute for full multi-feature causal-synergy estimation unless `interaction_analysis.parquet` is regenerated with explicit `combined_causal_importance`, `sum_individual_effects`, and `interaction_effect` columns.

The notebook syncs these outputs to:

```text
s3://pgxdatalake/gold/dashboard/visualizations/scenario_multidrug_profiles/
```
