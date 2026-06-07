# CH3 Physical-Therapy Rule Trace vs Regenerated Scenario Outputs

## Prior CH3 claims located

The CH3 manuscript source (CH_3/ch03_cts.qmd) contains explicit physical-therapy related claims:

- Line ~420: "Physical therapy absence (protective; mean |SHAP| = 0.11): absence of CPT 97110/97530 elevates risk..."
- Lines ~425-427: example rule: IF oxycodone count >= 2 AND ICD-10 M54.5 low back pain AND NOT CPT 97110/97530 physical therapy >= 1, THEN high OUD-ED risk.
- Lines ~560-565: missed care opportunities / absence of physical therapy or non-opioid alternatives.
- Line ~571: 25-44 band driven by oxycodone-gabapentin co-prescription with low back pain and physical therapy absence.
- Later discussion frames PT, CBT, and non-opioid analgesics as intervention-window additions, while explicitly preserving association-vs-causation limitations.

## Current regenerated scenario evidence

Current synced scenario outputs from s3://pgxdatalake/gold/dashboard/visualizations/scenario/ contain:

- 63 per-bin combined SHAP/FFA outputs.
- 124 PT-like CPT rows across all ranks.
- 16 bins with any PT-like CPT row.
- Only 3 PT-like rows in the top-20 combined SHAP/FFA features.
- PT-like rows are restricted to older opioid_ed strata: 55_64, 65_74, 75_84, and 85_114.
- Best PT-like rows are mostly CPT presence features, especially item_cpt_97110, item_cpt_97112, item_cpt_97162, item_cpt_97140, and item_cpt_97530.

## Why the old rule is no longer directly supported

1. The regenerated feature matrix encodes CPT presence (item_cpt_97110, item_cpt_97530) rather than explicit absence (no_physical_therapy). The old manuscript rule relied on a negated condition: NOT physical therapy >= 1.

2. Current combined importance is consensus-ranked across XGBoost SHAP and XGBoost FFA. PT features are present, but their FFA scores are generally much lower than their SHAP scores, so they rarely survive into top-20 consensus outputs.

3. Event-density binning splits the earlier global cohort signal into low/medium/high/extreme strata. The strongest current PT signal appears in older opioid strata rather than the prior emphasized 25-44 stratum.

4. The current top consensus features are dominated by PGx burden, drugs, chronic-pain ICDs, and encounter CPTs. These stronger features displace PT absence from the top consensus interpretation.

5. The hardened combine path now requires XGBoost-aligned SHAP and FFA inputs. If the earlier PT rule came from a hybrid/intermediate CatBoost SHAP or broader manuscript exploratory table, it is intentionally not part of the current strict XGBoost FFA scenario output.

## Current defensible wording

The regenerated scenario outputs do not support the prior strong claim that absence of physical therapy is a leading Consensus-Causal driver. They support a narrower statement:

> Physical-therapy-related CPT codes are detectable as secondary opioid_ed signals in older age strata, but they are not consistently top-ranked SHAP/FFA consensus features. The current feature representation captures PT code presence rather than explicit PT absence, so the prior NOT CPT 97110/97530 rule should be treated as an exploratory/historical rule unless explicit absence-of-PT features are regenerated and validated.

## Recommended manuscript action

- Replace strong "physical therapy absence elevates risk" language with the narrower secondary-signal wording above.
- If retaining the old rule, label it as an exploratory prior analysis not reproduced in the regenerated density-bin SHAP/FFA outputs.
- For a future rerun, add explicit features such as has_physical_therapy, no_physical_therapy, pt_visit_count, and days_to_first_pt before Step 6, then rerun Step 6-8 and Combine.
