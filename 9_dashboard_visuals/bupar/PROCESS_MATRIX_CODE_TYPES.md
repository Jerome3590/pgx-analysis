# Process Matrix and Code Types (Drug, ICD, CPT)

## How the process matrix handles code types

The BupaR event log has a **single `activity` column** with values prefixed by type:

- `DRUG:...` (e.g. `DRUG:HYDROCODONE`)
- `ICD:...` (e.g. `ICD:F1120`)
- `CPT:...` (e.g. `CPT:99213`)

[bupaR Process Matrix](https://bupaverse.github.io/docs/process_matrix.html) computes **directly-follows** flows: for each pair (antecedent → consequent) it counts how often that pair appears in sequence. It does **not** natively separate by code type. So:

- The **current single process matrix** is **all activities × all activities**: it includes Drug→Drug, Drug→ICD, Drug→CPT, ICD→Drug, ICD→ICD, ICD→CPT, CPT→Drug, CPT→ICD, CPT→CPT in one heatmap.
- Code type is only implied by the activity label prefix (`DRUG:`, `ICD:`, `CPT:`). The matrix does not “handle” types separately unless we filter.

## Do we need Drug×Drug, Drug×ICD, Drug×CPT?

**It depends on the analysis question:**

| Matrix | Question it answers |
|--------|---------------------|
| **Combined (current)** | What are the most common flows overall? (all types mixed) |
| **Drug × Drug** | Which drug tends to follow which drug? (prescribing sequences) |
| **Drug × ICD** | Which diagnoses tend to follow which drugs? Or which drugs precede which diagnoses? |
| **Drug × CPT** | Which procedures follow which drugs? |
| **ICD × ICD** | Which diagnosis tends to follow which? |
| **ICD × CPT** | Which procedure follows which diagnosis? |
| **CPT × CPT** | Which procedure follows which procedure? |

**Recommendation:**

- **Keep the single combined process matrix** for the main dashboard (overview of all flows).
- **Optionally add type-pair process matrices** (Drug×Drug, Drug×ICD, Drug×CPT, etc.) if you want to focus on specific code-type transitions. These can be produced by filtering the same `process_matrix()` result by antecedent/consequent prefix and saving separate PNGs (and optionally exposing them in the dashboard via a dropdown or extra panels).

## Implementation

- **Combined matrix:** One process matrix from the full event log → `{cohort}_{age_band}_process_matrix.png` (used on the main dashboard).
- **Type-pair (production):** The R scripts save only **Drug × Drug**: `{cohort}_{age_band}_process_matrix_drug_drug.png`. Filter is antecedent/consequent prefix `DRUG:` / `DRUG:`. This is the only type-pair produced for the final production pipeline (research-question focus on drug sequences).
