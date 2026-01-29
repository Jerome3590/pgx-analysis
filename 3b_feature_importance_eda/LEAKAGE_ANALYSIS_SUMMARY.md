# Post-F1120 Target Leakage Analysis Summary

## Overview

**348 features** (out of 1,029 total) have a **post-F1120 ratio ≥ 80%**, indicating they represent target leakage. These features capture events that occur **after** the F1120 (opioid dependence) diagnosis, which would not be available at prediction time.

## Distribution by Feature Type

- **Drugs**: 164 features (47.1%)
- **ICD Codes**: 100 features (28.7%)
- **CPT Codes**: 84 features (24.1%)

## Why These Features Represent Target Leakage

### 1. **Treatment Medications** (Post-Diagnosis Interventions)

These are medications prescribed **after** the F1120 diagnosis as part of treatment:

**Top Opioid Treatment Medications:**
- `SUBOXONE`: 6,082 post-F1120 events (100% post)
- `BUPRENORPHINE_HCL/NALOXON`: 3,032 post-F1120 events (100% post)
- `BUPRENORPHINE_HCL`: 2,562 post-F1120 events (100% post)
- `ZUBSOLV`: 400 post-F1120 events (100% post)
- `VIVITROL`: 106 post-F1120 events (100% post)
- `NARCAN`: 271 post-F1120 events (100% post)

**Why this is leakage:**
- These medications are prescribed **because** the patient was diagnosed with opioid dependence
- They are **consequences** of the diagnosis, not predictors
- At prediction time, we wouldn't know if a patient will receive these treatments

**Other Treatment-Related Drugs:**
- `BUPROPION_HYDROCHLORIDE_E`: 670 post-F1120 events (often used for depression/anxiety in addiction treatment)
- `BACLOFEN`: 127 post-F1120 events (muscle relaxant, sometimes used in addiction treatment)
- `CLONIDINE_HCL`: 75 post-F1120 events (used for withdrawal symptoms)

### 2. **Drug Screening Tests** (Post-Diagnosis Monitoring)

**CPT Code 80307** (Presumptive drug class screening): **3,636 post-F1120 events (100% post)**

**Why this is leakage:**
- Drug screening is ordered **after** diagnosis to monitor treatment compliance
- This is a **monitoring** activity, not a predictive signal
- The test wouldn't exist before the diagnosis

**Other Drug Screening Codes:**
- `80348`: 7 post-F1120 events
- `80362`: 6 post-F1120 events

### 3. **Mental Health Diagnoses** (Comorbidities/Complications)

**ICD Codes with F3 prefix** (Mental and behavioral disorders): **14 codes**

Examples:
- `F3341`: 89 post-F1120 events (Persistent depressive disorder)
- `F200`: 47 post-F1120 events (Schizophrenia)
- `F3110`: 36 post-F1120 events (Bipolar disorder, current episode manic)
- `F320`: 10 post-F1120 events (Major depressive disorder, single episode)
- `F349`: 13 post-F1120 events (Unspecified mood disorder)

**Why this is leakage:**
- These may be **comorbidities** diagnosed during treatment
- Or **complications** that arise after opioid dependence
- They represent downstream effects, not predictive signals

### 4. **Opioid-Related Diagnoses** (Recurring/Related Diagnoses)

**ICD Code F1120** itself: **57,377 post-F1120 events (100% post)**

**Why this is leakage:**
- This is the **target diagnosis itself** appearing again after the initial diagnosis
- Using it as a feature would be directly using the target to predict the target

**Other Opioid-Related Codes:**
- `F11120`: 12 post-F1120 events (Other opioid dependence with intoxication)

### 5. **Healthcare Services** (Post-Diagnosis Care)

**CPT Code H0020** (Alcohol and/or drug services): **9,630 post-F1120 events (100% post)**

**Why this is leakage:**
- This is a billing code for substance abuse services
- These services are provided **after** diagnosis as part of treatment
- They wouldn't exist before the diagnosis

**Other Service Codes:**
- `J2315`: 156 post-F1120 events (Injection, naltrexone)
- `90868`: 121 post-F1120 events (Psychotherapy)
- `99335`: 28 post-F1120 events (Nursing facility care)
- `99334`: 22 post-F1120 events (Nursing facility care)

### 6. **Complications and Adverse Events**

**ICD Codes indicating complications:**
- `T8142XA`: 5 post-F1120 events (Complications of procedures)
- `N907`: 5 post-F1120 events (Complications)
- `S83511D`: 31 post-F1120 events (Injury codes)
- `S88911A`: 23 post-F1120 events (Injury codes)

**Why this is leakage:**
- These represent **adverse events** that occur during or after treatment
- They are **consequences** of the condition and its treatment
- Not predictive signals

### 7. **Care Coordination Codes**

**ICD Z-codes** (Factors influencing health status):
- `Z3046`: 25 post-F1120 events (Encounter for aftercare)
- `Z3800`: 17 post-F1120 events (Encounter for other specified aftercare)

**Why this is leakage:**
- These are administrative codes indicating **follow-up care**
- They explicitly mark events as occurring **after** treatment initiation
- They are markers of post-diagnosis care, not predictors

## Key Patterns Identified

### Pattern 1: Treatment Cascade
1. F1120 diagnosis occurs
2. Treatment medications prescribed (SUBOXONE, BUPRENORPHINE, etc.)
3. Monitoring tests ordered (drug screening)
4. Follow-up care provided (therapy, case management)
5. Complications may arise (infections, injuries)

**All of these are post-diagnosis and represent leakage.**

### Pattern 2: Administrative/Procedural Codes
- CPT codes starting with `H` (Healthcare Common Procedure Coding System)
- ICD Z-codes (aftercare, follow-up)
- These are **billing/administrative** codes that mark post-diagnosis activities

### Pattern 3: Mental Health Comorbidities
- F3-prefix ICD codes (mental health disorders)
- These may be:
  - Pre-existing but only diagnosed during treatment
  - New diagnoses arising from substance use
  - Either way, they're captured post-diagnosis in this dataset

## Recommendations

### 1. **Filter All 348 Leakage Features**
Remove all features with `post_f1120_ratio >= 0.8` before model training.

### 2. **Use Pre-F1120 Features Only**
Focus on the **114 features** with `pre_f1120_ratio >= 0.8` as these are truly predictive.

### 3. **Review Mixed-Timing Features**
For the **567 features** with mixed timing (neither ≥80% pre nor ≥80% post):
- If they have significant pre-F1120 presence, they may be legitimate predictors
- Consider using only the pre-F1120 portion of these features
- Or set a threshold (e.g., require ≥50% pre-F1120) for inclusion

### 4. **Temporal Validation**
- Ensure all feature engineering uses only pre-F1120 data
- Use time-based train/test splits
- Validate that no future information leaks into training

### 5. **Documentation**
- Document which features were removed and why
- Maintain a list of known leakage features for future reference
- Update feature engineering pipelines to automatically flag potential leakage

## Example: Why CPT 80307 is Leakage

**CPT 80307**: Presumptive drug class screening
- **3,636 post-F1120 events**
- **0 pre-F1120 events**
- **100% post-F1120 ratio**

**Why this is leakage:**
1. Drug screening is ordered **after** opioid dependence diagnosis
2. It's used to **monitor** treatment compliance, not predict diagnosis
3. The test wouldn't exist in the patient's record before diagnosis
4. Using it as a feature would let the model "see into the future"

**Correct approach:**
- Remove `item_cpt_80307` from feature set
- If drug screening is needed, look for **pre-diagnosis** screening tests
- Or create a feature: "had any drug screening **before** F1120" (not after)

## Conclusion

The 348 leakage features identified represent:
- **Treatment interventions** (medications, procedures)
- **Monitoring activities** (drug tests, follow-ups)
- **Complications** (adverse events, infections)
- **Care coordination** (referrals, case management)
- **Administrative codes** (billing, aftercare markers)

All of these are **consequences** of the F1120 diagnosis, not predictors. They should be filtered out to prevent target leakage and ensure the model learns true predictive patterns rather than post-diagnosis artifacts.
