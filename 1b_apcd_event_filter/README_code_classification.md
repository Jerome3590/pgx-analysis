# ICD and CPT Code Classification: Administrative vs. Medical

## Overview

This document describes the methodology for classifying ICD-10-CM, CPT, and HCPCS codes as **administrative** vs. **medical** for use in event filtering (Step 4b). This classification is critical for filtering out administrative events (billing, scheduling, documentation) while preserving all medical/pharmacy events. Codes are identified in Step 3b (`0_icd_cpt_check`).

## Background: Why Code Classification Matters

Neither ICD-10-CM nor CPT natively labels codes as "administrative" or "medical." This classification must be inferred from code patterns, descriptions, and clinical context. The classification heuristics are based on:

1. **ICD-10-CM Chapter 21 (Z00-Z99) patterns**: Administrative examinations vs. clinical diagnoses
2. **CPT code sections and service descriptions**: Evaluation & Management (E/M) codes vs. diagnostic/therapeutic procedures
3. **HCPCS code prefixes**: G, H, S codes and their typical use cases

## Classification Heuristics

### ICD-10-CM Classification

#### Administrative Codes

**Z02.* codes** - Clearly administrative (administrative examinations):
- Z02.0 - Encounter for examination for admission to educational institution
- Z02.1 - Encounter for examination for admission to residential institution
- Z02.2 - Encounter for examination for admission to prison
- Z02.3 - Encounter for examination for recruitment to armed forces
- Z02.4 - Encounter for examination for driving license
- Z02.5 - Encounter for examination for participation in sport
- Z02.6 - Encounter for examination for insurance purposes
- Z02.7 - Encounter for issue of medical certificate
- Z02.8 - Encounter for other administrative examinations
- Z02.9 - Encounter for administrative examination, unspecified

**Z00-Z04 codes** - Potentially administrative (when used for third-party requirements):
- Z00.00 - Encounter for general adult medical examination without abnormal findings
- Z00.01 - Encounter for general adult medical examination with abnormal findings
- Z00.121 - Encounter for routine child health examination with abnormal findings
- Z00.129 - Encounter for routine child health examination without abnormal findings
- Z01.00 - Encounter for examination of eyes and vision without abnormal findings
- Z01.10 - Encounter for examination of ears and hearing without abnormal findings
- Z02.0-Z02.9 - Administrative examinations (see above)
- Z03.0-Z03.9 - Encounter for medical observation for suspected diseases and conditions
- Z04.0-Z04.9 - Encounter for examination and observation for other reasons

**Classification Rule for Z00-Z04:**
- If encounter is for third-party requirement (employment, school, insurance, legal) → **administrative**
- If encounter is for preventive care or patient complaint → **medical**

#### Medical Codes

**Z11-Z13 codes** - Screening codes (medical - preventive care):
- Z11.0-Z11.9 - Encounter for screening for infectious diseases
- Z12.0-Z12.9 - Encounter for screening for malignant neoplasms
- Z13.0-Z13.9 - Encounter for screening for other diseases and disorders

**Z55-Z65, Z59 codes** - Social determinants of health (medical - clinical context):
- Z55.0-Z55.9 - Problems related to education and literacy
- Z56.0-Z56.9 - Problems related to employment and unemployment
- Z57.0-Z57.9 - Occupational exposure to risk factors
- Z58.0-Z58.9 - Problems related to physical environment
- Z59.0-Z59.9 - Problems related to housing and economic circumstances
- Z60.0-Z60.9 - Problems related to social environment
- Z61.0-Z61.9 - Problems related to negative life events in childhood
- Z62.0-Z62.9 - Other problems related to upbringing
- Z63.0-Z63.9 - Other problems related to primary support group
- Z64.0-Z64.9 - Problems related to certain psychosocial circumstances
- Z65.0-Z65.9 - Problems related to other psychosocial circumstances

**All other codes (A00-Y99, other Z-codes)** - Medical diagnoses:
- All diagnosis codes outside the Z00-Z04 administrative range are classified as **medical**

### CPT Classification

#### Administrative Codes

**E/M codes (9920x, 9921x range)** - Potentially administrative:
- 99202-99205 - Office or other outpatient visit, new patient
- 99211-99215 - Office or other outpatient visit, established patient

**Classification Rule for E/M codes:**
- If visit is for third-party exam (pre-employment, independent medical examination, disability evaluation, legal, administrative, form completion) → **administrative**
- If visit is problem-oriented, preventive, or management visit → **medical**

#### Medical Codes

**Emergency Department E/M codes (9928x, 9929x)** - Medical:
- 99281-99285 - Emergency department visit
- 99291-99292 - Critical care

**Category I CPT codes** - Medical procedures:
- **Surgery (10021-69990)**: All surgical procedures
- **Radiology (70000-79999)**: Diagnostic imaging and radiology procedures
- **Pathology & Laboratory (80000-89999)**: Laboratory tests and pathology procedures
- **Medicine (90000-99999)**: Medical procedures and services

**Default**: Most CPT codes are medical procedures and should be classified as **medical**

### HCPCS Classification

**G codes** - Often administrative:
- G codes are frequently used for administrative services (care management, behavioral health services)
- Classification depends on specific code and context

**H codes** - Behavioral health services:
- Most H codes are medical (behavioral health services)

**S codes** - Temporary codes:
- Varies by code - requires case-by-case review

## Implementation

The classification is implemented in `research_icd_cpt_codes.py` with the following functions:

- `classify_icd_code()`: Classifies ICD-10/ICD-9 codes
- `classify_cpt_code()`: Classifies CPT codes
- `classify_hcpcs_code()`: Classifies HCPCS codes
- `classify_code_as_administrative()`: Main router function

## Web Search Integration

The script attempts to look up code meanings using web search with the following sources:

- **ICD codes**: [CDC ICD-10-CM Tool](https://icd10cmtool.cdc.gov/?fy=FY2026)
- **CPT codes**: [AMA CPT Codes](https://www.ama-assn.org/topics/cpt-codes)
- **HCPCS codes**: [CMS Physician Fee Schedule](https://www.cms.gov/medicare/physician-fee-schedule/search)

If web search fails or returns no results, the script falls back to heuristic classification based on the rules above.

## Lookup Table Structure

The lookup table (`administrative_codes_lookup.json`) contains:

```json
{
  "description": "Lookup table for classifying codes as administrative vs. medical/pharmacy",
  "version": "1.0",
  "last_updated": "YYYY-MM-DD",
  "administrative_codes": {
    "icd": ["Z02.0", "Z02.1", ...],
    "cpt": ["99213", "99214", ...],
    "hcpcs": ["G0483", "G0480", ...]
  },
  "notes": {
    "icd": "ICD codes for administrative/billing purposes",
    "cpt": "CPT codes for administrative procedures",
    "hcpcs": "HCPCS codes for administrative services"
  }
}
```

## Research Workflow

1. **Extract codes** from aggregated feature importance CSVs
2. **Look up code meanings** using web search (with fallback to heuristics)
3. **Classify codes** as administrative or medical
4. **Review classifications** using reference URLs
5. **Update lookup table** with confirmed administrative codes
6. **Use lookup table** in event filter (Step 4b) to remove administrative events

## References

### ICD-10-CM References

- [CDC ICD-10-CM Tool](https://icd10cmtool.cdc.gov/?fy=FY2026) - Official ICD-10-CM code lookup
- [Wolters Kluwer: Guide to ICD-10-CM Z Codes](https://www.wolterskluwer.com/en/expert-insights/guide-to-icd-10-cm-z-codes)
- [BCBSRI: Administrative Examination Diagnosis Codes](https://www.bcbsri.com/providers/update/icd-10-administrative-examination-diagnosis-codes)
- [AAPC: Z02 Codes](https://www.aapc.com/codes/icd-10-codes/Z02)
- [Solventum: Z Codes as Principal Diagnosis](https://www.solventum.com/en-us/home/health-information-technology/resources-education/blog/2024/3/z-codes-that-may-only-be-principal-first-listed-diagnosis/)

### CPT References

- [AMA: CPT Codes](https://www.ama-assn.org/topics/cpt-codes) - Official CPT code information
- [AAPC: E/M Codes](https://www.aapc.com/resources/what-are-e-m-codes) - Evaluation & Management codes
- [AMA: E/M Descriptors and Guidelines](https://www.ama-assn.org/system/files/2023-e-m-descriptors-guidelines.pdf)
- [CMS: Coding Classification Systems](https://www.cms.gov/cms-guide-medical-technology-companies-and-other-interested-parties/coding/overview-coding-classification-systems)

### General Medical Coding References

- [Medical Billing and Coding: Classification Systems](https://www.medicalbillingandcoding.org/qnas/what-are-the-different-types-of-medical-coding-classification-systems/)
- [AIMS Education: CPT Codes](https://aimseducation.edu/blog/what-are-cpt-codes)
- [DeVry: Medical Codes and Classification Systems](https://www.devry.edu/blog/understanding-medical-codes-and-coding-classification-systems.html)

## Usage

### Running the Research Script

```bash
# Extract codes and create lookup table
python 4b_event_filter/research_icd_cpt_codes.py

# Skip web search (faster, uses heuristics only)
python 4b_event_filter/research_icd_cpt_codes.py --skip-web-search
```

### Output Files

- `4b_event_filter/outputs/code_research/icd_cpt_hcpcs_codes_research.csv` - Research CSV with all codes, descriptions, and classifications
- `4b_event_filter/outputs/code_research/administrative_codes_lookup.json` - Lookup table for filtering
- `4b_event_filter/outputs/code_research/icd_codes_list.txt` - List of ICD codes
- `4b_event_filter/outputs/code_research/cpt_codes_list.txt` - List of CPT codes
- `4b_event_filter/outputs/code_research/hcpcs_codes_list.txt` - List of HCPCS codes

### Validating Classifications

1. Review the research CSV for codes classified as "administrative"
2. Use reference URLs to verify code meanings
3. Update classifications if needed
4. Copy `administrative_codes_lookup.json` to `4b_event_filter/administrative_codes_lookup.json` when ready

## Notes

- **Post-event leakage**: Events occurring on or after target event date are automatically classified as administrative (leakage) and do not need to be in the lookup table
- **Conservative approach**: When in doubt, classify as medical (don't filter it)
- **Context matters**: Some codes (especially E/M codes) may be administrative in one context but medical in another. The lookup table should reflect the most common use case in your data.
