# final_cohort_schema.py
"""
Schema for the final cohort output of create_cohort.py
- Order matches final_cohort_schema.json
- Includes descriptions and key comments for each field
- Updated to reflect Phase 3 output including time window target columns
"""

final_cohort_schema = [
    # Unique person identifier
    ("mi_person_key", "str", "Unique masked person key (primary identifier)"),
    
    # Event details
    ("event_date", "str", "Date of event (YYYY-MM-DD)"),
    ("event_type", "str", "Type of event: 'medical' or 'pharmacy'"),
    ("data_source", "str", "Source of event data: 'medical' or 'pharmacy'"),
    
    # Demographics (imputed)
    ("age_imputed", "int", "Imputed age at event (1-114)"),
    ("member_gender", "str", "Gender of member (imputed)"),
    ("member_race", "str", "Race/ethnicity of member (imputed)"),
    ("zip_imputed", "str", "ZIP code at date of service (imputed)"),
    ("county_imputed", "str", "County at date of service (imputed)"),
    ("payer_imputed", "str", "Type of insurance payer (imputed)"),
    
    # ALL ICD diagnosis codes (for ML feature discovery) - positions 1-10
    ("primary_icd_diagnosis_code", "str", "Primary ICD diagnosis code (medical events only)"),
    ("two_icd_diagnosis_code", "str", "Second ICD diagnosis code (medical events only)"),
    ("three_icd_diagnosis_code", "str", "Third ICD diagnosis code (medical events only)"),
    ("four_icd_diagnosis_code", "str", "Fourth ICD diagnosis code (medical events only)"),
    ("five_icd_diagnosis_code", "str", "Fifth ICD diagnosis code (medical events only)"),
    ("six_icd_diagnosis_code", "str", "Sixth ICD diagnosis code (medical events only)"),
    ("seven_icd_diagnosis_code", "str", "Seventh ICD diagnosis code (medical events only)"),
    ("eight_icd_diagnosis_code", "str", "Eighth ICD diagnosis code (medical events only)"),
    ("nine_icd_diagnosis_code", "str", "Ninth ICD diagnosis code (medical events only)"),
    ("ten_icd_diagnosis_code", "str", "Tenth ICD diagnosis code (medical events only)"),
    
    # ALL ICD procedure codes (for ML feature discovery) - positions 2-10
    ("two_icd_procedure_code", "str", "Second ICD procedure code (medical events only)"),
    ("three_icd_procedure_code", "str", "Third ICD procedure code (medical events only)"),
    ("four_icd_procedure_code", "str", "Fourth ICD procedure code (medical events only)"),
    ("five_icd_procedure_code", "str", "Fifth ICD procedure code (medical events only)"),
    ("six_icd_procedure_code", "str", "Sixth ICD procedure code (medical events only)"),
    ("seven_icd_procedure_code", "str", "Seventh ICD procedure code (medical events only)"),
    ("eight_icd_procedure_code", "str", "Eighth ICD procedure code (medical events only)"),
    ("nine_icd_procedure_code", "str", "Ninth ICD procedure code (medical events only)"),
    ("ten_icd_procedure_code", "str", "Tenth ICD procedure code (medical events only)"),
    
    # Drug event fields (may be NULL for medical events)
    ("drug_name", "str", "Drug name (pharmacy events only)"),
    ("therapeutic_class_1", "str", "Therapeutic class level 1 (pharmacy events only)"),
    
    # CPT/procedure codes (medical events only)
    ("procedure_code", "str", "Procedure code (medical events only)"),
    ("cpt_mod_1_code", "str", "CPT modifier 1 (medical events only)"),
    ("cpt_mod_2_code", "str", "CPT modifier 2 (medical events only)"),
    
    # HCG fields for ED visit identification (medical events only)
    ("hcg_setting", "str", "Healthcare setting (medical events only)"),
    ("hcg_line", "str", "Healthcare line code - used for ED visit identification (medical events only)"),
    ("hcg_detail", "str", "Healthcare detail (medical events only)"),
    
    # Event classification and sequence
    ("event_classification", "str", "Event classification: 'opioid_ed', 'ed_non_opioid', 'target', 'non_target'"),
    ("event_sequence", "int", "Sequential order of events per patient (globally ordered across medical and pharmacy)"),
    
    # Cohort metadata
    # NOTE: target column is legacy compatibility - use is_target_case for actual target/control distinction
    ("target", "int", "Legacy target column: 1 for OPIOID_ED/ED_NON_OPIOID cohorts (use is_target_case instead)"),
    ("cohort_name", "str", "Cohort group name: 'OPIOID_ED' or 'ED_NON_OPIOID'"),
    ("cohort", "str", "Cohort classification: 'OPIOID_ED', 'NON_OPIOID_ED', or 'NON_ED'"),
    
    # Target case indicators
    # NOTE: is_target_case uses the main time window (default 14 days for polypharmacy cohort)
    # NOTE: is_target_case_7d through is_target_case_45d are multiclass target columns for polypharmacy cohort only
    # NOTE: For OPIOID_ED cohort, only is_target_case is populated (time window columns are NULL)
    ("is_target_case", "int", "Main target case indicator: 1=target case, 0=control (uses default time window for polypharmacy)"),
    ("is_target_case_7d", "int", "Target case indicator for 7-day time window (polypharmacy cohort only, NULL for OPIOID_ED)"),
    ("is_target_case_14d", "int", "Target case indicator for 14-day time window (polypharmacy cohort only, NULL for OPIOID_ED)"),
    ("is_target_case_21d", "int", "Target case indicator for 21-day time window (polypharmacy cohort only, NULL for OPIOID_ED)"),
    ("is_target_case_30d", "int", "Target case indicator for 30-day time window (polypharmacy cohort only, NULL for OPIOID_ED)"),
    ("is_target_case_45d", "int", "Target case indicator for 45-day time window (polypharmacy cohort only, NULL for OPIOID_ED)"),
    
    # Cohort-specific event dates
    # NOTE: first_opioid_ed_date is populated for OPIOID_ED cohort only (NULL for ED_NON_OPIOID)
    # NOTE: first_ed_non_opioid_date is populated for ED_NON_OPIOID cohort only (NULL for OPIOID_ED)
    ("first_opioid_ed_date", "str", "Date of first opioid ED event (if any) - OPIOID_ED cohort only"),
    ("first_ed_non_opioid_date", "str", "Date of first non-opioid ED event (if any) - ED_NON_OPIOID cohort only"),
    
    # Temporal analysis
    # NOTE: days_to_target_event is NULL for OPIOID_ED cohort (can be calculated from event_date and first_opioid_ed_date)
    # NOTE: days_to_target_event is calculated for ED_NON_OPIOID cohort (used for time window filtering)
    ("days_to_target_event", "int", "Days from event to target event - NULL for OPIOID_ED, calculated for ED_NON_OPIOID")
]
