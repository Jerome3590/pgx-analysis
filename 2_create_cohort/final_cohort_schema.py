# final_cohort_schema.py
"""
Schema for the final cohort output of create_cohort.py
- Order matches final_cohort_schema.json (with 'target' at the end)
- Includes descriptions and key comments for each field
"""

final_cohort_schema = [
    # Unique person identifier
    ("mi_person_key", "str", "Unique masked person key (primary identifier)"),
    # Demographics and cohort info
    ("age_band", "str", "Age band (e.g., '65-74')"),
    ("event_year", "int", "Year of event (e.g., 2016)"),
    ("cohort_name", "str", "Cohort group: 'opioid_ed' or 'ed_non_opioid'"),
    # Event details
    ("event_date", "str", "Date of event (YYYY-MM-DD)"),
    ("event_type", "str", "Type of event: 'medical', 'drug', etc."),
    ("data_source", "str", "Source of event data (claims, pharmacy, etc.)"),
    # NOTE: days_to_target_event is NULL for OPIOID_ED cohort (can be calculated from event_date and first_opioid_ed_date)
    # NOTE: days_to_target_event is calculated for ED_NON_OPIOID cohort (used for 30-day lookback window filtering)
    ("days_to_target_event", "int", "Days from event to target event - NULL for OPIOID_ED, calculated for ED_NON_OPIOID"),
    # Demographics
    ("age_imputed", "int", "Imputed age at event (1-114)"),
    ("member_gender", "str", "Gender of member"),
    ("member_race", "str", "Race/ethnicity of member"),
    ("member_zip_code_dos", "str", "ZIP code at date of service"),
    ("member_county_dos", "str", "County at date of service"),
    ("payer_type", "str", "Type of insurance payer"),
    # Medical event fields (may be NULL for drug events)
    ("primary_icd_diagnosis_code", "str", "Primary ICD diagnosis code (medical events only)"),
    ("primary_icd_rollup", "str", "ICD code rollup (medical events only)"),
    ("primary_icd_ccs_level_1", "str", "CCS Level 1 (medical events only)"),
    ("primary_icd_ccs_level_2", "str", "CCS Level 2 (medical events only)"),
    ("primary_icd_ccs_level_3", "str", "CCS Level 3 (medical events only)"),
    ("hcg_setting", "str", "Healthcare setting (medical events only)"),
    ("hcg_line", "str", "Healthcare line (medical events only)"),
    ("hcg_detail", "str", "Healthcare detail (medical events only)"),
    ("place_of_service", "str", "Place of service (medical events only)"),
    ("admit_type", "str", "Admission type (medical events only)"),
    ("procedure_code", "str", "Procedure code (medical events only)"),
    ("procedure_name", "str", "Procedure name (medical events only)"),
    ("billing_provider_name", "str", "Billing provider name (medical events only)"),
    ("service_provider_name", "str", "Service provider name (medical events only)"),
    # Drug event fields (may be NULL for medical events)
    ("drug_name", "str", "Drug name (drug events only)"),
    ("therapeutic_class_1", "str", "Therapeutic class level 1 (drug events only)"),
    ("therapeutic_class_2", "str", "Therapeutic class level 2 (drug events only)"),
    ("therapeutic_class_3", "str", "Therapeutic class level 3 (drug events only)"),
    # Cohort-specific event dates
    # NOTE: first_opioid_ed_date is populated for OPIOID_ED cohort only (NULL for ED_NON_OPIOID)
    # NOTE: first_ed_non_opioid_date is populated for ED_NON_OPIOID cohort only (NULL for OPIOID_ED)
    ("first_opioid_ed_date", "str", "Date of first opioid ED event (if any) - OPIOID_ED cohort only"),
    ("first_ed_non_opioid_date", "str", "Date of first non-opioid ED event (if any) - ED_NON_OPIOID cohort only"),
    # Metadata
    ("created_at", "str", "Timestamp when row was created"),
    ("age_band_filter", "str", "Age band filter used for cohort selection"),
    ("event_year_filter", "int", "Event year filter used for cohort selection"),
    # Target variable (for ML)
    ("target", "int", "Target variable: 1=case, 0=control (place at end for ML)")
]
