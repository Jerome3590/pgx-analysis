# Environment-aware defaults
import os

# Default outputs directory for target artifacts. Can be overridden by setting
# the environment variable PGX_TARGET_OUTPUTS_DIR (useful on EC2 or CI).
DEFAULT_TARGET_OUTPUTS_DIR = os.environ.get(
    'PGX_TARGET_OUTPUTS_DIR', os.path.join('1a_apcd_input_data', 'outputs')
)

# Environment-driven target selection (these mirror PGX_TARGET_* env vars)
# Use these constants throughout the codebase instead of calling os.getenv() everywhere.
PGX_TARGET_NAME = os.environ.get('PGX_TARGET_NAME', '').strip()
PGX_TARGET_ICD_CODES = os.environ.get('PGX_TARGET_ICD_CODES', '').strip()
PGX_TARGET_CPT_CODES = os.environ.get('PGX_TARGET_CPT_CODES', '').strip()
PGX_TARGET_ICD_PREFIXES = os.environ.get('PGX_TARGET_ICD_PREFIXES', '').strip()
PGX_TARGET_CPT_PREFIXES = os.environ.get('PGX_TARGET_CPT_PREFIXES', '').strip()

# Richmond, VA zip codes
RICHMOND_ZIP_CODES = {
    '23173', '23218', '23219', '23220', '23221', '23222', '23223', '23224',
    '23225', '23232', '23240', '23241', '23249', '23260', '23261', '23284',
    '23285', '23298'
}

# Drug-name column: values to exclude from model training (not drugs or not useful as features).
# Used in Step 3b (filter_and_refine_features), Step 4 (create_model_data.get_important_items),
# and Step 6 (build_final_cohort_model_features). See READMEs and docs for full rationale.
#
# 1036F: Not a drug. CPT Category II tracking code used to document that a patient (18+) is a
#   current tobacco non-user, usually during preventive screenings; part of quality measures
#   for tobacco use assessment and preventive care.
# T401XA1: Not a drug. ICD-10-CM diagnosis code for "Poisoning by 4-aminophenol derivatives,
#   accidental (unintentional), initial encounter" — in practice usually unintentional overdose
#   or poisoning with acetaminophen (paracetamol) or closely related compounds, at the
#   patient's initial encounter.
# Narcan, Unknown, Fentanyl: excluded per model-training requirements.
DRUG_NAMES_EXCLUDED_MODEL_TRAINING = frozenset({
    "Narcan",
    "Unknown",
    "Fentanyl",
    "1036F",
    "T401XA1",
})

# Substrings to exclude from any feature name (case-insensitive).
# Syringe was removed; it is helpful for identifying diabetics.
FEATURE_SUBSTRINGS_EXCLUDED = frozenset()

# Codes to exclude (lagging variables)
EXCLUDED_CODES = {
    'F11',    # All F11 codes (opioid use disorder)
    'F1120',  # Opioid use disorder
    'HCG',    # VHI grouping code
    'hcg',    # Case-insensitive match
    'medical_supplies',  # Medical supplies and devices
    'freestyle_lancets'  # Blood glucose testing supplies
}

# Comprehensive Opioid-Related ICD Codes (Step 2 cohort target for opioid_ed).
# F10 (alcohol) and F19 (other psychoactive substance) are intentionally NOT included:
# cohort target is opioid-specific. In 3b feature importance, F10/F11/F19 are excluded
# as outcome-related (substance-use) features.
OPIOID_ICD_CODES = {
    # Opioid Use Disorder (F11.x)
    'F11.20', 'F11.21', 'F11.22', 'F11.23', 'F11.24', 'F11.25', 'F11.26', 'F11.27', 'F11.28', 'F11.29',
    'F1120', 'F1121', 'F1122', 'F1123', 'F1124', 'F1125', 'F1126', 'F1127', 'F1128', 'F1129',

    # Opioid Poisoning/Overdose (T40.x)
    'T40.0', 'T40.1', 'T40.2', 'T40.3', 'T40.4', 'T40.5', 'T40.6', 'T40.7', 'T40.8', 'T40.9',
    'T400', 'T401', 'T402', 'T403', 'T404', 'T405', 'T406', 'T407', 'T408', 'T409',

    # Opioid Abuse (F11.1x)
    'F11.10', 'F11.11', 'F11.12', 'F11.13', 'F11.14', 'F11.15', 'F11.16', 'F11.17', 'F11.18', 'F11.19',
    'F1110', 'F1111', 'F1112', 'F1113', 'F1114', 'F1115', 'F1116', 'F1117', 'F1118', 'F1119',

    # Opioid Intoxication (F11.0x)
    'F11.00', 'F11.01', 'F11.02', 'F11.03', 'F11.04', 'F11.05', 'F11.06', 'F11.07', 'F11.08', 'F11.09',
    'F1100', 'F1101', 'F1102', 'F1103', 'F1104', 'F1105', 'F1106', 'F1107', 'F1108', 'F1109',

    # Opioid Withdrawal (F11.3x)
    'F11.30', 'F11.31', 'F11.32', 'F11.33', 'F11.34', 'F11.35', 'F11.36', 'F11.37', 'F11.38', 'F11.39',
    'F1130', 'F1131', 'F1132', 'F1133', 'F1134', 'F1135', 'F1136', 'F1137', 'F1138', 'F1139',

    # Opioid-Induced Disorders (F11.8x, F11.9x)
    'F11.80', 'F11.81', 'F11.82', 'F11.83', 'F11.84', 'F11.85', 'F11.86', 'F11.87', 'F11.88', 'F11.89',
    'F11.90', 'F11.91', 'F11.92', 'F11.93', 'F11.94', 'F11.95', 'F11.96', 'F11.97', 'F11.98', 'F11.99',
    'F1180', 'F1181', 'F1182', 'F1183', 'F1184', 'F1185', 'F1186', 'F1187', 'F1188', 'F1189',
    'F1190', 'F1191', 'F1192', 'F1193', 'F1194', 'F1195', 'F1196', 'F1197', 'F1198', 'F1199',

    # Opioid-Related Complications (Y12.x)
    'Y12.0', 'Y12.1', 'Y12.2', 'Y12.3', 'Y12.4', 'Y12.5', 'Y12.6', 'Y12.7', 'Y12.8', 'Y12.9',
    'Y120', 'Y121', 'Y122', 'Y123', 'Y124', 'Y125', 'Y126', 'Y127', 'Y128', 'Y129',

    # Opioid-Related Adverse Effects (T40.6x - Narcotic antagonists)
    'T40.60', 'T40.61', 'T40.62', 'T40.63', 'T40.64', 'T40.65', 'T40.66', 'T40.67', 'T40.68', 'T40.69',
    'T4060', 'T4061', 'T4062', 'T4063', 'T4064', 'T4065', 'T4066', 'T4067', 'T4068', 'T4069'
}

# All ICD diagnosis code column names (positions 1-10)
ALL_ICD_DIAGNOSIS_COLUMNS = [
    'primary_icd_diagnosis_code',
    'two_icd_diagnosis_code',
    'three_icd_diagnosis_code',
    'four_icd_diagnosis_code',
    'five_icd_diagnosis_code',
    'six_icd_diagnosis_code',
    'seven_icd_diagnosis_code',
    'eight_icd_diagnosis_code',
    'nine_icd_diagnosis_code',
    'ten_icd_diagnosis_code'
]


def get_opioid_icd_sql_condition(table_alias=None):
    """
    Generate SQL condition to check for opioid ICD codes across ALL diagnosis code positions.
    
    Args:
        table_alias: Optional table alias (e.g., 'uef' for 'uef.primary_icd_diagnosis_code')
    
    Returns:
        SQL WHERE condition string checking all 10 ICD diagnosis columns
    
    Example:
        >>> get_opioid_icd_sql_condition()
        "(primary_icd_diagnosis_code IN ('F1120', ...) OR two_icd_diagnosis_code IN (...) OR ...)"
    """
    prefix = f"{table_alias}." if table_alias else ""
    codes_tuple = tuple(OPIOID_ICD_CODES)
    
    conditions = [f"{prefix}{col} IN {codes_tuple}" for col in ALL_ICD_DIAGNOSIS_COLUMNS]
    return "(" + " OR ".join(conditions) + ")"


def get_icd_codes_sql_condition(icd_codes, table_alias=None):
    """
    Generate SQL condition to check for specific ICD codes across ALL diagnosis code positions.
    
    Args:
        icd_codes: Set or list of ICD codes to check
        table_alias: Optional table alias
    
    Returns:
        SQL WHERE condition string checking all 10 ICD diagnosis columns
    """
    prefix = f"{table_alias}." if table_alias else ""
    codes_tuple = tuple(icd_codes)
    
    conditions = [f"{prefix}{col} IN {codes_tuple}" for col in ALL_ICD_DIAGNOSIS_COLUMNS]
    return "(" + " OR ".join(conditions) + ")"


# FpGrowth
TOP_K = 50
MIN_SUPPORT_THRESHOLD = 0.025
MIN_SUPPORT_FINAL = 0.01
MAX_ATTEMPTS = 5
TIMEOUT_SECONDS = 300

# Rule generation
MIN_CONFIDENCE_SMALL = 0.1
MIN_CONFIDENCE_MEDIUM = 0.25
MIN_CONFIDENCE_LARGE = 0.3
MIN_LIFT_SMALL = 0.5
MIN_LIFT_MEDIUM = 0.6
MIN_LIFT_LARGE = 0.7
MIN_SUPPORT_RULE = 0.025
FALLBACK_DELTA = 0.005
MIN_FALLBACK_CONFIDENCE = 0.1
MIN_FALLBACK_LIFT = 0.0


# Pattern metrics
METRIC_COLUMNS = ["support", "confidence", "lift", "certainty"]
MAX_PATTERN_COLUMNS = 15

# AWS configuration
S3_BUCKET = "pgxdatalake"
METRICS_BUCKET = "pgx-repository"
SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/535362115856/cohorts.fifo"
BASE_PATH_FEATURES = "s3://pgxdatalake/fpgrowth_features"
BASE_PATH_COHORT = "s3://pgxdatalake/cohorts"
MAX_RETRIES = 3
RETRY_DELAY = 2
AWS_REGION = "us-east-1" 

# Email configuration
NOTIFICATION_EMAIL = "jerome@mushinsolutions.com" 

# Age bands for cohort analysis (last band 85-114 combines former 85-94 and 95-114)
AGE_BANDS = [
    '0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114'
]

# Event years for cohort analysis
EVENT_YEARS = [
    '2016', '2017', '2018', '2019', '2020'
]

# Cohort names for feature importance analysis
COHORT_NAMES = [
    'opioid_ed', 'non_opioid_ed'
]

# Pipeline-supported (cohort, age_band): model_events and dashboard visuals only run for these.
# Each cohort has all age bands. Align with 3_model_train_shap_ffa.ipynb and 5_build_and_deploy.ipynb.
REQUIRED_COHORTS = {
    "opioid_ed": list(AGE_BANDS),
    "non_opioid_ed": list(AGE_BANDS),
}

# Age bands used for dashboard visualization tabs (BupaR, DTW, FP-Growth, Causal, Cohort PGx).
# Risk Assessment excludes 0-12 (min age 13); do not build or offer 0-12 in dashboard visuals.
DASHBOARD_VISUAL_AGE_BANDS = [b for b in AGE_BANDS if b != "0-12"]

# Helper function: convert age-band to filename-safe format
def age_band_uses_f1120_target(age_band: str) -> bool:
    """
    Determine if age band uses F11.20 target (opioid dependence) or HCG target (polypharmacy).
    
    Rules:
    - Age bands < 65 (13-24, 25-44, 45-54, 55-64): Use F11.20 target
    - Age bands >= 65 (65-74, 75-84, 85-114): Use first ED visit (HCG Setting) within 21 days of a prescription drug event (see NON_OPIOID_ED_TARGET_DESCRIPTION)
    
    Args:
        age_band: Age band string (e.g., "13-24", "65-74")
    
    Returns:
        True if age band uses F11.20 target, False if uses HCG target
    """
    # Parse age band to get lower bound
    try:
        parts = age_band.split('-')
        if len(parts) == 2:
            lower_bound = int(parts[0])
            # Age bands < 64 use F11.20, >= 65 use HCG
            return lower_bound < 65
        else:
            # Fallback: assume F11.20 if can't parse
            return True
    except (ValueError, AttributeError):
        # Fallback: assume F11.20 if can't parse
        return True


def get_target_name(age_band: str) -> str:
    """
    Get the target name for an age band.
    
    Args:
        age_band: Age band string (e.g., "13-24", "65-74")
    
    Returns:
        "F1120" for age bands < 65, "ED visit (HCG)" for age bands >= 65
    """
    if age_band_uses_f1120_target(age_band):
        return "F1120"
    else:
        return "ED visit (HCG)"


def get_cohort_slug(age_band: str) -> str:
    """
    Get the cohort slug for S3 paths based on age band.
    
    Rules:
    - Age bands < 65 (13-24, 25-44, 45-54, 55-64): Use "opioid" slug
    - Age bands >= 65 (65-74, 75-84, 85-114): Use "polypharmacy" slug
    
    Args:
        age_band: Age band string (e.g., "13-24", "65-74")
    
    Returns:
        "opioid" for age bands < 65, "polypharmacy" for age bands >= 65
    """
    if age_band_uses_f1120_target(age_band):
        return "opioid"
    else:
        return "polypharmacy"


def cohort_uses_f1120_target(cohort: str) -> bool:
    """
    Determine target by cohort (Step 2 / 3b convention).
    - opioid_ed: F1120 (first opioid use disorder ED)
    - non_opioid_ed: First ED visit (HCG Setting) within 21 days of a prescription drug event (see NON_OPIOID_ED_TARGET_DESCRIPTION)
    """
    return (cohort or "").strip().lower() == "opioid_ed"


def get_target_name_by_cohort(cohort: str) -> str:
    """Target name for display: opioid_ed -> F1120, non_opioid_ed -> ED visit (HCG)."""
    return "F1120" if cohort_uses_f1120_target(cohort) else "ED visit (HCG)"


# Canonical definition for non_opioid_ed (polypharmacy) cohort target - must stay in sync with
# 2_create_cohort (phase2 HCG classification, phase3 21-day window) and 4_model_data.
# Model_events target-date column for non_opioid_ed is first_o11_p_date; O11_P includes all qualifying ED codes (P51b, O11, P33).
NON_OPIOID_ED_TARGET_DESCRIPTION = (
    "First ED visit (identified by HCG Setting: P51/O11/P33) within 21 days of a prescription drug event."
)
NON_OPIOID_ED_TIME_WINDOW_DAYS = 21
# Max ED visits per year for polypharmacy target definition; patients with this many or more are excluded.
NON_OPIOID_ED_MAX_ED_VISITS_PER_YEAR = 7

# Age-band-specific parameters for non_opioid_ed cohort (relaxed for pediatric/geriatric)
# Rationale: Pediatric (0-12) and young adults (13-24) may have different adverse drug event patterns
# due to different drug metabolism, fewer prescriptions, and different healthcare utilization patterns.
NON_OPIOID_ED_AGE_BAND_PARAMS = {
    # Pediatric (0-12): Wider window (30 days) + more ED visits allowed (10/year)
    # Rationale: Slower metabolism, fewer prescriptions, parents may delay ED visits
    "0-12": {"time_window_days": 30, "max_ed_visits_per_year": 10},
    
    # Young adults (13-24): Slightly relaxed (28 days, 9 visits)
    # Rationale: Transitional age, medication adherence issues, fewer prescriptions
    "13-24": {"time_window_days": 28, "max_ed_visits_per_year": 9},
    
    # Adults (25-64): Standard parameters (21 days, 7 visits)
    # These are the baseline parameters validated by distribution analysis
    "25-44": {"time_window_days": 21, "max_ed_visits_per_year": 7},
    "45-54": {"time_window_days": 21, "max_ed_visits_per_year": 7},
    "55-64": {"time_window_days": 21, "max_ed_visits_per_year": 7},
    
    # Seniors (65-84): Standard parameters
    # Note: Polypharmacy is more common but we maintain strict filters for true adverse events
    "65-74": {"time_window_days": 21, "max_ed_visits_per_year": 7},
    "75-84": {"time_window_days": 21, "max_ed_visits_per_year": 7},
    
    # Elderly (85-114): Slightly relaxed window (25 days, still 7 visits)
    # Rationale: Slower drug metabolism, delayed symptom presentation
    "85-114": {"time_window_days": 25, "max_ed_visits_per_year": 7},
}

def get_non_opioid_ed_params(age_band: str) -> dict:
    """
    Get age-band-specific parameters for non_opioid_ed cohort.
    
    Args:
        age_band: Age band string (e.g., "0-12", "65-74")
    
    Returns:
        Dict with 'time_window_days' and 'max_ed_visits_per_year' keys
    
    Example:
        >>> get_non_opioid_ed_params("0-12")
        {'time_window_days': 30, 'max_ed_visits_per_year': 10}
    """
    return NON_OPIOID_ED_AGE_BAND_PARAMS.get(
        age_band,
        {"time_window_days": NON_OPIOID_ED_TIME_WINDOW_DAYS, 
         "max_ed_visits_per_year": NON_OPIOID_ED_MAX_ED_VISITS_PER_YEAR}
    )

# Expected empty cohorts: age bands where non_opioid_ed is expected to have insufficient data
# Used by dashboard visualizations to show appropriate messages instead of errors
NON_OPIOID_ED_EXPECTED_EMPTY_AGE_BANDS = set()
# Note: After implementing relaxed parameters, we expect all age bands to have some data.
# This set is kept for future reference if specific age bands consistently produce no results.


def get_cohort_slug_by_cohort(cohort: str) -> str:
    """Cohort slug for paths: opioid_ed -> opioid, non_opioid_ed -> polypharmacy."""
    return "opioid" if cohort_uses_f1120_target(cohort) else "polypharmacy"


def get_target_file_suffix(cohort: str) -> str:
    """File suffix for BupaR pre/post target outputs: opioid_ed -> f1120, non_opioid_ed -> target (no F1120 ref)."""
    return "f1120" if cohort_uses_f1120_target(cohort) else "target"


def age_band_to_fname(age_band: str) -> str:
    """Convert an age-band like '0-12' to a filename-safe form '0_12'."""
    return age_band.replace('-', '_') if isinstance(age_band, str) else str(age_band)


def get_physical_age_bands_for_gold(age_band: str) -> list:
    """
    Return the physical age-band partition(s) for gold COHORT data.
    For 85-114 we use the single partition 85-114 only.
    """
    if age_band == "85-114":
        return ["85-114"]
    return [age_band]


def get_physical_age_bands_for_medical_pharmacy(age_band: str) -> list:
    """
    Return the physical age-band partition(s) for gold MEDICAL and PHARMACY data.
    For 85-114, medical/pharmacy are stored as two sub-cohorts: 85-94 and 95-114.
    """
    if age_band == "85-114":
        return ["85-94", "95-114"]
    return [age_band]


def age_band_partition_candidates(physical_band: str) -> list:
    """
    Return candidate partition folder names for a physical age band (e.g. 85-94).
    Tries hyphen first (85-94), then underscore (85_94) so gold data stored either way is found.
    """
    candidates = [physical_band]
    if "-" in physical_band:
        candidates.append(physical_band.replace("-", "_"))
    return candidates

# Processing Configuration
LOCK_TIMEOUT_HOURS = 6  # Hours before considering a lock stale
DEFAULT_SAMPLE_RATIO = 5  # Default 5x controls per positive case

# Bloom filter configuration
BLOOM_FILTER_FALSE_POSITIVE_RATIO = 0.01  # 1% false positive ratio
DICTIONARY_SIZE_LIMIT_PERCENT = 10  # 10% of row group size (enables Bloom filters)

###############################################################################
# Healthcare Cost Group (HCG) System Documentation
###############################################################################

"""
Milliman HCG (Healthcare Cost Group) System:
A widely used system for categorizing and costing healthcare services. This system helps in
standardizing healthcare service classification and cost analysis across different providers
and settings.

Key Components:
1. HCG Line:
   - A specific code within the HCG system (e.g., "O11" for Emergency Room)
   - Used to identify the type of service provided
   - Based on Virginia APCD data description standards
   - Helps in precise service categorization

2. HCG Setting:
   - Broader categorization of services within the HCG system
   - Examples include: Inpatient, Outpatient, Emergency Room
   - Provides context for the service location and type
   - Used in conjunction with HCG Line for complete service classification

3. VHI Healthcare Pricing Report:
   - Utilizes the Milliman HCG system
   - Analyzes healthcare costs and utilization trends
   - Provides standardized cost comparisons across different service types
   - Helps in understanding healthcare service patterns and costs

Usage in Analysis:
- Service Classification: Using HCG Line and Setting for consistent service categorization
- Cost Analysis: Standardized cost comparisons across different service types
- Trend Analysis: Tracking healthcare utilization patterns
- Quality Metrics: Assessing service delivery patterns and outcomes
"""
