# Environment-aware defaults
import os

# Default outputs directory for target artifacts. Can be overridden by setting
# the environment variable PGX_TARGET_OUTPUTS_DIR (useful on EC2 or CI).
DEFAULT_TARGET_OUTPUTS_DIR = os.environ.get(
    'PGX_TARGET_OUTPUTS_DIR', os.path.join('1_apcd_input_data', 'outputs')
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

# Codes to exclude (lagging variables)
EXCLUDED_CODES = {
    'F11',    # All F11 codes (opioid use disorder)
    'F1120',  # Opioid use disorder
    'HCG',    # VHI grouping code
    'hcg',    # Case-insensitive match
    'medical_supplies',  # Medical supplies and devices
    'freestyle_lancets'  # Blood glucose testing supplies
}

# Comprehensive Opioid-Related ICD Codes
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

# Age bands for cohort analysis
AGE_BANDS = [
    '0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-94', '95-114'
]

# Event years for cohort analysis
EVENT_YEARS = [
    '2016', '2017', '2018', '2019', '2020'
]

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
