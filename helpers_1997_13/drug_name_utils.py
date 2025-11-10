"""
Drug name cleaning and mapping utilities.
"""

import sys
import os
import re
import logging
import pandas as pd
import string
import glob
import json
from collections import OrderedDict, defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional, Union


# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.common_imports import *
from helpers_1997_13.duckdb_utils import get_duckdb_connection

from helpers_1997_13.s3_utils import (
    get_output_paths,
    save_to_s3_parquet, 
    save_to_s3_json
)


def clean_drug_name(name: str, logger: Optional[logging.Logger] = None) -> str:
    """Clean and standardize drug name by removing prefixes and normalizing terms."""
    try:
        if not isinstance(name, str):
            if logger:
                logger.warning(f"Expected string for drug name, got: {type(name)}")
            return ""

        original_name = name  # Keep for logging

        # Remove 'drug_' prefix
        name = re.sub(r"^drug_", "", name, flags=re.IGNORECASE)

        # Standardize special characters
        name = name.replace('+', '_').replace('/', '_')

        # Replace '_hcl' with '_hydrochloride' (case insensitive)
        name = re.sub(r'_hcl(?=[^a-zA-Z]|$)', '_hydrochloride', name, flags=re.IGNORECASE)

        cleaned_name = name.strip()

        if logger:
            logger.debug(f"Cleaned drug name: '{original_name}' -> '{cleaned_name}'")

        return cleaned_name

    except Exception as e:
        if logger:
            logger.error(f"Error cleaning drug name '{name}': {str(e)}")
        else:
            print(f"Error cleaning drug name '{name}': {str(e)}")
        return ""


def count_syllables(word: str, logger: Optional[logging.Logger] = None) -> int:
    """Count the number of syllables in a word."""
    try:
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1

        if logger:
            logger.debug(f"Syllable count for '{word}': {count}")

        return count

    except Exception as e:
        if logger:
            logger.error(f"Error counting syllables in '{word}': {str(e)}")
        else:
            print(f"Error counting syllables in '{word}': {str(e)}")
        return 0


def count_chemical_suffixes(word: str) -> int:
    """Count common chemical suffixes in drug names."""
    suffixes = ['ine', 'ol', 'ate', 'ide', 'in', 'an', 'ic', 'al', 'um', 'on']
    count = 0
    word_lower = word.lower()
    for suffix in suffixes:
        if word_lower.endswith(suffix):
            count += 1
    return count


def count_consonant_clusters(word: str) -> int:
    """Count consecutive consonant sequences (clusters)."""
    consonants = 'bcdfghjklmnpqrstvwxyz'
    clusters = 0
    consecutive = 0
    
    for char in word.lower():
        if char in consonants:
            consecutive += 1
        else:
            if consecutive >= 2:  # Only count as cluster if 2+ consonants
                clusters += 1
            consecutive = 0
    
    # Check if word ends with consonant cluster
    if consecutive >= 2:
        clusters += 1
    
    return clusters


def calculate_repetition_factor(word: str) -> int:
    """Calculate repetition factor based on repeated letters."""
    if not word:
        return 0
    
    char_counts = {}
    for char in word.lower():
        if char.isalpha():
            char_counts[char] = char_counts.get(char, 0) + 1
    
    # Count letters that appear more than once
    repeated_chars = sum(1 for count in char_counts.values() if count > 1)
    return repeated_chars


def encode_drug_name(drug_name: str, fpgrowth_metrics: dict, logger: Optional[logging.Logger] = None) -> str:
    """
    Encode a drug name with its linguistic properties and FpGrowth metrics.

    Format:
    - First letter index (3-digit)
    - Length (2-digit)
    - Syllables (2-digit)
    - Consonants (2-digit)
    - Vowels (2-digit)
    - Hyphens/underscores (2-digit)
    - Chemical suffixes (2-digit)
    - Consonant clusters (2-digit)
    - Repetition factor (2-digit)
    - Support (4-digit, scaled x1000)
    - Number of rules (3-digit)
    - Number of drugs in rules (3-digit)
    Total: 3 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 4 + 3 + 3 = 29 characters
    """
    try:
        clean_name = clean_drug_name(drug_name, logger)

        # Get first character index in name (A=111, B=222, C=333, ..., Z=999, fallback 0)
        first_letter = next((ch.upper() for ch in clean_name if ch.isalpha()), 'X')
        # Create distinct 3-digit codes for each letter
        letter_codes = {
            'A': 111, 'B': 222, 'C': 333, 'D': 444, 'E': 555, 'F': 666, 'G': 777, 'H': 888, 'I': 999,
            'J': 101, 'K': 202, 'L': 303, 'M': 404, 'N': 505, 'O': 606, 'P': 707, 'Q': 808, 'R': 909,
            'S': 110, 'T': 220, 'U': 330, 'V': 440, 'W': 550, 'X': 660, 'Y': 770, 'Z': 880
        }
        first_index = str(letter_codes.get(first_letter, 0)).zfill(3)

        # Basic linguistic features
        length = str(len(clean_name)).zfill(2)
        syllables = str(count_syllables(clean_name)).zfill(2)
        consonants = str(len(re.findall(r'[BCDFGHJKLMNPQRSTVWXYZ]', clean_name.upper()))).zfill(2)
        
        # Additional linguistic features
        vowels = str(len(re.findall(r'[AEIOU]', clean_name.upper()))).zfill(2)
        hyphens_underscores = str(len(re.findall(r'[-_]', clean_name))).zfill(2)
        
        # Advanced linguistic features
        chemical_suffixes = str(count_chemical_suffixes(clean_name)).zfill(2)
        consonant_clusters = str(count_consonant_clusters(clean_name)).zfill(2)
        repetition_factor = str(calculate_repetition_factor(clean_name)).zfill(2)

        # Scale and format support metric
        support = str(int(fpgrowth_metrics.get('support', 0) * 1000)).zfill(4)
        
        # Get rule-related metrics
        num_rules = str(fpgrowth_metrics.get('num_rules', 0)).zfill(3)
        num_drugs_in_rules = str(fpgrowth_metrics.get('num_drugs_in_rules', 0)).zfill(3)

        result = f"{first_index}{length}{syllables}{consonants}{vowels}{hyphens_underscores}{chemical_suffixes}{consonant_clusters}{repetition_factor}{support}{num_rules}{num_drugs_in_rules}"
        
        # Debug logging
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Encoded '{drug_name}' -> '{result}' (first_letter: {first_letter}, first_index: {first_index})")
        
        return result
    
    except Exception as e:
        if logger:
            logger.error(f"Error encoding drug name '{drug_name}': {str(e)}")
            logger.exception(f"Full traceback for drug '{drug_name}':")
        return "00000000000000000000000000000"  # default fallback (29 characters)


def parse_encoding(encoding: str) -> List[int]:
    """Split a 29-character drug encoding into numeric segments."""
    if len(encoding) != 29:
        raise ValueError(f"Encoding must be 29 characters long, got {len(encoding)}: {encoding}")
    
    return [
        int(encoding[0:3]),   # first letter index (1-26)
        int(encoding[3:5]),   # length
        int(encoding[5:7]),   # syllables
        int(encoding[7:9]),   # consonants
        int(encoding[9:11]),  # vowels
        int(encoding[11:13]), # hyphens/underscores
        int(encoding[13:15]), # chemical suffixes
        int(encoding[15:17]), # consonant clusters
        int(encoding[17:19]), # repetition factor
        int(encoding[19:23]), # support (×1000)
        int(encoding[23:26]), # number of rules
        int(encoding[26:29])  # number of drugs in rules
    ]


def encode_pattern_numeric(pattern: Union[set, list], encoding_map: Dict[str, str]) -> List[float]:
    """Numerically encode a pattern by summing parsed encodings of all drugs."""
    numeric_vector = [0.0] * 12
    for drug in pattern:
        encoding = encoding_map.get(drug)
        if encoding:
            components = parse_encoding(encoding)
            numeric_vector = [x + y for x, y in zip(numeric_vector, components)]
    return numeric_vector


def save_drug_encoding_map(
    drug_encodings,
    cohort_name: str,
    age_band: str,
    event_year: str,
    bucket_name: str = "pgxdatalake",
    logger: Optional[logging.Logger] = None
) -> bool:
    """Save the drug encoding map to S3 for a specific cohort."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(list(drug_encodings.items()), columns=['drug_name', 'encoding'])

        # Get cohort-specific output paths
        paths = get_output_paths(cohort_name, age_band, event_year, bucket_name=bucket_name)

        # Define output file paths
        parquet_path = paths.get("drug_encoding_parquet")
        json_path = paths.get("drug_encoding_json")

        # Partition columns to exclude
        partition_cols = ["cohort_name", "age_band", "event_year"]

        # Save Parquet (excluding partition columns if present)
        save_to_s3_parquet(df, parquet_path, logger=logger, partition_cols=partition_cols)

        # Save JSON (also excluding partition keys if embedded in keys)
        save_to_s3_json(drug_encodings, json_path, logger=logger, partition_cols=partition_cols)

        if logger:
            logger.info(f"✓ Successfully saved drug encoding map to {parquet_path}")
        return True

    except Exception as e:
        msg = f"✗ Error saving drug encoding map: {str(e)}"
        if logger:
            logger.error(msg)
        return False


def replace_special_chars(text: str) -> str:
    """
    Replace all '+' and '/' characters with '_' in drug names.
    
    Args:
        text (str): The text to process
        
    Returns:
        str: Text with '+' and '/' replaced by '_'
    """
    return text.replace('+', '_').replace('/', '_')


def standardize_hcl(text: str) -> str:
    """
    Replace '_hcl' with '_hydrochloride' in drug names (case insensitive).
    
    Args:
        text (str): The text to process
        
    Returns:
        str: Text with '_hcl' standardized to '_hydrochloride'
    """
    return re.sub(r'_hcl(?=[^a-zA-Z]|$)', '_hydrochloride', text, flags=re.IGNORECASE)


def clean_unnest_patterns(text: str) -> str:
    """
    Extract and clean inner drug names from '{unnest:...}' or similar patterns.

    Args:
        text (str): Raw drug name possibly containing '{{unnest:...}}'

    Returns:
        str: Cleaned text with the inner value extracted and outer braces removed
    """
    if pd.isna(text):
        return text
    cleaned = re.sub(r"{[^}]*:\s*([^}]+)}", r"\1", text)
    return cleaned


def clean_drug_name_simple(drug_name: str) -> str:
    """
    Apply all drug name cleansing operations to a single drug name.
    
    Args:
        drug_name (str): The drug name to clean
        
    Returns:
        str: The cleaned drug name
    """
    if pd.isna(drug_name) or drug_name is None:
        return drug_name

    cleaned = str(drug_name)
    cleaned = clean_unnest_patterns(cleaned)
    cleaned = cleaned.lower()
    cleaned = replace_special_chars(cleaned)
    cleaned = re.sub(r'[^a-z0-9\s_]', '', cleaned)
    cleaned = re.sub(r'\s+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = re.sub(r'^_+|_+$', '', cleaned)
    cleaned = standardize_hcl(cleaned)
    return cleaned


def clean_drug_name_column(df: pd.DataFrame, drug_name_col: str = 'drug_name') -> pd.DataFrame:
    """
    Clean the drug_name column in a DataFrame by applying all cleansing operations.
    
    Args:
        df (pd.DataFrame): DataFrame containing the drug_name column
        drug_name_col (str): Name of the column containing drug names (default: 'drug_name')
        
    Returns:
        pd.DataFrame: DataFrame with cleaned drug_name column
    """
    if drug_name_col not in df.columns:
        raise ValueError(f"Column '{drug_name_col}' not found in DataFrame")

    df_cleaned = df.copy()
    df_cleaned[drug_name_col] = df_cleaned[drug_name_col].apply(clean_drug_name_simple)
    return df_cleaned
