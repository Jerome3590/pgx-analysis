"""
Drug name cleaning and mapping utilities.
"""

import sys
import os
import re
from typing import Any, Optional, Union
import logging
import pandas as pd
import string


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


def encode_drug_name(drug_name: str, fpgrowth_metrics: dict, logger: Optional[logging.Logger] = None) -> str:
    """
    Encode a drug name with its linguistic properties and FpGrowth metrics.

    Format:
    - First letter index (3-digit)
    - Length (2-digit)
    - Syllables (2-digit)
    - Consonants (2-digit)
    - Support (4-digit, scaled ×1000)
    - Confidence (4-digit, scaled ×1000)
    - Certainty (4-digit, scaled ×1000)
    Total: 3 + 2 + 2 + 2 + 4 + 4 + 4 = 21 characters
    """
    try:
        clean_name = clean_drug_name(drug_name, logger)

        # Get first character index in name (A=1, B=2, ..., Z=26, fallback 0)
        first_letter = next((ch.upper() for ch in clean_name if ch.isalpha()), 'X')
        first_index = str(ord(first_letter) - ord('A') + 1).zfill(3) if first_letter != 'X' else '000'

        # Basic linguistic features
        length = str(len(clean_name)).zfill(2)
        syllables = str(count_syllables(clean_name)).zfill(2)
        consonants = str(len(re.findall(r'[BCDFGHJKLMNPQRSTVWXYZ]', clean_name.upper()))).zfill(2)

        # Scale and format metrics
        support = str(int(fpgrowth_metrics.get('support', 0) * 1000)).zfill(4)
        confidence = str(int(fpgrowth_metrics.get('confidence', 0) * 1000)).zfill(4)
        certainty = str(int(fpgrowth_metrics.get('certainty', 0) * 1000)).zfill(4)

        return f"{first_index}{length}{syllables}{consonants}{support}{confidence}{certainty}"
    
    except Exception as e:
        if logger:
            logger.error(f"Error encoding drug name '{drug_name}': {str(e)}")
        return "0000000000000000000"  # default fallback


def parse_encoding(encoding: str) -> List[int]:
    """Split a 21-character drug encoding into numeric segments."""
    if len(encoding) != 21:
        raise ValueError(f"Encoding must be 21 characters long, got {len(encoding)}: {encoding}")
    
    return [
        int(encoding[0:3]),   # first letter index (1-26)
        int(encoding[3:5]),   # length
        int(encoding[5:7]),   # syllables
        int(encoding[7:9]),   # consonants
        int(encoding[9:13]),  # support (×1000)
        int(encoding[13:17]), # confidence (×1000)
        int(encoding[17:21])  # certainty (×1000)
    ]


def encode_pattern_numeric(pattern: Union[set, list], encoding_map: Dict[str, str]) -> List[float]:
    """Numerically encode a pattern by summing parsed encodings of all drugs."""
    numeric_vector = [0.0] * 7
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

