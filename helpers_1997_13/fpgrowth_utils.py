"""
FpGrowth algorithm utilities for feature engineering.
"""
import os
import sys
from typing import Any, Optional, Dict, List, Union
import logging
import json
import time
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import re
from multiprocessing import Process, Queue
import numpy as np


# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)


# Import utility modules
from helpers_1997_13.common_imports import (
    s3_client,
    S3_BUCKET,
    List,
    ClientError
)

from helpers_1997_13.constants import (
    TOP_K,
    MIN_SUPPORT_THRESHOLD,
    MIN_SUPPORT_FINAL,
    MAX_ATTEMPTS,
    TIMEOUT_SECONDS,
    MIN_CONFIDENCE_SMALL,
    MIN_CONFIDENCE_MEDIUM,
    MIN_CONFIDENCE_LARGE,
    MIN_LIFT_SMALL,
    MIN_LIFT_MEDIUM,
    MIN_LIFT_LARGE,
    MIN_SUPPORT_RULE,
    FALLBACK_DELTA,
    MIN_FALLBACK_CONFIDENCE,
    MIN_FALLBACK_LIFT,
    EXCLUDED_CODES,
    METRIC_COLUMNS,
    MAX_PATTERN_COLUMNS
)

from helpers_1997_13.data_utils import (
    clean_rules_dataframe,
    safe_mean,
    is_excluded_code
)

from helpers_1997_13.drug_name_utils import (
    clean_drug_name,
    encode_drug_name,
    encode_pattern_numeric,
    save_drug_encoding_map
)

from helpers_1997_13.s3_utils import (
    get_output_paths,
    parse_s3_path,
    save_to_s3_parquet,
    save_to_s3_json,
    load_from_s3_json
)

from helpers_1997_13.visualization_utils import create_network_visualization


def fpgrowth_with_timeout(token_df, min_support=0.025, timeout=300):
    def run_fpgrowth(queue):
        from mlxtend.frequent_patterns import fpgrowth  # ensure in subprocess
        try:
            result = fpgrowth(token_df, min_support=min_support, use_colnames=True)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    queue = Queue()
    p = Process(target=run_fpgrowth, args=(queue,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        return None  # timed out

    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


def generate_placeholder_rule_columns():
    return pd.DataFrame(columns=[
        "antecedents", "consequents",
        "support", "confidence", "lift", "leverage", "conviction",
        "certainty"
    ])



def convert_frozensets(obj: Any, logger: Optional[logging.Logger] = None) -> Any:
    """Recursively convert frozensets in a nested structure to lists."""
    try:
        if isinstance(obj, frozenset):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_frozensets(v, logger) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_frozensets(item, logger) for item in obj]
        return obj

    except Exception as e:
        msg = f"✗ Error converting frozensets: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise


def save_feature_artifacts(df: pd.DataFrame, itemsets: pd.DataFrame, rules: pd.DataFrame, 
                         manifest: Dict, paths: Dict, logger: Optional[logging.Logger] = None) -> None:
    """Save feature engineering artifacts to S3.
    
    Args:
        df: Enhanced DataFrame with features
        itemsets: DataFrame containing frequent itemsets
        rules: DataFrame containing association rules
        manifest: Dictionary containing processing metadata
        paths: Dictionary of S3 paths
        logger: Optional logger for error tracking
    """
    try:
        # Save enhanced DataFrame
        save_to_s3_parquet(df, paths['fpgrowth_features'], logger)
        
        # Save itemsets
        save_to_s3_parquet(itemsets, paths['itemsets'], logger)
        
        # Save rules
        save_to_s3_parquet(rules, paths['rules'], logger)
        
        # Save manifest
        save_to_s3_json(manifest, paths['manifest'], logger)
        
        if logger:
            logger.info(f"Saved feature artifacts for {manifest['cohort_name']}/{manifest['age_band']}/{manifest['event_year']}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error saving feature artifacts: {str(e)}")
        raise


def save_drug_encoding_map(
    drug_encodings: dict,
    cohort_name: str,
    age_band: str,
    event_year: str,
    s3_bucket: str = "pgxdatalake",
    logger: Optional[logging.Logger] = None
) -> bool:
    try:
        # Convert to DataFrame
        df = pd.DataFrame(list(drug_encodings.items()), columns=['drug_name', 'numeric_encoding'])

        partitions = f"cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}"
        parquet_path = f"s3://{s3_bucket}/drug_encoding_parquet/{partitions}/drug_encoding.parquet"
        json_path = f"s3://{s3_bucket}/drug_encoding_json/{partitions}/drug_encoding.json"

        save_to_s3_parquet(df, parquet_path, logger=logger)
        save_to_s3_json(drug_encodings, json_path, logger=logger)

        if logger:
            logger.info(f"✓ Successfully saved drug encoding map to {parquet_path}")
        return True

    except Exception as e:
        msg = f"✗ Error saving drug encoding map: {str(e)}"
        if logger:
            logger.error(msg)
        return False


def load_feature_manifest(cohort_name: str, age_band: str, event_year: str) -> Dict:
    """Load feature manifest from S3.
    
    Args:
        cohort_name: Name of the cohort
        age_band: Age band
        event_year: Event year
        
    Returns:
        Dictionary containing feature manifest
    """
    try:
        # Load from S3
        manifest_path = f"s3://pgxdatalake/features/{cohort_name}/{age_band}/{event_year}/manifest.json"
        manifest = load_from_s3_json(manifest_path)
        
        return manifest
        
    except Exception as e:
        print(f"Error loading feature manifest: {str(e)}")
        raise


def extract_tokens(row, token_type="drug", logger=None):
    tokens = []
    
    if token_type != "drug":
        if logger:
            logger.warning(f"Unsupported token type: {token_type}")
        return tokens

    drug_name = row.get('drug_name')
    if pd.notnull(drug_name):
        # drug_name is already cleaned
        parts = re.split(r"[ ,;]+", drug_name.strip())
        tokens = [f"drug_{drug}" for drug in parts if drug]

    return tokens


def extract_features(name: str) -> List[str]:
    try:
        # Split name into words
        words = name.lower().split()
        
        # Extract features
        features = []
        for word in words:
            # Remove special characters
            word = ''.join(c for c in word if c.isalnum())
            if word:
                features.append(word)
                
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise


def create_encoding(drug: str, features: List[str], fp_rules: Dict) -> str:
    """Create encoding for a drug based on its features and FpGrowth rules.
    
    Args:
        drug: Drug name
        features: List of features extracted from drug name
        fp_rules: Dictionary of FpGrowth rules
        
    Returns:
        Encoded value for the drug
    """
    try:
        # Create encoding
        encoding = []
        for feature in features:
            if feature in fp_rules:
                encoding.append(fp_rules[feature])
                
        return '_'.join(sorted(encoding))
        
    except Exception as e:
        print(f"Error creating encoding: {str(e)}")
        raise


def generate_rules_from_itemsets(itemsets_df, metric="confidence", min_threshold=0.01):
    """
    Generate association rules from a set of frequent itemsets.

    Args:
        itemsets_df (pd.DataFrame): DataFrame with 'itemsets' and 'support' columns.
        metric (str): Metric to evaluate if rule is of interest. e.g., "confidence".
        min_threshold (float): Minimal threshold for the evaluation metric.

    Returns:
        pd.DataFrame: Association rules with antecedents, consequents, and metrics.
    """
    if 'itemsets' not in itemsets_df or 'support' not in itemsets_df:
        raise ValueError("itemsets_df must contain 'itemsets' and 'support' columns")

    # Ensure itemsets are in frozenset format for mlxtend compatibility
    itemsets_df = itemsets_df.copy()
    itemsets_df['itemsets'] = itemsets_df['itemsets'].apply(lambda x: frozenset(x) if not isinstance(x, frozenset) else x)

    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            rules = association_rules(itemsets_df, metric=metric, min_threshold=min_threshold)
        except KeyError:
            # Fallback when antecedent/consequent supports are missing in the pruned itemsets
            rules = association_rules(itemsets_df, metric=metric, min_threshold=min_threshold, support_only=True)
        rules = rules.fillna(0)

    # Optional cleanup
    if rules.empty:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return rules
        

def check_collisions(encoded_values: Dict[str, str]) -> Dict[str, List[str]]:
    """Check for collisions in encoded values.
    
    Args:
        encoded_values: Dictionary mapping drug names to their encoded values
        
    Returns:
        Dictionary mapping encoded values to lists of drugs that share that encoding
    """
    try:
        # Check for collisions
        collisions = {}
        for drug, encoding in encoded_values.items():
            if encoding in collisions:
                collisions[encoding].append(drug)
            else:
                collisions[encoding] = [drug]
                
        # Filter out non-collisions
        return {k: v for k, v in collisions.items() if len(v) > 1}
        
    except Exception as e:
        print(f"Error checking collisions: {str(e)}")
        raise


def parse_drug_rule(encoded_value: str) -> Dict[str, Any]:
   
    try:
        # Split encoded value into components
        components = encoded_value.split('_')
        
        # Parse components
        rule = {
            'support': float(components[0]),
            'confidence': float(components[1]),
            'lift': float(components[2])
        }
        
        return rule
        
    except Exception as e:
        print(f"Error parsing drug rule: {str(e)}")
        raise
        

def validate_drug_rule(rule: Dict[str, Any], business_rules: Dict[str, Any]) -> bool:
    """Validate a drug rule against business rules.
    
    Args:
        rule: Dictionary containing rule information
        business_rules: Dictionary containing business rules
        
    Returns:
        Boolean indicating if rule is valid
    """
    try:
        # Check support threshold
        if rule['support'] < business_rules['min_support']:
            return False
            
        # Check confidence threshold
        if rule['confidence'] < business_rules['min_confidence']:
            return False
            
        # Check lift threshold
        if rule['lift'] < business_rules['min_lift']:
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating drug rule: {str(e)}")
        raise 


def merge_pattern_metrics(df, pattern_lookup, i, logger=None):
    col = f"pattern_{i}"
    df = df.merge(
        pattern_lookup.add_suffix(f"_{i}"),
        left_on=col,
        right_on=f"pattern_id_{i}",
        how="left"
    )
    if logger and df[[f"support_{i}", f"confidence_{i}", f"lift_{i}", f"certainty_{i}"\
]].isnull().any().any():
        logger.warning(f"Missing pattern metrics in some rows for pattern_{i}")
    return df


def expand_pattern_metrics(df, itemsets_df, rules_df, cohort_name, band, year, s3_bucket, encoding_map, logger=None):
    df = df.copy()
    slot_prefix = 'pattern_'
    max_patterns = 15
    encoding_dim = 7  # includes first letter position index

    slot_assignments = []
    pattern_manifest = []
    METRIC_COLUMNS = ['support', 'confidence', 'certainty']  # customize if needed

    # === Collect actual patterns ===
    for idx, row in itemsets_df.head(max_patterns).iterrows():
        items = frozenset(row['itemsets'])
        support = row['support']
        slot = f"{slot_prefix}{idx + 1}"
        numeric_encoding = encode_pattern_numeric(items, encoding_map)
        slot_assignments.append((slot, items, numeric_encoding))
        pattern_manifest.append({
            'slot': slot,
            'items': sorted(items),
            'support': support,
            'encoding': numeric_encoding
        })

    # === Pad up to max_patterns ===
    while len(slot_assignments) < max_patterns:
        slot_idx = len(slot_assignments) + 1
        slot = f"{slot_prefix}{slot_idx}"
        empty_pattern = frozenset()
        empty_encoding = [0.0] * encoding_dim
        slot_assignments.append((slot, empty_pattern, empty_encoding))
        pattern_manifest.append({
            'slot': slot,
            'items': [],
            'support': 0.0,
            'encoding': empty_encoding
        })

    # === Preallocate Encoding + Metric Columns ===
    encoding_cols = {}
    fill_cols = []

    for slot, pattern_set, pattern_encoding in slot_assignments:
        mask = df['drug_tokens'].apply(lambda toks: pattern_set.issubset(set(toks)))

        for j in range(encoding_dim):
            col = f"{slot}_enc_{j}"
            encoding_cols[col] = mask.apply(lambda matched: pattern_encoding[j] if matched else 0.0).tolist()
            fill_cols.append(col)

        # Initialize rule metrics with zero
        for metric in METRIC_COLUMNS:
            col = f"{slot}_{metric}"
            encoding_cols[col] = [0.0] * len(df)
            fill_cols.append(col)

    # === Compute Rule-Based Metrics ===
    rule_metrics = {}
    if rules_df is not None and not rules_df.empty:
        for metric in METRIC_COLUMNS:
            if metric not in rules_df.columns:
                if logger:
                    logger.warning(f"Missing metric column: {metric} in rules_df")
                continue

            for slot, pattern_set, _ in slot_assignments:
                try:
                    filtered = rules_df[
                        rules_df['antecedents'].apply(
                            lambda x: set(x) == pattern_set if isinstance(x, (list, set, frozenset)) else set([x]) == pattern_set
                        ) |
                        rules_df['consequents'].apply(
                            lambda x: set(x) == pattern_set if isinstance(x, (list, set, frozenset)) else set([x]) == pattern_set
                        )
                    ]
                    value = filtered[metric].mean() if not filtered.empty else 0.0
                    rule_metrics.setdefault(f"{slot}_{metric}", []).append(value)

                    col = f"{slot}_{metric}"
                    mask = [v > 0 for v in encoding_cols[f"{slot}_enc_0"]]
                    encoding_cols[col] = [value if m else 0.0 for m in mask]
                except Exception as e:
                    if logger:
                        logger.warning(f"Error computing metric {metric} for {slot}: {e}")
                    rule_metrics.setdefault(f"{slot}_{metric}", []).append(0.0)
    else:
        if logger:
            logger.warning("Rules contain only fallback — rule-based metrics will be skipped")

    # === Enforce column order ===
    ordered_encoding_cols = [
        f"{slot_prefix}{i+1}_enc_{j}"
        for i in range(max_patterns)
        for j in range(encoding_dim)
    ]

    ordered_metric_cols = [
        f"{slot_prefix}{i+1}_{metric}"
        for i in range(max_patterns)
        for metric in METRIC_COLUMNS
    ]

    ordered_cols = ordered_encoding_cols + ordered_metric_cols

    # Ensure all expected columns exist (pad with zeros if needed)
    for col in ordered_cols:
        if col not in encoding_cols:
            encoding_cols[col] = [0.0] * len(df)

    # === Merge all new columns at once ===
    df = pd.concat([df, pd.DataFrame({col: encoding_cols[col] for col in ordered_cols})], axis=1)

    # === Fill and format ===
    with pd.option_context("future.no_silent_downcasting", True):
        df[ordered_encoding_cols] = df[ordered_encoding_cols].fillna(0.0).astype(float)
        df[ordered_metric_cols] = df[ordered_metric_cols].fillna(0.0).infer_objects(copy=False)

    # === Save manifest ===
    manifest = {
        'patterns': pattern_manifest,
        'rule_metrics': rule_metrics,
        'cohort': cohort_name,
        'band': band,
        'year': year
    }

    return df, manifest


def run_fpgrowth_drug_token_with_fallback(
    grouped_df: pd.DataFrame,
    cohort_name: str,
    age_band: str,
    event_year: str,
    paths: dict,
    logger: Optional[logging.Logger] = None,
    support_start: float = None,
    TOP_K: int = 30,
    min_confidence: float = 0.3,
    timeout: int = 300
) -> tuple:
    """
    Run FP-Growth with fallback logic for min_support threshold.
    Tries decreasing min_support values on failure or empty result.
    Returns: (features_df, itemsets, rules)
    """
    if support_start is None:
        support_start = MIN_SUPPORT_THRESHOLD
    # Compose fallback thresholds: start high, decrease to MIN_SUPPORT_FINAL
    thresholds = [support_start]
    # Add decreasing steps down to MIN_SUPPORT_FINAL (exclusive)
    delta = FALLBACK_DELTA if FALLBACK_DELTA > 0 else 0.005
    current = support_start - delta
    while current >= MIN_SUPPORT_FINAL:
        thresholds.append(round(current, 5))
        current -= delta
    # Ensure MIN_SUPPORT_FINAL is included
    if MIN_SUPPORT_FINAL not in thresholds:
        thresholds.append(MIN_SUPPORT_FINAL)
    # Remove duplicates and sort descending
    thresholds = sorted(set(thresholds), reverse=True)

    for attempt, min_support in enumerate(thresholds, 1):
        if logger:
            logger.info(f"[FP-Growth] Attempt {attempt}: min_support={min_support}")
        try:
            # TransactionEncoder: convert tokens to one-hot
            te = TransactionEncoder()
            te_ary = te.fit_transform(grouped_df['drug_tokens'])
            token_df = pd.DataFrame(te_ary, columns=te.columns_)
            # Run fpgrowth with timeout
            itemsets = fpgrowth_with_timeout(token_df, min_support=min_support, timeout=timeout)
            if itemsets is None:
                if logger:
                    logger.warning(f"FP-Growth timed out at min_support={min_support}")
                continue
            if itemsets.empty:
                if logger:
                    logger.warning(f"FP-Growth found no itemsets at min_support={min_support}")
                continue
            # Limit to top K itemsets by support
            itemsets = itemsets.sort_values("support", ascending=False).head(TOP_K)
            # Generate rules
            rules = generate_rules_from_itemsets(itemsets, metric="confidence", min_threshold=min_confidence)
            # Clean rules if needed
            if 'clean_rules_dataframe' in globals():
                rules = clean_rules_dataframe(rules)
            # If rules are empty, fallback
            if rules is None or rules.empty:
                if logger:
                    logger.warning(f"FP-Growth found no rules at min_support={min_support}")
                continue
            # Compose features DataFrame (optional, can be expanded)
            features_df = grouped_df.copy()
            # Success: return
            if logger:
                logger.info(f"FP-Growth succeeded at min_support={min_support} with {len(itemsets)} itemsets and {len(rules)} rules.")
            return features_df, itemsets, rules
        except Exception as e:
            if logger:
                logger.error(f"FP-Growth error at min_support={min_support}: {e}")
            continue
    # If all attempts fail, return empty results
    if logger:
        logger.error(f"FP-Growth failed for all min_support thresholds: {thresholds}")
    return grouped_df.copy(), pd.DataFrame(), pd.DataFrame()

def generate_placeholder_rule_columns():
    return pd.DataFrame(columns=[
        "antecedents", "consequents",
        "support", "confidence", "lift", "leverage", "conviction",
        "certainty"  # If you calculate certainty later
    ])

