"""
Feature Utilities for Feature Importance EDA

Shared utility functions for feature name handling, normalization, and extraction.
Used by Step 3b Feature Importance EDA scripts.
"""

from typing import Set, Tuple
import pandas as pd
import re


def categorize_feature(feature: str) -> Tuple[str, str]:
    """
    Categorize a feature by type and extract the code.
    
    Args:
        feature: Feature name (e.g., "item_icd_F1120", "item_cpt_80307", "item_drug_SUBOXONE")
    
    Returns:
        Tuple of (type, code) where type is one of: 'ICD', 'CPT', 'Drug', 'Unknown'
    """
    if feature.startswith('item_icd_'):
        code = feature.replace('item_icd_', '')
        return ('ICD', code)
    elif feature.startswith('item_cpt_'):
        code = feature.replace('item_cpt_', '')
        return ('CPT', code)
    elif feature.startswith('item_drug_'):
        drug = feature.replace('item_drug_', '')
        return ('Drug', drug)
    else:
        return ('Unknown', feature)


def normalize_feature_name(feature: str) -> str:
    """
    Normalize feature name to match aggregated importance format.
    
    Aggregated importance uses: item_80307, item_SUBOXONE, item_F1120
    Safe filter uses: item_cpt_80307, item_drug_SUBOXONE, item_icd_F1120
    
    This function converts from safe filter format to aggregated importance format.
    
    Args:
        feature: Feature name in any format
    
    Returns:
        Normalized feature name (item_XXXX format)
    """
    if not feature.startswith('item_'):
        return feature
    
    # Remove item_ prefix
    code = feature[5:]
    
    # Check if it has type prefix (item_cpt_, item_drug_, item_icd_)
    if code.startswith('cpt_'):
        # item_cpt_80307 -> item_80307
        return f"item_{code[4:]}"
    elif code.startswith('drug_'):
        # item_drug_SUBOXONE -> item_SUBOXONE
        return f"item_{code[5:]}"
    elif code.startswith('icd_'):
        # item_icd_F1120 -> item_F1120
        return f"item_{code[4:]}"
    else:
        # Already in normalized format (item_80307)
        return feature


def normalize_feature_set(features: Set[str]) -> Set[str]:
    """
    Normalize a set of feature names.
    
    Args:
        features: Set of feature names in any format
    
    Returns:
        Set of normalized feature names
    """
    return {normalize_feature_name(f) for f in features}


def extract_features_from_traces(traces_df: pd.DataFrame) -> Set[str]:
    """
    Extract unique feature names from BupaR traces.
    
    Traces contain activity sequences like "ICD:F1120", "CPT:80307", "DRUG:SUBOXONE"
    
    Args:
        traces_df: DataFrame with 'trace' column containing activity sequences
    
    Returns:
        Set of feature names (e.g., {"item_icd_F1120", "item_cpt_80307", "item_drug_SUBOXONE"})
    """
    features = set()
    
    if traces_df.empty or 'trace' not in traces_df.columns:
        return features
    
    for trace in traces_df['trace']:
        if pd.isna(trace):
            continue
        
        # Split trace by separator (typically ">>" or ",")
        activities = str(trace).replace('>>', ',').split(',')
        for activity in activities:
            activity = activity.strip()
            if ':' in activity:
                # Extract code/drug name after prefix (ICD:, CPT:, DRUG:)
                parts = activity.split(':', 1)
                if len(parts) == 2:
                    prefix, code = parts
                    # Store as feature name (e.g., "item_icd_80307", "item_drug_SUBOXONE")
                    if prefix.upper() == 'ICD':
                        features.add(f"item_icd_{code.strip()}")
                    elif prefix.upper() == 'CPT':
                        features.add(f"item_cpt_{code.strip()}")
                    elif prefix.upper() == 'DRUG':
                        features.add(f"item_drug_{code.strip()}")
    
    return features


def extract_features_from_patient_features(features_df: pd.DataFrame) -> Set[str]:
    """
    Extract feature names from BupaR patient features DataFrame.
    
    Looks for columns that represent feature counts or indicators.
    
    Args:
        features_df: DataFrame with patient features
    
    Returns:
        Set of feature names found in the DataFrame
    """
    features = set()
    
    if features_df.empty:
        return features
    
    # Look for columns that might contain feature information
    # This is a simplified approach - may need adjustment based on actual BupaR output format
    for col in features_df.columns:
        if 'feature' in col.lower() or 'item' in col.lower():
            # If column contains feature names
            if features_df[col].dtype == 'object':
                features.update(features_df[col].dropna().unique())
    
    return features


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spaces and special characters in feature names with underscores.
    
    Args:
        df: DataFrame with potentially problematic feature names
    
    Returns:
        DataFrame with sanitized feature names
    """
    df = df.copy()
    
    # If DataFrame has an index with feature names, sanitize it
    if df.index.name or any('item' in str(idx).lower() for idx in df.index[:10] if len(df) > 0):
        df.index = [re.sub(r'[^a-zA-Z0-9_]', '_', str(idx)) for idx in df.index]
        df.index = [re.sub(r'_+', '_', str(idx)) for idx in df.index]
        df.index = [str(idx).strip('_') for idx in df.index]
    
    # If DataFrame has a column with feature names, sanitize it
    feature_cols = [col for col in df.columns if 'feature' in col.lower() or 'item' in col.lower()]
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '_', str(x)) if pd.notna(x) else x)
            df[col] = df[col].apply(lambda x: re.sub(r'_+', '_', str(x)) if pd.notna(x) else x)
            df[col] = df[col].apply(lambda x: str(x).strip('_') if pd.notna(x) else x)
    
    return df
