"""
Feature Utilities for Feature Importance EDA

Shared utility functions for feature name handling, normalization, and extraction.
Used by Step 3b Feature Importance EDA scripts.

Following cursor dev rules: Prefer DuckDB and Parquet over pandas DataFrames.
"""

from typing import Set, Tuple, Union
import re
import duckdb
from pathlib import Path


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


def read_csv_with_duckdb(csv_path: Union[str, Path]) -> duckdb.DuckDBPyRelation:
    """
    Read CSV file using DuckDB (preferred over pandas).
    
    Following cursor dev rules: Prefer DuckDB and Parquet over pandas DataFrames.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DuckDB relation (can be converted to DataFrame with .df() if needed)
    """
    con = duckdb.connect()
    csv_path_str = str(csv_path).replace("'", "''")
    return con.execute(f"SELECT * FROM read_csv_auto('{csv_path_str}')")


def extract_features_from_traces(traces_data: Union[duckdb.DuckDBPyRelation, Path, 'pd.DataFrame']) -> Set[str]:
    """
    Extract unique feature names from BupaR traces.
    
    Traces contain activity sequences like "ICD:F1120", "CPT:80307", "DRUG:SUBOXONE"
    
    Following cursor dev rules: Prefer DuckDB and Parquet over pandas DataFrames.
    This function accepts a DuckDB relation, file path, or pandas DataFrame (for compatibility).
    
    Args:
        traces_data: DuckDB relation, Path to CSV/Parquet file, or pandas DataFrame with 'trace' column
    
    Returns:
        Set of feature names (e.g., {"item_icd_F1120", "item_cpt_80307", "item_drug_SUBOXONE"})
    """
    features = set()
    
    # Handle file path - read with DuckDB (preferred)
    if isinstance(traces_data, (str, Path)):
        con = duckdb.connect()
        file_path_str = str(traces_data).replace("'", "''")
        if str(traces_data).endswith('.parquet'):
            traces_df = con.execute(f"SELECT trace FROM read_parquet('{file_path_str}')").df()
        else:
            traces_df = con.execute(f"SELECT trace FROM read_csv_auto('{file_path_str}')").df()
        con.close()
    elif hasattr(traces_data, 'df'):
        # DuckDB relation - convert to DataFrame for iteration
        traces_df = traces_data.df()
    else:
        # Assume pandas DataFrame (for backward compatibility)
        traces_df = traces_data
    
    if traces_df.empty or 'trace' not in traces_df.columns:
        return features
    
    # Use DuckDB for string operations if possible, but for complex parsing we use pandas
    # This is acceptable per dev rules: "Only use pandas when DuckDB operations are not feasible"
    import pandas as pd
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


def extract_features_from_patient_features(features_data: Union[duckdb.DuckDBPyRelation, Path, 'pd.DataFrame']) -> Set[str]:
    """
    Extract feature names from BupaR patient features.
    
    Looks for columns that represent feature counts or indicators.
    
    Following cursor dev rules: Prefer DuckDB and Parquet over pandas DataFrames.
    This function accepts a DuckDB relation, file path, or pandas DataFrame (for compatibility).
    
    Args:
        features_data: DuckDB relation, Path to CSV/Parquet file, or pandas DataFrame with patient features
    
    Returns:
        Set of feature names found in the data
    """
    features = set()
    
    # Handle file path - read with DuckDB (preferred)
    if isinstance(features_data, (str, Path)):
        con = duckdb.connect()
        file_path_str = str(features_data).replace("'", "''")
        if str(features_data).endswith('.parquet'):
            features_df = con.execute(f"SELECT * FROM read_parquet('{file_path_str}')").df()
        else:
            features_df = con.execute(f"SELECT * FROM read_csv_auto('{file_path_str}')").df()
        con.close()
    elif hasattr(features_data, 'df'):
        # DuckDB relation - convert to DataFrame for column inspection
        features_df = features_data.df()
    else:
        # Assume pandas DataFrame (for backward compatibility)
        features_df = features_data
    
    if features_df.empty:
        return features
    
    # Look for columns that might contain feature information
    # This is a simplified approach - may need adjustment based on actual BupaR output format
    import pandas as pd
    for col in features_df.columns:
        if 'feature' in col.lower() or 'item' in col.lower():
            # If column contains feature names
            if features_df[col].dtype == 'object':
                features.update(features_df[col].dropna().unique())
    
    return features


def sanitize_feature_names(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Replace spaces and special characters in feature names with underscores.
    
    Following cursor dev rules: Prefer DuckDB and Parquet over pandas DataFrames.
    However, this function requires complex string operations that are easier with pandas.
    This is acceptable per dev rules: "Only use pandas when DuckDB operations are not feasible"
    
    Args:
        df: DataFrame with potentially problematic feature names
    
    Returns:
        DataFrame with sanitized feature names
    """
    import pandas as pd
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
