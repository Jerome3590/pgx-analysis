import duckdb
import pandas as pd
from typing import Dict
import json
from pathlib import Path

def create_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection."""
    return duckdb.connect('drug_data.duckdb')

def load_pharmacy_data(conn: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    """Load pharmacy data from DuckDB table."""
    return conn.execute(f"SELECT * FROM {table_name}").df()

def sort_combination(drug_name: str) -> str:
    """Sort drug combinations alphabetically."""
    parts = drug_name.split('+')
    return '+'.join(sorted(parts))

def normalize_drug_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize drug names in the dataframe."""
    # Convert to lowercase
    df['standardized_drug_name'] = df['drug_name'].str.lower()
    
    # Remove trailing slashes
    df['standardized_drug_name'] = df['standardized_drug_name'].str.replace(r'/$', '', regex=True)
    
    # Replace spaces with underscores
    df['standardized_drug_name'] = df['standardized_drug_name'].str.replace(' ', '_')
    
    # Replace '/' with '+'
    df['standardized_drug_name'] = df['standardized_drug_name'].str.replace('/', '+')
    
    # Sort combinations if they contain '+'
    mask = df['standardized_drug_name'].str.contains('+', regex=False)
    df.loc[mask, 'standardized_drug_name'] = df.loc[mask, 'standardized_drug_name'].apply(sort_combination)
    
    return df

def filter_medical_supplies(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out medical supplies and mark as 'not_drug'."""
    medical_supplies = [
        'accu-chek', 'knee_brace', 'lancet', 'syringe', 'needle', 'test_strip', 'monitor',
        'lancing_device', 'insulin_pump', 'glucose_meter', 'blood_glucose', 'nebulizer',
        'inhaler', 'spacer', 'chamber', 'compressor', 'catheter', 'dressing', 'bandage',
        'gauze', 'tape', 'alcohol_prep', 'alcohol_swab', 'pen_needle', 'aerochamber', 'onetouch',
        'optichamber', 'infusion_set', 'breast', 'aerosol', 'control_solution', 'swab', 'pad', 
        'wipes', 'mask', 'peak_flow','humidifier', 'compressor', 'table_top', 'transmitter', 'sensor',
        'receiver', 'sharps', 'container','safety_syr', 'autoshield', 'collector', 'reservoir', 'meter',
        'test', 'strip', 'solution', 'reservoir', 'pen_tip', 'pen_needl', 'lancin', 'glucometer', 'flow_meter',
        'feeding', 'pump', 'tuberculin', 'piston', 'enema', 'walker', 'rollator', 'underwear', 'diaper', 
        'support', 'brace', 'peak', 'flow', 'compressor', 'nebulizer', 'aerosol', 'mask', 'humidifier',
        'chamber', 'holding', 'valved', 'device', 'container', 'collector', 'pad', 'dressing', 'bandage',
        'tape', 'gauze', 'wipes', 'swab', 'alcohol', 'prep', 'pen', 'needle', 'syringe', 'injection',
        'reservoir', 'cartridge', 'strip', 'test', 'meter', 'monitor', 'glucose',
        'blood_pressure', 'bp', 'thermometer', 'feeding', 'pump', 'breast', 'milk', 'storage'
    ]
    
    for item in medical_supplies:
        mask = df['standardized_drug_name'].str.contains(item, case=False, na=False)
        df.loc[mask, 'standardized_drug_name'] = 'not_drug'
    
    return df

def load_drug_mappings() -> Dict[str, Dict[str, str]]:
    """Load all drug mapping files from the drug_mappings directory."""
    mappings = {}
    mappings_dir = Path('drug_mappings')
    
    for mapping_file in mappings_dir.glob('*_mappings.json'):
        letter = mapping_file.stem[0]  # Get first letter from filename (e.g., 'a' from 'a_mappings.json')
        with open(mapping_file, 'r') as f:
            mappings[letter] = json.load(f)
    
    return mappings

def apply_drug_mappings(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Apply drug name mappings to standardize drug names."""
    # Create a combined mapping dictionary
    combined_mappings = {}
    for letter_mappings in mappings.values():
        combined_mappings.update(letter_mappings)
    
    # Apply mappings
    df['standardized_drug_name'] = df['standardized_drug_name'].map(combined_mappings).fillna(df['standardized_drug_name'])
    return df

def save_normalized_data(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, output_dir: str):
    """
    Save normalized data as partitioned Parquet files using DuckDB.
    The data will be partitioned by 'age_band' and 'event_year'.
    """
    # Register the DataFrame as a DuckDB view
    conn.register('normalized_df', df)
    
    # Optional: sort the data before writing (DuckDB will handle partitioning)
    conn.execute("""
        CREATE OR REPLACE TABLE sorted_df AS
        SELECT *
        FROM normalized_df
        ORDER BY age_band, event_year, mi_person_key, event_date
    """)
    
    # Write partitioned Parquet files
    conn.execute(f"""
        COPY sorted_df TO '{output_dir}'
        (FORMAT PARQUET, PARTITION_BY (age_band, event_year), OVERWRITE_OR_IGNORE 1)
    """)

def main():
    # Create connection
    conn = create_connection()
    
    # Load data
    df = load_pharmacy_data(conn, 'pharmacy')
    
    # Drop rows with missing drug_name
    df = df.dropna(subset=['drug_name'])
    df = df[df['drug_name'] != '']
    
    # Normalize drug names
    df = normalize_drug_names(df)
    
    # Filter medical supplies
    df = filter_medical_supplies(df)
    
    # Load and apply drug mappings
    drug_mappings = load_drug_mappings()
    df = apply_drug_mappings(df, drug_mappings)
    
    # Save normalized data
    save_normalized_data(conn, df, 'pharmacy-clean-drug-names/')
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main() 