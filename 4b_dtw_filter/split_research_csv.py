"""
Split the ICD/CPT/HCPCS research CSV into multiple files with 100 rows each.
"""
import pandas as pd
from pathlib import Path

def split_csv(input_path: Path, output_dir: Path, rows_per_file: int = 100):
    """
    Split a CSV file into multiple files with specified rows per file.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save split files
        rows_per_file: Number of data rows per file (header included in each)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading CSV file: {input_path}")
    df = pd.read_csv(input_path)
    
    total_rows = len(df)
    num_files = (total_rows + rows_per_file - 1) // rows_per_file  # Ceiling division
    
    print(f"Total rows: {total_rows}")
    print(f"Rows per file: {rows_per_file}")
    print(f"Number of files to create: {num_files}")
    print()
    
    base_name = input_path.stem
    extension = input_path.suffix
    
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)
        
        chunk_df = df.iloc[start_idx:end_idx]
        
        output_filename = f"{base_name}_part_{i+1:03d}{extension}"
        output_path = output_dir / output_filename
        
        chunk_df.to_csv(output_path, index=False)
        
        print(f"[{i+1}/{num_files}] Saved: {output_filename} (rows {start_idx+1}-{end_idx})")
    
    print()
    print(f"[OK] Split complete. Created {num_files} files in: {output_dir}")

if __name__ == "__main__":
    input_path = Path("4b_dtw_filter/outputs/code_research/icd_cpt_hcpcs_codes_research.csv")
    output_dir = Path("4b_dtw_filter/outputs/code_research/split_files")
    
    split_csv(input_path, output_dir, rows_per_file=100)
