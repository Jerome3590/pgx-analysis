import os
import json
import pandas as pd
import string
import sys

# Add parent directory to path to access helpers
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.s3_utils import save_to_s3_json, load_from_s3_json, save_to_s3_csv
from helpers_1997_13.drug_utils import encode_drug_name
import logging


def clean_drug_name_robust(drug_name: str) -> str:
    """
    Robustly clean drug names to handle various edge cases.
    
    Args:
        drug_name (str): Raw drug name
        
    Returns:
        str: Cleaned drug name
    """
    if not drug_name or pd.isna(drug_name):
        return ""
    
    # Convert to string and strip whitespace
    cleaned = str(drug_name).strip()
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove leading/trailing spaces again
    cleaned = cleaned.strip()
    
    # Handle empty strings after cleaning
    if not cleaned:
        return ""
    
    return cleaned


def validate_drug_name(drug_name: str) -> bool:
    """
    Validate that a drug name is properly formatted.
    
    Args:
        drug_name (str): Drug name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not drug_name or pd.isna(drug_name):
        return False
    
    # Check if it's a string
    if not isinstance(drug_name, str):
        return False
    
    # Check if it's not empty after cleaning
    cleaned = clean_drug_name_robust(drug_name)
    if not cleaned:
        return False
    
    # Check if it contains at least one alphabetic character
    if not any(c.isalpha() for c in cleaned):
        return False
    
    # Check if it's not too long (reasonable limit)
    if len(cleaned) > 100:
        return False
    
    return True


def calculate_word_features(word, fpgrowth_metrics=None, logger=None):
    # Clean the word first - remove extra spaces and normalize
    if word:
        word = word.strip()
    
    vowels = "aeiou"
    consonants = set(string.ascii_lowercase) - set(vowels)

    # Use the same letter mapping as encode_drug_name function
    # A=111, B=222, C=333, ..., Z=880
    letter_codes = {
        'A': 111, 'B': 222, 'C': 333, 'D': 444, 'E': 555, 'F': 666, 'G': 777, 'H': 888, 'I': 999,
        'J': 101, 'K': 202, 'L': 303, 'M': 404, 'N': 505, 'O': 606, 'P': 707, 'Q': 808, 'R': 909,
        'S': 110, 'T': 220, 'U': 330, 'V': 440, 'W': 550, 'X': 660, 'Y': 770, 'Z': 880
    }
    
    # Use the same cleaning as encode_drug_name to ensure consistency
    from helpers_1997_13.drug_utils import clean_drug_name
    cleaned_word = clean_drug_name(word) if word else ""
    
    # Get first letter from cleaned word (same as encode_drug_name)
    first_letter = next((ch.upper() for ch in cleaned_word if ch.isalpha()), 'X') if cleaned_word else 'X'
    first_letter_index = letter_codes.get(first_letter, 0)
    
    # Calculate linguistic features using cleaned word
    num_vowels = sum(1 for char in cleaned_word.lower() if char in vowels)
    num_consonants = sum(1 for char in cleaned_word.lower() if char in consonants)
    word_length = len(cleaned_word)
    
    # Additional linguistic features
    num_hyphens_underscores = sum(1 for char in cleaned_word if char in ['-', '_'])
    
    # Advanced linguistic features
    from helpers_1997_13.drug_utils import count_chemical_suffixes, count_consonant_clusters, calculate_repetition_factor
    num_chemical_suffixes = count_chemical_suffixes(cleaned_word)
    num_consonant_clusters = count_consonant_clusters(cleaned_word)
    repetition_factor = calculate_repetition_factor(cleaned_word)

    # Use encode_drug_name function to get the proper encoded name
    global_encoded_name = encode_drug_name(word, fpgrowth_metrics, logger) if word else ""
    
    # Debug logging for first few drugs
    if word and (len(word) <= 10 or word in ['acetaminophen', 'albuterol', 'amlodipine_besylate']):
        logger.debug(f"Word: '{word}' -> cleaned: '{cleaned_word}', first_letter_index: {first_letter_index}, global_encoded_name: {global_encoded_name}")
        if fpgrowth_metrics:
            logger.debug(f"  Metrics passed to encode_drug_name: {fpgrowth_metrics}")
        else:
            logger.debug(f"  No metrics passed to encode_drug_name")

    return first_letter_index, num_vowels, num_consonants, word_length, num_hyphens_underscores, num_chemical_suffixes, num_consonant_clusters, repetition_factor, global_encoded_name


def calculate_trend_features_by_year(df, drug_name, filename_year):
    """
    Calculate trend slope for drug support metric using year data from filename.
    
    Args:
        df (pd.DataFrame): Drug prescription data
        drug_name (str): Name of the drug to analyze
        filename_year (int): Year from filename
    
    Returns:
        dict: Trend slope value and year information
    """
    try:
        # Filter data for specific drug
        drug_data = df[df['drug_name'] == drug_name].copy()
        
        if len(drug_data) < 1:  # Need at least 1 data point
            return {'trend': 0.0, 'year': filename_year}
        
        # Check if support column exists
        if 'support' not in drug_data.columns:
            return {'trend': 0.0, 'year': filename_year}
        
        # Get support values for this drug
        support_values = drug_data['support'].values
        
        if len(support_values) < 1:
            return {'trend': 0.0, 'year': filename_year}
        
        # Calculate average support for this drug in this year
        avg_support = support_values.mean()
        
        return {
            'trend': 0.0,  # Will be calculated across years later
            'year': filename_year,
            'avg_support': avg_support
        }
        
    except Exception as e:
        # Return default value if trend calculation fails
        return {'trend': 0.0, 'year': filename_year}


def calculate_multi_year_trends(results):
    """
    Calculate trends across multiple years for each drug.
    
    Args:
        results (list): List of DataFrames, one per year/file
    
    Returns:
        dict: Dictionary mapping drug_name to trend information
    """
    drug_trends = {}
    
    # Collect all data points for each drug across years
    for result_df in results:
        if 'drug_name' in result_df.columns and 'year' in result_df.columns and 'avg_support' in result_df.columns:
            for _, row in result_df.iterrows():
                drug_name = row['drug_name']
                year = row['year']
                avg_support = row['avg_support']
                
                # Skip if year is None or not an integer
                if year is None or not isinstance(year, int):
                    continue
                
                if drug_name not in drug_trends:
                    drug_trends[drug_name] = {'years': [], 'supports': []}
                
                drug_trends[drug_name]['years'].append(year)
                drug_trends[drug_name]['supports'].append(avg_support)
    
    # Calculate trends for each drug
    trend_results = {}
    for drug_name, data in drug_trends.items():
        years = data['years']
        supports = data['supports']
        
        # Ensure we have valid numeric data
        if not years or not supports or len(years) != len(supports):
            continue
            
        # Filter out any None values that might have slipped through
        valid_data = [(year, support) for year, support in zip(years, supports) 
                     if isinstance(year, int) and isinstance(support, (int, float))]
        
        if len(valid_data) < 2:
            # Not enough valid data points
            trend_results[drug_name] = {
                'trend': 0.0,
                'years': years,
                'supports': supports,
                'trend_direction': 'insufficient_data'
            }
            continue
            
        # Unzip valid data
        years, supports = zip(*valid_data)
        
        if len(years) >= 2:
            # Calculate trend using linear regression
            # Simple linear regression: y = mx + b
            n = len(years)
            sum_x = sum(years)
            sum_y = sum(supports)
            sum_xy = sum(year * support for year, support in zip(years, supports))
            sum_x2 = sum(year * year for year in years)
            
            # Calculate slope
            numerator = n * sum_xy - sum_x * sum_y
            denominator = n * sum_x2 - sum_x * sum_x
            
            if denominator != 0:
                slope = numerator / denominator
                trend_results[drug_name] = {
                    'trend': slope,
                    'years': years,
                    'supports': supports,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
            else:
                trend_results[drug_name] = {
                    'trend': 0.0,
                    'years': years,
                    'supports': supports,
                    'trend_direction': 'stable'
                }
        else:
            # Single year data
            trend_results[drug_name] = {
                'trend': 0.0,
                'years': years,
                'supports': supports,
                'trend_direction': 'single_year'
            }
    
    return trend_results


def filter_by_year_range(df, start_event_year=None, end_event_year=None, date_column='event_date'):
    """Filter DataFrame by year range.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        start_event_year (int, optional): Start year filter (e.g., 2020)
        end_event_year (int, optional): End year filter (e.g., 2022)
        date_column (str): Column name containing dates
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if date_column not in df.columns:
        return df  # Return original if date column doesn't exist
    
    # Convert date column to datetime if it's not already
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except:
        return df  # Return original if conversion fails
    
    # Apply year filters
    if start_event_year:
        df = df[df[date_column].dt.year >= start_event_year]
    
    if end_event_year:
        df = df[df[date_column].dt.year <= end_event_year]
    
    return df

def extract_year_from_filename(filename):
    """Extract year from filename pattern.
    
    Args:
        filename (str): Filename to extract year from
    
    Returns:
        int or None: Year if found, None if not found
    """
    import re
    
    # Look for year pattern in filename (e.g., "drug_metrics_0-12_2020.json")
    year_match = re.search(r'_(\d{4})\.json$', filename)
    if year_match:
        return int(year_match.group(1))
    
    # Look for year range pattern (e.g., "drug_metrics_0-12_2016-2017.json")
    year_range_match = re.search(r'_(\d{4})-(\d{4})\.json$', filename)
    if year_range_match:
        # Use the first year in the range
        return int(year_range_match.group(1))
    
    return None


def filter_by_filename_year(filename, start_event_year=None, end_event_year=None):
    """Filter files by year based on filename pattern.
    
    Args:
        filename (str): Filename to check
        start_event_year (int, optional): Start year filter (e.g., 2020)
        end_event_year (int, optional): End year filter (e.g., 2022)
    
    Returns:
        bool: True if file should be processed, False if it should be skipped
    """
    file_year = extract_year_from_filename(filename)
    
    if file_year is None:
        # If no year found in filename, process the file
        return True
    
    # Apply year filters
    if start_event_year and file_year < start_event_year:
        return False
    
    if end_event_year and file_year > end_event_year:
        return False
    
    return True


def process_itemsets(directory, start_event_year=None, end_event_year=None, logger=None):
    """Iterate through JSON files and process itemsets (local version).
    
    Args:
        directory (str): Directory containing JSON files
        start_event_year (int, optional): Start year filter (e.g., 2020)
        end_event_year (int, optional): End year filter (e.g., 2022)
        logger: Logger instance (optional)
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting process_itemsets with directory: {directory}")
    logger.info(f"Year filters: start_event_year={start_event_year}, end_event_year={end_event_year}")
    
    # Collect all data from all files first
    all_dataframes = []
    total_files_processed = 0

    # Check if directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    # Get list of drug_metrics JSON files only
    json_files = [f for f in os.listdir(directory) if f.startswith("drug_metrics") and f.endswith(".json")]
    logger.info(f"Found {len(json_files)} drug_metrics files in directory: {json_files}")

    # Step 1: Collect all data from all files
    for filename in json_files:
        logger.info(f"Loading file: {filename}")
        
        # Apply filename-based year filtering first
        if start_event_year or end_event_year:
            if not filter_by_filename_year(filename, start_event_year, end_event_year):
                logger.info(f"Skipping {filename} - year filter: {start_event_year} to {end_event_year}")
                continue
            else:
                logger.info(f"Processing {filename} - year filter: {start_event_year} to {end_event_year}")
        
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, "r") as file:
                itemsets = json.load(file)
            
            logger.info(f"Loaded {len(itemsets)} itemsets from {filename}")
            
            # Check if itemsets is empty or None
            if not itemsets:
                logger.warning(f"No itemsets found in {filename}, skipping...")
                continue
            
            df = pd.DataFrame(itemsets)
            logger.info(f"Created DataFrame with shape: {df.shape}, columns: {list(df.columns)}")
            
            # Clean drug names to remove extra spaces and normalize
            if 'drug_name' in df.columns:
                df['drug_name'] = df['drug_name'].apply(clean_drug_name_robust)
                
                # Log any invalid drug names
                invalid_drugs = df[~df['drug_name'].apply(validate_drug_name)]['drug_name'].unique()
                if len(invalid_drugs) > 0:
                    logger.warning(f"Found {len(invalid_drugs)} invalid drug names in {filename}: {invalid_drugs[:5]}...")
                
                # Remove invalid drug names
                df = df[df['drug_name'].apply(validate_drug_name)]
                logger.info(f"Cleaned drug names in {filename}, removed {len(invalid_drugs)} invalid entries")
            
            # Check if DataFrame is empty
            if df.empty:
                logger.warning(f"DataFrame is empty for {filename}, skipping...")
                continue
            
            # Log initial drug count BEFORE any processing
            if 'drug_name' in df.columns:
                initial_drug_count = df['drug_name'].nunique()
                logger.info(f"Initial drug count in {filename}: {initial_drug_count} unique drugs")
            else:
                logger.error(f"'drug_name' column not found in {filename}. Available columns: {list(df.columns)}")
                logger.error(f"DataFrame shape: {df.shape}")
                if len(df) > 0:
                    logger.error(f"Sample data: {df.head(1).to_dict('records')}")
                continue
            
            # Log initial data sample
            if len(df) > 0:
                logger.debug(f"Sample data from {filename}: {df.head(1).to_dict('records')}")

            # Check if drug_name column exists
            if 'drug_name' not in df.columns:
                logger.error(f"'drug_name' column not found in {filename}. Available columns: {list(df.columns)}")
                logger.error(f"DataFrame shape: {df.shape}")
                if len(df) > 0:
                    logger.error(f"Sample data: {df.head(1).to_dict('records')}")
                continue
            
            # Get unique drug names before grouping
            unique_drugs = df['drug_name'].nunique()
            logger.info(f"Found {unique_drugs} unique drugs in {filename}")
            
            # Additional safety check before groupby
            if unique_drugs == 0:
                logger.warning(f"No unique drugs found in {filename}, skipping...")
                continue
            
            # Add source information to the DataFrame
            filename_year = extract_year_from_filename(filename)
            import re
            age_band_match = re.search(r'drug_metrics_(\d+-\d+)_', filename)
            age_band = age_band_match.group(1) if age_band_match else "unknown"
            
            df['source_age_band'] = age_band
            df['source_year'] = filename_year
            df['source_filename'] = filename
            
            all_dataframes.append(df)
            total_files_processed += 1
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            logger.exception(f"Exception details for {filename}")

    logger.info(f"Collected data from {total_files_processed} files")
    
    # Calculate total unique drugs across all files
    all_unique_drugs = set()
    total_records = 0
    for df in all_dataframes:
        all_unique_drugs.update(df['drug_name'].unique())
        total_records += len(df)
    
    logger.info(f"Total unique drugs across all files: {len(all_unique_drugs)}")
    logger.info(f"Total records across all files: {total_records}")
    
    # Analyze drug distribution across groups
    logger.info("Analyzing drug distribution across groups...")
    
    # Check drug distribution by age band
    age_band_drugs = {}
    year_drugs = {}
    filename_drugs = {}
    
    for df in all_dataframes:
        if 'source_age_band' in df.columns and 'drug_name' in df.columns:
            for age_band in df['source_age_band'].unique():
                if age_band not in age_band_drugs:
                    age_band_drugs[age_band] = set()
                age_band_drugs[age_band].update(df[df['source_age_band'] == age_band]['drug_name'].unique())
        
        if 'source_year' in df.columns and 'drug_name' in df.columns:
            for year in df['source_year'].unique():
                if pd.notna(year):
                    year_str = str(year)
                    if year_str not in year_drugs:
                        year_drugs[year_str] = set()
                    year_drugs[year_str].update(df[df['source_year'] == year]['drug_name'].unique())
        
        if 'source_filename' in df.columns and 'drug_name' in df.columns:
            for filename in df['source_filename'].unique():
                if filename not in filename_drugs:
                    filename_drugs[filename] = set()
                filename_drugs[filename].update(df[df['source_filename'] == filename]['drug_name'].unique())
    
    # Log distribution analysis
    logger.info(f"Drug distribution by age band:")
    for age_band, drugs in age_band_drugs.items():
        logger.info(f"  {age_band}: {len(drugs)} unique drugs")
    
    logger.info(f"Drug distribution by year:")
    for year, drugs in year_drugs.items():
        logger.info(f"  {year}: {len(drugs)} unique drugs")
    
    logger.info(f"Drug distribution by filename:")
    for filename, drugs in filename_drugs.items():
        logger.info(f"  {filename}: {len(drugs)} unique drugs")
    
    # Check for drugs that appear in multiple groups
    all_drugs_by_group = list(age_band_drugs.values()) + list(year_drugs.values()) + list(filename_drugs.values())
    if all_drugs_by_group:
        # Find drugs that appear in multiple groups
        drug_group_count = {}
        for group_drugs in all_drugs_by_group:
            for drug in group_drugs:
                if drug not in drug_group_count:
                    drug_group_count[drug] = 0
                drug_group_count[drug] += 1
        
        # Count drugs by how many groups they appear in
        group_appearance_count = {}
        for drug, count in drug_group_count.items():
            if count not in group_appearance_count:
                group_appearance_count[count] = 0
            group_appearance_count[count] += 1
        
        logger.info(f"Drug appearance across groups:")
        for group_count, drug_count in sorted(group_appearance_count.items()):
            logger.info(f"  Drugs appearing in {group_count} group(s): {drug_count} drugs")
        
        # Show some examples of drugs that appear in multiple groups
        multi_group_drugs = [drug for drug, count in drug_group_count.items() if count > 1]
        if multi_group_drugs:
            logger.info(f"Sample drugs appearing in multiple groups:")
            for drug in multi_group_drugs[:10]:  # Show first 10
                logger.info(f"  {drug}: appears in {drug_group_count[drug]} groups")
    

    
    if not all_dataframes:
        logger.warning("No data collected from any files")
        return []
    
    # Step 2: Combine all data and perform global aggregations
    logger.info("Combining all data and performing global aggregations...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    logger.info(f"Combined DataFrame unique drugs: {combined_df['drug_name'].nunique()}")
    logger.info(f"Combined DataFrame columns: {list(combined_df.columns)}")
    
    # FP-Growth Signal Analysis
    logger.info("=== FP-Growth Signal Analysis ===")
    logger.info(f"FP-Growth filtered drugs (with signal): {len(all_unique_drugs)}")
    
    # Analyze support distribution to understand FP-Growth filtering
    if 'support' in combined_df.columns:
        support_stats = combined_df['support'].describe()
        logger.info(f"Support metric distribution:")
        logger.info(f"  Min: {support_stats['min']:.6f}")
        logger.info(f"  25%: {support_stats['25%']:.6f}")
        logger.info(f"  50%: {support_stats['50%']:.6f}")
        logger.info(f"  75%: {support_stats['75%']:.6f}")
        logger.info(f"  Max: {support_stats['max']:.6f}")
        
        # Count drugs by support ranges
        support_ranges = {
            'Very Low (<0.001)': len(combined_df[combined_df['support'] < 0.001]),
            'Low (0.001-0.01)': len(combined_df[(combined_df['support'] >= 0.001) & (combined_df['support'] < 0.01)]),
            'Medium (0.01-0.1)': len(combined_df[(combined_df['support'] >= 0.01) & (combined_df['support'] < 0.1)]),
            'High (0.1-0.5)': len(combined_df[(combined_df['support'] >= 0.1) & (combined_df['support'] < 0.5)]),
            'Very High (>=0.5)': len(combined_df[combined_df['support'] >= 0.5])
        }
        
        logger.info(f"Drugs by support level:")
        for range_name, count in support_ranges.items():
            logger.info(f"  {range_name}: {count} drugs")
    
    # Load original pharmacy_clean dataset for comparison
    try:
        logger.info("Loading original pharmacy_clean dataset for comparison...")
        import boto3
        from io import BytesIO
        
        # Load sample of pharmacy_clean data (now under gold/) to get total drug universe
        s3_client = boto3.client('s3')
        bucket = 'pgxdatalake'
        prefix = 'gold/pharmacy_clean/'
        
        # List objects in pharmacy_clean
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        pharmacy_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]
        
        if pharmacy_files:
            # Load first few files to get drug universe
            total_pharmacy_drugs = set()
            files_checked = 0
            max_files_to_check = 5  # Limit to avoid long loading time
            
            for file_key in pharmacy_files[:max_files_to_check]:
                try:
                    logger.info(f"Loading pharmacy file: {file_key}")
                    response = s3_client.get_object(Bucket=bucket, Key=file_key)
                    df_pharmacy = pd.read_parquet(BytesIO(response['Body'].read()))
                    
                    if 'drug_name' in df_pharmacy.columns:
                        file_drugs = set(df_pharmacy['drug_name'].unique())
                        total_pharmacy_drugs.update(file_drugs)
                        logger.info(f"  Found {len(file_drugs)} unique drugs in {file_key}")
                        files_checked += 1
                    else:
                        logger.warning(f"  No 'drug_name' column in {file_key}")
                        
                except Exception as e:
                    logger.warning(f"  Error loading {file_key}: {e}")
                    continue
            
            logger.info(f"=== PHARMACY CLEAN vs FP-GROWTH COMPARISON ===")
            logger.info(f"Pharmacy Clean Dataset (sample from {files_checked} files):")
            logger.info(f"  Total unique drugs: {len(total_pharmacy_drugs)}")
            logger.info(f"FP-Growth Filtered Dataset:")
            logger.info(f"  Total unique drugs: {len(all_unique_drugs)}")
            logger.info(f"Filtering Ratio:")
            logger.info(f"  FP-Growth kept: {len(all_unique_drugs)} / {len(total_pharmacy_drugs)} = {len(all_unique_drugs)/len(total_pharmacy_drugs)*100:.1f}%")
            logger.info(f"  FP-Growth filtered out: {len(total_pharmacy_drugs) - len(all_unique_drugs)} drugs")
            
            # Show overlap
            overlap_drugs = all_unique_drugs.intersection(total_pharmacy_drugs)
            logger.info(f"Drugs in both datasets: {len(overlap_drugs)}")
            logger.info(f"Drugs only in pharmacy_clean: {len(total_pharmacy_drugs - all_unique_drugs)}")
            logger.info(f"Drugs only in FP-Growth: {len(all_unique_drugs - total_pharmacy_drugs)}")
            
        else:
            logger.warning("No pharmacy_clean files found in S3 gold/pharmacy_clean/")
            
    except Exception as e:
        logger.warning(f"Could not load pharmacy_clean dataset for comparison: {e}")
        logger.info("Manual comparison needed: FP-Growth filters out drugs without significant signal")
    
    # Check which metrics are available in the combined data
    available_metrics = []
    for metric in ['support', 'confidence', 'lift', 'certainty']:
        if metric in combined_df.columns:
            available_metrics.append(metric)
    
    logger.info(f"Available FP-Growth metrics in combined data: {available_metrics}")
    
    if not available_metrics:
        logger.warning("No FP-Growth metrics found in combined data")
        return []
    
    # Perform global aggregations across all files
    logger.info(f"Performing global aggregations for {combined_df['drug_name'].nunique()} unique drugs")
    
    # Global aggregation by drug_name
    global_avg_metrics = combined_df.groupby("drug_name")[available_metrics].mean().reset_index()
    logger.info(f"Calculated global averages for {len(global_avg_metrics)} drugs")
    
    # Handle itemsets column - collect all itemsets for each drug globally
    if 'itemsets' in combined_df.columns:
        logger.info("Processing itemsets column globally")
        # Group itemsets by drug_name and collect all itemsets across all files
        itemsets_by_drug = combined_df.groupby("drug_name")['itemsets'].apply(list).reset_index()
        # Remove duplicates: convert to list, use set to remove duplicates, then join
        itemsets_by_drug['itemsets'] = itemsets_by_drug['itemsets'].apply(lambda x: ';'.join(list(set(x))))
        global_avg_metrics = global_avg_metrics.merge(itemsets_by_drug, on='drug_name', how='left')
        logger.info(f"Processed global itemsets for {len(global_avg_metrics)} drugs")
    else:
        logger.warning("No 'itemsets' column found in combined data")
    
    # Ensure we have only one record per drug_name (remove any duplicates)
    pre_dedup_count = len(global_avg_metrics)
    global_avg_metrics = global_avg_metrics.drop_duplicates(subset=['drug_name']).reset_index(drop=True)
    post_dedup_count = len(global_avg_metrics)
    if pre_dedup_count != post_dedup_count:
        logger.info(f"Removed {pre_dedup_count - post_dedup_count} duplicate drug records")
    
    # Add global count of entries for each drug
    logger.info("Calculating global entry counts")
    global_drug_counts = combined_df.groupby("drug_name").size().reset_index(name='entry_count')
    global_avg_metrics = global_avg_metrics.merge(global_drug_counts, on='drug_name', how='left')
    
    # Add source information summary for each drug
    logger.info("Adding source information summary")
    source_info = combined_df.groupby("drug_name").agg({
        'source_age_band': lambda x: ';'.join(sorted(set(x.dropna()))),
        'source_year': lambda x: ';'.join(map(str, sorted(set(x.dropna())))),
        'source_filename': lambda x: ';'.join(sorted(set(x.dropna())))
    }).reset_index()
    
    global_avg_metrics = global_avg_metrics.merge(source_info, on='drug_name', how='left')
    
    # Calculate rule-related metrics for each drug
    logger.info("Calculating rule-related metrics")
    def calculate_rule_metrics(drug_name):
        """Calculate number of rules and drugs in rules for a specific drug."""
        drug_data = combined_df[combined_df['drug_name'] == drug_name]
        
        # Count unique itemsets (rules) for this drug
        num_rules = 0
        num_drugs_in_rules = 0
        
        if 'itemsets' in drug_data.columns:
            # Get all itemsets for this drug
            all_itemsets = []
            for itemset_str in drug_data['itemsets'].dropna():
                if isinstance(itemset_str, str):
                    # Split by semicolon to get individual itemsets
                    itemsets = itemset_str.split(';')
                    all_itemsets.extend(itemsets)
            
            # Enhanced deduplication: normalize and deduplicate itemsets
            normalized_itemsets = set()
            for itemset in all_itemsets:
                if itemset:
                    # Normalize the itemset (sort drugs, remove extra spaces)
                    drugs = [d.strip() for d in itemset.split(',') if d.strip()]
                    if drugs:
                        # Sort drugs alphabetically to ensure consistent representation
                        normalized_rule = ','.join(sorted(drugs))
                        normalized_itemsets.add(normalized_rule)
            
            # Count unique itemsets (rules)
            num_rules = len(normalized_itemsets)
            
            # Count total drugs across all unique rules
            total_drugs = 0
            for normalized_rule in normalized_itemsets:
                if normalized_rule:
                    # Count drugs in this normalized itemset
                    drugs_in_rule = len([d.strip() for d in normalized_rule.split(',') if d.strip()])
                    total_drugs += drugs_in_rule
            
            num_drugs_in_rules = total_drugs
        
        return {'num_rules': num_rules, 'num_drugs_in_rules': num_drugs_in_rules}
    
    # Calculate rule metrics for all drugs
    rule_metrics = {}
    total_rules_before_dedup = 0
    total_rules_after_dedup = 0
    
    for drug_name in global_avg_metrics['drug_name']:
        rule_metrics[drug_name] = calculate_rule_metrics(drug_name)
        total_rules_before_dedup += rule_metrics[drug_name].get('num_rules', 0)
    
    # Log deduplication summary
    logger.info(f"Rule deduplication summary:")
    logger.info(f"  - Total unique rules across all drugs: {total_rules_before_dedup}")
    logger.info(f"  - Average rules per drug: {total_rules_before_dedup / len(global_avg_metrics):.2f}")
    
    # Show some examples of drugs with many rules
    drugs_with_most_rules = sorted(rule_metrics.items(), key=lambda x: x[1].get('num_rules', 0), reverse=True)[:5]
    logger.info(f"  - Drugs with most rules:")
    for drug_name, metrics in drugs_with_most_rules:
        logger.info(f"    {drug_name}: {metrics.get('num_rules', 0)} rules, {metrics.get('num_drugs_in_rules', 0)} total drugs")
    
    # Add rule metrics to global results
    global_avg_metrics['num_rules'] = global_avg_metrics['drug_name'].map(
        lambda x: rule_metrics.get(x, {}).get('num_rules', 0)
    )
    global_avg_metrics['num_drugs_in_rules'] = global_avg_metrics['drug_name'].map(
        lambda x: rule_metrics.get(x, {}).get('num_drugs_in_rules', 0)
    )
    
    logger.info(f"After all merges: {len(global_avg_metrics)} drugs")
    
    # Step 3: Now encode with global averages
    logger.info(f"Encoding {len(global_avg_metrics)} drugs with global averages")
    def apply_word_features_with_global_metrics(row):
        drug_name = row['drug_name']
        # Create metrics dict for this drug with global averages
        metrics_dict = {}
        for metric in available_metrics:
            if metric in row:
                metrics_dict[metric] = row[metric]
        
        # Ensure support metric is present (only metric needed for encoding)
        if 'support' not in metrics_dict:
            metrics_dict['support'] = 0.0
            if len(global_avg_metrics) <= 5 or row.name < 5:
                logger.debug(f"Added missing 'support' metric with default value 0.0 for drug '{drug_name}'")
        
        # Add rule-related metrics
        metrics_dict['num_rules'] = row.get('num_rules', 0)
        metrics_dict['num_drugs_in_rules'] = row.get('num_drugs_in_rules', 0)
        
        # Debug logging for first few drugs
        if len(global_avg_metrics) <= 5 or row.name < 5:
            logger.debug(f"Processing drug '{drug_name}' with metrics: {metrics_dict}")
        
        return calculate_word_features(drug_name, metrics_dict, logger)
    
    word_features = global_avg_metrics.apply(apply_word_features_with_global_metrics, axis=1)
    global_avg_metrics["first_letter_index"], global_avg_metrics["num_vowels"], global_avg_metrics["num_consonants"], global_avg_metrics["word_length"], global_avg_metrics["num_hyphens_underscores"], global_avg_metrics["num_chemical_suffixes"], global_avg_metrics["num_consonant_clusters"], global_avg_metrics["repetition_factor"], global_avg_metrics["global_encoded_name"] = zip(*word_features)
    
    # Add source information summary
    global_avg_metrics['total_files_processed'] = total_files_processed
    global_avg_metrics['total_records_processed'] = len(combined_df)
    
    # Calculate trends using the combined data
    logger.info("Calculating trends from combined data...")
    
    # Create a function to calculate trends for each drug
    def calculate_drug_trend(drug_name):
        """Calculate trend for a specific drug using all available data points."""
        drug_data = combined_df[combined_df['drug_name'] == drug_name]
        
        if len(drug_data) < 2:  # Need at least 2 data points for trend
            return {'trend': 0.0, 'trend_direction': 'insufficient_data', 'years_analyzed': len(drug_data)}
        
        # Group by year and calculate average support for each year
        if 'source_year' in drug_data.columns and 'support' in drug_data.columns:
            yearly_data = drug_data.groupby('source_year')['support'].mean().reset_index()
            yearly_data = yearly_data.dropna()
            
            if len(yearly_data) < 2:
                return {'trend': 0.0, 'trend_direction': 'single_year', 'years_analyzed': len(yearly_data)}
            
            # Sort by year
            yearly_data = yearly_data.sort_values('source_year')
            years = yearly_data['source_year'].values
            supports = yearly_data['support'].values
            
            # Calculate linear regression trend
            n = len(years)
            sum_x = sum(years)
            sum_y = sum(supports)
            sum_xy = sum(year * support for year, support in zip(years, supports))
            sum_x2 = sum(year * year for year in years)
            
            # Calculate slope
            numerator = n * sum_xy - sum_x * sum_y
            denominator = n * sum_x2 - sum_x * sum_x
            
            if denominator != 0:
                slope = numerator / denominator
                trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                return {
                    'trend': slope,
                    'trend_direction': trend_direction,
                    'years_analyzed': len(yearly_data),
                    'years': years.tolist(),
                    'supports': supports.tolist()
                }
            else:
                return {'trend': 0.0, 'trend_direction': 'stable', 'years_analyzed': len(yearly_data)}
        else:
            return {'trend': 0.0, 'trend_direction': 'no_year_data', 'years_analyzed': 0}
    
    # Calculate trends for all drugs
    trend_results = {}
    for drug_name in global_avg_metrics['drug_name']:
        trend_results[drug_name] = calculate_drug_trend(drug_name)
    
    # Add trend information to global results
    global_avg_metrics['trend'] = global_avg_metrics['drug_name'].map(
        lambda x: trend_results.get(x, {}).get('trend', 0.0)
    )
    global_avg_metrics['trend_direction'] = global_avg_metrics['drug_name'].map(
        lambda x: trend_results.get(x, {}).get('trend_direction', 'unknown')
    )
    global_avg_metrics['years_analyzed'] = global_avg_metrics['drug_name'].map(
        lambda x: trend_results.get(x, {}).get('years_analyzed', 0)
    )
    
    # Log trend summary
    trend_directions = global_avg_metrics['trend_direction'].value_counts()
    logger.info(f"Trend calculation summary:")
    logger.info(f"  - Total drugs: {len(global_avg_metrics)}")
    for direction, count in trend_directions.items():
        logger.info(f"  - {direction}: {count}")
    
    # Log some trend examples
    trend_examples = global_avg_metrics[global_avg_metrics['trend_direction'].isin(['increasing', 'decreasing'])].head(5)
    if not trend_examples.empty:
        logger.info("Sample trend examples:")
        for _, row in trend_examples.iterrows():
            logger.info(f"  {row['drug_name']}: {row['trend_direction']} (slope: {row['trend']:.6f}, years: {row['years_analyzed']})")
    
    # Log final result summary
    logger.info(f"Successfully processed global data: {len(global_avg_metrics)} drugs with {len(global_avg_metrics.columns)} features")
    logger.debug(f"Final columns: {list(global_avg_metrics.columns)}")
    
    return [global_avg_metrics]


def save_processed_results_to_s3(results, s3_output_path, logger=None):
    """Save consolidated processed results to S3 as JSON."""
    if not logger:
        logger = logging.getLogger(__name__)
    
    try:
        # Convert consolidated results to JSON-serializable format
        processed_data = []
        for i, df in enumerate(results):
            df_dict = df.to_dict(orient='records')
            processed_data.extend(df_dict)
        
        # Save to S3
        logger.info(f"Saving consolidated processed results to S3: {s3_output_path}")
        save_to_s3_json(processed_data, s3_output_path, logger)
        logger.info(f"Successfully saved {len(processed_data)} consolidated records to S3")
        
    except Exception as e:
        logger.error(f"Error saving consolidated processed results to S3: {e}")
        raise


def save_processed_results_to_s3_csv(results, s3_output_path, logger=None):
    """Save consolidated processed results to S3 as CSV."""
    if not logger:
        logger = logging.getLogger(__name__)
    
    try:
        import pandas as pd
        
        # Concatenate all DataFrames into one
        consolidated_df = pd.concat(results, ignore_index=True)
        
        # Use the existing S3 utils function
        save_to_s3_csv(consolidated_df, s3_output_path, logger)
        
        logger.info(f"Successfully saved {len(consolidated_df)} consolidated records to S3")
        
    except Exception as e:
        logger.error(f"Error saving consolidated processed results to S3 as CSV: {e}")
        raise
    

def save_processed_results(results, output_directory):
    """Save processed results to local directory as a single consolidated file."""
    import datetime
    import pandas as pd
    
    os.makedirs(output_directory, exist_ok=True)
    
    # Add timestamp to prevent overwrites
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if results:
        # Concatenate all DataFrames into one
        consolidated_df = pd.concat(results, ignore_index=True)
        
        # Save as a single consolidated CSV file with descriptive name
        output_path = os.path.join(output_directory, f"drug_names_with_feature_metrics_{timestamp}.csv")
        consolidated_df.to_csv(output_path, index=False)
        print(f"Saved drug names with feature metrics to: {output_path}")
        print(f"Total records: {len(consolidated_df)}")
        print(f"Total unique drugs: {consolidated_df['drug_name'].nunique()}")
    else:
        print("No results to save.")


def create_itemset_lookup_table(results):
    """Create an enhanced lookup table for all unique itemsets with metrics.
    
    Args:
        results (list): List of DataFrames from process_itemsets
        
    Returns:
        dict: Enhanced lookup table with Drug_Pattern{row_num} = {itemset + metrics}
    """
    lookup_table = {}
    row_num = 1
    
    # Collect all unique itemsets and their metrics from all results
    itemset_metrics = {}
    
    for df in results:
        if 'itemsets' in df.columns:
            for idx, row in df.iterrows():
                itemset_string = row.get('itemsets', '')
                if pd.notna(itemset_string) and itemset_string:
                    # Split by semicolon to get individual itemsets
                    individual_itemsets = itemset_string.split(';')
                    for itemset in individual_itemsets:
                        itemset = itemset.strip()
                        if itemset:
                            if itemset not in itemset_metrics:
                                itemset_metrics[itemset] = {
                                    'total_occurrences': 0,
                                    'support_values': [],
                                    'years_seen': set(),
                                    'age_bands_seen': set(),
                                    'source_filenames': set()
                                }
                            
                            # Update metrics
                            itemset_metrics[itemset]['total_occurrences'] += 1
                            
                            # Add support value if available
                            if 'support' in row:
                                itemset_metrics[itemset]['support_values'].append(row['support'])
                            
                            # Add year and age band info
                            if 'source_year' in row and pd.notna(row['source_year']):
                                # Handle semicolon-separated years
                                years_str = str(row['source_year'])
                                if ';' in years_str:
                                    for year in years_str.split(';'):
                                        if year.strip() and year.strip().isdigit():
                                            itemset_metrics[itemset]['years_seen'].add(int(year.strip()))
                                else:
                                    if years_str.strip() and years_str.strip().isdigit():
                                        itemset_metrics[itemset]['years_seen'].add(int(years_str.strip()))
                            
                            if 'source_age_band' in row:
                                # Handle semicolon-separated age bands
                                age_bands_str = str(row['source_age_band'])
                                if ';' in age_bands_str:
                                    for age_band in age_bands_str.split(';'):
                                        if age_band.strip():
                                            itemset_metrics[itemset]['age_bands_seen'].add(age_band.strip())
                                else:
                                    if age_bands_str.strip():
                                        itemset_metrics[itemset]['age_bands_seen'].add(age_bands_str.strip())
                            
                            if 'source_filename' in row:
                                # Handle semicolon-separated filenames
                                filenames_str = str(row['source_filename'])
                                if ';' in filenames_str:
                                    for filename in filenames_str.split(';'):
                                        if filename.strip():
                                            itemset_metrics[itemset]['source_filenames'].add(filename.strip())
                                else:
                                    if filenames_str.strip():
                                        itemset_metrics[itemset]['source_filenames'].add(filenames_str.strip())
    
    # Create enhanced lookup table with metrics
    for itemset in sorted(itemset_metrics.keys()):
        metrics = itemset_metrics[itemset]
        
        # Calculate derived metrics
        pattern_length = len(itemset.split(';')) if ';' in itemset else 1
        avg_support = sum(metrics['support_values']) / len(metrics['support_values']) if metrics['support_values'] else 0
        max_support = max(metrics['support_values']) if metrics['support_values'] else 0
        min_support = min(metrics['support_values']) if metrics['support_values'] else 0
        
        # Convert sets to sorted lists for JSON serialization
        years_seen = sorted(list(metrics['years_seen']))
        age_bands_seen = sorted(list(metrics['age_bands_seen']))
        
        lookup_table[f"Drug_Pattern{row_num}"] = {
            'pattern': itemset,
            'total_occurrences': metrics['total_occurrences'],
            'pattern_length': pattern_length,
            'avg_support': int(avg_support * 1000),  # Multiply by 1000 to match encode_drug_name scaling
            'max_support': int(max_support * 1000),   # Multiply by 1000 to match encode_drug_name scaling
            'min_support': int(min_support * 1000),   # Multiply by 1000 to match encode_drug_name scaling
            'years_seen': years_seen,
            'age_bands_seen': age_bands_seen,
            'first_seen_year': min(years_seen) if years_seen else None,
            'last_seen_year': max(years_seen) if years_seen else None,
            'num_years_seen': len(years_seen),
            'num_age_bands_seen': len(age_bands_seen)
        }
        row_num += 1
    
    return lookup_table


def save_lookup_table_to_s3(lookup_table, s3_output_path, logger=None):
    """Save itemset lookup table to S3.
    
    Args:
        lookup_table (dict): Lookup table to save
        s3_output_path (str): S3 path to save the lookup table
        logger: Logger instance
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    try:
        # Save to S3
        logger.info(f"Saving itemset lookup table to S3: {s3_output_path}")
        save_to_s3_json(lookup_table, s3_output_path, logger)
        logger.info(f"Successfully saved lookup table with {len(lookup_table)} unique itemsets to S3")
        
    except Exception as e:
        logger.error(f"Error saving lookup table to S3: {e}")
        raise

def save_lookup_table_local(lookup_table, output_directory, filename="drug_patterns_with_feature_metrics.json"):
    """Save drug patterns lookup table to local directory.
    
    Args:
        lookup_table (dict): Lookup table to save
        output_directory (str): Local directory to save the file
        filename (str): Name of the output file
    """
    import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Add timestamp to prevent overwrites
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(filename)
    timestamped_filename = f"{base_name}_{timestamp}{ext}"
    output_path = os.path.join(output_directory, timestamped_filename)
    
    with open(output_path, 'w') as f:
        json.dump(lookup_table, f, indent=2)
    
    print(f"Drug patterns with feature metrics saved locally to: {output_path}")
    print(f"Total unique drug patterns: {len(lookup_table)}")


def process_global_itemsets():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Local input path - use the correct path for EC2 environment
    input_directory = "/home/pgx3874/pgx-analysis/fpgrowth_analysis/global_itemsets"
    
    # S3 output path
    s3_output_path = "s3://pgxdatalake/pgx_pipeline/fpgrowth_analysis/processed_itemsets/processed_itemsets.json"
    
    # Local output path (fallback) - create processed_itemsets directory in the same location
    output_directory = os.path.join(os.path.dirname(input_directory), "processed_itemsets")

    # Year filtering parameters (can be modified as needed)
    start_event_year = None  # e.g., 2020 (None = process all years)
    end_event_year = None    # e.g., 2022 (None = process all years)
    
    try:
        # Process from local directory with optional year filtering
        logger.info(f"Processing itemsets from local directory: {input_directory}")
        if start_event_year or end_event_year:
            logger.info(f"Applying year filter: {start_event_year} to {end_event_year}")
        processed_results = process_itemsets(input_directory, start_event_year, end_event_year, logger)
        
        # Log summary of processed results
        total_drugs = sum(len(result) for result in processed_results)
        logger.info(f"Processed {total_drugs} total drug records across all files")
        
        # Show consolidation summary
        if processed_results:
            unique_drugs = set()
            age_bands = set()
            years = set()
            for df in processed_results:
                unique_drugs.update(df['drug_name'].unique())
                age_bands.update(df['source_age_band'].unique())
                years.update(df['source_year'].dropna().unique())
            
            logger.info(f"Consolidation summary:")
            logger.info(f"  - Total records: {total_drugs}")
            logger.info(f"  - Unique drugs: {len(unique_drugs)}")
            logger.info(f"  - Age bands: {sorted(age_bands)}")
            logger.info(f"  - Years: {sorted(years)}")
        

        
        # Show available metrics in the first result
        if processed_results and len(processed_results) > 0:
            first_result = processed_results[0]
            available_columns = [col for col in first_result.columns if col not in ['drug_name', 'first_letter_index', 'num_vowels', 'num_consonants', 'word_length', 'global_encoded_name', 'entry_count']]
            logger.info(f"Available FP-Growth metrics: {available_columns}")
            
            # Check if itemsets were processed
            if 'itemsets' in first_result.columns:
                logger.info("✓ Itemsets column processed - drug combinations preserved")
            else:
                logger.info("⚠ Itemsets column not found in data")
        
        # Save to S3 with timestamp to prevent overwrites
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Using timestamp for this run: {timestamp}")
        
        # Create organized folder structure
        drug_names_dir = os.path.join(output_directory, "drug_names")
        drug_patterns_dir = os.path.join(output_directory, "drug_patterns")
        os.makedirs(drug_names_dir, exist_ok=True)
        os.makedirs(drug_patterns_dir, exist_ok=True)
        
        logger.info("Saving drug names with feature metrics to S3...")
        s3_output_path_timestamped = f"s3://pgxdatalake/pgx_pipeline/fpgrowth_analysis/processed_itemsets/drug_names/drug_names_with_feature_metrics_{timestamp}.csv"
        save_processed_results_to_s3_csv(processed_results, s3_output_path_timestamped, logger)
        print(f"Drug names with feature metrics saved to S3: {s3_output_path_timestamped}")
        
        # Also save locally in drug_names folder
        logger.info("Saving drug names with feature metrics locally...")
        save_processed_results(processed_results, drug_names_dir)
        
        # Create and save drug patterns lookup table
        logger.info("Creating drug patterns with feature metrics...")
        lookup_table = create_itemset_lookup_table(processed_results)
        
        if lookup_table:
            # Save drug patterns lookup table to S3 with timestamp
            s3_lookup_path_timestamped = f"s3://pgxdatalake/pgx_pipeline/fpgrowth_analysis/processed_itemsets/drug_patterns/drug_patterns_with_feature_metrics_{timestamp}.json"
            save_lookup_table_to_s3(lookup_table, s3_lookup_path_timestamped, logger)
            print(f"Drug patterns with feature metrics saved to S3: {s3_lookup_path_timestamped}")
            
            # Save lookup table locally in the drug_patterns directory
            save_lookup_table_local(lookup_table, drug_patterns_dir)
            
            # Log lookup table summary
            logger.info(f"Created lookup table with {len(lookup_table)} unique itemsets")
            logger.info("Sample lookup table entries:")
            for i, (key, value) in enumerate(list(lookup_table.items())[:5]):
                logger.info(f"  {key}: {value}")
        else:
            logger.warning("No itemsets found to create lookup table")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        
        # Check if we have processed results to save locally
        if 'processed_results' in locals():
            logger.warning("S3 save failed. Falling back to local save...")
            
            # Create organized folder structure for fallback
            drug_names_dir = os.path.join(output_directory, "drug_names")
            drug_patterns_dir = os.path.join(output_directory, "drug_patterns")
            os.makedirs(drug_names_dir, exist_ok=True)
            os.makedirs(drug_patterns_dir, exist_ok=True)
            
            # Fallback to local save in drug_names folder
            save_processed_results(processed_results, drug_names_dir)
            print(f"Processed results saved locally to: {drug_names_dir}")
            
            # Try to create lookup table even if S3 save failed
            try:
                logger.info("Creating itemset lookup table (local only)...")
                lookup_table = create_itemset_lookup_table(processed_results)
                if lookup_table:
                    save_lookup_table_local(lookup_table, drug_patterns_dir)
            except Exception as lookup_error:
                logger.error(f"Failed to create lookup table: {lookup_error}")
        else:
            logger.error("Processing failed completely. No results to save.")
            raise

# Run it!
process_global_itemsets()
