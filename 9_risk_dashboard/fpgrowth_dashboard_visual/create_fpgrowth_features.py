#!/usr/bin/env python3
"""
Create patient-level features from FP-Growth itemsets and rules.

This script extracts features from FP-Growth analysis:
- Itemsets: Frequent co-occurring items (drugs, ICD codes, CPT codes)
- Rules: Association rules (antecedent → consequent)

Features created:
- Binary indicators for frequent itemsets (does patient have this itemset?)
- Binary indicators for association rules (does patient match this rule?)
- Support/confidence scores for matched itemsets/rules
- Count of matched itemsets/rules per patient
- Top N itemsets/rules features

Output:
- Saves to: outputs/feature_engineering/fpgrowth_features_{cohort}_{age_band}.csv
- This intermediate file is then merged with other features by add_fpgrowth_features_to_model_data.py
"""

import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import duckdb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Item types to process
ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code', 'medical_code']


def load_json_file(json_path: Path) -> List[Dict]:
    """Load JSON file and return as list of dictionaries."""
    if not json_path.exists():
        logger.warning(f"JSON file not found: {json_path}")
        return []
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {json_path}: {e}")
        return []


def extract_patient_transactions(
    model_data_path: Path,
    item_type: str,
    cohort_name: str
) -> pd.DataFrame:
    """
    Extract patient transactions from model_data parquet file.
    
    Returns DataFrame with columns: mi_person_key, items (list of items)
    """
    if not model_data_path.exists():
        logger.warning(f"Model data file not found: {model_data_path}")
        return pd.DataFrame()
    
    con = duckdb.connect()
    
    # Build query based on item_type
    if item_type == 'drug_name':
        item_query = "SELECT mi_person_key, drug_name as item FROM read_parquet('{path}') WHERE drug_name IS NOT NULL AND drug_name != ''"
    elif item_type == 'icd_code':
        item_query = """
        WITH all_icds AS (
            SELECT mi_person_key, primary_icd_diagnosis_code as icd FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, two_icd_diagnosis_code as icd FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, three_icd_diagnosis_code as icd FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, four_icd_diagnosis_code as icd FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, five_icd_diagnosis_code as icd FROM read_parquet('{path}')
        )
        SELECT mi_person_key, icd as item FROM all_icds
        WHERE icd IS NOT NULL AND icd != ''
        """
    elif item_type == 'cpt_code':
        item_query = "SELECT mi_person_key, procedure_code as item FROM read_parquet('{path}') WHERE procedure_code IS NOT NULL AND procedure_code != ''"
    elif item_type == 'medical_code':
        item_query = """
        WITH all_codes AS (
            SELECT mi_person_key, primary_icd_diagnosis_code as code FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, two_icd_diagnosis_code as code FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, three_icd_diagnosis_code as code FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, four_icd_diagnosis_code as code FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, five_icd_diagnosis_code as code FROM read_parquet('{path}')
            UNION ALL
            SELECT mi_person_key, procedure_code as code FROM read_parquet('{path}')
        )
        SELECT mi_person_key, code as item FROM all_codes WHERE code IS NOT NULL AND code != ''
        """
    else:
        logger.error(f"Unknown item_type: {item_type}")
        con.close()
        return pd.DataFrame()
    
    query = item_query.format(path=str(model_data_path))
    
    # Get patient transactions
    transactions_df = con.execute(query).df()
    con.close()
    
    if transactions_df.empty:
        logger.warning(f"No transactions found for {item_type}")
        return pd.DataFrame()
    
    # Group by patient to create transaction lists
    patient_transactions = (
        transactions_df
        .groupby('mi_person_key')['item']
        .apply(list)
        .reset_index(name='items')
    )
    
    # Convert items to set for faster matching
    patient_transactions['items_set'] = patient_transactions['items'].apply(set)
    
    logger.info(f"Extracted transactions for {len(patient_transactions)} patients ({item_type})")
    
    return patient_transactions


def match_itemset(patient_items: Set[str], itemset: List[str]) -> bool:
    """Check if patient's items contain all items in the itemset."""
    itemset_set = set(itemset)
    return itemset_set.issubset(patient_items)


def match_rule(patient_items: Set[str], antecedents: List[str], consequents: List[str]) -> bool:
    """Check if patient matches an association rule (has antecedents AND consequents)."""
    antecedents_set = set(antecedents)
    consequents_set = set(consequents)
    return antecedents_set.issubset(patient_items) and consequents_set.issubset(patient_items)


def create_itemset_features(
    patient_transactions: pd.DataFrame,
    itemsets: List[Dict],
    item_type: str,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Create patient-level features from itemsets.
    
    Features:
    - Binary indicators for top N itemsets
    - Support scores for matched itemsets
    - Count of matched itemsets
    """
    if not itemsets or patient_transactions.empty:
        logger.warning(f"No itemsets or patient transactions for {item_type}")
        return pd.DataFrame(columns=['mi_person_key'])
    
    # Sort by support and take top N
    sorted_itemsets = sorted(itemsets, key=lambda x: x.get('support', 0), reverse=True)[:top_n]
    
    # Initialize features dataframe
    features_df = patient_transactions[['mi_person_key']].copy()
    
    # Feature 1: Binary indicators for top itemsets
    for idx, itemset_data in enumerate(sorted_itemsets):
        itemset = itemset_data.get('itemsets', [])
        support = itemset_data.get('support', 0)
        
        if not itemset:
            continue
        
        # Check if patient has this itemset (binary indicator)
        features_df[f'{item_type}_itemset_{idx}_match'] = (
            patient_transactions['items_set'].apply(lambda x, itemset=itemset: match_itemset(x, itemset)).astype(int)
        )
        
        # Store support value for later aggregation (not as individual feature)
        # We'll use this to compute max_support across matched itemsets
    
    # Feature 2: Count of matched itemsets
    itemset_match_cols = [col for col in features_df.columns if col.endswith('_match')]
    if itemset_match_cols:
        features_df[f'{item_type}_itemsets_matched_count'] = features_df[itemset_match_cols].sum(axis=1)
    else:
        features_df[f'{item_type}_itemsets_matched_count'] = 0
    
    # Feature 3: Maximum support among matched itemsets
    # Compute directly from match indicators and itemset support values
    # This aggregates across itemsets and provides additional signal beyond binary match
    max_support_values = []
    for idx, itemset_data in enumerate(sorted_itemsets):
        itemset = itemset_data.get('itemsets', [])
        support = itemset_data.get('support', 0)
        match_col = f'{item_type}_itemset_{idx}_match'
        if match_col in features_df.columns:
            # Only consider support if patient matched this itemset
            max_support_values.append(features_df[match_col] * support)
    
    if max_support_values:
        features_df[f'{item_type}_itemsets_max_support'] = pd.concat(max_support_values, axis=1).max(axis=1).fillna(0)
    else:
        features_df[f'{item_type}_itemsets_max_support'] = 0
    
    logger.info(f"Created {len(features_df.columns) - 1} itemset features for {item_type}")
    
    return features_df


def create_rule_features(
    patient_transactions: pd.DataFrame,
    rules: List[Dict],
    item_type: str,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Create patient-level features from association rules.
    
    Features:
    - Binary indicators for top N rules
    - Confidence scores for matched rules
    - Count of matched rules
    """
    if not rules or patient_transactions.empty:
        logger.warning(f"No rules or patient transactions for {item_type}")
        return pd.DataFrame(columns=['mi_person_key'])
    
    # Sort by lift (or confidence) and take top N
    sorted_rules = sorted(rules, key=lambda x: x.get('lift', x.get('confidence', 0)), reverse=True)[:top_n]
    
    # Initialize features dataframe
    features_df = patient_transactions[['mi_person_key']].copy()
    
    # Feature 1: Binary indicators for top rules
    for idx, rule_data in enumerate(sorted_rules):
        antecedents = rule_data.get('antecedents', [])
        consequents = rule_data.get('consequents', [])
        confidence = rule_data.get('confidence', 0)
        lift = rule_data.get('lift', 0)
        
        if not antecedents or not consequents:
            continue
        
        # Check if patient matches this rule
        features_df[f'{item_type}_rule_{idx}_match'] = (
            patient_transactions['items_set'].apply(
                lambda x, antecedents=antecedents, consequents=consequents: match_rule(x, antecedents, consequents)
            ).astype(int)
        )
        
        # Confidence score if matched
        features_df[f'{item_type}_rule_{idx}_confidence'] = (
            features_df[f'{item_type}_rule_{idx}_match'] * confidence
        )
        
        # Lift score if matched
        features_df[f'{item_type}_rule_{idx}_lift'] = (
            features_df[f'{item_type}_rule_{idx}_match'] * lift
        )
    
    # Feature 2: Count of matched rules
    rule_match_cols = [col for col in features_df.columns if col.endswith('_match')]
    if rule_match_cols:
        features_df[f'{item_type}_rules_matched_count'] = features_df[rule_match_cols].sum(axis=1)
    else:
        features_df[f'{item_type}_rules_matched_count'] = 0
    
    # Feature 3: Maximum confidence among matched rules
    rule_confidence_cols = [col for col in features_df.columns if col.endswith('_confidence')]
    if rule_confidence_cols:
        features_df[f'{item_type}_rules_max_confidence'] = features_df[rule_confidence_cols].max(axis=1).fillna(0)
    else:
        features_df[f'{item_type}_rules_max_confidence'] = 0
    
    # Feature 4: Maximum lift among matched rules
    rule_lift_cols = [col for col in features_df.columns if col.endswith('_lift')]
    if rule_lift_cols:
        features_df[f'{item_type}_rules_max_lift'] = features_df[rule_lift_cols].max(axis=1).fillna(0)
    else:
        features_df[f'{item_type}_rules_max_lift'] = 0
    
    logger.info(f"Created {len(features_df.columns) - 1} rule features for {item_type}")
    
    return features_df


def create_all_fpgrowth_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    split_type: str = "combined",
    event_year: str = "train",
    top_n_itemsets: int = 30,
    top_n_rules: int = 30
) -> pd.DataFrame:
    """
    Create all FP-Growth features (itemsets + rules) for all item types.
    
    Parameters:
    -----------
    project_root : Path
        Project root directory
    cohort_name : str
        Cohort name (e.g., opioid_ed)
    age_band : str
        Age band (e.g., 0-12)
    split_type : str
        Split type (combined or target)
    event_year : str
        Event year label (train, 2019, etc.)
    top_n_itemsets : int
        Number of top itemsets to create features for
    top_n_rules : int
        Number of top rules to create features for
        
    Returns:
    --------
    pd.DataFrame
        Combined patient-level FP-Growth features
    """
    age_band_fname = age_band.replace("-", "_")
    
    # FP-Growth output directory (step-local, under 10b_fpgrowth_dashboard_visual)
    fpgrowth_output_dir = (
        project_root / "10b_fpgrowth_dashboard_visual" / "outputs" / cohort_name / split_type / age_band_fname / event_year
    )
    
    # Model data path: use canonical 4a_model_data (model-ready cases + controls).
    # We no longer read from the legacy model_data/ tree for feature engineering.
    model_data_dir = (
        project_root
        / "4a_model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    model_data_filtered = model_data_dir / "model_events_no_protocols.parquet"
    model_data_path = (
        model_data_filtered
        if model_data_filtered.exists()
        else model_data_dir / "model_events.parquet"
    )
    
    if not model_data_path.exists():
        logger.error(f"Model data not found: {model_data_path}")
        return pd.DataFrame()
    
    # Get base patient list (both target and control)
    con = duckdb.connect()
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key
        FROM read_parquet('{model_data_path}')
        WHERE target IN (0, 1)
        """
    ).df()
    con.close()
    
    if base_df.empty:
        logger.error("No patients found in model_data")
        return pd.DataFrame()
    
    logger.info(f"Creating FP-Growth features for {len(base_df)} patients (target + control)")
    
    # Collect all feature dataframes
    all_features = [base_df.copy()]
    
    # Process each item type
    for item_type in ITEM_TYPES:
        logger.info(f"\nProcessing {item_type}...")
        
        # Load itemsets
        itemsets_path = fpgrowth_output_dir / f"{item_type}_itemsets.json"
        itemsets = load_json_file(itemsets_path)
        
        # Load rules
        rules_path = fpgrowth_output_dir / f"{item_type}_rules.json"
        rules = load_json_file(rules_path)
        
        if not itemsets and not rules:
            logger.warning(f"No itemsets or rules found for {item_type}, skipping")
            continue
        
        # Extract patient transactions
        patient_transactions = extract_patient_transactions(
            model_data_path=model_data_path,
            item_type=item_type,
            cohort_name=cohort_name
        )
        
        if patient_transactions.empty:
            logger.warning(f"No patient transactions for {item_type}, skipping")
            continue
        
        # Merge with base_df to ensure all patients are included
        patient_transactions = base_df.merge(patient_transactions, on='mi_person_key', how='left')
        patient_transactions['items'] = patient_transactions['items'].fillna('').apply(lambda x: [] if x == '' else x)
        patient_transactions['items_set'] = patient_transactions['items'].apply(set)
        
        # Create itemset features
        if itemsets:
            itemset_features = create_itemset_features(
                patient_transactions=patient_transactions,
                itemsets=itemsets,
                item_type=item_type,
                top_n=top_n_itemsets
            )
            if not itemset_features.empty:
                all_features.append(itemset_features.drop(columns=['mi_person_key'], errors='ignore'))
        
        # Create rule features
        if rules:
            rule_features = create_rule_features(
                patient_transactions=patient_transactions,
                rules=rules,
                item_type=item_type,
                top_n=top_n_rules
            )
            if not rule_features.empty:
                all_features.append(rule_features.drop(columns=['mi_person_key'], errors='ignore'))
    
    # Combine all features
    combined_features = all_features[0].copy()
    for feature_df in all_features[1:]:
        if 'mi_person_key' in feature_df.columns:
            combined_features = combined_features.merge(
                feature_df,
                on='mi_person_key',
                how='left'
            )
        else:
            # Merge by index if no mi_person_key
            combined_features = pd.concat([combined_features, feature_df], axis=1)
    
    # Fill NaN values with 0 for numeric columns
    for col in combined_features.columns:
        if col != 'mi_person_key':
            if combined_features[col].dtype in ['float64', 'int64']:
                combined_features[col] = combined_features[col].fillna(0)
    
    logger.info(f"\nCreated {len(combined_features.columns) - 1} FP-Growth features for {len(combined_features)} patients")
    
    return combined_features


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create FP-Growth features from itemsets and rules")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--split_type", default="combined", help="Split type (combined or target)")
    parser.add_argument("--event_year", default="train", help="Event year label (train, 2019, etc.)")
    parser.add_argument("--top_n_itemsets", type=int, default=30, help="Number of top itemsets to use")
    parser.add_argument("--top_n_rules", type=int, default=30, help="Number of top rules to use")
    parser.add_argument("--output", help="Output CSV path (optional)")
    
    args = parser.parse_args()
    
    project_root = PROJECT_ROOT
    
    # Create FP-Growth features
    fpgrowth_features = create_all_fpgrowth_features(
        project_root=project_root,
        cohort_name=args.cohort,
        age_band=args.age_band,
        split_type=args.split_type,
        event_year=args.event_year,
        top_n_itemsets=args.top_n_itemsets,
        top_n_rules=args.top_n_rules
    )
    
    if fpgrowth_features.empty:
        logger.error("No features created. Check inputs and logs.")
        return
    
    # Set output path - intermediate file for FP-Growth features only
    if not args.output:
        age_band_fname = args.age_band.replace("-", "_")
        feature_eng_dir = project_root / "10b_fpgrowth_dashboard_visual" / "outputs" / "feature_engineering"
        feature_eng_dir.mkdir(parents=True, exist_ok=True)
        args.output = feature_eng_dir / f"fpgrowth_features_{args.cohort}_{age_band_fname}.csv"
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpgrowth_features.to_csv(output_path, index=False)

    # Mirror to central 5_feature_engineering/feature_engineering_outputs directory for easy access
    try:
        fe_root = (
            PROJECT_ROOT
            / "5_feature_engineering"
            / "feature_engineering_outputs"
            / "4_fpgrowth"
            / args.cohort
            / args.age_band
        )
        fe_root.mkdir(parents=True, exist_ok=True)
        mirror_path = fe_root / output_path.name
        print(f"[INFO] Copying FP-Growth features to {mirror_path}")
        shutil.copy2(output_path, mirror_path)
    except Exception as e:  # pragma: no cover - best-effort mirror
        logger.warning(f"Could not mirror FP-Growth features to feature_engineering_outputs: {e}")
    
    print(f"\nCreated {len(fpgrowth_features.columns) - 1} FP-Growth features for {len(fpgrowth_features)} patients")
    print("Output format: Ready for merging with other features (uses mi_person_key)")
    print(f"Saved to: {output_path}")
    
    # Upload to S3 gold location (intermediate file)
    # Validate inputs to prevent command injection (cohort and age_band should be from constants)
    if not args.cohort.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid cohort name: {args.cohort}")
    if not args.age_band.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid age band: {args.age_band}")
    
    age_band_fname = args.age_band.replace("-", "_")
    s3_path = f"s3://pgxdatalake/gold/feature_engineering/4_fpgrowth/{args.cohort}/{args.age_band}/fpgrowth_features_{args.cohort}_{age_band_fname}.csv"
    
    # Check for AWS CLI
    aws_cli = shutil.which("aws")
    if aws_cli:
        try:
            print(f"\nUploading to S3: {s3_path}")
            # Using list form of subprocess.run prevents shell injection
            # Inputs validated above to ensure safe S3 path construction
            # aws_cli from shutil.which() is trusted system binary
            subprocess.run(  # noqa: S603
                [aws_cli, "s3", "cp", str(output_path), s3_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"S3 upload successful: {s3_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"S3 upload failed: {e.stderr}")
            print(f"Warning: Could not upload to S3: {e.stderr}")
    else:
        logger.info("AWS CLI not found, skipping S3 upload")
        print("Note: AWS CLI not found, skipping S3 upload")
    
    print(f"\nFeature columns ({len(fpgrowth_features.columns)} total):")
    for col in fpgrowth_features.columns[:20]:  # Show first 20
        print(f"  - {col}")
    if len(fpgrowth_features.columns) > 20:
        print(f"  ... and {len(fpgrowth_features.columns) - 20} more")


if __name__ == "__main__":
    main()

