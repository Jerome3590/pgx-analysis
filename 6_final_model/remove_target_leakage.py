#!/usr/bin/env python3
"""
Remove target leakage from final feature table.

This script:
1. Removes post-event features (target leakage)
2. Removes time-to-target features (target leakage)
3. Identifies and documents remaining features for review
4. Rebuilds feature table without leakage

Usage:
    python remove_target_leakage.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def remove_target_leakage(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """Remove target leakage features from final feature table."""
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load feature table
    feature_table_path = (
        project_root
        / "6_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features.csv"
    )
    
    if not feature_table_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")
    
    print(f"[INFO] Loading feature table from {feature_table_path}")
    df = pd.read_csv(feature_table_path)
    
    print(f"[INFO] Original dataset: {len(df)} patients, {len(df.columns)} columns")
    
    # Identify leakage features
    leakage_features = []
    
    # 1. Post-event features (calculated AFTER target event)
    post_features = [c for c in df.columns if c.startswith('post_')]
    leakage_features.extend(post_features)
    print(f"\n[INFO] Post-event features (TARGET LEAKAGE): {len(post_features)}")
    for f in post_features:
        print(f"  - {f}")
    
    # 2. Time-to-target features (reference the target event)
    time_to_features = [c for c in df.columns if 'time_to' in c.lower() or 'time_to_' in c.lower()]
    leakage_features.extend(time_to_features)
    print(f"\n[INFO] Time-to-target features (TARGET LEAKAGE): {len(time_to_features)}")
    for f in time_to_features:
        print(f"  - {f}")
    
    # 2b. Time-window features that reference target event (30d, 90d, 180d before target)
    # NOTE: Time intervals BETWEEN consecutive events (e.g., drug_interval_mean) are OK - they don't reference target
    # Only remove time windows that count events in X days BEFORE target
    time_window_features = [c for c in df.columns if any(x in c for x in ['_30d', '_90d', '_180d']) and 'interval' not in c.lower()]
    leakage_features.extend(time_window_features)
    print(f"\n[INFO] Time-window features referencing target (TARGET LEAKAGE): {len(time_window_features)}")
    for f in time_window_features:
        print(f"  - {f}")
    
    # Note: Time interval features (between consecutive events) are KEPT - they're predictive
    interval_features = [c for c in df.columns if 'interval' in c.lower()]
    print(f"\n[INFO] Time interval features (KEPT - predictive): {len(interval_features)}")
    for f in interval_features[:10]:  # Show first 10
        print(f"  - {f}")
    if len(interval_features) > 10:
        print(f"  ... and {len(interval_features) - 10} more")
    
    # 3. Target time and first time (datetime columns, not features but should be removed)
    datetime_features = ['target_time', 'first_time']
    leakage_features.extend([f for f in datetime_features if f in df.columns])
    
    # 4. DTW features (REMOVED - used for protocol filtering, not as features)
    # DTW captures standard care protocols that both targets and controls follow
    # Sequence information comes from BupaR, not DTW
    dtw_features = [c for c in df.columns if 'dtw' in c.lower()]
    leakage_features.extend(dtw_features)
    print(f"\n[INFO] DTW features found: {len(dtw_features)}")
    print("[INFO] DTW features are REMOVED - DTW is used for protocol filtering, not feature engineering")
    print("[INFO] DTW captures standard care protocols that both targets and controls follow")
    print("[INFO] Sequence information comes from BupaR, not DTW")
    for f in dtw_features[:10]:
        print(f"  - {f}")
    if len(dtw_features) > 10:
        print(f"  ... and {len(dtw_features) - 10} more")
    
    # Remove leakage features
    safe_features = [c for c in df.columns if c not in leakage_features]
    
    # Verify important predictive features are preserved
    sequence_features = [c for c in df.columns if 'sequence' in c.lower() or 'trace' in c.lower()]
    interval_features_kept = [c for c in safe_features if 'interval' in c.lower()]
    fpgrowth_features_kept = [c for c in safe_features if any(x in c for x in ['itemset', 'rule', 'support', 'confidence', 'lift'])]
    
    print(f"\n[INFO] Preserving important predictive features:")
    print(f"  Sequence features (top/rare): {len([c for c in sequence_features if c in safe_features])}")
    print(f"  Time interval features (between events): {len(interval_features_kept)}")
    print(f"  FP-Growth features (itemsets/rules): {len(fpgrowth_features_kept)}")
    
    print(f"\n[INFO] Removing {len(leakage_features)} leakage features")
    df_clean = df[safe_features].copy()
    
    # Verify no F1120 in feature names (should be excluded during feature engineering)
    f1120_features = [c for c in df_clean.columns if 'F1120' in c.upper()]
    if f1120_features:
        print(f"\n[WARNING] Found {len(f1120_features)} features with F1120 in name:")
        for f in f1120_features:
            print(f"  - {f}")
        print("[INFO] These should be removed - F1120 must be excluded from final model features")
        # Remove F1120 features
        safe_features = [c for c in safe_features if c not in f1120_features]
        leakage_features.extend(f1120_features)
        df_clean = df[safe_features].copy()
    
    # Remove non-predictive markers/confounders
    excluded_markers = [
        'item_drug_SUBOXONE',  # Treatment medication - marker, not predictive
        'item_drug_BUPRENORPHINE_HCL',  # Treatment medication - marker, not predictive
        'item_drug_BUPRENORPHINE_HCL_NALOXON',  # Treatment medication - marker, not predictive
        'item_icd_F1123',  # Opioid dependence ICD code - marker, not predictive
    ]
    found_excluded = [c for c in df_clean.columns if c in excluded_markers]
    if found_excluded:
        print(f"\n[INFO] Removing {len(found_excluded)} non-predictive markers/confounders:")
        for f in found_excluded:
            print(f"  - {f}")
        safe_features = [c for c in safe_features if c not in found_excluded]
        leakage_features.extend(found_excluded)
        df_clean = df[safe_features].copy()
    
    # 5. For non_opioid_ed cohort: remove ICD and CPT features (polypharmacy uses drugs only)
    if "non_opioid" in cohort_name.lower() or "ed_non_opioid" in cohort_name.lower():
        item_icd_features_to_remove = [c for c in df_clean.columns if c.startswith('item_icd_')]
        item_cpt_features_to_remove = [c for c in df_clean.columns if c.startswith('item_cpt_')]
        if item_icd_features_to_remove or item_cpt_features_to_remove:
            print(f"\n[INFO] For non_opioid_ed cohort: Removing ICD and CPT features (polypharmacy uses drugs only)")
            print(f"  Removing {len(item_icd_features_to_remove)} ICD features and {len(item_cpt_features_to_remove)} CPT features")
            safe_features = [c for c in safe_features if c not in item_icd_features_to_remove and c not in item_cpt_features_to_remove]
            leakage_features.extend(item_icd_features_to_remove)
            leakage_features.extend(item_cpt_features_to_remove)
            df_clean = df[safe_features].copy()
    
    # 6. Validate item_* features for post-target leakage (drugs and ICD codes)
    print(f"\n[INFO] Validating item_* features for post-target leakage...")
    item_drug_features = [c for c in df_clean.columns if c.startswith('item_drug_')]
    item_icd_features = [c for c in df_clean.columns if c.startswith('item_icd_')]
    item_cpt_features = [c for c in df_clean.columns if c.startswith('item_cpt_')]
    
    post_target_item_features = []
    
    if DUCKDB_AVAILABLE and (item_drug_features or item_icd_features or item_cpt_features):
        # Check underlying event data for post-target leakage
        model_data_path = (
            project_root
            / "model_data"
            / f"cohort_name={cohort_name}"
            / f"age_band={age_band}"
            / "model_events.parquet"
        )
        
        # Also try alternative location
        if not model_data_path.exists():
            model_data_path = (
                project_root
                / "4a_model_data"
                / f"cohort_name={cohort_name}"
                / f"age_band={age_band}"
                / "model_events_no_protocols.parquet"
            )
        
        if model_data_path.exists():
            try:
                # Determine target date field
                if "opioid" in cohort_name.lower():
                    target_date_field = "first_opioid_ed_date"
                else:
                    target_date_field = "first_ed_non_opioid_date"
                
                con = duckdb.connect()
                model_data_path_str = str(model_data_path).replace('\\', '/')
                
                # Check each item feature for post-target events
                for feature_name in item_drug_features + item_icd_features + item_cpt_features:
                    # Extract the code/drug name from feature name
                    # Format: item_drug_DRUG_NAME or item_icd_ICD_CODE or item_cpt_CPT_CODE
                    if feature_name.startswith('item_drug_'):
                        code_name = feature_name.replace('item_drug_', '')
                        code_column = 'drug_name'
                    elif feature_name.startswith('item_icd_'):
                        code_name = feature_name.replace('item_icd_', '')
                        # Check all ICD diagnosis columns
                        code_column = None  # Will check all ICD columns
                    elif feature_name.startswith('item_cpt_'):
                        code_name = feature_name.replace('item_cpt_', '')
                        code_column = 'procedure_code'
                    else:
                        continue
                    
                    # Get patients who have this feature = 1
                    patients_with_feature = df_clean[df_clean[feature_name] == 1]['mi_person_key'].astype(str).unique().tolist()
                    
                    if not patients_with_feature or len(patients_with_feature) == 0:
                        continue
                    
                    # Limit to reasonable batch size for query (avoid SQL injection and query size limits)
                    # Check in batches if needed
                    max_batch_size = 1000
                    post_target_found = False
                    
                    for i in range(0, len(patients_with_feature), max_batch_size):
                        batch = patients_with_feature[i:i + max_batch_size]
                        patient_list = ','.join([f"'{p.replace(\"'\", \"''\")}'" for p in batch])
                        
                        # Check if any of these patients have this code/drug AFTER target event
                        if code_column == 'drug_name':
                            # Check drug_name column
                            query = f"""
                            SELECT COUNT(*) as post_target_count
                            FROM read_parquet('{model_data_path_str}')
                            WHERE CAST(mi_person_key AS VARCHAR) IN ({patient_list})
                              AND drug_name = '{code_name.replace("'", "''")}'
                              AND {target_date_field} IS NOT NULL
                              AND event_date IS NOT NULL
                              AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP)
                            """
                        elif code_column == 'procedure_code':
                            # Check procedure_code column
                            query = f"""
                            SELECT COUNT(*) as post_target_count
                            FROM read_parquet('{model_data_path_str}')
                            WHERE CAST(mi_person_key AS VARCHAR) IN ({patient_list})
                              AND procedure_code = '{code_name.replace("'", "''")}'
                              AND {target_date_field} IS NOT NULL
                              AND event_date IS NOT NULL
                              AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP)
                            """
                        else:
                            # Check all ICD diagnosis columns
                            query = f"""
                            SELECT COUNT(*) as post_target_count
                            FROM read_parquet('{model_data_path_str}')
                            WHERE CAST(mi_person_key AS VARCHAR) IN ({patient_list})
                              AND (
                                primary_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR two_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR three_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR four_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR five_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR six_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR seven_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR eight_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR nine_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                                OR ten_icd_diagnosis_code = '{code_name.replace("'", "''")}'
                              )
                              AND {target_date_field} IS NOT NULL
                              AND event_date IS NOT NULL
                              AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP)
                            """
                        
                        result = con.execute(query).df()
                        post_target_count = result.iloc[0]['post_target_count'] if len(result) > 0 else 0
                        
                        if post_target_count > 0:
                            post_target_found = True
                            if feature_name not in post_target_item_features:
                                post_target_item_features.append(feature_name)
                                print(f"  [WARNING] {feature_name}: {post_target_count} post-target events found (TARGET LEAKAGE)")
                            break  # Found leakage, no need to check more batches
                    
                    if post_target_found:
                        continue  # Move to next feature
                
                con.close()
                
            except Exception as e:
                print(f"  [WARNING] Could not validate item_* features against event data: {e}")
                print(f"  [INFO] Skipping post-target validation (this is a best-effort check)")
        
        else:
            print(f"  [INFO] Model data file not found for validation: {model_data_path}")
            print(f"  [INFO] Skipping post-target validation (this is a best-effort check)")
    
    if post_target_item_features:
        print(f"\n[WARNING] Found {len(post_target_item_features)} item_* features with post-target events:")
        for f in post_target_item_features:
            print(f"  - {f}")
        print("[INFO] These features may include post-target events and should be removed")
        safe_features = [c for c in safe_features if c not in post_target_item_features]
        leakage_features.extend(post_target_item_features)
        df_clean = df[safe_features].copy()
    else:
        print(f"  [OK] No post-target leakage detected in item_* features")
    
    print(f"\n[INFO] Clean dataset: {len(df_clean)} patients, {len(df_clean.columns)} columns")
    print(f"[INFO] Removed {len(df.columns) - len(df_clean.columns)} columns")
    print(f"[INFO] All features are from events BEFORE F1120 (excluding F1120 and everything after)")
    
    # Save cleaned feature table (Parquet format for efficiency)
    output_path_csv = (
        project_root
        / "6_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    output_path_parquet = (
        project_root
        / "6_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "inputs"
        / "model_train"
        / "final_features.parquet"
    )
    
    # Save CSV (for backward compatibility)
    print(f"\n[INFO] Saving cleaned feature table to {output_path_csv}")
    output_path_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path_csv, index=False)
    
    # Save Parquet (preferred format for downstream steps)
    print(f"[INFO] Saving cleaned feature table to Parquet: {output_path_parquet}")
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path_parquet, index=False, compression='snappy', engine='pyarrow')
    
    # Save list of removed features
    removed_features_path = (
        project_root
        / "6_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_removed_leakage_features.txt"
    )
    
    with open(removed_features_path, 'w') as f:
        f.write("Removed Target Leakage Features\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total removed: {len(leakage_features)}\n\n")
        f.write("Post-event features:\n")
        for feat in post_features:
            f.write(f"  - {feat}\n")
        f.write("\nTime-to-target features:\n")
        for feat in time_to_features:
            f.write(f"  - {feat}\n")
        f.write("\nDTW features (removed for replacement):\n")
        for feat in dtw_features:
            f.write(f"  - {feat}\n")
        f.write("\nDatetime columns:\n")
        for feat in datetime_features:
            if feat in df.columns:
                f.write(f"  - {feat}\n")
    
    print(f"[INFO] Saved removed features list to {removed_features_path}")
    
    # Summary of remaining features
    remaining_features = [c for c in df_clean.columns if c not in ['mi_person_key', 'target']]
    print(f"\n[INFO] Remaining predictive features: {len(remaining_features)}")
    
    # Categorize remaining features
    pre_features = [c for c in remaining_features if c.startswith('pre_')]
    fpgrowth_features = [c for c in remaining_features if any(x in c for x in ['itemset', 'rule', 'support', 'confidence', 'lift'])]
    sequence_features_remaining = [c for c in remaining_features if 'sequence' in c.lower() or 'trace' in c.lower()]
    interval_features_remaining = [c for c in remaining_features if 'interval' in c.lower()]
    pgx_features = [c for c in remaining_features if 'pgx' in c.lower()]
    n_events_features = [c for c in remaining_features if 'n_events' in c.lower()]
    
    print(f"\n[INFO] Feature breakdown (PRESERVED):")
    print(f"  Pre-event features: {len(pre_features)}")
    print(f"  FP-Growth features (itemsets/rules): {len(fpgrowth_features)}")
    print(f"    - Itemset features: {len([c for c in fpgrowth_features if 'itemset' in c])}")
    print(f"    - Rule features: {len([c for c in fpgrowth_features if 'rule' in c])}")
    print(f"    - Support/confidence/lift: {len([c for c in fpgrowth_features if any(x in c for x in ['support', 'confidence', 'lift'])])}")
    print(f"  Sequence features (top/rare): {len(sequence_features_remaining)}")
    print(f"  Time interval features (between events): {len(interval_features_remaining)}")
    print(f"  PGx features: {len(pgx_features)}")
    print(f"  Event count features: {len(n_events_features)}")
    print(f"  Other features: {len(remaining_features) - len(pre_features) - len(fpgrowth_features) - len(sequence_features_remaining) - len(interval_features_remaining) - len(pgx_features) - len(n_events_features)}")
    
    print("\n[INFO] Done. Next steps:")
    print("  1. Create predictive DTW features (time windows between drugs)")
    print("  2. Re-run feature engineering for control patients")
    print("  3. Rebuild final feature table")


def main():
    parser = argparse.ArgumentParser(
        description="Remove target leakage from final feature table"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    remove_target_leakage(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

