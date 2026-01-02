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
        / "8_final_model"
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
    
    print(f"\n[INFO] Clean dataset: {len(df_clean)} patients, {len(df_clean.columns)} columns")
    print(f"[INFO] Removed {len(df.columns) - len(df_clean.columns)} columns")
    print(f"[INFO] All features are from events BEFORE F1120 (excluding F1120 and everything after)")
    
    # Save cleaned feature table
    output_path = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    
    print(f"\n[INFO] Saving cleaned feature table to {output_path}")
    df_clean.to_csv(output_path, index=False)
    
    # Save list of removed features
    removed_features_path = (
        project_root
        / "8_final_model"
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

