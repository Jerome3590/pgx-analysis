#!/usr/bin/env python3
"""
Create DTW trajectory visualizations.

This script creates visualizations from DTW trajectory analysis:
- Trajectory length distribution
- Most common items in trajectories
- Sample trajectory timelines
- DTW distance distribution
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available. Visualizations will be skipped.")


def create_dtw_visualizations(
    patient_trajectories: Dict[str, List[str]],
    dtw_features_df: pd.DataFrame,
    cohort_name: str,
    age_band: str,
    output_dir: Path,
    n_sample_trajectories: int = 10
) -> List[Path]:
    """
    Create DTW trajectory visualizations.
    
    Parameters:
    -----------
    patient_trajectories : Dict[str, List[str]]
        Dictionary mapping patient IDs to trajectory sequences
    dtw_features_df : pd.DataFrame
        DataFrame with DTW features (must include trajectory_length, trajectory_diversity)
    cohort_name : str
        Cohort name (e.g., 'opioid_ed')
    age_band : str
        Age band (e.g., '13-24')
    output_dir : Path
        Directory to save visualizations
    n_sample_trajectories : int
        Number of sample trajectories to visualize
    
    Returns:
    --------
    List[Path]
        List of paths to created visualization files
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping visualizations.")
        return []
    
    if not patient_trajectories:
        logger.warning("No patient trajectories provided. Skipping visualizations.")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    logger.info(f"Creating DTW visualizations for {cohort_name} / {age_band}")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    age_band_fname = age_band.replace("-", "_")
    
    # 1. Trajectory Length Distribution
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'DTW Trajectory Analysis - {cohort_name} / {age_band}', 
                    fontsize=16, fontweight='bold')
        
        # Panel 1: Trajectory Length Distribution
        ax1 = axes[0, 0]
        if 'trajectory_length' in dtw_features_df.columns:
            lengths = dtw_features_df['trajectory_length'].dropna()
            if len(lengths) > 0:
                ax1.hist(lengths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
                ax1.set_title('Trajectory Length Distribution')
                ax1.set_xlabel('Trajectory Length (number of events)')
                ax1.set_ylabel('Number of Patients')
                ax1.axvline(lengths.median(), color='red', linestyle='--', 
                           label=f'Median: {lengths.median():.1f}')
                ax1.legend()
        else:
            # Fallback: compute from trajectories
            lengths = [len(traj) for traj in patient_trajectories.values()]
            if lengths:
                ax1.hist(lengths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
                ax1.set_title('Trajectory Length Distribution')
                ax1.set_xlabel('Trajectory Length (number of events)')
                ax1.set_ylabel('Number of Patients')
                ax1.axvline(np.median(lengths), color='red', linestyle='--', 
                           label=f'Median: {np.median(lengths):.1f}')
                ax1.legend()
        
        # Panel 2: Trajectory Diversity Distribution
        ax2 = axes[0, 1]
        if 'trajectory_diversity' in dtw_features_df.columns:
            diversity = dtw_features_df['trajectory_diversity'].dropna()
            if len(diversity) > 0:
                ax2.hist(diversity, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
                ax2.set_title('Trajectory Diversity Distribution')
                ax2.set_xlabel('Number of Unique Items')
                ax2.set_ylabel('Number of Patients')
                ax2.axvline(diversity.median(), color='red', linestyle='--', 
                           label=f'Median: {diversity.median():.1f}')
                ax2.legend()
        else:
            # Fallback: compute from trajectories
            diversity = [len(set(traj)) for traj in patient_trajectories.values()]
            if diversity:
                ax2.hist(diversity, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
                ax2.set_title('Trajectory Diversity Distribution')
                ax2.set_xlabel('Number of Unique Items')
                ax2.set_ylabel('Number of Patients')
                ax2.axvline(np.median(diversity), color='red', linestyle='--', 
                           label=f'Median: {np.median(diversity):.1f}')
                ax2.legend()
        
        # Panel 3: Most Common Items
        ax3 = axes[1, 0]
        all_items = []
        for traj in patient_trajectories.values():
            all_items.extend(traj)
        
        if all_items:
            item_counts = Counter(all_items)
            top_items = item_counts.most_common(20)
            items, counts = zip(*top_items) if top_items else ([], [])
            
            if items:
                # Truncate long item names for display
                display_items = [item[:30] + '...' if len(item) > 30 else item for item in items]
                y_pos = np.arange(len(display_items))
                ax3.barh(y_pos, counts, color='lightgreen', alpha=0.7)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(display_items, fontsize=8)
                ax3.set_xlabel('Frequency')
                ax3.set_title('Top 20 Most Common Items in Trajectories')
                ax3.invert_yaxis()
        
        # Panel 4: DTW Distance Distribution
        ax4 = axes[1, 1]
        distance_cols = [col for col in dtw_features_df.columns 
                        if col.startswith('dtw_distance_to_prototype_')]
        if distance_cols:
            # Get minimum distance for each patient
            min_distances = dtw_features_df[distance_cols].min(axis=1).dropna()
            if len(min_distances) > 0:
                # Filter out infinite values
                finite_distances = min_distances[np.isfinite(min_distances)]
                if len(finite_distances) > 0:
                    ax4.hist(finite_distances, bins=50, color='plum', alpha=0.7, edgecolor='black')
                    ax4.set_title('DTW Distance to Nearest Prototype')
                    ax4.set_xlabel('DTW Distance')
                    ax4.set_ylabel('Number of Patients')
                    ax4.axvline(finite_distances.median(), color='red', linestyle='--', 
                               label=f'Median: {finite_distances.median():.2f}')
                    ax4.legend()
        
        plt.tight_layout()
        
        output_path = output_dir / f"dtw_trajectory_analysis_{cohort_name}_{age_band_fname}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_files.append(output_path)
        logger.info(f"Created visualization: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating trajectory visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Sample Trajectory Timeline Visualization
    try:
        if len(patient_trajectories) > 0:
            # Select sample trajectories (longest, shortest, median length)
            traj_lengths = [(pid, len(traj)) for pid, traj in patient_trajectories.items()]
            traj_lengths.sort(key=lambda x: x[1])
            
            sample_indices = []
            n_trajs = len(traj_lengths)
            if n_trajs >= 3:
                sample_indices = [
                    traj_lengths[0][0],  # Shortest
                    traj_lengths[n_trajs // 2][0],  # Median
                    traj_lengths[-1][0]  # Longest
                ]
            else:
                sample_indices = [traj_lengths[i][0] for i in range(min(n_trajs, n_sample_trajectories))]
            
            # Add random samples if needed
            if len(sample_indices) < n_sample_trajectories and n_trajs > len(sample_indices):
                remaining = [pid for pid, _ in traj_lengths if pid not in sample_indices]
                np.random.seed(42)
                additional = np.random.choice(remaining, 
                                            size=min(n_sample_trajectories - len(sample_indices), 
                                                    len(remaining)), 
                                            replace=False)
                sample_indices.extend(additional.tolist())
            
            fig, ax = plt.subplots(figsize=(14, max(8, len(sample_indices) * 0.8)))
            
            y_pos = 0
            for pid in sample_indices[:n_sample_trajectories]:
                traj = patient_trajectories.get(pid, [])
                if not traj:
                    continue
                
                # Truncate item names for display
                display_traj = [item[:20] + '...' if len(item) > 20 else item for item in traj]
                
                # Create timeline
                x_positions = np.arange(len(traj))
                ax.scatter(x_positions, [y_pos] * len(traj), s=50, alpha=0.6)
                
                # Add labels for first and last items
                if len(traj) > 0:
                    ax.text(-0.5, y_pos, f"P{pid[:8]}...", fontsize=8, 
                           verticalalignment='center', fontweight='bold')
                    ax.text(len(traj) - 0.5, y_pos, f"({len(traj)} items)", 
                           fontsize=7, verticalalignment='center', style='italic')
                
                # Add item labels for key positions
                if len(traj) <= 10:
                    for i, item in enumerate(display_traj):
                        ax.text(i, y_pos + 0.15, item, fontsize=6, 
                               rotation=45, ha='left', va='bottom')
                else:
                    # Show first, middle, last
                    ax.text(0, y_pos + 0.15, display_traj[0], fontsize=6, 
                           rotation=45, ha='left', va='bottom')
                    ax.text(len(traj) // 2, y_pos + 0.15, display_traj[len(traj) // 2], 
                           fontsize=6, rotation=45, ha='left', va='bottom')
                    ax.text(len(traj) - 1, y_pos + 0.15, display_traj[-1], 
                           fontsize=6, rotation=45, ha='left', va='bottom')
                
                y_pos += 1
            
            ax.set_xlabel('Event Position in Trajectory')
            ax.set_ylabel('Sample Patient Trajectories')
            ax.set_title(f'Sample Patient Trajectories - {cohort_name} / {age_band}')
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = output_dir / f"dtw_sample_trajectories_{cohort_name}_{age_band_fname}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(output_path)
            logger.info(f"Created sample trajectory visualization: {output_path}")
            
    except Exception as e:
        logger.error(f"Error creating sample trajectory visualization: {e}")
        import traceback
        traceback.print_exc()
    
    return created_files


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create DTW trajectory visualizations")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 13-24)")
    parser.add_argument("--features-csv", required=True, help="Path to DTW features CSV")
    parser.add_argument("--trajectories-json", help="Path to trajectories JSON (optional)")
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    parser.add_argument("--n-samples", type=int, default=10, 
                        help="Number of sample trajectories to visualize")
    
    args = parser.parse_args()
    
    if not VISUALIZATION_AVAILABLE:
        logger.error("matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
        return
    
    project_root = PROJECT_ROOT
    
    # Load DTW features
    features_path = Path(args.features_csv)
    if not features_path.exists():
        logger.error(f"Features CSV not found: {features_path}")
        return
    
    dtw_features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(dtw_features_df)} patient features")
    
    # Load trajectories if provided
    patient_trajectories = {}
    if args.trajectories_json:
        traj_path = Path(args.trajectories_json)
        if traj_path.exists():
            import json
            with open(traj_path, 'r') as f:
                patient_trajectories = json.load(f)
            logger.info(f"Loaded {len(patient_trajectories)} patient trajectories")
    
    # If trajectories not provided, try to reconstruct from features
    if not patient_trajectories and 'trajectory_length' in dtw_features_df.columns:
        logger.warning("Trajectories not provided. Visualizations will be limited.")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        age_band_fname = args.age_band.replace("-", "_")
        output_dir = project_root / "3b_feature_importance_eda" / "outputs" / args.cohort / age_band_fname / "plots"
    
    # Create visualizations
    created_files = create_dtw_visualizations(
        patient_trajectories=patient_trajectories,
        dtw_features_df=dtw_features_df,
        cohort_name=args.cohort,
        age_band=args.age_band,
        output_dir=output_dir,
        n_sample_trajectories=args.n_samples
    )
    
    if created_files:
        print(f"\nCreated {len(created_files)} visualization files:")
        for f in created_files:
            print(f"  - {f}")
    else:
        print("\nNo visualizations were created.")


if __name__ == "__main__":
    main()
