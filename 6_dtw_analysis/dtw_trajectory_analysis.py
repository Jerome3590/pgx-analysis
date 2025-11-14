"""
Enhanced DTW Analysis for Patient Trajectory Development

This module extends DTW analysis to:
1. Develop patient trajectories using temporal sequences (drugs, ICD codes, CPT codes)
2. Integrate with cohort temporal fields (days_to_target_event)
3. Create trajectory archetypes and patterns
4. Enable trajectory-based risk prediction

Key Advantages over FPGrowth/BupaR:
- Handles variable-length sequences with temporal warping
- Identifies similar trajectories even with timing differences
- Creates trajectory clusters for personalized medicine
- Enables trajectory-based outcome prediction
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import duckdb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.common_imports import (
    s3_client, 
    S3_BUCKET, 
    get_logger
)
from helpers_1997_13.duckdb_utils import get_duckdb_connection
from helpers_1997_13.s3_utils import (
    get_output_paths,
    get_cohort_parquet_path,
    save_to_s3_parquet,
    save_to_s3_json,
    s3_exists
)

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. Install with: pip install dtaidistance")


class PatientTrajectoryAnalyzer:
    """
    Enhanced DTW analyzer for developing patient trajectories from cohort data.
    
    Creates trajectories from:
    - Drug sequences (with temporal positioning via days_to_target_event)
    - ICD code sequences
    - CPT code sequences
    - Combined multi-modal trajectories
    """
    
    def __init__(self, cohort_name: str, age_band: str, event_year: str, 
                 item_type: str = 'drug'):
        """
        Initialize trajectory analyzer.
        
        Args:
            cohort_name: Cohort name (opioid_ed, ed_non_opioid)
            age_band: Age band (e.g., '65-74')
            event_year: Event year (e.g., '2020')
            item_type: Type of items for trajectory ('drug', 'icd', 'cpt', 'combined')
        """
        self.cohort_name = cohort_name.lower()
        self.age_band = age_band
        self.event_year = str(event_year)
        self.item_type = item_type
        self.logger = get_logger("dtw_trajectory", age_band, event_year)
        
        # Initialize DuckDB connection
        self.conn = get_duckdb_connection(self.logger)
        self.conn.sql("INSTALL httpfs; LOAD httpfs;")
        self.conn.sql("CALL load_aws_credentials('');")
        
        # Trajectory storage
        self.patient_trajectories = {}
        self.trajectory_sequences = {}
        self.similarity_matrix = None
        self.trajectory_clusters = {}
        self.archetype_trajectories = {}
        
    def load_cohort_data(self) -> pd.DataFrame:
        """Load cohort data with temporal fields."""
        self.logger.info(f"Loading {self.cohort_name} cohort data...")
        
        cohort_path = get_cohort_parquet_path(
            self.cohort_name, self.age_band, self.event_year
        )
        
        # Build query based on item type and cohort
        if self.item_type == 'drug':
            if self.cohort_name == 'ed_non_opioid':
                # ED_NON_OPIOID: Include drugs with temporal positioning
                query = f"""
                SELECT 
                    mi_person_key,
                    event_date,
                    drug_name as item_name,
                    days_to_target_event,
                    first_ed_non_opioid_date,
                    is_target_case,
                    event_type
                FROM read_parquet('{cohort_path}')
                WHERE drug_name IS NOT NULL
                  AND event_type = 'pharmacy'
                  AND (
                      (is_target_case = 1)
                      OR (is_target_case = 0 AND days_to_target_event IS NOT NULL 
                          AND days_to_target_event >= 0 AND days_to_target_event <= 30)
                  )
                ORDER BY mi_person_key, days_to_target_event DESC NULLS LAST, event_date
                """
            else:
                # OPIOID_ED: All drugs (no temporal filtering)
                query = f"""
                SELECT 
                    mi_person_key,
                    event_date,
                    drug_name as item_name,
                    NULL as days_to_target_event,
                    first_opioid_ed_date,
                    is_target_case,
                    event_type
                FROM read_parquet('{cohort_path}')
                WHERE drug_name IS NOT NULL
                  AND event_type = 'pharmacy'
                ORDER BY mi_person_key, event_date
                """
        
        elif self.item_type == 'icd':
            query = f"""
            SELECT 
                mi_person_key,
                event_date,
                primary_icd_diagnosis_code as item_name,
                NULL as days_to_target_event,
                first_opioid_ed_date,
                is_target_case,
                event_type
            FROM read_parquet('{cohort_path}')
            WHERE primary_icd_diagnosis_code IS NOT NULL
              AND event_type = 'medical'
            ORDER BY mi_person_key, event_date
            """
        
        elif self.item_type == 'cpt':
            query = f"""
            SELECT 
                mi_person_key,
                event_date,
                procedure_code as item_name,
                NULL as days_to_target_event,
                first_opioid_ed_date,
                is_target_case,
                event_type
            FROM read_parquet('{cohort_path}')
            WHERE procedure_code IS NOT NULL
              AND event_type = 'medical'
            ORDER BY mi_person_key, event_date
            """
        
        else:
            raise ValueError(f"Unsupported item_type: {self.item_type}")
        
        df = self.conn.sql(query).df()
        self.logger.info(
            f"Loaded {len(df):,} {self.item_type} records for "
            f"{df['mi_person_key'].nunique():,} patients"
        )
        
        return df
    
    def create_temporal_trajectories(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Create patient trajectories with temporal positioning.
        
        For ED_NON_OPIOID: Uses days_to_target_event for temporal alignment
        For OPIOID_ED: Uses event_date for temporal ordering
        
        Returns:
            Dictionary mapping patient_id to trajectory (list of events with timing)
        """
        self.logger.info("Creating temporal trajectories...")
        
        trajectories = {}
        
        for patient_id in df['mi_person_key'].unique():
            patient_data = df[df['mi_person_key'] == patient_id].copy()
            
            # Sort by temporal position
            if self.cohort_name == 'ed_non_opioid' and 'days_to_target_event' in patient_data.columns:
                # Use days_to_target_event for temporal alignment (descending: 30 -> 0)
                patient_data = patient_data.sort_values(
                    'days_to_target_event', 
                    ascending=False, 
                    na_position='last'
                )
                temporal_key = 'days_to_target_event'
            else:
                # Use event_date for temporal ordering
                patient_data = patient_data.sort_values('event_date')
                temporal_key = 'event_date'
            
            # Create trajectory with temporal information
            trajectory = []
            for _, row in patient_data.iterrows():
                event = {
                    'item': str(row['item_name']),
                    'temporal_position': (
                        row[temporal_key] if pd.notnull(row[temporal_key]) 
                        else None
                    ),
                    'event_date': str(row['event_date']),
                    'is_target': bool(row.get('is_target_case', False))
                }
                trajectory.append(event)
            
            if trajectory:
                trajectories[patient_id] = trajectory
        
        self.logger.info(f"Created {len(trajectories)} patient trajectories")
        self.patient_trajectories = trajectories
        
        return trajectories
    
    def encode_trajectory_sequence(self, trajectory: List[Dict]) -> Tuple[List[int], List[float]]:
        """
        Encode trajectory to numerical sequence for DTW.
        
        Returns:
            Tuple of (encoded_items, temporal_positions)
        """
        # Create encoding map if not exists
        if not hasattr(self, 'item_encoding_map'):
            all_items = set()
            for traj in self.patient_trajectories.values():
                for event in traj:
                    all_items.add(event['item'])
            
            self.item_encoding_map = {
                item: idx for idx, item in enumerate(sorted(all_items))
            }
            self.logger.info(
                f"Created encoding map for {len(self.item_encoding_map)} unique items"
            )
        
        # Encode sequence
        encoded_items = []
        temporal_positions = []
        
        for event in trajectory:
            item = event['item']
            encoded_items.append(self.item_encoding_map.get(item, -1))
            
            # Handle temporal position
            temp_pos = event.get('temporal_position')
            if temp_pos is not None:
                if isinstance(temp_pos, (int, float)):
                    temporal_positions.append(float(temp_pos))
                else:
                    temporal_positions.append(0.0)  # Default if invalid
            else:
                temporal_positions.append(0.0)
        
        # Filter out invalid encodings
        valid_indices = [i for i, x in enumerate(encoded_items) if x != -1]
        encoded_items = [encoded_items[i] for i in valid_indices]
        temporal_positions = [temporal_positions[i] for i in valid_indices]
        
        return encoded_items, temporal_positions
    
    def calculate_trajectory_similarity_matrix(self) -> np.ndarray:
        """Calculate DTW similarity matrix for all trajectories."""
        if not DTW_AVAILABLE:
            raise ImportError(
                "dtaidistance package not available. "
                "Install with: pip install dtaidistance"
            )
        
        self.logger.info("Calculating DTW similarity matrix for trajectories...")
        
        patient_ids = list(self.patient_trajectories.keys())
        n_patients = len(patient_ids)
        
        # Encode all trajectories
        encoded_sequences = {}
        for pid in patient_ids:
            items, temps = self.encode_trajectory_sequence(
                self.patient_trajectories[pid]
            )
            encoded_sequences[pid] = (items, temps)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_patients, n_patients))
        
        # Calculate pairwise DTW distances
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                seq1_items, seq1_temps = encoded_sequences[patient_ids[i]]
                seq2_items, seq2_temps = encoded_sequences[patient_ids[j]]
                
                if seq1_items and seq2_items:
                    # Calculate DTW distance on item sequences
                    distance = dtw.distance(seq1_items, seq2_items)
                    similarity_matrix[i][j] = distance
                    similarity_matrix[j][i] = distance
                else:
                    similarity_matrix[i][j] = np.inf
                    similarity_matrix[j][i] = np.inf
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processed {i + 1}/{n_patients} patients")
        
        self.similarity_matrix = similarity_matrix
        self.patient_ids = patient_ids
        
        self.logger.info("DTW similarity matrix calculation complete")
        return similarity_matrix
    
    def cluster_trajectories(self, n_clusters: int = 5) -> Dict:
        """Cluster patients based on trajectory similarity."""
        self.logger.info(f"Clustering trajectories into {n_clusters} groups...")
        
        if self.similarity_matrix is None:
            self.calculate_trajectory_similarity_matrix()
        
        # Handle infinite distances
        finite_matrix = np.where(
            np.isinf(self.similarity_matrix),
            np.max(self.similarity_matrix[~np.isinf(self.similarity_matrix)]) * 2,
            self.similarity_matrix
        )
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(finite_matrix)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(
            finite_matrix, cluster_labels, metric='precomputed'
        )
        
        # Create cluster mapping
        patient_cluster_map = dict(zip(self.patient_ids, cluster_labels))
        
        # Extract archetype trajectories (centroid-like trajectories per cluster)
        archetypes = self.extract_archetype_trajectories(
            patient_cluster_map, n_clusters
        )
        
        cluster_results = {
            'cluster_labels': cluster_labels.tolist(),
            'patient_ids': self.patient_ids,
            'patient_cluster_map': patient_cluster_map,
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette_avg),
            'cluster_sizes': np.bincount(cluster_labels).tolist(),
            'archetype_trajectories': archetypes
        }
        
        self.trajectory_clusters = cluster_results
        self.archetype_trajectories = archetypes
        
        self.logger.info(
            f"Clustering complete. Silhouette score: {silhouette_avg:.3f}"
        )
        self.logger.info(
            f"Cluster sizes: {dict(enumerate(cluster_results['cluster_sizes']))}"
        )
        
        return cluster_results
    
    def extract_archetype_trajectories(
        self, 
        patient_cluster_map: Dict[str, int], 
        n_clusters: int
    ) -> Dict[int, List[Dict]]:
        """
        Extract archetype (representative) trajectory for each cluster.
        
        Uses median trajectory or most common pattern per cluster.
        """
        archetypes = {}
        
        for cluster_id in range(n_clusters):
            cluster_patients = [
                pid for pid, cid in patient_cluster_map.items() 
                if cid == cluster_id
            ]
            
            if not cluster_patients:
                continue
            
            # Get trajectories for this cluster
            cluster_trajectories = [
                self.patient_trajectories[pid] 
                for pid in cluster_patients
            ]
            
            # Find most common trajectory pattern
            # (simplified: use trajectory with median length)
            lengths = [len(t) for t in cluster_trajectories]
            median_length = int(np.median(lengths))
            
            # Find trajectory closest to median length
            closest_traj = min(
                cluster_trajectories,
                key=lambda t: abs(len(t) - median_length)
            )
            
            archetypes[cluster_id] = closest_traj
        
        return archetypes
    
    def analyze_trajectory_patterns(self) -> Dict:
        """Analyze patterns within trajectory clusters."""
        self.logger.info("Analyzing trajectory patterns...")
        
        if not self.trajectory_clusters:
            raise ValueError("Must run clustering first")
        
        patterns = {}
        patient_cluster_map = self.trajectory_clusters['patient_cluster_map']
        
        for cluster_id in range(self.trajectory_clusters['n_clusters']):
            cluster_patients = [
                pid for pid, cid in patient_cluster_map.items() 
                if cid == cluster_id
            ]
            
            cluster_trajectories = [
                self.patient_trajectories[pid] 
                for pid in cluster_patients
            ]
            
            # Analyze common patterns
            # 1. Most common items
            all_items = []
            for traj in cluster_trajectories:
                all_items.extend([e['item'] for e in traj])
            
            from collections import Counter
            item_counts = Counter(all_items)
            
            # 2. Average trajectory length
            avg_length = np.mean([len(t) for t in cluster_trajectories])
            
            # 3. Temporal characteristics (if available)
            temporal_info = []
            for traj in cluster_trajectories:
                temps = [
                    e['temporal_position'] 
                    for e in traj 
                    if e.get('temporal_position') is not None
                ]
                if temps:
                    temporal_info.extend(temps)
            
            patterns[cluster_id] = {
                'n_patients': len(cluster_patients),
                'avg_trajectory_length': float(avg_length),
                'most_common_items': dict(item_counts.most_common(10)),
                'avg_temporal_position': (
                    float(np.mean(temporal_info)) if temporal_info else None
                ),
                'archetype_trajectory': self.archetype_trajectories.get(cluster_id, [])
            }
        
        return patterns
    
    def save_trajectory_results(self):
        """Save trajectory analysis results to S3."""
        self.logger.info("Saving trajectory results...")
        
        # Prepare results
        results = {
            'metadata': {
                'cohort_name': self.cohort_name,
                'age_band': self.age_band,
                'event_year': self.event_year,
                'item_type': self.item_type,
                'analysis_timestamp': datetime.now().isoformat(),
                'n_patients': len(self.patient_trajectories),
                'n_unique_items': len(self.item_encoding_map) if hasattr(self, 'item_encoding_map') else 0
            },
            'trajectory_clusters': self.trajectory_clusters,
            'archetype_trajectories': self.archetype_trajectories,
            'trajectory_patterns': self.analyze_trajectory_patterns(),
            'item_encoding_map': (
                self.item_encoding_map if hasattr(self, 'item_encoding_map') else {}
            )
        }
        
        # Save to S3
        base_path = (
            f"s3://{S3_BUCKET}/dtw_trajectories/"
            f"{self.cohort_name}/{self.age_band}/{self.event_year}"
        )
        
        json_path = f"{base_path}/trajectory_results_{self.item_type}.json"
        save_to_s3_json(results, json_path, self.logger)
        
        # Save patient trajectories as Parquet
        trajectory_records = []
        for pid, traj in self.patient_trajectories.items():
            cluster_id = self.trajectory_clusters.get(
                'patient_cluster_map', {}
            ).get(pid, -1)
            
            trajectory_records.append({
                'mi_person_key': pid,
                'cluster_id': cluster_id,
                'trajectory_length': len(traj),
                'trajectory_items': [e['item'] for e in traj],
                'temporal_positions': [
                    e.get('temporal_position') for e in traj
                ]
            })
        
        traj_df = pd.DataFrame(trajectory_records)
        parquet_path = f"{base_path}/patient_trajectories_{self.item_type}.parquet"
        save_to_s3_parquet(traj_df, parquet_path, self.logger)
        
        self.logger.info(f"Results saved:")
        self.logger.info(f"  - JSON: {json_path}")
        self.logger.info(f"  - Parquet: {parquet_path}")
    
    def run_analysis(self, n_clusters: int = 5):
        """Run complete trajectory analysis pipeline."""
        self.logger.info(
            f"Starting trajectory analysis for {self.cohort_name} "
            f"({self.item_type})..."
        )
        
        try:
            # Step 1: Load data
            df = self.load_cohort_data()
            
            # Step 2: Create trajectories
            trajectories = self.create_temporal_trajectories(df)
            
            if not trajectories:
                self.logger.warning("No trajectories found. Analysis cannot proceed.")
                return None
            
            # Step 3: Calculate similarity matrix
            similarity_matrix = self.calculate_trajectory_similarity_matrix()
            
            # Step 4: Cluster trajectories
            cluster_results = self.cluster_trajectories(n_clusters)
            
            # Step 5: Analyze patterns
            patterns = self.analyze_trajectory_patterns()
            
            # Step 6: Save results
            self.save_trajectory_results()
            
            self.logger.info("Trajectory analysis completed successfully!")
            
            return {
                'cluster_results': cluster_results,
                'patterns': patterns,
                'archetype_trajectories': self.archetype_trajectories,
                'n_patients': len(trajectories)
            }
            
        except Exception as e:
            self.logger.error(f"Error in trajectory analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


def main():
    """Main function to run trajectory analysis."""
    parser = argparse.ArgumentParser(
        description='DTW Trajectory Analysis for Patient Sequences'
    )
    parser.add_argument(
        '--cohort',
        required=True,
        choices=['opioid_ed', 'ed_non_opioid'],
        help='Cohort to analyze'
    )
    parser.add_argument(
        '--age-band',
        required=True,
        help='Age band (e.g., "65-74")'
    )
    parser.add_argument(
        '--event-year',
        type=str,
        required=True,
        help='Event year (e.g., "2020")'
    )
    parser.add_argument(
        '--item-type',
        choices=['drug', 'icd', 'cpt'],
        default='drug',
        help='Type of items for trajectory (default: drug)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of clusters to create (default: 5)'
    )
    
    args = parser.parse_args()
    
    if not DTW_AVAILABLE:
        print("Error: dtaidistance package not available.")
        print("Install with: pip install dtaidistance")
        sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = PatientTrajectoryAnalyzer(
        args.cohort,
        args.age_band,
        args.event_year,
        args.item_type
    )
    
    try:
        results = analyzer.run_analysis(n_clusters=args.n_clusters)
        
        print(f"\nTrajectory Analysis completed successfully!")
        print(f"Cohort: {args.cohort}")
        print(f"Age Band: {args.age_band}")
        print(f"Event Year: {args.event_year}")
        print(f"Item Type: {args.item_type}")
        print(f"Number of Patients: {results['n_patients']}")
        print(f"Number of Clusters: {args.n_clusters}")
        print(
            f"Silhouette Score: "
            f"{results['cluster_results']['silhouette_score']:.3f}"
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

