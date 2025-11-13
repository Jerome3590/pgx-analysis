#!/usr/bin/env python3
"""
Dynamic Time Warping (DTW) Analysis for Drug Sequence Similarity

This script performs DTW analysis on drug exposure sequences from cohort datasets
to identify similar patient patterns and cluster patients based on drug histories.

Author: PGx Analysis Team
Date: 2024
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import duckdb
import boto3

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project utilities
from helpers_1997_13.common_imports import (
    s3_client, 
    S3_BUCKET, 
    get_logger
)

from helpers_1997_13.duckdb_utils import (
    get_duckdb_connection,
    execute_duckdb_query
)

from helpers_1997_13.s3_utils import (
    get_output_paths,
    save_to_s3_parquet,
    save_to_s3_json
)

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. Install with: pip install dtaidistance")


class DTWCohortAnalyzer:
    """
    Dynamic Time Warping analyzer for drug sequence similarity in cohort datasets.
    """
    
    def __init__(self, age_band: str, event_year: int, cohort_name: str):
        """
        Initialize DTW analyzer for a specific cohort.
        
        Args:
            age_band: Age band to analyze (e.g., "0-12", "65-74")
            event_year: Event year to analyze (e.g., 2020)
            cohort_name: Cohort name ("opioid_ed" or "ed_non_opioid")
        """
        self.age_band = age_band
        self.event_year = event_year
        self.cohort_name = cohort_name
        self.logger = get_logger("dtw_analysis", age_band, event_year)
        
        # Initialize DuckDB connection
        self.conn = get_duckdb_connection(self.logger)
        self.conn.sql("INSTALL httpfs; LOAD httpfs;")
        self.conn.sql("CALL load_aws_credentials('');")
        
        # Analysis results storage
        self.similarity_matrix = None
        self.patient_sequences = {}
        self.cluster_results = {}
        self.analysis_metrics = {}
        
    def load_cohort_data(self) -> pd.DataFrame:
        """
        Load cohort data from S3.
        
        Returns:
            DataFrame containing cohort data with drug exposure information
        """
        self.logger.info(f"Loading {self.cohort_name} cohort data...")
        
        # Get cohort path
        paths = get_output_paths(self.cohort_name, self.age_band, self.event_year)
        cohort_path = paths["cohort_parquet"]
        
        # Load data using DuckDB
        query = f"""
        SELECT 
            mi_person_key,
            event_date,
            drug_name,
            therapeutic_class_1,
            therapeutic_class_2,
            therapeutic_class_3,
            days_to_ade,
            days_to_opioid_ed,
            data_source,
            Event,
            First_Event
        FROM read_parquet('{cohort_path}')
        WHERE data_source = 'drug_exposure'
        ORDER BY mi_person_key, event_date
        """
        
        df = self.conn.sql(query).df()
        self.logger.info(f"Loaded {len(df)} drug exposure records for {df['mi_person_key'].nunique()} patients")
        
        return df
    
    def create_drug_sequences(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create drug sequences for each patient.
        
        Args:
            df: DataFrame with drug exposure data
            
        Returns:
            Dictionary mapping patient IDs to their drug sequences
        """
        self.logger.info("Creating drug sequences for each patient...")
        
        sequences = {}
        
        for patient_id in df['mi_person_key'].unique():
            patient_data = df[df['mi_person_key'] == patient_id].sort_values('event_date')
            
            # Create sequence of drug names
            drug_sequence = patient_data['drug_name'].dropna().tolist()
            
            if drug_sequence:  # Only include patients with drug data
                sequences[patient_id] = drug_sequence
        
        self.logger.info(f"Created sequences for {len(sequences)} patients")
        self.patient_sequences = sequences
        
        return sequences
    
    def encode_drug_sequence(self, drug_sequence: List[str]) -> List[int]:
        """
        Encode drug sequence to numerical representation for DTW analysis.
        
        Args:
            drug_sequence: List of drug names
            
        Returns:
            List of encoded drug IDs
        """
        # Create drug encoding map if not exists
        if not hasattr(self, 'drug_encoding_map'):
            all_drugs = set()
            for seq in self.patient_sequences.values():
                all_drugs.update(seq)
            
            self.drug_encoding_map = {drug: idx for idx, drug in enumerate(sorted(all_drugs))}
            self.logger.info(f"Created encoding map for {len(self.drug_encoding_map)} unique drugs")
        
        # Encode sequence
        encoded_sequence = [self.drug_encoding_map.get(drug, -1) for drug in drug_sequence]
        return [x for x in encoded_sequence if x != -1]  # Remove unknown drugs
    
    def calculate_dtw_similarity_matrix(self, sequences: Dict[str, List[str]]) -> np.ndarray:
        """
        Calculate DTW similarity matrix for all patient sequences.
        
        Args:
            sequences: Dictionary of patient drug sequences
            
        Returns:
            Similarity matrix as numpy array
        """
        if not DTW_AVAILABLE:
            raise ImportError("dtaidistance package not available. Install with: pip install dtaidistance")
        
        self.logger.info("Calculating DTW similarity matrix...")
        
        patient_ids = list(sequences.keys())
        n_patients = len(patient_ids)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_patients, n_patients))
        
        # Calculate pairwise DTW distances
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                # Encode sequences
                seq1 = self.encode_drug_sequence(sequences[patient_ids[i]])
                seq2 = self.encode_drug_sequence(sequences[patient_ids[j]])
                
                if seq1 and seq2:  # Only calculate if sequences are not empty
                    # Calculate DTW distance
                    distance = dtw.distance(seq1, seq2)
                    similarity_matrix[i][j] = distance
                    similarity_matrix[j][i] = distance
                else:
                    # Handle empty sequences
                    similarity_matrix[i][j] = np.inf
                    similarity_matrix[j][i] = np.inf
            
            # Progress logging
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{n_patients} patients")
        
        self.similarity_matrix = similarity_matrix
        self.patient_ids = patient_ids
        
        self.logger.info("DTW similarity matrix calculation complete")
        return similarity_matrix
    
    def cluster_patients(self, similarity_matrix: np.ndarray, n_clusters: int = 5) -> Dict:
        """
        Cluster patients based on DTW similarity matrix.
        
        Args:
            similarity_matrix: DTW similarity matrix
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary containing clustering results
        """
        self.logger.info(f"Clustering patients into {n_clusters} groups...")
        
        # Handle infinite distances (replace with large value)
        finite_matrix = np.where(np.isinf(similarity_matrix), 
                                np.max(similarity_matrix[~np.isinf(similarity_matrix)]) * 2, 
                                similarity_matrix)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(finite_matrix)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(finite_matrix, cluster_labels, metric='precomputed')
        
        # Create results dictionary
        cluster_results = {
            'cluster_labels': cluster_labels,
            'patient_ids': self.patient_ids,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_sizes': np.bincount(cluster_labels),
            'similarity_matrix': similarity_matrix
        }
        
        self.cluster_results = cluster_results
        
        self.logger.info(f"Clustering complete. Silhouette score: {silhouette_avg:.3f}")
        self.logger.info(f"Cluster sizes: {dict(enumerate(cluster_results['cluster_sizes']))}")
        
        return cluster_results
    
    def analyze_cluster_characteristics(self, df: pd.DataFrame, cluster_results: Dict) -> Dict:
        """
        Analyze characteristics of each cluster.
        
        Args:
            df: Original cohort data
            cluster_results: Results from clustering
            
        Returns:
            Dictionary with cluster characteristics
        """
        self.logger.info("Analyzing cluster characteristics...")
        
        # Create patient-cluster mapping
        patient_cluster_map = dict(zip(cluster_results['patient_ids'], 
                                      cluster_results['cluster_labels']))
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = df_with_clusters['mi_person_key'].map(patient_cluster_map)
        
        cluster_characteristics = {}
        
        for cluster_id in range(cluster_results['n_clusters']):
            cluster_patients = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            if len(cluster_patients) == 0:
                continue
            
            # Calculate cluster characteristics
            characteristics = {
                'n_patients': cluster_patients['mi_person_key'].nunique(),
                'n_drug_events': len(cluster_patients),
                'avg_sequence_length': cluster_patients.groupby('mi_person_key').size().mean(),
                'most_common_drugs': cluster_patients['drug_name'].value_counts().head(5).to_dict(),
                'therapeutic_class_distribution': cluster_patients['therapeutic_class_1'].value_counts().head(5).to_dict(),
                'avg_days_to_event': cluster_patients['days_to_ade'].mean() if 'days_to_ade' in cluster_patients.columns else None,
                'avg_days_to_opioid_ed': cluster_patients['days_to_opioid_ed'].mean() if 'days_to_opioid_ed' in cluster_patients.columns else None
            }
            
            cluster_characteristics[cluster_id] = characteristics
        
        self.logger.info("Cluster characteristics analysis complete")
        return cluster_characteristics
    
    def create_visualizations(self, cluster_results: Dict, cluster_characteristics: Dict):
        """
        Create visualizations for DTW analysis results.
        
        Args:
            cluster_results: Results from clustering
            cluster_characteristics: Cluster characteristics
        """
        self.logger.info("Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'DTW Analysis Results - {self.cohort_name} Cohort ({self.age_band}, {self.event_year})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Similarity matrix heatmap
        ax1 = axes[0, 0]
        similarity_matrix = cluster_results['similarity_matrix']
        finite_matrix = np.where(np.isinf(similarity_matrix), 
                                np.max(similarity_matrix[~np.isinf(similarity_matrix)]) * 2, 
                                similarity_matrix)
        
        im1 = ax1.imshow(finite_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('DTW Similarity Matrix')
        ax1.set_xlabel('Patient Index')
        ax1.set_ylabel('Patient Index')
        plt.colorbar(im1, ax=ax1, label='DTW Distance')
        
        # 2. Cluster size distribution
        ax2 = axes[0, 1]
        cluster_sizes = cluster_results['cluster_sizes']
        ax2.bar(range(len(cluster_sizes)), cluster_sizes, color='skyblue', alpha=0.7)
        ax2.set_title('Cluster Size Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Patients')
        ax2.set_xticks(range(len(cluster_sizes)))
        
        # 3. Average sequence length by cluster
        ax3 = axes[1, 0]
        cluster_ids = list(cluster_characteristics.keys())
        avg_lengths = [cluster_characteristics[cid]['avg_sequence_length'] for cid in cluster_ids]
        ax3.bar(cluster_ids, avg_lengths, color='lightcoral', alpha=0.7)
        ax3.set_title('Average Drug Sequence Length by Cluster')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Average Sequence Length')
        
        # 4. Most common drugs across clusters
        ax4 = axes[1, 1]
        all_drugs = set()
        for char in cluster_characteristics.values():
            all_drugs.update(char['most_common_drugs'].keys())
        
        # Select top 10 most common drugs overall
        drug_counts = {}
        for char in cluster_characteristics.values():
            for drug, count in char['most_common_drugs'].items():
                drug_counts[drug] = drug_counts.get(drug, 0) + count
        
        top_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        drugs, counts = zip(*top_drugs)
        
        ax4.barh(range(len(drugs)), counts, color='lightgreen', alpha=0.7)
        ax4.set_title('Top 10 Most Common Drugs Across All Clusters')
        ax4.set_xlabel('Total Count')
        ax4.set_yticks(range(len(drugs)))
        ax4.set_yticklabels(drugs)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = f"dtw_analysis_results_{self.cohort_name}_{self.age_band}_{self.event_year}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to {output_path}")
        
        plt.show()
    
    def save_results(self, cluster_results: Dict, cluster_characteristics: Dict):
        """
        Save DTW analysis results to S3.
        
        Args:
            cluster_results: Results from clustering
            cluster_characteristics: Cluster characteristics
        """
        self.logger.info("Saving DTW analysis results...")
        
        # Prepare results for saving
        results = {
            'metadata': {
                'cohort_name': self.cohort_name,
                'age_band': self.age_band,
                'event_year': self.event_year,
                'analysis_timestamp': datetime.now().isoformat(),
                'n_patients': len(self.patient_sequences),
                'n_unique_drugs': len(self.drug_encoding_map) if hasattr(self, 'drug_encoding_map') else 0
            },
            'clustering_results': {
                'n_clusters': cluster_results['n_clusters'],
                'silhouette_score': cluster_results['silhouette_score'],
                'cluster_sizes': cluster_results['cluster_sizes'].tolist(),
                'patient_cluster_mapping': dict(zip(cluster_results['patient_ids'], 
                                                   cluster_results['cluster_labels']))
            },
            'cluster_characteristics': cluster_characteristics,
            'drug_encoding_map': self.drug_encoding_map if hasattr(self, 'drug_encoding_map') else {}
        }
        
        # Save to S3
        output_paths = get_output_paths(self.cohort_name, self.age_band, self.event_year)
        
        # Save JSON results
        json_path = f"s3://{S3_BUCKET}/dtw_analysis/{self.cohort_name}/{self.age_band}/{self.event_year}/dtw_results.json"
        save_to_s3_json(results, json_path)
        
        # Save patient sequences as Parquet
        sequences_df = pd.DataFrame([
            {'mi_person_key': pid, 'drug_sequence': seq, 'sequence_length': len(seq)}
            for pid, seq in self.patient_sequences.items()
        ])
        
        parquet_path = f"s3://{S3_BUCKET}/dtw_analysis/{self.cohort_name}/{self.age_band}/{self.event_year}/patient_sequences.parquet"
        save_to_s3_parquet(sequences_df, parquet_path)
        
        self.logger.info(f"Results saved to S3:")
        self.logger.info(f"  - JSON: {json_path}")
        self.logger.info(f"  - Parquet: {parquet_path}")
    
    def run_analysis(self, n_clusters: int = 5, create_plots: bool = True):
        """
        Run complete DTW analysis pipeline.
        
        Args:
            n_clusters: Number of clusters to create
            create_plots: Whether to create and save visualizations
        """
        self.logger.info(f"Starting DTW analysis for {self.cohort_name} cohort...")
        
        try:
            # Step 1: Load data
            df = self.load_cohort_data()
            
            # Step 2: Create drug sequences
            sequences = self.create_drug_sequences(df)
            
            if not sequences:
                self.logger.warning("No drug sequences found. Analysis cannot proceed.")
                return
            
            # Step 3: Calculate DTW similarity matrix
            similarity_matrix = self.calculate_dtw_similarity_matrix(sequences)
            
            # Step 4: Cluster patients
            cluster_results = self.cluster_patients(similarity_matrix, n_clusters)
            
            # Step 5: Analyze cluster characteristics
            cluster_characteristics = self.analyze_cluster_characteristics(df, cluster_results)
            
            # Step 6: Create visualizations
            if create_plots:
                self.create_visualizations(cluster_results, cluster_characteristics)
            
            # Step 7: Save results
            self.save_results(cluster_results, cluster_characteristics)
            
            self.logger.info("DTW analysis completed successfully!")
            
            return {
                'cluster_results': cluster_results,
                'cluster_characteristics': cluster_characteristics,
                'similarity_matrix': similarity_matrix,
                'patient_sequences': sequences
            }
            
        except Exception as e:
            self.logger.error(f"Error in DTW analysis: {str(e)}")
            raise


def main():
    """Main function to run DTW analysis."""
    parser = argparse.ArgumentParser(description='DTW Analysis for Drug Sequence Similarity')
    parser.add_argument('--cohort', required=True, choices=['opioid_ed', 'ed_non_opioid'],
                       help='Cohort to analyze')
    parser.add_argument('--age-band', required=True, 
                       help='Age band (e.g., "0-12", "65-74")')
    parser.add_argument('--event-year', type=int, required=True,
                       help='Event year (e.g., 2020)')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters to create (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating visualizations')
    
    args = parser.parse_args()
    
    # Check if DTW is available
    if not DTW_AVAILABLE:
        print("Error: dtaidistance package not available.")
        print("Install with: pip install dtaidistance")
        sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = DTWCohortAnalyzer(args.age_band, args.event_year, args.cohort)
    
    try:
        results = analyzer.run_analysis(
            n_clusters=args.n_clusters,
            create_plots=not args.no_plots
        )
        
        print(f"\nDTW Analysis completed successfully!")
        print(f"Cohort: {args.cohort}")
        print(f"Age Band: {args.age_band}")
        print(f"Event Year: {args.event_year}")
        print(f"Number of Patients: {len(results['patient_sequences'])}")
        print(f"Number of Clusters: {args.n_clusters}")
        print(f"Silhouette Score: {results['cluster_results']['silhouette_score']:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 