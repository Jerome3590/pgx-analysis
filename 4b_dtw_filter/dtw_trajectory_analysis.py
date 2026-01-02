"""
Enhanced DTW Analysis for Patient Trajectory Development

This module extends DTW analysis to:
1. Develop patient trajectories using temporal sequences (drugs, ICD codes, CPT codes)
2. Integrate with cohort temporal fields (days_to_target_event)
3. Create trajectory archetypes and patterns
4. Enable trajectory-based risk prediction
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

from py_helpers.common_imports import (  # type: ignore[attr-defined]
    s3_client,
    S3_BUCKET,
    get_logger,
)
from py_helpers.duckdb_utils import get_duckdb_connection  # type: ignore[attr-defined]
from py_helpers.s3_utils import (  # type: ignore[attr-defined]
    get_output_paths,
    get_cohort_parquet_path,
    save_to_s3_parquet,
    save_to_s3_json,
    s3_exists,
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

    def __init__(
        self,
        cohort_name: str,
        age_band: str,
        event_year: str,
        item_type: str = "drug",
    ):
        """
        Initialize trajectory analyzer.
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
        self.patient_trajectories: Dict[str, List[Dict[str, object]]] = {}
        self.trajectory_sequences: Dict[str, object] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.trajectory_clusters: Dict[str, object] = {}
        self.archetype_trajectories: Dict[int, List[Dict[str, object]]] = {}

    def load_cohort_data(self) -> pd.DataFrame:
        """Load cohort data with temporal fields."""
        self.logger.info("Loading {0} cohort data...".format(self.cohort_name))

        cohort_path = get_cohort_parquet_path(
            self.cohort_name, self.age_band, self.event_year
        )

        if self.item_type == "drug":
            if self.cohort_name == "ed_non_opioid":
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

        elif self.item_type == "icd":
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

        elif self.item_type == "cpt":
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
            raise ValueError("Unsupported item_type: {0}".format(self.item_type))

        df = self.conn.sql(query).df()
        self.logger.info(
            "Loaded {0:,} {1} records for {2:,} patients".format(
                len(df), self.item_type, df["mi_person_key"].nunique()
            )
        )

        return df

    def create_temporal_trajectories(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Create patient trajectories with temporal positioning.
        """
        self.logger.info("Creating temporal trajectories...")

        trajectories: Dict[str, List[Dict[str, object]]] = {}

        for patient_id in df["mi_person_key"].unique():
            patient_data = df[df["mi_person_key"] == patient_id].copy()

            if (
                self.cohort_name == "ed_non_opioid"
                and "days_to_target_event" in patient_data.columns
            ):
                patient_data = patient_data.sort_values(
                    "days_to_target_event",
                    ascending=False,
                    na_position="last",
                )
                temporal_key = "days_to_target_event"
            else:
                patient_data = patient_data.sort_values("event_date")
                temporal_key = "event_date"

            trajectory: List[Dict[str, object]] = []
            for _, row in patient_data.iterrows():
                event = {
                    "item": str(row["item_name"]),
                    "temporal_position": (
                        row[temporal_key] if pd.notnull(row[temporal_key]) else None
                    ),
                    "event_date": str(row["event_date"]),
                    "is_target": bool(row.get("is_target_case", False)),
                }
                trajectory.append(event)

            if trajectory:
                trajectories[patient_id] = trajectory

        self.logger.info("Created {0} patient trajectories".format(len(trajectories)))
        self.patient_trajectories = trajectories

        return trajectories

    def encode_trajectory_sequence(
        self, trajectory: List[Dict[str, object]]
    ) -> Tuple[List[int], List[float]]:
        """
        Encode trajectory to numerical sequence for DTW.
        """
        if not hasattr(self, "item_encoding_map"):
            all_items = set()
            for traj in self.patient_trajectories.values():
                for event in traj:
                    all_items.add(event["item"])  # type: ignore[index]

            self.item_encoding_map = {
                item: idx for idx, item in enumerate(sorted(all_items))
            }
            self.logger.info(
                "Created encoding map for {0} unique items".format(
                    len(self.item_encoding_map)
                )
            )

        encoded_items: List[int] = []
        temporal_positions: List[float] = []

        for event in trajectory:
            item = event["item"]  # type: ignore[index]
            encoded_items.append(self.item_encoding_map.get(item, -1))

            temp_pos = event.get("temporal_position")
            if temp_pos is not None:
                if isinstance(temp_pos, (int, float)):
                    temporal_positions.append(float(temp_pos))
                else:
                    temporal_positions.append(0.0)
            else:
                temporal_positions.append(0.0)

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

        encoded_sequences: Dict[str, Tuple[List[int], List[float]]] = {}
        for pid in patient_ids:
            items, temps = self.encode_trajectory_sequence(self.patient_trajectories[pid])
            encoded_sequences[pid] = (items, temps)

        similarity_matrix = np.zeros((n_patients, n_patients))

        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                seq1_items, _ = encoded_sequences[patient_ids[i]]
                seq2_items, _ = encoded_sequences[patient_ids[j]]

                if seq1_items and seq2_items:
                    distance = dtw.distance(seq1_items, seq2_items)
                    similarity_matrix[i][j] = distance
                    similarity_matrix[j][i] = distance
                else:
                    similarity_matrix[i][j] = np.inf
                    similarity_matrix[j][i] = np.inf

            if (i + 1) % 50 == 0:
                self.logger.info(
                    "Processed {0}/{1} patients".format(i + 1, n_patients)
                )

        self.similarity_matrix = similarity_matrix
        self.patient_ids = patient_ids

        self.logger.info("DTW similarity matrix calculation complete")
        return similarity_matrix

    def cluster_trajectories(self, n_clusters: int = 5) -> Dict[str, object]:
        """Cluster patients based on trajectory similarity."""
        self.logger.info(
            "Clustering trajectories into {0} groups...".format(n_clusters)
        )

        if self.similarity_matrix is None:
            self.calculate_trajectory_similarity_matrix()

        assert self.similarity_matrix is not None

        finite_matrix = np.where(
            np.isinf(self.similarity_matrix),
            np.max(self.similarity_matrix[~np.isinf(self.similarity_matrix)]) * 2,
            self.similarity_matrix,
        )

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="ward",
        )

        cluster_labels = clustering.fit_predict(finite_matrix)

        silhouette_avg = silhouette_score(
            finite_matrix,
            cluster_labels,
            metric="precomputed",
        )

        patient_cluster_map = dict(zip(self.patient_ids, cluster_labels))

        archetypes = self.extract_archetype_trajectories(patient_cluster_map, n_clusters)

        cluster_results: Dict[str, object] = {
            "cluster_labels": cluster_labels.tolist(),
            "patient_ids": self.patient_ids,
            "patient_cluster_map": patient_cluster_map,
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette_avg),
            "cluster_sizes": np.bincount(cluster_labels).tolist(),
            "archetype_trajectories": archetypes,
        }

        self.trajectory_clusters = cluster_results
        self.archetype_trajectories = archetypes

        self.logger.info(
            "Clustering complete. Silhouette score: {0:.3f}".format(silhouette_avg)
        )
        self.logger.info(
            "Cluster sizes: {0}".format(
                dict(enumerate(cluster_results["cluster_sizes"]))  # type: ignore[index]
            )
        )

        return cluster_results

    def extract_archetype_trajectories(
        self,
        patient_cluster_map: Dict[str, int],
        n_clusters: int,
    ) -> Dict[int, List[Dict[str, object]]]:
        """
        Extract archetype (representative) trajectory for each cluster.
        """
        archetypes: Dict[int, List[Dict[str, object]]] = {}

        for cluster_id in range(n_clusters):
            cluster_patients = [
                pid
                for pid, cid in patient_cluster_map.items()
                if cid == cluster_id
            ]

            if not cluster_patients:
                continue

            cluster_trajectories = [
                self.patient_trajectories[pid] for pid in cluster_patients
            ]

            lengths = [len(t) for t in cluster_trajectories]
            median_length = int(np.median(lengths))

            closest_traj = min(
                cluster_trajectories,
                key=lambda t: abs(len(t) - median_length),
            )

            archetypes[cluster_id] = closest_traj

        return archetypes

    def analyze_trajectory_patterns(self) -> Dict[int, object]:
        """Analyze patterns within trajectory clusters."""
        self.logger.info("Analyzing trajectory patterns...")

        if not self.trajectory_clusters:
            raise ValueError("Must run clustering first")

        patterns: Dict[int, object] = {}
        patient_cluster_map = self.trajectory_clusters["patient_cluster_map"]  # type: ignore[index]

        for cluster_id in range(self.trajectory_clusters["n_clusters"]):  # type: ignore[index]
            cluster_patients = [
                pid
                for pid, cid in patient_cluster_map.items()
                if cid == cluster_id
            ]

            cluster_trajectories = [
                self.patient_trajectories[pid] for pid in cluster_patients
            ]

            all_items: List[str] = []
            for traj in cluster_trajectories:
                all_items.extend([str(e["item"]) for e in traj])

            from collections import Counter

            item_counts = Counter(all_items)

            avg_length = np.mean([len(t) for t in cluster_trajectories])

            temporal_info: List[float] = []
            for traj in cluster_trajectories:
                temps = [
                    e.get("temporal_position")  # type: ignore[union-attr]
                    for e in traj
                    if e.get("temporal_position") is not None
                ]
                if temps:
                    temporal_info.extend([float(x) for x in temps])

            patterns[cluster_id] = {
                "n_patients": len(cluster_patients),
                "avg_trajectory_length": float(avg_length),
                "most_common_items": dict(item_counts.most_common(10)),
                "avg_temporal_position": (
                    float(np.mean(temporal_info)) if temporal_info else None
                ),
                "archetype_trajectory": self.archetype_trajectories.get(cluster_id, []),
            }

        return patterns

    def save_trajectory_results(self) -> None:
        """Save trajectory analysis results to S3."""
        self.logger.info("Saving trajectory results...")

        results = {
            "metadata": {
                "cohort_name": self.cohort_name,
                "age_band": self.age_band,
                "event_year": self.event_year,
                "item_type": self.item_type,
                "analysis_timestamp": datetime.now().isoformat(),
                "n_patients": len(self.patient_trajectories),
                "n_unique_items": len(self.item_encoding_map)
                if hasattr(self, "item_encoding_map")
                else 0,
            },
            "trajectory_clusters": self.trajectory_clusters,
            "archetype_trajectories": self.archetype_trajectories,
            "trajectory_patterns": self.analyze_trajectory_patterns(),
            "item_encoding_map": (
                self.item_encoding_map if hasattr(self, "item_encoding_map") else {}
            ),
        }

        base_path = (
            f"s3://{S3_BUCKET}/dtw_trajectories/"
            f"{self.cohort_name}/{self.age_band}/{self.event_year}"
        )

        json_path = f"{base_path}/trajectory_results_{self.item_type}.json"
        save_to_s3_json(results, json_path, self.logger)

        trajectory_records: List[Dict[str, object]] = []
        patient_cluster_map = self.trajectory_clusters.get(
            "patient_cluster_map", {}
        )

        for pid, traj in self.patient_trajectories.items():
            cluster_id = patient_cluster_map.get(pid, -1)

            trajectory_records.append(
                {
                    "mi_person_key": pid,
                    "cluster_id": cluster_id,
                    "trajectory_length": len(traj),
                    "trajectory_items": [e["item"] for e in traj],  # type: ignore[index]
                    "temporal_positions": [
                        e.get("temporal_position") for e in traj
                    ],
                }
            )

        traj_df = pd.DataFrame(trajectory_records)
        parquet_path = f"{base_path}/patient_trajectories_{self.item_type}.parquet"
        save_to_s3_parquet(traj_df, parquet_path, self.logger)

        self.logger.info("Results saved:")
        self.logger.info(f"  - JSON: {json_path}")
        self.logger.info(f"  - Parquet: {parquet_path}")

    def run_analysis(self, n_clusters: int = 5) -> Optional[Dict[str, object]]:
        """Run complete trajectory analysis pipeline."""
        self.logger.info(
            "Starting trajectory analysis for {0} ({1})...".format(
                self.cohort_name, self.item_type
            )
        )

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
            "cluster_results": cluster_results,
            "patterns": patterns,
            "archetype_trajectories": self.archetype_trajectories,
            "n_patients": len(trajectories),
        }


def main() -> None:
    """Main function to run trajectory analysis."""
    parser = argparse.ArgumentParser(
        description="DTW Trajectory Analysis for Patient Sequences"
    )
    parser.add_argument(
        "--cohort",
        required=True,
        choices=["opioid_ed", "ed_non_opioid"],
        help="Cohort to analyze",
    )
    parser.add_argument(
        "--age-band",
        required=True,
        help='Age band (e.g., "65-74")',
    )
    parser.add_argument(
        "--event-year",
        type=str,
        required=True,
        help='Event year (e.g., "2020")',
    )
    parser.add_argument(
        "--item-type",
        choices=["drug", "icd", "cpt"],
        default="drug",
        help="Type of items for trajectory (default: drug)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters to create (default: 5)",
    )

    args = parser.parse_args()

    if not DTW_AVAILABLE:
        print("Error: dtaidistance package not available.")
        print("Install with: pip install dtaidistance")
        sys.exit(1)

    analyzer = PatientTrajectoryAnalyzer(
        args.cohort,
        args.age_band,
        args.event_year,
        args.item_type,
    )

    try:
        results = analyzer.run_analysis(n_clusters=args.n_clusters)

        if results is None:
            print("No trajectories were analyzed.")
            sys.exit(1)

        print("\nTrajectory Analysis completed successfully!")
        print(f"Cohort: {args.cohort}")
        print(f"Age Band: {args.age_band}")
        print(f"Event Year: {args.event_year}")
        print(f"Item Type: {args.item_type}")
        print(f"Number of Patients: {results['n_patients']}")
        print(f"Number of Clusters: {args.n_clusters}")
        print(
            "Silhouette Score: "
            f"{results['cluster_results']['silhouette_score']:.3f}"  # type: ignore[index]
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

