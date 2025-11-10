# catboost_exp_explainer.py

import json
from itertools import count
from pysat.examples.hitman import Hitman
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from catboost import CatBoostClassifier
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class PathConfig:
    def __init__(self, 
                 model_path: str,
                 data_dir: str,
                 output_dir: str,
                 tree_rules_path: str = None,
                 age_band: str = None):
        """
        Initialize path configuration for S3 paths.
        
        Args:
            model_path: Path to the CatBoost model JSON file in S3
            data_dir: Base directory containing datasets in S3
            output_dir: Directory for saving outputs in S3
            tree_rules_path: Path to the tree rules JSON file
            age_band: Age band for the cohort (e.g., "0-12", "13-24", etc.)
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tree_rules_path = tree_rules_path
        self.age_band = age_band
        
    @property
    def train_data_path(self) -> str:
        return os.path.join(self.data_dir, f'train_data_{self.age_band}.csv')
        
    @property
    def test_data_path(self) -> str:
        return os.path.join(self.data_dir, f'test_data_{self.age_band}.csv')
        
    @property
    def axp_output_dir(self) -> str:
        return os.path.join(self.output_dir, 'axp')
    
    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read parquet file from S3."""
        return pd.read_parquet(path)
    
    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write parquet file to S3."""
        df.to_parquet(path)
    
    def read_json(self, path: str) -> Dict:
        """Read JSON file from S3."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def write_json(self, data: Dict, path: str) -> None:
        """Write JSON file to S3."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_plot(self, fig: plt.Figure, path: str) -> None:
        """Save plot to S3."""
        fig.savefig(path, bbox_inches='tight', dpi=300)
    
    def ensure_dir_exists(self, path: str) -> None:
        """Ensure S3 directory exists."""
        os.makedirs(path, exist_ok=True)


class AnalysisConfig:
    """Configuration class for analysis parameters."""
    def __init__(self,
                 top_k: int = 10,
                 min_coverage: float = 0.8,
                 significance_threshold: float = 0.05,
                 n_permutations: int = 1000,
                 visualization_params: Dict[str, Any] = None):
        self.top_k = top_k
        self.min_coverage = min_coverage
        self.significance_threshold = significance_threshold
        self.n_permutations = n_permutations
        self.visualization_params = visualization_params or {
            'figsize': (12, 8),
            'dpi': 300,
            'fontsize': 10
        }

class CatBoostAXPExplainer:
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config
        self.model = None
        self.feature_names = {}
        self.condition_id_map = {}
        self.id_condition_map = {}
        self.rule_clauses = []
        self.rule_predictions = []
        self._id_gen = count(1)
        self.setup_logging()
    
    def setup_logging(self, log_file: str = None, level: int = logging.INFO) -> None:
        """Setup logging configuration"""
        if log_file is None:
            log_file = os.path.join(self.path_config.output_dir, 'axp_analysis.log')
        
        logging.basicConfig(
            filename=log_file,
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> CatBoostClassifier:
        """Load the CatBoost model"""
        try:
            model = CatBoostClassifier()
            model.load_model(self.path_config.model_path)
            self.model = model
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def compute_feature_attribution(self, 
                                  X: Union[np.ndarray, pd.DataFrame],
                                  predictions: np.ndarray,
                                  class_labels: Optional[List[int]] = None) -> Dict[int, pd.DataFrame]:
        """Compute feature attribution for each class"""
        if class_labels is None:
            class_labels = [0, 1]
        
        results = {}
        for class_label in class_labels:
            # Get instances of this class
            class_mask = predictions == class_label
            X_class = X[class_mask]
            
            # Compute AXPs for each instance
            axps = self._batch_compute_axps(X_class, class_label)
            
            # Compute feature metrics
            metrics = self._compute_feature_metrics(axps)
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics)
            results[class_label] = metrics_df
        
        return results
    
    def validate_explanations(self, X: Union[np.ndarray, pd.DataFrame], 
                             predictions: np.ndarray,
                             threshold: float = 0.8) -> Dict[str, Any]:
        """Validate the generated explanations"""
        validation_results = {
            'coverage': {},
            'reliability': {},
            'stability': {}
        }
        
        for class_label in [0, 1]:
            class_mask = predictions == class_label
            X_class = X[class_mask]
            
            # Compute coverage
            axps = self._batch_compute_axps(X_class, class_label)
            coverage = len(axps) / len(X_class)
            
            # Compute reliability
            reliable_count = sum(1 for axp in axps if len(axp['conditions']) > 0)
            reliability = reliable_count / len(axps) if axps else 0
            
            # Check stability
            unstable_features = self._check_stability(X_class, class_label)
            
            validation_results['coverage'][f'class_{class_label}'] = coverage
            validation_results['reliability'][f'class_{class_label}'] = reliability
            validation_results['stability'][f'class_{class_label}'] = unstable_features
        
        return validation_results
    
    def test_feature_significance(self,
                                X: pd.DataFrame,
                                predictions: np.ndarray,
                                n_permutations: int = 1000,
                                alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
        """Test significance of features using permutation tests"""
        significance_results = {0: {}, 1: {}}
        
        for class_label in [0, 1]:
            class_mask = predictions == class_label
            X_class = X[class_mask]
            
            # Get original feature importance
            original_importance = self._compute_feature_importance(X_class, class_label)
            
            # Permutation test
            for feature in X_class.columns:
                p_value = self._permutation_test(
                    X_class, feature, class_label, 
                    original_importance[feature],
                    n_permutations
                )
                significance_results[class_label][feature] = p_value
        
        return significance_results
    
    def compare_with_native_importance(self, 
                                     catboost_model: CatBoostClassifier,
                                     X: pd.DataFrame) -> pd.DataFrame:
        """Compare AXP-based importance with CatBoost's native importance"""
        native_importance = pd.DataFrame({
            'feature': catboost_model.feature_names_,
            'native_importance': catboost_model.get_feature_importance()
        })
        
        # Get AXP-based importance
        axp_importance = self._compute_feature_importance(X, 1)  # Use class 1 for comparison
        
        # Combine results
        comparison = pd.DataFrame({
            'feature': list(axp_importance.keys()),
            'axp_importance': list(axp_importance.values())
        })
        
        comparison = comparison.merge(native_importance, on='feature', how='outer')
        comparison = comparison.fillna(0)
        
        return comparison
    
    def export_results(self, results: Dict[str, Any], output_dir: str, formats: List[str] = ['json', 'parquet']) -> None:
        """Export analysis results in specified formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        for format_type in formats:
            if format_type == 'json':
                with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
            elif format_type == 'parquet':
                for class_label, metrics_df in results['feature_attribution'].items():
                    metrics_df.to_parquet(
                        os.path.join(output_dir, f'feature_metrics_class_{class_label}.parquet')
                    )
    
    def _batch_compute_axps(self, X: np.ndarray, target_class: int) -> List[Dict]:
        """Compute AXPs for a batch of instances"""
        axps = []
        for i, x in enumerate(X):
            try:
                literals = self.explain_literals(x, target_class)
                conditions = [self._literal_to_text(lit) for lit in literals]
                axps.append({
                    'instance_idx': i,
                    'conditions': conditions,
                    'literals': literals
                })
            except Exception as e:
                self.logger.warning(f"Error computing AXP for instance {i}: {str(e)}")
                continue
        return axps
    
    def _compute_feature_metrics(self, axps: List[Dict]) -> List[Dict]:
        """Compute metrics for features based on AXPs"""
        feature_stats = {}
        
        for axp in axps:
            for condition in axp['conditions']:
                feature = condition.split()[0]
                if feature not in feature_stats:
                    feature_stats[feature] = {
                        'count': 0,
                        'instances': set(),
                        'conditions': []
                    }
                
                feature_stats[feature]['count'] += 1
                feature_stats[feature]['instances'].add(axp['instance_idx'])
                feature_stats[feature]['conditions'].append(condition)
        
        # Convert to list of metrics
        metrics = []
        for feature, stats in feature_stats.items():
            metrics.append({
                'feature': feature,
                'support': stats['count'],
                'coverage': len(stats['instances']) / len(axps),
                'conditions': stats['conditions']
            })
        
        return metrics
    
    def _check_stability(self, X: np.ndarray, class_label: int) -> List[str]:
        """Check stability of features across instances"""
        axps = self._batch_compute_axps(X, class_label)
        feature_counts = {}
        
        for axp in axps:
            for condition in axp['conditions']:
                feature = condition.split()[0]
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Identify unstable features (present in less than 10% of instances)
        threshold = len(X) * 0.1
        unstable = [f for f, count in feature_counts.items() if count < threshold]
        
        return unstable
    
    def _permutation_test(self, X: pd.DataFrame, feature: str, class_label: int,
                         original_importance: float, n_permutations: int) -> float:
        """Perform permutation test for feature significance"""
        permuted_importances = []
        
        for _ in range(n_permutations):
            # Permute feature values
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            
            # Compute importance with permuted values
            permuted_importance = self._compute_feature_importance(X_permuted, class_label)[feature]
            permuted_importances.append(permuted_importance)
        
        # Compute p-value
        p_value = sum(1 for imp in permuted_importances if imp >= original_importance) / n_permutations
        
        return p_value
    
    def _compute_feature_importance(self, X: pd.DataFrame, class_label: int) -> Dict[str, float]:
        """Compute feature importance based on AXPs"""
        axps = self._batch_compute_axps(X, class_label)
        metrics = self._compute_feature_metrics(axps)
        
        # Convert to dictionary of feature importance
        importance = {m['feature']: m['support'] for m in metrics}
        
        return importance

class CatBoostSymbolicExplainer:
  
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config
        self.condition_id_map = {}
        self.id_condition_map = {}
        self.rule_clauses = []
        self.rule_predictions = []
        self.feature_names = {}
        self.model_json = None
        self.tree_rules = None
        self._id_gen = count(1)  # SAT literals start at 1
        self.setup_logging()

   
    def _get_condition_literal(self, feat_idx, threshold, direction):
        key = (feat_idx, threshold, direction)
        if key not in self.condition_id_map:
            lit = next(self._id_gen)
            self.condition_id_map[key] = lit
            self.id_condition_map[lit] = key
        return self.condition_id_map[key]

    
    def fit_from_model_json(self, model_json):
        """Parse non-oblivious CatBoost model and build symbolic CNF clauses for FFA."""
        self.model_json = model_json
        self.rule_clauses.clear()
        self.rule_predictions.clear()
        self.condition_id_map.clear()
        self.id_condition_map.clear()
    
        # Validate feature map
        self.feature_names = {
            f["flat_feature_index"]: f["feature_id"]
            for f in model_json["features_info"]["float_features"]
        }
    
        # === Safety check: Reject oblivious trees ===
        if "oblivious_trees" in model_json:
            raise ValueError("❌ Model was trained using oblivious trees. "
                             "Please retrain with grow_policy='Lossguide' and boosting_type='Plain'.")
    
        # === Require non-oblivious trees ===
        if "non_oblivious_trees" not in model_json:
            raise ValueError("❌ Model JSON missing 'non_oblivious_trees'. "
                             "Ensure the model was trained with grow_policy='Lossguide'.")
    
        # === Process each non-oblivious tree ===
        for tree in model_json["non_oblivious_trees"]:
            self._traverse_non_oblivious_tree(tree["nodes"], path=[], conditions=[])

    def _traverse_non_oblivious_tree(self, node, path, conditions):
        """Recursively parse a non-oblivious tree and extract CNF rules."""
        if "leaf_value" in node:
            # Base case: reached a leaf
            pred = 1 if node["leaf_value"] > 0 else 0
            clause = [self._get_condition_literal(f, t, d) for (f, t, d) in conditions]
            self.rule_clauses.append(clause)
            self.rule_predictions.append(pred)
            return
    
        # Recursive case: internal node with split
        split = node["split_condition"]
        feature_index = split["float_feature_index"]
        threshold = split["border"]
    
        # Left child: condition is "feature <= threshold" => direction=0
        self._traverse_non_oblivious_tree(
            node["left_child"],
            path + [0],
            conditions + [(feature_index, threshold, 0)]
        )
    
        # Right child: condition is "feature > threshold" => direction=1
        self._traverse_non_oblivious_tree(
            node["right_child"],
            path + [1],
            conditions + [(feature_index, threshold, 1)]
        )
    

    def _satisfied_rules(self, instance, target_class):
        """Return rule indexes satisfied by instance and match class."""
        matched = []
        for idx, clause in enumerate(self.rule_clauses):
            if self.rule_predictions[idx] != target_class:
                continue
            if all(
                (instance[feat] <= thresh if dir == 0 else instance[feat] > thresh)
                for (feat, thresh, dir) in (self.id_condition_map[lit] for lit in clause)
            ):
                matched.append(idx)
        return matched

    
    def _compute_axp(self, rule_ids):
        """Compute minimal hitting set (AXP) over matching rule IDs."""
        h = Hitman(solver="m22")
        for ridx in rule_ids:
            h.hit(self.rule_clauses[ridx])
        return h.get()

    
    def explain_literals(self, instance, predicted_class):
        """Return minimal AXP literals for instance."""
        matched = self._satisfied_rules(instance, predicted_class)
        if not matched:
            return []
        return self._compute_axp(matched)

    
    def explain_instance(self, instance, predicted_class=None):
        """Return readable explanation (AXP) for instance."""
        if predicted_class is None:
            raise ValueError("You must provide the predicted class.")

        literals = self.explain_literals(instance, predicted_class)
        return [self._literal_to_text(lit) for lit in literals]

    
    def _literal_to_text(self, lit):
        feat_idx, thresh, direction = self.id_condition_map[lit]
        feat = self.feature_names[feat_idx]
        op = "<=" if direction == 0 else ">"
        return f"{feat} {op} {thresh}"
      

    def explain_dataset(self, X, predictions=None, return_df=True, show_progress=True):
        """
        Generate AXP explanations for a dataset.
        
        Parameters:
        - X: numpy array or DataFrame
        - predictions: list of predicted classes (optional)
        - return_df: whether to return a pandas DataFrame (default: True)
        - show_progress: whether to show progress bar (default: True)
        
        Returns:
        - List[Dict] or DataFrame: One explanation per row
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if predictions is None:
            raise ValueError("Please provide the predicted class labels for each instance.")

        results = []
        iterator = enumerate(zip(X, predictions))
        if show_progress:
            iterator = tqdm(iterator, total=len(X), desc="Generating explanations")

        for i, (x, yhat) in iterator:
            axp = self.explain_instance(x, predicted_class=yhat)
            results.append({
                "index": i,
                "predicted_class": yhat,
                "axp": axp
            })

        return pd.DataFrame(results) if return_df else results
      
      
    def debug_rule_match(instance, target_class, rule_clauses, rule_preds, id_condition_map):
        for idx, clause in enumerate(rule_clauses):
            if rule_preds[idx] != target_class:
                continue
    
            print(f"\nRule {idx}:")
            all_matched = True
            for lit in clause:
                feat_idx, thresh, direction = id_condition_map[lit]
                val = instance[feat_idx]
                cond = (val <= thresh) if direction == 0 else (val > thresh)
                print(f" - Feature {feat_idx} ({'≤' if direction == 0 else '>'} {thresh}): {val} → {cond}")
                if not cond:
                    all_matched = False
            print(" → MATCH" if all_matched else " → NO MATCH")

    def _batch_compute_axps(self, X: np.ndarray, target_class: int) -> List[Dict]:
        """
        Compute AXPs for multiple instances in batch.
        
        Args:
            X: Feature matrix
            target_class: Target class to explain
            
        Returns:
            List of dictionaries containing instance index and AXPs
        """
        results = []
        
        for i, x in enumerate(X):
            clauses = self._satisfied_clauses_for_instance(x, target_class)
            if not clauses:
                continue
            
            axp_literal_sets = self._enumerate_axps(clauses)
            seen = set()  # track unique AXPs for this instance
            
            for axp_literals in axp_literal_sets:
                axp_readable = [self._literal_to_text(lit) for lit in axp_literals]
                
                # Normalize and de-duplicate
                axp_str = str(sorted(axp_readable))
                if axp_str in seen:
                    continue
                
                seen.add(axp_str)
                results.append({
                    "instance": i,
                    "axp": axp_readable
                })
        
        return results

    def _satisfied_clauses_for_instance(self, x: np.ndarray, target_class: int) -> List[List[int]]:
        """
        Find all clauses satisfied by an instance for a target class.
        
        Args:
            x: Feature vector
            target_class: Target class
            
        Returns:
            List of satisfied clauses (lists of literals)
        """
        matched_clauses = []
        for i, clause in enumerate(self.rule_clauses):
            if self.rule_predictions[i] != target_class:
                continue
            if all(self._literal_condition_holds(x, lit) for lit in clause):
                matched_clauses.append(clause)
        return matched_clauses

    def _literal_condition_holds(self, x: np.ndarray, lit: int) -> bool:
        """
        Check if a literal's condition holds for an instance.
        
        Args:
            x: Feature vector
            lit: Literal ID
            
        Returns:
            Boolean indicating if condition holds
        """
        feat_idx, threshold, direction = self.id_condition_map[lit]
        value = x[feat_idx]
        return value <= threshold if direction == 0 else value > threshold

    def _enumerate_axps(self, clauses: List[List[int]]) -> List[List[int]]:
        """
        Enumerate all minimal hitting sets (AXPs) for a set of clauses.
        
        Args:
            clauses: List of clauses
            
        Returns:
            List of AXPs (lists of literals)
        """
        h = Hitman(solver="m22", htype='sorted')
        for clause in clauses:
            h.hit(clause)
        return list(h.enumerate())

    def _compute_feature_metrics(self, axps: List[Dict]) -> List[Dict]:
        """
        Compute comprehensive feature attribution metrics from AXPs.
        """
        # Existing metric initialization
        instance_groups = defaultdict(list)
        for axp in axps:
            instance_groups[axp["instance"]].append(axp["axp"])
        
        instance_count = len(instance_groups)
        
        # Initialize enhanced metric trackers
        essentiality = defaultdict(int)
        contrastiveness = defaultdict(int)
        support_counter = Counter()
        coverage_map = defaultdict(set)
        specificity_map = defaultdict(list)
        position_map = defaultdict(list)
        stability_map = defaultdict(list)
        
        # Compute metrics for each instance
        for instance_id, axp_list in instance_groups.items():
            all_features = [cond.split()[0] for axp in axp_list for cond in axp]
            unique_features = set(all_features)
            feature_sets = [set(cond.split()[0] for cond in axp) for axp in axp_list]
            
            # Compute essential features
            essential_feats = set.intersection(*feature_sets) if feature_sets else set()
            
            # Update metrics
            for f in essential_feats:
                essentiality[f] += 1
            
            for f in unique_features:
                if any(f not in fs for fs in feature_sets):
                    contrastiveness[f] += 1
                
                support_counter[f] += 1
                coverage_map[f].add(instance_id)
                
                # Track positions and specificity
                axp_lengths = []
                positions = []
                for axp in axp_list:
                    if f in {c.split()[0] for c in axp}:
                        axp_lengths.append(len(axp))
                        positions.append(next(i for i, c in enumerate(axp) if c.split()[0] == f))
                
                specificity_map[f].extend(axp_lengths)
                position_map[f].extend(positions)
                stability_map[f].append(np.std(positions) if positions else 0)
        
        # Compile enhanced metrics
        metrics = []
        all_features = set(support_counter.keys())
        
        for feature in all_features:
            metrics.append({
                # Existing metrics
                "feature": feature,
                "support": support_counter[feature],
                "coverage": len(coverage_map[feature]),
                "specificity": np.mean(specificity_map[feature]) if specificity_map[feature] else 0,
                "essentiality_ratio": essentiality[feature] / instance_count if instance_count > 0 else 0,
                "contrastive_instances": contrastiveness[feature],
                # New metrics
                "stability": np.mean(stability_map[feature]) if stability_map[feature] else 0,
                "relative_importance": support_counter[feature] / sum(support_counter.values()),
                "avg_position": np.mean(position_map[feature]) if position_map[feature] else 0,
                "position_std": np.std(position_map[feature]) if position_map[feature] else 0
            })
        
        return metrics

    def _plot_bar_importance(self, feature_metrics: pd.DataFrame, 
                           class_label: int,
                           top_k: int = 10,
                           save_path: Optional[str] = None) -> None:
        """Plot bar chart of feature importance."""
        df = feature_metrics.sort_values('support', ascending=False).head(top_k)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(df)), df['support'], color='lightcoral' if class_label == 1 else 'skyblue')
        plt.xticks(range(len(df)), df['feature'], rotation=45, ha='right')
        plt.title(f'Top {top_k} Features by Support (Class {class_label})')
        plt.xlabel('Feature')
        plt.ylabel('Support Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_mirror_importance(self, feature_metrics_0: pd.DataFrame,
                              feature_metrics_1: pd.DataFrame,
                              top_k: int = 10,
                              save_path: Optional[str] = None) -> None:
        """Plot mirror bar chart comparing feature importance between classes."""
        # Combine metrics
        df_0 = feature_metrics_0.copy()
        df_1 = feature_metrics_1.copy()
        df_0['class'] = 0
        df_1['class'] = 1
        
        # Get top features by combined support
        features = pd.concat([df_0, df_1])
        top_features = features.groupby('feature')['support'].sum().nlargest(top_k).index
        
        # Filter and prepare for plotting
        plot_data = pd.concat([df_0, df_1])
        plot_data = plot_data[plot_data['feature'].isin(top_features)]
        
        # Create mirror plot
        plt.figure(figsize=(12, 8))
        
        # Plot bars
        for _, row in plot_data.iterrows():
            plt.barh(
                y=row['feature'],
                width=row['specificity'] * (-1 if row['class'] == 0 else 1),
                color='skyblue' if row['class'] == 0 else 'lightcoral',
                alpha=0.8,
                edgecolor='black'
            )
            
        plt.axvline(x=0, color='black', linewidth=1)
        plt.xlabel('Specificity (Lower is Better)')
        plt.title('Feature Importance Comparison Between Classes')
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_cattail_distribution(self, feature_metrics: pd.DataFrame,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 class_label: int,
                                 top_k: int = 10,
                                 save_path: Optional[str] = None) -> None:
        """Plot cattail distribution of feature values."""
        # Get top features
        top_features = feature_metrics.nlargest(top_k, 'support')['feature'].tolist()
        
        # Prepare data
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names.values())
        
        # Melt data for plotting
        X_melted = X[top_features].melt(var_name='Feature', value_name='Value')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Box plot
        sns.boxplot(y='Feature', x='Value', data=X_melted,
                   whis=1.5, fliersize=0, color='lightgray')
        
        # Overlay points
        sns.stripplot(y='Feature', x='Value', data=X_melted,
                     hue='Value', palette='coolwarm',
                     jitter=0.25, size=3, alpha=0.7)
        
        plt.title(f'Feature Value Distribution (Class {class_label})')
        plt.xlabel('Raw Feature Value')
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def compute_feature_attribution(self, 
                                  X: Union[np.ndarray, pd.DataFrame],
                                  predictions: np.ndarray,
                                  class_labels: Optional[List[int]] = None) -> Dict[int, pd.DataFrame]:
        """
        Compute comprehensive feature attribution metrics for specified classes.
        
        Args:
            X: Feature matrix
            predictions: Model predictions
            class_labels: List of class labels to analyze (default: [0, 1])
            
        Returns:
            Dictionary mapping class labels to feature attribution DataFrames
        """
        if class_labels is None:
            class_labels = [0, 1]
            
        results = {}
        for label in class_labels:
            # Filter data for class
            mask = (predictions == label)
            X_class = X[mask] if isinstance(X, np.ndarray) else X.loc[mask]
            
            # Compute AXPs
            axps = self._batch_compute_axps(X_class, label)
            
            # Calculate metrics
            metrics = self._compute_feature_metrics(axps)
            results[label] = pd.DataFrame(metrics)
            
        return results

    def load_model(self):
        """Load model from configured S3 path"""
        self.model_json = self.path_config.read_json(self.path_config.model_path)
        self.fit_from_model_json(self.model_json)

    def save_analysis(self, class_label: int, metrics_df: pd.DataFrame, plot_path: Optional[str] = None):
        """Save analysis results to configured S3 output directory"""
        self.path_config.ensure_dir_exists(self.path_config.axp_output_dir)
        
        # Save metrics
        metrics_path = f"{self.path_config.axp_output_dir}/axp_metrics_class{class_label}.parquet"
        self.path_config.write_parquet(metrics_df, metrics_path)
        
        # Save plot if provided
        if plot_path:
            plot_save_path = f"{self.path_config.axp_output_dir}/{plot_path}"
            self.path_config.save_plot(plt.gcf(), plot_save_path)

    def validate_explanations(self, X: Union[np.ndarray, pd.DataFrame], 
                             predictions: np.ndarray,
                             threshold: float = 0.8) -> Dict[str, Any]:
        """
        Validate explanation coverage and reliability.
        
        Args:
            X: Feature matrix
            predictions: Model predictions
            threshold: Minimum acceptable coverage ratio
            
        Returns:
            Dictionary containing validation metrics
        """
        validation_results = {
            'coverage': {},
            'stability': {},
            'reliability': {}
        }
        
        # Check explanation coverage
        for class_label in [0, 1]:
            mask = (predictions == class_label)
            X_class = X[mask]
            explained = self.explain_dataset(X_class, predictions=predictions[mask])
            coverage = len(explained) / len(X_class)
            
            validation_results['coverage'][f'class_{class_label}'] = coverage
            if coverage < threshold:
                print(f"Warning: Low explanation coverage for class {class_label}: {coverage:.2f}")
        
        # Check feature stability
        feature_metrics = self.compute_feature_attribution(X, predictions)
        for class_label, metrics_df in feature_metrics.items():
            unstable_features = metrics_df[metrics_df['stability'] > 0.5]['feature'].tolist()
            validation_results['stability'][f'class_{class_label}'] = unstable_features
            
            if unstable_features:
                print(f"Warning: Unstable features in class {class_label}: {unstable_features}")
        
        # Compute reliability score
        for class_label, metrics_df in feature_metrics.items():
            reliability = np.mean(metrics_df['essentiality_ratio'])
            validation_results['reliability'][f'class_{class_label}'] = reliability
            
        return validation_results

    def compare_with_native_importance(self, 
                                     catboost_model: CatBoostClassifier,
                                     X: pd.DataFrame) -> pd.DataFrame:
        """
        Compare symbolic explanations with native CatBoost feature importance.
        
        Args:
            catboost_model: Trained CatBoost model
            X: Feature matrix
            
        Returns:
            DataFrame comparing different importance metrics
        """
        # Get native feature importance
        native_importance = pd.DataFrame({
            'feature': X.columns,
            'native_importance': catboost_model.feature_importances_
        })
        
        # Get symbolic importance
        predictions = catboost_model.predict(X)
        symbolic_metrics = self.compute_feature_attribution(X, predictions)
        
        # Combine metrics for both classes
        symbolic_importance = pd.DataFrame()
        for class_label, metrics in symbolic_metrics.items():
            df = pd.DataFrame(metrics)
            df['class'] = class_label
            symbolic_importance = pd.concat([symbolic_importance, df])
        
        # Compute correlation
        merged = pd.merge(native_importance, 
                         symbolic_importance.groupby('feature')['relative_importance'].mean().reset_index(),
                         on='feature')
        
        correlation = np.corrcoef(merged['native_importance'], 
                                merged['relative_importance'])[0,1]
        
        merged['correlation'] = correlation
        
        return merged

    def test_feature_significance(self,
                                X: pd.DataFrame,
                                predictions: np.ndarray,
                                n_permutations: int = 1000,
                                alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
        """
        Perform permutation tests to assess feature significance.
        
        Args:
            X: Feature matrix
            predictions: Model predictions
            n_permutations: Number of permutations for the test
            alpha: Significance level
            
        Returns:
            Dictionary containing p-values for each feature
        """
        base_metrics = self.compute_feature_attribution(X, predictions)
        significance_scores = {0: {}, 1: {}}
        
        for class_label in [0, 1]:
            metrics_df = pd.DataFrame(base_metrics[class_label])
            
            for feature in X.columns:
                null_distribution = []
                X_permuted = X.copy()
                
                for _ in range(n_permutations):
                    X_permuted[feature] = np.random.permutation(X_permuted[feature])
                    permuted_metrics = self.compute_feature_attribution(X_permuted, predictions)
                    null_distribution.append(
                        permuted_metrics[class_label][
                            permuted_metrics[class_label]['feature'] == feature
                        ]['support'].iloc[0]
                    )
                
                # Calculate p-value
                actual_value = metrics_df[metrics_df['feature'] == feature]['support'].iloc[0]
                p_value = sum(null_dist >= actual_value for null_dist in null_distribution) / n_permutations
                
                significance_scores[class_label][feature] = p_value
                
                if p_value < alpha:
                    print(f"Feature '{feature}' is significant for class {class_label} (p={p_value:.4f})")
        
        return significance_scores

    def plot_feature_interactions(self,
                                axps: List[Dict],
                                top_k: int = 10,
                                min_weight: int = 2,
                                save_path: Optional[str] = None) -> None:
        """
        Visualize feature interactions in explanations.
        
        Args:
            axps: List of AXP dictionaries
            top_k: Number of top features to include
            min_weight: Minimum interaction weight to show
            save_path: Path to save the plot
        """
        # Create interaction graph
        G = nx.Graph()
        interactions = defaultdict(int)
        
        # Count feature co-occurrences
        for axp in axps:
            features = [cond.split()[0] for cond in axp['axp']]
            for f1, f2 in itertools.combinations(features, 2):
                interactions[(f1, f2)] += 1
        
        # Filter by minimum weight
        significant_interactions = {k: v for k, v in interactions.items() if v >= min_weight}
        
        # Create graph
        for (f1, f2), weight in significant_interactions.items():
            G.add_edge(f1, f2, weight=weight)
        
        # Plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw edges with varying widths
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w/max(weights)*5 for w in weights],
                              edge_color='gray', alpha=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=1000, alpha=0.6)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Feature Interaction Network')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def validate_input_data(self, X: Union[np.ndarray, pd.DataFrame], predictions: np.ndarray) -> None:
        """Validate input data format and consistency."""
        if len(X) != len(predictions):
            raise ValueError("Length mismatch between features and predictions")
        
        if isinstance(X, pd.DataFrame):
            missing_features = set(self.feature_names.values()) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
        
        unique_classes = np.unique(predictions)
        if not all(c in [0, 1] for c in unique_classes):
            raise ValueError("Predictions must be binary (0 or 1)")

    @lru_cache(maxsize=128)
    def _compute_axp_cached(self, rule_ids_tuple: Tuple[int, ...]) -> List[int]:
        """Cached version of AXP computation for repeated patterns."""
        return self._compute_axp(list(rule_ids_tuple))

    def setup_logging(self, log_file: str = None, level: int = logging.INFO) -> None:
        """Configure logging for analysis process."""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file) if log_file else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_large_dataset(self, X: pd.DataFrame, batch_size: int = 1000) -> Dict[str, Any]:
        """Process large datasets in batches to manage memory."""
        results = defaultdict(list)
        
        for i in range(0, len(X), batch_size):
            batch_X = X.iloc[i:i+batch_size]
            batch_results = self.compute_feature_attribution(batch_X)
            
            for class_label, metrics in batch_results.items():
                results[class_label].extend(metrics)
        
        return dict(results)

    def export_results(self, results: Dict[str, Any], output_dir: str, formats: List[str] = ['parquet', 'json']) -> None:
        """Export analysis results in multiple formats to S3."""
        self.path_config.ensure_dir_exists(output_dir)
        
        for class_label, class_results in results.items():
            metrics_df = pd.DataFrame(class_results['metrics'])
            
            if 'parquet' in formats:
                self.path_config.write_parquet(
                    metrics_df, 
                    f"{output_dir}/metrics_class{class_label}.parquet"
                )
            if 'json' in formats:
                self.path_config.write_json(
                    metrics_df.to_dict(orient='records'),
                    f"{output_dir}/metrics_class{class_label}.json"
                )

    def load_tree_rules(self) -> None:
        """Load tree rules from JSON file and transform into logic rules."""
        if not self.path_config.tree_rules_path:
            raise ValueError("Tree rules path not specified in PathConfig")
            
        try:
            self.tree_rules = self.path_config.read_json(self.path_config.tree_rules_path)
            self.logger.info("Successfully loaded tree rules from JSON")
            
            # Extract feature names from tree rules
            if "float_features" in self.tree_rules:
                self.feature_names = {
                    f["flat_feature_index"]: f["feature_id"]
                    for f in self.tree_rules["float_features"]
                }
            
            # Process trees and build logic rules
            self._process_tree_rules()
            
        except Exception as e:
            self.logger.error(f"Error loading tree rules: {str(e)}")
            raise

    def _process_tree_rules(self) -> None:
        """Process tree rules and build logic rules using PySAT."""
        self.rule_clauses.clear()
        self.rule_predictions.clear()
        self.condition_id_map.clear()
        self.id_condition_map.clear()
        
        if not self.tree_rules:
            raise ValueError("Tree rules not loaded. Call load_tree_rules() first.")
            
        # Process each tree in the rules
        for tree in self.tree_rules.get("trees", []):
            self._process_tree(tree)
            
        self.logger.info(f"Processed {len(self.rule_clauses)} logic rules")

    def _process_tree(self, tree: Dict) -> None:
        """Process a single tree and extract logic rules."""
        def traverse(node: Dict, conditions: List[Tuple] = None) -> None:
            if conditions is None:
                conditions = []
                
            if "value" in node:
                # Leaf node - create rule
                pred = 1 if node["value"] > 0 else 0
                clause = [self._get_condition_literal(f, t, d) for (f, t, d) in conditions]
                self.rule_clauses.append(clause)
                self.rule_predictions.append(pred)
                return
                
            if "split" not in node:
                return
                
            split = node["split"]
            split_type = split.get("split_type", "")
            
            if split_type == "FloatFeature":
                feature_index = split["float_feature_index"]
                threshold = split["border"]
                
                # Left child: feature <= threshold
                traverse(node["left"], conditions + [(feature_index, threshold, 0)])
                # Right child: feature > threshold
                traverse(node["right"], conditions + [(feature_index, threshold, 1)])
                
            elif split_type == "OneHotFeature":
                feature_index = split["cat_feature_index"]
                value = split["value"]
                
                # Left child: feature == value
                traverse(node["left"], conditions + [(feature_index, value, 0)])
                # Right child: feature != value
                traverse(node["right"], conditions + [(feature_index, value, 1)])
                
            elif split_type == "OnlineCtr":
                # Handle online CTR splits
                split_index = split["split_index"]
                threshold = split["border"]
                
                # Left child: ctr <= threshold
                traverse(node["left"], conditions + [(split_index, threshold, 0)])
                # Right child: ctr > threshold
                traverse(node["right"], conditions + [(split_index, threshold, 1)])
        
        traverse(tree)

class FeatureVisualization:
    @staticmethod
    def plot_bar_importance(feature_metrics: pd.DataFrame,
                           class_label: int,
                           top_k: int = 10,
                           save_path: Optional[str] = None) -> None:
        """Plot bar chart of feature importance."""
        df = feature_metrics.sort_values('support', ascending=False).head(top_k)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(df)), df['support'], 
                color='lightcoral' if class_label == 1 else 'skyblue')
        plt.xticks(range(len(df)), df['feature'], rotation=45, ha='right')
        plt.title(f'Top {top_k} Features by Support (Class {class_label})')
        plt.xlabel('Feature')
        plt.ylabel('Support Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_mirror_chart(metrics_0: List[Dict],
                         metrics_1: List[Dict],
                         top_k: int = 10,
                         save_path: Optional[str] = None) -> None:
        """Plot mirror chart comparing metrics between classes."""
        df_0 = pd.DataFrame(metrics_0)
        df_1 = pd.DataFrame(metrics_1)
        
        # Get top features by combined support
        features = pd.concat([df_0.assign(class_=0), df_1.assign(class_=1)])
        top_features = features.groupby('feature')['support'].sum().nlargest(top_k).index
        
        plt.figure(figsize=(12, 8))
        
        # Plot bars for both classes
        for _, row in features[features['feature'].isin(top_features)].iterrows():
            plt.barh(
                y=row['feature'],
                width=row['specificity'] * (-1 if row['class_'] == 0 else 1),
                color='skyblue' if row['class_'] == 0 else 'lightcoral',
                alpha=0.8,
                edgecolor='black'
            )
            
            # Add coverage annotations
            plt.text(
                x=row['specificity'] * (-1 if row['class_'] == 0 else 1),
                y=row['feature'],
                s=f" {int(row['coverage'])}",
                va='center',
                ha='left' if row['class_'] == 1 else 'right',
                fontsize=8
            )
        
        plt.axvline(x=0, color='black', linewidth=1)
        plt.xlabel('Specificity (Lower is Better)')
        plt.title('Feature Importance Comparison Between Classes')
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', edgecolor='black', label='Class 0'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Class 1')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_cattail_distribution(feature_metrics: pd.DataFrame,
                                X: Union[np.ndarray, pd.DataFrame],
                                feature_names: Dict[int, str],
                                class_label: int,
                                top_k: int = 10,
                                save_path: Optional[str] = None) -> None:
        """Plot cattail distribution of feature values."""
        # Get top features
        top_features = feature_metrics.nlargest(top_k, 'support')['feature'].tolist()
        
        # Prepare data
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_names.values())
        
        # Melt data for plotting
        X_melted = X[top_features].melt(var_name='Feature', value_name='Value')
        
        plt.figure(figsize=(12, 8))
        
        # Box plot base
        sns.boxplot(y='Feature', x='Value', data=X_melted,
                   whis=1.5, fliersize=0, color='lightgray')
        
        # Cattail overlay
        sns.stripplot(y='Feature', x='Value', data=X_melted,
                     hue='Value', palette='coolwarm',
                     jitter=0.25, size=3, alpha=0.7)
        
        plt.title(f'Feature Value Distribution (Class {class_label})')
        plt.xlabel('Raw Feature Value')
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_normalized_importance(feature_metrics: pd.DataFrame,
                             class_label: int,
                             top_k: int = 10,
                             save_path: Optional[str] = None) -> None:
        """Plot normalized feature importance bar chart."""
        df = feature_metrics.sort_values('support', ascending=False).head(top_k)
        # Normalize support values
        df['normalized_support'] = df['support'] / df['support'].max()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(df)), df['normalized_support'],
                color='lightcoral' if class_label == 1 else 'skyblue')
        plt.xticks(range(len(df)), df['feature'], rotation=45, ha='right')
        plt.title(f'Normalized Feature Importance (Class {class_label})')
        plt.xlabel('Feature')
        plt.ylabel('Normalized Support')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_mirror_frequency(metrics_0: List[Dict],
                             metrics_1: List[Dict],
                             X: Union[np.ndarray, pd.DataFrame],
                             top_k: int = 10,
                             save_path: Optional[str] = None) -> None:
        """Plot mirror chart of feature value frequency statistics."""
        df_0 = pd.DataFrame(metrics_0)
        df_1 = pd.DataFrame(metrics_1)
        
        # Get top features
        features = pd.concat([df_0.assign(class_=0), df_1.assign(class_=1)])
        top_features = features.groupby('feature')['support'].sum().nlargest(top_k).index
        
        # Calculate statistics for each feature
        stats = []
        for feature in top_features:
            values = X[feature] if isinstance(X, pd.DataFrame) else X[:, list(X.columns).index(feature)]
            stats.append({
                'feature': feature,
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean()
            })
        stats_df = pd.DataFrame(stats)
        
        plt.figure(figsize=(12, 8))
        
        # Plot frequency bars with statistics
        for _, row in features[features['feature'].isin(top_features)].iterrows():
            plt.barh(
                y=row['feature'],
                width=row['support'] * (-1 if row['class_'] == 0 else 1),
                color='skyblue' if row['class_'] == 0 else 'lightcoral',
                alpha=0.8,
                edgecolor='black'
            )
            
            # Add statistics annotations
            stat = stats_df[stats_df['feature'] == row['feature']].iloc[0]
            plt.text(
                x=0,
                y=row['feature'],
                s=f" μ={stat['mean']:.2f}\n min={stat['min']:.2f}\n max={stat['max']:.2f}",
                va='center',
                ha='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
        
        plt.axvline(x=0, color='black', linewidth=1)
        plt.xlabel('Feature Value Frequency')
        plt.title('Feature Importance by Frequency with Statistics')
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_mirror_specificity_coverage(metrics_0: List[Dict],
                                         metrics_1: List[Dict],
                                         top_k: int = 10,
                                         save_path: Optional[str] = None) -> None:
        """Plot mirror chart comparing specificity vs coverage."""
        df_0 = pd.DataFrame(metrics_0)
        df_1 = pd.DataFrame(metrics_1)
        
        # Get top features
        features = pd.concat([df_0.assign(class_=0), df_1.assign(class_=1)])
        top_features = features.groupby('feature')['support'].sum().nlargest(top_k).index
        
        plt.figure(figsize=(12, 8))
        
        # Plot bars for both metrics
        for _, row in features[features['feature'].isin(top_features)].iterrows():
            # Plot specificity
            plt.barh(
                y=row['feature'],
                width=row['specificity'] * (-1 if row['class_'] == 0 else 1),
                color='skyblue' if row['class_'] == 0 else 'lightcoral',
                alpha=0.8,
                edgecolor='black'
            )
            
            # Add coverage as text
            plt.text(
                x=row['specificity'] * (-1 if row['class_'] == 0 else 1),
                y=row['feature'],
                s=f" Cov: {int(row['coverage'])}",
                va='center',
                ha='left' if row['class_'] == 1 else 'right',
                fontsize=8
            )
        
        plt.axvline(x=0, color='black', linewidth=1)
        plt.xlabel('Specificity (with Coverage Annotations)')
        plt.title('Feature Importance: Specificity vs Coverage')
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_all_visualizations(results: Dict[int, Dict],
                              X: Union[np.ndarray, pd.DataFrame],
                              feature_names: Dict[int, str],
                              output_dir: str,
                              predictions: np.ndarray) -> None:
        """Generate all visualizations and save to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalized importance bar charts for each class
        for class_label in [0, 1]:
            FeatureVisualization.plot_normalized_importance(
                pd.DataFrame(results[class_label]['metrics']),
                class_label,
                save_path=f"{output_dir}/normalized_importance_class{class_label}.png"
            )
        
        # Cattail distribution plots for each class
        for class_label in [0, 1]:
            mask = (predictions == class_label)
            X_class = X[mask] if isinstance(X, np.ndarray) else X.loc[mask]
            
            FeatureVisualization.plot_cattail_distribution(
                pd.DataFrame(results[class_label]['metrics']),
                X_class,
                feature_names,
                class_label,
                save_path=f"{output_dir}/cattail_distribution_class{class_label}.png"
            )
        
        # Mirror charts
        FeatureVisualization.plot_mirror_frequency(
            results[0]['metrics'],
            results[1]['metrics'],
            X,
            save_path=f"{output_dir}/mirror_frequency_stats.png"
        )
        
        FeatureVisualization.plot_mirror_specificity_coverage(
            results[0]['metrics'],
            results[1]['metrics'],
            save_path=f"{output_dir}/mirror_specificity_coverage.png"
        )

def analyze_model(path_config: PathConfig, model, analysis_config: AnalysisConfig = None) -> Dict[str, Any]:
    """
    Run complete model analysis using configured S3 paths.
    
    Args:
        path_config: PathConfig instance with all required S3 paths
        model: Trained CatBoost model
        analysis_config: Configuration for analysis parameters
        
    Returns:
        Dictionary containing analysis results
    """
    if analysis_config is None:
        analysis_config = AnalysisConfig()
    
    # Initialize explainer and logging
    explainer = CatBoostSymbolicExplainer(path_config)
    explainer.load_model()
    
    # Load and validate data
    test_data = path_config.read_parquet(path_config.test_data_path)
    X_test = test_data.iloc[:, :-1]
    y_pred = model.predict(X_test)
    explainer.validate_input_data(X_test, y_pred)
    
    try:
        # Process large dataset in batches if needed
        if len(X_test) > 1000:
            results = explainer.process_large_dataset(X_test)
        else:
            results = {}
            for class_label in [0, 1]:
                explainer.logger.info(f"Processing class {class_label}")
                
                # Filter data for class
                mask = (y_pred == class_label)
                X_class = X_test[mask]
                
                # Generate explanations with progress bar
                df_axps = explainer.explain_dataset(X_class, predictions=y_pred[mask], show_progress=True)
                
                # Compute metrics
                metrics = explainer._compute_feature_metrics(df_axps)
                
                # Export results in multiple formats
                explainer.export_results(
                    {class_label: {'metrics': metrics}},
                    path_config.axp_output_dir,
                    formats=['parquet', 'json']
                )
                
                results[class_label] = {
                    'axps': df_axps,
                    'metrics': metrics
                }
        
        # Generate all visualizations with configured parameters
        FeatureVisualization.plot_all_visualizations(
            results,
            X_test,
            explainer.feature_names,
            path_config.axp_output_dir,
            y_pred
        )
        
        # Validate explanations
        validation_results = explainer.validate_explanations(
            X_test, 
            y_pred,
            threshold=analysis_config.min_coverage
        )
        
        # Test feature significance
        significance_results = explainer.test_feature_significance(
            X_test,
            y_pred,
            n_permutations=analysis_config.n_permutations,
            alpha=analysis_config.significance_threshold
        )
        
        # Compare with native CatBoost importance
        native_comparison = explainer.compare_with_native_importance(model, X_test)
        
        # Add additional results
        results['validation'] = validation_results
        results['significance'] = significance_results
        results['native_comparison'] = native_comparison
        
        explainer.logger.info("Analysis completed successfully")
        
    except Exception as e:
        explainer.logger.error(f"Error during analysis: {str(e)}")
        raise
    
    return results

