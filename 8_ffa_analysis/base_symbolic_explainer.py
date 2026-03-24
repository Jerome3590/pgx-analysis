"""
Base class for unified symbolic explainers (CatBoost, XGBoost, XGBoost RF).

This module provides a unified schema and interface for all model types, ensuring
consistent behavior and easier maintenance.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from itertools import count
from abc import ABC, abstractmethod

try:
    from pysat.examples.hitman import Hitman
except ImportError:
    Hitman = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _explain_instance_worker(task: Tuple[int, List[float], int, Dict, Optional[Dict]]) -> Dict:
    """
    Worker function for parallel explanation generation.
    This function must be at module level to be picklable.
    
    Args:
        task: Tuple of (index, instance_values, predicted_class, explainer_state, instance_shap_values)
              instance_shap_values: Optional dict mapping feature_name -> SHAP value for this specific instance
        
    Returns:
        Dictionary with explanation results
    """
    # Import numpy at function level to ensure it's available in multiprocessing context
    import numpy as np

    idx, instance_values, predicted_class, explainer_state, instance_shap_values = task
    
    # Reconstruct minimal explainer state for this instance
    rule_clauses = explainer_state['rule_clauses']
    rule_predictions = explainer_state['rule_predictions']
    id_condition_map = explainer_state['id_condition_map']
    feature_names = explainer_state['feature_names']
    _class_rule_indices = explainer_state.get('_class_rule_indices', {})
    shap_importance_map = explainer_state.get('shap_importance_map', {})
    rule_frequencies = explainer_state.get('rule_frequencies', {})  # NEW: Get rule frequencies for Set 5
    
    if not shap_importance_map:
        raise ValueError("shap_importance_map is required in explainer_state. Only rules with SHAP importance > 0 will be used.")
    
    # Require instance-specific SHAP values - do not fall back to global
    if not instance_shap_values:
        raise ValueError(
            f"Instance-specific SHAP values are required for instance {idx}. "
            f"Cannot use global SHAP importance for instance-specific rule filtering. "
            f"Please ensure individual SHAP values are loaded from Step 7 (SHAP Analysis)."
        )
    
    # Convert instance back to numpy array (ensure it's 1D)
    # instance_values should be a list from x.tolist()
    if isinstance(instance_values, (int, float)):
        # Scalar - shouldn't happen, but handle it
        raise ValueError(f"Expected list, got scalar: {instance_values}")
    elif isinstance(instance_values, list):
        # Convert list to numpy array
        instance = np.array(instance_values, dtype=np.float64)
        # Ensure it's 1D
        if instance.ndim == 0:
            raise ValueError(f"List converted to scalar: {instance_values[:5]}...")
        elif instance.ndim > 1:
            instance = instance.flatten()
    else:
        # Try to convert whatever it is
        instance = np.array(instance_values, dtype=np.float64)
        if instance.ndim == 0:
            raise ValueError(f"Expected list, got scalar: {instance_values}")
        elif instance.ndim > 1:
            instance = instance.flatten()
    
    # Ensure it's 1D
    if instance.ndim != 1:
        raise ValueError(f"Expected 1D array, got {instance.ndim}D array with shape {instance.shape}, instance_values type: {type(instance_values)}")
    
    # Validate length (should match number of features, typically 137)
    if len(instance) < 100:  # Sanity check - should have many features
        raise ValueError(f"Instance array too short: {len(instance)} elements, expected ~137. First few: {instance[:5]}")
    
    # Get rule indices for this class (cached)
    if predicted_class not in _class_rule_indices:
        _class_rule_indices[predicted_class] = [
            i for i, pred in enumerate(rule_predictions) if pred == predicted_class
        ]
    rule_indices = _class_rule_indices[predicted_class]
    
    # Find satisfied rules
    matched = []
    for idx_rule in rule_indices:
        clause = rule_clauses[idx_rule]
        if all(
            (instance[feat] <= thresh if dir == 0 else instance[feat] > thresh)
            for (feat, thresh, dir) in (id_condition_map[lit] for lit in clause)
        ):
            matched.append(idx_rule)
    
    # Score rules by SHAP using instance-specific SHAP values (required)
    def score_rule_by_shap(rule_id):
        clause = rule_clauses[rule_id]
        score = 0.0
        features_in_rule = set()
        for lit in clause:
            feat_idx, _, _ = id_condition_map[lit]
            feat_name = feature_names.get(feat_idx, f"f{feat_idx}")
            features_in_rule.add(feat_name)
        for feat_name in features_in_rule:
            # Use instance-specific SHAP value (required - no fallback)
            if feat_name in instance_shap_values:
                # Use absolute value of instance-specific SHAP
                score += abs(instance_shap_values[feat_name])
            else:
                # Log warning but continue (feature might not be in SHAP values)
                # This can happen if feature wasn't in the SHAP evaluation set
                pass
        return score
    
    # Set 1: First 100 rules (from all matched rules)
    max_rules = 100
    first_rules = matched[:max_rules] if len(matched) > max_rules else matched
    
    # Set 2: Random sample of 100 rules (from all matched rules, seed for reproducibility)
    import random
    random.seed(42)
    if len(matched) > max_rules:
        random_rules = random.sample(matched, max_rules)
    else:
        random_rules = matched.copy()
    
    # Set 3: Top K rules by SHAP importance with percentile threshold fallback
    # Score all matched rules and take top K OR all above percentile (whichever is larger)
    rule_scores = [(rid, score_rule_by_shap(rid)) for rid in matched]
    rule_scores.sort(key=lambda x: x[1], reverse=True)  # Sort all rules by score (including score = 0)
    
    # Separate rules with SHAP > 0 from rules with SHAP = 0
    shap_positive_rules = [(rid, score) for rid, score in rule_scores if score > 0]
    shap_zero_rules = [rid for rid, score in rule_scores if score == 0]

    # Take top 300 from SHAP > 0 rules
    top_300 = [rid for rid, score in shap_positive_rules[:300]]

    # Also include all rules above 10th percentile (safety net for important rules)
    if len(shap_positive_rules) > 0:
        import numpy as np
        scores = [score for _, score in shap_positive_rules]
        percentile_10 = np.percentile(scores, 10.0)
        percentile_rules = [rid for rid, score in shap_positive_rules if score >= percentile_10]
        # Use larger set to ensure coverage
        shap_filtered_matched = percentile_rules if len(percentile_rules) > len(top_300) else top_300
    else:
        shap_filtered_matched = top_300
    
    # Set 4: Fallback - include SHAP = 0 rules that aren't already covered
    # This ensures we don't miss rules that might be important for FFA but have SHAP = 0
    covered_rules = set(first_rules) | set(random_rules) | set(shap_filtered_matched)
    missing_shap_zero_rules = [rid for rid in shap_zero_rules if rid not in covered_rules]
    # Add up to 100 additional SHAP = 0 rules for completeness
    max_fallback_rules = 100
    fallback_rules = missing_shap_zero_rules[:max_fallback_rules]
    
    # Set 5: Top 100 most frequent rules (across dataset)
    # This ensures frequent patterns are captured, matching reference implementation's implicit weighting
    frequent_rules = []
    if rule_frequencies:
        # Sort rules by frequency (descending)
        sorted_by_freq = sorted(rule_frequencies.items(), key=lambda x: x[1], reverse=True)
        # Take top 100 frequent rules that match this instance
        max_frequent_rules = 100
        frequent_rules = [rule_id for rule_id, freq in sorted_by_freq[:max_frequent_rules] if rule_id in matched]
    
    # Union all five sets to get final unique rule set
    combined_rule_ids = list(set(first_rules) | set(random_rules) | set(shap_filtered_matched) | set(fallback_rules) | set(frequent_rules))
    
    # Compute AXP if we have any rules
    if not combined_rule_ids:
        axp_literals = []
    else:
        if Hitman is None:
            raise ImportError("pysat is required for AXP computation")
        
        h = Hitman(solver="m22")
        for ridx in combined_rule_ids:
            h.hit(rule_clauses[ridx])
        axp_literals = h.get() or []
    
    # Convert literals to text
    axp_text = []
    for lit in axp_literals:
        feat_idx, thresh, direction = id_condition_map[lit]
        feat = feature_names.get(feat_idx, f"f{feat_idx}")
        op = "<=" if direction == 0 else ">"
        axp_text.append(f"{feat} {op} {thresh}")
    
    return {
        "index": idx,
        "predicted_class": predicted_class,
        "axp": axp_text
    }


# Unified DataFrame Schema for Tree Paths
TREE_PATH_SCHEMA = [
    'tree_idx',      # Index of the tree in the ensemble
    'path_idx',      # Unique identifier for this path (leaf index or path number)
    'step_in_path',  # Step number within this path (0-based)
    'feature_idx',   # Feature index (integer)
    'feature_name',  # Feature name (string)
    'threshold',     # Split threshold (float)
    'direction',     # 0 for <= (left), 1 for > (right)
    'depth',         # Depth in tree (0 = root)
    'leaf_value',    # Leaf value (float)
    'prediction',    # Binary prediction (0 or 1)
    'path_length'    # Total length of this path
]


class BaseSymbolicExplainer(ABC):
    """
    Base class for all symbolic explainers.
    
    Provides unified interface and common functionality for:
    - CatBoost (oblivious and non-oblivious trees)
    - XGBoost (standard and RF mode)
    
    All explainers use the same:
    - DataFrame schema for tree paths
    - Rule creation logic
    - AXP computation
    - Explanation generation
    """
    
    def __init__(self, path_config=None, shap_importance_map: Dict[str, float] = None,
                 shap_values_df: Optional[pd.DataFrame] = None):
        """
        Initialize base explainer with common attributes.
        
        Args:
            path_config: Path configuration object
            shap_importance_map: Required dict mapping feature_name -> mean_abs_shap value.
                                 Used as fallback and for validation.
            shap_values_df: REQUIRED DataFrame with individual SHAP values per instance.
                           Indexed by instance index, columns are feature names.
                           Each row contains SHAP values for one instance.
        
        Raises:
            ValueError: If shap_importance_map or shap_values_df is None or empty
        """
        if not shap_importance_map:
            raise ValueError("shap_importance_map is required. Only rules with SHAP importance > 0 will be used.")
        
        if shap_values_df is None or len(shap_values_df) == 0:
            raise ValueError(
                "shap_values_df is REQUIRED. Individual SHAP values per instance are required "
                "for accurate instance-specific rule filtering. Please load individual SHAP values "
                "from Step 7 (SHAP Analysis) parquet file."
            )
        
        self.path_config = path_config
        self.condition_id_map = {}  # Maps (feature_idx, threshold, direction) -> literal_id
        self.id_condition_map = {}  # Maps literal_id -> (feature_idx, threshold, direction)
        self.rule_clauses = []      # List of rule clauses (each clause is list of literal_ids)
        self.rule_predictions = []  # List of predictions (0 or 1) for each rule
        self.feature_names = {}     # Maps feature_idx -> feature_name
        self.model_json = None      # Model JSON structure
        self._id_gen = count(1)     # Generator for literal IDs (start at 1 for SAT)
        self._class_rule_indices = {}  # Cache: maps class -> list of rule indices for that class
        self.shap_importance_map = shap_importance_map  # Feature name -> SHAP importance (for validation)
        self.shap_values_df = shap_values_df  # Individual SHAP values per instance (REQUIRED)
        self.rule_frequencies = {}  # Maps rule_id -> frequency (how often rule matches across dataset)
        self._std_log_path = None  # repo logs/8_ffa_analysis/ffa_*.log (Step 8 standard path; mirrors to S3)
        self.setup_logging()
    
    def setup_logging(self, log_file: Optional[str] = None, level: int = logging.INFO) -> None:
        """
        Configure Step 8 logging like Step 7 SHAP: one named logger, file under logs/8_ffa_analysis/,
        plus stderr — no logging.basicConfig (avoids silent drops when root already has handlers).
        """
        from pathlib import Path as _Path

        pc = getattr(self, "path_config", None)
        cohort = getattr(pc, "cohort", None) if pc else None
        age_band = getattr(pc, "age_band", None) if pc else None
        density_bin = getattr(pc, "density_bin", None) if pc else None

        _repo_root = _Path(__file__).resolve().parents[1]
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if cohort and age_band:
            ab = str(age_band).replace("-", "_")
            bin_suffix = f"_{density_bin}" if density_bin else ""
            log_name = f"8_ffa_analysis.{cohort}.{ab}{bin_suffix}"
            _log_dir = _repo_root / "logs" / "8_ffa_analysis"
            _log_dir.mkdir(parents=True, exist_ok=True)
            std_path = _log_dir / f"ffa_{cohort}_{ab}{bin_suffix}.log"
        else:
            log_name = f"8_ffa_analysis.{self.__class__.__name__}"
            std_path = None

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(level)

        def _norm_path(p: str) -> str:
            try:
                return os.path.normcase(os.path.abspath(p))
            except Exception:
                return p

        def _has_file_target(target: str) -> bool:
            t = _norm_path(target)
            for h in self.logger.handlers:
                if isinstance(h, logging.FileHandler):
                    bf = getattr(h, "baseFilename", None)
                    if bf and _norm_path(str(bf)) == t:
                        return True
            return False

        # Primary file: same layout as 7_shap_analysis (logs/<step>/<step>_<cohort>_<age>.log)
        if std_path is not None:
            sp = str(std_path)
            if not _has_file_target(sp):
                fh = logging.FileHandler(sp, mode="a", encoding="utf-8")
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)
            self._std_log_path = std_path

        if log_file and not _has_file_target(log_file):
            fh2 = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh2.setFormatter(fmt)
            self.logger.addHandler(fh2)

        if not any(type(h) is logging.StreamHandler for h in self.logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)

        self.logger.propagate = False

    def mirror_logs_to_s3(self) -> None:
        """Mirror the standard logs/8_ffa_analysis/ log file to S3 (best-effort). Call after analysis completes."""
        std_log = getattr(self, '_std_log_path', None)
        if std_log is None:
            return
        from pathlib import Path as _Path
        std_log = _Path(std_log)
        if not std_log.exists():
            return
        path_config = getattr(self, 'path_config', None)
        cohort = getattr(path_config, 'cohort', None) if path_config else None
        age_band = getattr(path_config, 'age_band', None) if path_config else None
        if not cohort or not age_band:
            return
        try:
            import sys
            _repo_root = _Path(__file__).resolve().parents[1]
            if str(_repo_root) not in sys.path:
                sys.path.insert(0, str(_repo_root))
            from py_helpers.fe_monitor import mirror_log_to_s3
            mirror_log_to_s3("8_ffa_analysis", cohort, age_band, std_log, self.logger)
        except Exception:
            pass
    
    def _get_condition_literal(self, feat_idx: int, threshold: float, direction: int) -> int:
        """
        Get or create a literal ID for a condition.
        
        Args:
            feat_idx: Feature index
            threshold: Split threshold
            direction: 0 for <=, 1 for >
            
        Returns:
            Literal ID (positive integer)
        """
        key = (feat_idx, threshold, direction)
        if key not in self.condition_id_map:
            lit = next(self._id_gen)
            self.condition_id_map[key] = lit
            self.id_condition_map[lit] = key
        return self.condition_id_map[key]
    
    @abstractmethod
    def fit_from_model_json(self, model_json: Dict[str, Any]) -> None:
        """
        Parse model JSON and build symbolic CNF clauses.
        
        This method should:
        1. Extract feature names
        2. Parse trees (using model-specific logic)
        3. Explode trees to DataFrame
        4. Create rules from DataFrame
        
        Args:
            model_json: Model JSON structure (format depends on model type)
        """
        pass
    
    @abstractmethod
    def _explode_tree_to_dataframe(self, tree: Dict, tree_idx: int = 0) -> pd.DataFrame:
        """
        Explode a tree structure into a unified DataFrame schema.
        
        Args:
            tree: Tree structure (format depends on model type)
            tree_idx: Index of tree in ensemble
            
        Returns:
            DataFrame conforming to TREE_PATH_SCHEMA
        """
        pass
    
    def _create_rules_from_dataframe(self, df_paths: pd.DataFrame) -> None:
        """
        Create rules from exploded tree DataFrame.
        
        This is the unified rule creation logic used by all model types.
        Each unique path_idx represents one rule (one path from root to leaf).
        
        Args:
            df_paths: DataFrame conforming to TREE_PATH_SCHEMA
        """
        if len(df_paths) == 0:
            self.logger.warning("Empty paths DataFrame, no rules to create")
            return
        
        # Group by path_idx to get complete paths
        unique_paths = df_paths.groupby('path_idx').first()
        
        for path_idx, path_info in unique_paths.iterrows():
            # Get all steps for this path (excluding the LEAF row)
            path_steps = df_paths[
                (df_paths['path_idx'] == path_idx) & 
                (df_paths['feature_name'] != 'LEAF')
            ].sort_values('step_in_path')
            
            if len(path_steps) == 0:
                continue
            
            # Extract conditions
            conditions = []
            for _, step in path_steps.iterrows():
                feat_idx = step['feature_idx']
                threshold = step['threshold']
                direction = step['direction']
                
                if feat_idx is not None and threshold is not None and direction is not None:
                    try:
                        # Create literal to ensure it exists in mapping
                        self._get_condition_literal(int(feat_idx), float(threshold), int(direction))
                        conditions.append((int(feat_idx), float(threshold), int(direction)))
                    except Exception as e:
                        self.logger.error(f"Error creating literal for path {path_idx}, step: {step.to_dict()}, error: {e}")
                        continue
            
            if conditions:
                # Create clause from conditions
                clause = [self._get_condition_literal(f, t, d) for (f, t, d) in conditions]
                pred = int(path_info['prediction'])
                
                self.rule_clauses.append(clause)
                self.rule_predictions.append(pred)
                
                if len(self.rule_clauses) <= 5:
                    self.logger.debug(f"Created rule {len(self.rule_clauses)-1} from path {path_idx}: "
                                    f"{len(clause)} conditions, prediction={pred}")
    
    def _get_rule_indices_for_class(self, target_class: int) -> List[int]:
        """Get cached list of rule indices for a given class."""
        if target_class not in self._class_rule_indices:
            # Build cache on first access
            self._class_rule_indices[target_class] = [
                idx for idx, pred in enumerate(self.rule_predictions) 
                if pred == target_class
            ]
        return self._class_rule_indices[target_class]
    
    def _satisfied_rules(self, instance: np.ndarray, target_class: int) -> List[int]:
        """
        Return rule indexes satisfied by instance and match class.
        Optimized to only check rules for the target class.
        
        Args:
            instance: Feature vector
            target_class: Target class (0 or 1)
            
        Returns:
            List of rule indices that match
        """
        import time
        start_time = time.time()
        matched = []
        
        # Get only rules for this class (cached)
        rule_indices = self._get_rule_indices_for_class(target_class)
        total_rules_for_class = len(rule_indices)
        
        # Log start
        if hasattr(self, 'logger'):
            self.logger.info(f"_satisfied_rules: Starting check of {total_rules_for_class} rules for class {target_class}")
        
        # Only iterate through rules for this class; log first 10 progress points then only errors
        progress_log_count = [0]  # use list so inner scope can mutate
        for check_idx, idx in enumerate(rule_indices, 1):
            clause = self.rule_clauses[idx]
            
            # Log progress at first 10 milestones (every 500 rules) only
            if hasattr(self, 'logger') and check_idx % 500 == 0 and progress_log_count[0] < 10:
                progress_log_count[0] += 1
                elapsed = time.time() - start_time
                self.logger.info(f"_satisfied_rules: Checked {check_idx}/{total_rules_for_class} rules, matched {len(matched)}, elapsed {elapsed:.2f}s")
            
            # Check if all conditions in clause are satisfied
            # Use _literal_condition_holds for proper binary/categorical handling
            if all(
                self._literal_condition_holds(instance, lit)
                for lit in clause
            ):
                matched.append(idx)
        
        duration = time.time() - start_time
        if hasattr(self, 'logger'):
            self.logger.info(f"_satisfied_rules: Completed - checked {total_rules_for_class} rules, matched {len(matched)}, took {duration:.4f}s")
        
        return matched
    
    def _score_rule_by_shap(self, rule_id: int) -> float:
        """
        Score a rule based on SHAP importance of features it contains.
        
        Args:
            rule_id: Rule index
            
        Returns:
            SHAP-based score (sum of mean_abs_shap for features in rule)
        """
        clause = self.rule_clauses[rule_id]
        score = 0.0
        
        # Get unique features in this rule
        features_in_rule = set()
        for lit in clause:
            feat_idx, _, _ = self.id_condition_map[lit]
            feat_name = self.feature_names.get(feat_idx, f"f{feat_idx}")
            features_in_rule.add(feat_name)
        
        # Sum SHAP importance for features in rule
        for feat_name in features_in_rule:
            score += self.shap_importance_map.get(feat_name, 0.0)
        
        return score
    
    def _get_features_in_rule(self, rule_id: int) -> Set[str]:
        """Get set of feature names in a rule."""
        clause = self.rule_clauses[rule_id]
        features = set()
        for lit in clause:
            feat_idx, _, _ = self.id_condition_map[lit]
            feat_name = self.feature_names.get(feat_idx, f"f{feat_idx}")
            features.add(feat_name)
        return features
    
    def _filter_rules_by_shap(self, rule_ids: List[int], top_k: int = 300, min_shap_score: float = 0.0, 
                              percentile_threshold: float = None, ensure_shap_coverage: bool = True) -> List[int]:
        """
        Filter rules using hybrid approach: top-K OR threshold-based (whichever captures more rules).
        
        NEW: Also ensures features with SHAP > 0 are represented in the filtered rule set.

        This reduces the number of rules while ensuring we don't miss important rules that might
        be ranked lower globally but are still significant (above threshold).

        Strategy:
        - Score all rules by SHAP importance
        - Take top K rules by score
        - Also take all rules above percentile threshold (if provided) or min_shap_score
        - NEW: Ensure all features with SHAP > 0 are represented (add rules containing them if missing)
        - Return union of all sets

        Args:
            rule_ids: List of rule indices to filter
            top_k: Maximum number of top rules to return (default: 300)
            min_shap_score: Minimum SHAP score threshold (default: 0.0, filters out zero-importance rules)
            percentile_threshold: Optional percentile threshold (0-100). If provided, also includes
                                 all rules above this percentile, even if beyond top_k.
            ensure_shap_coverage: If True, ensures all features with SHAP > 0 are represented (default: True)

        Returns:
            List of rule indices: top K rules OR all rules above threshold (whichever set is larger),
            plus rules needed to cover features with SHAP > 0
        """
        # Score all rules
        rule_scores = []
        for rid in rule_ids:
            shap_score = self._score_rule_by_shap(rid)
            if shap_score > min_shap_score:
                rule_scores.append((rid, shap_score))

        if not rule_scores:
            return []

        # Sort by SHAP score (descending)
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        scores = [score for _, score in rule_scores]

        # Strategy 1: Take top K rules
        top_k_rules = [rid for rid, score in rule_scores[:top_k]]

        # Strategy 2: Also include rules above percentile threshold (if provided)
        threshold_rules = top_k_rules.copy()  # Start with top K
        if percentile_threshold is not None and percentile_threshold > 0:
            import numpy as np
            threshold_value = np.percentile(scores, percentile_threshold)
            # Add all rules above percentile threshold (even if beyond top_k)
            threshold_rules = [rid for rid, score in rule_scores if score >= threshold_value]
            # Deduplicate
            threshold_rules = list(dict.fromkeys(threshold_rules))  # Preserves order

        # Use the larger set from strategies 1 and 2
        if len(threshold_rules) > len(top_k_rules):
            selected_rules = threshold_rules.copy()
            strategy = f"percentile_threshold (>{percentile_threshold}th percentile)"
        else:
            selected_rules = top_k_rules.copy()
            strategy = f"top_k={top_k}"

        # Strategy 3: Ensure features with SHAP > 0 are represented
        if ensure_shap_coverage and self.shap_importance_map:
            # Get all features with SHAP > 0
            features_with_shap = {
                feat for feat, score in self.shap_importance_map.items() 
                if score > 0
            }
            
            # Get features already covered by selected rules
            covered_features = set()
            for rid in selected_rules:
                covered_features.update(self._get_features_in_rule(rid))
            
            # Find missing features (SHAP > 0 but not in selected rules)
            missing_features = features_with_shap - covered_features
            
            if missing_features and hasattr(self, 'logger'):
                self.logger.info(
                    f"_filter_rules_by_shap: {len(missing_features)} features with SHAP > 0 "
                    f"not covered by selected rules. Adding rules to ensure coverage."
                )
            
            # For each missing feature, find at least one rule that contains it
            # Prefer rules with higher SHAP scores
            coverage_rules = []
            for feat in missing_features:
                # Find all rules containing this feature
                rules_with_feat = []
                for rid, score in rule_scores:
                    if feat in self._get_features_in_rule(rid):
                        rules_with_feat.append((rid, score))
                
                # Sort by score and take the best one
                if rules_with_feat:
                    rules_with_feat.sort(key=lambda x: x[1], reverse=True)
                    best_rule_id = rules_with_feat[0][0]
                    if best_rule_id not in selected_rules:
                        coverage_rules.append(best_rule_id)
                        covered_features.add(feat)
            
            # Add coverage rules to selected set
            if coverage_rules:
                selected_rules.extend(coverage_rules)
                selected_rules = list(dict.fromkeys(selected_rules))  # Deduplicate
                strategy += f" + {len(coverage_rules)} coverage rules for SHAP > 0 features"
                
                if hasattr(self, 'logger'):
                    final_coverage = len(covered_features) / len(features_with_shap) * 100 if features_with_shap else 0
                    self.logger.info(
                        f"_filter_rules_by_shap: Coverage: {len(covered_features)}/{len(features_with_shap)} "
                        f"features with SHAP > 0 ({final_coverage:.1f}%)"
                    )

        if hasattr(self, 'logger'):
            max_score = scores[0] if scores else 0.0
            self.logger.info(
                f"_filter_rules_by_shap: Filtered {len(rule_ids)} rules -> "
                f"{len(selected_rules)} rules using {strategy} "
                f"(min_score={min_shap_score:.6f}, max_score={max_score:.6f})"
            )

        return selected_rules
    
    def _compute_axp(self, rule_ids: List[int]) -> List[int]:
        """
        Compute minimal hitting set (AXP) over matching rule IDs.
        
        Process:
        1. Take first 100 rules (from all matched rules)
        2. Take random sample of 100 rules (from all matched rules)
        3. Take top 300 SHAP-scored rules OR all above 10th percentile
        4. Take up to 100 SHAP=0 rules as fallback
        5. Take top 100 most frequent rules (across dataset) - Set 5
        6. Union all five sets to get final unique rule set
        7. Compute AXP from the union
        
        Args:
            rule_ids: List of rule indices
            
        Returns:
            List of literal IDs forming the minimal hitting set
        """
        import time
        import random
        start_time = time.time()
        
        if Hitman is None:
            raise ImportError("pysat is required for AXP computation. Install with: pip install python-sat")
        
        if hasattr(self, 'logger'):
            self.logger.info(f"_compute_axp: Starting computation for {len(rule_ids)} matched rules")
        
        max_rules = 100  # Limit to prevent hanging
        
        # Set 1: First 100 rules (from all matched rules)
        first_rules = rule_ids[:max_rules] if len(rule_ids) > max_rules else rule_ids
        
        # Set 2: Random sample of 100 rules (from all matched rules, seed for reproducibility)
        random.seed(42)
        if len(rule_ids) > max_rules:
            random_rules = random.sample(rule_ids, max_rules)
        else:
            random_rules = rule_ids.copy()
        
        # Set 3: Top K rules by SHAP importance with percentile threshold fallback
        # Uses top 300 OR all rules above 10th percentile (whichever is larger)
        # NEW: Also ensures all features with SHAP > 0 are represented
        # This ensures we don't miss important rules while still reducing computation
        shap_filtered_rules = self._filter_rules_by_shap(
            rule_ids, 
            top_k=300, 
            min_shap_score=0.0,
            percentile_threshold=10.0,  # Include all rules above 10th percentile as safety net
            ensure_shap_coverage=True  # Ensure features with SHAP > 0 are represented
        )
        
        # Set 4: Fallback - include SHAP = 0 rules that aren't already covered
        # Score all matched rules to identify SHAP = 0 rules
        rule_scores = [(rid, self._score_rule_by_shap(rid)) for rid in rule_ids]
        shap_zero_rules = [rid for rid, score in rule_scores if score == 0]
        covered_rules = set(first_rules) | set(random_rules) | set(shap_filtered_rules)
        missing_shap_zero_rules = [rid for rid in shap_zero_rules if rid not in covered_rules]
        max_fallback_rules = 100
        fallback_rules = missing_shap_zero_rules[:max_fallback_rules]
        
        # Set 5: Top 100 most frequent rules (across dataset)
        # This ensures frequent patterns are captured, matching reference implementation's implicit weighting
        frequent_rules = []
        if hasattr(self, 'rule_frequencies') and self.rule_frequencies:
            # Sort rules by frequency (descending)
            sorted_by_freq = sorted(self.rule_frequencies.items(), key=lambda x: x[1], reverse=True)
            # Take top 100 frequent rules that match this instance
            max_frequent_rules = 100
            frequent_rules = [rule_id for rule_id, freq in sorted_by_freq[:max_frequent_rules] if rule_id in rule_ids]
        
        if hasattr(self, 'logger'):
            self.logger.info(f"_compute_axp: First 100: {len(first_rules)}, Random 100: {len(random_rules)}, Top SHAP: {len(shap_filtered_rules)}, Fallback: {len(fallback_rules)}, Frequent: {len(frequent_rules)}")
        
        # Union all five sets to get final unique rule set
        combined_rule_ids = list(set(first_rules) | set(random_rules) | set(shap_filtered_rules) | set(fallback_rules) | set(frequent_rules))
        
        if not combined_rule_ids:
            if hasattr(self, 'logger'):
                self.logger.warning("_compute_axp: No rules after combining sets, returning empty AXP")
            return []
        
        if hasattr(self, 'logger'):
            self.logger.info(f"_compute_axp: Combined sets -> {len(combined_rule_ids)} unique rules (deduplicated)")
        
        # Compute AXP from the union of all three sets
        h = Hitman(solver="m22")
        for ridx in combined_rule_ids:
            h.hit(self.rule_clauses[ridx])
        
        if hasattr(self, 'logger'):
            self.logger.info(f"_compute_axp: Computing AXP from {len(combined_rule_ids)} unique rules...")
        
        result = h.get()
        if result is None:
            result = []
            if hasattr(self, 'logger'):
                self.logger.warning(f"_compute_axp: Hitman.get() returned None for {len(combined_rule_ids)} rules - no valid explanation found")
        
        duration = time.time() - start_time
        if hasattr(self, 'logger'):
            self.logger.info(f"_compute_axp: {len(combined_rule_ids)} unique rules -> {len(result)} literals, took {duration:.4f}s")
        
        return result
    
    def explain_literals(self, instance: np.ndarray, predicted_class: int) -> List[int]:
        """
        Return minimal AXP literals for instance.
        
        Args:
            instance: Feature vector
            predicted_class: Predicted class (0 or 1)
            
        Returns:
            List of literal IDs forming the AXP
        """
        import time
        start_time = time.time()
        
        matched = self._satisfied_rules(instance, predicted_class)
        if not matched:
            if hasattr(self, 'logger'):
                self.logger.info(f"explain_literals: no matched rules for class {predicted_class}")
            return []
        
        result = self._compute_axp(matched)
        duration = time.time() - start_time
        
        if hasattr(self, 'logger'):
            self.logger.info(f"explain_literals: total time {duration:.4f}s, {len(matched)} matched -> {len(result)} literals")
        
        return result
    
    def explain_instance(self, instance: Union[np.ndarray, pd.DataFrame], predicted_class: Optional[int] = None,
                        instance_index: Optional[int] = None) -> List[str]:
        """
        Return readable explanation (AXP) for instance.
        
        Args:
            instance: Feature vector (numpy array or DataFrame row)
            predicted_class: Predicted class (0 or 1), required
            instance_index: Optional index of instance (for looking up individual SHAP values)
            
        Returns:
            List of human-readable condition strings
        """
        if predicted_class is None:
            raise ValueError("You must provide the predicted class.")
        
        if isinstance(instance, pd.DataFrame):
            instance = instance.values[0] if len(instance) == 1 else instance.values
        
        # Get instance-specific SHAP values (required - no fallback)
        if instance_index is None:
            raise ValueError("instance_index is required for explain_instance when using individual SHAP values")
        
        if self.shap_values_df is None:
            error_msg = (
                "ERROR: Individual SHAP values DataFrame is not available. "
                "Instance-specific SHAP values are REQUIRED for accurate rule filtering. "
                "Please ensure individual SHAP values are loaded from Step 7 (SHAP Analysis)."
            )
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            if instance_index in self.shap_values_df.index:
                instance_shap_row = self.shap_values_df.loc[instance_index]
                instance_shap_values = instance_shap_row.to_dict()
            elif len(self.shap_values_df) > instance_index:
                instance_shap_row = self.shap_values_df.iloc[instance_index]
                instance_shap_values = instance_shap_row.to_dict()
            else:
                raise IndexError(f"Instance index {instance_index} out of range for SHAP values DataFrame (length: {len(self.shap_values_df)})")
        except (KeyError, IndexError) as e:
            error_msg = (
                f"ERROR: Could not find individual SHAP values for instance {instance_index}: {e}. "
                f"Individual SHAP values per instance are REQUIRED for accurate rule filtering. "
                f"SHAP values DataFrame has {len(self.shap_values_df)} rows, "
                f"requested instance index: {instance_index}. "
                f"Please ensure SHAP values are loaded correctly from Step 7 (SHAP Analysis)."
            )
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Temporarily override shap_importance_map for this instance using individual SHAP values
        original_shap_map = self.shap_importance_map
        # Create a temporary map using absolute values of instance-specific SHAP
        temp_shap_map = {feat: abs(val) for feat, val in instance_shap_values.items()}
        # Merge with global map (instance-specific takes precedence)
        self.shap_importance_map = {**original_shap_map, **temp_shap_map}
        
        try:
            literals = self.explain_literals(instance, predicted_class)
            return [self._literal_to_text(lit) for lit in literals]
        finally:
            # Restore original shap_importance_map
            self.shap_importance_map = original_shap_map
    
    def _literal_to_text(self, lit: int) -> str:
        """
        Convert a literal to human-readable text.
        
        Args:
            lit: Literal ID
            
        Returns:
            Human-readable condition string (e.g., "feature_name <= 0.5")
        """
        feat_idx, thresh, direction = self.id_condition_map[lit]
        feat = self.feature_names.get(feat_idx, f"f{feat_idx}")
        op = "<=" if direction == 0 else ">"
        return f"{feat} {op} {thresh}"
    
    def compute_rule_frequencies(self, X: Union[np.ndarray, pd.DataFrame], 
                                 predictions: np.ndarray) -> Dict[int, int]:
        """
        Compute how often each rule matches across the dataset.
        
        This is used to identify frequent rules for Set 5 (rule frequency-based selection).
        Rule frequencies are computed once and reused for all instances.
        
        Args:
            X: Feature matrix (numpy array or DataFrame)
            predictions: Array of predicted classes (required)
            
        Returns:
            Dictionary mapping rule_id -> frequency (number of instances where rule matches)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if predictions is None:
            raise ValueError("Please provide the predicted class labels for each instance.")
        
        from collections import defaultdict
        rule_frequencies = defaultdict(int)
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Computing rule frequencies across {len(X)} instances...")
        
        for i, (instance, predicted_class) in enumerate(zip(X, predictions, strict=True)):
            # Ensure instance is 1D numpy array
            if isinstance(instance, pd.Series):
                instance = instance.values
            if not isinstance(instance, np.ndarray):
                instance = np.array(instance)
            if instance.ndim > 1:
                instance = instance.flatten()
            
            # Get matching rules for this instance
            matched_rules = self._satisfied_rules(instance, predicted_class)
            
            # Count frequency of each matched rule
            for rule_id in matched_rules:
                rule_frequencies[rule_id] += 1
            
            # Log progress (more often on Windows — avoids "hung" appearance under Jupyter)
            _freq_every = 500 if os.name == "nt" else 1000
            if hasattr(self, 'logger') and (i + 1) % _freq_every == 0:
                self.logger.info(f"  Processed {i+1}/{len(X)} instances for rule frequency computation...")
                for _h in getattr(self.logger, "handlers", []):
                    try:
                        _h.flush()
                    except Exception:
                        pass
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Computed frequencies for {len(rule_frequencies)} unique rules")
            if rule_frequencies:
                max_freq = max(rule_frequencies.values())
                self.logger.info(f"  Max frequency: {max_freq}, Mean frequency: {sum(rule_frequencies.values()) / len(rule_frequencies):.2f}")
        
        return dict(rule_frequencies)
    
    def explain_dataset(self, X: Union[np.ndarray, pd.DataFrame], 
                       predictions: Optional[np.ndarray] = None,
                       return_df: bool = True,
                       show_progress: bool = True,
                       n_jobs: int = -1) -> Union[pd.DataFrame, List[Dict]]:
        """
        Generate AXP explanations for a dataset.
        
        Args:
            X: Feature matrix (numpy array or DataFrame)
            predictions: Array of predicted classes (required)
            return_df: Whether to return DataFrame (default: True)
            show_progress: Whether to show progress bar (default: True)
            n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
            
        Returns:
            DataFrame or List[Dict] with explanations
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if predictions is None:
            raise ValueError("Please provide the predicted class labels for each instance.")
        
        # Compute rule frequencies if not already computed (for Set 5)
        if not self.rule_frequencies:
            self.rule_frequencies = self.compute_rule_frequencies(X, predictions)
        
        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
            # Windows spawn + many workers: long quiet periods between progress logs; cap workers.
            if os.name == "nt":
                n_jobs = max(1, min(n_jobs, 4))
        
        # Use parallel processing if n_jobs > 1 and dataset is large enough
        # For small datasets, sequential is faster due to overhead
        if n_jobs > 1 and len(X) > 5:
            return self._explain_dataset_parallel(X, predictions, return_df, show_progress, n_jobs)
        else:
            return self._explain_dataset_sequential(X, predictions, return_df, show_progress)
    
    def _explain_dataset_sequential(self, X: np.ndarray, predictions: np.ndarray,
                                    return_df: bool, show_progress: bool) -> Union[pd.DataFrame, List[Dict]]:
        """Sequential explanation generation."""
        results = []
        iterator = enumerate(zip(X, predictions, strict=True))
        # Disable tqdm on Windows to avoid stderr issues
        use_tqdm = show_progress and os.name != 'nt'
        if use_tqdm:
            try:
                iterator = tqdm(iterator, total=len(X), desc="Generating explanations")
            except Exception:
                use_tqdm = False
        
        if show_progress and not use_tqdm and hasattr(self, 'logger'):
            self.logger.info(f"Processing {len(X)} instances sequentially...")
        
        for i, (x, yhat) in iterator:
            import time
            inst_start = time.time()
            
            # Pass instance index for individual SHAP value lookup
            axp = self.explain_instance(x, predicted_class=yhat, instance_index=i)
            
            inst_duration = time.time() - inst_start
            if hasattr(self, 'logger') and i < 5:
                self.logger.info(f"Instance {i+1}/{len(X)}: {len(axp)} conditions, took {inst_duration:.4f}s")
            
            results.append({
                "index": i,
                "predicted_class": yhat,
                "axp": axp
            })
        
        return pd.DataFrame(results) if return_df else results
    
    def _explain_dataset_parallel(self, X: np.ndarray, predictions: np.ndarray,
                                 return_df: bool, show_progress: bool, n_jobs: int) -> Union[pd.DataFrame, List[Dict]]:
        """Parallel explanation generation using multiprocessing."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Using parallel processing with {n_jobs} workers for {len(X)} instances")
        
        # Prepare data for parallel processing
        # We need to pass the explainer state since it can't be pickled directly
        explainer_state = {
            'rule_clauses': self.rule_clauses,
            'rule_predictions': self.rule_predictions,
            'id_condition_map': self.id_condition_map,
            'feature_names': self.feature_names,
            '_class_rule_indices': self._class_rule_indices,
            'shap_importance_map': self.shap_importance_map,
            'rule_frequencies': self.rule_frequencies  # NEW: Include rule frequencies for Set 5
        }
        
        # Create tasks
        # Ensure x is 1D before converting to list
        tasks = []
        for i, (x, yhat) in enumerate(zip(X, predictions, strict=True)):
            # Ensure x is a numpy array
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            
            # Ensure x is 1D numpy array (flatten if needed)
            if x.ndim == 0:
                raise ValueError(f"Instance {i}: x is a scalar, expected 1D array")
            elif x.ndim > 1:
                x_1d = x.flatten()
            else:
                x_1d = x
            
            # Validate length - get expected feature count from explainer if available
            expected_features = getattr(explainer_state, 'n_features', None)
            if expected_features is None and hasattr(explainer_state, 'feature_names'):
                expected_features = len(explainer_state.feature_names) if explainer_state.feature_names else None
            
            # If we have expected features, validate match
            if expected_features and len(x_1d) != expected_features:
                raise ValueError(f"Instance {i}: x has {len(x_1d)} elements, expected {expected_features}. Shape: {x.shape}. "
                               f"First few values: {x_1d[:5] if len(x_1d) >= 5 else x_1d}")
            elif not expected_features:
                # No expected features - just do a basic sanity check
                # Allow any reasonable number of features (model might have been trained with different feature set)
                if len(x_1d) < 1:
                    raise ValueError(f"Instance {i}: x has only {len(x_1d)} elements, which is invalid. Shape: {x.shape}")
            
            x_list = x_1d.tolist()
            
            # Get instance-specific SHAP values (required - no fallback)
            try:
                # Try to get SHAP values for this instance index
                # shap_values_df should be indexed by instance index (from original data)
                if i in self.shap_values_df.index:
                    instance_shap_row = self.shap_values_df.loc[i]
                    # Convert to dict: feature_name -> SHAP value
                    instance_shap_values = instance_shap_row.to_dict()
                elif len(self.shap_values_df) > i:
                    # If index doesn't match, try positional access
                    instance_shap_row = self.shap_values_df.iloc[i]
                    instance_shap_values = instance_shap_row.to_dict()
                else:
                    raise IndexError(f"Instance index {i} out of range for SHAP values DataFrame (length: {len(self.shap_values_df)})")
            except (KeyError, IndexError) as e:
                # Log error - individual SHAP values are required
                error_msg = (
                    f"ERROR: Could not find individual SHAP values for instance {i}: {e}. "
                    f"Individual SHAP values per instance are REQUIRED for accurate rule filtering. "
                    f"SHAP values DataFrame has {len(self.shap_values_df)} rows, "
                    f"requested instance index: {i}. "
                    f"Please ensure SHAP values are loaded correctly from Step 7 (SHAP Analysis)."
                )
                if hasattr(self, 'logger'):
                    self.logger.error(error_msg)
                raise ValueError(error_msg) from e
            
            tasks.append((i, x_list, int(yhat), explainer_state, instance_shap_values))
        
        results = [None] * len(X)
        completed = 0
        _prog_every = 10 if os.name == "nt" else 50
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_explain_instance_worker, task): task[0] 
                for task in tasks
            }
            
            # Collect results with progress bar
            # Disable tqdm on Windows to avoid stderr issues
            use_tqdm = show_progress and os.name != 'nt'
            if use_tqdm:
                try:
                    iterator = tqdm(as_completed(future_to_idx), total=len(X), desc="Generating explanations")
                except Exception:
                    use_tqdm = False
            
            if not use_tqdm:
                iterator = as_completed(future_to_idx)
                if show_progress and hasattr(self, 'logger'):
                    self.logger.info(
                        f"Processing {len(X)} instances in parallel with {n_jobs} workers "
                        f"(progress every {_prog_every} completions; file log under logs/8_ffa_analysis/)..."
                    )
                    for _h in getattr(self.logger, "handlers", []):
                        try:
                            _h.flush()
                        except Exception:
                            pass
            
            for future in iterator:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    completed += 1
                    
                    if hasattr(self, 'logger') and (completed <= 5 or completed % _prog_every == 0):
                        self.logger.info(f"Completed {completed}/{len(X)} instances")
                        for _h in getattr(self.logger, "handlers", []):
                            try:
                                _h.flush()
                            except Exception:
                                pass
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Error processing instance {idx}: {e}")
                    results[idx] = {
                        "index": idx,
                        "predicted_class": predictions[idx],
                        "axp": []
                    }
        
        # Filter out None results (shouldn't happen, but safety check)
        results = [r for r in results if r is not None]
        
        return pd.DataFrame(results) if return_df else results
    
    def _literal_condition_holds(self, x: np.ndarray, lit: int) -> bool:
        """
        Check if a literal's condition holds for an instance.
        
        Handles binary features (0/1) and categorical variables correctly:
        - For binary features: XGBoost uses threshold=0.5 to split 0 vs 1
          * direction=0 (<=): feature <= 0.5 matches feature == 0
          * direction=1 (>): feature > 0.5 matches feature == 1
        - For categorical (one-hot encoded): same as binary
        - For continuous: standard threshold comparison
        
        Args:
            x: Feature vector
            lit: Literal ID
            
        Returns:
            Boolean indicating if condition holds
        """
        feat_idx, threshold, direction = self.id_condition_map[lit]
        value = x[feat_idx]
        
        # Standard comparison works correctly for both binary and continuous features
        # For binary features with threshold=0.5:
        # - value=0, direction=0 (<= 0.5): 0 <= 0.5 = True ✓
        # - value=1, direction=0 (<= 0.5): 1 <= 0.5 = False ✓
        # - value=0, direction=1 (> 0.5): 0 > 0.5 = False ✓
        # - value=1, direction=1 (> 0.5): 1 > 0.5 = True ✓
        # 
        # For edge cases with threshold=0.0 or threshold=1.0:
        # - threshold=0.0, direction=1 (> 0.0): matches value=1 only ✓
        # - threshold=1.0, direction=0 (<= 1.0): matches both 0 and 1 (always true)
        #   This is correct behavior - if threshold=1.0, the condition is always satisfied
        
        if direction == 0:
            return value <= threshold
        else:  # direction == 1
            return value > threshold
    
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
    
    def _enumerate_axps(self, clauses: List[List[int]]) -> List[List[int]]:
        """
        Enumerate all minimal hitting sets (AXPs) for a set of clauses.
        
        Args:
            clauses: List of clauses
            
        Returns:
            List of AXPs (lists of literals)
        """
        if Hitman is None:
            raise ImportError("pysat is required for AXP enumeration. Install with: pip install python-sat")
        
        h = Hitman(solver="m22", htype='sorted')
        for clause in clauses:
            h.hit(clause)
        return list(h.enumerate())
    
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

