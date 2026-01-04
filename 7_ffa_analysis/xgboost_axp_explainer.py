# xgboost_axp_explainer.py

import json
import itertools
from itertools import count
from collections import defaultdict, Counter
from pysat.examples.hitman import Hitman  # noqa: F401 (pysat is a valid package name)
import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import xgboost as xgb

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # Create a dummy class for type hints when xgboost is not available
    class xgb:
        class Booster:
            pass
        class XGBClassifier:
            pass
from functools import lru_cache
import matplotlib.pyplot as plt

# Import base class - add current directory to path if needed
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
from base_symbolic_explainer import BaseSymbolicExplainer, TREE_PATH_SCHEMA
try:
    import networkx as nx  # pyright: ignore[reportMissingImports]
except ImportError:
    # Fallback if networkx is not available (optional dependency)
    nx = None
try:
    import seaborn as sns
except ImportError:
    # Fallback if seaborn is not available
    sns = None
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available - create a no-op wrapper
    def tqdm(iterable, *args, **kwargs):
        return iterable
    tqdm = tqdm  # Make it available in module scope


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
            model_path: Path to the XGBoost model JSON file in S3
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


class XGBoostSymbolicExplainer(BaseSymbolicExplainer):
    """XGBoost model explainer using unified symbolic AXP schema."""
  
    def __init__(self, path_config: PathConfig):
        super().__init__(path_config)
        self.path_config = path_config
        self.tree_rules = None
        
        # Override logging setup to use file if path_config provides output_dir
        if path_config and hasattr(path_config, 'output_dir'):
            log_file = os.path.join(path_config.output_dir, 'axp_analysis.log')
            self.setup_logging(log_file=log_file)

    def fit_from_model_json(self, model_json):
        """Parse XGBoost model JSON and build symbolic CNF clauses for FFA."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
            
        self.model_json = model_json
        self.rule_clauses.clear()
        self.rule_predictions.clear()
        self.condition_id_map.clear()
        self.id_condition_map.clear()
    
        # Extract feature names from model JSON unless they were already provided.
        if getattr(self, "feature_names", None):
            # Feature names were set externally (e.g., from the DataFrame columns);
            # keep them as-is.
            pass
        elif "feature_names" in model_json:
            # Feature names are provided as a list
            self.feature_names = {
                i: name for i, name in enumerate(model_json["feature_names"])
            }
        elif "feature_names" not in model_json and "trees" in model_json:
            # Infer feature names from tree structure
            # XGBoost uses f0, f1, f2, etc. as default feature names
            max_feature_idx = 0
            for tree in model_json.get("trees", []):
                max_feature_idx = max(max_feature_idx, self._get_max_feature_index(tree))
            
            self.feature_names = {
                i: f"f{i}" for i in range(max_feature_idx + 1)
            }
        else:
            raise ValueError("Could not determine feature names from model JSON")
    
        # Process each tree in the model
        if "trees" in model_json:
            trees_processed = 0
            total_trees = len(model_json["trees"])
            self.logger.info(f"Processing {total_trees} trees from model JSON")
            
            for tree_idx, tree in enumerate(model_json["trees"]):
                if isinstance(tree, dict) and "tree_dump" in tree:
                    # Tree is wrapped in a dict with tree_dump
                    tree_dump = tree["tree_dump"]
                    self.logger.debug(f"Tree {tree_idx}: Found tree_dump (length: {len(tree_dump)} chars)")
                elif isinstance(tree, str):
                    # Tree is a string dump
                    tree_dump = tree
                    self.logger.debug(f"Tree {tree_idx}: Tree is string dump (length: {len(tree_dump)} chars)")
                else:
                    # Tree is already parsed
                    self.logger.debug(f"Tree {tree_idx}: Tree is already parsed, traversing directly")
                    initial_rule_count = len(self.rule_clauses)
                    self._traverse_xgboost_tree(tree, conditions=[])
                    rules_added = len(self.rule_clauses) - initial_rule_count
                    if rules_added > 0:
                        trees_processed += 1
                        self.logger.debug(f"Tree {tree_idx}: Added {rules_added} rules from pre-parsed tree")
                    continue
                
                # Parse tree dump string into structured format
                initial_rule_count = len(self.rule_clauses)
                parsed_tree = self._parse_xgboost_tree_dump(tree_dump)
                
                if parsed_tree:
                    # Check if tree has any nodes
                    has_nodes = bool(parsed_tree)
                    has_leaf = "leaf_value" in parsed_tree or "leaf" in parsed_tree
                    has_feature = "feature" in parsed_tree
                    
                    self.logger.debug(f"Tree {tree_idx}: Parsed successfully - has_nodes={has_nodes}, "
                                    f"has_leaf={has_leaf}, has_feature={has_feature}")
                    
                    if has_nodes:
                        # Try dataframe approach first (easier to debug)
                        try:
                            df_paths = self._explode_tree_to_dataframe(parsed_tree, tree_idx)
                            if len(df_paths) > 0:
                                self.logger.debug(f"Tree {tree_idx}: Exploded to {len(df_paths)} path rows, "
                                                f"{df_paths['path_idx'].nunique()} unique paths")
                                self._create_rules_from_dataframe(df_paths)
                                rules_added = len(self.rule_clauses) - initial_rule_count
                                
                                if rules_added > 0:
                                    trees_processed += 1
                                    self.logger.debug(f"Tree {tree_idx}: Added {rules_added} rules via dataframe method")
                                else:
                                    self.logger.warning(f"Tree {tree_idx}: DataFrame method created no rules, "
                                                      f"falling back to traversal")
                                    # Fallback to traversal method
                                    self._traverse_xgboost_tree(parsed_tree, conditions=[])
                                    rules_added = len(self.rule_clauses) - initial_rule_count
                                    if rules_added > 0:
                                        trees_processed += 1
                            else:
                                self.logger.warning(f"Tree {tree_idx}: DataFrame explosion returned empty, "
                                                  f"falling back to traversal")
                                self._traverse_xgboost_tree(parsed_tree, conditions=[])
                                rules_added = len(self.rule_clauses) - initial_rule_count
                                if rules_added > 0:
                                    trees_processed += 1
                        except Exception as e:
                            self.logger.warning(f"Tree {tree_idx}: DataFrame method failed: {e}, "
                                              f"falling back to traversal")
                            import traceback
                            self.logger.debug(traceback.format_exc())
                            # Fallback to traversal method
                            self._traverse_xgboost_tree(parsed_tree, conditions=[])
                            rules_added = len(self.rule_clauses) - initial_rule_count
                            if rules_added > 0:
                                trees_processed += 1
                    else:
                        self.logger.warning(f"Tree {tree_idx}: Parsed tree is empty")
                else:
                    self.logger.warning(f"Tree {tree_idx}: Failed to parse tree dump")
                
                # Print progress for first few trees and every 50th tree
                if tree_idx < 3 or (tree_idx + 1) % 50 == 0:
                    self.logger.info(f"Tree {tree_idx+1}/{total_trees}: "
                                   f"Total rules so far: {len(self.rule_clauses)}")
            
            self.logger.info(f"Finished processing: {trees_processed}/{total_trees} trees created rules, "
                           f"Total rules: {len(self.rule_clauses)}")
            
            if len(self.rule_clauses) == 0:
                self.logger.error("No rules were created from any trees! Check tree parsing and traversal logic.")
        else:
            raise ValueError("Model JSON missing 'trees' field. Ensure model was exported correctly.")

    def _get_max_feature_index(self, tree: Dict) -> int:
        """Get maximum feature index from a tree structure."""
        max_idx = -1
        if isinstance(tree, dict):
            if "split" in tree:
                split = tree["split"]
                if "feature" in split:
                    max_idx = max(max_idx, split["feature"])
                elif "float_feature_index" in split:
                    max_idx = max(max_idx, split["float_feature_index"])
            if "children" in tree:
                for child in tree["children"]:
                    max_idx = max(max_idx, self._get_max_feature_index(child))
            if "left_child" in tree:
                max_idx = max(max_idx, self._get_max_feature_index(tree["left_child"]))
            if "right_child" in tree:
                max_idx = max(max_idx, self._get_max_feature_index(tree["right_child"]))
        return max_idx

    def _parse_xgboost_tree_dump(self, tree_dump: str) -> Dict:
        """Parse XGBoost tree dump string into structured format."""
        # XGBoost tree dump format can use either:
        # 1. Feature indices: "0:[f0<0.5] yes=1,no=2,missing=1"
        # 2. Feature names: "0:[icd_code_itemset_2_match<1] yes=1,no=2,missing=1"
        
        lines = tree_dump.strip().split('\n')
        tree = {}
        node_map = {}
        nodes_parsed = 0
        leaves_parsed = 0
        
        # Create reverse mapping: feature_name -> feature_index
        feature_name_to_idx = {name: idx for idx, name in self.feature_names.items()}
        self.logger.debug(f"Feature name mapping: {len(feature_name_to_idx)} features available")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for leaf node first: "11:leaf=0.000641238759,cover=54.7450752"
            if ':' in line and 'leaf=' in line:
                parts = line.split(':', 1)
                node_id = int(parts[0])
                leaf_part = parts[1]
                # Extract leaf value: "leaf=0.000641238759,cover=54.7450752"
                leaf_value = float(leaf_part.split('leaf=')[1].split(',')[0])
                node_map[node_id] = {'leaf': leaf_value}
                leaves_parsed += 1
                continue
                
            # Parse node line: "0:[f0<0.5] yes=1,no=2,missing=1" or "0:[feature_name<0.5] yes=1,no=2,missing=1"
            if ':' in line and '[' in line:
                parts = line.split(':', 1)
                node_id = int(parts[0])
                
                # Extract condition
                condition_part = parts[1].split(']')[0] + ']'
                if '<' in condition_part:
                    # Format: [f0<0.5] or [feature_name<0.5]
                    feat_part, threshold_part = condition_part[1:-1].split('<')
                    
                    # Try to parse as feature index (f0, f1, etc.) first
                    if feat_part.startswith('f') and feat_part[1:].isdigit():
                        feature_idx = int(feat_part[1:])  # Remove 'f' prefix
                    elif feat_part in feature_name_to_idx:
                        # Feature name found in mapping
                        feature_idx = feature_name_to_idx[feat_part]
                    else:
                        # Try to find feature by name (case-insensitive partial match)
                        # This handles cases where feature names might have slight variations
                        matching_idx = None
                        for name, idx in feature_name_to_idx.items():
                            if name == feat_part or name.endswith(feat_part) or feat_part in name:
                                matching_idx = idx
                                break
                        if matching_idx is not None:
                            feature_idx = matching_idx
                        else:
                            # Fallback: use feature name as-is and let _traverse handle it
                            # We'll need to store the feature name and map it later
                            feature_idx = feat_part  # Store as string for now
                            self.logger.warning(f"Feature '{feat_part}' not found in feature_names. Using as-is.")
                    
                    threshold = float(threshold_part)
                    direction = 0  # <=
                elif '>' in condition_part:
                    # Format: [f0>0.5] or [feature_name>0.5]
                    feat_part, threshold_part = condition_part[1:-1].split('>')
                    
                    # Try to parse as feature index (f0, f1, etc.) first
                    if feat_part.startswith('f') and feat_part[1:].isdigit():
                        feature_idx = int(feat_part[1:])  # Remove 'f' prefix
                    elif feat_part in feature_name_to_idx:
                        # Feature name found in mapping
                        feature_idx = feature_name_to_idx[feat_part]
                    else:
                        # Try to find feature by name
                        matching_idx = None
                        for name, idx in feature_name_to_idx.items():
                            if name == feat_part or name.endswith(feat_part) or feat_part in name:
                                matching_idx = idx
                                break
                        if matching_idx is not None:
                            feature_idx = matching_idx
                        else:
                            feature_idx = feat_part  # Store as string for now
                            self.logger.warning(f"Feature '{feat_part}' not found in feature_names. Using as-is.")
                    
                    threshold = float(threshold_part)
                    direction = 1  # >
                else:
                    # This shouldn't happen if we check for leaf nodes above
                    # But keep as fallback
                    continue
                
                # Extract children
                children_part = parts[1].split(']')[1]
                yes_id = int(children_part.split('yes=')[1].split(',')[0])
                no_id = int(children_part.split('no=')[1].split(',')[0])
                
                node_map[node_id] = {
                    'feature': feature_idx,
                    'threshold': threshold,
                    'left_child': yes_id,
                    'right_child': no_id
                }
                nodes_parsed += 1
                
                # Log feature name resolution for first few nodes
                if nodes_parsed <= 3:
                    if isinstance(feature_idx, str):
                        self.logger.debug(f"Node {node_id}: Feature '{feature_idx}' stored as string (will resolve during traversal)")
                    else:
                        feat_name = self.feature_names.get(feature_idx, f"f{feature_idx}")
                        self.logger.debug(f"Node {node_id}: Feature index {feature_idx} ({feat_name}), threshold={threshold}")
        
        self.logger.debug(f"Parsed {nodes_parsed} internal nodes and {leaves_parsed} leaf nodes from tree dump")
        
        # Build tree structure starting from root (node 0)
        def build_node(node_id: int, depth: int = 0) -> Dict:
            if node_id in node_map:
                node = node_map[node_id]
                if 'leaf' in node:
                    if depth < 3:
                        self.logger.debug(f"Building leaf node {node_id} at depth {depth}, value={node['leaf']}")
                    return {'leaf_value': node['leaf']}
                else:
                    if depth < 3:
                        feat_info = node.get('feature', 'unknown')
                        self.logger.debug(f"Building internal node {node_id} at depth {depth}, "
                                        f"feature={feat_info}, threshold={node.get('threshold')}")
                    
                    left_child = build_node(node['left_child'], depth + 1)
                    right_child = build_node(node['right_child'], depth + 1)
                    
                    result = {
                        'feature': node['feature'],
                        'threshold': node['threshold'],
                        'left_child': left_child,
                        'right_child': right_child
                    }
                    
                    # Check if children are empty (shouldn't happen if parsing worked)
                    if not left_child and not right_child:
                        self.logger.warning(f"Node {node_id} has empty left and right children")
                    
                    return result
            else:
                if depth < 3:
                    self.logger.debug(f"Node {node_id} not found in node_map at depth {depth}")
                return {}
        
        root_tree = build_node(0) if 0 in node_map else {}
        if not root_tree:
            self.logger.error("Failed to build tree: root node 0 not found in node_map")
            self.logger.debug(f"Available node IDs: {sorted(list(node_map.keys()))[:10]}...")
        
        return root_tree

    def _explode_tree_to_dataframe(self, tree: Dict, tree_idx: int = 0) -> pd.DataFrame:
        """
        Explode a tree structure into a DataFrame where each row represents a path from root to leaf.
        This makes it easier to debug and create rules.
        
        Returns:
            DataFrame with columns: tree_idx, path_depth, feature_idx, threshold, direction, leaf_value, prediction
        """
        paths = []
        
        def traverse_path(node: Dict, path: List[Dict] = None, depth: int = 0):
            if path is None:
                path = []
            
            if "leaf_value" in node or "leaf" in node:
                # Reached a leaf - save the complete path
                leaf_value = node.get("leaf_value", node.get("leaf", 0))
                pred = 1 if leaf_value > 0 else 0
                
                # Create a row for each step in the path
                for step_idx, step in enumerate(path):
                    paths.append({
                        'tree_idx': tree_idx,
                        'path_idx': len(paths),  # Unique path identifier
                        'step_in_path': step_idx,
                        'feature_idx': step.get('feature_idx'),
                        'feature_name': step.get('feature_name'),
                        'threshold': step.get('threshold'),
                        'direction': step.get('direction'),  # 0 for <=, 1 for >
                        'depth': step.get('depth'),
                        'leaf_value': leaf_value,
                        'prediction': pred,
                        'path_length': len(path)
                    })
                
                # Also add a summary row for the leaf
                paths.append({
                    'tree_idx': tree_idx,
                    'path_idx': len(paths) - len(path),  # Same path_idx as the steps
                    'step_in_path': len(path),  # Leaf is the last step
                    'feature_idx': None,
                    'feature_name': 'LEAF',
                    'threshold': None,
                    'direction': None,
                    'depth': depth,
                    'leaf_value': leaf_value,
                    'prediction': pred,
                    'path_length': len(path)
                })
                return
            
            # Internal node - add to path and recurse
            if "feature" in node and "threshold" in node:
                feature_ref = node["feature"]
                threshold = node["threshold"]
                
                # Resolve feature name to index
                if isinstance(feature_ref, str):
                    feature_index = None
                    feature_name = feature_ref
                    for idx, name in self.feature_names.items():
                        if name == feature_ref:
                            feature_index = idx
                            break
                    if feature_index is None:
                        self.logger.warning(f"Feature '{feature_ref}' not found in feature_names during explode")
                        return
                else:
                    feature_index = feature_ref
                    feature_name = self.feature_names.get(feature_index, f"f{feature_index}")
                
                # Traverse left child (direction 0: <=)
                if "left_child" in node:
                    new_path = path + [{
                        'feature_idx': feature_index,
                        'feature_name': feature_name,
                        'threshold': threshold,
                        'direction': 0,
                        'depth': depth
                    }]
                    traverse_path(node["left_child"], new_path, depth + 1)
                
                # Traverse right child (direction 1: >)
                if "right_child" in node:
                    new_path = path + [{
                        'feature_idx': feature_index,
                        'feature_name': feature_name,
                        'threshold': threshold,
                        'direction': 1,
                        'depth': depth
                    }]
                    traverse_path(node["right_child"], new_path, depth + 1)
        
        traverse_path(tree)
        return pd.DataFrame(paths)

    # _create_rules_from_dataframe is inherited from BaseSymbolicExplainer

    def _traverse_xgboost_tree(self, node: Dict, conditions: List[Tuple] = None, depth: int = 0):
        """Recursively parse an XGBoost tree and extract CNF rules."""
        if conditions is None:
            conditions = []
        
        # Limit logging depth to avoid spam
        log_this_node = depth < 3 or len(self.rule_clauses) < 10
        
        if "leaf_value" in node or "leaf" in node:
            # Base case: reached a leaf
            leaf_value = node.get("leaf_value", node.get("leaf", 0))
            pred = 1 if leaf_value > 0 else 0
            
            if log_this_node:
                self.logger.debug(f"Leaf at depth {depth}: value={leaf_value}, pred={pred}, "
                                f"conditions={len(conditions)}")
            
            try:
                clause = [self._get_condition_literal(f, t, d) for (f, t, d) in conditions]
                self.rule_clauses.append(clause)
                self.rule_predictions.append(pred)
                
                if log_this_node:
                    self.logger.debug(f"Created rule {len(self.rule_clauses)-1}: {len(clause)} conditions, "
                                    f"prediction={pred}")
            except Exception as e:
                self.logger.error(f"Error creating rule from leaf: {e}, conditions={conditions}")
                import traceback
                self.logger.error(traceback.format_exc())
            return
    
        # Recursive case: internal node with split
        if "feature" in node and "threshold" in node:
            feature_ref = node["feature"]  # Can be int (index) or str (feature name)
            threshold = node["threshold"]
            
            if log_this_node:
                self.logger.debug(f"Internal node at depth {depth}: feature_ref={feature_ref} "
                                f"(type={type(feature_ref).__name__}), threshold={threshold}")
            
            # Convert feature name to index if needed
            if isinstance(feature_ref, str):
                # Feature is a name, need to find its index
                feature_index = None
                for idx, name in self.feature_names.items():
                    if name == feature_ref:
                        feature_index = idx
                        break
                if feature_index is None:
                    self.logger.warning(f"Feature '{feature_ref}' not found in feature_names. "
                                      f"Available features: {list(self.feature_names.values())[:5]}... "
                                      f"Skipping split.")
                    return
                
                if log_this_node:
                    self.logger.debug(f"Resolved feature name '{feature_ref}' to index {feature_index}")
            else:
                feature_index = feature_ref
                if log_this_node:
                    feat_name = self.feature_names.get(feature_index, f"f{feature_index}")
                    self.logger.debug(f"Using feature index {feature_index} ({feat_name})")
            
            # Left child: condition is "feature <= threshold" => direction=0
            if "left_child" in node:
                left_child = node["left_child"]
                if log_this_node:
                    self.logger.debug(f"Traversing left child (depth {depth+1})")
                self._traverse_xgboost_tree(
                    left_child,
                    conditions + [(feature_index, threshold, 0)],
                    depth + 1
                )
            else:
                if log_this_node:
                    self.logger.debug(f"No left_child found in node at depth {depth}")
            
            # Right child: condition is "feature > threshold" => direction=1
            if "right_child" in node:
                right_child = node["right_child"]
                if log_this_node:
                    self.logger.debug(f"Traversing right child (depth {depth+1})")
                self._traverse_xgboost_tree(
                    right_child,
                    conditions + [(feature_index, threshold, 1)],
                    depth + 1
                )
            else:
                if log_this_node:
                    self.logger.debug(f"No right_child found in node at depth {depth}")
        elif "split" in node:
            # Alternative format with split dict
            split = node["split"]
            if log_this_node:
                self.logger.debug(f"Node at depth {depth} uses 'split' format: {list(split.keys())}")
            
            if "feature" in split:
                feature_index = split["feature"]
            elif "float_feature_index" in split:
                feature_index = split["float_feature_index"]
            else:
                self.logger.warning(f"Split dict at depth {depth} has no 'feature' or 'float_feature_index'")
                return
                
            threshold = split.get("threshold", split.get("border", 0))
            
            if "left_child" in node:
                self._traverse_xgboost_tree(
                    node["left_child"],
                    conditions + [(feature_index, threshold, 0)],
                    depth + 1
                )
            
            if "right_child" in node:
                self._traverse_xgboost_tree(
                    node["right_child"],
                    conditions + [(feature_index, threshold, 1)],
                    depth + 1
                )
        else:
            # Node doesn't match expected formats
            if log_this_node:
                self.logger.warning(f"Node at depth {depth} doesn't match expected formats. "
                                  f"Keys: {list(node.keys())}")

    # All common methods (_satisfied_rules, _compute_axp, explain_literals, 
    # explain_instance, _literal_to_text, explain_dataset) are inherited from BaseSymbolicExplainer

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
            axps = self._batch_compute_axps(X_class.values if isinstance(X_class, pd.DataFrame) else X_class, label)
            
            # Calculate metrics
            metrics = self._compute_feature_metrics(axps)
            results[label] = pd.DataFrame(metrics)
            
        return results

    def load_model(self):
        """Load model from configured path (JSON format)."""
        self.model_json = self.path_config.read_json(self.path_config.model_path)
        self.fit_from_model_json(self.model_json)

    def load_model_from_booster(self, booster: "xgb.Booster"):
        """Load model from XGBoost Booster object."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
            
        # Get feature names
        feature_names = booster.feature_names if hasattr(booster, 'feature_names') else None
        
        # Get tree dumps
        tree_dumps = booster.get_dump(with_stats=True)
        
        # Build model JSON structure
        model_json = {
            "feature_names": feature_names or [f"f{i}" for i in range(booster.num_feature())],
            "trees": [{"tree_dump": dump} for dump in tree_dumps]
        }
        
        self.fit_from_model_json(model_json)

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

    @lru_cache(maxsize=128)  # noqa: B019
    def _compute_axp_cached(self, rule_ids_tuple: Tuple[int, ...]) -> List[int]:
        """Cached version of AXP computation for repeated patterns.
        
        Note: lru_cache on methods can retain self references, but this is acceptable
        here as the cache is bounded (maxsize=128) and improves performance for
        repeated AXP computations.
        """
        return self._compute_axp(list(rule_ids_tuple))

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
            X_class = X[mask] if isinstance(X, np.ndarray) else X.loc[mask]
            explained = self.explain_dataset(X_class.values if isinstance(X_class, pd.DataFrame) else X_class, 
                                          predictions=predictions[mask])
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
                                     xgboost_model: Union["xgb.Booster", "xgb.XGBClassifier"],
                                     X: pd.DataFrame) -> pd.DataFrame:
        """
        Compare symbolic explanations with native XGBoost feature importance.
        
        Args:
            xgboost_model: Trained XGBoost model (Booster or XGBClassifier)
            X: Feature matrix
            
        Returns:
            DataFrame comparing different importance metrics
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
            
        # Get native feature importance
        if isinstance(xgboost_model, xgb.Booster):
            booster = xgboost_model
        elif hasattr(xgboost_model, 'get_booster'):
            booster = xgboost_model.get_booster()
        else:
            raise ValueError("Model must be XGBoost Booster or XGBClassifier")
        
        # Get feature importance (weight, gain, or cover)
        importance_dict = booster.get_score(importance_type='weight')
        
        native_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'native_importance': list(importance_dict.values())
        })
        
        # Get symbolic importance
        predictions = booster.predict(xgb.DMatrix(X)) > 0.5
        symbolic_metrics = self.compute_feature_attribution(X, predictions.astype(int))
        
        # Combine metrics for both classes
        symbolic_importance = pd.DataFrame()
        for class_label, metrics in symbolic_metrics.items():
            df = pd.DataFrame(metrics)
            df['class'] = class_label
            symbolic_importance = pd.concat([symbolic_importance, df])
        
        # Compute correlation
        merged = pd.merge(native_importance, 
                         symbolic_importance.groupby('feature')['relative_importance'].mean().reset_index(),
                         on='feature', how='outer')
        merged = merged.fillna(0)
        
        if len(merged) > 1:
            correlation = np.corrcoef(merged['native_importance'], 
                                    merged['relative_importance'])[0,1]
            merged['correlation'] = correlation
        
        return merged

    def export_results(self, results: Dict[str, Any], output_dir: str, formats: Optional[List[str]] = None) -> None:
        """Export analysis results in multiple formats to S3."""
        if formats is None:
            formats = ['parquet', 'json']
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


# Reuse visualization classes from CatBoost explainer
# (These are model-agnostic and work with any explainer)
from catboost_axp_explainer import FeatureVisualization  # noqa: E402


def analyze_model(path_config: PathConfig, model: Union["xgb.Booster", "xgb.XGBClassifier"], 
                  analysis_config: AnalysisConfig = None) -> Dict[str, Any]:
    """
    Run complete model analysis using configured paths.
    
    Args:
        path_config: PathConfig instance with all required paths
        model: Trained XGBoost model (Booster or XGBClassifier)
        analysis_config: Configuration for analysis parameters
        
    Returns:
        Dictionary containing analysis results
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
        
    if analysis_config is None:
        analysis_config = AnalysisConfig()
    
    # Initialize explainer and logging
    explainer = XGBoostSymbolicExplainer(path_config)
    
    # Load model
    if isinstance(model, xgb.Booster) or hasattr(model, 'get_booster'):
        explainer.load_model_from_booster(model)
    else:
        explainer.load_model()
    
    # Load and validate data
    test_data = path_config.read_parquet(path_config.test_data_path)
    X_test = test_data.iloc[:, :-1]
    
    # Get predictions
    if isinstance(model, xgb.Booster):
        booster = model
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
    else:
        raise ValueError("Model must be XGBoost Booster or XGBClassifier")
    
    y_pred = (booster.predict(xgb.DMatrix(X_test)) > 0.5).astype(int)
    explainer.validate_input_data(X_test, y_pred)
    
    try:
        results = {}
        for class_label in [0, 1]:
            explainer.logger.info(f"Processing class {class_label}")
            
            # Filter data for class
            mask = (y_pred == class_label)
            X_class = X_test[mask]
            
            # Generate explanations with progress bar
            df_axps = explainer.explain_dataset(X_class, predictions=y_pred[mask], show_progress=True)
            
            # Compute metrics
            metrics = explainer._compute_feature_metrics(df_axps.to_dict('records'))
            
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
        
        # Compare with native XGBoost importance
        native_comparison = explainer.compare_with_native_importance(model, X_test)
        
        # Add additional results
        results['validation'] = validation_results
        results['native_comparison'] = native_comparison
        
        explainer.logger.info("Analysis completed successfully")
        
    except Exception as e:
        explainer.logger.error(f"Error during analysis: {str(e)}")
        raise
    
    return results

