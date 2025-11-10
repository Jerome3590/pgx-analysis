import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from ast import literal_eval
import boto3
from catboost import CatBoostClassifier
from pysat.examples.hitman import Hitman
from pysat.card import EncType
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss, 
    accuracy_score, 
    log_loss, 
    f1_score, 
    precision_score, 
    recall_score, 
    average_precision_score, 
    confusion_matrix
)

# Core data structures
import collections
import collections.abc 
collections.Mapping = collections.abc.Mapping
collections.Sequence = collections.abc.Sequence

# System monitoring
import psutil


def validate_explainer_structure(explainer):
    """Validate the structure of the explainer before processing.
    
    Args:
        explainer: CatBoostSymbolicExplainer instance
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("\n=== Validating Explainer Structure ===")
    
    # 1. Check feature names
    print("\n1. Feature Names Validation:")
    print(f"Total feature names: {len(explainer.feature_names)}")
    print("First 5 feature names:", explainer.feature_names[:5])
    
    # 2. Check rule structure
    print("\n2. Rule Structure Validation:")
    print(f"Total rules: {len(explainer.rule_clauses)}")
    print(f"Total rule predictions: {len(explainer.rule_predictions)}")
    
    # 3. Check condition mapping
    print("\n3. Condition Mapping Validation:")
    print(f"Total conditions in mapping: {len(explainer.id_condition_map)}")
    
    # 4. Validate feature indices in rules
    print("\n4. Feature Index Validation:")
    max_feat_idx = -1
    invalid_indices = set()
    
    for i, clause in enumerate(explainer.rule_clauses):
        for lit in clause:
            try:
                feat_idx, _, _ = explainer.id_condition_map[lit]
                max_feat_idx = max(max_feat_idx, feat_idx)
                if feat_idx >= len(explainer.feature_names):
                    invalid_indices.add(feat_idx)
            except KeyError:
                print(f"Warning: Literal {lit} not found in condition mapping")
    
    print(f"Maximum feature index found: {max_feat_idx}")
    if invalid_indices:
        print(f"Found {len(invalid_indices)} invalid feature indices: {sorted(invalid_indices)}")
        print("This indicates a mismatch between rule conditions and feature names")
        return False
    
    # 5. Validate rule predictions
    print("\n5. Rule Prediction Validation:")
    unique_predictions = set(explainer.rule_predictions)
    print(f"Unique prediction classes: {unique_predictions}")
    
    # 6. Check for empty rules
    empty_rules = [i for i, clause in enumerate(explainer.rule_clauses) if not clause]
    if empty_rules:
        print(f"Warning: Found {len(empty_rules)} empty rules at indices: {empty_rules}")
        return False
    
    print("\n✓ All validation checks passed!")
    return True


def analyze_ctr_hash_maps(ctr_data):
    """Analyze the hash map structure in CTR data.
    
    Args:
        ctr_data: The CTR data from the model JSON
        
    Returns:
        dict: Analysis results of the hash maps
    """
    print("\n=== Analyzing CTR Hash Maps ===")
    
    analysis = {
        'total_entries': 0,
        'hash_map_patterns': defaultdict(int),
        'feature_stats': defaultdict(lambda: {
            'total_values': 0,
            'unique_hashes': set(),
            'value_counts': defaultdict(int)
        })
    }
    
    for ctr_key, ctr_value in ctr_data.items():
        try:
            # Parse the CTR key
            ctr_info = json.loads(ctr_key)
            if 'identifier' not in ctr_info:
                continue
                
            # Get feature index
            for identifier in ctr_info['identifier']:
                if 'cat_feature_index' not in identifier:
                    continue
                    
                feat_idx = identifier['cat_feature_index']
                
                # Analyze hash map if present
                if isinstance(ctr_value, dict) and 'hash_map' in ctr_value:
                    hash_map = ctr_value['hash_map']
                    analysis['total_entries'] += len(hash_map)
                    
                    # Update feature stats
                    feat_stats = analysis['feature_stats'][feat_idx]
                    feat_stats['total_values'] += len(hash_map)
                    
                    # Analyze hash-value pairs
                    for i in range(0, len(hash_map), 2):
                        if i + 1 < len(hash_map):
                            hash_val = hash_map[i]
                            count = hash_map[i + 1]
                            feat_stats['unique_hashes'].add(hash_val)
                            feat_stats['value_counts'][count] += 1
                    
                    # Record pattern
                    pattern = f"hash_map_size_{len(hash_map)}"
                    analysis['hash_map_patterns'][pattern] += 1
                    
        except json.JSONDecodeError:
            continue
    
    # Print analysis results
    print(f"\nTotal hash map entries across all features: {analysis['total_entries']}")
    
    print("\nHash Map Size Patterns:")
    for pattern, count in sorted(analysis['hash_map_patterns'].items()):
        print(f"- {pattern}: {count} features")
    
    print("\nFeature Statistics:")
    for feat_idx, stats in analysis['feature_stats'].items():
        print(f"\nFeature {feat_idx}:")
        print(f"- Total values: {stats['total_values']}")
        print(f"- Unique hashes: {len(stats['unique_hashes'])}")
        
        # Show value count distribution
        print("- Value count distribution:")
        for count, freq in sorted(stats['value_counts'].items())[:5]:
            print(f"  * Count {count}: {freq} occurrences")
        if len(stats['value_counts']) > 5:
            print(f"  * ... and {len(stats['value_counts']) - 5} more unique counts")
    
    return analysis


def print_json_key_structure(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}")
            print_json_key_structure(v, prefix + "  ")
    elif isinstance(d, list):
        for i, item in enumerate(d[:1]):  # just preview 1st item of lists
            print_json_key_structure(item, prefix + "  ")


def validate_model_json_parsing(model_json):
    """Validate the model JSON structure and feature indices parsing.
    
    Args:
        model_json: The loaded model JSON data
        
    Returns:
        dict: Validation results including feature indices and structure
    """
    print("\n=== Validating Model JSON Structure ===")
    
    validation_results = {
        'has_features': False,
        'feature_count': 0,
        'feature_indices': set(),
        'ctr_data': {},
        'parsing_errors': []
    }
    
    try:
        # 1. Check basic structure
        print("\n1. Basic Structure Validation:")
        if 'ctr_data' not in model_json:
            validation_results['parsing_errors'].append("Missing 'ctr_data' in model JSON")
            return validation_results
            
        ctr_data = model_json['ctr_data']
        print(f"Found CTR data with {len(ctr_data)} entries")
        
        # 2. Analyze CTR data structure
        print("\n2. CTR Data Analysis:")
        for ctr_key, ctr_value in ctr_data.items():
            try:
                # Parse the CTR key which contains feature information
                ctr_info = json.loads(ctr_key)
                if 'identifier' in ctr_info:
                    for identifier in ctr_info['identifier']:
                        if 'cat_feature_index' in identifier:
                            feat_idx = identifier['cat_feature_index']
                            validation_results['feature_indices'].add(feat_idx)
                            validation_results['ctr_data'][feat_idx] = {
                                'type': ctr_info.get('type', 'Unknown'),
                                'hash_map_size': len(ctr_value.get('hash_map', [])) if isinstance(ctr_value, dict) else 0
                            }
            except json.JSONDecodeError:
                print(f"Warning: Could not parse CTR key: {ctr_key}")
                continue
        
        # 3. Validate feature indices
        print("\n3. Feature Index Validation:")
        if validation_results['feature_indices']:
            print(f"Total unique feature indices found: {len(validation_results['feature_indices'])}")
            print(f"Feature indices range: {min(validation_results['feature_indices'])} to {max(validation_results['feature_indices'])}")
            
            # Print CTR data summary
            print("\nCTR Data Summary:")
            for feat_idx, info in validation_results['ctr_data'].items():
                print(f"Feature {feat_idx}:")
                print(f"- Type: {info['type']}")
                print(f"- Hash map size: {info['hash_map_size']}")
        else:
            print("No feature indices found in CTR data")
            validation_results['parsing_errors'].append("No feature indices found in CTR data")
        
        # 4. Check for model info
        print("\n4. Model Info Check:")
        if 'model_info' in model_json:
            model_info = model_json['model_info']
            validation_results['has_features'] = True
            if 'feature_names' in model_info:
                validation_results['feature_count'] = len(model_info['feature_names'])
                print(f"Found {validation_results['feature_count']} feature names in model info")
                print("First 5 feature names:", model_info['feature_names'][:5])
        else:
            print("No model_info found in JSON")
            validation_results['parsing_errors'].append("Missing 'model_info' in model JSON")
        
        # 5. Analyze hash maps
        print("\n5. Hash Map Analysis:")
        hash_map_analysis = analyze_ctr_hash_maps(ctr_data)
        validation_results['hash_map_analysis'] = hash_map_analysis
        
    except Exception as e:
        validation_results['parsing_errors'].append(f"Error during validation: {str(e)}")
        print(f"Error during validation: {str(e)}")
    
    return validation_results


def extract_tree_structure(model_path):
    """Extract tree structure from CatBoost model."""
    print("\n=== Extracting Tree Structure ===")
    
    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Get tree count
    tree_count = model.tree_count_
    print(f"Found {tree_count} trees in model")
    
    # Load tree rules JSON - use the already downloaded file
    tree_rules_path = os.path.join(os.path.dirname(model_path), 'tree_rules.json')
    if not os.path.exists(tree_rules_path):
        print(f"❌ Error: tree_rules.json not found at {tree_rules_path}")
        raise FileNotFoundError(f"tree_rules.json not found at {tree_rules_path}")
    
    with open(tree_rules_path, 'r') as f:
        tree_rules = json.load(f)
    
    print("\n=== Tree Rules Structure ===")
    print("Top-level keys:", list(tree_rules.keys()))
    
    # Examine each top-level key
    for key in tree_rules.keys():
        print(f"\n{key} structure:")
        if isinstance(tree_rules[key], dict):
            print(f"- Keys: {list(tree_rules[key].keys())}")
            # If it's a dictionary, show first item's structure
            if tree_rules[key]:
                first_item = next(iter(tree_rules[key].values()))
                if isinstance(first_item, dict):
                    print(f"- First item keys: {list(first_item.keys())}")
        elif isinstance(tree_rules[key], list):
            print(f"- Length: {len(tree_rules[key])}")
            if tree_rules[key]:
                print(f"- First item type: {type(tree_rules[key][0])}")
                if isinstance(tree_rules[key][0], dict):
                    print(f"- First item keys: {list(tree_rules[key][0].keys())}")
    
    # Get feature importances
    feature_importances = model.get_feature_importance()
    
    # Create feature name mapping
    feature_names = model.feature_names_
    
    # Process trees to extract structure
    processed_trees = []
    for tree_idx, tree in enumerate(tree_rules.get('trees', [])):
        print(f"\nProcessing tree {tree_idx}:")
        print(f"Tree keys: {list(tree.keys())}")
        
        # Extract tree structure based on available keys
        tree_info = {
            'index': tree_idx,
            'split': [],
            'leaf_values': []
        }
        
        # Process splits recursively
        def process_node(node, depth=0):
            if 'split' in node:
                split = node['split']
                print(f"Split at depth {depth}:", split)
                split_info = {
                    'feature_index': split.get('float_feature_index'),
                    'feature_name': feature_names[split.get('float_feature_index')],
                    'border': split.get('border'),
                    'left_child': split.get('left_child'),
                    'right_child': split.get('right_child')
                }
                tree_info['split'].append(split_info)
                
                # Process children
                if 'left' in node:
                    process_node(node['left'], depth + 1)
                if 'right' in node:
                    process_node(node['right'], depth + 1)
            else:
                # Leaf node
                tree_info['leaf_values'].append(node.get('value', 0))
        
        # Start processing from root
        process_node(tree)
        
        processed_trees.append(tree_info)
        print(f"Tree {tree_idx} processed - {len(tree_info['split'])} split, {len(tree_info['leaf_values'])} leaves")
    
    # Create features_info structure
    features_info = {
        'float_features': [
            {
                'flat_feature_index': i,
                'feature_id': name
            }
            for i, name in enumerate(feature_names)
        ]
    }
    
    tree_structure = {
        'trees': processed_trees,
        'feature_importances': feature_importances,
        'feature_names': feature_names,
        'features_info': features_info
    }
    
    # Validate tree structure
    print("\nValidating tree structure:")
    print(f"- Number of trees: {len(tree_structure['trees'])}")
    print(f"- Number of features: {len(tree_structure['feature_names'])}")
    print(f"- Feature importance shape: {feature_importances.shape}")
    print(f"- Features info keys: {list(tree_structure['features_info'].keys())}")
    
    # Print sample of first tree
    if processed_trees:
        first_tree = processed_trees[0]
        print("\nSample of first tree:")
        print(f"- Number of splits: {len(first_tree['split'])}")
        if first_tree['split']:
            print("- First split:", first_tree['split'][0])
        print(f"- Number of leaf values: {len(first_tree['leaf_values'])}")
        if first_tree['leaf_values']:
            print("- First leaf value:", first_tree['leaf_values'][0])
    
    return tree_structure


# === Function: build_decision_rules ===
def build_decision_rules(tree_structure):
    trees = tree_structure.get("trees", [])
    features_info = tree_structure.get("features_info", {})
    ctr_data = tree_structure.get("ctr_data", {})

    # Extract float and categorical feature names
    float_feature_names = {
        f.get("feature_index", f.get("flat_feature_index")): f.get("feature_name") or f.get("feature_id") or f"float_feature_{f.get('feature_index')}"
        for f in features_info.get("float_features", [])
    }

    cat_feature_names = {
        f.get("feature_index", f.get("flat_feature_index")): f.get("feature_name") or f.get("feature_id") or f"cat_feature_{f.get('feature_index')}"
        for f in features_info.get("categorical_features", [])
    }

    # Map OnlineCtr split index to readable feature names
    ctr_feature_name_map = {}
    ctr_data_keys = list(ctr_data.keys())
    for idx, key in enumerate(ctr_data_keys):
        try:
            parsed_key = json.loads(key)
            cat_idx = parsed_key["identifier"][0]["cat_feature_index"]
            ctr_type = parsed_key.get("type", "CTR")
            base_name = cat_feature_names.get(cat_idx, f"unknown_feature_{cat_idx}")
            ctr_feature_name_map[idx] = f"{ctr_type}_{base_name}"
        except Exception:
            ctr_feature_name_map[idx] = f"CTR_unknown_feature_{idx}"

    rules = []
    print(f"Processing {len(trees)} trees\n")

    def recurse_tree(node, conditions):
        if "value" in node:
            pred = node["value"]
            rule = {
                "conditions": conditions[:],
                "prediction": pred
            }
            rules.append(rule)
            return

        split = node.get("split", {})
        split_type = split.get("split_type")
        split_index = split.get("split_index")
        threshold = split.get("border")

        if split_type == "FloatFeature":
            feature_name = float_feature_names.get(split_index, f"unknown_feature_{split_index}")
        elif split_type == "OnlineCtr":
            feature_name = ctr_feature_name_map.get(split_index, f"CTR_unknown_feature_{split_index}")
        else:
            feature_name = f"unknown_feature_{split_index}"

        # Go left: feature <= threshold
        recurse_tree(node["left"], conditions + [f"{feature_name} <= {threshold}"])
        # Go right: feature > threshold
        recurse_tree(node["right"], conditions + [f"{feature_name} > {threshold}"])

    for tree_idx, tree in enumerate(trees):
        print(f"Processing tree {tree_idx}:")
        recurse_tree(tree, [])

    print("\n=== Symbolic Decision Rules (Top 15) ===")
    for i, rule in enumerate(rules[:15]):
        cond_str = " AND ".join(rule["conditions"])
        print(f"Rule {i+1}: IF {cond_str} THEN prediction = {rule['prediction']}")

    return rules


def setup_ffa_environment():
    """Set up the FFA environment with model and data."""
    print("\n=== Setting up FFA Environment ===")
    
    # Download model files
    model_path = download_model_files()
    
    # Extract tree structure
    tree_structure = extract_tree_structure(model_path)
    
    # Build decision rules
    rules = build_decision_rules(tree_structure)
    
    # Print sample rules
    print("\n=== Sample Decision Rules ===")
    for i, rule in enumerate(rules[:5]):
        print(f"\nRule {i+1}:")
        print(f"Conditions: {rule['conditions']}")
        print(f"Prediction: {rule['prediction']}")
        print(f"Support: {rule['support']:.3f}")
        print(f"Coverage: {rule['coverage']:.3f}")
        print(f"Confidence: {rule['confidence']:.3f}")
    
    # Initialize explainer
    explainer = FFAExplainer(
        model_path=model_path,
        tree_structure=tree_structure,
        rules=rules
    )
    
    return explainer, OUTPUT_DIR


def apply_ffa_rules(explainer, model, X_test, y_pred):
    """Apply FFA rules to test dataset and generate AXP explanations."""
    # Class 1 analysis
    class1_indices = np.where(y_pred == 1)[0]
    unmatched = []
    for i in class1_indices:
        x = X_test[i] if isinstance(X_test, np.ndarray) else X_test.iloc[i].values
        matched = explainer._satisfied_rules(x, target_class=1)
        if len(matched) == 0:
            unmatched.append(i)

    print(f"Total class 1 predictions: {len(class1_indices)}")
    print(f"Class 1 predictions with NO supporting rules: {len(unmatched)}")

    # Generate AXP explanations
    X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
    mask1 = (y_pred == 1)
    X_class1 = X_test_df.loc[mask1]
    y_class1 = y_pred[mask1]

    df_axps1 = explainer.explain_dataset(X_class1, predictions=y_class1)
    df_axps1.to_csv("symbolic_axps_class1.csv", index=False)
    
    return df_axps1


def analyze_feature_importance(df_axps, output_dir):
    """Analyze and visualize feature importance from AXP explanations."""
    # Process AXP explanations
    valid_axps = df_axps["axp"].dropna().astype(str)
    total_explanations = len(valid_axps)

    # Extract features
    all_features = []
    for cond_list in valid_axps:
        try:
            parsed = eval(cond_list)
            unique_feats = set(condition.split()[0] for condition in parsed)
            all_features.extend(unique_feats)
        except Exception as e:
            print("Parse error:", cond_list)

    # Calculate normalized importance
    feature_counts = Counter(all_features)
    df_norm = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['raw_count'])
    df_norm['normalized'] = df_norm['raw_count'] / total_explanations
    df_norm = df_norm.sort_values(by="normalized", ascending=False)

    # Plot normalized feature importance
    plt.figure(figsize=(10, 6))
    df_norm['normalized'].plot(kind='bar', legend=False)
    plt.title("Normalized AXP Feature Frequency (Class 1)")
    plt.ylabel("Fraction of Explanations")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "axp_feature_importance_normalized_class1.png"), dpi=300)
    plt.close()

    return df_norm


def perform_causal_analysis(model, X_test, y_pred, df_axps):
    """Perform causal analysis by flipping features and measuring impact."""
    # Extract Class 1 test instances
    X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=model.feature_names_)
    X_class1 = X_test_df.loc[y_pred == 1].reset_index(drop=True)

    # Determine feature types
    feature_stats = {}
    for col in X_test_df.columns:
        unique_vals = X_test_df[col].dropna().unique()
        if len(unique_vals) == 2 and sorted(unique_vals) in [[0, 1], [False, True]]:
            feature_stats[col] = {"type": "binary"}
        else:
            feature_stats[col] = {
                "type": "numeric",
                "mean": X_test_df[col].mean(),
                "std": X_test_df[col].std()
            }

    # Store causal effects
    causal_effects = defaultdict(int)
    total_instances = len(X_class1)

    df_axps["parsed_axp"] = df_axps["axp"].apply(literal_eval)

    for idx, row in X_class1.iterrows():
        original = row.values.copy()
        axp = df_axps.loc[idx, "parsed_axp"]
        if not isinstance(axp, list):
            continue

        # Parse feature names in AXP
        axp_features = [cond.split()[0] for cond in axp]

        for feat in axp_features:
            if feat not in X_test_df.columns:
                continue

            feat_idx = X_test_df.columns.get_loc(feat)
            flipped = original.copy()

            # Smart flipping
            ftype = feature_stats[feat]["type"]
            if ftype == "binary":
                flipped[feat_idx] = 1 - original[feat_idx]  # flip 0 ↔ 1
            elif ftype == "numeric":
                flipped[feat_idx] = original[feat_idx] + np.sign(original[feat_idx]) * feature_stats[feat]["std"]
            else:
                continue

            # Re-predict
            new_pred = model.predict(flipped.reshape(1, -1))[0]

            # If prediction flips, count the feature
            if new_pred != 1:
                causal_effects[feat] += 1

    # Create causal summary
    causal_summary = pd.DataFrame([
        {
            "feature": f,
            "causal_responsibility": causal_effects[f] / total_instances
        }
        for f in causal_effects
    ]).sort_values(by="causal_responsibility", ascending=False)

    return causal_summary


class FFAAnalyzer:
    def __init__(self, train_df, test_df, model, explainer):
        """
        Initialize FFA Analyzer with data and model
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            model: Trained model object
            explainer: FFA explainer object
        """
        self.train_df = train_df
        self.test_df = test_df
        self.model = model
        self.explainer = explainer
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def prepare_data(self):
        """Prepare test data for analysis"""
        print("\n=== Preparing Data ===")
        print(f"Input test_df shape: {self.test_df.shape}")
        
        # Assuming last column is target
        self.X_test = self.test_df.iloc[:, :-1]
        self.y_test = self.test_df.iloc[:, -1]
        
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(f"X_test type: {type(self.X_test)}")
        
        # Convert to numpy array to match the original implementation
        if isinstance(self.X_test, pd.DataFrame):
            print("Converting DataFrame to numpy array")
            self.X_test = self.X_test.values
            print(f"After conversion - X_test type: {type(self.X_test)}")
            print(f"X_test dtype: {self.X_test.dtype}")
            
        print("Data preparation complete\n")
        
    def calibrate_model(self):
        """Calibrate model and get predictions"""
        print("\n=== Calibrating Model ===")
        if self.X_test is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
            
        # Get predictions
        print("Getting model predictions...")
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print(f"Predictions shape: {self.y_pred.shape}")
        print(f"Prediction probabilities shape: {self.y_pred_proba.shape}")
        print(f"Unique prediction values: {np.unique(self.y_pred)}")
        
        # Calculate optimal threshold using ROC
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal threshold: {optimal_threshold}")
        
        # Apply threshold
        self.y_pred = (self.y_pred_proba >= optimal_threshold).astype(int)
        print(f"After threshold - Unique prediction values: {np.unique(self.y_pred)}")
        print("Model calibration complete\n")
        
        return optimal_threshold
        
    def calculate_metrics(self):
        """Calculate and return model performance metrics"""
        metrics = {
            'AUC': roc_auc_score(self.y_test, self.y_pred_proba),
            'Brier Score': brier_score_loss(self.y_test, self.y_pred_proba),
            'Accuracy': accuracy_score(self.y_test, self.y_pred),
            'Log Loss': log_loss(self.y_test, self.y_pred_proba),
            'F1 Score': f1_score(self.y_test, self.y_pred),
            'Precision': precision_score(self.y_test, self.y_pred),
            'Recall': recall_score(self.y_test, self.y_pred),
            'AUPR': average_precision_score(self.y_test, self.y_pred_proba)
        }
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics.update({
            'True Negative': tn,
            'False Positive': fp,
            'False Negative': fn,
            'True Positive': tp
        })
        
        return pd.DataFrame([metrics])
        
    def analyze_class_predictions(self, target_class):
        """Analyze predictions for a specific class"""
        print(f"\n=== Analyzing Class {target_class} Predictions ===")
        if self.y_pred is None:
            raise ValueError("Model not calibrated. Call calibrate_model() first.")
            
        class_indices = np.where(self.y_pred == target_class)[0]
        print(f"Found {len(class_indices)} instances of class {target_class}")
        
        unmatched = []
        for i in class_indices:
            x = self.X_test[i] if isinstance(self.X_test, np.ndarray) else self.X_test[i].values
            matched = self.explainer._satisfied_rules(x, target_class=target_class)
            if len(matched) == 0:
                unmatched.append(i)
                
        print(f"Total class {target_class} predictions: {len(class_indices)}")
        print(f"Class {target_class} predictions with NO supporting rules: {len(unmatched)}")
        
        return {
            'total_predictions': len(class_indices),
            'unmatched_predictions': len(unmatched)
        }
        
    def generate_axp_explanations(self, target_class):
        """Generate AXP explanations for a specific class"""
        print(f"\n=== Generating AXP Explanations for Class {target_class} ===")
        if self.y_pred is None:
            raise ValueError("Model not calibrated. Call calibrate_model() first.")
            
        mask = (self.y_pred == target_class)
        X_class = self.X_test[mask]
        y_class = self.y_pred[mask]
        
        print(f"X_class shape: {X_class.shape}")
        print(f"y_class shape: {y_class.shape}")
        
        print("Generating explanations...")
        df_axps = self.explainer.explain_dataset(X_class, predictions=y_class)
        print(f"Generated {len(df_axps)} explanations")
        print("First few explanations:")
        print(df_axps.head())
        
        return df_axps
        
    def analyze_feature_importance(self, df_axps):
        """Analyze feature importance from AXP explanations"""
        valid_axps = df_axps["axp"].dropna().astype(str)
        total_explanations = len(valid_axps)
        
        all_features = []
        for cond_list in valid_axps:
            try:
                parsed = eval(cond_list)
                unique_feats = set(condition.split()[0] for condition in parsed)
                all_features.extend(unique_feats)
            except Exception as e:
                print(f"Parse error: {cond_list}")
                
        feature_counts = Counter(all_features)
        df_norm = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['raw_count'])
        df_norm['normalized'] = df_norm['raw_count'] / total_explanations
        df_norm = df_norm.sort_values(by="normalized", ascending=False)
        
        return df_norm
        
    def plot_feature_importance(self, df_norm, class_label, save_path=None):
        """Plot normalized feature importance"""
        plt.figure(figsize=(10, 6))
        df_norm['normalized'].plot(kind='bar', legend=False)
        plt.title(f"Normalized AXP Feature Frequency (Class {class_label})")
        plt.ylabel("Fraction of Explanations")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
    def analyze_rule_metrics(self, df_metrics, manifest, save_path=None):
        """Analyze rule metrics with enhanced causal analysis."""
        print("\n=== Analyzing Rule Metrics ===")
        
        # Calculate causal importance
        causal_importance = self.calculate_causal_importance(df_metrics, self.X_test, self.y_pred)
        df_metrics['causal_importance'] = causal_importance
        
        # Validate causal importance
        validation_results = self.validate_causal_importance(df_metrics, self.X_test, self.y_test)
        df_metrics = df_metrics.merge(validation_results, on='rule_id', how='left')
        
        # Plot causal relationships
        feature_importance = self.plot_causal_relationships(df_metrics, self.X_test, save_path)
        
        # Create rule table with enhanced metrics
        rule_table = self.create_rule_table(df_metrics, manifest)
        
        # Save results
        if save_path:
            # Save metrics
            metrics_path = os.path.join(os.path.dirname(save_path), 'rule_metrics.csv')
            df_metrics.to_csv(metrics_path, index=False)
            print(f"\nSaved rule metrics to {metrics_path}")
            
            # Save rule table
            table_path = os.path.join(os.path.dirname(save_path), 'rule_table.csv')
            rule_table.to_csv(table_path, index=False)
            print(f"Saved rule table to {table_path}")
            
            # Upload to S3
            try:
                from s3_utils import save_to_s3_parquet
                # Save metrics
                s3_key = f"ffa_analysis/rule_metrics/rule_metrics.parquet"
                save_to_s3_parquet(df_metrics, "pgxdatalake", s3_key)
                print(f"Uploaded rule metrics to s3://pgxdatalake/{s3_key}")
                
                # Save rule table
                s3_key = f"ffa_analysis/rule_tables/rule_table.parquet"
                save_to_s3_parquet(rule_table, "pgxdatalake", s3_key)
                print(f"Uploaded rule table to s3://pgxdatalake/{s3_key}")
            except Exception as e:
                print(f"Warning: Failed to upload to S3: {str(e)}")
        
        return df_metrics, rule_table, feature_importance

    def create_rule_table(self, df_metrics, manifest):
        """Create comprehensive rule table with patterns and metrics.
        
        Args:
            df_metrics: DataFrame with rule metrics
            manifest: Feature manifest dictionary
            
        Returns:
            pd.DataFrame: Enhanced rule table
        """
        print("\n=== Creating Rule Table ===")
        
        # Extract pattern information
        pattern_info = {}
        for pattern in manifest.get('patterns', []):
            pattern_id = pattern['slot'].split('_')[1]
            pattern_info[pattern_id] = {
                'drugs': [item.replace('drug_', '').replace('_', ' ').title() 
                         for item in pattern['items'] if item.startswith('drug_')],
                'support': pattern.get('support', 0),
                'hash': pattern.get('hash', '')
            }
        
        # Create enhanced rule table
        rule_table = []
        for _, row in df_metrics.iterrows():
            rule = {
                'rule_id': row['rule_id'],
                'prediction': row['prediction'],
                'support': row['support'],
                'coverage': row['coverage'],
                'confidence': row['confidence'],
                'samples_satisfied': row['samples_satisfied'],
                'samples_correct': row['samples_correct'],
                'rule_type': row['rule_type']
            }
            
            # Extract patterns and drugs from conditions
            patterns = []
            drugs = []
            for condition in row['raw_conditions'].split(' AND '):
                if 'pattern_' in condition:
                    pattern_id = condition.split('_')[1]
                    if pattern_id in pattern_info:
                        patterns.append({
                            'id': pattern_id,
                            'drugs': pattern_info[pattern_id]['drugs'],
                            'support': pattern_info[pattern_id]['support'],
                            'hash': pattern_info[pattern_id]['hash']
                        })
                elif any(drug in condition.lower() for drug in ['drug_', 'medication']):
                    drugs.append(condition)
            
            rule['patterns'] = patterns
            rule['drugs'] = drugs
            rule_table.append(rule)
        
        return pd.DataFrame(rule_table)

    def calculate_causal_importance(self, df_metrics, X_test, y_pred):
        """Calculate causal importance of each rule by measuring prediction changes."""
        causal_importance = []
        
        for _, row in df_metrics.iterrows():
            # Get rule conditions
            conditions = row['raw_conditions']
            
            # Find samples that satisfy the rule
            rule_mask = self.evaluate_rule_conditions(X_test, conditions)
            rule_samples = X_test[rule_mask].copy()
            
            if len(rule_samples) == 0:
                causal_importance.append(0)
                continue
                
            # Calculate feature correlations
            feature_correlations = rule_samples.corr()
            
            # Calculate original predictions
            original_preds = y_pred[rule_mask]
            
            # Modify feature values to break the rule
            modified_samples = rule_samples.copy()
            for condition in conditions:
                feature, operator, value = condition
                # Get correlated features
                correlated_features = feature_correlations[feature][
                    abs(feature_correlations[feature]) > 0.3
                ].index
                
                # Modify main feature
                if operator == '>':
                    modified_samples[feature] = value - 1e-6
                elif operator == '<':
                    modified_samples[feature] = value + 1e-6
                elif operator == '==':
                    modified_samples[feature] = value + 1
                    
                # Adjust correlated features proportionally
                for corr_feature in correlated_features:
                    if corr_feature != feature:
                        corr = feature_correlations.loc[feature, corr_feature]
                        modified_samples[corr_feature] += corr * (
                            modified_samples[feature] - rule_samples[feature]
                        )
                    
            # Get new predictions
            new_preds = self.model.predict(modified_samples)
            
            # Calculate importance as average prediction change
            importance = np.mean(np.abs(original_preds - new_preds))
            causal_importance.append(importance)
            
        return causal_importance

    def validate_causal_importance(self, df_metrics, X_test, y_test):
        """Validate causal importance with additional metrics."""
        print("\n=== Validating Causal Importance ===")
        validation_results = []
        
        for _, row in df_metrics.iterrows():
            conditions = row['raw_conditions']
            rule_mask = self.evaluate_rule_conditions(X_test, conditions)
            
            if len(rule_mask) == 0:
                continue
                
            # Calculate rule stability
            rule_samples = X_test[rule_mask]
            rule_preds = self.model.predict(rule_samples)
            rule_stability = np.std(rule_preds)
            
            # Calculate rule coverage
            rule_coverage = len(rule_mask) / len(X_test)
            
            # Calculate rule accuracy
            rule_accuracy = np.mean(rule_preds == y_test[rule_mask])
            
            # Calculate feature importance stability
            feature_stability = {}
            for condition in conditions:
                feature, operator, value = condition
                feature_values = rule_samples[feature]
                feature_stability[feature] = {
                    'std': np.std(feature_values),
                    'range': np.ptp(feature_values),
                    'mean': np.mean(feature_values)
                }
            
            validation_results.append({
                'rule_id': row['rule_id'],
                'stability': rule_stability,
                'coverage': rule_coverage,
                'accuracy': rule_accuracy,
                'feature_stability': feature_stability
            })
        
        return pd.DataFrame(validation_results)

    def plot_causal_relationships(self, df_metrics, X_test, save_path=None):
        """Plot causal relationships between features and outcomes."""
        print("\n=== Plotting Causal Relationships ===")
        
        # Create feature importance plot
        feature_importance = []
        for _, row in df_metrics.iterrows():
            conditions = row['raw_conditions']
            for condition in conditions:
                feature, operator, value = condition
                feature_importance.append({
                    'feature': feature,
                    'importance': row['causal_importance'],
                    'operator': operator,
                    'value': value
                })
        
        feature_df = pd.DataFrame(feature_importance)
        feature_df = feature_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 2)
        
        # Plot feature importance
        ax1 = fig.add_subplot(gs[0, 0])
        feature_df.head(20).plot(kind='bar', ax=ax1)
        ax1.set_title('Top 20 Features by Causal Importance')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot feature value distributions
        ax2 = fig.add_subplot(gs[0, 1])
        for feature in feature_df.head(5).index:
            sns.kdeplot(data=X_test[feature], label=feature, ax=ax2)
        ax2.set_title('Distribution of Top 5 Causal Features')
        ax2.legend()
        
        # Plot feature correlations
        ax3 = fig.add_subplot(gs[1, :])
        top_features = feature_df.head(10).index
        corr_matrix = X_test[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax3)
        ax3.set_title('Correlation Matrix of Top 10 Causal Features')
        
        plt.tight_layout()
        
        if save_path:
            causal_plot_path = os.path.join(os.path.dirname(save_path), 
                                          'causal_relationships.png')
            plt.savefig(causal_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance to CSV
        if save_path:
            feature_importance_path = os.path.join(os.path.dirname(save_path), 
                                                 'feature_importance.csv')
            feature_df.to_csv(feature_importance_path)
            print(f"\nSaved feature importance to {feature_importance_path}")
            
            # Upload to S3
            try:
                from s3_utils import save_to_s3_parquet
                s3_key = f"ffa_analysis/feature_importance/feature_importance.parquet"
                save_to_s3_parquet(feature_df, "pgxdatalake", s3_key)
                print(f"Uploaded feature importance to s3://pgxdatalake/{s3_key}")
            except Exception as e:
                print(f"Warning: Failed to upload to S3: {str(e)}")
        
        return feature_df


def check_explainer_rules(explainer, num_rules=10):
    """Print out a specified number of rules from the explainer for inspection.
    
    Args:
        explainer: The CatBoostSymbolicExplainer instance
        num_rules: Number of rules to print (default: 10)
    """
    print(f"\n=== First {num_rules} Rules ===")
    for i, (clause, pred) in enumerate(zip(explainer.rule_clauses[:num_rules], explainer.rule_predictions[:num_rules])):
        conditions = []
        for lit in clause:
            feat_idx, thresh, direction = explainer.id_condition_map[lit]
            feat = explainer.feature_names[feat_idx]
            op = "<=" if direction == 0 else ">"
            conditions.append(f"{feat} {op} {thresh}")
        print(f"\nRule {i+1} (Class {pred}):")
        print(" AND ".join(conditions))


def load_feature_manifest(cohort_name, age_band, event_year):
    """Load feature manifest from S3.
    
    Args:
        cohort_name: Name of the cohort (e.g., 'ed_non_opioid')
        age_band: Age band (e.g., '0-12')
        event_year: Event year (e.g., '2016')
        
    Returns:
        dict: Feature manifest data with patterns and metrics
    """
    try:
        from s3_utils import parse_s3_path
        import duckdb
        
        # Construct S3 path
        s3_path = f"s3://pgxdatalake/feature_manifest/cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}/feature_manifest.json"
        
        # Load using DuckDB
        con = duckdb.connect(database=':memory:')
        con.sql("INSTALL httpfs; LOAD httpfs;")
        con.sql("CALL load_aws_credentials();")
        
        query = f"SELECT * FROM read_json_auto('{s3_path}')"
        manifest_df = con.sql(query).df()
        con.close()
        
        # Convert to dictionary
        manifest = manifest_df.to_dict('records')[0] if not manifest_df.empty else {}
        print(f"Loaded feature manifest for {cohort_name}/{age_band}/{event_year}")
        return manifest
        
    except Exception as e:
        print(f"Warning: Could not load feature manifest: {str(e)}")
        return {}


def get_pattern_description(pattern_id, manifest):
    """Get pattern description from manifest.
    
    Args:
        pattern_id: Pattern ID (e.g., '1' for pattern_1)
        manifest: Feature manifest dictionary
        
    Returns:
        str: Pattern description
    """
    try:
        # Find pattern in patterns array
        for pattern in manifest.get('patterns', []):
            if pattern['slot'] == f'pattern_{pattern_id}':
                # Get drug names from items
                drug_names = []
                for item in pattern['items']:
                    if item.startswith('drug_'):
                        drug_name = item.replace('drug_', '').replace('_', ' ').title()
                        drug_names.append(drug_name)
                
                # Get pattern metrics
                support = pattern.get('support', 0)
                
                # Format description
                if drug_names:
                    return f"{' + '.join(drug_names)} (Support: {support:.1%})"
                return f"Empty Pattern (Support: {support:.1%})"
        return f"Pattern {pattern_id}"
    except Exception as e:
        print(f"Warning: Error getting pattern description: {str(e)}")
        return f"Pattern {pattern_id}"

