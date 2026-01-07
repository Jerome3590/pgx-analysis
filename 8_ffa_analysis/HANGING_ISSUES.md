# FFA Analysis Hanging Issues and Optimizations

## Identified Issues

### 1. **Causal Analysis is Extremely Slow** ⚠️ CRITICAL

**Location:** `run_full_ffa_analysis.py:897-948`

**Problem:**
- Calls `explain_dataset()` **twice** for each feature (original + modified)
- Uses `show_progress=False` and `n_jobs=1` (no progress visibility)
- For 100 features, this means **200 full explanation runs**
- Each explanation run processes 100 samples (causal_sample_size)
- **Total: 20,000 explanation instances** (100 features × 2 runs × 100 samples)

**Current Code:**
```python
for feat_idx, feat_name in enumerate(tqdm(available_features, desc="Causal analysis")):
    # ... 
    original_explanations = explainer.explain_dataset(
        X_sample,  # 100 samples
        predictions=y_sample,
        return_df=True,
        show_progress=False,  # ❌ No progress visibility
        n_jobs=1  # ❌ Single-threaded
    )
    modified_explanations = explainer.explain_dataset(
        X_modified,  # Another 100 samples
        predictions=y_sample,
        return_df=True,
        show_progress=False,  # ❌ No progress visibility
        n_jobs=1  # ❌ Single-threaded
    )
```

**Impact:** If each explanation takes 1 second per instance, this would take **5.5 hours** (20,000 seconds).

**Fix:**
1. Add progress logging for each feature
2. Add timeout mechanism
3. Reduce causal_sample_size further (currently 100, could be 50 or even 20)
4. Add early exit if taking too long
5. Make causal analysis optional/skippable

### 2. **Multi-Feature Causal Analysis Could Hang** ⚠️ CRITICAL

**Location:** `run_full_ffa_analysis.py:1000-1145`

**Problem:**
- Generates combinations of features (2-way, 3-way interactions)
- **COMBINATORIAL EXPLOSION**: Without filtering, testing all combinations of all features is impossible
  - Example: 11,060 features → C(11,060, 2) = **61 million pairs** (impossible!)
  - Even with 100 features: C(100, 2) = 4,950 pairs, C(100, 3) = 161,700 triplets

**Solution Implemented:**
- **SHAP/FFA/Causal Filtering**: Only include features with ANY importance > 0
  - SHAP importance > 0 (model-level), OR
  - FFA importance > 0 (explanation-based), OR  
  - Causal importance > 0 (individual causal effect)
- **Impact**: Reduces feature set from thousands to typically 20-100 important features
  - Example: 50 important features → C(50, 2) = 1,225 pairs (manageable!)
  - Example: 100 important features → C(100, 2) = 4,950 pairs (still manageable)
- **No Max Limit**: Tests ALL combinations of important features (no arbitrary cutoff)
- This dramatically reduces combinatorial explosion while ensuring comprehensive coverage

**Previous Problem:**
- For `interaction_top_k=20` features:
  - 2-way combinations: C(20,2) = 190 combinations
  - 3-way combinations: C(20,3) = 1,140 combinations
- Each combination calls `explain_dataset()` **twice**
- **Total: 2,660 explanation runs** (if all combinations processed)

**Current Code:**
```python
for interaction_size in [2, 3]:
    feature_combinations = list(combinations(top_features[:interaction_top_k], interaction_size))
    # Could be 190 + 1,140 = 1,330 combinations
    
    for combo_idx, feature_combo in enumerate(tqdm(feature_combinations, desc=f"Size {interaction_size}")):
        original_explanations = explainer.explain_dataset(...)  # ❌ Slow
        modified_explanations = explainer.explain_dataset(...)  # ❌ Slow
```

**Impact:** This could take **days** to complete.

**Fix:**
1. Significantly reduce `interaction_top_k` (default 20 → 10 or even 5)
2. Add maximum combination limit (e.g., max 50 combinations total)
3. Add timeout per combination
4. Skip multi-feature analysis if single-feature analysis takes too long

### 3. **No Progress Visibility in Critical Sections** ⚠️ HIGH

**Problem:**
- `show_progress=False` in causal analysis (line 934, 945)
- No intermediate logging during `explain_dataset` calls
- Hard to tell if it's hanging or just slow

**Fix:**
- Enable `show_progress=True` for causal analysis
- Add per-feature timing logs
- Add estimated time remaining

### 4. **Large Rule Sets Could Slow AXP Computation** ⚠️ MEDIUM

**Location:** `base_symbolic_explainer.py:526-600`

**Problem:**
- `_compute_axp` processes matched rules
- Uses Hitman solver which can be slow for large rule sets
- Current limit is 100 rules per set, but union could be ~300-500 rules

**Current Code:**
```python
# Set 1: First 100 rules
# Set 2: Random 100 rules  
# Set 3: SHAP-filtered rules (up to 300)
# Union = potentially 300-500 unique rules
```

**Fix:**
- Add timeout to AXP computation
- Further limit rule sets if computation takes too long
- Add progress logging in `_compute_axp`

### 5. **Memory Cleanup May Not Be Sufficient** ⚠️ MEDIUM

**Location:** `run_full_ffa_analysis.py:950-953`

**Problem:**
- Calls `gc.collect()` but may not be enough
- Large DataFrames may persist in memory

**Fix:**
- More aggressive memory cleanup
- Process features in smaller batches
- Clear intermediate results immediately

## Recommended Immediate Fixes

### Priority 1: Add Progress Logging and Timeouts

```python
# In perform_causal_analysis()
for feat_idx, feat_name in enumerate(available_features):
    logger.info(f"[{feat_idx+1}/{len(available_features)}] Analyzing {feat_name}...")
    feat_start = time.time()
    
    # Add timeout wrapper
    try:
        original_explanations = explainer.explain_dataset(
            X_sample,
            predictions=y_sample,
            return_df=True,
            show_progress=True,  # ✅ Enable progress
            n_jobs=1
        )
        logger.info(f"  Original explanations: {time.time() - feat_start:.2f}s")
        
        modified_explanations = explainer.explain_dataset(
            X_modified,
            predictions=y_sample,
            return_df=True,
            show_progress=True,  # ✅ Enable progress
            n_jobs=1
        )
        logger.info(f"  Modified explanations: {time.time() - feat_start:.2f}s")
        
        if time.time() - feat_start > 300:  # If > 5 minutes per feature
            logger.warning(f"  Feature {feat_name} took >5 minutes, skipping remaining features")
            break
    except Exception as e:
        logger.error(f"  Error analyzing {feat_name}: {e}")
        continue
```

### Priority 2: Reduce Sample Sizes

```python
# Reduce causal_sample_size from 100 to 20-50
causal_sample_size = min(20, len(X_class))  # Was 100

# Reduce interaction_top_k from 20 to 5-10
'interaction_top_k': 5,  # Was 20
```

### Priority 3: Add Early Exit Options

```python
# Add command-line flag to skip causal analysis
parser.add_argument('--skip-causal', action='store_true', 
                    help='Skip causal analysis (can be very slow)')

# Add maximum time limit
parser.add_argument('--max-causal-time', type=int, default=3600,
                    help='Maximum time (seconds) for causal analysis (default: 1 hour)')
```

### Priority 4: Optimize Multi-Feature Analysis

```python
# Limit total combinations
MAX_INTERACTION_COMBINATIONS = 50  # Hard limit

for interaction_size in [2, 3]:
    feature_combinations = list(combinations(top_features[:interaction_top_k], interaction_size))
    
    # Limit combinations
    if len(feature_combinations) > MAX_INTERACTION_COMBINATIONS:
        logger.warning(f"Limiting to {MAX_INTERACTION_COMBINATIONS} combinations (found {len(feature_combinations)})")
        feature_combinations = feature_combinations[:MAX_INTERACTION_COMBINATIONS]
```

## Quick Diagnostic Commands

To check where it's hanging:

1. **Check log file:**
```bash
tail -f 8_ffa_analysis/logs/ffa_analysis_*.log
```

2. **Check process:**
```bash
ps aux | grep run_full_ffa_analysis
```

3. **Check memory:**
```bash
top -p $(pgrep -f run_full_ffa_analysis)
```

4. **Add debug logging:**
```python
# Add at start of each major section
logger.info(f"[DEBUG] Starting section X at {time.time()}")
```

## Configuration Recommendations

Update `ANALYSIS_CONFIG` defaults:

```python
ANALYSIS_CONFIG = {
    'max_explanation_samples': 1000,  # Limit explanation samples
    'n_jobs': 1,  # Keep single-threaded for memory
    'batch_size': 50,  # Smaller batches
    'causal_sample_size': 20,  # Reduce from 100
    'interaction_top_k': 5,  # Reduce from 20
    'max_interaction_combinations': 50,  # New limit
    'skip_causal': False,  # Option to skip
    'max_causal_time': 3600,  # 1 hour max
}
```

