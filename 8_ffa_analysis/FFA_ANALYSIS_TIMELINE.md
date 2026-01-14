# FFA Analysis Timeline and Phases

## Overview

The FFA analysis goes through several phases before reaching multi-feature interaction analysis. Here's what to expect:

## Phase Breakdown

### **Step 1: Load Model JSON** (~1-2 minutes)
- **CPU Usage**: 1-2 cores
- **What it does**: Loads the trained model JSON file
- **Parallelization**: None (single-threaded)

### **Step 2: Extract Feature Mappings** (~1-2 minutes)
- **CPU Usage**: 1-2 cores
- **What it does**: Extracts feature names and mappings from model
- **Parallelization**: None (single-threaded)

### **Step 3: Load Data** (~2-5 minutes)
- **CPU Usage**: 1-2 cores
- **What it does**: Loads training data (Parquet or CSV)
- **Parallelization**: None (single-threaded, I/O bound)

### **Step 4: Initialize Explainer** (~2-5 minutes)
- **CPU Usage**: 1-2 cores
- **What it does**: Extracts rules from model, builds explainer
- **Parallelization**: None (single-threaded)

### **Step 5: Generate AXP Explanations** (~10-30 minutes) ⚡ **PARALLEL**
- **CPU Usage**: ~28 cores (with fix)
- **What it does**: Generates explanations for all instances
- **Parallelization**: ✅ Uses ProcessPoolExecutor (28 workers)
- **This is where you'll first see high CPU usage**

### **Step 6: Calculate Feature Importance** (~2-5 minutes)
- **CPU Usage**: 1-2 cores
- **What it does**: Aggregates explanations into feature importance scores
- **Parallelization**: None (single-threaded aggregation)

### **Step 7: Perform Causal Analysis** (~5-15 minutes) ⚡ **PARALLEL**
- **CPU Usage**: ~28 cores (with fix)
- **What it does**: Tests how removing/modifying each feature affects explanations
- **Parallelization**: ✅ Uses ProcessPoolExecutor (28 workers)
- **This is the second phase with high CPU usage**

### **Step 7.5/8: Multi-Feature Interaction Analysis** (~10-30 minutes) ⚡ **PARALLEL**
- **CPU Usage**: ~28 cores (with fix)
- **What it does**: Tests combinations of features for synergy/antagonism
- **Parallelization**: ✅ Uses ProcessPoolExecutor (28 workers)
- **This is the third and final phase with high CPU usage**

### **Step 8: Save Results** (~1-2 minutes)
- **CPU Usage**: 1-2 cores
- **What it does**: Saves all results to disk and uploads to S3
- **Parallelization**: None (I/O bound)

## Total Timeline

| Phase | Duration | CPU Usage |
|-------|----------|-----------|
| Steps 1-4 (Setup) | ~6-14 minutes | Low (1-2 cores) |
| Step 5 (Explanations) | ~10-30 minutes | **High (28 cores)** |
| Step 6 (Importance) | ~2-5 minutes | Low (1-2 cores) |
| Step 7 (Causal) | ~5-15 minutes | **High (28 cores)** |
| Step 8 (Interactions) | ~10-30 minutes | **High (28 cores)** |
| Step 9 (Save) | ~1-2 minutes | Low (1-2 cores) |
| **Total** | **~34-96 minutes** | **~45-60 minutes at high CPU** |

## What You'll See

### **Early Phases (Steps 1-4):**
```
CPU Usage: 6% (2 cores at 100%)
Processes: 2 Python processes
Status: Normal - single-threaded setup phases
```

### **Parallel Phases (Steps 5, 7, 8):**
```
CPU Usage: 87% (28 cores at 80-100%)
Processes: ~29 Python processes (1 main + 28 workers)
Status: Normal - parallel processing active
```

### **Between Parallel Phases:**
```
CPU Usage: Drops to 6% (1-2 cores)
Processes: ~2 processes
Status: Normal - aggregation/saving phases
```

## Why It Takes Time

1. **Steps 1-4 are sequential** - Must complete before parallel phases
2. **Step 5 must complete** - Explanations needed for causal/interaction analysis
3. **Step 7 must complete** - Causal scores needed to select features for interactions
4. **Step 8 is computationally intensive** - Tests many feature combinations

## Monitoring Progress

### Check Current Step:
```bash
tail -f 8_ffa_analysis/logs/ffa_analysis_*.log | grep -E "Step [0-9]"
```

### Expected Log Messages:
- **Step 1**: "Step 1: Loading model JSON..."
- **Step 2**: "Step 2: Extracting feature mappings..."
- **Step 3**: "Step 3: Loading data..."
- **Step 4**: "Step 4: Initializing explainer..."
- **Step 5**: "Step 5: Generating explanations..." + "Using 28 parallel workers"
- **Step 6**: "Step 6: Calculating feature importance..."
- **Step 7**: "Step 7: Performing causal analysis..." + "Using 28 parallel workers"
- **Step 8**: "Step 7.5: Performing multi-feature interaction analysis..." + "Using 28 parallel workers"

## Tips

1. **Be patient during Steps 1-4** - These are necessary setup phases
2. **Watch for Step 5** - First parallel phase, should see ~28 cores active
3. **Step 7 is faster** - Uses grouped comparison optimization
4. **Step 8 can be longest** - Tests many combinations, but now uses 28 workers

## Summary

**Yes, it takes time to reach Step 8 (interactions)** because:
- Steps 1-4 must complete first (~6-14 minutes)
- Step 5 must complete (~10-30 minutes)
- Step 7 must complete (~5-15 minutes)
- **Total: ~21-59 minutes before Step 8 starts**

But once Step 8 starts, you should see ~28 cores active with the fix!
