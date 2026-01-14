

## Overview

- **XGBoost R GPU on Windows from source is effectively not supported in a stable way** with current toolchains (R 4.4.x/4.5.x + VS 2022 + CUDA 13.x). We reached repeated header / ABI issues in R’s own headers even after wiring CUDA and MSVC correctly.
- The most reliable path on Windows is to use the **official Python XGBoost GPU wheel** and keep R on a CPU‑only build, matching the versions used on EC2.
- As of this setup, the latest working Python XGBoost wheel in our environment is **3.1.2**, while the last CRAN Windows binary for R is more than two years behind (around the **2.0.0 era**), so trying to “match” R and Python feature parity on Windows via a custom R GPU build introduces more risk than value.

In short: **do not rely on a custom R GPU build from source on Windows** for production or reproducible work. Use Python for GPU training and R for data prep / analysis.

## Python XGBoost GPU on Windows (Supported Path)

In addition to the R build, we use Python XGBoost with GPU enabled from a project‑local virtual environment. This relies on the fact that the official Windows wheel already bundles GPU support when a compatible CUDA driver/toolkit is present.

### Prerequisites

- **NVIDIA driver + CUDA toolkit**:
  - Updated NVIDIA Game Ready driver (e.g., 591.44) installed on Windows.
  - CUDA runtime/toolkit installed so that:
    - `nvidia-smi` shows the GPU (e.g., RTX 3080 Ti Laptop GPU) and a CUDA version.
    - `nvcc --version` works from the CUDA `bin` directory.
  - This ensures the **OS and CUDA stack see the GPU**, which XGBoost requires for `device = "cuda"` [as described in the GPU docs](https://xgboost.readthedocs.io/en/stable/gpu/index.html).
- **Python virtual environment**:
  - A venv in the project root, e.g. `.venv`, created with:

```bash
    python -m venv .venv
    ```

  - GPU‑enabled Python packages installed into the venv:

```bash
    .venv/Scripts/python.exe -m pip install --upgrade \
      xgboost==3.1.2 catboost optuna shap matplotlib
    ```

### Verifying GPU access from Python

Run from the project root:

```bash
cd C:\Projects\pgx-analysis
.venv\Scripts\python.exe - << "EOF"
import xgboost as xgb
import numpy as np

print("XGBoost version:", xgb.__version__)

n, p = 2000, 50
X = np.random.rand(n, p).astype("float32")
y = (X.sum(axis=1) > p / 2).astype("int32")

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 4,
    "eta": 0.1,
}

print("Starting GPU training (Python XGBoost)...")
dtrain = xgb.DMatrix(X, label=y)
booster = xgb.train(params, dtrain, num_boost_round=20)
print("Finished training.")
EOF
```

If this script runs to completion without errors or “device changed from GPU to CPU” warnings, the **Python XGBoost wheel is successfully using the GPU** on Windows.

## CPU XGBoost Settings (Windows & EC2)

Even when a GPU is available, most of our production and EC2 runs use **CPU XGBoost** for stability and parity with R analyses. The main knobs are:

- **`tree_method="hist"`**  
  - Default for recent XGBoost versions; fast and memory‑efficient for tabular data.  
  - Use this for both CPU and GPU runs unless you have a specific reason to change it.

- **Thread control (`nthread` / `n_jobs`)**  
  - For the core XGBoost API, set `nthread` (or `n_jobs` in sklearn wrappers) to control CPU usage.
  - Recommended caps:
    - **Laptop / dev**: `nthread = 4–8`
    - **Workstation (16 cores)**: `nthread = 8–16`
    - **EC2 (32 cores)**: `nthread = 16–24` (leave headroom for other workers / I/O)
  - Example params for CPU training:

```python
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 4,
    "eta": 0.1,
    "nthread": 16,  # adjust for your machine
}
booster = xgb.train(params, dtrain, num_boost_round=200)
```

- **Device selection**  
  - **CPU-only**: omit `device` (or explicitly set `device = "cpu"`).  
  - **GPU**: set `device = "cuda"` as in the section above; if the GPU is not available, XGBoost may silently fall back to CPU.

- **Consistency with the final-model pipelines**  
  - The final-model Python code (EC2) assumes **CPU XGBoost by default**, with `tree_method="hist"` and a capped `nthread` based on instance size.  
  - GPU runs should be treated as **optional accelerators** for experimentation, not a hard dependency for the risk calculator. 

## XGBoost Random Forest Mode Settings

We also use XGBoost in **Random Forest (RF) mode** as part of the model ensemble. The key differences vs boosted trees are the **randomization settings** and **number of trees**:

- **How similar is it to a traditional Random Forest?**
  - Conceptually very close: many **independent trees**, each trained on **row and feature subsamples**, aggregated by averaging their predictions.  
  - The main implementation differences are:
    - XGBoost RF uses the same optimized tree builder as boosted trees (with `tree_method="hist"`), so it is often **faster** and more memory‑efficient than many sklearn RF implementations.  
    - Trees are built via the boosting API with `num_boost_round=1` and `num_parallel_tree = N`, instead of `n_estimators = N` as in classic RF.  
    - You still control depth, row/feature sampling, and evaluation metrics in almost the same way as a standard Random Forest.

- **Core RF parameters**
  - `num_parallel_tree`: number of trees grown in parallel per boosting iteration (RF uses many trees with `num_boost_round = 1`).  
  - `subsample`: row subsampling per tree (e.g., `0.7–0.9`).  
  - `colsample_bytree`: feature subsampling per tree (e.g., `0.5–0.8`).  
  - `eta`: often set to `1.0` in RF mode (no need for small learning rate because trees are not boosted sequentially).

- **Typical RF CPU config we use**

```python
params_rf = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 1.0,              # RF mode (no shrinkage over rounds)
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "num_parallel_tree": 200, # total trees in the forest
    "nthread": 16,            # adjust to machine/EC2 size
}

rf_model = xgb.train(params_rf, dtrain, num_boost_round=1)
```

- **GPU RF mode**
  - Same idea as CPU RF, but add `device = "cuda"` to `params_rf` and ensure the GPU test above passes.  
  - Keep `tree_method="hist"`; XGBoost will route trees to the GPU when `device="cuda"` is set.  
  - For EC2, we generally **start with CPU RF** and only enable GPU RF for experimentation, not as a production requirement. 