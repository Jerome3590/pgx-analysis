# FPGrowth Analysis - Installation & Execution Guide

## ⚠️ Administrator Access Required

Your system Python requires admin rights to install packages. Follow these steps:

## 🚀 Quick Start (3 Steps)

### Step 1: Install mlxtend (Run as Administrator)
```powershell
# Right-click PowerShell -> "Run as Administrator"
python -m pip install mlxtend
```

### Step 2: Run the Analysis
```bash
cd C:\Projects\pgx-analysis
./run_fpgrowth_optimized.sh
```

### Step 3: Monitor Progress
```bash
# Watch live
tail -f fpgrowth_execution.log

# Or use monitor script  
./monitor_fpgrowth.sh
```

---

## 📊 Your System Configuration

**Detected Hardware:**
- CPU: Intel 14 cores / 20 threads @ 2.5GHz
- RAM: 32GB (12.8GB available)
- GPU: NVIDIA RTX 3080 Ti (16GB VRAM)

**Optimized Settings:**
- Global Analysis: Sequential (handles 7GB dataset)
- Cohort Analysis: 10 parallel workers
- Total Estimated Time: 3-5 hours

---

## 📂 Expected Outputs

### S3 Structure:
```
s3://pgxdatalake/gold/fpgrowth/
├── global/
│   ├── drug_name/
│   │   ├── encoding_map.json
│   │   ├── itemsets.json
│   │   ├── rules.json
│   │   └── metrics.json
│   ├── icd_code/ (same files)
│   └── cpt_code/ (same files)
│
└── cohort/
    ├── drug_name/
    │   └── cohort_name=*/age_band=*/event_year=*/
    ├── icd_code/ (same structure)
    └── cpt_code/ (same structure)
```

---

## 🐛 Troubleshooting

### If mlxtend still not found after pip install:
```bash
# Verify installation
python -c "import mlxtend; print(f'mlxtend {mlxtend.__version__} OK')"

# If still fails, try:
python -m pip install --force-reinstall mlxtend
```

### If permission errors persist:
```bash
# Alternative: Create virtual environment
python -m venv fpgrowth_env
source fpgrowth_env/bin/activate  # or fpgrowth_env\Scripts\activate on Windows
pip install mlxtend pandas numpy scipy scikit-learn
./run_fpgrowth_optimized.sh
```

---

## ⏱️ Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| **Global drug_name** | 15-20 min | 4 files |
| **Global icd_code** | 15-20 min | 4 files |
| **Global cpt_code** | 10-15 min | 4 files |
| **Cohort processing** | 2-4 hours | 270 result sets |
| **Total** | **3-5 hours** | Full analysis complete |

---

## ✅ Success Indicators

Watch for these in the logs:
```
✓ mlxtend imported successfully
✓ Extracted X unique drugs
✓ Created Y transactions
✓ Found Z frequent itemsets
✓ Generated W association rules
✓ Saved to S3: s3://pgxdatalake/...
```

---

## 📝 After Completion

The executed notebooks will be saved with all outputs and errors:
- `3_fpgrowth_analysis/executed_global_fpgrowth.ipynb`
- `3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb`

You can open these in Jupyter to review results and any errors.

---

**Ready to start? Run PowerShell as Administrator and execute Step 1!** 🚀

