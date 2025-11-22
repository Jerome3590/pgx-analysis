# Windows Quick Start - FPGrowth Analysis

## 🚀 One-Command Execution

### Step 1: Open PowerShell as Administrator
- Press `Win + X`
- Select "Windows PowerShell (Admin)" or "Terminal (Admin)"

### Step 2: Navigate and Run
```powershell
cd C:\Projects\pgx-analysis\3_fpgrowth_analysis
.\QUICK_START.ps1
```

That's it! The script will:
✅ Install mlxtend automatically
✅ Run global analysis (~45 min)
✅ Run cohort analysis (~3 hours)
✅ Save all results to S3

---

## 📊 What It Does

**Hardware Detected:**
- CPU: 14 cores / 20 threads
- RAM: 32GB
- GPU: RTX 3080 Ti (idle - FP-Growth is CPU-only)

**Processing:**
- **Global**: 3 item types (drug_name, icd_code, cpt_code)
- **Cohort**: 270 jobs (90 cohorts × 3 types) with 10 parallel workers

**Total Time:** 3-4 hours

---

## 📂 Outputs

### S3 Locations:
```
s3://pgxdatalake/gold/fpgrowth/
├── global/
│   ├── drug_name/
│   ├── icd_code/
│   └── cpt_code/
└── cohort/
    ├── drug_name/
    ├── icd_code/
    └── cpt_code/
```

### Local Files (with outputs/errors):
- `executed_global_fpgrowth.ipynb`
- `executed_cohort_fpgrowth.ipynb`

---

## ⚠️ If Script Fails

### Issue: "mlxtend not found"
```powershell
# Install manually:
python -m pip install mlxtend
```

### Issue: "Access denied" 
- Make sure you ran PowerShell as **Administrator**
- Alternative: `python -m pip install --user mlxtend`

### Issue: Want to see progress
```powershell
# Open another PowerShell window and run:
cd C:\Projects\pgx-analysis\3_fpgrowth_analysis
Get-Content ..\fpgrowth_execution.log -Wait
```

---

## 🔍 Manual Execution (if you prefer)

```powershell
# Install mlxtend (one-time)
python -m pip install mlxtend

# Run notebooks
cd C:\Projects\pgx-analysis

# Global analysis
jupyter nbconvert --to notebook --execute `
  --ExecutePreprocessor.timeout=7200 `
  3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb

# Cohort analysis  
jupyter nbconvert --to notebook --execute `
  --ExecutePreprocessor.timeout=18000 `
  3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb
```

---

**Ready? Just run `.\QUICK_START.ps1` as Administrator!** 🚀

