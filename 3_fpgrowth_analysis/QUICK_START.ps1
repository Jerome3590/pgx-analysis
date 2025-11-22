# FPGrowth Analysis - PowerShell Quick Start
# Run this script as Administrator

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "FPGrowth Analysis - Quick Start" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install mlxtend
Write-Host "Step 1: Installing mlxtend..." -ForegroundColor Yellow
try {
    python -m pip install mlxtend --quiet
    Write-Host "✓ mlxtend installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install mlxtend" -ForegroundColor Red
    Write-Host "  Please run PowerShell as Administrator" -ForegroundColor Yellow
    exit 1
}

# Step 2: Verify installation
Write-Host ""
Write-Host "Step 2: Verifying installation..." -ForegroundColor Yellow
$result = python -c "import mlxtend; print(f'mlxtend {mlxtend.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $result" -ForegroundColor Green
} else {
    Write-Host "✗ mlxtend not found - installation may have failed" -ForegroundColor Red
    Write-Host "  Try running: python -m pip install --user mlxtend" -ForegroundColor Yellow
    exit 1
}

# Step 3: Run the analysis
Write-Host ""
Write-Host "Step 3: Starting FPGrowth analysis..." -ForegroundColor Yellow
Write-Host "  This will take 3-5 hours to complete" -ForegroundColor Cyan
Write-Host "  Output will be saved to: fpgrowth_execution.log" -ForegroundColor Cyan
Write-Host ""

# Change to parent directory to run
Set-Location ..

# Run global analysis
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "GLOBAL ANALYSIS (30-60 minutes)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

jupyter nbconvert `
    --to notebook `
    --execute `
    --ExecutePreprocessor.timeout=7200 `
    --output 3_fpgrowth_analysis/executed_global_fpgrowth.ipynb `
    3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Global analysis complete!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Global analysis failed" -ForegroundColor Red
    Write-Host "  Check: 3_fpgrowth_analysis/executed_global_fpgrowth.ipynb" -ForegroundColor Yellow
    exit 1
}

# Run cohort analysis
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "COHORT ANALYSIS (2-4 hours)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

jupyter nbconvert `
    --to notebook `
    --execute `
    --ExecutePreprocessor.timeout=18000 `
    --output 3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb `
    3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Cohort analysis complete!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Cohort analysis failed" -ForegroundColor Red
    Write-Host "  Check: 3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "ANALYSIS COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to S3:" -ForegroundColor Cyan
Write-Host "  s3://pgxdatalake/gold/fpgrowth/global/{drug_name,icd_code,cpt_code}/" -ForegroundColor White
Write-Host "  s3://pgxdatalake/gold/fpgrowth/cohort/{drug_name,icd_code,cpt_code}/" -ForegroundColor White
Write-Host ""
Write-Host "Executed notebooks (with outputs):" -ForegroundColor Cyan
Write-Host "  3_fpgrowth_analysis/executed_global_fpgrowth.ipynb" -ForegroundColor White
Write-Host "  3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb" -ForegroundColor White
Write-Host ""

