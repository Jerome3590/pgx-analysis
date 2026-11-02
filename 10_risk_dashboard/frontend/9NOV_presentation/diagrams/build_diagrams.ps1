# Build script for D2 cohort pipeline diagrams
# Compiles all .d2 files in this folder to .svg and .png

Write-Host "Compiling D2 Cohort Pipeline Diagrams..." -ForegroundColor Cyan

$diagrams = @(
    "cohort_pipeline_opioid_ed",
    "cohort_pipeline_ed_non_opioid",
    "cohort_pipeline_dual_overview"
)

foreach ($d in $diagrams) {
    Write-Host "  -> Rendering $d.d2 to SVG and PNG..." -ForegroundColor Yellow
    d2 "$PSScriptRoot\$d.d2" "$PSScriptRoot\$d.svg"
    d2 "$PSScriptRoot\$d.d2" "$PSScriptRoot\$d.png"
}

Write-Host "`nAll diagrams compiled successfully!" -ForegroundColor Green
Get-ChildItem -Path $PSScriptRoot -Filter "cohort_pipeline_*" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
