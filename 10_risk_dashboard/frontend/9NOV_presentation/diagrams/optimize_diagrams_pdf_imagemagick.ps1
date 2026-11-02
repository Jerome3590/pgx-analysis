# optimize_diagrams_pdf_imagemagick.ps1
# End-to-end PDF font & image placement optimization pipeline:
# 1. D2 compiles .d2 to vector PDF (--pad 20)
# 2. pdftoppm renders vector PDF at 300 DPI (high-fidelity font rasterization)
# 3. ImageMagick trims whitespace (-trim +repage) to optimize slide placement
# 4. ImageMagick sharpens font contrast (-unsharp 0x0.6+0.8+0.005)
# 5. ImageMagick adds balanced padding (-border 36x36) and exports optimized PNG & PDF

$ErrorActionPreference = "Stop"
$diagDir = "c:\Projects\action\vcu\exam\presentation\diagrams"
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")

$diagrams = @(
    "cohort_pipeline_opioid_ed",
    "cohort_pipeline_ed_non_opioid",
    "cohort_pipeline_dual_overview"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Running D2 -> PDF -> ImageMagick Optimization Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

foreach ($name in $diagrams) {
    Write-Host "`n[Processing: $name]" -ForegroundColor Yellow

    $d2File = "$diagDir\$name.d2"
    $pdfFile = "$diagDir\$name.pdf"
    $tempPrefix = "$diagDir\temp_$name"
    $optPng = "$diagDir\$name.png"
    $optPdf = "$diagDir\${name}_optimized.pdf"

    # Step 1: Compile D2 directly to vector PDF
    Write-Host "  Step 1: Compiling D2 -> vector PDF..." -ForegroundColor Gray
    d2 --pad 20 "$d2File" "$pdfFile"

    # Step 2: Render PDF at 300 DPI via pdftoppm (superior font engine)
    Write-Host "  Step 2: Rendering PDF at 300 DPI via pdftoppm..." -ForegroundColor Gray
    pdftoppm -png -r 300 "$pdfFile" "$tempPrefix"

    $renderedPng = "$tempPrefix-1.png"
    if (-not (Test-Path $renderedPng)) {
        throw "Failed to render $pdfFile via pdftoppm"
    }

    # Step 3 & 4: ImageMagick font sharpening, trim bounding box, and add clean framing border
    Write-Host "  Step 3: ImageMagick optimizing fonts, trimming whitespace & framing..." -ForegroundColor Gray
    magick "$renderedPng" `
        -trim +repage `
        -bordercolor "#FFFFFF" `
        -border 36x36 `
        -unsharp 0x0.6+0.8+0.005 `
        "$optPng"

    # Step 5: Export optimized publication PDF
    Write-Host "  Step 4: Packaging into optimized PDF..." -ForegroundColor Gray
    magick "$optPng" -quality 100 -compress zip "$optPdf"

    # Cleanup temp render
    Remove-Item "$renderedPng" -ErrorAction SilentlyContinue

    Write-Host "  ? Completed: $name.png and ${name}_optimized.pdf" -ForegroundColor Green
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "All Cohort Pipeline Diagrams Successfully Optimized!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

Get-ChildItem -Path $diagDir -Filter "cohort_pipeline_*" | 
    Select-Object Name, Length, LastWriteTime | 
    Format-Table -AutoSize
