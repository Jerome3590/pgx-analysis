# refresh_after_download.ps1
# Run this after download_pmc_articles.py and unpaywall_lookup.py complete
# to refresh PRISMA counts, ontology organisation, and NLP pre-screening.
#
# Run from: manuscript/CH_1/Literature_Review/
#   .\scripts\refresh_after_download.ps1

Set-Location $PSScriptRoot\..
$Rscript = "C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
$python  = "python"

Write-Host "`n═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Post-Download Refresh Pipeline" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════`n" -ForegroundColor Cyan

# ── Phase 4: Refresh PRISMA counts ───────────────────────────────────────────
Write-Host "[Phase 4] Refreshing PRISMA counts..." -ForegroundColor Yellow
& $Rscript -e "here::i_am('lit_review.qmd'); source('scripts/prisma_tracker.R')"
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 4 failed"; exit 1 }
Write-Host "[Phase 4] Done.`n" -ForegroundColor Green

# ── Phase 6: Re-organise ontology ────────────────────────────────────────────
Write-Host "[Phase 6] Re-organising ontology..." -ForegroundColor Yellow
& $Rscript -e "here::i_am('lit_review.qmd'); source('scripts/organize_by_ontology.R')"
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 6 failed"; exit 1 }
Write-Host "[Phase 6] Done.`n" -ForegroundColor Green

# ── Phase 7: Re-run NLP pre-screening with full text ─────────────────────────
Write-Host "[Phase 7] Re-running NLP screening (full text available now)..." -ForegroundColor Yellow
& $python scripts/screen_articles.py --threshold 0.05
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 7 failed"; exit 1 }
Write-Host "[Phase 7] Done.`n" -ForegroundColor Green

# ── Phase 5: Re-render PRISMA chart ──────────────────────────────────────────
Write-Host "[Phase 5] Re-rendering PRISMA chart..." -ForegroundColor Yellow
quarto render lit_review.qmd --to html --no-execute
if ($LASTEXITCODE -ne 0) { Write-Warning "Phase 5 render warning (non-fatal)" }
Write-Host "[Phase 5] Done.`n" -ForegroundColor Green

Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Refresh complete: $(Get-Date -Format 'HH:mm')" -ForegroundColor Cyan
Write-Host "  Review: data/ontology/articles_screened.csv" -ForegroundColor Cyan
Write-Host "  PRISMA: output/CH_1/Literature_Review/lit_review.html" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════`n" -ForegroundColor Cyan
