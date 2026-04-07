# build_submission.ps1
# Creates self-contained submission ZIPs for journal upload.
# Each ZIP contains: .tex (bib paths patched), .cls, .bst, .bib files, figure TIFFs.
#
# Usage:
#   .\build_submission.ps1           # all chapters
#   .\build_submission.ps1 -Chapter 1  # single chapter

param([int]$Chapter = 0)

$Root    = $PSScriptRoot
$SubDir  = Join-Path $Root "output\submission"
if (-not (Test-Path $SubDir)) { New-Item -ItemType Directory $SubDir | Out-Null }

$Chapters = @(
    @{ Num=1; ChDir="CH_1"; Tex="ch01_cts.tex";  Figs="ch01"; Bib="discipline,bmic-jpm,cpt-psp,cts";        Journal="cts"     },
    @{ Num=2; ChDir="CH_2"; Tex="ch02_psp.tex";  Figs="ch02"; Bib="discipline,bmic-jpm,cpt-psp,cts";        Journal="cpt_psp" },
    @{ Num=3; ChDir="CH_3"; Tex="ch03_cts.tex";  Figs="ch03"; Bib="discipline,bmic-jpm,cpt-psp,cts";        Journal="cts"     },
    @{ Num=4; ChDir="CH_4"; Tex="ch04_psp.tex";  Figs="ch04"; Bib="discipline,bmic-jpm,cpt-psp,cts";        Journal="cpt_psp" },
    @{ Num=5; ChDir="CH_5"; Tex="ch05_cpt.tex";  Figs="ch05"; Bib="discipline,bmic-jpm,cpt-psp,cts";        Journal="cpt"     }
)

function Build-Zip {
    param($ch)

    $chDir   = Join-Path $Root $ch.ChDir
    $texSrc  = Join-Path $chDir $ch.Tex
    $figDir  = Join-Path $Root "figures\$($ch.Figs)"
    $outDir  = Join-Path $SubDir $ch.Journal
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory $outDir | Out-Null }
    $zipPath = Join-Path $outDir ($ch.Tex -replace '\.tex$', '_submission.zip')

    Write-Host "`n==> Building submission ZIP: $($ch.Tex)" -ForegroundColor Cyan

    # ── Stage in temp dir ────────────────────────────────────────────────────
    $stage = Join-Path $env:TEMP "wiley_stage_$($ch.Num)"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory $stage | Out-Null

    # 1. Patch .tex: replace ../refs/X with X (flat bib references)
    $texContent = Get-Content $texSrc -Raw
    $texContent = $texContent -replace '\.\./refs/', ''
    $patchedTex = Join-Path $stage $ch.Tex
    Set-Content -Path $patchedTex -Value $texContent -NoNewline -Encoding UTF8

    # 2. .cls and .bst files from chapter dir
    Get-ChildItem $chDir -Filter "*.cls" | Copy-Item -Destination $stage
    Get-ChildItem $chDir -Filter "*.bst" | Copy-Item -Destination $stage

    # 3. .bib files
    foreach ($bib in ($ch.Bib -split ',')) {
        $bibSrc = Join-Path $Root "refs\$bib.bib"
        if (Test-Path $bibSrc) {
            Copy-Item $bibSrc $stage
        } else {
            Write-Warning "  Missing bib: $bibSrc"
        }
    }

    # 4. Figure TIFFs
    if (Test-Path $figDir) {
        $tifs = Get-ChildItem $figDir -Filter "*.tif"
        foreach ($t in $tifs) { Copy-Item $t.FullName $stage }
        Write-Host "  Figures: $($tifs.Count) TIF files"
    }

    # 5. ZIP
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path "$stage\*" -DestinationPath $zipPath
    $sizekb = [int]((Get-Item $zipPath).Length / 1024)
    Write-Host "  OK -> $zipPath  ($sizekb KB)" -ForegroundColor Green

    Remove-Item $stage -Recurse -Force
}

if ($Chapter -eq 0) {
    foreach ($ch in $Chapters) { Build-Zip $ch }
} else {
    $target = $Chapters | Where-Object { $_.Num -eq $Chapter }
    if ($null -eq $target) { Write-Error "Unknown chapter: $Chapter"; exit 1 }
    Build-Zip $target
}

Write-Host "`nAll done. ZIPs -> $SubDir" -ForegroundColor Green
