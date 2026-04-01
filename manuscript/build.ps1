# build.ps1 – Windows PowerShell build script for dissertation manuscripts
#
# Usage:
#   .\build.ps1              # build all chapters → output/ (journal PDFs)
#   .\build.ps1 -Chapter 1  # build Chapter 1 only
#   .\build.ps1 -Docx        # build all chapters → edits/ (Word .docx for advisor)
#   .\build.ps1 -Docx -Chapter 2  # single chapter to edits/
#   .\build.ps1 -Draft       # plain article class, no journal template
#   .\build.ps1 -Clean       # remove output/ and edits/ artifacts
#   .\build.ps1 -Full        # full dissertation → output/dissertation_dixon_<yyyyMMdd_HHmmss>.pdf
#   .\build.ps1 -Full -Docx    # full dissertation → edits/dissertation_dixon_<yyyyMMdd_HHmmss>.docx
#
# Prerequisites:
#   Quarto CLI  : https://quarto.org/docs/get-started/
#   LaTeX       : quarto install tinytex  OR  MiKTeX  OR  TeXLive
#   MDPI cls    : templates/Definitions/mdpi.cls  (already bundled)
#   Wiley cls   : bundled via _extensions/ramiromagno/wiley-njd/

param(
    [int]$Chapter  = 0,      # 0 = all chapters
    [switch]$Draft  = $false,
    [switch]$Docx   = $false,
    [switch]$Clean  = $false,
    [switch]$Full   = $false  # build full dissertation PDF
)

$Root    = $PSScriptRoot
$Output  = Join-Path $Root "output"
$Edits   = Join-Path $Root "edits"

# ── TEXINPUTS: lets xelatex find templates/Definitions/mdpi.cls ─────────────
$env:TEXINPUTS = ".;$Root\templates\;;$env:TEXINPUTS"
# ── BSTINPUTS: lets bibtex find Definitions/mdpi.bst from any chapter dir ────
$env:BSTINPUTS = ".;$Root\templates\Definitions\;;$env:BSTINPUTS"
# ── BIBINPUTS: lets bibtex find refs/*.bib regardless of working directory ───
$env:BIBINPUTS = ".;$Root\refs\;;$env:BIBINPUTS"

# ── Helper ───────────────────────────────────────────────────────────────────
function Build-Chapter {
    param([string]$QmdPath, [string]$Format, [string]$PdfName, [string]$DocxName)

    $FullQmd = Join-Path $Root $QmdPath
    if (-not (Test-Path $FullQmd)) {
        Write-Warning "QMD not found: $FullQmd"
        return
    }

    if ($Docx) {
        # ── Word / Google Docs output for advisor review ──────────────────────
        Write-Host "`n==> Building $DocxName (docx) ..." -ForegroundColor Magenta
        quarto render $FullQmd --to docx `
            --output-dir $Edits `
            --output $DocxName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    OK  -> $Edits\$DocxName" -ForegroundColor Green
        } else {
            Write-Error "    FAILED: $DocxName (exit $LASTEXITCODE)"
        }
        return
    }

    # ── Journal PDF output ────────────────────────────────────────────────────
    Write-Host "`n==> Building $PdfName ..." -ForegroundColor Cyan

    if ($Draft) {
        quarto render $FullQmd --to pdf `
            --output-dir $Output `
            --output $PdfName `
            -M "format.pdf.template=" `
            -M "format.pdf.documentclass=article"
    } else {
        quarto render $FullQmd --to $Format `
            --output-dir $Output `
            --output $PdfName
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK  -> $Output\$PdfName" -ForegroundColor Green
    } else {
        Write-Error "    FAILED: $PdfName (exit $LASTEXITCODE)"
    }
}

# ── Full dissertation ────────────────────────────────────────────────────────
if ($Full) {
    $FullQmd = Join-Path $Root "full_dissertation\full_dissertation.qmd"
    if (-not (Test-Path $FullQmd)) {
        Write-Error "Full dissertation QMD not found: $FullQmd"
        exit 1
    }
    if (-not (Test-Path $Output)) { New-Item -ItemType Directory $Output | Out-Null }
    if (-not (Test-Path $Edits))  { New-Item -ItemType Directory $Edits  | Out-Null }

    $DissertationStamp = Get-Date -Format "yyyyMMdd_HHmmss"

    if ($Docx) {
        $DissertationDocx = "dissertation_dixon_$DissertationStamp.docx"
        Write-Host "`n==> Building full dissertation DOCX ..." -ForegroundColor Magenta
        Write-Host "    Output file: $DissertationDocx" -ForegroundColor DarkGray
        quarto render $FullQmd --to docx `
            --output-dir $Edits `
            --output $DissertationDocx
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    OK  -> $Edits\$DissertationDocx" -ForegroundColor Green
        } else {
            Write-Error "    FAILED: full dissertation docx (exit $LASTEXITCODE)"
        }
        exit 0
    }

    $DissertationPdf = "dissertation_dixon_$DissertationStamp.pdf"
    Write-Host "`n==> Building full dissertation PDF ..." -ForegroundColor Cyan
    Write-Host "    Output file: $DissertationPdf" -ForegroundColor DarkGray
    quarto render $FullQmd --to pdf `
        --output-dir $Output `
        --output $DissertationPdf
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK  -> $Output\$DissertationPdf" -ForegroundColor Green
    } else {
        Write-Error "    FAILED: full dissertation (exit $LASTEXITCODE)"
    }
    exit 0
}

# ── Clean ────────────────────────────────────────────────────────────────────
if ($Clean) {
    Write-Host "Cleaning output/ and edits/ ..." -ForegroundColor Yellow
    Get-ChildItem $Output -Include *.pdf,*.tex,*.log -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem $Edits  -Include *.docx -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    # Quarto may leave a PDF in the chapter cwd on failed moves
    Get-ChildItem $Root -Filter "ch*_*.pdf" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "Done. (Close any open PDFs in output/ if files could not be deleted.)" -ForegroundColor Green
    exit 0
}

# ── Ensure output directories exist ──────────────────────────────────────────
if (-not (Test-Path $Output)) { New-Item -ItemType Directory $Output | Out-Null }
if (-not (Test-Path $Edits))  { New-Item -ItemType Directory $Edits  | Out-Null }

# ── Chapter definitions ───────────────────────────────────────────────────────
# Format: pdf for MDPI/plain; wiley-njd-pdf for Wiley chapters (extension required)
$Chapters = @(
    @{ Num=1; Qmd="CH_1\ch01_bmic.qmd";       Format="pdf";           Pdf="ch01_bmic_jpm.pdf";   Docx="ch01_bmic_draft.docx"      },
    @{ Num=2; Qmd="CH_2\ch02_psp.qmd";        Format="wiley-njd-pdf"; Pdf="ch02_psp.pdf";        Docx="ch02_psp_draft.docx"        },
    @{ Num=3; Qmd="CH_3\ch03_cts.qmd";        Format="wiley-njd-pdf"; Pdf="ch03_cts.pdf";        Docx="ch03_cts_draft.docx"        },
    @{ Num=4; Qmd="CH_4\ch04_psp.qmd";        Format="wiley-njd-pdf"; Pdf="ch04_psp.pdf";        Docx="ch04_psp_draft.docx"        },
    @{ Num=5; Qmd="CH_5\ch05_bmic.qmd";       Format="pdf";           Pdf="ch05_bmic_jpm.pdf";   Docx="ch05_bmic_draft.docx"       },
    @{ Num=6; Qmd="CH_6\ch06_conclusion.qmd"; Format="pdf";           Pdf="ch06_conclusion.pdf"; Docx="ch06_conclusion_draft.docx" }
)

# ── Build ────────────────────────────────────────────────────────────────────
if ($Chapter -eq 0) {
    foreach ($ch in $Chapters) {
        Build-Chapter -QmdPath $ch.Qmd -Format $ch.Format -PdfName $ch.Pdf -DocxName $ch.Docx
    }
} else {
    $target = $Chapters | Where-Object { $_.Num -eq $Chapter }
    if ($null -eq $target) {
        Write-Error "Unknown chapter: $Chapter  (valid: 1-6)"
        exit 1
    }
    Build-Chapter -QmdPath $target.Qmd -Format $target.Format -PdfName $target.Pdf -DocxName $target.Docx
}

if ($Docx) {
    Write-Host "`nAll done. Drafts in: $Edits" -ForegroundColor Magenta
} else {
    Write-Host "`nAll done. PDFs in: $Output" -ForegroundColor Green
}
