# build.ps1 – Windows PowerShell build script for dissertation manuscripts
#
# Usage:
#   .\build.ps1              # build all chapters → output/ (journal PDFs)
#   .\build.ps1 -Chapter 1  # build Chapter 1 only
#   .\build.ps1 -Docx        # build all chapters → output/edits/ (Word .docx for advisor)
#   .\build.ps1 -Docx -Chapter 2  # single chapter to output/edits/
#   .\build.ps1 -Submit         # full submission package: DOCX + TIFFs → output/final_submission/
#   .\build.ps1 -Submit -Chapter 2  # single chapter submission package
#   .\build.ps1 -Draft       # plain article class, no journal template
#   .\build.ps1 -Clean       # remove output/ artifacts
#   .\build.ps1 -Full        # full dissertation → output/dissertation_dixon_<yyyyMMdd_HHmmss>.pdf
#   .\build.ps1 -Full -Docx    # full dissertation → output/edits/dissertation_dixon_<yyyyMMdd_HHmmss>.docx
#   .\build.ps1 -ExportFigures    # export PSP-ready TIFF figures only → output/final_submission/
#   .\build.ps1 -ExportFigures -Chapter 4  # single PSP chapter TIFF export
#
# Output folder structure:
#   output/
#   ├── edits/<journal>/              ← DOCX drafts for advisor review
#   ├── final_submission/<journal>/chNN/   ← submission-ready packages (DOCX + TIFFs + supp)
#   ├── submission/<journal>/         ← LaTeX+TIFF ZIP packages (build_submission.ps1)
#   └── <journal>/                    ← compiled journal PDFs
#
# Prerequisites:
#   Quarto CLI  : https://quarto.org/docs/get-started/
#   LaTeX       : quarto install tinytex  OR  MiKTeX  OR  TeXLive
#   MDPI cls    : templates/Definitions/mdpi.cls  (already bundled)
#   Wiley cls   : bundled via _extensions/ramiromagno/wiley-njd/

param(
    [int]$Chapter         = 0,      # 0 = all chapters
    [switch]$Draft         = $false,
    [switch]$Docx          = $false,
    [switch]$Submit        = $false,  # DOCX + TIFFs → output/final_submission/ (one-step submission)
    [switch]$Clean         = $false,
    [switch]$Full          = $false,  # build full dissertation PDF
    [switch]$ExportFigures = $false   # export PSP-ready TIFFs only → output/final_submission/cpt_psp/
)

# -Submit implies -Docx (builds DOCX first, then exports figures)
$IsSubmit = $Submit.IsPresent
if ($IsSubmit) { $Docx = $true }

$Root    = $PSScriptRoot
$Output  = Join-Path $Root "output"
$Edits   = Join-Path $Output "edits"
$SubmitDir  = Join-Path $Output "final_submission"

# ── TEXINPUTS: lets xelatex find templates/Definitions/mdpi.cls ─────────────
$env:TEXINPUTS = ".;$Root\templates\;;$env:TEXINPUTS"
# ── BSTINPUTS: lets bibtex find Definitions/mdpi.bst from any chapter dir ────
$env:BSTINPUTS = ".;$Root\templates\Definitions\;;$env:BSTINPUTS"
# ── BIBINPUTS: lets bibtex find refs/*.bib regardless of working directory ───
$env:BIBINPUTS = ".;$Root\refs\;;$env:BIBINPUTS"

# ── Helper ───────────────────────────────────────────────────────────────────
function Build-Chapter {
    param([string]$QmdPath, [string]$Format, [string]$PdfName, [string]$DocxName, [string]$SubDir = "", [int]$ChapterNum = 0)

    $FullQmd = Join-Path $Root $QmdPath
    if (-not (Test-Path $FullQmd)) {
        Write-Warning "QMD not found: $FullQmd"
        return
    }

    # Resolve journal subdirectory paths
    $PdfDir  = if ($SubDir) { Join-Path $Output $SubDir } else { $Output }
    $DocxDir = if ($SubDir) { Join-Path $Edits  $SubDir } else { $Edits  }
    if (-not (Test-Path $PdfDir))  { New-Item -ItemType Directory $PdfDir  | Out-Null }
    if (-not (Test-Path $DocxDir)) { New-Item -ItemType Directory $DocxDir | Out-Null }
    if (-not (Test-Path $SubmitDir))  { New-Item -ItemType Directory $SubmitDir  | Out-Null }

    if ($Docx) {
        Write-Host "`n==> Building $DocxName (docx) ..." -ForegroundColor Magenta
        $ChapterDir = Join-Path $Root (Split-Path $QmdPath -Parent)

        if ($SubDir -eq "cpt_psp" -or $SubDir -eq "cpt") {
            # ── PSP / CPT submission pipeline ─────────────────────────────────
            # Figures must NOT be embedded — submitted as separate files.
            # Pass 1: render with suppression Lua filter (no image embedding)
            # Pass 2: replace [IMAGE:...] with [Figure N near here]; move legends to end
            $LuaFilter = "$Root\templates\suppress_images_psp.lua"
            quarto render $FullQmd --to docx `
                --output-dir $DocxDir `
                --output $DocxName `
                --lua-filter $LuaFilter
            if ($LASTEXITCODE -eq 0) {
                # Pass 2: format callouts + move figure legends to end
                python "$Root\templates\format_psp_manuscript.py" `
                    "$DocxDir\$DocxName" --chapter $ChapterNum
                # Pass 2b: table left-alignment + mark TOC fields dirty
                python "$Root\templates\fix_docx.py" "$DocxDir\$DocxName"
                # Pass 2c: move embedded title page block before abstract
                python "$Root\templates\move_titlepage.py" "$DocxDir\$DocxName"
                # Pass 2d: generate supplementary table DOCX files
                python "$Root\templates\make_supp_tables.py" --chapter $ChapterNum
                # Pass 2e: copy DOCX into final_submission package folder
                if ($SubDir) {
                    $ChNN = "ch{0:d2}" -f $ChapterNum
                    $PkgDir = Join-Path $SubmitDir "$SubDir\$ChNN"
                    if (-not (Test-Path $PkgDir)) { New-Item -ItemType Directory $PkgDir | Out-Null }
                    Copy-Item "$DocxDir\$DocxName" "$PkgDir\$DocxName" -Force
                }
                Write-Host "    OK  -> $DocxDir\$DocxName" -ForegroundColor Green
            } else {
                Write-Error "    FAILED: $DocxName (exit $LASTEXITCODE)"
            }
        } else {
            # ── Non-PSP pipeline (CTS, CPT, dissertation) ─────────────────────
            # Images are embedded for reviewer convenience.
            # Pass 1: render with standard Lua filter (image placeholders)
            # Pass 2: insert real images from source files
            $LuaFilter = "$Root\templates\suppress_images.lua"
            quarto render $FullQmd --to docx `
                --output-dir $DocxDir `
                --output $DocxName `
                --lua-filter $LuaFilter
            if ($LASTEXITCODE -eq 0) {
                # Pass 2: insert source images at placeholder positions
                python "$Root\templates\insert_docx_images.py" `
                    "$DocxDir\$DocxName" `
                    --chapter-dir $ChapterDir
                # Pass 2b: table left-alignment + mark TOC fields dirty
                python "$Root\templates\fix_docx.py" "$DocxDir\$DocxName"
                # Pass 2c: move embedded title page block before abstract
                python "$Root\templates\move_titlepage.py" "$DocxDir\$DocxName"
                # Pass 2d: generate supplementary table DOCX/CSV files
                python "$Root\templates\make_supp_tables.py" --chapter $ChapterNum
                # Pass 2e: copy DOCX into final_submission package folder
                if ($SubDir) {
                    $ChNN = "ch{0:d2}" -f $ChapterNum
                    $PkgDir = Join-Path $SubmitDir "$SubDir\$ChNN"
                    if (-not (Test-Path $PkgDir)) { New-Item -ItemType Directory $PkgDir | Out-Null }
                    Copy-Item "$DocxDir\$DocxName" "$PkgDir\$DocxName" -Force
                }
                Write-Host "    OK  -> $DocxDir\$DocxName" -ForegroundColor Green
            } else {
                Write-Error "    FAILED: $DocxName (exit $LASTEXITCODE)"
            }
        }
        return
    }

    # ── Journal PDF output ────────────────────────────────────────────────────
    Write-Host "`n==> Building $PdfName ..." -ForegroundColor Cyan

    if ($Draft) {
        quarto render $FullQmd --to pdf `
            --output-dir $PdfDir `
            --output $PdfName `
            -M "format.pdf.template=" `
            -M "format.pdf.documentclass=article"
    } else {
        quarto render $FullQmd --to $Format `
            --output-dir $PdfDir `
            --output $PdfName
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK  -> $PdfDir\$PdfName" -ForegroundColor Green
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
    Write-Host "Cleaning output/ (edits, final_submission, submission, PDFs) ..." -ForegroundColor Yellow
    Get-ChildItem $Output -Include *.pdf,*.tex,*.log -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem $Edits  -Include *.docx -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem $SubmitDir -Recurse -ErrorAction SilentlyContinue |
        Where-Object { -not $_.PSIsContainer } |
        Remove-Item -Force -ErrorAction SilentlyContinue
    $ZipDir = Join-Path $Output "submission"
    Get-ChildItem $ZipDir -Include *.zip -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    # Quarto may leave a PDF in the chapter cwd on failed moves
    Get-ChildItem $Root -Filter "ch*_*.pdf" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "Done. (Close any open PDFs/DOCXs in output/ if files could not be deleted.)" -ForegroundColor Green
    exit 0
}

# ── ExportFigures: PSP-ready TIFF conversion ─────────────────────────────────
if ($ExportFigures) {
    $chapArg = if ($Chapter -gt 0) { "--chapter $Chapter" } else { "" }
    Write-Host "`n==> Exporting PSP-ready TIFF figures ..." -ForegroundColor Cyan
    python "$Root\templates\export_figures_psp.py" $chapArg
    Write-Host "`nDone. TIFFs in: $SubmitDir\cpt_psp\" -ForegroundColor Green
    exit 0
}

# ── Ensure output directories exist ──────────────────────────────────────────
if (-not (Test-Path $Output)) { New-Item -ItemType Directory $Output | Out-Null }
if (-not (Test-Path $Edits))  { New-Item -ItemType Directory $Edits  | Out-Null }

# ── Chapter definitions ───────────────────────────────────────────────────────
# Format: pdf for plain article; wiley-njd-pdf for all Wiley/ASCPT journals
# SubDir: journal subdirectory under output/ and edits/
$Chapters = @(
    @{ Num=1; Qmd="CH_1\ch01_cts.qmd";        Format="wiley-njd-pdf"; Pdf="ch01_cts.pdf";        Docx="ch01_cts_draft.docx";        SubDir="cts"     },
    @{ Num=2; Qmd="CH_2\ch02_psp.qmd";        Format="wiley-njd-pdf"; Pdf="ch02_psp.pdf";        Docx="ch02_psp_draft.docx";        SubDir="cpt_psp" },
    @{ Num=3; Qmd="CH_3\ch03_cts.qmd";        Format="wiley-njd-pdf"; Pdf="ch03_cts.pdf";        Docx="ch03_cts_draft.docx";        SubDir="cts"     },
    @{ Num=4; Qmd="CH_4\ch04_psp.qmd";        Format="wiley-njd-pdf"; Pdf="ch04_psp.pdf";        Docx="ch04_psp_draft.docx";        SubDir="cpt_psp" },
    @{ Num=5; Qmd="CH_5\ch05_cpt.qmd";        Format="wiley-njd-pdf"; Pdf="ch05_cpt.pdf";        Docx="ch05_cpt_draft.docx";        SubDir="cpt"     },
    @{ Num=6; Qmd="CH_6\ch06_conclusion.qmd"; Format="pdf";           Pdf="ch06_conclusion.pdf"; Docx="ch06_conclusion_draft.docx"; SubDir=""        }
)

# ── Build ────────────────────────────────────────────────────────────────────
if ($Chapter -eq 0) {
    foreach ($ch in $Chapters) {
        Build-Chapter -QmdPath $ch.Qmd -Format $ch.Format -PdfName $ch.Pdf -DocxName $ch.Docx -SubDir $ch.SubDir -ChapterNum $ch.Num
    }
} else {
    $target = $Chapters | Where-Object { $_.Num -eq $Chapter }
    if ($null -eq $target) {
        Write-Error "Unknown chapter: $Chapter  (valid: 1-6)"
        exit 1
    }
    Build-Chapter -QmdPath $target.Qmd -Format $target.Format -PdfName $target.Pdf -DocxName $target.Docx -SubDir $target.SubDir -ChapterNum $target.Num
}

if ($IsSubmit) {
    Write-Host "`n==> Exporting PSP/CPT TIFF figures for final submission ..." -ForegroundColor Cyan
    $chapArg = if ($Chapter -gt 0) { "--chapter", "$Chapter" } else { @() }
    python "$Root\templates\export_figures_psp.py" @chapArg
    Write-Host "`nSubmission package ready:" -ForegroundColor Green
    Write-Host "  Drafts  -> $Edits" -ForegroundColor Magenta
    Write-Host "  Package -> $SubmitDir" -ForegroundColor Cyan
} elseif ($Docx) {
    Write-Host "`nAll done. Drafts  -> $Edits" -ForegroundColor Magenta
    Write-Host "         Packages -> $SubmitDir" -ForegroundColor Cyan
} else {
    Write-Host "`nAll done. PDFs in: $Output" -ForegroundColor Green
}
