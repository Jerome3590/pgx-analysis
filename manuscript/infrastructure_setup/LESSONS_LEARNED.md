# Build Infrastructure — Lessons Learned

Issues encountered getting Quarto → XeLaTeX → PDF working with MDPI and Wiley NJD
journal class files on Windows / TinyTeX 2026. Organized by category.

> **System context:** Windows 11, Quarto 1.7+, TinyTeX 2026, XeLaTeX, R 4.5.2 at
> `C:\Program Files\R\R-4.5.2\bin\Rscript.exe`

---

## 1. Quarto YAML Reserved Fields

These field names are **reserved by Quarto 1.7** and will throw YAML validation
errors if used with string values when Quarto expects an object or a file path.

| Field used | Error | Fix |
|:-----------|:------|:----|
| `journal: "Journal Name"` | `Field "journal" must be an object` | Rename to `target-journal:` |
| `abbreviations: >` | `withBinaryFile: invalid argument` | Rename to `manuscript-abbreviations:` — Quarto 1.7 treats `abbreviations:` as a path to a glossary file |
| `articletype: "review"` | Potential conflict | Rename to `target-articletype:` |

**Rule:** Before adding any top-level YAML key, check if it is a known Quarto
metadata field. When in doubt, prefix with `target-` or `manuscript-`.

---

## 2. Pandoc Template Syntax

These issues arise in `.tex` Pandoc templates (passed via `template:` in the QMD).

### 2a. Bare `$` in LaTeX comments

Pandoc parses the entire template file for `$variable$` syntax — including comment
lines. Any `$` that is NOT part of a `$variable$` expression must be escaped as `$$`.

```latex
%% WRONG — Pandoc tries to parse $TEXINPUTS as a variable name:
%%   export TEXINPUTS=".:path//:$TEXINPUTS"
%%   $env:TEXINPUTS = ".;C:\..."

%% CORRECT — escape with $$:
%%   export TEXINPUTS=".:path//:$$TEXINPUTS"
%%   $$env:TEXINPUTS = ".;C:\..."
```

### 2b. LaTeX math superscripts in template code

The LaTeX math delimiter `$^{n}$` conflicts with Pandoc template variable
syntax. Pandoc sees `$^{` and tries to parse a variable named `^{`.

```latex
%% WRONG — Pandoc template parse error:
  $^{$it.number$}$\quad $it.name$

%% CORRECT — use \textsuperscript:
  \textsuperscript{$it.number$}\quad $it.name$
```

### 2c. `$header-includes$` ordering matters

Quarto injects float infrastructure (including `\newcommand*\listoflistings{}`)
via `$header-includes$`. Any compatibility shims that need to fire BEFORE that
code must be placed in the template **before** `$if(header-includes)$$header-includes$$endif$`.

---

## 3. MDPI Class (`mdpi.cls`) — `bmic_jpm_template.tex`

### 3a. `\listoflistings` pre-defined by newfloat

`mdpi.cls` calls `\DeclareFloatingEnvironment[]{listing}` via the `newfloat`
package. The `newfloat` package automatically creates `\listoflistings` (prefixed
`listof` + pluralized name). Quarto then injects `\newcommand*\listoflistings{...}`
via `$header-includes$`, which fails because the command already exists.

**Fix:** Add `\let\listoflistings\relax` in the template **before**
`$if(header-includes)$` to make the command redefine-able:

```latex
%% mdpi.cls pre-defines \listoflistings via newfloat; undefine so Quarto can redefine it
\let\listoflistings\relax
$if(header-includes)$
$header-includes$
$endif$
```

### 3b. `\history{}` not defined in bundled cls version

Newer MDPI templates use `\history{Received: ...; Accepted: ...; Published: ...}`.
The bundled `mdpi.cls` uses separate commands: `\datereceived{}`, `\dateaccepted{}`,
`\datepublished{}`.

**Fix:** Add `\providecommand{\history}[1]{}` in the template preamble.

### 3c. `\abbreviations` — two arguments, and body-only

The cls defines:
```latex
\newcommand{\abbreviations}[2]{\vspace{12pt}\noindent{\textbf{#1}}{...#2...}}
```

Two issues:
1. Takes **two** arguments: `\abbreviations{Title}{content}` — not one.
2. Uses `\vspace`/`\noindent` — it is a **body command**, not preamble metadata.
   Calling it before `\begin{document}` causes `! Missing \begin{document}`.

**Fix:** Move `\abbreviations` call to inside the document body (after `$body$`,
before references), and pass both arguments:

```latex
$if(manuscript-abbreviations)$
\abbreviations{Abbreviations}{$manuscript-abbreviations$}
$endif$
```

### 3e. `\@datepublished` never initialized

`mdpi.cls` defines `\newcommand{\datepublished}[1]{\gdef\@datepublished{#1}}` at
line 649 but unlike every other date command, **forgets** the preceding
`\def\@datepublished{}` initializer. `\maketitle` then calls
`\ifthenelse{\equal{\@datepublished}{}}` and crashes with
`! Undefined control sequence` at `\begin{document}`.

**Fix:** Add to the template preamble:
```latex
\makeatletter\def\@datepublished{}\makeatother
```

### 3f. Non-numeric placeholders in `\pubvolume` / `\issuenum` / `\articlenumber`

`mdpi.cls` processes these three fields through `\cutdigits{}`, which strips
non-digit characters. Passing `"xx"` / `"x"` produces an empty result; LaTeX
then crashes with `! A number should have been here; I inserted '0'` midway
through the title page.

**Fix:** Use integer defaults in the template:
```latex
\pubvolume{$if(pubvolume)$$pubvolume$$else$1$endif$}
\issuenum{$if(issuenum)$$issuenum$$else$1$endif$}
\articlenumber{$if(articlenumber)$$articlenumber$$else$1$endif$}
```

### 3h. Bibliography file names must not contain underscores

`natbib`/bibtex receives bibliography paths from the `.aux` file. LaTeX's font
encoding machinery escapes `_` as `\protect\T1\textunderscore` when writing to
the aux file, producing a path bibtex cannot resolve.

**Fix:** Use hyphens in all `.bib` filenames:
```
refs/bmic_jpm.bib  →  refs/bmic-jpm.bib
refs/cpt_psp.bib   →  refs/cpt-psp.bib
```
Update all `bibliography:` entries in every `.qmd` accordingly.

### 3i. `\bibliographystyle` must not use a directory-prefixed path

`mdpi.cls` originally called `\bibliographystyle{Definitions/mdpi}`. When TeX
writes this to the `.aux` file, the long path wraps across a line, producing:
```
\bibstyle{Definitions/mdpi
                          }
```
bibtex cannot parse the multi-line token and reports
`I found no style file`. The `BSTINPUTS` environment variable is ignored.

**Fix:** Patch `templates/Definitions/mdpi.cls` lines 396/399/402 to use bare
names (`mdpi_apacite`, `mdpi_chicago`, `mdpi`) and set `BSTINPUTS` so bibtex
finds the `.bst` files by name:
```powershell
$env:BSTINPUTS = ".;$Root\templates\Definitions\;;"
```

### 3j. Remove `biblio-style:` from MDPI chapter YAML

`mdpi.cls` calls `\bibliographystyle{mdpi}` internally. If the QMD also has
`biblio-style: mdpi`, Quarto emits a second `\bibliographystyle`, causing
`Illegal, another \bibstyle command` in bibtex.

**Fix:** Remove `biblio-style:` from `CH_1` and `CH_5` format blocks entirely.
The cls handles it.

### 3k. Remove `$if(natbib)$` reference block from template

When `cite-method: natbib` is set in Quarto, `\bibliographystyle{}` and
`\bibliography{}` are already emitted inside `$body$`. Adding them again via a
`$if(natbib)$` block in the template produces duplicates.

**Fix:** Delete the natbib/biblatex reference block from `bmic_jpm_template.tex`.
Add a comment explaining why:
```latex
%% References emitted by Quarto inside $body$ — do NOT add \bibliography here.
```

### 3g. `\keyword` loop syntax

Inside `$for(keywords)$`, the current item is `$keywords$` (same name as the
loop variable), not `$it$`. Verified working:

```latex
\keyword{$for(keywords)$$keywords$$sep$; $endfor$}
```

---

## 4. Wiley NJD Class (`WileyNJDv5.cls`) — Wiley extension

All fixes applied directly to `_extensions/ramiromagno/wiley-njd/wiley-njd-v5/WileyNJDv5.cls`.

### 4a. `\reserveinserts{28}` crashes under XeLaTeX

`WileyNJDv5.cls` loads `etex.sty` then calls `\reserveinserts{28}`. Under
XeLaTeX, eTeX is already active, so `etex.sty` detects this, prints a warning
("Extended allocation already in use"), and **skips defining `\reserveinserts`**.
The bare call then fails with `! Undefined control sequence`.

**Fix:** Guard the call:
```latex
\usepackage{etex}%
\ifdefined\reserveinserts\reserveinserts{28}\fi
```

### 4b. `\usepackage[english]{babel}` crashes under babel 2026

babel 2026 (v26.4) changed its option API for XeLaTeX. The option `english`
is no longer accepted directly; it must be `main=english`. However, for a
monolingual English manuscript, babel is not needed at all.

**Fix:** Comment out the babel load:
```latex
%\usepackage[main=english]{babel}% disabled: babel 2026 API incompatible with XeLaTeX
```

### 4c. `\tightlist` undefined

`WileyNJDv5.cls` does not define `\tightlist`, which Pandoc generates for
compact bullet/numbered lists.

**Fix:** Add to `_extensions/ramiromagno/wiley-njd/partials/pandoc.tex`:
```latex
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
```

### 4d. `longtable` package commented out

`%\RequirePackage{longtable}%` is commented out in the cls. Pandoc converts
all markdown tables to `longtable` by default.

**Fix:** Uncomment: `\RequirePackage{longtable}%`

---

## 5. TinyTeX / LaTeX Distribution

### 5a. TinyTeX 2025 → 2026 upgrade

After a new TeX Live year is released, `tlmgr` 2025 refuses to install packages
from the 2026 repository. Error: `Local TeX Live (2025) is older than remote repository (2026)`.

**Fix:** Upgrade via R:
```r
# R must be installed; path on this machine:
# C:\Program Files\R\R-4.5.2\bin\Rscript.exe
Rscript.exe -e "tinytex::install_tinytex(force = TRUE, repository = 'illinois')"
```

### 5b. `mathastext.sty` missing

`WileyNJDv5.cls` requires `mathastext`, which is not installed by default in
TinyTeX. After upgrading to 2026, install it:

```r
Rscript.exe -e "tinytex::tlmgr_install('mathastext')"
```

### 5c. `fmtutil-sys --all` error after auto-install

Quarto auto-installs missing LaTeX packages mid-render. Occasionally the
post-install format rebuild (`fmtutil-sys --all`) fails with
`[non-error-thrown] Problem running fmtutil-sys`. The packages ARE installed;
the error is in the rebuild hook.

**Fix:** Simply rerun the render command — the second attempt succeeds because
the packages are already in place.

---

## 6. Figure Formats

### 6a. PDF vs PNG vs JPEG

Both `\includegraphics{fig.pdf}` and `\includegraphics{fig.png}` work with
XeLaTeX via the `graphicx` package (loaded by both `mdpi.cls` and `WileyNJDv5.cls`).

| Figure type | Format | Notes |
|:------------|:-------|:------|
| matplotlib / seaborn plots | `.pdf` | Use `savefig('fig.pdf')` |
| R ggplot2 | `.pdf` | Use `ggsave('fig.pdf', device='pdf')` |
| SHAP summary / beeswarm | `.pdf` | Vector scales cleanly |
| Architecture diagrams (draw.io, PPT) | `.pdf` | Export as PDF from source app |
| Network graphs (FP-Growth, BupaR) | `.pdf` preferred, `.png` @ 300 DPI fallback | |
| Dashboard screenshots | `.png` @ ≥300 DPI | `savefig(..., dpi=300, format='png')` |

**Never use JPEG** for scientific figures — lossy compression produces visible
artifacts and fails journal production QC checks.

### 6b. Why PDF is preferred over PNG

- **Vector:** scales to any size without pixelation — essential for two-column
  Wiley layout where figures are often resized at production
- **Required or strongly preferred** by MDPI and Wiley for plots/diagrams
- **Smaller file size** than 300 DPI PNG for most line-art figures

### 6c. DPI for PNG figures

If PDF is not possible, minimum **300 DPI** for print. 600 DPI for fine
line-art (SHAP beeswarm with small dot density). Verify with:
```python
from PIL import Image
img = Image.open('fig.png')
print(img.info['dpi'])   # must be (300.0, 300.0) or higher
```

---

## 7. General Build Rules (Windows)

- Always set `TEXINPUTS` before rendering MDPI chapters so xelatex finds
  `templates/Definitions/mdpi.cls`:
  ```powershell
  $env:TEXINPUTS = ".;$PWD\templates\;;"
  quarto render CH_1\ch01_bmic.qmd --to pdf --output-dir output
  ```
- Wiley chapters use the `wiley-njd-pdf` format (bundled extension); no
  `TEXINPUTS` override needed.
- Use `.\build.ps1` or `make all` to automate — both scripts set `TEXINPUTS`.
- After any TinyTeX upgrade, run one chapter first to trigger auto-installs,
  then rebuild if `fmtutil-sys` fails.
