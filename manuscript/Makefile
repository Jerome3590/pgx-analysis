# Manuscript Build Makefile
# Builds journal-ready PDFs for all dissertation chapters via Quarto.
#
# Usage:
#   make all          - build every chapter (journal PDFs → output/)
#   make ch01         - build Chapter 1 only
#   make docx         - build all chapters as Word .docx → edits/ (for advisor)
#   make docx-full    - full dissertation .docx → edits/ (timestamped filename)
#   make docx-ch01    - single chapter to edits/
#   make clean        - remove compiled PDFs from output/ and .docx from edits/
#   make draft        - build all chapters in draft mode (plain article class)
#
# Prerequisites:
#   - Quarto CLI:  https://quarto.org/docs/get-started/
#   - LaTeX:       TinyTeX (quarto install tinytex) | MiKTeX | TeXLive
#   - Journal cls: templates/Definitions/mdpi.cls is already bundled.
#                  WileyNJD-v2.cls: see templates/INSTALL.md

OUTPUT := output
EDITS  := edits

# TEXINPUTS: lets xelatex find Definitions/mdpi.cls from any chapter dir.
export TEXINPUTS := .:$(CURDIR)/templates//:$(TEXINPUTS)
# BSTINPUTS: lets bibtex find Definitions/mdpi.bst from any chapter dir.
export BSTINPUTS := .:$(CURDIR)/templates/Definitions//:$(BSTINPUTS)

.PHONY: all ch01 ch02 ch03 ch04 ch05 ch06 clean draft \
        docx docx-ch01 docx-ch02 docx-ch03 docx-ch04 docx-ch05 docx-ch06

all: ch01 ch02 ch03 ch04 ch05 ch06

# ── Chapter 1: SQLR → BMIC / Journal of Personalized Medicine ────────────────
ch01:
	quarto render CH_1/ch01_bmic.qmd --to pdf \
	  --output-dir $(OUTPUT) \
	  --output ch01_bmic_jpm.pdf

# ── Chapter 2: Methodology → CPT: Pharmacometrics & Systems Pharmacology ──────
ch02:
	quarto render CH_2/ch02_psp.qmd --to wiley-njd-pdf \
	  --output-dir $(OUTPUT) \
	  --output ch02_psp.pdf

# ── Chapter 3: Opioid ED → Clinical and Translational Science ────────────────
ch03:
	quarto render CH_3/ch03_cts.qmd --to wiley-njd-pdf \
	  --output-dir $(OUTPUT) \
	  --output ch03_cts.pdf

# ── Chapter 4: Polypharmacy → CPT: Pharmacometrics & Systems Pharmacology ─────
ch04:
	quarto render CH_4/ch04_psp.qmd --to wiley-njd-pdf \
	  --output-dir $(OUTPUT) \
	  --output ch04_psp.pdf

# ── Chapter 5: PGx Dashboard → BMIC / Journal of Personalized Medicine ────────
ch05:
	quarto render CH_5/ch05_bmic.qmd --to pdf \
	  --output-dir $(OUTPUT) \
	  --output ch05_bmic_jpm.pdf

# ── Chapter 6: Conclusion (dissertation synthesis) ────────────────────────────
ch06:
	quarto render CH_6/ch06_conclusion.qmd --to pdf \
	  --output-dir $(OUTPUT) \
	  --output ch06_conclusion.pdf

# ── Draft mode: renders all chapters without journal templates ─────────────────
draft:
	for qmd in CH_1/ch01_bmic.qmd CH_2/ch02_psp.qmd CH_3/ch03_cts.qmd \
	            CH_4/ch04_psp.qmd CH_5/ch05_bmic.qmd CH_6/ch06_conclusion.qmd; do \
	  quarto render $$qmd --to pdf \
	    -M "format.pdf.template=" \
	    -M "format.pdf.documentclass=article" \
	    --output-dir $(OUTPUT); \
	done

# ── Word .docx for advisor review (Google Docs compatible) ───────────────────
docx: docx-ch01 docx-ch02 docx-ch03 docx-ch04 docx-ch05 docx-ch06

# Full dissertation (matches build.ps1 -Full -Docx)
docx-full:
	mkdir -p $(EDITS)
	quarto render full_dissertation/full_dissertation.qmd --to docx \
	  --output-dir $(EDITS) --output dissertation_dixon_$$(date +%Y%m%d_%H%M%S).docx

docx-ch01:
	mkdir -p $(EDITS)
	quarto render CH_1/ch01_bmic.qmd --to docx \
	  --output-dir $(EDITS) --output ch01_bmic_draft.docx

docx-ch02:
	mkdir -p $(EDITS)
	quarto render CH_2/ch02_psp.qmd --to docx \
	  --output-dir $(EDITS) --output ch02_psp_draft.docx

docx-ch03:
	mkdir -p $(EDITS)
	quarto render CH_3/ch03_cts.qmd --to docx \
	  --output-dir $(EDITS) --output ch03_cts_draft.docx

docx-ch04:
	mkdir -p $(EDITS)
	quarto render CH_4/ch04_psp.qmd --to docx \
	  --output-dir $(EDITS) --output ch04_psp_draft.docx

docx-ch05:
	mkdir -p $(EDITS)
	quarto render CH_5/ch05_bmic.qmd --to docx \
	  --output-dir $(EDITS) --output ch05_bmic_draft.docx

docx-ch06:
	mkdir -p $(EDITS)
	quarto render CH_6/ch06_conclusion.qmd --to docx \
	  --output-dir $(EDITS) --output ch06_conclusion_draft.docx

# ── Clean compiled output ──────────────────────────────────────────────────────
clean:
	rm -f $(OUTPUT)/*.pdf $(OUTPUT)/*.tex $(OUTPUT)/*.log
	rm -f $(EDITS)/*.docx
