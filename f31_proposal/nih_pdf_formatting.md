

## Page size, margins, layout

- Paper size: Standard US letter, 8.5 × 11 inches.[^2][^3]
- Margins: At least 0.5 inch on all sides (top, bottom, left, right). No text, figures, or page numbers in the margins.[^3][^5][^2]
- Columns: Single‑column layout only; multi‑column formatting is discouraged.[^6][^5]
- Headers/footers: Do not add your own headers or footers; NIH systems will add page numbers, PI name, and document title.[^5][^2][^6]
- File type: PDF attachments only for all narrative sections (Specific Aims, Research Strategy, Training Plan, etc.).[^4][^6]

In LaTeX/Quarto, that means: geometry with 0.5in margins, no custom headers/footers, and avoid multi‑column environments.

## Font, size, density, spacing

NIH cares about legibility, type density, and line spacing more than the exact TeX engine.

- Allowed fonts (recommended): Arial, Georgia, Helvetica, or Palatino Linotype.
- Font size: 11 pt or larger for main text.
    - You may use a Symbol font for Greek/special characters, but still at a size consistent with 11 pt.[^2]
- Type density: No more than 15 characters per linear inch (including spaces).
- Line spacing: No more than six lines of type per vertical inch.[^1][^3][^5]
    - NIH allows slightly more white space than strict single‑spacing; many institutional guides recommend “at least single, not compressed” spacing so you comfortably stay under six lines/inch.

In LaTeX/Quarto, the safe pattern is: 11 pt base font, one of the NIH fonts via `mainfont`, and avoid `\small` or tighter line‑spreading for main narrative text.

## Figures, tables, and images

- Figures, graphics, charts, and tables must be legible when printed on an 8.5 × 11 page at 100% scale.
- Figures should follow the same page size and margin rules; they count toward page limits when embedded in sections like Research Strategy.
- Font in figures can be smaller than 11 pt, but must still be readable at 100% zoom in the PDF.
- NIH suggests image sizes on the order of ~1200 × 1500 pixels and use of standard formats (e.g., PNG, JPEG) so they render cleanly in the compiled PDF.

Practically, for LaTeX: use vector graphics where possible (PDF, EPS) or high‑resolution raster, and keep axis labels and legend text reasonably large.

## File naming and general attachment rules

- Attachments must be PDF, with file names of 50 characters or fewer, using only: letters, numbers, underscore, hyphen, space, and period. No `&`, `%`, `#`, `/`, etc.[^8]
- Headings within the text are encouraged for readability; they do not violate any rules.
- Hyperlinks/URLs: allowed only where specifically permitted by the FOA and SF424 guidance; NIH generally discourages using URLs to bypass page limits.

For Quarto, you can control the final PDF name and keep it FOA‑appropriate; inside the document, just avoid exotic characters in filenames and keep section headings normal.

## Minimal Quarto/LaTeX setup (conceptual)

Translating these rules into a Quarto PDF config, our YAML can follow these constraints:

- Use `geometry: "margin=0.5in"` for margins.
- Use an allowed main font at 11 pt.
- Avoid LaTeX packages or settings that add headers/footers or multi‑column text.