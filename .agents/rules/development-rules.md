# Development Rules & Workspace Guidelines

## 1. Architecture Separation
- Maintain a flat, modular code structure separating UI components, state management, and data access.
- Keep API schemas and contracts strictly defined before extending implementation.

## 2. Code Consistency & Patterns
- Require explicit type definitions (TypeScript / Python type hints) across all interfaces.
- Follow existing repository formatting and linting conventions.
- Write sample code or initial patterns for the agent to replicate when introducing new modules.

## 3. Automated Testing & Verification
- Generate comprehensive unit and integration test suites alongside feature implementation.
- Run tests deterministically via the terminal or runner before finalizing code changes.
- Produce walkthrough artifacts detailing changed files, test output, and visual or functional verification.

## 4. Project Documentation & Context Preservation
- Maintain a `CONTEXT.md` or `README.md` file in each major folder describing module purpose and dependencies.
- Keep context documentation updated whenever underlying source files change.
- Track multi-step initiatives using structured `task_list.md` and `implementation_plan.md` artifacts.

## 5. Execution Permissions & Sub-Agent Delegation
- Non-destructive commands (linters, static checks, test runners) can execute automatically.
- Structural modifications, external network requests, or database migrations require explicit review.
- Break complex refactoring or multi-file builds into parallel sub-agents.


## 6. PDF Generation & Type 3 Font Elimination Standards (`Type3: 0` / IDE `pdf.js` Compatibility — `RC-31`)
- **Zero `/Subtype /Type3` Fonts Mandate (`Type3: 0`)**: Every automated PDF build script must inspect the generated PDF binary after `page.pdf()` and verify `len(re.findall(r"/Subtype\s+/Type3", raw_ascii)) == 0`.
- **Static TrueType Fonts Only**: Never use Windows Variable Fonts (`"Cascadia Code"`), Color Emojis (`"Segoe UI Emoji"`), or dynamic MathJax CHTML webfonts in PDF print stylesheets; always specify `"Segoe UI", Arial, sans-serif` for body text and `"Consolas", "Courier New", monospace` for code blocks.
- **No Synthetic Bold on `Cambria Math`**: Verify all non-ASCII characters against the TrueType `cmap` tables of `segoeui.ttf` and `consola.ttf` (`SAFE_CPS`) and render mathematical expressions using native HTML/CSS (`<span class="frac">`, `.overline`, `.hat`, `<sub>`, `<sup>`).
