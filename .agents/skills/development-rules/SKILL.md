---
name: development-rules
description: Universal development guidelines, architectural patterns, test generation standards, and workspace workflows migrated from Cursor rules. Use when writing code, structuring modules, running tests, or planning refactors.
---

# Development Rules & Workspace Guidelines
*(Migrated from Cursor `.cursorrules` configuration)*

## 1. Skill Metadata
- **Skill Name**: `development-rules`
- **Trigger**: Universal coding, architectural, and workflow directives for Antigravity agents.
- **Locations**:
  - Global: `~/.gemini/config/skills/development-rules/SKILL.md`
  - Workspace: `.agents/skills/development-rules/SKILL.md`

---

## 2. Core Architecture & Coding Guidelines
- **Architecture Separation**:
  - Maintain a flat, modular code structure separating UI components, state management, and data access.
  - Keep API schemas and contracts strictly defined before extending implementation.
- **Code Consistency & Patterns**:
  - Require explicit type definitions (TypeScript / Python type hints) across all interfaces.
  - Follow existing repository formatting and linting conventions.
  - Write sample code or initial patterns for the agent to replicate when introducing new modules.

---

## 3. Automated Testing & Verification
- **Test Generation & Execution**:
  - Generate comprehensive unit and integration test suites alongside feature implementation.
  - Run tests deterministically via the terminal or runner before finalizing code changes.
- **Verification Walkthroughs**:
  - Produce walkthrough artifacts detailing changed files, test output, and visual or functional verification.

---

## 4. Project Documentation & Context Preservation
- **Directory Context Files**:
  - Maintain a `CONTEXT.md` or `README.md` file in each major folder describing module purpose and dependencies.
  - Keep context documentation updated whenever underlying source files change.
- **Task & Plan Artifacts**:
  - Track multi-step initiatives using structured `task_list.md` and `implementation_plan.md` artifacts.

---

## 5. Execution Permissions & Sub-Agent Rules
- **Execution Controls**:
  - Non-destructive commands (linters, static checks, test runners) can execute automatically.
  - Structural modifications, external network requests, or database migrations require explicit review.
- **Sub-Agent Delegation**:
  - Break complex refactoring or multi-file builds into parallel sub-agents (e.g., separate front-end, back-end, and QA sub-agents).
