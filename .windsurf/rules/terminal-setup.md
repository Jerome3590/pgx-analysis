---
trigger: always_on
---

# Terminal & Shell Execution Rules

## Windows Terminal Configuration
- **Default integrated terminal**: WSL (for pipeline/Python scripts)
- **PowerShell profile**: PowerShell 7 (`C:\Program Files\PowerShell\7\pwsh.exe`)
- **Automation profile**: `pwsh.exe -NoProfile`
- **External terminal exec**: `C:\Program Files\PowerShell\7\pwsh.exe`

## Script Execution via Cascade (run_command)
Cascade executes commands through **WSL bash**, not PowerShell. For `.ps1` scripts, always invoke explicitly:
```bash
pwsh.exe -File "/mnt/c/Projects/pgx-analysis/path/to/script.ps1"
```

## User-level settings (`%APPDATA%\Windsurf\User\settings.json`)
Must be kept in sync with `.vscode/settings.json` for:
- `terminal.integrated.defaultProfile.windows`
- `terminal.integrated.profiles.windows > PowerShell > path`
- `terminal.integrated.automationProfile.windows > path`
- `terminal.external.windowsExec`

All four should point to `C:\Program Files\PowerShell\7\pwsh.exe`.

## Project workspace settings (`.vscode/settings.json`)
Tracked in git. Workspace default terminal is **WSL** (pipeline uses bash/Python).
PowerShell 7 profile is available as a named profile for `.ps1` test scripts.
