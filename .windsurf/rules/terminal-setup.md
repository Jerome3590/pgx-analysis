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

## WSL Environment Setup (one-time, already configured)

### AWS CLI
- AWS CLI v2 is installed in WSL (`/usr/bin/aws`)
- Credentials symlinked from Windows: `~/.aws/credentials → /mnt/c/Projects/credentials`
- Config symlinked from Windows: `~/.aws/config → /mnt/c/Projects/config`
- All profiles available: `default`, `pgx`, `imat`, `mushin`, `cana`, etc.
- **`aws` commands work directly in WSL/Windsurf `run_command` — no `pwsh.exe` wrapper needed**
- Verify: `aws sts get-caller-identity` or `aws sts get-caller-identity --profile pgx`

### Python
- WSL has Python 3.10.12 at `/usr/bin/python3`
- `boto3` is installed (sufficient for deployment scripts)
- `requirements.txt` requires Python 3.11 — full install only works on EC2 (`/home/pgx3874/jupyter-env/bin/python3.11`)
- Dashboard deployment scripts (`sync_frontend_to_s3.py` etc.) only need `boto3` — already available

### Windows username vs WSL username
- Windows profile: `C:\Users\jerom` (username `jerom`)
- WSL username: `jerome3590`
- Windows drives: `/mnt/c/`

## When to use WSL vs PowerShell 7

| Task | Use |
|---|---||
| `aws` CLI commands | WSL ✅ (credentials symlinked) |
| `python3` deployment scripts | WSL ✅ (boto3 installed) |
| Puppeteer / Jest tests | WSL ✅ (`npx jest` — Linux Chromium at `~/.cache/puppeteer/`) |
| `git` | WSL ✅ |
