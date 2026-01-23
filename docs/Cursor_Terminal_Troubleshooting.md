# Cursor Terminal Command Execution Troubleshooting Guide

## Issue
Terminal commands executed through Cursor's AI agent are timing out or failing to spawn, even though commands work fine in the user's own terminal.

## Root Cause Analysis

### Likely Causes
1. **Cursor AI Agent Sandbox Restrictions**
   - Commands may run in a restricted sandbox environment
   - Network access may be blocked
   - Git writes require explicit permissions
   - File system access may be limited

2. **System-Level Cursor Settings**
   - Settings are not in workspace files (`.vscode/settings.json`)
   - Settings are in Cursor's application-level configuration
   - May be controlled by Cursor's AI agent permissions

3. **Timeout Configurations**
   - Commands may have execution time limits
   - Large repositories may trigger timeouts
   - Complex commands may exceed limits

4. **Workspace Path Issues**
   - Workspace path changed (noted in system messages)
   - Path resolution may be affected

## Configuration Files Checked

### ✅ Files Found
- `.cursorrules` - Contains coding rules, not terminal settings
- `.vscode/settings.json` - Created with terminal settings (may not be used by Cursor)
- `.gitignore` - No terminal-related exclusions

### ❌ Files Not Found
- `.cursor/settings.json` - Does not exist
- Cursor-specific configuration files - Not found in workspace

## Settings to Check in Cursor

### 1. Open Cursor Settings
- Press `Ctrl+,` (Windows/Linux) or `Cmd+,` (Mac)
- Or: File → Preferences → Settings

### 2. Search for These Settings

#### Terminal Settings
- `terminal.integrated.defaultProfile.windows`
- `terminal.integrated.profiles.windows`
- `terminal.integrated.enablePersistentSessions`
- `terminal.integrated.commandsToSkipShell`

#### AI Agent / Command Execution Settings
- `cursor.agent.commandTimeout`
- `cursor.agent.sandbox.enabled`
- `cursor.agent.permissions`
- `cursor.agent.networkAccess`
- `cursor.agent.gitAccess`

#### Timeout Settings
- `cursor.commandTimeout`
- `cursor.executionTimeout`
- Any settings with "timeout" in the name

#### Permission Settings
- `cursor.permissions.git`
- `cursor.permissions.network`
- `cursor.permissions.filesystem`

### 3. Check Workspace Settings
Look for a `.cursor/` directory in your workspace or user settings directory:
- Windows: `%APPDATA%\Cursor\User\settings.json`
- Mac: `~/Library/Application Support/Cursor/User/settings.json`
- Linux: `~/.config/Cursor/User/settings.json`

## Workarounds

### Option 1: Use Your Terminal Directly
For git operations and other commands, use your own terminal instead of the AI agent:
```bash
# In your terminal (Git Bash, PowerShell, etc.)
cd C:\Projects\pgx-analysis
git add 2_create_cohort/sample_hcg_values.py
git commit -m "Add HCG verification scripts"
git push
```

### Option 2: Request Explicit Permissions
When running commands through the AI agent, explicitly request permissions:
- `required_permissions: ['git_write']` for git operations
- `required_permissions: ['network']` for network operations
- `required_permissions: ['all']` to disable sandbox (use with caution)

### Option 3: Break Down Commands
Instead of complex commands, break them into smaller steps:
- Instead of: `cd dir && git add file && git commit`
- Use: Separate commands or file operations directly

### Option 4: Use File Operations Directly
For file operations, use the tool's file read/write capabilities instead of terminal commands:
- ✅ `read_file()` - Read files
- ✅ `write()` - Write files
- ✅ `search_replace()` - Edit files
- ❌ `cat`, `echo`, `git add` - Use tools instead

## Created Configuration Files

### `.vscode/settings.json`
Created with the following settings (may help if Cursor respects VS Code settings):
- Terminal profile configurations
- Python and Git path settings
- File watcher exclusions

**Note**: Cursor may not use VS Code settings. Check Cursor's own settings.

## Testing Commands

### Simple Test Commands
Try these in order of complexity:

1. **Basic command:**
   ```bash
   echo "test"
   ```

2. **Git version:**
   ```bash
   git --version
   ```

3. **Python version:**
   ```bash
   python --version
   ```

4. **Git status:**
   ```bash
   git status --short
   ```

### Expected Behavior
- ✅ Commands should execute within 1-2 seconds
- ✅ Output should be returned
- ❌ Timeouts or "failed to spawn" errors indicate the issue

## Next Steps

1. **Check Cursor Settings UI**
   - Open Settings (`Ctrl+,`)
   - Search for "terminal", "timeout", "sandbox", "agent"
   - Document any relevant settings found

2. **Check Cursor Documentation**
   - Look for AI agent configuration documentation
   - Check for sandbox/permission settings
   - Look for timeout configurations

3. **Contact Cursor Support**
   - If settings are not accessible
   - If issue persists after configuration changes
   - Report the command execution failures

4. **Use Workarounds**
   - Continue using your own terminal for git operations
   - Use file operations directly instead of terminal commands
   - Break complex operations into smaller steps

## Related Files

- `.vscode/settings.json` - Workspace terminal settings (created)
- `.cursorrules` - Coding rules (not terminal-related)
- `2_create_cohort/sample_hcg_values.py` - HCG verification script
- `2_create_cohort/verify_hcg_codes.py` - HCG code verification script

## Additional Resources

- [Cursor Documentation](https://cursor.sh/docs)
- [VS Code Terminal Settings](https://code.visualstudio.com/docs/terminal/basics)
- Git configuration: Check `.git/config` for repository-specific settings
