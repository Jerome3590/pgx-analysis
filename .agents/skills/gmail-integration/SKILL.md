---
name: gmail-integration
description: Interact with Gmail across multiple accounts (personal/SurgicalED, VCU, Canal LLC, Mushin) using the dedicated Gmail MCP server. Use when searching emails, reading threads, sending messages, managing labels, or handling attachments.
---

# Gmail MCP Integration

Antigravity integrates with the user's Gmail accounts via the local Model Context Protocol (MCP) server.

## 1. Account Routing & Identity Mapping

Always route tasks to the specific inbox MCP server based on context and recipient:

| Account Slug | Target Email | MCP Server Name | Primary Context & Use Cases |
|---|---|---|---|
| `jerome_dixon90` | `jerome.dixon90@gmail.com` | `gmail-jerome-dixon90` | SurgicalED / `vr-sling` / Dr. Siff / Personal Gmail |
| `dixonrj_vcu` | `dixonrj@vcu.edu` | `gmail-dixonrj-vcu` | VCU academic, comps, grant communications |
| `jdixon_canallc` | `jdixon@canallc.com` | `gmail-jdixon-canallc` | Canal LLC client & contract operations |
| `jerome_mushinsolutions` | `jerome@mushinsolutions.com` | `gmail-jerome-mushinsolutions` | Mushin Solutions ops / AWS admin identity |

## 2. Security & Operational Rules
- **Never open browser Gmail** when MCP servers are connected; use the MCP tools directly.
- **Strict Token Isolation**: Each account has an independent token stored in `%USERPROFILE%\.gmail-mcp\tokens\<slug>.json`. Never reuse or cross-link token files.
- **Zero Credential Exposure**: Never print, copy, or commit token JSON or OAuth client secret contents.
- **Matching Server Selection**: Before invoking Gmail operations, verify you are calling the tool mapped to the correct inbox (`gmail-jerome-dixon90`, `gmail-dixonrj-vcu`, `gmail-jdixon-canallc`, or `gmail-jerome-mushinsolutions`).

## 3. Re-Authentication & Token Recovery
If an MCP tool returns `401 Unauthorized` or indicates token expiration:
1. Do not fall back to another account.
2. Direct the user to run the interactive OAuth script:
   ```powershell
   powershell -File C:\Projects\project_utility_scripts\scripts\gmail_mcp\auth_account.ps1 <slug>
   ```
3. To list current token status across all accounts:
   ```powershell
   powershell -File C:\Projects\project_utility_scripts\scripts\gmail_mcp\auth_account.ps1 -List
   ```

## 4. Common Workflows
- **Search Emails**: Query messages using standard Gmail search syntax (`from:`, `to:`, `subject:`, `after:`, `has:attachment`).
- **Read Threads**: Fetch message details and parse headers/body.
- **Drafts & Sending**: Create drafts for review before sending external communications.
