# Gmail MCP - One Token Store Per Inbox

Do **not** use the browser to open Gmail when these MCP servers are connected. Route by **email**, then call that server only.

| Email | MCP Server | Token File (do not read) | Use |
|---|---|---|---|
| `jerome.dixon90@gmail.com` | `gmail-jerome-dixon90` | `%USERPROFILE%\.gmail-mcp\tokens\jerome_dixon90.json` | SurgicalED / `vr-sling` / Dr. Siff / Personal |
| `dixonrj@vcu.edu` | `gmail-dixonrj-vcu` | `%USERPROFILE%\.gmail-mcp\tokens\dixonrj_vcu.json` | VCU |
| `jdixon@canallc.com` | `gmail-jdixon-canallc` | `%USERPROFILE%\.gmail-mcp\tokens\jdixon_canallc.json` | Canal LLC |
| `jerome@mushinsolutions.com` | `gmail-jerome-mushinsolutions` | `%USERPROFILE%\.gmail-mcp\tokens\jerome_mushinsolutions.json` | Mushin ops |

Shared Desktop OAuth client (not copied): `C:\Projects\notebookLM-edit-content\secrets\gdrive_jerome_dixon90_client_secret.json` via env `GMAIL_MCP_CLIENT_SECRET`. Manifest: `%USERPROFILE%\.gmail-mcp\accounts.json`.

## Auth Commands
```powershell
powershell -File C:\Projects\project_utility_scripts\scripts\gmail_mcp\auth_account.ps1 -List
powershell -File C:\Projects\project_utility_scripts\scripts\gmail_mcp\auth_account.ps1 jerome_dixon90
```

## Agent Rules
- Call tools on the matching `gmail-*` server before any Gmail action.
- Never print, copy, or commit token JSON / client secrets.
- Never reuse one token file for a second inbox.
- `vr-sling` / Lauren / SurgicalED mail --> `gmail-jerome-dixon90`.
- If a server is `needsAuth` or tools return 401, re-run `auth_account.ps1 <slug>` - do not fall back to another account.
