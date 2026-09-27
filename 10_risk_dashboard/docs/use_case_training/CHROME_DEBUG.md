# NotebookLM Chrome remote debugging (port 9226)

Attach script (Drive **file URL paste**, not Drive picker / website scrape):

```powershell
cd C:\Projects\pgx-analysis\11_testing\puppeteer
node notebooklm_pgx_ucs.js --browser-url http://127.0.0.1:9226 --only UC01
```

Operator steps: `NOTEBOOKLM_URL_PASTE.md`. Resolve URLs first: `utility_scripts/resolve_pgx_uc_drive_urls.py`.

## Current session (2026-09-26)

- **PID:** 51872
- **Port:** 9226 (`http://127.0.0.1:9226`)
- **`/json/version`:** OK (`Chrome/153.0.8010.54`, Protocol 1.3)
- **Login needed:** yes — dedicated profile opened on Google Sign-in (sign in as `jerome.dixon90@gmail.com` in **this** window only)
- Left running: everyday Default Chrome (not debug) and VCU debug on **9223**

Do **not** use CANA / `*canallc*`. Preferred account in this window: `jerome.dixon90@gmail.com`.

## Why a dedicated profile

The user's everyday Chrome (`--profile-directory=Default`) is already running and **not** on a debug port. Reusing that User Data dir would require killing Chrome and would drop other work tabs. This launch uses a **separate** `--user-data-dir` so existing windows stay open.

## Exact launch command (Windows)

```bat
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9226 --remote-debugging-address=127.0.0.1 --user-data-dir="%TEMP%\chrome-nlm-debug" --profile-directory=Default --no-first-run --no-default-browser-check --disable-popup-blocking https://notebooklm.google.com/
```

PowerShell equivalent:

```powershell
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$userData = Join-Path $env:TEMP "chrome-nlm-debug"
Start-Process -FilePath $chrome -ArgumentList @(
  "--remote-debugging-port=9226",
  "--remote-debugging-address=127.0.0.1",
  "--user-data-dir=$userData",
  "--profile-directory=Default",
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-popup-blocking",
  "https://notebooklm.google.com/"
)
```

## Attach

- CDP URL: `http://127.0.0.1:9226`
- Version check: `curl http://127.0.0.1:9226/json/version`
- Playwright: `chromium.connect_over_cdp("http://127.0.0.1:9226")`

## Ports already in use (leave them)

- **9223** — existing VCU NotebookLM debug window (`%TEMP%\nl-vcu-chrome`). Do not kill.
- **9222–9225** — reserved / used by earlier sessions. Use **9226** for this Gmail NotebookLM attach.

## First-time login

This dedicated dir starts unsigned-in. In the **new** Chrome window only, sign in as `jerome.dixon90@gmail.com` (never canallc). After that, the session persists in `%TEMP%\chrome-nlm-debug` until that folder is deleted.

## Fallback profile dir (if TEMP is wiped)

```bat
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9226 --remote-debugging-address=127.0.0.1 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\User Data NLMDebug" --profile-directory=Default --no-first-run --no-default-browser-check https://notebooklm.google.com/
```
