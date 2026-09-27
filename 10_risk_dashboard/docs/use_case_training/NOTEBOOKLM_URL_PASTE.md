# NotebookLM — Drive URL paste (UC01–UC09)

One **Gemini Notebook** per use case. Sources are **Google Drive file view URLs** for that UC’s `README.md` plus `screenshots/*.png`.

Do **not** use the Drive picker iframe. Do **not** use Fast Research / “Search the web” scrape. Do **not** sign in as CANA / `*canallc*`. Do **not** treat VCU `I:\` as primary.

Account: `jerome.dixon90@gmail.com`  
Pack: `G:\My Drive\PGx_Dashboard_Use_Cases\`  
Repo mirror: `10_risk_dashboard/docs/use_case_training\`

## 1. Launch debug Chrome (dedicated user-data-dir)

Everyday Chrome Default is usually already running and is **not** on a debug port. This window uses `%TEMP%\chrome-nlm-debug`, so you must sign in as `jerome.dixon90@gmail.com` in **this** window.

```bat
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9226 --remote-debugging-address=127.0.0.1 --user-data-dir="%TEMP%\chrome-nlm-debug" --profile-directory=Default --no-first-run --no-default-browser-check --disable-popup-blocking https://notebooklm.google.com/
```

PowerShell:

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

More launch notes: `CHROME_DEBUG.md`.

Check: `http://127.0.0.1:9226/json/version`

## 2. Sign in

In the **new** window only:

1. Open https://notebooklm.google.com/
2. Sign in as **`jerome.dixon90@gmail.com`**
3. Never pick `*canallc*`
4. Wait until the notebook list (not the Google password/chooser page) is visible

## 3. Resolve Drive view URLs (real IDs only)

```powershell
cd C:\Projects\pgx-analysis
.\.venv\Scripts\python.exe utility_scripts\resolve_pgx_uc_drive_urls.py
```

Writes `10_risk_dashboard/docs/use_case_training/notebooklm_drive_urls.json`.

This reads Google Drive for desktop metadata for the Gmail / `G:\` tree. It does **not** invent IDs.

If DriveFS is missing, use notebookLM-automation OAuth (`jerome_dixon90`) after a one-time auth:

```powershell
cd C:\Projects\notebookLM-automation
.\.venv\Scripts\python.exe -c "from notebooklm_automation.gdrive import get_account, get_credentials; get_credentials(get_account('config/gdrive_accounts.yaml','jerome_dixon90'))"
cd C:\Projects\pgx-analysis
.\.venv\Scripts\python.exe utility_scripts\resolve_pgx_uc_drive_urls.py --source api
```

`gdrive list` is that resolver (`--source api` or DriveFS). Skip `notebooklm\` and extra markdown (`NOTEBOOKLM_PROMPTS.md`, etc.).

## 4. Run the attach script (sequential, one notebook per UC)

```powershell
cd C:\Projects\pgx-analysis\11_testing\puppeteer
node notebooklm_pgx_ucs.js --browser-url http://127.0.0.1:9226 --only UC01
```

All nine, after UC01 looks right:

```powershell
node notebooklm_pgx_ucs.js --browser-url http://127.0.0.1:9226
```

Dry-run (print URLs, no browser):

```powershell
node notebooklm_pgx_ucs.js --dry-run
```

The script **connects** to the debug Chrome and **disconnects** when finished. It does not close the browser.

Per UC it:

1. Creates **New notebook** named `PGx UC## — …`
2. **Add sources → Websites** and pastes each Drive view URL
3. Pastes `NOTEBOOKLM_PROMPTS.md` plus the screenshot-only rule
4. Clicks **Audio Overview** / **Video Overview** / **Slide Deck** if those buttons exist
5. Copies downloads into both `notebooklm\` folders when Chrome yields files

## 5. Downloads

If Studio files stay in Chrome’s download folder, copy them to:

- `G:\My Drive\PGx_Dashboard_Use_Cases\<UC>\notebooklm\`
- `C:\Projects\pgx-analysis\10_risk_dashboard\docs\use_case_training\<UC>\notebooklm\`

## Notebook names

| Notebook | Folder | Sources |
|---|---|---|
| PGx UC01 — Cohort risk | `UC01_cohort_risk` | `README.md` + 7 PNGs |
| PGx UC02 — Scenario analysis | `UC02_scenario_analysis` | `README.md` + 5 PNGs |
| PGx UC03 — Density bin exploration | `UC03_density_bin_exploration` | `README.md` + 6 PNGs |
| PGx UC04 — Feature importance | `UC04_feature_importance` | `README.md` + 3 PNGs |
| PGx UC05 — Pattern and process | `UC05_pattern_process` | `README.md` + 4 PNGs |
| PGx UC06 — Claims PGx card | `UC06_claims_pgx_card` | `README.md` + 3 PNGs |
| PGx UC07 — Personalized PGx card | `UC07_personalized_pgx_card` | `README.md` + 5 PNGs |
| PGx UC08 — Cohort vs card | `UC08_cohort_vs_card` | `README.md` + 3 PNGs |
| PGx UC09 — Documentation | `UC09_documentation` | `README.md` + 5 PNGs |
