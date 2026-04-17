# Puppeteer E2E Testing — Lessons Learned

**Date:** 2026-04-17  
**Context:** First full Puppeteer E2E run against production dashboard after implementing `available.json` / `GET /available` Lambda endpoint and density-bin risk model updates.

---

## Error 1 — `page.waitForTimeout` removed in Puppeteer v22

### Symptom
All 160 tests failed immediately with:
```
TypeError: page.waitForTimeout is not a function
```
at `helpers/browser.js:54`, `tests/viz.test.js:52`, `tests/pgx-card.test.js:35`, `tests/combinatorial.test.js:176`.

### Root Cause
`page.waitForTimeout()` was deprecated in Puppeteer v21 and **removed** in v22. The project upgraded to `puppeteer@^22.0.0`.

### Fix
Added a `sleep()` helper in `helpers/browser.js` and replaced all `page.waitForTimeout(ms)` calls:

```js
// helpers/browser.js
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
```

Files updated:
- `helpers/browser.js` — `selectCohort()`
- `tests/combinatorial.test.js` — error-handling tests
- `tests/viz.test.js` — `loadVizTab()` and missing-param guard
- `tests/pgx-card.test.js` — `openPgxCardTab()` and `submitVariants()`

---

## Error 2 — Jest can't find config when run via `npx --prefix`

### Symptom
```
Error: Could not find a config file based on provided values:
path: "C:\Projects\pgx-analysis"
```

### Root Cause
`run_prod_tests.ps1` used `npx --prefix $puppeteerDir jest`, which sets the npm prefix path but does **not** change the working directory for Jest. Jest traverses from `cwd` (repo root) looking for `jest.config.js` or a `package.json` with a `"jest"` key — neither exists at the repo root.

### Fix
Changed `run_prod_tests.ps1` to `Push-Location` into the puppeteer directory before invoking Jest, then `Pop-Location` after:

```powershell
Push-Location $puppeteerDir
& npx jest --testPathPattern=tests/ --forceExit
$puppeteerExit = $LASTEXITCODE
Pop-Location
if ($puppeteerExit -ne 0) { ... }
```

---

## Error 3 — `$env:VAR` eaten by WSL bash when passed via `-Command`

### Symptom
PowerShell env var assignments like `$env:DASHBOARD_URL='...'` passed through WSL `powershell.exe -Command "..."` were stripped by bash variable expansion:
```
:DASHBOARD_URL=https://... : The term ':DASHBOARD_URL=...' is not recognized...
```
Bash interprets `$env` as an empty variable, leaving `:DASHBOARD_URL=...` as a bare string.

### Root Cause
WSL bash expands `$env` before passing the command string to PowerShell.

### Fix
Moved all env var assignments into dedicated `.ps1` script files (`run_puppeteer_by_cohort.ps1`, `run_puppeteer.ps1`, `debug_one_test.ps1`) invoked via `powershell.exe -ExecutionPolicy Bypass -File`. Inside `.ps1` files, `$env:` assignments work correctly and are inherited by child processes.

---

## Error 4 — em-dash in PowerShell string causes parse error

### Symptom
```
The string is missing the terminator: ".
```
at line 135 of `run_prod_tests.ps1`.

### Root Cause
The em-dash character (`—`, U+2014) in the string `"ONE OR MORE SUITES FAILED — review output above"` caused a PowerShell string parse error when the file was saved/read with certain encodings.

### Fix
Replaced em-dash with ASCII hyphen: `"ONE OR MORE SUITES FAILED - review output above"`.

---

## Error 5 — `injectCodes` codes lost: `updateCodeLists()` scope closure resets selects

### Symptom
All non-baseline combinatorial scenarios (`low`, `medium`, `high`, `extreme`) failed with:
```
expect(data.is_baseline).not.toBe(true)
Expected: not true
```
The Lambda returned `is_baseline: true` for every scenario that had injected codes (drugs/icds/cpts), as if 0 codes were submitted.

### Root Cause
The original `injectCodes()` helper wrote directly to the `<select>` DOM elements. However, inside `calculateRisk()`, the call sequence is:

```js
if (!currentMetadata || currentCohort !== cohort) {
    await loadMetadata(cohort);
}
currentAgeBand = ageBand;
updateCodeLists();          // ← repopulates selects from metadata, clearing injected options
const drugs = getMultiSelectValues(drugsEl);   // ← reads now-empty selects
```

`updateCodeLists()` is a **local function in the same script scope**, not on `window`. It clears the selects and repopulates them with metadata codes (all unselected). By the time `getMultiSelectValues()` runs, the injected codes are gone. Lambda receives `drugs=[], icds=[], cpts=[]` → `n_events=0` → `is_baseline=true`.

A secondary fix attempt (stubbing `window.updateCodeLists`) failed because `calculateRisk()` references `updateCodeLists` via direct **scope closure**, not via `window`.

### Fix
Replaced DOM injection with **`window.fetch` interception**. The stub wraps the outgoing POST `/risk` request body and forces the correct `drugs`/`icds`/`cpts` values, bypassing the DOM entirely. The interceptor is one-shot (restores original `fetch` immediately after firing):

```js
async function injectCodes(page, drugs, icds, cpts) {
  await page.evaluate((d, i, c) => {
    const _orig = window.fetch;
    window.fetch = async function (url, opts, ...rest) {
      if (
        window.fetch !== _orig &&
        url && String(url).includes("/risk") &&
        opts && opts.method === "POST" &&
        !String(url).includes("/comparison") &&
        !String(url).includes("/drug_contributions")
      ) {
        window.fetch = _orig;   // one-shot: restore immediately
        try {
          const body = JSON.parse(opts.body || "{}");
          body.drugs = d;
          body.icds  = i;
          body.cpts  = c;
          opts = { ...opts, body: JSON.stringify(body) };
        } catch (_) {}
      }
      return _orig.call(this, url, opts, ...rest);
    };
  }, drugs, icds, cpts);
}
```

**Result:** All combinatorial scenarios (baseline, low, medium, high, extreme) pass for all cohort × age-band combinations.

---

## Error 6 — Lambda `GET /available` returns 500 instead of 404 when `available.json` missing

### Symptom
```json
{
  "error": "Internal server error",
  "message": "An error occurred (AccessDenied) when calling the GetObject operation:
  ...not authorized to perform: s3:ListBucket..."
}
```

### Root Cause
When `available.json` does not exist in S3 **and** the Lambda execution role lacks `s3:ListBucket`, AWS returns `AccessDenied` (not `NoSuchKey`) from `GetObject`. The original exception handler only caught `NoSuchKey`/`404`/`NotFound`, so `AccessDenied` fell through to the generic `raise` → 500 response.

### Fix
Added `"AccessDenied"` and `"403"` to the caught error codes in `handle_available()`:

```python
if code in ("NoSuchKey", "404", "NotFound", "AccessDenied", "403"):
    return _response(404, {"error": "available.json not found — run notebook 5 to generate it"})
```

Fix deployed to `s3://pgxdatalake/gold/dashboard/code/lambda_function.py`. Lambda will pick it up on next cold start (env vars `CODE_S3_KEY` + `PGX_RESULTS_BUCKET` now set).

---

## Error 7 — PGx Card: stale selectors + lazy tab content loading

### Symptom
```
Error: failed to find element matching selector "#pgx-variants-input"
```
All 3 PGx Card tests fail immediately.

### Root Cause (3 issues)
1. **Stale selector**: test used `#pgx-variants-input` → actual id is `#snp-input` (in `tabs/pgx-card.html`).
2. **Stale button selector**: test used `#btnGeneratePgxCard` → actual id is `#btnGenerateCard`.
3. **Lazy tab loading**: tab content is fetched from `tabs/pgx-card.html` on first switch. A bare `sleep(400)` was not enough to guarantee the HTML was injected. Test should use `waitForSelector("#pgx-card-cohort")`.
4. **Missing prerequisite**: the SNP refinement section (`#pgx-snp-refine-section`) is hidden until `#btnLoadPgxCardProfile` is clicked and the cohort profile loads. The test must complete this step before writing to `#snp-input`.
5. **Wrong input format**: the textarea expects plain text lines (`CYP2D6,*1,*2`), not JSON.

### Fix
Rewrote `pgx-card.test.js`:
- `openPgxCardTab()` → waits for `#pgx-card-cohort` via `waitForSelector`
- Added `loadCohortProfile(cohort, ageBand)` → sets selects, clicks `#btnLoadPgxCardProfile`, waits for `#pgx-snp-refine-section` to become visible
- `submitVariants(lines[])` → writes plain-text lines to `#snp-input`, clicks `#btnGenerateCard`
- Called `loadCohortProfile()` in `beforeAll` so SNP section is ready for all tests

### Design Rule
> When a UI section requires a prerequisite action (e.g., load a profile), always complete that prerequisite in `beforeAll` and wait for the dependent element before asserting. Never rely on a fixed sleep for lazy-loaded content — use `waitForSelector`.

---

## Error 8 — Viz `cohort-pgx` test intercepts S3 HTML instead of Lambda JSON

### Symptom
```
expect(body !== null && typeof body === "object").toBe(true)
Expected: true   Received: false
```
for `non_opioid_ed/{13-24,45-54,75-84} — cohort-pgx-visualizations`, even though `GET /visualizations/cohort_pgx` returns a valid JSON object.

### Root Cause
`waitForResponse` filters by `resp.url().includes("/cohort_pgx")`. After the Lambda responds, the frontend fetches `network_topology.html` **directly from S3** at:
```
https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/
  visualizations/cohort_pgx/networks/non_opioid_ed/13-24/network_topology.html
```
This URL also contains `/cohort_pgx`. `waitForResponse` may resolve on this S3 HTML response. `response.json()` throws on HTML → `body = null` → assertion fails.

### Fix
Narrowed the `waitForResponse` filter to require the URL also starts with `API_BASE_URL` (the Lambda API Gateway endpoint). S3 sub-resource URLs have a different host and are excluded:

```js
resp => resp.url().includes(pathFrag) &&
        resp.request().method() === "GET" &&
        (!API_BASE || resp.url().startsWith(API_BASE))
```

### Design Rule
> When intercepting API responses via `waitForResponse`, always scope the filter to the API host. Never rely on path fragment alone — the page may fetch static assets from CDN/S3 whose URLs share the same path segment.

---

## Test Infrastructure: New Scripts Added

| Script | Purpose |
|--------|---------|
| `run_puppeteer.ps1` | Run Puppeteer suite by name pattern against prod; sets env vars correctly |
| `run_puppeteer_by_cohort.ps1` | Run one cohort + one age band at a time; writes per-cohort markdown + per-failure `.log` files |
| `debug_one_test.ps1` | Run a targeted single test pattern and dump raw ANSI-stripped output to `results/debug_one_test.log` |

---

## Key Design Principle

> **Never test DOM state that the application will overwrite.** If the SUT's internal flow resets a DOM element between your setup and the assertion, inject at the network/protocol layer instead (fetch interception, request interception, or payload construction).
