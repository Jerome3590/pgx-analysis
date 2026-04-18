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

## Error 9 — `xgboost_rf` missing → `models_failed` instead of `models_absent`

### Symptom
The model breakdown chart rendered nothing for `xgboost_rf`-weighted age bands. Lambda logs showed `AccessDenied` on `GetObject` for the `xgboost_rf.joblib` key, which fell through to `models_failed` — treated as a genuine prediction error.

### Root Cause
Two compounding issues:
1. **IAM missing `s3:ListBucket`**: AWS returns `403 AccessDenied` (not `404 NoSuchKey`) when a key doesn't exist and the role lacks `s3:ListBucket`. The `_load_from_s3_bytes` helper only caught `NoSuchKey/404/NotFound`.
2. **`models_absent` not tracked**: `predict_risk` had no concept of "expected missing" vs "unexpected failure". Any `FileNotFoundError` on a non-sole-best model went into `models_failed`.

### Fix
Two-part fix in `lambda_function.py`:
1. In `_load_from_s3_bytes`, treat `AccessDenied`/`403` as key-not-found (return `None`) so `load_model` raises `FileNotFoundError`:
```python
if code in ("NoSuchKey", "404", "NotFound", "AccessDenied", "403"):
    return None
```
2. In `predict_risk`, track `models_absent` separately — `FileNotFoundError` on a model with `weight < 1.0` routes to `models_absent`, not `models_failed`. Response now includes both fields so callers can distinguish expected absence from unexpected failure.

### Verification
```python
ei = response['ensemble_info']
assert 'xgboost_rf' in ei['models_absent']
assert ei['models_failed'] == []
```

---

## Error 10 — `GET /performance` returns 500: `ModuleNotFoundError: No module named 'py_helpers'`

### Symptom
```
HTTP Error 500: Internal Server Error
ModuleNotFoundError: No module named 'py_helpers'
```

### Root Cause
`handle_performance()` imported `REQUIRED_COHORTS` from `py_helpers.constants` inside the function body. `py_helpers` is a local dev/EC2 module — it is **not** installed in the Lambda container image.

### Fix
Replaced the import with an inline constant:
```python
# Before (breaks in Lambda)
from py_helpers.constants import REQUIRED_COHORTS

# After
COHORTS = ["opioid_ed", "non_opioid_ed"]
```

### Design Rule
> Never import from `py_helpers` (or any local project module) inside Lambda handler functions. Lambda containers only have packages explicitly installed in `requirements.txt` or the base image. Use inline literals or constants defined at module top-level from safe stdlib/third-party imports.

---

## Error 11 — `/performance` returns empty after EC2 rebuild (wrong operation order)

### Symptom
`/performance` returned `{"by_cohort": {"opioid_ed": {}, "non_opioid_ed": {}}}` even after EC2 rebuild.

### Root Cause
`docker_build.sh` runs `prepare_lambda_dir.py` which **copies** `feature_schema.json` files from `10_risk_dashboard/outputs/models/`. If `prepare_models.py --all` has not been run first, those schemas don't have the `model_scores` key yet. Notebook 5 / `docker_build.sh` alone is not sufficient.

### Correct EC2 Operation Order
```bash
# 1. Embed model_scores into feature schemas (reads MC-CV CSVs from 6_final_model/outputs)
python 10_risk_dashboard/data_preparation/prepare_models.py --all

# 2. Build + push ECR (runs prepare_lambda_dir internally, then docker build + push + Lambda update)
bash 10_risk_dashboard/deployment/docker_build.sh
```

### Design Rule
> `docker_build.sh` is a deploy script, not a data prep script. Data artifacts (feature schemas with embedded metrics) must be generated by `prepare_models.py` before `docker_build.sh` copies them into the container.

---

## Error 12 — Testing with cohort-defining target code as an ICD input

### Symptom
Test payloads for `opioid_ed` included `"icds": ["F1120"]` (Opioid dependence, uncomplicated). F1120 is the **target variable** for `opioid_ed` — using it as a predictor is circular (data leakage).

### Root Cause
Test code copied a generic ICD code without checking cohort-specific exclusions.

### Verification Pattern
Always verify target codes are excluded at three layers before writing test payloads:
1. **Feature schema**: `item_F1120` must not appear in `feature_schema.json` features list
2. **Metadata**: F1120 must not appear in `/metadata?cohort=opioid_ed` ICD list
3. **Runtime**: `codes_used.icds` must not contain F1120 even if submitted

```python
# Confirm clean at all layers
assert not any('F1120' in f.upper() for f in schema['features'])
assert not any(c['code'].upper() == 'F1120' for c in metadata_icds)
assert 'F1120' not in response['codes_used']['icds']
```

### Design Rule
> Never use cohort-defining target codes as test inputs. For each cohort, check the README / cohort creation documentation for the target definition before constructing test payloads.

---

## Error 13 — Causal Analysis filter bug not caught: test never selected codes

### Symptom
Causal Analysis tab showed no output when codes were selected, but all Puppeteer tests passed.

### Root Cause
`viz.test.js` combinatorial matrix clicks `btnLoadCausal` with **no codes selected**. When `selectedDrugs/Icds/Cpts` are empty arrays, `selectedFeatureSet.size === 0` → the filter is skipped entirely and all causal factors are returned unfiltered. The broken code path (`"item_" + code.toUpperCase()` producing `item_DRUG_GABAPENTIN` instead of `item_drug_GABAPENTIN`) was never exercised.

The assertions only checked HTTP 200 + valid JSON — not that `causal_factors` was non-empty or that submitted codes appeared in the results.

### Fix
Added a dedicated regression test in `viz.test.js`:
1. Selects cohort + age first, waits for `updateCodeLists()` to populate the selects
2. Uses `page.evaluate` to select `drug_GABAPENTIN` and `icd_B1920` by value in the DOM multi-selects (safe here — causal handler reads selects directly, no `updateCodeLists()` between DOM write and read)
3. Sets `causal-n-event-bin` to `medium` to force the per-bin Lambda API path (always fires a GET, not a static fetch)
4. Asserts `causal_factors.length > 0` AND that at least one feature name matches `item_drug_GABAPENTIN` or `item_icd_B1920` (guards against the `item_DRUG_GABAPENTIN` regression)

### Design Rule
> For any tab that filters or transforms user-selected codes, write at least one test that actually **selects codes** before triggering the load action. A test that exercises the tab with no selection only validates the no-filter path.

### Note on DOM injection vs. fetch interception
Unlike `POST /risk` (where `updateCodeLists()` in the same closure scope resets selects before they are read — see Error 5), the causal handler reads `getMultiSelectValues(drugsEl)` **directly when the button is clicked** without calling `updateCodeLists()` first. DOM injection of `option.selected = true` is therefore safe and sufficient for this tab.

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
