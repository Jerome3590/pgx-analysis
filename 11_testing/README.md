# 11. Testing

Central place for **test plans**, **all tests**, and **running tests** from your local machine (including Windows).

## Contents

| Item | Description |
|------|--------------|
| [TEST_PLAN_FINAL_DASHBOARD.md](TEST_PLAN_FINAL_DASHBOARD.md) | Test plan for the final dashboard: required artifacts, Lambda/API Gateway, per-component checks |
| [DASHBOARD_VALIDATION.md](DASHBOARD_VALIDATION.md) | Validation checklist for frontend changes and S3 deploys: tabs, visual headings, S3 path-style URLs, manifest alignment, CORS |
| **tests/** | All project tests: dashboard (final) + dashboard_visuals (BupaR/DTW/FP-Growth). See [tests/README.md](tests/README.md). |
| `run_tests.ps1` | PowerShell script to run all tests from repo root (Windows) |
| `run_tests.bat` | Batch file to run tests (cmd or double-click) |

## Running tests on Windows

**Prerequisites:** Python 3.11+, pytest. From repo root: `pip install -r requirements.txt` (optional: `pip install requests` for live API tests).

### Option 1: PowerShell (recommended)

From a PowerShell prompt, from anywhere in the repo:

```powershell
.\11_testing\run_tests.ps1
```

To run with a live API URL (e.g. after deploy):

```powershell
$env:BASE_URL = "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod"
.\11_testing\run_tests.ps1
```

### Option 2: Batch file

From Command Prompt or Explorer (double-click):

```cmd
11_testing\run_tests.bat
```

### Option 3: pytest directly

From repo root:

```cmd
pytest 11_testing/tests/ -v
```

With live API:

```cmd
set BASE_URL=https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod
pytest 11_testing/tests/ -v
```

Run a single file (e.g. Risk Assessment tab):

```cmd
pytest 11_testing/tests/test_risk_assessment_tab.py -v
```

Run only dashboard visuals tests:

```cmd
pytest 11_testing/tests/dashboard_visuals/ -v
```

## What the tests cover

All tests live under **11_testing/tests/**:

- **Dashboard (final):** artifacts, CORS, and per-tab Lambda/API tests (Risk Assessment, PGx Card, Documentation, Feature Importance, Causal Analysis, BupaR, DTW, FP-Growth), plus live API when `BASE_URL` is set. See `tests/README.md` and `tests/test_final_dashboard.py` docstring.
- **Dashboard visuals:** prerequisite (allowed_codes_shap_ffa) and output structure for BupaR, DTW, FP-Growth; optional integration when `RUN_VISUALS_INTEGRATION=1`.

This folder (`11_testing`) holds the plan, run scripts, and the single tests directory.
