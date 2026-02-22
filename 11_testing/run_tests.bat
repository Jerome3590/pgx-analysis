@echo off
REM Run dashboard tests from repo root (Windows cmd).
REM Usage: 11_testing\run_tests.bat
REM Optional: set BASE_URL=https://...execute-api.../prod before running for live API tests.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."
if not exist "11_testing\tests" (
    echo Error: test directory not found: 11_testing\tests
    exit /b 1
)

echo Repo root: %CD%
echo Running: pytest 11_testing\tests -v (dashboard + dashboard_visuals)
echo.

python -m pytest 11_testing\tests -v %*
exit /b %ERRORLEVEL%
