$env:DASHBOARD_URL = "https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html"
$env:API_BASE_URL  = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"

$puppeteerDir = Join-Path $PSScriptRoot "puppeteer"
$logFile      = Join-Path $PSScriptRoot "results\debug_pgx_card.log"

Push-Location $puppeteerDir
$out = & npx jest --testPathPattern=tests/pgx-card --forceExit --verbose 2>&1
$exit = $LASTEXITCODE
Pop-Location

($out -replace '\x1B\[[0-9;]*[A-Za-z]','') | Set-Content $logFile -Encoding UTF8
Write-Host "Exit: $exit   Log: $logFile"
