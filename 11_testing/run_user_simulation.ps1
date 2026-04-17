<#
.SYNOPSIS
    Smoke-run the user-simulation test suite and write results to results/results_user_simulation.md.
#>
$ErrorActionPreference = "Continue"
$env:DASHBOARD_URL = "https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html"
$env:API_BASE_URL  = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"

$puppeteerDir = Join-Path $PSScriptRoot "puppeteer"
$resultsDir   = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

Push-Location $puppeteerDir
$out  = & npx jest --testPathPattern="tests/user-simulation" --forceExit --verbose 2>&1
$code = $LASTEXITCODE
Pop-Location

$summary  = ($out | Where-Object { $_ -match "^Tests:" } | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
$time     = ($out  | Where-Object { $_ -match "^Time:"  } | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
$icon     = if ($code -eq 0) { "PASS" } else { "FAIL" }
$mdFile   = Join-Path $resultsDir "results_user_simulation.md"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

"# Test Results: Full UI Simulation (Real User Workflow)" | Set-Content  $mdFile
""                                                         | Add-Content $mdFile
"**Run:** $timestamp"                                      | Add-Content $mdFile
"**Dashboard:** $env:DASHBOARD_URL"                        | Add-Content $mdFile
"**API:** $env:API_BASE_URL"                               | Add-Content $mdFile
""                                                         | Add-Content $mdFile
"**$icon** $summary  $time"                                | Add-Content $mdFile
""                                                         | Add-Content $mdFile

if ($code -ne 0) {
    $logFile = Join-Path $resultsDir "user_simulation.log"
    ($out -replace '\x1B\[[0-9;]*[A-Za-z]','') | Set-Content $logFile -Encoding UTF8
    "  [raw log: results/user_simulation.log]" | Add-Content $mdFile
    "" | Add-Content $mdFile

    # Extract failure summaries
    $failures = @{}; $cur = $null; $errs = @()
    foreach ($line in $out) {
        $plain = $line -replace '\x1B\[[0-9;]*m', ''
        if ($plain -match '^\s+●\s+(.+)$') {
            if ($cur -and $errs.Count) { $failures[$cur] = $errs[0..([Math]::Min(5,$errs.Count-1))] }
            $cur = $Matches[1].Trim(); $errs = @()
        } elseif ($cur -and $plain -match '\S') { $errs += $plain.Trim() }
    }
    if ($cur -and $errs.Count) { $failures[$cur] = $errs[0..([Math]::Min(5,$errs.Count-1))] }

    if ($failures.Count) {
        "#### Failures ($($failures.Count))" | Add-Content $mdFile
        "" | Add-Content $mdFile
        foreach ($t in $failures.Keys) {
            "- **$t**" | Add-Content $mdFile
            $first = ($failures[$t] | Where-Object { $_ -match "Error|expect|TypeError|net::" } | Select-Object -First 1)
            if ($first) { "  ``$first``" | Add-Content $mdFile }
        }
        "" | Add-Content $mdFile
    }
}

Write-Host "`n$icon  $summary" -ForegroundColor $(if ($code -eq 0) {"Green"} else {"Red"})
Write-Host "Results: $mdFile" -ForegroundColor Yellow
