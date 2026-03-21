# PowerShell script to set R_LIBS_USER environment variable
# Run this script as Administrator to set system-wide, or run normally for user-only

# For R 4.5.2, the library path is:
$userProfile = $env:USERPROFILE
$rLibPath = "$userProfile\R\win-library\4.5"

Write-Host "Setting R_LIBS_USER to: $rLibPath" -ForegroundColor Green

# Set for current user (no admin required)
[System.Environment]::SetEnvironmentVariable("R_LIBS_USER", $rLibPath, [System.EnvironmentVariableTarget]::User)

Write-Host "R_LIBS_USER environment variable set successfully!" -ForegroundColor Green
Write-Host "You may need to restart R/RStudio for this to take effect." -ForegroundColor Yellow

# Display current value
$currentValue = [System.Environment]::GetEnvironmentVariable("R_LIBS_USER", [System.EnvironmentVariableTarget]::User)
Write-Host "Current R_LIBS_USER value: $currentValue" -ForegroundColor Cyan
