# PowerShell script to change permissions on R library folder
# WARNING: This modifies Program Files permissions - use at your own risk
# Run this script as Administrator

$rLibPath = "C:\Program Files\R\R-4.5.2\library"

# Check if the folder exists
if (Test-Path $rLibPath) {
    Write-Host "Found R library folder: $rLibPath" -ForegroundColor Green
    
    # Get current user
    $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    
    Write-Host "Granting full control to: $currentUser" -ForegroundColor Yellow
    
    # Take ownership first (requires admin)
    takeown /F "$rLibPath" /R /D Y
    
    # Grant full control to current user
    icacls "$rLibPath" /grant "${currentUser}:(OI)(CI)F" /T
    
    Write-Host "Permissions updated successfully!" -ForegroundColor Green
    Write-Host "You should now be able to install packages to this location." -ForegroundColor Green
} else {
    Write-Host "R library folder not found at: $rLibPath" -ForegroundColor Red
    Write-Host "Please update the path in this script to match your R installation." -ForegroundColor Yellow
}
