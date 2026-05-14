# NeuroScan Pro - Robust Launch Script
# This script handles broken environments and runs the app reliably.

$VenvPath = "venv"
$VenvPython = "$VenvPath\Scripts\python.exe"
$ActivateScript = "$VenvPath\Scripts\Activate.ps1"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       NeuroScan Pro - AI MRI Scan        " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Detect broken or missing venv
if (-Not (Test-Path $ActivateScript) -or -Not (Test-Path $VenvPython)) {
    Write-Host "[!] Virtual environment is missing or broken. Recreating..." -ForegroundColor Yellow
    if (Test-Path $VenvPath) {
        # Try to remove, but don't fail if files are locked (user might have it open)
        try { Remove-Item -Recurse -Force $VenvPath -ErrorAction Stop } 
        catch { Write-Host "[!] Could not clear old venv. Proceeding with repair..." -ForegroundColor Gray }
    }
    python -m venv $VenvPath
}

# 2. Verify venv python exists
if (-Not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Could not find or create venv python at $VenvPython" -ForegroundColor Red
    Write-Host "Please ensure Python 3 is installed and added to your Windows PATH." -ForegroundColor Red
    exit 1
}

# 3. Install/Update dependencies using venv python directly
# This avoids activation policy issues
Write-Host "[*] Checking dependencies..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r backend/requirements.txt

# 4. Run the application
Write-Host ""
Write-Host "[!] Starting NeuroScan Pro Application..." -ForegroundColor Green
Write-Host "[!] The application will be available at: http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "[!] Press Ctrl+C to stop the server." -ForegroundColor Red
Write-Host ""

& $VenvPython backend/app.py
