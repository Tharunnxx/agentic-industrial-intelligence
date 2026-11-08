<#
 Starts the backend in a new window, waits until it answers, then opens the demo UI in your browser.
   .\run_demo.ps1
   .\run_demo.ps1 -Python "C:\path\to\python.exe" -Port 8000
 The first /health call loads the joblib model, which can take a while for large Random Forests.
#>
param([string]$Python = "python", [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$url = "http://127.0.0.1:$Port"
$backend = Join-Path $PSScriptRoot "run_backend.ps1"
Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$backend`"", "-Python", "`"$Python`"", "-Port", "$Port")
Write-Host "Waiting for the backend (and model load) ..."
$ready = $false
for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 120
        $ready = $true
    } catch { }
}
if (-not $ready) { Write-Error "Backend did not respond. Look at the backend window for errors."; exit 1 }
if (-not $h.model_loaded) { Write-Warning "Backend is up but the model is NOT loaded: $($h.model_error.message)" }
Start-Process $url
Write-Host "Demo opened at $url . Close the backend window to stop the server."
