<#
 Starts the FastAPI backend (http://127.0.0.1:8000). Run from anywhere:
   .\run_backend.ps1                      # uses `python` on PATH
   .\run_backend.ps1 -Python "C:\path\to\python.exe" -Port 8001
 Use the SAME Python environment you trained the joblib models with.
#>
param([string]$Python = "python", [int]$Port = 8000, [string]$BindHost = "127.0.0.1", [switch]$Reload)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$uvArgs = @("-m", "uvicorn", "agentic_app.main:app", "--host", $BindHost, "--port", "$Port")
if ($Reload) { $uvArgs += "--reload" }
Write-Host "Starting backend on http://${BindHost}:$Port  (Ctrl+C to stop)"
& $Python @uvArgs
