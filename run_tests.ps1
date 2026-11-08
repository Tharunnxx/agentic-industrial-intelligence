<#
 Runs the automated tests.
   .\run_tests.ps1                 # fast unit tests (no model, no dataset)
   .\run_tests.ps1 -Integration    # adds integration/API tests (train a tiny SYNTHETIC model in a temp folder)
#>
param([string]$Python = "python", [switch]$Integration)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$paths = @("tests/unit")
if ($Integration) { $paths += "tests/integration" }
& $Python -m pytest @paths -q
exit $LASTEXITCODE
