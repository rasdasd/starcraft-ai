# Freeze the 64-bit Python brain with PyInstaller (onedir, next to numpy binaries).
[CmdletBinding()]
param(
    [string]$Bot = 'mybot',
    [string]$OutDir = ''
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$PyRoot = Join-Path $Root 'python'
$VenvPy = Join-Path $PyRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) { throw "venv not found at $VenvPy (run scripts\setup_windows.ps1)" }
if (-not $OutDir) { $OutDir = Join-Path $Root 'dist\bot' }

& $VenvPy -m pip install -q pyinstaller
$work = Join-Path $Root 'dist\pyi'
New-Item -ItemType Directory -Force -Path $OutDir, $work | Out-Null

& $VenvPy -m PyInstaller `
    --noconfirm --clean --onedir --name bot `
    --distpath $OutDir --workpath $work --specpath $work `
    --paths $PyRoot `
    --collect-submodules bwbot `
    --collect-submodules mybot `
    --collect-submodules goliath `
    --collect-submodules learned `
    --collect-submodules blackboard `
    --collect-submodules adjutant `
    --hidden-import mybot `
    --hidden-import goliath `
    --hidden-import learned `
    --hidden-import blackboard `
    --hidden-import adjutant `
    (Join-Path $PyRoot 'packaging\run_bot.py')
if ($LASTEXITCODE) { throw "PyInstaller failed ($LASTEXITCODE)" }

Write-Host "Frozen bot: $(Join-Path $OutDir 'bot\bot.exe')  (pass '$Bot' as the first argument)"
