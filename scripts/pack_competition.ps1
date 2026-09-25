# Build an AIIDE/BASIL run folder: shim_module.dll + frozen bot + run_proxy.bat.
[CmdletBinding()]
param(
    [string]$Name = 'bwbot',
    [string]$Bot = 'adjutant',       # profile: set BWBOT_PROFILE in run_proxy.bat or ship a profile JSON
    [switch]$SkipShim,
    [switch]$SkipFreeze
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Pack = Join-Path $Root "dist\competition\$Name"
$Ai = Join-Path $Pack 'AI'
$DllSrc = Join-Path $Root 'shim\build\shim_module.dll'

if (-not $SkipShim) { & (Join-Path $PSScriptRoot 'build_shim.ps1') }
if (-not (Test-Path $DllSrc)) { throw "shim_module.dll not found at $DllSrc" }

if (-not $SkipFreeze) { & (Join-Path $PSScriptRoot 'freeze_bot.ps1') -Bot $Bot -OutDir (Join-Path $Root 'dist\bot') }

New-Item -ItemType Directory -Force -Path $Ai, (Join-Path $Pack 'read'), (Join-Path $Pack 'write') | Out-Null
Copy-Item $DllSrc (Join-Path $Ai "$Name.dll") -Force
$frozen = Join-Path $Root 'dist\bot\bot'
if (Test-Path $frozen) {
    Copy-Item $frozen (Join-Path $Ai 'bot') -Recurse -Force
}
# Learned models (.npz). The loader also checks bwapi-data\read\ so updated models can be dropped in later.
$models = Get-ChildItem (Join-Path $Root 'python\models') -Filter *.npz -ErrorAction SilentlyContinue
if ($models) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Ai 'models') | Out-Null
    $models | Copy-Item -Destination (Join-Path $Ai 'models') -Force
}
# Your builds (builds\) and learned ones (python\models\builds\); the built-ins are frozen into bot.exe.
foreach ($src in @((Join-Path $Root 'builds'), (Join-Path $Root 'python\models\builds'))) {
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Ai 'models\builds') | Out-Null
        Copy-Item (Join-Path $src '*') (Join-Path $Ai 'models\builds') -Recurse -Force
    }
}

$proxy = @"
@echo off
REM Tournament CWD is the StarCraft folder. Do not cd.
start "bwbot" /min bwapi-data\AI\bot\bot.exe $Bot --host 127.0.0.1 --port 8765 --no-gui
"@
Set-Content (Join-Path $Ai 'run_proxy.bat') $proxy -Encoding ASCII

Copy-Item (Join-Path $Root 'docs\competition.md') (Join-Path $Pack 'README.md') -Force

Write-Host ""
Write-Host "Competition folder: $Pack"
Write-Host "  AI\$Name.dll"
Write-Host "  AI\bot\bot.exe $Bot"
Write-Host "  AI\run_proxy.bat"
Write-Host "Zip this folder for AIIDE/BASIL."
