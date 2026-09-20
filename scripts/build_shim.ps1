# Builds the 32-bit shim (client exe + AI module dll) with MSVC + Ninja into shim/build.
# Uses the CMake/Ninja bundled with Visual Studio if none are on PATH.
[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')] [string]$Config = 'Release',
    [switch]$Clean
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root 'shim\build'
if ($Clean -and (Test-Path $BuildDir)) { Remove-Item $BuildDir -Recurse -Force }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'No Visual Studio C++ toolset found (run scripts\setup_windows.ps1).' }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
$flatc = Join-Path $Root 'tools\flatc\flatc.exe'
$flatcArg = if (Test-Path $flatc) { "-DFLATC_EXECUTABLE=`"$($flatc -replace '\\','/')`"" } else { '' }

# Run configure + build inside the x86 developer environment (cmake/ninja come with VS).
$cmd = @"
call "$vcvars" x86 >nul
cmake -S "$Root\shim" -B "$BuildDir" -G Ninja -DCMAKE_BUILD_TYPE=$Config -DSHIM_BACKEND=bwapi $flatcArg || exit /b 1
cmake --build "$BuildDir" --config $Config || exit /b 1
"@
$bat = Join-Path $env:TEMP 'bwshim_build.cmd'
Set-Content $bat $cmd -Encoding ASCII
cmd /c $bat
if ($LASTEXITCODE) { throw "shim build failed ($LASTEXITCODE)" }
Write-Host "`nBuilt:" -ForegroundColor Green
Get-ChildItem $BuildDir -Include shim.exe, shim_module.dll -Recurse | ForEach-Object { Write-Host "  $($_.FullName)  ($([math]::Round($_.Length/1KB)) KB)" }
