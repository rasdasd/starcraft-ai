<#
.SYNOPSIS
  Launch the full native stack on Windows: shim (BWAPI client) -> StarCraft 1.16.1 + BWAPI -> Python bot.

  Each component opens in its own console window so their logs are separate. Ctrl+C in this
  window (or -Stop) tears everything down.

.EXAMPLE
  scripts\run_native.ps1                                  # Adjutant, default settings
  scripts\run_native.ps1 -Race Zerg -FrameSkip 1 -Speed 42
  scripts\run_native.ps1 -BotProfile search               # pick a bot profile (BWBOT_PROFILE)
  scripts\run_native.ps1 -Games 5                         # play 5 games, then shut everything down
  scripts\run_native.ps1 -NoBot                           # only StarCraft + shim; run your bot by hand
  scripts\run_native.ps1 -Stop                            # kill StarCraft / shim / bot
#>
[CmdletBinding()]
param(
    [string]$Bot = 'adjutant',
    [int]$Port = 8765,
    [int]$FrameSkip = -1,        # -1 = bot default
    [int]$Speed = -2,            # -2 = bot default, -1 = game default, 0 = fastest, 42 = normal
    [string]$Map = '',           # override bwapi.ini map (relative to game/, e.g. maps/BroodWar/sscai/(4)Python.scx)
    [string]$Race = '',          # Terran | Protoss | Zerg | Random
    [string]$EnemyRace = '',
    [Alias('Profile')]
    [string]$BotProfile = '',    # bot profile name or JSON path (sets BWBOT_PROFILE for the bot)
    [int]$Games = 0,             # play exactly N games, then shut everything down (0 = forever)
    [switch]$NoBot,
    [switch]$NoGame,
    [switch]$Stop,
    [switch]$Wait                # block until the bot process exits
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Game = Join-Path $Root 'game'
$Shim = Join-Path $Root 'shim\build\shim.exe'
$Py   = Join-Path $Root 'python\.venv\Scripts\python.exe'

function Stop-All {
    foreach ($n in 'StarCraft', 'shim') { Get-Process $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'bwbot\.run' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
if ($Stop) { Stop-All; Write-Host 'stopped'; return }

if (-not (Test-Path (Join-Path $Game 'StarCraft.exe'))) { throw "game/StarCraft.exe missing - run scripts\setup_windows.ps1" }
if (-not (Test-Path $Shim)) { throw "shim not built - run scripts\build_shim.ps1" }

# --- optional bwapi.ini overrides ---------------------------------------------
$ini = Join-Path $Game 'bwapi-data\bwapi.ini'
$c = Get-Content $ini -Raw
if ($Map)       { $c = $c -replace '(?m)^map\s*=.*$',        "map = $Map" }
if ($Race)      { $c = $c -replace '(?m)^race\s*=.*$',       "race = $Race" }
if ($EnemyRace) { $c = $c -replace '(?m)^enemy_race\s*=.*$', "enemy_race = $EnemyRace" }
$c = $c -replace '(?m)^ai\s*=.*$', 'ai     = '   # client mode: no AI module dll
Set-Content $ini -Value $c -NoNewline

Stop-All
Start-Sleep -Milliseconds 300

# --- 1. shim (BWAPI client). It waits for StarCraft, then for the bot on $Port. --
Write-Host "starting shim on port $Port"
$shimProc = Start-Process -FilePath $Shim -ArgumentList @('--port', $Port) -WorkingDirectory (Split-Path $Shim) -PassThru

# --- 2. StarCraft + BWAPI via Injectory (windowed) --------------------------------
if (-not $NoGame) {
    # StarCraft 1.16.1 keeps "Show Tips at Startup" in the registry (tip=0x100 on a fresh
    # install). The tips dialog pauses the match until dismissed, so turn it off every run.
    $reg = 'HKCU:\Software\Blizzard Entertainment\Starcraft'
    if (-not (Test-Path $reg)) { New-Item $reg -Force | Out-Null }
    Set-ItemProperty $reg -Name tip -Value 0 -Type DWord
    Set-ItemProperty $reg -Name tipnum -Value 0 -Type DWord

    Write-Host 'starting StarCraft 1.16.1 with BWAPI 4.4.0 injected'
    Start-Process -FilePath (Join-Path $Game 'injectory_x86.exe') `
        -ArgumentList @('--launch', 'StarCraft.exe', '--inject', 'bwapi-data\BWAPI.dll', 'wmode.dll') `
        -WorkingDirectory $Game | Out-Null
}

# --- 3. python bot -----------------------------------------------------------------
if (-not $NoBot) {
    if (-not (Test-Path $Py)) { throw "python venv missing - run scripts\setup_windows.ps1" }
    $args = @('-m', 'bwbot.run', $Bot, '--port', $Port)
    if ($FrameSkip -ge 1) { $args += @('--frame-skip', $FrameSkip) }
    if ($Speed -ge -1)    { $args += @('--speed', $Speed) }
    if ($Games -gt 0)     { $args += @('--games', $Games) }
    Write-Host "starting bot: python $($args -join ' ')$(if ($BotProfile) { " (BWBOT_PROFILE=$BotProfile)" })"
    $prevProfile = $env:BWBOT_PROFILE
    if ($BotProfile) { $env:BWBOT_PROFILE = $BotProfile }
    try {
        $botProc = Start-Process -FilePath $Py -ArgumentList $args -WorkingDirectory (Join-Path $Root 'python') -PassThru
    } finally {
        $env:BWBOT_PROFILE = $prevProfile
    }
    if ($Games -gt 0) {
        # Bounded session: wait for the bot to finish its N games, then tear down StarCraft + shim,
        # otherwise the game auto-restarts and sits there waiting for a bot that never comes.
        Write-Host "playing $Games game(s); everything shuts down when the bot exits (Ctrl+C to abort)"
        try { $botProc.WaitForExit() } finally { Stop-All }
        Write-Host "done: $Games game(s) played, exit code $($botProc.ExitCode)"
        return
    }
    if ($Wait) { $botProc.WaitForExit() }
}

Write-Host "`nRunning. Stop everything with:  scripts\run_native.ps1 -Stop"
