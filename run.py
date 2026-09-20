#!/usr/bin/env python3
"""Launch the bot against a visible StarCraft window. Works from Windows or WSL.

    python run.py                          # example bot, windowed StarCraft, default settings
    python run.py --bot mybots.zerg --speed 42
    python run.py --games 3                # play exactly 3 games, then shut everything down
    python run.py --stop                   # tear everything down

The visible game (StarCraft 1.16.1 + BWAPI) only exists on Windows, so this script always drives
scripts/run_native.ps1 with PowerShell. From WSL it goes through Windows interop (powershell.exe),
which puts the game, shim and bot windows on the Windows desktop. For headless OpenBW games use
scripts/run_openbw.sh instead.

Add new options to `build_parser()` and translate them in `to_ps_args()`.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE_SCRIPT = ROOT / "scripts" / "run_native.ps1"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", default="examples.basic_terran", help="python module holding the Bot subclass")
    p.add_argument("--port", type=int, default=8765, help="shim <-> bot TCP port")
    p.add_argument("--frame-skip", type=int, help="bot decides every N frames (default: bot config)")
    p.add_argument("--speed", type=int, help="game speed: -1 game default, 0 fastest, 42 normal (default: bot config)")
    p.add_argument("--map", help="map relative to game/, e.g. maps/BroodWar/sscai/(4)Python.scx")
    p.add_argument("--race", choices=["Terran", "Protoss", "Zerg", "Random"], help="bot race")
    p.add_argument("--enemy-race", choices=["Terran", "Protoss", "Zerg", "Random"], help="built-in AI race")
    p.add_argument("--games", type=int, help="play exactly N games, then shut down StarCraft/shim/bot (default: forever)")
    p.add_argument("--no-bot", action="store_true", help="start only StarCraft + shim; run the bot yourself")
    p.add_argument("--no-game", action="store_true", help="start only shim + bot (StarCraft already running)")
    p.add_argument("--wait", action="store_true", help="block until the bot process exits")
    p.add_argument("--stop", action="store_true", help="kill StarCraft, shim and bot, then exit")
    return p


def to_ps_args(a: argparse.Namespace) -> list[str]:
    args: list[str] = []
    if a.stop:
        return ["-Stop"]
    args += ["-Bot", a.bot, "-Port", str(a.port)]
    if a.frame_skip is not None:
        args += ["-FrameSkip", str(a.frame_skip)]
    if a.speed is not None:
        args += ["-Speed", str(a.speed)]
    if a.map:
        args += ["-Map", a.map]
    if a.race:
        args += ["-Race", a.race]
    if a.enemy_race:
        args += ["-EnemyRace", a.enemy_race]
    if a.games is not None:
        if a.games < 1:
            sys.exit("--games must be >= 1")
        args += ["-Games", str(a.games)]
    for flag, name in ((a.no_bot, "-NoBot"), (a.no_game, "-NoGame"), (a.wait, "-Wait")):
        if flag:
            args.append(name)
    return args


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def powershell_invocation(script: Path) -> tuple[str, str]:
    """Return (powershell executable, script path as PowerShell will see it)."""
    if os.name == "nt":
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return exe, str(script)
    if is_wsl():
        exe = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        win_path = subprocess.run(["wslpath", "-w", str(script)], check=True, capture_output=True, text=True).stdout.strip()
        return exe, win_path
    sys.exit("run.py needs Windows or WSL (the visible game is Windows-only). "
             "For headless games on Linux use scripts/run_openbw.sh.")


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    exe, script = powershell_invocation(NATIVE_SCRIPT)
    cmd = [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, *to_ps_args(a)]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
