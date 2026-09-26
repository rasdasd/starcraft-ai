#!/usr/bin/env python3
"""Launch the bot against a visible StarCraft window. Works from Windows or WSL.

    python run.py                          # Adjutant, windowed StarCraft, default settings
    python run.py --race Zerg --speed 42
    python run.py --profile search         # pick a profile (see --list)
    python run.py --games 3                # play exactly 3 games, then shut everything down
    python run.py --opponent Dave_Churchill            # watch a game against a published bot
    python run.py --opponent Locutus --race Zerg --map "maps/BroodWar/aiide/(2)Destination.scx"
    python run.py --stop                   # tear everything down
    python run.py --list                   # list valid --bot values, profiles and local opponents

The visible game (StarCraft 1.16.1 + BWAPI) only exists on Windows, so this script always drives
scripts/run_native.ps1 with PowerShell. From WSL it goes through Windows interop (powershell.exe),
which puts the game, shim and bot windows on the Windows desktop. For headless OpenBW games use
scripts/run_openbw.sh instead.

With --opponent the enemy is a published SSCAIT bot instead of the built-in AI: this runs
`harness.winematch --watch` in WSL (as root, for its network namespaces). Both clients run under
Wine; the opponent headless, ours in a WSLg window at a watchable speed. Bots in bots/ are used
as-is; any other SSCAIT name is downloaded on first use.

Add new options to `build_parser()` and translate them in `to_ps_args()`.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE_SCRIPT = ROOT / "scripts" / "run_native.ps1"
PYTHON_DIR = ROOT / "python"
BOTS_DIR = ROOT / "bots"
WSL = "wsl.exe"
WSL_DISTRO = os.environ.get("BWBOT_WSL_DISTRO", "Ubuntu")
WATCH_SPEED = 42            # ms per frame: the game's own "Fastest", what human games are played at
SKIP_DIRS = {".venv", "tests", "__pycache__", "build", "dist"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", default="adjutant", help="python module holding the Bot subclass")
    p.add_argument("--profile", help="bot profile name or JSON file (sets BWBOT_PROFILE; see --list)")
    p.add_argument("--port", type=int, default=8765, help="shim <-> bot TCP port")
    p.add_argument("--frame-skip", type=int, help="bot decides every N frames (default: bot config)")
    p.add_argument("--speed", type=int, help="game speed: -1 game default, 0 fastest, 42 normal (default: bot config)")
    p.add_argument("--map", help="map relative to game/, e.g. maps/BroodWar/sscai/(4)Python.scx")
    p.add_argument("--race", choices=["Terran", "Protoss", "Zerg", "Random"], help="bot race")
    p.add_argument("--enemy-race", choices=["Terran", "Protoss", "Zerg", "Random"], help="built-in AI race")
    p.add_argument("--opponent", help="play a published bot instead of the built-in AI, e.g. Dave_Churchill "
                                      f"(speed defaults to {WATCH_SPEED}; see --list)")
    p.add_argument("--games", type=int, help="play exactly N games, then shut down StarCraft/shim/bot (default: forever)")
    p.add_argument("--no-bot", action="store_true", help="start only StarCraft + shim; run the bot yourself")
    p.add_argument("--no-game", action="store_true", help="start only shim + bot (StarCraft already running)")
    p.add_argument("--wait", action="store_true", help="block until the bot process exits")
    p.add_argument("--stop", action="store_true", help="kill StarCraft, shim and bot, then exit")
    p.add_argument("--list", action="store_true", help="list valid --bot values and their profiles, then exit")
    return p


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PYTHON_DIR).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _dict_keys(tree: ast.Module, var: str) -> list[str]:
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == var and isinstance(node.value, ast.Dict):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return []


def discover_bots() -> list[dict]:
    """Statically scan python/ for modules exposing `BOT` (what `bwbot.run.load_bot` uses by default).

    No imports, so this works without the bot venv (e.g. from WSL). Returns one entry per module:
    {module, default, classes: [(name, profile, doc)], profiles: [names]}.
    """
    trees: dict[str, ast.Module] = {}
    files: dict[str, Path] = {}
    for path in sorted(PYTHON_DIR.rglob("*.py")):
        if SKIP_DIRS & set(path.relative_to(PYTHON_DIR).parts):
            continue
        try:
            trees[_module_name(path)] = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        files[_module_name(path)] = path

    bases: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, set()).update(_base_name(b) for b in node.bases)

    def is_bot(name: str, seen: frozenset = frozenset()) -> bool:
        if name == "Bot":
            return True
        return name not in seen and any(is_bot(b, seen | {name}) for b in bases.get(name, ()))

    bots = []
    for mod, tree in trees.items():
        default = next((n.value.id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
                        and any(isinstance(t, ast.Name) and t.id == "BOT" for t in n.targets)), None)
        if default is None or not is_bot(default):
            continue
        classes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_") and is_bot(node.name):
                profile = next((s.value.value for s in node.body
                                if isinstance(s, (ast.Assign, ast.AnnAssign)) and isinstance(s.value, ast.Constant)
                                and any(isinstance(t, ast.Name) and t.id == "profile"
                                        for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))), None)
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                classes.append((node.name, profile, doc[0] if doc else ""))
        profiles_file = files[mod].parent / "profiles.py"
        profiles = []
        if profiles_file.exists():
            profiles = sorted(_dict_keys(ast.parse(profiles_file.read_text(encoding="utf-8")), "BUILTINS"))
        bots.append({"module": mod, "default": default, "classes": classes, "profiles": profiles})
    return sorted(bots, key=lambda b: b["module"])


def print_bots() -> None:
    bots = discover_bots()
    if not bots:
        print(f"no bots found under {PYTHON_DIR}")
        return
    rows = []
    for b in bots:
        rows.append((b["module"], f"-> {b['default']}"))
        for name, profile, doc in b["classes"]:
            note = " ".join(x for x in (f"[profile: {profile}]" if profile else "", doc) if x)
            rows.append((f"{b['module']}:{name}", note))
    width = max(len(r[0]) for r in rows)
    print("Valid --bot values:")
    for spec, note in rows:
        print(f"  {spec.ljust(width)}  {note}".rstrip())
    for b in bots:
        if b["profiles"]:
            print(f"\nProfiles for {b['module']} (select with --profile <name>):")
            for name in b["profiles"]:
                print(f"  {name}")
    opponents = local_opponents()
    if opponents:
        print("\nOpponents in bots/ (select with --opponent <name>; other SSCAIT names are downloaded):")
        for name, race in opponents:
            print(f"  {name.ljust(20)}  {race}")


def local_opponents() -> list[tuple[str, str]]:
    out = []
    for meta in sorted(BOTS_DIR.glob("*/bot.json")):
        try:
            race = json.loads(meta.read_text(encoding="utf-8")).get("race", "")
        except (OSError, ValueError):
            continue
        out.append((meta.parent.name, race))
    return out


def wsl_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    drive, rest = str(path).split(":", 1)
    return f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}"


def wsl_python() -> str:
    """The WSL venv winematch runs in: $BWBOT_WSL_PYTHON, else ~/.venvs/bwbot of the default WSL user."""
    if os.environ.get("BWBOT_WSL_PYTHON"):
        return os.environ["BWBOT_WSL_PYTHON"]
    home = subprocess.run([WSL, "-d", WSL_DISTRO, "--", "sh", "-c", "echo $HOME"], capture_output=True,
                          text=True).stdout.strip()
    if not home:
        sys.exit(f"--opponent needs WSL ({WSL_DISTRO}); set BWBOT_WSL_DISTRO / BWBOT_WSL_PYTHON if it lives elsewhere")
    return f"{home}/.venvs/bwbot/bin/python"


def watch_cmd(a: argparse.Namespace) -> list[str]:
    """One `harness.winematch --watch` game: the opponent headless under Wine, our client in a WSLg
    window at a watchable speed, and a wall-clock timeout that fits the frame limit at that speed."""
    if a.no_bot or a.no_game:
        sys.exit("--opponent starts both games and our bot itself; --no-bot / --no-game don't apply")
    spec = a.bot + (f"@{a.profile}" if a.profile else "") + (f"/{a.race}" if a.race else "")
    speed = WATCH_SPEED if a.speed is None else a.speed
    max_frames = 24 * 60 * 25
    minutes = max_frames * max(speed, 42) / 1000 / 60 + 10
    brain = [f"--speed={speed}"] + ([f"--frame-skip={a.frame_skip}"] if a.frame_skip is not None else [])
    cmd = [WSL, "-d", WSL_DISTRO, "-u", "root", "--cd", wsl_path(PYTHON_DIR), "--", wsl_python(),
           "-m", "harness.winematch", "--watch", "--p1", spec,
           "--opponent", a.opponent, "--games", str(a.games or 1), "--max-frames", str(max_frames),
           "--timeout-min", f"{minutes:.0f}", "--run-id", time.strftime("watch%Y%m%d-%H%M%S"),
           *(f"--brain-args={b}" for b in brain)]
    if a.map:
        cmd += ["--maps", a.map]
    return cmd


def to_ps_args(a: argparse.Namespace) -> list[str]:
    args: list[str] = []
    if a.stop:
        return ["-Stop"]
    args += ["-Bot", a.bot, "-Port", str(a.port)]
    if a.profile:
        args += ["-BotProfile", a.profile]
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


def resolve_profile(bot: str, profile: str) -> str:
    """Validate a profile name against the bot's built-ins; turn a JSON path into an absolute Windows path."""
    path = Path(profile)
    if path.suffix == ".json" or path.exists():
        if not path.is_file():
            sys.exit(f"--profile: file not found: {profile}")
        path = path.resolve()
        if is_wsl():
            return subprocess.run(["wslpath", "-w", str(path)], check=True, capture_output=True, text=True).stdout.strip()
        return str(path)
    module = bot.partition(":")[0]
    entry = next((b for b in discover_bots() if b["module"] == module), None)
    if entry is None or not entry["profiles"]:
        sys.exit(f"--profile: bot {bot!r} has no profiles (see `python run.py --list`)")
    if profile not in entry["profiles"]:
        sys.exit(f"--profile: unknown profile {profile!r} for {module}; valid: {', '.join(entry['profiles'])}")
    return profile


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
    if a.list:
        print_bots()
        return 0
    if a.profile and not a.stop:
        a.profile = resolve_profile(a.bot, a.profile)
    if a.opponent and not a.stop:
        if os.name != "nt" and not is_wsl():
            sys.exit("--opponent needs Windows or WSL (the visible game is Windows-only)")
        cmd = watch_cmd(a)
        print("+", " ".join(cmd), flush=True)
        return subprocess.call(cmd)
    exe, script = powershell_invocation(NATIVE_SCRIPT)
    cmd = [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, *to_ps_args(a)]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
