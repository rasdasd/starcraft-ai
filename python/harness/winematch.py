"""Fast games against published bots: StarCraft under Wine in WSL, the two clients on a UDP LAN
between Linux network namespaces (the sc-docker layout without Docker).

    wsl -d Ubuntu -u root -- /home/<you>/.venvs/bwbot/bin/python -m harness.winematch \\
        --opponent Locutus --opponent Stardust --games 8 --parallel 4

Why: on native Windows two clients can only meet over "Local PC", which paces a game at ~64
frames/s and is machine-wide (one game at a time). Here every game slot gets its own bridge
`scbr<k>` (10.77.<k>.0/24) with namespaces `sc<k>a` (us, host) and `sc<k>b` (opponent, joins
the first LAN game it sees), its own Xvfb displays (started inside the namespaces: X's abstract
sockets are per network namespace and WSLg owns /tmp/.X11-unix) and its own Wine prefixes, so
slots neither share UDP 6111 nor see each other's games. Needs root (namespaces) and the Linux
packages wine, wine32, xvfb (`scripts/setup_wsl.sh` does not install them).

`--watch` puts our client in a window on $DISPLAY instead (WSLg's X server: its path socket is
reachable from the namespaces) and renders every frame; `python run.py --opponent NAME` uses it.

Bots, bwapi.ini, the tournament module (speed 0 + frame skip in both clients) and the results
format are shared with `harness.botmatch`; instances and prefixes live under /root/sc (ext4).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .botmatch import (GAME, ROOT, SHIM_DLL, bwapi_ini, ensure_bwta_dlls, game_row, install_tournament_module,
                       load_bot, safe_name, seed_character)
from .selfplay import DEFAULT_MAPS, Player

WORK = Path(os.environ.get("WINEMATCH_DIR", "/root/sc"))
WINE_ENV = {"WINEDEBUG": "-all", "WINEDLLOVERRIDES": "mscoree,mshtml="}
SC_REG = "HKEY_CURRENT_USER\\Software\\Blizzard Entertainment\\Starcraft"
RENDERER = os.environ.get("WINEMATCH_RENDERER", "gdi")    # ~30% faster than wined3d's GL on llvmpipe


def sh(*cmd: str, check: bool = False, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------- network
def ns_name(slot: int, side: str) -> str:
    return f"sc{slot}{side}"


def setup_net(slot: int) -> None:
    """Bridge scbr<k> + namespaces sc<k>a (.10) / sc<k>b (.11); idempotent."""
    br = f"scbr{slot}"
    if sh("ip", "link", "show", br).returncode != 0:
        sh("ip", "link", "add", br, "type", "bridge", check=True)
        sh("ip", "addr", "add", f"10.77.{slot}.1/24", "dev", br, check=True)
        sh("ip", "link", "set", br, "up", check=True)
    have = sh("ip", "netns", "list").stdout.split()
    for side, host in (("a", 10), ("b", 11)):
        ns = ns_name(slot, side)
        if ns in have:
            continue
        vh, vn = f"v{slot}h{side}", f"v{slot}n{side}"
        sh("ip", "link", "del", vh)
        sh("ip", "netns", "add", ns, check=True)
        sh("ip", "link", "add", vh, "type", "veth", "peer", "name", vn, check=True)
        sh("ip", "link", "set", vn, "netns", ns, check=True)
        sh("ip", "link", "set", vh, "master", br, "up", check=True)
        sh("ip", "-n", ns, "addr", "add", f"10.77.{slot}.{host}/24", "dev", vn, check=True)
        sh("ip", "-n", ns, "link", "set", vn, "up", check=True)
        sh("ip", "-n", ns, "link", "set", "lo", "up", check=True)
        sh("ip", "-n", ns, "route", "add", "default", "via", f"10.77.{slot}.1", check=True)


# ---------------------------------------------------------------------------- wine + instances
def ensure_prefix(prefix: Path) -> None:
    """A Wine prefix, copied from a template made once with wineboot (a fresh boot takes ~20 s)."""
    if prefix.exists():
        return
    tpl = WORK / "prefix_template"
    if not tpl.exists():
        env = dict(os.environ, WINEPREFIX=str(tpl), **WINE_ENV)
        env.pop("DISPLAY", None)
        subprocess.run(["wineboot", "-i"], env=env, capture_output=True, check=True)
        subprocess.run(["wineserver", "-w"], env=env, capture_output=True)
    shutil.copytree(tpl, prefix, symlinks=True)


def set_registry(prefix: Path, inst: Path) -> None:
    """InstallPath (BWAPI <= 4.1.2 finds bwapi.ini through it) and no tips/intro/music."""
    win = "Z:" + str(inst).replace("/", "\\\\") + "\\\\"
    reg = prefix / "starcraft.reg"
    reg.write_text("REGEDIT4\n\n[" + SC_REG + "]\n" + f'"InstallPath"="{win}"\n' + "".join(
        f'"{k}"=dword:{v:08x}\n' for k, v in (("tip", 0), ("tipnum", 0), ("intro", 0x200), ("introX", 0),
                                               ("CPUThrottle", 0), ("music", 0), ("sfx", 0)))
        + f'\n[HKEY_CURRENT_USER\\Software\\Wine\\Direct3D]\n"renderer"="{RENDERER}"\n', encoding="utf-8")
    env = dict(os.environ, WINEPREFIX=str(prefix), **WINE_ENV)
    env.pop("DISPLAY", None)
    subprocess.run(["wine", "regedit", str(reg)], env=env, capture_output=True)


def game_cache() -> Path:
    """`game/` copied once onto ext4 (drvfs is slow and has no Linux permissions)."""
    cache = WORK / "game"
    stamp = cache / ".stamp"
    src_stamp = str(int((GAME / "StarCraft.exe").stat().st_mtime))
    if not stamp.exists() or stamp.read_text() != src_stamp:
        if cache.exists():
            shutil.rmtree(cache)
        shutil.copytree(GAME, cache, ignore=shutil.ignore_patterns("AI", "logs", "replays", "wmode.dll", "WMode.dll"))
        stamp.write_text(src_stamp)
    return cache


def prepare_instance(inst: Path, bwapi_dll: Path, ai_files: list[Path], character: str) -> None:
    if not inst.exists():
        shutil.copytree(game_cache(), inst, symlinks=True)
    bd = inst / "bwapi-data"
    if (bd / "AI").exists():
        shutil.rmtree(bd / "AI")
    for sub in ("AI", "read", "write", "logs", "BWTA", "BWTA2", "replays"):
        (bd / sub).mkdir(parents=True, exist_ok=True)
    for stale in ("TournamentModule.dll", "tm_settings.ini"):
        (bd / stale).unlink(missing_ok=True)
    seed_character(inst, character)
    shutil.copy2(bwapi_dll, bd / "BWAPI.dll")
    for f in ai_files:
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, bd / "AI" / f.name)


def lan_ini(*a, window: Optional[tuple[int, int]] = None, **kw) -> str:
    """bwapi.ini for a LAN client: full screen on its Xvfb, or a `window` (w, h) BWAPI scales the game to."""
    ini = bwapi_ini(*a, lan_mode="Local Area Network (UDP)", join="JOIN_FIRST", **kw)
    if window is None:
        return ini.replace("windowed = ON", "windowed = OFF")
    return (ini.replace("width = 640", f"width = {window[0]}").replace("height = 480", f"height = {window[1]}")
            .replace("left = 0", "left = 40"))


def raise_window(display: str, title: str = "Brood War", timeout: float = 90.0) -> None:
    """Bring our client's window to the front once it exists (new WSLg windows can open behind others)."""
    if shutil.which("xdotool") is None:
        return
    env = dict(os.environ, DISPLAY=display)
    end = time.time() + timeout
    while time.time() < end:
        found = sh("xdotool", "search", "--name", f"^{title}$", env=env).stdout.split()
        if found:
            for wid in found:
                sh("xdotool", "windowactivate", wid, env=env)
                sh("xdotool", "windowraise", wid, env=env)
            return
        time.sleep(1.0)


# ---------------------------------------------------------------------------- match
def play(index: int, me: Player, opp_name: str, map_path: str, run_id: str, run_dir: Path,
         args: argparse.Namespace, slot: int) -> dict:
    bot_dir, meta = load_bot(opp_name)
    if meta["botType"] not in ("AI_MODULE", "EXE"):
        raise SystemExit(f"{meta['name']}: {meta['botType']} bots are not supported under Wine yet")
    gdir = run_dir / "games" / f"{index:04d}"
    gdir.mkdir(parents=True, exist_ok=True)
    setup_net(slot)
    base = WORK / f"s{slot}"
    inst = {"a": base / "a", "b": base / "b"}
    prefix = {s: base / f"prefix_{s}" for s in "ab"}
    names = {"a": safe_name(me.name)[:20], "b": safe_name(meta["name"])[:20]}
    prepare_instance(inst["a"], GAME / "bwapi-data" / "BWAPI.dll", [SHIM_DLL], names["a"])
    prepare_instance(inst["b"], bot_dir / "BWAPI.dll", list((bot_dir / "AI").iterdir()), names["b"])
    client = meta["botType"] != "AI_MODULE"
    game_id = f"{run_id}-g{index:04d}"
    replay = f"bwapi-data/replays/{game_id}.rep" if args.replays else ""
    tm = {s: install_tournament_module(inst[s], args.tm_frame_skip) if args.tm and not (args.watch and s == "a")
          else "" for s in "ab"}
    (inst["a"] / "bwapi-data" / "bwapi.ini").write_text(
        lan_ini(f"bwapi-data/AI/{SHIM_DLL.name}", me.race or "Terran", names["a"], map_path, True, replay, 0,
                tm["a"], window=args.watch_size if args.watch else None), encoding="utf-8")
    (inst["b"] / "bwapi-data" / "bwapi.ini").write_text(
        lan_ini("" if client else f"bwapi-data/AI/{meta['file']}", meta.get("race", "Random"),
                names["b"], map_path, False, "", 0, tm["b"]), encoding="utf-8")
    for s in "ab":
        ensure_prefix(prefix[s])
        set_registry(prefix[s], inst[s])

    procs: list[subprocess.Popen] = []
    logs = []
    t0 = time.time()
    timed_out = True
    started = False

    def spawn(side: str, cmd: list[str], log: str, env: dict, cwd: Path) -> subprocess.Popen:
        fh = open(gdir / log, "wb")
        logs.append(fh)
        p = subprocess.Popen(["ip", "netns", "exec", ns_name(slot, side), *cmd], cwd=cwd, env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
        procs.append(p)
        return p

    shown = {"a"} if args.watch else set()

    def wine_env(side: str, **extra: str) -> dict:
        display = os.environ.get("DISPLAY", ":0") if side in shown else f":{100 + 2 * slot + (side == 'b')}"
        return dict(os.environ, DISPLAY=display, WINEPREFIX=str(prefix[side]), **WINE_ENV, **extra)

    try:
        for s in "ab":
            if s not in shown:
                spawn(s, ["Xvfb", wine_env(s)["DISPLAY"], "-screen", "0", "800x600x24", "-nolisten", "tcp"],
                      f"xvfb_{s}.log", dict(os.environ), inst[s])
        time.sleep(1.0)
        launch = ["wine", "injectory_x86.exe", "--launch", "StarCraft.exe", "--inject", "bwapi-data/BWAPI.dll"]
        spawn("a", launch, "wine_a.log",
              wine_env("a", BWBOT_PORT=str(args.port), BWBOT_HOST="127.0.0.1", BWBOT_NO_SPAWN="1"), inst["a"])
        if "a" in shown:
            threading.Thread(target=raise_window, args=(wine_env("a")["DISPLAY"],), daemon=True).start()
        benv = dict(os.environ, BWBOT_RESULT=str(gdir / "result_a.json"), BWBOT_LOG_DIR=str(gdir / "logs"),
                    BWBOT_GAME_ID=game_id, BWBOT_SIDE="a", PYTHONPATH=str(ROOT / "python"))
        if me.profile:
            benv["BWBOT_PROFILE"] = me.profile
        benv.update({k: str(v) for k, v in me.env.items()})
        brain = spawn("a", [sys.executable, "-m", "harness.brain", me.spec, "--port", str(args.port), "--games", "1",
                            "--max-frames", str(args.max_frames), "--connect-timeout", "180", "--no-apm-hud",
                            *me.args, *args.brain_args], "brain_a.log", benv, ROOT / "python")
        time.sleep(args.join_delay)
        spawn("b", launch, "wine_b.log", wine_env("b"), inst["b"])
        if client:
            time.sleep(5.0)
            spawn("b", ["wine", f"bwapi-data/AI/{meta['file']}"], "opponent.log", wine_env("b"), inst["b"])
        deadline = t0 + args.timeout_min * 60
        lobby_deadline = t0 + args.lobby_timeout
        while time.time() < deadline and brain.poll() is None:
            time.sleep(0.5)
            if not started and time.time() > lobby_deadline:
                started = b"game start:" in (gdir / "brain_a.log").read_bytes()
                if not started:
                    break
        timed_out = brain.poll() is None
    finally:
        for s in "ab":
            subprocess.run(["wineserver", "-k"], env=wine_env(s), capture_output=True)
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for fh in logs:
            fh.close()
        if replay and (inst["a"] / replay).is_file():
            shutil.move(str(inst["a"] / replay), str(gdir / "replay.rep"))
    row = game_row(index, game_id, map_path, me, meta, gdir, args.max_frames, timed_out, t0, "winematch")
    if not started and row["frames"] == 0:
        row["reason"] = "no_start"
    return row


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default="adjutant", help="our player (module[:Class][@profile])")
    ap.add_argument("--opponent", action="append", default=[], help="bot name in bots/ or on SSCAIT (rotated)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1, help="game slots at once (each ~2 CPU cores)")
    ap.add_argument("--maps", nargs="*")
    ap.add_argument("--max-frames", type=int, default=24 * 60 * 25)
    ap.add_argument("--timeout-min", type=float, default=20)
    ap.add_argument("--lobby-timeout", type=float, default=120, help="seconds until the game must have started")
    ap.add_argument("--port", type=int, default=8795, help="brain <-> shim port (per namespace, so shared)")
    ap.add_argument("--join-delay", type=float, default=8.0, help="seconds between host and joiner start")
    ap.add_argument("--run-id", default=time.strftime("wm%Y%m%d-%H%M%S"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--no-replays", dest="replays", action="store_false")
    ap.add_argument("--brain-args", nargs="*", action="extend", default=[],
                    help="extra bwbot.run options for our brain (repeatable: --brain-args=--speed=42)")
    ap.add_argument("--no-tm", dest="tm", action="store_false", help="don't load the Tournament Manager module")
    ap.add_argument("--tm-frame-skip", type=int, default=256, help="render every N frames (tournament module)")
    ap.add_argument("--watch", action="store_true",
                    help="show our client in a window on $DISPLAY (WSLg), every frame rendered; the opponent "
                         "stays headless. Pace it with --brain-args=--speed=42. Implies --parallel 1")
    ap.add_argument("--watch-size", default="960x720",
                    help="--watch window size, WxH (scaled in software: larger windows cap the frame rate)")
    args = ap.parse_args(argv)
    try:
        args.watch_size = tuple(int(v) for v in args.watch_size.lower().split("x", 1))
    except ValueError:
        ap.error("--watch-size wants WxH, e.g. 1280x960")

    if not args.opponent:
        ap.error("at least one --opponent")
    if args.watch:
        args.parallel = 1
    if os.name == "nt" or os.geteuid() != 0:
        raise SystemExit("winematch needs root in WSL/Linux: wsl -d Ubuntu -u root -- <venv>/bin/python -m "
                         "harness.winematch ...")
    for tool in ("wine", "Xvfb", "ip"):
        if shutil.which(tool) is None:
            raise SystemExit(f"missing {tool} (apt install wine wine32 xvfb iproute2)")
    for p in (GAME / "StarCraft.exe", GAME / "injectory_x86.exe", SHIM_DLL):
        if not p.exists():
            raise SystemExit(f"missing {p} (scripts/setup_windows.ps1, scripts/build_shim.ps1)")
    ensure_bwta_dlls()
    game_cache()
    me = Player.parse(args.p1)
    maps = args.maps or DEFAULT_MAPS
    run_dir = Path(args.out) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for opp in dict.fromkeys(args.opponent):
        load_bot(opp)
    ensure_prefix(WORK / "s0" / "prefix_a")          # builds the template before threads race for it
    rows = []
    out_lock = threading.Lock()
    free: queue.Queue[int] = queue.Queue()
    for s in range(max(1, args.parallel)):
        free.put(s)

    def one(i: int) -> None:
        slot = free.get()
        opp = args.opponent[i % len(args.opponent)]
        try:
            time.sleep(2.0 * slot if i < args.parallel else 0.0)
            row = play(i, me, opp, maps[i % len(maps)], args.run_id, run_dir, args, slot)
        finally:
            free.put(slot)
        with out_lock:
            rows.append(row)
            with (run_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[winematch] g{i:04d} {me.name} vs {opp} on {Path(row['map']).stem}: "
                  f"{row['winner_name'] or '-'} ({row['reason']}, {row['frames']} frames, {row['wall_s']:.0f}s)",
                  flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(one, range(args.games)))
    rows.sort(key=lambda r: r["game"])
    from adjutant.learn.report import print_report
    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
