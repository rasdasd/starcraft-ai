"""Games against published bots on native Windows StarCraft 1.16.1 (the Tournament Manager route).

    python -m harness.botmatch --list                       # SSCAIT bots (name, race, type)
    python -m harness.botmatch --fetch "Dave Churchill" --fetch Stardust
    python -m harness.botmatch --p1 adjutant --opponent "Dave Churchill" --opponent Stardust --games 6

Docker is not required: each side gets its own StarCraft folder under `runs/botmatch/inst_<side>`
(hard links to `game/`, so it costs no disk), with its own `bwapi-data/BWAPI.dll` — the opponent
runs on the BWAPI version it was built for, like on SSCAIT — and its own `bwapi.ini`. Both clients
meet over the "Local PC" network: our side hosts, the opponent joins (`game = JOIN_FIRST`).

Our side loads `shim/build/shim_module.dll` as its AI module with `BWBOT_NO_SPAWN=1`, and the brain
(`python -m harness.brain`, Windows venv) connects to it. The opponent has no result file of its
own, so frame-limit games are judged on BWAPI scores (ours vs. the enemy's, both from our brain).
Results go to `runs/<run-id>/results.jsonl` in the same format as `harness.selfplay`, so
`adjutant.learn.report` reads both. One match at a time (Local PC is machine-wide).

Opponent bots live in `bots/<name>/`: `bot.json` (name, race, botType, file), `AI/` (the bot's
files, copied to `bwapi-data/AI/`), `BWAPI.dll`. Supported: AI_MODULE (DLL) bots, and EXE / JAVA
client bots on a best-effort basis (they attach through BWAPI shared memory).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .selfplay import DEFAULT_MAPS, Player, judge, log_summary

ROOT = Path(__file__).resolve().parents[2]
GAME = ROOT / "game"
BOTS = ROOT / "bots"
SHIM_DLL = ROOT / "shim" / "build" / "shim_module.dll"
API = "https://sscaitournament.com/api/bots.php"


# ---------------------------------------------------------------------------- bots
def sscait_bots() -> list[dict]:
    return json.loads(_download(API).decode("utf-8"))


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "adjutant-botmatch"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch(name: str, bots: Optional[list[dict]] = None) -> Path:
    bots = bots if bots is not None else sscait_bots()
    info = next((b for b in bots if b["name"].lower() == name.lower()), None)
    if info is None:
        raise SystemExit(f"bot {name!r} not on SSCAIT")
    d = BOTS / safe_name(info["name"])
    if d.exists():
        shutil.rmtree(d)
    (d / "AI").mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(_download(info["botBinary"]))) as z:
        z.extractall(d / "AI")
    (d / "BWAPI.dll").write_bytes(_download(info["bwapiDLL"]))
    exts = {"AI_MODULE": ".dll", "EXE": ".exe", "JAVA_JNI": ".jar", "JAVA_MIRROR": ".jar"}
    ext = exts.get(info.get("botType", ""), ".dll")
    files = sorted((d / "AI").glob(f"*{ext}"), key=lambda p: (p.stem.lower() != safe_name(info["name"]).lower(), p.name))
    meta = {"name": info["name"], "race": info.get("race", "Random"), "botType": info.get("botType", "AI_MODULE"),
            "file": files[0].name if files else "", "source": info["botBinary"]}
    (d / "bot.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[botmatch] fetched {info['name']} ({meta['race']}, {meta['botType']}, {meta['file']}) -> {d}")
    return d


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def load_bot(name: str) -> tuple[Path, dict]:
    d = BOTS / safe_name(name)
    if not (d / "bot.json").is_file():
        d = fetch(name)
    return d, json.loads((d / "bot.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- instances
def _link_tree(src: Path, dst: Path, skip: set[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name.lower() in skip:
            continue
        q = dst / p.name
        if p.is_dir():
            if not q.exists():
                subprocess.run(["cmd", "/c", "mklink", "/J", str(q), str(p)], check=True, capture_output=True)
        elif not q.exists():
            try:
                os.link(p, q)
            except OSError:
                shutil.copy2(p, q)


def prepare_instance(inst: Path, bwapi_dll: Path, ai_files: list[Path]) -> None:
    """A StarCraft folder sharing `game/` via hard links/junctions, with a private bwapi-data."""
    _link_tree(GAME, inst, skip={"bwapi-data", "characters", "maps"})
    if not (inst / "maps").exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(inst / "maps"), str(GAME / "maps")], check=True,
                       capture_output=True)
    if (GAME / "characters").is_dir() and not (inst / "characters").exists():
        shutil.copytree(GAME / "characters", inst / "characters")
    bd = inst / "bwapi-data"
    if (GAME / "bwapi-data" / "data").is_dir() and not (bd / "data").exists():
        shutil.copytree(GAME / "bwapi-data" / "data", bd / "data")
    if (bd / "AI").exists():
        shutil.rmtree(bd / "AI")
    for sub in ("AI", "read", "write", "logs", "BWTA", "BWTA2", "replays"):
        (bd / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(bwapi_dll, bd / "BWAPI.dll")
    for f in ai_files:
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, bd / "AI" / f.name)


def bwapi_ini(ai: str, race: str, name: str, map_path: str, host: bool, replay: str, left: int,
              tournament: str = "", lan_mode: str = "Local PC", join: str = "JOIN_FIRST") -> str:
    return f"""[ai]
ai = {ai}
ai_dbg =
tournament = {tournament}

[auto_menu]
auto_menu = LAN
pause_dbg = OFF
lan_mode = {lan_mode}
auto_restart = OFF
map = {map_path if host else ''}
game = {'' if host else join}
mapiteration = RANDOM
race = {race}
enemy_count = 1
enemy_race = Default
game_type = MELEE
save_replay = {replay}
wait_for_min_players = 2
wait_for_max_players = 2
wait_for_time = 60000
character_name = {name[:20]}

[config]
holiday = OFF
shared_memory = ON
console_attach_on_startup = FALSE
console_alloc_on_startup = FALSE
console_attach_auto = TRUE
console_alloc_auto = TRUE

[window]
windowed = ON
left = {left}
top = 40
width = 640
height = 480

[starcraft]
sound = OFF
screenshots = gif
drop_players = ON
"""


SC_KEY = r"Software\Blizzard Entertainment\Starcraft"
BWMIRROR_JAR = "https://github.com/vjurenka/BWMirror/raw/master/dist/bwmirror_v2_5.jar"
BWTA_DLLS = ("libgmp-10.dll", "libmpfr-4.dll")


def _no_tips() -> None:
    try:
        import winreg
        k = winreg.CreateKey(winreg.HKEY_CURRENT_USER, SC_KEY)
        winreg.SetValueEx(k, "tip", 0, winreg.REG_DWORD, 0)
        winreg.SetValueEx(k, "tipnum", 0, winreg.REG_DWORD, 0)
    except OSError:
        pass


def install_path(value: Optional[str]) -> Optional[str]:
    """Set (or delete, for None) HKCU ...\\Starcraft\\InstallPath; returns the previous value.
    BWAPI 4.1.2 and older find bwapi-data/bwapi.ini through it, not through the working directory."""
    import winreg
    k = winreg.CreateKey(winreg.HKEY_CURRENT_USER, SC_KEY)
    try:
        prev = winreg.QueryValueEx(k, "InstallPath")[0]
    except OSError:
        prev = None
    if value is None:
        try:
            winreg.DeleteValue(k, "InstallPath")
        except OSError:
            pass
    else:
        winreg.SetValueEx(k, "InstallPath", 0, winreg.REG_SZ, value)
    return prev


def read_install_path() -> Optional[str]:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, SC_KEY) as k:
            return winreg.QueryValueEx(k, "InstallPath")[0]
    except OSError:
        return None


def ensure_bwta_dlls() -> None:
    """BWTA2 bots (UAlbertaBot, many 4.1.2 bots) need libgmp/libmpfr next to StarCraft.exe; the
    tournament images ship them, `game/` does not. BWMirror's jar bundles both."""
    if all((GAME / n).exists() for n in BWTA_DLLS):
        return
    with zipfile.ZipFile(io.BytesIO(_download(BWMIRROR_JAR))) as z:
        for info in z.infolist():
            if Path(info.filename).name in BWTA_DLLS:
                (GAME / Path(info.filename).name).write_bytes(z.read(info))
    print(f"[botmatch] installed {', '.join(BWTA_DLLS)} into {GAME}")


TM_REQUIRED = ("https://github.com/davechurchill/StarcraftAITournamentManager/raw/master/server/required/"
               "Required_BWAPI_{v}.zip")
TM_VERSIONS = ("440", "420", "412", "401B", "374")
TM_DIR = ROOT / "bots" / "_tournament_module"


def _sha1(path: Path) -> str:
    import hashlib
    return hashlib.sha1(path.read_bytes()).hexdigest()


def tournament_module(bwapi_dll: Path) -> Optional[Path]:
    """The Tournament Manager's TournamentModule.dll built for the BWAPI version of `bwapi_dll`
    (matched by the BWAPI.dll the TM ships with each module; downloaded into bots/ on first use)."""
    want = _sha1(bwapi_dll)
    for v in TM_VERSIONS:
        d = TM_DIR / v
        if not (d / "TournamentModule.dll").exists():
            with zipfile.ZipFile(io.BytesIO(_download(TM_REQUIRED.format(v=v)))) as z:
                d.mkdir(parents=True, exist_ok=True)
                for name in ("bwapi-data/BWAPI.dll", "bwapi-data/TournamentModule.dll"):
                    (d / Path(name).name).write_bytes(z.read(name))
        if _sha1(d / "BWAPI.dll") == want:
            return d / "TournamentModule.dll"
    return None


def install_tournament_module(inst: Path, frame_skip: int) -> str:
    """Copies the matching tournament module into `inst` with speed 0 + `frame_skip` rendering and no
    timeouts / frame limit (the harness judges those). Returns the bwapi.ini `tournament` value."""
    bd = inst / "bwapi-data"
    tm = tournament_module(bd / "BWAPI.dll")
    if tm is None:
        print(f"[botmatch] no tournament module for {bd / 'BWAPI.dll'}; that client renders every frame")
        return ""
    shutil.copy2(tm, bd / "TournamentModule.dll")
    (bd / "tm_settings.ini").write_text(
        f"LocalSpeed 0\nFrameSkip {frame_skip}\nGameFrameLimit 0\nDrawUnitInfo false\n"
        f"DrawTournamentInfo false\nDrawBotNames false\n", encoding="utf-8")
    return "bwapi-data/TournamentModule.dll"


def starcraft_pids() -> set[int]:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq StarCraft.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return {int(line.split('","')[1]) for line in out.splitlines() if line.startswith('"StarCraft.exe"')}


def launch(inst: Path, env: Optional[dict] = None, timeout: float = 20.0) -> set[int]:
    """Start StarCraft + BWAPI in `inst` via injectory; returns the new StarCraft pid(s)."""
    before = starcraft_pids()
    subprocess.Popen([str(inst / "injectory_x86.exe"), "--launch", "StarCraft.exe", "--inject",
                      "bwapi-data\\BWAPI.dll", "wmode.dll"], cwd=inst, env=env)
    end = time.time() + timeout
    while time.time() < end:
        new = starcraft_pids() - before
        if new:
            return new
        time.sleep(0.5)
    raise RuntimeError(f"StarCraft did not start in {inst}")


def kill_pids(pids: set[int]) -> None:
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def win_python() -> str:
    p = ROOT / "python" / ".venv" / "Scripts" / "python.exe"
    return str(p) if p.exists() else sys.executable


# ---------------------------------------------------------------------------- match
LAUNCH_LOCK = threading.Lock()      # InstallPath is one registry value: launches must not overlap


def play(index: int, me: Player, opp_name: str, map_path: str, run_id: str, run_dir: Path,
         args: argparse.Namespace, slot: int = 0) -> dict:
    bot_dir, meta = load_bot(opp_name)
    gdir = run_dir / "games" / f"{index:04d}"
    gdir.mkdir(parents=True, exist_ok=True)
    base = Path(args.instances) / f"s{slot}"
    inst_a, inst_b = base / "inst_a", base / "inst_b"
    host_name = f"{me.name[:14]}-{run_id[-3:]}{slot}"      # unique game name: the joiner looks it up
    prepare_instance(inst_a, GAME / "bwapi-data" / "BWAPI.dll", [SHIM_DLL])
    ai_files = [p for p in (bot_dir / "AI").iterdir()]
    prepare_instance(inst_b, bot_dir / "BWAPI.dll", ai_files)
    client = meta["botType"] != "AI_MODULE"
    game_id = f"{run_id}-g{index:04d}"
    replay = f"bwapi-data/replays/{game_id}.rep" if args.replays else ""
    tm_a = install_tournament_module(inst_a, args.tm_frame_skip) if args.tm else ""
    tm_b = install_tournament_module(inst_b, args.tm_frame_skip) if args.tm else ""
    (inst_a / "bwapi-data" / "bwapi.ini").write_text(
        bwapi_ini(f"bwapi-data/AI/{SHIM_DLL.name}", me.race or "Terran", host_name, map_path, True, replay,
                  20 + 40 * slot, tm_a, args.lan_mode), encoding="utf-8")
    (inst_b / "bwapi-data" / "bwapi.ini").write_text(
        bwapi_ini("" if client else f"bwapi-data/AI/{meta['file']}", meta.get("race", "Random"), meta["name"],
                  map_path, False, "", 680 + 40 * slot, tm_b, args.lan_mode, join=host_name), encoding="utf-8")
    _no_tips()
    port = args.port + slot
    env_a = dict(os.environ, BWBOT_PORT=str(port), BWBOT_HOST="127.0.0.1", BWBOT_NO_SPAWN="1")
    logs = []
    t0 = time.time()
    procs = []
    pids: set[int] = set()
    LAUNCH_LOCK.acquire()
    holding = True
    try:
        install_path(str(inst_a) + "\\")
        pids |= launch(inst_a, env_a)
        benv = dict(os.environ, BWBOT_RESULT=str(gdir / "result_a.json"), BWBOT_LOG_DIR=str(gdir / "logs"),
                    BWBOT_GAME_ID=game_id, BWBOT_SIDE="a", PYTHONPATH=str(ROOT / "python"))
        if me.profile:
            benv["BWBOT_PROFILE"] = me.profile
        benv.update({k: str(v) for k, v in me.env.items()})
        fh = open(gdir / "brain_a.log", "wb")
        logs.append(fh)
        brain = subprocess.Popen([win_python(), "-m", "harness.brain", me.spec, "--port", str(port), "--games", "1",
                                  "--max-frames", str(args.max_frames), "--connect-timeout", "180", "--no-apm-hud",
                                  *me.args, *args.brain_args], cwd=ROOT / "python", env=benv, stdout=fh,
                                 stderr=subprocess.STDOUT)
        procs.append(brain)
        time.sleep(args.join_delay)
        install_path(str(inst_b) + "\\")
        pids |= launch(inst_b)
        time.sleep(3.0)                # BWAPI reads bwapi.ini (via InstallPath) while injecting
        LAUNCH_LOCK.release()
        holding = False
        if client:
            procs.append(_start_client(inst_b, meta, gdir))
        deadline = t0 + args.timeout_min * 60
        while time.time() < deadline and brain.poll() is None:
            time.sleep(1.0)
        timed_out = brain.poll() is None
    finally:
        if holding:
            LAUNCH_LOCK.release()
        for p in procs:
            if p.poll() is None:
                p.kill()
        kill_pids(pids)
        time.sleep(1.0)
        for fh in logs:
            fh.close()
        if replay and (inst_a / replay).is_file():
            shutil.move(str(inst_a / replay), str(gdir / "replay.rep"))
    return game_row(index, game_id, map_path, me, meta, gdir, args.max_frames, timed_out, t0, "botmatch")


def game_row(index: int, game_id: str, map_path: str, me: Player, meta: dict, gdir: Path, max_frames: int,
             timed_out: bool, t0: float, harness: str) -> dict:
    """The results.jsonl row for one game, judged from our brain's result file (the opponent has none)."""
    a = {"name": me.name, "spec": me.spec, "profile": me.profile, "race": me.race}
    ra = None
    if (gdir / "result_a.json").is_file():
        ra = json.loads((gdir / "result_a.json").read_text(encoding="utf-8"))
        a["result"] = ra
        if ra.get("log") and Path(ra["log"]).is_file():
            a.update(log_summary(Path(ra["log"])))
    b = {"name": meta["name"], "spec": f"sscait:{meta['name']}", "profile": None, "race": meta.get("race"),
         "botType": meta["botType"]}
    rb = None if ra is None else {"won": not ra["won"], "frame": ra["frame"], "total": ra.get("enemy_total", 0)}
    if ra is not None and ra["frame"] >= max_frames - 48 and not ra.get("enemy_total"):
        rb["total"] = ra.get("total", 0)
    winner, reason = judge(ra, rb, max_frames)
    if timed_out and reason != "elimination":
        reason = "timeout"
    return {"game": index, "game_id": game_id, "map": map_path, "a": a, "b": b, "winner": winner,
            "winner_name": {"a": a["name"], "b": b["name"]}.get(winner), "reason": reason,
            "frames": (ra or {}).get("frame", 0), "wall_s": round(time.time() - t0, 1), "dir": str(gdir),
            "harness": harness}


def _start_client(inst: Path, meta: dict, gdir: Path) -> subprocess.Popen:
    f = inst / "bwapi-data" / "AI" / meta["file"]
    cmd = ["java", "-jar", str(f)] if f.suffix == ".jar" else [str(f)]
    return subprocess.Popen(cmd, cwd=inst, stdout=open(gdir / "opponent.log", "wb"), stderr=subprocess.STDOUT)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list enabled SSCAIT bots")
    ap.add_argument("--fetch", action="append", default=[], help="download a bot from SSCAIT into bots/")
    ap.add_argument("--p1", default="adjutant", help="our player (module[:Class][@profile])")
    ap.add_argument("--opponent", action="append", default=[], help="SSCAIT bot name (repeatable; rotated)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1,
                    help="matches at once (each Local PC game runs at ~64 frames/s, so this is the speed-up)")
    ap.add_argument("--maps", nargs="*")
    ap.add_argument("--max-frames", type=int, default=24 * 60 * 25)
    ap.add_argument("--timeout-min", type=float, default=40)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--join-delay", type=float, default=8.0, help="seconds between host and joiner start")
    ap.add_argument("--run-id", default=time.strftime("bm%Y%m%d-%H%M%S"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--instances", default=str(ROOT / "runs" / "botmatch"))
    ap.add_argument("--no-replays", dest="replays", action="store_false")
    ap.add_argument("--brain-args", nargs="*", default=[], help="extra bwbot.run options for our brain")
    ap.add_argument("--no-tm", dest="tm", action="store_false",
                    help="don't load the Tournament Manager module (speed 0 + frame skip in both clients)")
    ap.add_argument("--tm-frame-skip", type=int, default=256, help="render every N frames (tournament module)")
    ap.add_argument("--lan-mode", default="Local PC", help='multiplayer network: "Local PC", '
                    '"Local Area Network (UDP)", "Direct IP"')
    args = ap.parse_args(argv)

    if args.list:
        for b in sorted(sscait_bots(), key=lambda b: b["name"].lower()):
            if b.get("status") == "Enabled":
                print(f"{b['name']:<28} {b.get('race', ''):<8} {b.get('botType', '')}")
        return 0
    bots = sscait_bots() if args.fetch else None
    for name in args.fetch:
        fetch(name, bots)
    if not args.opponent:
        return 0
    if os.name != "nt":
        raise SystemExit("botmatch drives Windows StarCraft; run it from Windows (self-play runs in WSL)")
    for p in (GAME / "StarCraft.exe", GAME / "injectory_x86.exe", SHIM_DLL):
        if not p.exists():
            raise SystemExit(f"missing {p} (scripts/setup_windows.ps1, scripts/build_shim.ps1)")
    if starcraft_pids():
        raise SystemExit("StarCraft is already running; close it first (the Local PC network would pick it up)")
    ensure_bwta_dlls()
    me = Player.parse(args.p1)
    maps = args.maps or DEFAULT_MAPS
    run_dir = Path(args.out) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for opp in dict.fromkeys(args.opponent):
        load_bot(opp)
    rows = []
    out_lock = threading.Lock()
    free = queue.Queue()
    for s in range(max(1, args.parallel)):
        free.put(s)
    prev_install = read_install_path()

    def one(i: int) -> None:
        slot = free.get()
        try:
            opp = args.opponent[i % len(args.opponent)]
            row = play(i, me, opp, maps[i % len(maps)], args.run_id, run_dir, args, slot)
        finally:
            free.put(slot)
        with out_lock:
            rows.append(row)
            with (run_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[botmatch] g{i:04d} {me.name} vs {opp} on {Path(row['map']).stem}: {row['winner_name'] or '-'} "
                  f"({row['reason']}, {row['frames']} frames, {row['wall_s']:.0f}s)", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            list(pool.map(one, range(args.games)))
    finally:
        install_path(prev_install)
    rows.sort(key=lambda r: r["game"])
    from adjutant.learn.report import print_report
    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
