"""Parallel headless OpenBW 1v1 games between Python bots (run inside WSL/Linux).

    python -m harness.selfplay --p1 adjutant@explore --p2 goliath --p2 adjutant:Parity --games 40 --parallel 4
    python -m harness.selfplay --pool pool.json --games 100 --parallel 6 --max-frames 28800

Each game runs two BWAPILauncher processes that meet in an OpenBW LAN game over a private unix
socket (`OPENBW_LAN_MODE=LOCAL`), each hosting the shim module on its own TCP port, plus one brain
per side (`python -m harness.brain`). Both brains share `BWBOT_GAME_ID`, so their recorder logs can
be joined later (e.g. to label the opponent's strategy).

A game ends by elimination, or at `--max-frames` when both brains leave and the higher in-game
score wins. A side whose brain dies loses. Everything lands in `runs/<run-id>/`:
`results.jsonl` (one row per game, with profile names and slot/model descriptions), per-game
launcher/brain output, recorder logs, and replays.

Player syntax: `module[:Class][@profile][/Race]`, e.g. `adjutant@explore`, `sparring.zerg/Zerg`.
A pool file is a JSON list of {"name", "spec", "profile", "race", "env", "args", "weight"}.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
WSL = ROOT / "wsl"
GAME_DIR = WSL / "game"
LAUNCHER = WSL / "bwapi" / "build" / "bin" / "BWAPILauncher"
LIBDIR = WSL / "bwapi" / "build" / "lib"
SHIM_SO = ROOT / "shim" / "build-openbw" / "shim_module.so"

DEFAULT_MAPS = [
    "maps/BroodWar/aiide/(2)Benzene.scx",
    "maps/BroodWar/aiide/(2)Destination.scx",
    "maps/BroodWar/aiide/(2)Heartbreak Ridge.scx",
    "maps/BroodWar/aiide/(3)Aztec.scx",
    "maps/BroodWar/aiide/(3)Tau Cross.scx",
    "maps/BroodWar/aiide/(4)Andromeda.scx",
    "maps/BroodWar/aiide/(4)Circuit Breaker.scx",
    "maps/BroodWar/aiide/(4)Fortress.scx",
    "maps/BroodWar/aiide/(4)Python.scx",
]


@dataclass
class Player:
    spec: str
    profile: Optional[str] = None
    race: Optional[str] = None
    name: str = ""
    env: dict = field(default_factory=dict)
    args: list = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.spec + (f"@{self.profile}" if self.profile else "")
        if not self.race:
            self.race = bot_race(self.spec)

    @classmethod
    def parse(cls, text: str) -> "Player":
        race = None
        if "/" in text:
            text, race = text.rsplit("/", 1)
        profile = None
        if "@" in text:
            text, profile = text.split("@", 1)
        return cls(spec=text, profile=profile, race=race)


def bot_race(spec: str) -> str:
    """The bot class's `race` attribute (sparring bots declare one); Terran otherwise."""
    import importlib
    mod_name, _, cls_name = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return "Terran"
    cls = getattr(mod, cls_name or "BOT", None)
    return str(getattr(cls, "race", "Terran") or "Terran")


def load_pool(path: Path) -> list[Player]:
    return [Player(**p) for p in json.loads(Path(path).read_text(encoding="utf-8"))]


def brain_python() -> str:
    env = os.environ.get("BWBOT_BRAIN_PYTHON")
    if env:
        return env
    marker = WSL / "brain_venv"
    if marker.is_file():
        p = Path(marker.read_text().strip()) / "bin" / "python"
        if p.exists():
            return str(p)
    return sys.executable


@dataclass
class GameSpec:
    index: int
    map: str
    a: Player
    b: Player


class Match:
    """One LAN game: two launchers + two brains, supervised until both brains exit or timeout."""

    def __init__(self, g: GameSpec, run_id: str, run_dir: Path, slot: int, args: argparse.Namespace) -> None:
        self.g, self.run_id, self.slot, self.args = g, run_id, slot, args
        self.dir = run_dir / "games" / f"{g.index:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.game_id = f"{run_id}-g{g.index:04d}"
        self.ports = (args.base_port + 2 * slot, args.base_port + 2 * slot + 1)
        self.sock = f"/tmp/bwbot-{run_id}-{slot}.sock"
        self.procs: list[subprocess.Popen] = []
        self._files: list = []

    def _launcher_env(self, side: str, p: Player, port: int) -> dict:
        env = dict(os.environ)
        env.update({
            "LD_LIBRARY_PATH": f"{LIBDIR}:{env.get('LD_LIBRARY_PATH', '')}",
            "BWAPI_CONFIG_AI__AI": str(GAME_DIR / "bwapi-data" / "AI" / "shim_module.so"),
            "BWAPI_CONFIG_AUTO_MENU__AUTO_MENU": "LAN",
            "BWAPI_CONFIG_AUTO_MENU__MAP": self.g.map,
            "BWAPI_CONFIG_AUTO_MENU__RACE": p.race,
            "BWAPI_CONFIG_AUTO_MENU__CHARACTER_NAME": f"{side}_{p.name}"[:24],
            "BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART": "OFF",
            "BWAPI_CONFIG_AUTO_MENU__WAIT_FOR_MIN_PLAYERS": "2",
            "BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY": str(self.dir / f"replay_{side}.rep") if self.args.replays else "",
            "OPENBW_LAN_MODE": "LOCAL",
            "OPENBW_LOCAL_PATH": self.sock,
            "OPENBW_ENABLE_UI": "0",
            "BWBOT_HOST": "127.0.0.1",
            "BWBOT_PORT": str(port),
            "BWBOT_NO_SPAWN": "1",
        })
        return env

    def _brain_env(self, side: str, p: Player) -> dict:
        env = dict(os.environ)
        env.update({
            "BWBOT_RESULT": str(self.dir / f"result_{side}.json"),
            "BWBOT_LOG_DIR": str(self.dir / "logs"),
            "BWBOT_GAME_ID": self.game_id,
            "BWBOT_SIDE": side,
            "PYTHONPATH": f"{ROOT / 'python'}:{env.get('PYTHONPATH', '')}",
        })
        if p.profile:
            env["BWBOT_PROFILE"] = p.profile
        env.update({k: str(v) for k, v in p.env.items()})
        return env

    def _spawn(self, cmd: list[str], env: dict, log_name: str, cwd: Path) -> subprocess.Popen:
        fh = open(self.dir / log_name, "wb")
        self._files.append(fh)
        p = subprocess.Popen(cmd, env=env, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append(p)
        return p

    def run(self) -> dict:
        g, a = self.g, self.args
        t0 = time.time()
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        py = brain_python()
        brains = []
        try:
            for side, p, port in (("a", g.a, self.ports[0]), ("b", g.b, self.ports[1])):
                self._spawn([str(LAUNCHER)], self._launcher_env(side, p, port), f"launcher_{side}.log", GAME_DIR)
                time.sleep(0.5)
            for side, p, port in (("a", g.a, self.ports[0]), ("b", g.b, self.ports[1])):
                cmd = [py, "-m", "harness.brain", p.spec, "--port", str(port), "--games", "1", "--no-gui",
                       "--max-frames", str(a.max_frames), "--connect-timeout", "120", "--no-apm-hud", *p.args]
                brains.append(self._spawn(cmd, self._brain_env(side, p), f"brain_{side}.log", ROOT / "python"))
            deadline = t0 + a.timeout_min * 60
            while time.time() < deadline and any(b.poll() is None for b in brains):
                time.sleep(1.0)
            timed_out = any(b.poll() is None for b in brains)
        finally:
            self._stop()
        return self._result(time.time() - t0, timed_out)

    def _stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        end = time.time() + 5
        for p in self.procs:
            try:
                p.wait(timeout=max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        for fh in self._files:
            fh.close()
        if os.path.exists(self.sock):
            try:
                os.unlink(self.sock)
            except OSError:
                pass

    def _side(self, side: str, p: Player) -> dict:
        out = {"name": p.name, "spec": p.spec, "profile": p.profile, "race": p.race}
        rf = self.dir / f"result_{side}.json"
        if rf.is_file():
            out["result"] = json.loads(rf.read_text(encoding="utf-8"))
            log = out["result"].get("log")
            if log and Path(log).is_file():
                out.update(log_summary(Path(log)))
        return out

    def _result(self, wall: float, timed_out: bool) -> dict:
        sa, sb = self._side("a", self.g.a), self._side("b", self.g.b)
        ra, rb = sa.get("result"), sb.get("result")
        winner, reason = judge(ra, rb, self.args.max_frames)
        if timed_out and reason != "elimination":
            reason = "timeout"
        frames = max((r or {}).get("frame", 0) for r in (ra, rb))
        return {"game": self.g.index, "game_id": self.game_id, "map": self.g.map, "a": sa, "b": sb,
                "winner": winner, "winner_name": {"a": sa["name"], "b": sb["name"]}.get(winner),
                "reason": reason, "frames": frames, "wall_s": round(wall, 1), "dir": str(self.dir)}


def log_summary(path: Path) -> dict:
    """Slot descriptions and strategy templates (first + all, in order) from a recorder log."""
    from blackboard.recorder import read_game
    g = read_game(path)
    templates = [r["to"] for r in g["rows"] if r.get("slot") == "strategy" and r.get("k") == "switch"]
    return {"slots": g["header"].get("slots"), "log": str(path),
            "template": templates[0] if templates else None, "templates": templates}


def judge(ra: Optional[dict], rb: Optional[dict], max_frames: int) -> tuple[Optional[str], str]:
    """(winner side or None, reason)."""
    if ra is None and rb is None:
        return None, "crash"
    if ra is None:
        return "b", "crash_a"
    if rb is None:
        return "a", "crash_b"
    frame = max(ra["frame"], rb["frame"])
    if frame < max_frames - 48 and ra["won"] != rb["won"]:
        return ("a" if ra["won"] else "b"), "elimination"
    ta, tb = ra.get("total", 0), rb.get("total", 0)
    if ta == tb:
        return None, "draw"
    return ("a" if ta > tb else "b"), "score"


def schedule(args: argparse.Namespace, rng: random.Random) -> list[GameSpec]:
    pool = load_pool(Path(args.pool)) if args.pool else []
    opponents = [Player.parse(s) for s in (args.p2 or [])] + pool
    learners = [Player.parse(s) for s in (args.p1 or [])]
    if not learners and not opponents:
        raise SystemExit("need --p1/--p2 or --pool")
    maps = args.maps or DEFAULT_MAPS
    games = []
    for i in range(args.games):
        if learners:
            a = learners[i % len(learners)]
            b = rng.choices(opponents or learners, weights=[p.weight for p in (opponents or learners)])[0]
        else:
            a, b = rng.sample(opponents, 2) if len(opponents) > 1 else (opponents[0], opponents[0])
        if args.swap and rng.random() < 0.5:
            a, b = b, a
        games.append(GameSpec(i, maps[i % len(maps)], a, b))
    return games


def check_env() -> None:
    missing = [str(p) for p in (LAUNCHER, GAME_DIR / "bwapi-data" / "bwapi.ini") if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run scripts/setup_wsl.sh first")
    if os.name == "nt":
        raise SystemExit("run the self-play harness inside WSL/Linux (OpenBW LAN mode needs unix sockets)")
    dst = GAME_DIR / "bwapi-data" / "AI" / "shim_module.so"
    if SHIM_SO.exists() and (not dst.exists() or SHIM_SO.stat().st_mtime > dst.stat().st_mtime):
        dst.write_bytes(SHIM_SO.read_bytes())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", action="append", help="player under test (repeatable; games alternate)")
    ap.add_argument("--p2", action="append", help="opponent (repeatable; sampled per game)")
    ap.add_argument("--pool", help="JSON opponent pool")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--maps", nargs="*", help="map paths relative to wsl/game (default: AIIDE pool)")
    ap.add_argument("--max-frames", type=int, default=24 * 60 * 20, help="frame limit (default 20 game minutes)")
    ap.add_argument("--timeout-min", type=float, default=30, help="wall-clock limit per game")
    ap.add_argument("--base-port", type=int, default=9100)
    ap.add_argument("--run-id", default=time.strftime("sp%Y%m%d-%H%M%S"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--swap", action="store_true", help="randomize which player hosts (side a)")
    ap.add_argument("--no-replays", dest="replays", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    check_env()
    rng = random.Random(args.seed)
    games = schedule(args, rng)
    run_dir = Path(args.out) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**vars(args), "games_spec": [{"map": g.map, "a": asdict(g.a), "b": asdict(g.b)} for g in games]},
        indent=1, default=str), encoding="utf-8")
    results = run_dir / "results.jsonl"
    lock = threading.Lock()
    slots = list(range(args.parallel))
    slot_lock = threading.Lock()

    def play(g: GameSpec) -> dict:
        with slot_lock:
            slot = slots.pop()
        try:
            return Match(g, args.run_id, run_dir, slot, args).run()
        finally:
            with slot_lock:
                slots.append(slot)

    print(f"[selfplay] {len(games)} games, {args.parallel} parallel -> {run_dir}", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(play, g) for g in games]
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            with lock, results.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[selfplay] g{row['game']:04d} {row['a']['name']} vs {row['b']['name']} on "
                  f"{Path(row['map']).stem}: {row['winner_name'] or '-'} ({row['reason']}, {row['frames']} frames, "
                  f"{row['wall_s']:.0f}s)", flush=True)
    from adjutant.learn.report import print_report
    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
