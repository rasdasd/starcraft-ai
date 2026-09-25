"""Self-generated builds: mutate existing builds into new JSON builds, score builds from match
results, and prune the generated ones that lose.

    python -m adjutant.learn.builds mutate mech_expand --n 4          # -> python/models/builds/
    python -m adjutant.learn.builds stats runs/wm1 runs/wm2            # win rate per build
    python -m adjutant.learn.builds prune runs/wm1 runs/wm2 --below 0.25 --min-games 6

The loop: mutate, play them (the `explore` profile tries every build of our race; winematch or
botmatch against the pool), `stats`/`prune`, retrain the strategy model (`adjutant.learn.train
strategy`), which scores builds from their descriptors, so generated builds are picked like any other.
Only builds with a `parent` (generated ones) are ever deleted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from ..strategies import TEMPLATES, BuildSpec, SpecError, reload

OUT = Path(__file__).resolve().parents[2] / "models" / "builds"
NOT_PRODUCTION = ("Supply_Depot", "Pylon", "Overlord", "Refinery", "Assimilator", "Extractor",
                  "Command_Center", "Nexus", "Hatchery")


# ---------------------------------------------------------------------------- mutations
def _op_timing(d: dict, rng: random.Random, scale: float) -> bool:
    steps = d.get("opening") or []
    if not steps:
        return False
    i = rng.randrange(len(steps))
    lo = steps[i - 1][0] if i else 4
    hi = steps[i + 1][0] if i + 1 < len(steps) else 200
    new = max(lo, min(hi, steps[i][0] + rng.choice((-2, -1, 1, 2))))
    if new == steps[i][0]:
        return False
    steps[i] = [new, steps[i][1]]
    return True


def _op_swap(d: dict, rng: random.Random, scale: float) -> bool:
    steps = d.get("opening") or []
    if len(steps) < 2:
        return False
    i = rng.randrange(len(steps) - 1)
    if steps[i][1] == steps[i + 1][1]:
        return False
    steps[i], steps[i + 1] = [steps[i][0], steps[i + 1][1]], [steps[i + 1][0], steps[i][1]]
    return True


def _op_drop(d: dict, rng: random.Random, scale: float) -> bool:
    steps = d.get("opening") or []
    if len(steps) < 3:
        return False
    steps.pop(rng.randrange(1, len(steps)))
    return True


def _op_extra_producer(d: dict, rng: random.Random, scale: float) -> bool:
    """One more copy of a production building in the opening, a little later than the first."""
    steps = d.get("opening") or []
    prod = [s for s in steps if s[1].split("_", 1)[-1] not in NOT_PRODUCTION and s[1] not in NOT_PRODUCTION]
    if not prod:
        return False
    src = rng.choice(prod)
    steps.append([src[0] + rng.choice((1, 2, 3, 4)), src[1]])
    steps.sort(key=lambda s: s[0])
    return True


def _op_posture(d: dict, rng: random.Random, scale: float) -> bool:
    a = int(d.get("attack_supply", 40))
    r = int(d.get("retreat_supply", 12))
    a2 = max(6, round(a * rng.uniform(1 - scale, 1 + scale)))
    r2 = max(2, min(a2 - 4, round(r * rng.uniform(1 - scale, 1 + scale))))
    if (a2, r2) == (a, r):
        return False
    d["attack_supply"], d["retreat_supply"] = a2, r2
    return True


def _op_goal(d: dict, rng: random.Random, scale: float) -> bool:
    """Scale one plain-number unit or building count of the goal."""
    goal = d.get("goal", {})
    slots = [(k, n) for k in ("units", "buildings") for n, v in goal.get(k, {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
    if not slots:
        return False
    k, n = rng.choice(slots)
    v = goal[k][n]
    new = max(1, round(v * rng.uniform(1 - scale, 1 + scale) + rng.choice((-1, 0, 1))))
    if new == v:
        return False
    goal[k][n] = new
    return True


OPS = {"timing": _op_timing, "swap": _op_swap, "drop": _op_drop, "extra_producer": _op_extra_producer,
       "posture": _op_posture, "goal": _op_goal}


def _digest(d: dict) -> str:
    body = {k: v for k, v in d.items() if k not in ("name", "description", "parent", "source")}
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:6]


def mutate(doc: dict, rng: random.Random, scale: float = 0.25, max_ops: int = 3,
           tries: int = 20) -> Optional[dict]:
    """A valid variant of `doc` with 1..max_ops mutations, or None."""
    base = str(doc["name"]).split("~")[0]
    for _ in range(tries):
        d = copy.deepcopy(doc)
        d.pop("source", None)
        d.pop("default_vs", None)        # rule defaults stay with hand-written builds
        names = rng.sample(sorted(OPS), rng.randint(1, max_ops))
        applied = [n for n in names if OPS[n](d, rng, scale)]
        if not applied:
            continue
        d["parent"] = doc["name"]
        d["name"] = f"{base}~{_digest(d)}"
        d["description"] = f"{doc['name']} + {', '.join(applied)}"
        try:
            BuildSpec(d)
        except SpecError:
            continue
        return d
    return None


# ---------------------------------------------------------------------------- results
def _rows(runs: Iterable[Path]) -> Iterable[dict]:
    for run in runs:
        p = Path(run) / "results.jsonl" if Path(run).is_dir() else Path(run)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def _local(path: str) -> Path:
    p = Path(path)
    if not p.exists() and path.startswith("/mnt/") and len(path) > 6:
        p = Path(f"{path[5].upper()}:/{path[7:]}")
    return p


def game_build(log: Path) -> Optional[str]:
    """The build a game's strategy followed most (report snapshots' `tmpl`)."""
    c: Counter = Counter()
    try:
        for line in log.open(encoding="utf-8"):
            if '"snap"' not in line:
                continue
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("tmpl"):
                c[x["tmpl"]] += 1
    except OSError:
        return None
    return c.most_common(1)[0][0] if c else None


def stats(runs: Iterable[Path]) -> dict[str, dict]:
    """build -> {games, wins, rate (Beta(1,1) posterior mean), opponents}."""
    out: dict[str, dict] = {}
    for r in _rows(runs):
        log = (r.get("a") or {}).get("result", {}).get("log")
        b = game_build(_local(log)) if log else None
        if b is None:
            continue
        s = out.setdefault(b, {"games": 0, "wins": 0, "opponents": Counter()})
        s["games"] += 1
        s["wins"] += int(r.get("winner") == "a")
        s["opponents"][(r.get("b") or {}).get("name", "?")] += 1
    for s in out.values():
        s["rate"] = (s["wins"] + 1) / (s["games"] + 2)
    return out


def generated(dirs: Iterable[Path]) -> dict[str, Path]:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")) if Path(d).is_dir() else ():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("parent"):
                out[doc["name"]] = p
    return out


# ---------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="adjutant.learn.builds", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mutate", help="write N variants of a build")
    m.add_argument("parent", nargs="+")
    m.add_argument("--n", type=int, default=4)
    m.add_argument("--scale", type=float, default=0.25)
    m.add_argument("--max-ops", type=int, default=3)
    m.add_argument("--seed", type=int)
    m.add_argument("--out", type=Path, default=OUT)
    s = sub.add_parser("stats", help="win rate per build over runs")
    s.add_argument("runs", nargs="+", type=Path)
    p = sub.add_parser("prune", help="delete generated builds that lose")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--dir", type=Path, default=OUT)
    p.add_argument("--min-games", type=int, default=6)
    p.add_argument("--below", type=float, default=0.25, help="posterior win rate to drop at")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "mutate":
        rng = random.Random(a.seed)
        a.out.mkdir(parents=True, exist_ok=True)
        for name in a.parent:
            t = TEMPLATES.get(name)
            if not isinstance(t, BuildSpec):
                raise SystemExit(f"unknown JSON build {name!r} (have {sorted(TEMPLATES)})")
            made = 0
            for _ in range(a.n * 5):
                if made >= a.n:
                    break
                d = mutate(t.to_doc(), rng, a.scale, a.max_ops)
                if d is None or d["name"] in TEMPLATES:
                    continue
                path = a.out / f"{d['name'].replace('~', '__')}.json"
                path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
                TEMPLATES[d["name"]] = BuildSpec(d, source=str(path))
                print(f"{d['name']}: {d['description']} -> {path}")
                made += 1
    elif a.cmd == "stats":
        st = stats(a.runs)
        for b, s in sorted(st.items(), key=lambda kv: -kv[1]["rate"]):
            opp = ", ".join(f"{k} {v}" for k, v in s["opponents"].most_common(4))
            print(f"{b:<28} {s['wins']:>3}/{s['games']:<3} rate {s['rate']:.2f}   {opp}")
    elif a.cmd == "prune":
        st = stats(a.runs)
        for name, path in generated([a.dir]).items():
            s = st.get(name)
            if s is None or s["games"] < a.min_games or s["rate"] >= a.below:
                continue
            print(f"{'would drop' if a.dry_run else 'drop'} {name}: {s['wins']}/{s['games']} ({path})")
            if not a.dry_run:
                path.unlink()
        reload()


if __name__ == "__main__":
    main()
