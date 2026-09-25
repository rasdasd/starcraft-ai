"""Belief learning: fog-limited features, truth labels, the self-play label join, and training.

Rows: `belief/sample` (written by the `BeliefLog` REPORT component in games with complete map
information) carry `x` = belief_features(bb) (only what a normal game shows) and truth targets:
`yc` log1p enemy counts per UNITS type, `yt` enemy tech present per TECH type, `yb` enemy bases.
The opponent's template comes from joining the other side's log of the same self-play game
(same BWBOT_GAME_ID, other side): `opponent_template(frame)`.

    python -m adjutant.learn.train belief --logs runs/truth1 --out models/
      -> belief_counts.npz (regression), belief_tech.npz (multi-label), belief_bases.npz,
         belief_opponent.npz (softmax over opponent templates, self-play games only)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from blackboard import Blackboard
from blackboard.models import FeatureSpec
from blackboard.recorder import iter_games
from bwbot import Race, UnitType as U
from mybot.opponent import OPENING_NAMES

BELIEF_VERSION = 1

UNITS = tuple(int(t) for t in (
    U.Terran_SCV, U.Terran_Marine, U.Terran_Firebat, U.Terran_Medic, U.Terran_Vulture,
    U.Terran_Siege_Tank_Tank_Mode, U.Terran_Siege_Tank_Siege_Mode, U.Terran_Goliath, U.Terran_Wraith,
    U.Terran_Dropship, U.Terran_Science_Vessel, U.Terran_Battlecruiser, U.Terran_Valkyrie,
    U.Zerg_Drone, U.Zerg_Overlord, U.Zerg_Zergling, U.Zerg_Hydralisk, U.Zerg_Lurker, U.Zerg_Mutalisk,
    U.Zerg_Scourge, U.Zerg_Ultralisk, U.Zerg_Defiler, U.Zerg_Guardian,
    U.Protoss_Probe, U.Protoss_Zealot, U.Protoss_Dragoon, U.Protoss_High_Templar, U.Protoss_Dark_Templar,
    U.Protoss_Archon, U.Protoss_Reaver, U.Protoss_Shuttle, U.Protoss_Observer, U.Protoss_Corsair,
    U.Protoss_Scout, U.Protoss_Carrier, U.Protoss_Arbiter,
))
TECH = tuple(int(t) for t in (
    U.Terran_Command_Center, U.Terran_Barracks, U.Terran_Factory, U.Terran_Starport, U.Terran_Academy,
    U.Terran_Armory, U.Terran_Engineering_Bay, U.Terran_Science_Facility, U.Terran_Machine_Shop,
    U.Terran_Control_Tower, U.Terran_Bunker, U.Terran_Missile_Turret,
    U.Zerg_Hatchery, U.Zerg_Lair, U.Zerg_Hive, U.Zerg_Spawning_Pool, U.Zerg_Hydralisk_Den, U.Zerg_Spire,
    U.Zerg_Queens_Nest, U.Zerg_Evolution_Chamber, U.Zerg_Defiler_Mound, U.Zerg_Ultralisk_Cavern,
    U.Zerg_Sunken_Colony, U.Zerg_Spore_Colony,
    U.Protoss_Nexus, U.Protoss_Gateway, U.Protoss_Cybernetics_Core, U.Protoss_Forge, U.Protoss_Photon_Cannon,
    U.Protoss_Citadel_of_Adun, U.Protoss_Templar_Archives, U.Protoss_Robotics_Facility, U.Protoss_Observatory,
    U.Protoss_Stargate, U.Protoss_Fleet_Beacon, U.Protoss_Arbiter_Tribunal, U.Protoss_Robotics_Support_Bay,
))
RACES = (int(Race.Terran), int(Race.Zerg), int(Race.Protoss))

_names = (["minute"] + [f"race_{r}" for r in RACES]
          + [f"n_{t}" for t in UNITS] + [f"dead_{t}" for t in UNITS]
          + [f"seen_{t}" for t in TECH] + [f"inf_{t}" for t in TECH]
          + ["bases", "stale", "army_age", "own_army", "own_workers"]
          + [f"p_{n}" for n in OPENING_NAMES])
BELIEF_SPEC = FeatureSpec("belief", BELIEF_VERSION, tuple(_names))
COUNT_LABELS = tuple(str(t) for t in UNITS)
TECH_LABELS = tuple(str(t) for t in TECH)


def belief_features(bb: Blackboard) -> np.ndarray:
    b, w = bb.belief, bb.world
    frame = w.frame
    race = [float(b.enemy_race == r) for r in RACES]
    counts = [np.log1p(b.counts.get(t, 0.0)) for t in UNITS]
    dead = [np.log1p(b.dead.get(t, 0)) for t in UNITS]
    inferred = b.inferred
    seen = [float(t in b.tech and t not in inferred) for t in TECH]
    inf = [float(t in inferred) for t in TECH]
    stale = np.mean(list(b.staleness.values())) / (24 * 300) if b.staleness else 1.0
    age = (frame - b.army_seen_frame) / (24 * 120) if b.army_seen_frame >= 0 else 1.0
    probs = [float(b.opening_probs.get(n, 0.0)) for n in OPENING_NAMES]
    x = ([frame / (24 * 60) / 20] + race + counts + dead + seen + inf
         + [sum(1 for e in b.bases if e.alive) / 4, min(stale, 2.0), min(age, 2.0), w.army_supply / 100,
            len(w.workers) / 60] + probs)
    return np.asarray(x, np.float32)


def truth_targets(bb: Blackboard) -> tuple[list[float], list[int], int]:
    t = bb.truth
    yc = [round(float(np.log1p(t.counts.get(u, 0))), 4) for u in UNITS]
    yt = [int(u in t.buildings) for u in TECH]
    return yc, yt, len(t.bases)


# ---------------------------------------------------------------------------- label join
def _template_timeline(game: dict) -> list[tuple[int, str]]:
    out = []
    for r in game["rows"]:
        if r.get("slot") == "strategy" and r.get("k") == "switch" and r.get("to"):
            out.append((int(r["f"]), str(r["to"])))
    return out


def join_opponents(games: list[dict]) -> dict[str, list[tuple[int, str]]]:
    """path -> the other side's template timeline, for self-play games where both sides logged."""
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for g in games:
        gid = g["header"].get("game_id")
        if gid:
            by_id[gid][g["header"].get("side", "")] = g
    out = {}
    for sides in by_id.values():
        if len(sides) != 2:
            continue
        (sa, ga), (sb, gb) = sides.items()
        out[ga["path"]] = _template_timeline(gb)
        out[gb["path"]] = _template_timeline(ga)
    return out


def template_at(timeline: list[tuple[int, str]], frame: int) -> Optional[str]:
    cur = None
    for f, t in timeline:
        if f <= frame:
            cur = t
    return cur


def belief_dataset(dirs: Sequence[Path]):
    games = list(iter_games(*dirs))
    opp = join_opponents(games)
    X, YC, YT, YB, groups, OPP = [], [], [], [], [], []
    skipped = defaultdict(int)
    for g in games:
        rows = [r for r in g["rows"] if r.get("slot") == "belief" and r.get("k") == "sample"]
        if not rows:
            skipped["no_rows"] += 1
            continue
        tl = opp.get(g["path"])
        for r in rows:
            if len(r["x"]) != BELIEF_SPEC.dim or len(r["yc"]) != len(UNITS) or len(r["yt"]) != len(TECH):
                skipped["width"] += 1
                continue
            X.append(r["x"])
            YC.append(r["yc"])
            YT.append(r["yt"])
            YB.append(r["yb"])
            groups.append(g["path"])
            OPP.append(template_at(tl, int(r["f"])) if tl else None)
    return (np.asarray(X, np.float32), np.asarray(YC, np.float32), np.asarray(YT, np.float32),
            np.asarray(YB, np.float32), np.asarray(groups), OPP, dict(skipped), len(games))


def train_belief(args) -> int:
    import sys

    from blackboard.train import fit
    X, YC, YT, YB, groups, OPP, skipped, n_games = belief_dataset([Path(p) for p in args.logs])
    print(f"games {n_games}  samples {len(X)}  skipped {skipped}")
    if len(X) == 0:
        print("no belief rows (play games with complete_map_information and the BeliefLog report)", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(spec=BELIEF_SPEC, groups=groups, hidden=args.hidden, epochs=args.epochs, l2=args.l2, seed=args.seed)
    m, rep = fit(X, YC, task="regression", labels=COUNT_LABELS, **kw)
    print(f"counts   {rep}  -> {m.save(out / 'belief_counts.npz')}")
    keep = YT.max(0) > 0                    # tech never present in the data cannot be learned
    m, rep = fit(X, YT, task="binary", labels=TECH_LABELS, **kw)
    print(f"tech     {rep}  ({int(keep.sum())}/{len(TECH)} types present) -> {m.save(out / 'belief_tech.npz')}")
    m, rep = fit(X, YB, task="regression", labels=("bases",), **kw)
    print(f"bases    {rep}  -> {m.save(out / 'belief_bases.npz')}")
    lab = [o for o in OPP if o]
    if len(set(lab)) >= 2:
        classes = sorted(set(lab))
        mask = np.array([o is not None for o in OPP])
        y = np.array([classes.index(o) for o in OPP if o is not None])
        m, rep = fit(X[mask], y, task="softmax", labels=classes, n_classes=len(classes),
                     **{**kw, "groups": groups[mask]})
        print(f"opponent {rep}  classes {classes} -> {m.save(out / 'belief_opponent.npz')}")
    else:
        print("opponent: need self-play games where both sides log (>=2 opponent templates)")
    return 0
