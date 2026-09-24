"""Belief implementations.

`LegacyBelief` wraps mybot's InformationManager + OpponentModel.
`ScriptedBelief` (v2) keeps those (fog memory, `services["info"]` for the legacy scout/combat,
opening classification) and adds: per-id unit tracking with kills from UnitDestroy, tech inferred
through the tech tree (a Vulture implies Factory and Barracks), enemy bases over BWEM bases with
start-location elimination, per-base staleness, soft opening probabilities and the enemy army's
last position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from blackboard.sections import EnemyBase
from bwbot import EventType, Race, UnitType as U
from bwbot.observation import UnitTypeFlag
from mybot.information import RESOURCE_DEPOTS, InformationManager
from mybot.opponent import OPENING_NAMES, RUSH_WINDOW, OpponentModel

from ..techtree import TechTree
from ..units import AIR_COMBAT, AIR_TECH, CLOAKERS


@register("LegacyBelief")
class LegacyBelief(Component):
    phase = Phase.SENSE
    reads = ("world",)
    writes = ("belief",)
    order = 20

    def __init__(self) -> None:
        self.info = InformationManager()
        self.opponent = OpponentModel()

    def on_start(self, bb: Blackboard) -> None:
        self.info.on_start(bb.game)
        self.opponent.on_start(bb.game)
        bb.services["info"] = self.info
        b = bb.belief
        b.enemy_start = self.info.enemy_start
        b.start_candidates = {tuple(s): 1.0 / max(1, len(self.info.other_starts)) for s in self.info.other_starts}

    def tick(self, bb: Blackboard) -> None:
        obs, game = bb.obs, bb.game
        prev_start = self.info.enemy_start
        self.info.update(obs, game)
        self.opponent.update(obs, game, self.info)
        snap = self.opponent.snapshot()
        b = bb.belief
        prev_open = b.opening
        b.enemy_race = snap.race
        b.enemy_start = self.info.enemy_start
        b.buildings = self.info.buildings()
        b.counts = {int(t): float(n) for t, n in snap.counts.items()}
        b.first_seen = dict(self.opponent.first)
        b.tech = {t for t, _, _ in b.buildings} | {
            int(t) for t in snap.counts if int(game.unit_types["flags"][int(t)]) & UnitTypeFlag.Building}
        b.air = float(snap.air_units)
        b.army_supply = float(snap.ground_army + snap.air_units)
        b.proxy = snap.proxy
        b.opening = OPENING_NAMES[snap.opening] if 0 <= snap.opening < len(OPENING_NAMES) else "unknown"
        b.memory = snap
        if self.info.has_enemy_base() and self.info.enemy_start != prev_start:
            bb.raise_event("enemy_base_found")
        if b.opening != prev_open:
            bb.raise_event("opening_changed")
        if "enemy_air" not in bb.stats and any(int(t) in AIR_COMBAT or int(t) in AIR_TECH
                                               for t, n in snap.counts.items() if n):
            bb.stats["enemy_air"] = bb.frame
            bb.raise_event("enemy_air")


@dataclass
class Track:
    type: int
    x: int
    y: int
    frame: int              # last seen
    building: bool


BASE_RADIUS = 10 * 32       # an enemy depot this close to a BWEM base centre owns it
ARMY_FRESH = 24 * 20        # army position from units seen in the last 20 s


def _opening_scores(b, race: int, first: dict[int, int], counts: dict[int, float], frame: int) -> dict[str, float]:
    """Evidence scores per opening class (log-odds-like); softmax gives `opening_probs`."""
    s = {n: 0.0 for n in OPENING_NAMES}
    s["unknown"] = 1.0
    early = lambda t: 0 <= first.get(int(t), -1) < RUSH_WINDOW  # noqa: E731
    if b.proxy:
        s["cheese"] += 5
    if race == Race.Zerg:
        if early(U.Zerg_Spawning_Pool) or early(U.Zerg_Zergling):
            s["rush"] += 3
        hatch = counts.get(int(U.Zerg_Hatchery), 0) + counts.get(int(U.Zerg_Lair), 0)
        if hatch >= 2:
            s["two_base"] += 2 + (1 if not early(U.Zerg_Spawning_Pool) else 0)
        if counts.get(int(U.Zerg_Spire), 0) or counts.get(int(U.Zerg_Mutalisk), 0):
            s["air"] += 4
        if counts.get(int(U.Zerg_Hydralisk_Den), 0):
            s["unknown"] += 0.5
    elif race == Race.Protoss:
        gates = counts.get(int(U.Protoss_Gateway), 0)
        if gates >= 2 and early(U.Protoss_Gateway):
            s["rush"] += 3
        if counts.get(int(U.Protoss_Forge), 0) and not gates:
            s["cheese"] += 2
            s["two_base"] += 1
        if counts.get(int(U.Protoss_Nexus), 0) >= 2:
            s["two_base"] += 3
        if counts.get(int(U.Protoss_Stargate), 0) or counts.get(int(U.Protoss_Fleet_Beacon), 0):
            s["air"] += 4
    elif race == Race.Terran:
        rax = counts.get(int(U.Terran_Barracks), 0)
        if rax >= 2:
            s["bio" if frame > RUSH_WINDOW or not early(U.Terran_Barracks) else "rush"] += 3
        if counts.get(int(U.Terran_Factory), 0):
            s["mech"] += 3
        if counts.get(int(U.Terran_Starport), 0):
            s["air"] += 3
        if counts.get(int(U.Terran_Command_Center), 0) >= 2:
            s["two_base"] += 3
    if early(U.Terran_Barracks) or early(U.Protoss_Gateway):
        s["rush"] += 1
    return s


@register("ScriptedBelief")
class ScriptedBelief(Component):
    """Belief v2 (see module doc). Also registers `services["info"]` like LegacyBelief."""

    phase = Phase.SENSE
    reads = ("world", "meta")
    writes = ("belief",)
    order = 20

    def __init__(self, temperature: float = 1.0) -> None:
        self.info = InformationManager()
        self.opponent = OpponentModel()
        self.temperature = temperature

    def on_start(self, bb: Blackboard) -> None:
        g = bb.game
        self.info.on_start(g)
        self.opponent.on_start(g)
        bb.services["info"] = self.info
        self.tree = TechTree(g)
        self.tracks: dict[int, Track] = {}
        self.dead: dict[int, int] = {}
        self.seen_max: dict[int, int] = {}
        self.first: dict[int, int] = {}
        self.base_seen: dict[int, int] = {b.id: -1 for b in g.bases}
        self.enemy_bases: dict[int, EnemyBase] = {}
        me = tuple(g.self_player.start_location)
        self.all_starts = [tuple(s) for s in g.start_locations.tolist() if tuple(s) != me]
        self.candidates = set(self.all_starts)
        b = bb.belief
        b.start_candidates = {s: 1.0 / max(1, len(self.candidates)) for s in self.candidates}
        b.enemy_start = self.info.enemy_start
        enemy = g.enemies[0] if g.enemies else None
        if enemy is not None and enemy.race in (int(Race.Zerg), int(Race.Terran), int(Race.Protoss)):
            b.enemy_race = enemy.race

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        obs, g, b = bb.obs, bb.game, bb.belief
        prev = (b.opening, self.info.enemy_start, len([e for e in self.enemy_bases.values() if e.alive]),
                b.cloak, bool(b.air))
        self.info.update(obs, g)
        self.opponent.update(obs, g, self.info)
        frame = obs.frame_count
        ut = g.unit_types
        n_types = len(ut)

        for e in obs.iter_events(EventType.UnitDestroy):
            tr = self.tracks.pop(int(e.unit), None)
            if tr is not None:
                self.dead[tr.type] = self.dead.get(tr.type, 0) + 1

        enemies = obs.enemy_units
        visible_now: dict[int, int] = {}
        for u in enemies:
            t = int(u["type"])
            uid = int(u["id"])
            bld = bool(ut["flags"][min(t, n_types - 1)] & UnitTypeFlag.Building)
            self.tracks[uid] = Track(t, int(u["x"]), int(u["y"]), frame, bld)
            visible_now[t] = visible_now.get(t, 0) + 1
            self.first.setdefault(t, frame)
        for t, n in visible_now.items():
            self.seen_max[t] = max(self.seen_max.get(t, 0), n)

        vis = obs.visible if obs.tiles.size else None
        if vis is not None:
            # buildings we should see but do not are gone (moved/destroyed out of sight)
            for uid, tr in list(self.tracks.items()):
                if tr.building and tr.frame != frame:
                    tx, ty = tr.x // 32, tr.y // 32
                    if 0 <= ty < vis.shape[0] and 0 <= tx < vis.shape[1] and vis[ty, tx]:
                        del self.tracks[uid]
            self._bases(bb, vis, frame)
            self._starts(bb, vis)

        counts: dict[int, float] = {}
        for tr in self.tracks.values():
            counts[tr.type] = counts.get(tr.type, 0) + 1
        b.counts = counts
        b.seen_max = dict(self.seen_max)
        b.dead = dict(self.dead)
        b.first_seen = dict(self.first)
        seen_b = {t for t in self.first if ut["flags"][min(t, n_types - 1)] & UnitTypeFlag.Building}
        inferred = self._infer(set(self.first)) - seen_b
        b.tech = seen_b | inferred
        b.inferred = inferred
        b.buildings = [(tr.type, tr.x, tr.y) for tr in self.tracks.values() if tr.building]
        b.units = {uid: (tr.type, tr.x, tr.y, tr.frame) for uid, tr in self.tracks.items()}
        b.enemy_race = int(self.opponent.race) if int(self.opponent.race) != int(Race.Unknown) else b.enemy_race
        b.enemy_start = next(iter(self.candidates)) if len(self.candidates) == 1 else self.info.enemy_start
        self._army(b, frame, ut)
        b.proxy = self.opponent.proxy
        b.cloak = any(t in CLOAKERS and t != int(U.Protoss_Observer) for t in self.first) or \
            any(t in b.tech for t in (int(U.Protoss_Templar_Archives), int(U.Terran_Covert_Ops),
                                       int(U.Protoss_Arbiter_Tribunal)))
        snap = self.opponent.snapshot()
        b.opening = OPENING_NAMES[snap.opening] if 0 <= snap.opening < len(OPENING_NAMES) else "unknown"
        scores = _opening_scores(b, b.enemy_race, self.first, counts, frame)
        if b.opening in scores:
            scores[b.opening] += 1.0
        z = np.array(list(scores.values())) / max(self.temperature, 1e-6)
        p = np.exp(z - z.max())
        p /= p.sum()
        b.opening_probs = {k: round(float(v), 3) for k, v in zip(scores, p)}
        b.bases = sorted(self.enemy_bases.values(), key=lambda e: e.base_id)
        b.staleness = {bid: (frame - f if f >= 0 else frame) for bid, f in self.base_seen.items()}
        b.memory = snap

        if self.info.enemy_start != prev[1] and self.info.has_enemy_base():
            bb.raise_event("enemy_base_found")
        if b.opening != prev[0]:
            bb.raise_event("opening_changed")
        alive = len([e for e in self.enemy_bases.values() if e.alive])
        if alive > prev[2] and prev[2] >= 1:
            bb.raise_event("enemy_expanded")
        if b.cloak and not prev[3]:
            bb.raise_event("enemy_cloak")
        if b.air and not prev[4]:
            bb.raise_event("enemy_air")

    # ------------------------------------------------------------------ pieces
    def _infer(self, types: set[int]) -> set[int]:
        out: set[int] = set()
        stack = list(types)
        while stack:
            t = stack.pop()
            for r in self.tree.unit_requires(t):
                if r not in out and self.tree.is_building(r):
                    out.add(r)
                    stack.append(r)
        return out

    def _bases(self, bb: Blackboard, vis: np.ndarray, frame: int) -> None:
        g = bb.game
        depots = [tr for tr in self.tracks.values() if tr.building and tr.type in RESOURCE_DEPOTS]
        for base in g.bases:
            cx, cy = base.center
            tx, ty = cx // 32, cy // 32
            seen = 0 <= ty < vis.shape[0] and 0 <= tx < vis.shape[1] and bool(vis[ty, tx])
            if seen:
                self.base_seen[base.id] = frame
            own = [d for d in depots if (d.x - cx) ** 2 + (d.y - cy) ** 2 <= BASE_RADIUS ** 2]
            eb = self.enemy_bases.get(base.id)
            if own:
                last = max(d.frame for d in own)
                if eb is None:
                    self.enemy_bases[base.id] = EnemyBase(base.id, base.tile, last, True)
                else:
                    eb.last_seen, eb.alive = last, True
            elif eb is not None and seen:
                eb.alive = False
                eb.last_seen = frame

    def _starts(self, bb: Blackboard, vis: np.ndarray) -> None:
        b = bb.belief
        depots = [(tr.x // 32, tr.y // 32) for tr in self.tracks.values() if tr.building and tr.type in RESOURCE_DEPOTS]
        for s in self.all_starts:
            if any((s[0] + 2 - x) ** 2 + (s[1] + 1 - y) ** 2 <= 8 ** 2 for x, y in depots):
                self.candidates = {s}
                b.start_candidates = {s: 1.0}
                return
        for s in list(self.candidates):
            x, y = s[0] + 2, s[1] + 1
            if 0 <= y < vis.shape[0] and 0 <= x < vis.shape[1] and vis[y, x] and len(self.candidates) > 1:
                near = [tr for tr in self.tracks.values() if tr.building
                        and (tr.x // 32 - x) ** 2 + (tr.y // 32 - y) ** 2 <= 12 ** 2]
                if not near:
                    self.candidates.discard(s)
        n = max(1, len(self.candidates))
        b.start_candidates = {s: 1.0 / n for s in self.candidates}
        if not self.info.has_enemy_base() and self.candidates:
            # InformationManager defaults to the first other start; keep it on a live candidate
            if self.info.enemy_start not in self.candidates:
                self.info.enemy_start = sorted(self.candidates)[0]

    def _army(self, b, frame: int, ut) -> None:
        n_types = len(ut)
        supply = 0.0
        air = 0
        xs, ys = [], []
        last = -1
        for tr in self.tracks.values():
            if tr.building or tr.type in (int(U.Terran_SCV), int(U.Zerg_Drone), int(U.Protoss_Probe),
                                          int(U.Zerg_Larva), int(U.Zerg_Egg), int(U.Zerg_Overlord)):
                continue
            t = min(tr.type, n_types - 1)
            if not ut["flags"][t] & UnitTypeFlag.CanAttack and tr.type not in (int(U.Protoss_Carrier),
                                                                             int(U.Protoss_Reaver)):
                continue
            supply += int(ut["supply_required"][t]) / 2
            if tr.type in AIR_COMBAT:
                air += 1
            if frame - tr.frame <= ARMY_FRESH:
                xs.append(tr.x)
                ys.append(tr.y)
                last = max(last, tr.frame)
        b.army_supply = supply
        b.air = float(air)
        if xs:
            b.army_pos = (int(np.median(xs)), int(np.median(ys)))
            b.army_seen_frame = last
