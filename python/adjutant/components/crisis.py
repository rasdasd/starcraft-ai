"""Crisis Defense: scripted detectors with high-priority responses through the normal arbiters.

`Crisis` (DECIDE) writes `threats` and posts requests; it never commands units itself:
- worker_rush: several enemy workers in our main early -> pull workers.
- early_rush / overrun: enemy army near our bases that the engagement estimate says our army (with
  static defense) loses to -> defensive posture for `defend_s`, cancel an unstarted expansion, a
  bunker at the front (Terran, when barracks exist), and a worker pull if it reaches a mineral line.
- proxy / cannon_rush: enemy static defense or production buildings close to our bases early ->
  pull workers onto them while they are weak.
- cloak: enemy cloaked units near our army or bases without detection (or cloak tech in belief)
  -> comsat + turrets requested, scans requested on visible cloaked units.
- drop: enemy ground units inside our main while the army is away -> threat for the defense squad.
- air: enemy air near our bases and no anti-air -> turrets.

`WorkerDefense` (ACT, CRISIS priority) executes "units/worker_pull" requests: leases the nearest
healthy workers, attacks the threat's units, and releases workers when hurt or when it is over.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from blackboard.sections import Threat
from bwbot import Race, UnitType as U
from bwbot.observation import UnitFlag, UnitTypeFlag

from .. import engage as E
from .engagement import _Enemy
from .executors import _sync_leases

log = logging.getLogger("adjutant.crisis")

FPS = 24
STATIC_D = {int(U.Protoss_Photon_Cannon), int(U.Terran_Bunker), int(U.Zerg_Sunken_Colony),
            int(U.Zerg_Creep_Colony), int(U.Terran_Missile_Turret), int(U.Zerg_Spore_Colony)}
PROXY_OK = {int(U.Protoss_Pylon), int(U.Protoss_Gateway), int(U.Terran_Barracks), int(U.Zerg_Hatchery),
            int(U.Protoss_Forge)}
TRANSPORTS = {int(U.Terran_Dropship), int(U.Protoss_Shuttle)}
DETECTION = {int(U.Terran_Missile_Turret), int(U.Terran_Science_Vessel), int(U.Terran_Comsat_Station),
             int(U.Protoss_Observer), int(U.Protoss_Photon_Cannon), int(U.Zerg_Overlord), int(U.Zerg_Spore_Colony)}


def _d2(a, b) -> float:
    return float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


@register("Crisis")
class Crisis(Component):
    phase = Phase.DECIDE
    reads = ("world", "belief", "meta")
    writes = ("threats",)
    priority = Priority.CRISIS
    order = 0

    def __init__(self, base_radius_tiles: int = 18, early_s: int = 360, defend_s: int = 45,
                 max_pull: int = 12, lose_prob: float = 0.4, pull_prob: float = 0.5) -> None:
        self.base_px = base_radius_tiles * 32
        self.pull_prob = pull_prob
        self.early = early_s * FPS
        self.defend_frames = defend_s * FPS
        self.max_pull = max_pull
        self.lose_prob = lose_prob

    def on_start(self, bb: Blackboard) -> None:
        self.table = E.TypeTable(bb.game)
        self.seen: set[str] = set()

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        w, frame, thr = bb.world, bb.frame, bb.threats
        homes = [(w.main_tile[0] * 32 + 64, w.main_tile[1] * 32 + 48)] + [tuple(d) for d in w.depots]
        enemies = list(w.enemies) if len(w.enemies) else []
        near = [u for u in enemies if any(_d2((int(u["x"]), int(u["y"])), h) <= self.base_px ** 2 for h in homes)]
        active: list[Threat] = []

        self._worker_rush(bb, near, frame, active)
        self._army_threat(bb, near, homes, frame, active)
        self._proxy(bb, enemies, homes, frame, active)
        self._cloak(bb, enemies, homes, frame, active)
        self._air(bb, near, homes, frame, active)

        thr.active = active
        thr.level = max((t.severity for t in active), default=0.0)
        if any(t.kind in ("worker_rush", "early_rush", "overrun", "cannon_rush") and t.severity >= 0.5
               for t in active):
            thr.posture_override = "defend"
            thr.override_until = frame + self.defend_frames
        elif frame > thr.override_until:
            thr.posture_override = None
        for t in active:
            if t.kind not in self.seen:
                self.seen.add(t.kind)
                bb.raise_event(f"threat_{t.kind}")
                bb.record("crisis", "threat", threat=t.kind, severity=round(t.severity, 2), x=t.x, y=t.y,
                          n=len(t.units))
                log.info("f%d threat %s sev %.2f at (%d,%d)", frame, t.kind, t.severity, t.x, t.y)

    # ------------------------------------------------------------------ detectors
    def _worker_rush(self, bb, near, frame, active) -> None:
        ut = bb.game.unit_types
        workers = [u for u in near if ut["flags"][int(u["type"])] & UnitTypeFlag.Worker]
        if len(workers) < 3 or frame > self.early:
            return
        cx, cy = int(np.mean([u["x"] for u in workers])), int(np.mean([u["y"] for u in workers]))
        ids = [int(u["id"]) for u in workers]
        active.append(Threat("worker_rush", cx, cy, min(1.0, len(workers) / 5), frame, ids))
        self._pull(bb, "worker_rush", (cx, cy), len(workers) + 2)

    def _army_threat(self, bb, near, homes, frame, active) -> None:
        ut = bb.game.unit_types
        army = [u for u in near if self.table.get(int(u["type"])).combat
                and not ut["flags"][int(u["type"])] & (UnitTypeFlag.Worker | UnitTypeFlag.Building)]
        if not army:
            return
        cx, cy = int(np.mean([u["x"] for u in army])), int(np.mean([u["y"] for u in army]))
        ev = bb.services.get("engage")
        if ev is None:
            return
        w = bb.world
        own_rows = list(w.army) + [u for u in w.buildings if self.table.get(int(u["type"])).combat
                                   and _d2((int(u["x"]), int(u["y"])), (cx, cy)) <= (12 * 32) ** 2]
        enemy = ev.enemy_side([_Enemy(int(u["id"]), int(u["type"]), int(u["x"]), int(u["y"]),
                                      float(u["hit_points"]) + float(u["shields"]), False) for u in army])
        est = ev.evaluate(ev.own_side(own_rows), enemy)
        ids = [int(u["id"]) for u in army]
        in_main = self._in_main(bb, (cx, cy))
        transports = any(int(u["type"]) in TRANSPORTS for u in bb.world.enemies)
        if in_main and (transports or frame > self.early) and len(army) <= 12 and self._army_away(bb, (cx, cy)):
            active.append(Threat("drop", cx, cy, 0.8, frame, ids))
        if est.win_prob >= self.lose_prob:
            if est.win_prob < 0.7:
                active.append(Threat("pressure", cx, cy, 0.5, frame, ids))
            return
        kind = "early_rush" if frame <= self.early else "overrun"
        sev = min(1.0, 0.5 + (self.lose_prob - est.win_prob))
        active.append(Threat(kind, cx, cy, sev, frame, ids))
        hall = {int(Race.Terran): int(U.Terran_Command_Center), int(Race.Protoss): int(U.Protoss_Nexus),
                int(Race.Zerg): int(U.Zerg_Hatchery)}.get(bb.meta.self_race)
        bm = bb.services.get("buildings")
        if hall is not None and bm is not None and bm.starting(hall):
            bb.request("production", self.slot, ttl=self.defend_frames, priority=int(Priority.CRISIS),
                       type_id=hall, item="cancel")
        if bb.meta.self_race == int(Race.Terran) and w.count_completed(U.Terran_Barracks) and \
                w.count(U.Terran_Bunker) == 0:
            front = w.main_choke if w.main_choke is not None else homes[0]
            bb.request("production", self.slot, ttl=FPS * 30, priority=int(Priority.CRISIS),
                       type_id=int(U.Terran_Bunker), item="build",
                       near=(int(front[0]) // 32, int(front[1]) // 32))
        if any(_d2((cx, cy), h) <= (10 * 32) ** 2 for h in homes) and len(w.workers):
            n = min(self.max_pull, 2 * len(army) + 2)
            wk = sorted(w.workers, key=lambda u: _d2((int(u["x"]), int(u["y"])), (cx, cy)))[:n]
            with_workers = ev.evaluate(ev.own_side(own_rows + list(wk)), enemy)
            if with_workers.win_prob >= self.pull_prob:
                self._pull(bb, kind, (cx, cy), n)

    def _proxy(self, bb, enemies, homes, frame, active) -> None:
        if frame > self.early:
            return
        blds = [u for u in enemies if int(u["type"]) in STATIC_D | PROXY_OK
                and any(_d2((int(u["x"]), int(u["y"])), h) <= (self.base_px * 1.3) ** 2 for h in homes)]
        blds = [u for u in blds if int(u["type"]) not in (int(U.Zerg_Hatchery),) or frame < 24 * 240]
        if not blds:
            return
        static = [u for u in blds if int(u["type"]) in STATIC_D]
        kind = "cannon_rush" if static else "proxy"
        cx, cy = int(np.mean([u["x"] for u in blds])), int(np.mean([u["y"] for u in blds]))
        ids = [int(u["id"]) for u in blds]
        workers = [u for u in enemies if bb.game.unit_types["flags"][int(u["type"])] & UnitTypeFlag.Worker
                   and _d2((int(u["x"]), int(u["y"])), (cx, cy)) <= (10 * 32) ** 2]
        ids += [int(u["id"]) for u in workers]
        active.append(Threat(kind, cx, cy, 0.9 if static else 0.6, frame, ids))
        self._pull(bb, kind, (cx, cy), min(self.max_pull, 3 * len(blds) + len(workers)))

    def _cloak(self, bb, enemies, homes, frame, active) -> None:
        w = bb.world
        detection = any(w.count_completed(t) for t in DETECTION)
        cloaked = [u for u in enemies if (int(u["flags"]) & int(UnitFlag.Cloaked | UnitFlag.Burrowed))
                   and not int(u["flags"]) & int(UnitFlag.Detected)]
        if not cloaked and not bb.belief.cloak:
            return
        if cloaked:
            c = cloaked[0]
            cx, cy = int(c["x"]), int(c["y"])
            active.append(Threat("cloak", cx, cy, 1.0 if not detection else 0.6, frame,
                                 [int(u["id"]) for u in cloaked]))
            bb.request("scan", self.slot, ttl=FPS * 3, priority=int(Priority.CRISIS), x=cx, y=cy)
        else:
            active.append(Threat("cloak_tech", homes[0][0], homes[0][1], 0.5, frame, []))
        if bb.meta.self_race == int(Race.Terran):
            if w.count(U.Terran_Comsat_Station) == 0:
                bb.request("production", self.slot, ttl=FPS * 30, priority=int(Priority.CRISIS),
                           type_id=int(U.Terran_Comsat_Station), item="addon")
            if w.count(U.Terran_Missile_Turret) < len(w.depots):
                mt = w.main_tile
                bb.request("production", self.slot, ttl=FPS * 30, priority=int(Priority.CRISIS) - 5,
                           type_id=int(U.Terran_Missile_Turret), item="build", count=len(w.depots),
                           near=(mt[0], mt[1] + 3))

    def _air(self, bb, near, homes, frame, active) -> None:
        ut = bb.game.unit_types
        air = [u for u in near if ut["flags"][int(u["type"])] & UnitTypeFlag.Flyer
               and self.table.get(int(u["type"])).combat]
        if not air:
            return
        w = bb.world
        aa = [u for u in list(w.army) + list(w.buildings) if self.table.get(int(u["type"])).air is not None]
        if len(aa) >= len(air):
            return
        cx, cy = int(air[0]["x"]), int(air[0]["y"])
        active.append(Threat("air", cx, cy, 0.7, frame, [int(u["id"]) for u in air]))
        if bb.meta.self_race == int(Race.Terran):
            mt = w.main_tile
            bb.request("production", self.slot, ttl=FPS * 30, priority=int(Priority.CRISIS) - 5,
                       type_id=int(U.Terran_Missile_Turret), item="build", count=len(w.depots) + 1,
                       near=(mt[0], mt[1] + 3))

    # ------------------------------------------------------------------ helpers
    def _pull(self, bb: Blackboard, why: str, at: tuple[int, int], n: int) -> None:
        bb.request("units", self.slot, ttl=FPS * 2, priority=int(Priority.CRISIS), purpose="worker_pull",
                   count=int(n), x=int(at[0]), y=int(at[1]))

    def _in_main(self, bb: Blackboard, p) -> bool:
        mt = bb.world.main_tile
        return _d2(p, (mt[0] * 32 + 64, mt[1] * 32 + 48)) <= (14 * 32) ** 2

    def _army_away(self, bb: Blackboard, p) -> bool:
        army = bb.world.army
        if len(army) == 0:
            return True
        d = (army["x"] - p[0]) ** 2 + (army["y"] - p[1]) ** 2
        return bool((d > (20 * 32) ** 2).all())


@register("WorkerDefense")
class WorkerDefense(Component):
    """Executes worker-pull requests (see `Crisis`)."""

    phase = Phase.ACT
    reads = ("world", "threats")
    priority = Priority.CRISIS
    order = 5

    def __init__(self, release_hp: float = 0.35, leash_tiles: int = 14) -> None:
        self.release_hp = release_hp
        self.leash_px = leash_tiles * 32

    def on_start(self, bb: Blackboard) -> None:
        self.pulled: set[int] = set()

    def tick(self, bb: Blackboard) -> None:
        req = next((r for r in bb.requests.active("units") if r.purpose == "worker_pull"), None)
        w, obs = bb.world, bb.obs
        alive = {int(u["id"]): u for u in w.workers} if len(w.workers) else {}
        self.pulled &= set(alive)
        if req is None:
            if self.pulled:
                log.info("f%d worker pull over (%d released)", bb.frame, len(self.pulled))
            self.pulled.clear()
            _sync_leases(bb, self.slot, set(), self.priority, "pull")
            return
        ut = bb.game.unit_types
        for uid in list(self.pulled):
            u = alive[uid]
            if float(u["hit_points"]) < self.release_hp * float(ut["max_hit_points"][int(u["type"])]):
                self.pulled.discard(uid)
        want = int(req.count)
        if len(self.pulled) < want:
            cands = []
            for uid, u in alive.items():
                if uid in self.pulled:
                    continue
                owner = bb.leases.owner(uid)
                if owner not in (None, self.slot):
                    lease = bb.leases.get(uid)
                    if lease is not None and lease.purpose in ("build", "scout"):
                        continue
                if float(u["hit_points"]) < 0.6 * float(ut["max_hit_points"][int(u["type"])]):
                    continue
                cands.append((_d2((int(u["x"]), int(u["y"])), (req.x, req.y)), uid))
            cands.sort()
            for _, uid in cands[: want - len(self.pulled)]:
                self.pulled.add(uid)
        elif len(self.pulled) > want:
            for uid in sorted(self.pulled)[want:]:
                self.pulled.discard(uid)
        _sync_leases(bb, self.slot, self.pulled, self.priority, "pull")

        targets = set()
        for t in bb.threats.active:
            if _d2((t.x, t.y), (req.x, req.y)) <= (12 * 32) ** 2:
                targets |= set(t.units)
        enemies = [e for e in w.enemies if int(e["id"]) in targets] if len(w.enemies) else []
        enemies = [e for e in enemies if _d2((int(e["x"]), int(e["y"])), (req.x, req.y)) <= self.leash_px ** 2]
        for uid in self.pulled:
            u = alive[uid]
            if enemies:
                e = min(enemies, key=lambda e: (float(e["hit_points"]) + float(e["shields"]),
                                                _d2((int(e["x"]), int(e["y"])), (int(u["x"]), int(u["y"])))))
                if int(u["order_target"]) != int(e["id"]):
                    bb.act.attack(u, e)
            elif _d2((int(u["x"]), int(u["y"])), (req.x, req.y)) > (4 * 32) ** 2:
                bb.act.attack_move(u, req.x, req.y)

    def summary(self) -> str:
        return f"pulled {len(self.pulled)}"
