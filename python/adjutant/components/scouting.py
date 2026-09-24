"""Scouting slot, driven by belief staleness and strategy needs.

1. Initial worker scout (mybot ScoutManager): an SCV walks the remaining start candidates, then
   watches the enemy main until threatened.
2. Re-scouts: bases are ranked by value x staleness (known enemy bases > the enemy natural >
   other expansions near the enemy); one cheap unit (vulture, marine, ... or an SCV early) is
   leased just above COMBAT priority (taken from the squad pool, never from the defense squad), so
   the army cannot pull it back, and sent to the best target. It
   returns to the army when it arrives, is threatened, or times out.
3. Comsat scans: before an attack when the target is stale, and periodically on the stalest
   enemy base while keeping `scan_reserve` energy for detection.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from bwbot import TechType, UnitFlag, UnitType as U
from mybot.scout import ScoutManager

from .. import compat

log = logging.getLogger("adjutant.scouting")

RESCOUT_TYPES = (U.Terran_Vulture, U.Terran_Marine, U.Terran_Goliath, U.Terran_Wraith)
RESCOUT_PRIORITY = int(Priority.COMBAT) + 1        # above the squad executor, below crisis
ARRIVE = 6 * 32
THREAT = 7 * 32


@register("Scouting")
class Scouting(Component):
    phase = Phase.ACT
    reads = ("world", "belief", "strategy", "meta", "squads")
    writes = ("scouting",)
    priority = Priority.SCOUT
    order = 10

    def __init__(self, worker_scout: bool = True, stale_s: int = 120, rescout_every_s: int = 45,
                 timeout_s: int = 75, unit_types: Optional[Sequence] = None, scan: bool = True,
                 scan_reserve: int = 50, scan_every_s: int = 150) -> None:
        self.worker_scout = worker_scout
        self.stale = stale_s * 24
        self.rescout_every = rescout_every_s * 24
        self.timeout = timeout_s * 24
        types = unit_types if unit_types is not None else RESCOUT_TYPES
        self.unit_types = [int(getattr(U, t)) if isinstance(t, str) else int(t) for t in types]
        self.scan = scan
        self.scan_reserve = scan_reserve
        self.scan_every = scan_every_s * 24
        self.initial = ScoutManager()

    def on_start(self, bb: Blackboard) -> None:
        self.initial.on_start(bb.game, bb.services["info"])
        if not self.worker_scout:
            self.initial.done = True
        self.unit_id: Optional[int] = None
        self.target: Optional[tuple[int, int]] = None
        self.target_base: Optional[int] = None
        self.sent = -10 ** 9
        self.next_rescout = 0
        self.last_periodic_scan = 0
        self.was_attacking = False
        self.want_attack_scan = False

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        wm = compat.worker_manager(bb)
        sc = bb.scouting
        sc.scouts = {}
        if not self.initial.done:
            self.initial.update(s, bb.act, wm, bb.services["info"])
            wid = self.initial.worker_id
            if wid is not None:
                bb.leases.lease(wid, self.slot, self.priority, "scout", bb.frame)
                i = self.initial
                dest = i.targets[i.ti] if i.ti < len(i.targets) else (0, 0)
                sc.scouts[wid] = (dest[0] * 32 + 64, dest[1] * 32 + 48)
            if self.initial.done and wid is not None:
                bb.leases.release(wid, self.slot)
        sc.initial_done = self.initial.done

        ranked = self.rank_targets(bb)
        sc.targets = [c for _, c, _ in ranked]
        if self.initial.done:
            self._rescout(bb, s, ranked)
        if self.unit_id is not None and self.target is not None:
            sc.scouts[self.unit_id] = self.target
        if self.scan:
            self._scans(bb, ranked)

    # ------------------------------------------------------------------ targets
    def rank_targets(self, bb: Blackboard) -> list[tuple[float, tuple[int, int], int]]:
        """(score, centre px, base id), best first; only bases staler than `stale_s`."""
        g, b, w = bb.game, bb.belief, bb.world
        enemy_alive = {e.base_id for e in b.bases if e.alive}
        es = b.enemy_start
        emain = None
        if es is not None:
            emain = min(g.bases, key=lambda base: (base.tile[0] - es[0]) ** 2 + (base.tile[1] - es[1]) ** 2)
        stale_limit = self.stale // 2 if (b.opening == "unknown" and bb.frame > 24 * 240) else self.stale
        out = []
        for base in g.bases:
            cx, cy = base.center
            if any((cx - x) ** 2 + (cy - y) ** 2 < (8 * 32) ** 2 for x, y in w.depots):
                continue
            age = b.staleness.get(base.id, bb.frame)
            if age < stale_limit:
                continue
            if base.id in enemy_alive:
                value = 3.0
            elif emain is not None and base.id != emain.id:
                d = ((cx - emain.center[0]) ** 2 + (cy - emain.center[1]) ** 2) ** 0.5
                value = 2.0 if d < 25 * 32 else 1.0 + max(0.0, 1.0 - d / (100 * 32))
            elif emain is not None:
                value = 2.5                        # enemy main itself
            else:
                value = 1.5 if base.starting else 0.5
            out.append((value * min(age / max(1, stale_limit), 3.0), (int(cx), int(cy)), base.id))
        out.sort(key=lambda r: -r[0])
        return out

    # ------------------------------------------------------------------ unit re-scouts
    def _rescout(self, bb: Blackboard, s, ranked) -> None:
        obs = bb.obs
        if self.unit_id is not None:
            u = obs.unit(self.unit_id)
            if u is None:
                self._release(bb, "lost")
                self.next_rescout = bb.frame + self.rescout_every * 2
                return
            x, y = int(u["x"]), int(u["y"])
            tx, ty = self.target
            arrived = (x - tx) ** 2 + (y - ty) ** 2 < ARRIVE ** 2
            fresh = self.target_base is not None and bb.belief.staleness.get(self.target_base, 10 ** 9) < 24 * 2
            if arrived or fresh or bb.frame - self.sent > self.timeout or self._threatened(s, x, y):
                self._release(bb, "arrived" if arrived or fresh else "timeout/threat")
                self.next_rescout = bb.frame + self.rescout_every
                return
            if bool(int(u["flags"]) & int(UnitFlag.Idle)) or \
                    abs(int(u["order_target_x"]) - tx) + abs(int(u["order_target_y"]) - ty) > 64:
                bb.act.move(u, tx, ty)
            return
        if not ranked or bb.frame < self.next_rescout:
            return
        _, (tx, ty), bid = ranked[0]
        u = self._pick_unit(bb, tx, ty)
        if u is None:
            return
        uid = int(u["id"])
        if not bb.leases.lease(uid, self.slot, RESCOUT_PRIORITY, "rescout", bb.frame):
            return
        self.unit_id, self.target, self.target_base, self.sent = uid, (tx, ty), bid, bb.frame
        bb.act.move(u, tx, ty)
        bb.record("scouting", "rescout", unit=bb.game.type_name(int(u["type"])), base=bid)
        log.info("f%d rescout %s #%d -> base %d", bb.frame, bb.game.type_name(int(u["type"])), uid, bid)

    def _pick_unit(self, bb: Blackboard, tx: int, ty: int):
        obs = bb.obs
        for t in self.unit_types:
            units = obs.my_completed(t)
            if len(units) == 0:
                continue
            free = [u for u in units if self._takeable(bb, int(u["id"]))]
            if not free:
                continue
            d = [(int(u["x"]) - tx) ** 2 + (int(u["y"]) - ty) ** 2 for u in free]
            return free[int(np.argmin(d))]
        return None

    def _takeable(self, bb: Blackboard, uid: int) -> bool:
        """Free, ours, or an idle-ish army unit held by the squad executor (not a defender)."""
        lease = bb.leases.get(uid)
        if lease is None or lease.owner == self.slot:
            return True
        if lease.purpose != "squad" or lease.priority >= RESCOUT_PRIORITY:
            return False
        d = bb.squads.get("defense")
        return d is None or uid not in d.units

    def _release(self, bb: Blackboard, why: str) -> None:
        if self.unit_id is not None:
            bb.leases.release(self.unit_id, self.slot)
            log.debug("f%d rescout #%d done (%s)", bb.frame, self.unit_id, why)
        self.unit_id = self.target = self.target_base = None

    @staticmethod
    def _threatened(s, x: int, y: int) -> bool:
        e = s.enemies
        if len(e) == 0:
            return False
        return bool((((e["x"] - x) ** 2 + (e["y"] - y) ** 2) < THREAT ** 2).any())

    # ------------------------------------------------------------------ scans
    def _scans(self, bb: Blackboard, ranked) -> None:
        obs, sc = bb.obs, bb.scouting
        stance = bb.strategy.posture.stance
        attacking = stance in ("attack", "all_in")
        if attacking and not self.was_attacking:
            self.want_attack_scan = True
        self.was_attacking = attacking
        comsats = obs.my_completed(U.Terran_Comsat_Station)
        req = next(iter(bb.requests.active("scan")), None)     # e.g. crisis: cloaked units in our base
        cooldown = 24 * 8 if req is not None else 24 * 10
        if len(comsats) == 0 or bb.frame - sc.last_scan_frame < cooldown:
            return
        best = comsats[int(np.argmax(comsats["energy"]))]
        energy = int(best["energy"])
        if energy < 50:
            return
        target = None
        why = ""
        if req is not None:
            target, why = (req.x, req.y), f"request:{req.source}"
        es = bb.belief.enemy_start
        if target is None and self.want_attack_scan and attacking and es is not None:
            self.want_attack_scan = False
            base = min(bb.game.bases, key=lambda b: (b.tile[0] - es[0]) ** 2 + (b.tile[1] - es[1]) ** 2)
            if bb.belief.staleness.get(base.id, bb.frame) > 24 * 60:
                target, why = base.center, "pre-attack"
        if target is None and energy >= 50 + self.scan_reserve and ranked \
                and bb.frame - self.last_periodic_scan >= self.scan_every:
            target, why = ranked[0][1], "periodic"
            self.last_periodic_scan = bb.frame
        if target is None:
            return
        bb.act.use_tech_pos(best, int(TechType.Scanner_Sweep), target[0], target[1])
        sc.last_scan_frame = bb.frame
        bb.record("scouting", "scan", why=why, x=target[0], y=target[1])
        log.info("f%d scan (%s) at %s", bb.frame, why, target)

    def summary(self) -> str:
        return f"scout {self.unit_id} -> {self.target_base}"
