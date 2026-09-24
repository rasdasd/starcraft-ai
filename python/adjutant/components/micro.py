"""Combat Micro: executes squad orders unit by unit.

Owns (leases) every unit in `squads`. Per unit:
- Fighting (enemies it can hit within weapon range + `engage_tiles`): focus fire without overkill
  (targets scored by threat per remaining hit point; a target stops attracting shooters once the
  volleys assigned to it this decision cover its hit points), kiting (ranged units step away from
  melee units while their weapon cools down), harass squads prefer workers.
- Siege tanks siege when ground enemies are in cannon range (and none is on top of them) or when
  they reach a hold/defend/contain point, and unsiege once nothing has been near for `unsiege_s`
  and the squad has somewhere else to be.
- Otherwise: attack-move (or move, for retreats) to the squad's point.
- Command throttling: an identical command to the same unit is not repeated within `repeat_frames`
  (or at all while the unit is already doing it), and at most `max_cmds` commands go out per
  decision, most urgent first (retreats and fights before walking).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from bwbot.enums import Order, TechType, UnitType as U
from bwbot.observation import UnitFlag, UnitTypeFlag

from .. import engage as E
from .executors import _sync_leases

TANK, SIEGED = int(U.Terran_Siege_Tank_Tank_Mode), int(U.Terran_Siege_Tank_Siege_Mode)
STATIC_POINTS = ("hold", "defend", "contain")
FPS = 24


@register("Micro")
class Micro(Component):
    phase = Phase.ACT
    reads = ("world", "squads", "belief")
    priority = Priority.COMBAT
    order = 40

    def __init__(self, max_cmds: int = 60, repeat_frames: int = 24, engage_tiles: int = 4, kite: bool = True,
                 focus: bool = True, siege: bool = True, unsiege_s: int = 4, arrive_tiles: int = 3) -> None:
        self.max_cmds = max_cmds
        self.repeat = repeat_frames
        self.engage_px = engage_tiles * 32
        self.kite = kite
        self.focus = focus
        self.siege = siege
        self.unsiege_frames = unsiege_s * FPS
        self.arrive_px = arrive_tiles * 32

    def on_start(self, bb: Blackboard) -> None:
        self.table = E.TypeTable(bb.game)
        self.etable = E.TypeTable(bb.game)
        self.last: dict[int, tuple[int, tuple]] = {}
        self.enemy_near: dict[int, int] = {}
        self.sent = 0
        self.skipped = 0

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        obs, w = bb.obs, bb.world
        frame = bb.frame
        me = getattr(obs, "me", None)
        if me is not None and getattr(me.upgrade_level, "size", 0) and \
                not np.array_equal(self.table.upgrades, me.upgrade_level):
            self.table.upgrades = me.upgrade_level.copy()
            self.table._cache.clear()
        researched = me.has_researched if me is not None else np.zeros(0)
        siege_ok = self.siege and int(TechType.Tank_Siege_Mode) < researched.size and \
            bool(researched[int(TechType.Tank_Siege_Mode)])

        squads = bb.squads.squads
        owned = set().union(*(s.units for s in squads.values())) if squads else set()
        _sync_leases(bb, self.slot, owned, self.priority, "squad")
        if not owned:
            return
        by_id = {uid: u for uid in owned if (u := obs.unit(uid)) is not None}
        enemies = self._targets(w.enemies)
        e_xy = np.array([(int(e["x"]), int(e["y"])) for e in enemies], np.float32).reshape(-1, 2)
        assigned: dict[int, float] = {}
        cmds: list[tuple[float, int, tuple, Callable[[], None]]] = []
        for sq in squads.values():
            for uid in sq.units:
                u = by_id.get(uid)
                if u is None:
                    continue
                c = self._unit(bb, u, sq, enemies, e_xy, assigned, siege_ok, frame)
                if c is not None:
                    cmds.append((c[0], uid, c[1], c[2]))
        cmds.sort(key=lambda c: -c[0])
        for urgency, uid, key, fn in cmds[: self.max_cmds]:
            fn()
            self.last[uid] = (frame, key)
            self.sent += 1
        self.skipped += max(0, len(cmds) - self.max_cmds)
        alive = set(by_id)
        for uid in [k for k in self.last if k not in alive]:
            del self.last[uid]

    # ------------------------------------------------------------------ per unit
    def _unit(self, bb: Blackboard, u, sq, enemies, e_xy, assigned, siege_ok: bool, frame: int):
        act = bb.act
        t = int(u["type"])
        info = self.table.get(t)
        pos = (int(u["x"]), int(u["y"]))
        o = sq.order
        dest = (o.x, o.y)

        if o.kind == "retreat":
            if t == SIEGED:
                return self._cmd(u, 5, ("unsiege",), frame, lambda: act.unsiege(u))
            if _d2(pos, dest) <= self.arrive_px ** 2:
                return None
            return self._cmd(u, 5, ("move", dest[0] // 64, dest[1] // 64), frame, lambda: act.move(u, *dest))

        d2 = ((e_xy - pos) ** 2).sum(axis=1) if len(e_xy) else np.zeros(0)
        if t in (TANK, SIEGED):
            c = self._tank(bb, u, t, sq, enemies, d2, siege_ok, frame, dest)
            if c is not None or t == SIEGED:
                return c

        if not info.combat:
            # medics, vessels: stay with the squad
            if _d2(pos, dest) <= self.arrive_px ** 2:
                return None
            return self._cmd(u, 1, ("amove", dest[0] // 96, dest[1] // 96), frame,
                             lambda: act.attack_move(u, *dest))

        reach = max(_range(info.ground), _range(info.air)) + self.engage_px
        near = [i for i in np.nonzero(d2 <= reach * reach)[0] if self._can_hit(info, enemies[i])] if len(d2) else []
        if near:
            if self.kite:
                k = self._kite(u, info, enemies, near, d2, act, frame)
                if k is not None:
                    return k
            if self.focus:
                tgt = self._pick(u, info, enemies, near, d2, assigned, harass=o.kind == "harass")
                if tgt is not None:
                    tid = int(tgt["id"])
                    if int(u["order_target"]) == tid or int(u["target"]) == tid:
                        return None
                    return self._cmd(u, 4, ("atk", tid), frame, lambda: act.attack(u, tgt))
            return None

        if o.kind == "hunt_air" and o.target >= 0 and info.air is not None:
            tgt = next((e for e in enemies if int(e["id"]) == o.target), None)
            if tgt is not None:
                return self._cmd(u, 3, ("atk", o.target), frame, lambda: act.attack(u, tgt))
        if _d2(pos, dest) <= self.arrive_px ** 2 and o.kind in STATIC_POINTS + ("gather",):
            return None
        if int(u["order"]) == int(Order.AttackMove) and \
                _d2((int(u["order_target_x"]), int(u["order_target_y"])), dest) <= (3 * 32) ** 2:
            return None
        return self._cmd(u, 2, ("amove", dest[0] // 96, dest[1] // 96), frame, lambda: act.attack_move(u, *dest))

    def _tank(self, bb, u, t, sq, enemies, d2, siege_ok, frame, dest):
        act = bb.act
        uid = int(u["id"])
        ground = [i for i in range(len(enemies)) if not _flyer(bb, enemies[i]) and
                  self.etable.get(int(enemies[i]["type"])).combat]
        in_cannon = [i for i in ground if d2[i] <= (12 * 32) ** 2]
        close = [i for i in ground if d2[i] <= (3 * 32) ** 2]
        if in_cannon:
            self.enemy_near[uid] = frame
        at_point = _d2((int(u["x"]), int(u["y"])), dest) <= self.arrive_px ** 2 and sq.order.kind in STATIC_POINTS
        if t == TANK:
            if siege_ok and ((in_cannon and not close) or (at_point and not close)):
                return self._cmd(u, 4, ("siege",), frame, lambda: act.siege(u))
            return None
        quiet = frame - self.enemy_near.get(uid, -10_000) > self.unsiege_frames
        if quiet and not at_point:
            return self._cmd(u, 3, ("unsiege",), frame, lambda: act.unsiege(u))
        return None

    def _kite(self, u, info, enemies, near, d2, act, frame):
        w = info.ground
        if w is None or w.max_range < 3 * 32 or int(u["ground_weapon_cooldown"]) <= 0:
            return None
        melee = [i for i in near if (lambda ei: ei.ground is not None and ei.ground.max_range <= 32 and
                                     not ei.building)(self.etable.get(int(enemies[i]["type"])))]
        if not melee:
            return None
        i = min(melee, key=lambda i: d2[i])
        if d2[i] > (80 + 16) ** 2:
            return None
        ex, ey = int(enemies[i]["x"]), int(enemies[i]["y"])
        dx, dy = int(u["x"]) - ex, int(u["y"]) - ey
        n = math.hypot(dx, dy) or 1.0
        tx, ty = int(u["x"] + dx / n * 96), int(u["y"] + dy / n * 96)
        return self._cmd(u, 4.5, ("kite", tx // 32, ty // 32), frame, lambda: act.move(u, tx, ty), repeat=6)

    def _pick(self, u, info, enemies, near, d2, assigned, harass: bool):
        best, best_s, best_w = None, -math.inf, None
        fallback, fallback_s = None, -math.inf
        for i in near:
            e = enemies[i]
            ei = self.etable.get(int(e["type"]))
            w = info.air if ei.flyer else info.ground
            if w is None:
                continue
            hp = float(e["hit_points"]) + float(e["shields"])
            if ei.worker:
                prio = 4.0 if harass else 1.5
            elif ei.combat:
                prio = 3.0 if not ei.building else 2.0
            else:
                prio = 0.3
            dist = math.sqrt(d2[i])
            s = prio * 100.0 / (hp + 30.0) - (0.0 if dist <= w.max_range + 32 else 1.0) - dist / 640.0
            if assigned.get(int(e["id"]), 0.0) >= hp:
                if s > fallback_s:
                    fallback, fallback_s = e, s
                continue
            if s > best_s:
                best, best_s, best_w = e, s, (w, ei)
        if best is None:
            return fallback
        w, ei = best_w
        volley = w.per_frame(ei.armor, ei.size) * w.cooldown
        assigned[int(best["id"])] = assigned.get(int(best["id"]), 0.0) + volley
        return best

    # ------------------------------------------------------------------ helpers
    def _cmd(self, u, urgency: float, key: tuple, frame: int, fn, repeat: Optional[int] = None):
        last = self.last.get(int(u["id"]))
        if last is not None and last[1] == key and frame - last[0] < (self.repeat if repeat is None else repeat):
            return None
        return urgency, key, fn

    def _targets(self, enemies):
        if len(enemies) == 0:
            return enemies
        flags = enemies["flags"].astype(np.int64)
        hidden = ((flags & int(UnitFlag.Cloaked | UnitFlag.Burrowed)) != 0) & ((flags & int(UnitFlag.Detected)) == 0)
        return enemies[~hidden]

    def _can_hit(self, info: E.TypeInfo, e) -> bool:
        ei = self.etable.get(int(e["type"]))
        return (info.air if ei.flyer else info.ground) is not None

    def describe(self) -> dict:
        return {"impl": self.name, "max_cmds": self.max_cmds, "kite": self.kite, "focus": self.focus,
                "siege": self.siege}


def _range(w: Optional[E.Weapon]) -> int:
    return w.max_range if w is not None else 0


def _d2(a, b) -> float:
    return float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _flyer(bb: Blackboard, e) -> bool:
    return bool(bb.game.unit_types["flags"][int(e["type"])] & UnitTypeFlag.Flyer)
