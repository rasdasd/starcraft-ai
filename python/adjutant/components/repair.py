"""Repair: one SCV repairs the most damaged mechanical unit (or bunker / turret) near home."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from bwbot import UnitType
from bwbot.enums import Order
from bwbot.observation import UnitTypeFlag

from ..macro.workers import pick_worker

REPAIR_ORDERS = (int(Order.Repair), int(Order.Move), int(Order.PlayerGuard), int(Order.Guard))


@register("Repair")
class Repair(Component):
    phase = Phase.ACT
    reads = ("world",)
    priority = Priority.NORMAL
    order = 20

    def __init__(self, unit_types: Optional[Sequence] = None, radius_tiles: int = 25, threshold: float = 0.75,
                 structures: bool = True) -> None:
        self.unit_types = None if unit_types is None else {int(getattr(UnitType, t)) if isinstance(t, str) else int(t)
                                                            for t in unit_types}
        self.radius = radius_tiles * 32
        self.threshold = threshold
        self.structures = structures
        self.scv: Optional[int] = None
        self.target: Optional[int] = None
        self._issued = 0

    def tick(self, bb: Blackboard) -> None:
        obs = bb.obs
        target = self._target(bb)
        scv = obs.unit(self.scv) if self.scv is not None else None
        if scv is not None and (target is None or bb.leases.owner(self.scv) not in (None, self.slot)
                                or (int(scv["order"]) not in REPAIR_ORDERS and self.target is not None
                                    and bb.frame - self._issued > 24)):
            scv = None
        if target is None or scv is None:
            if self.scv is not None:
                bb.leases.release(self.scv, self.slot)
            self.scv = self.target = None
        if target is None:
            return
        if scv is None:
            allowed = {int(u["id"]) for u in bb.world.workers if bb.leases.owner(int(u["id"])) is None}
            scv = pick_worker(obs, bb.world.workers, int(target["x"]), int(target["y"]), allowed)
            if scv is None:
                return
            self.scv = int(scv["id"])
            bb.leases.lease(self.scv, self.slot, self.priority, "repair", bb.frame)
        if self.target != int(target["id"]) or int(scv["order"]) != int(Order.Repair):
            bb.act.repair(scv, target)
            self.target, self._issued = int(target["id"]), bb.frame

    def _target(self, bb: Blackboard):
        game, obs = bb.game, bb.obs
        mine = obs.completed(obs.my_units)
        if len(mine) == 0:
            return None
        types = np.clip(mine["type"], 0, len(game.unit_types) - 1)
        flags = game.unit_types["flags"][types]
        if self.unit_types is not None:
            cand = np.isin(mine["type"], list(self.unit_types))
        else:
            mech = ((flags & UnitTypeFlag.Mechanical) != 0) & ((flags & UnitTypeFlag.Worker) == 0)
            cand = mech & ((flags & UnitTypeFlag.Building) == 0) & ((flags & UnitTypeFlag.CanAttack) != 0)
            if self.structures:
                cand = cand | np.isin(mine["type"], [int(UnitType.Terran_Bunker), int(UnitType.Terran_Missile_Turret)])
        units = mine[cand]
        if len(units) == 0:
            return None
        hurt = units[units["hit_points"] < game.unit_types["max_hit_points"][units["type"]] * self.threshold]
        if len(hurt) == 0:
            return None
        homes = [(bb.world.main_tile[0] * 32, bb.world.main_tile[1] * 32)] + [(x, y) for x, y in bb.world.depots]
        near = np.zeros(len(hurt), bool)
        for hx, hy in homes:
            near |= (hurt["x"] - hx) ** 2 + (hurt["y"] - hy) ** 2 <= self.radius ** 2
        hurt = hurt[near]
        if len(hurt) == 0:
            return None
        return hurt[int(hurt["hit_points"].argmin())]
