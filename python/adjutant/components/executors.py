"""ACT-phase executors over the mybot managers: construction/production, economy, repair.

Worker claims made through the shared WorkerManager are mirrored into unit leases so other
components (and the command filter) see who owns which SCV; workers leased by anyone else
(e.g. a crisis worker pull) are handed to the WorkerManager as `external` so it leaves them alone.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from bwbot import Actions, Color, UnitFlag, UnitType
from bwbot.observation import UnitTypeFlag
from mybot.buildings import BuildingManager
from mybot.macro import Placer
from mybot.policy import BuildAddon, Research, Train, Upgrade
from mybot.production import ProductionManager

from .. import compat


class SpreadProductionManager(ProductionManager):
    """Trains on a different idle producer for each Train intent in the same decision, and allows
    several addons of one type (`addon_targets[type]` = wanted total; default 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.addon_targets: dict[int, int] = {}

    def update(self, s, act, buildings) -> None:
        self._used: set[int] = set()
        super().update(s, act, buildings)

    def _addon(self, unit_type, s, act, budget) -> None:
        unit_type = int(unit_type)
        if not budget.can_afford(s.game, unit_type) or s.count(unit_type) >= self.addon_targets.get(unit_type, 1):
            return
        parent_type = int(s.game.unit_types["what_builds"][unit_type])
        for parent in s.obs.my_completed(parent_type):
            if (int(parent["addon"]) >= 0 or (int(parent["flags"]) & int(UnitFlag.Lifted))
                    or int(parent["id"]) in self._used or int(parent["train_queue_count"]) > 0):
                continue
            act.build_addon(parent, unit_type)
            self._used.add(int(parent["id"]))
            budget.spend(s.game, unit_type)
            return

    def _train(self, unit_type, s, act, budget) -> None:
        if not budget.can_afford(s.game, unit_type):
            return
        producer_type = int(s.game.unit_types["what_builds"][int(unit_type)])
        producers = s.obs.my_completed(producer_type)
        if producer_type == UnitType.Zerg_Larva:
            free = [p for p in producers if int(p["id"]) not in self._used]
            if free:
                act.morph(free[0], unit_type)
                self._used.add(int(free[0]["id"]))
                budget.spend(s.game, unit_type)
            return
        ok = (producers["train_queue_count"] == 0) & ((producers["flags"] & UnitFlag.Lifted) == 0)
        for p in producers[ok]:
            if int(p["id"]) in self._used:
                continue
            act.train(p, unit_type)
            self._used.add(int(p["id"]))
            budget.spend(s.game, unit_type)
            return


def _sync_leases(bb: Blackboard, owner: str, ids: set[int], priority: int, purpose: str) -> None:
    for uid in ids:
        bb.leases.lease(uid, owner, priority, purpose, bb.frame)
    for uid in bb.leases.owned(owner) - ids:
        bb.leases.release(uid, owner)


@register("Construction")
class Construction(Component):
    """Executes `plan`: building jobs (queue + BuildingManager + Placer), trains, addons,
    upgrades, research. Plan items are already in priority order."""

    phase = Phase.ACT
    reads = ("world", "plan")
    priority = Priority.CONSTRUCTION
    order = 0

    def __init__(self, spread: bool = True, draw: bool = True) -> None:
        self.production = SpreadProductionManager() if spread else ProductionManager()
        self.buildings = BuildingManager()
        self.placer = Placer()
        self.draw = draw

    def on_start(self, bb: Blackboard) -> None:
        self.production.reset()
        self.buildings.reset()
        self.placer.reset()
        bb.services.update(production=self.production, buildings=self.buildings, placer=self.placer)

    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        wm = compat.worker_manager(bb)
        plan = bb.plan
        intents = []
        if isinstance(self.production, SpreadProductionManager):
            self.production.addon_targets = {it.type_id: max(1, it.count) for it in plan.items if it.kind == "addon"}
        for it in plan.items:
            if it.kind == "build":
                self.production.ensure_build(it.type_id, self.buildings, front=it.priority >= Priority.SUPPLY,
                                             near=it.near, exact=it.exact)
            elif it.kind == "train":
                intents += [Train(it.type_id)] * max(1, it.count)
            elif it.kind == "addon":
                intents.append(BuildAddon(it.type_id))
            elif it.kind == "upgrade":
                intents.append(Upgrade(it.type_id))
            elif it.kind == "research":
                intents.append(Research(it.type_id))
        for t in plan.cancel:
            self.production.cancel(t, self.buildings, wm, self.placer, bb.game)
        self.production.ingest(intents, self.buildings)
        m, g = self._reserved_elsewhere(bb)
        if m or g:
            s = dataclasses.replace(s, minerals=s.minerals - m, gas=s.gas - g)
        self.production.update(s, bb.act, self.buildings)
        self.buildings.update(s, bb.act, wm, self.placer)
        _sync_leases(bb, self.slot, set(wm.build), self.priority, "build")
        if self.draw:
            self._draw(bb, s, bb.act)

    def _reserved_elsewhere(self, bb: Blackboard) -> tuple[int, int]:
        m = g = 0
        for key, (rm, rg, _, rp) in bb.budget.reservations.items():
            if not key.startswith(self.slot) and rp > self.priority:
                m, g = m + rm, g + rg
        return m, g

    def _draw(self, bb: Blackboard, s, act: Actions) -> None:
        game = bb.game
        for task in self.buildings.tasks:
            if task.tile is not None:
                act.draw_tile_box(task.tile[0], task.tile[1], int(game.unit_types["tile_width"][task.unit_type]),
                                  int(game.unit_types["tile_height"][task.unit_type]))
            if task.worker_id is not None:
                u = s.obs.unit(task.worker_id)
                if u is not None:
                    act.draw_circle(int(u["x"]), int(u["y"]), 12, color=Color.Yellow)

    def summary(self) -> str:
        jobs = ",".join(f"{t.unit_type}:{t.status[0]}" for t in self.buildings.tasks) or "-"
        return f"queue {self.production.queue} jobs {jobs}"


@register("Economy")
class Economy(Component):
    """Mineral/gas assignment for every worker nobody else owns."""

    phase = Phase.ACT
    reads = ("world",)
    priority = Priority.ECONOMY
    order = 30

    def __init__(self, local_mining: Optional[bool] = None) -> None:
        self.local_mining = local_mining

    def on_start(self, bb: Blackboard) -> None:
        wm = compat.worker_manager(bb)
        wm.reset()
        if self.local_mining is not None:
            wm.local_only = self.local_mining

    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        wm = compat.worker_manager(bb)
        managed = wm.build | wm.repair | wm.scout
        wm.external = {int(w["id"]) for w in s.workers
                       if int(w["id"]) not in managed and bb.leases.owner(int(w["id"])) not in (None, self.slot)}
        wm.update(s, bb.act)


@register("Repair")
class Repair(Component):
    """One SCV repairs the most damaged mechanical unit (or bunker/turret) near home."""

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

    def tick(self, bb: Blackboard) -> None:
        s = compat.state(bb)
        wm = compat.worker_manager(bb)
        target = self._target(bb, s)
        if target is not None:
            scv = wm.claim_repair(s, int(target["x"]), int(target["y"]))
            if scv is not None:
                bb.act.repair(scv, target)
        _sync_leases(bb, self.slot, set(wm.repair), self.priority, "repair")

    def _target(self, bb: Blackboard, s):
        game = bb.game
        mine = s.obs.completed(s.obs.my_units)
        if len(mine) == 0:
            return None
        types = np.clip(mine["type"], 0, len(game.unit_types) - 1)
        flags = game.unit_types["flags"][types]
        if self.unit_types is not None:
            cand = np.isin(mine["type"], list(self.unit_types))
        else:
            mech = ((flags & UnitTypeFlag.Mechanical) != 0) & ((flags & UnitTypeFlag.Worker) == 0)
            army = mech & ((flags & UnitTypeFlag.Building) == 0) & ((flags & UnitTypeFlag.CanAttack) != 0)
            cand = army
            if self.structures:
                cand = cand | np.isin(mine["type"], [int(UnitType.Terran_Bunker), int(UnitType.Terran_Missile_Turret)])
        units = mine[cand]
        if len(units) == 0:
            return None
        max_hp = game.unit_types["max_hit_points"][units["type"]]
        hurt = units[units["hit_points"] < max_hp * self.threshold]
        if len(hurt) == 0:
            return None
        mx, my = s.main_tile[0] * 32, s.main_tile[1] * 32
        homes = [(mx, my)] + [(x, y) for x, y in bb.world.depots]
        near = np.zeros(len(hurt), bool)
        for hx, hy in homes:
            near |= (hurt["x"] - hx) ** 2 + (hurt["y"] - hy) ** 2 <= self.radius ** 2
        hurt = hurt[near]
        if len(hurt) == 0:
            return None
        return hurt[int(hurt["hit_points"].argmin())]
