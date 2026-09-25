"""WorkerManager: stable mineral / gas / build / repair jobs.

Builders and repairers are claimed by other managers and are not yanked back to mine.
Gas stays at three workers per completed refinery, none while gas is heavily over-banked
relative to minerals. Mineral patches are filled to two
before a third is piled on. Gather commands are issued only when the assignment is new
or the worker is idle — not every decision.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from bwbot import Actions, Race, UnitType
from bwbot.enums import Order
from bwbot.observation import UnitTypeFlag

from .state import State

log = logging.getLogger("mybot.workers")

GAS_ORDERS = {int(Order.MoveToGas), int(Order.WaitForGas), int(Order.HarvestGas), int(Order.ReturnGas)}
MINERAL_ORDERS = {
    int(Order.MoveToMinerals), int(Order.WaitForMinerals), int(Order.MiningMinerals),
    int(Order.ReturnMinerals), int(Order.Harvest1), int(Order.Harvest2),
    int(Order.Harvest3), int(Order.Harvest4),
}
REPAIR_ORDERS = {int(Order.Repair), int(Order.MoveToRepair)}
GAS_PER_REFINERY = 3
REFINERY = {int(Race.Terran): int(UnitType.Terran_Refinery), int(Race.Zerg): int(UnitType.Zerg_Extractor),
            int(Race.Protoss): int(UnitType.Protoss_Assimilator)}
MINERALS_PER_PATCH = 2


LOCAL_MINERAL_PX = 12 * 32


class WorkerManager:
    def __init__(self, local_only: bool = False, gas_bank: tuple[int, int] = (300, 600)) -> None:
        self.gas_bank = gas_bank
        self.gas_per = GAS_PER_REFINERY
        self.build: set[int] = set()
        self.repair: set[int] = set()
        self.scout: set[int] = set()
        self.external: set[int] = set()        # workers owned by someone else (e.g. a crisis pull); set by callers
        self.gas: dict[int, int] = {}          # worker id -> refinery id
        self.mineral: dict[int, int] = {}      # worker id -> patch id
        self.local_only = local_only           # mine only patches near our own resource depots
        self.refinery_type = int(UnitType.Terran_Refinery)
        self._claimed_this_frame: set[int] = set()

    def reset(self) -> None:
        self.build.clear()
        self.repair.clear()
        self.scout.clear()
        self.external.clear()
        self.gas.clear()
        self.mineral.clear()
        self._claimed_this_frame.clear()
        self.gas_per = GAS_PER_REFINERY

    def claimed(self) -> set[int]:
        return self.build | self.repair | self.scout | self.external

    def claim_builder(self, s: State, near_tile: tuple[int, int]) -> Optional[np.void]:
        w = self._pick_free(s, near_tile[0] * 32, near_tile[1] * 32)
        if w is None:
            return None
        wid = int(w["id"])
        self.build.add(wid)
        self._claimed_this_frame.add(wid)
        self.gas.pop(wid, None)
        self.mineral.pop(wid, None)
        log.debug("claim builder #%d", wid)
        return w

    def claim_repair(self, s: State, x: int, y: int) -> Optional[np.void]:
        if self.repair:
            return None
        w = self._pick_free(s, x, y)
        if w is None:
            return None
        wid = int(w["id"])
        self.repair.add(wid)
        self._claimed_this_frame.add(wid)
        self.gas.pop(wid, None)
        self.mineral.pop(wid, None)
        return w

    def claim_scout(self, s: State, near_tile: tuple[int, int]) -> Optional[np.void]:
        w = self._pick_free(s, near_tile[0] * 32, near_tile[1] * 32)
        if w is None:
            return None
        wid = int(w["id"])
        self.scout.add(wid)
        self._claimed_this_frame.add(wid)
        self.gas.pop(wid, None)
        self.mineral.pop(wid, None)
        return w

    def release(self, worker_id: int) -> None:
        self.build.discard(int(worker_id))
        self.repair.discard(int(worker_id))
        self.scout.discard(int(worker_id))

    def update(self, s: State, act: Actions) -> None:
        alive = {int(w["id"]) for w in s.workers}
        for bucket in (self.build, self.repair, self.scout):
            for wid in list(bucket):
                if wid not in alive:
                    bucket.discard(wid)
        self.gas = {w: r for w, r in self.gas.items() if w in alive}
        self.mineral = {w: p for w, p in self.mineral.items() if w in alive}

        for wid in list(self.repair):
            if wid in self._claimed_this_frame:
                continue
            u = s.obs.unit(wid)
            if u is None or int(u["order"]) not in REPAIR_ORDERS:
                self.repair.discard(wid)

        self._assign_gas(s, act)
        self._assign_minerals(s, act)
        self._claimed_this_frame.clear()

    def _pick_free(self, s: State, x: int, y: int) -> Optional[np.void]:
        locked = self.claimed()
        prefer = []
        fallback = []
        for i, w in enumerate(s.workers):
            wid = int(w["id"])
            if wid in locked:
                continue
            fallback.append(i)
            if int(w["order"]) not in GAS_ORDERS and int(w["carry_resource_type"]) == 0:
                prefer.append(i)
        pool = prefer or fallback
        if not pool:
            return None
        return s.obs.nearest(s.workers[np.array(pool, dtype=np.intp)], x, y)

    def _gas_target(self, s: State) -> int:
        """Workers per refinery: none while gas is banked far beyond minerals, back to full once
        it has been spent down (hysteresis between the two `gas_bank` levels)."""
        low, high = self.gas_bank
        if s.gas >= high and s.gas > 2 * s.minerals:
            self.gas_per = 0
        elif s.gas < low:
            self.gas_per = GAS_PER_REFINERY
        return self.gas_per

    def _assign_gas(self, s: State, act: Actions) -> None:
        refs = s.obs.my_completed(REFINERY.get(int(s.game.self_race), self.refinery_type))
        if len(refs) == 0:
            self.gas.clear()
            return
        per = self._gas_target(s)
        ref_ids = {int(r["id"]) for r in refs}
        self.gas = {w: r for w, r in self.gas.items() if r in ref_ids and w not in self.claimed()}

        counts: dict[int, int] = {rid: 0 for rid in ref_ids}
        for wid, rid in list(self.gas.items()):
            if counts[rid] >= per:
                del self.gas[wid]              # back to minerals (_assign_minerals sees a non-mineral order)
                continue
            counts[rid] += 1
        for w in s.workers:
            wid = int(w["id"])
            if wid in self.claimed() or wid in self.gas:
                continue
            if int(w["order"]) in GAS_ORDERS:
                tgt = int(w["order_target"]) if int(w["order_target"]) >= 0 else int(w["target"])
                if tgt in counts and counts[tgt] < per:
                    self.gas[wid] = tgt
                    counts[tgt] += 1

        mineral_pool = [w for w in s.workers
                        if int(w["id"]) not in self.claimed() and int(w["id"]) not in self.gas]
        for r in refs:
            rid = int(r["id"])
            while counts.get(rid, 0) < per and mineral_pool:
                w = mineral_pool.pop(0)
                wid = int(w["id"])
                self.gas[wid] = rid
                self.mineral.pop(wid, None)
                act.gather(w, r)
                counts[rid] = counts.get(rid, 0) + 1

    def _assign_minerals(self, s: State, act: Actions) -> None:
        fields = s.obs.minerals_fields
        if self.local_only:
            fields = _near_depots(s, fields)
        if len(fields) == 0:
            return
        load: dict[int, int] = {int(p["id"]): 0 for p in fields}
        for pid in self.mineral.values():
            if pid in load:
                load[pid] += 1

        busy = self.claimed() | set(self.gas)
        for w in s.workers:
            wid = int(w["id"])
            if wid in busy:
                continue
            assigned = self.mineral.get(wid)
            idle = int(w["order"]) not in MINERAL_ORDERS
            if assigned is not None and assigned in load and not idle:
                continue
            patch = _least_loaded(fields, load, int(w["x"]), int(w["y"]))
            if patch is None:
                continue
            pid = int(patch["id"])
            self.mineral[wid] = pid
            load[pid] = load.get(pid, 0) + 1
            act.gather(w, patch)


def _near_depots(s: State, fields: np.ndarray) -> np.ndarray:
    """Patches within LOCAL_MINERAL_PX of one of our completed resource depots."""
    mine = s.obs.completed(s.obs.my_units)
    if len(mine) == 0 or len(fields) == 0:
        return fields[:0]
    flags = s.game.unit_types["flags"][np.clip(mine["type"], 0, len(s.game.unit_types) - 1)]
    depots = mine[(flags & UnitTypeFlag.ResourceDepot) != 0]
    if len(depots) == 0:
        return fields[:0]
    d2 = (fields["x"][:, None] - depots["x"][None, :]) ** 2 + (fields["y"][:, None] - depots["y"][None, :]) ** 2
    return fields[(d2 < LOCAL_MINERAL_PX ** 2).any(axis=1)]


def _least_loaded(fields: np.ndarray, load: dict[int, int], x: int, y: int) -> Optional[np.void]:
    best = None
    best_key = None
    for p in fields:
        pid = int(p["id"])
        n = load.get(pid, 0)
        if n >= MINERALS_PER_PATCH and best is not None:
            continue
        d = (int(p["x"]) - x) ** 2 + (int(p["y"]) - y) ** 2
        key = (n, d)
        if best_key is None or key < best_key:
            best_key = key
            best = p
    if best is not None:
        return best
    d = (fields["x"] - x) ** 2 + (fields["y"] - y) ** 2
    return fields[int(np.argmin(d))]
