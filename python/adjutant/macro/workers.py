"""Worker pool: mineral and gas assignment over our bases.

Workers mine only at our completed bases (no long-distance mining). Each base is filled to two per
patch, the least saturated first, and up to three per patch once every base is full; when a base is
over two per patch while another is under, a few workers move per decision. Gas takes three per
completed refinery at our bases, none while gas is banked far beyond minerals. Workers another
component leases (builders, scouts, repairers, worker pulls) are not the pool's; gather commands go
out only for new assignments and idle workers.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from bwbot import UnitFlag
from bwbot.enums import Order
from bwbot.observation import UnitTypeFlag

GAS_ORDERS = {int(Order.MoveToGas), int(Order.WaitForGas), int(Order.HarvestGas), int(Order.ReturnGas)}
MINERAL_ORDERS = {int(Order.MoveToMinerals), int(Order.WaitForMinerals), int(Order.MiningMinerals),
                  int(Order.ReturnMinerals), int(Order.Harvest1), int(Order.Harvest2), int(Order.Harvest3),
                  int(Order.Harvest4)}
GAS_PER_REFINERY = 3
PER_PATCH, MAX_PER_PATCH = 2, 3
TRANSFERS_PER_DECISION = 4
REFINERY_PX = 12 * 32


class WorkerPool:
    def __init__(self, gas_bank: tuple[int, int] = (300, 600)) -> None:
        self.gas_bank = gas_bank
        self.gas_per = GAS_PER_REFINERY
        self.gas: dict[int, int] = {}          # worker id -> refinery id
        self.mineral: dict[int, int] = {}      # worker id -> patch id
        self.miners: dict[int, int] = {}       # base id -> mineral workers (last update)
        self.gassers: dict[int, int] = {}      # base id -> gas workers

    def release(self, worker_id: int) -> None:
        self.gas.pop(int(worker_id), None)
        self.mineral.pop(int(worker_id), None)

    def gas_target(self, minerals: int, gas: int) -> int:
        """Workers per refinery: none while gas is banked far beyond minerals, back to full once it has
        been spent down (hysteresis between the two `gas_bank` levels)."""
        low, high = self.gas_bank
        if gas >= high and gas > 2 * minerals:
            self.gas_per = 0
        elif gas < low:
            self.gas_per = GAS_PER_REFINERY
        return self.gas_per

    # ------------------------------------------------------------------ update
    def update(self, obs, act, workers: np.ndarray, free: set[int], bases, minerals: int, gas: int) -> None:
        """`free`: ids of the workers the pool may use this decision; `bases`: a BaseTracker."""
        pool = [w for w in workers if int(w["id"]) in free]
        ids = {int(w["id"]) for w in pool}
        self.gas = {w: r for w, r in self.gas.items() if w in ids}
        self.mineral = {w: p for w, p in self.mineral.items() if w in ids}
        done = {b.base_id: b for b in bases.bases if b.completed}
        self._gas(obs, act, pool, done, self.gas_target(minerals, gas))
        self._minerals(obs, act, pool, done, bases)

    def _gas(self, obs, act, pool, done, per: int) -> None:
        g = obs.game
        mine = obs.completed(obs.my_units)
        flags = g.unit_types["flags"][np.clip(mine["type"], 0, len(g.unit_types) - 1)] if len(mine) else np.zeros(0)
        refs = [r for r in (mine[(flags & UnitTypeFlag.Refinery) != 0] if len(mine) else [])
                if any((int(r["x"]) - b.center[0]) ** 2 + (int(r["y"]) - b.center[1]) ** 2 <= REFINERY_PX ** 2
                       for b in done.values())]
        ref_ids = {int(r["id"]) for r in refs}
        counts = {rid: 0 for rid in ref_ids}
        for wid, rid in list(self.gas.items()):
            if rid not in counts or counts[rid] >= per:
                del self.gas[wid]              # back to minerals: its order is not a mineral order
                continue
            counts[rid] += 1
        for w in pool:
            wid = int(w["id"])
            if wid in self.gas or int(w["order"]) not in GAS_ORDERS:
                continue
            tgt = int(w["order_target"]) if int(w["order_target"]) >= 0 else int(w["target"])
            if tgt in counts and counts[tgt] < per:
                self.gas[wid] = tgt
                counts[tgt] += 1
        spare = [w for w in pool if int(w["id"]) not in self.gas]
        for r in refs:
            rid = int(r["id"])
            while counts[rid] < per and spare:
                w = min(spare, key=lambda u: (int(u["carry_resource_type"]) != 0,
                                              (int(u["x"]) - int(r["x"])) ** 2 + (int(u["y"]) - int(r["y"])) ** 2))
                spare.remove(w)
                self.gas[int(w["id"])] = rid
                self.mineral.pop(int(w["id"]), None)
                act.gather(w, r)
                counts[rid] += 1
        self.gassers = {}
        for rid, n in counts.items():
            r = next(x for x in refs if int(x["id"]) == rid)
            bid = min(done.values(), key=lambda b: (int(r["x"]) - b.center[0]) ** 2 + (int(r["y"]) - b.center[1]) ** 2).base_id
            self.gassers[bid] = self.gassers.get(bid, 0) + n

    def _minerals(self, obs, act, pool, done, bases) -> None:
        fields = [f for f in obs.minerals_fields if bases.base_of_patch(int(f["id"])) in done]
        self.miners = {bid: 0 for bid in done}
        if not fields:
            self.mineral.clear()
            return
        by_id = {int(f["id"]): f for f in fields}
        base_of = {pid: bases.base_of_patch(pid) for pid in by_id}
        patches: dict[int, list[int]] = {bid: [] for bid in done}
        for pid, bid in base_of.items():
            patches[bid].append(pid)
        load = {pid: 0 for pid in by_id}
        self.mineral = {w: p for w, p in self.mineral.items() if p in by_id}
        for pid in self.mineral.values():
            load[pid] += 1
        base_load = lambda bid: sum(load[p] for p in patches[bid])  # noqa: E731

        def assign(w, bid: Optional[int] = None) -> None:
            x, y = int(w["x"]), int(w["y"])
            if bid is None:
                open_ = [b for b in patches if patches[b] and base_load(b) < PER_PATCH * len(patches[b])]
                cands = open_ or [b for b in patches if patches[b]]
                if not open_:
                    cands = [min(cands, key=lambda b: base_load(b) / len(patches[b]))]
                bid = min(cands, key=lambda b: (done[b].center[0] - x) ** 2 + (done[b].center[1] - y) ** 2)
            pid = min(patches[bid], key=lambda p: (load[p], (int(by_id[p]["x"]) - x) ** 2 + (int(by_id[p]["y"]) - y) ** 2))
            old = self.mineral.get(int(w["id"]))
            if old is not None:
                load[old] -= 1
            self.mineral[int(w["id"])] = pid
            load[pid] += 1
            act.gather(w, by_id[pid])

        for w in pool:
            wid = int(w["id"])
            if wid in self.gas:
                continue
            idle = int(w["order"]) not in MINERAL_ORDERS
            if wid in self.mineral and not idle:
                continue
            if wid in self.mineral and idle:
                act.gather(w, by_id[self.mineral[wid]])
                continue
            assign(w)

        moved = 0
        for src in sorted(patches, key=lambda b: -base_load(b)):
            for dst in sorted(patches, key=base_load):
                if src == dst or not patches[src] or not patches[dst]:
                    continue
                over = base_load(src) - PER_PATCH * len(patches[src])
                under = PER_PATCH * len(patches[dst]) - base_load(dst)
                n = min(over, under, TRANSFERS_PER_DECISION - moved)
                if n <= 0:
                    continue
                movers = [w for w in pool if self.mineral.get(int(w["id"])) in patches[src]]
                movers.sort(key=lambda u: int(u["carry_resource_type"]) != 0)
                for w in movers[:n]:
                    assign(w, dst)
                    moved += 1
        for pid, n in load.items():
            self.miners[base_of[pid]] += n


def pick_worker(obs, workers: np.ndarray, x: int, y: int, allowed: Iterable[int],
                prefer_minerals: Optional[set[int]] = None) -> Optional[np.void]:
    """Nearest allowed worker to (x, y): mining minerals and not carrying first, gas workers last."""
    allowed = set(allowed)
    rows = [w for w in workers if int(w["id"]) in allowed
            and not int(w["flags"]) & int(UnitFlag.Constructing) and int(w["order"]) != int(Order.ConstructingBuilding)]
    if not rows:
        return None
    minerals = prefer_minerals if prefer_minerals is not None else set()

    def key(w):
        on_gas = int(w["order"]) in GAS_ORDERS
        return (on_gas, int(w["carry_resource_type"]) != 0,
                bool(minerals) and int(w["id"]) not in minerals,
                (int(w["x"]) - x) ** 2 + (int(w["y"]) - y) ** 2)
    return min(rows, key=key)
