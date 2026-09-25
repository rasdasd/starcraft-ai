"""Forward economy simulator for build-order search (BOSS-style), from GameInfo costs and times.

State: resources, mineral/gas workers, supply, completed and started counts, producer slots
(the frame each producer is next free), research slots, the builder pipeline, and a heap of
completion events. `when(state, action)` is the earliest frame the action can start given only
what is already in progress; `do(state, action)` fast-forwards to that frame and starts it.

Approximations: constant income per worker (mineral workers saturate at `per_patch` per patch),
new workers mine minerals, each completed refinery takes 3 gas workers, a Terran building takes
one SCV off mining for `travel` + build time, the construction executor places one building at a
time (`builder_gap` frames between building starts), units needing an addon use producers that
have it, and research/upgrades use separate slots from production. Terran and Protoss only (Zerg
larva are not modelled, so larva units are never legal).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bwbot import Order, Race, UnitType as U
from bwbot.observation import UnitFlag

from .techtree import TechTree

FPS = 24
GAS_ORDERS = {int(Order.MoveToGas), int(Order.WaitForGas), int(Order.HarvestGas), int(Order.ReturnGas)}


@dataclass(frozen=True)
class Action:
    kind: str          # "unit" (incl. buildings and addons) | "upgrade" | "tech"
    type_id: int
    level: int = 1

    def __str__(self) -> str:
        return f"{self.kind}:{self.type_id}" + (f"@{self.level}" if self.kind == "upgrade" else "")


@dataclass
class SimState:
    frame: int
    minerals: float
    gas: float
    m_workers: int
    g_workers: int
    supply_used: int                      # BWAPI units (x2)
    supply_total: int
    done: dict[int, int]                  # completed counts per unit type
    started: dict[int, int]               # completed + in progress
    slots: dict[int, list[int]]           # producer type -> free-at frame per completed producer
    research: dict[int, list[int]]        # researching building type -> free-at frames
    upgrades: dict[int, int]              # upgrade -> level done or in progress
    techs: set[int]
    events: list = field(default_factory=list)   # heap of (frame, seq, kind, type_id, level)
    builder_free: int = 0
    patches: int = 8
    seq: int = 0

    def copy(self) -> "SimState":
        return SimState(self.frame, self.minerals, self.gas, self.m_workers, self.g_workers, self.supply_used,
                        self.supply_total, dict(self.done), dict(self.started),
                        {k: list(v) for k, v in self.slots.items()}, {k: list(v) for k, v in self.research.items()},
                        dict(self.upgrades), set(self.techs), list(self.events), self.builder_free, self.patches,
                        self.seq)


class EconSim:
    def __init__(self, tree: TechTree, race: int, m_rate: float = 0.045, g_rate: float = 0.037,
                 per_patch: float = 2.0, travel: int = 96, builder_gap: int = 48) -> None:
        self.tree = tree
        self.race = int(race)
        self.m_rate = m_rate            # minerals per mining worker per frame
        self.g_rate = g_rate
        self.per_patch = per_patch
        self.travel = travel
        self.builder_gap = builder_gap
        self.worker = {int(Race.Terran): int(U.Terran_SCV), int(Race.Protoss): int(U.Protoss_Probe),
                       int(Race.Zerg): int(U.Zerg_Drone)}.get(self.race, int(U.Terran_SCV))
        self.refinery = {int(Race.Terran): int(U.Terran_Refinery), int(Race.Protoss): int(U.Protoss_Assimilator),
                         int(Race.Zerg): int(U.Zerg_Extractor)}.get(self.race, int(U.Terran_Refinery))
        ut = tree.ut
        self.provides = ut["supply_provided"].astype(np.int64)

    # ------------------------------------------------------------------ income
    def income(self, s: SimState) -> tuple[float, float]:
        cap = self.per_patch * s.patches
        m = min(s.m_workers, cap) + 0.3 * max(0.0, min(s.m_workers - cap, s.patches))
        return m * self.m_rate, s.g_workers * self.g_rate

    def advance(self, s: SimState, frame: int) -> None:
        """Move time forward to `frame`, applying completions on the way."""
        while s.events and s.events[0][0] <= frame:
            f = s.events[0][0]
            self._accrue(s, f)
            _, _, kind, t, lvl = heapq.heappop(s.events)
            self._complete(s, kind, t, lvl, f)
        self._accrue(s, frame)

    def _accrue(self, s: SimState, frame: int) -> None:
        dt = frame - s.frame
        if dt <= 0:
            return
        m, g = self.income(s)
        s.minerals += m * dt
        s.gas += g * dt
        s.frame = frame

    def _complete(self, s: SimState, kind: str, t: int, lvl: int, f: int) -> None:
        tree = self.tree
        if kind == "unit":
            s.done[t] = s.done.get(t, 0) + 1
            s.supply_total = min(400, s.supply_total + int(self.provides[t]))
            if t == self.worker:
                s.m_workers += 1
            elif t == self.refinery:
                moved = min(3, s.m_workers)
                s.m_workers -= moved
                s.g_workers += moved
            if tree.is_building(t):
                if tree.is_addon(t):
                    parent = tree.builder(t)
                    pool = s.slots.get(parent, [])
                    if pool:                              # the parent now produces as an addon holder
                        pool.remove(max(pool))
                    s.slots.setdefault(t, []).append(f)
                else:
                    s.slots.setdefault(t, []).append(f)
                s.research.setdefault(t, []).append(f)
            if tree.is_depot(t) and tree.is_building(t):
                s.patches += 8
        elif kind == "builder":
            s.m_workers += 1

    # ------------------------------------------------------------------ actions
    def reqs(self, a: Action) -> list[int]:
        tree = self.tree
        if a.kind == "unit":
            return [r for r in tree.unit_requires(a.type_id) if r != self.worker and r != int(U.Zerg_Larva)]
        if a.kind == "upgrade":
            return tree.upgrade_requires(a.type_id, a.level)
        return tree.tech_requires(a.type_id)

    def cost(self, a: Action) -> tuple[int, int]:
        if a.kind == "unit":
            return self.tree.cost(a.type_id)
        if a.kind == "upgrade":
            return self.tree.upgrade_cost(a.type_id, a.level)
        return self.tree.tech_cost(a.type_id)

    def duration(self, a: Action) -> int:
        if a.kind == "unit":
            return max(1, self.tree.time(a.type_id))
        if a.kind == "upgrade":
            return max(1, self.tree.upgrade_time(a.type_id, a.level))
        i = self.tree.techs[a.type_id]
        return max(1, int(i["research_time"]))

    def _pools(self, s: SimState, a: Action) -> Optional[list[list[int]]]:
        """Producer free-at lists that can make `a` (None: needs no producer slot)."""
        tree = self.tree
        if a.kind != "unit":
            r = self.reqs(a)
            return [s.research.get(r[0], [])] if r else None
        t = a.type_id
        b = tree.builder(t)
        if b == self.worker or b == -1:
            return None
        if tree.is_addon(t):
            return [s.slots.get(b, [])]
        addons = [r for r in tree.unit_requires(t) if tree.is_addon(r) and tree.builder(r) == b]
        if addons:
            return [s.slots.get(addons[0], [])]
        pools = [s.slots.get(b, [])]
        for other, lst in s.slots.items():                 # parents with an addon still produce
            if other != b and tree.is_addon(other) and tree.builder(other) == b:
                pools.append(lst)
        return pools

    def legal(self, s: SimState, a: Action) -> bool:
        """Can happen eventually without other new actions (prereqs started, a producer exists)."""
        for r in self.reqs(a):
            if s.started.get(r, 0) <= 0:
                return False
        m, g = self.cost(a)
        if g > 0 and s.g_workers == 0 and s.gas < g and s.started.get(self.refinery, 0) == 0:
            return False
        if a.kind == "unit":
            need = int(self.tree.ut["supply_required"][a.type_id])
            if need > 0 and s.supply_used + need > self._supply_eventually(s):
                return False
            if a.type_id == int(U.Zerg_Larva):
                return False
        if a.kind == "upgrade" and s.upgrades.get(a.type_id, 0) != a.level - 1:
            return False
        if a.kind == "tech" and a.type_id in s.techs:
            return False
        pools = self._pools(s, a)
        if pools is not None and not any(pools):
            return any(s.started.get(r, 0) > s.done.get(r, 0) for r in self.reqs(a))   # producer on the way
        return True

    def _supply_eventually(self, s: SimState) -> int:
        pending = sum(int(self.provides[e[3]]) for e in s.events if e[2] == "unit")
        return min(400, s.supply_total + pending)

    def when(self, s: SimState, a: Action, limit: int = 24 * 60 * 10) -> Optional[int]:
        """Earliest start frame of `a` (None if not within `limit` frames)."""
        if not self.legal(s, a):
            return None
        t, copied = s, False               # read-only until time has to move forward
        m, g = self.cost(a)
        need = int(self.tree.ut["supply_required"][a.type_id]) if a.kind == "unit" else 0
        end = s.frame + limit
        while True:
            f = t.frame
            ok = all(t.done.get(r, 0) > 0 for r in self.reqs(a))
            if ok and need and t.supply_used + need > t.supply_total:
                ok = False
            pools = self._pools(t, a) if ok else None
            if ok and pools is not None:
                free = min((min(p) for p in pools if p), default=None)
                if free is None:
                    ok = False
                else:
                    f = max(f, free)
            if ok and a.kind == "unit" and self.tree.is_building(a.type_id) and not self.tree.is_addon(a.type_id):
                f = max(f, t.builder_free)
            if ok:
                mi, gi = self.income(t)
                wait_m = 0 if t.minerals >= m else (math.inf if mi <= 0 else (m - t.minerals) / mi)
                wait_g = 0 if t.gas >= g else (math.inf if gi <= 0 else (g - t.gas) / gi)
                f = max(f, t.frame + int(math.ceil(max(wait_m, wait_g))) if math.isfinite(max(wait_m, wait_g))
                        else end + 1)
                nxt = t.events[0][0] if t.events else None
                if nxt is None or f <= nxt:
                    return f if f <= end else None
            nxt = t.events[0][0] if t.events else None
            if nxt is None or nxt > end:
                return None
            if not copied:
                t, copied = s.copy(), True
            self.advance(t, nxt)

    def do(self, s: SimState, a: Action, at: Optional[int] = None) -> Optional[int]:
        """Start `a` at its earliest frame (or `at`); returns the start frame."""
        f = self.when(s, a) if at is None else at
        if f is None:
            return None
        self.advance(s, f)
        m, g = self.cost(a)
        s.minerals -= m
        s.gas -= g
        dur = self.duration(a)
        pools = self._pools(s, a)
        if pools is not None:
            pool = min((p for p in pools if p), key=min)
            i = pool.index(min(pool))
            pool[i] = f + dur
        if a.kind == "unit":
            t = a.type_id
            s.supply_used += int(self.tree.ut["supply_required"][t])
            s.started[t] = s.started.get(t, 0) + 1
            if self.tree.is_building(t) and not self.tree.is_addon(t):
                s.builder_free = f + self.builder_gap
                if self.tree.builder(t) == self.worker:
                    if self.race == int(Race.Terran):
                        s.m_workers = max(0, s.m_workers - 1)
                        self._push(s, f + self.travel + dur, "builder", t, 0)
                    elif self.race == int(Race.Zerg):
                        s.m_workers = max(0, s.m_workers - 1)
                        s.supply_used -= int(self.tree.ut["supply_required"][self.worker])
            self._push(s, f + dur + (self.travel if self.tree.is_building(t) and not self.tree.is_addon(t) else 0),
                       "unit", t, 0)
        elif a.kind == "upgrade":
            s.upgrades[a.type_id] = a.level
            self._push(s, f + dur, "upgrade", a.type_id, a.level)
        else:
            s.techs.add(a.type_id)
            self._push(s, f + dur, "tech", a.type_id, 0)
        return f

    def _push(self, s: SimState, f: int, kind: str, t: int, lvl: int) -> None:
        s.seq += 1
        heapq.heappush(s.events, (int(f), s.seq, kind, int(t), int(lvl)))

    def finish(self, s: SimState) -> int:
        """Frame at which everything in progress is complete."""
        return max((e[0] for e in s.events), default=s.frame)

    # ------------------------------------------------------------------ from the live game
    def from_game(self, bb) -> SimState:
        obs, w, g = bb.obs, bb.world, bb.game
        frame = int(bb.frame)
        mine = obs.my_units
        done: dict[int, int] = {}
        started: dict[int, int] = {}
        slots: dict[int, list[int]] = {}
        research: dict[int, list[int]] = {}
        s = SimState(frame, float(w.minerals), float(w.gas), 0, 0, 2 * int(w.supply_used), 2 * int(w.supply_total),
                     done, started, slots, research, {}, set())
        m_workers = g_workers = 0
        for u in mine:
            t = int(u["type"])
            flags = int(u["flags"])
            started[t] = started.get(t, 0) + 1
            if not flags & int(UnitFlag.Completed):
                rem = max(1, int(u["remaining_build_time"]))
                self._push(s, frame + rem, "unit", t, 0)
                continue
            done[t] = done.get(t, 0) + 1
            if t == self.worker:
                if int(u["order"]) in GAS_ORDERS:
                    g_workers += 1
                elif not flags & int(UnitFlag.Constructing):
                    m_workers += 1
            if self.tree.is_building(t):
                busy = frame + max(int(u["remaining_train_time"]), 0) if int(u["train_queue_count"]) > 0 else frame
                if self.tree.is_addon(t):
                    slots.setdefault(t, []).append(frame)
                else:
                    slots.setdefault(t, []).append(busy)
                r_busy = frame + max(int(u["remaining_research_time"]), int(u["remaining_upgrade_time"]), 0)
                research.setdefault(t, []).append(r_busy)
            # queued trains in progress
            if int(u["train_queue_count"]) > 0 and flags & int(UnitFlag.Completed):
                q = int(u["train_queue_0"])
                if 0 <= q < len(self.tree.ut):
                    started[q] = started.get(q, 0) + 1
                    self._push(s, frame + max(1, int(u["remaining_train_time"])), "unit", q, 0)
        # parents that carry an addon produce from the addon's pool (which takes their busy times)
        for t in list(slots):
            if self.tree.is_addon(t):
                pool = slots.get(self.tree.builder(t), [])
                for i in range(min(len(pool), len(slots[t]))):
                    busiest = max(pool)
                    pool.remove(busiest)
                    slots[t][i] = busiest
        s.m_workers, s.g_workers = m_workers, g_workers
        s.patches = max(8, 8 * len(w.depots))
        me = obs.me
        for u in range(min(len(me.upgrade_level), 70)):
            lvl = int(me.upgrade_level[u]) + (1 if me.is_upgrading.size > u and me.is_upgrading[u] else 0)
            if lvl:
                s.upgrades[u] = lvl
        for t in range(min(len(me.has_researched), 50)):
            if me.has_researched[t] or (me.is_researching.size > t and me.is_researching[t]):
                s.techs.add(t)
        if m_workers and w.income_minerals > 0:
            per = w.income_minerals / (24 * 60) / m_workers
            self.m_rate = float(min(0.06, max(0.03, per)))
        return s
