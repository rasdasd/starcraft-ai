"""Arbiters for contended resources: units (leases), money (budget), requests, and commands.

Commands: each component gets a `ScopedActions` that shares the frame's command lists but tags every
unit command with the component's owner name and priority, and drops commands for units another
component leased at equal or higher priority. `CommandBus.flush` then keeps only the winning owner's
commands per unit and sorts by priority, so the runner's APM trim cuts the least important first.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from bwbot.commands import Actions, _uid

from .component import Priority

log = logging.getLogger("blackboard.arbiter")


# ---------------------------------------------------------------------------- requests
@dataclass
class Request:
    kind: str                         # "production" | "units" | "posture"
    source: str
    priority: int = Priority.NORMAL
    type_id: int = -1                 # production: unit/upgrade/tech id; units: unit type filter or -1
    item: str = "build"               # production: build | train | addon | upgrade | research
    count: int = 1
    x: int = 0
    y: int = 0
    purpose: str = ""
    near: Optional[tuple[int, int]] = None
    exact: bool = False
    expires: int = 0

    @property
    def key(self) -> tuple:
        return self.kind, self.source, self.item, self.type_id, self.purpose


class RequestQueue:
    """Multi-writer queue. Posting the same key again refreshes it; requests expire after `ttl`."""

    def __init__(self) -> None:
        self._items: dict[tuple, Request] = {}

    def post(self, req: Request, frame: int, ttl: int = 48) -> None:
        req.expires = frame + max(1, ttl)
        self._items[req.key] = req

    def withdraw(self, source: str, kind: Optional[str] = None) -> None:
        for k in [k for k, r in self._items.items() if r.source == source and (kind is None or r.kind == kind)]:
            del self._items[k]

    def expire(self, frame: int) -> None:
        for k in [k for k, r in self._items.items() if r.expires < frame]:
            del self._items[k]

    def active(self, kind: str) -> list[Request]:
        return sorted((r for r in self._items.values() if r.kind == kind), key=lambda r: -r.priority)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------- unit leases
@dataclass
class Lease:
    owner: str
    priority: int
    purpose: str
    since: int


class UnitLeases:
    """Explicit unit ownership. Units without a lease belong to their category's default owner."""

    def __init__(self) -> None:
        self._leases: dict[int, Lease] = {}

    def lease(self, uid: int, owner: str, priority: int, purpose: str = "", frame: int = 0) -> bool:
        uid = int(uid)
        cur = self._leases.get(uid)
        if cur is not None and cur.owner != owner and cur.priority >= priority:
            return False
        if cur is None or cur.owner != owner:
            self._leases[uid] = Lease(owner, int(priority), purpose, frame)
        else:
            cur.priority, cur.purpose = int(priority), purpose
        return True

    def release(self, uid: int, owner: Optional[str] = None) -> None:
        cur = self._leases.get(int(uid))
        if cur is not None and (owner is None or cur.owner == owner):
            del self._leases[int(uid)]

    def release_owner(self, owner: str) -> None:
        for uid in [u for u, l in self._leases.items() if l.owner == owner]:
            del self._leases[uid]

    def owner(self, uid: int) -> Optional[str]:
        cur = self._leases.get(int(uid))
        return cur.owner if cur is not None else None

    def get(self, uid: int) -> Optional[Lease]:
        return self._leases.get(int(uid))

    def owned(self, owner: str) -> set[int]:
        return {u for u, l in self._leases.items() if l.owner == owner}

    def leased(self, exclude_owner: Optional[str] = None) -> set[int]:
        return {u for u, l in self._leases.items() if l.owner != exclude_owner}

    def allowed(self, uid: int, owner: str, priority: int) -> bool:
        cur = self._leases.get(int(uid))
        return cur is None or cur.owner == owner or priority > cur.priority

    def sync(self, alive: Iterable[int]) -> None:
        alive = set(int(a) for a in alive)
        for uid in [u for u in self._leases if u not in alive]:
            del self._leases[uid]

    def __len__(self) -> int:
        return len(self._leases)


# ---------------------------------------------------------------------------- budget
class Budget:
    """Per-decision money/supply ledger with persistent reservations.

    A reservation blocks spending by anything of lower priority; `spend` draws from the pool left
    this decision. Executors spend in plan order, which already is priority order.
    """

    def __init__(self) -> None:
        self.minerals = 0
        self.gas = 0
        self.supply = 0
        self.reservations: dict[str, tuple[int, int, int, int]] = {}   # key -> (m, g, s, priority)

    def reset(self, minerals: int, gas: int, supply_left: int) -> None:
        self.minerals, self.gas, self.supply = int(minerals), int(gas), int(supply_left)

    def reserve(self, key: str, minerals: int, gas: int, supply: int = 0, priority: int = Priority.NORMAL) -> None:
        self.reservations[key] = (int(minerals), int(gas), int(supply), int(priority))

    def release(self, key: str) -> None:
        self.reservations.pop(key, None)

    def available(self, priority: int = 0) -> tuple[int, int, int]:
        m, g, s = self.minerals, self.gas, self.supply
        for rm, rg, rs, rp in self.reservations.values():
            if rp > priority:
                m, g, s = m - rm, g - rg, s - rs
        return m, g, s

    def can_afford(self, minerals: int, gas: int, supply: int = 0, priority: int = 0) -> bool:
        m, g, s = self.available(priority)
        return m >= minerals and g >= gas and s >= supply

    def spend(self, minerals: int, gas: int, supply: int = 0) -> None:
        self.minerals -= int(minerals)
        self.gas -= int(gas)
        self.supply -= int(supply)


# ---------------------------------------------------------------------------- commands
class ScopedActions(Actions):
    """`Actions` view for one component: shares the frame's lists, tags priority, respects leases."""

    def __init__(self, root: Actions, bus: "CommandBus", owner: str, priority: int) -> None:
        super().__init__(frame_count=root.frame_count, unit_cmds=root.unit_cmds, game_cmds=root.game_cmds,
                         draws=root.draws, place_qs=root.place_qs)
        self._bus = bus
        self._owner = owner
        self._priority = int(priority)

    def command(self, unit, ctype, target=None, x=0, y=0, extra=0, queued=False) -> None:
        uid = _uid(unit)
        if not self._bus.leases.allowed(uid, self._owner, self._priority):
            self._bus.note_drop(self._owner, uid)
            return
        super().command(unit, ctype, target, x, y, extra, queued)
        c = self.unit_cmds[-1]
        c.prio = self._priority
        c.owner = self._owner
        c.seq = next(self._bus.seq)


class CommandBus:
    def __init__(self, leases: UnitLeases) -> None:
        self.leases = leases
        self.seq = itertools.count()
        self.dropped = 0
        self.conflicts = 0
        self._drop_log: set[tuple[str, int]] = set()

    def scope(self, root: Actions, owner: str, priority: int) -> ScopedActions:
        return ScopedActions(root, self, owner, priority)

    def note_drop(self, owner: str, uid: int) -> None:
        self.dropped += 1
        key = (owner, uid)
        if key not in self._drop_log and len(self._drop_log) < 200:
            self._drop_log.add(key)
            log.info("%s: command for #%d dropped (leased by %s)", owner, uid, self.leases.owner(uid))

    def flush(self, act: Actions) -> None:
        cmds = act.unit_cmds
        if not cmds:
            return
        winner: dict[int, tuple[int, str]] = {}
        for c in cmds:
            p = getattr(c, "prio", int(Priority.NORMAL))
            o = getattr(c, "owner", "")
            cur = winner.get(c.unit)
            if cur is None or p > cur[0]:
                winner[c.unit] = (p, o)
        keep = [c for c in cmds if winner[c.unit][1] == getattr(c, "owner", "")]
        self.conflicts += len(cmds) - len(keep)
        keep.sort(key=lambda c: -getattr(c, "prio", int(Priority.NORMAL)))
        cmds[:] = keep
