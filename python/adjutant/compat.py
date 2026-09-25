"""Bridge to the `mybot` managers, which take a legacy `State`.

Perception stores `perceive(obs, game, Memory(...))` without enemy memory in `world.state`; this
merges in what Belief knows (enemy start, last-known buildings, opponent snapshot) once per
decision. Keeping the merge here avoids a Perception <-> Belief dependency cycle.
"""
from __future__ import annotations

import dataclasses

from blackboard import Blackboard
from mybot.opponent import OpponentSnapshot
from mybot.state import State


def state(bb: Blackboard) -> State:
    s = bb.cache.get("state")
    if s is not None:
        return s
    base: State = bb.world.state
    if base is None:
        raise RuntimeError("compat.state() before Perception ran")
    b = bb.belief
    snap = b.memory if isinstance(b.memory, OpponentSnapshot) else snapshot_from_belief(bb)
    s = dataclasses.replace(base, enemy_start=b.enemy_start, enemy_buildings=list(b.buildings), opponent=snap)
    bb.cache["state"] = s
    return s


def snapshot_from_belief(bb: Blackboard) -> OpponentSnapshot:
    from mybot.opponent import OPENING_NAMES
    b = bb.belief
    counts = {int(t): int(round(n)) for t, n in b.counts.items()}
    opening = OPENING_NAMES.index(b.opening) if b.opening in OPENING_NAMES else 0
    return OpponentSnapshot(race=int(b.enemy_race), opening=opening, air_units=int(round(b.air)),
                            ground_army=int(round(b.army_supply)), proxy=bool(b.proxy), counts=counts,
                            buildings=len(b.buildings))


def worker_manager(bb: Blackboard):
    """The shared WorkerManager (created on first use; Economy owns its update)."""
    wm = bb.services.get("workers")
    if wm is None:
        from mybot.workers import WorkerManager
        wm = bb.services["workers"] = WorkerManager(local_only=bool(bb.config.get("local_mining", False)))
    return wm
