"""Perception: turn a raw `Observation` into a compact `State`.

This is the only place that reads `obs` on the policy's behalf. Keep everything a policy needs
here so that a learned policy can consume `State.as_features()` while a scripted one reads the
named attributes. Nothing in this module issues commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bwbot import GameInfo, Observation, UnitFlag
from bwbot.observation import UnitTypeFlag


@dataclass
class Memory:
    """Facts that persist across frames (the observation itself is per-frame only)."""

    enemy_start: Optional[tuple[int, int]] = None      # tile coords, guessed from start locations
    enemy_buildings_seen: dict[int, tuple[int, int]] = field(default_factory=dict)  # unit id -> pixel pos
    last_attacked_frame: int = -10_000


@dataclass
class State:
    frame: int
    minerals: int
    gas: int
    supply_used: int          # displayed supply (e.g. 9/17)
    supply_total: int
    counts: np.ndarray        # own units per UnitType incl. in production (index = UnitType id)
    completed: np.ndarray     # own completed units per UnitType
    army: np.ndarray          # own completed units that can attack and are not workers/buildings (UNIT_DTYPE rows)
    workers: np.ndarray       # own completed workers
    enemies: np.ndarray       # visible enemy units
    main_tile: tuple[int, int]                # own start location (tiles)
    enemy_start: Optional[tuple[int, int]]    # best guess (tiles)
    under_attack: bool        # something of ours was hit recently
    game: GameInfo
    obs: Observation          # escape hatch for scripted policies; a learned policy should not need it

    @property
    def supply_left(self) -> int:
        return self.supply_total - self.supply_used

    def count(self, unit_type: int) -> int:
        return int(self.counts[int(unit_type)])

    def count_completed(self, unit_type: int) -> int:
        return int(self.completed[int(unit_type)])

    def as_features(self) -> np.ndarray:
        """Flat float vector for a learned policy. Extend as needed; keep it deterministic."""
        return np.concatenate([
            np.array([self.frame / 10_000, self.minerals / 1_000, self.gas / 1_000,
                      self.supply_used / 200, self.supply_total / 200, float(self.under_attack),
                      len(self.army) / 100, len(self.enemies) / 100], dtype=np.float32),
            self.counts.astype(np.float32) / 50,
        ])


def perceive(obs: Observation, game: GameInfo, mem: Memory) -> State:
    mine = obs.completed(obs.my_units)
    types = np.clip(mine["type"], 0, len(game.unit_types) - 1)
    tflags = game.unit_types["flags"][types]
    is_worker = (tflags & UnitTypeFlag.Worker) != 0
    is_army = ((tflags & UnitTypeFlag.CanAttack) != 0) & ~is_worker & ((tflags & UnitTypeFlag.Building) == 0)

    if (obs.my_units["flags"] & UnitFlag.RecentlyAttacked).any():
        mem.last_attacked_frame = obs.frame_count

    enemies = obs.enemy_units
    etypes = np.clip(enemies["type"], 0, len(game.unit_types) - 1)
    for u in enemies[(game.unit_types["flags"][etypes] & UnitTypeFlag.Building) != 0]:
        mem.enemy_buildings_seen[int(u["id"])] = (int(u["x"]), int(u["y"]))

    me = obs.me
    n = len(game.unit_types)
    counts = me.all_unit_count if me.all_unit_count.size >= n else np.zeros(n, dtype=np.int32)
    completed = me.completed_unit_count if me.completed_unit_count.size >= n else np.zeros(n, dtype=np.int32)

    return State(
        frame=obs.frame_count,
        minerals=obs.minerals,
        gas=obs.gas,
        supply_used=obs.supply_used,
        supply_total=obs.supply_total,
        counts=counts,
        completed=completed,
        army=mine[is_army],
        workers=mine[is_worker],
        enemies=enemies,
        main_tile=tuple(game.self_player.start_location),
        enemy_start=mem.enemy_start,
        under_attack=obs.frame_count - mem.last_attacked_frame < 24 * 5,
        game=game,
        obs=obs,
    )
