"""Perception: turn a raw `Observation` into a compact `State`.

This is the only place that reads `obs` on the policy's behalf. Keep everything a policy needs
here so that a learned policy can consume `State.as_features()` while a scripted one reads the
named attributes. Nothing in this module issues commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bwbot import GameInfo, Observation, UnitType
from bwbot.observation import UnitTypeFlag

from .opponent import OpponentSnapshot

# units morphing inside these count as their `build_type` (BWAPI counts them as the egg)
EGGS = (int(UnitType.Zerg_Egg), int(UnitType.Zerg_Lurker_Egg), int(UnitType.Zerg_Cocoon))


@dataclass
class Memory:
    """Facts that persist across frames (the observation itself is per-frame only)."""

    enemy_start: Optional[tuple[int, int]] = None      # tile coords, guessed from start locations
    enemy_buildings_seen: dict[int, tuple[int, int]] = field(default_factory=dict)  # unit id -> pixel pos
    enemy_building_list: list[tuple[int, int, int]] = field(default_factory=list)  # (type, x, y)
    last_attacked_frame: int = -10_000
    opponent: OpponentSnapshot = field(default_factory=OpponentSnapshot)


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
    natural_tile: Optional[tuple[int, int]]   # BWEM natural (tiles)
    main_choke: Optional[tuple[int, int]]     # pixels, choke between main and natural
    enemy_start: Optional[tuple[int, int]]    # best guess (tiles)
    under_attack: bool        # something of ours was hit recently
    enemy_flyers: np.ndarray  # visible enemy units with the Flyer flag
    enemy_buildings: list[tuple[int, int, int]]  # last-known fog buildings (type, x, y)
    upgrade_level: np.ndarray # own upgrade levels, indexed by UpgradeType
    is_upgrading: np.ndarray  # own in-progress upgrades, indexed by UpgradeType
    opponent: OpponentSnapshot
    game: GameInfo
    obs: Observation          # escape hatch for scripted policies; a learned policy should not need it

    @property
    def supply_left(self) -> int:
        return self.supply_total - self.supply_used

    def count(self, unit_type: int) -> int:
        return int(self.counts[int(unit_type)])

    def count_completed(self, unit_type: int) -> int:
        return int(self.completed[int(unit_type)])

    def has_upgrade(self, upgrade_type: int, level: int = 1) -> bool:
        arr = self.upgrade_level
        return arr.size > int(upgrade_type) and int(arr[int(upgrade_type)]) >= level

    def upgrading(self, upgrade_type: int) -> bool:
        arr = self.is_upgrading
        return arr.size > int(upgrade_type) and bool(arr[int(upgrade_type)])

    @property
    def rally_point(self) -> tuple[int, int]:
        if self.main_choke is not None:
            return self.main_choke
        if self.natural_tile is not None:
            return self.natural_tile[0] * 32 + 64, self.natural_tile[1] * 32 + 48
        return self.main_tile[0] * 32 + 64, self.main_tile[1] * 32 + 48

    def as_features(self) -> np.ndarray:
        """Fixed vector for a learned policy. See features.FEATURE_NAMES."""
        from .features import encode
        return encode(self)


def perceive(obs: Observation, game: GameInfo, mem: Memory) -> State:
    mine = obs.completed(obs.my_units)
    types = np.clip(mine["type"], 0, len(game.unit_types) - 1)
    tflags = game.unit_types["flags"][types]
    is_worker = (tflags & UnitTypeFlag.Worker) != 0
    is_army = ((tflags & UnitTypeFlag.CanAttack) != 0) & ~is_worker & ((tflags & UnitTypeFlag.Building) == 0)

    enemies = obs.enemy_units
    etypes = np.clip(enemies["type"], 0, len(game.unit_types) - 1)
    eflags = game.unit_types["flags"][etypes]

    me = obs.me
    n = len(game.unit_types)
    counts = me.all_unit_count if me.all_unit_count.size >= n else np.zeros(n, dtype=np.int32)
    completed = me.completed_unit_count if me.completed_unit_count.size >= n else np.zeros(n, dtype=np.int32)
    own = obs.my_units
    eggs = own[np.isin(own["type"], EGGS) & (own["build_type"] >= 0) & (own["build_type"] < n)]
    if len(eggs):
        counts = counts.copy()
        for bt in eggs["build_type"]:
            counts[int(bt)] += 2 if int(game.unit_types["flags"][int(bt)]) & UnitTypeFlag.TwoUnitsInOneEgg else 1

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
        natural_tile=game.natural.tile if game.natural is not None else None,
        main_choke=game.main_choke.center if game.main_choke is not None else None,
        enemy_start=mem.enemy_start,
        under_attack=obs.frame_count - mem.last_attacked_frame < 24 * 5,
        enemy_flyers=enemies[(eflags & UnitTypeFlag.Flyer) != 0],
        enemy_buildings=list(mem.enemy_building_list),
        upgrade_level=me.upgrade_level,
        is_upgrading=me.is_upgrading,
        opponent=mem.opponent,
        game=game,
        obs=obs,
    )
