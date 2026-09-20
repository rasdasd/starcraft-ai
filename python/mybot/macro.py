"""Low-level helpers for executing intents: where to put a building, which worker builds it.

Deliberately simple (spiral search on the buildable grid, nearest free worker). Replace with
proper base/region analysis when placement starts to matter. All coordinates: tiles unless the
name says pixels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from bwbot import GameInfo, Observation, UnitFlag, UnitType
from bwbot.enums import Order
from bwbot.observation import UnitTypeFlag

ADDON_TYPES = {UnitType.Terran_Barracks, UnitType.Terran_Factory, UnitType.Terran_Starport,
               UnitType.Terran_Science_Facility, UnitType.Terran_Command_Center}


def occupancy(obs: Observation) -> np.ndarray:
    """(h, w) bool grid of tiles covered by buildings/resources (resources padded to keep mining lanes clear)."""
    g = obs.game
    occ = np.zeros((g.map_height, g.map_width), dtype=bool)
    types = np.clip(obs.units["type"], 0, len(g.unit_types) - 1)
    is_bld = (g.unit_types["flags"][types] & (UnitTypeFlag.Building | UnitTypeFlag.ResourceContainer)) != 0
    for u, t in zip(obs.units[is_bld], types[is_bld]):
        tw, th = int(g.unit_types["tile_width"][t]), int(g.unit_types["tile_height"][t])
        tx = int(u["x"]) // 32 - tw // 2
        ty = int(u["y"]) // 32 - th // 2
        pad = 3 if (g.unit_types["flags"][t] & UnitTypeFlag.ResourceContainer) else 0
        occ[max(0, ty - pad):ty + th + pad, max(0, tx - pad):tx + tw + pad] = True
    return occ


def spiral(cx: int, cy: int, r_min: int, r_max: int):
    """Deterministic ring-by-ring walk around (cx, cy)."""
    for r in range(r_min, r_max + 1):
        for dx in range(-r, r + 1):
            yield cx + dx, cy - r
            yield cx + dx, cy + r
        for dy in range(-r + 1, r):
            yield cx - r, cy + dy
            yield cx + r, cy + dy


def find_build_tile(obs: Observation, building: int, near_tile: tuple[int, int],
                    skip: frozenset[tuple[int, int]] = frozenset(),
                    occ: Optional[np.ndarray] = None) -> Optional[tuple[int, int]]:
    """Top-left tile for `building` near `near_tile`, with a 1-tile margin (plus addon space).

    Candidates must be buildable, unoccupied, on the same ground level as `near_tile` (stay on
    the main plateau) and explored (StarCraft refuses to place structures in the black), and
    not in `skip` (tiles that failed before).
    """
    g: GameInfo = obs.game
    tw = int(g.unit_types["tile_width"][building])
    th = int(g.unit_types["tile_height"][building])
    extra_w = 2 if building in ADDON_TYPES else 0
    if occ is None:
        occ = occupancy(obs)
    level = g.ground_height[near_tile[1], near_tile[0]] // 2     # BWAPI height: 0/1 low, 2/3 high, 4/5 very high
    explored = obs.explored if obs.tiles.size else None
    for tx, ty in spiral(near_tile[0], near_tile[1], 3, 18):
        if (tx, ty) in skip:
            continue
        if tx < 1 or ty < 1 or tx + tw + extra_w + 1 >= g.map_width or ty + th + 1 >= g.map_height:
            continue
        sl = (slice(ty - 1, ty + th + 1), slice(tx - 1, tx + tw + extra_w + 1))
        if not g.buildable[sl].all() or occ[sl].any():
            continue
        if (g.ground_height[sl] // 2 != level).any():
            continue
        if explored is not None and not explored[sl].all():
            continue
        return tx, ty
    return None


def pick_builder(obs: Observation, workers: np.ndarray, tile: tuple[int, int]) -> Optional[np.void]:
    """Nearest worker that is not already constructing or carrying resources."""
    busy = ((workers["flags"] & UnitFlag.Constructing) != 0) | (workers["order"] == Order.ConstructingBuilding)
    free = workers[~busy & (workers["carry_resource_type"] == 0)]
    if len(free) == 0:
        free = workers[~busy]
    return obs.nearest(free, tile[0] * 32, tile[1] * 32)
