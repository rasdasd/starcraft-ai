"""Building placement: reserved tiles, mineral-line exclusion, geyser refineries.

A `Placer` is the stateful object BuildingManager uses. Free functions remain for
one-shot searches (tests, examples). All coordinates are tiles unless the name says pixels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from bwbot import GameInfo, Observation, UnitFlag, UnitType
from bwbot.enums import Order
from bwbot.observation import UnitTypeFlag

ADDON_TYPES = {UnitType.Terran_Barracks, UnitType.Terran_Factory, UnitType.Terran_Starport,
               UnitType.Terran_Science_Facility, UnitType.Terran_Command_Center}
RESOURCE_PAD = 3
SEARCH_R_MIN = 2
SEARCH_R_MAX = 32


def occupancy(obs: Observation) -> np.ndarray:
    """(h, w) bool grid of tiles covered by buildings/resources (resources padded)."""
    g = obs.game
    occ = np.zeros((g.map_height, g.map_width), dtype=bool)
    types = np.clip(obs.units["type"], 0, len(g.unit_types) - 1)
    is_bld = (g.unit_types["flags"][types] & (UnitTypeFlag.Building | UnitTypeFlag.ResourceContainer)) != 0
    for u, t in zip(obs.units[is_bld], types[is_bld]):
        tw, th = int(g.unit_types["tile_width"][t]), int(g.unit_types["tile_height"][t])
        tx = int(u["x"]) // 32 - tw // 2
        ty = int(u["y"]) // 32 - th // 2
        pad = RESOURCE_PAD if (g.unit_types["flags"][t] & UnitTypeFlag.ResourceContainer) else 0
        occ[max(0, ty - pad):ty + th + pad, max(0, tx - pad):tx + tw + pad] = True
    return occ


def block_mineral_line(obs: Observation, occ: np.ndarray) -> None:
    """Mark the lane between each resource depot and its nearby patches as unbuildable."""
    g = obs.game
    flags = g.unit_types["flags"][np.clip(obs.units["type"], 0, len(g.unit_types) - 1)]
    depots = obs.units[(obs.units["player"] == obs.self_id)
                       & ((flags & UnitTypeFlag.ResourceDepot) != 0)]
    fields = obs.minerals_fields
    if len(depots) == 0 or len(fields) == 0:
        return
    for d in depots:
        dx, dy = int(d["x"]), int(d["y"])
        near = fields[(fields["x"] - dx) ** 2 + (fields["y"] - dy) ** 2 < (12 * 32) ** 2]
        if len(near) == 0:
            continue
        mx, my = int(near["x"].mean()), int(near["y"].mean())
        x0, x1 = sorted((dx // 32, mx // 32))
        y0, y1 = sorted((dy // 32, my // 32))
        occ[max(0, y0 - 2):min(g.map_height, y1 + 3), max(0, x0 - 2):min(g.map_width, x1 + 3)] = True


def spiral(cx: int, cy: int, r_min: int, r_max: int):
    """Deterministic ring-by-ring walk around (cx, cy)."""
    for r in range(r_min, r_max + 1):
        for dx in range(-r, r + 1):
            yield cx + dx, cy - r
            yield cx + dx, cy + r
        for dy in range(-r + 1, r):
            yield cx - r, cy + dy
            yield cx + r, cy + dy


def footprint(game: GameInfo, building: int, tile: tuple[int, int]) -> list[tuple[int, int]]:
    """Tiles occupied by `building` at `tile`, including addon room."""
    tw = int(game.unit_types["tile_width"][building])
    th = int(game.unit_types["tile_height"][building])
    extra_w = 2 if building in ADDON_TYPES else 0
    tx, ty = tile
    return [(tx + x, ty + y) for y in range(th) for x in range(tw + extra_w)]


def unit_tile(game: GameInfo, u: np.void) -> tuple[int, int]:
    """Top-left tile of a unit from its pixel center and type dimensions."""
    t = int(u["type"])
    tw = int(game.unit_types["tile_width"][t])
    th = int(game.unit_types["tile_height"][t])
    return int(u["x"]) // 32 - tw // 2, int(u["y"]) // 32 - th // 2


def find_build_tile(obs: Observation, building: int, near_tile: tuple[int, int],
                    skip: frozenset[tuple[int, int]] = frozenset(),
                    occ: Optional[np.ndarray] = None,
                    reserved: frozenset[tuple[int, int]] = frozenset()) -> Optional[tuple[int, int]]:
    """Top-left tile for `building` near `near_tile`, with a 1-tile margin (plus addon space)."""
    g: GameInfo = obs.game
    tw = int(g.unit_types["tile_width"][building])
    th = int(g.unit_types["tile_height"][building])
    extra_w = 2 if building in ADDON_TYPES else 0
    if occ is None:
        occ = occupancy(obs)
        block_mineral_line(obs, occ)
    if reserved:
        for rx, ry in reserved:
            if 0 <= ry < occ.shape[0] and 0 <= rx < occ.shape[1]:
                occ[ry, rx] = True
    level = g.ground_height[near_tile[1], near_tile[0]] // 2
    explored = obs.explored if obs.tiles.size else None
    for tx, ty in spiral(near_tile[0], near_tile[1], SEARCH_R_MIN, SEARCH_R_MAX):
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


def find_refinery_tile(obs: Observation, near_tile: tuple[int, int],
                       skip: frozenset[tuple[int, int]] = frozenset()) -> Optional[tuple[int, int]]:
    """Top-left tile of the nearest unclaimed geyser. Refineries must be placed on a geyser."""
    geysers = obs.geysers
    if len(geysers) == 0:
        return None
    px, py = near_tile[0] * 32, near_tile[1] * 32
    order = np.argsort((geysers["x"] - px) ** 2 + (geysers["y"] - py) ** 2)
    for u in geysers[order]:
        tile = unit_tile(obs.game, u)
        if tile not in skip:
            return tile
    return None


def pick_builder(obs: Observation, workers: np.ndarray, tile: tuple[int, int]) -> Optional[np.void]:
    """Nearest worker that is not already constructing or carrying resources."""
    busy = ((workers["flags"] & UnitFlag.Constructing) != 0) | (workers["order"] == Order.ConstructingBuilding)
    free = workers[~busy & (workers["carry_resource_type"] == 0)]
    if len(free) == 0:
        free = workers[~busy]
    return obs.nearest(free, tile[0] * 32, tile[1] * 32)


class Placer:
    """Stateful reserved-tile grid used by BuildingManager across frames."""

    def __init__(self) -> None:
        self.reserved: set[tuple[int, int]] = set()
        self.failed: set[tuple[int, int]] = set()

    def reset(self) -> None:
        self.reserved.clear()
        self.failed.clear()

    def reserve(self, game: GameInfo, building: int, tile: tuple[int, int]) -> None:
        self.reserved.update(footprint(game, building, tile))

    def release(self, game: GameInfo, building: int, tile: tuple[int, int]) -> None:
        self.reserved.difference_update(footprint(game, building, tile))

    def fail(self, tile: tuple[int, int], radius: int = 0) -> None:
        """Skip `tile` as a top-left from now on (and the tiles within `radius`: whatever blocks a
        builder standing at the site, e.g. sieged tanks, usually covers the neighbours too)."""
        x, y = tile
        self.failed.update((x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1))

    def find(self, obs: Observation, building: int, near_tile: tuple[int, int]) -> Optional[tuple[int, int]]:
        skip = frozenset(self.failed)
        if obs.game.type_flags(building) & UnitTypeFlag.Refinery:
            return find_refinery_tile(obs, near_tile, skip=skip)
        return find_build_tile(obs, building, near_tile, skip=skip, reserved=frozenset(self.reserved))
