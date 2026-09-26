"""Building placement: reserved tiles, mineral-line exclusion, free base sites, geysers at our bases.

All coordinates are tiles unless the name says pixels.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from bwbot import GameInfo, Observation, UnitType as U
from bwbot.observation import TileFlag, UnitTypeFlag

ADDON_PARENTS = {int(U.Terran_Factory), int(U.Terran_Starport), int(U.Terran_Science_Facility),
                 int(U.Terran_Command_Center)}
HALL_W, HALL_H = 4, 3
PSI_HALF_W, PSI_HALF_H = 7.5, 4.5        # pylon field, slightly shrunk so BWAPI's edge cases are skipped
RESOURCE_PAD = 3
GEYSER_RANGE_PX = 12 * 32                 # a geyser this close to one of our halls is ours to take
SEARCH_R_MIN, SEARCH_R_MAX = 2, 32


def unit_tile(game: GameInfo, u) -> tuple[int, int]:
    """Top-left tile of a unit from its pixel centre and type dimensions."""
    t = int(u["type"])
    return (int(u["x"]) // 32 - int(game.unit_types["tile_width"][t]) // 2,
            int(u["y"]) // 32 - int(game.unit_types["tile_height"][t]) // 2)


def footprint(game: GameInfo, building: int, tile: tuple[int, int]) -> list[tuple[int, int]]:
    """Tiles `building` covers at `tile`, including addon room."""
    tw = int(game.unit_types["tile_width"][building]) + (2 if int(building) in ADDON_PARENTS else 0)
    th = int(game.unit_types["tile_height"][building])
    return [(tile[0] + x, tile[1] + y) for y in range(th) for x in range(tw)]


def occupancy(obs: Observation, keep_free: Iterable[tuple[int, int]] = ()) -> np.ndarray:
    """(h, w) grid of tiles we must not build on: buildings, padded resources, the lane between each
    of our depots and its minerals, and the hall site of every base in `keep_free`."""
    g = obs.game
    occ = np.zeros((g.map_height, g.map_width), dtype=bool)
    types = np.clip(obs.units["type"], 0, len(g.unit_types) - 1)
    flags = g.unit_types["flags"][types]
    solid = (flags & (UnitTypeFlag.Building | UnitTypeFlag.ResourceContainer)) != 0
    for u, t in zip(obs.units[solid], types[solid]):
        tw, th = int(g.unit_types["tile_width"][t]), int(g.unit_types["tile_height"][t])
        tx, ty = int(u["x"]) // 32 - tw // 2, int(u["y"]) // 32 - th // 2
        pad = RESOURCE_PAD if g.unit_types["flags"][t] & UnitTypeFlag.ResourceContainer else 0
        occ[max(0, ty - pad):ty + th + pad, max(0, tx - pad):tx + tw + pad] = True
    depots = obs.units[(obs.units["player"] == obs.self_id) & ((flags & UnitTypeFlag.ResourceDepot) != 0)]
    fields = obs.minerals_fields
    for d in depots if len(fields) else ():
        dx, dy = int(d["x"]), int(d["y"])
        near = fields[(fields["x"] - dx) ** 2 + (fields["y"] - dy) ** 2 < (12 * 32) ** 2]
        if len(near):
            x0, x1 = sorted((dx // 32, int(near["x"].mean()) // 32))
            y0, y1 = sorted((dy // 32, int(near["y"].mean()) // 32))
            occ[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = True
    for tx, ty in keep_free:
        occ[max(0, ty - 1):ty + HALL_H + 1, max(0, tx - 1):tx + HALL_W + 1] = True
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


def powered(pylons, cx: float, cy: float) -> bool:
    """A building centred at tile (cx, cy) is inside a pylon's psi field."""
    return any(((cx - px) / PSI_HALF_W) ** 2 + ((cy - py) / PSI_HALF_H) ** 2 <= 1.0 for px, py in pylons)


def find_build_tile(obs: Observation, building: int, near_tile: tuple[int, int], occ: np.ndarray,
                    skip: frozenset = frozenset()) -> Optional[tuple[int, int]]:
    """Top-left tile for `building` near `near_tile` with a one-tile margin (plus addon room), on
    the same ground level, explored, on creep / in pylon power when the type needs it."""
    g = obs.game
    tw = int(g.unit_types["tile_width"][building])
    th = int(g.unit_types["tile_height"][building])
    extra_w = 2 if int(building) in ADDON_PARENTS else 0
    nx = min(max(near_tile[0], 0), g.map_width - 1)
    ny = min(max(near_tile[1], 0), g.map_height - 1)
    level = g.ground_height[ny, nx] // 2
    explored = obs.explored if obs.tiles.size else None
    tflags = g.type_flags(building)
    creep = (obs.tiles & TileFlag.Creep) != 0 if tflags & UnitTypeFlag.RequiresCreep and obs.tiles.size else None
    pylons = None
    if tflags & UnitTypeFlag.RequiresPsi:
        pylons = [(int(p["x"]) / 32, int(p["y"]) / 32) for p in obs.my_completed(U.Protoss_Pylon)]
        if not pylons:
            return None
    for tx, ty in spiral(nx, ny, SEARCH_R_MIN, SEARCH_R_MAX):
        if (tx, ty) in skip:
            continue
        if tx < 1 or ty < 1 or tx + tw + extra_w + 1 >= g.map_width or ty + th + 1 >= g.map_height:
            continue
        sl = (slice(ty - 1, ty + th + 1), slice(tx - 1, tx + tw + extra_w + 1))
        if not g.buildable[sl].all() or occ[sl].any() or (g.ground_height[sl] // 2 != level).any():
            continue
        if explored is not None and not explored[sl].all():
            continue
        if creep is not None and not creep[ty:ty + th, tx:tx + tw].all():
            continue
        if pylons is not None and not powered(pylons, tx + tw / 2, ty + th / 2):
            continue
        return tx, ty
    return None


class Placer:
    """Tiles held for building jobs that have not placed yet, and tiles that failed."""

    def __init__(self) -> None:
        self.reserved: set[tuple[int, int]] = set()
        self.failed: set[tuple[int, int]] = set()

    def reserve(self, game: GameInfo, building: int, tile: tuple[int, int]) -> None:
        self.reserved.update(footprint(game, building, tile))

    def release(self, game: GameInfo, building: int, tile: tuple[int, int]) -> None:
        self.reserved.difference_update(footprint(game, building, tile))

    def fail(self, tile: tuple[int, int], radius: int = 0) -> None:
        """Skip `tile` as a top-left from now on, and its neighbours within `radius` (whatever blocks
        a builder standing at the site, e.g. sieged tanks, usually covers them too)."""
        x, y = tile
        self.failed.update((x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1))

    def find(self, obs: Observation, building: int, near_tile: tuple[int, int],
             keep_free: Iterable[tuple[int, int]] = (), halls: Iterable[tuple[int, int]] = ()) -> Optional[tuple[int, int]]:
        """A tile for `building`: a free geyser next to one of `halls` (pixels) for a refinery, else a
        spiral search near `near_tile` that leaves `keep_free` base sites open."""
        if obs.game.type_flags(building) & UnitTypeFlag.Refinery:
            return self._geyser(obs, list(halls), near_tile)
        occ = occupancy(obs, keep_free)
        for rx, ry in self.reserved:
            if 0 <= ry < occ.shape[0] and 0 <= rx < occ.shape[1]:
                occ[ry, rx] = True
        return find_build_tile(obs, building, near_tile, occ, skip=frozenset(self.failed))

    def _geyser(self, obs: Observation, halls: list[tuple[int, int]], near_tile) -> Optional[tuple[int, int]]:
        geysers = obs.geysers
        if len(geysers) == 0 or not halls:
            return None
        best, best_d = None, None
        for u in geysers:
            tile = unit_tile(obs.game, u)
            if tile in self.failed or any(p in self.reserved for p in footprint(obs.game, int(u["type"]), tile)):
                continue
            d = min((int(u["x"]) - hx) ** 2 + (int(u["y"]) - hy) ** 2 for hx, hy in halls)
            if d > GEYSER_RANGE_PX ** 2:
                continue
            d += ((tile[0] - near_tile[0]) ** 2 + (tile[1] - near_tile[1]) ** 2) * 32 * 32
            if best_d is None or d < best_d:
                best, best_d = tile, d
        return best
