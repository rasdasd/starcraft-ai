"""A Barracks + Supply Depot pair at the top of the main ramp, touching, with a gap a Marine fits
through and a Zealot does not: zealots walk around the pair while marines step through the gap.

Which side each building goes on comes from the unit type dimensions (pixel extents inside the tile
footprint), not a fixed layout: with BWAPI's numbers a Barracks left of a Depot leaves 7 + 10 = 17 px
(Marine 17 wide, Zealot 23), the other way round 9 + 16 = 25 px. The pair must leave the ramp open
for bigger units (SCVs and tanks still have to get out), checked on the walk grid.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bwbot import GameInfo, UnitType as U

log = logging.getLogger("adjutant.macro")

SEARCH_R = 8                # tiles around the anchor for the left building's top-left
ANCHOR_IN = 3               # anchor: this many tiles from the ramp toward the main hall
WINDOW = 14                 # tiles around the choke checked for a way around
CLEAR_WALK = 4              # walk tiles (32 px) a unit needs to get around: an SCV / tank, not a Marine


@dataclass
class Wall:
    tiles: dict[int, tuple[int, int]] = field(default_factory=dict)    # unit type -> top-left tile
    gap: int = 0                                                          # pixels between the pair

    def tile(self, t: int) -> Optional[tuple[int, int]]:
        return self.tiles.get(int(t))


def _dims(game: GameInfo, t: int) -> tuple[int, int, int, int, int, int]:
    ut = game.unit_types
    t = int(t)
    return (int(ut["tile_width"][t]), int(ut["tile_height"][t]), int(ut["dimension_left"][t]),
            int(ut["dimension_up"][t]), int(ut["dimension_right"][t]), int(ut["dimension_down"][t]))


def box(game: GameInfo, t: int, tile: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1), inclusive, a building occupies at top-left `tile`."""
    tw, th, left, up, right, down = _dims(game, t)
    cx, cy = tile[0] * 32 + tw * 16, tile[1] * 32 + th * 16
    return cx - left, cy - up, cx + right, cy + down


def side_gap(game: GameInfo, left_t: int, right_t: int) -> int:
    """Free pixels between `left_t` and `right_t` placed in adjacent tile columns."""
    lw = _dims(game, left_t)[0]
    return box(game, right_t, (lw, 0))[0] - box(game, left_t, (0, 0))[2] - 1


def width(game: GameInfo, t: int) -> int:
    d = _dims(game, t)
    return d[2] + d[4] + 1


def arrangement(game: GameInfo, a: int, b: int, fits: int, blocked: int) -> Optional[tuple[int, int, int]]:
    """(left, right, gap) of `a` and `b` side by side whose gap lets `fits` walk through vertically
    and not `blocked`; None if neither order does."""
    lo, hi = width(game, fits), width(game, blocked)
    for left, right in ((a, b), (b, a)):
        gap = side_gap(game, left, right)
        if lo <= gap < hi:
            return int(left), int(right), gap
    return None


def plan_wall(game: GameInfo, occupied: Optional[np.ndarray] = None) -> Optional[Wall]:
    """Barracks + Depot pair near the top of our main ramp, or None (no ramp, no room, no layout)."""
    choke, main, nat = game.main_choke, game.main, game.natural
    if choke is None or main is None or nat is None:
        return None
    lay = arrangement(game, U.Terran_Barracks, U.Terran_Supply_Depot, U.Terran_Marine, U.Protoss_Zealot)
    if lay is None:
        return None
    left, right, gap = lay
    lw, lh = _dims(game, left)[:2]
    rw, rh = _dims(game, right)[:2]
    cx, cy = choke.center
    mx, my = main.center
    d = math.hypot(mx - cx, my - cy) or 1.0
    ux, uy = (mx - cx) / d, (my - cy) / d
    ax, ay = int((cx + ux * ANCHOR_IN * 32) // 32), int((cy + uy * ANCHOR_IN * 32) // 32)
    level = int(game.ground_height[main.tile[1], main.tile[0]]) // 2
    inside = (cx + ux * (ANCHOR_IN + 6) * 32, cy + uy * (ANCHOR_IN + 6) * 32)
    nx, ny = nat.center
    dn = math.hypot(nx - cx, ny - cy) or 1.0
    outside = (cx + (nx - cx) / dn * 4 * 32, cy + (ny - cy) / dn * 4 * 32)

    def ok(t: int, tile: tuple[int, int]) -> bool:
        tw, th = _dims(game, t)[:2]
        x, y = tile
        if x < 1 or y < 1 or x + tw >= game.map_width or y + th >= game.map_height:
            return False
        sl = (slice(y, y + th), slice(x, x + tw))
        if not game.buildable[sl].all() or (game.ground_height[sl] // 2 != level).any():
            return False
        return occupied is None or not occupied[sl].any()

    cands = []
    for dx in range(-SEARCH_R, SEARCH_R + 1):
        for dy in range(-SEARCH_R, SEARCH_R + 1):
            lt = (ax + dx, ay + dy)
            for off in dict.fromkeys((0, lh - rh)):
                rt = (lt[0] + lw, lt[1] + off)
                if not (ok(left, lt) and ok(right, rt)):
                    continue
                gx = (lt[0] + lw) * 32
                gy = (max(lt[1], rt[1]) * 32 + min(lt[1] + lh, rt[1] + rh) * 32) / 2
                # the gap faces the ramp best when the ramp is above or below it
                score = math.hypot(gx - cx, gy - cy) + 0.5 * abs(gx - cx)
                cands.append((score, lt, rt))
    cands.sort()
    if cands and not _way_around(game, [], choke.center, inside, outside):
        return None                      # the check can't tell a sealed ramp from an open one here
    for _, lt, rt in cands:
        if _way_around(game, [box(game, left, lt), box(game, right, rt)], choke.center, inside, outside):
            wall = Wall({left: lt, right: rt}, gap)
            log.info("wall: %s at %s, %s at %s (gap %d px)", game.type_name(left), lt, game.type_name(right),
                     rt, gap)
            return wall
    return None


def _way_around(game: GameInfo, boxes, center, inside, outside) -> bool:
    """Units CLEAR_WALK walk tiles wide can still get from `outside` (below the ramp) to `inside`
    (the main) within WINDOW tiles of `center`, with `boxes` (pixels) built."""
    wx0 = max(0, int(center[0]) // 8 - WINDOW * 4)
    wy0 = max(0, int(center[1]) // 8 - WINDOW * 4)
    wx1 = min(game.walkable.shape[1], int(center[0]) // 8 + WINDOW * 4)
    wy1 = min(game.walkable.shape[0], int(center[1]) // 8 + WINDOW * 4)
    free = game.walkable[wy0:wy1, wx0:wx1].copy()
    for x0, y0, x1, y1 in boxes:
        free[max(0, y0 // 8 - wy0):max(0, y1 // 8 + 1 - wy0), max(0, x0 // 8 - wx0):max(0, x1 // 8 + 1 - wx0)] = False
    k = CLEAR_WALK
    h, w = free.shape
    if h < k or w < k:
        return False
    s = np.pad(np.cumsum(np.cumsum(free.astype(np.int32), 0), 1), ((1, 0), (1, 0)))
    room = (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) == k * k      # k x k block free at (y, x)

    def cell(p) -> Optional[tuple[int, int]]:
        py, px = int(p[1]) // 8 - wy0, int(p[0]) // 8 - wx0
        best = None
        for r in range(0, 12):
            for y in range(py - r, py + r + 1):
                for x in range(px - r, px + r + 1):
                    if 0 <= y < room.shape[0] and 0 <= x < room.shape[1] and room[y, x]:
                        if best is None or (y - py) ** 2 + (x - px) ** 2 < (best[0] - py) ** 2 + (best[1] - px) ** 2:
                            best = (y, x)
            if best is not None:
                return best
        return None

    start, goal = cell(outside), cell(inside)
    if start is None or goal is None:
        return False
    seen = np.zeros_like(room)
    seen[start] = True
    q = deque([start])
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            return True
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < room.shape[0] and 0 <= nx < room.shape[1] and room[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return False
