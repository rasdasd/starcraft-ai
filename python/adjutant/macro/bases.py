"""Our bases and where to expand next.

A base is ours when one of our resource depots (completed or not) stands on its BWEM hall site;
Zerg macro hatcheries elsewhere are not bases. Mineral fields and geysers belong to the nearest base
centre within `RESOURCE_PX`. The next expansion is the free base closest by ground to our bases
(the natural first), skipping bases the enemy holds or has buildings near, bases that are not
ground-reachable from the main without crossing a blocking choke (a mineral wall or neutral
building), and sites that were recently found blocked or dangerous.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from blackboard.sections import OwnBase
from bwbot import UnitFlag
from bwbot.observation import UnitTypeFlag

from ..mapgraph import MapGraph

log = logging.getLogger("adjutant.macro")

RESOURCE_PX = 12 * 32
HALL_PX = 4 * 32             # a depot this close to a base centre stands on its hall site
ENEMY_NEAR_PX = 14 * 32      # enemy buildings this close make a base unavailable
UNREACHABLE_FRAMES = 24 * 600  # a site builders could not walk to is skipped this long


class BaseTracker:
    def __init__(self, blocked_frames: int = 24 * 180, danger_frames: int = 24 * 120) -> None:
        self.blocked_frames = blocked_frames
        self.danger_frames = danger_frames
        self.avoid: dict[int, int] = {}          # base id -> frame until which it is skipped
        self.bases: list[OwnBase] = []
        self.patch_base: dict[int, int] = {}      # mineral field id -> base id
        self.graph: Optional[MapGraph] = None
        self._walled: set[int] = set()

    def on_start(self, game, graph: Optional[MapGraph] = None) -> None:
        self.game = game
        self.graph = graph or MapGraph(game)
        self.avoid.clear()
        self.bases = []
        self.patch_base = {}
        main = game.main
        self.main_area = main.area_id if main is not None else -1
        self.main_center = main.center if main is not None else (0, 0)

    # ------------------------------------------------------------------ per decision
    def update(self, obs, frame: int) -> list[OwnBase]:
        g = self.game
        centers = np.array([b.center for b in g.bases], dtype=np.int64).reshape(-1, 2)
        fields = obs.minerals_fields
        self.patch_base = {}
        per_base: dict[int, list] = {}
        if len(fields) and len(centers):
            d2 = (fields["x"][:, None] - centers[None, :, 0]) ** 2 + (fields["y"][:, None] - centers[None, :, 1]) ** 2
            nearest = d2.argmin(axis=1)
            for f, i, dd in zip(fields, nearest, d2[np.arange(len(fields)), nearest]):
                if dd <= RESOURCE_PX ** 2:
                    bid = g.bases[int(i)].id
                    self.patch_base[int(f["id"])] = bid
                    per_base.setdefault(bid, []).append(f)
        mine = obs.my_units
        flags = g.unit_types["flags"][np.clip(mine["type"], 0, len(g.unit_types) - 1)] if len(mine) else np.zeros(0)
        depots = mine[(flags & UnitTypeFlag.ResourceDepot) != 0] if len(mine) else mine
        refineries = mine[(flags & UnitTypeFlag.Refinery) != 0] if len(mine) else mine
        geysers = obs.geysers
        out: list[OwnBase] = []
        for base in g.bases:
            cx, cy = base.center
            on_site = [d for d in depots if (int(d["x"]) - cx) ** 2 + (int(d["y"]) - cy) ** 2 <= HALL_PX ** 2]
            if not on_site:
                continue
            done = any(obs.has_flag(d, UnitFlag.Completed) for d in on_site)
            fs = per_base.get(base.id, [])
            near = lambda rows: [r for r in rows if (int(r["x"]) - cx) ** 2 + (int(r["y"]) - cy) ** 2 <= RESOURCE_PX ** 2]  # noqa: E731
            refs = near(refineries)
            out.append(OwnBase(
                base_id=base.id, tile=tuple(base.tile), center=(int(cx), int(cy)), completed=done,
                patches=len(fs), minerals=int(sum(int(f["resources"]) for f in fs)),
                geysers=len(near(geysers)) + len(refs),
                refineries=sum(1 for r in refs if obs.has_flag(r, UnitFlag.Completed))))
        self.bases = out
        for bid in [b for b, until in self.avoid.items() if until <= frame]:
            del self.avoid[bid]
        return out

    def base_of_patch(self, patch_id: int) -> Optional[int]:
        return self.patch_base.get(int(patch_id))

    def owned(self, base_id: int) -> bool:
        return any(b.base_id == base_id for b in self.bases)

    # ------------------------------------------------------------------ expansion choice
    def mark_blocked(self, base_id: int, frame: int) -> None:
        self.avoid[int(base_id)] = frame + self.blocked_frames

    def mark_unreachable(self, base_id: int, frame: int) -> None:
        self.avoid[int(base_id)] = frame + UNREACHABLE_FRAMES

    def mark_dangerous(self, base_id: int, frame: int) -> None:
        self.avoid[int(base_id)] = max(self.avoid.get(int(base_id), 0), frame + self.danger_frames)

    def next_base(self, belief, natural_id: int = -1) -> Optional[int]:
        g, graph = self.game, self.graph
        enemy_tiles = [e.tile for e in belief.bases if e.alive]
        enemy_blds = [(x, y) for _, x, y in belief.buildings]
        homes = [b.center for b in self.bases] or [self.main_center]
        es = belief.enemy_start
        enemy_main = None
        if es is not None and g.bases:
            enemy_main = min(g.bases, key=lambda b: (b.tile[0] - es[0]) ** 2 + (b.tile[1] - es[1]) ** 2)
        best, best_score = None, math.inf
        for base in g.bases:
            if self.owned(base.id) or base.id in self.avoid or base.minerals <= 0:
                continue
            if enemy_main is not None and base.id == enemy_main.id:
                continue
            tx, ty = base.tile
            if any((tx - ex) ** 2 + (ty - ey) ** 2 < 12 ** 2 for ex, ey in enemy_tiles):
                continue
            cx, cy = base.center
            if any((cx - x) ** 2 + (cy - y) ** 2 < ENEMY_NEAR_PX ** 2 for x, y in enemy_blds):
                continue
            if graph is not None and self.main_area >= 0 \
                    and not graph.connected(self.main_area, base.area_id, open_only=True):
                if base.id not in self._walled and graph.connected(self.main_area, base.area_id):
                    self._walled.add(base.id)
                    log.info("base %d at %s is behind a blocking choke: not an expansion", base.id, base.tile)
                continue                        # an island, or behind a mineral wall / neutral building
            if base.id == natural_id:
                return base.id
            d_home = min(self._ground(h, base) for h in homes)
            score = d_home * (0.9 if base.geysers else 1.0)
            if enemy_main is not None:
                score -= 0.25 * self._ground(enemy_main.center, base, enemy_main.area_id)
            if score < best_score:
                best, best_score = base.id, score
        return best

    def _ground(self, p, base, area: Optional[int] = None) -> float:
        if self.graph is None:
            return math.hypot(p[0] - base.center[0], p[1] - base.center[1])
        if area is None:
            area = self._area_of(p)
        return self.graph.distance(p, area, base.center, base.area_id)

    def _area_of(self, p) -> int:
        g = self.game
        b = min(g.bases, key=lambda b: (b.center[0] - p[0]) ** 2 + (b.center[1] - p[1]) ** 2)
        return b.area_id
