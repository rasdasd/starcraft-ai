"""Ground distances over the BWEM area/choke graph from GameInfo (no pathfinding queries needed).

A path from a point in area A to a point in area B goes point -> choke -> ... -> choke -> point,
each leg a straight line; chokes are linked when they border a common area. Blocking chokes
(neutral buildings / mineral walls) cost `blocking_penalty` extra. Disconnected pairs (islands, or
maps whose chokes the analysis could not link) fall back to straight-line distance x `detour`.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Optional

from bwbot.observation import GameInfo


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MapGraph:
    def __init__(self, game: GameInfo, blocking_penalty: float = 2000.0, detour: float = 1.3) -> None:
        self.game = game
        self.detour = detour
        self.chokes = list(game.chokes)
        self.by_area: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(self.chokes):
            self.by_area[c.area_a].append(i)
            self.by_area[c.area_b].append(i)
        self.cost = [blocking_penalty if c.blocking else 0.0 for c in self.chokes]
        self._cache: dict[tuple, tuple[float, bool]] = {}

    def distance(self, p: tuple[int, int], area_p: int, q: tuple[int, int], area_q: int) -> float:
        return self.route(p, area_p, q, area_q)[0]

    def connected(self, area_p: int, area_q: int) -> bool:
        if area_p == area_q:
            return True
        return self.route((0, 0), area_p, (0, 0), area_q)[1]

    def route(self, p, area_p: int, q, area_q: int) -> tuple[float, bool]:
        """(distance in pixels, found a choke path)."""
        key = (tuple(p), area_p, tuple(q), area_q)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if area_p == area_q:
            out = (_d(p, q), True)
        else:
            out = self._dijkstra(p, area_p, q, area_q)
        self._cache[key] = out
        return out

    def _dijkstra(self, p, area_p, q, area_q) -> tuple[float, bool]:
        dist: dict[int, float] = {}
        heap: list[tuple[float, int]] = []
        for i in self.by_area.get(area_p, ()):
            d0 = _d(p, self.chokes[i].center) + self.cost[i]
            if d0 < dist.get(i, math.inf):
                dist[i] = d0
                heapq.heappush(heap, (d0, i))
        targets = set(self.by_area.get(area_q, ()))
        best = math.inf
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist.get(i, math.inf) or d >= best:
                continue
            c = self.chokes[i]
            if i in targets:
                best = min(best, d + _d(c.center, q))
            for area in (c.area_a, c.area_b):
                for j in self.by_area.get(area, ()):
                    if j == i:
                        continue
                    nd = d + _d(c.center, self.chokes[j].center) + self.cost[j]
                    if nd < dist.get(j, math.inf):
                        dist[j] = nd
                        heapq.heappush(heap, (nd, j))
        if best < math.inf:
            return best, True
        return _d(p, q) * self.detour, False

    def base_distance(self, a_id: int, b_id: int) -> float:
        a, b = self.game.base(a_id), self.game.base(b_id)
        if a is None or b is None:
            return 0.0
        return self.distance(a.center, a.area_id, b.center, b.area_id)


def graph_for(game: GameInfo, cache: Optional[dict] = None) -> MapGraph:
    if cache is not None:
        g = cache.get("mapgraph")
        if g is None or g.game is not game:
            g = cache["mapgraph"] = MapGraph(game)
        return g
    return MapGraph(game)
