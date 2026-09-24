"""Meta slot: what is known before (and barely changes during) the game: map identity and
geometry, matchup, opponent name. No cross-game memory: everything comes from this game's
GameInfo and observations.

`meta.features` is a fixed, named numeric vector for learned strategy models (see META_FEATURES);
`meta.map_analysis` holds per-analyzer results. Add analyzers with `register_analyzer(name, fn)`
where `fn(game, graph) -> dict`; they run once at game start.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from bwbot import Race
from bwbot.observation import GameInfo

from ..mapgraph import MapGraph

log = logging.getLogger("adjutant.meta")

RACE_LETTER = {int(Race.Zerg): "Z", int(Race.Terran): "T", int(Race.Protoss): "P"}
KNOWN = (int(Race.Zerg), int(Race.Terran), int(Race.Protoss))

META_FEATURES = (
    "map_w", "map_h", "starts", "bases", "bases_per_start", "areas", "chokes",
    "rush_min", "rush_max", "rush_mean", "nat_dist", "main_choke_w", "nat_open", "island_bases",
    "self_T", "self_Z", "self_P", "enemy_T", "enemy_Z", "enemy_P", "enemy_random",
)

ANALYZERS: dict[str, Callable[[GameInfo, MapGraph], dict]] = {}


def register_analyzer(name: str):
    def deco(fn):
        ANALYZERS[name] = fn
        return fn
    return deco


@register_analyzer("bases")
def _bases(game: GameInfo, graph: MapGraph) -> dict:
    main = game.main
    out = []
    for b in game.bases:
        d = graph.distance(main.center, main.area_id, b.center, b.area_id) if main else 0.0
        out.append({"id": b.id, "tile": b.tile, "dist": round(d), "minerals": b.minerals, "geysers": b.geysers,
                    "starting": b.starting, "reachable": graph.connected(main.area_id, b.area_id) if main else True})
    out.sort(key=lambda r: r["dist"])
    return {"by_distance": out}


@register_analyzer("chokes")
def _chokes(game: GameInfo, graph: MapGraph) -> dict:
    widths = [c.width for c in game.chokes]
    return {"n": len(widths), "mean_width": float(np.mean(widths)) if widths else 0.0,
            "blocking": sum(1 for c in game.chokes if c.blocking)}


def clean_map_name(name: str) -> str:
    """Strip StarCraft colour/control codes (e.g. '\\x06B\\x05e...')."""
    return "".join(ch for ch in name if ch >= " ").strip()


def matchup(self_race: int, enemy_race: int) -> str:
    return f"{RACE_LETTER.get(self_race, 'R')}v{RACE_LETTER.get(enemy_race, 'R')}"


@register("MetaAnalysis")
class MetaAnalysis(Component):
    phase = Phase.SENSE
    writes = ("meta",)
    order = 10                          # after Perception's fog filter

    def on_start(self, bb: Blackboard) -> None:
        g, m = bb.game, bb.meta
        self.graph = MapGraph(g)
        bb.services["mapgraph"] = self.graph
        m.map_name, m.map_hash = clean_map_name(g.map_name), g.map_hash
        m.map_size = (g.map_width, g.map_height)
        m.n_starts, m.n_bases = len(g.start_locations), len(g.bases)
        m.self_race = int(g.self_race)
        enemy = g.enemies[0] if g.enemies else None
        m.enemy_name = enemy.name if enemy else ""
        m.enemy_race = enemy.race if enemy and enemy.race in KNOWN else int(Race.Unknown)
        self._random = not (enemy and enemy.race in KNOWN)
        mc = g.main_choke
        m.main_choke_width = mc.width if mc else 0
        main, nat = g.main, g.natural
        m.natural_distance = self.graph.base_distance(main.id, nat.id) if main and nat else 0.0
        m.enemy_start_distances = []
        for sb in g.start_bases:
            if main is not None and sb.base_id != main.id:
                m.enemy_start_distances.append(self.graph.base_distance(main.id, sb.base_id))
        m.rush_distance = min(m.enemy_start_distances) if m.enemy_start_distances else 0.0
        m.map_analysis = {}
        for name, fn in ANALYZERS.items():
            try:
                m.map_analysis[name] = fn(g, self.graph)
            except Exception:  # an analyzer bug must not stop the game
                log.exception("map analyzer %s failed", name)
        self._features(bb)

    def tick(self, bb: Blackboard) -> None:
        m = bb.meta
        if m.enemy_race in KNOWN:
            return
        obs, g = bb.obs, bb.game
        enemy = obs.units[obs.units["player"] == g.enemy_id]
        if len(enemy):
            races = g.unit_types["race"][np.clip(enemy["type"], 0, len(g.unit_types) - 1)]
            known = [int(r) for r in races if int(r) in KNOWN]
            if known:
                m.enemy_race = known[0]
                self._features(bb)
                bb.raise_event("enemy_race")

    def _features(self, bb: Blackboard) -> None:
        g, m = bb.game, bb.meta
        rush = m.enemy_start_distances or [0.0]
        bases = m.map_analysis.get("bases", {}).get("by_distance", [])
        mc = g.main_choke
        f = {
            "map_w": g.map_width / 256, "map_h": g.map_height / 256,
            "starts": m.n_starts / 4, "bases": m.n_bases / 20,
            "bases_per_start": m.n_bases / max(1, m.n_starts) / 5,
            "areas": len(g.areas) / 50, "chokes": len(g.chokes) / 50,
            "rush_min": min(rush) / 32 / 200, "rush_max": max(rush) / 32 / 200,
            "rush_mean": float(np.mean(rush)) / 32 / 200,
            "nat_dist": m.natural_distance / 32 / 50,
            "main_choke_w": m.main_choke_width / 32 / 10,
            "nat_open": float(sum(1 for c in g.chokes if g.natural is not None
                                  and g.natural.area_id in (c.area_a, c.area_b)
                                  and (mc is None or c.id != mc.id))) / 4,
            "island_bases": sum(1 for b in bases if not b["reachable"]) / 10,
            "self_T": float(m.self_race == Race.Terran), "self_Z": float(m.self_race == Race.Zerg),
            "self_P": float(m.self_race == Race.Protoss),
            "enemy_T": float(m.enemy_race == Race.Terran), "enemy_Z": float(m.enemy_race == Race.Zerg),
            "enemy_P": float(m.enemy_race == Race.Protoss), "enemy_random": float(self._random),
        }
        m.features = {k: float(f[k]) for k in META_FEATURES}

    def vector(self, bb: Blackboard) -> np.ndarray:
        return np.array([bb.meta.features.get(k, 0.0) for k in META_FEATURES], np.float32)

    def summary(self) -> str:
        return "meta"
