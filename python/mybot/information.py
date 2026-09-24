"""InformationManager: remember the enemy after they disappear into fog.

BWAPI only reports units you can see. Other bots keep a register of last-known
positions (especially buildings) and treat a building as gone only when they have
vision on its tile and it is not there, or when they get UnitDestroy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from bwbot import EventType, GameInfo, Observation, UnitFlag, UnitType
from bwbot.observation import UnitTypeFlag

from .state import Memory

log = logging.getLogger("mybot.info")

RESOURCE_DEPOTS = {
    int(UnitType.Terran_Command_Center),
    int(UnitType.Zerg_Hatchery), int(UnitType.Zerg_Lair), int(UnitType.Zerg_Hive),
    int(UnitType.Protoss_Nexus),
}


@dataclass
class SeenUnit:
    id: int
    type: int
    x: int
    y: int
    frame: int
    building: bool


@dataclass
class InformationManager:
    enemy_start: Optional[tuple[int, int]] = None
    last_attacked_frame: int = -10_000
    seen: dict[int, SeenUnit] = field(default_factory=dict)
    other_starts: list[tuple[int, int]] = field(default_factory=list)

    def reset(self) -> None:
        self.enemy_start = None
        self.last_attacked_frame = -10_000
        self.seen.clear()
        self.other_starts.clear()

    def on_start(self, game: GameInfo) -> None:
        self.reset()
        me = tuple(game.self_player.start_location)
        self.other_starts = [tuple(s) for s in game.start_locations.tolist() if tuple(s) != me]
        if self.other_starts:
            self.enemy_start = self.other_starts[0]

    def as_memory(self) -> Memory:
        return Memory(
            enemy_start=self.enemy_start,
            enemy_buildings_seen={u.id: (u.x, u.y) for u in self.seen.values() if u.building},
            enemy_building_list=self.buildings(),
            last_attacked_frame=self.last_attacked_frame,
        )

    def buildings(self) -> list[tuple[int, int, int]]:
        """Last-known enemy buildings as (type, x, y) pixels."""
        return [(u.type, u.x, u.y) for u in self.seen.values() if u.building]

    def has_enemy_base(self) -> bool:
        return any(u.building and u.type in RESOURCE_DEPOTS for u in self.seen.values())

    def nearest_building(self, x: int, y: int) -> Optional[tuple[int, int]]:
        best = None
        best_d = None
        for u in self.seen.values():
            if not u.building:
                continue
            d = (u.x - x) ** 2 + (u.y - y) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best = (u.x, u.y)
        return best

    def update(self, obs: Observation, game: GameInfo) -> None:
        if len(obs.my_units) and (obs.my_units["flags"] & UnitFlag.RecentlyAttacked).any():
            self.last_attacked_frame = obs.frame_count

        for e in obs.iter_events(EventType.UnitDestroy):
            self.seen.pop(e.unit, None)

        visible_ids: set[int] = set()
        enemies = obs.enemy_units
        if len(enemies):
            etypes = enemies["type"].clip(0, len(game.unit_types) - 1)
            eflags = game.unit_types["flags"][etypes]
            is_bld = (eflags & UnitTypeFlag.Building) != 0
            for u, building in zip(enemies, is_bld):
                uid = int(u["id"])
                visible_ids.add(uid)
                self.seen[uid] = SeenUnit(
                    uid, int(u["type"]), int(u["x"]), int(u["y"]), obs.frame_count, bool(building),
                )
                if building and int(u["type"]) in RESOURCE_DEPOTS:
                    tile = (int(u["x"]) // 32, int(u["y"]) // 32)
                    if self.enemy_start != tile:
                        self.enemy_start = tile
                        log.info("f%d enemy base at tile %s (%s)", obs.frame_count, tile,
                                 game.type_name(int(u["type"])))

        visible = obs.visible if obs.tiles.size else None
        if visible is None:
            return
        for uid, su in list(self.seen.items()):
            if uid in visible_ids or not su.building:
                continue
            tx, ty = su.x // 32, su.y // 32
            if 0 <= ty < visible.shape[0] and 0 <= tx < visible.shape[1] and visible[ty, tx]:
                self.seen.pop(uid, None)
