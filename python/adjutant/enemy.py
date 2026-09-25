"""Enemy memory and interpretation used by the belief slot.

`FogMemory` keeps last-known enemy buildings and the guessed enemy start: a building is gone once
its tile is visible and it is not there, or on UnitDestroy. `OpeningModel` records first-seen
frames and composition and classifies the opening (`OPENING_NAMES`) and proxies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from bwbot import EventType, GameInfo, Observation, Race, UnitFlag, UnitType as U
from bwbot.observation import UnitTypeFlag

log = logging.getLogger("adjutant.enemy")

RESOURCE_DEPOTS = {int(U.Terran_Command_Center), int(U.Zerg_Hatchery), int(U.Zerg_Lair), int(U.Zerg_Hive),
                   int(U.Protoss_Nexus)}
OPENING_NAMES = ["unknown", "rush", "bio", "mech", "air", "two_base", "cheese"]
RUSH_WINDOW = 24 * 100   # first ~100 game-seconds

AIR_UNITS = {
    int(U.Terran_Wraith), int(U.Terran_Valkyrie), int(U.Terran_Battlecruiser), int(U.Terran_Science_Vessel),
    int(U.Terran_Dropship), int(U.Zerg_Mutalisk), int(U.Zerg_Scourge), int(U.Zerg_Guardian), int(U.Zerg_Devourer),
    int(U.Zerg_Overlord), int(U.Zerg_Queen), int(U.Protoss_Scout), int(U.Protoss_Carrier), int(U.Protoss_Corsair),
    int(U.Protoss_Arbiter), int(U.Protoss_Shuttle), int(U.Protoss_Observer),
}
FIRST_MILITARY = (U.Terran_Barracks, U.Terran_Bunker, U.Terran_Factory, U.Zerg_Spawning_Pool, U.Protoss_Gateway,
                  U.Protoss_Photon_Cannon)
PLAYABLE = (int(Race.Zerg), int(Race.Terran), int(Race.Protoss))


@dataclass
class Seen:
    id: int
    type: int
    x: int
    y: int
    frame: int
    building: bool


@dataclass
class FogMemory:
    enemy_start: Optional[tuple[int, int]] = None      # tiles
    last_attacked_frame: int = -10_000
    seen: dict[int, Seen] = field(default_factory=dict)
    other_starts: list[tuple[int, int]] = field(default_factory=list)

    def on_start(self, game: GameInfo) -> None:
        self.seen.clear()
        self.last_attacked_frame = -10_000
        me = tuple(game.self_player.start_location)
        self.other_starts = [tuple(s) for s in game.start_locations.tolist() if tuple(s) != me]
        self.enemy_start = self.other_starts[0] if self.other_starts else None

    def buildings(self) -> list[tuple[int, int, int]]:
        """Last-known enemy buildings as (type, x, y) pixels."""
        return [(u.type, u.x, u.y) for u in self.seen.values() if u.building]

    def has_enemy_base(self) -> bool:
        return any(u.building and u.type in RESOURCE_DEPOTS for u in self.seen.values())

    def update(self, obs: Observation, game: GameInfo) -> None:
        if len(obs.my_units) and (obs.my_units["flags"] & UnitFlag.RecentlyAttacked).any():
            self.last_attacked_frame = obs.frame_count
        for e in obs.iter_events(EventType.UnitDestroy):
            self.seen.pop(e.unit, None)
        visible_ids: set[int] = set()
        enemies = obs.enemy_units
        if len(enemies):
            flags = game.unit_types["flags"][enemies["type"].clip(0, len(game.unit_types) - 1)]
            for u, building in zip(enemies, (flags & UnitTypeFlag.Building) != 0):
                uid, t = int(u["id"]), int(u["type"])
                visible_ids.add(uid)
                self.seen[uid] = Seen(uid, t, int(u["x"]), int(u["y"]), obs.frame_count, bool(building))
                if building and t in RESOURCE_DEPOTS:
                    tile = (int(u["x"]) // 32, int(u["y"]) // 32)
                    if self.enemy_start != tile:
                        self.enemy_start = tile
                        log.info("f%d enemy base at tile %s (%s)", obs.frame_count, tile, game.type_name(t))
        visible = obs.visible if obs.tiles.size else None
        if visible is None:
            return
        for uid, su in list(self.seen.items()):
            if uid in visible_ids or not su.building:
                continue
            tx, ty = su.x // 32, su.y // 32
            if 0 <= ty < visible.shape[0] and 0 <= tx < visible.shape[1] and visible[ty, tx]:
                self.seen.pop(uid, None)


class OpeningModel:
    """Race, first-seen frames, max-seen composition, proxy flag, opening class."""

    def __init__(self) -> None:
        self.race = Race.Unknown
        self.first: dict[int, int] = {}
        self.counts: dict[int, int] = {}
        self.proxy = False
        self._our_start: Optional[tuple[int, int]] = None

    def on_start(self, game: GameInfo) -> None:
        self.__init__()
        self._our_start = tuple(game.self_player.start_location)
        for p in game.enemies:
            if p.race in PLAYABLE:
                self.race = Race(p.race)
                break

    def update(self, obs: Observation, game: GameInfo, fog: FogMemory) -> None:
        visible: dict[int, int] = {}
        for u in obs.enemy_units:
            visible[int(u["type"])] = visible.get(int(u["type"]), 0) + 1
            self._note(int(u["type"]), obs.frame_count, game)
        for su in fog.seen.values():            # fog buildings still count toward composition / first-seen
            self._note(su.type, su.frame, game)
            if su.building:
                visible[su.type] = max(visible.get(su.type, 0), 1)
                if not self.proxy and self._looks_proxy(su, fog):
                    log.info("f%d proxy %s at %s", obs.frame_count, game.type_name(su.type), (su.x, su.y))
                    self.proxy = True
        for t, n in visible.items():
            self.counts[t] = max(self.counts.get(t, 0), n)

    def count(self, t) -> int:
        return int(self.counts.get(int(t), 0))

    def opening(self) -> str:
        if self.proxy:
            return "cheese"
        first = [self.first[int(t)] for t in FIRST_MILITARY if int(t) in self.first]
        early = bool(first) and min(first) < RUSH_WINDOW
        c = self.count
        if self.race is Race.Zerg:
            pool = self.first.get(int(U.Zerg_Spawning_Pool), -1)
            hatch = max(c(U.Zerg_Hatchery) + c(U.Zerg_Lair) + c(U.Zerg_Hive), 1)
            if c(U.Zerg_Spire) or c(U.Zerg_Mutalisk):
                return "air"
            if hatch >= 2 and (pool < 0 or pool > RUSH_WINDOW):
                return "two_base"
            if 0 <= pool < RUSH_WINDOW and hatch <= 1:
                return "rush"
        elif self.race is Race.Protoss:
            if c(U.Protoss_Forge) and c(U.Protoss_Gateway) == 0:
                return "cheese"
            if c(U.Protoss_Stargate) or c(U.Protoss_Corsair):
                return "air"
            if c(U.Protoss_Nexus) >= 2:
                return "two_base"
        elif self.race is Race.Terran:
            if c(U.Terran_Starport) or c(U.Terran_Wraith):
                return "air"
            if c(U.Terran_Factory):
                return "mech"
            if c(U.Terran_Barracks) >= 2:
                return "rush" if early else "bio"
        return "rush" if early else "unknown"

    def _note(self, t: int, frame: int, game: GameInfo) -> None:
        self.first.setdefault(t, frame)
        if self.race is Race.Unknown:
            race = int(game.unit_types["race"][min(t, len(game.unit_types) - 1)])
            if race in PLAYABLE:
                self.race = Race(race)

    def _looks_proxy(self, su: Seen, fog: FogMemory) -> bool:
        if self._our_start is None or su.type in RESOURCE_DEPOTS:
            return False
        ox, oy = self._our_start[0] * 32, self._our_start[1] * 32
        d_us = (su.x - ox) ** 2 + (su.y - oy) ** 2
        if fog.enemy_start is None:
            return d_us < (24 * 32) ** 2
        ex, ey = fog.enemy_start[0] * 32, fog.enemy_start[1] * 32
        return d_us < (su.x - ex) ** 2 + (su.y - ey) ** 2 and d_us < (30 * 32) ** 2
