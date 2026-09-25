"""Opponent model: race, first-seen timings, composition, opening guess.

This is the persistent picture a learned policy is allowed to condition on.
It is not fog memory (that stays in InformationManager); it is *interpretation*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from bwbot import GameInfo, Observation, Race, UnitType

RESOURCE_DEPOTS = {
    int(UnitType.Terran_Command_Center),
    int(UnitType.Zerg_Hatchery), int(UnitType.Zerg_Lair), int(UnitType.Zerg_Hive),
    int(UnitType.Protoss_Nexus),
}

log = logging.getLogger("mybot.opponent")

# Buildings whose first-seen frame is enough to classify most ladder openings.
TRACKED = (
    UnitType.Terran_Supply_Depot, UnitType.Terran_Barracks, UnitType.Terran_Refinery,
    UnitType.Terran_Factory, UnitType.Terran_Starport, UnitType.Terran_Bunker,
    UnitType.Terran_Engineering_Bay, UnitType.Terran_Armory, UnitType.Terran_Academy,
    UnitType.Zerg_Hatchery, UnitType.Zerg_Lair, UnitType.Zerg_Hive,
    UnitType.Zerg_Spawning_Pool, UnitType.Zerg_Extractor, UnitType.Zerg_Hydralisk_Den,
    UnitType.Zerg_Spire, UnitType.Zerg_Greater_Spire,
    UnitType.Protoss_Pylon, UnitType.Protoss_Gateway, UnitType.Protoss_Assimilator,
    UnitType.Protoss_Cybernetics_Core, UnitType.Protoss_Forge, UnitType.Protoss_Stargate,
    UnitType.Protoss_Robotics_Facility, UnitType.Protoss_Nexus, UnitType.Protoss_Photon_Cannon,
)

AIR_UNITS = {
    int(UnitType.Terran_Wraith), int(UnitType.Terran_Valkyrie), int(UnitType.Terran_Battlecruiser),
    int(UnitType.Terran_Science_Vessel), int(UnitType.Terran_Dropship),
    int(UnitType.Zerg_Mutalisk), int(UnitType.Zerg_Scourge), int(UnitType.Zerg_Guardian),
    int(UnitType.Zerg_Devourer), int(UnitType.Zerg_Overlord), int(UnitType.Zerg_Queen),
    int(UnitType.Protoss_Scout), int(UnitType.Protoss_Carrier), int(UnitType.Protoss_Corsair),
    int(UnitType.Protoss_Arbiter), int(UnitType.Protoss_Shuttle), int(UnitType.Protoss_Observer),
}

RUSH_WINDOW = 24 * 100   # first ~100 game-seconds


class OpeningGuess(IntEnum):
    UNKNOWN = 0
    RUSH = 1
    BIO = 2
    MECH = 3
    AIR = 4
    TWO_BASE = 5
    CHEESE = 6


OPENING_NAMES = [g.name.lower() for g in OpeningGuess]


@dataclass
class OpponentSnapshot:
    """Frozen view stuffed into State / the feature vector. Safe to log."""

    race: int = int(Race.Unknown)
    opening: int = int(OpeningGuess.UNKNOWN)
    first_military: int = -1          # frame of first fighting building, or -1
    buildings: int = 0
    air_units: int = 0
    ground_army: int = 0
    workers: int = 0
    attacks: int = 0
    proxy: bool = False
    counts: dict[int, int] = field(default_factory=dict)

    def count(self, unit_type: int) -> int:
        return int(self.counts.get(int(unit_type), 0))


class OpponentModel:
    def __init__(self) -> None:
        self.race = Race.Unknown
        self.first: dict[int, int] = {}
        self.counts: dict[int, int] = {}
        self.attacks = 0
        self.proxy = False
        self._last_attacked = -10_000
        self._our_start: Optional[tuple[int, int]] = None

    def reset(self) -> None:
        self.__init__()

    def on_start(self, game: GameInfo) -> None:
        self.reset()
        self._our_start = tuple(game.self_player.start_location)
        for p in game.enemies:
            if p.race in (int(Race.Zerg), int(Race.Terran), int(Race.Protoss)):
                self.race = Race(p.race)
                break

    def snapshot(self) -> OpponentSnapshot:
        air, ground, workers = 0, 0, 0
        for t, n in self.counts.items():
            if t in AIR_UNITS:
                air += n
            elif t in (int(UnitType.Terran_SCV), int(UnitType.Zerg_Drone), int(UnitType.Protoss_Probe)):
                workers += n
            else:
                ground += n
        return OpponentSnapshot(
            race=int(self.race),
            opening=int(self.guess_opening()),
            first_military=self._first_military(),
            buildings=sum(1 for t, n in self.counts.items() if n and _is_building_id(t)),
            air_units=air,
            ground_army=ground,
            workers=workers,
            attacks=self.attacks,
            proxy=self.proxy,
            counts=dict(self.counts),
        )

    def update(self, obs: Observation, game: GameInfo, info) -> None:
        if info.last_attacked_frame > self._last_attacked:
            self.attacks += 1
            self._last_attacked = info.last_attacked_frame

        visible = {int(u["type"]): 0 for u in obs.enemy_units} if len(obs.enemy_units) else {}
        for u in obs.enemy_units:
            visible[int(u["type"])] = visible.get(int(u["type"]), 0) + 1
            self._note(int(u["type"]), obs.frame_count, game)

        # Fog buildings still count toward composition / first-seen.
        for su in info.seen.values():
            self._note(su.type, su.frame, game)
            if su.building:
                visible[su.type] = max(visible.get(su.type, 0), 1)
            if su.building and self._looks_proxy(su, info):
                if not self.proxy:
                    log.info("f%d proxy %s at %s", obs.frame_count, game.type_name(su.type), (su.x, su.y))
                self.proxy = True

        for t, n in visible.items():
            self.counts[t] = max(self.counts.get(t, 0), n)

        if self.race is Race.Unknown:
            self._infer_race(game)

    def guess_opening(self) -> OpeningGuess:
        if self.proxy:
            return OpeningGuess.CHEESE
        first = self._first_military()
        early = first >= 0 and first < RUSH_WINDOW

        if self.race is Race.Zerg:
            pool = self.first.get(int(UnitType.Zerg_Spawning_Pool), -1)
            hatch = max(self.count(UnitType.Zerg_Hatchery) + self.count(UnitType.Zerg_Lair)
                        + self.count(UnitType.Zerg_Hive), 1)
            if self.count(UnitType.Zerg_Spire) or self.count(UnitType.Zerg_Mutalisk):
                return OpeningGuess.AIR
            if hatch >= 2 and (pool < 0 or pool > RUSH_WINDOW):
                return OpeningGuess.TWO_BASE
            if pool >= 0 and pool < RUSH_WINDOW and hatch <= 1:
                return OpeningGuess.RUSH
            if early:
                return OpeningGuess.RUSH
            return OpeningGuess.UNKNOWN

        if self.race is Race.Protoss:
            if self.count(UnitType.Protoss_Forge) and self.count(UnitType.Protoss_Gateway) == 0:
                return OpeningGuess.CHEESE
            if self.count(UnitType.Protoss_Stargate) or self.count(UnitType.Protoss_Corsair):
                return OpeningGuess.AIR
            if self.count(UnitType.Protoss_Nexus) >= 2:
                return OpeningGuess.TWO_BASE
            if self.count(UnitType.Protoss_Gateway) >= 2 and early:
                return OpeningGuess.RUSH
            if early:
                return OpeningGuess.RUSH
            return OpeningGuess.UNKNOWN

        if self.race is Race.Terran:
            if self.count(UnitType.Terran_Starport) or self.count(UnitType.Terran_Wraith):
                return OpeningGuess.AIR
            if self.count(UnitType.Terran_Factory):
                return OpeningGuess.MECH
            if self.count(UnitType.Terran_Barracks) >= 2:
                return OpeningGuess.BIO if not early else OpeningGuess.RUSH
            if early:
                return OpeningGuess.RUSH
            return OpeningGuess.UNKNOWN

        if early:
            return OpeningGuess.RUSH
        return OpeningGuess.UNKNOWN

    def count(self, unit_type: int) -> int:
        return int(self.counts.get(int(unit_type), 0))

    def _note(self, unit_type: int, frame: int, game: GameInfo) -> None:
        if unit_type not in self.first:
            self.first[unit_type] = frame
            if unit_type in {int(t) for t in TRACKED}:
                log.info("f%d first seen %s", frame, game.type_name(unit_type))
        if self.race is Race.Unknown:
            race = int(game.unit_types["race"][min(unit_type, len(game.unit_types) - 1)])
            if race in (int(Race.Zerg), int(Race.Terran), int(Race.Protoss)):
                self.race = Race(race)

    def _first_military(self) -> int:
        keys = (
            UnitType.Terran_Barracks, UnitType.Terran_Bunker, UnitType.Terran_Factory,
            UnitType.Zerg_Spawning_Pool, UnitType.Protoss_Gateway, UnitType.Protoss_Photon_Cannon,
        )
        frames = [self.first[int(t)] for t in keys if int(t) in self.first]
        return min(frames) if frames else -1

    def _infer_race(self, game: GameInfo) -> None:
        for t in self.counts:
            race = int(game.unit_types["race"][min(t, len(game.unit_types) - 1)])
            if race in (int(Race.Zerg), int(Race.Terran), int(Race.Protoss)):
                self.race = Race(race)
                return

    def _looks_proxy(self, su, info) -> bool:
        if not su.building or self._our_start is None:
            return False
        if su.type in RESOURCE_DEPOTS:
            return False
        ox, oy = self._our_start[0] * 32, self._our_start[1] * 32
        d_us = (su.x - ox) ** 2 + (su.y - oy) ** 2
        if info.enemy_start is None:
            return d_us < (24 * 32) ** 2
        ex, ey = info.enemy_start[0] * 32, info.enemy_start[1] * 32
        d_them = (su.x - ex) ** 2 + (su.y - ey) ** 2
        return d_us < d_them and d_us < (30 * 32) ** 2


def _is_building_id(unit_type: int) -> bool:
    return unit_type >= int(UnitType.Terran_Command_Center)
