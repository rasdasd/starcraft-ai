"""Decision-making: a `Policy` maps a `State` to a list of `Intent`s.

`Build` is idempotent: ProductionManager queues it once. Openings are data
(`opening.MARINE` / `opening.GOLIATH`); army stance is Attack/Rally and
CombatCommander turns that into defend / push / hunt-air.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Union

from bwbot import Race, UnitType

from . import opening
from .state import State

log = logging.getLogger("mybot.policy")


@dataclass(frozen=True)
class Train:
    unit_type: int


@dataclass(frozen=True)
class Build:
    unit_type: int


@dataclass(frozen=True)
class Attack:
    x: int
    y: int


@dataclass(frozen=True)
class Rally:
    x: int
    y: int


@dataclass(frozen=True)
class BuildAddon:
    unit_type: int


@dataclass(frozen=True)
class Upgrade:
    upgrade_type: int


@dataclass(frozen=True)
class Research:
    tech_type: int


Intent = Union[Train, Build, Attack, Rally, BuildAddon, Upgrade, Research]


class Policy(Protocol):
    def decide(self, s: State) -> list[Intent]: ...
    def on_game_end(self, s: Optional[State], won: bool) -> None: ...


SCV = UnitType.Terran_SCV
DEPOT = UnitType.Terran_Supply_Depot
RAX = UnitType.Terran_Barracks
MARINE = UnitType.Terran_Marine


class ScriptedPolicy:
    """Deterministic Terran bio: `opening.MARINE` then more rax / marines."""

    MAX_WORKERS = 20
    ATTACK_ARMY_SIZE = 20
    RETREAT_ARMY_SIZE = 6

    def __init__(self) -> None:
        self.opening = opening.Opening(opening.MARINE)
        self.attacking = False

    def reset(self) -> None:
        self.__init__()

    def decide(self, s: State) -> list[Intent]:
        if s.game.self_race != Race.Terran:
            log.warning("ScriptedPolicy is Terran-only; game race is %s (set race = Terran in bwapi.ini)",
                        s.game.self_race.name)
            return []
        out: list[Intent] = []

        nxt = self.opening.next_build(s)
        if nxt is not None:
            out.append(Build(nxt))
        elif self.opening.done:
            depots_in_progress = s.count(DEPOT) - s.count_completed(DEPOT)
            if s.supply_left <= 3 and depots_in_progress == 0 and s.supply_total < 200:
                out.append(Build(DEPOT))
            elif s.minerals >= 400:
                out.append(Build(RAX))

        if s.count(SCV) < self.MAX_WORKERS:
            out.append(Train(SCV))
        if s.count_completed(RAX) > 0:
            out.append(Train(MARINE))

        n = len(s.army)
        if n >= self.ATTACK_ARMY_SIZE:
            self.attacking = True
        elif n < self.RETREAT_ARMY_SIZE:
            self.attacking = False

        target = self._army_target(s)
        if target is not None:
            out.append(Attack(*target))
        else:
            out.append(Rally(*s.rally_point))
        return out

    def _army_target(self, s: State) -> Optional[tuple[int, int]]:
        if len(s.enemies) and (s.under_attack or self.attacking):
            e = s.obs.nearest(s.enemies, s.main_tile[0] * 32, s.main_tile[1] * 32)
            return int(e["x"]), int(e["y"])
        if self.attacking:
            if s.enemy_buildings:
                return s.enemy_buildings[0][1], s.enemy_buildings[0][2]
            if s.enemy_start is not None:
                return s.enemy_start[0] * 32 + 64, s.enemy_start[1] * 32 + 48
        return None

    def on_game_end(self, s: Optional[State], won: bool) -> None:
        log.info("game over: %s after %d opening steps", "WIN" if won else "LOSS", self.opening.i)
        self.reset()
