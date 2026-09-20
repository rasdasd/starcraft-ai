"""Decision-making: a `Policy` maps a `State` to a list of `Intent`s.

Intents are high-level and race-agnostic ("train a Marine", "build a Barracks", "attack there").
`bot.py` turns them into concrete BWAPI commands (which producer, which SCV, which tile). That
split is what lets you replace `ScriptedPolicy` with a learned one later:

    class LearnedPolicy:
        def __init__(self, model): self.model = model
        def decide(self, s: State) -> list[Intent]:
            action_id = int(self.model(s.as_features()).argmax())
            return [ACTION_TABLE[action_id]]          # e.g. a fixed menu of Train/Build/Attack intents
        def on_game_end(self, s, won): ...            # record the episode outcome

Nothing here touches the game: policies are plain Python and can be unit-tested with fabricated
`State`s or replayed from logged ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Union

from bwbot import Race, UnitType

from .state import State

log = logging.getLogger("mybot.policy")


# --------------------------------------------------------------------------- intents
@dataclass(frozen=True)
class Train:
    unit_type: int                # what to produce (bot picks an idle producer)


@dataclass(frozen=True)
class Build:
    unit_type: int                # structure to place near the main (bot picks tile + builder)


@dataclass(frozen=True)
class Attack:
    x: int                        # pixel position for the whole army to attack-move to
    y: int


@dataclass(frozen=True)
class Rally:
    x: int                        # pixel position for idle army units to gather at
    y: int


Intent = Union[Train, Build, Attack, Rally]


class Policy(Protocol):
    def decide(self, s: State) -> list[Intent]: ...
    def on_game_end(self, s: Optional[State], won: bool) -> None: ...


# --------------------------------------------------------------------------- scripted
SCV = UnitType.Terran_SCV
CC = UnitType.Terran_Command_Center
DEPOT = UnitType.Terran_Supply_Depot
RAX = UnitType.Terran_Barracks
MARINE = UnitType.Terran_Marine


class ScriptedPolicy:
    """Deterministic Terran opener + simple rules. Same State in -> same Intents out."""

    # (displayed supply at which to start it, structure). Executed strictly in order.
    BUILD_ORDER: list[tuple[int, int]] = [(9, DEPOT), (11, RAX), (13, RAX), (16, DEPOT), (20, DEPOT)]
    MAX_WORKERS = 20
    ATTACK_ARMY_SIZE = 20
    RETREAT_ARMY_SIZE = 6

    def __init__(self) -> None:
        self.step = 0            # index into BUILD_ORDER
        self.attacking = False

    def reset(self) -> None:
        self.__init__()

    def decide(self, s: State) -> list[Intent]:
        if s.game.self_race != Race.Terran:
            log.warning("ScriptedPolicy is Terran-only; game race is %s (set race = Terran in bwapi.ini)",
                        s.game.self_race.name)
            return []
        out: list[Intent] = []

        # 1. build order: advance when the structure exists (counts include in-progress ones).
        if self.step < len(self.BUILD_ORDER):
            supply_at, structure = self.BUILD_ORDER[self.step]
            want = sum(1 for _, t in self.BUILD_ORDER[: self.step + 1] if t == structure)
            if s.count(structure) >= want:
                self.step += 1
            elif s.supply_used >= supply_at:
                out.append(Build(structure))
        # 2. after the opener: never get supply blocked, keep adding barracks.
        else:
            depots_in_progress = s.count(DEPOT) - s.count_completed(DEPOT)
            if s.supply_left <= 3 and depots_in_progress == 0 and s.supply_total < 200:
                out.append(Build(DEPOT))
            elif s.minerals >= 400:          # floating minerals -> more production
                out.append(Build(RAX))

        # 3. production: workers to saturation, marines forever.
        if s.count(SCV) < self.MAX_WORKERS:
            out.append(Train(SCV))
        if s.count_completed(RAX) > 0:
            out.append(Train(MARINE))

        # 4. army: rally in front of the first barracks until big enough, then attack.
        n = len(s.army)
        if n >= self.ATTACK_ARMY_SIZE:
            self.attacking = True
        elif n < self.RETREAT_ARMY_SIZE:
            self.attacking = False

        target = self._army_target(s)
        if target is not None:
            out.append(Attack(*target))
        else:
            rax = s.obs.my_completed(RAX)
            if len(rax):
                out.append(Rally(int(rax[0]["x"]) + 96, int(rax[0]["y"]) + 32))
        return out

    def _army_target(self, s: State) -> Optional[tuple[int, int]]:
        """Defend if we're being hit, otherwise push when attacking; None = stay home."""
        if len(s.enemies) and (s.under_attack or self.attacking):
            e = s.obs.nearest(s.enemies, s.main_tile[0] * 32, s.main_tile[1] * 32)
            return int(e["x"]), int(e["y"])
        if self.attacking and s.enemy_start is not None:
            return s.enemy_start[0] * 32 + 64, s.enemy_start[1] * 32 + 48
        return None

    def on_game_end(self, s: Optional[State], won: bool) -> None:
        log.info("game over: %s after %d build-order steps", "WIN" if won else "LOSS", self.step)
        self.reset()
