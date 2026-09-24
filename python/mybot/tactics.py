"""Discrete army tactics. A learned policy picks one of these, not a pixel.

CombatCommander still enforces defend / no-suicide / hunt-air. These labels are what
we log and train on so the model cannot invent illegal micro.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional, Union

from .policy import Attack, Intent, Rally
from .state import State

TACTIC_NAMES = ("hold", "defend", "push", "hunt_air")


class Tactic(IntEnum):
    HOLD = 0
    DEFEND = 1
    PUSH = 2
    HUNT_AIR = 3


ArmyIntent = Union[Attack, Rally]


def from_intents(intents: list[Intent], s: State) -> Tactic:
    """Teacher label: what the scripted policy asked the army to do this decision."""
    attacking = any(isinstance(i, Attack) for i in intents)
    if not attacking:
        return Tactic.HOLD
    if s.under_attack:
        return Tactic.DEFEND
    if len(s.enemy_flyers):
        return Tactic.HUNT_AIR
    return Tactic.PUSH


def to_intent(tactic: Tactic, s: State) -> Optional[ArmyIntent]:
    if tactic is Tactic.HOLD:
        return Rally(*s.rally_point)
    if tactic is Tactic.DEFEND:
        if len(s.enemies):
            e = s.obs.nearest(s.enemies, s.main_tile[0] * 32, s.main_tile[1] * 32)
            if e is not None:
                return Attack(int(e["x"]), int(e["y"]))
        return Rally(*s.rally_point)
    if tactic is Tactic.HUNT_AIR and len(s.enemy_flyers):
        e = s.obs.nearest(s.enemy_flyers, s.main_tile[0] * 32, s.main_tile[1] * 32)
        if e is not None:
            return Attack(int(e["x"]), int(e["y"]))
    if s.enemy_buildings:
        return Attack(s.enemy_buildings[0][1], s.enemy_buildings[0][2])
    if s.enemy_start is not None:
        return Attack(s.enemy_start[0] * 32 + 64, s.enemy_start[1] * 32 + 48)
    if len(s.enemies):
        e = s.obs.nearest(s.enemies, s.main_tile[0] * 32, s.main_tile[1] * 32)
        if e is not None:
            return Attack(int(e["x"]), int(e["y"]))
    return Rally(*s.rally_point)


def replace_army(intents: list[Intent], tactic: Tactic, s: State) -> list[Intent]:
    rest = [i for i in intents if not isinstance(i, (Attack, Rally))]
    army = to_intent(tactic, s)
    return rest + ([army] if army is not None else [])
