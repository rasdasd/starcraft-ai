"""CombatCommander: defend the main, push with the army, hunt air.

Policy still says Attack or Rally. This manager applies the rules other bots treat
as non-negotiable: do not leave home with a handful of units, peel air-capable
units onto flyers, and walk toward last-known buildings when the enemy is in fog.
"""
from __future__ import annotations

from typing import Optional, Union

from bwbot import Actions, UnitFlag
from bwbot.enums import Order, WeaponType

from .information import InformationManager
from .policy import Attack, Rally
from .state import State

ArmyOrder = Union[Attack, Rally]

HOME_PX = 28 * 32
MIN_PUSH = 3          # never leave home with fewer than this (except to defend)
REISSUE_PX = 160


class CombatCommander:
    def update(self, s: State, act: Actions, order: Optional[ArmyOrder],
               info: InformationManager) -> str:
        """Issue army commands. Returns the stance name for the HUD."""
        if len(s.army) == 0:
            return "none"
        homes = [(s.main_tile[0] * 32, s.main_tile[1] * 32)]
        if s.natural_tile is not None:
            homes.append((s.natural_tile[0] * 32, s.natural_tile[1] * 32))
        home_x, home_y = homes[0]
        at_home = any(len(_enemies_near(s, x, y, HOME_PX)) > 0 for x, y in homes)
        defending = s.under_attack or at_home

        if defending:
            tgt = _visible_target(s, home_x, home_y) or _fog_target(s, info, home_x, home_y)
            if tgt is not None:
                self._squad(s, act, tgt, hunt_air=True, peel=True)
            return "defend"

        pushing = isinstance(order, Attack) and len(s.army) >= MIN_PUSH
        if pushing:
            ax = int(s.army["x"].mean())
            ay = int(s.army["y"].mean())
            tgt = _visible_target(s, ax, ay) or _fog_target(s, info, ax, ay)
            if tgt is None and isinstance(order, Attack):
                tgt = (order.x, order.y)
            if tgt is not None:
                self._squad(s, act, tgt, hunt_air=True)
            return "attack"

        if isinstance(order, Rally):
            self._rally(s, act, order.x, order.y)
            return "rally"
        return "hold"

    def _squad(self, s: State, act: Actions, dest: tuple[int, int], hunt_air: bool,
               peel: bool = False) -> None:
        flyers = s.enemy_flyers
        focus = None
        if hunt_air and len(flyers) and len(s.army):
            ax, ay = int(s.army["x"].mean()), int(s.army["y"].mean())
            focus = s.obs.nearest(flyers, ax, ay)
        for u in s.army:
            if not peel and not (int(u["flags"]) & int(UnitFlag.Idle)):
                continue
            if peel and not _needs_peel(u, dest):
                continue
            if focus is not None and _has_air_weapon(s.game, int(u["type"])):
                act.attack(u, focus)
            elif len(s.enemies):
                e = s.obs.nearest(s.enemies, int(u["x"]), int(u["y"]))
                if e is not None:
                    act.attack(u, e)
                else:
                    act.attack_move(u, dest[0], dest[1])
            else:
                act.attack_move(u, dest[0], dest[1])

    def _rally(self, s: State, act: Actions, x: int, y: int) -> None:
        for u in s.obs.idle(s.army):
            if abs(int(u["x"]) - x) + abs(int(u["y"]) - y) > REISSUE_PX:
                act.move(u, x, y)


def _enemies_near(s: State, x: int, y: int, radius: int):
    if len(s.enemies) == 0:
        return s.enemies
    d = (s.enemies["x"] - x) ** 2 + (s.enemies["y"] - y) ** 2
    return s.enemies[d < radius * radius]


def _visible_target(s: State, x: int, y: int) -> Optional[tuple[int, int]]:
    if len(s.enemies) == 0:
        return None
    e = s.obs.nearest(s.enemies, x, y)
    return (int(e["x"]), int(e["y"])) if e is not None else None


def _fog_target(s: State, info: InformationManager, x: int, y: int) -> Optional[tuple[int, int]]:
    b = info.nearest_building(x, y)
    if b is not None:
        return b
    if s.enemy_start is not None:
        return s.enemy_start[0] * 32 + 64, s.enemy_start[1] * 32 + 48
    return None


def _has_air_weapon(game, unit_type: int) -> bool:
    weapon = int(game.unit_types["air_weapon"][int(unit_type)])
    return weapon >= 0 and weapon != int(WeaponType.None_)


def _needs_peel(u, dest: tuple[int, int]) -> bool:
    """True if this unit is not already fighting near `dest` (so it should come home)."""
    order = int(u["order"])
    if order == int(Order.AttackUnit):
        return False
    if order == int(Order.AttackMove):
        dx = int(u["order_target_x"]) - dest[0]
        dy = int(u["order_target_y"]) - dest[1]
        return dx * dx + dy * dy > (12 * 32) ** 2
    return True
