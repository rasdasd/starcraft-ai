"""GoliathPolicy: Terran mech that techs to Goliaths and fights with them.

The opener is `opening.GOLIATH` (depot, rax, gas, factory, depot). Midgame rules
add armory, more factories, Goliaths, Charon Boosters. Combat stance is Attack/Rally;
CombatCommander handles defend / hunt-air / no-suicide.
"""
from __future__ import annotations

import logging
from typing import Optional

from bwbot import Race, UnitType, UpgradeType

from mybot import opening
from mybot.policy import Attack, Build, BuildAddon, Intent, Rally, Train, Upgrade
from mybot.state import State

log = logging.getLogger("goliath.policy")

SCV = UnitType.Terran_SCV
DEPOT = UnitType.Terran_Supply_Depot
RAX = UnitType.Terran_Barracks
REFINERY = UnitType.Terran_Refinery
FACTORY = UnitType.Terran_Factory
ARMORY = UnitType.Terran_Armory
MACHINE_SHOP = UnitType.Terran_Machine_Shop
GOLIATH = UnitType.Terran_Goliath
VULTURE = UnitType.Terran_Vulture
MARINE = UnitType.Terran_Marine

CHARON = UpgradeType.Charon_Boosters
VEHICLE_WEAPONS = UpgradeType.Terran_Vehicle_Weapons
VEHICLE_PLATING = UpgradeType.Terran_Vehicle_Plating


class GoliathPolicy:
    MAX_WORKERS = 24
    MAX_FACTORIES = 4
    EARLY_MARINES = 4
    ATTACK_GOLIATHS = 8
    RETREAT_GOLIATHS = 3
    HUNT_AIR_GOLIATHS = 2

    def __init__(self) -> None:
        self.opening = opening.Opening(opening.GOLIATH)
        self.attacking = False

    def reset(self) -> None:
        self.__init__()

    def decide(self, s: State) -> list[Intent]:
        if s.game.self_race != Race.Terran:
            log.warning("GoliathPolicy is Terran-only; game race is %s (set race = Terran in bwapi.ini)",
                        s.game.self_race.name)
            return []
        out: list[Intent] = []

        nxt = self.opening.next_build(s)
        if nxt is not None:
            out.append(Build(nxt))
        elif self.opening.done:
            depots_in_progress = s.count(DEPOT) - s.count_completed(DEPOT)
            if s.supply_left <= 6 and depots_in_progress == 0 and s.supply_total < 200:
                out.append(Build(DEPOT))

        if s.count_completed(FACTORY) > 0 and s.count(ARMORY) == 0:
            out.append(Build(ARMORY))
        if (s.count_completed(ARMORY) > 0 and s.count(FACTORY) < self.MAX_FACTORIES
                and s.minerals >= 300):
            out.append(Build(FACTORY))
        if s.count_completed(FACTORY) >= 2 and s.count(REFINERY) < 2 and len(s.obs.geysers):
            out.append(Build(REFINERY))

        workers_wanted = 16 + 3 * max(1, s.count(REFINERY))
        if s.count(SCV) < min(self.MAX_WORKERS, workers_wanted):
            out.append(Train(SCV))

        if s.count_completed(ARMORY) > 0:
            for _ in range(s.count_completed(FACTORY)):
                out.append(Train(GOLIATH))
        elif s.count_completed(FACTORY) > 0:
            out.append(Train(VULTURE))
        elif s.count_completed(RAX) > 0 and s.count(MARINE) < self.EARLY_MARINES:
            out.append(Train(MARINE))

        if (s.count_completed(ARMORY) > 0 and s.count_completed(FACTORY) > 0
                and s.count(MACHINE_SHOP) == 0 and s.count_completed(GOLIATH) >= 2):
            out.append(BuildAddon(MACHINE_SHOP))

        if s.count_completed(ARMORY) > 0:
            if (s.count_completed(MACHINE_SHOP) > 0 and not s.has_upgrade(CHARON)
                    and not s.upgrading(CHARON)):
                out.append(Upgrade(CHARON))
            elif (s.count_completed(GOLIATH) >= 4 and not s.upgrading(VEHICLE_WEAPONS)
                    and not s.has_upgrade(VEHICLE_WEAPONS, 1)):
                out.append(Upgrade(VEHICLE_WEAPONS))
            elif (s.has_upgrade(VEHICLE_WEAPONS, 1) and not s.upgrading(VEHICLE_PLATING)
                    and not s.has_upgrade(VEHICLE_PLATING, 1)):
                out.append(Upgrade(VEHICLE_PLATING))

        n_goliaths = s.count_completed(GOLIATH)
        if n_goliaths >= self.ATTACK_GOLIATHS:
            self.attacking = True
        elif n_goliaths < self.RETREAT_GOLIATHS:
            self.attacking = False
        hunt_air = len(s.enemy_flyers) > 0 and n_goliaths >= self.HUNT_AIR_GOLIATHS

        target = self._army_target(s, hunt_air)
        if target is not None:
            out.append(Attack(*target))
        else:
            out.append(Rally(*s.rally_point))
        return out

    def _army_target(self, s: State, hunt_air: bool) -> Optional[tuple[int, int]]:
        if hunt_air:
            e = s.obs.nearest(s.enemy_flyers, s.main_tile[0] * 32, s.main_tile[1] * 32)
            if e is not None:
                return int(e["x"]), int(e["y"])
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
        goliaths = s.count_completed(GOLIATH) if s is not None else 0
        log.info("game over: %s after %d opening steps, %d goliaths",
                 "WIN" if won else "LOSS", self.opening.i, goliaths)
        self.reset()
