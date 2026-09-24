"""Built-in Terran templates: goliath_1fact, bio_2rax, anti_rush, mech_expand."""
from __future__ import annotations

from blackboard import Blackboard
from blackboard.sections import Goal, Posture
from bwbot import Race, TechType as T, UnitType as U, UpgradeType as Up
from mybot import opening
from mybot.opening import OpeningStep

from .base import Template, count, done, template, workers_for

SCV, MARINE, MEDIC, FIREBAT = U.Terran_SCV, U.Terran_Marine, U.Terran_Medic, U.Terran_Firebat
VULTURE, TANK, GOLIATH = U.Terran_Vulture, U.Terran_Siege_Tank_Tank_Mode, U.Terran_Goliath
DEPOT, RAX, REF, FACT = U.Terran_Supply_Depot, U.Terran_Barracks, U.Terran_Refinery, U.Terran_Factory
ARMORY, ACADEMY, EBAY, BUNKER = U.Terran_Armory, U.Terran_Academy, U.Terran_Engineering_Bay, U.Terran_Bunker
SHOP, CC, TURRET, COMSAT = U.Terran_Machine_Shop, U.Terran_Command_Center, U.Terran_Missile_Turret, U.Terran_Comsat_Station


def _i(d: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in d.items() if v > 0}


@template
class Goliath1Fact(Template):
    """One-base Goliaths (the `goliath` bot's GoliathPolicy as a template)."""

    name = "goliath_1fact"
    opening = opening.GOLIATH
    tags = frozenset({"anti_air", "mech", "one_base"})
    attack_supply = 16          # 8 goliaths
    retreat_supply = 6          # 3 goliaths

    def goal(self, bb: Blackboard) -> Goal:
        armory = done(bb, ARMORY) > 0
        fact_done = done(bb, FACT)
        return Goal(
            workers=min(24, 16 + 3 * max(1, count(bb, REF))),
            bases=1,
            buildings=_i({RAX: 1, FACT: 4 if armory else 1, ARMORY: 1 if fact_done else 0,
                          REF: 2 if fact_done >= 2 else 1}),
            units=_i({MARINE: 4, VULTURE: 0 if armory else count(bb, VULTURE) + 1, GOLIATH: 60}),
            addons=_i({SHOP: 1 if done(bb, GOLIATH) >= 2 else 0}),
            upgrades=[(int(Up.Charon_Boosters), 1), (int(Up.Terran_Vehicle_Weapons), 1),
                      (int(Up.Terran_Vehicle_Plating), 1)],
        )

    def posture(self, bb: Blackboard, prev: Posture) -> Posture:
        n = done(bb, GOLIATH)
        stance = prev.stance
        if n >= 8:
            stance = "attack"
        elif n < 3 or stance != "attack":
            stance = "defend" if bb.world.under_attack else "hold"
        return Posture(stance=stance, attack_supply=self.attack_supply, retreat_supply=self.retreat_supply)


@template
class Bio2Rax(Template):
    """Two-barracks marines into stim/medics."""

    name = "bio_2rax"
    opening = opening.MARINE
    tags = frozenset({"bio", "aggressive", "one_base"})
    attack_supply = 20
    retreat_supply = 6

    def goal(self, bb: Blackboard) -> Goal:
        rax = done(bb, RAX)
        academy = done(bb, ACADEMY) > 0
        zerg = bb.belief.enemy_race == int(Race.Zerg)
        return Goal(
            workers=min(22, workers_for(bb)),
            bases=1,
            buildings=_i({RAX: 5 if academy else 3, REF: 1 if rax >= 2 else 0, ACADEMY: 1 if rax >= 2 else 0,
                          EBAY: 1 if count(bb, MARINE) >= 12 else 0}),
            units=_i({MARINE: 60, MEDIC: 8 if academy else 0, FIREBAT: 4 if academy and zerg else 0}),
            upgrades=[(int(Up.U_238_Shells), 1), (int(Up.Terran_Infantry_Weapons), 1)],
            techs=[int(T.Stim_Packs)],
        )


@template
class AntiRush(Template):
    """Safe opening against rushes: early bunker at the choke, marines, SCV repair; then factory."""

    name = "anti_rush"
    opening = (OpeningStep(9, DEPOT), OpeningStep(10, RAX), OpeningStep(12, RAX), OpeningStep(13, BUNKER),
               OpeningStep(15, DEPOT))
    tags = frozenset({"rush_safe", "defensive", "one_base"})
    attack_supply = 30
    retreat_supply = 10

    def goal(self, bb: Blackboard) -> Goal:
        rax = done(bb, RAX)
        safe = bb.world.army_supply >= 16
        return Goal(
            workers=min(20 if not safe else 26, workers_for(bb)),
            bases=1,
            buildings=_i({RAX: 2, BUNKER: 1, REF: 1 if safe else 0, FACT: 1 if safe else 0,
                          ACADEMY: 1 if rax >= 2 and safe else 0}),
            units=_i({MARINE: 20, VULTURE: 6 if safe else 0, MEDIC: 4 if done(bb, ACADEMY) else 0}),
            upgrades=[(int(Up.U_238_Shells), 1)],
        )

    def posture(self, bb: Blackboard, prev: Posture) -> Posture:
        p = super().posture(bb, prev)
        if bb.frame < 24 * 60 * 5 and p.stance == "attack" and bb.world.army_supply < 40:
            p.stance = "hold"
        return p


@template
class MechExpand(Template):
    """Factory expand into tanks/vultures (goliaths if air), up to three bases."""

    name = "mech_expand"
    opening = (OpeningStep(9, DEPOT), OpeningStep(11, RAX), OpeningStep(12, REF), OpeningStep(16, FACT),
               OpeningStep(20, CC), OpeningStep(21, DEPOT))
    tags = frozenset({"mech", "expand", "macro"})
    attack_supply = 70
    retreat_supply = 30

    def goal(self, bb: Blackboard) -> Goal:
        bases = 2 if count(bb, SCV) < 40 else 3
        air = bb.belief.air
        facts = done(bb, FACT)
        return Goal(
            workers=min(60, workers_for(bb)),
            bases=bases,
            buildings=_i({RAX: 1, FACT: min(6, 2 + 2 * max(0, len(bb.world.depots) - 1)),
                          REF: max(1, min(len(bb.world.depots), 3)), ARMORY: 1 if facts >= 2 else 0,
                          EBAY: 1 if air > 0 else 0, TURRET: 2 if air > 2 else 0,
                          ACADEMY: 1 if facts >= 3 else 0}),
            units=_i({VULTURE: 8, TANK: 16, GOLIATH: 6 + int(2 * air)}),
            addons=_i({SHOP: min(3, facts), COMSAT: 1 if done(bb, ACADEMY) else 0}),
            upgrades=[(int(Up.Ion_Thrusters), 1), (int(Up.Terran_Vehicle_Weapons), 1),
                      (int(Up.Terran_Vehicle_Plating), 1), (int(Up.Charon_Boosters), 1),
                      (int(Up.Terran_Vehicle_Weapons), 2)],
            techs=[int(T.Tank_Siege_Mode), int(T.Spider_Mines)],
        )
