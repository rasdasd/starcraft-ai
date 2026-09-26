"""Strategy templates ("builds"): an opening plus goal/posture rules over the board.

A template never issues commands or picks individual builds; it says what the army and tech should
look like (`Goal`) and when to fight (`Posture`). The production planner turns that into builds.
Templates are the builds a selector chooses among or blends. Most are JSON documents
(`spec.BuildSpec`, loaded from `spec.build_dirs()`); a subclass of `Template` registered with
`@template` can express rules the JSON format cannot.
"""
from __future__ import annotations

from typing import Sequence

from blackboard import Blackboard
from blackboard.sections import Goal, Posture
from bwbot import Race, UnitType as U

from .opening import OpeningStep


class Template:
    name: str = "base"
    race: int = int(Race.Terran)
    opening: Sequence[OpeningStep] = ()
    tags: frozenset[str] = frozenset()
    default_vs: tuple[str, ...] = ()      # enemy races ("Zerg", ..., "Unknown") the rule selector picks it for
    description: str = ""
    source: str = "python"
    attack_supply: int = 40
    retreat_supply: int = 12

    def goal(self, bb: Blackboard) -> Goal:
        raise NotImplementedError

    def posture(self, bb: Blackboard, prev: Posture) -> Posture:
        """Default: attack at `attack_supply`, fall back below `retreat_supply` (hysteresis)."""
        army = bb.world.army_supply
        stance = prev.stance
        if army >= self.attack_supply:
            stance = "attack"
        elif army < self.retreat_supply or stance not in ("attack", "contain"):
            stance = "defend" if bb.world.under_attack else "hold"
        return Posture(stance=stance, attack_supply=self.attack_supply, retreat_supply=self.retreat_supply)

    def opening_steps(self) -> list[tuple[int, int]]:
        return [(s.at_supply, int(s.unit_type)) for s in self.opening]

    def __repr__(self) -> str:
        return f"Template({self.name})"


def count(bb: Blackboard, t: int) -> int:
    return bb.world.count(t)


def done(bb: Blackboard, t: int) -> int:
    return bb.world.count_completed(t)


REFINERY = {int(Race.Terran): int(U.Terran_Refinery), int(Race.Zerg): int(U.Zerg_Extractor),
            int(Race.Protoss): int(U.Protoss_Assimilator)}
HALL = {int(Race.Terran): int(U.Terran_Command_Center), int(Race.Zerg): int(U.Zerg_Hatchery),
        int(Race.Protoss): int(U.Protoss_Nexus)}


def workers_for(bb: Blackboard, per_base: int = 16, cap: int = 60, per_gas: int = 3, ahead: int = 1) -> int:
    """Mineral saturation for our finished mining bases plus `ahead` more (the next expansion, or
    the one under construction), plus `per_gas` per refinery (of our race): workers are ready when
    a hall finishes instead of being made for it afterwards."""
    mb = bb.macro.bases
    if mb:
        bases = sum(1 for b in mb if b.completed and b.patches > 0)
    else:
        hall = HALL.get(int(bb.game.self_race), int(U.Terran_Command_Center))
        bases = done(bb, hall)
    bases = max(1, bases) + ahead
    ref = REFINERY.get(int(bb.game.self_race), int(U.Terran_Refinery))
    return min(cap, per_base * bases + per_gas * max(1, count(bb, ref)))


TEMPLATES: dict[str, Template] = {}


def template(cls):
    TEMPLATES[cls.name] = cls()
    return cls


def get(name: str) -> Template:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; have {sorted(TEMPLATES)}")
    return TEMPLATES[name]


def validate_goal(goal: Goal, tree, race: int) -> list[str]:
    """Contract for Goal: units are trainable non-buildings of our race, buildings are non-addon
    buildings, addons are addons, upgrade levels exist, techs exist, counts are non-negative."""
    errs = []
    name = tree.game.type_name

    def ours(t: int) -> bool:
        return tree.known_unit(t) and tree.race(t) == race

    for t, n in goal.units.items():
        if not ours(t) or tree.is_building(t):
            errs.append(f"units: {name(t)} is not a {race} unit")
        if n < 0:
            errs.append(f"units: negative count for {name(t)}")
    for t, n in goal.buildings.items():
        if not ours(t) or not tree.is_building(t) or tree.is_addon(t):
            errs.append(f"buildings: {name(t)} is not a buildable structure")
        if n < 0:
            errs.append(f"buildings: negative count for {name(t)}")
    for t, n in goal.addons.items():
        if not ours(t) or not tree.is_addon(t):
            errs.append(f"addons: {name(t)} is not an addon")
    for u, lvl in goal.upgrades:
        if not 1 <= lvl <= tree.max_level(u):
            errs.append(f"upgrades: {u} level {lvl} out of range")
    for t in goal.techs:
        if t not in tree.techs:
            errs.append(f"techs: unknown tech {t}")
    if goal.workers < 0 or goal.bases < 1:
        errs.append("workers must be >= 0 and bases >= 1")
    return errs


def validate_template(t: Template, tree) -> list[str]:
    """Static checks: opening steps are buildings of the template's race with non-decreasing supply."""
    errs = []
    last = 0
    for supply, ut in t.opening_steps():
        if not tree.known_unit(ut) or tree.race(ut) != t.race:
            errs.append(f"opening: {ut} is not a {t.race} type")
        if supply < last:
            errs.append(f"opening: supply {supply} after {last}")
        last = supply
    if t.retreat_supply > t.attack_supply:
        errs.append("retreat_supply > attack_supply")
    return errs


def blend_goals(goals: Sequence[Goal], weights: Sequence[float]) -> Goal:
    """Weighted mix of goals: counts are rounded weighted means; upgrade/tech order follows the
    heaviest template, with others' items appended."""
    tot = sum(weights) or 1.0
    w = [x / tot for x in weights]

    def mix(key: str) -> dict[int, int]:
        keys = set().union(*(getattr(g, key).keys() for g in goals))
        out = {}
        for k in keys:
            v = sum(wi * getattr(g, key).get(k, 0) for g, wi in zip(goals, w))
            if v >= 0.5:
                out[k] = int(round(v))
        return out

    order = sorted(range(len(goals)), key=lambda i: -w[i])
    ups: list[tuple[int, int]] = []
    techs: list[int] = []
    for i in order:
        if w[i] < 0.15 and i != order[0]:
            continue
        ups += [u for u in goals[i].upgrades if u not in ups]
        techs += [t for t in goals[i].techs if t not in techs]
    return Goal(units=mix("units"), buildings=mix("buildings"), addons=mix("addons"), upgrades=ups, techs=techs,
                workers=int(round(sum(g.workers * wi for g, wi in zip(goals, w)))),
                bases=max(1, int(round(sum(g.bases * wi for g, wi in zip(goals, w))))))
