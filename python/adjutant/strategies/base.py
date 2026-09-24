"""Strategy templates: an opening plus goal/posture rules over the board.

A template never issues commands or picks individual builds; it says what the army and tech should
look like (`Goal`) and when to fight (`Posture`). The production planner turns that into builds.
Templates are the "pre-made strategies" a learned selector chooses among or blends.
"""
from __future__ import annotations

from typing import Sequence

from blackboard import Blackboard
from blackboard.sections import Goal, Posture
from bwbot import Race, UnitType as U
from mybot.opening import OpeningStep


class Template:
    name: str = "base"
    race: int = int(Race.Terran)
    opening: Sequence[OpeningStep] = ()
    tags: frozenset[str] = frozenset()
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


def workers_for(bb: Blackboard, per_base: int = 16, cap: int = 60) -> int:
    """Mineral saturation per base plus three per refinery."""
    bases = max(1, len(bb.world.depots))
    return min(cap, per_base * bases + 3 * max(1, count(bb, U.Terran_Refinery)))


TEMPLATES: dict[str, Template] = {}


def template(cls):
    TEMPLATES[cls.name] = cls()
    return cls


def get(name: str) -> Template:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; have {sorted(TEMPLATES)}")
    return TEMPLATES[name]


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
