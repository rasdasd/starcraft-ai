"""Openings as data: a list of (supply, unit_type) steps walked in order.

Policies share `Opening` instead of each copying `if supply >= 15: Build(...)`.
Midgame rules (more depots, extra factories) stay in the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from bwbot import UnitType

from .state import State

DEPOT = UnitType.Terran_Supply_Depot
RAX = UnitType.Terran_Barracks
REFINERY = UnitType.Terran_Refinery
FACTORY = UnitType.Terran_Factory


@dataclass(frozen=True)
class OpeningStep:
    at_supply: int
    unit_type: int


# Marine bio: two rax, then depots.
MARINE = (
    OpeningStep(9, DEPOT),
    OpeningStep(11, RAX),
    OpeningStep(13, RAX),
    OpeningStep(16, DEPOT),
    OpeningStep(20, DEPOT),
)

# One-base Goliath: gas and factory in the opener; armory is a midgame rule.
GOLIATH = (
    OpeningStep(9, DEPOT),
    OpeningStep(11, RAX),
    OpeningStep(12, REFINERY),
    OpeningStep(15, FACTORY),
    OpeningStep(16, DEPOT),
)


class Opening:
    """Walks `steps` in order. `next_build` is idempotent for ProductionManager.

    `base` (unit counts at game start, e.g. `State.counts`) is not counted toward the steps, so a
    Hatchery / Command Center / Overlord step means one more than we started with."""

    def __init__(self, steps: Sequence[OpeningStep], base=None) -> None:
        self.steps = list(steps)
        self.i = 0
        self.base = base

    def reset(self) -> None:
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps)

    def next_build(self, s: State) -> Optional[int]:
        """Return the unit type we should be building now, or None if waiting / finished."""
        while self.i < len(self.steps):
            step = self.steps[self.i]
            want = sum(1 for st in self.steps[: self.i + 1] if st.unit_type == step.unit_type)
            if self.base is not None:
                want += int(self.base[int(step.unit_type)])
            if s.count(step.unit_type) >= want:
                self.i += 1
                continue
            if s.supply_used >= step.at_supply:
                return step.unit_type
            return None
        return None
