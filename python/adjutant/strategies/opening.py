"""Openings as data: (supply, unit_type) steps walked in order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class OpeningStep:
    at_supply: int
    unit_type: int


class Opening:
    """Walks `steps` in order against `world` counts (which include units in production).

    `base` (unit counts when the opening started) is not counted toward the steps, so a Hatchery /
    Command Center / Overlord step means one more than we started with."""

    def __init__(self, steps: Sequence[OpeningStep], base=None) -> None:
        self.steps = list(steps)
        self.i = 0
        self.base = base

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps)

    def next_build(self, world) -> Optional[int]:
        """The unit type the opening wants started now, or None while waiting for supply / finished."""
        while self.i < len(self.steps):
            step = self.steps[self.i]
            want = sum(1 for st in self.steps[: self.i + 1] if st.unit_type == step.unit_type)
            if self.base is not None:
                want += int(self.base[int(step.unit_type)])
            if world.count(step.unit_type) >= want:
                self.i += 1
                continue
            return step.unit_type if world.supply_used >= step.at_supply else None
        return None
