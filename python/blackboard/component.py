"""Component (knowledge source) base class, decision phases, and shared priority levels.

A component fills one *slot* ("strategy", "belief", ...). It declares the board sections it reads
and writes; the scheduler validates those contracts and orders components so writers run before
their same-phase readers. Any implementation (scripted, learned, hybrid) can fill a slot.
"""
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .board import Blackboard


class Phase(IntEnum):
    SENSE = 0     # observation -> world / meta / belief
    DECIDE = 1    # crisis, strategy, engagement, tactics
    PLAN = 2      # production plan, scouting targets
    ACT = 3       # executors that own units and issue commands
    REPORT = 4    # recorder, HUD


class Priority(IntEnum):
    """Shared scale for unit leases, production requests, and command ordering under the APM cap."""

    BACKGROUND = 10
    ECONOMY = 30
    SCOUT = 40
    NORMAL = 50
    PRODUCTION = 60
    CONSTRUCTION = 70
    COMBAT = 80
    SUPPLY = 90
    CRISIS = 100


class Component:
    slot: str = ""
    phase: Phase = Phase.DECIDE
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    period_frames: int = 0             # 0 = every decision
    triggers: tuple[str, ...] = ()     # board events that force a tick even if not due
    priority: int = Priority.NORMAL    # leases and command ordering
    order: int = 0                     # tie-break inside a phase when there is no data dependency
    fallback: Optional["Component"] = None   # used after repeated failures (learned -> scripted)

    @property
    def name(self) -> str:
        return type(self).__name__

    def on_start(self, bb: "Blackboard") -> None:
        """Called once per game, after the board is created."""

    def tick(self, bb: "Blackboard") -> None:
        """Called on decisions where the component is due."""

    def on_end(self, bb: "Blackboard", won: bool) -> None:
        """Called when the game ends."""

    def describe(self) -> dict:
        """Recorded in the game log header (implementation name, model hash, ...)."""
        return {"impl": self.name}

    def __repr__(self) -> str:
        return f"{self.name}(slot={self.slot!r})"
