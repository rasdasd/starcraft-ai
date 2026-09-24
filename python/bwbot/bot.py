"""Bot base class."""
from __future__ import annotations

from .apm import ApmMeter
from .commands import Actions
from .observation import GameInfo, Observation
from .protocol import ClientConfig

__all__ = ["Bot", "ClientConfig"]


class Bot:
    """Subclass and override the hooks. One instance may play many consecutive games.

    Attributes:
        config: `ClientConfig` sent to the shim at the start of every match (frame_skip, speed, ...).
        game:   `GameInfo` for the current match (set before `on_start`).
        apm:        `ApmMeter` counting the unit commands this bot issued (maintained by the runner;
                    `apm.current` = trailing game-minute, `apm.average` = whole game).
        apm_budget: trailing-minute cap on unit commands (default 400). `None` or `<= 0` = unlimited.
                    The runner trims `act.unit_cmds` after `on_frame` so later human play is not a rewrite.
    """

    config: ClientConfig = ClientConfig()
    game: GameInfo
    apm: ApmMeter = ApmMeter()
    # Trailing-minute cap on unit commands. None = unlimited. 400 is high-master human range.
    apm_budget: float | None = 400.0

    def on_start(self, game: GameInfo) -> None:
        """Called once per match after GameStart is received."""

    def on_frame(self, obs: Observation, act: Actions) -> None:
        """Called every `frame_skip` frames. Queue commands on `act`; they are sent when this returns."""

    def on_end(self, is_winner: bool) -> None:
        """Called when the match ends (before the next Hello, if the game auto-restarts)."""
