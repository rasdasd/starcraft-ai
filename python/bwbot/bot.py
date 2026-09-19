"""Bot base class."""
from __future__ import annotations

from .commands import Actions
from .observation import GameInfo, Observation
from .protocol import ClientConfig

__all__ = ["Bot", "ClientConfig"]


class Bot:
    """Subclass and override the hooks. One instance may play many consecutive games.

    Attributes:
        config: `ClientConfig` sent to the shim at the start of every match (frame_skip, speed, ...).
        game:   `GameInfo` for the current match (set before `on_start`).
    """

    config: ClientConfig = ClientConfig()
    game: GameInfo

    def on_start(self, game: GameInfo) -> None:
        """Called once per match after GameStart is received."""

    def on_frame(self, obs: Observation, act: Actions) -> None:
        """Called every `frame_skip` frames. Queue commands on `act`; they are sent when this returns."""

    def on_end(self, is_winner: bool) -> None:
        """Called when the match ends (before the next Hello, if the game auto-restarts)."""
