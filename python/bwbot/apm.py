"""APM (actions per minute) accounting for the bot's own commands.

The runner records every unit command the bot queues on `Actions` and exposes the meter as
`bot.apm`. A minute is game time (24 frames/s at "fastest"), not wall time, so numbers are
comparable across `local_speed` settings and between native StarCraft and OpenBW.

Two views are kept:
  - `current`: actions over the last `window_frames` (default 1 game minute) - what a human
    APM counter shows; scaled up while less than a window has elapsed.
  - `average`: actions over the whole game so far.

Only unit commands (`act.move`, `act.train`, `act.build`, ...) count. Game commands (speed,
text) and debug drawing are free. `Observation.game_apm` is the game's own counter
(`Broodwar->getAPM()`), useful as a cross-check; a future APM budget will be enforced on this
meter since it counts exactly what the bot emitted.
"""
from __future__ import annotations

from collections import deque

FRAMES_PER_MINUTE = 24 * 60


class ApmMeter:
    def __init__(self, window_frames: int = FRAMES_PER_MINUTE, min_window_frames: int = 24 * 10) -> None:
        self.window_frames = window_frames
        self.min_window_frames = min_window_frames    # avoid absurd APM in the first seconds
        self.reset()

    def reset(self, start_frame: int = 0) -> None:
        self.start_frame = start_frame
        self.frame = start_frame
        self.total = 0                                # actions this game
        self.last = 0                                 # actions on the most recent decision
        self._window: deque[tuple[int, int]] = deque()   # (frame, actions)
        self._window_sum = 0

    def record(self, frame: int, actions: int) -> None:
        self.frame = frame
        self.last = actions
        self.total += actions
        if actions:
            self._window.append((frame, actions))
            self._window_sum += actions
        cutoff = frame - self.window_frames
        while self._window and self._window[0][0] <= cutoff:
            self._window_sum -= self._window.popleft()[1]

    @property
    def elapsed_frames(self) -> int:
        return max(self.frame - self.start_frame, 0)

    @property
    def current(self) -> float:
        """APM over the trailing window (game time)."""
        span = max(min(self.elapsed_frames, self.window_frames), self.min_window_frames)
        return self._window_sum * FRAMES_PER_MINUTE / span

    @property
    def average(self) -> float:
        """APM over the whole game so far."""
        return self.total * FRAMES_PER_MINUTE / max(self.elapsed_frames, self.min_window_frames)

    def __repr__(self) -> str:
        return f"ApmMeter(current={self.current:.0f}, average={self.average:.0f}, total={self.total})"
