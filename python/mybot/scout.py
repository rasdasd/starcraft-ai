"""ScoutManager: send one SCV to the other start, then watch.

Starts once we have a small mineral line. The scout is a claimed worker so
WorkerManager will not yank it back to mine. After a resource depot is seen,
the SCV stays near it. If the guessed start is empty, we walk the remaining
start locations. A threatened scout is recalled to mine.
"""
from __future__ import annotations

import logging
from typing import Optional

from bwbot import Actions, GameInfo, UnitFlag

from .information import InformationManager
from .state import State
from .workers import WorkerManager

log = logging.getLogger("mybot.scout")

MIN_WORKERS = 9
ARRIVE_PX = 8 * 32
THREAT_PX = 8 * 32
REISSUE_PX = 64


class ScoutManager:
    def __init__(self) -> None:
        self.worker_id: Optional[int] = None
        self.targets: list[tuple[int, int]] = []
        self.ti = 0
        self.done = False

    def reset(self) -> None:
        self.worker_id = None
        self.targets = []
        self.ti = 0
        self.done = False

    def on_start(self, game: GameInfo, info: InformationManager) -> None:
        self.reset()
        me = tuple(game.self_player.start_location)
        self.targets = [tuple(s) for s in game.start_locations.tolist() if tuple(s) != me]
        if info.enemy_start in self.targets:
            self.targets.remove(info.enemy_start)
            self.targets.insert(0, info.enemy_start)

    def update(self, s: State, act: Actions, workers: WorkerManager, info: InformationManager) -> None:
        if self.done:
            return
        if self.worker_id is None:
            dest = self._dest(info)
            if dest is None:
                self._finish(s, workers)
                return
            if len(s.workers) < MIN_WORKERS:
                return
            w = workers.claim_scout(s, dest)
            if w is None:
                return
            self.worker_id = int(w["id"])
            log.info("f%d scout #%d -> %s", s.frame, self.worker_id, dest)

        w = s.obs.unit(self.worker_id)
        if w is None:
            workers.release(self.worker_id)
            self.worker_id = None
            return

        if _threatened(s, int(w["x"]), int(w["y"])):
            log.info("f%d scout #%d threatened; recalling", s.frame, self.worker_id)
            self._finish(s, workers)
            return

        if info.has_enemy_base() and info.enemy_start is not None:
            self._go(act, w, info.enemy_start)
            return

        if self.ti >= len(self.targets):
            self._finish(s, workers)
            return

        dest = self.targets[self.ti]
        if _near(w, dest, ARRIVE_PX):
            log.info("f%d scout arrived %s (base=%s)", s.frame, dest, info.has_enemy_base())
            self.ti += 1
            return
        self._go(act, w, dest)

    def _dest(self, info: InformationManager) -> Optional[tuple[int, int]]:
        if info.has_enemy_base() and info.enemy_start is not None:
            return info.enemy_start
        if self.ti < len(self.targets):
            return self.targets[self.ti]
        return None

    def _go(self, act: Actions, w, dest: tuple[int, int]) -> None:
        tx, ty = dest[0] * 32 + 64, dest[1] * 32 + 48
        idle = bool(int(w["flags"]) & int(UnitFlag.Idle))
        if idle or abs(int(w["order_target_x"]) - tx) + abs(int(w["order_target_y"]) - ty) > REISSUE_PX:
            act.move(w, tx, ty)

    def _finish(self, s: State, workers: WorkerManager) -> None:
        if self.worker_id is not None:
            workers.release(self.worker_id)
            log.info("f%d scout #%d recalled", s.frame, self.worker_id)
            self.worker_id = None
        self.done = True


def _near(w, dest: tuple[int, int], radius: int) -> bool:
    dx = dest[0] * 32 + 64 - int(w["x"])
    dy = dest[1] * 32 + 48 - int(w["y"])
    return dx * dx + dy * dy < radius * radius


def _threatened(s: State, x: int, y: int) -> bool:
    if len(s.enemies) == 0:
        return False
    d = (s.enemies["x"] - x) ** 2 + (s.enemies["y"] - y) ** 2
    return bool((d < THREAT_PX * THREAT_PX).any())
