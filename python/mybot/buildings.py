"""BuildingManager: construction as a state machine, not a one-shot command.

A task lives until the structure is completed (or the match ends). Timeouts change the
tile and re-issue; they do not drop the building. A dead Terran SCV is replaced so an
incomplete building can be finished.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bwbot import Actions, UnitFlag
from bwbot.enums import Order

from . import macro
from .macro import Placer
from .state import State
from .workers import WorkerManager

log = logging.getLogger("mybot.buildings")

UNASSIGNED = "unassigned"
ASSIGNED = "assigned"
CONSTRUCTING = "constructing"

REISSUE_FRAMES = 8
TILE_TIMEOUT_FRAMES = 24 * 15      # at the site without construction starting
TRAVEL_TIMEOUT_FRAMES = 24 * 60    # builder never reached the site
ARRIVE_PX = 5 * 32
WALK_TILES = 6                     # move (not Build) until this close to the site centre
EXACT_RETRIES = 3                  # then search near the exact tile instead


@dataclass
class BuildTask:
    unit_type: int
    status: str = UNASSIGNED
    worker_id: Optional[int] = None
    tile: Optional[tuple[int, int]] = None
    building_id: Optional[int] = None
    started_frame: int = 0
    last_issue_frame: int = -10_000
    query_id: Optional[int] = None
    near: Optional[tuple[int, int]] = None   # preferred tile (default: main)
    exact: bool = False                      # `near` is the exact top-left tile (skip the placer)
    arrived_frame: int = 0
    failures: int = 0


class BuildingManager:
    def __init__(self) -> None:
        self.tasks: list[BuildTask] = []
        self._next_query = 1

    def reset(self) -> None:
        self.tasks.clear()
        self._next_query = 1

    def add(self, unit_type: int, near: Optional[tuple[int, int]] = None, exact: bool = False) -> BuildTask:
        task = BuildTask(int(unit_type), near=near, exact=exact and near is not None)
        self.tasks.append(task)
        return task

    def cancel(self, unit_type: int, workers: Optional[WorkerManager] = None,
               placer: Optional[Placer] = None, game=None) -> int:
        """Drop tasks of `unit_type` that have not started construction. Returns how many."""
        keep, dropped = [], 0
        for t in self.tasks:
            if t.unit_type == int(unit_type) and t.status != CONSTRUCTING:
                dropped += 1
                if t.worker_id is not None and workers is not None:
                    workers.release(t.worker_id)
                if t.tile is not None and placer is not None and game is not None and not t.exact:
                    placer.release(game, t.unit_type, t.tile)
            else:
                keep.append(t)
        self.tasks = keep
        return dropped

    def pending_count(self, unit_type: int) -> int:
        return sum(1 for t in self.tasks if t.unit_type == int(unit_type))

    def starting(self, unit_type: int) -> bool:
        """True if we have a task of this type that has not begun construction yet."""
        return any(t.unit_type == int(unit_type) and t.status != CONSTRUCTING for t in self.tasks)

    def reserved_minerals(self, game) -> int:
        return sum(int(game.unit_types["mineral_price"][t.unit_type])
                   for t in self.tasks if t.status != CONSTRUCTING)

    def reserved_gas(self, game) -> int:
        return sum(int(game.unit_types["gas_price"][t.unit_type])
                   for t in self.tasks if t.status != CONSTRUCTING)

    def update(self, s: State, act: Actions, workers: WorkerManager, placer: Placer) -> None:
        self._apply_placement_results(s, placer)
        keep: list[BuildTask] = []
        for task in self.tasks:
            if self._step(task, s, act, workers, placer):
                keep.append(task)
            else:
                self._finish(task, s, workers, placer)
        self.tasks = keep

    def _apply_placement_results(self, s: State, placer: Placer) -> None:
        results = getattr(s.obs, "placement_results", None)
        if not results:
            return
        by_id = {t.query_id: t for t in self.tasks if t.query_id is not None and t.tile is None}
        for req_id, ok, tx, ty in results:
            task = by_id.get(req_id)
            if task is None or not ok:
                continue
            tile = (int(tx), int(ty))
            if tile in placer.failed:
                continue
            task.tile = tile
            placer.reserve(s.game, task.unit_type, tile)
            log.info("f%d shim placed %s at %s", s.frame, s.game.type_name(task.unit_type), tile)

    def _step(self, task: BuildTask, s: State, act: Actions, workers: WorkerManager,
              placer: Placer) -> bool:
        if self._is_complete(task, s):
            return False

        started = self._find_started(task, s)
        if started is not None:
            bid = int(started["id"])
            if task.status != CONSTRUCTING:
                log.info("f%d construction started %s #%d", s.frame, s.game.type_name(task.unit_type), bid)
            task.status = CONSTRUCTING
            task.building_id = bid
            if task.tile is None:
                task.tile = macro.unit_tile(s.game, started)
            return True

        if task.status == CONSTRUCTING:
            self._replace_dead_builder(task, s, act, workers)
            return True

        if task.tile is None:
            if task.exact and task.near is not None:
                tile = task.near
            else:
                tile = placer.find(s.obs, task.unit_type, task.near or s.main_tile)
            if tile is None:
                self._request_shim_tile(task, s, act)
                return True
            task.tile = tile
            placer.reserve(s.game, task.unit_type, tile)

        if task.worker_id is None or s.obs.unit(task.worker_id) is None:
            if task.worker_id is not None:
                workers.release(task.worker_id)
            w = workers.claim_builder(s, task.tile)
            if w is None:
                return True
            task.worker_id = int(w["id"])
            task.status = ASSIGNED
            if task.started_frame == 0:
                task.started_frame = s.frame

        w = s.obs.unit(task.worker_id)
        constructing = (w is not None and (
            (int(w["flags"]) & int(UnitFlag.Constructing))
            or int(w["order"]) == int(Order.ConstructingBuilding)
            or int(w["build_type"]) == task.unit_type))
        if w is not None and not constructing and s.frame - task.last_issue_frame >= REISSUE_FRAMES:
            # BWAPI refuses Build on unexplored tiles: walk there first
            if self._needs_walk(task, s, w):
                ut = s.game.unit_types
                act.move(w, task.tile[0] * 32 + int(ut["tile_width"][task.unit_type]) * 16,
                         task.tile[1] * 32 + int(ut["tile_height"][task.unit_type]) * 16)
            else:
                act.build(w, task.unit_type, task.tile[0], task.tile[1])
            task.last_issue_frame = s.frame

        if w is not None and not task.arrived_frame:
            ut = s.game.unit_types
            cx = task.tile[0] * 32 + int(ut["tile_width"][task.unit_type]) * 16
            cy = task.tile[1] * 32 + int(ut["tile_height"][task.unit_type]) * 16
            if (int(w["x"]) - cx) ** 2 + (int(w["y"]) - cy) ** 2 <= ARRIVE_PX ** 2:
                task.arrived_frame = s.frame
        at_site = task.arrived_frame and s.frame - task.arrived_frame > TILE_TIMEOUT_FRAMES
        travel = task.started_frame and s.frame - task.started_frame > TRAVEL_TIMEOUT_FRAMES
        if at_site or travel:
            task.failures += 1
            if task.exact and task.failures >= EXACT_RETRIES:
                task.exact = False                       # search near it from now on
                how = "searching near it"
            else:
                how = "the same tile" if task.exact else "another tile"
            wpos = f"({int(w['x']) // 32},{int(w['y']) // 32}) order {int(w['order'])}" if w is not None else "none"
            log.warning("f%d %s at %s timed out (%s, builder %s); retrying %s", s.frame,
                        s.game.type_name(task.unit_type), task.tile, "at site" if at_site else "travel", wpos, how)
            placer.release(s.game, task.unit_type, task.tile)
            if not task.exact:
                placer.fail(task.tile)
            task.tile = None
            task.query_id = None
            task.status = UNASSIGNED
            task.started_frame = s.frame
            task.arrived_frame = 0
        return True

    @staticmethod
    def _needs_walk(task: BuildTask, s: State, w) -> bool:
        """Walk to far sites and only then Build: BWAPI (OpenBW especially) refuses long-distance
        or unexplored/unaffordable Build orders, and a refused order leaves the SCV mining."""
        tx, ty = task.tile
        ut = s.game.unit_types
        cx = tx * 32 + int(ut["tile_width"][task.unit_type]) * 16
        cy = ty * 32 + int(ut["tile_height"][task.unit_type]) * 16
        return (int(w["x"]) - cx) ** 2 + (int(w["y"]) - cy) ** 2 > (WALK_TILES * 32) ** 2

    def _request_shim_tile(self, task: BuildTask, s: State, act: Actions) -> None:
        if task.query_id is not None or not hasattr(act, "get_build_location"):
            return
        task.query_id = self._next_query
        self._next_query += 1
        near = task.near or s.main_tile
        act.get_build_location(task.query_id, task.unit_type, near)
        log.info("f%d asking shim for %s near %s", s.frame, s.game.type_name(task.unit_type), near)

    def _replace_dead_builder(self, task: BuildTask, s: State, act: Actions,
                              workers: WorkerManager) -> None:
        if task.worker_id is not None and s.obs.unit(task.worker_id) is not None:
            return
        if task.worker_id is not None:
            workers.release(task.worker_id)
            task.worker_id = None
        if task.building_id is None:
            return
        b = s.obs.unit(task.building_id)
        if b is None:
            return
        w = workers.claim_builder(s, (int(b["x"]) // 32, int(b["y"]) // 32))
        if w is None:
            return
        task.worker_id = int(w["id"])
        act.right_click(w, b)
        log.info("f%d replacement SCV #%d finishing %s #%d", s.frame, task.worker_id,
                 s.game.type_name(task.unit_type), task.building_id)

    def _is_complete(self, task: BuildTask, s: State) -> bool:
        if task.building_id is None:
            return False
        u = s.obs.unit(task.building_id)
        if u is None:
            return s.count_completed(task.unit_type) > 0 and task.status == CONSTRUCTING
        return bool(int(u["flags"]) & int(UnitFlag.Completed))

    def _find_started(self, task: BuildTask, s: State):
        claimed = {t.building_id for t in self.tasks if t.building_id is not None}
        for u in s.obs.my_units:
            if int(u["type"]) != task.unit_type:
                continue
            if int(u["flags"]) & int(UnitFlag.Completed):
                continue
            uid = int(u["id"])
            if uid in claimed and uid != task.building_id:
                continue
            if task.tile is not None:
                ut = macro.unit_tile(s.game, u)
                if abs(ut[0] - task.tile[0]) > 2 or abs(ut[1] - task.tile[1]) > 2:
                    continue
            return u
        if task.worker_id is not None:
            w = s.obs.unit(task.worker_id)
            if w is not None and int(w["build_type"]) == task.unit_type:
                return None  # money spent / order accepted; building unit may appear next event
            if w is not None and (int(w["flags"]) & int(UnitFlag.Constructing)
                                  or int(w["order"]) == int(Order.ConstructingBuilding)):
                return None
        return None

    def _finish(self, task: BuildTask, s: State, workers: WorkerManager, placer: Placer) -> None:
        log.info("f%d finished %s", s.frame, s.game.type_name(task.unit_type))
        if task.worker_id is not None:
            workers.release(task.worker_id)
        if task.tile is not None:
            placer.release(s.game, task.unit_type, task.tile)
