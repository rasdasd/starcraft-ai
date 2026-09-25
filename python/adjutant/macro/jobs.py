"""Building jobs: one structure from dispatch until it is finished.

A job is created with a tile and a builder and holds its cost until the structure is placed. The
builder walks to the site and is only given the Build order once the money is banked (BWAPI refuses
an unaffordable or far-away Build, and a refused order leaves the worker mining). A Terran SCV stays
until the structure is finished (a dead or reassigned one is replaced); Protoss and Zerg builders
are done once the structure is placed. A site that does not take the structure within
`site_timeout` fails the tile (an exact site, e.g. an expansion, after `EXACT_RETRIES`); builders
dying on the way make the job `dangerous`. Either ends the job without a building.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bwbot import Race, UnitFlag
from bwbot.enums import Order

from .placement import unit_tile

log = logging.getLogger("adjutant.macro")

WALKING, PLACED = "walking", "placed"
REISSUE_FRAMES = 8
WALK_TILES = 6             # Move (not Build) until this close to the site centre
ARRIVE_PX = 5 * 32
EXACT_RETRIES = 3
BUILDER_DEATHS = 2


@dataclass
class BuildJob:
    unit_type: int
    tile: tuple[int, int]
    exact: bool = False              # tile is fixed (a base's hall site, a geyser)
    base_id: Optional[int] = None    # the expansion this job makes
    worker_id: Optional[int] = None
    building_id: Optional[int] = None
    status: str = WALKING
    created: int = 0
    arrived: int = 0
    last_issue: int = -10_000
    retries: int = 0
    deaths: int = 0
    failed: str = ""                 # "blocked" | "dangerous" | "destroyed" once the job is over

    def cost(self, game) -> tuple[int, int]:
        ut = game.unit_types
        return int(ut["mineral_price"][self.unit_type]), int(ut["gas_price"][self.unit_type])

    def center(self, game) -> tuple[int, int]:
        ut = game.unit_types
        return (self.tile[0] * 32 + int(ut["tile_width"][self.unit_type]) * 16,
                self.tile[1] * 32 + int(ut["tile_height"][self.unit_type]) * 16)

    def label(self, game) -> str:
        name = game.type_name(self.unit_type).split("_", 1)[-1]
        return f"{name}:{self.status}"


class JobRunner:
    """Steps jobs; `claim(job)` gets a new builder from the executor (or None)."""

    def __init__(self, site_timeout: int = 24 * 15, travel_timeout: int = 24 * 90) -> None:
        self.site_timeout = site_timeout
        self.travel_timeout = travel_timeout

    def step(self, job: BuildJob, obs, act, minerals: int, gas: int, claim, placer, claimed: set[int]) -> bool:
        """Advance `job` one decision. False once it is over (finished or failed)."""
        g, frame = obs.game, obs.frame_count
        if job.building_id is not None:
            b = obs.unit(job.building_id)
            if b is None:
                job.failed = "destroyed"
                return False
            if obs.has_flag(b, UnitFlag.Completed):
                log.info("f%d finished %s", frame, g.type_name(job.unit_type))
                return False
            self._keep_builder(job, obs, act, b, claim)
            return True

        placed = self._placed(job, obs, claimed)
        if placed is not None:
            job.building_id, job.status = int(placed["id"]), PLACED
            log.info("f%d placed %s at %s", frame, g.type_name(job.unit_type), job.tile)
            if int(g.unit_types["race"][job.unit_type]) != int(Race.Terran):
                job.worker_id = None
            return True

        w = obs.unit(job.worker_id) if job.worker_id is not None else None
        if w is None:
            if job.worker_id is not None:
                job.deaths += 1
                if job.deaths >= BUILDER_DEATHS:
                    job.failed = "dangerous"
                    log.warning("f%d %s at %s: %d builders lost on the way", frame, g.type_name(job.unit_type),
                                job.tile, job.deaths)
                    return False
            job.worker_id = None
            w = claim(job)
            if w is None:
                return True
            job.worker_id = int(w["id"])
            job.last_issue = -10_000
        building = (int(w["flags"]) & int(UnitFlag.Constructing) or int(w["order"]) == int(Order.PlaceBuilding)
                    or int(w["order"]) == int(Order.ConstructingBuilding))
        cx, cy = job.center(g)
        d2 = (int(w["x"]) - cx) ** 2 + (int(w["y"]) - cy) ** 2
        if not job.arrived and d2 <= ARRIVE_PX ** 2:
            job.arrived = frame
        m, gg = job.cost(g)
        if not building and frame - job.last_issue >= REISSUE_FRAMES:
            if d2 > (WALK_TILES * 32) ** 2 or minerals < m or gas < gg:
                if d2 > (2 * 32) ** 2 and (abs(int(w["order_target_x"]) - cx) + abs(int(w["order_target_y"]) - cy) > 64
                                          or int(w["order"]) != int(Order.Move)):
                    act.move(w, cx, cy)
            else:
                act.build(w, job.unit_type, job.tile[0], job.tile[1])
            job.last_issue = frame
        stuck = job.arrived and minerals >= m and gas >= gg and frame - job.arrived > self.site_timeout
        lost = frame - job.created > self.travel_timeout and not job.arrived
        if stuck or lost:
            return self._retry(job, obs, placer, "at site" if stuck else "travel")
        return True

    def _retry(self, job: BuildJob, obs, placer, why: str) -> bool:
        g, frame = obs.game, obs.frame_count
        job.retries += 1
        log.warning("f%d %s at %s timed out (%s, try %d)", frame, g.type_name(job.unit_type), job.tile, why, job.retries)
        if job.exact:
            if job.retries >= EXACT_RETRIES:
                job.failed = "blocked"
                return False
        else:
            placer.release(g, job.unit_type, job.tile)
            placer.fail(job.tile, radius=2 if why == "at site" else 0)
            job.failed = "blocked"
            return False
        job.arrived, job.created, job.last_issue = 0, frame, -10_000
        return True

    @staticmethod
    def _placed(job: BuildJob, obs, claimed: set[int]):
        """Our unfinished structure of the job's type at (about) its tile."""
        g = obs.game
        for u in obs.my_units_of_type(job.unit_type):
            if obs.has_flag(u, UnitFlag.Completed) or int(u["id"]) in claimed:
                continue
            tx, ty = unit_tile(g, u)
            if abs(tx - job.tile[0]) <= 2 and abs(ty - job.tile[1]) <= 2:
                return u
        return None

    def _keep_builder(self, job: BuildJob, obs, act, building, claim) -> None:
        """Terran construction stops without its SCV: send it (or a replacement) back."""
        g, frame = obs.game, obs.frame_count
        if int(g.unit_types["race"][job.unit_type]) != int(Race.Terran):
            return
        w = obs.unit(job.worker_id) if job.worker_id is not None else None
        if w is not None and (int(w["flags"]) & int(UnitFlag.Constructing)
                              or int(w["order"]) == int(Order.ConstructingBuilding)):
            return
        if frame - job.last_issue < REISSUE_FRAMES:
            return
        job.last_issue = frame
        if w is None:
            job.worker_id = None
            w = claim(job)
            if w is None:
                return
            job.worker_id = int(w["id"])
            log.info("f%d SCV #%d takes over %s #%d", frame, job.worker_id, g.type_name(job.unit_type), job.building_id)
        act.right_click(w, building)
