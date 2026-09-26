"""Macro executor: the one component that turns the production plan into commands.

Each decision it updates our bases, steps building jobs, applies the plan's cancels, then walks
`plan.items` (highest priority first) with one budget and one rule for every kind of item:

* ready and affordable: start it (train, addon, morph, research, or dispatch a building job);
* a building whose builder would arrive by the time the money does: dispatch it now;
* ready but not affordable: hold its cost, so lower-priority items cannot spend it (only the short
  resource when minerals alone would cover it);
* blocked (no free producer, missing prerequisite, no site, no supply, gas without any refinery):
  skip it without holding money.

Dispatched jobs hold their cost until placed. Workers nobody else leases mine at our bases (see
`WorkerPool`). The result goes to the `macro` section: our bases and their saturation, jobs not yet
placed per type, the next expansion, and what was blocked, so planners count from one ledger.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from blackboard.sections import MacroState, PlanItem
from bwbot import Color, Race, UnitFlag, UnitType as U
from bwbot.observation import UnitTypeFlag as F

from ..mapgraph import MapGraph
from ..techtree import TechTree
from .bases import BaseTracker
from .jobs import WALKING, BuildJob, JobRunner
from .placement import Placer, footprint, occupancy
from .wall import Wall, plan_wall
from .workers import WorkerPool, pick_worker

log = logging.getLogger("adjutant.macro")

HALL = {int(Race.Terran): int(U.Terran_Command_Center), int(Race.Zerg): int(U.Zerg_Hatchery),
        int(Race.Protoss): int(U.Protoss_Nexus)}
REFINERIES = (int(U.Terran_Refinery), int(U.Protoss_Assimilator), int(U.Zerg_Extractor))
SHIM_QUERY_FRAMES = 24 * 2
NO_SITE_RETRY_FRAMES = 24 * 2     # a full spiral search that found nothing is not repeated sooner
WALL_TYPES = (int(U.Terran_Barracks), int(U.Terran_Supply_Depot))
WALL_VS = (int(Race.Protoss), int(Race.Random), int(Race.Unknown))


def _building_something(u) -> bool:
    """Constructing an addon or morphing (BWAPI reports UnitTypes::None when idle)."""
    return 0 <= int(u["build_type"]) < int(U.None_)


class Budget:
    def __init__(self, minerals: int, gas: int, supply: int) -> None:
        self.m, self.g, self.s = minerals, gas, supply

    def can(self, m: int, g: int, s: int = 0) -> bool:
        return self.m >= m and self.g >= g and self.s >= s

    def spend(self, m: int, g: int, s: int = 0) -> None:
        self.m, self.g, self.s = self.m - m, self.g - g, self.s - s

    def hold(self, m: int, g: int) -> None:
        """Save for an unaffordable item: gas only if minerals already cover it."""
        if self.m >= m:
            self.g -= g
        else:
            self.m, self.g = self.m - m, self.g - g


@register("Macro")
class Macro(Component):
    phase = Phase.ACT
    reads = ("world", "plan", "belief", "meta")
    writes = ("macro",)
    priority = Priority.CONSTRUCTION
    order = 0

    def __init__(self, max_jobs: int = 3, early_dispatch: bool = True, gas_bank: Optional[tuple[int, int]] = None,
                 site_timeout_s: float = 15, draw: bool = True, wall: bool = True) -> None:
        self.max_jobs = max_jobs              # building jobs walking at once
        self.use_wall = wall                  # Terran vs Protoss: Barracks + Depot marine gap at the ramp
        self.early_dispatch = early_dispatch
        self.gas_bank = tuple(gas_bank) if gas_bank else None   # (resume, pull) gas levels; None: always mine gas
        self.site_timeout = int(site_timeout_s * 24)
        self.draw = draw

    def on_start(self, bb: Blackboard) -> None:
        g = bb.game
        self.tree = TechTree(g)
        self.race = int(g.self_race)
        self.hall = HALL.get(self.race, -1)
        self.bases = BaseTracker()
        self.bases.on_start(g, bb.services.get("mapgraph") or MapGraph(g))
        self.pool = WorkerPool(gas_bank=self.gas_bank)
        self.placer = Placer()
        self.runner = JobRunner(site_timeout=self.site_timeout)
        self.jobs: list[BuildJob] = []
        self.cancelled: set[int] = set()
        self._no_site: dict[int, int] = {}
        self.shim: dict[int, tuple[int, Optional[tuple[int, int]]]] = {}   # type -> (query frame, answer)
        self._query = 0
        enemy = g.player(g.enemy_id)
        self.wall: Optional[Wall] = None
        self._wall_pending = (self.use_wall and self.race == int(Race.Terran) and
                              (enemy is None or int(enemy.race) in WALL_VS))
        wt = int(U.Terran_SCV if self.race == int(Race.Terran) else U.Protoss_Probe
                 if self.race == int(Race.Protoss) else U.Zerg_Drone)
        self.worker_speed = max(1.0, float(g.unit_types["top_speed"][wt]))

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        obs, w, frame = bb.obs, bb.world, bb.frame
        self.bases.update(obs, frame)
        self._shim_answers(obs)
        self._step_jobs(bb)
        self.cancelled = {int(t) for t in bb.plan.cancel}
        for t in self.cancelled:
            self.cancel(bb, t)

        reserved = self._reserved()
        m, g = self._reserved_elsewhere(bb)
        budget = Budget(w.minerals - reserved[0] - m, w.gas - reserved[1] - g, w.supply_left)
        self.used: set[int] = set()              # producers given an order this decision
        blocked: list[str] = []
        for it in bb.plan.items:
            why = self._item(bb, it, budget)
            if why:
                blocked.append(f"{it.kind}:{bb.game.type_name(it.type_id) if it.kind not in ('upgrade', 'research') else it.type_id}:{why}")

        builders = {j.worker_id for j in self.jobs if j.worker_id is not None}
        bb.leases.hold(self.slot, builders, self.priority, "build", frame)
        free = {int(u["id"]) for u in w.workers if bb.leases.owner(int(u["id"])) in (None, self.slot)} - builders
        self.pool.update(obs, bb.act, w.workers, free, self.bases, w.minerals, w.gas)
        self._publish(bb, reserved, blocked)
        if self.draw:
            self._draw(bb)

    # ------------------------------------------------------------------ plan items
    def _item(self, bb: Blackboard, it: PlanItem, budget: Budget) -> str:
        """Start / hold for / skip one item. Returns why it was skipped ('' if started or held)."""
        kind, t = it.kind, int(it.type_id)
        if kind in ("build", "expand"):
            if t in self.cancelled:
                return "cancelled"
            return self._build(bb, it, budget)
        if kind == "train":
            return self._train(bb, t, max(1, it.count), budget)
        if kind == "addon":
            return self._addon(bb, t, max(1, it.count), budget)
        if kind == "upgrade":
            return self._upgrade(bb, t, budget)
        if kind == "research":
            return self._research(bb, t, budget)
        return "unknown kind"

    def _build(self, bb: Blackboard, it: PlanItem, budget: Budget) -> str:
        tree, obs, t = self.tree, bb.obs, int(it.type_id)
        expand = it.kind == "expand"
        mine = [j for j in self.jobs if j.status == WALKING and (j.base_id is not None if expand else j.unit_type == t)]
        if len(mine) >= max(1, it.count):
            return ""
        src = tree.builder(t)
        if src != -1 and tree.is_building(src):            # Lair, Hive, Sunken / Spore Colony, Greater Spire
            return self._morph_building(bb, t, src, budget)
        if not self._reqs_done(bb, t):
            return "prereq"
        m, g = tree.cost(t)
        if self._gas_blocked(bb, g, budget):
            return "no gas"
        if sum(1 for j in self.jobs if j.status == WALKING) >= self.max_jobs:
            return "jobs"
        base_id = None
        if expand:
            base_id = self.bases.next_base(bb.belief, bb.game.self_natural_id)
            if base_id is None:
                return "no base"
            tile, exact = tuple(bb.game.base(base_id).tile), True
        elif it.exact and it.near is not None:
            tile, exact = tuple(it.near), True
        else:
            tile = self._site(bb, t, it.near)
            exact = tree.is_refinery(t)
            if tile is None:
                if bb.game.type_flags(t) & F.RequiresPsi:
                    return self._power(bb, it, budget)
                return "no site"
        worker = self._pick(bb, tile, t)
        if worker is None:
            return "no builder"
        eta = self._eta(bb, worker, tile, t, base_id)
        if not budget.can(m, g):
            per = 1.0 / (24 * 60)
            soon = (budget.m + bb.world.income_minerals * per * eta >= m
                    and budget.g + bb.world.income_gas * per * eta >= g)
            if not (self.early_dispatch and soon):
                budget.hold(m, g)
                return ""
        job = BuildJob(t, tile, exact=exact, base_id=base_id, worker_id=int(worker["id"]), created=bb.frame,
                       eta=int(eta) if math.isfinite(eta) else 0)
        self.jobs.append(job)
        if not exact:
            self.placer.reserve(bb.game, t, tile)
        budget.spend(m, g)
        self.pool.release(job.worker_id)
        bb.leases.lease(job.worker_id, self.slot, self.priority, "build", bb.frame)
        log.info("f%d dispatch %s at %s%s (#%d)", bb.frame, bb.game.type_name(t), tile,
                 f" base {base_id}" if base_id is not None else "", job.worker_id)
        return ""

    def _power(self, bb: Blackboard, it: PlanItem, budget: Budget) -> str:
        """No powered room for a Protoss building: put down a pylon for it (one at a time)."""
        pylon = int(U.Protoss_Pylon)
        if any(j.unit_type == pylon for j in self.jobs) or bb.world.count(pylon) > bb.world.count_completed(pylon):
            return "no site (powering)"
        why = self._build(bb, PlanItem("build", pylon, it.priority, "power"), budget)
        return f"no site (pylon: {why})" if why else "no site (pylon)"

    def _morph_building(self, bb: Blackboard, t: int, src: int, budget: Budget) -> str:
        if not self._reqs_done(bb, t):
            return "prereq"
        srcs = [u for u in bb.obs.my_completed(src) if int(u["id"]) not in self.used
                and not _building_something(u) and int(u["remaining_research_time"]) == 0
                and int(u["remaining_upgrade_time"]) == 0]
        if not srcs:
            return "no source"
        m, g = self.tree.cost(t)
        if not budget.can(m, g):
            budget.hold(m, g)
            return ""
        bb.act.morph(srcs[0], t)
        self.used.add(int(srcs[0]["id"]))
        budget.spend(m, g)
        log.info("f%d morph %s", bb.frame, bb.game.type_name(t))
        return ""

    def _train(self, bb: Blackboard, t: int, n: int, budget: Budget) -> str:
        tree = self.tree
        producer = tree.builder(t)
        if producer == -1 or not self._reqs_done(bb, t):
            return "prereq"
        m, g = tree.cost(t)
        s = tree.supply(t)
        if self._gas_blocked(bb, g, budget):
            return "no gas"
        free = self._producers(bb, t, producer)
        if not free:
            return "no producer"
        started = 0
        for p in free[:n]:
            if s > budget.s:
                return "" if started else "supply"
            if not budget.can(m, g):
                budget.hold(m, g)
                return ""
            if producer == int(U.Zerg_Larva) or not tree.is_building(producer):
                bb.act.morph(p, t)
            else:
                bb.act.train(p, t)
            self.used.add(int(p["id"]))
            budget.spend(m, g, s)
            started += 1
        return ""

    def _producers(self, bb: Blackboard, t: int, producer: int) -> list:
        """Completed producers of `producer` type that can take `t` now (idle, not lifted, with the
        addon `t` needs), nearest to the main first."""
        obs, tree = bb.obs, self.tree
        units = [u for u in obs.my_completed(producer) if int(u["id"]) not in self.used]
        if producer == int(U.Zerg_Larva):
            return units
        needs_addon = [r for r in tree.unit_requires(t) if tree.is_addon(r) and tree.builder(r) == producer]
        out = []
        for u in units:
            if int(u["flags"]) & int(UnitFlag.Lifted):
                continue
            if tree.is_building(producer) and (int(u["train_queue_count"]) > 0 or _building_something(u)
                                               or int(u["remaining_research_time"]) or int(u["remaining_upgrade_time"])):
                continue
            if needs_addon:
                a = obs.unit(int(u["addon"])) if int(u["addon"]) >= 0 else None
                if a is None or int(a["type"]) not in needs_addon or not obs.has_flag(a, UnitFlag.Completed):
                    continue
            out.append(u)
        mx, my = bb.world.main_tile[0] * 32, bb.world.main_tile[1] * 32
        out.sort(key=lambda u: (int(u["x"]) - mx) ** 2 + (int(u["y"]) - my) ** 2)
        return out

    def _addon(self, bb: Blackboard, t: int, n: int, budget: Budget) -> str:
        if bb.world.count(t) >= n:
            return ""
        if not self._reqs_done(bb, t):
            return "prereq"
        parent = self.tree.builder(t)
        parents = [u for u in bb.obs.my_completed(parent) if int(u["id"]) not in self.used and int(u["addon"]) < 0
                   and not int(u["flags"]) & int(UnitFlag.Lifted) and int(u["train_queue_count"]) == 0
                   and not _building_something(u)]
        if not parents:
            return "no parent"
        m, g = self.tree.cost(t)
        if not budget.can(m, g):
            budget.hold(m, g)
            return ""
        bb.act.build_addon(parents[0], t)
        self.used.add(int(parents[0]["id"]))
        budget.spend(m, g)
        return ""

    def _upgrade(self, bb: Blackboard, u: int, budget: Budget) -> str:
        me, tree = bb.obs.me, self.tree
        info = tree.upgrades.get(int(u))
        if info is None:
            return "unknown"
        cur = int(me.upgrade_level[u]) if me.upgrade_level.size > u else 0
        if (me.is_upgrading.size > u and me.is_upgrading[u]) or cur >= tree.max_level(u):
            return ""
        return self._research_at(bb, int(info["what_upgrades"]), tree.upgrade_cost(u, cur + 1), budget,
                                 lambda b: bb.act.upgrade(b, u))

    def _research(self, bb: Blackboard, tech: int, budget: Budget) -> str:
        me, tree = bb.obs.me, self.tree
        info = tree.techs.get(int(tech))
        if info is None:
            return "unknown"
        if (me.has_researched.size > tech and me.has_researched[tech]) or \
                (me.is_researching.size > tech and me.is_researching[tech]):
            return ""
        return self._research_at(bb, int(info["what_researches"]), tree.tech_cost(tech), budget,
                                 lambda b: bb.act.research(b, tech))

    def _research_at(self, bb: Blackboard, where: int, cost: tuple[int, int], budget: Budget, issue) -> str:
        blds = [b for b in bb.obs.my_completed(where) if int(b["id"]) not in self.used
                and int(b["remaining_research_time"]) == 0 and int(b["remaining_upgrade_time"]) == 0
                and int(b["train_queue_count"]) == 0 and not int(b["flags"]) & int(UnitFlag.Lifted)]
        if not blds:
            return "no building"
        if not budget.can(*cost):
            budget.hold(*cost)
            return ""
        issue(blds[0])
        self.used.add(int(blds[0]["id"]))
        budget.spend(*cost)
        return ""

    # ------------------------------------------------------------------ helpers
    def _reqs_done(self, bb: Blackboard, t: int) -> bool:
        w = bb.world
        for r in self.tree.unit_requires(t):
            if self.tree.flags(r) & F.Worker or r == int(U.Zerg_Larva):
                continue
            if w.count_completed(r) == 0:
                return False
        return True

    def _gas_blocked(self, bb: Blackboard, gas: int, budget: Budget) -> bool:
        """Needs more gas than we have and nothing is (or will be) mining any."""
        if gas <= 0 or budget.g >= gas:
            return False
        if any(bb.world.count(r) for r in REFINERIES):
            return False
        return not any(j.unit_type in REFINERIES for j in self.jobs)

    def _wall_site(self, bb: Blackboard, t: int) -> Optional[tuple[int, int]]:
        """The wall's tile for `t` while it is still free (the first Barracks / Depot go there)."""
        if self._wall_pending:
            self._wall_pending = False
            self.wall = plan_wall(bb.game, occupancy(bb.obs))
        tile = self.wall.tile(t) if self.wall is not None else None
        if tile is None or tile in self.placer.failed:
            return None
        cells = footprint(bb.game, t, tile)
        if any(c in self.placer.reserved for c in cells):
            return None
        occ = occupancy(bb.obs)
        if any(occ[y, x] for x, y in cells):
            return None
        return tile

    def _site(self, bb: Blackboard, t: int, near: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        obs, g = bb.obs, bb.game
        if near is None and (self.wall is not None or self._wall_pending) and t in WALL_TYPES:
            tile = self._wall_site(bb, t)
            if tile is not None:
                return tile
        anchors = [tuple(near)] if near is not None else \
            [tuple(bb.world.main_tile)] + [tuple(b.tile) for b in self.bases.bases if b.completed]
        if self.tree.is_refinery(t) or bb.frame - self._no_site.get(t, -10 ** 9) >= NO_SITE_RETRY_FRAMES:
            free = [tuple(b.tile) for b in g.bases if not self.bases.owned(b.id)]
            halls = [b.center for b in self.bases.bases]
            for anchor in dict.fromkeys(anchors):
                tile = self.placer.find(obs, t, anchor, keep_free=free, halls=halls)
                if tile is not None or self.tree.is_refinery(t):
                    return tile
            self._no_site[t] = bb.frame
        near = anchors[0]
        asked, answer = self.shim.get(t, (-10 ** 9, None))
        if answer is not None and answer not in self.placer.failed:
            self.shim.pop(t, None)
            return answer
        if bb.frame - asked >= SHIM_QUERY_FRAMES and hasattr(bb.act, "get_build_location"):
            self._query += 1
            self.shim[t] = (bb.frame, None)
            self._queries = getattr(self, "_queries", {})
            self._queries[self._query] = t
            bb.act.get_build_location(self._query, t, near)
        return None

    def _shim_answers(self, obs) -> None:
        for req_id, ok, tx, ty in getattr(obs, "placement_results", None) or ():
            t = getattr(self, "_queries", {}).pop(req_id, None)
            if t is not None and ok:
                self.shim[t] = (obs.frame_count, (int(tx), int(ty)))

    def _pick(self, bb: Blackboard, tile: tuple[int, int], t: int):
        w = bb.world
        busy = {j.worker_id for j in self.jobs if j.worker_id is not None}
        allowed = {int(u["id"]) for u in w.workers if bb.leases.owner(int(u["id"])) in (None, self.slot)} - busy
        return pick_worker(bb.obs, w.workers, tile[0] * 32 + 16, tile[1] * 32 + 16, allowed,
                           prefer_minerals=set(self.pool.mineral))

    def _eta(self, bb: Blackboard, worker, tile, t: int, base_id: Optional[int]) -> float:
        """Frames for `worker` to reach the site (ground distance for expansions)."""
        ut = bb.game.unit_types
        cx = tile[0] * 32 + int(ut["tile_width"][t]) * 16
        cy = tile[1] * 32 + int(ut["tile_height"][t]) * 16
        x, y = int(worker["x"]), int(worker["y"])
        d = math.hypot(cx - x, cy - y)
        graph = self.bases.graph
        if base_id is not None and graph is not None:
            here = min(bb.game.bases, key=lambda b: (b.center[0] - x) ** 2 + (b.center[1] - y) ** 2)
            d = max(d, graph.distance((x, y), here.area_id, (cx, cy), bb.game.base(base_id).area_id))
        return d / self.worker_speed

    def _claim(self, bb: Blackboard):
        def claim(job: BuildJob):
            w = self._pick(bb, job.tile, job.unit_type)
            if w is not None:
                self.pool.release(int(w["id"]))
                bb.leases.lease(int(w["id"]), self.slot, self.priority, "build", bb.frame)
            return w
        return claim

    def _step_jobs(self, bb: Blackboard) -> None:
        obs, w = bb.obs, bb.world
        claim = self._claim(bb)
        claimed = {j.building_id for j in self.jobs if j.building_id is not None}
        keep = []
        for job in self.jobs:
            if job.worker_id is not None and bb.leases.owner(job.worker_id) not in (None, self.slot):
                job.worker_id = None               # taken by a higher priority (a worker pull): replace it
            if self.runner.step(job, obs, bb.act, w.minerals, w.gas, claim, self.placer, claimed):
                keep.append(job)
                if job.building_id is not None:
                    claimed.add(job.building_id)
                continue
            if not job.exact:
                self.placer.release(bb.game, job.unit_type, job.tile)
                if job.failed == "dangerous":
                    self.placer.fail(job.tile, radius=3)
            if job.base_id is not None and job.failed in ("blocked", "unreachable", "dangerous", "destroyed"):
                mark = {"blocked": self.bases.mark_blocked, "unreachable": self.bases.mark_unreachable}
                mark.get(job.failed, self.bases.mark_dangerous)(job.base_id, bb.frame)
                bb.record("macro", "expand_failed", base=job.base_id, why=job.failed)
                log.warning("f%d expansion to base %d %s", bb.frame, job.base_id, job.failed)
        self.jobs = keep

    def cancel(self, bb: Blackboard, t: int) -> int:
        """Drop jobs of type `t` that have not placed yet (an expansion under attack, ...)."""
        keep, dropped = [], 0
        for j in self.jobs:
            if j.unit_type == t and j.status == WALKING:
                dropped += 1
                if not j.exact:
                    self.placer.release(bb.game, j.unit_type, j.tile)
                if j.worker_id is not None:
                    bb.leases.release(j.worker_id, self.slot)
            else:
                keep.append(j)
        self.jobs = keep
        if dropped:
            log.info("f%d cancelled %d %s job(s)", bb.frame, dropped, bb.game.type_name(t))
        return dropped

    def _reserved(self) -> tuple[int, int]:
        m = g = 0
        for j in self.jobs:
            if j.status == WALKING:
                jm, jg = j.cost(self.tree.game)
                m, g = m + jm, g + jg
        return m, g

    def _reserved_elsewhere(self, bb: Blackboard) -> tuple[int, int]:
        m = g = 0
        for key, (rm, rg, _, rp) in bb.budget.reservations.items():
            if not key.startswith(self.slot) and rp > self.priority:
                m, g = m + rm, g + rg
        return m, g

    def _publish(self, bb: Blackboard, reserved: tuple[int, int], blocked: list[str]) -> None:
        ms = MacroState()
        for b in self.bases.bases:
            b.miners = self.pool.miners.get(b.base_id, 0)
            b.gas_workers = self.pool.gassers.get(b.base_id, 0)
        ms.bases = list(self.bases.bases)
        for j in self.jobs:
            if j.status == WALKING:
                ms.pending[j.unit_type] = ms.pending.get(j.unit_type, 0) + 1
        ms.jobs = [j.label(bb.game) for j in self.jobs]
        ms.reserved = reserved
        ms.expanding = next((j.base_id for j in self.jobs if j.base_id is not None), None)
        ms.next_base = self.bases.next_base(bb.belief, bb.game.self_natural_id)
        ms.worker_target = sum(2 * b.patches + 3 * b.refineries for b in ms.bases)
        ms.blocked = blocked
        bb.put("macro", ms)

    def _draw(self, bb: Blackboard) -> None:
        ut = bb.game.unit_types
        for j in self.jobs:
            bb.act.draw_tile_box(j.tile[0], j.tile[1], int(ut["tile_width"][j.unit_type]),
                                 int(ut["tile_height"][j.unit_type]))
            u = bb.obs.unit(j.worker_id) if j.worker_id is not None else None
            if u is not None:
                bb.act.draw_circle(int(u["x"]), int(u["y"]), 12, color=Color.Yellow)

    def summary(self) -> str:
        return ",".join(j.label(self.tree.game) for j in self.jobs) or "-"
