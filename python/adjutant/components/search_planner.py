"""Production slot, v2: build-order search with a forward economy simulator.

Same slot and contract as `GreedyPlanner` (it inherits its helpers, the opening, request handling
and cancels). After the opening, each `every_frames` it builds a target from `strategy.goal`
(buildings, addons, bases, refineries up to the geysers we own, workers, upgrades, techs, and the
next chunk of army sized by the minimum-army rule), seeds `EconSim` from the live game (building
jobs already walking count as started; the executor's build queue is forced first because it
runs FIFO) and runs `BuildSearch` within `budget_ms`.

The plan is the prefix of the best sequence that the simulator starts within `slack_frames`:
everything after it is left out, so the executor never spends money the sequence needs sooner.
Plan priorities follow the sequence order.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from blackboard import Blackboard, Phase, Priority
from blackboard.profile import register
from blackboard.sections import PlanItem
from mybot.buildings import CONSTRUCTING

from ..buildsearch import BuildSearch, Target
from ..econsim import Action, EconSim
from .planner import HALL, REFINERY, SUPPLY, WORKER, GreedyPlanner

log = logging.getLogger("adjutant.search_planner")

P_TOP = Priority.PRODUCTION + 30


@register("SearchPlanner")
class SearchPlanner(GreedyPlanner):
    phase = Phase.PLAN

    def __init__(self, every_frames: int = 24, budget_ms: float = 12.0, slack_frames: int = 24,
                 worker_chunk: int = 8, army_chunk: int = 8, max_army_chunk: int = 24, float_minerals: int = 400,
                 **kw) -> None:
        super().__init__(**kw)
        self.float_minerals = float_minerals
        self.every = every_frames
        self.budget_ms = budget_ms
        self.slack = slack_frames
        self.worker_chunk = worker_chunk
        self.army_chunk = army_chunk
        self.max_army_chunk = max_army_chunk

    def on_start(self, bb: Blackboard) -> None:
        super().on_start(bb)
        self.sim = EconSim(self.tree, self.race)
        self.search = BuildSearch(self.sim, SUPPLY.get(self.race, -1), REFINERY.get(self.race, -1))
        self.last = -10 ** 9
        self.best = None
        self.stats = {"searches": 0, "ms_max": 0.0, "ms_sum": 0.0, "fallbacks": 0}

    # ------------------------------------------------------------------ tick
    def tick(self, bb: Blackboard) -> None:
        st = bb.strategy
        if getattr(self, "sim", None) is None or self.tree.game is not bb.game:
            self.on_start(bb)
        opening = not st.opening_done and st.opening_next is not None
        if opening or bb.obs is None or self.race not in WORKER:
            return super().tick(bb)
        if bb.frame - self.last >= self.every or self.best is None:
            self.last = bb.frame
            try:
                self._search(bb)
            except Exception:                  # never leave the bot without a plan
                log.exception("search failed; greedy plan this decision")
                self.stats["fallbacks"] += 1
                self.best = None
                return super().tick(bb)
        self._emit(bb)

    def _search(self, bb: Blackboard) -> None:
        s = self.sim.from_game(bb)
        bm, prod = bb.services.get("buildings"), bb.services.get("production")
        if bm is not None:
            for task in bm.tasks:
                if task.status == CONSTRUCTING:        # already a unit in the world
                    continue
                s.started[task.unit_type] = s.started.get(task.unit_type, 0) + 1
                self.sim._push(s, bb.frame + self.sim.travel + self.tree.time(task.unit_type), "unit",
                               task.unit_type, 0)
            s.minerals -= bm.reserved_minerals(bb.game)
            s.gas -= bm.reserved_gas(bb.game)
        forced = [Action("unit", int(t)) for t in (prod.queue if prod is not None else [])]
        target = self.target(bb, s)
        self.best = self.search.search(s, target, self.budget_ms, forced=forced)
        self.target_ = target
        st = self.stats
        st["searches"] += 1
        st["ms_sum"] += self.best.ms
        st["ms_max"] = max(st["ms_max"], self.best.ms)

    # ------------------------------------------------------------------ target
    def target(self, bb: Blackboard, s) -> Target:
        tree, goal, w = self.tree, bb.strategy.goal, bb.world
        worker, hall, refinery = WORKER[self.race], HALL[self.race], REFINERY[self.race]
        counts: dict[int, int] = {}
        order: list[Action] = []

        def want(t: int, n: int) -> None:
            t = int(t)
            if not tree.known_unit(t):
                return
            if n > s.started.get(t, 0):
                counts[t] = max(counts.get(t, 0), n)
                if Action("unit", t) not in order:
                    order.append(Action("unit", t))

        # army first when short, then economy, then tech, then the rest of the army chunk
        need = self.min_army(bb)
        short = w.army_supply < need
        army = self._army_chunk(bb, s, goal, need)
        if short:
            for t, n in army.items():
                want(t, n)
        want(worker, min(goal.workers, s.started.get(worker, 0) + self.worker_chunk))
        if goal.bases > len(w.depots) + (s.started.get(hall, 0) - s.done.get(hall, 0)):
            want(hall, s.started.get(hall, 0) + 1)
        geysers = self.owned_geysers(bb)
        for t, n in goal.buildings.items():
            want(t, min(n, geysers) if t == refinery else n)
        for t, n in goal.addons.items():
            want(t, n)
        ups: dict[int, int] = {}
        techs: set[int] = set()
        if not short:
            for u, lvl in goal.upgrades:
                if lvl > s.upgrades.get(int(u), 0) and lvl == s.upgrades.get(int(u), 0) + 1:
                    ups[int(u)] = lvl
                    order.append(Action("upgrade", int(u), lvl))
            for t in goal.techs:
                if int(t) not in s.techs:
                    techs.add(int(t))
                    order.append(Action("tech", int(t)))
        for t, n in army.items():
            want(t, n)
        if w.minerals >= 2 * self.float_minerals and army:
            main = max(army, key=lambda t: army[t] * max(1, tree.supply(t)))
            producer = tree.builder(main)
            if producer != -1 and tree.is_building(producer) and s.started.get(producer, 0) == s.done.get(producer, 0):
                want(producer, s.started.get(producer, 0) + 1)
        return Target(counts, ups, techs, order)

    def _army_chunk(self, bb: Blackboard, s, goal, need: float) -> dict[int, int]:
        """Target started counts per goal unit type: the next `army_chunk` supply (more when short)."""
        tree, worker = self.tree, WORKER[self.race]
        types = {int(t): n for t, n in goal.units.items() if n > 0 and int(t) != worker}
        if not types:
            return {}
        deficit = max(0.0, need - bb.world.army_supply)
        chunk = min(self.max_army_chunk, max(self.army_chunk, deficit))
        # beyond the goal counts when short on army or floating money
        extend = bb.world.army_supply < need or bb.world.minerals >= self.float_minerals
        total = sum(types.values())
        out: dict[int, int] = {}
        for t, n in types.items():
            k = max(1, int(round(chunk * n / total / max(1, tree.supply(t)))))
            have = s.started.get(t, 0)
            if have < n or extend:
                out[t] = have + k
        return out

    # ------------------------------------------------------------------ plan
    def _emit(self, bb: Blackboard) -> None:
        items: list[PlanItem] = []
        emitted: set[tuple[str, int]] = set()

        def add(kind: str, t: int, prio: int, reason: str, count: int = 1, near=None, exact=False,
                cost=None) -> None:
            if (kind, t) in emitted:
                return
            emitted.add((kind, t))
            items.append(PlanItem(kind, int(t), int(prio), reason, count, near, exact))

        cancels = self._requests(bb, self.tree, add, WORKER[self.race])
        hall = HALL[self.race]
        horizon = bb.frame + self.slack
        trains: Counter = Counter()
        prio = {}
        best = self.best
        for i, (a, f) in enumerate(best.steps if best else []):
            if f > horizon:
                break
            p = int(P_TOP) - min(i, 29)
            if a.kind == "upgrade":
                add("upgrade", a.type_id, p, "search")
            elif a.kind == "tech":
                add("research", a.type_id, p, "search")
            elif self.tree.is_addon(a.type_id):
                add("addon", a.type_id, p, "search", count=max(1, bb.world.count(a.type_id) + 1))
            elif self.tree.is_building(a.type_id):
                if a.type_id == hall:
                    if hall in cancels:
                        continue
                    tile = self.next_base(bb)
                    add("build", hall, p, "search expand", near=tile, exact=tile is not None)
                else:
                    add("build", a.type_id, p, "search")
            else:
                trains[a.type_id] += 1
                prio.setdefault(a.type_id, p)
        for t, n in trains.items():
            add("train", t, prio[t], "search", count=n)
        items.sort(key=lambda it: -it.priority)
        plan = bb.plan
        plan.items = items
        plan.notes = [f"search {best.makespan - bb.frame if best else 0}f {best.ms if best else 0:.1f}ms "
                      f"{len(best.steps) if best else 0} steps"]
        if best:
            plan.notes += [f"{self._name(a)}+{f - bb.frame}" for a, f in best.steps[:4]]
        plan.army_order = None
        plan.cancel = cancels
        plan.replace_queue = True

    def _name(self, a: Action) -> str:
        if a.kind == "unit":
            return self.tree.game.type_name(a.type_id).replace("Terran_", "").replace("Protoss_", "")
        return str(a)

    def summary(self) -> str:
        st = self.stats
        n = max(1, st["searches"])
        return f"search n={st['searches']} avg {st['ms_sum'] / n:.1f}ms max {st['ms_max']:.1f}ms"

    def describe(self) -> dict:
        return {"impl": self.name, "budget_ms": self.budget_ms, "every": self.every}
