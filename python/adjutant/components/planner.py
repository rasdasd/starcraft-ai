"""Production slot, v1: greedy, tech-tree aware planner from `strategy.goal`.

Each decision it rewrites `plan.items` in priority order:
  supply (projected from production capacity) > opening step > workers > missing prerequisites >
  goal buildings / expansions / addons > upgrades and research > army trains.
Army trains only spend what is left after reserving the cost of the next unstarted building and
of ready upgrades/research/addons, so production does not starve tech. Builds are only emitted
once their own prerequisites are complete, so the executor's queue head never blocks on them.
"""
from __future__ import annotations

import math
from typing import Optional

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from blackboard.sections import PlanItem
from bwbot import Race, UnitType as U

from ..techtree import TechTree

WORKER = {int(Race.Terran): int(U.Terran_SCV), int(Race.Zerg): int(U.Zerg_Drone), int(Race.Protoss): int(U.Protoss_Probe)}
SUPPLY = {int(Race.Terran): int(U.Terran_Supply_Depot), int(Race.Zerg): int(U.Zerg_Overlord),
          int(Race.Protoss): int(U.Protoss_Pylon)}
HALL = {int(Race.Terran): int(U.Terran_Command_Center), int(Race.Zerg): int(U.Zerg_Hatchery),
        int(Race.Protoss): int(U.Protoss_Nexus)}
REFINERY = {int(Race.Terran): int(U.Terran_Refinery), int(Race.Zerg): int(U.Zerg_Extractor),
            int(Race.Protoss): int(U.Protoss_Assimilator)}

P_SUPPLY = Priority.SUPPLY
P_OPENING = Priority.PRODUCTION + 20
P_WORKER = Priority.PRODUCTION + 12
P_CHAIN = Priority.PRODUCTION + 8
P_EXPAND = Priority.PRODUCTION + 6
P_TECH = Priority.PRODUCTION + 4
P_BUILDING = Priority.PRODUCTION + 2
P_UPGRADE = Priority.PRODUCTION + 1
P_ARMY = Priority.PRODUCTION
P_ARMY_URGENT = Priority.PRODUCTION + 7


@register("GreedyPlanner")
class GreedyPlanner(Component):
    phase = Phase.PLAN
    reads = ("world", "meta", "belief", "strategy", "threats")
    writes = ("plan",)

    def __init__(self, supply_lead: float = 1.0, max_supply: int = 200, army_from_min: float = 3.0,
                 army_per_min: float = 4.0, army_cap: float = 30.0, enemy_army_factor: float = 1.0) -> None:
        self.supply_lead = supply_lead      # supply buffer, in "decisions of full production" units
        self.max_supply = max_supply
        self.army_from_min = army_from_min  # minimum army supply: army_per_min per minute after this
        self.army_per_min = army_per_min
        self.army_cap = army_cap
        self.enemy_army_factor = enemy_army_factor   # ... and at least this x the known enemy army

    # ------------------------------------------------------------------ helpers
    def _queued(self, bb: Blackboard, t: int) -> int:
        prod, bm = bb.services.get("production"), bb.services.get("buildings")
        n = 0
        if prod is not None:
            n += sum(1 for q in prod.queue if q == t)
        if bm is not None:
            n += bm.pending_count(t)
        return n

    def have(self, bb: Blackboard, t: int) -> int:
        """Owned (incl. in production) + queued / not yet placed."""
        return bb.world.count(t) + self._queued(bb, t)

    def done(self, bb: Blackboard, t: int) -> int:
        return bb.world.count_completed(t)

    def _reqs_done(self, bb: Blackboard, tree: TechTree, reqs) -> bool:
        return all(self.done(bb, r) > 0 for r in reqs if r != WORKER.get(self.race))

    # ------------------------------------------------------------------ tick
    def on_start(self, bb: Blackboard) -> None:
        self.tree = TechTree(bb.game)
        self.race = int(bb.game.self_race)

    def tick(self, bb: Blackboard) -> None:
        tree = getattr(self, "tree", None)
        if tree is None or tree.game is not bb.game:
            self.on_start(bb)
            tree = self.tree
        w, st = bb.world, bb.strategy
        goal = st.goal
        items: list[PlanItem] = []
        notes: list[str] = []
        emitted: set[tuple[str, int]] = set()
        held: dict[int, int] = {}           # producer type -> buildings kept idle for an addon
        reserve = [0, 0]
        first_build_reserved = [False]

        def add(kind: str, t: int, prio: int, reason: str, count: int = 1, near=None, exact=False,
                cost: Optional[tuple[int, int]] = None) -> None:
            if (kind, t) in emitted:
                return
            emitted.add((kind, t))
            items.append(PlanItem(kind, int(t), int(prio), reason, count, near, exact))
            if kind == "build":
                bm = bb.services.get("buildings")
                started = bm is not None and any(tk.unit_type == t for tk in bm.tasks)
                if not started and not first_build_reserved[0]:
                    m, g = tree.cost(t)
                    reserve[0] += m
                    reserve[1] += g
                    first_build_reserved[0] = True
            elif cost is not None:
                reserve[0] += cost[0]
                reserve[1] += cost[1]
            if kind == "addon":
                parent = tree.builder(t)
                held[parent] = held.get(parent, 0) + 1

        worker, supply_t, hall = WORKER.get(self.race), SUPPLY.get(self.race), HALL.get(self.race)
        refinery = REFINERY.get(self.race)

        # 1. supply
        if supply_t is not None and w.supply_total < self.max_supply:
            if self._need_supply(bb, tree, supply_t, hall):
                add("build", supply_t, P_SUPPLY, "supply")
                notes.append("supply")

        # 2. opening
        opening = not st.opening_done and st.opening_next is not None
        if opening:
            t = int(st.opening_next)
            if self._reqs_done(bb, tree, tree.unit_requires(t)):
                add("build", t, P_OPENING, "opening")

        # 3. workers
        if worker is not None:
            deficit = goal.workers - self.have(bb, worker)
            halls = self.done(bb, hall) if hall is not None else 0
            if deficit > 0 and halls > 0:
                add("train", worker, P_WORKER, "workers", count=min(deficit, halls))

        if not opening:
            # 4. prerequisites for everything the goal asks for
            needs: list[int] = []
            for t, n in goal.units.items():
                if n > 0:
                    needs += [r for r in tree.unit_requires(t) if r != worker]
            for t, n in list(goal.buildings.items()) + list(goal.addons.items()):
                if n > 0:
                    needs += [r for r in tree.unit_requires(t) if r != worker]
            for u, lvl in goal.upgrades:
                needs += tree.upgrade_requires(u, lvl)
            for tech in goal.techs:
                needs += tree.tech_requires(tech)
            for t in tree.missing(needs, lambda x: self.have(bb, x)):
                if t == worker or t == hall and self.done(bb, hall) > 0:
                    continue
                if not self._reqs_done(bb, tree, tree.unit_requires(t)):
                    continue
                if tree.is_addon(t):
                    if not self._free_parent(bb, tree, t):
                        continue
                    add("addon", t, P_CHAIN, "prereq", count=max(1, goal.addons.get(t, 1)), cost=tree.cost(t))
                elif tree.is_building(t):
                    add("build", t, P_CHAIN, "prereq")
                notes.append(f"prereq {bb.game.type_name(t)}")

            # 4b. too little army for the game time / the enemy we know about: units before
            # expansions and tech (and before their money is reserved)
            need = self.min_army(bb)
            if w.army_supply < need:
                self._army(bb, tree, goal, add, reserve, worker, held, prio=P_ARMY_URGENT)
                notes.append(f"army {w.army_supply}<{need:.0f}")

            # 5. expansions
            bases_now = self.have(bb, hall) if hall is not None else 1
            if hall is not None and goal.bases > bases_now:
                tile = self.next_base(bb)
                if tile is not None:
                    add("build", hall, P_EXPAND, "expand", near=tile, exact=True)
                    notes.append(f"expand {tile}")

            # 6. goal buildings
            geysers = self.owned_geysers(bb)
            for t, n in goal.buildings.items():
                if t == refinery:
                    n = min(n, geysers)
                if n <= self.have(bb, t) or not self._reqs_done(bb, tree, tree.unit_requires(t)):
                    continue
                producer = any(int(tree.builder(u)) == t for u in goal.units)
                add("build", t, P_BUILDING if producer else P_TECH, "goal")

            # 7. addons
            for t, n in goal.addons.items():
                if n > self.have(bb, t) and self._reqs_done(bb, tree, tree.unit_requires(t)):
                    if self._free_parent(bb, tree, t):
                        add("addon", t, P_TECH, "goal", count=n, cost=tree.cost(t))

            # 8. upgrades / research
            me = bb.obs.me if bb.obs is not None else None
            for u, lvl in goal.upgrades:
                if me is None:
                    break
                cur = int(me.upgrade_level[u]) if me.upgrade_level.size > u else 0
                busy = bool(me.is_upgrading[u]) if me.is_upgrading.size > u else False
                if cur >= lvl or busy or lvl != cur + 1 or ("upgrade", u) in emitted:
                    continue
                if self._reqs_done(bb, tree, tree.upgrade_requires(u, lvl)):
                    add("upgrade", u, P_UPGRADE, "goal", cost=tree.upgrade_cost(u, lvl))
            for tech in goal.techs:
                if me is None:
                    break
                has = bool(me.has_researched[tech]) if me.has_researched.size > tech else False
                busy = bool(me.is_researching[tech]) if me.is_researching.size > tech else False
                if has or busy:
                    continue
                if self._reqs_done(bb, tree, tree.tech_requires(tech)):
                    add("research", tech, P_UPGRADE, "goal", cost=tree.tech_cost(tech))

        # 9. army
        self._army(bb, tree, goal, add, reserve, worker, held)

        items.sort(key=lambda it: -it.priority)
        plan = bb.plan
        plan.items = items
        plan.notes = notes
        plan.army_order = None
        plan.cancel = []

    # ------------------------------------------------------------------ pieces
    def _need_supply(self, bb: Blackboard, tree: TechTree, supply_t: int, hall: Optional[int]) -> bool:
        w = bb.world
        bm = bb.services.get("buildings")
        pending = w.count(supply_t) - self.done(bb, supply_t) + (bm.pending_count(supply_t) if bm else 0)
        provided = int(bb.game.unit_types["supply_provided"][supply_t]) // 2
        future = w.supply_left + pending * provided
        if hall is not None:
            future += (w.count(hall) - self.done(bb, hall)) * (int(bb.game.unit_types["supply_provided"][hall]) // 2)
        producers = self.done(bb, hall) if hall is not None else 1
        for t in bb.strategy.goal.units:
            b = tree.builder(t)
            if b != -1 and tree.is_building(b):
                producers += self.done(bb, b)
        need = 2 + math.ceil(self.supply_lead * 2 * producers)
        if w.supply_total <= 10 and w.supply_used < 8:
            return False
        return future <= need

    def _free_parent(self, bb: Blackboard, tree: TechTree, addon: int) -> bool:
        parent = tree.builder(addon)
        obs = bb.obs
        if obs is None or parent == -1:
            return False
        ps = obs.my_completed(parent)
        return bool(len(ps)) and bool((ps["addon"] < 0).any())

    def min_army(self, bb: Blackboard) -> float:
        minutes = bb.frame / (24 * 60)
        by_time = min(self.army_cap, max(0.0, (minutes - self.army_from_min) * self.army_per_min))
        return max(by_time, self.enemy_army_factor * bb.belief.army_supply)

    def _army(self, bb: Blackboard, tree: TechTree, goal, add, reserve, worker, held,
              prio: int = P_ARMY) -> None:
        w = bb.world
        minerals = w.minerals - reserve[0]
        gas = w.gas - reserve[1]
        supply = w.supply_left
        by_producer: dict[int, list[int]] = {}
        for t, n in goal.units.items():
            if n <= 0 or t == worker:
                continue
            if not self._reqs_done(bb, tree, tree.unit_requires(t)):
                continue
            by_producer.setdefault(tree.builder(t), []).append(t)
        counts: dict[int, int] = {}
        for producer, types in by_producer.items():
            slots = self.done(bb, producer) - held.get(producer, 0)
            deficits = {t: goal.units[t] - self.have(bb, t) for t in types}
            for _ in range(slots):
                open_ = [t for t in types if deficits[t] > 0]
                if not open_:
                    break
                t = min(open_, key=lambda x: (self.have(bb, x) + counts.get(x, 0)) / max(1, goal.units[x]))
                m, g = tree.cost(t)
                s = tree.supply(t)
                if m > minerals or g > gas or s > supply:
                    break
                minerals, gas, supply = minerals - m, gas - g, supply - s
                deficits[t] -= 1
                counts[t] = counts.get(t, 0) + 1
        for t, n in counts.items():
            m, g = tree.cost(t)
            add("train", t, prio, "goal" if prio == P_ARMY else "min army", count=n, cost=(m * n, g * n))

    def owned_geysers(self, bb: Blackboard) -> int:
        total = 0
        for base in bb.game.bases:
            cx, cy = base.center
            if any((cx - x) ** 2 + (cy - y) ** 2 < (8 * 32) ** 2 for x, y in bb.world.depots):
                total += base.geysers
        return max(total, 1)

    def next_base(self, bb: Blackboard) -> Optional[tuple[int, int]]:
        """Closest free base to the main (the natural first), not near a known enemy base."""
        g = bb.game
        mx, my = bb.world.main_tile
        enemy = [b.tile for b in bb.belief.bases if b.alive]
        best, best_d = None, None
        for base in g.bases:
            cx, cy = base.center
            if any((cx - x) ** 2 + (cy - y) ** 2 < (8 * 32) ** 2 for x, y in bb.world.depots):
                continue
            tx, ty = base.tile
            if any((tx - ex) ** 2 + (ty - ey) ** 2 < 12 ** 2 for ex, ey in enemy):
                continue
            if base.minerals <= 0:
                continue
            d = (tx - mx) ** 2 + (ty - my) ** 2
            if base.id == g.self_natural_id:
                d = -1
            if best_d is None or d < best_d:
                best, best_d = (int(tx), int(ty)), d
        return best

    def summary(self) -> str:
        return "greedy"
