"""Production slot, v1: greedy, tech-tree aware planner from `strategy.goal`.

Each decision it rewrites `plan.items` in priority order:
  supply (projected from production capacity) > opening step > workers > missing prerequisites >
  goal buildings / expansions / addons > upgrades and research > army trains.
Army trains only spend what is left after reserving the cost of the next unstarted building and
of ready upgrades/research/addons, so production does not starve tech. Counts include the macro
executor's dispatched jobs (`macro.pending`); expansions are `expand` items (the executor picks the
base) while `macro.base_count` is below the goal.
"""
from __future__ import annotations

import math
from typing import Optional

from blackboard import Blackboard, Component, Phase, Priority
from blackboard.profile import register
from blackboard.sections import PlanItem
from bwbot import Race, UnitType as U
from bwbot.observation import UnitTypeFlag as F

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

SATURATION_SLACK = 2        # this close to the worker target counts as saturated
FLOAT_MINERALS = 250        # ... and a saturated economy floating this much takes another base
LARVA_FLOAT = 600           # Zerg banking this much with no larva adds a hatchery


@register("GreedyPlanner")
class GreedyPlanner(Component):
    phase = Phase.PLAN
    reads = ("world", "meta", "belief", "strategy", "threats", "macro")
    writes = ("plan",)

    def __init__(self, supply_lead: float = 1.0, max_supply: int = 200, army_from_min: float = 3.0,
                 army_per_min: float = 4.0, army_cap: float = 30.0, enemy_army_factor: float = 1.0,
                 float_minerals: int = 500, float_short: int = 400, producers_per_base: int = 3,
                 float_train: int = 200) -> None:
        self.float_train = float_train          # bank (after goal trains) spent on filler units
        self.float_minerals = float_minerals    # bank (after this decision's spending) that adds a producer
        self.float_short = float_short          # ... while the army is short
        self.producers_per_base = producers_per_base
        self.supply_lead = supply_lead      # supply buffer, in "decisions of full production" units
        self.max_supply = max_supply
        self.army_from_min = army_from_min  # minimum army supply: army_per_min per minute after this
        self.army_per_min = army_per_min
        self.army_cap = army_cap
        self.enemy_army_factor = enemy_army_factor   # ... and at least this x the known enemy army

    # ------------------------------------------------------------------ helpers
    def have(self, bb: Blackboard, t: int) -> int:
        """Owned (incl. in production) + dispatched to a builder but not placed yet."""
        return bb.world.count(t) + bb.macro.pending_count(t)

    def bases(self, bb: Blackboard, hall: Optional[int]) -> int:
        """Mining bases, including one being taken (the tracker has none before the first update)."""
        if bb.macro.bases or hall is None:
            return bb.macro.base_count
        return self.have(bb, hall)

    def want_bases(self, bb: Blackboard, goal, hall: Optional[int]) -> int:
        """The goal's bases, or one more than we mine once those are saturated and minerals pile
        up: a build's base count is a floor, not a cap."""
        m, w = bb.macro, bb.world
        now = self.bases(bb, hall)
        if goal.bases > now or hall is None or m.expanding is not None or m.worker_target <= 0:
            return goal.bases
        target = min(m.worker_target, goal.workers) if goal.workers > 0 else m.worker_target
        if len(w.workers) >= target - SATURATION_SLACK and w.minerals >= FLOAT_MINERALS \
                and not w.under_attack:
            return now + 1
        return goal.bases

    def _larva_starved(self, bb: Blackboard, hall: int) -> bool:
        """Zerg floating minerals with no larva and no hatchery on the way: another hatchery pays."""
        w = bb.world
        return (self.race == int(Race.Zerg) and w.minerals >= LARVA_FLOAT and w.count(int(U.Zerg_Larva)) == 0
                and w.count(hall) == self.done(bb, hall) and not bb.macro.pending_count(hall))

    def done(self, bb: Blackboard, t: int) -> int:
        return bb.world.count_completed(t)

    def _reqs_done(self, bb: Blackboard, tree: TechTree, reqs) -> bool:
        skip = (WORKER.get(self.race), int(U.Zerg_Larva))     # always there again soon
        return all(self.done(bb, r) > 0 for r in reqs if r not in skip)

    # ------------------------------------------------------------------ tick
    def on_start(self, bb: Blackboard) -> None:
        self.tree = TechTree(bb.game)
        self.race = int(bb.game.self_race)
        producers = {self.tree.builder(t) for t in range(len(self.tree.ut)) if self.tree.known_unit(t)}
        self.filler_map = {p: f for p in sorted(producers) if p != -1 and (f := self.tree.fillers(p, self.race))
                           and (self.tree.is_building(p) or p == int(U.Zerg_Larva))}

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
            if kind in ("build", "expand"):
                if not bb.macro.pending_count(t) and not first_build_reserved[0]:
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
                self._add_supply(tree, add, supply_t)
                notes.append("supply")

        # 1b. requests from other slots (crisis: bunker / turret / comsat, cancel an expansion)
        cancels = self._requests(bb, tree, add, worker)
        cancel_hall = hall is not None and hall in cancels

        # 2. opening
        opening = not st.opening_done and st.opening_next is not None
        if opening:
            t = int(st.opening_next)
            if self._reqs_done(bb, tree, tree.unit_requires(t)):
                if not tree.is_building(t):
                    add("train", t, P_OPENING, "opening", cost=tree.cost(t))
                elif t == hall and self.done(bb, hall) > 0:
                    if not cancel_hall:
                        add("expand", t, P_OPENING, "opening expand")
                else:
                    add("build", t, P_OPENING, "opening")

        # 3. workers
        if worker is not None:
            deficit = goal.workers - self.have(bb, worker)
            halls = self.done(bb, hall) if hall is not None else 0
            if deficit > 0 and halls > 0:
                add("train", worker, P_WORKER, "workers", count=min(deficit, halls))

        # types the opening still has to make (it may be waiting for supply): the goal does not
        # start them ahead of it
        later = set() if st.opening_done else {int(t) for _, t in st.opening[st.opening_index:]}

        # 3a. addons: before any army trains, so their parent is held idle for them
        if not opening:
            for t, n in goal.addons.items():
                if n > self.have(bb, t) and self._reqs_done(bb, tree, tree.unit_requires(t)):
                    if self._free_parent(bb, tree, t):
                        add("addon", t, P_TECH, "goal", count=n, cost=tree.cost(t))

        # 3b. too little army for the game time / the enemy we know about: units before
        # expansions and tech (and before their money is reserved), also during the opening
        need = self.min_army(bb)
        short = w.army_supply < need
        if short:
            severe = w.army_supply < 0.5 * need and (need >= 8 or bb.belief.army_supply > 0)
            prio = P_WORKER + 1 if severe else P_ARMY_URGENT
            left, used = self._army(bb, tree, goal, add, reserve, worker, held, prio=prio)
            # whatever idle producers can make now, while the goal's units are not available yet
            self._filler(bb, tree, add, left, used, held, prio=prio, start=0, keep=0)
            notes.append(f"army {w.army_supply}<{need:.0f}")

        if not opening:
            # 4. prerequisites for everything the goal asks for
            needs: list[int] = []
            for t, n in goal.units.items():
                if n > 0:
                    needs += [r for r in tree.unit_requires(t) if r != worker]
            for t, n in list(goal.buildings.items()) + list(goal.addons.items()):
                if n > 0:
                    needs += [r for r in tree.unit_requires(t) if r != worker]
            me = bb.obs.me if bb.obs is not None else None
            tech_now = not short and st.opening_done
            for u, lvl in goal.upgrades if tech_now else ():
                cur = int(me.upgrade_level[u]) if me is not None and me.upgrade_level.size > u else 0
                if lvl == cur + 1:              # only the next level's requirements
                    needs += tree.upgrade_requires(u, lvl)
            for tech in goal.techs if tech_now else ():
                needs += tree.tech_requires(tech)
            for t in tree.missing(needs, lambda x: self.have(bb, x)):
                if t == worker or t == hall and self.done(bb, hall) > 0 or t in later:
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
            # 5. expansions
            bases_now, bases_want = self.bases(bb, hall), self.want_bases(bb, goal, hall)
            if hall is not None and bases_want > bases_now and not cancel_hall and hall not in later:
                add("expand", hall, P_EXPAND, "expand")
                notes.append(f"expand {bases_now}->{bases_want}")
            elif hall is not None and not cancel_hall and self._larva_starved(bb, hall):
                if bb.macro.next_base is not None:
                    add("expand", hall, P_EXPAND, "larva")
                else:
                    add("build", hall, P_EXPAND, "macro hatch")
                notes.append("larva")

            # 6. goal buildings
            geysers = max(1, bb.macro.geysers)
            for t, n in goal.buildings.items():
                if t == refinery:
                    n = min(n, geysers)
                if n <= self.have(bb, t) or t in later or not self._reqs_done(bb, tree, tree.unit_requires(t)):
                    continue
                producer = any(int(tree.builder(u)) == t for u in goal.units)
                if not producer and short:
                    continue
                add("build", t, P_BUILDING if producer else P_TECH, "goal")

            # 8. upgrades / research
            me = bb.obs.me if bb.obs is not None and not short else None
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

        # 9. army, then 10. filler units from producers the goal leaves idle while money piles up
        left, used = self._army(bb, tree, goal, add, reserve, worker, held)
        self._filler(bb, tree, add, left, used, held)

        items.sort(key=lambda it: -it.priority)
        plan = bb.plan
        plan.items = items
        plan.notes = notes
        plan.cancel = cancels

    # ------------------------------------------------------------------ pieces
    def _requests(self, bb: Blackboard, tree: TechTree, add, worker) -> list[int]:
        cancels: list[int] = []
        for r in bb.requests.active("production"):
            t = int(r.type_id)
            if r.item == "cancel":
                cancels.append(t)
                continue
            why = f"req:{r.source}"
            if r.item in ("build", "addon", "train"):
                if r.item != "train" and self.have(bb, t) >= r.count:
                    continue
                reqs = tree.unit_requires(t)
                if not self._reqs_done(bb, tree, reqs):
                    for p in tree.missing(reqs, lambda x: self.have(bb, x)):
                        if p == worker or not tree.is_building(p) or tree.is_addon(p):
                            continue
                        if self._reqs_done(bb, tree, tree.unit_requires(p)):
                            add("build", p, r.priority - 1, why + " prereq")
                    continue
                if r.item == "build":
                    add("build", t, r.priority, why, near=r.near, exact=r.exact)
                elif r.item == "addon":
                    if self._free_parent(bb, tree, t):
                        add("addon", t, r.priority, why, count=r.count, cost=tree.cost(t))
                else:
                    m, g = tree.cost(t)
                    add("train", t, r.priority, why, count=r.count, cost=(m * r.count, g * r.count))
            elif r.item == "upgrade":
                add("upgrade", t, r.priority, why)
            elif r.item == "research":
                add("research", t, r.priority, why, cost=tree.tech_cost(t))
        return cancels

    @staticmethod
    def _add_supply(tree: TechTree, add, supply_t: int) -> None:
        """Depots and pylons are built; overlords are trained from larva."""
        if tree.is_building(supply_t):
            add("build", supply_t, P_SUPPLY, "supply")
        else:
            add("train", supply_t, P_SUPPLY, "supply", cost=tree.cost(supply_t))

    def _need_supply(self, bb: Blackboard, tree: TechTree, supply_t: int, hall: Optional[int]) -> bool:
        w = bb.world
        pending = w.count(supply_t) - self.done(bb, supply_t) + bb.macro.pending_count(supply_t)
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
              prio: int = P_ARMY) -> tuple[tuple[int, int, int], dict[int, int]]:
        """Goal-mix trains on free producers. Returns the (minerals, gas, supply) left and the
        producer slots used, per producer type."""
        w = bb.world
        minerals = w.minerals - bb.macro.reserved[0] - reserve[0]
        gas = w.gas - bb.macro.reserved[1] - reserve[1]
        supply = w.supply_left
        by_producer: dict[int, list[int]] = {}
        for t, n in goal.units.items():
            if n <= 0 or t == worker:
                continue
            if not self._reqs_done(bb, tree, tree.unit_requires(t)):
                continue
            by_producer.setdefault(tree.builder(t), []).append(t)
        urgent = prio != P_ARMY
        counts: dict[int, int] = {}
        for producer, types in by_producer.items():
            slots = self.done(bb, producer) - held.get(producer, 0)
            # short on army: keep producing the goal mix past its counts
            deficits = {t: 10 ** 6 if urgent and tree.flags(t) & F.CanAttack else goal.units[t] - self.have(bb, t)
                        for t in types}
            for _ in range(slots):
                open_ = [t for t in types if deficits[t] > 0]
                open_.sort(key=lambda x: (self.have(bb, x) + counts.get(x, 0)) / max(1, goal.units[x]))
                pick = None
                for t in open_ if urgent else open_[:1]:
                    m, g = tree.cost(t)
                    if m <= minerals and (g == 0 or g <= gas) and tree.supply(t) <= supply:
                        pick = t
                        break
                if pick is None:
                    break
                m, g = tree.cost(pick)
                minerals, gas, supply = minerals - m, gas - g, supply - tree.supply(pick)
                deficits[pick] -= 1
                counts[pick] = counts.get(pick, 0) + 1
        if by_producer and minerals >= (self.float_short if urgent else self.float_minerals):
            # floating with every producer busy: another production building for the main army
            # type (a macro hatchery when the producer is larva), up to `producers_per_base`
            main = max(goal.units, key=lambda t: goal.units[t] * tree.supply(t) if t != worker else -1)
            producer = tree.builder(main)
            if producer != -1 and not tree.is_building(producer):
                producer = HALL.get(self.race, -1)
            cap = self.producers_per_base * max(1, len(w.depots))
            if producer != -1 and tree.is_building(producer) and not tree.is_addon(producer) \
                    and self.have(bb, producer) - self.done(bb, producer) == 0 and self.have(bb, producer) < cap:
                add("build", producer, P_BUILDING, "min army producer" if urgent else "float producer")
        used: dict[int, int] = {}
        for t, n in counts.items():
            m, g = tree.cost(t)
            add("train", t, prio, "goal" if prio == P_ARMY else "min army", count=n, cost=(m * n, g * n))
            used[tree.builder(t)] = used.get(tree.builder(t), 0) + n
        return (minerals, gas, supply), used

    def _filler(self, bb: Blackboard, tree: TechTree, add, left, used, held, prio: int = P_ARMY - 1,
                start: Optional[int] = None, keep: Optional[int] = None) -> None:
        """Spend a surplus above `start` (default `float_train`), down to `keep`, on the cheapest
        combat unit of each idle producer (marines from a barracks the build does not use yet,
        zealots, zerglings)."""
        start = self.float_train if start is None else start
        keep = self.float_train - 100 if keep is None else keep
        minerals, _, supply = left
        if minerals < start:
            return
        for producer, types in self.filler_map.items():
            slots = self.done(bb, producer) - held.get(producer, 0) - used.get(producer, 0)
            if producer == int(U.Zerg_Larva):     # larva is shared with drones and the goal mix
                slots = min(slots, max(1, int(minerals // 400)))
            n, pick = 0, None
            for t in types:
                if tree.cost(t)[1] == 0 and self._reqs_done(bb, tree, tree.unit_requires(t)):
                    pick = t
                    break
            if pick is None:
                continue
            m, g = tree.cost(pick)
            while n < slots and minerals - m >= keep and tree.supply(pick) <= supply:
                minerals, supply, n = minerals - m, supply - tree.supply(pick), n + 1
            if n:
                add("train", pick, prio, "filler", count=n, cost=(m * n, 0))

    def summary(self) -> str:
        return "greedy"
