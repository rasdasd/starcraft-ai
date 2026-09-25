"""Build-order search over `EconSim`: the shortest-makespan way to reach a target.

`Target` = unit counts (started, incl. buildings/addons/workers), upgrade levels and techs.
Candidates at a state are the actions that make progress: types below their count (or the first
startable missing prerequisite of one), the next upgrade level, missing techs, a supply building
when supply would block, and a refinery when the rest of the target needs gas.

Search: every candidate first action is completed by two greedy rollout policies (earliest start
first; target order among the actions that can start within `order_slack` of the earliest) and scored by the frame the whole target is done plus the mean start
offset (so among equally long plans the one that starts things sooner wins). Both policies make a
worker whenever one can start within `order_slack` of the earliest action (constant worker
production: a makespan objective over a short target undervalues workers). The best first action
and its rollout are kept, then the second action is improved the same way while time is left.
With the deadline hit it returns the best sequence found so far (at least the greedy one).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .econsim import Action, EconSim, SimState


@dataclass
class Target:
    counts: dict[int, int] = field(default_factory=dict)
    upgrades: dict[int, int] = field(default_factory=dict)
    techs: set[int] = field(default_factory=set)
    order: list[Action] = field(default_factory=list)       # preference order for the "target order" policy

    def done(self, s: SimState) -> bool:
        return (all(s.started.get(t, 0) >= n for t, n in self.counts.items())
                and all(s.upgrades.get(u, 0) >= lvl for u, lvl in self.upgrades.items())
                and all(t in s.techs for t in self.techs))


@dataclass
class Plan:
    steps: list[tuple[Action, int]]      # (action, start frame)
    makespan: int
    evaluated: int
    ms: float
    score: float = 0.0                   # makespan + start_weight x mean start offset (lower is better)


class BuildSearch:
    def __init__(self, sim: EconSim, supply_type: int, refinery: int, max_steps: int = 60,
                 start_weight: float = 1.0, order_slack: int = 48) -> None:
        self.sim = sim
        self.start_weight = start_weight
        self.order_slack = order_slack
        self.worker_action = Action("unit", sim.worker)
        self.supply_type = supply_type
        self.refinery = refinery
        self.max_steps = max_steps
        self.evaluated = 0

    # ------------------------------------------------------------------ candidates
    def candidates(self, s: SimState, target: Target) -> list[Action]:
        sim, tree = self.sim, self.sim.tree
        out: list[Action] = []
        seen: set[Action] = set()

        def push(a: Action) -> None:
            if a not in seen and sim.legal(s, a):
                seen.add(a)
                out.append(a)

        gas_needed = False
        for a in target.order:
            if a.kind == "unit":
                if s.started.get(a.type_id, 0) >= target.counts.get(a.type_id, 0):
                    continue
            elif a.kind == "upgrade":
                if s.upgrades.get(a.type_id, 0) >= target.upgrades.get(a.type_id, 0):
                    continue
                a = Action("upgrade", a.type_id, s.upgrades.get(a.type_id, 0) + 1)
            elif a.type_id in s.techs:
                continue
            if sim.cost(a)[1] > 0:
                gas_needed = True
            reqs = sim.reqs(a)
            missing = tree.missing(reqs, lambda x: s.started.get(x, 0))
            if not missing:
                push(a)
            for m in missing:
                if all(s.started.get(r, 0) > 0 for r in sim.reqs(Action("unit", m))):
                    if sim.cost(Action("unit", m))[1] > 0:
                        gas_needed = True
                    push(Action("unit", m))
                    break
        if gas_needed and s.started.get(self.refinery, 0) == 0:
            push(Action("unit", self.refinery))
        if self._supply_short(s, out):
            push(Action("unit", self.supply_type))
        return out

    def _supply_short(self, s: SimState, cands: list[Action]) -> bool:
        ut = self.sim.tree.ut
        eventually = self.sim._supply_eventually(s)
        if eventually >= 400:
            return False
        need = max((int(ut["supply_required"][a.type_id]) for a in cands if a.kind == "unit"), default=0)
        producers = sum(len(v) for k, v in s.slots.items() if not self.sim.tree.is_addon(k)) or 1
        return eventually - s.supply_used < max(need, 2) * (1 + producers)

    # ------------------------------------------------------------------ rollouts
    def rollout(self, s: SimState, target: Target, policy: str, first: Optional[list[Action]] = None,
                deadline: float = float("inf")) -> Optional[Plan]:
        t0 = s.frame
        s = s.copy()
        steps: list[tuple[Action, int]] = []
        for a in first or ():
            f = self.sim.do(s, a)
            if f is None:
                return None
            steps.append((a, f))
        for _ in range(self.max_steps):
            if target.done(s):
                break
            cands = self.candidates(s, target)
            if not cands:
                break
            times = []
            for i, a in enumerate(cands):
                f = self.sim.when(s, a)
                self.evaluated += 1
                if f is not None:
                    times.append((f, i, a))
            if not times:
                break
            first = min(times)
            worker = next((t for t in times if t[2] == self.worker_action and t[0] <= first[0] + self.order_slack),
                          None)
            if worker is not None:         # constant worker production, as every build order does
                f, _, pick = worker
            elif policy == "order":        # target order among what can start about as soon as the earliest
                f, _, pick = min((t for t in times if t[0] <= first[0] + self.order_slack), key=lambda t: t[1])
            else:
                f, _, pick = first
            self.sim.do(s, pick, at=f)
            steps.append((pick, f))
            if time.perf_counter() > deadline:
                break
        penalty = 0 if target.done(s) else 24 * 60 * 20
        makespan = self.sim.finish(s) + penalty
        mean_start = sum(f - t0 for _, f in steps) / len(steps) if steps else 0.0
        return Plan(steps, makespan, 0, 0.0, makespan + self.start_weight * mean_start)

    def search(self, s: SimState, target: Target, budget_ms: float = 15.0,
               forced: Optional[list[Action]] = None) -> Plan:
        t0 = time.perf_counter()
        deadline = t0 + budget_ms / 1000
        self.evaluated = 0
        forced = list(forced or [])
        best: Optional[Plan] = None
        for policy in ("earliest", "order"):
            p = self.rollout(s, target, policy, forced, deadline=float("inf") if best is None else deadline)
            if p is not None and (best is None or p.score < best.score):
                best = p
        prefix = list(forced)
        for depth in range(2):
            if best is None or time.perf_counter() > deadline:
                break
            base = s.copy()
            for a in prefix:
                if self.sim.do(base, a) is None:
                    break
            improved = False
            for a in self.candidates(base, target):
                if time.perf_counter() > deadline:
                    break
                for policy in ("earliest", "order"):
                    p = self.rollout(s, target, policy, prefix + [a], deadline)
                    if p is not None and p.score < best.score:
                        best, improved = p, True
            if len(best.steps) <= len(prefix):
                break
            prefix = [a for a, _ in best.steps[:len(prefix) + 1]]
            if not improved and depth > 0:
                break
        if best is None:
            best = Plan([], s.frame, 0, 0.0)
        best.evaluated = self.evaluated
        best.ms = (time.perf_counter() - t0) * 1000
        return best
