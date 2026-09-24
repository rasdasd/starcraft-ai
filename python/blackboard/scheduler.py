"""Phased scheduler: SENSE -> DECIDE -> PLAN -> ACT -> REPORT once per decision.

Inside a phase, a component that writes a section runs before same-phase components that read it;
remaining ties use `Component.order`. A component is due when its period elapsed, when one of its
trigger events was raised, or on the first decision. When the decision's time budget is spent,
periodic DECIDE/PLAN/REPORT components are deferred (they stay due). A component that raises
`max_failures` times in a row, or overruns its own time limit `max_slow` times in a row, is replaced
by its `fallback` for the rest of the game.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .board import Blackboard
from .component import Component, Phase
from .sections import PRIVILEGED

log = logging.getLogger("blackboard.scheduler")


class ContractError(ValueError):
    pass


@dataclass
class ComponentStats:
    ticks: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    failures: int = 0
    consecutive_failures: int = 0
    slow: int = 0
    deferred: int = 0
    violations: int = 0
    replaced_by: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.ticks if self.ticks else 0.0


def validate(components: Iterable[Component], schema: dict[str, type]) -> None:
    """Static checks: known sections, one writer slot per section, unique slots, privileged reads."""
    writers: dict[str, str] = {}
    slots: dict[str, Component] = {}
    for c in components:
        key = c.slot or c.name
        if key in slots:
            raise ContractError(f"slot {key!r} filled twice ({slots[key].name}, {c.name})")
        slots[key] = c
        for s in (*c.reads, *c.writes):
            if s not in schema:
                raise ContractError(f"{c.name}: unknown section {s!r}")
        for s in c.writes:
            if s in writers and writers[s] != key:
                raise ContractError(f"section {s!r} has two writers: {writers[s]} and {key}")
            writers[s] = key
        if c.phase != Phase.REPORT:
            bad = PRIVILEGED.intersection(c.reads)
            if bad:
                raise ContractError(f"{c.name}: only REPORT components may read {sorted(bad)}")
        fb = c.fallback
        if fb is not None:
            if (fb.slot or fb.name) != key or fb.phase != c.phase or not set(fb.writes) <= set(c.writes):
                raise ContractError(f"{c.name}: fallback {fb.name} must fill the same slot/phase and write a subset")


def order_components(components: list[Component]) -> list[Component]:
    out: list[Component] = []
    for phase in Phase:
        group = sorted((c for c in components if c.phase == phase), key=lambda c: c.order)
        out.extend(_topo(group))
    return out


def _topo(group: list[Component]) -> list[Component]:
    writers = {s: c for c in group for s in c.writes}
    deps = {id(c): {id(writers[s]) for s in c.reads if s in writers and writers[s] is not c} for c in group}
    done: set[int] = set()
    out: list[Component] = []
    pending = list(group)
    while pending:
        ready = [c for c in pending if deps[id(c)] <= done]
        if not ready:
            log.warning("dependency cycle among %s; using declared order", [c.name for c in pending])
            ready = pending[:1]
        c = ready[0]
        out.append(c)
        done.add(id(c))
        pending.remove(c)
    return out


class Scheduler:
    def __init__(self, components: list[Component], time_budget_ms: float = 40.0, max_failures: int = 3,
                 component_limit_ms: float = 25.0, max_slow: int = 20, strict: bool = False) -> None:
        self.time_budget_ms = time_budget_ms
        self.max_failures = max_failures
        self.component_limit_ms = component_limit_ms
        self.max_slow = max_slow
        self.strict = strict                     # re-raise component exceptions and contract violations
        self.components: list[Component] = list(components)
        self.stats: dict[str, ComponentStats] = {}
        self._last_tick: dict[int, int] = {}
        self._schema: dict[str, type] = {}
        self.last_decision_ms = 0.0
        self.max_decision_ms = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self, bb: Blackboard) -> None:
        validate(self.components, bb.schema)
        self._schema = bb.schema
        self.components = order_components(self.components)
        self.stats = {self._key(c): ComponentStats() for c in self.components}
        self._last_tick.clear()
        for c in list(self.components):
            try:
                c.on_start(bb)
            except Exception as e:
                if self.strict:
                    raise
                log.exception("%s.on_start failed", c.name)
                self._fail(bb, c, e)

    def end(self, bb: Blackboard, won: bool) -> None:
        for c in self.components:
            try:
                c.on_end(bb, won)
            except Exception:
                if self.strict:
                    raise
                log.exception("%s.on_end failed", c.name)

    def slot(self, key: str) -> Optional[Component]:
        return next((c for c in self.components if self._key(c) == key), None)

    # ------------------------------------------------------------------ decision
    def run(self, bb: Blackboard) -> None:
        t0 = time.perf_counter()
        root = bb.root_act
        for c in list(self.components):
            if c not in self.components:          # replaced during this decision
                continue
            if not self._due(bb, c):
                continue
            key = self._key(c)
            st = self.stats[key]
            spent = (time.perf_counter() - t0) * 1000.0
            if spent > self.time_budget_ms and c.phase in (Phase.DECIDE, Phase.PLAN, Phase.REPORT) \
                    and c.period_frames > 0 and id(c) in self._last_tick:
                st.deferred += 1
                continue
            bb.act = bb.bus.scope(root, key, c.priority)
            before = {n: id(v) for n, v in bb.sections.items()}
            t1 = time.perf_counter()
            try:
                c.tick(bb)
                st.consecutive_failures = 0
            except Exception as e:
                if self.strict:
                    raise
                log.exception("%s.tick failed at frame %d", c.name, bb.frame)
                self._fail(bb, c, e)
            dt = (time.perf_counter() - t1) * 1000.0
            self._last_tick[id(c)] = bb.frame
            st.ticks += 1
            st.total_ms += dt
            st.last_ms = dt
            st.max_ms = max(st.max_ms, dt)
            self._check_writes(c, st, before, bb)
            if dt > self.component_limit_ms:
                st.slow += 1
                if st.slow >= self.max_slow and c.fallback is not None:
                    log.warning("%s too slow (%.1fms); switching to %s", c.name, dt, c.fallback.name)
                    self._replace(bb, c)
            else:
                st.slow = 0
        bb.act = root
        bb.bus.flush(root)
        self.last_decision_ms = (time.perf_counter() - t0) * 1000.0
        self.max_decision_ms = max(self.max_decision_ms, self.last_decision_ms)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _key(c: Component) -> str:
        return c.slot or c.name

    def _due(self, bb: Blackboard, c: Component) -> bool:
        last = self._last_tick.get(id(c))
        if last is None or c.period_frames <= 0:
            return True
        if bb.frame - last >= c.period_frames:
            return True
        return any(t in bb.events for t in c.triggers)

    def _check_writes(self, c: Component, st: ComponentStats, before: dict, bb: Blackboard) -> None:
        for n, v in bb.sections.items():
            if id(v) != before.get(n) and n not in c.writes:
                st.violations += 1
                msg = f"{c.name} replaced section {n!r} it does not write"
                if self.strict:
                    raise ContractError(msg)
                log.error(msg)

    def _fail(self, bb: Blackboard, c: Component, e: Exception) -> None:
        st = self.stats.setdefault(self._key(c), ComponentStats())
        st.failures += 1
        st.consecutive_failures += 1
        if len(st.errors) < 5:
            st.errors.append(f"f{bb.frame}: {type(e).__name__}: {e}")
        bb.note(f"fail.{self._key(c)}")
        if st.consecutive_failures >= self.max_failures and c.fallback is not None:
            log.warning("%s failed %d times; switching to %s", c.name, st.consecutive_failures, c.fallback.name)
            self._replace(bb, c)

    def _replace(self, bb: Blackboard, c: Component) -> None:
        fb = c.fallback
        assert fb is not None
        key = self._key(c)
        i = self.components.index(c)
        self.components[i] = fb
        st = self.stats[key]
        st.replaced_by = fb.name
        st.consecutive_failures = 0
        st.slow = 0
        bb.note(f"fallback.{key}")
        bb.say(f"{key}: {c.name} -> {fb.name}")
        try:
            fb.on_start(bb)
        except Exception:
            log.exception("fallback %s.on_start failed", fb.name)
