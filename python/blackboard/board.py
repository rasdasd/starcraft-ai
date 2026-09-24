"""The blackboard: typed sections, request queues, arbiters, and a per-decision event list."""
from __future__ import annotations

from typing import Any, Optional

from bwbot.commands import Actions
from bwbot.observation import GameInfo, Observation

from .arbiter import Budget, CommandBus, Request, RequestQueue, UnitLeases
from .recorder import Recorder
from .sections import STANDARD_SCHEMA


class Blackboard:
    def __init__(self, game: Optional[GameInfo] = None, schema: Optional[dict[str, type]] = None,
                 config: Optional[dict] = None) -> None:
        self.game = game
        self.schema = dict(STANDARD_SCHEMA if schema is None else schema)
        self.sections: dict[str, Any] = {name: cls() for name, cls in self.schema.items()}
        self.config: dict = dict(config or {})
        self.requests = RequestQueue()
        self.leases = UnitLeases()
        self.budget = Budget()
        self.bus = CommandBus(self.leases)
        self.obs: Optional[Observation] = None
        self.act: Actions = Actions()          # the running component's scoped view
        self.root_act: Actions = self.act
        self.frame = 0
        self.decision = 0
        self.events: set[str] = set()          # raised this decision; triggers for next due check
        self._next_events: set[str] = set()
        self.cache: dict[str, Any] = {}        # per-decision derived values (cleared each decision)
        self.services: dict[str, Any] = {}     # shared executor objects (worker manager, placer), not knowledge
        self.stats: dict[str, Any] = {}        # per-game counters for the recorder/report
        self.log_lines: list[str] = []
        self.recorder = Recorder(enabled=False)

    # ------------------------------------------------------------------ sections
    def __getattr__(self, name: str) -> Any:
        sections = self.__dict__.get("sections")
        if sections is not None and name in sections:
            return sections[name]
        raise AttributeError(name)

    def get(self, name: str) -> Any:
        return self.sections[name]

    def put(self, name: str, value: Any) -> None:
        if name not in self.sections:
            raise KeyError(f"unknown section {name!r}")
        self.sections[name] = value

    def reset_section(self, name: str) -> Any:
        self.sections[name] = self.schema[name]()
        return self.sections[name]

    # ------------------------------------------------------------------ events
    def raise_event(self, name: str) -> None:
        """Visible to components later in this decision and to due-checks in the next one."""
        self.events.add(name)
        self._next_events.add(name)

    def begin_decision(self, obs: Observation, act: Actions) -> None:
        self.obs = obs
        self.root_act = act
        self.act = act
        self.frame = obs.frame_count
        self.decision += 1
        self.events = self._next_events
        self._next_events = set()
        self.cache.clear()
        self.requests.expire(self.frame)
        self.budget.reset(obs.minerals, obs.gas, obs.supply_total - obs.supply_used)

    # ------------------------------------------------------------------ requests
    def request(self, kind: str, source: str, ttl: int = 48, **kw) -> Request:
        req = Request(kind=kind, source=source, **kw)
        self.requests.post(req, self.frame, ttl)
        return req

    def record(self, slot: str, kind: str, every: int = 0, **data) -> bool:
        return self.recorder.record(slot, kind, self.frame, every=every, **data)

    def note(self, key: str, inc: float = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + inc

    def say(self, text: str) -> None:
        self.log_lines.append(f"[{self.frame}] {text}")
        if len(self.log_lines) > 200:
            del self.log_lines[:100]
