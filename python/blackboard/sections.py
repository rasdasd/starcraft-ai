"""Typed blackboard sections: the contracts between slots.

Every section has exactly one writer slot (see `STANDARD_WRITERS`). Unit and upgrade ids are raw
BWAPI ids, so the contracts are race-agnostic. Sections are plain dataclasses the writer mutates or
replaces; readers treat them as read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

_EMPTY = np.zeros(0)


# ---------------------------------------------------------------------------- world (Perception)
@dataclass
class World:
    frame: int = 0
    minerals: int = 0
    gas: int = 0
    supply_used: int = 0                    # displayed supply
    supply_total: int = 0
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))      # incl. in production
    completed: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    workers: np.ndarray = field(default_factory=lambda: _EMPTY)   # UNIT_DTYPE rows, completed
    army: np.ndarray = field(default_factory=lambda: _EMPTY)
    buildings: np.ndarray = field(default_factory=lambda: _EMPTY)
    enemies: np.ndarray = field(default_factory=lambda: _EMPTY)  # visible enemy units
    depots: list[tuple[int, int]] = field(default_factory=list)  # own resource depots (pixels)
    main_tile: tuple[int, int] = (0, 0)
    natural_tile: Optional[tuple[int, int]] = None
    main_choke: Optional[tuple[int, int]] = None
    under_attack: bool = False
    income_minerals: float = 0.0            # per game minute, smoothed
    income_gas: float = 0.0
    army_supply: int = 0

    @property
    def supply_left(self) -> int:
        return self.supply_total - self.supply_used

    def count(self, unit_type: int) -> int:
        t = int(unit_type)
        return int(self.counts[t]) if 0 <= t < self.counts.size else 0

    def count_completed(self, unit_type: int) -> int:
        t = int(unit_type)
        return int(self.completed[t]) if 0 <= t < self.completed.size else 0

    def summary(self) -> str:
        return (f"f{self.frame} {self.minerals}m {self.gas}g {self.supply_used}/{self.supply_total} "
                f"wrk {len(self.workers)} army {len(self.army)} inc {self.income_minerals:.0f}/{self.income_gas:.0f}")


# ---------------------------------------------------------------------------- meta (Meta)
@dataclass
class Meta:
    map_name: str = ""
    map_hash: str = ""
    map_size: tuple[int, int] = (0, 0)
    n_starts: int = 0
    n_bases: int = 0
    self_race: int = 8
    enemy_race: int = 8                     # Race.Unknown until seen
    enemy_name: str = ""
    main_choke_width: int = 0
    natural_distance: float = 0.0           # pixels, ground (BWEM graph) from main
    enemy_start_distances: list[float] = field(default_factory=list)   # pixels, ground
    rush_distance: float = 0.0              # min ground distance to a possible enemy start
    features: dict[str, float] = field(default_factory=dict)          # numeric map/matchup features
    map_analysis: dict[str, Any] = field(default_factory=dict)        # extension point (future work)

    def summary(self) -> str:
        return f"{self.map_name} starts {self.n_starts} rush {self.rush_distance / 32:.0f}t enemy r{self.enemy_race}"


# ---------------------------------------------------------------------------- belief (Belief)
@dataclass
class EnemyBase:
    base_id: int
    tile: tuple[int, int]
    last_seen: int
    alive: bool = True


@dataclass
class Belief:
    enemy_race: int = 8
    enemy_start: Optional[tuple[int, int]] = None          # tiles, best guess
    start_candidates: dict[tuple[int, int], float] = field(default_factory=dict)
    counts: dict[int, float] = field(default_factory=dict)   # estimated alive enemy units per type
    seen_max: dict[int, int] = field(default_factory=dict)
    dead: dict[int, int] = field(default_factory=dict)
    tech: set[int] = field(default_factory=set)              # enemy building types seen or inferred
    inferred: set[int] = field(default_factory=set)          # subset of tech that was only inferred
    first_seen: dict[int, int] = field(default_factory=dict)
    bases: list[EnemyBase] = field(default_factory=list)
    buildings: list[tuple[int, int, int]] = field(default_factory=list)   # last-known (type, x, y)
    units: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)  # id -> last (type, x, y, frame)
    army_supply: float = 0.0
    army_pos: Optional[tuple[int, int]] = None
    army_seen_frame: int = -1
    air: float = 0.0
    cloak: bool = False
    proxy: bool = False
    opening: str = "unknown"
    opening_probs: dict[str, float] = field(default_factory=dict)
    staleness: dict[int, int] = field(default_factory=dict)   # base id -> frames since seen
    predicted: dict[str, Any] = field(default_factory=dict)   # learned-model outputs, if any

    def count(self, unit_type: int) -> float:
        return float(self.counts.get(int(unit_type), 0.0))

    def summary(self) -> str:
        return (f"race {self.enemy_race} open {self.opening} army~{self.army_supply:.0f} air {self.air:.0f} "
                f"bases {sum(b.alive for b in self.bases)} tech {len(self.tech)} proxy {int(self.proxy)}")


# ---------------------------------------------------------------------------- threats (Crisis)
@dataclass
class Threat:
    kind: str
    x: int
    y: int
    severity: float                         # 0..1
    frame: int
    units: list[int] = field(default_factory=list)


@dataclass
class Threats:
    active: list[Threat] = field(default_factory=list)
    level: float = 0.0
    posture_override: Optional[str] = None  # e.g. "defend"
    override_until: int = -1

    def has(self, kind: str) -> bool:
        return any(t.kind == kind for t in self.active)

    def summary(self) -> str:
        return ",".join(f"{t.kind}:{t.severity:.1f}" for t in self.active) or "-"


# ---------------------------------------------------------------------------- strategy (Strategy)
@dataclass
class Goal:
    units: dict[int, int] = field(default_factory=dict)        # UnitType -> target count
    buildings: dict[int, int] = field(default_factory=dict)    # production / tech buildings
    upgrades: list[tuple[int, int]] = field(default_factory=list)   # (UpgradeType, level), in order
    techs: list[int] = field(default_factory=list)              # TechType
    addons: dict[int, int] = field(default_factory=dict)       # addon type -> count
    workers: int = 20
    bases: int = 1


STANCES = ("defend", "hold", "contain", "attack", "all_in")


@dataclass
class Posture:
    stance: str = "hold"                    # one of STANCES
    attack_supply: int = 40                 # army supply to start a push
    retreat_supply: int = 12
    harass: bool = False


@dataclass
class StrategyState:
    template: str = ""
    goal: Goal = field(default_factory=Goal)
    posture: Posture = field(default_factory=Posture)
    opening: list[tuple[int, int]] = field(default_factory=list)   # (supply, unit_type) steps
    opening_index: int = 0
    opening_done: bool = False
    opening_next: Optional[int] = None      # unit type the opening wants started now, if any
    switched_frame: int = 0
    history: list[tuple[int, str]] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)          # model scores per template

    def summary(self) -> str:
        return f"{self.template} {self.posture.stance} opening {'done' if self.opening_done else self.opening_index}"


# ---------------------------------------------------------------------------- engagements
@dataclass
class Engagement:
    squad: str
    center: tuple[int, int]
    own_strength: float
    enemy_strength: float
    win_prob: float
    enemy_ids: list[int] = field(default_factory=list)
    own_ids: list[int] = field(default_factory=list)
    ratio: float = 1.0                      # own / enemy Lanchester strength
    own_left: float = 1.0                   # expected surviving fraction of our units
    contact: bool = False                   # the two groups are within weapon reach


@dataclass
class Engagements:
    by_squad: dict[str, Engagement] = field(default_factory=dict)   # own army clusters ("c0", ...)
    global_ratio: float = 1.0               # whole army vs estimated enemy army
    global_win_prob: float = 0.5
    fights: int = 0                         # fights logged this game

    def summary(self) -> str:
        return " ".join(f"{k}:{e.win_prob:.2f}" for k, e in self.by_squad.items()) or f"global {self.global_ratio:.2f}"


# ---------------------------------------------------------------------------- squads (Tactics)
@dataclass
class SquadOrder:
    kind: str = "hold"                      # attack | defend | hold | retreat | hunt_air | contain | gather
    x: int = 0
    y: int = 0
    target: int = -1


@dataclass
class Squad:
    name: str
    units: set[int] = field(default_factory=set)
    order: SquadOrder = field(default_factory=SquadOrder)
    priority: int = 80


@dataclass
class Squads:
    squads: dict[str, Squad] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Squad]:
        return self.squads.get(name)

    def summary(self) -> str:
        return " ".join(f"{s.name}[{len(s.units)}]:{s.order.kind}" for s in self.squads.values()) or "-"


# ---------------------------------------------------------------------------- plan (Production)
@dataclass
class PlanItem:
    kind: str                               # build | expand | train | addon | upgrade | research
    type_id: int
    priority: int = 50
    reason: str = ""
    count: int = 1                          # trains: how many producers may take it this decision
    near: Optional[tuple[int, int]] = None  # build: preferred tile
    exact: bool = False                     # build: `near` is the exact top-left tile


@dataclass
class ProductionPlan:
    """Everything production wants started now, highest priority first. The macro executor spends
    in this order; a building job, once dispatched, lives until it is built or cancelled."""
    items: list[PlanItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cancel: list[int] = field(default_factory=list)   # building types whose unplaced jobs to drop

    def summary(self) -> str:
        head = self.items[:4]
        return " ".join(f"{i.kind[0]}{i.type_id}@{i.priority}" for i in head) or "-"


# ---------------------------------------------------------------------------- macro (Macro executor)
@dataclass
class OwnBase:
    base_id: int
    tile: tuple[int, int]                   # hall top-left
    center: tuple[int, int]                 # pixels
    completed: bool                         # hall finished (else under construction)
    patches: int = 0                        # mineral fields left
    minerals: int = 0                       # minerals left in them
    geysers: int = 0                        # geysers, with or without a refinery
    refineries: int = 0                     # our completed refineries
    miners: int = 0
    gas_workers: int = 0

    @property
    def saturation(self) -> float:
        return self.miners / (2 * self.patches) if self.patches else 1.0


@dataclass
class MacroState:
    bases: list[OwnBase] = field(default_factory=list)   # our halls at BWEM bases (not macro hatcheries)
    pending: dict[int, int] = field(default_factory=dict)   # building jobs dispatched but not placed yet
    jobs: list[str] = field(default_factory=list)            # "Type:status" per job, for logs
    reserved: tuple[int, int] = (0, 0)       # money held for dispatched, unplaced jobs
    next_base: Optional[int] = None          # base id an "expand" item would go to
    expanding: Optional[int] = None          # base id of the expansion job in flight, if any
    worker_target: int = 0                   # 2 per mineral patch + 3 per refinery at our bases
    blocked: list[str] = field(default_factory=list)         # plan items skipped last decision, and why

    def pending_count(self, unit_type: int) -> int:
        return int(self.pending.get(int(unit_type), 0))

    @property
    def mining_bases(self) -> int:
        return sum(1 for b in self.bases if b.completed and b.patches > 0)

    @property
    def base_count(self) -> int:
        """Bases with minerals left, counting ones under construction and an expansion in flight."""
        n = sum(1 for b in self.bases if b.patches > 0)
        return n + (1 if self.expanding is not None and all(b.base_id != self.expanding for b in self.bases) else 0)

    @property
    def geysers(self) -> int:
        return sum(b.geysers for b in self.bases)

    def summary(self) -> str:
        sat = " ".join(f"{b.base_id}:{b.miners}/{2 * b.patches}" for b in self.bases)
        return f"bases {sat or '-'} jobs {','.join(self.jobs) or '-'} next {self.next_base}"


# ---------------------------------------------------------------------------- scouting
@dataclass
class ScoutingState:
    scouts: dict[int, tuple[int, int]] = field(default_factory=dict)   # unit id -> destination (px)
    targets: list[tuple[int, int]] = field(default_factory=list)       # pixels, most wanted first
    last_scan_frame: int = -10_000
    initial_done: bool = False

    def summary(self) -> str:
        return f"scouts {len(self.scouts)} targets {len(self.targets)}"


# ---------------------------------------------------------------------------- truth (training only)
@dataclass
class Truth:
    enabled: bool = False
    frame: int = 0
    counts: dict[int, int] = field(default_factory=dict)    # all enemy units per type
    buildings: set[int] = field(default_factory=set)
    bases: list[tuple[int, int]] = field(default_factory=list)
    army_supply: float = 0.0


STANDARD_SCHEMA: dict[str, type] = {
    "world": World,
    "meta": Meta,
    "belief": Belief,
    "threats": Threats,
    "strategy": StrategyState,
    "engagements": Engagements,
    "squads": Squads,
    "plan": ProductionPlan,
    "macro": MacroState,
    "scouting": ScoutingState,
    "truth": Truth,
}

# Sections only REPORT-phase components may read.
PRIVILEGED = frozenset({"truth"})
