"""ProductionManager: a real queue plus per-frame train / addon / upgrade.

Strategy emits Intents every decision. `Build` is treated as "ensure this is on the
queue or already being built" — it is not re-issued as a new construction job. The
queue hands one building to BuildingManager at a time until that building has started.
Addons, trains and upgrades (in that order) spend whatever is left after construction
reservations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from bwbot import Actions, GameInfo, UnitFlag, UnitType
from bwbot.observation import UnitTypeFlag

from .buildings import CONSTRUCTING, BuildingManager
from .policy import Attack, Build, BuildAddon, Intent, Rally, Research, Train, Upgrade
from .state import State

log = logging.getLogger("mybot.production")

ArmyOrder = Union[Attack, Rally]


@dataclass
class Budget:
    minerals: int
    gas: int
    supply: int

    def can_afford(self, game: GameInfo, unit_type: int) -> bool:
        t = game.unit_types[int(unit_type)]
        return (self.minerals >= t["mineral_price"] and self.gas >= t["gas_price"]
                and self.supply >= t["supply_required"] // 2)

    def spend(self, game: GameInfo, unit_type: int) -> None:
        t = game.unit_types[int(unit_type)]
        self.minerals -= int(t["mineral_price"])
        self.gas -= int(t["gas_price"])
        self.supply -= int(t["supply_required"]) // 2

    def can_afford_upgrade(self, game: GameInfo, upgrade_type: int, level: int = 0) -> bool:
        info = upgrade_info(game, upgrade_type)
        if info is None:
            return False
        return (self.minerals >= info["mineral_price"] + level * info["mineral_price_factor"]
                and self.gas >= info["gas_price"] + level * info["gas_price_factor"])

    def spend_upgrade(self, game: GameInfo, upgrade_type: int, level: int = 0) -> None:
        info = upgrade_info(game, upgrade_type)
        if info is None:
            return
        self.minerals -= int(info["mineral_price"] + level * info["mineral_price_factor"])
        self.gas -= int(info["gas_price"] + level * info["gas_price_factor"])

    def can_afford_tech(self, game: GameInfo, tech_type: int) -> bool:
        info = tech_info(game, tech_type)
        return info is not None and self.minerals >= info["mineral_price"] and self.gas >= info["gas_price"]

    def spend_tech(self, game: GameInfo, tech_type: int) -> None:
        info = tech_info(game, tech_type)
        if info is not None:
            self.minerals -= int(info["mineral_price"])
            self.gas -= int(info["gas_price"])


class ProductionManager:
    def __init__(self, max_starting: int = 1) -> None:
        self.max_starting = max_starting     # building jobs walking to their site at once
        self.queue: list[int] = []
        self.targets: dict[int, tuple[Optional[tuple[int, int]], bool]] = {}   # queued type -> (near, exact)
        self._trains: list[int] = []
        self._addons: list[int] = []
        self._upgrades: list[int] = []
        self._researches: list[int] = []

    def reset(self) -> None:
        self.queue.clear()
        self.targets.clear()
        self._trains.clear()
        self._addons.clear()
        self._upgrades.clear()
        self._researches.clear()

    def ingest(self, intents: list[Intent], buildings: BuildingManager) -> Optional[ArmyOrder]:
        """Fold this frame's intents into the queue. Returns the last army order, if any."""
        self._trains.clear()
        self._addons.clear()
        self._upgrades.clear()
        self._researches.clear()
        army: Optional[ArmyOrder] = None
        for intent in intents:
            if isinstance(intent, Build):
                self.ensure_build(intent.unit_type, buildings)
            elif isinstance(intent, Train):
                self._trains.append(intent.unit_type)
            elif isinstance(intent, BuildAddon):
                self._addons.append(intent.unit_type)
            elif isinstance(intent, Upgrade):
                self._upgrades.append(intent.upgrade_type)
            elif isinstance(intent, Research):
                self._researches.append(intent.tech_type)
            elif isinstance(intent, (Attack, Rally)):
                army = intent
        return army

    def ensure_build(self, unit_type: int, buildings: BuildingManager, front: bool = False,
                     near: Optional[tuple[int, int]] = None, exact: bool = False) -> None:
        """Queue `unit_type` unless it is queued or starting. `front` jumps the queue (urgent items);
        `near` is the preferred tile, or the exact top-left tile when `exact` (e.g. a CC at a base)."""
        unit_type = int(unit_type)
        if buildings.starting(unit_type):
            return
        if unit_type in self.queue:
            if front and self.queue[0] != unit_type:
                self.queue.remove(unit_type)
                self.queue.insert(0, unit_type)
            return
        if front:
            self.queue.insert(0, unit_type)
        else:
            self.queue.append(unit_type)
        if near is not None:
            self.targets[unit_type] = (near, exact)
        log.info("queued building type %d (queue %d%s)", unit_type, len(self.queue), ", front" if front else "")

    def cancel(self, unit_type: int, buildings: BuildingManager, workers=None, placer=None, game=None) -> None:
        """Drop queued and not-yet-started jobs of `unit_type`."""
        unit_type = int(unit_type)
        if unit_type in self.queue:
            self.queue.remove(unit_type)
        self.targets.pop(unit_type, None)
        buildings.cancel(unit_type, workers, placer, game)

    def update(self, s: State, act: Actions, buildings: BuildingManager) -> None:
        budget = Budget(
            s.minerals - buildings.reserved_minerals(s.game),
            s.gas - buildings.reserved_gas(s.game),
            s.supply_left,
        )
        if self.queue and self._morph_building(self.queue[0], s, act, budget):
            self.queue.pop(0)
        starting = sum(1 for t in buildings.tasks if t.status != CONSTRUCTING)
        while starting < self.max_starting and self.queue:
            nxt = self.queue[0]
            if not (budget.can_afford(s.game, nxt) and _prereqs_ready(s, nxt)):
                break
            near, exact = self.targets.pop(nxt, (None, False))
            buildings.add(nxt, near=near, exact=exact)
            self.queue.pop(0)
            budget.spend(s.game, nxt)
            starting += 1
            log.info("f%d start build job %s", s.frame, s.game.type_name(nxt))

        self._addon_parents: set[int] = set()
        for unit_type in self._addons:         # before trains, which would take the idle parent
            self._addon(unit_type, s, act, budget)
        for unit_type in self._trains:
            self._train(unit_type, s, act, budget)
        for upgrade_type in self._upgrades:
            self._upgrade(upgrade_type, s, act, budget)
        for tech_type in self._researches:
            self._research(tech_type, s, act, budget)

    def _morph_building(self, unit_type: int, s: State, act: Actions, budget: Budget) -> bool:
        """Buildings made from another building (Lair, Hive, Sunken/Spore Colony, Greater Spire):
        morph a completed source instead of sending a worker. False for worker-built types."""
        src_type = int(s.game.unit_types["what_builds"][int(unit_type)])
        if not 0 <= src_type < len(s.game.unit_types) or \
                not int(s.game.unit_types["flags"][src_type]) & int(UnitTypeFlag.Building):
            return False
        if not budget.can_afford(s.game, unit_type) or not _prereqs_ready(s, unit_type):
            return False
        src = s.obs.my_completed(src_type)
        if not len(src):
            return False
        act.morph(src[0], unit_type)
        budget.spend(s.game, unit_type)
        log.info("f%d morph %s", s.frame, s.game.type_name(unit_type))
        return True

    def _train(self, unit_type: int, s: State, act: Actions, budget: Budget) -> None:
        if not budget.can_afford(s.game, unit_type):
            return
        producer_type = int(s.game.unit_types["what_builds"][int(unit_type)])
        producers = s.obs.my_completed(producer_type)
        if producer_type == UnitType.Zerg_Larva:
            if len(producers):
                act.morph(producers[0], unit_type)
                budget.spend(s.game, unit_type)
            return
        idle = [p for p in producers[producers["train_queue_count"] == 0]
                if int(p["id"]) not in getattr(self, "_addon_parents", ())]
        if idle:
            act.train(idle[0], unit_type)
            budget.spend(s.game, unit_type)

    def _addon(self, unit_type: int, s: State, act: Actions, budget: Budget) -> None:
        if not budget.can_afford(s.game, unit_type) or s.count(unit_type) > 0:
            return
        parent_type = int(s.game.unit_types["what_builds"][int(unit_type)])
        for parent in s.obs.my_completed(parent_type):
            if int(parent["addon"]) >= 0 or (int(parent["flags"]) & int(UnitFlag.Lifted)):
                continue
            act.build_addon(parent, unit_type)
            budget.spend(s.game, unit_type)
            self._addon_parents.add(int(parent["id"]))
            log.info("f%d addon %s on #%d", s.frame, s.game.type_name(unit_type), int(parent["id"]))
            return

    def _upgrade(self, upgrade_type: int, s: State, act: Actions, budget: Budget) -> None:
        if s.has_upgrade(upgrade_type) or s.upgrading(upgrade_type):
            return
        level = int(s.upgrade_level[int(upgrade_type)]) if s.upgrade_level.size > int(upgrade_type) else 0
        if not budget.can_afford_upgrade(s.game, upgrade_type, level):
            return
        info = upgrade_info(s.game, upgrade_type)
        if info is None:
            return
        researchers = s.obs.my_completed(int(info["what_upgrades"]))
        idle = researchers[(researchers["remaining_research_time"] == 0)
                           & (researchers["remaining_upgrade_time"] == 0)]
        if len(idle) == 0:
            return
        act.upgrade(idle[0], upgrade_type)
        budget.spend_upgrade(s.game, upgrade_type, level)
        log.info("f%d upgrade %s", s.frame, info["name"])

    def _research(self, tech_type: int, s: State, act: Actions, budget: Budget) -> None:
        me = s.obs.me
        t = int(tech_type)
        if (me.has_researched.size > t and me.has_researched[t]) or (me.is_researching.size > t and me.is_researching[t]):
            return
        if not budget.can_afford_tech(s.game, t):
            return
        info = tech_info(s.game, t)
        if info is None:
            return
        researchers = s.obs.my_completed(int(info["what_researches"]))
        idle = researchers[(researchers["remaining_research_time"] == 0)
                           & (researchers["remaining_upgrade_time"] == 0)]
        if len(idle) == 0:
            return
        act.research(idle[0], t)
        budget.spend_tech(s.game, t)
        log.info("f%d research %s", s.frame, info["name"])


def upgrade_info(game: GameInfo, upgrade_type: int) -> Optional[dict]:
    for info in game.upgrade_types:
        if int(info["id"]) == int(upgrade_type):
            return info
    return None


def tech_info(game: GameInfo, tech_type: int) -> Optional[dict]:
    for info in game.tech_types:
        if int(info["id"]) == int(tech_type):
            return info
    return None


def _prereqs_ready(s: State, unit_type: int) -> bool:
    """True if every required unit type is completed (BWAPI required_units)."""
    reqs = s.game.unit_type_required_units
    if not (0 <= int(unit_type) < len(reqs)):
        return True
    for req in reqs[int(unit_type)]:
        if req < 0:
            continue
        if s.count_completed(req) == 0 and s.count(req) == 0:
            return False
        # Factory requires Barracks completed in practice; count_completed is the safe check
        # for tech buildings. Workers/depots as requirements are already satisfied.
        if s.game.type_flags(req) & UnitTypeFlag.Building:
            if s.count_completed(req) == 0:
                return False
    return True
