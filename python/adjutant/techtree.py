"""Tech tree derived from GameInfo: builders, prerequisites, costs, missing-prerequisite chains.

BWAPI's `requiredUnits` (GameInfo.unit_type_required_units) includes the builder. Upgrade level
requirements (`whatsRequired(level)`) are not serialized, so the few that matter are listed here.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from bwbot import UnitType as U, UpgradeType as Up
from bwbot.observation import GameInfo, UnitTypeFlag as F

_LEVEL_23 = {Up.Terran_Infantry_Weapons, Up.Terran_Infantry_Armor, Up.Terran_Vehicle_Weapons,
             Up.Terran_Vehicle_Plating, Up.Terran_Ship_Weapons, Up.Terran_Ship_Plating}
UPGRADE_EXTRA: dict[tuple[int, int], list[int]] = {(int(Up.Charon_Boosters), 1): [int(U.Terran_Armory)]}
for _u in _LEVEL_23:
    for _lvl in (2, 3):
        UPGRADE_EXTRA[(int(_u), _lvl)] = [int(U.Terran_Science_Facility)]

NONE = -1


class TechTree:
    def __init__(self, game: GameInfo) -> None:
        self.game = game
        self.ut = game.unit_types
        self.req = game.unit_type_required_units
        self.upgrades = {int(d["id"]): d for d in game.upgrade_types}
        self.techs = {int(d["id"]): d for d in game.tech_types}

    # ------------------------------------------------------------------ unit types
    def flags(self, t: int) -> int:
        return int(self.ut["flags"][t])

    def is_building(self, t: int) -> bool:
        return bool(self.flags(t) & F.Building)

    def is_addon(self, t: int) -> bool:
        return bool(self.flags(t) & F.Addon)

    def is_refinery(self, t: int) -> bool:
        return bool(self.flags(t) & F.Refinery)

    def is_depot(self, t: int) -> bool:
        return bool(self.flags(t) & F.ResourceDepot)

    def race(self, t: int) -> int:
        return int(self.ut["race"][t])

    def builder(self, t: int) -> int:
        b = int(self.ut["what_builds"][t])
        return b if 0 <= b < len(self.ut) else NONE

    def cost(self, t: int) -> tuple[int, int]:
        return int(self.ut["mineral_price"][t]), int(self.ut["gas_price"][t])

    def time(self, t: int) -> int:
        return int(self.ut["build_time"][t])

    def supply(self, t: int) -> int:
        """Displayed supply used by one unit (BWAPI units / 2)."""
        return int(self.ut["supply_required"][t]) // 2

    def unit_requires(self, t: int) -> list[int]:
        out = [int(r) for r in self.req[t]] if t < len(self.req) else []
        b = self.builder(t)
        if b != NONE and b not in out:
            out.append(b)
        return out

    def known_unit(self, t: int) -> bool:
        return 0 <= t < len(self.ut) and int(self.ut["id"][t]) == t and self.builder(t) != NONE

    # ------------------------------------------------------------------ upgrades / techs
    def upgrade_requires(self, u: int, level: int = 1) -> list[int]:
        info = self.upgrades.get(int(u))
        if info is None:
            return []
        return [int(info["what_upgrades"])] + UPGRADE_EXTRA.get((int(u), int(level)), [])

    def upgrade_cost(self, u: int, level: int = 1) -> tuple[int, int]:
        i = self.upgrades[int(u)]
        k = max(0, level - 1)
        return (int(i["mineral_price"]) + k * int(i["mineral_price_factor"]),
                int(i["gas_price"]) + k * int(i["gas_price_factor"]))

    def upgrade_time(self, u: int, level: int = 1) -> int:
        i = self.upgrades[int(u)]
        return int(i["upgrade_time"]) + max(0, level - 1) * int(i["upgrade_time_factor"])

    def max_level(self, u: int) -> int:
        i = self.upgrades.get(int(u))
        return int(i["max_repeats"]) if i else 0

    def tech_requires(self, t: int) -> list[int]:
        info = self.techs.get(int(t))
        return [int(info["what_researches"])] if info else []

    def tech_cost(self, t: int) -> tuple[int, int]:
        i = self.techs[int(t)]
        return int(i["mineral_price"]), int(i["gas_price"])

    # ------------------------------------------------------------------ chains
    def missing(self, needs: Iterable[int], have: Callable[[int], int]) -> list[int]:
        """Unit types to make, in dependency order, so that every type in `needs` exists.
        `have(t)` counts owned-or-in-progress units of type t."""
        out: list[int] = []
        seen: set[int] = set()

        def visit(t: int, depth: int = 0) -> None:
            if t in seen or depth > 12:
                return
            seen.add(t)
            if have(t) > 0:
                return
            for r in self.unit_requires(t):
                if r != t and not self._is_worker_or_larva(r):
                    visit(r, depth + 1)
            out.append(t)

        for t in needs:
            visit(int(t))
        return out

    def _is_worker_or_larva(self, t: int) -> bool:
        return bool(self.flags(t) & F.Worker) or t == int(U.Zerg_Larva)

    def ready(self, needs: Iterable[int], done: Callable[[int], int]) -> bool:
        """All of `needs` exist as completed units."""
        return all(done(int(t)) > 0 for t in needs)


def tree_for(game: GameInfo, cache: Optional[dict] = None) -> TechTree:
    if cache is not None:
        t = cache.get("techtree")
        if t is None or t.game is not game:
            t = cache["techtree"] = TechTree(game)
        return t
    return TechTree(game)
