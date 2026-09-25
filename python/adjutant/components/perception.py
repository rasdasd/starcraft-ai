"""Perception: observation -> `world` (and `truth` in training games with full map information)."""
from __future__ import annotations

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from bwbot import UnitFlag, UnitType as U
from bwbot.observation import UnitTypeFlag

INCOME_ALPHA = 0.05      # EMA per decision
UNDER_ATTACK_FRAMES = 24 * 5
# units morphing inside these count as their `build_type` (BWAPI counts them as the egg)
EGGS = (int(U.Zerg_Egg), int(U.Zerg_Lurker_Egg), int(U.Zerg_Cocoon))


def unit_counts(obs, game) -> tuple[np.ndarray, np.ndarray]:
    """(all incl. in production, completed) own units per type; eggs count as what they hatch into."""
    me, n = obs.me, len(game.unit_types)
    counts = me.all_unit_count if me.all_unit_count.size >= n else np.zeros(n, dtype=np.int32)
    completed = me.completed_unit_count if me.completed_unit_count.size >= n else np.zeros(n, dtype=np.int32)
    own = obs.my_units
    eggs = own[np.isin(own["type"], EGGS) & (own["build_type"] >= 0) & (own["build_type"] < n)]
    if len(eggs):
        counts = counts.copy()
        for bt in eggs["build_type"]:
            counts[int(bt)] += 2 if int(game.unit_types["flags"][int(bt)]) & UnitTypeFlag.TwoUnitsInOneEgg else 1
    return counts, completed


@register("Perception")
class Perception(Component):
    """Writes `world`. With `fog_filter` (default on under complete map information) enemy units we
    could not actually see are removed from the observation before anything else reads it; the
    full enemy state goes to `truth`, which only REPORT components may read."""

    phase = Phase.SENSE
    writes = ("world", "truth")
    order = 0

    def __init__(self, fog_filter: bool = True) -> None:
        self.fog_filter = fog_filter
        self._last_attacked = -10_000
        self._gathered = (0, 0, 0)
        self._enemy_seen = False
        self._was_under_attack = False

    def on_start(self, bb: Blackboard) -> None:
        self._last_attacked = -10_000
        self._gathered = (0, 0, 0)
        self._enemy_seen = False
        self._was_under_attack = False
        bb.truth.enabled = bool(bb.config.get("complete_map_information", False))

    def tick(self, bb: Blackboard) -> None:
        obs, game = bb.obs, bb.game
        if bb.truth.enabled:
            from .truth import capture_truth, fog_filter
            capture_truth(bb)
            if self.fog_filter:
                fog_filter(obs)
        mine = obs.my_units
        if len(mine) and (mine["flags"] & UnitFlag.RecentlyAttacked).any():
            self._last_attacked = obs.frame_count

        w = bb.world
        w.frame, w.minerals, w.gas = obs.frame_count, obs.minerals, obs.gas
        w.supply_used, w.supply_total = obs.supply_used, obs.supply_total
        w.counts, w.completed = unit_counts(obs, game)
        w.enemies = obs.enemy_units
        w.main_tile = tuple(game.self_player.start_location)
        w.natural_tile = game.natural.tile if game.natural is not None else None
        w.main_choke = game.main_choke.center if game.main_choke is not None else None
        w.under_attack = obs.frame_count - self._last_attacked < UNDER_ATTACK_FRAMES

        done = obs.completed(mine)
        types = np.clip(done["type"], 0, len(game.unit_types) - 1)
        flags = game.unit_types["flags"][types]
        worker = (flags & UnitTypeFlag.Worker) != 0
        w.workers = done[worker]
        w.army = done[((flags & UnitTypeFlag.CanAttack) != 0) & ~worker & ((flags & UnitTypeFlag.Building) == 0)]
        w.buildings = done[(flags & UnitTypeFlag.Building) != 0]
        depots = done[(flags & UnitTypeFlag.ResourceDepot) != 0]
        w.depots = [(int(d["x"]), int(d["y"])) for d in depots]
        if len(w.army):
            w.army_supply = int(game.unit_types["supply_required"][w.army["type"]].sum()) // 2
        else:
            w.army_supply = 0

        me = obs.me
        f0, m0, g0 = self._gathered
        if f0 and obs.frame_count > f0:
            per_min = 24 * 60 / (obs.frame_count - f0)
            w.income_minerals += INCOME_ALPHA * ((me.gathered_minerals - m0) * per_min - w.income_minerals)
            w.income_gas += INCOME_ALPHA * ((me.gathered_gas - g0) * per_min - w.income_gas)
        self._gathered = (obs.frame_count, me.gathered_minerals, me.gathered_gas)

        bb.leases.sync(int(i) for i in mine["id"])
        if len(w.enemies) and not self._enemy_seen:
            self._enemy_seen = True
            bb.raise_event("enemy_seen")
        if w.under_attack and not self._was_under_attack:
            bb.raise_event("under_attack")
        self._was_under_attack = w.under_attack
