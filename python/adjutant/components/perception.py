"""Perception: observation -> `world` (and `truth` in training games with full map information)."""
from __future__ import annotations

import numpy as np

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from bwbot import UnitFlag
from bwbot.observation import UnitTypeFlag
from mybot.state import Memory, perceive

INCOME_ALPHA = 0.05      # EMA per decision


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
        s = perceive(obs, game, Memory(last_attacked_frame=self._last_attacked))

        w = bb.world
        w.state = s
        w.frame, w.minerals, w.gas = s.frame, s.minerals, s.gas
        w.supply_used, w.supply_total = s.supply_used, s.supply_total
        w.counts, w.completed = s.counts, s.completed
        w.workers, w.army, w.enemies = s.workers, s.army, s.enemies
        w.main_tile, w.natural_tile, w.main_choke = s.main_tile, s.natural_tile, s.main_choke
        w.under_attack = s.under_attack

        done = obs.completed(mine)
        types = np.clip(done["type"], 0, len(game.unit_types) - 1)
        flags = game.unit_types["flags"][types]
        w.buildings = done[(flags & UnitTypeFlag.Building) != 0]
        depots = done[(flags & UnitTypeFlag.ResourceDepot) != 0]
        w.depots = [(int(d["x"]), int(d["y"])) for d in depots]
        if len(s.army):
            w.army_supply = int(game.unit_types["supply_required"][s.army["type"]].sum()) // 2
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
        if len(s.enemies) and not self._enemy_seen:
            self._enemy_seen = True
            bb.raise_event("enemy_seen")
        if s.under_attack and not self._was_under_attack:
            bb.raise_event("under_attack")
        self._was_under_attack = s.under_attack
