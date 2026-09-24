"""Training games with full map information: capture ground truth, then hide it again.

`capture_truth` fills the privileged `truth` section from every enemy unit. `fog_filter` then
removes enemy units whose `visible_mask` bit for us is not set (what a normal game would show)
and blanks the enemy player's economy, so no component can cheat by accident.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from blackboard import Blackboard
from bwbot import Observation
from bwbot.observation import UnitTypeFlag


def capture_truth(bb: Blackboard) -> None:
    obs, game, t = bb.obs, bb.game, bb.truth
    enemies = obs.enemy_units
    t.frame = obs.frame_count
    t.counts = {}
    t.buildings = set()
    t.bases = []
    t.army_supply = 0.0
    if len(enemies) == 0:
        return
    types = np.clip(enemies["type"], 0, len(game.unit_types) - 1)
    flags = game.unit_types["flags"][types]
    for u, ty, fl in zip(enemies, types, flags):
        ty = int(ty)
        t.counts[ty] = t.counts.get(ty, 0) + 1
        if fl & UnitTypeFlag.Building:
            t.buildings.add(ty)
            if fl & UnitTypeFlag.ResourceDepot:
                t.bases.append((int(u["x"]) // 32, int(u["y"]) // 32))
        elif not (fl & UnitTypeFlag.Worker):
            t.army_supply += int(game.unit_types["supply_required"][ty]) / 2


def fog_filter(obs: Observation) -> None:
    units = obs.units
    enemy_ids = [p.id for p in obs.game.enemies]
    is_enemy = np.isin(units["player"], enemy_ids)
    seen = (units["visible_mask"] & np.uint32(1 << obs.self_id)) != 0
    obs.units = units[~is_enemy | seen]
    obs._by_id = None
    players = obs.players
    for pid in enemy_ids:
        p = players.get(pid)
        if p is not None:
            players[pid] = dataclasses.replace(p, minerals=0, gas=0, gathered_minerals=0, gathered_gas=0,
                                               supply_used=0, supply_total=0, unit_score=0, kill_score=0,
                                               building_score=0, razing_score=0)
