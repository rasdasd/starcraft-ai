"""Belief implementations. `LegacyBelief` wraps mybot's InformationManager + OpponentModel."""
from __future__ import annotations

from blackboard import Blackboard, Component, Phase
from blackboard.profile import register
from bwbot.observation import UnitTypeFlag
from mybot.information import InformationManager
from mybot.opponent import OPENING_NAMES, OpponentModel

from ..units import AIR_COMBAT, AIR_TECH


@register("LegacyBelief")
class LegacyBelief(Component):
    phase = Phase.SENSE
    reads = ("world",)
    writes = ("belief",)
    order = 20

    def __init__(self) -> None:
        self.info = InformationManager()
        self.opponent = OpponentModel()

    def on_start(self, bb: Blackboard) -> None:
        self.info.on_start(bb.game)
        self.opponent.on_start(bb.game)
        bb.services["info"] = self.info
        b = bb.belief
        b.enemy_start = self.info.enemy_start
        b.start_candidates = {tuple(s): 1.0 / max(1, len(self.info.other_starts)) for s in self.info.other_starts}

    def tick(self, bb: Blackboard) -> None:
        obs, game = bb.obs, bb.game
        prev_start = self.info.enemy_start
        self.info.update(obs, game)
        self.opponent.update(obs, game, self.info)
        snap = self.opponent.snapshot()
        b = bb.belief
        prev_open = b.opening
        b.enemy_race = snap.race
        b.enemy_start = self.info.enemy_start
        b.buildings = self.info.buildings()
        b.counts = {int(t): float(n) for t, n in snap.counts.items()}
        b.first_seen = dict(self.opponent.first)
        b.tech = {t for t, _, _ in b.buildings} | {
            int(t) for t in snap.counts if int(game.unit_types["flags"][int(t)]) & UnitTypeFlag.Building}
        b.air = float(snap.air_units)
        b.army_supply = float(snap.ground_army + snap.air_units)
        b.proxy = snap.proxy
        b.opening = OPENING_NAMES[snap.opening] if 0 <= snap.opening < len(OPENING_NAMES) else "unknown"
        b.memory = snap
        if self.info.has_enemy_base() and self.info.enemy_start != prev_start:
            bb.raise_event("enemy_base_found")
        if b.opening != prev_open:
            bb.raise_event("opening_changed")
        if "enemy_air" not in bb.stats and any(int(t) in AIR_COMBAT or int(t) in AIR_TECH
                                               for t, n in snap.counts.items() if n):
            bb.stats["enemy_air"] = bb.frame
            bb.raise_event("enemy_air")
