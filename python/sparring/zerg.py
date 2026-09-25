"""Zerg sparring partners. `sparring.zerg` = 9-pool speedlings; `sparring.zerg:Hydra` = hatch-first hydras."""
from __future__ import annotations

from bwbot import UnitType as U
from bwbot.enums import UpgradeType

from .base import LARVA, ScriptBot


class _Zerg(ScriptBot):
    race = "Zerg"
    larva_based = True
    worker, hall, supply, refinery = U.Zerg_Drone, U.Zerg_Hatchery, U.Zerg_Overlord, U.Zerg_Extractor


class Pool(_Zerg):
    build_order = [(9, U.Zerg_Spawning_Pool), (9, U.Zerg_Extractor), (12, U.Zerg_Hatchery)]
    army = [(U.Zerg_Zergling, LARVA, 1.0)]
    upgrades = [(U.Zerg_Spawning_Pool, UpgradeType.Metabolic_Boost, False)]
    max_workers = 11
    attack_at = 12
    retreat_below = 4


class Hydra(_Zerg):
    build_order = [(12, U.Zerg_Hatchery), (11, U.Zerg_Spawning_Pool), (12, U.Zerg_Extractor),
                   (18, U.Zerg_Hydralisk_Den)]
    army = [(U.Zerg_Hydralisk, LARVA, 3.0), (U.Zerg_Zergling, LARVA, 1.0)]
    upgrades = [(U.Zerg_Hydralisk_Den, UpgradeType.Grooved_Spines, False),
                (U.Zerg_Hydralisk_Den, UpgradeType.Muscular_Augments, False)]
    max_workers = 20
    attack_at = 14
    retreat_below = 5

    def _produce(self, obs, act):
        if obs.count(U.Zerg_Hydralisk_Den, completed_only=True) == 0 and obs.count(U.Zerg_Zergling) >= 6:
            if self.minerals < 150 + 50 * (obs.count(U.Zerg_Hydralisk_Den) == 0):
                return self._drones_only(obs, act)
        super()._produce(obs, act)

    def _drones_only(self, obs, act):
        if obs.count(self.worker) >= self.max_workers or obs.supply_used >= obs.supply_total:
            return
        for larva in obs.my_units_of_type(LARVA)[:1]:
            if self.minerals >= 50:
                act.morph(larva, self.worker)
                self.minerals -= 50


BOT = Pool
