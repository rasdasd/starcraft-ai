"""Protoss sparring partners. `sparring.protoss` = 2-gate zealots; `sparring.protoss:Dragoon` = 1-gate core dragoons."""
from __future__ import annotations

from bwbot import UnitType as U
from bwbot.enums import UpgradeType

from .base import ScriptBot


class _Protoss(ScriptBot):
    race = "Protoss"
    worker, hall, supply, refinery = U.Protoss_Probe, U.Protoss_Nexus, U.Protoss_Pylon, U.Protoss_Assimilator


class Zealot(_Protoss):
    build_order = [(8, U.Protoss_Pylon), (10, U.Protoss_Gateway), (12, U.Protoss_Gateway), (15, U.Protoss_Pylon),
                   (22, U.Protoss_Gateway)]
    army = [(U.Protoss_Zealot, U.Protoss_Gateway, 1.0)]
    max_workers = 16
    gas_workers = 0
    attack_at = 8
    retreat_below = 3


class Dragoon(_Protoss):
    build_order = [(8, U.Protoss_Pylon), (10, U.Protoss_Gateway), (11, U.Protoss_Assimilator),
                   (13, U.Protoss_Cybernetics_Core), (16, U.Protoss_Pylon), (18, U.Protoss_Gateway),
                   (24, U.Protoss_Gateway)]
    army = [(U.Protoss_Dragoon, U.Protoss_Gateway, 3.0), (U.Protoss_Zealot, U.Protoss_Gateway, 1.0)]
    upgrades = [(U.Protoss_Cybernetics_Core, UpgradeType.Singularity_Charge, False)]
    max_workers = 20
    attack_at = 10
    retreat_below = 4


BOT = Zealot
