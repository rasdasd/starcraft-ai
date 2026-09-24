"""Fixed-size feature vector for a learned Policy.

`encode(state)` is the only feature function. Training and inference must use the same
`FEATURE_NAMES` / `FEATURE_DIM`. Managers (placement, workers, APM trim) stay out of it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bwbot import Race, UnitType, UpgradeType

from .opponent import OPENING_NAMES, OpeningGuess, OpponentSnapshot

if TYPE_CHECKING:
    from .state import State

CHARON = UpgradeType.Charon_Boosters

OWN_TYPES = (
    UnitType.Terran_SCV, UnitType.Terran_Marine, UnitType.Terran_Vulture, UnitType.Terran_Goliath,
    UnitType.Terran_Supply_Depot, UnitType.Terran_Barracks, UnitType.Terran_Refinery,
    UnitType.Terran_Factory, UnitType.Terran_Armory, UnitType.Terran_Bunker,
    UnitType.Terran_Machine_Shop,
)

FEATURE_NAMES: list[str] = [
    "frame", "minerals", "gas", "supply_used", "supply_total", "supply_left",
    "workers", "army", "enemies", "under_attack", "has_natural", "has_choke",
    "has_charon",
    *[f"own_{int(t)}" for t in OWN_TYPES],
    "opp_unk", "opp_zerg", "opp_terran", "opp_protoss",
    *[f"open_{n}" for n in OPENING_NAMES],
    "opp_first_military", "opp_buildings", "opp_air", "opp_ground", "opp_workers",
    "opp_attacks", "opp_proxy",
]


FEATURE_DIM = len(FEATURE_NAMES)


def encode(s: State) -> np.ndarray:
    opp = s.opponent
    race = int(opp.race) if opp is not None else int(Race.Unknown)
    opening = int(opp.opening) if opp is not None else int(OpeningGuess.UNKNOWN)
    own = [s.count(t) / 20.0 for t in OWN_TYPES]
    race_oh = [0.0, 0.0, 0.0, 0.0]
    if race == int(Race.Unknown):
        race_oh[0] = 1.0
    elif race == int(Race.Zerg):
        race_oh[1] = 1.0
    elif race == int(Race.Terran):
        race_oh[2] = 1.0
    elif race == int(Race.Protoss):
        race_oh[3] = 1.0
    open_oh = [0.0] * len(OPENING_NAMES)
    if 0 <= opening < len(open_oh):
        open_oh[opening] = 1.0
    raw = [
        s.frame / 10_000,
        s.minerals / 1_000,
        s.gas / 1_000,
        s.supply_used / 200,
        s.supply_total / 200,
        s.supply_left / 200,
        len(s.workers) / 30,
        len(s.army) / 50,
        len(s.enemies) / 50,
        float(s.under_attack),
        float(s.natural_tile is not None),
        float(s.main_choke is not None),
        float(s.has_upgrade(CHARON)),
        *own,
        *race_oh,
        *open_oh,
        ((opp.first_military / 10_000) if opp and opp.first_military >= 0 else 1.0),
        (opp.buildings / 20) if opp else 0.0,
        (opp.air_units / 20) if opp else 0.0,
        (opp.ground_army / 30) if opp else 0.0,
        (opp.workers / 20) if opp else 0.0,
        min((opp.attacks / 10) if opp else 0.0, 1.0),
        float(opp.proxy) if opp else 0.0,
    ]
    x = np.asarray(raw, dtype=np.float32)
    if x.size != FEATURE_DIM:
        raise RuntimeError(f"feature length {x.size} != FEATURE_DIM {FEATURE_DIM}")
    return x


def empty_opponent() -> OpponentSnapshot:
    return OpponentSnapshot()
