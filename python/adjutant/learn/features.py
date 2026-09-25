"""Versioned feature encoders for learned components.

Strategy: a context vector (meta features + a snapshot of the board at decision time), a build
descriptor (tags and opening shape, `build_features`), and the model input for "how likely do we
win if build B is active in this context":
    [ctx, bf(B), ctx * bf(B)]   (outer product, flattened)
Builds are described rather than one-hot encoded, so a model scores builds it never saw (your own
or generated ones) by their resemblance to the ones it did, and adding a build needs no retrain.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from blackboard import Blackboard
from blackboard.models import FeatureSpec
from bwbot import UnitType as U
from ..enemy import OPENING_NAMES

from ..components.meta import META_FEATURES
from ..units import AIR_TECH, CLOAKERS

STRATEGY_VERSION = 1

BOARD_FEATURES = (
    "minute", "workers", "army", "bases_own", "income_m", "income_g", "supply",
    "enemy_army", "enemy_air", "enemy_cloak", "enemy_proxy", "enemy_bases", "enemy_tech", "enemy_air_tech",
    "army_ratio", "under_attack", "threat",
    *[f"open_{n}" for n in OPENING_NAMES],
)
CTX_NAMES = tuple(META_FEATURES) + BOARD_FEATURES
CTX_SPEC = FeatureSpec("strategy_ctx", STRATEGY_VERSION, CTX_NAMES)


def strategy_context(bb: Blackboard) -> np.ndarray:
    w, b, m = bb.world, bb.belief, bb.meta
    meta = [float(m.features.get(k, 0.0)) for k in META_FEATURES]
    open_oh = [1.0 if b.opening == n else 0.0 for n in OPENING_NAMES]
    enemy_army = float(b.army_supply)
    board = [
        w.frame / (24 * 60) / 20,
        len(w.workers) / 60,
        w.army_supply / 100,
        len(w.depots) / 4,
        w.income_minerals / 2000,
        w.income_gas / 1000,
        w.supply_used / 200,
        enemy_army / 100,
        float(b.air) / 20,
        float(b.cloak or any(t in CLOAKERS for t in b.counts if b.counts[t] > 0)),
        float(b.proxy),
        sum(1 for eb in b.bases if eb.alive) / 4,
        len(b.tech) / 15,
        float(any(t in AIR_TECH for t in b.tech)),
        (w.army_supply + 1) / (w.army_supply + enemy_army + 2),
        float(w.under_attack),
        float(bb.threats.level),
    ]
    return np.array(meta + board + open_oh, np.float32)


BUILD_VERSION = 1
# Tag vocabulary; tags outside it are allowed in builds but carry no features.
BUILD_TAGS = ("rush", "aggressive", "timing", "all_in", "cheese", "one_base", "expand", "macro", "fast_expand",
              "defensive", "rush_safe", "anti_air", "air", "mech", "bio", "tech", "harass", "detection",
              "economic", "late_game")
_HALLS = {int(U.Terran_Command_Center), int(U.Zerg_Hatchery), int(U.Protoss_Nexus)}
_GAS = {int(U.Terran_Refinery), int(U.Zerg_Extractor), int(U.Protoss_Assimilator)}
BUILD_NAMES = (*[f"tag_{t}" for t in BUILD_TAGS], "attack_supply", "retreat_supply", "open_expand",
               "open_expand_supply", "open_gas", "open_gas_supply", "open_steps", "bias")
BUILD_SPEC = FeatureSpec("build", BUILD_VERSION, BUILD_NAMES)


def build_features(t) -> np.ndarray:
    """Race-agnostic descriptor of a build (a `Template`): its tags plus the shape of its opening.
    The constant `bias` makes the ctx x build product include ctx itself."""
    steps = t.opening_steps()
    hall = next((s for s, u in steps if u in _HALLS), None)
    gas = next((s for s, u in steps if u in _GAS), None)
    return np.array([
        *[1.0 if tag in t.tags else 0.0 for tag in BUILD_TAGS],
        t.attack_supply / 100, t.retreat_supply / 100,
        float(hall is not None), (hall or 0) / 30, float(gas is not None), (gas or 0) / 30,
        len(steps) / 8, 1.0,
    ], np.float32)


def strategy_spec() -> FeatureSpec:
    names = list(CTX_NAMES) + list(BUILD_NAMES) + [f"{c}*{b}" for c in CTX_NAMES for b in BUILD_NAMES]
    return FeatureSpec("strategy_build", STRATEGY_VERSION * 100 + BUILD_VERSION, tuple(names))


def strategy_input(ctx: np.ndarray, bf: np.ndarray) -> np.ndarray:
    return np.concatenate([ctx, bf, np.outer(ctx, bf).ravel()]).astype(np.float32)


def strategy_inputs(ctx: np.ndarray, bfs: Sequence[np.ndarray]) -> np.ndarray:
    """One row per candidate build: (n_builds, dim)."""
    return np.stack([strategy_input(ctx, bf) for bf in bfs])


def blend_features(builds: dict, weights: dict[str, float]) -> np.ndarray:
    """Weighted mean descriptor of the active builds (blend mode)."""
    tot = sum(w for n, w in weights.items() if n in builds) or 1.0
    return sum(build_features(builds[n]) * (w / tot) for n, w in weights.items() if n in builds)
