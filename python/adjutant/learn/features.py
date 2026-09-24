"""Versioned feature encoders for learned components.

Strategy: a context vector (meta features + a snapshot of the board at decision time) and the
model input for "how likely do we win if template T is active in this context":
    [ctx, onehot(T), ctx * onehot(T)]
i.e. one linear win model per template plus a shared part, in a single `.npz`. The template list
is part of the spec name/hash, so a model trained on a different template set is rejected.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from blackboard import Blackboard
from blackboard.models import FeatureSpec
from mybot.opponent import OPENING_NAMES

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


def strategy_spec(templates: Sequence[str]) -> FeatureSpec:
    names = list(CTX_NAMES) + [f"t_{t}" for t in templates] + [f"{c}*{t}" for t in templates for c in CTX_NAMES]
    return FeatureSpec("strategy:" + ",".join(templates), STRATEGY_VERSION, tuple(names))


def strategy_input(ctx: np.ndarray, template_index: int, n_templates: int) -> np.ndarray:
    oh = np.zeros(n_templates, np.float32)
    oh[template_index] = 1.0
    cross = np.zeros((n_templates, len(ctx)), np.float32)
    cross[template_index] = ctx
    return np.concatenate([ctx, oh, cross.ravel()])


def strategy_inputs(ctx: np.ndarray, n_templates: int) -> np.ndarray:
    """All templates at once: (n_templates, dim)."""
    return np.stack([strategy_input(ctx, i, n_templates) for i in range(n_templates)])
